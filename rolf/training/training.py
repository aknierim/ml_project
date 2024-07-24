from pathlib import Path

import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn.metrics import roc_auc_score

from rolf.architecture import ResNet

MODEL_DICT = {"ResNet": ResNet}

OPTIMIZERS = {"Adam": optim.AdamW, "SGD": optim.SGD}


def _create_model(model_name, model_hparams):
    if model_name in MODEL_DICT:
        return MODEL_DICT[model_name](**model_hparams)
    else:
        raise ValueError(
            f"Unknown model name '{model_name}'. "
            f"Available models are: {str(MODEL_DICT.keys())}"
        )


class TrainModule(L.LightningModule):
    "Choo choo choo choo choo choo choo"

    def __init__(
        self,
        model_name: str,
        model_hparams: dict,
        optimizer_name: str,
        optimizer_hparams: dict,
        class_weights: list = None,
    ) -> None:
        """Training Module

        Parameters
        ----------
        model_name : str
            Name of the model/CNN to run. Used for creating the model.
        model_hparams : dict
            Dictionary of hyperparameters for the model.
        optimizer_name : str
            Name of the optimizer to use. Either 'Adam or SGD'.
        optimizer_hparams : dict
            Hyperparameters for the optimizer, as dictionary.
            This includes learning rate, weight decay, etc.
        """
        super().__init__()
        self.save_hyperparameters()

        self.model = _create_model(model_name, model_hparams)
        self.loss_module = nn.CrossEntropyLoss(weight=class_weights)
        self.val_loss_module = nn.CrossEntropyLoss()

        # Example input for visualizing the graph in Tensorboard
        self.example_input_array = torch.zeros((1, 1, 300, 300))

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self) -> tuple[list, list]:
        if self.hparams.optimizer_name in OPTIMIZERS:
            if self.hparams.optimizer_name == "Adam":
                self.hparams.optimizer_hparams.pop("momentum", None)
            optimizer = OPTIMIZERS[self.hparams.optimizer_name](
                self.parameters(), **self.hparams.optimizer_hparams
            )
        else:
            avail_opt = list(OPTIMIZERS.keys())
            raise ValueError(
                f"Unknown optimizer '{self.hparams.optimizer_name}'! "
                f"Available optimizers are: {avail_opt}"
            )

        # Reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[100, 150], gamma=0.1
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        """Log training roc auc and loss (per epoch by default)"""
        imgs, labels = batch
        preds = self.model(imgs).softmax(dim=1)
        loss = self.loss_module(preds, labels)

        roc_auc = roc_auc_score(
            y_true=labels.cpu().detach().numpy(),
            y_score=preds.cpu().detach().numpy(),
            multi_class="ovo",
            average="macro",
            labels=[0, 1, 2, 3],
        )
        # roc_auc = MulticlassAUROC(num_classes=4, average="weighted", thresholds=None)

        # self.log("train_roc_auc", roc_auc(preds, labels), on_step=False, on_epoch=True)
        self.log("train_roc_auc", roc_auc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        """Log validation roc auc and loss (per epoch by default)"""
        imgs, labels = batch
        preds = self.model(imgs).softmax(dim=1)
        loss = self.val_loss_module(preds, labels)

        # roc_auc = MulticlassAUROC(num_classes=4, average="weighted", thresholds=None)
        roc_auc = roc_auc_score(
            y_true=labels.cpu().numpy(),
            y_score=preds.cpu().numpy(),
            multi_class="ovo",
            average="macro",
            labels=[0, 1, 2, 3],
        )

        # self.log("val_roc_auc", roc_auc(preds, labels), on_step=False, on_epoch=True)
        self.log("val_roc_auc", roc_auc, on_step=False, on_epoch=True)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx) -> None:
        """Log test roc auc (per epoch by default)"""
        imgs, labels = batch
        preds = self.model(imgs).softmax(dim=1)
        # roc_auc = MulticlassAUROC(num_classes=4, average="weighted", thresholds=None)
        roc_auc = roc_auc_score(
            labels.cpu().detach().numpy(),
            preds.cpu().detach().numpy(),
            multi_class="ovo",
            average="macro",
            labels=[0, 1, 2, 3],
        )

        # self.log("test_roc_auc", roc_auc(preds, labels), on_step=False, on_epoch=True)
        self.log("test_roc_auc", roc_auc, on_step=False, on_epoch=True)


def train_model(
    model_name: str,
    train_loader: data.DataLoader,
    val_loader: data.DataLoader,
    test_loader: data.DataLoader,
    checkpoint_path: str | Path,
    save_name: str | Path = "",
    epochs: int = 300,
    class_weights: list = None,
    **kwargs,
):
    """Train the model passed via 'model_name'.

    Parameters
    ----------
    model_name : str
        Name of the model you want to run. Is used
        to look up the class in 'MODEL_DICT'
    train_loader : DataLoader
    save_name : str, optional
        If specified, this name will be used
        for creating the checkpoint and logging directory.
    """
    if not save_name:
        save_name = model_name

    if not isinstance(checkpoint_path, Path):
        checkpoint_path = Path(checkpoint_path)

    trainer = L.Trainer(
        default_root_dir=Path(checkpoint_path / save_name),
        accelerator="auto",
        devices=1,
        max_epochs=epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_roc_auc"),
            LearningRateMonitor("epoch"),
        ],
    )

    trainer.logger._log_graph = True
    trainer.logger._default_hp_metric = None

    pretrained_filename = Path(checkpoint_path / save_name).with_suffix(".ckpt")

    if pretrained_filename.exists():
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = TrainModule.load_from_checkpoint(pretrained_filename)
    else:
        L.seed_everything(42)
        model = TrainModule(
            model_name=model_name, class_weights=class_weights, **kwargs
        )
        trainer.fit(model, train_loader, val_loader)
        model = TrainModule.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )

    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)

    result = {
        "test": test_result[0]["test_roc_auc"],
        "val": val_result[0]["test_roc_auc"],
    }

    return model, result, trainer


def main() -> None:
    pass
