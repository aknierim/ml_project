import sys

import click


def save_on_interrupt(model_path):
    """Utility function that saves the model
    on keyboard interrupt.
    """

    interrupt = click.confirm(
        "KeyboardInterrupt: Do you wish to stop training and save the model?",
        default=False,
        show_default=True,
        abort=False,
    )

    if interrupt:
        print(f"Saving model to file {model_path} after {'EPOCHS'} epoch(s).")

    sys.exit(1)
