"""Optimization visualization."""

#
# This module is an adapted version of the visualization
# module of optuna. This allows us to have more control
# over the visuals of the plots.
#
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from optuna.importance._base import BaseImportanceEvaluator
from optuna.study import Study
from optuna.trial import FrozenTrial
from optuna.visualization._parallel_coordinate import (
    _get_parallel_coordinate_info,
)
from optuna.visualization._param_importances import (
    _get_importances_infos,
    _ImportancesInfo,
)

AXES_PADDING_RATIO = 1.05


def plot_param_importances(
    study: Study,
    ax,
    fig,
    evaluator: BaseImportanceEvaluator | None = None,
    params: list[str] | None = None,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
    fontsize=12,
):
    importances_infos = _get_importances_infos(
        study, evaluator, params, target, target_name
    )

    height = 0.8 / len(importances_infos)

    ax.set_title("Hyperparameter Importances", loc="left", fontsize=fontsize * 1.2)
    ax.set_xlabel("Hyperparameter Importance", fontsize=fontsize)
    ax.set_ylabel("Hyperparameter", fontsize=fontsize)
    ax.tick_params(axis="x", labelsize=fontsize * 0.9)

    for objective_id, info in enumerate(importances_infos):
        param_names = info.param_names
        pos = np.arange(len(param_names))
        offset = height * objective_id
        importance_values = info.importance_values

        if not importance_values:
            continue

        ax.barh(
            pos + offset,
            importance_values,
            height=height,
            align="center",
            label=info.target_name,
        )

        _set_bar_labels(info, fig, ax, offset)
        ax.set_yticks(pos + offset / 2, param_names, fontsize=fontsize * 0.9)

    return ax


def _set_bar_labels(
    info: _ImportancesInfo, fig, ax, offset: float, fontsize=14
) -> None:
    renderer = fig.canvas.get_renderer()
    for idx, (val, label) in enumerate(
        zip(info.importance_values, info.importance_labels)
    ):
        text = ax.text(val + 0.01, idx + offset, label, va="center", fontsize=fontsize)

        # Sometimes horizontal axis needs to be re-scaled
        # to avoid text going over plot area.
        bbox = text.get_window_extent(renderer)
        bbox = bbox.transformed(ax.transData.inverted())
        _, plot_xmax = ax.get_xlim()
        bbox_xmax = bbox.xmax

        if bbox_xmax > plot_xmax:
            ax.set_xlim(xmax=AXES_PADDING_RATIO * bbox_xmax)


def plot_parallel_coordinate(
    study: Study,
    ax,
    fig,
    params: list[str] | None = None,
    target: Callable[[FrozenTrial], float] | None = None,
    target_name: str = "Objective Value",
    cmap="inferno",
    fontsize=12,
):
    info = _get_parallel_coordinate_info(study, params, target, target_name)
    reversescale = info.reverse_scale
    target_name = info.target_name

    # Set up the graph style.
    cmap = plt.get_cmap(f"{cmap}_r" if reversescale else cmap)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Prepare data for plotting.
    if len(info.dims_params) == 0 or len(info.dim_objective.values) == 0:
        return ax

    obj_min = info.dim_objective.range[0]
    obj_max = info.dim_objective.range[1]
    obj_w = obj_max - obj_min
    dims_obj_base = [[o] for o in info.dim_objective.values]
    for dim in info.dims_params:
        p_min = dim.range[0]
        p_max = dim.range[1]
        p_w = p_max - p_min

        if p_w == 0.0:
            center = obj_w / 2 + obj_min
            for i in range(len(dim.values)):
                dims_obj_base[i].append(center)
        else:
            for i, v in enumerate(dim.values):
                dims_obj_base[i].append((v - p_min) / p_w * obj_w + obj_min)

    # Draw multiple line plots and axes.
    # Ref: https://stackoverflow.com/a/50029441
    n_params = len(info.dims_params)

    xs = [range(n_params + 1) for _ in range(len(dims_obj_base))]
    segments = [np.column_stack([x, y]) for x, y in zip(xs, dims_obj_base)]

    lc = LineCollection(segments, cmap=cmap)
    lc.set_array(np.asarray(info.dim_objective.values))

    cbar = fig.colorbar(lc, pad=0.1, ax=ax)
    cbar.set_label(target_name, fontsize=fontsize)
    cbar.ax.set_yticks(np.linspace(np.min(dims_obj_base), np.max(dims_obj_base), 10))
    cbar.ax.set_yticklabels(labels=cbar.ax.get_yticklabels(), fontsize=fontsize * 0.9)

    var_names = [info.dim_objective.label] + [dim.label for dim in info.dims_params]

    for i, dim in enumerate(info.dims_params):
        ax2 = ax.twinx()
        if dim.is_log:
            ax2.set_ylim(np.power(10, dim.range[0]), np.power(10, dim.range[1]))
            ax2.set_yscale("log")
        else:
            ax2.set_ylim(dim.range[0], dim.range[1])
        ax2.spines["top"].set_visible(False)
        ax2.spines["bottom"].set_visible(False)
        ax2.xaxis.set_visible(False)
        ax2.spines["right"].set_position(("axes", (i + 1) / n_params))
        if dim.is_cat:
            ax2.set_yticks(dim.tickvals)
            ax2.set_yticklabels(dim.ticktext)

        ax2.tick_params(axis="y", labelsize=fontsize)

    ax.add_collection(lc)

    ax.set_title("Parallel Coordinate Plot", fontsize=fontsize * 1.2, y=1.05)
    ax.set(
        xlim=(0, n_params),
        ylim=(info.dim_objective.range[0], info.dim_objective.range[1]),
    )
    ax.tick_params(axis="y", labelsize=fontsize)
    ax.set_xticks(range(n_params + 1))
    ax.set_xticklabels(
        var_names, rotation=45, fontsize=fontsize, ha="right", rotation_mode="anchor"
    )

    return ax
