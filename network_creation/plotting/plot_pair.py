from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter, NullLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.ndimage import gaussian_filter
from scipy.stats import binned_statistic_2d


@dataclass(frozen=True)
class _AxisSpec:
    values: np.ndarray
    is_ordered_discrete: bool
    limits: tuple[float, float]
    levels: np.ndarray | None = None
    edges: np.ndarray | None = None


def _discrete_edges(levels: np.ndarray) -> np.ndarray:
    if levels.size == 1:
        level = float(levels[0])
        return np.array([level - 0.5, level + 0.5], dtype=float)
    midpoints = (levels[:-1] + levels[1:]) / 2.0
    return np.concatenate(
        (
            [levels[0] - (midpoints[0] - levels[0])],
            midpoints,
            [levels[-1] + (levels[-1] - midpoints[-1])],
        )
    )


def _build_axis_spec(
    values: np.ndarray,
    is_ordered_discrete: bool,
    bounds: tuple[float, float] | None = None,
) -> _AxisSpec:
    values = np.asarray(values)
    if is_ordered_discrete:
        levels = np.asarray(np.unique(values), dtype=float)
        edges = _discrete_edges(levels)
        return _AxisSpec(
            values, True, (float(edges[0]), float(edges[-1])), levels, edges
        )
    if bounds is None:
        bounds = (float(values.min()), float(values.max()))
    return _AxisSpec(values, False, (float(bounds[0]), float(bounds[1])))


def _bin_indices(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.searchsorted(edges, values, side="right") - 1, 0, edges.size - 2)


def _axis_grid(spec: _AxisSpec, resolution: int) -> tuple[np.ndarray, np.ndarray]:
    if spec.is_ordered_discrete:
        assert spec.edges is not None
        assert spec.levels is not None
        return spec.edges, np.searchsorted(spec.levels, spec.values)
    edges = np.linspace(spec.limits[0], spec.limits[1], resolution + 1)
    return edges, _bin_indices(spec.values, edges)


def _aggregate_mean_grid(
    x_indices: np.ndarray,
    y_indices: np.ndarray,
    z: np.ndarray,
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    z_sum = np.zeros(shape, dtype=float)
    z_count = np.zeros(shape, dtype=float)
    np.add.at(z_sum, (x_indices, y_indices), z)
    np.add.at(z_count, (x_indices, y_indices), 1.0)
    return z_sum, z_count


def _apply_axis_metadata(ax: Axes, x_spec: _AxisSpec, y_spec: _AxisSpec) -> None:
    ax.set_xlim(*x_spec.limits)
    ax.set_ylim(*y_spec.limits)
    if x_spec.levels is not None:
        ax.set_xticks(x_spec.levels)
        ax.xaxis.set_minor_locator(NullLocator())
        assert x_spec.edges is not None
        for edge in x_spec.edges[1:-1]:
            ax.axvline(
                float(edge),
                color="w",
                linewidth=2.0,
                alpha=0.25,
                zorder=2.5,
            )
    if y_spec.levels is not None:
        ax.set_yticks(y_spec.levels)
        ax.yaxis.set_minor_locator(NullLocator())
        assert y_spec.edges is not None
        for edge in y_spec.edges[1:-1]:
            ax.axhline(
                float(edge),
                color="w",
                linewidth=2.0,
                alpha=0.25,
                zorder=2.5,
            )


def _render_descriptor(
    x_spec: _AxisSpec,
    y_spec: _AxisSpec,
    z: np.ndarray,
    method: str,
    resolution: int,
    ax: Axes,
    norm: Normalize,
    cmap: str,
    point_size: float,
    alpha: float,
    sigma: float,
) -> Axes:
    if method == "points":
        ax.scatter(
            x_spec.values,
            y_spec.values,
            c=z,
            cmap=cmap,
            norm=norm,
            s=point_size,
            alpha=alpha,
            linewidths=0,
        )
        _apply_axis_metadata(ax, x_spec, y_spec)
        return ax

    if not x_spec.is_ordered_discrete and not y_spec.is_ordered_discrete:
        z_sum, _, _, _ = binned_statistic_2d(
            x_spec.values,
            y_spec.values,
            z,
            statistic="sum",
            bins=resolution,
            range=[list(x_spec.limits), list(y_spec.limits)],
        )
        z_count, _, _, _ = binned_statistic_2d(
            x_spec.values,
            y_spec.values,
            z,
            statistic="count",
            bins=resolution,
            range=[list(x_spec.limits), list(y_spec.limits)],
        )
        zi = gaussian_filter(z_sum, sigma=sigma, mode="nearest") / np.maximum(
            gaussian_filter(z_count, sigma=sigma, mode="nearest"),
            1e-12,
        )
        ax.imshow(
            zi.T,
            cmap=cmap,
            norm=norm,
            origin="lower",
            extent=[*x_spec.limits, *y_spec.limits],
            interpolation="nearest",
            aspect="auto",
        )
        _apply_axis_metadata(ax, x_spec, y_spec)
        return ax

    x_edges, x_indices = _axis_grid(x_spec, resolution)
    y_edges, y_indices = _axis_grid(y_spec, resolution)
    z_sum, z_count = _aggregate_mean_grid(
        x_indices=x_indices,
        y_indices=y_indices,
        z=z,
        shape=(x_edges.size - 1, y_edges.size - 1),
    )
    sigma_x = 0.0 if x_spec.is_ordered_discrete else sigma
    sigma_y = 0.0 if y_spec.is_ordered_discrete else sigma
    if sigma_x or sigma_y:
        z_sum = gaussian_filter(z_sum, sigma=(sigma_x, sigma_y), mode="nearest")
        z_count = gaussian_filter(z_count, sigma=(sigma_x, sigma_y), mode="nearest")

    ax.pcolormesh(
        x_edges,
        y_edges,
        (z_sum / np.maximum(z_count, 1e-12)).T,
        cmap=cmap,
        norm=norm,
        shading="flat",
    )
    _apply_axis_metadata(ax, x_spec, y_spec)
    return ax


def plot_2d_descriptor(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    method: str,
    resolution: int,
    ax: Axes,
    norm: Normalize,
    bounds=None,
    cmap: str = "Greys",
    point_size: float = 6,
    alpha: float = 0.5,
    sigma: float = 1.5,
    x_is_ordered_discrete: bool = False,
    y_is_ordered_discrete: bool = False,
):
    if method not in {"points", "gaussian"}:
        raise ValueError(f"{method} is not a valid method for plotting")
    if bounds is None:
        bounds = [x.min(), x.max(), y.min(), y.max()]

    return _render_descriptor(
        x_spec=_build_axis_spec(
            x,
            is_ordered_discrete=x_is_ordered_discrete,
            bounds=(float(bounds[0]), float(bounds[1])),
        ),
        y_spec=_build_axis_spec(
            y,
            is_ordered_discrete=y_is_ordered_discrete,
            bounds=(float(bounds[2]), float(bounds[3])),
        ),
        z=z,
        method=method,
        resolution=resolution,
        ax=ax,
        norm=norm,
        cmap=cmap,
        point_size=point_size,
        alpha=alpha,
        sigma=sigma,
    )


def plot_pair(
    X: np.ndarray,
    y: np.ndarray,
    labels=None,
    ordered_discrete_keys: list[str] | None = None,
    base_fig_size=2.4,
    cmap="Greys",
    label_y="y",
    atol=0.0,
    method="points",
    resolution=100,
    point_size=6,
    alpha=0.5,
    sigma=1.5,
    norm: Normalize | None = None,
):
    d = X.shape[1]
    if labels is None:
        labels = [f"x{i}" for i in range(d)]
    elif len(labels) != d:
        raise ValueError(f"`labels` must have length {d}, got {len(labels)}")
    else:
        labels = list(labels)

    if method not in {"points", "gaussian"}:
        raise ValueError(f"{method} is not a valid method for plotting")

    discrete_keys = (
        set() if ordered_discrete_keys is None else set(ordered_discrete_keys)
    )
    unknown_keys = discrete_keys.difference(labels)
    if unknown_keys:
        raise ValueError(
            "`ordered_discrete_keys` contains unknown labels: "
            + ", ".join(sorted(unknown_keys))
        )

    keep = np.var(X, axis=0) > atol
    if not np.any(keep):
        raise ValueError("All X dimensions are constant; nothing to plot.")

    X = X[:, keep]
    labels = [label for label, keep_dim in zip(labels, keep) if keep_dim]
    discrete_flags = [label in discrete_keys for label in labels]
    if X.shape[1] < 2:
        raise ValueError("Need at least two varying X dimensions to build pair plots.")

    resolved_norm = norm or Normalize(vmin=float(np.min(y)), vmax=float(np.max(y)))
    axis_specs = [
        _build_axis_spec(X[:, i], discrete_flags[i]) for i in range(X.shape[1])
    ]
    n = X.shape[1] - 1
    fig, axes = plt.subplots(
        n,
        n,
        figsize=(base_fig_size * n, base_fig_size * n),
        layout="constrained",
    )
    axes = np.array([[axes]]) if n == 1 else axes

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue

            _render_descriptor(
                x_spec=axis_specs[j],
                y_spec=axis_specs[i + 1],
                z=y,
                method=method,
                resolution=resolution,
                ax=ax,
                norm=resolved_norm,
                cmap=cmap,
                point_size=point_size,
                alpha=alpha,
                sigma=sigma,
            )
            ax.set_box_aspect(1)
            ax.grid(False)
            if i == n - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.tick_params(
                    axis="x", which="both", bottom=False, top=False, labelbottom=False
                )
            if j == 0:
                ax.set_ylabel(labels[i + 1])
            else:
                ax.tick_params(
                    axis="y", which="both", left=False, right=False, labelleft=False
                )

    cax = (
        inset_axes(
            axes[0, -1], width="4%", height="90%", loc="center right", borderpad=-2.5
        )
        if n > 1
        else fig.add_axes([1.02, 0.2, 0.02, 0.7])
    )
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=resolved_norm), cax=cax)
    cbar.set_label(label_y)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    return fig, axes
