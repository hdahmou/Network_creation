import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import binned_statistic_2d
from scipy.ndimage import gaussian_filter
from matplotlib.axes import Axes


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
):
    if method not in {"points", "gaussian"}:
        raise ValueError(f"{method} is not a valid method for plotting")

    if bounds is None:
        bounds = [x.min(), x.max(), y.min(), y.max()]

    xmin, xmax, ymin, ymax = bounds

    if method == "points":
        ax.scatter(
            x,
            y,
            c=z,
            cmap=cmap,
            norm=norm,
            s=point_size,
            alpha=alpha,
            linewidths=0,
        )
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        return ax

    z_sum, _, _, _ = binned_statistic_2d(
        x,
        y,
        z,
        statistic="sum",
        bins=resolution,
        range=[[xmin, xmax], [ymin, ymax]],
    )

    z_count, _, _, _ = binned_statistic_2d(
        x,
        y,
        z,
        statistic="count",
        bins=resolution,
        range=[[xmin, xmax], [ymin, ymax]],
    )

    z_sum_s = gaussian_filter(z_sum, sigma=sigma, mode="nearest")
    z_count_s = gaussian_filter(z_count, sigma=sigma, mode="nearest")
    zi = z_sum_s / np.maximum(z_count_s, 1e-12)

    ax.imshow(
        zi.T,
        cmap=cmap,
        norm=norm,
        origin="lower",
        extent=bounds,
        interpolation="nearest",
        aspect="auto",
    )
    return ax
def plot_pair(
    X: np.ndarray,
    y: np.ndarray,
    labels=None,
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
    if len(labels) != d:
        raise ValueError(f"`labels` must have length {d}, got {len(labels)}")

    if method not in {"points", "gaussian"}:
        raise ValueError(f"{method} is not a valid method for plotting")

    keep = np.var(X, axis=0) > atol
    if not np.any(keep):
        raise ValueError("All X dimensions are constant; nothing to plot.")

    X = X[:, keep]
    labels = [lab for lab, k in zip(labels, keep) if k]

    if X.shape[1] < 2:
        raise ValueError("Need at least two varying X dimensions to build pair plots.")

    n = X.shape[1] - 1
    resolved_norm = norm
    if resolved_norm is None:
        resolved_norm = Normalize(
            vmin=float(np.min(y)),
            vmax=float(np.max(y)),
        )

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

            xj = X[:, j]
            yi = X[:, i + 1]
            bounds = [xj.min(), xj.max(), yi.min(), yi.max()]

            plot_2d_descriptor(
                x=xj,
                y=yi,
                z=y,
                method=method,
                resolution=resolution,
                ax=ax,
                norm=resolved_norm,
                bounds=bounds,
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
                    axis="x",
                    which="both",
                    bottom=False,
                    top=False,
                    labelbottom=False,
                )

            if j == 0:
                ax.set_ylabel(labels[i + 1])
            else:
                ax.tick_params(
                    axis="y",
                    which="both",
                    left=False,
                    right=False,
                    labelleft=False,
                )

    mappable = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=resolved_norm,
    )

    if n > 1:
        colorbar_axis = inset_axes(
            axes[0, -1],
            width="4%",
            height="90%",
            loc="center right",
            borderpad=-2.5,
        )
    else:
        colorbar_axis = fig.add_axes([1.02, 0.2, 0.02, 0.7])

    cbar = fig.colorbar(mappable, cax=colorbar_axis)
    cbar.set_label(label_y)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    return fig, axes
