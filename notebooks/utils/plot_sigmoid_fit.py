from scipy.special import expit
import numpy as np
import matplotlib.pyplot as plt


def plot_sigmoid_fit(
    raw,
    A,
    k,
    t0,
    t=None,
    *,
    ax=None,
    data_label="data",
    fit_label="fit",
    data_color=None,
    fit_color=None,
    fit_linestyle="-",
    show_t0=True,
    show_legend=True,
):
    def zero_start_sigmoid_np(t, A, k, t0):
        s = expit(k * (t - t0))
        s0 = expit(k * (t[0] - t0))
        return A * (s - s0) / (1.0 - s0 + 1e-8)

    t = np.arange(raw.shape[0], dtype=np.float32) if t is None else np.asarray(t)
    pred = zero_start_sigmoid_np(t, A, k, t0)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, raw, label=data_label, color=data_color)
    ax.plot(t, pred, label=fit_label, color=fit_color, linestyle=fit_linestyle)
    if show_t0:
        ax.axvline(t0, color="gray", linestyle="--", alpha=0.5)
    if show_legend:
        ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    return ax
