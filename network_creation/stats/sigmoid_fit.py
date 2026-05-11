from __future__ import annotations

from collections.abc import Sequence
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from numpy.typing import ArrayLike

jax.config.update("jax_platforms", "cpu")


@partial(
    jax.jit,
    static_argnames=(
        "steps",
        "lr",
        "min_k",
        "max_k",
        "huber_delta",
        "min_points",
        "min_amp",
        "min_r2",
        "max_nrmse",
    ),
)
def _fit_zero_start_sigmoids_core(
    Y: jax.Array,
    t_values: jax.Array,
    *,
    steps: int,
    lr: float,
    min_k: float,
    max_k: float,
    huber_delta: float,
    min_points: int,
    min_amp: float,
    min_r2: float,
    max_nrmse: float,
):
    _, T = Y.shape
    t_grid = t_values[None, :]
    t_min = t_values[0]
    t_span = jnp.maximum(t_values[-1] - t_values[0], 1.0)
    min_abs_k = abs(float(min_k))
    max_abs_k = abs(float(max_k))

    obs = jnp.isfinite(Y)
    Y0 = jnp.where(obs, Y, 0.0)
    row_indices = jnp.arange(Y.shape[0])

    eps = jnp.asarray(1e-6, dtype=Y.dtype)

    def logit(x):
        x = jnp.clip(x, eps, 1.0 - eps)
        return jnp.log(x) - jnp.log1p(-x)

    observed_indices = jnp.where(obs, jnp.arange(T)[None, :], -1)
    last_observed_index = jnp.max(observed_indices, axis=1)
    last_observed_index = jnp.maximum(last_observed_index, 0)
    last_observed = Y0[row_indices, last_observed_index]

    peak_index = jnp.argmax(jnp.where(obs, jnp.abs(Y0), -jnp.inf), axis=1)
    peak_observed = Y0[row_indices, peak_index]
    A0 = jnp.where(
        jnp.abs(last_observed) >= 0.5 * jnp.abs(peak_observed),
        last_observed,
        peak_observed,
    )
    A0 = jnp.where(
        jnp.abs(A0) >= min_amp,
        A0,
        jnp.where(A0 < 0.0, -min_amp, min_amp),
    )

    half = 0.5 * A0[:, None]
    dist_to_half = jnp.where(obs, jnp.abs(Y0 - half), jnp.inf)
    t00 = t_values[jnp.argmin(dist_to_half, axis=1)]

    if T > 1:
        dy = jnp.diff(Y0, axis=1)
        dt = jnp.diff(t_values)[None, :]
        valid_slope = obs[:, 1:] & obs[:, :-1]
        slope = jnp.where(valid_slope, dy / dt, 0.0)
        max_abs_slope_index = jnp.argmax(jnp.abs(slope), axis=1)
        signed_slope = slope[row_indices, max_abs_slope_index]
    else:
        signed_slope = jnp.zeros(Y.shape[0], dtype=Y.dtype)

    k0 = 4.0 * signed_slope / jnp.where(jnp.abs(A0) >= min_amp, A0, min_amp)
    k0 = jnp.clip(k0, -max_abs_k * 0.5, max_abs_k * 0.5)
    k0 = jnp.where(
        jnp.abs(k0) >= min_abs_k * 10.0,
        k0,
        jnp.where(k0 < 0.0, -min_abs_k * 10.0, min_abs_k * 10.0),
    )

    raw = {
        "A": A0,
        "k": jnp.arctanh(jnp.clip(k0 / max_abs_k, -0.999, 0.999)),
        "t0": logit((t00 - t_min) / t_span),
    }

    def unpack(raw):
        A = raw["A"][:, None]
        k = max_abs_k * jnp.tanh(raw["k"])[:, None]
        t0 = t_min + t_span * jax.nn.sigmoid(raw["t0"])[:, None]
        return A, k, t0

    def predict(raw):
        A, k, t0 = unpack(raw)
        s = jax.nn.sigmoid(k * (t_grid - t0))
        s_start = jax.nn.sigmoid(k * (t_min - t0))
        return A * (s - s_start) / (1.0 - s_start + eps)

    def loss_fn(raw):
        pred = predict(raw)
        r = pred - Y0

        abs_r = jnp.abs(r)
        huber = jnp.where(
            abs_r <= huber_delta,
            0.5 * r**2,
            huber_delta * (abs_r - 0.5 * huber_delta),
        )

        huber = jnp.where(obs, huber, 0.0)

        count = jnp.maximum(jnp.sum(obs, axis=1), 1)
        row_loss = jnp.sum(huber, axis=1) / count

        valid_row = count >= min_points
        return jnp.sum(row_loss * valid_row) / jnp.maximum(jnp.sum(valid_row), 1)

    opt = optax.adam(lr)
    opt_state = opt.init(raw)

    def body_fun(_, carry):
        raw, opt_state = carry
        _, grads = jax.value_and_grad(loss_fn)(raw)
        updates, opt_state = opt.update(grads, opt_state, raw)
        raw = optax.apply_updates(raw, updates)
        return raw, opt_state

    raw, opt_state = jax.lax.fori_loop(
        0,
        steps,
        body_fun,
        (raw, opt_state),
    )

    pred = predict(raw)
    A, k, t0 = unpack(raw)
    params = jnp.concatenate([A, k, t0], axis=1)

    count = jnp.maximum(jnp.sum(obs, axis=1), 1)

    residual = jnp.where(obs, pred - Y0, 0.0)
    mse = jnp.sum(residual**2, axis=1) / count
    rmse = jnp.sqrt(mse)

    y_mean = jnp.sum(jnp.where(obs, Y0, 0.0), axis=1) / count
    ss_res = jnp.sum(jnp.where(obs, (pred - Y0) ** 2, 0.0), axis=1)
    ss_tot = jnp.sum(jnp.where(obs, (Y0 - y_mean[:, None]) ** 2, 0.0), axis=1)

    r2 = 1.0 - ss_res / jnp.maximum(ss_tot, eps)
    nrmse = rmse / jnp.maximum(jnp.abs(params[:, 0]), min_amp)

    amp = jnp.max(jnp.where(obs, Y0, -jnp.inf), axis=1) - jnp.min(
        jnp.where(obs, Y0, jnp.inf), axis=1
    )

    good = (
        (count >= min_points) & (amp >= min_amp) & (r2 >= min_r2) & (nrmse <= max_nrmse)
    )

    return params, good


def fit_zero_start_sigmoids(
    Y: ArrayLike,
    *,
    t: Sequence[float] | ArrayLike,
    steps: int = 2000,
    lr: float = 5e-2,
    min_k: float = 1e-4,
    max_k: float = 10.0,
    huber_delta: float = 0.03,
    min_points: int = 5,
    min_amp: float = 0.02,
    min_r2: float = 0.80,
    max_nrmse: float = 0.20,
):
    Y = jnp.asarray(Y, dtype=jnp.float32)
    if Y.ndim != 2:
        raise ValueError("Y must have shape (series, time)")
    if max_k == 0.0:
        raise ValueError("max_k must be non-zero")

    _, T = Y.shape
    t_values_np = _time_values(t, int(T))
    t_values = jnp.asarray(t_values_np, dtype=jnp.float32)

    return _fit_zero_start_sigmoids_core(
        Y,
        t_values,
        steps=steps,
        lr=lr,
        min_k=min_k,
        max_k=max_k,
        huber_delta=huber_delta,
        min_points=min_points,
        min_amp=min_amp,
        min_r2=min_r2,
        max_nrmse=max_nrmse,
    )


def _time_values(t: Sequence[float] | ArrayLike, n_steps: int) -> np.ndarray:
    if n_steps <= 0:
        raise ValueError("Y must include at least one time point")

    values = np.asarray(t, dtype=np.float32)
    if values.shape != (n_steps,):
        raise ValueError(f"t must have shape ({n_steps},)")
    if not np.all(np.isfinite(values)):
        raise ValueError("t must contain only finite values")
    if n_steps > 1 and not np.all(np.diff(values) > 0):
        raise ValueError("t must be strictly increasing")
    return values

__all__ = ["fit_zero_start_sigmoids"]
