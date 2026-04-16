import numpy as np

_ARGMAX_P_GRID = np.linspace(0.0, 1.0, 1000, dtype=float)


def _expected_aggregate_payoff_for_p(
    N: int,
    v_k: float,
    p: float,
    delta: float,
    epsilon: float = 1e-6,
) -> float:
    normalization = N * (2 * N - 1)
    pair_count = N * (N - 1)

    if p == 0.0:
        return (v_k + N - 1) / (2 * N - 1)

    if p == 1.0:
        expected_benefit = N + pair_count * delta
        expected_edges = pair_count
        return (v_k * expected_benefit - expected_edges + pair_count) / normalization

    q = 1.0 - p
    expected_edges = p * pair_count
    shell_tail_prev = 1.0
    shell_tail_curr = q
    expected_decay_sum = 0.0

    for distance in range(1, N):
        shell_probability = shell_tail_prev - shell_tail_curr
        if shell_probability < epsilon:
            break

        expected_decay_sum += shell_probability * np.power(delta, distance)
        shell_tail_prev, shell_tail_curr = (
            shell_tail_curr,
            shell_tail_curr * np.power(q, (N - 1) * shell_probability),
        )

    expected_benefit = N + pair_count * expected_decay_sum
    return (v_k * expected_benefit - expected_edges + pair_count) / normalization


def expected_aggregate_payoff(
    N: int,
    v_k: float,
    p: float | None,
    delta: float,
    epsilon: float = 1e-6,
) -> float | tuple[float, float]:
    if p is None:
        candidate_ps = _ARGMAX_P_GRID
        payoffs = np.array(
            [
                _expected_aggregate_payoff_for_p(N, v_k, candidate_p, delta, epsilon)
                for candidate_p in candidate_ps
            ],
            dtype=float,
        )
        best_idx = int(np.argmax(payoffs))
        return float(payoffs[best_idx]), float(candidate_ps[best_idx])

    return _expected_aggregate_payoff_for_p(N, v_k, p, delta, epsilon)
