# Rank-Payoff Heterogeneity Analysis

## Question

Determine whether an agent's parameter rank within its population has a
consistent association with its payoff. The aim is not to estimate one pooled
rank effect, because the direction and shape of the relationship may differ
across population configurations.

This is a descriptive analysis of relative position. It should not be
interpreted as a causal rank effect because agents do not observe parameter
ranks directly, and rank is related to the absolute parameter value and the
surrounding population.

## Data Preparation

Perform the analysis separately for each parameter and population size.

For every simulated population:

1. Average each agent's terminal payoff across episodes. Treat the resulting
   agent-level mean as one observation; do not treat episodes sharing the same
   sampled population as independent populations.
2. Express each agent's payoff relative to its peers:

   $$
   \Delta y_{ig}
   = y_{ig} - \operatorname{mean}_{j \ne i}(y_{jg}),
   $$

   where $i$ identifies the agent and $g$ identifies the population.
3. Rank the sampled parameter values within the population. Use average ranks
   for tied values, because clipping at parameter boundaries can produce
   genuine ties.
4. Convert ranks to within-population percentiles and classify agents into
   bottom, middle, and top thirds. Record the number of agents in each group,
   since boundary ties can make the groups unequal in size.
5. Calculate the mean relative payoff in each rank group:

   $$
   L_g = E[\Delta y \mid \text{bottom third}],
   \qquad
   M_g = E[\Delta y \mid \text{middle third}],
   \qquad
   H_g = E[\Delta y \mid \text{top third}].
   $$

## Population-Level Contrasts

Calculate two contrasts for every population.

### Direction

$$
D_g = H_g - L_g.
$$

- $D_g > 0$: high-ranked agents have a payoff advantage.
- $D_g < 0$: low-ranked agents have a payoff advantage.
- $D_g \approx 0$: there is no clear top-versus-bottom difference.

### Curvature

$$
C_g = M_g - \frac{H_g + L_g}{2}.
$$

- $C_g > 0$: middle-ranked agents have an advantage and the extremes perform
  relatively poorly.
- $C_g < 0$: extreme-ranked agents have an advantage over middle-ranked
  agents.
- $C_g \approx 0$: there is no clear middle-versus-extremes difference.

The curvature contrast is necessary because a low payoff at both high and low
parameter values can produce a top-minus-bottom contrast near zero even when a
strong nonlinear relationship is present.

## Primary Figure

Plot one point per population with:

- x-axis: directional contrast $D_g$;
- y-axis: curvature contrast $C_g$;
- horizontal and vertical reference lines at zero;
- separate panels for each parameter and population size.

Use a hexbin or two-dimensional count plot if there are too many populations
for a readable scatter plot. Do not apply Gaussian smoothing to this figure.

The quadrants have direct interpretations:

| Region | Interpretation |
| --- | --- |
| $D>0$, $C>0$ | High ranks outperform low ranks, with an additional middle-rank advantage |
| $D>0$, $C<0$ | High ranks outperform low ranks, and the extremes outperform the middle |
| $D<0$, $C>0$ | Low ranks outperform high ranks, with an additional middle-rank advantage |
| $D<0$, $C<0$ | Low ranks outperform high ranks, and the extremes outperform the middle |

## Numerical Summary

For every parameter and population size, report:

- the median and interquartile range of $D_g$;
- the median and interquartile range of $C_g$;
- the proportion of populations with $D_g>0$;
- the proportion of populations with $C_g>0$;
- population-level bootstrap intervals for the median contrasts or sign
  proportions.

Resample complete populations in the bootstrap. Do not resample agents as if
they came from independent populations.

## Interpretation

If both contrasts are tightly concentrated around zero, conclude that rank has
little practically meaningful association with payoff.

If the contrasts are broadly distributed across quadrants, conclude that
rank-payoff relationships exist but differ in direction and shape across
population contexts. A pooled rank coefficient would conceal this
heterogeneity.

The analysis does not establish that rank itself causes payoff differences.
Absolute parameter values and population composition remain possible
explanations. The intended conclusion is therefore about the consistency, or
lack of consistency, of within-population relative-position associations.
