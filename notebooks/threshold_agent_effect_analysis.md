# Threshold-Agent Effect: Analysis Checklist

This checklist tracks the questions investigated in
`threshold_agent_effect.ipynb`. The treatment replaces one best-response agent
with one threshold-link agent and is compared with the paired all-best-response
baseline for networks of 10, 50, and 100 agents.

## 1. Aggregate-Welfare Level and Dynamics

**Tool:** A descriptive Bayesian quadratic location-scale model fitted with
NumPyro to the paired welfare differences. NumPyro will need to be added as a
project dependency before implementation.

For network size $n$, parameter row $j$, and normalized time $x=t/40$, start
with the following model:

$$
\Delta W_{nj}(x) \sim \operatorname{Normal}\left(
    \beta_{1n}x + \beta_{2n}x^2,\; x\sigma_n
\right).
$$

The intercept is fixed at zero because each paired run has the same initial
network. Treat the observations at $x=0$ as deterministic and exclude them from
the likelihood, where a zero Normal scale would be invalid. A quadratic is the
simplest mean curve that can represent one change in direction. The scale
parameter describes variation across the fixed parameter rows, and multiplying
it by $x$ respects the zero initial effect. Fit a separate set of three
parameters $(\beta_{1n},\beta_{2n},\sigma_n)$ for each network size with weakly
regularizing priors.

**Goal:** Jointly estimate the average welfare-effect trajectory, the direction
and magnitude of the endpoint effect, the amount of endpoint heterogeneity,
and whether and when the average effect reverses.

The posterior summary should report the following derived quantities for each
network size:

- endpoint mean effect, $\beta_{1n}+\beta_{2n}$;
- endpoint effect distribution and probability that the effect is positive;
- average effect over the full horizon, $\beta_{1n}/2+\beta_{2n}/3$;
- turning time, $40[-\beta_{1n}/(2\beta_{2n})]$, when it lies in the observed
  interval; and
- nonzero crossing time, $40(-\beta_{1n}/\beta_{2n})$, when it lies in the
  observed interval.

Use posterior predictive checks to compare the fitted mean and spread with the
empirical trajectories and endpoint distribution. Keep this quadratic Normal
model if it is adequate; only move to a spline or heavier-tailed likelihood if
those checks expose a material failure. Posterior intervals are descriptive
model summaries of the fixed Sobol design, not sampling confidence intervals.

## 2. Parameter-Space Heterogeneity

**Tool:** Binned conditional means, marginal-effect profiles, and interaction
heatmaps.

**Goal:** Determine how link threshold, visibility, outside optimism, and
exploration change the welfare effect.

## 3. Nonlinearities and Interactions

**Tool:** A held-out histogram gradient-boosting surrogate, compared with a
threshold-only linear regression, plus permutation importance.

**Goal:** Assess whether nonlinear relationships and parameter interactions
explain the effect better than a single threshold slope.

## 4. Network-Structure Mechanisms

**Tool:** Paired changes in mean indegree and central point of dominance, with
Spearman correlations against the welfare effect.

**Goal:** Investigate whether welfare changes are associated more strongly with
link abundance or with network centralization.

## 5. Distribution of Benefits and Costs

**Tool:** Policy-specific payoff comparisons for the threshold-link agent,
treated best-response peers, and the all-best-response baseline.

**Goal:** Establish which type of agent gains or loses after the replacement.

## 6. Private Cost Versus Social Benefit

**Tool:** Threshold-binned profiles comparing aggregate welfare effects with
policy-specific payoffs.

**Goal:** Determine whether aggregate gains in larger networks coexist with a
private payoff cost for the threshold-link agent.

## 7. Reporting Limitations

**Tool:** Experimental-design and artifact-metadata review.

**Goal:** Document the limits of interpretation, including the fixed Sobol QMC
design, unavailable episode-level variance, descriptive surrogate analysis,
and the inapplicability of conventional Sobol sensitivity indices.
