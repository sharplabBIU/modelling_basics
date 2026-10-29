# Syllabus: lesson-by-lesson outline

Each entry lists the sections, the exercises (type in brackets; *code* exercises
name the reference function, usually in `rlcourse.models` / `rlcourse.fitting`)
and the sources to cite.  Lesson 3 (`lessons_src/lesson03_rescorla_wagner.txt`)
is the exemplar for tone, structure and exercise design.

Exercise ids are `N.k`; aim for 8–10 per lesson with all three types (≥ 2
conceptual [choice or free text], ≥ 2 maths [numeric], ≥ 3 coding).

---

## Lesson 1 — Thinking like a modeller: tasks, agents and simulation

*Goal:* make the student comfortable with the idea that a model is a recipe that generates behaviour, and with simulating tasks and agents before anything is fitted.

1. **What is a model?** Ingredients: a task (generative process), an agent (hypothesis about the computations), parameters, and noise. The modelling cycle of Wilson & Collins (2019): design → simulate → fit → recover → compare → validate; where each lesson of the course sits.
2. **Simulating a task.** `tasks.Bandit`; Bernoulli outcomes; the law of large numbers (running mean converges, plot several seeds). [code] `running_mean(x)` → `models.running_mean`. [numeric] expected reward of a random chooser on a (0.7, 0.3) bandit = 0.5. [numeric] probability that a 70%-option fails three times in a row = 0.027.
3. **Simulating simple agents.** A random agent and a win-stay/lose-shift (WSLS) agent; the observe → choose → outcome loop. [code] `simulate_random_agent(probs, n_trials, rng)` returning `{"choices", "rewards"}` (check: lengths, values in range, rewards 0/1, reward rate near the mean of probs). [code] `simulate_wsls(probs, n_trials, rng)` (deterministic rule: repeat after a reward, switch after no reward; check by *consistency* of choices with the rule, as in lesson 3 ex. 3.8).
4. **Describing behaviour without a model.** Learning curves, reward rate, and stay probabilities as model-agnostic summaries. [code] `stay_probabilities_bandit(choices, rewards)` → `(P(stay | reward), P(stay | no reward))` (deterministic).
5. **Reproducibility and seeds.** `np.random.default_rng`, why a seed, why pass `rng` around. [choice] why simulate before fitting (identify parameters' behavioural signatures / detect bugs / check recoverability).
6. **The modelling cycle in practice.** [free text] what can go wrong if you fit a model you never simulated (concepts: unrecoverable parameters, model mis-specification, bugs, no behavioural signature).

*Cite:* Wilson & Collins 2019; Palminteri et al. 2017; Sutton & Barto 2018 ch. 2; Daw 2011.

## Lesson 2 — Probability and Bayes for modellers

*Goal:* every probability concept the course needs, built on a coin/bandit example: distributions, likelihood, log-likelihood, MLE, Bayes' rule, conjugate updating, posterior summaries, posterior predictive.

1. **Distributions you need.** Bernoulli, binomial, Gaussian, Beta (plot the Beta family); random variables vs parameters.
2. **Likelihood.** Fix the data, vary the parameter: plot L(p) for 7 successes in 10. [code] `bernoulli_log_likelihood(p, outcomes)` → `models.bernoulli_log_likelihood`. [numeric] log-likelihood of (1,1,1,0) at p = 0.75 = 3 ln 0.75 + ln 0.25 ≈ −2.249. [numeric] the MLE for 7/10 = 0.7. [choice] why work with logs (products of tiny numbers underflow; sums are easier to optimise).
3. **Bayes' rule on a grid.** posterior ∝ prior × likelihood; [code] `grid_posterior(prior, likelihood)` → `models.grid_posterior`; the effect of the prior with little vs much data (plots).
4. **Conjugacy: Beta–Bernoulli.** [code] `beta_posterior(a0, b0, outcomes)` → `models.beta_posterior`; posterior mean a/(a+b), mode, 95% credible interval with `scipy.stats.beta`. [numeric] posterior mean after a Beta(1,1) prior and 7 successes in 10 = 8/12 ≈ 0.667. [numeric] P(p > 0.5 | Beta(8, 4)) via `scipy.stats.beta.sf` (compute the exact value when writing the exercise).
5. **Sequential updating is learning.** Update the posterior outcome by outcome (animation-like plot); the posterior mean moves by (r − m)/(a+b+1): a learning rate that *shrinks* — the bridge to lesson 3 (constant α) and lesson 11 (Bayesian learners).
6. **Posterior predictive.** P(next outcome = 1) = a/(a+b); prediction vs estimation.
7. **Likelihood vs posterior.** [choice] which object is a distribution over the parameter; [free text] explain the difference between P(data | θ) and P(θ | data) (concepts: likelihood is a function of θ not a distribution; posterior needs a prior; normalisation).

*Cite:* Lee & Wagenmakers 2013; McElreath 2020; Gelman et al. 2013; Daw 2011.

## Lesson 3 — Model-free learning I (written: the exemplar)

## Lesson 4 — Model-free learning II: temporal-difference learning, control and dopamine

1. **From bandits to sequences.** MDPs (states, actions, transitions, rewards), returns, discounting γ; `tasks.MDP`, `tasks.random_walk_chain`. [numeric] discounted return of rewards (0, 0, 1) with γ = 0.9 = 0.81.
2. **Value functions and the Bellman equation.** V^π(s) = E[r + γ V^π(s')]; solve a 3-state chain by hand. [numeric] a small Bellman computation.
3. **TD(0).** Bootstrapping; [code] `td0_update(V, s, r, s_next, alpha, gamma, done)` → `models.td0_update`; reproduce Sutton & Barto Example 6.2 (values converge to 1/6 … 5/6) with `models.run_td_prediction`.
4. **Dopamine and the TD error.** Schultz, Dayan & Montague (1997): before learning the response is at the reward, after learning at the cue, omission gives a dip. [code] `td_pavlovian(reward_sequences, alpha, gamma)` → `models.td_pavlovian` (state the convention: δ_t = r_t + γV[t] − V[t−1], V before the cue = 0, update V[t−1]; see the reference docstring) and plot the three classic panels with `tasks.PavlovianTrace`. [choice] what the cue response represents after learning.
5. **Control: SARSA and Q-learning.** [code] `q_learning_update(Q, s, a, r, s_next, alpha, gamma, done)` → `models.q_learning_update`; [code] `sarsa_update(...)` → `models.sarsa_update`; demonstration on `tasks.gridworld` with `models.run_td_control`; on-policy vs off-policy. [choice] which algorithm learns the value of the policy it follows.
6. **Eligibility traces.** TD(λ) with accumulating traces; [code] `td_lambda_episode(V, states, rewards, alpha, gamma, lam)` → `models.td_lambda_episode`; the λ of the two-step model (lesson 9) is this λ. [numeric] trace of the start state two steps later with γ = 0.9, λ = 0.5 = 0.2025.
7. **Actor–critic.** [code] `actor_critic_update(V, theta, s, a, r, s_next, alpha_critic, alpha_actor, gamma, done)` → `models.actor_critic_update`; the basal-ganglia mapping (Joel, Niv & Ruppin 2002). [free text] how TD solves temporal credit assignment (concepts: bootstrapping, propagating value backwards, prediction error at the cue).

*Cite:* Sutton 1988; Sutton & Barto 2018 ch. 3, 6, 12, 13; Watkins & Dayan 1992; Montague, Dayan & Sejnowski 1996; Schultz, Dayan & Montague 1997; Joel, Niv & Ruppin 2002; Niv 2009.

## Lesson 5 — Fitting models to behaviour: maximum likelihood

1. **Inverting the simulation.** From parameters → choices to choices → parameters. The likelihood of a choice sequence ∏_t P(c_t | history_t, θ): the model is *replayed* on the participant's actual choices and rewards; it never samples its own choices. [free text] why we cannot fit by simulating choices and comparing them with the data (concepts: randomness of simulated choices, likelihood uses the observed history, probabilities not samples).
2. **The negative log-likelihood of the RW model.** [code] `nll_rw(params, choices, rewards, n_arms=2, q0=0.5)` → `models.nll_rw`; bug variants: update before computing the probability, forgetting the minus sign, summing probabilities. [numeric] by hand: q0 = 0.5, α = 0.5, β = 2, choices (0, 0), rewards (1, 1): nll = −ln 0.5 − ln σ(2·0.25) ≈ 0.693 + 0.474 = 1.167.
3. **The likelihood surface.** `fitting.grid_nll` → contour plot over (α, β); the maximum; flat directions. [choice] what a flat likelihood means for parameter estimation.
4. **Optimisation.** `scipy.optimize.minimize`; local minima and random restarts; parameter transforms. [code] `to_constrained(x, kinds)` → `fitting.to_constrained`. [code] `fit_rw(choices, rewards, n_starts, rng)` returning `(alpha, beta, nll)` (compare: the student's nll must be ≤ the reference's + 1e-3 on the same data; params inside their ranges).
5. **How good is the estimate?** Fit simulated subjects with 50, 200, 1000 trials; plot estimate vs truth. [choice] why multiple restarts.
6. **Fitting many subjects.** Loop over subjects, collect a DataFrame; compare with a chance model (nll = n ln 2); pseudo-R² = 1 − nll/nll_chance. [numeric] pseudo-R² for nll = 100, n = 200 = 1 − 100/(200 ln 2) ≈ 0.279.
7. **Pitfalls.** Estimates at the boundary, β → ∞ for near-deterministic data, identifiability (preview of lesson 6).

*Cite:* Daw 2011; Wilson & Collins 2019; Myung, I. J. (2003). Tutorial on maximum likelihood estimation. *Journal of Mathematical Psychology, 47*(1), 90–100. https://doi.org/10.1016/S0022-2496(02)00028-7

## Lesson 6 — Trust but verify: parameter recovery, identifiability and model recovery

1. **Why fits can lie.** Rules 4–6 of Wilson & Collins; Palminteri's falsification argument.
2. **Parameter recovery.** Simulate N agents with known (α, β) → fit → scatter and correlation (`plotting.plot_parameter_recovery`). [code] `simulate_and_fit(alpha, beta, n_trials, rng, n_starts)` using `models.simulate_rw_bandit` and `fitting.fit_mle` (deterministic given the rng; compare with a loose tolerance). [code] `recovery_summary(true, recovered)` → per-parameter correlation and mean bias (deterministic).
3. **Identifiability and trade-offs.** Correlation between recovered α and β; effect of trial number and of task design (0.7/0.3 vs 0.55/0.45). [choice] which design recovers better and why. [numeric] e.g. correlation computed from a given small table.
4. **Model recovery.** Simulate from {bias, WSLS, RW, RW+stickiness} (`models.simulate_model`), fit all with `fitting.fit_mle` and `models.MODEL_SPECS`, pick the winner by BIC (define BIC briefly; lesson 7 explains). [code] `confusion_matrix(sim_labels, fit_labels, n_models)` (deterministic; rows = simulated). Confusion vs inversion matrix (`plotting.plot_confusion`). [choice] interpretation of a row vs a column.
5. **Posterior predictive check (preview).** Simulate from the fitted parameters and compare learning curves with the data.
6. [free text] what to do when recovery is poor (concepts: more trials, redesign task, simplify/fix parameters, hierarchical priors).

*Cite:* Wilson & Collins 2019; Palminteri et al. 2017; Daw 2011.

## Lesson 7 — Model comparison

1. **The problem.** Nested models: more parameters never fit worse (demonstrate).
2. **Information criteria.** [code] `aic(nll, k)`, `bic(nll, k, n)` → `fitting.aic`, `fitting.bic`. [numeric] BIC for nll = 100, k = 3, n = 200 ≈ 215.9; ΔBIC and Bayes-factor heuristics (Kass & Raftery 1995). [choice] why BIC penalises more than AIC for large n.
3. **Cross-validation for sequential models.** Score only held-out trials while replaying the whole sequence. [code] `nll_rw_on_trials(params, choices, rewards, test_mask)` (deterministic). Leave-one-block-out.
4. **Group-level comparison.** Fixed effects (summed evidence) vs random effects (Stephan et al. 2009). [code] `bms_alpha(log_evidence, alpha0=1.0)` implementing the variational iteration and returning the Dirichlet `alpha` (compare with `fitting.bms(L)["alpha"]`). Exceedance probabilities by Dirichlet sampling; protected exceedance probabilities and the Bayes omnibus risk (Rigoux et al. 2014) via `fitting.bms`. [numeric] expected frequency r_k = α_k / Σα for a given α. [choice] what an exceedance probability means.
5. **Absolute fit: posterior predictive checks.** Simulate from fitted parameters; compare summary statistics; a model can win and still be wrong (Palminteri et al. 2017). [free text].
6. **Worked comparison.** RW vs RW-asymmetric vs WSLS vs bias on simulated data; a results table with BIC, PXP.

*Cite:* Akaike 1974; Schwarz 1978; Kass & Raftery 1995 (*JASA, 90*, 773–795, https://doi.org/10.1080/01621459.1995.10476572); Stephan et al. 2009; Rigoux et al. 2014; Vehtari et al. 2017; Palminteri et al. 2017; Wilson & Collins 2019; Piray et al. 2019.

## Lesson 8 — Bayesian parameter estimation and hierarchical models

1. **From point estimates to posteriors.** Uncertainty, regularisation, boundary estimates.
2. **Priors and MAP.** Log posterior = log likelihood + log prior in unconstrained space. [code] `log_posterior_rw(x, choices, rewards, mu, sd)` (uses `models.nll_rw`, `fitting.to_constrained`, `scipy.stats.norm.logpdf`; deterministic). MAP with `fitting.fit_mle(..., log_prior=...)`; shrinkage with few trials.
3. **Laplace approximation.** Gaussian around the MAP; `fitting.laplace_covariance`. [numeric] posterior sd from a Hessian entry (e.g. H = 25 → sd = 0.2).
4. **MCMC by hand.** [code] `metropolis_hastings(log_target, x0, n_samples, proposal_sd, rng)` → `fitting.metropolis_hastings` (compare by statistics of the samples on a known Gaussian target, and acceptance rate within a range). Trace plots, burn-in, acceptance rate, autocorrelation; posterior of (α, β) for one subject vs the MLE. [choice] what a very high acceptance rate indicates.
5. **Hierarchical models.** Group distribution, partial pooling, shrinkage. Empirical-Bayes EM (Huys et al. 2011; Gershman 2016): [code] `em_m_step(x_hat, var_hat)` → `(mu, sd)` (deterministic); `fitting.em_hierarchical` on 30 simulated subjects; recovery MLE vs EM (shrinkage plot). [numeric] shrinkage weight τ²/(τ² + σ²).
6. **Full hierarchical Bayes with PyMC (optional).** Wrap in `try/except ImportError`; set `os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")` *before* importing pymc and sample with `nuts_sampler="numpyro"` when available (fall back to the default sampler); scan-based likelihood following Vieira (2022); ArviZ summaries, R-hat. Keep it small (≤ 20 subjects, ≤ 500 draws).
7. **Tools used in the field.** HBI (Piray et al. 2019), hBayesDM (Ahn et al. 2017), Stan/PyMC. [free text] when hierarchical fitting helps and when it can mislead (concepts: shrinkage, few trials, group differences require the group structure in the model).

*Cite:* Gelman et al. 2013; Lee & Wagenmakers 2013; Huys et al. 2011; Gershman 2016; Piray et al. 2019; Ahn et al. 2017; Vieira 2022 (PyMC example, MIT licence); Daw 2011.

## Lesson 9 — Model-based RL and the two-step task

1. **Habits and goals.** MF caches values; MB plans with a model (transitions T, rewards R). Daw, Niv & Dayan 2005; Dolan & Dayan 2013; Gläscher et al. 2010 (state prediction errors).
2. **Planning with a model.** [code] `value_iteration(P, R, gamma, n_iter)` on `tasks.MDP` (deterministic). [code] `two_step_mb_values(Q2, transitions)` → `models.two_step_mb_values`. [numeric] Q_MB for Q2 = [[0.8, 0.2], [0.4, 0.6]] and T = [[0.7, 0.3], [0.3, 0.7]]: Q_MB(a = 0) = 0.74.
3. **The two-step task.** Structure, drifting rewards, `tasks.TwoStepTask`; the stay-probability logic (MF: main effect of reward; MB: reward × transition interaction) with a figure of the two idealised patterns.
4. **The hybrid model of Daw et al. (2011).** All equations: SARSA(λ) at both stages, Q_MB from the known transition matrix, Q_net = w Q_MB + (1 − w) Q_MF, softmax with β1 (+ perseveration p) and β2. [code] `two_step_simulate(params, task, rng)` → `models.two_step_simulate` (compare by structure + behavioural signature: with w = 1 the interaction pattern, with w = 0 the main-effect pattern). [code] `stay_probabilities(data)` → `models.stay_probabilities`. Plots for w ∈ {0, 0.5, 1}.
5. **Likelihood and fitting.** [code] `two_step_nll(params, data)` → `models.two_step_nll`; fit one simulated subject with `fitting.fit_mle` and `models.TWO_STEP_PARAM_KINDS`; brief recovery of w. [choice] which parameter indexes model-based control. The reparameterisation with separate β_MB, β_MF used by Gillan et al. (2016).
6. **Caveats.** Akam et al. 2015 (confounds), Kool et al. 2016 (MB does not pay in the original task), Otto et al. 2013 (working memory), Keramati et al. 2016 (planning depth), arbitration (Lee et al. 2014). [free text] why rare-transition trials are diagnostic.

*Cite:* Daw et al. 2005, 2011; Gläscher et al. 2010; Dolan & Dayan 2013; Gillan et al. 2016; Akam et al. 2015; Kool et al. 2016; Otto et al. 2013; Lee et al. 2014; Keramati et al. 2016; Sutton 1990 (Dyna).

## Lesson 10 — The successor representation

1. **Between model-free and model-based.** M(s, s') = expected discounted future occupancy; V = M R (Dayan 1993). [code] `sr_analytic(T, gamma)` → `models.sr_analytic`; visualise with `plotting.plot_matrix` for `tasks.random_walk_chain` and `tasks.three_branch_graph`. [numeric] an SR entry for a tiny chain (e.g. deterministic s0 → s1 terminal: M[0, 1] = γ).
2. **Learning the SR by TD.** [code] `sr_td_update(M, s, s_next, alpha, gamma)` → `models.sr_td_update`; [code] `sr_reward_update(R_hat, s, r, alpha)` → `models.sr_reward_update`; `models.sr_value`; learned vs analytic with `models.run_sr_agent`.
3. **Revaluation experiments.** Phases on `tasks.three_branch_graph`: latent learning (random exploration), reward learning, *reward revaluation* (terminal rewards change, observed directly) and *transition revaluation* (which intermediate state leads where changes, observed only locally). MF (Q-learning), MB (value iteration on learned T and R) and SR agents; the signature: MB adapts to both, SR only to reward revaluation, MF to neither (Momennejad et al. 2017). [code] `mb_values_from_model(T_hat, R_hat, gamma)` (value iteration, deterministic). [choice] which agent fails which revaluation and why. [numeric] V after a reward change computed from a given M.
4. **SR in brains and people.** Stachenfeld et al. 2017 (hippocampus); Momennejad et al. 2017 (humans are partly sensitive to transition revaluation → SR + MB hybrid); Russek et al. 2017 (SR-TD vs SR-Dyna; SR in the two-step task). [free text] in what sense the SR is model-based-like and model-free-like (concepts: cached predictions of states, flexible to reward change, inflexible to transition/policy change).

*Cite:* Dayan 1993; Momennejad et al. 2017; Russek et al. 2017; Gershman 2018; Stachenfeld et al. 2017.

## Lesson 11 — Bayesian learners I: uncertainty-aware learning and exploration

1. **When is a constant learning rate right?** The Bayesian view: the learning rate is the relative uncertainty of the estimate vs the observation.
2. **Beta–Bernoulli learner.** [code] `beta_update(a, b, reward)` → `models.beta_update`; the implied learning rate 1/(a + b + 1). [numeric] learning rate after 9 outcomes from a Beta(1, 1) prior = 1/11. Forgetting: `models.beta_update_forgetful`.
3. **The Kalman filter.** Gaussian belief, process and observation noise; [code] `kalman_update(mu, var, reward, obs_var, process_var)` → `models.kalman_update`; the gain as a learning rate that decays to a steady state > 0 when the world drifts (Daw et al. 2006; the Kalman Rescorla–Wagner of Gershman 2015). [numeric] gain on the first update with prior variance 1, observation variance 1, no process noise = 0.5.
4. **Exploration guided by uncertainty.** `models.kalman_bandit_step`; [code] `ucb_values(mu, var, c)` → `models.ucb_values`; Thompson sampling (`models.thompson_choice`); compare softmax-on-means, UCB and Thompson on `tasks.RestlessBandit`. Directed vs random exploration (Wilson et al. 2014; Gershman 2018). [choice] which strategy is "directed".
5. **Change-points.** [code] `grid_changepoint_filter(outcomes, hazard, grid)` → `models.grid_changepoint_filter`; the effective learning rate jumps after surprising outcomes (Nassar et al. 2010; Behrens et al. 2007 as the ideal observer that also learns the hazard rate).
6. **Pearce–Hall.** [code] `pearce_hall_update(v, assoc, reward, eta, kappa)` → `models.pearce_hall_update`; surprise-modulated learning rates as a heuristic approximation (Yu & Dayan 2005). [free text] why learning rates should be high after change and low in stable periods (concepts: uncertainty of the estimate vs outcome noise, change-points).

*Cite:* Kalman 1960; Daw et al. 2006; Gershman 2015; Gershman 2018; Wilson et al. 2014; Speekenbrink & Konstantinidis 2015; Behrens et al. 2007; Nassar et al. 2010; Pearce & Hall 1980; Yu, A. J., & Dayan, P. (2005). Uncertainty, neuromodulation, and attention. *Neuron, 46*(4), 681–692. https://doi.org/10.1016/j.neuron.2005.04.026

## Lesson 12 — Bayesian learners II: volatility and the volatile Kalman filter

1. **Volatility.** Expected noise vs volatility; the generative model of Behrens et al. (2007); humans raise their learning rate in volatile blocks.
2. **The continuous VKF (Piray & Daw 2020, Eqs. 9–13).** [code] `vkf_continuous(outcomes, lam, v0, sigma2)` → `models.vkf_continuous` (deterministic; compare the returned dict); learning rate rises after change-points.
3. **The binary VKF (Eqs. 14–19).** [code] `vkf_bin(outcomes, lam, v0, omega)` → `models.vkf_bin`; demonstration on `tasks.ReversalTask.stable_volatile`: predictions, learning rates by block. [numeric] learning rate on the first trial = √(ω + v0). [choice] which parameter sets how fast the volatility estimate changes.
4. **Fitting the VKF to choices.** [code] `vkf_bin_nll(params, outcomes, choices)` → `models.vkf_bin_nll`; fit a simulated subject; recovery of λ; VKF vs RW (`models.rw_reversal_nll`) by BIC on a few subjects of `datasets.load("reversal_anxiety")`.
5. **Learning-rate adaptation.** Fit RW separately to the stable and volatile blocks; α_volatile > α_stable (Browning et al. 2015); preview of lesson 14.
6. **The HGF and other volatility models.** Mathys et al. 2011, 2014 (hierarchy of Gaussian random walks); how the VKF relates (a Kalman-filter structure with a volatility update); pointers to toolboxes (TAPAS HGF, pyhgf). [free text] why noise and volatility have opposite effects on the learning rate. [choice] which learner wins in which environment.

*Cite:* Piray & Daw 2020; Behrens et al. 2007; Mathys et al. 2011, 2014; Browning et al. 2015; Gagne et al. 2020; Pulcu & Browning 2019; Nassar et al. 2010.

## Lesson 13 — Distributional reinforcement learning

1. **Beyond the mean.** Why a distribution (risk, variance, skew); the machine-learning motivation (Bellemare et al. 2017) and the neural one (Dabney et al. 2020).
2. **Expectiles.** Definition; τ = 0.5 is the mean. [code] `expectile(samples, tau)` → `models.expectile` (deterministic). [numeric] the τ-expectile of a 50/50 outcome of 0 or 1 equals τ (e.g. 0.8).
3. **Distributional TD with asymmetric learning rates.** [code] `expectile_td_update(values, reward, alpha_pos, alpha_neg)` → `models.expectile_td_update`; `models.taus_from_alphas`; a population of predictors converges to the expectiles of a bimodal reward distribution (plot); reversal points. [numeric] τ for α+ = 0.3, α− = 0.1 = 0.75.
4. **Decoding the distribution.** [code] `expectile_loss(e, samples, tau)` → `models.expectile_loss`; `models.decode_distribution` recovers the reward distribution from (expectile, τ) pairs. [choice] what pattern in dopamine neurons supports a distributional code (diverse asymmetric scaling of positive vs negative RPEs; reversal points ordered by asymmetry).
5. **Quantile TD.** [code] `quantile_td_update(values, reward, alpha, taus)` → `models.quantile_td_update` (Dabney et al. 2018).
6. **Asymmetric learning in humans.** Lefebvre et al. 2017 (α+ > α−), Niv et al. 2012 (risk sensitivity), Pulcu & Browning 2017 (affective bias as adaptation); fit `models.nll_rw_asym` to a few subjects of `datasets.load("bandit_anhedonia")`; the single asymmetric learner converges to an expectile (optimistic/pessimistic value). [free text] relate asymmetric learning rates to the expectile view and to mood.

*Cite:* Bellemare et al. 2017; Dabney et al. 2018, 2020; Lowet et al. 2020; Lefebvre et al. 2017; Niv et al. 2012; Pulcu & Browning 2017; Palminteri & Lebreton 2022; Eldar et al. 2016.

## Lesson 14 — Computational psychiatry: linking parameters to psychopathology

1. **Parameters as phenotypes.** Montague et al. 2012; Huys et al. 2016; Patzelt et al. 2018. The logic and the pitfalls, up front.
2. **Worked example 1: compulsivity and model-based control** (`datasets.load("two_step_compulsivity")`, cf. Gillan et al. 2016). Model-agnostic index: per-subject logistic regression stay ~ reward × transition (statsmodels); [code] `stay_regression_coefficients(data)` → (intercept, reward, transition, interaction) (deterministic). Model-based index: w from the hybrid model (fit a few subjects live; load cached fits for all from `data/two_step_compulsivity_fits.csv`, produced by `scripts/make_fits.py`). Correlate with compulsivity controlling for age and anxiety (OLS); scatter with CI. Compare with the ground truth in `datasets.truth`.
3. **Worked example 2: anxiety and learning-rate adaptation** (`datasets.load("reversal_anxiety")`, cf. Browning et al. 2015). Fit RW per block for every subject (fast); Δα = logit(α_vol) − logit(α_stable) vs trait anxiety.
4. **Worked example 3: anhedonia and positive learning rate** (`datasets.load("bandit_anhedonia")`, cf. Huys et al. 2013; Lefebvre et al. 2017). Fit RW-asym; relate α+ to anhedonia; check the α–β trade-off (Eckstein et al. 2021).
5. **Reliability.** Split-half reliability of fitted parameters; [code] `split_half_reliability(x_half1, x_half2)` → Spearman–Brown corrected r (`fitting.spearman_brown`); [numeric] Spearman–Brown for r = 0.5 = 0.667; attenuation: [numeric] max observable r with reliabilities 0.5 and 0.8 = √0.4 ≈ 0.632 (Brown et al. 2020; Zorowitz & Niv 2023; Hedge et al. 2018).
6. **Power by simulation.** [code] `power_by_simulation(r_true, n, n_sims, rng, alpha=0.05)` (compare within ± 0.1 of the reference).
7. **Pitfalls and better practice.** Identifiability and trade-offs (Katahira 2015; Eckstein et al. 2021); regression on point estimates ignores uncertainty → embed covariates in the hierarchical model (Piray et al. 2019; optional PyMC sketch); group vs dimensional analyses; dysregulation vs deficit (Sharp et al. 2023); time and context (Hitchcock et al. 2022). [choice] on attenuation; [free text] why a significant parameter–symptom correlation is not enough (concepts: recovery, reliability, confounds, alternative models). A checklist.

*Cite:* Montague et al. 2012; Huys et al. 2016; Patzelt et al. 2018; Gillan et al. 2016; Voon et al. 2015; Browning et al. 2015; Gagne et al. 2020; Huys et al. 2013; Lefebvre et al. 2017; Eckstein et al. 2021; Katahira 2015; Brown et al. 2020; Zorowitz & Niv 2023; Hedge et al. 2018; Piray et al. 2019; Sharp et al. 2022, 2023; Mkrtchian et al. 2017; Hitchcock et al. 2022; Collins & Frank 2012.

## Lesson 15 — Capstone project

A guided end-to-end project on `datasets.load("bandit_anhedonia")` (the other two data sets are offered as alternatives at the end). Milestones, each with a checked exercise where possible:

1. Describe the data model-agnostically (learning curves by anhedonia tertile, stay probabilities). [code] a specific summary function (deterministic).
2. Candidate models: RW, RW-asymmetric, RW + stickiness, WSLS, bias (`models.MODEL_SPECS`). [code] `fit_subject_all_models(choices, rewards, rng)` → dict of nll per model (deterministic with the given rng and `n_starts`); fit all subjects (use a cached `data/bandit_anhedonia_fits.csv` from `scripts/make_fits.py` with a `RUN_FULL = False` switch).
3. Model recovery (small: 10 simulations per model). [choice] on reading the confusion matrix.
4. Model comparison: BIC, `fitting.bms` → PXP. [numeric] ΔBIC between two given values.
5. Parameter recovery for the winning model in this exact design.
6. Relate α+ (and α−, β) to anhedonia controlling for age; split-half reliability; attenuation-corrected correlation. [numeric].
7. Posterior predictive check.
8. Write-up: a markdown template (Methods / Results / Limitations) to fill in; [free text] limitations (concepts: simulated data, reliability, identifiability, causality). Then reveal the ground truth and compare.

*Cite:* the lessons' sources as needed; Wilson & Collins 2019 for the workflow.
