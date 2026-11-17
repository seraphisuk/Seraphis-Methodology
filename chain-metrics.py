import numpy as np

EPS = 1e-9


def stationary(P):
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1))])
    pi = np.clip(pi / pi.sum(), 0, None)
    return pi / pi.sum()


def chain_to_metrics(P, punitive=(2, 3), risky=(3, 4), max_lag=50):
    """Map a row-stochastic chain to six behavioural proxies.
    State numbers are 1-indexed. State 3 is both punitive and risky."""
    P = np.asarray(P, float)
    n = len(P)
    pun = np.array([s - 1 for s in punitive])
    rsk = np.array([s - 1 for s in risky])
    safe = np.array([i for i in range(n) if i not in set(pun) | set(rsk)])
    pi = stationary(P)

    # 1. BART choice consistency: 1 - normalised entropy rate  (0 = random, 1 = deterministic)
    row_H = -(P * np.log(P + EPS)).sum(1)
    consistency = 1 - (pi @ row_H) / np.log(n)

    # 2. BART risk preference: long-run mass on risky states
    risk_pref = pi[rsk].sum()

    # 3. IGT loss aversion: after a punitive state, flee to safe vs chase into risky
    w = pi[pun] / pi[pun].sum()
    after = w @ P[pun]
    loss_aversion = (after[safe].sum() + EPS) / (after[rsk].sum() + EPS)

    # 4. IGT learning rate: spectral gap (modulus handles complex eigenvalues)
    mags = np.sort(np.abs(np.linalg.eigvals(P)))[::-1]
    learning_rate = 1 - mags[1]

    # 5. Two-stage model-based weight: 1 - habit, habit = stationary-weighted self-loop mass
    mb_weight = 1 - pi @ np.diag(P)

    # 6. Eligibility trace: persistence of the punitive signal.
    #    Integrated autocorrelation time tau -> lambda = 1 - 1/tau
    f = np.zeros(n)
    f[pun] = 1
    m = pi @ f
    var = pi @ f**2 - m**2
    tau = 1.0
    if var > EPS:
        Pk, g = np.eye(n), pi * f
        for _ in range(max_lag):
            Pk = Pk @ P
            tau += 2 * ((g @ Pk @ f - m**2) / var)
    tau = max(tau, 1.0)
    elig_trace = 1 - 1 / tau

    return dict(consistency=consistency, risk_pref=risk_pref,
                loss_aversion=loss_aversion, learning_rate=learning_rate,
                mb_weight=mb_weight, elig_trace=elig_trace)


def plausibility_error(P, targets, scales=None, **kw):
    """Weighted squared error between a chain's implied metrics and known
    behavioural metrics (dict, any subset of the six). Lower = more plausible.
    scales: per-metric SD (e.g. across participants) so units are comparable."""
    got = chain_to_metrics(P, **kw)
    err = 0.0
    for k, t in targets.items():
        s = (scales or {}).get(k, 1.0)
        a, b = got[k], t
        if k == "loss_aversion":  # ratio: compare on log scale
            a, b = np.log(a), np.log(b)
        err += ((a - b) / s) ** 2
    return err



# ---------------------------------------------------------------------------
# Vector form: Phi(M) and the error function against observed Theta
# Order: risk pref, choice consistency, loss aversion, learning rate,
#        model-based weight, eligibility trace
# ---------------------------------------------------------------------------
ORDER = ["risk_pref", "consistency", "loss_aversion", "learning_rate", "mb_weight", "elig_trace"]
THETA = np.array([INSERT 6-VECTOR OF VALUES])
LOG_IDX = [ORDER.index("loss_aversion")]  # ratio-scaled, compared in log space


def Phi(M, **kw):
    d = chain_to_metrics(M, **kw)
    return np.array([d[k] for k in ORDER])


def _to_space(v):
    v = np.array(v, float)
    v[LOG_IDX] = np.log(np.clip(v[LOG_IDX], EPS, None))
    return v


def error(M, theta=THETA, sigma=0.15, cov=None, return_parts=False, **kw):
    """Negative log-likelihood of observed theta given chain M (lower = more plausible).
    Model: theta ~ N(Phi(M), Sigma), computed in log space for loss aversion.
      sigma: scalar or length-6 vector of per-metric noise SDs (in the transformed space;
             use cohort SDs if you have them). Ignored if cov is given.
      cov:   full 6x6 covariance (e.g. cohort covariance of the six metrics), which makes
             this a Mahalanobis distance and handles the learning-rate / trace overlap.
    exp(-error) is the likelihood, so error differences are log likelihood ratios."""
    d = _to_space(theta) - _to_space(Phi(M, **kw))
    S = np.asarray(cov, float) if cov is not None else np.diag(np.broadcast_to(sigma, 6) ** 2)
    nll = 0.5 * d @ np.linalg.solve(S, d) + 0.5 * np.linalg.slogdet(2 * np.pi * S)[1]
    return (nll, d) if return_parts else nll



# ---------------------------------------------------------------------------
# Environmental friction term and composite error
# ---------------------------------------------------------------------------
E = np.array([
    INSERT ECOLOGICAL MATRIX
])


def friction(M, E=E):
    """sqrt(sum_ij (E ⊙ M)_ij^2): Frobenius norm of the Hadamard product."""
    return float(np.sqrt(np.sum((E * np.asarray(M)) ** 2)))


def composite_error(M, theta=THETA, w_friction=1.0, **kw):
    """Metric NLL + w_friction * friction. w_friction=1 is the plain sum you asked for;
    raise/lower it to trade environmental rationality against the behavioural match."""
    return error(M, theta=theta, **kw) + w_friction * friction(M)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    M = rng.dirichlet(np.ones(7), size=7)
    print("metric NLL:", round(error(M), 3), " friction:", round(friction(M), 3),
          " composite:", round(composite_error(M), 3))
