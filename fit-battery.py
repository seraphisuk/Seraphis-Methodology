#!/usr/bin/env python
"""
fit_battery.py
==============
Extracts six cognitive-model parameters from the BART / IGT / Two-Step workbook.

    BART      risk preference, choice consistency      (Wallsten-style 2-parameter model, hierarchical Bayes, PyMC)
    IGT       loss aversion, learning rate             (PVL-Delta, hierarchical Bayes, PyMC)
    Two-Step  model-based weight, eligibility trace    (Daw et al. 2011 hybrid, per-participant maximum likelihood)

USAGE (Windows, Anaconda/Miniforge Prompt, inside the folder containing this file):

    python fit_battery.py data.xlsx
    python fit_battery.py data.xlsx --quick          (fast test run, rough numbers)
    python fit_battery.py data.xlsx --only twostep   (run just one task: bart, igt or twostep)

Output: results.xlsx (per participant, aggregate, correlations, diagnostics, model notes).

Design facts taken from the task pages
    BART  : 20 balloons, breakpoint ~ Uniform{1..128}; pump number b pops a balloon whose breakpoint is b,
            so on a popped balloon `pumps` includes the bursting pump; on a banked balloon it is the number
            of successful pumps.
    IGT   : 100 trials, classic Bechara schedule, `win` and `loss` stored as positive magnitudes.
    2SM   : 75 trials, rocket 1 -> planet 1 (aliens 1,2) and rocket 2 -> planet 2 (aliens 3,4) with
            probability 0.7; alien reward probabilities follow bounded random walks (reflecting 0.25..0.75).
"""
import argparse
import math
import os
import re
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm

# =====================================================================
# SETTINGS (you normally do not need to change anything here)
# =====================================================================
SEED = 2026

# MCMC (BART and IGT)
MCMC_DRAWS = 2000
MCMC_TUNE = 2000
MCMC_CHAINS = 4
TARGET_ACCEPT = 0.95

# Two-Step maximum likelihood
TWOSTEP_STARTS = 20              # random starting points per participant
TWOSTEP_SEPARATE_ALPHAS = False  # False: one learning rate for both stages (recommended with 75 trials)
P_COMMON = 0.7                   # known common-transition probability (from the task page)
N_BOOT = 5000                    # bootstrap resamples for the Two-Step aggregate interval

IGT_OUTCOME_SCALE = 100.0        # outcomes are divided by this before fitting (numerical stability)

# Two-Step parameter bounds
TS_BOUNDS = {
    "alpha1": (0.0, 1.0), "alpha2": (0.0, 1.0),
    "beta1": (0.0, 20.0), "beta2": (0.0, 20.0),
    "lam": (0.0, 1.0), "w": (0.0, 1.0), "pers": (-1.0, 3.0),
}


# =====================================================================
# GENERIC HELPERS
# =====================================================================
def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def norm_name(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_workbook(path):
    """Return {'bart': df, 'igt': df, 'twostep': df}; missing tabs are simply absent."""
    raw = pd.read_excel(path, sheet_name=None)
    out = {}
    for name, df in raw.items():
        key = norm_name(name)
        task = "bart" if "bart" in key else "igt" if "igt" in key else "twostep" if "twostep" in key or "2sm" in key else None
        if task is None:
            continue
        df = df.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        out[task] = df
        log(f"Read tab '{name}': {len(df)} rows -> {task}")
    return out


def num(df, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan
    return df


def trial_rows_only(df):
    """Drop session_summary rows; keep everything else. Cleans the session id."""
    d = df.copy()
    if "event_type" in d.columns:
        d = d[~d["event_type"].astype(str).str.strip().str.lower().eq("session_summary")]
    d = d[d["session_id"].notna()].copy()
    d["session_id"] = d["session_id"].astype(str).str.strip()
    d = d[d["session_id"] != ""]
    return d


def dedupe(d, keys, label):
    n0 = len(d)
    d = d.sort_values(keys).drop_duplicates(keys, keep="first")
    if len(d) < n0:
        log(f"  {label}: ignored {n0 - len(d)} repeated trial row(s)")
    return d


def all_session_ids(tabs):
    ids = set()
    for df in tabs.values():
        if "session_id" in df.columns:
            s = df["session_id"].dropna().astype(str).str.strip()
            ids |= set(s[s != ""])
    return sorted(ids)


# =====================================================================
# SUMMARY HELPERS
# =====================================================================
def bayes_group_row(task, label, group_samples, indiv_means, note=""):
    g = np.asarray(group_samples).reshape(-1)
    im = np.asarray(indiv_means, dtype=float)
    return {
        "Task": task, "Parameter": label, "Method": "Hierarchical Bayes (posterior)",
        "Group estimate": float(np.mean(g)),
        "95% interval low": float(np.percentile(g, 2.5)),
        "95% interval high": float(np.percentile(g, 97.5)),
        "Mean of individual estimates": float(np.nanmean(im)),
        "Median of individual estimates": float(np.nanmedian(im)),
        "SD of individual estimates": float(np.nanstd(im, ddof=1)) if np.sum(~np.isnan(im)) > 1 else np.nan,
        "N participants": int(np.sum(~np.isnan(im))), "Note": note,
    }


def ml_group_row(task, label, values, rng, note=""):
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return {"Task": task, "Parameter": label, "Method": "Per-participant ML (mean)", "N participants": 0, "Note": note}
    boots = np.array([np.mean(rng.choice(v, size=len(v), replace=True)) for _ in range(N_BOOT)])
    return {
        "Task": task, "Parameter": label, "Method": "Per-participant ML (mean)",
        "Group estimate": float(np.mean(v)),
        "95% interval low": float(np.percentile(boots, 2.5)),
        "95% interval high": float(np.percentile(boots, 97.5)),
        "Mean of individual estimates": float(np.mean(v)),
        "Median of individual estimates": float(np.median(v)),
        "SD of individual estimates": float(np.std(v, ddof=1)) if len(v) > 1 else np.nan,
        "N participants": int(len(v)), "Note": note,
    }


# =====================================================================
# PyMC plumbing
# =====================================================================
def check_pymc():
    import pymc as pm  # noqa: F401
    import pytensor
    if not pytensor.config.cxx:
        log("WARNING: PyMC cannot find a C++ compiler, so it will run in a much slower fallback mode.")
        log("         The IGT fit could take many hours. See the 'Troubleshooting' section of the instructions.")


def sample_model(model, label, args):
    import pymc as pm
    draws = 400 if args.quick else MCMC_DRAWS
    tune = 400 if args.quick else MCMC_TUNE
    chains = 2 if args.quick else MCMC_CHAINS
    kw = dict(draws=draws, tune=tune, chains=chains, target_accept=TARGET_ACCEPT,
              random_seed=SEED, progressbar=True, idata_kwargs={"log_likelihood": False})
    log(f"Sampling {label}: {chains} chains x ({tune} tuning + {draws} draws)")
    with model:
        try:
            return pm.sample(cores=min(chains, os.cpu_count() or 1), **kw)
        except Exception as e:  # multiprocessing problems on some Windows set-ups
            log(f"Parallel sampling failed ({type(e).__name__}); retrying one chain at a time (slower).")
            return pm.sample(cores=1, **kw)


def diagnostics_row(idata, var_names, label):
    row = {"Task": label, "Max R-hat": np.nan, "Min effective sample size": np.nan, "Divergent transitions": np.nan}
    try:
        import arviz as az
        s = az.summary(idata, var_names=var_names)
        row["Max R-hat"] = float(s["r_hat"].max())
        row["Min effective sample size"] = float(s["ess_bulk"].min())
        row["Divergent transitions"] = int(idata.sample_stats["diverging"].sum())
    except Exception as e:
        row["Note"] = f"Could not compute: {e}"
    if row["Max R-hat"] == row["Max R-hat"] and row["Max R-hat"] > 1.05:
        row["Note"] = "R-hat above 1.05: chains have not fully converged; treat this task's estimates with caution."
    return row


def post_array(idata, name):
    """posterior[name] as (samples, participants)."""
    a = idata.posterior[name].values
    return a.reshape(-1, a.shape[-1])


# =====================================================================
# BART  --  Wallsten-style two-parameter model
#   P(pump at opportunity k) = 1 / (1 + exp(beta * (k - omega)))
#   omega = target number of pumps (risk preference); beta = sensitivity (choice consistency)
# =====================================================================
def prep_bart(df):
    d = trial_rows_only(df)
    d = num(d, ["balloon_index", "pumps"])
    d["outcome"] = d["outcome"].astype(str).str.strip().str.lower() if "outcome" in d.columns else ""
    d = d.dropna(subset=["balloon_index", "pumps"])
    d = d[d["outcome"].isin(["popped", "banked"])]
    d = dedupe(d, ["session_id", "balloon_index"], "BART")
    return d


def bart_decisions(d, sessions):
    """Every pump / stop decision as rows (participant index, opportunity k, y=1 pump / 0 stop)."""
    index = {s: i for i, s in enumerate(sessions)}
    P, K, Y = [], [], []
    for s, g in d.groupby("session_id"):
        i = index[s]
        for n, out in zip(g["pumps"].astype(int).clip(lower=0), g["outcome"]):
            if n > 0:
                K.append(np.arange(1, n + 1)); Y.append(np.ones(n)); P.append(np.full(n, i))
            if out == "banked":                       # chose to stop at opportunity n+1
                K.append(np.array([n + 1])); Y.append(np.array([0.0])); P.append(np.array([i]))
    return np.concatenate(P), np.concatenate(K).astype(float), np.concatenate(Y)


def bart_ll_np(omega, beta, k, y):
    logit = beta * (omega - k)
    return float(np.sum(y * logit - np.logaddexp(0.0, logit)))


def fit_bart(tab, args):
    import pymc as pm
    import pytensor.tensor as pt

    d = prep_bart(tab)
    sessions = sorted(d["session_id"].unique())
    if not sessions:
        log("BART: no usable trials found, skipping.")
        return None
    log(f"BART: {len(sessions)} participants, {len(d)} balloons")
    P, K, Y = bart_decisions(d, sessions)
    n = len(sessions)

    with pm.Model() as model:
        mu_lw = pm.Normal("mu_logomega", 3.5, 1.0)         # exp(3.5) ~ 33 pumps; the optimum is 64
        sd_lw = pm.HalfNormal("sd_logomega", 1.0)
        mu_lb = pm.Normal("mu_logbeta", -1.5, 1.5)
        sd_lb = pm.HalfNormal("sd_logbeta", 1.0)
        zw = pm.Normal("z_omega", 0.0, 1.0, shape=n)
        zb = pm.Normal("z_beta", 0.0, 1.0, shape=n)
        omega = pm.Deterministic("omega", pt.exp(mu_lw + sd_lw * zw))
        beta = pm.Deterministic("beta", pt.exp(mu_lb + sd_lb * zb))
        logit_p = beta[P] * (omega[P] - K)
        pm.Bernoulli("y", logit_p=logit_p, observed=Y)

    idata = sample_model(model, "BART", args)

    om, be = post_array(idata, "omega"), post_array(idata, "beta")
    om_m, be_m = om.mean(0), be.mean(0)
    per = pd.DataFrame({
        "session_id": sessions,
        "risk_preference_BART": om_m, "risk_preference_BART_sd": om.std(0),
        "choice_consistency_BART": be_m, "choice_consistency_BART_sd": be.std(0),
    })
    n_dec = [int(np.sum(P == i)) for i in range(n)]
    ll = [bart_ll_np(om_m[i], be_m[i], K[P == i], Y[P == i]) for i in range(n)]
    per["BART_n_balloons"] = d.groupby("session_id").size().reindex(sessions).values
    per["BART_loglik"] = ll
    per["BART_pseudoR2"] = [1 - l / (nd * math.log(0.5)) if nd else np.nan for l, nd in zip(ll, n_dec)]

    g_om = np.exp(idata.posterior["mu_logomega"].values)
    g_be = np.exp(idata.posterior["mu_logbeta"].values)
    rows = [
        bayes_group_row("BART", "Risk preference (omega, target pumps)", g_om, om_m,
                        "Group estimate = typical (median) participant. Optimal strategy for this task is about 64 pumps."),
        bayes_group_row("BART", "Choice consistency (beta)", g_be, be_m,
                        "Higher = more deterministic pump/stop decisions."),
    ]
    diag = diagnostics_row(idata, ["mu_logomega", "sd_logomega", "mu_logbeta", "sd_logbeta", "omega", "beta"], "BART")
    return per, rows, diag


# =====================================================================
# IGT  --  PVL-Delta (Ahn et al., 2008; parameter labels and bounds as in Steingroever et al., 2013)
#   u(x) = x^A (gains), -w |x|^A (losses);  Ev_k += a (u - Ev_k) for the chosen deck;
#   P(choice) = softmax(theta * Ev), theta = 3^c - 1
#   A = utility curvature (0-1), w = loss aversion (0-5), a = learning rate (0-1), c = consistency (0-5)
# =====================================================================
def prep_igt(df):
    d = trial_rows_only(df)
    d = num(d, ["trial", "win", "loss", "net"])
    d["deck"] = d["deck"].astype(str).str.strip().str.upper() if "deck" in d.columns else ""
    d = d[d["deck"].isin(list("ABCD"))]
    d = d.dropna(subset=["trial"])
    d["x"] = (d["win"] - d["loss"].abs()).where(d["win"].notna(), d["net"])
    d["x"] = d["x"].fillna(d["net"])
    d = d.dropna(subset=["x"])
    d = dedupe(d, ["session_id", "trial"], "IGT")
    return d


def igt_ll_np(A, w, a, c, choices, x):
    theta = 3.0 ** c - 1.0
    ev = np.zeros(4)
    ll = 0.0
    for ch, xt in zip(choices, x):
        z = theta * ev
        z = z - z.max()
        ll += z[ch] - math.log(np.exp(z).sum())
        mag = (abs(xt) + 1e-6) ** A
        u = mag if xt >= 0 else -w * mag
        ev[ch] += a * (u - ev[ch])
    return ll


def fit_igt(tab, args):
    import pymc as pm
    import pytensor
    import pytensor.tensor as pt

    d = prep_igt(tab)
    sessions = sorted(d["session_id"].unique())
    if not sessions:
        log("IGT: no usable trials found, skipping.")
        return None
    n = len(sessions)
    seqs = {}
    for s, g in d.sort_values(["session_id", "trial"]).groupby("session_id"):
        seqs[s] = (g["deck"].map({"A": 0, "B": 1, "C": 2, "D": 3}).values.astype(int),
                   (g["x"].values / IGT_OUTCOME_SCALE).astype(float))
    T = max(len(v[0]) for v in seqs.values())
    log(f"IGT: {n} participants, up to {T} trials each")

    onehot = np.zeros((T, n, 4))
    xmat = np.zeros((T, n))
    for j, s in enumerate(sessions):
        ch, xs = seqs[s]
        onehot[np.arange(len(ch)), j, ch] = 1.0      # padded trials stay all-zero = ignored
        xmat[:len(ch), j] = xs

    with pm.Model() as model:
        mu_pr = pm.Normal("mu_pr", 0.0, 1.0, shape=4)
        sd_pr = pm.HalfNormal("sd_pr", 0.5, shape=4)
        z = pm.Normal("z", 0.0, 1.0, shape=(4, n))
        probs = pm.math.invprobit(mu_pr[:, None] + sd_pr[:, None] * z)
        A = pm.Deterministic("A", probs[0])
        w = pm.Deterministic("w", probs[1] * 5.0)
        a = pm.Deterministic("a", probs[2])
        c = pm.Deterministic("c", probs[3] * 5.0)
        theta = pt.power(3.0, c) - 1.0

        oh = pt.as_tensor_variable(onehot)
        xs_t = pt.as_tensor_variable(xmat)

        def step(oh_t, x_t, ev, A_, w_, a_, theta_):
            logits = theta_[:, None] * ev
            logp = logits - pt.logsumexp(logits, axis=1, keepdims=True)
            ll_t = (logp * oh_t).sum(axis=1)
            mag = (pt.abs(x_t) + 1e-6) ** A_
            u = pt.switch(x_t >= 0, mag, -w_ * mag)
            ev_new = ev + a_[:, None] * (u[:, None] - ev) * oh_t
            return ev_new, ll_t

        (_, ll_seq), _ = pytensor.scan(
            step, sequences=[oh, xs_t],
            outputs_info=[pt.zeros((n, 4)), None],
            non_sequences=[A, w, a, theta])
        pm.Potential("loglik", ll_seq.sum())

    idata = sample_model(model, "IGT (this is the slow one)", args)

    pA, pw, pa, pc = (post_array(idata, k) for k in ["A", "w", "a", "c"])
    mA, mw, ma, mc = pA.mean(0), pw.mean(0), pa.mean(0), pc.mean(0)
    per = pd.DataFrame({
        "session_id": sessions,
        "loss_aversion_IGT": mw, "loss_aversion_IGT_sd": pw.std(0),
        "learning_rate_IGT": ma, "learning_rate_IGT_sd": pa.std(0),
        "IGT_utility_curvature_A": mA, "IGT_consistency_c": mc,
        "IGT_n_trials": [len(seqs[s][0]) for s in sessions],
    })
    ll = [igt_ll_np(mA[j], mw[j], ma[j], mc[j], seqs[s][0], seqs[s][1]) for j, s in enumerate(sessions)]
    per["IGT_loglik"] = ll
    per["IGT_pseudoR2"] = [1 - l / (len(seqs[s][0]) * math.log(0.25)) if len(seqs[s][0]) else np.nan
                           for l, s in zip(ll, sessions)]

    mu = idata.posterior["mu_pr"].values            # (chain, draw, 4)
    rows = [
        bayes_group_row("IGT", "Loss aversion (w)", norm.cdf(mu[..., 1]) * 5.0, mw,
                        "Group estimate = typical participant. 1 = losses and gains weighted equally."),
        bayes_group_row("IGT", "Learning rate (a)", norm.cdf(mu[..., 2]), ma),
        bayes_group_row("IGT", "Utility curvature (A)", norm.cdf(mu[..., 0]), mA,
                        "Outcomes were divided by 100 before fitting."),
        bayes_group_row("IGT", "Choice consistency (c)", norm.cdf(mu[..., 3]) * 5.0, mc),
    ]
    diag = diagnostics_row(idata, ["mu_pr", "sd_pr", "A", "w", "a", "c"], "IGT")
    return per, rows, diag


# =====================================================================
# TWO-STEP  --  Daw et al. (2011) hybrid, fitted per participant by maximum likelihood
#   Stage-1 value = w * Q_MB + (1 - w) * Q_MF, softmax(beta1 * (value + pers * [same rocket as last trip]))
#   Stage-2 softmax(beta2 * Q2). SARSA(lambda) updates; Q initialised at 0; transition probability known (0.7).
# =====================================================================
def prep_twostep(df):
    d = trial_rows_only(df)
    d = num(d, ["trial", "stage1_choice", "stage2_choice", "reward"])
    d = d.dropna(subset=["trial", "stage1_choice", "stage2_choice", "reward"])
    d = d[d["stage1_choice"].isin([1, 2]) & d["stage2_choice"].isin([1, 2, 3, 4]) & d["reward"].isin([0, 1])]
    d = dedupe(d, ["session_id", "trial"], "Two-Step")
    return d


def _logsig(z):
    return -math.log1p(math.exp(-z)) if z >= 0 else z - math.log1p(math.exp(z))


def twostep_nll(theta, trials, sep_alpha):
    if sep_alpha:
        a1, a2, b1, b2, lam, w, pers = theta
    else:
        a1, b1, b2, lam, w, pers = theta
        a2 = a1
    pc = P_COMMON
    q1 = [0.0, 0.0]
    q2 = [0.0, 0.0, 0.0, 0.0]
    ll = 0.0
    prev = -1
    for c1, al, r in trials:
        v_p1 = q2[0] if q2[0] > q2[1] else q2[1]
        v_p2 = q2[2] if q2[2] > q2[3] else q2[3]
        mb0 = pc * v_p1 + (1 - pc) * v_p2
        mb1 = (1 - pc) * v_p1 + pc * v_p2
        x0 = b1 * (w * mb0 + (1 - w) * q1[0] + (pers if prev == 0 else 0.0))
        x1 = b1 * (w * mb1 + (1 - w) * q1[1] + (pers if prev == 1 else 0.0))
        ll += _logsig(x1 - x0) if c1 == 1 else _logsig(x0 - x1)
        pl = al // 2
        xa, xb = b2 * q2[2 * pl], b2 * q2[2 * pl + 1]
        ll += _logsig(xb - xa) if al == 2 * pl + 1 else _logsig(xa - xb)
        d1 = q2[al] - q1[c1]
        q1[c1] += a1 * d1
        d2 = r - q2[al]
        q2[al] += a2 * d2
        q1[c1] += a1 * lam * d2
        prev = c1
    return -ll


def fit_twostep_one(trials, rng, sep_alpha):
    names = (["alpha1", "alpha2"] if sep_alpha else ["alpha1"]) + ["beta1", "beta2", "lam", "w", "pers"]
    bounds = [TS_BOUNDS[k] for k in names]
    best = None
    for s in range(TWOSTEP_STARTS):
        x0 = []
        for k in names:
            lo, hi = TS_BOUNDS[k]
            if k.startswith("beta"):
                x0.append(rng.uniform(0.5, 8.0))
            elif k == "pers":
                x0.append(rng.uniform(-0.3, 1.0))
            else:
                x0.append(rng.uniform(lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo)))
        try:
            res = minimize(twostep_nll, x0, args=(trials, sep_alpha), method="L-BFGS-B", bounds=bounds)
        except Exception:
            continue
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return None
    est = dict(zip(names, best.x))
    if not sep_alpha:
        est["alpha2"] = est["alpha1"]
    return est, -best.fun, names


def fit_twostep(tab, args):
    d = prep_twostep(tab)
    sessions = sorted(d["session_id"].unique())
    if not sessions:
        log("Two-Step: no usable trials found, skipping.")
        return None
    log(f"Two-Step: {len(sessions)} participants, {len(d)} trials (maximum likelihood, {TWOSTEP_STARTS} starts each)")
    rng = np.random.default_rng(SEED)
    rows = []
    for i, s in enumerate(sessions, 1):
        g = d[d["session_id"] == s].sort_values("trial")
        trials = [(int(c) - 1, int(a) - 1, int(r)) for c, a, r in zip(g["stage1_choice"], g["stage2_choice"], g["reward"])]
        fit = fit_twostep_one(trials, rng, TWOSTEP_SEPARATE_ALPHAS)
        log(f"  Two-Step participant {i}/{len(sessions)}")
        if fit is None:
            rows.append({"session_id": s, "TwoStep_n_trials": len(trials)})
            continue
        est, ll, names = fit
        k = len(names)
        nobs = 2 * len(trials)
        at_bound = [n_ for n_, v in zip(names, [est[x] for x in names])
                    if min(abs(v - TS_BOUNDS[n_][0]), abs(v - TS_BOUNDS[n_][1])) < 1e-3 * (TS_BOUNDS[n_][1] - TS_BOUNDS[n_][0])]
        rows.append({
            "session_id": s,
            "model_based_weight_2SM": est["w"], "eligibility_trace_2SM": est["lam"],
            "TwoStep_learning_rate": est["alpha1"], "TwoStep_learning_rate_stage2": est["alpha2"],
            "TwoStep_beta_stage1": est["beta1"], "TwoStep_beta_stage2": est["beta2"],
            "TwoStep_perseveration": est["pers"],
            "TwoStep_n_trials": len(trials), "TwoStep_loglik": ll,
            "TwoStep_AIC": 2 * k - 2 * ll, "TwoStep_BIC": k * math.log(nobs) - 2 * ll,
            "TwoStep_pseudoR2": 1 - ll / (nobs * math.log(0.5)),
            "TwoStep_params_at_bounds": ", ".join(at_bound),
        })
    per = pd.DataFrame(rows)

    boot_rng = np.random.default_rng(SEED + 1)
    note_w = "Mean of per-participant ML estimates; interval is a bootstrap over participants."
    note_l = note_w + " Lambda is weakly identified with only 75 trials per person; many individual estimates may sit at 0 or 1."
    agg = [
        ml_group_row("Two-Step", "Model-based weight (w)", per.get("model_based_weight_2SM", []), boot_rng, note_w),
        ml_group_row("Two-Step", "Eligibility trace (lambda)", per.get("eligibility_trace_2SM", []), boot_rng, note_l),
        ml_group_row("Two-Step", "Learning rate (alpha)", per.get("TwoStep_learning_rate", []), boot_rng),
        ml_group_row("Two-Step", "Inverse temperature, stage 1 (beta1)", per.get("TwoStep_beta_stage1", []), boot_rng),
        ml_group_row("Two-Step", "Inverse temperature, stage 2 (beta2)", per.get("TwoStep_beta_stage2", []), boot_rng),
        ml_group_row("Two-Step", "Perseveration", per.get("TwoStep_perseveration", []), boot_rng),
    ]
    diag = {"Task": "Two-Step",
            "Note": f"ML fits. Participants with at least one parameter at a bound: "
                    f"{int((per.get('TwoStep_params_at_bounds', pd.Series(dtype=str)).fillna('') != '').sum())} of {len(per)}"}
    return per, agg, diag


# =====================================================================
# MAIN
# =====================================================================
MODEL_NOTES = [
    ("BART", "Per pump opportunity k on each balloon: P(pump) = 1 / (1 + exp(beta * (k - omega))). omega = risk preference "
             "(the number of pumps the person is aiming for; the payoff-maximising value here is about 64). beta = choice consistency "
             "(slope; larger = more deterministic). Popped balloons contribute only 'pump' decisions (including the bursting pump); "
             "banked balloons contribute 'pump' decisions plus one final 'stop'. Hierarchical: log omega and log beta are normally "
             "distributed across participants (non-centred). Priors: mu_log_omega ~ N(3.5,1), mu_log_beta ~ N(-1.5,1.5), group SDs ~ HalfNormal(1)."),
    ("IGT", "PVL-Delta (Ahn et al. 2008; parameter labels and bounds as in Steingroever et al. 2013; probit-transformed hierarchy). Outcome = win - loss; u = x^A for gains, -w*|x|^A for losses; "
            "delta-rule update of the chosen deck's expectancy with learning rate a; softmax with theta = 3^c - 1. Expectancies start at 0. "
            "Outcomes divided by 100 before fitting. Hierarchical with probit-scaled, non-centred parameters: A in (0,1), w in (0,5), a in (0,1), c in (0,5). "
            "Priors: group means on the probit scale ~ N(0,1), group SDs ~ HalfNormal(0.5)."),
    ("Two-Step", "Daw et al. (2011) hybrid with known transition probability 0.7. Stage-1 value = w*Q_MB + (1-w)*Q_MF; SARSA(lambda) "
                 "updates (lambda = eligibility trace); softmax with beta1 at stage 1 (plus a perseveration bonus for repeating the previous rocket) "
                 "and beta2 at stage 2. One shared learning rate for both stages (a 75-trial task cannot support separate ones). "
                 "Q-values start at 0. Both stage-1 and stage-2 choices enter the likelihood. Fitted per participant by maximum likelihood "
                 f"(L-BFGS-B, {TWOSTEP_STARTS} random starts). Bounds: alpha 0-1, beta 0-20, lambda 0-1, w 0-1, perseveration -1 to 3."),
    ("Aggregation", "BART and IGT group estimates are the population-level (hyper)parameters of the hierarchical model, i.e. the typical participant, "
                    "with a 95% credible interval. Two-Step group estimates are the mean of the per-participant ML estimates with a bootstrap 95% interval. "
                    "Per-participant values for BART and IGT are posterior means and are shrunk toward the group; correlations between them are therefore "
                    "approximate."),
    ("Sessions", "Every session_id is treated as one participant. No participants were excluded. Repeated trial rows (same session and trial number) were ignored; "
                 "malformed rows were skipped."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", nargs="?", default="data.xlsx")
    ap.add_argument("--out", default="results.xlsx")
    ap.add_argument("--only", choices=["bart", "igt", "twostep"], default=None)
    ap.add_argument("--quick", action="store_true", help="short MCMC runs for testing")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Could not find '{args.input}'. Put the workbook in the same folder as this script, or give the full path.")
    warnings.filterwarnings("ignore", category=FutureWarning)

    tabs = load_workbook(args.input)
    if not tabs:
        sys.exit("No tabs named like 'BART Data', 'IGT Data', 'TwoStep Data' were found.")
    sessions = all_session_ids(tabs)
    log(f"{len(sessions)} distinct session_ids across all tabs")

    if args.only in (None, "bart", "igt"):
        try:
            check_pymc()
        except ImportError:
            sys.exit("PyMC is not installed in this Python environment. Follow step 1 of the instructions, "
                     "then open the 'PyMC' environment before running this script.")

    per_tables, agg_rows, diag_rows = [], [], []
    jobs = [("bart", fit_bart), ("igt", fit_igt), ("twostep", fit_twostep)]
    for key, fn in jobs:
        if args.only and args.only != key:
            continue
        if key not in tabs:
            log(f"No tab found for {key}; skipping.")
            continue
        try:
            res = fn(tabs[key], args)
        except Exception as e:
            log(f"!! {key} failed: {type(e).__name__}: {e}")
            diag_rows.append({"Task": key, "Note": f"FAILED: {type(e).__name__}: {e}"})
            continue
        if res is None:
            continue
        per, rows, diag = res
        per_tables.append(per)
        agg_rows += rows
        diag_rows.append(diag)

    # ---- assemble per-participant table (every session appears; blanks where a task is missing) ----
    out = pd.DataFrame({"session_id": sessions})
    for t in per_tables:
        out = out.merge(t, on="session_id", how="left")
    main_cols = ["risk_preference_BART", "choice_consistency_BART", "loss_aversion_IGT", "learning_rate_IGT",
                 "model_based_weight_2SM", "eligibility_trace_2SM"]
    for c in main_cols:
        if c not in out.columns:
            out[c] = np.nan
    rest = [c for c in out.columns if c not in main_cols + ["session_id"]]
    out = out[["session_id"] + main_cols + rest]

    # ---- aggregate: the six requested parameters first ----
    agg = pd.DataFrame(agg_rows)
    if not agg.empty:
        priority = ["Risk preference (omega, target pumps)", "Choice consistency (beta)", "Loss aversion (w)",
                    "Learning rate (a)", "Model-based weight (w)", "Eligibility trace (lambda)"]
        agg["_o"] = agg["Parameter"].apply(lambda p: priority.index(p) if p in priority else 99)
        agg = agg.sort_values(["_o"], kind="stable").drop(columns="_o")

    corr_p = out[main_cols].apply(pd.to_numeric, errors="coerce").corr(method="pearson")
    corr_s = out[main_cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")

    with pd.ExcelWriter(args.out, engine="openpyxl") as xw:
        out.to_excel(xw, sheet_name="Per participant", index=False)
        agg.to_excel(xw, sheet_name="Aggregate", index=False)
        corr_p.to_excel(xw, sheet_name="Correlations (Pearson)")
        corr_s.to_excel(xw, sheet_name="Correlations (Spearman)")
        pd.DataFrame(diag_rows).to_excel(xw, sheet_name="Diagnostics", index=False)
        pd.DataFrame(MODEL_NOTES, columns=["Topic", "Description"]).to_excel(xw, sheet_name="Model notes", index=False)
        for ws in xw.book.worksheets:
            for col in ws.columns:
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col[:60])
                ws.column_dimensions[col[0].column_letter].width = min(max(12, width + 2), 70)

    log(f"Done. Results written to {os.path.abspath(args.out)}")
    if not agg.empty:
        print("\nAGGREGATE RESULTS")
        print(agg[["Task", "Parameter", "Group estimate", "95% interval low", "95% interval high"]].to_string(index=False))


if __name__ == "__main__":
    main()
