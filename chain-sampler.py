#!/usr/bin/env python3
"""Find the 7x7 Markov chain minimising composite_error (metric NLL + friction).

Parametrisation: each row is softmax(logits), so every row is a valid distribution.
Algorithm:  simulated-annealing Metropolis (several restarts) -> optional Nelder-free
            local polish (scipy L-BFGS if available) -> optional posterior sampling at T=1.

Usage:
    python sample_chain.py                       # defaults
    python sample_chain.py --steps 60000 --restarts 8 --seed 1
    python sample_chain.py --w-friction 0.5 --sigma 0.2 --posterior 5000
"""
import argparse
import numpy as np
from chain_metrics import Phi, THETA, ORDER, composite_error, error, friction

FLOOR = 1e-4  # keeps the chain ergodic so the stationary distribution is unique


def to_chain(L):
    Z = np.exp(L - L.max(1, keepdims=True))
    M = Z / Z.sum(1, keepdims=True)
    M = np.maximum(M, FLOOR)
    return M / M.sum(1, keepdims=True)


def make_energy(args):
    def U(L):
        try:
            return composite_error(to_chain(L), w_friction=args.w_friction, sigma=args.sigma)
        except Exception:
            return np.inf
    return U


def anneal(U, rng, steps, T0, T1, step0):
    L = rng.normal(0, 1, (7, 7))
    e = U(L)
    best_L, best_e = L.copy(), e
    for t in range(steps):
        frac = t / steps
        T = T0 * (T1 / T0) ** frac
        s = step0 * (1 - 0.9 * frac)
        P = L.copy()
        if rng.random() < 0.7:                      # perturb one row
            i = rng.integers(7)
            P[i] += rng.normal(0, s, 7)
        else:                                       # or a single entry, bigger kick
            P[rng.integers(7), rng.integers(7)] += rng.normal(0, 3 * s)
        eP = U(P)
        if eP < e or rng.random() < np.exp((e - eP) / T):
            L, e = P, eP
            if e < best_e:
                best_L, best_e = L.copy(), e
    return best_L, best_e


def polish(U, L):
    try:
        from scipy.optimize import minimize
    except ImportError:
        return L, U(L)
    r = minimize(lambda x: U(x.reshape(7, 7)), L.ravel(), method="L-BFGS-B")
    L2 = r.x.reshape(7, 7)
    return (L2, r.fun) if r.fun < U(L) else (L, U(L))


def posterior(U, rng, L, n, step=0.05):
    """Plain Metropolis at T=1 around the optimum: how wide is the plausible set?"""
    e, out, acc = U(L), [], 0
    for _ in range(n):
        P = L.copy()
        i = rng.integers(7)
        P[i] += rng.normal(0, step, 7)
        eP = U(P)
        if eP < e or rng.random() < np.exp(e - eP):
            L, e, acc = P, eP, acc + 1
        out.append(Phi(to_chain(L)))
    return np.array(out), acc / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--restarts", type=int, default=5)
    ap.add_argument("--T0", type=float, default=20.0)
    ap.add_argument("--T1", type=float, default=0.02)
    ap.add_argument("--step", type=float, default=0.6)
    ap.add_argument("--sigma", type=float, default=0.15)
    ap.add_argument("--w-friction", type=float, default=1.0)
    ap.add_argument("--posterior", type=int, default=0, help="n Metropolis samples at T=1 after optimising")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="best_chain.npy")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    U = make_energy(args)

    best = (None, np.inf)
    for r in range(args.restarts):
        L, e = anneal(U, rng, args.steps, args.T0, args.T1, args.step)
        L, e = polish(U, L)
        print(f"restart {r + 1}/{args.restarts}: composite = {e:.4f}")
        if e < best[1]:
            best = (L, e)

    L, e = best
    M = to_chain(L)
    np.save(args.out, M)

    np.set_printoptions(precision=3, suppress=True, linewidth=120)
    print("\nBest chain (rows = from, cols = to):\n", M)
    print("\nrow sums:", M.sum(1))
    print(f"\ncomposite = {e:.4f}   metric NLL = {error(M, sigma=args.sigma):.4f}   "
          f"friction = {friction(M):.4f}")
    print(f"\n{'metric':15s} {'Phi(M)':>8s} {'Theta':>8s}")
    for k, a, b in zip(ORDER, Phi(M), THETA):
        print(f"{k:15s} {a:8.4f} {b:8.4f}")
    print(f"\nsaved to {args.out}")

    if args.posterior:
        S, acc = posterior(U, rng, L, args.posterior)
        print(f"\nposterior (T=1) acceptance {acc:.2f}; Phi mean ± sd over samples:")
        for k, m, s in zip(ORDER, S.mean(0), S.std(0)):
            print(f"{k:15s} {m:8.4f} ± {s:.4f}")


if __name__ == "__main__":
    main()
