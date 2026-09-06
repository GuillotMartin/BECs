#!/usr/bin/env python
"""Wall-clock benchmark and bit-exactness harness for the split-step solvers.

Two jobs, both needed while optimizing:

  * 'time'  -- run a fixed sweep and report seconds per accepted step, so a change can be shown to
               have helped rather than assumed to have.
  * 'save'  -- store the resulting wavefunctions to an .npz, and
    'check' -- compare a later run against that file with np.array_equal.

The comparison is deliberately exact and not np.allclose. The adaptive controller is chaotic: a
one-ulp change in the error estimate flips a step from accepted to rejected and sends the whole run
down a different path (see the note in BECs.groundstate.distance). So a change that is supposed to
be pure redundancy elimination must reproduce every bit, and 'allclose' would hide exactly the
mistakes this is meant to catch.

Usage:
    python benchmarks/bench_solvers.py time                 # both solvers, 256^2
    python benchmarks/bench_solvers.py time --n 512         # the size the heavy notebooks use
    python benchmarks/bench_solvers.py save --out base.npz  # before a change
    python benchmarks/bench_solvers.py check --ref base.npz # after it
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from bloch_schrodinger.potential import Potential, create_parameter

from BECs.potentialT import AnalyticPotential, PotentialT
from BECs.rssfm import rSSFM
from BECs.ssfm import SSFM

# Kept small and fixed: the point is a stable basis for comparison, not physics. Every quantity
# here is hard-coded rather than defaulted so that two runs on different days are comparable.
#
# The trap strength matters more than it looks. V = TRAP * r^2 is a harmonic trap of frequency
# omega = sqrt(2*TRAP), whose ground state has width 1/sqrt(omega). Start the run from a Gaussian
# far from that width and the packet breathes hard, the controller shortens the step to follow it,
# and the benchmark spends its time on violent dynamics rather than on representative ones -- an
# early version of this file used TRAP = 500 with SIGMA = 1.5, a factor of eight mismatch, and took
# 85 s per case at 128^2 instead of well under a second. These values are close to matched, so the
# packet breathes gently and the step size stays in a sane range.
BOX = 20.0
TRAP = 2.0                     # omega = 2, ground-state width 0.71
SIGMA = 1.0                    # mildly wider than that, so there is something to propagate
T_FINAL = 1.5                  # about half a trap period
N_SAMPLES = 20
TOL = 1e-6
G_VALUES = np.linspace(0.0, 0.5, 3)


def gaussian_psi0(pot) -> "np.ndarray":
    """A normalized Gaussian on a potential's grid, shared by all three cases."""
    psi0 = np.exp(-((pot.x / SIGMA) ** 2 + (pot.y / SIGMA) ** 2)).astype(complex)
    return psi0 / (abs(psi0) ** 2).sum(["a1", "a2"]) ** 0.5


def build_ssfm(n: int) -> SSFM:
    """A harmonic trap that is switched off part way, so the time-dependent path is exercised."""
    trap = PotentialT(unitvecs=[[BOX, 0], [0, BOX]], resolution=(n, n), v0=0)
    trap.add_shape("trap", (trap.x**2 + trap.y**2) * TRAP)
    trap.step("switch_off", ts=T_FINAL / 2, sigma=T_FINAL / 20, vi=1, vf=0)
    trap.add_term("trap * switch_off")

    return SSFM(
        potential=trap, psi0=gaussian_psi0(trap), g=create_parameter("g", G_VALUES)
    )


def build_static_ssfm(n: int) -> SSFM:
    """The same solver on a potential with no time dependence at all.

    Split out because that is the case the static fast path targets, and it is most of the runs in
    the notebooks -- a speedup there will not show up in 'build_ssfm'.
    """
    trap = Potential(unitvecs=[[BOX, 0], [0, BOX]], resolution=(n, n), v0=0)
    trap.add((trap.x**2 + trap.y**2) * TRAP)

    return SSFM(
        potential=trap, psi0=gaussian_psi0(trap), g=create_parameter("g", G_VALUES)
    )


def build_rssfm(n: int) -> rSSFM:
    """A free expansion in the rescaling frame, which is what rSSFM is for."""
    pot = AnalyticPotential(unitvecs=[[BOX, 0], [0, BOX]], resolution=(n, n))
    pot.add_function("flat", function=lambda t, x, y: x * 0, parameters={})
    pot.add_term("flat")

    return rSSFM(
        potential=pot, psi0=gaussian_psi0(pot), g=create_parameter("g", G_VALUES)
    )


CASES = {
    "ssfm": build_ssfm,
    "ssfm_static": build_static_ssfm,
    "rssfm": build_rssfm,
}


def run(case: str, n: int, order: int = 2) -> tuple[np.ndarray, float]:
    """Run one case and return its result together with the wall time it took.

    'order' selects the splitting for the two SSFM cases; rSSFM has only its own second-order one,
    so it ignores the argument rather than failing on it.
    """
    solver = CASES[case](n)
    t_samples = create_parameter("t", np.linspace(0, T_FINAL, N_SAMPLES))

    kwargs = {} if case == "rssfm" else {"order": order}
    t0 = time.perf_counter()
    psi = solver.solve(
        t_init=0,
        t_final=T_FINAL,
        t_samples=t_samples,
        tol=TOL,
        parallel=False,  # serial: this measures the solver, not the job scheduler
        verbose=False,
        **kwargs,
    )
    elapsed = time.perf_counter() - t0
    return np.asarray(psi), elapsed


def cmd_time(args: argparse.Namespace) -> int:
    # Best-of-N, not the mean. Timing noise on a desktop is one-sided -- something else stealing
    # the CPU can only make a run slower, never faster -- so the minimum is the closest estimate
    # of what the code costs. It matters here: repeated runs of identical code on this machine
    # came out 36 s, 50 s and 41 s, a spread far wider than the changes being measured.
    print(f"grid {args.n}x{args.n}, {len(G_VALUES)} values of g, tol {TOL:g}, serial")
    print(f"best of {args.repeat}\n")
    header = f"{'case':<14} {'wall (s)':>10} {'s / sample':>12} {'spread':>9}"
    print(header)
    for case in args.cases:
        times = [run(case, args.n, args.order)[1] for _ in range(args.repeat)]
        best = min(times)
        spread = (max(times) - best) / best * 100 if best else 0.0
        print(f"{case:<14} {best:10.2f} {best / N_SAMPLES:12.3f} {spread:8.0f}%")
    return 0


def cmd_save(args: argparse.Namespace) -> int:
    results = {}
    for case in args.cases:
        psi, elapsed = run(case, args.n, args.order)
        results[case] = psi
        print(f"{case:<14} {elapsed:8.2f} s  shape {psi.shape}")
    np.savez_compressed(args.out, n=args.n, **results)
    print(f"\nwrote {args.out}")
    return 0


def cmd_check(args: argparse.Namespace) -> int:
    ref = np.load(args.ref)
    n = int(ref["n"])
    print(f"comparing against {args.ref} (grid {n}x{n})\n")

    failed = False
    for case in args.cases:
        if case not in ref:
            print(f"{case:<14} SKIP   not in the reference file")
            continue
        psi, elapsed = run(case, n, args.order)
        expected = ref[case]
        if psi.shape != expected.shape:
            print(f"{case:<14} FAIL   shape {psi.shape} != {expected.shape}")
            failed = True
        elif np.array_equal(psi, expected):
            print(f"{case:<14} exact  ({elapsed:.2f} s)")
        else:
            # Report how far off it is, since that separates "a step was rejected differently"
            # from "the arithmetic is actually wrong"
            diff = np.abs(psi - expected)
            scale = np.abs(expected).max()
            print(
                f"{case:<14} DIFFERS  max |d| {diff.max():.3e} "
                f"({diff.max() / scale:.2e} relative)  ({elapsed:.2f} s)"
            )
            failed = True

    if failed:
        print(
            "\nA difference is only acceptable for a change that was expected to alter the\n"
            "arithmetic. For a redundancy-elimination change it means something else moved too."
        )
    return 1 if failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("time", "save", "check"))
    parser.add_argument("--n", type=int, default=256, help="grid points per axis (default 256)")
    parser.add_argument(
        "--cases",
        nargs="+",
        default=list(CASES),
        choices=list(CASES),
        help="which solvers to run (default: all)",
    )
    parser.add_argument(
        "--repeat", type=int, default=3, help="runs per case for 'time'; the best is reported"
    )
    parser.add_argument(
        "--order", type=int, default=2, choices=(2, 4),
        help="splitting order for the SSFM cases: 2 Strang, 4 Yoshida (default 2)",
    )
    parser.add_argument("--out", default="benchmarks/baseline.npz", help="for 'save'")
    parser.add_argument("--ref", default="benchmarks/baseline.npz", help="for 'check'")
    args = parser.parse_args()

    return {"time": cmd_time, "save": cmd_save, "check": cmd_check}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
