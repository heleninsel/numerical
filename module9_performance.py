# ============================================================
# MODULE 9: Performance Analysis, Numerical Stability
#           and Error Handling
# Course: 155-4007 Numerical Methods in Engineering
# Student: Fatma Helen Insel - 24220030019
# ============================================================
# This module demonstrates:
#   1. Timing and benchmarking numerical algorithms
#   2. Numerical stability analysis (condition number)
#   3. Robust error handling with try-except blocks
#   4. Adaptive step size strategy
#   5. Convergence testing across methods
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import time

# ============================================================
# PART 1: Algorithm Timing & Benchmarking
# ============================================================
def benchmark_matrix_operations():
    """
    Benchmarks matrix operations at different sizes.
    Shows how computation time scales with problem size.
    Demonstrates why algorithm choice matters for large systems.
    """
    print("\n" + "=" * 60)
    print("  PART 1: MATRIX OPERATION BENCHMARKS")
    print("=" * 60)

    sizes       = [10, 50, 100, 200, 500]
    times_solve = []
    times_inv   = []
    times_lu    = []

    print(f"\n  {'Size':<8} {'linalg.solve':<18} "
          f"{'Matrix Inverse':<18} {'LU Factor'}")
    print("  " + "-" * 58)

    for n in sizes:
        A = np.random.randn(n, n) + n * np.eye(n)  # well-conditioned
        b = np.random.randn(n)

        # Method 1: Direct solve
        reps = max(1, 200 // n)
        t0 = time.perf_counter()
        for _ in range(reps):
            np.linalg.solve(A, b)
        t1 = time.perf_counter()
        t_solve = (t1 - t0) / reps * 1000
        times_solve.append(t_solve)

        # Method 2: Matrix inverse (generally NOT recommended)
        t0 = time.perf_counter()
        for _ in range(reps):
            np.linalg.inv(A) @ b
        t1 = time.perf_counter()
        t_inv = (t1 - t0) / reps * 1000
        times_inv.append(t_inv)

        # Method 3: LU decomposition
        t0 = time.perf_counter()
        for _ in range(reps):
            lu = linalg.lu_factor(A)
            linalg.lu_solve(lu, b)
        t1 = time.perf_counter()
        t_lu = (t1 - t0) / reps * 1000
        times_lu.append(t_lu)

        print(f"  {n:<8} {t_solve:<18.4f} {t_inv:<18.4f} {t_lu:.4f}  (ms)")

    return sizes, times_solve, times_inv, times_lu


# ============================================================
# PART 2: Numerical Stability — Condition Number
# ============================================================
def analyze_condition_number():
    """
    The condition number of a matrix measures how sensitive
    the solution is to small changes in the input.

    High condition number = ill-conditioned = numerically unstable.
    Rule of thumb: if cond(A) ~ 10^k, you lose k digits of accuracy.

    Demonstrates with well-conditioned vs ill-conditioned systems.
    """
    print("\n" + "=" * 60)
    print("  PART 2: NUMERICAL STABILITY — CONDITION NUMBER")
    print("=" * 60)

    # Well-conditioned system
    A_good = np.array([[4.0, 1.0],
                        [1.0, 3.0]])
    b_good = np.array([1.0, 2.0])
    cond_good = np.linalg.cond(A_good)
    x_good    = np.linalg.solve(A_good, b_good)

    # Ill-conditioned system (Hilbert matrix — notoriously unstable)
    n = 8
    A_hilbert = np.array([[1.0/(i+j+1) for j in range(n)]
                           for i in range(n)])
    b_hilbert = np.ones(n)
    cond_hilbert = np.linalg.cond(A_hilbert)
    x_hilbert    = np.linalg.solve(A_hilbert, b_hilbert)

    # Perturb b slightly and see how much solution changes
    epsilon       = 1e-6
    b_perturbed   = b_hilbert + epsilon * np.random.randn(n)
    x_perturbed   = np.linalg.solve(A_hilbert, b_perturbed)
    solution_change = np.linalg.norm(x_hilbert - x_perturbed)

    print(f"\n  Well-conditioned system:")
    print(f"    Condition number : {cond_good:.4f}")
    print(f"    Solution x       : {x_good}")
    print(f"    Interpretation   : Stable — small input change = small output change")

    print(f"\n  Ill-conditioned system (Hilbert matrix, n={n}):")
    print(f"    Condition number : {cond_hilbert:.4e}")
    print(f"    Input perturbation  : {epsilon:.2e}")
    print(f"    Solution change     : {solution_change:.4e}")
    print(f"    Interpretation   : UNSTABLE — tiny input change causes huge output change!")
    print(f"    Lost digits      : ~{int(np.log10(cond_hilbert))} decimal digits of accuracy")

    return cond_good, cond_hilbert


# ============================================================
# PART 3: Error Handling with try-except
# ============================================================
def robust_root_finder(func, a, b, tol=1e-10, max_iter=1000):
    """
    A robust bisection root finder with comprehensive error handling.
    Demonstrates how production-quality numerical code handles:
      - Invalid intervals
      - Non-bracketing inputs
      - Convergence failures
      - Singular/degenerate cases

    This is the kind of defensive programming required in
    real engineering software.
    """
    print("\n" + "=" * 60)
    print("  PART 3: ROBUST ERROR HANDLING IN NUMERICAL CODE")
    print("=" * 60)

    test_cases = [
        # (a, b, description, expected)
        (0.0,  np.pi, "Valid bracket for sin(x)",        "should converge"),
        (1.0,  2.0,   "No root in [1, 2] for sin(x)",   "should raise error"),
        (0.0,  0.0,   "Zero-length interval",             "should raise error"),
        (-1.0, 1.0,   "Valid bracket, root at x=0",      "should converge"),
        (0.5,  10.0,  "Valid bracket for sin(x) root",   "should converge"),
    ]

    for a_test, b_test, description, expected in test_cases:
        print(f"\n  Test: {description}")
        print(f"  Expected: {expected}")
        try:
            # Input validation
            if a_test == b_test:
                raise ValueError(f"Interval [a,b] has zero length: a=b={a_test}")

            fa = func(a_test)
            fb = func(b_test)

            if fa * fb > 0:
                raise ValueError(
                    f"f(a)={fa:.4f} and f(b)={fb:.4f} have the same sign. "
                    f"No root guaranteed in [{a_test}, {b_test}]."
                )

            # Bisection
            for i in range(max_iter):
                mid = (a_test + b_test) / 2.0
                fm  = func(mid)

                if abs(fm) < tol or (b_test - a_test) / 2 < tol:
                    print(f"  Result: Root found at x = {mid:.8f} "
                          f"in {i+1} iterations")
                    break
                if fa * fm < 0:
                    b_test = mid
                    fb     = fm
                else:
                    a_test = mid
                    fa     = fm
            else:
                raise RuntimeError(
                    f"Bisection did not converge in {max_iter} iterations."
                )

        except ValueError as ve:
            print(f"  ValueError caught: {ve}")
        except RuntimeError as re:
            print(f"  RuntimeError caught: {re}")
        except Exception as e:
            print(f"  Unexpected error: {type(e).__name__}: {e}")


# ============================================================
# PART 4: Adaptive Step Size Strategy
# ============================================================
def adaptive_integration(func, a, b, tol=1e-6):
    """
    Adaptive integration: automatically refines the step size
    in regions where the function changes rapidly.
    Compares adaptive vs fixed step size in terms of accuracy
    and number of function evaluations.

    Strategy: if the error estimate for a subinterval exceeds
    tolerance, split it further (recursively).
    """
    print("\n" + "=" * 60)
    print("  PART 4: ADAPTIVE STEP SIZE STRATEGY")
    print("=" * 60)

    def adaptive_simpson(f, a, b, tol, depth=0, max_depth=50):
        """Recursive adaptive Simpson's rule."""
        c      = (a + b) / 2
        h      = b - a
        fa, fb, fc = f(a), f(b), f(c)

        # Coarse estimate
        S1 = h / 6 * (fa + 4*fc + fb)

        # Refined estimate (two subintervals)
        d  = (a + c) / 2
        e  = (c + b) / 2
        fd, fe = f(d), f(e)
        S2 = h / 12 * (fa + 4*fd + 2*fc + 4*fe + fb)

        error_est = abs(S2 - S1)

        if depth >= max_depth or error_est < 15 * tol:
            return S2 + (S2 - S1) / 15  # Richardson extrapolation
        else:
            return (adaptive_simpson(f, a, c, tol/2, depth+1, max_depth) +
                    adaptive_simpson(f, c, b, tol/2, depth+1, max_depth))

    # Test function: rapidly oscillating
    def test_func(x):
        return np.sin(10 * x) * np.exp(-x)

    exact = (-10*np.cos(10*b)*np.exp(-b) + np.sin(10*b)*np.exp(-b) -
             (-10*np.cos(10*a)*np.exp(-a) + np.sin(10*a)*np.exp(-a))) / 101

    # Fixed step (n=100)
    from scipy.integrate import trapezoid as sci_trap
    x_fixed = np.linspace(a, b, 100)
    fixed_result = sci_trap(test_func(x_fixed), x_fixed)
    fixed_error  = abs(fixed_result - exact)

    # Adaptive
    adaptive_result = adaptive_simpson(test_func, a, b, tol)
    adaptive_error  = abs(adaptive_result - exact)

    print(f"\n  Integrating sin(10x)*exp(-x) from {a} to {b}")
    print(f"  Exact value       : {exact:.10f}")
    print(f"\n  Fixed step (n=100):")
    print(f"    Result : {fixed_result:.10f}")
    print(f"    Error  : {fixed_error:.2e}")
    print(f"\n  Adaptive Simpson:")
    print(f"    Result : {adaptive_result:.10f}")
    print(f"    Error  : {adaptive_error:.2e}")
    print(f"\n  Adaptive is {fixed_error/adaptive_error:.1f}x more accurate")


# ============================================================
# PART 5: Convergence Testing
# ============================================================
def convergence_testing():
    """
    Systematic convergence testing: verifies that our numerical
    methods achieve their theoretical convergence rates.
    Forward diff: O(h), Central diff: O(h^2), Simpson: O(h^4).
    """
    print("\n" + "=" * 60)
    print("  PART 5: CONVERGENCE RATE VERIFICATION")
    print("=" * 60)

    def f(x):       return np.sin(x) * np.exp(-0.1 * x)
    def df_exact(x): return np.cos(x)*np.exp(-0.1*x) - 0.1*np.sin(x)*np.exp(-0.1*x)

    x0     = 1.5
    exact  = df_exact(x0)
    h_vals = [0.1, 0.05, 0.01, 0.005, 0.001]

    print(f"\n  Derivative of sin(x)*exp(-0.1x) at x={x0}")
    print(f"  Exact value: {exact:.10f}")
    print(f"\n  {'h':<10} {'Fwd Error':<16} {'Fwd Rate':<12} "
          f"{'Cen Error':<16} {'Cen Rate'}")
    print("  " + "-" * 65)

    fwd_errors = []
    cen_errors = []

    for h in h_vals:
        fwd = (f(x0+h) - f(x0)) / h
        cen = (f(x0+h) - f(x0-h)) / (2*h)
        fwd_errors.append(abs(fwd - exact))
        cen_errors.append(abs(cen - exact))

    for i, h in enumerate(h_vals):
        fwd_rate = (np.log(fwd_errors[i-1]) - np.log(fwd_errors[i])) / \
                   (np.log(h_vals[i-1]) - np.log(h)) if i > 0 else 0
        cen_rate = (np.log(cen_errors[i-1]) - np.log(cen_errors[i])) / \
                   (np.log(h_vals[i-1]) - np.log(h)) if i > 0 else 0
        print(f"  {h:<10.3f} {fwd_errors[i]:<16.2e} "
              f"{fwd_rate:<12.2f} {cen_errors[i]:<16.2e} {cen_rate:.2f}")

    print(f"\n  Forward difference converges at rate ~1 (O(h))")
    print(f"  Central difference converges at rate ~2 (O(h^2))")


# ============================================================
# VISUALIZATION
# ============================================================
def plot_benchmarks(sizes, times_solve, times_inv, times_lu):
    """Plots matrix operation benchmark results."""
    plt.figure(figsize=(10, 6))
    plt.loglog(sizes, times_solve, 'o-', color='steelblue',
               linewidth=2, label='linalg.solve')
    plt.loglog(sizes, times_inv,   's-', color='crimson',
               linewidth=2, label='Matrix Inverse')
    plt.loglog(sizes, times_lu,    '^-', color='forestgreen',
               linewidth=2, label='LU Decomposition')
    plt.xlabel("Matrix Size n", fontsize=13)
    plt.ylabel("Time per solve (ms, log scale)", fontsize=13)
    plt.title("Algorithm Performance Benchmark\n"
              "Computation time vs matrix size",
              fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plot_module9_benchmark.png", dpi=150)
    plt.show()
    print("Plot saved: plot_module9_benchmark.png")


def plot_condition_numbers():
    """Visualizes condition number effect on solution sensitivity."""
    ns       = range(2, 12)
    cond_nums = []

    for n in ns:
        H = np.array([[1.0/(i+j+1) for j in range(n)] for i in range(n)])
        cond_nums.append(np.linalg.cond(H))

    plt.figure(figsize=(10, 5))
    plt.semilogy(list(ns), cond_nums, 'o-', color='crimson', linewidth=2)
    plt.xlabel("Hilbert Matrix Size n", fontsize=13)
    plt.ylabel("Condition Number (log scale)", fontsize=13)
    plt.title("Condition Number of Hilbert Matrix vs Size\n"
              "Larger n = more ill-conditioned = less accurate solution",
              fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plot_module9_condition_number.png", dpi=150)
    plt.show()
    print("Plot saved: plot_module9_condition_number.png")


# ============================================================
# MAIN
# ============================================================
def run_module9():
    """
    Main function for Module 9.
    Covers performance benchmarking, numerical stability,
    error handling, adaptive step size, and convergence testing.
    """
    print("\n" + "=" * 60)
    print("  MODULE 9: PERFORMANCE, STABILITY & ERROR HANDLING")
    print("=" * 60)

    sizes, t_solve, t_inv, t_lu = benchmark_matrix_operations()
    analyze_condition_number()
    robust_root_finder(np.sin, 0, np.pi)
    adaptive_integration(np.sin, 0, np.pi, tol=1e-8)
    convergence_testing()

    plot_benchmarks(sizes, t_solve, t_inv, t_lu)
    plot_condition_numbers()

    print("\n[OK] Module 9 complete.\n")


if __name__ == "__main__":
    run_module9()