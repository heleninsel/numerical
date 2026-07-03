# ============================================================
# MODULE 6: Linear Systems & LU Decomposition
# Course: 155-4007 Numerical Methods in Engineering
# Student: Fatma Helen Insel - 24220030019
# ============================================================
# Real-world problem: Heat distribution along a metal rod.
# We divide the rod into segments and write energy balance
# equations — this produces a system of linear equations
# A*x = b, where x = temperatures at each segment.
#
# Methods compared:
#   1. numpy.linalg.solve   - direct solver
#   2. LU Decomposition     - efficient for repeated solves
#   3. Iterative (Gauss-Seidel) - for large sparse systems
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import time

# ============================================================
# HEAT DISTRIBUTION PROBLEM SETUP
# ============================================================
def build_heat_system(n=10, T_left=100.0, T_right=20.0):
    """
    Builds the linear system A*x = b for 1D heat distribution.
    A metal rod has fixed temperatures at both ends.
    Interior points satisfy: T[i-1] - 2*T[i] + T[i+1] = 0
    (finite difference discretization of d^2T/dx^2 = 0)

    Parameters:
        n       : number of interior points
        T_left  : temperature at left end (Celsius)
        T_right : temperature at right end (Celsius)

    Returns:
        A : coefficient matrix (n x n)
        b : right-hand side vector (n,)
    """
    # Build tridiagonal matrix
    # Diagonal = -2, off-diagonals = 1
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = -2.0
        if i > 0:
            A[i, i-1] = 1.0
        if i < n - 1:
            A[i, i+1] = 1.0

    # Right-hand side: boundary conditions appear only at endpoints
    b = np.zeros(n)
    b[0]  = -T_left   # left boundary condition
    b[-1] = -T_right  # right boundary condition

    return A, b


# ============================================================
# METHOD 1: Direct Solve (numpy.linalg.solve)
# ============================================================
def direct_solve(A, b):
    """
    Direct solution using numpy's built-in solver.
    Uses LU decomposition internally (LAPACK routines).
    Best for one-time solves of moderate-sized systems.

    Returns:
        x    : solution vector
        time : computation time in seconds
    """
    t0 = time.perf_counter()
    x  = np.linalg.solve(A, b)
    t1 = time.perf_counter()
    return x, t1 - t0


# ============================================================
# METHOD 2: LU Decomposition (scipy)
# ============================================================
def lu_decomposition_solve(A, b_list):
    """
    LU Decomposition: factors A = P*L*U once, then solves
    A*x = b for multiple right-hand side vectors efficiently.

    Key advantage: When solving the same system with different
    b vectors (e.g., different boundary conditions), LU
    factorization is done ONCE and reused — much faster
    than repeated direct solves.

    Parameters:
        A      : coefficient matrix
        b_list : list of right-hand side vectors to solve

    Returns:
        solutions : list of solution vectors
        time_lu   : time for LU factorization
        time_solve: time for all back-substitutions
    """
    # Step 1: Factorize A = P*L*U (done once)
    t0 = time.perf_counter()
    lu_factor = linalg.lu_factor(A)
    t1 = time.perf_counter()
    time_lu = t1 - t0

    # Step 2: Solve for each b (cheap back-substitution)
    solutions  = []
    t0 = time.perf_counter()
    for b in b_list:
        x = linalg.lu_solve(lu_factor, b)
        solutions.append(x)
    t1 = time.perf_counter()
    time_solve = t1 - t0

    return solutions, time_lu, time_solve


# ============================================================
# METHOD 3: Gauss-Seidel Iterative Solver
# ============================================================
def gauss_seidel(A, b, tol=1e-10, max_iter=1000):
    """
    Gauss-Seidel iterative method: starts with an initial
    guess and iteratively improves the solution.

    Advantage: Memory-efficient for large sparse systems
               (does not need to store the full matrix).
    Disadvantage: May not converge for all matrices.
    Convergence guaranteed when A is diagonally dominant.

    Parameters:
        A        : coefficient matrix
        b        : right-hand side vector
        tol      : convergence tolerance
        max_iter : maximum iterations allowed

    Returns:
        x           : solution vector
        iterations  : number of iterations used
        error_hist  : error history per iteration
    """
    n          = len(b)
    x          = np.zeros(n)   # initial guess: all zeros
    error_hist = []

    for iteration in range(max_iter):
        x_old = x.copy()

        for i in range(n):
            # Sum all terms except diagonal
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            if abs(A[i, i]) < 1e-15:
                raise ValueError(f"Zero diagonal at row {i} — "
                                 "Gauss-Seidel cannot proceed.")
            x[i] = (b[i] - sigma) / A[i, i]

        # Check convergence: max change between iterations
        error = np.max(np.abs(x - x_old))
        error_hist.append(error)

        if error < tol:
            return x, iteration + 1, error_hist

    return x, max_iter, error_hist


# ============================================================
# PERFORMANCE COMPARISON: Direct vs LU for repeated solves
# ============================================================
def compare_direct_vs_lu(n=50, num_rhs=20):
    """
    Compares performance of direct solve vs LU decomposition
    when solving the same system with many different b vectors.
    LU factorizes once and reuses — much faster for many solves.
    """
    print("\n" + "=" * 60)
    print(f"  DIRECT vs LU: {num_rhs} right-hand sides, n={n}")
    print("=" * 60)

    A, b0 = build_heat_system(n)

    # Generate multiple b vectors (different boundary conditions)
    b_list = [b0 + np.random.randn(n) * 5 for _ in range(num_rhs)]

    # Direct solve repeated num_rhs times
    t0 = time.perf_counter()
    for b in b_list:
        np.linalg.solve(A, b)
    t1 = time.perf_counter()
    time_direct = t1 - t0

    # LU: factorize once, solve many times
    solutions, time_lu, time_solve = lu_decomposition_solve(A, b_list)
    time_lu_total = time_lu + time_solve

    print(f"\n  Direct solve ({num_rhs} times): {time_direct*1000:.4f} ms")
    print(f"  LU factorization (once)      : {time_lu*1000:.4f} ms")
    print(f"  LU back-substitution x{num_rhs}    : {time_solve*1000:.4f} ms")
    print(f"  LU total                     : {time_lu_total*1000:.4f} ms")

    if time_lu_total < time_direct:
        speedup = time_direct / time_lu_total
        print(f"\n  LU is {speedup:.2f}x FASTER than repeated direct solve")
    else:
        print(f"\n  Direct solve was faster (small n={n} case)")

    return time_direct, time_lu_total


# ============================================================
# VISUALIZATION
# ============================================================
def plot_heat_distribution(n=20):
    """
    Solves the heat distribution problem and plots the
    temperature profile along the rod.
    Also compares all three methods visually.
    """
    A, b = build_heat_system(n, T_left=100.0, T_right=20.0)

    # Solve with all three methods
    x_direct, _  = direct_solve(A, b)
    x_lu, _, _   = lu_decomposition_solve(A, [b])
    x_lu         = x_lu[0]
    x_gs, iters, error_hist = gauss_seidel(A, b)

    # Full rod including boundary points
    positions    = np.linspace(0, 1, n + 2)
    T_full_direct = np.concatenate([[100.0], x_direct, [20.0]])
    T_full_lu     = np.concatenate([[100.0], x_lu,     [20.0]])
    T_full_gs     = np.concatenate([[100.0], x_gs,     [20.0]])

    # Analytical solution: linear temperature profile
    T_analytical  = 100.0 + (20.0 - 100.0) * positions

    print("\n" + "=" * 60)
    print("  HEAT DISTRIBUTION RESULTS (n=20 interior points)")
    print("=" * 60)
    print(f"\n  Gauss-Seidel converged in {iters} iterations")
    print(f"\n  Max difference (Direct vs LU)      : "
          f"{np.max(np.abs(x_direct - x_lu)):.2e}")
    print(f"  Max difference (Direct vs G-Seidel): "
          f"{np.max(np.abs(x_direct - x_gs)):.2e}")

    # Plot temperature profile
    plt.figure(figsize=(11, 6))
    plt.plot(positions, T_analytical,  'k--',  linewidth=2,
             label='Analytical (linear)', zorder=5)
    plt.plot(positions, T_full_direct, 'o-',   color='steelblue',
             linewidth=2, markersize=4, label='Direct Solve')
    plt.plot(positions, T_full_lu,     's--',  color='forestgreen',
             linewidth=2, markersize=4, label='LU Decomposition')
    plt.plot(positions, T_full_gs,     '^:',   color='crimson',
             linewidth=2, markersize=4, label='Gauss-Seidel')
    plt.xlabel("Position along rod (normalized)", fontsize=13)
    plt.ylabel("Temperature (Celsius)", fontsize=13)
    plt.title("Heat Distribution in a Metal Rod\n"
              "All three methods produce identical results",
              fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plot_module6_heat_distribution.png", dpi=150)
    plt.show()
    print("Plot saved: plot_module6_heat_distribution.png")

    # Plot Gauss-Seidel convergence
    plt.figure(figsize=(10, 5))
    plt.semilogy(range(1, len(error_hist)+1), error_hist,
                 color='crimson', linewidth=2)
    plt.xlabel("Iteration", fontsize=13)
    plt.ylabel("Max Change (log scale)", fontsize=13)
    plt.title("Gauss-Seidel Convergence History\n"
              "Error decreases with each iteration",
              fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plot_module6_gauss_seidel_convergence.png", dpi=150)
    plt.show()
    print("Plot saved: plot_module6_gauss_seidel_convergence.png")


def plot_performance_comparison(n_values=None):
    """
    Shows how LU decomposition speedup grows with system size.
    """
    if n_values is None:
        n_values = [10, 20, 50, 100, 200]

    direct_times = []
    lu_times     = []
    num_rhs      = 30

    print("\n" + "=" * 60)
    print("  PERFORMANCE SCALING WITH SYSTEM SIZE")
    print("=" * 60)

    for n in n_values:
        A, b0   = build_heat_system(n)
        b_list  = [b0 + np.random.randn(n) * 5 for _ in range(num_rhs)]

        t0 = time.perf_counter()
        for b in b_list:
            np.linalg.solve(A, b)
        t1 = time.perf_counter()
        direct_times.append((t1 - t0) * 1000)

        _, time_lu, time_solve = lu_decomposition_solve(A, b_list)
        lu_times.append((time_lu + time_solve) * 1000)

        print(f"  n={n:<5}: Direct={direct_times[-1]:.3f} ms  "
              f"LU={lu_times[-1]:.3f} ms  "
              f"Speedup={direct_times[-1]/lu_times[-1]:.2f}x")

    plt.figure(figsize=(10, 6))
    plt.plot(n_values, direct_times, 'o-', color='steelblue',
             linewidth=2, label=f'Direct Solve x{num_rhs}')
    plt.plot(n_values, lu_times,     's-', color='forestgreen',
             linewidth=2, label=f'LU Decomposition x{num_rhs}')
    plt.xlabel("System Size (n)", fontsize=13)
    plt.ylabel("Total Time (ms)", fontsize=13)
    plt.title("Performance: Direct Solve vs LU Decomposition\n"
              "LU advantage grows with system size and number of solves",
              fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plot_module6_performance.png", dpi=150)
    plt.show()
    print("Plot saved: plot_module6_performance.png")


# ============================================================
# MAIN
# ============================================================
def run_module6():
    """
    Main function for Module 6.
    Solves heat distribution problem using three methods
    and compares direct vs LU decomposition performance.
    """
    print("\n" + "=" * 60)
    print("  MODULE 6: LINEAR SYSTEMS & LU DECOMPOSITION")
    print("=" * 60)

    plot_heat_distribution(n=20)
    compare_direct_vs_lu(n=50, num_rhs=20)
    plot_performance_comparison()

    print("\n[OK] Module 6 complete.\n")


if __name__ == "__main__":
    run_module6()