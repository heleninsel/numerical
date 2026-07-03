# ============================================================
# MODULE 1: Error Analysis and Floating Point Precision
# Course: 155-4007 Numerical Methods in Engineering
# Student: Fatma Helen Insel - 24220030019
# ============================================================
# This module demonstrates the natural limitations of computer
# arithmetic, including rounding errors and finite precision,
# and shows how errors can accumulate in iterative computations.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

def demonstrate_floating_point_limits():
    """
    Demonstrates that computers cannot represent all real numbers
    exactly due to finite memory (IEEE 754 double precision).
    """
    print("=" * 60)
    print("FLOATING POINT PRECISION LIMITS")
    print("=" * 60)

    # Machine epsilon: smallest number such that 1.0 + eps != 1.0
    machine_epsilon = np.finfo(float).eps
    print(f"Machine Epsilon (float64): {machine_epsilon:.2e}")
    # This means any number smaller than this gets lost when
    # added to 1.0 — it simply disappears due to rounding.

    # Classic example: 0.1 + 0.2 is NOT exactly 0.3 in binary
    a = 0.1
    b = 0.2
    c = 0.3
    print(f"\n0.1 + 0.2 = {a + b:.20f}")
    print(f"0.3       = {c:.20f}")
    print(f"Are they equal? {a + b == c}")
    print(f"Absolute difference: {abs((a + b) - c):.2e}")
    # Explanation: 0.1 and 0.2 have no exact binary representation.
    # The tiny difference (~2.8e-17) is a rounding error.

    # Catastrophic cancellation: subtracting two nearly equal numbers
    x = 1.000000001
    y = 1.000000000
    result = x - y
    print(f"\nCatastrophic Cancellation:")
    print(f"x = {x}, y = {y}")
    print(f"x - y = {result:.2e}  (expected: 1e-9)")
    # When two nearly equal numbers are subtracted, significant
    # digits cancel out, leaving only noisy low-order digits.


def demonstrate_error_propagation(n_iterations=50):
    """
    Shows how a tiny initial rounding error grows through
    repeated (iterative) calculations — error propagation.
    
    We compute x_n = (4/3)*x_{n-1} - (1/3)*x_{n-2}
    Analytically the answer should always be 1.0,
    but rounding errors accumulate and diverge.
    """
    print("\n" + "=" * 60)
    print("ERROR PROPAGATION IN ITERATIVE COMPUTATION")
    print("=" * 60)

    # True analytical values: x should stay at 1.0 forever
    x_prev2 = 1.0
    x_prev1 = 1.0 + np.finfo(float).eps  # tiny initial perturbation

    errors = []
    iterations = list(range(n_iterations))

    for i in iterations:
        x_new = (4.0 / 3.0) * x_prev1 - (1.0 / 3.0) * x_prev2
        error = abs(x_new - 1.0)  # true value is 1.0
        errors.append(error)
        x_prev2 = x_prev1
        x_prev1 = x_new

    print(f"Initial perturbation: {np.finfo(float).eps:.2e}")
    print(f"Error after {n_iterations} iterations: {errors[-1]:.2e}")
    print("→ A microscopic error grew to a significant value!")

    # Mitigation strategy: use higher precision (float128 if available)
    # or restructure the algorithm to reduce error accumulation.
    print("\nMitigation Strategy:")
    print("  - Use numerically stable algorithms (e.g., Kahan summation)")
    print("  - Avoid subtracting nearly equal numbers")
    print("  - Use higher precision types when needed (np.float128)")

    return iterations, errors


def demonstrate_rounding_in_summation():
    """
    Compares naive summation vs Kahan compensated summation.
    Kahan summation is a mitigation strategy that reduces
    accumulated rounding error in long sums.
    """
    print("\n" + "=" * 60)
    print("NAIVE vs KAHAN COMPENSATED SUMMATION")
    print("=" * 60)

    N = 1_000_000
    small_value = 0.1

    # Naive summation: errors accumulate with each addition
    naive_sum = 0.0
    for _ in range(N):
        naive_sum += small_value

    # Kahan compensated summation: tracks the lost bits
    kahan_sum = 0.0
    compensation = 0.0
    for _ in range(N):
        y = small_value - compensation
        t = kahan_sum + y
        compensation = (t - kahan_sum) - y
        kahan_sum = t

    true_value = N * small_value  # analytically: 100000.0
    naive_error = abs(naive_sum - true_value)
    kahan_error = abs(kahan_sum - true_value)

    print(f"True value:            {true_value:.6f}")
    print(f"Naive sum:             {naive_sum:.6f}  | Error: {naive_error:.2e}")
    print(f"Kahan compensated sum: {kahan_sum:.6f}  | Error: {kahan_error:.2e}")
    ratio = naive_error / kahan_error if kahan_error > 0 else float('inf')
    print(f"Kahan is {ratio:.0f}x more accurate" if kahan_error > 0 else "Kahan is infinitely more accurate (zero error)")


def plot_error_propagation(iterations, errors):
    """
    Visualizes how rounding errors grow over iterations.
    This plot will be included in the project report.
    """
    plt.figure(figsize=(10, 5))
    plt.semilogy(iterations, errors, color='crimson', linewidth=2)
    plt.xlabel("Iteration Number", fontsize=13)
    plt.ylabel("Absolute Error (log scale)", fontsize=13)
    plt.title("Error Propagation in Iterative Computation\n"
              "A microscopic initial rounding error grows exponentially",
              fontsize=14)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plot_module1_error_propagation.png", dpi=150)
    plt.show()
    print("\nPlot saved: plot_module1_error_propagation.png")


def run_module1():
    """
    Main function for Module 1.
    Runs all error analysis demonstrations.
    """
    print("\n" + "=" * 60)
    print("  MODULE 1: ERROR ANALYSIS & FLOATING POINT PRECISION")
    print("=" * 60)

    demonstrate_floating_point_limits()
    iterations, errors = demonstrate_error_propagation()
    demonstrate_rounding_in_summation()
    plot_error_propagation(iterations, errors)

    print("\n✓ Module 1 complete.\n")


# Run this module directly for testing
if __name__ == "__main__":
    run_module1()