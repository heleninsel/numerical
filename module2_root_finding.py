# ============================================================
# MODULE 2: Root Finding (Equation Solving)
# Course: 155-4007 Numerical Methods in Engineering
# Student: Fatma Helen Insel - 24220030019
# ============================================================
# Real-world problem: Finding the operating point of a diode
# in an electrical circuit. The diode current is described by
# the Shockley equation: f(V) = I_s*(exp(V/V_t) - 1) - I_load = 0
# We need to find voltage V where this equation equals zero.
#
# Also demonstrates break-even point in a business simulation.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
import time

# ── Circuit / Diode parameters ──────────────────────────────
I_S  = 1e-12   # Saturation current (Amperes)
V_T  = 0.02585 # Thermal voltage at room temperature (Volts)
I_LOAD = 0.01  # Load current we want the diode to carry (Amperes)

def diode_equation(V):
    """
    Shockley diode equation rearranged to f(V) = 0.
    Root of this function = operating voltage of the diode.
    """
    return I_S * (np.exp(V / V_T) - 1) - I_LOAD

def diode_equation_derivative(V):
    """
    Analytical derivative of the diode equation.
    Required by the Newton-Raphson method.
    """
    return (I_S / V_T) * np.exp(V / V_T)

# ── Break-even function (business simulation) ───────────────
def breakeven_equation(x):
    """
    Break-even point: Revenue - Cost = 0
    Revenue = 50*x, Cost = 1000 + 20*x + 0.01*x^2
    Root = units to sell to break even.
    """
    revenue = 50 * x
    cost    = 1000 + 20 * x + 0.01 * x ** 2
    return revenue - cost


# ============================================================
# METHOD 1: Bisection Method
# ============================================================
def bisection(func, a, b, tol=1e-10, max_iter=1000):
    """
    Bisection method: repeatedly halves the interval [a, b]
    where the root lies (guaranteed by sign change).

    Convergence: slow but very robust — always converges
    as long as f(a) and f(b) have opposite signs.

    Parameters:
        func     : function to find root of
        a, b     : interval endpoints (must bracket the root)
        tol      : tolerance (stop when interval < tol)
        max_iter : maximum number of iterations

    Returns:
        root, iterations_used, error_history
    """
    # Validate that the root is actually bracketed
    if func(a) * func(b) > 0:
        raise ValueError("f(a) and f(b) must have opposite signs "
                         "(root not bracketed in this interval).")

    error_history = []

    for i in range(max_iter):
        midpoint = (a + b) / 2.0
        error    = abs(b - a) / 2.0
        error_history.append(error)

        if func(midpoint) == 0 or error < tol:
            return midpoint, i + 1, error_history

        # Keep the half that contains the sign change
        if func(a) * func(midpoint) < 0:
            b = midpoint
        else:
            a = midpoint

    return midpoint, max_iter, error_history


# ============================================================
# METHOD 2: Newton-Raphson Method
# ============================================================
def newton_raphson(func, dfunc, x0, tol=1e-10, max_iter=1000):
    """
    Newton-Raphson method: uses the derivative to jump directly
    toward the root. Much faster than bisection but can diverge
    if the initial guess is poor or derivative is near zero.

    Convergence: quadratic (very fast near the root).

    Parameters:
        func   : function to find root of
        dfunc  : derivative of func
        x0     : initial guess
        tol    : tolerance
        max_iter: maximum iterations

    Returns:
        root, iterations_used, error_history
    """
    x = x0
    error_history = []

    for i in range(max_iter):
        fx  = func(x)
        dfx = dfunc(x)

        # Safety check: avoid division by zero
        if abs(dfx) < 1e-15:
            raise ValueError(f"Derivative too close to zero at x={x:.6f}. "
                             "Newton-Raphson cannot continue.")

        x_new = x - fx / dfx
        error = abs(x_new - x)
        error_history.append(error)

        if error < tol:
            return x_new, i + 1, error_history

        x = x_new

    return x, max_iter, error_history


# ============================================================
# METHOD 3: Secant Method
# ============================================================
def secant(func, x0, x1, tol=1e-10, max_iter=1000):
    """
    Secant method: approximates the derivative using two points.
    Does NOT require an analytical derivative (unlike Newton-Raphson).
    Convergence: superlinear (~1.618, golden ratio order).

    Parameters:
        func     : function to find root of
        x0, x1   : two initial guesses (close to root)
        tol      : tolerance
        max_iter : maximum iterations

    Returns:
        root, iterations_used, error_history
    """
    error_history = []

    for i in range(max_iter):
        f0 = func(x0)
        f1 = func(x1)

        # Safety check: avoid division by zero
        if abs(f1 - f0) < 1e-15:
            raise ValueError("Secant method: division by zero risk "
                             "(f(x1) ≈ f(x0)). Try different initial guesses.")

        x2    = x1 - f1 * (x1 - x0) / (f1 - f0)
        error = abs(x2 - x1)
        error_history.append(error)

        if error < tol:
            return x2, i + 1, error_history

        x0, x1 = x1, x2

    return x1, max_iter, error_history


# ============================================================
# COMPARISON & VISUALIZATION
# ============================================================
def compare_methods(func, dfunc, a, b, x0_nr, x0_sec, x1_sec, label):
    """
    Runs all three methods on the same problem, compares
    convergence speed, accuracy and computation time.
    """
    print(f"\n{'=' * 60}")
    print(f"  ROOT FINDING: {label}")
    print(f"{'=' * 60}")

    results = {}

    # --- Bisection ---
    try:
        t0 = time.perf_counter()
        root_b, iters_b, errors_b = bisection(func, a, b)
        t1 = time.perf_counter()
        results['Bisection'] = {
            'root': root_b, 'iters': iters_b,
            'errors': errors_b, 'time': t1 - t0
        }
        print(f"\nBisection Method:")
        print(f"  Root found : {root_b:.10f}")
        print(f"  Iterations : {iters_b}")
        print(f"  Final error: {errors_b[-1]:.2e}")
        print(f"  Time       : {(t1-t0)*1000:.4f} ms")
    except Exception as e:
        print(f"  Bisection failed: {e}")

    # --- Newton-Raphson ---
    try:
        t0 = time.perf_counter()
        root_n, iters_n, errors_n = newton_raphson(func, dfunc, x0_nr)
        t1 = time.perf_counter()
        results['Newton-Raphson'] = {
            'root': root_n, 'iters': iters_n,
            'errors': errors_n, 'time': t1 - t0
        }
        print(f"\nNewton-Raphson Method:")
        print(f"  Root found : {root_n:.10f}")
        print(f"  Iterations : {iters_n}")
        print(f"  Final error: {errors_n[-1]:.2e}")
        print(f"  Time       : {(t1-t0)*1000:.4f} ms")
    except Exception as e:
        print(f"  Newton-Raphson failed: {e}")

    # --- Secant ---
    try:
        t0 = time.perf_counter()
        root_s, iters_s, errors_s = secant(func, x0_sec, x1_sec)
        t1 = time.perf_counter()
        results['Secant'] = {
            'root': root_s, 'iters': iters_s,
            'errors': errors_s, 'time': t1 - t0
        }
        print(f"\nSecant Method:")
        print(f"  Root found : {root_s:.10f}")
        print(f"  Iterations : {iters_s}")
        print(f"  Final error: {errors_s[-1]:.2e}")
        print(f"  Time       : {(t1-t0)*1000:.4f} ms")
    except Exception as e:
        print(f"  Secant failed: {e}")

    return results


def plot_convergence(results, title, filename):
    """
    Plots convergence speed of all three methods on a log scale.
    Fewer iterations to reach low error = faster convergence.
    """
    plt.figure(figsize=(10, 6))
    colors = {'Bisection': 'blue', 'Newton-Raphson': 'red', 'Secant': 'green'}

    for method, data in results.items():
        plt.semilogy(
            range(1, len(data['errors']) + 1),
            data['errors'],
            label=f"{method} ({data['iters']} iters)",
            color=colors.get(method, 'black'),
            linewidth=2
        )

    plt.xlabel("Iteration Number", fontsize=13)
    plt.ylabel("Absolute Error (log scale)", fontsize=13)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"Plot saved: {filename}")


def plot_diode_function():
    """
    Plots the diode equation f(V) to visually show where the root is.
    """
    V_range = np.linspace(0.3, 0.7, 500)
    f_values = [diode_equation(V) for V in V_range]

    plt.figure(figsize=(10, 5))
    plt.plot(V_range, f_values, color='darkorange', linewidth=2)
    plt.axhline(0, color='black', linewidth=1, linestyle='--')
    plt.xlabel("Voltage V (Volts)", fontsize=13)
    plt.ylabel("f(V) = I_s·exp(V/V_t) - I_s - I_load", fontsize=12)
    plt.title("Diode Equation: Finding Operating Voltage\n"
              "Root = voltage where f(V) = 0", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plot_module2_diode_function.png", dpi=150)
    plt.show()
    print("Plot saved: plot_module2_diode_function.png")


def run_module2():
    """
    Main function for Module 2.
    Solves two real-world root-finding problems and compares methods.
    """
    print("\n" + "=" * 60)
    print("  MODULE 2: ROOT FINDING — EQUATION SOLVING")
    print("=" * 60)

    # Problem 1: Diode circuit operating point
    plot_diode_function()
    results_diode = compare_methods(
        func    = diode_equation,
        dfunc   = diode_equation_derivative,
        a       = 0.3,    # lower bound (f(0.3) < 0)
        b       = 0.9,    # upper bound (f(0.9) > 0)
        x0_nr   = 0.6,    # Newton-Raphson initial guess
        x0_sec  = 0.5,    # Secant first point
        x1_sec  = 0.6,    # Secant second point
        label   = "Diode Circuit Operating Voltage"
    )
    plot_convergence(
        results_diode,
        title    = "Convergence Comparison — Diode Equation\n"
                   "Newton-Raphson converges fastest (quadratic)",
        filename = "plot_module2_convergence_diode.png"
    )

    # Problem 2: Business break-even point
    results_breakeven = compare_methods(
        func    = breakeven_equation,
        dfunc   = lambda x: 50 - 20 - 0.02 * x,  # d/dx of breakeven
        a       = 10,
        b       = 100,
        x0_nr   = 50.0,
        x0_sec  = 40.0,
        x1_sec  = 60.0,
        label   = "Business Break-Even Point (units sold)"
    )
    plot_convergence(
        results_breakeven,
        title    = "Convergence Comparison — Break-Even Equation",
        filename = "plot_module2_convergence_breakeven.png"
    )

    print("\n[OK] Module 2 complete.\n")


if __name__ == "__main__":
    run_module2()