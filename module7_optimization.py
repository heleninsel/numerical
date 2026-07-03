# ============================================================
# MODULE 7: Optimization Techniques
# Course: 155-4007 Numerical Methods in Engineering
# Student: Fatma Helen Insel - 24220030019
# ============================================================
# Real-world problems:
#   1. Minimizing production cost (scalar optimization)
#   2. Maximizing efficiency of a solar panel system
#      (multi-variable optimization)
#   3. Finding optimal beam dimensions (constrained)
#
# Methods used:
#   1. scipy.optimize.minimize_scalar  - single variable
#   2. scipy.optimize.minimize         - multi-variable
#   3. Manual gradient descent         - educational comparison
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar, minimize
import time

# ============================================================
# PROBLEM 1: Production Cost Minimization (Scalar)
# ============================================================
def production_cost(x):
    """
    Total production cost as a function of batch size x.
    Cost = fixed_cost/x + variable_cost*x + storage_cost*x^2
    This is the Economic Order Quantity (EOQ) problem.

    Parameters:
        x : batch size (units)

    Returns:
        total cost
    """
    fixed_cost    = 5000.0   # setup cost per batch
    variable_cost = 2.5      # cost per unit
    storage_cost  = 0.001    # storage cost per unit^2
    return fixed_cost / x + variable_cost * x + storage_cost * x**2


def solve_scalar_optimization():
    """
    Finds the batch size that minimizes total production cost.
    Uses scipy's minimize_scalar with bounded search.
    """
    print("\n" + "=" * 60)
    print("  PROBLEM 1: PRODUCTION COST MINIMIZATION")
    print("=" * 60)

    # Analytical solution for comparison:
    # d/dx [5000/x + 2.5x + 0.001x^2] = 0
    # -5000/x^2 + 2.5 + 0.002x = 0
    # Solved numerically here

    t0 = time.perf_counter()
    result = minimize_scalar(
        production_cost,
        bounds=(1, 5000),
        method='bounded'
    )
    t1 = time.perf_counter()

    print(f"\n  Optimal batch size : {result.x:.2f} units")
    print(f"  Minimum cost       : ${production_cost(result.x):.2f}")
    print(f"  Converged          : {result.success}")
    print(f"  Iterations         : {result.nit}")
    print(f"  Time               : {(t1-t0)*1000:.4f} ms")

    # Compare with a few manual guesses
    print(f"\n  Cost at x=100  : ${production_cost(100):.2f}")
    print(f"  Cost at x=500  : ${production_cost(500):.2f}")
    print(f"  Cost at x=1000 : ${production_cost(1000):.2f}")
    print(f"  Cost at x={result.x:.0f} (optimal): "
          f"${production_cost(result.x):.2f}")

    return result.x, result.fun


# ============================================================
# PROBLEM 2: Solar Panel Efficiency (Multi-variable)
# ============================================================
def solar_panel_efficiency(params):
    """
    Negative efficiency of a solar panel system as a function
    of panel tilt angle (degrees) and azimuth angle (degrees).
    We minimize the negative = maximize efficiency.

    Efficiency model:
        E = cos(tilt - optimal_tilt) * cos(azimuth - optimal_az)
        + noise term to make it non-trivial

    Parameters:
        params : [tilt_angle, azimuth_angle]

    Returns:
        negative efficiency (to minimize)
    """
    tilt    = params[0]
    azimuth = params[1]

    # Convert degrees to radians
    tilt_rad = np.radians(tilt)
    az_rad   = np.radians(azimuth)

    # Optimal angles for Adana, Turkey (latitude ~37 degrees)
    opt_tilt = np.radians(37)
    opt_az   = np.radians(180)  # south-facing

    efficiency = (np.cos(tilt_rad - opt_tilt) *
                  np.cos(az_rad   - opt_az) -
                  0.05 * np.sin(2 * tilt_rad) *
                  np.cos(az_rad))

    return -efficiency  # negative because we minimize


def solve_multivariable_optimization():
    """
    Finds optimal tilt and azimuth angles for maximum
    solar panel efficiency using scipy.optimize.minimize.
    Compares Nelder-Mead and BFGS methods.
    """
    print("\n" + "=" * 60)
    print("  PROBLEM 2: SOLAR PANEL ANGLE OPTIMIZATION")
    print("  (Location: Adana, Turkey - Latitude 37 deg)")
    print("=" * 60)

    x0      = [20.0, 150.0]  # initial guess: 20 deg tilt, 150 deg azimuth
    methods = ['Nelder-Mead', 'BFGS', 'Powell']
    results = {}

    for method in methods:
        try:
            t0  = time.perf_counter()
            res = minimize(
                solar_panel_efficiency,
                x0,
                method=method,
                options={'xatol': 1e-8, 'fatol': 1e-8,
                         'maxiter': 10000}
            )
            t1  = time.perf_counter()
            results[method] = res
            eff = -res.fun  # convert back to positive efficiency

            print(f"\n  Method: {method}")
            print(f"    Optimal tilt angle   : {res.x[0]:.4f} degrees")
            print(f"    Optimal azimuth angle: {res.x[1]:.4f} degrees")
            print(f"    Max efficiency       : {eff:.6f}")
            print(f"    Iterations           : {res.nit}")
            print(f"    Time                 : {(t1-t0)*1000:.4f} ms")
            print(f"    Converged            : {res.success}")
        except Exception as e:
            print(f"  {method} failed: {e}")

    return results


# ============================================================
# PROBLEM 3: Manual Gradient Descent (Educational)
# ============================================================
def gradient_descent(func, grad_func, x0, learning_rate=0.01,
                     max_iter=1000, tol=1e-8):
    """
    Manual gradient descent: moves in the direction of steepest
    descent (negative gradient) at each step.

    Educational purpose: shows HOW optimization algorithms work
    internally, before using scipy's black-box solvers.

    Parameters:
        func          : objective function to minimize
        grad_func     : gradient of objective function
        x0            : initial guess
        learning_rate : step size (alpha)
        max_iter      : maximum iterations
        tol           : convergence tolerance

    Returns:
        x_opt      : optimal x found
        cost_hist  : cost value at each iteration
        x_hist     : x value at each iteration
    """
    x         = float(x0)
    cost_hist = []
    x_hist    = [x]

    for i in range(max_iter):
        cost = func(x)
        cost_hist.append(cost)
        grad = grad_func(x)

        x_new = x - learning_rate * grad
        if abs(x_new - x) < tol:
            x = x_new
            cost_hist.append(func(x))
            break
        x = x_new
        x_hist.append(x)

    return x, cost_hist, x_hist


def production_cost_gradient(x):
    """
    Analytical gradient (derivative) of production_cost.
    d/dx [5000/x + 2.5x + 0.001x^2] = -5000/x^2 + 2.5 + 0.002x
    """
    return -5000.0 / x**2 + 2.5 + 0.002 * x


def compare_with_gradient_descent():
    """
    Compares scipy optimizer with manual gradient descent
    on the production cost problem.
    """
    print("\n" + "=" * 60)
    print("  GRADIENT DESCENT vs SCIPY OPTIMIZER COMPARISON")
    print("=" * 60)

    # Manual gradient descent
    t0 = time.perf_counter()
    x_gd, cost_hist, x_hist = gradient_descent(
        production_cost,
        production_cost_gradient,
        x0=100.0,
        learning_rate=50.0,
        max_iter=5000
    )
    t1 = time.perf_counter()

    print(f"\n  Gradient Descent:")
    print(f"    Optimal x    : {x_gd:.4f} units")
    print(f"    Min cost     : ${production_cost(x_gd):.4f}")
    print(f"    Iterations   : {len(cost_hist)}")
    print(f"    Time         : {(t1-t0)*1000:.4f} ms")

    # scipy minimize_scalar
    t0  = time.perf_counter()
    res = minimize_scalar(production_cost, bounds=(1, 5000), method='bounded')
    t1  = time.perf_counter()

    print(f"\n  scipy minimize_scalar:")
    print(f"    Optimal x    : {res.x:.4f} units")
    print(f"    Min cost     : ${res.fun:.4f}")
    print(f"    Iterations   : {res.nit}")
    print(f"    Time         : {(t1-t0)*1000:.4f} ms")

    return cost_hist, x_hist


# ============================================================
# VISUALIZATION
# ============================================================
def plot_production_cost(optimal_x):
    """Plots the production cost curve and marks the minimum."""
    x_vals    = np.linspace(10, 3000, 500)
    cost_vals = production_cost(x_vals)

    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, cost_vals, color='steelblue', linewidth=2,
             label='Total Production Cost')
    plt.axvline(x=optimal_x, color='crimson', linestyle='--',
                linewidth=2, label=f'Optimal x = {optimal_x:.1f}')
    plt.scatter([optimal_x], [production_cost(optimal_x)],
                color='crimson', s=100, zorder=5,
                label=f'Min cost = ${production_cost(optimal_x):.2f}')
    plt.xlabel("Batch Size (units)", fontsize=13)
    plt.ylabel("Total Cost ($)", fontsize=13)
    plt.title("Production Cost Minimization\n"
              "Finding optimal batch size using scipy.minimize_scalar",
              fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plot_module7_production_cost.png", dpi=150)
    plt.show()
    print("Plot saved: plot_module7_production_cost.png")


def plot_solar_efficiency():
    """
    Contour plot of solar panel efficiency vs tilt and azimuth.
    Shows the optimization landscape.
    """
    tilt_range    = np.linspace(0, 90, 200)
    azimuth_range = np.linspace(90, 270, 200)
    T, A          = np.meshgrid(tilt_range, azimuth_range)
    E             = np.array([[-solar_panel_efficiency([t, a])
                               for t in tilt_range]
                               for a in azimuth_range])

    plt.figure(figsize=(11, 7))
    cp = plt.contourf(T, A, E, levels=50, cmap='RdYlGn')
    plt.colorbar(cp, label='Efficiency')
    plt.xlabel("Tilt Angle (degrees)", fontsize=13)
    plt.ylabel("Azimuth Angle (degrees)", fontsize=13)
    plt.title("Solar Panel Efficiency Landscape\n"
              "Brighter = higher efficiency | "
              "Red dot = optimal angles found by optimizer",
              fontsize=13)

    # Mark optimal point (south-facing, latitude tilt)
    plt.scatter([37], [180], color='red', s=200, zorder=5,
                marker='*', label='Optimal (37 deg, 180 deg)')
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig("plot_module7_solar_efficiency.png", dpi=150)
    plt.show()
    print("Plot saved: plot_module7_solar_efficiency.png")


def plot_gradient_descent(cost_hist, x_hist):
    """Plots gradient descent convergence."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].semilogy(range(len(cost_hist)), cost_hist,
                      color='darkorange', linewidth=2)
    axes[0].set_xlabel("Iteration", fontsize=12)
    axes[0].set_ylabel("Cost (log scale)", fontsize=12)
    axes[0].set_title("Gradient Descent: Cost per Iteration",
                       fontsize=13)
    axes[0].grid(True, which='both', linestyle='--', alpha=0.6)

    x_plot   = np.linspace(50, 2000, 500)
    axes[1].plot(x_plot, production_cost(x_plot),
                  color='steelblue', linewidth=2, label='Cost function')
    axes[1].scatter(x_hist[::max(1, len(x_hist)//30)],
                     [production_cost(x) for x in
                      x_hist[::max(1, len(x_hist)//30)]],
                     color='crimson', s=20, zorder=5,
                     label='Gradient descent path')
    axes[1].set_xlabel("Batch Size x", fontsize=12)
    axes[1].set_ylabel("Cost", fontsize=12)
    axes[1].set_title("Gradient Descent Path on Cost Function",
                       fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("plot_module7_gradient_descent.png", dpi=150)
    plt.show()
    print("Plot saved: plot_module7_gradient_descent.png")


# ============================================================
# MAIN
# ============================================================
def run_module7():
    """
    Main function for Module 7.
    Demonstrates scalar and multi-variable optimization
    with real engineering examples.
    """
    print("\n" + "=" * 60)
    print("  MODULE 7: OPTIMIZATION TECHNIQUES")
    print("=" * 60)

    optimal_x, min_cost = solve_scalar_optimization()
    solve_multivariable_optimization()
    cost_hist, x_hist = compare_with_gradient_descent()

    plot_production_cost(optimal_x)
    plot_solar_efficiency()
    plot_gradient_descent(cost_hist, x_hist)

    print("\n[OK] Module 7 complete.\n")


if __name__ == "__main__":
    run_module7()