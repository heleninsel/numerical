# ============================================================
# MODULE 5: Numerical Integration
# Course: 155-4007 Numerical Methods in Engineering
# Student: Fatma Helen Insel - 24220030019
# ============================================================
# Real-world problems:
#   1. Computing distance from acceleration sensor data
#      (area under velocity-time curve)
#   2. Computing work done by a variable force
#
# Methods compared:
#   1. Trapezoidal Rule  - O(h^2) accuracy
#   2. Simpson's Rule    - O(h^4) accuracy, more accurate
#   3. scipy.integrate.quad - adaptive, most accurate
#   4. scipy.integrate.trapezoid - for discrete data
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import time

# ── Test function with known exact integral ──────────────────
# f(x) = sin(x) + 0.5*x^2  on [0, pi]
# Exact integral = [-cos(x) + x^3/6] from 0 to pi
# = (-cos(pi) + pi^3/6) - (-cos(0) + 0)
# = (1 + pi^3/6) - (-1) = 2 + pi^3/6

def f(x):
    """f(x) = sin(x) + 0.5*x^2"""
    return np.sin(x) + 0.5 * x**2

EXACT_INTEGRAL = 2 + (np.pi**3) / 6  # exact value for comparison

# ── Acceleration sensor data (simulated) ────────────────────
# A vehicle's velocity recorded every second
TIME_DATA = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # seconds
VEL_DATA  = np.array([0, 4.2, 8.1, 11.5, 13.8, 15.0,
                       14.2, 12.1, 8.9, 5.1, 1.2])          # m/s


# ============================================================
# METHOD 1: Trapezoidal Rule
# ============================================================
def trapezoidal_rule(func, a, b, n):
    """
    Trapezoidal Rule: approximates the area under a curve
    by dividing it into n trapezoids.

    Formula: h/2 * [f(a) + 2*f(x1) + 2*f(x2) + ... + f(b)]
    Error order: O(h^2) — second order accuracy.

    Parameters:
        func : integrand function
        a, b : integration limits
        n    : number of subintervals (more = more accurate)

    Returns:
        approximate integral value
    """
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)
    # First and last terms get weight 1, middle terms get weight 2
    result = h / 2 * (y[0] + 2 * np.sum(y[1:-1]) + y[-1])
    return result


# ============================================================
# METHOD 2: Simpson's Rule
# ============================================================
def simpsons_rule(func, a, b, n):
    """
    Simpson's Rule: uses quadratic polynomials instead of
    straight lines to approximate the area — more accurate
    than the trapezoidal rule for smooth functions.

    Formula: h/3 * [f(a) + 4*f(x1) + 2*f(x2) + 4*f(x3) + ... + f(b)]
    Error order: O(h^4) — fourth order accuracy.
    Requirement: n must be even.

    Parameters:
        func : integrand function
        a, b : integration limits
        n    : number of subintervals (MUST be even)

    Returns:
        approximate integral value
    """
    if n % 2 != 0:
        n += 1  # ensure n is even
        print(f"  [Note] n adjusted to {n} (must be even for Simpson's rule)")

    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = func(x)

    # Alternating weights: 1, 4, 2, 4, 2, ..., 4, 1
    weights      = np.ones(n + 1)
    weights[1:-1:2] = 4  # odd indices get weight 4
    weights[2:-2:2] = 2  # even indices get weight 2

    result = h / 3 * np.dot(weights, y)
    return result


# ============================================================
# METHOD 3: scipy.integrate.quad (adaptive)
# ============================================================
def scipy_quad_integration(func, a, b):
    """
    scipy's adaptive quadrature: automatically adjusts the
    step size to achieve high accuracy. Uses Gaussian
    quadrature with error estimation.
    Returns both the result and an error estimate.
    """
    result, error_estimate = integrate.quad(func, a, b)
    return result, error_estimate


# ============================================================
# CONVERGENCE ANALYSIS
# ============================================================
def convergence_analysis(a=0, b=np.pi):
    """
    Shows how accuracy improves as we increase n (number of intervals).
    Trapezoid error decreases as O(h^2), Simpson as O(h^4).
    """
    print("\n" + "=" * 60)
    print("  CONVERGENCE ANALYSIS (exact = {:.8f})".format(EXACT_INTEGRAL))
    print("=" * 60)
    print(f"  {'n':<8} {'Trapezoid':<16} {'Trap Error':<16} "
          f"{'Simpson':<16} {'Simp Error'}")
    print("  " + "-" * 72)

    n_values      = [4, 8, 16, 32, 64, 128, 256, 512]
    trap_errors   = []
    simp_errors   = []

    for n in n_values:
        trap = trapezoidal_rule(f, a, b, n)
        simp = simpsons_rule(f, a, b, n)
        trap_err = abs(trap - EXACT_INTEGRAL)
        simp_err = abs(simp - EXACT_INTEGRAL)
        trap_errors.append(trap_err)
        simp_errors.append(simp_err)
        print(f"  {n:<8} {trap:<16.8f} {trap_err:<16.2e} "
              f"{simp:<16.8f} {simp_err:.2e}")

    return n_values, trap_errors, simp_errors


# ============================================================
# DISCRETE DATA INTEGRATION (sensor data)
# ============================================================
def integrate_sensor_data():
    """
    Computes total distance traveled from velocity-time data
    using the trapezoidal rule on discrete sensor measurements.
    Area under v-t curve = displacement.
    """
    print("\n" + "=" * 60)
    print("  VEHICLE DISTANCE FROM VELOCITY SENSOR DATA")
    print("=" * 60)

    # scipy trapezoid works directly on discrete arrays
    distance_trap = integrate.trapezoid(VEL_DATA, TIME_DATA)

    # Manual trapezoidal on discrete data
    distance_manual = 0.0
    for i in range(len(TIME_DATA) - 1):
        dt = TIME_DATA[i+1] - TIME_DATA[i]
        distance_manual += 0.5 * (VEL_DATA[i] + VEL_DATA[i+1]) * dt

    print(f"\n  Total distance (scipy trapezoid) : {distance_trap:.4f} meters")
    print(f"  Total distance (manual trapezoid): {distance_manual:.4f} meters")
    print(f"  Peak velocity                    : {max(VEL_DATA):.1f} m/s "
          f"at t = {TIME_DATA[np.argmax(VEL_DATA)]} s")

    return distance_trap


# ============================================================
# FULL METHOD COMPARISON
# ============================================================
def compare_all_methods(a=0, b=np.pi, n=100):
    """
    Compares all integration methods at n=100 subintervals.
    """
    print("\n" + "=" * 60)
    print(f"  FULL METHOD COMPARISON (n={n}, a={a:.2f}, b={b:.4f})")
    print("=" * 60)
    print(f"  Exact integral value: {EXACT_INTEGRAL:.10f}")

    methods = {}

    t0 = time.perf_counter()
    trap = trapezoidal_rule(f, a, b, n)
    t1 = time.perf_counter()
    methods['Trapezoidal'] = {'result': trap, 'time': t1-t0}

    t0 = time.perf_counter()
    simp = simpsons_rule(f, a, b, n)
    t1 = time.perf_counter()
    methods['Simpson'] = {'result': simp, 'time': t1-t0}

    t0 = time.perf_counter()
    quad_result, quad_err = scipy_quad_integration(f, a, b)
    t1 = time.perf_counter()
    methods['scipy.quad'] = {'result': quad_result, 'time': t1-t0}

    print(f"\n  {'Method':<16} {'Result':<18} {'Error':<16} {'Time (ms)'}")
    print("  " + "-" * 62)
    for name, data in methods.items():
        err = abs(data['result'] - EXACT_INTEGRAL)
        print(f"  {name:<16} {data['result']:<18.10f} "
              f"{err:<16.2e} {data['time']*1000:.4f}")

    return methods


# ============================================================
# VISUALIZATION
# ============================================================
def plot_integration_methods(a=0, b=np.pi, n=8):
    """
    Visualizes how trapezoid and Simpson's rule approximate
    the area under the curve with n=8 subintervals.
    """
    x_fine = np.linspace(a, b, 500)
    y_fine = f(x_fine)
    x_pts  = np.linspace(a, b, n+1)
    y_pts  = f(x_pts)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Trapezoidal
    axes[0].plot(x_fine, y_fine, 'k-', linewidth=2, label='f(x)')
    axes[0].fill_between(x_pts, y_pts, alpha=0.3, color='steelblue',
                          step='post', label='Trapezoids')
    for i in range(n):
        axes[0].plot([x_pts[i], x_pts[i+1]], [y_pts[i], y_pts[i+1]],
                      'b-', linewidth=1)
    axes[0].scatter(x_pts, y_pts, color='blue', zorder=5, s=40)
    axes[0].set_title(f"Trapezoidal Rule (n={n})\nStraight lines between points",
                       fontsize=13)
    axes[0].set_xlabel("x", fontsize=12)
    axes[0].set_ylabel("f(x)", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Simpson's (show smooth parabolic arcs)
    axes[1].plot(x_fine, y_fine, 'k-', linewidth=2, label='f(x)')
    axes[1].fill_between(x_fine, y_fine, alpha=0.3, color='forestgreen',
                          label="Area (Simpson's)")
    axes[1].scatter(x_pts, y_pts, color='green', zorder=5, s=40)
    axes[1].set_title(f"Simpson's Rule (n={n})\nQuadratic arcs — more accurate",
                       fontsize=13)
    axes[1].set_xlabel("x", fontsize=12)
    axes[1].set_ylabel("f(x)", fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle("Numerical Integration: Trapezoidal vs Simpson's Rule",
                  fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plot_module5_methods.png", dpi=150)
    plt.show()
    print("Plot saved: plot_module5_methods.png")


def plot_convergence(n_values, trap_errors, simp_errors):
    """Plots how error decreases as n increases."""
    plt.figure(figsize=(10, 6))
    plt.loglog(n_values, trap_errors, 'o-', color='steelblue',
               linewidth=2, label='Trapezoidal O(h^2)')
    plt.loglog(n_values, simp_errors, 's-', color='forestgreen',
               linewidth=2, label="Simpson's O(h^4)")
    plt.xlabel("Number of Subintervals (n)", fontsize=13)
    plt.ylabel("Absolute Error (log scale)", fontsize=13)
    plt.title("Convergence of Numerical Integration Methods\n"
              "Simpson's rule converges much faster", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plot_module5_convergence.png", dpi=150)
    plt.show()
    print("Plot saved: plot_module5_convergence.png")


def plot_sensor_data():
    """Plots vehicle velocity and shaded area = distance."""
    plt.figure(figsize=(10, 5))
    plt.plot(TIME_DATA, VEL_DATA, 'o-', color='darkorange',
             linewidth=2, markersize=7, label='Velocity (m/s)')
    plt.fill_between(TIME_DATA, VEL_DATA, alpha=0.3,
                      color='darkorange', label='Area = Distance traveled')
    plt.xlabel("Time (seconds)", fontsize=13)
    plt.ylabel("Velocity (m/s)", fontsize=13)
    plt.title("Vehicle Distance from Velocity Sensor Data\n"
              "Numerical Integration: Area under v-t curve = displacement",
              fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plot_module5_sensor.png", dpi=150)
    plt.show()
    print("Plot saved: plot_module5_sensor.png")


# ============================================================
# MAIN
# ============================================================
def run_module5():
    """
    Main function for Module 5.
    Demonstrates numerical integration methods and comparisons.
    """
    print("\n" + "=" * 60)
    print("  MODULE 5: NUMERICAL INTEGRATION")
    print("=" * 60)
    print(f"\n  Integrating f(x) = sin(x) + 0.5*x^2 from 0 to pi")
    print(f"  Exact value: {EXACT_INTEGRAL:.10f}")

    compare_all_methods()
    n_values, trap_errors, simp_errors = convergence_analysis()
    integrate_sensor_data()

    plot_integration_methods()
    plot_convergence(n_values, trap_errors, simp_errors)
    plot_sensor_data()

    print("\n[OK] Module 5 complete.\n")


if __name__ == "__main__":
    run_module5()