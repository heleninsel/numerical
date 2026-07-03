# ============================================================
# MODULE 3: Interpolation Techniques
# Course: 155-4007 Numerical Methods in Engineering
# Student: Fatma Helen Insel - 24220030019
# ============================================================
# Real-world problem: A temperature sensor records data every
# 10 minutes. We use interpolation to estimate temperatures
# at every minute (filling the gaps between measurements).
#
# Methods compared:
#   1. Linear Interpolation       - simple, fast, less smooth
#   2. Cubic Spline Interpolation - smooth curves, more accurate
#   3. Polynomial Interpolation   - can oscillate (Runge's phenomenon)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

# Known sensor measurements (every 10 minutes)
TIME_POINTS = np.array([0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120])
TEMP_POINTS = np.array([15.0, 17.5, 21.3, 25.8, 28.1, 29.4, 28.7,
                         26.3, 23.1, 20.4, 18.9, 17.2, 16.5])

# Dense time axis for smooth curves (every ~0.12 min)
TIME_DENSE = np.linspace(0, 120, 1000)


# ============================================================
# METHOD 1: Linear Interpolation
# ============================================================
def linear_interpolation(x_known, y_known, x_query):
    """
    Connects each pair of known points with a straight line.
    Simple and fast but produces sharp corners at data points.
    """
    return np.interp(x_query, x_known, y_known)


# ============================================================
# METHOD 2: Cubic Spline Interpolation
# ============================================================
def cubic_spline_interpolation(x_known, y_known, x_query):
    """
    Fits smooth cubic polynomials between each pair of points.
    Ensures smooth transitions - best for physical phenomena
    like temperature, pressure, or velocity profiles.
    """
    cs = interpolate.CubicSpline(x_known, y_known)
    return cs(x_query)


# ============================================================
# METHOD 3: Polynomial Interpolation
# ============================================================
def polynomial_interpolation(x_known, y_known, x_query):
    """
    Fits a single high-degree polynomial through all points.
    Warning: causes wild oscillations with many points
    (Runge's Phenomenon) - shown here as a cautionary example.
    """
    coeffs = np.polyfit(x_known, y_known, deg=len(x_known) - 1)
    poly   = np.poly1d(coeffs)
    return poly(x_query)


# ============================================================
# ACCURACY EVALUATION
# ============================================================
def evaluate_accuracy():
    """
    Evaluates each method at the known data points.
    A perfect interpolation passes exactly through known points.
    """
    print("\nAccuracy at known data points:")

    linear     = linear_interpolation(TIME_POINTS, TEMP_POINTS, TIME_POINTS)
    spline_cs  = interpolate.CubicSpline(TIME_POINTS, TEMP_POINTS)
    spline     = spline_cs(TIME_POINTS)
    coeffs     = np.polyfit(TIME_POINTS, TEMP_POINTS, deg=len(TIME_POINTS) - 1)
    polynomial = np.poly1d(coeffs)(TIME_POINTS)

    for name, estimated in [("Linear    ", linear),
                             ("Spline    ", spline),
                             ("Polynomial", polynomial)]:
        errors  = np.abs(estimated - TEMP_POINTS)
        max_err = np.max(errors)
        avg_err = np.mean(errors)
        print(f"  {name}: Max error = {max_err:.6f} C  |  "
              f"Avg error = {avg_err:.6f} C")


# ============================================================
# VISUALIZATION - Separate plot for each method
# ============================================================
def plot_interpolation_comparison(linear, spline, polynomial):
    """
    Saves each interpolation method as a separate clean plot.
    """
    methods = [
        (linear,     'Linear Interpolation',
         'steelblue',   'Sharp corners at data points - not smooth',
         'plot_module3_linear.png'),
        (spline,     'Cubic Spline Interpolation',
         'forestgreen', 'Smooth curve - best for physical measurements',
         'plot_module3_spline.png'),
        (polynomial, 'Polynomial Interpolation (Lagrange)',
         'crimson',     "High-degree oscillation - Runge's Phenomenon",
         'plot_module3_polynomial.png'),
    ]

    for (y_interp, title, color, note, filename) in methods:
        plt.figure(figsize=(12, 6))
        plt.plot(TIME_DENSE, y_interp, color=color,
                 linewidth=2, label='Interpolated curve')
        plt.scatter(TIME_POINTS, TEMP_POINTS, color='black',
                    zorder=5, s=60, label='Sensor measurements')
        plt.xlabel("Time (minutes)", fontsize=12)
        plt.ylabel("Temperature (°C)", fontsize=12)
        plt.title(f"{title}\n{note}", fontsize=13)
        plt.legend(fontsize=11)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.ylim([min(TEMP_POINTS) - 5, max(TEMP_POINTS) + 5])
        plt.tight_layout()
        plt.savefig(filename, dpi=150)
        plt.show()
        print(f"Plot saved: {filename}")


def plot_interpolation_overlay(linear, spline, polynomial):
    """
    Overlays all three methods on one plot for direct comparison.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(TIME_DENSE, linear,     color='steelblue',
             linewidth=2, label='Linear', linestyle='--')
    plt.plot(TIME_DENSE, spline,     color='forestgreen',
             linewidth=2, label='Cubic Spline')
    plt.plot(TIME_DENSE, polynomial, color='crimson',
             linewidth=2, label='Polynomial (Lagrange)', linestyle=':')
    plt.scatter(TIME_POINTS, TEMP_POINTS, color='black',
                zorder=5, s=80, label='Sensor measurements')
    plt.xlabel("Time (minutes)", fontsize=13)
    plt.ylabel("Temperature (°C)", fontsize=13)
    plt.title("Interpolation Methods - Overlay Comparison\n"
              "Cubic Spline produces the smoothest and most realistic curve",
              fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plot_module3_overlay.png", dpi=150)
    plt.show()
    print("Plot saved: plot_module3_overlay.png")


# ============================================================
# MAIN
# ============================================================
def run_module3():
    """
    Main function for Module 3.
    Runs all interpolation methods and compares results.
    """
    print("\n" + "=" * 60)
    print("  MODULE 3: INTERPOLATION TECHNIQUES")
    print("=" * 60)
    print(f"\nKnown data points : {len(TIME_POINTS)} sensor readings")
    print(f"Time range        : 0 to 120 minutes")
    print(f"Interpolated at   : {len(TIME_DENSE)} points (every ~0.12 min)")

    # Compute all three interpolations on dense time axis
    linear     = linear_interpolation(TIME_POINTS, TEMP_POINTS, TIME_DENSE)
    spline     = cubic_spline_interpolation(TIME_POINTS, TEMP_POINTS, TIME_DENSE)
    polynomial = polynomial_interpolation(TIME_POINTS, TEMP_POINTS, TIME_DENSE)

    # Estimate temperature at t = 35 minutes (between measurements)
    t_query  = 35
    lin_est  = np.interp(t_query, TIME_POINTS, TEMP_POINTS)
    spl_est  = float(interpolate.CubicSpline(TIME_POINTS, TEMP_POINTS)(t_query))
    poly_est = float(np.poly1d(np.polyfit(TIME_POINTS, TEMP_POINTS,
                     deg=len(TIME_POINTS) - 1))(t_query))

    print(f"\nEstimated temperature at t = {t_query} minutes:")
    print(f"  Linear Interpolation : {lin_est:.4f} C")
    print(f"  Cubic Spline         : {spl_est:.4f} C")
    print(f"  Polynomial           : {poly_est:.4f} C")

    # Accuracy check at known points
    evaluate_accuracy()

    # Plots
    plot_interpolation_comparison(linear, spline, polynomial)
    plot_interpolation_overlay(linear, spline, polynomial)

    print("\n[OK] Module 3 complete.\n")


if __name__ == "__main__":
    run_module3()