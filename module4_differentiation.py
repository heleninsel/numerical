# ============================================================
# MODULE 4: Numerical Differentiation
# Course: 155-4007 Numerical Methods in Engineering
# Student: Fatma Helen Insel - 24220030019
# ============================================================
# Approximating derivatives using finite difference schemes.
# Real-world example: analyzing velocity and acceleration
# of an athlete from position-time data (sports performance).
#
# Methods compared:
#   1. Forward  Difference : uses f(x+h) - f(x)
#   2. Backward Difference : uses f(x) - f(x-h)
#   3. Central  Difference : uses f(x+h) - f(x-h)  <- most accurate
#
# Also analyzes how step size h affects accuracy.
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

# ── Test function with known analytical derivative ───────────
# f(x) = sin(x) + 0.5*x^2
# f'(x) = cos(x) + x        (exact derivative for comparison)

def f(x):
    """Test function: f(x) = sin(x) + 0.5*x^2"""
    return np.sin(x) + 0.5 * x**2

def f_exact_derivative(x):
    """Exact (analytical) derivative: f'(x) = cos(x) + x"""
    return np.cos(x) + x


# ── Athlete position data (simulated GPS measurements) ───────
# Position recorded every 0.5 seconds during a 100m sprint
TIME_DATA = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
                       3.5, 4.0, 4.5, 5.0, 5.5, 6.0])  # seconds
POS_DATA  = np.array([0.0, 1.2, 4.5, 9.8, 17.2, 26.1, 36.0,
                       46.5, 57.2, 67.8, 78.1, 88.0, 97.5])  # meters


# ============================================================
# METHOD 1: Forward Difference
# ============================================================
def forward_difference(func, x, h):
    """
    Forward difference approximation of f'(x).
    Formula: f'(x) ≈ [f(x+h) - f(x)] / h
    Error order: O(h) — first order accuracy.
    Uses only the current and next point.
    """
    return (func(x + h) - func(x)) / h


# ============================================================
# METHOD 2: Backward Difference
# ============================================================
def backward_difference(func, x, h):
    """
    Backward difference approximation of f'(x).
    Formula: f'(x) ≈ [f(x) - f(x-h)] / h
    Error order: O(h) — first order accuracy.
    Uses current and previous point.
    """
    return (func(x) - func(x - h)) / h


# ============================================================
# METHOD 3: Central Difference
# ============================================================
def central_difference(func, x, h):
    """
    Central difference approximation of f'(x).
    Formula: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
    Error order: O(h^2) — second order accuracy.
    More accurate than forward/backward for same step size.
    This is the preferred method in most engineering applications.
    """
    return (func(x + h) - func(x - h)) / (2 * h)


# ============================================================
# STEP SIZE ANALYSIS
# ============================================================
def analyze_step_size_effect(x_point=1.0):
    """
    Analyzes how step size h affects the accuracy of each method.
    Too large h: truncation error dominates.
    Too small h: round-off error dominates (cancellation).
    There is an optimal h in between.
    """
    print("\n" + "=" * 60)
    print("  STEP SIZE EFFECT ON ACCURACY")
    print("=" * 60)
    print(f"  Evaluating at x = {x_point}")
    print(f"  Exact derivative : {f_exact_derivative(x_point):.10f}")
    print()

    h_values     = np.logspace(-1, -14, 50)  # from 0.1 down to 1e-14
    errors_fwd   = []
    errors_bwd   = []
    errors_cen   = []
    exact        = f_exact_derivative(x_point)

    for h in h_values:
        errors_fwd.append(abs(forward_difference(f, x_point, h)  - exact))
        errors_bwd.append(abs(backward_difference(f, x_point, h) - exact))
        errors_cen.append(abs(central_difference(f, x_point, h)  - exact))

    # Find optimal h for each method
    opt_h_fwd = h_values[np.argmin(errors_fwd)]
    opt_h_cen = h_values[np.argmin(errors_cen)]

    print(f"  Optimal h for Forward  Difference: {opt_h_fwd:.2e}")
    print(f"  Optimal h for Central  Difference: {opt_h_cen:.2e}")
    print(f"  Central difference is more accurate at same h")

    return h_values, errors_fwd, errors_bwd, errors_cen


# ============================================================
# ATHLETE PERFORMANCE ANALYSIS
# ============================================================
def analyze_athlete_performance():
    """
    Computes velocity (1st derivative of position) and
    acceleration (2nd derivative) from athlete GPS data
    using central differences where possible, and
    forward/backward at the endpoints.
    """
    print("\n" + "=" * 60)
    print("  ATHLETE PERFORMANCE ANALYSIS")
    print("=" * 60)

    n        = len(TIME_DATA)
    dt       = TIME_DATA[1] - TIME_DATA[0]  # time step = 0.5 s
    velocity = np.zeros(n)
    accel    = np.zeros(n)

    for i in range(n):
        if i == 0:
            # Forward difference at the start
            velocity[i] = (POS_DATA[i+1] - POS_DATA[i]) / dt
        elif i == n - 1:
            # Backward difference at the end
            velocity[i] = (POS_DATA[i] - POS_DATA[i-1]) / dt
        else:
            # Central difference everywhere else (most accurate)
            velocity[i] = (POS_DATA[i+1] - POS_DATA[i-1]) / (2 * dt)

    for i in range(n):
        if i == 0:
            accel[i] = (POS_DATA[i+2] - 2*POS_DATA[i+1] + POS_DATA[i]) / dt**2
        elif i == n - 1:
            accel[i] = (POS_DATA[i] - 2*POS_DATA[i-1] + POS_DATA[i-2]) / dt**2
        else:
            # Central second difference
            accel[i] = (POS_DATA[i+1] - 2*POS_DATA[i] + POS_DATA[i-1]) / dt**2

    print(f"\n  {'Time(s)':<10} {'Position(m)':<15} "
          f"{'Velocity(m/s)':<17} {'Accel(m/s2)'}")
    print("  " + "-" * 57)
    for i in range(n):
        print(f"  {TIME_DATA[i]:<10.1f} {POS_DATA[i]:<15.1f} "
              f"{velocity[i]:<17.4f} {accel[i]:.4f}")

    peak_v = np.max(velocity)
    peak_t = TIME_DATA[np.argmax(velocity)]
    print(f"\n  Peak velocity : {peak_v:.4f} m/s at t = {peak_t:.1f} s")

    return velocity, accel


# ============================================================
# COMPARISON AT FIXED h
# ============================================================
def compare_methods_at_fixed_h(h=0.01):
    """
    Compares all three methods at a fixed step size h=0.01
    across a range of x values. Shows central difference
    is consistently more accurate.
    """
    print("\n" + "=" * 60)
    print(f"  METHOD COMPARISON AT FIXED h = {h}")
    print("=" * 60)
    print(f"  {'x':<8} {'Exact':<14} {'Forward':<14} "
          f"{'Backward':<14} {'Central'}")
    print("  " + "-" * 65)

    x_values = np.linspace(0.5, 3.0, 6)
    for x in x_values:
        exact = f_exact_derivative(x)
        fwd   = forward_difference(f, x, h)
        bwd   = backward_difference(f, x, h)
        cen   = central_difference(f, x, h)
        print(f"  {x:<8.2f} {exact:<14.8f} {fwd:<14.8f} "
              f"{bwd:<14.8f} {cen:.8f}")


# ============================================================
# VISUALIZATION
# ============================================================
def plot_step_size_analysis(h_values, errors_fwd, errors_bwd, errors_cen):
    """Plots error vs step size for all three methods."""
    plt.figure(figsize=(10, 6))
    plt.loglog(h_values, errors_fwd, color='steelblue',
               linewidth=2, label='Forward Difference O(h)')
    plt.loglog(h_values, errors_bwd, color='darkorange',
               linewidth=2, label='Backward Difference O(h)', linestyle='--')
    plt.loglog(h_values, errors_cen, color='forestgreen',
               linewidth=2, label='Central Difference O(h^2)')
    plt.xlabel("Step Size h", fontsize=13)
    plt.ylabel("Absolute Error (log scale)", fontsize=13)
    plt.title("Effect of Step Size on Numerical Differentiation Accuracy\n"
              "Central difference achieves lower error for same h",
              fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plot_module4_step_size.png", dpi=150)
    plt.show()
    print("Plot saved: plot_module4_step_size.png")


def plot_athlete_analysis(velocity, accel):
    """Plots position, velocity and acceleration of the athlete."""
    fig, axes = plt.subplots(3, 1, figsize=(11, 10))
    plt.subplots_adjust(hspace=0.45)

    axes[0].plot(TIME_DATA, POS_DATA, 'o-', color='steelblue',
                 linewidth=2, markersize=6)
    axes[0].set_ylabel("Position (m)", fontsize=12)
    axes[0].set_title("Athlete Sprint Analysis using Numerical Differentiation",
                      fontsize=13)
    axes[0].grid(True, linestyle='--', alpha=0.6)

    axes[1].plot(TIME_DATA, velocity, 's-', color='forestgreen',
                 linewidth=2, markersize=6)
    axes[1].set_ylabel("Velocity (m/s)", fontsize=12)
    axes[1].set_title("Velocity = 1st Derivative of Position (Central Difference)",
                      fontsize=12)
    axes[1].grid(True, linestyle='--', alpha=0.6)

    axes[2].plot(TIME_DATA, accel, '^-', color='crimson',
                 linewidth=2, markersize=6)
    axes[2].set_ylabel("Acceleration (m/s^2)", fontsize=12)
    axes[2].set_xlabel("Time (seconds)", fontsize=12)
    axes[2].set_title("Acceleration = 2nd Derivative of Position",
                      fontsize=12)
    axes[2].grid(True, linestyle='--', alpha=0.6)

    plt.savefig("plot_module4_athlete.png", dpi=150)
    plt.show()
    print("Plot saved: plot_module4_athlete.png")


# ============================================================
# MAIN
# ============================================================
def run_module4():
    """
    Main function for Module 4.
    Demonstrates numerical differentiation with real examples.
    """
    print("\n" + "=" * 60)
    print("  MODULE 4: NUMERICAL DIFFERENTIATION")
    print("=" * 60)

    compare_methods_at_fixed_h(h=0.01)
    h_values, errors_fwd, errors_bwd, errors_cen = analyze_step_size_effect()
    velocity, accel = analyze_athlete_performance()

    plot_step_size_analysis(h_values, errors_fwd, errors_bwd, errors_cen)
    plot_athlete_analysis(velocity, accel)

    print("\n[OK] Module 4 complete.\n")


if __name__ == "__main__":
    run_module4()