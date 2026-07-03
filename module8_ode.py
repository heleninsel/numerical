# ============================================================
# MODULE 8: Ordinary Differential Equations (ODE)
# Course: 155-4007 Numerical Methods in Engineering
# Student: Fatma Helen Insel - 24220030019
# ============================================================
# Real-world problems:
#   1. Simple pendulum motion (nonlinear ODE)
#   2. Projectile motion with air resistance
#
# Solvers compared:
#   1. RK45  - Runge-Kutta 4th/5th order (default, accurate)
#   2. RK23  - Runge-Kutta 2nd/3rd order (faster, less accurate)
#   3. DOP853 - 8th order (most accurate, slowest)
#   4. Euler - manual implementation (educational baseline)
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time

# ============================================================
# PROBLEM 1: Simple Pendulum
# ============================================================
# d^2(theta)/dt^2 + (g/L)*sin(theta) = 0
# Rewritten as system:
#   y[0] = theta        (angle)
#   y[1] = d(theta)/dt  (angular velocity)
#   dy[0]/dt = y[1]
#   dy[1]/dt = -(g/L)*sin(y[0])

G = 9.81   # gravitational acceleration (m/s^2)
L = 1.0    # pendulum length (meters)

def pendulum_ode(t, y):
    """
    ODE system for a simple pendulum.
    y[0] = angle (radians), y[1] = angular velocity (rad/s)
    Returns derivatives [dy0/dt, dy1/dt].
    """
    theta     = y[0]
    omega     = y[1]
    dtheta_dt = omega
    domega_dt = -(G / L) * np.sin(theta)
    return [dtheta_dt, domega_dt]


def pendulum_ode_linear(t, y):
    """
    Linearized pendulum ODE (small angle approximation).
    sin(theta) ≈ theta for small angles.
    Analytical solution: theta(t) = theta0 * cos(sqrt(g/L)*t)
    Used for accuracy comparison.
    """
    theta     = y[0]
    omega     = y[1]
    dtheta_dt = omega
    domega_dt = -(G / L) * theta  # linearized
    return [dtheta_dt, domega_dt]


# ============================================================
# PROBLEM 2: Projectile with Air Resistance
# ============================================================
# Equations of motion:
#   dx/dt  = vx
#   dy/dt  = vy
#   dvx/dt = -k * v * vx      (drag in x)
#   dvy/dt = -g - k * v * vy  (gravity + drag in y)
# where v = sqrt(vx^2 + vy^2), k = drag coefficient

MASS = 0.5    # kg
DRAG = 0.05   # drag coefficient (kg/m)

def projectile_ode(t, y):
    """
    ODE system for projectile with air resistance.
    y = [x, y_pos, vx, vy]
    Returns derivatives [dx/dt, dy/dt, dvx/dt, dvy/dt].
    """
    x, y_pos, vx, vy = y
    v     = np.sqrt(vx**2 + vy**2)  # total speed
    k     = DRAG / MASS              # drag per unit mass

    dx_dt  = vx
    dy_dt  = vy
    dvx_dt = -k * v * vx
    dvy_dt = -G - k * v * vy

    return [dx_dt, dy_dt, dvx_dt, dvy_dt]


def projectile_ode_no_drag(t, y):
    """
    Projectile ODE without air resistance (for comparison).
    Analytical solution exists: x = v0*cos(a)*t,
    y = v0*sin(a)*t - 0.5*g*t^2
    """
    x, y_pos, vx, vy = y
    return [vx, vy, 0, -G]


# ============================================================
# EULER METHOD (manual, educational)
# ============================================================
def euler_method(ode_func, y0, t_span, dt):
    """
    Simple Euler method: the most basic ODE solver.
    y_{n+1} = y_n + dt * f(t_n, y_n)

    First-order accuracy O(dt). Used here as educational
    baseline to show why higher-order methods (RK45) are better.

    Parameters:
        ode_func : ODE function f(t, y)
        y0       : initial conditions
        t_span   : (t_start, t_end)
        dt       : fixed time step

    Returns:
        t_arr : time array
        y_arr : solution array
    """
    t_start, t_end = t_span
    t_arr = np.arange(t_start, t_end + dt, dt)
    y_arr = np.zeros((len(t_arr), len(y0)))
    y_arr[0] = y0

    for i in range(1, len(t_arr)):
        t_curr      = t_arr[i-1]
        y_curr      = y_arr[i-1]
        derivatives = ode_func(t_curr, y_curr)
        y_arr[i]    = y_curr + dt * np.array(derivatives)

    return t_arr, y_arr


# ============================================================
# SOLVER COMPARISON
# ============================================================
def compare_pendulum_solvers():
    """
    Solves the pendulum ODE with multiple scipy solvers
    and the manual Euler method. Compares accuracy and speed.
    """
    print("\n" + "=" * 60)
    print("  PENDULUM ODE - SOLVER COMPARISON")
    print("=" * 60)

    # Initial conditions: 30 degree initial angle, at rest
    theta0 = np.radians(30)   # 30 degrees in radians
    y0     = [theta0, 0.0]    # [initial angle, initial angular velocity]
    t_span = (0, 10)          # simulate 10 seconds
    t_eval = np.linspace(0, 10, 1000)

    # Analytical solution (small angle approximation)
    omega_n     = np.sqrt(G / L)
    theta_exact = theta0 * np.cos(omega_n * t_eval)

    solvers = ['RK45', 'RK23', 'DOP853']
    results = {}

    for solver in solvers:
        t0  = time.perf_counter()
        sol = solve_ivp(
            pendulum_ode,
            t_span,
            y0,
            method=solver,
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10
        )
        t1  = time.perf_counter()

        # Compare nonlinear solution vs linearized analytical
        sol_linear = solve_ivp(
            pendulum_ode_linear,
            t_span,
            y0,
            method=solver,
            t_eval=t_eval,
            rtol=1e-8,
            atol=1e-10
        )

        max_diff = np.max(np.abs(
            sol.y[0] - sol_linear.y[0]
        ))

        results[solver] = {
            'sol'     : sol,
            'time'    : t1 - t0,
            'max_diff': max_diff
        }

        print(f"\n  Solver: {solver}")
        print(f"    Time steps used : {len(sol.t)}")
        print(f"    Computation time: {(t1-t0)*1000:.4f} ms")
        print(f"    Max diff vs linearized: {max_diff:.6f} rad")
        print(f"    Success         : {sol.success}")

    # Euler method comparison
    print(f"\n  Solver: Euler (manual, dt=0.01)")
    t0         = time.perf_counter()
    t_eu, y_eu = euler_method(pendulum_ode, y0, t_span, dt=0.01)
    t1         = time.perf_counter()
    print(f"    Time steps used : {len(t_eu)}")
    print(f"    Computation time: {(t1-t0)*1000:.4f} ms")

    return results, t_eval, theta_exact, t_eu, y_eu, y0


def solve_projectile():
    """
    Solves projectile motion with and without air resistance.
    Shows how drag significantly changes the trajectory.
    """
    print("\n" + "=" * 60)
    print("  PROJECTILE MOTION WITH/WITHOUT AIR RESISTANCE")
    print("=" * 60)

    # Launch conditions
    v0    = 50.0              # initial speed (m/s)
    angle = np.radians(45)   # launch angle
    vx0   = v0 * np.cos(angle)
    vy0   = v0 * np.sin(angle)
    y0    = [0.0, 0.0, vx0, vy0]  # [x, y, vx, vy]

    # Simulate until projectile hits the ground (y = 0)
    def hit_ground(t, y): return y[1]  # y position
    hit_ground.terminal  = True
    hit_ground.direction = -1

    # With air resistance (RK45)
    t0   = time.perf_counter()
    sol_drag = solve_ivp(
        projectile_ode,
        (0, 20),
        y0,
        method='RK45',
        events=hit_ground,
        max_step=0.01,
        rtol=1e-8
    )
    t1   = time.perf_counter()
    time_drag = t1 - t0

    # Without air resistance (RK45)
    t0       = time.perf_counter()
    sol_nodrag = solve_ivp(
        projectile_ode_no_drag,
        (0, 20),
        y0,
        method='RK45',
        events=hit_ground,
        max_step=0.01,
        rtol=1e-8
    )
    t1         = time.perf_counter()
    time_nodrag = t1 - t0

    # Analytical solution (no drag)
    t_analytical  = np.linspace(0, sol_nodrag.t[-1], 500)
    x_analytical  = vx0 * t_analytical
    y_analytical  = vy0 * t_analytical - 0.5 * G * t_analytical**2

    print(f"\n  Launch: v0={v0} m/s, angle=45 degrees")
    print(f"\n  WITH air resistance (RK45):")
    print(f"    Range          : {sol_drag.y[0,-1]:.2f} m")
    print(f"    Max height     : {np.max(sol_drag.y[1]):.2f} m")
    print(f"    Flight time    : {sol_drag.t[-1]:.2f} s")
    print(f"    Compute time   : {time_drag*1000:.4f} ms")

    print(f"\n  WITHOUT air resistance (RK45):")
    print(f"    Range          : {sol_nodrag.y[0,-1]:.2f} m")
    print(f"    Max height     : {np.max(sol_nodrag.y[1]):.2f} m")
    print(f"    Flight time    : {sol_nodrag.t[-1]:.2f} s")
    print(f"    Compute time   : {time_nodrag*1000:.4f} ms")

    range_reduction = (1 - sol_drag.y[0,-1] / sol_nodrag.y[0,-1]) * 100
    print(f"\n  Air resistance reduces range by: {range_reduction:.1f}%")

    return sol_drag, sol_nodrag, x_analytical, y_analytical


# ============================================================
# VISUALIZATION
# ============================================================
def plot_pendulum(results, t_eval, theta_exact, t_eu, y_eu, y0):
    """Plots pendulum angle over time for all solvers."""
    theta0 = y0[0]

    plt.figure(figsize=(12, 6))

    colors = {'RK45': 'steelblue', 'RK23': 'darkorange', 'DOP853': 'purple'}
    for solver, data in results.items():
        plt.plot(t_eval, np.degrees(data['sol'].y[0]),
                 label=f'{solver}', color=colors[solver],
                 linewidth=2)

    plt.plot(t_eu, np.degrees(y_eu[:, 0]),
             label='Euler (dt=0.01)', color='crimson',
             linewidth=1.5, linestyle=':')
    plt.plot(t_eval, np.degrees(theta_exact),
             label='Analytical (linear approx)', color='black',
             linewidth=1.5, linestyle='--')

    plt.xlabel("Time (seconds)", fontsize=13)
    plt.ylabel("Pendulum Angle (degrees)", fontsize=13)
    plt.title("Simple Pendulum: ODE Solver Comparison\n"
              "RK45/DOP853 agree closely; Euler accumulates error",
              fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plot_module8_pendulum.png", dpi=150)
    plt.show()
    print("Plot saved: plot_module8_pendulum.png")


def plot_projectile(sol_drag, sol_nodrag, x_analytical, y_analytical):
    """Plots projectile trajectories with and without drag."""
    plt.figure(figsize=(12, 6))

    plt.plot(sol_nodrag.y[0], sol_nodrag.y[1],
             color='steelblue', linewidth=2,
             label='No air resistance (RK45)')
    plt.plot(x_analytical, y_analytical,
             color='black', linewidth=1.5, linestyle='--',
             label='No drag (analytical)')
    plt.plot(sol_drag.y[0], sol_drag.y[1],
             color='crimson', linewidth=2,
             label='With air resistance (RK45)')

    plt.xlabel("Horizontal Distance (m)", fontsize=13)
    plt.ylabel("Height (m)", fontsize=13)
    plt.title("Projectile Motion: With vs Without Air Resistance\n"
              "Air resistance significantly reduces range and height",
              fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(bottom=0)
    plt.tight_layout()
    plt.savefig("plot_module8_projectile.png", dpi=150)
    plt.show()
    print("Plot saved: plot_module8_projectile.png")


def plot_solver_performance(results):
    """Bar chart comparing solver computation times."""
    solvers = list(results.keys()) + ['Euler']
    times   = [results[s]['time'] * 1000 for s in results.keys()]
    times.append(None)  # Euler time not stored here, skip

    solvers = list(results.keys())
    times   = [results[s]['time'] * 1000 for s in solvers]
    colors  = ['steelblue', 'darkorange', 'purple']

    plt.figure(figsize=(8, 5))
    bars = plt.bar(solvers, times, color=colors, edgecolor='black',
                   linewidth=1.2)
    for bar, t in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.01,
                 f'{t:.3f} ms', ha='center', fontsize=12)
    plt.ylabel("Computation Time (ms)", fontsize=13)
    plt.title("ODE Solver Performance Comparison\n"
              "Pendulum problem, 1000 time points",
              fontsize=14)
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig("plot_module8_solver_performance.png", dpi=150)
    plt.show()
    print("Plot saved: plot_module8_solver_performance.png")


# ============================================================
# MAIN
# ============================================================
def run_module8():
    """
    Main function for Module 8.
    Simulates pendulum and projectile using ODE solvers,
    compares RK45, RK23, DOP853 and manual Euler method.
    """
    print("\n" + "=" * 60)
    print("  MODULE 8: ORDINARY DIFFERENTIAL EQUATIONS (ODE)")
    print("=" * 60)

    results, t_eval, theta_exact, t_eu, y_eu, y0 = \
        compare_pendulum_solvers()
    sol_drag, sol_nodrag, x_an, y_an = solve_projectile()

    plot_pendulum(results, t_eval, theta_exact, t_eu, y_eu, y0)
    plot_projectile(sol_drag, sol_nodrag, x_an, y_an)
    plot_solver_performance(results)

    print("\n[OK] Module 8 complete.\n")


if __name__ == "__main__":
    run_module8()