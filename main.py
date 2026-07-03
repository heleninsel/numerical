# ============================================================
# MAIN - Numerical Methods in Engineering
# Course : 155-4007 Numerical Methods in Engineering
# Student: Fatma Helen Insel - 24220030019
# ============================================================
# This is the entry point of the project.
# It runs all 9 modules sequentially and prints a summary.
#
# Project Structure:
#   module1_error_analysis.py    - Floating point & error propagation
#   module2_root_finding.py      - Bisection, Newton-Raphson, Secant
#   module3_interpolation.py     - Linear, Cubic Spline, Polynomial
#   module4_differentiation.py   - Forward, Backward, Central difference
#   module5_integration.py       - Trapezoid, Simpson, scipy.quad
#   module6_linear_systems.py    - Direct, LU, Gauss-Seidel
#   module7_optimization.py      - Scalar, Multi-variable, Gradient descent
#   module8_ode.py               - RK45, RK23, DOP853, Euler
#   module9_performance.py       - Benchmarks, stability, error handling
# ============================================================

import time
import traceback

# Import all modules
from module1_error_analysis   import run_module1
from module2_root_finding     import run_module2
from module3_interpolation    import run_module3
from module4_differentiation  import run_module4
from module5_integration      import run_module5
from module6_linear_systems   import run_module6
from module7_optimization     import run_module7
from module8_ode              import run_module8
from module9_performance      import run_module9


def print_header():
    print("\n" + "=" * 65)
    print("  NUMERICAL METHODS IN ENGINEERING — FINAL PROJECT")
    print("  Course  : 155-4007")
    print("  Student : Fatma Helen Insel")
    print("  ID      : 24220030019")
    print("  Date    : 2026")
    print("=" * 65)
    print("\n  This project demonstrates 9 core topics in numerical")
    print("  methods applied to real-world engineering problems.")
    print("  Each module includes visualizations and comparisons.")
    print("=" * 65 + "\n")


def run_all_modules():
    """
    Runs all modules sequentially with timing and error handling.
    If a module fails, the error is caught and logged,
    and execution continues with the next module.
    """
    modules = [
        (1,  "Error Analysis & Floating Point",    run_module1),
        (2,  "Root Finding",                        run_module2),
        (3,  "Interpolation Techniques",            run_module3),
        (4,  "Numerical Differentiation",           run_module4),
        (5,  "Numerical Integration",               run_module5),
        (6,  "Linear Systems & LU Decomposition",  run_module6),
        (7,  "Optimization Techniques",             run_module7),
        (8,  "Ordinary Differential Equations",     run_module8),
        (9,  "Performance & Stability Analysis",    run_module9),
    ]

    results   = {}
    t_project = time.perf_counter()

    for num, name, func in modules:
        print(f"\n{'#' * 65}")
        print(f"  RUNNING MODULE {num}: {name}")
        print(f"{'#' * 65}")

        t0 = time.perf_counter()
        try:
            func()
            t1      = time.perf_counter()
            elapsed = t1 - t0
            results[num] = {
                'name'   : name,
                'status' : 'PASSED',
                'time'   : elapsed
            }
        except Exception as e:
            t1      = time.perf_counter()
            elapsed = t1 - t0
            results[num] = {
                'name'   : name,
                'status' : 'FAILED',
                'time'   : elapsed,
                'error'  : str(e)
            }
            print(f"\n  [ERROR] Module {num} failed: {e}")
            print(f"  Traceback:")
            traceback.print_exc()

    t_end = time.perf_counter()
    return results, t_end - t_project


def print_summary(results, total_time):
    """
    Prints a final summary table of all modules:
    status, time taken, and any errors.
    """
    print("\n" + "=" * 65)
    print("  PROJECT EXECUTION SUMMARY")
    print("=" * 65)
    print(f"  {'Module':<6} {'Name':<38} {'Status':<10} {'Time'}")
    print("  " + "-" * 60)

    passed = 0
    failed = 0

    for num, data in results.items():
        status = data['status']
        t      = data['time']
        name   = data['name'][:37]
        marker = "[OK]" if status == 'PASSED' else "[FAIL]"
        print(f"  {num:<6} {name:<38} {marker:<10} {t:.2f}s")

        if status == 'PASSED':
            passed += 1
        else:
            failed += 1
            print(f"         Error: {data.get('error', 'Unknown')[:55]}")

    print("  " + "-" * 60)
    print(f"  Total modules : {len(results)}")
    print(f"  Passed        : {passed}")
    print(f"  Failed        : {failed}")
    print(f"  Total time    : {total_time:.2f} seconds")
    print("=" * 65)

    if failed == 0:
        print("\n  All modules completed successfully.")
        print("  Project is ready for submission.")
    else:
        print(f"\n  {failed} module(s) need attention before submission.")

    print("=" * 65 + "\n")


def print_topics_covered():
    """
    Prints which required topics from the assignment are covered
    and which module covers each one.
    """
    print("\n" + "=" * 65)
    print("  REQUIRED TOPICS COVERAGE")
    print("=" * 65)

    topics = [
        ("1.  Error Analysis & Floating Point Precision",
         "Module 1 — machine epsilon, propagation, Kahan sum"),
        ("2.  Root Finding (Equation Solving)",
         "Module 2 — bisection, Newton-Raphson, secant"),
        ("3.  Interpolation Techniques",
         "Module 3 — linear, cubic spline, polynomial"),
        ("4.  Numerical Differentiation",
         "Module 4 — forward, backward, central difference"),
        ("5.  Numerical Integration",
         "Module 5 — trapezoid, Simpson, scipy.quad, adaptive"),
        ("6.  Linear Systems Solution",
         "Module 6 — numpy.linalg.solve, Gauss-Seidel"),
        ("7.  LU Decomposition",
         "Module 6 — scipy lu_factor / lu_solve, performance"),
        ("8.  Optimization Techniques",
         "Module 7 — minimize_scalar, minimize, gradient descent"),
        ("9.  ODE Solution",
         "Module 8 — RK45, RK23, DOP853, Euler comparison"),
        ("10. Performance & Stability Analysis",
         "Module 9 — benchmarks, condition number, try-except"),
        ("11. Visualization & Documentation",
         "All modules — matplotlib plots, inline comments"),
        ("12. Comparative Analysis & Case Studies",
         "All modules — method comparisons, real problems"),
    ]

    for topic, coverage in topics:
        print(f"\n  {topic}")
        print(f"    -> {coverage}")

    print("\n" + "=" * 65)
    print("  All 12 required topics are covered.")
    print("=" * 65 + "\n")


# ============================================================
# ENTRY POINT
# ============================================================
if __name__ == "__main__":
    print_header()
    print_topics_covered()

    input("\n  Press ENTER to start running all modules...\n")

    results, total_time = run_all_modules()
    print_summary(results, total_time)