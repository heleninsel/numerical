# ============================================================
# GUI APPLICATION - Numerical Methods in Engineering
# Course : 155-4007 Numerical Methods in Engineering
# Student: Fatma Helen Insel - 24220030019
# ============================================================

import tkinter as tk
from tkinter import ttk, scrolledtext
import numpy as np
from scipy import integrate, interpolate, linalg, optimize
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import math
import random

# ── Cyberpunk Palette ─────────────────────────────────────────
BG_VOID     = "#050510"
BG_PANEL    = "#0a0a1f"
BG_CARD     = "#0f0f2e"
NEON_CYAN   = "#00f5ff"
NEON_PINK   = "#ff006e"
NEON_GREEN  = "#00ff9f"
NEON_YELLOW = "#ffe600"
NEON_PURPLE = "#bf5fff"
DIM_CYAN    = "#003d40"
TEXT_BRIGHT = "#e0f7ff"
TEXT_DIM    = "#4a6a7a"
GRID_LINE   = "#0d1f2d"


# ============================================================
# MATRIX RAIN CANVAS
# ============================================================
class MatrixRain(tk.Canvas):
    """Falling katakana/number characters — matrix rain effect."""
    CHARS = "01アイウエオカキクケコサシスセソタチツテトナニヌネノ∑∫∂∇λπ"

    def __init__(self, parent, height=55):
        super().__init__(parent, height=height, bg=BG_VOID,
                          highlightthickness=0)
        self.pack(fill="x")
        self._cols   = []
        self._width  = 0
        self._height = height
        self._phase  = 0
        self.bind("<Configure>", self._on_resize)
        self._animate()

    def _on_resize(self, e):
        self._width = e.width
        col_w = 16
        n = max(1, self._width // col_w)
        self._cols = [
            {
                "x":      i * col_w + 8,
                "y":      random.randint(-200, 0),
                "speed":  random.uniform(0.4, 1.2),
                "chars":  [random.choice(self.CHARS) for _ in range(20)],
                "bright": random.randint(0, 19),
            }
            for i in range(n)
        ]

    def _animate(self):
        self.delete("rain")
        for col in self._cols:
            col["y"] += col["speed"]
            if col["y"] > self._height + 200:
                col["y"] = random.randint(-200, -20)
                col["speed"] = random.uniform(0.4, 1.2)
                col["chars"] = [random.choice(self.CHARS)
                                  for _ in range(20)]

            for j, ch in enumerate(col["chars"]):
                y_pos = col["y"] - j * 12
                if not (0 < y_pos < self._height):
                    continue
                # Head = bright white, tail fades to dark green
                if j == 0:
                    color = "#ffffff"
                elif j < 3:
                    color = NEON_GREEN
                else:
                    fade = max(0, 120 - j * 9)
                    color = f"#00{fade:02x}{fade//3:02x}"
                self.create_text(
                    col["x"], y_pos, text=ch,
                    font=("Courier", 8), fill=color,
                    tags="rain"
                )
        self.after(40, self._animate)


# ============================================================
# ANIMATED CYBER HEADER
# ============================================================
class CyberHeader(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG_VOID)
        self.pack(fill="x")

        # Matrix rain background
        self._rain = MatrixRain(self, height=50)

        # Title overlay (on top of rain)
        title_bar = tk.Frame(self, bg=BG_VOID, height=36)
        title_bar.pack(fill="x")

        self._title_var = tk.StringVar(
            value="NUMERICAL METHODS IN ENGINEERING")
        tk.Label(
            title_bar,
            textvariable=self._title_var,
            font=("Courier", 15, "bold"),
            bg=BG_VOID, fg=NEON_CYAN
        ).pack(side="left", padx=20, pady=6)

        tk.Label(
            title_bar,
            text="155-4007  ·  Fatma Helen Insel  ·  24220030019",
            font=("Courier", 9),
            bg=BG_VOID, fg=NEON_PURPLE
        ).pack(side="right", padx=20, pady=10)

        # Neon border
        border = tk.Frame(self, bg=NEON_CYAN, height=1)
        border.pack(fill="x")

        self._phase = 0
        self._pulse_border(border)

    def _pulse_border(self, border):
        self._phase += 0.08
        r = int(0 + 0   * abs(math.sin(self._phase)))
        g = int(200 + 55 * abs(math.sin(self._phase)))
        b = int(200 + 55 * abs(math.cos(self._phase)))
        border.config(bg=f"#{r:02x}{g:02x}{b:02x}")
        self.after(50, lambda: self._pulse_border(border))


# ============================================================
# LIVE COUNTER WIDGET
# ============================================================
class LiveCounter(tk.Label):
    """Animates a number from 0 to target value."""
    def __init__(self, parent, label="", color=NEON_CYAN,
                 fmt="{:.4f}", **kwargs):
        super().__init__(
            parent,
            text=f"{label}: —",
            font=("Courier", 9, "bold"),
            bg=BG_PANEL, fg=color,
            **kwargs
        )
        self._label = label
        self._fmt   = fmt
        self._color = color

    def animate_to(self, target, steps=30, delay=20, fmt=None):
        """Counts up from 0 to target over `steps` frames."""
        use_fmt = fmt if fmt is not None else self._fmt
        def _step(current, step_n):
            val = target * (step_n / steps)
            self.config(
                text=f"{self._label}: {use_fmt.format(val)}",
                fg=NEON_YELLOW if step_n < steps else self._color
            )
            if step_n < steps:
                self.after(delay, lambda: _step(current, step_n+1))
        _step(0, 1)


# ============================================================
# NEON BUTTON
# ============================================================
class NeonButton(tk.Button):
    def __init__(self, parent, text, color=NEON_CYAN,
                 command=None, **kwargs):
        super().__init__(
            parent, text=text,
            font=("Courier", 10, "bold"),
            bg=BG_VOID, fg=color,
            activebackground=BG_CARD,
            activeforeground=color,
            relief="flat", bd=0,
            cursor="hand2", command=command,
            highlightthickness=1,
            highlightbackground=color,
            highlightcolor=color,
            padx=8, pady=5, **kwargs
        )
        self._color = color
        self.bind("<Enter>", lambda e: self.config(
            bg=self._hex_dim(color)))
        self.bind("<Leave>", lambda e: self.config(bg=BG_VOID))

    @staticmethod
    def _hex_dim(h):
        r=int(h[1:3],16)//4
        g=int(h[3:5],16)//4
        b=int(h[5:7],16)//4
        return f"#{r:02x}{g:02x}{b:02x}"


# ============================================================
# HELPERS
# ============================================================
def cyber_label(parent, text):
    tk.Label(parent, text=f"▸ {text}",
              bg=BG_PANEL, fg=TEXT_DIM,
              font=("Courier", 8)
              ).pack(anchor="w", padx=14, pady=(5,0))

def cyber_entry(parent, default=""):
    f = tk.Frame(parent, bg=NEON_CYAN, pady=1)
    f.pack(fill="x", padx=14, pady=(1,0))
    e = tk.Entry(f, bg=BG_CARD, fg=NEON_GREEN,
                  insertbackground=NEON_GREEN,
                  font=("Courier", 10), relief="flat", bd=4)
    e.insert(0, default)
    e.pack(fill="x")
    return e

def cyber_section(parent, text):
    tk.Label(parent, text=f"══ {text} ══",
              bg=BG_PANEL, fg=NEON_CYAN,
              font=("Courier", 8, "bold")
              ).pack(anchor="w", padx=10, pady=(10,2))

def style_axis(ax, fig):
    ax.set_facecolor(BG_VOID)
    fig.patch.set_facecolor(BG_CARD)
    ax.tick_params(colors=TEXT_DIM, labelsize=8)
    ax.xaxis.label.set_color(TEXT_DIM)
    ax.yaxis.label.set_color(TEXT_DIM)
    ax.title.set_color(NEON_CYAN)
    ax.title.set_fontsize(10)
    for spine in ax.spines.values():
        spine.set_edgecolor(DIM_CYAN)
    ax.grid(True, color=GRID_LINE, linestyle="--",
             linewidth=0.5, alpha=0.7)


# ============================================================
# BASE TAB  (with tab-flash on switch + live counters)
# ============================================================
class BaseTab(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent, bg=BG_VOID)
        self._flash_count = 0
        self._build_layout()

    def flash(self):
        """Called when this tab is selected — neon flash effect."""
        self._flash_count = 0
        self._do_flash()

    def _do_flash(self):
        colors = [NEON_CYAN, BG_VOID, NEON_CYAN, BG_VOID, BG_VOID]
        if self._flash_count < len(colors):
            self.config(bg=colors[self._flash_count])
            self._flash_count += 1
            self.after(60, self._do_flash)

    def _build_layout(self):
        self.left = tk.Frame(self, bg=BG_PANEL, width=265,
                              highlightthickness=1,
                              highlightbackground=DIM_CYAN)
        self.left.pack(side="left", fill="y", padx=(6,3), pady=6)
        self.left.pack_propagate(False)

        self.right = tk.Frame(self, bg=BG_VOID)
        self.right.pack(side="left", fill="both",
                         expand=True, padx=(3,6), pady=6)

        self.fig = plt.Figure(figsize=(7, 4.2), dpi=90)
        self.fig.patch.set_facecolor(BG_CARD)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.right)
        self.canvas.get_tk_widget().configure(
            bg=BG_CARD, highlightthickness=0)
        self.canvas.get_tk_widget().pack(
            fill="both", expand=True, pady=(0,3))

        cf = tk.Frame(self.right, bg=NEON_CYAN, pady=1)
        cf.pack(fill="x")
        self.console = scrolledtext.ScrolledText(
            cf, height=8, bg=BG_VOID, fg=NEON_GREEN,
            font=("Courier", 8), insertbackground=NEON_GREEN,
            relief="flat", bd=0, selectbackground=DIM_CYAN)
        self.console.pack(fill="x")

        # Live counters strip
        self.counter_frame = tk.Frame(self.left, bg=BG_PANEL)
        self.counter_frame.pack(fill="x", padx=8, pady=(4,0))

        self._build_params()

        btn_f = tk.Frame(self.left, bg=BG_PANEL)
        btn_f.pack(fill="x", padx=10, pady=(8,6))
        NeonButton(btn_f, "▶  RUN", color=NEON_GREEN,
                    command=self._run_threaded).pack(fill="x", pady=(0,4))
        NeonButton(btn_f, "✕  CLEAR", color=NEON_PINK,
                    command=self._clear).pack(fill="x")

    def _build_params(self): pass

    def _add_counter(self, label, color=NEON_CYAN, fmt="{:.4f}"):
        c = LiveCounter(self.counter_frame, label=label,
                         color=color, fmt=fmt)
        c.pack(anchor="w", padx=4, pady=1)
        return c

    def _run_threaded(self):
        threading.Thread(target=self._safe_run, daemon=True).start()

    def _safe_run(self):
        try:
            self._run()
        except Exception as e:
            self._log(f"\n[ERR] {type(e).__name__}: {e}\n")

    def _run(self): pass

    def _clear(self):
        self.console.delete("1.0", tk.END)
        self.fig.clear()
        self.canvas.draw()

    def _log(self, text):
        self.console.insert(tk.END, text)
        self.console.see(tk.END)

    def _draw(self):
        self.fig.tight_layout(pad=1.5)
        self.canvas.draw()


# ============================================================
# TAB 1: Error Analysis
# ============================================================
class ErrorAnalysisTab(BaseTab):
    def _build_params(self):
        cyber_section(self.left, "M1: ERROR ANALYSIS")
        tk.Label(self.left,
                  text="Floating point limits,\nerror propagation,\nKahan summation.",
                  bg=BG_PANEL, fg=TEXT_DIM,
                  font=("Courier", 8), justify="left"
                  ).pack(anchor="w", padx=14, pady=3)
        cyber_label(self.left, "Iterations (propagation):")
        self.e_iters = cyber_entry(self.left, "50")
        cyber_label(self.left, "N (Kahan test):")
        self.e_n = cyber_entry(self.left, "1000000")

        self.c_eps    = self._add_counter("Machine ε", NEON_CYAN,  "{:.2e}")
        self.c_err    = self._add_counter("Final err", NEON_PINK,  "{:.2e}")
        self.c_kahan  = self._add_counter("Kahan err", NEON_GREEN, "{:.2e}")

    def _run(self):
        n_iters = int(self.e_iters.get())
        N       = int(self.e_n.get())
        self._log("\n◈ MODULE 1: ERROR ANALYSIS\n" + "─"*38+"\n")

        eps = np.finfo(float).eps
        self.c_eps.animate_to(eps, fmt="{:.2e}")
        self._log(f"Machine epsilon   : {eps:.2e}\n")
        self._log(f"0.1+0.2 == 0.3    : {0.1+0.2==0.3}\n")
        self._log(f"Difference        : {abs(0.1+0.2-0.3):.2e}\n\n")

        x0, x1 = 1.0, 1.0 + eps
        errors  = []
        for _ in range(n_iters):
            xn = (4/3)*x1 - (1/3)*x0
            errors.append(abs(xn - 1.0))
            x0, x1 = x1, xn
        self.c_err.animate_to(errors[-1], fmt="{:.2e}")
        self._log(f"Error after {n_iters} iters: {errors[-1]:.2e}\n")

        ks, c = 0.0, 0.0
        for _ in range(N):
            y=0.1-c; t=ks+y; c=(t-ks)-y; ks=t
        naive = sum(0.1 for _ in range(N))
        true_v = N * 0.1
        self.c_kahan.animate_to(abs(ks-true_v), fmt="{:.2e}")
        self._log(f"\nKahan (N={N:,}):\n")
        self._log(f"  Naive error : {abs(naive-true_v):.2e}\n")
        self._log(f"  Kahan error : {abs(ks-true_v):.2e}\n")

        self.fig.clear()
        ax = self.fig.add_subplot(111)
        ax.semilogy(range(len(errors)), errors,
                     color=NEON_CYAN, linewidth=1.5)
        ax.fill_between(range(len(errors)), errors,
                         alpha=0.12, color=NEON_CYAN)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Absolute Error")
        ax.set_title("ERROR PROPAGATION IN ITERATIVE COMPUTATION")
        style_axis(ax, self.fig)
        self._draw()
        self._log("\n[DONE]\n")


# ============================================================
# TAB 2: Root Finding
# ============================================================
class RootFindingTab(BaseTab):
    def _build_params(self):
        cyber_section(self.left, "M2: ROOT FINDING")
        tk.Label(self.left,
                  text="Bisection · Newton-Raphson · Secant\nDiode circuit equation.",
                  bg=BG_PANEL, fg=TEXT_DIM,
                  font=("Courier", 8), justify="left"
                  ).pack(anchor="w", padx=14, pady=3)
        cyber_label(self.left, "Interval left a:")
        self.e_a   = cyber_entry(self.left, "0.3")
        cyber_label(self.left, "Interval right b:")
        self.e_b   = cyber_entry(self.left, "0.9")
        cyber_label(self.left, "Newton initial guess x0:")
        self.e_x0  = cyber_entry(self.left, "0.6")
        cyber_label(self.left, "Tolerance:")
        self.e_tol = cyber_entry(self.left, "1e-10")

        self.c_root  = self._add_counter("Root V",    NEON_CYAN,  "{:.8f}")
        self.c_bis   = self._add_counter("Bisect itr",NEON_PINK,  "{:.0f}")
        self.c_nwt   = self._add_counter("Newton itr",NEON_GREEN, "{:.0f}")

    def _run(self):
        a=float(self.e_a.get()); b=float(self.e_b.get())
        x0=float(self.e_x0.get()); tol=float(self.e_tol.get())
        IS,VT,IL=1e-12,0.02585,0.01
        def f(V):  return IS*(np.exp(V/VT)-1)-IL
        def df(V): return (IS/VT)*np.exp(V/VT)

        self._log("\n◈ MODULE 2: ROOT FINDING\n"+"─"*38+"\n")

        aa,bb=a,b; errs_b=[]
        for i in range(10000):
            mid=(aa+bb)/2; errs_b.append(abs(bb-aa)/2)
            if f(aa)*f(mid)<0: bb=mid
            else: aa=mid
            if errs_b[-1]<tol: break
        bis_iters=i+1

        x=x0; errs_n=[]
        for i in range(10000):
            xn=x-f(x)/df(x); errs_n.append(abs(xn-x)); x=xn
            if errs_n[-1]<tol: break
        nwt_iters=i+1

        x0s,x1s=a,b; errs_s=[]
        for i in range(10000):
            x2=x1s-f(x1s)*(x1s-x0s)/(f(x1s)-f(x0s))
            errs_s.append(abs(x2-x1s)); x0s,x1s=x1s,x2
            if errs_s[-1]<tol: break

        self.c_root.animate_to(mid, fmt="{:.8f}")
        self.c_bis.animate_to(bis_iters, fmt="{:.0f}")
        self.c_nwt.animate_to(nwt_iters, fmt="{:.0f}")

        self._log(f"Bisection      : x={mid:.8f} ({bis_iters} iters)\n")
        self._log(f"Newton-Raphson : x={x:.8f} ({nwt_iters} iters)\n")
        self._log(f"Secant         : x={x2:.8f} ({i+1} iters)\n")

        self.fig.clear()
        ax1=self.fig.add_subplot(121)
        Vr=np.linspace(a,b,300)
        ax1.plot(Vr,[f(v) for v in Vr],color=NEON_CYAN,linewidth=1.8)
        ax1.axhline(0,color=NEON_PINK,linestyle='--',linewidth=1)
        ax1.scatter([mid],[0],color=NEON_GREEN,s=80,zorder=5)
        ax1.set_title("DIODE EQUATION f(V)")
        ax1.set_xlabel("Voltage V")
        style_axis(ax1,self.fig)

        ax2=self.fig.add_subplot(122)
        for (nm,errs),col in zip(
            [("Bisection",errs_b),("Newton",errs_n),("Secant",errs_s)],
            [NEON_CYAN,NEON_GREEN,NEON_PINK]):
            ax2.semilogy(range(1,len(errs)+1),errs,
                          label=nm,color=col,linewidth=1.8)
        ax2.set_title("CONVERGENCE")
        ax2.set_xlabel("Iteration")
        ax2.legend(fontsize=7,facecolor=BG_CARD,
                    labelcolor=TEXT_BRIGHT,edgecolor=DIM_CYAN)
        style_axis(ax2,self.fig)
        self._draw()
        self._log("\n[DONE]\n")


# ============================================================
# TAB 3: Interpolation
# ============================================================
class InterpolationTab(BaseTab):
    def _build_params(self):
        cyber_section(self.left, "M3: INTERPOLATION")
        tk.Label(self.left,
                  text="IoT sensor temperature data.\nLinear · Spline · Polynomial.",
                  bg=BG_PANEL, fg=TEXT_DIM,
                  font=("Courier", 8), justify="left"
                  ).pack(anchor="w", padx=14, pady=3)
        cyber_label(self.left, "Query time (minutes):")
        self.e_t = cyber_entry(self.left, "35")
        cyber_label(self.left, "Method:")
        self.mv = tk.StringVar(value="All")
        for m in ["All","Linear","Cubic Spline","Polynomial"]:
            tk.Radiobutton(self.left,text=m,variable=self.mv,value=m,
                            bg=BG_PANEL,fg=NEON_CYAN,selectcolor=BG_VOID,
                            activebackground=BG_PANEL,
                            font=("Courier",8)).pack(anchor="w",padx=20)

        self.c_lin  = self._add_counter("Linear  T", NEON_CYAN,  "{:.4f}")
        self.c_spl  = self._add_counter("Spline  T", NEON_GREEN, "{:.4f}")
        self.c_poly = self._add_counter("Poly    T", NEON_PINK,  "{:.4f}")

    def _run(self):
        t_q=float(self.e_t.get()); method=self.mv.get()
        TP=np.array([0,10,20,30,40,50,60,70,80,90,100,110,120])
        VP=np.array([15.0,17.5,21.3,25.8,28.1,29.4,28.7,
                      26.3,23.1,20.4,18.9,17.2,16.5])
        td=np.linspace(0,120,500)
        self._log("\n◈ MODULE 3: INTERPOLATION\n"+"─"*38+"\n")

        lin_est  = float(np.interp(t_q,TP,VP))
        spl_est  = float(interpolate.CubicSpline(TP,VP)(t_q))
        poly_est = float(np.poly1d(np.polyfit(TP,VP,len(TP)-1))(t_q))

        self.c_lin.animate_to(lin_est,  fmt="{:.4f}")
        self.c_spl.animate_to(spl_est,  fmt="{:.4f}")
        self.c_poly.animate_to(poly_est, fmt="{:.4f}")

        self._log(f"Query t={t_q} min:\n")
        self._log(f"  Linear      : {lin_est:.4f} °C\n")
        self._log(f"  Cubic Spline: {spl_est:.4f} °C\n")
        self._log(f"  Polynomial  : {poly_est:.4f} °C\n")

        self.fig.clear()
        ax=self.fig.add_subplot(111)
        ax.scatter(TP,VP,color="white",zorder=5,s=40,label="Sensor data")
        defs={
            "Linear":      (NEON_CYAN,  np.interp(td,TP,VP)),
            "Cubic Spline":(NEON_GREEN, interpolate.CubicSpline(TP,VP)(td)),
            "Polynomial":  (NEON_PINK,  np.poly1d(np.polyfit(TP,VP,len(TP)-1))(td)),
        }
        to_plot=list(defs.keys()) if method=="All" else [method]
        for m in to_plot:
            col,yv=defs[m]
            ax.plot(td,yv,color=col,linewidth=1.8,label=m,alpha=0.9)
        ax.axvline(t_q,color=NEON_YELLOW,linestyle=":",linewidth=1,alpha=0.7)
        ax.set_xlabel("Time (min)"); ax.set_ylabel("Temperature (°C)")
        ax.set_title("SENSOR DATA INTERPOLATION")
        ax.legend(fontsize=7,facecolor=BG_CARD,
                   labelcolor=TEXT_BRIGHT,edgecolor=DIM_CYAN)
        style_axis(ax,self.fig)
        self._draw()
        self._log("\n[DONE]\n")


# ============================================================
# TAB 4: Differentiation
# ============================================================
class DifferentiationTab(BaseTab):
    def _build_params(self):
        cyber_section(self.left, "M4: DIFFERENTIATION")
        tk.Label(self.left,
                  text="Forward · Backward · Central\nfinite difference schemes.",
                  bg=BG_PANEL, fg=TEXT_DIM,
                  font=("Courier", 8), justify="left"
                  ).pack(anchor="w", padx=14, pady=3)
        cyber_label(self.left, "Evaluation point x:")
        self.e_x = cyber_entry(self.left, "1.0")
        cyber_label(self.left, "Step size h:")
        self.e_h = cyber_entry(self.left, "0.01")

        self.c_exact = self._add_counter("Exact f'", NEON_CYAN,  "{:.8f}")
        self.c_cen   = self._add_counter("Central ", NEON_GREEN, "{:.8f}")
        self.c_err   = self._add_counter("Cen err ", NEON_PINK,  "{:.2e}")

    def _run(self):
        x=float(self.e_x.get()); h=float(self.e_h.get())
        def f(x):  return np.sin(x)+0.5*x**2
        def df(x): return np.cos(x)+x
        exact=df(x)
        fwd=(f(x+h)-f(x))/h
        bwd=(f(x)-f(x-h))/h
        cen=(f(x+h)-f(x-h))/(2*h)

        self.c_exact.animate_to(exact, fmt="{:.8f}")
        self.c_cen.animate_to(cen,     fmt="{:.8f}")
        self.c_err.animate_to(abs(cen-exact), fmt="{:.2e}")

        self._log("\n◈ MODULE 4: DIFFERENTIATION\n"+"─"*38+"\n")
        self._log(f"f(x)=sin(x)+0.5x²  x={x}, h={h}\n\n")
        self._log(f"Exact    : {exact:.10f}\n")
        self._log(f"Forward  : {fwd:.10f}  err={abs(fwd-exact):.2e}\n")
        self._log(f"Backward : {bwd:.10f}  err={abs(bwd-exact):.2e}\n")
        self._log(f"Central  : {cen:.10f}  err={abs(cen-exact):.2e}\n")

        h_vals=np.logspace(-1,-12,40)
        e_fwd=[abs((f(x+hh)-f(x))/hh-exact) for hh in h_vals]
        e_cen=[abs((f(x+hh)-f(x-hh))/(2*hh)-exact) for hh in h_vals]

        self.fig.clear()
        ax=self.fig.add_subplot(111)
        ax.loglog(h_vals,e_fwd,color=NEON_CYAN,linewidth=1.8,label="Forward O(h)")
        ax.loglog(h_vals,e_cen,color=NEON_GREEN,linewidth=1.8,label="Central O(h²)")
        ax.axvline(h,color=NEON_PINK,linestyle="--",linewidth=1.5,label=f"h={h}")
        ax.set_xlabel("Step size h"); ax.set_ylabel("Absolute Error")
        ax.set_title("STEP SIZE vs ACCURACY")
        ax.legend(fontsize=7,facecolor=BG_CARD,
                   labelcolor=TEXT_BRIGHT,edgecolor=DIM_CYAN)
        style_axis(ax,self.fig)
        self._draw()
        self._log("\n[DONE]\n")


# ============================================================
# TAB 5: Integration
# ============================================================
class IntegrationTab(BaseTab):
    def _build_params(self):
        cyber_section(self.left, "M5: INTEGRATION")
        tk.Label(self.left,
                  text="Trapezoid · Simpson · scipy.quad\nf(x) = sin(x) + 0.5x²",
                  bg=BG_PANEL, fg=TEXT_DIM,
                  font=("Courier", 8), justify="left"
                  ).pack(anchor="w", padx=14, pady=3)
        cyber_label(self.left, "Lower limit a:")
        self.e_a = cyber_entry(self.left, "0")
        cyber_label(self.left, "Upper limit b:")
        self.e_b = cyber_entry(self.left, "3.14159")
        cyber_label(self.left, "Intervals n:")
        self.e_n = cyber_entry(self.left, "100")

        self.c_trap = self._add_counter("Trapezoid", NEON_CYAN,  "{:.8f}")
        self.c_simp = self._add_counter("Simpson  ", NEON_GREEN, "{:.8f}")
        self.c_quad = self._add_counter("scipy.quad", NEON_PINK, "{:.8f}")

    def _run(self):
        a=float(self.e_a.get()); b=float(self.e_b.get()); n=int(self.e_n.get())
        def f(x): return np.sin(x)+0.5*x**2
        exact=2+np.pi**3/6
        x=np.linspace(a,b,n+1); y=f(x); h=(b-a)/n
        trap=h/2*(y[0]+2*np.sum(y[1:-1])+y[-1])
        if n%2!=0: n+=1
        x2=np.linspace(a,b,n+1); y2=f(x2); h2=(b-a)/n
        w=np.ones(n+1); w[1:-1:2]=4; w[2:-2:2]=2
        simp=h2/3*np.dot(w,y2)
        quad_r,_=integrate.quad(f,a,b)

        self.c_trap.animate_to(trap,   fmt="{:.8f}")
        self.c_simp.animate_to(simp,   fmt="{:.8f}")
        self.c_quad.animate_to(quad_r, fmt="{:.8f}")

        self._log("\n◈ MODULE 5: INTEGRATION\n"+"─"*38+"\n")
        self._log(f"Exact     : {exact:.10f}\n")
        self._log(f"Trapezoid : {trap:.10f}  err={abs(trap-exact):.2e}\n")
        self._log(f"Simpson   : {simp:.10f}  err={abs(simp-exact):.2e}\n")
        self._log(f"scipy.quad: {quad_r:.10f}  err={abs(quad_r-exact):.2e}\n")

        n_vals=[4,8,16,32,64,128,256]; te=[]; se=[]
        for nn in n_vals:
            xx=np.linspace(a,b,nn+1); yy=f(xx); hh=(b-a)/nn
            te.append(abs(hh/2*(yy[0]+2*np.sum(yy[1:-1])+yy[-1])-exact))
            nn2=nn if nn%2==0 else nn+1
            xx2=np.linspace(a,b,nn2+1); yy2=f(xx2); hh2=(b-a)/nn2
            ww=np.ones(nn2+1); ww[1:-1:2]=4; ww[2:-2:2]=2
            se.append(abs(hh2/3*np.dot(ww,yy2)-exact))

        self.fig.clear()
        ax=self.fig.add_subplot(111)
        ax.loglog(n_vals,te,'o-',color=NEON_CYAN,linewidth=1.8,label="Trapezoid O(h²)")
        ax.loglog(n_vals,se,'s-',color=NEON_GREEN,linewidth=1.8,label="Simpson O(h⁴)")
        ax.set_xlabel("Intervals n"); ax.set_ylabel("Absolute Error")
        ax.set_title("INTEGRATION CONVERGENCE")
        ax.legend(fontsize=7,facecolor=BG_CARD,
                   labelcolor=TEXT_BRIGHT,edgecolor=DIM_CYAN)
        style_axis(ax,self.fig)
        self._draw()
        self._log("\n[DONE]\n")


# ============================================================
# TAB 6: Linear Systems
# ============================================================
class LinearSystemsTab(BaseTab):
    def _build_params(self):
        cyber_section(self.left, "M6: LINEAR SYSTEMS")
        tk.Label(self.left,
                  text="Heat distribution in a metal rod.\nDirect · LU · Gauss-Seidel.",
                  bg=BG_PANEL, fg=TEXT_DIM,
                  font=("Courier", 8), justify="left"
                  ).pack(anchor="w", padx=14, pady=3)
        cyber_label(self.left, "Interior points n:")
        self.e_n  = cyber_entry(self.left, "20")
        cyber_label(self.left, "Left temperature (°C):")
        self.e_tl = cyber_entry(self.left, "100")
        cyber_label(self.left, "Right temperature (°C):")
        self.e_tr = cyber_entry(self.left, "20")

        self.c_gs   = self._add_counter("GS iters", NEON_CYAN,  "{:.0f}")
        self.c_diff = self._add_counter("GS err  ", NEON_PINK,  "{:.2e}")

    def _run(self):
        n=int(self.e_n.get()); TL=float(self.e_tl.get()); TR=float(self.e_tr.get())
        A=np.zeros((n,n))
        for i in range(n):
            A[i,i]=-2.0
            if i>0: A[i,i-1]=1.0
            if i<n-1: A[i,i+1]=1.0
        b=np.zeros(n); b[0]=-TL; b[-1]=-TR
        xd=np.linalg.solve(A,b)
        lu=linalg.lu_factor(A); xl=linalg.lu_solve(lu,b)
        xg=np.zeros(n); iters=0
        for it in range(5000):
            xo=xg.copy()
            for i in range(n):
                s=sum(A[i,j]*xg[j] for j in range(n) if j!=i)
                xg[i]=(b[i]-s)/A[i,i]
            iters+=1
            if np.max(np.abs(xg-xo))<1e-10: break

        self.c_gs.animate_to(iters, fmt="{:.0f}")
        self.c_diff.animate_to(np.max(np.abs(xd-xg)), fmt="{:.2e}")

        self._log("\n◈ MODULE 6: LINEAR SYSTEMS\n"+"─"*38+"\n")
        self._log(f"n={n}, TL={TL}°C, TR={TR}°C\n")
        self._log(f"Gauss-Seidel: {iters} iterations\n")
        self._log(f"Direct vs LU: {np.max(np.abs(xd-xl)):.2e}\n")
        self._log(f"Direct vs GS: {np.max(np.abs(xd-xg)):.2e}\n")

        pos=np.linspace(0,1,n+2)
        Td=np.concatenate([[TL],xd,[TR]])
        Ta=TL+(TR-TL)*pos
        self.fig.clear(); ax=self.fig.add_subplot(111)
        ax.plot(pos,Ta,'--',color=TEXT_DIM,linewidth=1,label="Analytical")
        ax.plot(pos,Td,'o-',color=NEON_CYAN,linewidth=1.8,markersize=3,label="Direct/LU")
        ax.plot(pos,np.concatenate([[TL],xg,[TR]]),'s:',
                 color=NEON_GREEN,linewidth=1.8,markersize=3,label="Gauss-Seidel")
        ax.set_xlabel("Position"); ax.set_ylabel("Temperature (°C)")
        ax.set_title(f"HEAT DISTRIBUTION  {TL}° → {TR}°")
        ax.legend(fontsize=7,facecolor=BG_CARD,
                   labelcolor=TEXT_BRIGHT,edgecolor=DIM_CYAN)
        style_axis(ax,self.fig)
        self._draw()
        self._log("\n[DONE]\n")


# ============================================================
# TAB 7: Optimization
# ============================================================
class OptimizationTab(BaseTab):
    def _build_params(self):
        cyber_section(self.left, "M7: OPTIMIZATION")
        tk.Label(self.left,
                  text="Production cost minimization.\nAdjust parameters live.",
                  bg=BG_PANEL, fg=TEXT_DIM,
                  font=("Courier", 8), justify="left"
                  ).pack(anchor="w", padx=14, pady=3)
        cyber_label(self.left, "Fixed cost per batch ($):")
        self.e_fc = cyber_entry(self.left, "5000")
        cyber_label(self.left, "Variable cost ($/unit):")
        self.e_vc = cyber_entry(self.left, "2.5")
        cyber_label(self.left, "Storage cost ($/unit²):")
        self.e_sc = cyber_entry(self.left, "0.001")
        cyber_label(self.left, "Search range max:")
        self.e_mx = cyber_entry(self.left, "5000")

        self.c_xopt = self._add_counter("Opt batch", NEON_CYAN,  "{:.2f}")
        self.c_copt = self._add_counter("Min cost $", NEON_GREEN, "{:.2f}")
        self.c_iter = self._add_counter("Iterations", NEON_PINK,  "{:.0f}")

    def _run(self):
        fc=float(self.e_fc.get()); vc=float(self.e_vc.get())
        sc=float(self.e_sc.get()); mx=float(self.e_mx.get())
        def cost(x): return fc/x+vc*x+sc*x**2
        res=optimize.minimize_scalar(cost,bounds=(1,mx),method='bounded')
        xo=res.x; co=res.fun

        self.c_xopt.animate_to(xo, fmt="{:.2f}")
        self.c_copt.animate_to(co, fmt="{:.2f}")
        self.c_iter.animate_to(res.nit, fmt="{:.0f}")

        self._log("\n◈ MODULE 7: OPTIMIZATION\n"+"─"*38+"\n")
        self._log(f"Optimal batch : {xo:.2f} units\n")
        self._log(f"Minimum cost  : ${co:.2f}\n")
        self._log(f"Iterations    : {res.nit}\n")

        xv=np.linspace(1,mx,500)
        self.fig.clear(); ax=self.fig.add_subplot(111)
        ax.plot(xv,cost(xv),color=NEON_CYAN,linewidth=1.8)
        ax.fill_between(xv,cost(xv),alpha=0.07,color=NEON_CYAN)
        ax.axvline(xo,color=NEON_GREEN,linestyle='--',
                    linewidth=1.5,label=f"Optimal x={xo:.1f}")
        ax.scatter([xo],[co],color=NEON_GREEN,s=100,zorder=5)
        ax.set_xlabel("Batch Size"); ax.set_ylabel("Total Cost ($)")
        ax.set_title(f"COST MINIMIZATION  →  ${co:.2f} @ x={xo:.1f}")
        ax.legend(fontsize=7,facecolor=BG_CARD,
                   labelcolor=TEXT_BRIGHT,edgecolor=DIM_CYAN)
        style_axis(ax,self.fig)
        self._draw()
        self._log("\n[DONE]\n")


# ============================================================
# TAB 8: ODE
# ============================================================
class ODETab(BaseTab):
    def _build_params(self):
        cyber_section(self.left, "M8: ODE SOLVER")
        tk.Label(self.left,
                  text="Pendulum simulation.\nRK45 · RK23 · DOP853 · Euler",
                  bg=BG_PANEL, fg=TEXT_DIM,
                  font=("Courier", 8), justify="left"
                  ).pack(anchor="w", padx=14, pady=3)
        cyber_label(self.left, "Initial angle (degrees):")
        self.e_theta = cyber_entry(self.left, "30")
        cyber_label(self.left, "Pendulum length L (m):")
        self.e_L = cyber_entry(self.left, "1.0")
        cyber_label(self.left, "Simulation time (s):")
        self.e_T = cyber_entry(self.left, "10")
        cyber_label(self.left, "Solver:")
        self.sv = tk.StringVar(value="RK45")
        for s in ["RK45","RK23","DOP853"]:
            tk.Radiobutton(self.left,text=s,variable=self.sv,value=s,
                            bg=BG_PANEL,fg=NEON_CYAN,selectcolor=BG_VOID,
                            activebackground=BG_PANEL,
                            font=("Courier",8)).pack(anchor="w",padx=20)

        self.c_steps = self._add_counter("Steps used", NEON_CYAN,  "{:.0f}")
        self.c_time  = self._add_counter("Comp time ", NEON_GREEN, "{:.2f}")

    def _run(self):
        theta0=np.radians(float(self.e_theta.get()))
        L=float(self.e_L.get()); T=float(self.e_T.get())
        solver=self.sv.get(); G=9.81
        def pend(t,y): return [y[1],-(G/L)*np.sin(y[0])]
        y0=[theta0,0.0]; t_eval=np.linspace(0,T,1000)
        t0=time.perf_counter()
        sol=solve_ivp(pend,(0,T),y0,method=solver,
                       t_eval=t_eval,rtol=1e-8,atol=1e-10)
        elapsed=(time.perf_counter()-t0)*1000

        dt=T/1000; t_eu=np.arange(0,T+dt,dt)
        y_eu=np.zeros((len(t_eu),2)); y_eu[0]=y0
        for i in range(1,len(t_eu)):
            d=pend(t_eu[i-1],y_eu[i-1])
            y_eu[i]=y_eu[i-1]+dt*np.array(d)
        theta_a=theta0*np.cos(np.sqrt(G/L)*t_eval)

        self.c_steps.animate_to(len(sol.t), fmt="{:.0f}")
        self.c_time.animate_to(elapsed,     fmt="{:.2f}")

        self._log("\n◈ MODULE 8: ODE SOLVER\n"+"─"*38+"\n")
        self._log(f"L={L}m, θ₀={np.degrees(theta0):.1f}°, T={T}s\n")
        self._log(f"Solver: {solver}\n")
        self._log(f"Steps : {len(sol.t)}\n")
        self._log(f"Time  : {elapsed:.3f} ms\n")

        self.fig.clear(); ax=self.fig.add_subplot(111)
        ax.plot(t_eval,np.degrees(sol.y[0]),
                 color=NEON_CYAN,linewidth=1.8,label=solver)
        ax.plot(t_eu,np.degrees(y_eu[:,0]),
                 color=NEON_PINK,linewidth=1.2,
                 linestyle=':',label="Euler",alpha=0.8)
        ax.plot(t_eval,np.degrees(theta_a),
                 color=TEXT_DIM,linewidth=1.2,
                 linestyle='--',label="Analytical")
        ax.fill_between(t_eval,np.degrees(sol.y[0]),
                          alpha=0.08,color=NEON_CYAN)
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Angle (degrees)")
        ax.set_title(f"PENDULUM — {solver} vs EULER")
        ax.legend(fontsize=7,facecolor=BG_CARD,
                   labelcolor=TEXT_BRIGHT,edgecolor=DIM_CYAN)
        style_axis(ax,self.fig)
        self._draw()
        self._log("\n[DONE]\n")


# ============================================================
# TAB 9: Performance
# ============================================================
class PerformanceTab(BaseTab):
    def _build_params(self):
        cyber_section(self.left, "M9: PERFORMANCE")
        tk.Label(self.left,
                  text="Benchmark matrix solvers.\nDirect · Inverse · LU",
                  bg=BG_PANEL, fg=TEXT_DIM,
                  font=("Courier", 8), justify="left"
                  ).pack(anchor="w", padx=14, pady=3)
        cyber_label(self.left, "Max matrix size:")
        self.e_max  = cyber_entry(self.left, "300")
        cyber_label(self.left, "Reps per size:")
        self.e_reps = cyber_entry(self.left, "10")

        self.c_best  = self._add_counter("Best time", NEON_CYAN,  "{:.3f}")
        self.c_speed = self._add_counter("LU speedup", NEON_GREEN, "{:.1f}")

    def _run(self):
        mx=int(self.e_max.get()); reps=int(self.e_reps.get())
        sizes=sorted(set([10,30,50,100,150,200,min(mx,500)]))
        ts_l=[]; ti_l=[]; tl_l=[]

        self._log("\n◈ MODULE 9: PERFORMANCE\n"+"─"*38+"\n")
        self._log(f"{'n':<6}{'Direct':>9}{'Inverse':>9}{'LU':>9}  ms\n")
        self._log("─"*34+"\n")

        for n in sizes:
            A=np.random.randn(n,n)+n*np.eye(n); b=np.random.randn(n)
            t0=time.perf_counter()
            for _ in range(reps): np.linalg.solve(A,b)
            ts=(time.perf_counter()-t0)/reps*1000
            t0=time.perf_counter()
            for _ in range(reps): np.linalg.inv(A)@b
            ti=(time.perf_counter()-t0)/reps*1000
            t0=time.perf_counter()
            for _ in range(reps):
                lu=linalg.lu_factor(A); linalg.lu_solve(lu,b)
            tl=(time.perf_counter()-t0)/reps*1000
            ts_l.append(ts); ti_l.append(ti); tl_l.append(tl)
            self._log(f"{n:<6}{ts:>9.3f}{ti:>9.3f}{tl:>9.3f}\n")

        best = min(tl_l)
        speedup = max(ts_l) / min(tl_l)
        self.c_best.animate_to(best,    fmt="{:.3f}")
        self.c_speed.animate_to(speedup, fmt="{:.1f}")

        self.fig.clear(); ax=self.fig.add_subplot(111)
        ax.loglog(sizes,ts_l,'o-',color=NEON_CYAN,linewidth=1.8,label="Direct")
        ax.loglog(sizes,ti_l,'s-',color=NEON_PINK,linewidth=1.8,label="Inverse")
        ax.loglog(sizes,tl_l,'^-',color=NEON_GREEN,linewidth=1.8,label="LU")
        ax.set_xlabel("Matrix size n"); ax.set_ylabel("Time (ms)")
        ax.set_title("SOLVER PERFORMANCE BENCHMARK")
        ax.legend(fontsize=7,facecolor=BG_CARD,
                   labelcolor=TEXT_BRIGHT,edgecolor=DIM_CYAN)
        style_axis(ax,self.fig)
        self._draw()
        self._log("\n[DONE]\n")


# ============================================================
# TAB 10: Summary Dashboard
# ============================================================
class SummaryTab(BaseTab):
    def _build_params(self):
        cyber_section(self.left, "MISSION CONTROL")
        tk.Label(self.left,
                  text="Run all 9 modules at once.\nSee pass/fail and timing\nfor each module.",
                  bg=BG_PANEL, fg=TEXT_DIM,
                  font=("Courier", 8), justify="left"
                  ).pack(anchor="w", padx=14, pady=3)

        self.c_pass  = self._add_counter("PASSED  ", NEON_GREEN, "{:.0f}")
        self.c_fail  = self._add_counter("FAILED  ", NEON_PINK,  "{:.0f}")
        self.c_total = self._add_counter("TOTAL s ", NEON_CYAN,  "{:.2f}")

    def _run(self):
        self._log("\n◈ MISSION CONTROL — ALL MODULES\n"+"═"*38+"\n")

        modules = [
            ("M1 Error Analysis",   self._run_m1),
            ("M2 Root Finding",     self._run_m2),
            ("M3 Interpolation",    self._run_m3),
            ("M4 Differentiation",  self._run_m4),
            ("M5 Integration",      self._run_m5),
            ("M6 Linear Systems",   self._run_m6),
            ("M7 Optimization",     self._run_m7),
            ("M8 ODE Solver",       self._run_m8),
            ("M9 Performance",      self._run_m9),
        ]

        results = []
        t_start = time.perf_counter()

        for name, func in modules:
            t0 = time.perf_counter()
            try:
                func()
                elapsed = time.perf_counter() - t0
                results.append((name, True, elapsed))
                self._log(f"  [OK]   {name:<22} {elapsed:.3f}s\n")
            except Exception as e:
                elapsed = time.perf_counter() - t0
                results.append((name, False, elapsed))
                self._log(f"  [FAIL] {name:<22} {str(e)[:25]}\n")

        total = time.perf_counter() - t_start
        passed = sum(1 for _,ok,_ in results if ok)
        failed = len(results) - passed

        self.c_pass.animate_to(passed, fmt="{:.0f}")
        self.c_fail.animate_to(failed, fmt="{:.0f}")
        self.c_total.animate_to(total,  fmt="{:.2f}")

        self._log("\n"+"═"*38+"\n")
        self._log(f"PASSED : {passed}/9\n")
        self._log(f"FAILED : {failed}/9\n")
        self._log(f"TOTAL  : {total:.2f}s\n")
        if failed == 0:
            self._log("\n✦ ALL SYSTEMS NOMINAL ✦\n")

        # Summary bar chart
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        names   = [r[0].replace(" ","\\n") for r in results]
        times   = [r[2] for r in results]
        colors  = [NEON_GREEN if r[1] else NEON_PINK for r in results]
        short   = [r[0][:8] for r in results]
        bars    = ax.bar(range(len(results)), times,
                          color=colors, edgecolor=DIM_CYAN,
                          linewidth=0.8)
        for bar, t in zip(bars, times):
            ax.text(bar.get_x()+bar.get_width()/2,
                     bar.get_height()+0.01,
                     f"{t:.2f}s", ha="center",
                     fontsize=7, color=TEXT_BRIGHT)
        ax.set_xticks(range(len(results)))
        ax.set_xticklabels(short, fontsize=7, rotation=30)
        ax.set_ylabel("Time (s)")
        ax.set_title("MODULE EXECUTION TIMES  ■ GREEN=PASS  ■ RED=FAIL")
        style_axis(ax, self.fig)
        self._draw()

    # ── Quick runners (no plots, just compute) ────────────────
    def _run_m1(self):
        eps=np.finfo(float).eps
        x0,x1=1.0,1.0+eps
        for _ in range(50):
            xn=(4/3)*x1-(1/3)*x0; x0,x1=x1,xn

    def _run_m2(self):
        IS,VT,IL=1e-12,0.02585,0.01
        def f(V): return IS*(np.exp(V/VT)-1)-IL
        a,b=0.3,0.9
        for _ in range(10000):
            m=(a+b)/2
            if f(a)*f(m)<0: b=m
            else: a=m
            if abs(b-a)<1e-10: break

    def _run_m3(self):
        TP=np.array([0,10,20,30,40,50,60,70,80,90,100,110,120])
        VP=np.array([15.,17.5,21.3,25.8,28.1,29.4,28.7,
                      26.3,23.1,20.4,18.9,17.2,16.5])
        _ = interpolate.CubicSpline(TP,VP)(35)

    def _run_m4(self):
        def f(x): return np.sin(x)+0.5*x**2
        h=0.01; x=1.0
        _ = (f(x+h)-f(x-h))/(2*h)

    def _run_m5(self):
        def f(x): return np.sin(x)+0.5*x**2
        _,_ = integrate.quad(f,0,np.pi)

    def _run_m6(self):
        n=20; A=np.zeros((n,n))
        for i in range(n):
            A[i,i]=-2.
            if i>0: A[i,i-1]=1.
            if i<n-1: A[i,i+1]=1.
        b=np.zeros(n); b[0]=-100.; b[-1]=-20.
        _ = np.linalg.solve(A,b)

    def _run_m7(self):
        def cost(x): return 5000/x+2.5*x+0.001*x**2
        _ = optimize.minimize_scalar(cost,bounds=(1,5000),method='bounded')

    def _run_m8(self):
        def pend(t,y): return [y[1],-9.81*np.sin(y[0])]
        _ = solve_ivp(pend,(0,5),[np.radians(30),0],method='RK45')

    def _run_m9(self):
        A=np.random.randn(50,50)+50*np.eye(50)
        b=np.random.randn(50)
        _ = np.linalg.solve(A,b)


# ============================================================
# MAIN APPLICATION
# ============================================================
class NumericalMethodsApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(
            "NUMERICAL METHODS — Fatma Helen Insel | 24220030019")
        self.geometry("1300x800")
        self.configure(bg=BG_VOID)
        self.resizable(True, True)

        CyberHeader(self)
        self._build_notebook()
        self._build_statusbar()

    def _build_notebook(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Cyber.TNotebook",
                          background=BG_VOID, borderwidth=0)
        style.configure("Cyber.TNotebook.Tab",
                          background=BG_PANEL,
                          foreground=TEXT_DIM,
                          padding=[10,5],
                          font=("Courier", 8, "bold"))
        style.map("Cyber.TNotebook.Tab",
                   background=[("selected", DIM_CYAN)],
                   foreground=[("selected", NEON_CYAN)])

        self.nb = ttk.Notebook(self, style="Cyber.TNotebook")
        self.nb.pack(fill="both", expand=True, padx=5, pady=(3,0))

        self._tab_objects = []
        tabs = [
            ("M1 ERROR",   ErrorAnalysisTab),
            ("M2 ROOTS",   RootFindingTab),
            ("M3 INTERP",  InterpolationTab),
            ("M4 DERIV",   DifferentiationTab),
            ("M5 INTEG",   IntegrationTab),
            ("M6 LINEAR",  LinearSystemsTab),
            ("M7 OPTIM",   OptimizationTab),
            ("M8 ODE",     ODETab),
            ("M9 PERF",    PerformanceTab),
            ("◈ MISSION",  SummaryTab),
        ]
        for label, Cls in tabs:
            f = Cls(self.nb)
            self._tab_objects.append(f)
            self.nb.add(f, text=f"  {label}  ")

        # Flash effect on tab switch
        self.nb.bind("<<NotebookTabChanged>>", self._on_tab_change)

    def _on_tab_change(self, event):
        idx = self.nb.index(self.nb.select())
        if 0 <= idx < len(self._tab_objects):
            self._tab_objects[idx].flash()

    def _build_statusbar(self):
        bar = tk.Frame(self, bg=BG_PANEL, height=22)
        bar.pack(fill="x", side="bottom")
        tk.Label(bar,
                  text="  ◈ 155-4007  ·  Select tab  ·  "
                       "Set parameters  ·  Press RUN",
                  font=("Courier", 8),
                  bg=BG_PANEL, fg=TEXT_DIM
                  ).pack(side="left", pady=3)
        tk.Label(bar, text="READY ●  ",
                  font=("Courier", 8, "bold"),
                  bg=BG_PANEL, fg=NEON_GREEN
                  ).pack(side="right", pady=3)


if __name__ == "__main__":
    app = NumericalMethodsApp()
    app.mainloop()