import tkinter as tk
from tkinter import messagebox
import re
import numpy as np
import sympy as sp
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from typing import Callable, Dict, Optional


# -----------------------------------------------------------------------------
# Theme
# -----------------------------------------------------------------------------
class Theme:
    BG = "#0B0F17"
    CARD = "#111827"
    CARD_LGHT = "#1F2937"
    BORDER = "#1E293B"
    TEXT = "#F1F5F9"
    TEXT_DIM = "#94A3B8"
    ACCENT = "#38BDF8"
    ACCENT_HVR = "#0EA5E9"
    SUCCESS = "#10B981"
    ERROR = "#EF4444"
    GRID = "#1E293B"
    AXIS = "#334155"
    COLORS = [
        "#F38BA8", "#89B4FA", "#A6E3A1", "#F9E2AF",
        "#CBA6F7", "#FAB387", "#94E2D5", "#F5C2E7",
    ]


# -----------------------------------------------------------------------------
# Hover Tooltip
# -----------------------------------------------------------------------------
class HoverTip:
    def __init__(self, widget: tk.Widget, text: str, delay: int = 350):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._tip = None
        self._id = None
        widget.bind("<Enter>", self._schedule)
        widget.bind("<Leave>", self._hide)

    def _schedule(self, _=None):
        self._id = self.widget.after(self.delay, self._show)

    def _show(self):
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 2

        self._tip = tk.Toplevel(self.widget)
        self._tip.wm_overrideredirect(True)
        self._tip.wm_attributes("-topmost", True)
        self._tip.wm_geometry(f"+{x}+{y}")

        tk.Label(
            self._tip,
            text=self.text,
            bg="#0B1220",
            fg=Theme.TEXT,
            font=("Segoe UI", 9),
            relief=tk.SOLID,
            borderwidth=1,
            highlightbackground=Theme.BORDER,
            padx=8,
            pady=3,
        ).pack()

        self._tip.update_idletasks()
        w = self._tip.winfo_width()
        self._tip.wm_geometry(f"+{x - w // 2}+{y}")

    def _hide(self, _=None):
        if self._id:
            self.widget.after_cancel(self._id)
            self._id = None
        if self._tip:
            self._tip.destroy()
            self._tip = None


# -----------------------------------------------------------------------------
# Recursive-Descent Parser
# -----------------------------------------------------------------------------
class ExprParser:
    _FUNCS = frozenset({
        "sin", "cos", "tan", "cot", "sec", "csc",
        "asin", "acos", "atan", "acot", "asec", "acsc",
        "sinh", "cosh", "tanh", "coth", "sech", "csch",
        "asinh", "acosh", "atanh", "acoth", "asech", "acsch",
        "log", "ln", "abs", "sqrt", "exp", "gamma", "sign",
        "floor", "ceil", "ceiling",
    })
    _EXTRA_KEYS = frozenset({"pi", "e", "x", "y", "z", "a", "b", "c", "d"})
    _OP_TAILS = frozenset({"+", "-", "*", "/", "^", "**", ")", ",", None})

    def __init__(self, text: str):
        text = text.lower().strip()
        keywords = sorted(self._FUNCS | self._EXTRA_KEYS, key=len, reverse=True)
        kw_pat = "|".join(re.escape(k) for k in keywords)
        self.tokens = re.findall(
            rf"(?:{kw_pat})|\d+\.\d+|\d+|\*\*|[+\-*/^(),]|[a-zA-Z_]\w*", text
        )
        self.pos = 0
        self.n = len(self.tokens)

    def _peek(self):
        return self.tokens[self.pos] if self.pos < self.n else None

    def _consume(self) -> str:
        t = self.tokens[self.pos]
        self.pos += 1
        return t

    def _can_start(self, tok):
        return tok is not None and tok not in self._OP_TAILS

    def parse(self) -> str:
        return self._expr()

    def _expr(self) -> str:
        node = self._term()
        while self._peek() in ("+", "-"):
            op = self._consume()
            node = f"{node}{op}{self._term()}"
        return node

    def _term(self) -> str:
        node = self._factor()
        while True:
            tok = self._peek()
            if tok in ("*", "/"):
                self._consume()
                node = f"{node}{tok}{self._factor()}"
            elif self._can_start(tok):
                node = f"{node}*{self._factor()}"
            else:
                break
        return node

    def _factor(self) -> str:
        return self._power()

    def _power(self) -> str:
        node = self._unary()
        if self._peek() in ("^", "**"):
            self._consume()
            node = f"{node}**{self._power()}"
        return node

    def _unary(self) -> str:
        if self._peek() in ("+", "-"):
            return f"{self._consume()}{self._unary()}"
        return self._primary()

    def _primary(self) -> str:
        tok = self._peek()
        if tok is None:
            return ""

        if tok == "(":
            self._consume()
            inner = self._expr()
            if self._peek() == ")":
                self._consume()
            return f"({inner})"

        if tok in self._FUNCS:
            self._consume()
            name = tok
            if name == "ln":
                name = "log"
            elif name == "ceil":
                name = "ceiling"
            if self._peek() == "(":
                self._consume()
                inner = self._expr()
                if self._peek() == ")":
                    self._consume()
                return f"{name}({inner})"
            return f"{name}({self._unary()})"

        if tok == "pi":
            self._consume()
            return "pi"

        if tok == "e":
            self._consume()
            return "E"

        self._consume()
        return tok


# -----------------------------------------------------------------------------
# Data Model
# -----------------------------------------------------------------------------
class PlottedFunction:
    __slots__ = (
        "fid", "expr_raw", "expr_display", "color",
        "lambda_fn", "sym_expr", "artist", "deriv_artist",
        "visible", "show_deriv", "deriv_lambda",
        "_ui_card", "_deriv_btn", "_vis_btn",
    )

    def __init__(
        self,
        fid: int,
        expr_raw: str,
        expr_display: str,
        color: str,
        lambda_fn: Callable,
        sym_expr,
    ):
        self.fid = fid
        self.expr_raw = expr_raw
        self.expr_display = expr_display
        self.color = color
        self.lambda_fn = lambda_fn
        self.sym_expr = sym_expr
        self.artist = None
        self.deriv_artist = None
        self.visible = True
        self.show_deriv = False
        self.deriv_lambda = None
        self._ui_card = None
        self._deriv_btn = None
        self._vis_btn = None


# -----------------------------------------------------------------------------
# Scrollable helper
# -----------------------------------------------------------------------------
class ScrollableFrame(tk.Frame):
    def __init__(self, parent, bg: str, **kw):
        super().__init__(parent, bg=bg, **kw)
        self.canvas = tk.Canvas(self, bg=bg, highlightthickness=0)
        self.scrollbar = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=bg)

        self.inner.bind(
            "<Configure>", lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self._cw = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.bind("<Configure>", self._fit_width)

    def _fit_width(self, event):
        self.canvas.itemconfig(self._cw, width=event.width)


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
class GraphingCalculator(tk.Tk):
    PARAMS = ("a", "b", "c", "d", "y", "z")
    DEFAULTS = {"a": 1.0, "b": 1.0, "c": 0.0, "d": 0.0, "y": 1.0, "z": 0.0}
    SYM_TUPLE = sp.symbols("x a b c d y z")

    def __init__(self):
        super().__init__()
        self.title("Graphware ” Graphing Calculator")
        self.configure(bg=Theme.BG)
        self.geometry("1400x900")
        self.minsize(1200, 750)

        self.next_fid = 1
        self.functions: Dict[int, PlottedFunction] = {}
        self.color_ptr = 0
        self._x_data = np.linspace(-10, 10, 2000)
        self._status_after = None

        self.vars = {p: tk.DoubleVar(value=self.DEFAULTS[p]) for p in self.PARAMS}
        self.var_labels: Dict[str, tk.Label] = {}
        self.xmin_var = tk.StringVar(value="-10")
        self.xmax_var = tk.StringVar(value="10")
        self.ymin_var = tk.StringVar(value="-10")
        self.ymax_var = tk.StringVar(value="10")
        self.autoscale_var = tk.BooleanVar(value=True)

        self._build_ui()
        self._build_figure()
        self._bind_events()
        self.add_function("a * sin(b * x) + c")

    # ========================================================================
    # UI Builders
    # ========================================================================
    def _styled_frame(self, parent, **kw):
        return tk.Frame(parent, bg=Theme.CARD, highlightbackground=Theme.BORDER, highlightthickness=1, **kw)

    def _accent_btn(self, parent, text, cmd, width=None):
        return tk.Button(
            parent,
            text=text,
            bg=Theme.ACCENT,
            fg=Theme.BG,
            font=("Segoe UI", 10, "bold"),
            relief=tk.FLAT,
            activebackground=Theme.ACCENT_HVR,
            activeforeground=Theme.BG,
            cursor="hand2",
            width=width,
            command=cmd,
        )

    def _ghost_btn(self, parent, text, cmd):
        return tk.Button(
            parent,
            text=text,
            bg=Theme.CARD_LGHT,
            fg=Theme.TEXT_DIM,
            font=("Segoe UI", 9),
            relief=tk.FLAT,
            activebackground=Theme.BORDER,
            activeforeground=Theme.TEXT,
            cursor="hand2",
            command=cmd,
        )

    def _section_label(self, parent, text):
        return tk.Label(
            parent, text=text, bg=Theme.CARD, fg=Theme.TEXT_DIM,
            font=("Segoe UI", 10, "bold"), anchor="w",
        )

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=0, minsize=340)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        self._build_sidebar()
        self._build_graph_area()
        self._build_bottom_bar()
        self._build_status_bar()

    def _build_sidebar(self):
        self.sidebar = self._styled_frame(self, padx=12, pady=12)
        self.sidebar.grid(row=0, column=0, sticky="nsew", padx=(0, 1))

        tk.Label(
            self.sidebar, text="Graphware", bg=Theme.CARD,
            fg=Theme.TEXT, font=("Segoe UI", 18, "bold"), anchor="w",
        ).grid(row=0, column=0, sticky="ew", pady=(0, 2))
        tk.Label(
            self.sidebar, text="Graphing Calculator", bg=Theme.CARD,
            fg=Theme.TEXT_DIM, font=("Segoe UI", 10), anchor="w",
        ).grid(row=1, column=0, sticky="ew", pady=(0, 14))

        # Entry
        in_frm = tk.Frame(self.sidebar, bg=Theme.CARD)
        in_frm.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        in_frm.grid_columnconfigure(0, weight=1)

        self.entry = tk.Entry(
            in_frm,
            bg=Theme.BG,
            fg=Theme.TEXT,
            insertbackground=Theme.TEXT_DIM,
            font=("JetBrains Mono", 12),
            relief=tk.FLAT,
            highlightthickness=1,
            highlightcolor=Theme.ACCENT,
            highlightbackground=Theme.BORDER,
        )
        self.entry.grid(row=0, column=0, sticky="ew", ipady=6)
        self.entry.insert(0, "a * sin(b * x) + c")
        self.entry.bind("<Return>", lambda e: self._on_plot())
        self.entry.focus_set()

        self._accent_btn(in_frm, "Plot", self._on_plot, width=8).grid(
            row=0, column=1, padx=(8, 0), ipady=4
        )

        # Derivative / Integral quick actions
        calc_frm = tk.Frame(self.sidebar, bg=Theme.CARD)
        calc_frm.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        calc_frm.grid_columnconfigure(0, weight=1)
        calc_frm.grid_columnconfigure(1, weight=1)

        self._ghost_btn(calc_frm, "Plot f '(x)", self._on_deriv).grid(
            row=0, column=0, sticky="ew", padx=(0, 3), ipady=3
        )
        self._ghost_btn(calc_frm, "Plot integral", self._on_integral).grid(
            row=0, column=1, sticky="ew", padx=(3, 0), ipady=3
        )

        # Keypad
        self._build_keypad(self.sidebar).grid(row=4, column=0, sticky="ew", pady=(0, 10))

        # Parameters
        self._build_params(self.sidebar).grid(row=5, column=0, sticky="ew", pady=(0, 10))

        # Active functions
        self._section_label(self.sidebar, "Active Functions").grid(
            row=6, column=0, sticky="w", pady=(0, 6)
        )

        self.active_list = ScrollableFrame(self.sidebar, bg=Theme.CARD)
        self.active_list.grid(row=7, column=0, sticky="nsew")
        self.sidebar.grid_rowconfigure(7, weight=1)

    def _build_keypad(self, parent):
        frm = tk.Frame(parent, bg=Theme.CARD)
        for c in range(6):
            frm.grid_columnconfigure(c, weight=1)

        styles = {
            "num":  {"bg": "#1F2937", "fg": "#F8FAFC", "hv": "#374151"},
            "op":   {"bg": "#334155", "fg": "#38BDF8", "hv": "#475569"},
            "func": {"bg": "#1E293B", "fg": "#94A3B8", "hv": "#334155"},
            "var":  {"bg": "#0F172A", "fg": "#A6E3A1", "hv": "#1E293B"},
            "act":  {"bg": "#450A0A", "fg": "#FCA5A5", "hv": "#7F1D1D"},
            "plot": {"bg": Theme.ACCENT, "fg": Theme.BG, "hv": Theme.ACCENT_HVR},
        }

        def _mk(r, c, txt, val, sty_key, tip=None, cs=1):
            st = styles[sty_key]
            if isinstance(val, str):
                cmd = lambda v=val: self._insert_text(v)
            else:
                cmd = val
            btn = tk.Button(
                frm, text=txt, bg=st["bg"], fg=st["fg"],
                relief=tk.FLAT, bd=0,
                font=("Segoe UI", 9, "bold" if sty_key == "plot" else ""),
                cursor="hand2",
                activebackground=st["hv"],
                activeforeground=st["fg"],
                command=cmd,
            )
            btn.grid(row=r, column=c, columnspan=cs, padx=1, pady=1, sticky="nsew", ipady=1)
            btn.bind("<Enter>", lambda e, b=btn, h=st["hv"]: b.config(bg=h))
            btn.bind("<Leave>", lambda e, b=btn, d=st["bg"]: b.config(bg=d))
            if tip:
                HoverTip(btn, tip)
            return btn

        # Row 0 â€“ basic trig
        _mk(0, 0, "sin",  "sin(",  "func", "Sine")
        _mk(0, 1, "cos",  "cos(",  "func", "Cosine")
        _mk(0, 2, "tan",  "tan(",  "func", "Tangent")
        _mk(0, 3, "asin", "asin(", "func", "Arcsine")
        _mk(0, 4, "acos", "acos(", "func", "Arccosine")
        _mk(0, 5, "atan", "atan(", "func", "Arctangent")
        # Row 1
        _mk(1, 0, "cot",  "cot(",  "func", "Cotangent")
        _mk(1, 1, "sec",  "sec(",  "func", "Secant")
        _mk(1, 2, "csc",  "csc(",  "func", "Cosecant")
        _mk(1, 3, "acot", "acot(", "func", "Arccotangent")
        _mk(1, 4, "asec", "asec(", "func", "Arcsecant")
        _mk(1, 5, "acsc", "acsc(", "func", "Arccosecant")
        # Row 2 â€“ hyperbolic
        _mk(2, 0, "sinh", "sinh(", "func", "Hyperbolic sine")
        _mk(2, 1, "cosh", "cosh(", "func", "Hyperbolic cosine")
        _mk(2, 2, "tanh", "tanh(", "func", "Hyperbolic tangent")
        _mk(2, 3, "asinh", "asinh(", "func", "Inv. hyperbolic sine")
        _mk(2, 4, "acosh", "acosh(", "func", "Inv. hyperbolic cosine")
        _mk(2, 5, "atanh", "atanh(", "func", "Inv. hyperbolic tangent")
        # Row 3
        _mk(3, 0, "coth", "coth(", "func", "Hyperbolic cotangent")
        _mk(3, 1, "sech", "sech(", "func", "Hyperbolic secant")
        _mk(3, 2, "csch", "csch(", "func", "Hyperbolic cosecant")
        _mk(3, 3, "acoth", "acoth(", "func", "Inv. hyperbolic cotangent")
        _mk(3, 4, "asech", "asech(", "func", "Inv. hyperbolic secant")
        _mk(3, 5, "acsch", "acsch(", "func", "Inv. hyperbolic cosecant")
        # Row 4 â€“ misc funcs
        _mk(4, 0, "abs",   "abs(",   "func", "Absolute value")
        _mk(4, 1, "sqrt",  "sqrt(",  "func", "Square root")
        _mk(4, 2, "log",   "log(",   "func", "Natural logarithm")
        _mk(4, 3, "ln",    "ln(",    "func", "Natural logarithm (alias)")
        _mk(4, 4, "exp",   "exp(",   "func", "Exponential")
        _mk(4, 5, "gamma", "gamma(", "func", "Gamma function")
        # Row 5 â€“ numbers 7-9
        _mk(5, 0, "7", "7", "num")
        _mk(5, 1, "8", "8", "num")
        _mk(5, 2, "9", "9", "num")
        _mk(5, 3, "/", "/", "op")
        _mk(5, 4, "C", self._clear_entry, "act", "Clear entry")
        _mk(5, 5, "<-", self._backspace, "act", "Backspace")
        # Row 6
        _mk(6, 0, "4", "4", "num")
        _mk(6, 1, "5", "5", "num")
        _mk(6, 2, "6", "6", "num")
        _mk(6, 3, "*", "*", "op")
        _mk(6, 4, "x", "x", "var", "Variable x")
        _mk(6, 5, "y", "y", "var", "Parameter y")
        # Row 7
        _mk(7, 0, "1", "1", "num")
        _mk(7, 1, "2", "2", "num")
        _mk(7, 2, "3", "3", "num")
        _mk(7, 3, "-", "-", "op")
        _mk(7, 4, "z", "z", "var", "Parameter z")
        _mk(7, 5, "a", "a", "var", "Parameter a")
        # Row 8
        _mk(8, 0, "0", "0", "num")
        _mk(8, 1, ".", ".", "num")
        _mk(8, 2, "pi", "pi", "func", "Pi constant")
        _mk(8, 3, "e", "e", "func", "Euler's number")
        _mk(8, 4, "b", "b", "var", "Parameter b")
        _mk(8, 5, "c", "c", "var", "Parameter c")
        # Row 9
        _mk(9, 0, "(", "(", "op")
        _mk(9, 1, ")", ")", "op")
        _mk(9, 2, "^", "^", "op", "Power")
        _mk(9, 3, "+", "+", "op")
        _mk(9, 4, "d", "d", "var", "Parameter d")
        _mk(9, 5, "Plot", self._on_plot, "plot")

        return frm

    def _build_params(self, parent):
        frm = tk.Frame(parent, bg=Theme.CARD)
        self._section_label(frm, "Parameters").pack(anchor="w", pady=(0, 6))

        grid = tk.Frame(frm, bg=Theme.CARD)
        grid.pack(fill=tk.X)
        grid.grid_columnconfigure(0, weight=1)
        grid.grid_columnconfigure(1, weight=1)

        for idx, name in enumerate(self.PARAMS):
            r, c = divmod(idx, 2)
            cell = tk.Frame(grid, bg=Theme.CARD)
            cell.grid(row=r, column=c, sticky="nsew", padx=3, pady=3)
            cell.grid_columnconfigure(1, weight=1)

            tk.Label(
                cell, text=name, bg=Theme.CARD, fg=Theme.ACCENT,
                font=("JetBrains Mono", 9, "bold"), width=2,
            ).grid(row=0, column=0)

            val_lbl = tk.Label(
                cell, text=f"{self.DEFAULTS[name]:.1f}", bg=Theme.CARD,
                fg=Theme.TEXT, font=("JetBrains Mono", 8), width=4,
            )
            val_lbl.grid(row=0, column=2)
            self.var_labels[name] = val_lbl

            scl = tk.Scale(
                cell,
                from_=-10.0,
                to=10.0,
                resolution=0.1,
                orient=tk.HORIZONTAL,
                bg=Theme.CARD,
                fg=Theme.TEXT,
                troughcolor=Theme.BORDER,
                highlightthickness=0,
                borderwidth=0,
                activebackground=Theme.ACCENT,
                sliderlength=14,
                width=8,
                showvalue=0,
                variable=self.vars[name],
                command=lambda v, n=name: self._on_slider(n),
            )
            scl.set(self.DEFAULTS[name])
            scl.grid(row=0, column=1, sticky="ew", padx=4)

        return frm

    def _build_graph_area(self):
        self.graph_frame = self._styled_frame(self)
        self.graph_frame.grid(row=0, column=1, sticky="nsew")
        self.graph_frame.grid_rowconfigure(0, weight=1)
        self.graph_frame.grid_columnconfigure(0, weight=1)

    def _build_bottom_bar(self):
        bar = self._styled_frame(self, padx=14, pady=12)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=0, pady=(1, 0))

        vp = tk.Frame(bar, bg=Theme.CARD)
        vp.pack(side=tk.LEFT, fill=tk.Y)

        self._section_label(vp, "Viewport (X / Y)").pack(anchor="w", pady=(0, 8))

        grid = tk.Frame(vp, bg=Theme.CARD)
        grid.pack()

        pairs = [
            ("X min", self.xmin_var),
            ("X max", self.xmax_var),
            ("Y min", self.ymin_var),
            ("Y max", self.ymax_var),
        ]
        for i, (lbl, var) in enumerate(pairs):
            r, c = divmod(i, 2)
            tk.Label(grid, text=lbl, bg=Theme.CARD, fg=Theme.TEXT_DIM, font=("Segoe UI", 9)).grid(
                row=r * 2, column=c, sticky="w", padx=4
            )
            e = tk.Entry(
                grid,
                textvariable=var,
                bg=Theme.BG,
                fg=Theme.TEXT,
                font=("JetBrains Mono", 9),
                relief=tk.FLAT,
                highlightthickness=1,
                highlightcolor=Theme.ACCENT,
                highlightbackground=Theme.BORDER,
                width=10,
            )
            e.grid(row=r * 2 + 1, column=c, sticky="w", padx=4, pady=(0, 6))
            e.bind("<Return>", lambda _, v=var: self._apply_viewport())

        btns = tk.Frame(vp, bg=Theme.CARD)
        btns.pack(fill=tk.X, pady=(4, 0))

        self._accent_btn(btns, "Apply", self._apply_viewport, width=8).pack(
            side=tk.LEFT, padx=(0, 6), ipady=3
        )
        self._ghost_btn(btns, "Reset", lambda: self._on_view_btn("Reset")).pack(
            side=tk.LEFT, padx=(0, 6), ipady=3
        )
        self._ghost_btn(btns, "Fit", lambda: self._on_view_btn("Fit")).pack(
            side=tk.LEFT, padx=(0, 6), ipady=3
        )
        self._ghost_btn(btns, "Square", lambda: self._on_view_btn("Square")).pack(
            side=tk.LEFT, ipady=3
        )

        right = tk.Frame(bar, bg=Theme.CARD)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        tk.Checkbutton(
            right,
            text="Auto-scale Y",
            variable=self.autoscale_var,
            bg=Theme.CARD,
            fg=Theme.TEXT_DIM,
            selectcolor=Theme.ACCENT,
            activebackground=Theme.CARD,
            activeforeground=Theme.TEXT,
            font=("Segoe UI", 9),
            bd=0,
            highlightthickness=0,
        ).pack(anchor="e", pady=(4, 0))

    def _build_status_bar(self):
        bar = tk.Frame(self, bg=Theme.BG, height=26)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew")

        self.coord_lbl = tk.Label(
            bar, text="", bg=Theme.BG, fg=Theme.TEXT_DIM, font=("JetBrains Mono", 9)
        )
        self.coord_lbl.pack(side=tk.LEFT, padx=12)

        self.status_lbl = tk.Label(
            bar, text="Ready", bg=Theme.BG, fg=Theme.TEXT_DIM, font=("Segoe UI", 9)
        )
        self.status_lbl.pack(side=tk.RIGHT, padx=12)

    def _build_figure(self):
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor=Theme.BG, tight_layout=True)
        self.ax = self.fig.add_subplot(111, facecolor=Theme.BG)

        self.ax.tick_params(colors=Theme.TEXT_DIM, which="both", labelsize=9)
        for spine in self.ax.spines.values():
            spine.set_color(Theme.AXIS)

        self.ax.grid(True, color=Theme.GRID, alpha=0.5, linestyle="-")
        self.ax.axhline(0, color=Theme.AXIS, linewidth=1.0)
        self.ax.axvline(0, color=Theme.AXIS, linewidth=1.0)
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-10, 10)
        self.ax.set_xlabel("x", color=Theme.TEXT_DIM, fontsize=9)
        self.ax.set_ylabel("y", color=Theme.TEXT_DIM, fontsize=9)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        w = self.canvas.get_tk_widget()
        w.configure(bg=Theme.BG)
        w.grid(row=0, column=0, sticky="nsew")

        toolbar = NavigationToolbar2Tk(self.canvas, self.graph_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.configure(bg=Theme.CARD)
        toolbar.grid(row=1, column=0, sticky="ew", pady=(2, 0))

    def _bind_events(self):
        self.canvas.mpl_connect("motion_notify_event", self._on_mouse_move)

    # ========================================================================
    # Math Engine
    # ========================================================================
    def _compile(self, raw: str):
        expr = raw.strip()
        expr = re.sub(r"^[a-zA-Z_]\w*\s*\(\s*x\s*\)\s*=\s*", "", expr)
        expr = re.sub(r"^[a-zA-Z_]\w*\s*=\s*", "", expr)

        py_expr = ExprParser(expr).parse()

        x, a, b, c, d, y, z = self.SYM_TUPLE
        ns = {
            x: x, a: a, b: b, c: c, d: d, "y": y, "z": z,
            "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
            "cot": sp.cot, "sec": sp.sec, "csc": sp.csc,
            "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
            "acot": sp.acot, "asec": sp.asec, "acsc": sp.acsc,
            "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
            "coth": sp.coth, "sech": sp.sech, "csch": sp.csch,
            "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
            "acoth": sp.acoth, "asech": sp.asech, "acsch": sp.acsch,
            "exp": sp.exp, "log": sp.log, "sqrt": sp.sqrt, "abs": sp.Abs,
            "gamma": sp.gamma, "sign": sp.sign,
            "floor": sp.floor, "ceiling": sp.ceiling,
            "pi": sp.pi, "E": sp.E,
        }
        parsed = sp.sympify(py_expr, locals=ns)
        allowed = set(self.SYM_TUPLE)
        bad = parsed.free_symbols - allowed
        if bad:
            names = ", ".join(sorted(str(s) for s in bad))
            raise ValueError(f"Unknown symbols: {names}. Use x, a-d, y, z.")

        lam = sp.lambdify(self.SYM_TUPLE, parsed, modules="numpy")
        return lam, parsed

    # ========================================================================
    # Function Management
    # ========================================================================
    def _next_color(self) -> str:
        c = Theme.COLORS[self.color_ptr % len(Theme.COLORS)]
        self.color_ptr += 1
        return c

    def add_function(self, raw: str, display: Optional[str] = None):
        raw = raw.strip()
        if not raw:
            return
        try:
            lam, sym = self._compile(raw)
        except Exception as exc:
            self._set_status(f"Syntax error: {exc}", Theme.ERROR)
            messagebox.showwarning(
                "Syntax Error",
                f"Could not parse expression:\n{exc}\n\nHints:\n"
                "  â€¢ Implicit multiplication: 3sinx, (x+1)(x+2)\n"
                "  â€¢ Functions need no parens: sinx, logx, absx\n"
                "  â€¢ Use x as independent variable; a-f, y, z as sliders",
            )
            return

        fid = self.next_fid
        self.next_fid += 1

        pf = PlottedFunction(
            fid=fid,
            expr_raw=raw,
            expr_display=display if display else raw,
            color=self._next_color(),
            lambda_fn=lam,
            sym_expr=sym,
        )
        self.functions[fid] = pf
        self._render_card(pf)
        self._redraw()
        self._set_status("Function plotted", Theme.SUCCESS)

    def _render_card(self, pf: PlottedFunction):
        card = tk.Frame(self.active_list.inner, bg=Theme.CARD, padx=0, pady=0)
        card.pack(fill=tk.X, pady=(0, 1))
        pf._ui_card = card

        bar = tk.Frame(card, bg=pf.color, width=3)
        bar.pack(side=tk.LEFT, fill=tk.Y)

        txt = tk.Label(
            card,
            text=pf.expr_display,
            bg=Theme.CARD,
            fg=Theme.TEXT,
            font=("JetBrains Mono", 10),
            anchor="w",
            padx=8,
            pady=5,
        )
        txt.pack(side=tk.LEFT, fill=tk.X, expand=True)

        acts = tk.Frame(card, bg=Theme.CARD)
        acts.pack(side=tk.RIGHT, padx=(0, 6))

        deriv = tk.Button(
            acts,
            text="f'",
            bg="#1E293B",
            fg=Theme.TEXT_DIM,
            relief=tk.FLAT,
            bd=0,
            font=("JetBrains Mono", 9, "bold"),
            cursor="hand2",
            command=lambda: self._toggle_derivative(pf.fid),
        )
        deriv.pack(side=tk.LEFT, padx=2)
        pf._deriv_btn = deriv
        HoverTip(deriv, "Plot derivative")

        vis = tk.Button(
            acts,
            text="Hide",
            bg=Theme.CARD,
            fg=Theme.TEXT_DIM,
            relief=tk.FLAT,
            bd=0,
            font=("Segoe UI", 9),
            cursor="hand2",
            command=lambda: self._toggle_visibility(pf.fid),
        )
        vis.pack(side=tk.LEFT, padx=2)
        pf._vis_btn = vis

        delete = tk.Button(
            acts,
            text="x",
            bg=Theme.CARD,
            fg=Theme.ERROR,
            relief=tk.FLAT,
            bd=0,
            font=("Segoe UI", 11, "bold"),
            cursor="hand2",
            command=lambda: self._delete_function(pf.fid),
        )
        delete.pack(side=tk.LEFT)

    def _toggle_derivative(self, fid: int):
        pf = self.functions.get(fid)
        if not pf or pf.sym_expr is None:
            return
        pf.show_deriv = not pf.show_deriv

        if pf._deriv_btn:
            if pf.show_deriv:
                pf._deriv_btn.config(bg=pf.color, fg=Theme.BG)
            else:
                pf._deriv_btn.config(bg="#1E293B", fg=Theme.TEXT_DIM)

        if pf.show_deriv and pf.deriv_lambda is None:
            deriv_expr = sp.diff(pf.sym_expr, self.SYM_TUPLE[0])
            pf.deriv_lambda = sp.lambdify(self.SYM_TUPLE, deriv_expr, modules="numpy")

        if not pf.show_deriv and pf.deriv_artist:
            pf.deriv_artist.set_data([], [])

        self._redraw()

    def _toggle_visibility(self, fid: int):
        pf = self.functions.get(fid)
        if not pf:
            return
        pf.visible = not pf.visible
        if pf._vis_btn:
            pf._vis_btn.config(text="Hide" if pf.visible else "Show")
        if not pf.visible:
            for art in (pf.artist, pf.deriv_artist):
                if art:
                    art.set_data([], [])
        self._redraw()

    def _delete_function(self, fid: int):
        pf = self.functions.pop(fid, None)
        if not pf:
            return
        for art in (pf.artist, pf.deriv_artist):
            if art:
                try:
                    art.remove()
                except Exception:
                    pass
        pf.artist = pf.deriv_artist = None
        if pf._ui_card:
            pf._ui_card.destroy()
        self._redraw()

    # ========================================================================
    # Derivative / Integral from entry
    # ========================================================================
    def _on_deriv(self):
        raw = self.entry.get().strip()
        if not raw:
            return
        try:
            _, sym = self._compile(raw)
            deriv = sp.diff(sym, self.SYM_TUPLE[0])
            dstr = str(deriv)
            if dstr == "0":
                self._set_status("Derivative is zero", Theme.WARN)
            self.add_function(dstr, display=f"d/dx({raw})")
        except Exception as exc:
            self._set_status(f"Derivative error: {exc}", Theme.ERROR)
            messagebox.showwarning("Derivative Error", str(exc))

    def _on_integral(self):
        raw = self.entry.get().strip()
        if not raw:
            return
        try:
            _, sym = self._compile(raw)
            integral = sp.integrate(sym, self.SYM_TUPLE[0])
            if isinstance(integral, sp.Integral):
                raise ValueError("Cannot integrate symbolically.")
            istr = str(integral)
            self.add_function(istr, display=f"integral({raw})")
        except Exception as exc:
            self._set_status(f"Integral error: {exc}", Theme.ERROR)
            messagebox.showwarning("Integral Error", str(exc))

    # ========================================================================
    # Rendering
    # ========================================================================
    def _sanitize(self, y: np.ndarray) -> np.ndarray:
        if np.iscomplexobj(y):
            y = np.where(np.isclose(y.imag, 0, atol=1e-9), y.real, np.nan)
        y = np.asarray(y, dtype=float)
        y[~np.isfinite(y)] = np.nan
        return y

    def _redraw(self):
        vals = [self.vars[p].get() for p in self.PARAMS]
        any_finite = False

        for pf in self.functions.values():
            if not pf.visible:
                for art in (pf.artist, pf.deriv_artist):
                    if art:
                        art.set_data([], [])
                continue

            # Main curve
            try:
                with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                    y = pf.lambda_fn(self._x_data, *vals)
                y = self._sanitize(y)

                if pf.artist is None:
                    (pf.artist,) = self.ax.plot(
                        self._x_data,
                        y,
                        color=pf.color,
                        linewidth=2.2,
                        solid_capstyle="round",
                    )
                else:
                    pf.artist.set_data(self._x_data, y)
                    pf.artist.set_color(pf.color)

                if np.any(np.isfinite(y)):
                    any_finite = True
            except Exception:
                if pf.artist:
                    pf.artist.set_data([], [])

            # Derivative overlay
            if pf.show_deriv and pf.deriv_lambda:
                try:
                    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                        yd = pf.deriv_lambda(self._x_data, *vals)
                    yd = self._sanitize(yd)

                    if pf.deriv_artist is None:
                        (pf.deriv_artist,) = self.ax.plot(
                            self._x_data,
                            yd,
                            color=pf.color,
                            linewidth=1.4,
                            linestyle="--",
                            alpha=0.65,
                            solid_capstyle="round",
                        )
                    else:
                        pf.deriv_artist.set_data(self._x_data, yd)
                        pf.deriv_artist.set_color(pf.color)
                except Exception:
                    if pf.deriv_artist:
                        pf.deriv_artist.set_data([], [])
            elif pf.deriv_artist:
                pf.deriv_artist.set_data([], [])

        if any_finite and self.autoscale_var.get():
            self._run_autoscale()

        self.canvas.draw_idle()

    def _run_autoscale(self):
        segments = []
        for pf in self.functions.values():
            if not pf.visible:
                continue
            for art in (pf.artist, pf.deriv_artist):
                if art is None:
                    continue
                y = art.get_ydata()
                ok = y[np.isfinite(y)]
                if len(ok):
                    segments.append(ok)
        if not segments:
            return
        all_y = np.concatenate(segments)
        if len(all_y) == 0:
            return
        q1, q99 = np.percentile(all_y, [1, 99])
        span = q99 - q1
        if span < 1e-6:
            span = 2.0
        margin = span * 0.15
        self.ax.set_ylim(q1 - margin, q99 + margin)

    def _on_slider(self, name: str):
        val = self.vars[name].get()
        self.var_labels[name].config(text=f"{val:.1f}")
        self._redraw()

    # ========================================================================
    # Viewport
    # ========================================================================
    def _apply_viewport(self):
        try:
            xmin, xmax = float(self.xmin_var.get()), float(self.xmax_var.get())
            ymin, ymax = float(self.ymin_var.get()), float(self.ymax_var.get())
            if xmin >= xmax or ymin >= ymax:
                raise ValueError("Min must be less than max.")
            self._x_data = np.linspace(xmin, xmax, 2000)
            self.ax.set_xlim(xmin, xmax)
            self.ax.set_ylim(ymin, ymax)
            self.autoscale_var.set(False)
            self._redraw()
            self._set_status("Viewport applied", Theme.SUCCESS)
        except Exception as exc:
            self._set_status(f"Viewport error: {exc}", Theme.ERROR)
            messagebox.showwarning("Viewport Error", str(exc))

    def _on_view_btn(self, label: str):
        if label == "Reset":
            self.xmin_var.set("-10")
            self.xmax_var.set("10")
            self.ymin_var.set("-10")
            self.ymax_var.set("10")
            self._apply_viewport()
        elif label == "Fit":
            self.autoscale_var.set(True)
            self._redraw()
        elif label == "Square":
            self.ax.set_aspect("equal", adjustable="box")
            self.canvas.draw_idle()
            self._set_status("Square aspect enabled", Theme.SUCCESS)

    # ========================================================================
    # Keypad & Entry helpers
    # ========================================================================
    def _insert_text(self, text: str):
        self.entry.insert(tk.INSERT, text)
        self.entry.focus_set()
        self.entry.icursor(self.entry.index(tk.INSERT))

    def _backspace(self):
        pos = self.entry.index(tk.INSERT)
        if pos != "0":
            self.entry.delete(pos + "-1c", pos)
        self.entry.focus_set()

    def _clear_entry(self):
        self.entry.delete(0, tk.END)
        self.entry.focus_set()

    def _on_plot(self):
        self.add_function(self.entry.get())

    # ========================================================================
    # Interaction
    # ========================================================================
    def _on_mouse_move(self, event):
        if event.inaxes:
            self.coord_lbl.config(
                text=f"x = {event.xdata:.3f}    y = {event.ydata:.3f}"
            )
        else:
            self.coord_lbl.config(text="")

    def _set_status(self, text: str, color: str = Theme.TEXT_DIM):
        self.status_lbl.config(text=text, fg=color)
        if self._status_after:
            self.after_cancel(self._status_after)
        self._status_after = self.after(
            4000, lambda: self.status_lbl.config(text="Ready", fg=Theme.TEXT_DIM)
        )


# -----------------------------------------------------------------------------
# Entry
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    app = GraphingCalculator()
    app.mainloop()