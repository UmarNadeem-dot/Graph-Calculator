# 📈 Graphware — Graphing Calculator

**Graphware** is a high-performance desktop graphing calculator built in Python. Designed with a custom dark UI theme, it combines a custom mathematical parser, **SymPy** for symbolic algebra/calculus, **NumPy** for vector computation, and **Matplotlib** integrated into a **Tkinter** desktop wrapper.

---

## 🌟 Key Features

### 🧮 Advanced Mathematical Engine
- **Implicit Multiplication & Syntax Flexibility:** Automatically parses mathematical expressions like `3x`, `a*sin(b*x)`, `(x+1)(x-2)`, and `pi`.
- **Comprehensive Function Library:** Full support for trigonometric, inverse trigonometric, hyperbolic, inverse hyperbolic, exponential, logarithmic (`ln`/`log`), square root, absolute value, floor, ceiling, and gamma functions.
- **Dynamic Parameter Sliders:** Live control sliders for $a, b, c, d, y, z$ parameters allowing real-time transformation and parametric exploration.

### 📐 Symbolic Calculus Suite
- **Interactive Derivatives:** Toggle derivative curves ($f'(x)$) as dashed overlays directly on active functions.
- **Quick Action Calculus:** Compute and plot symbolic derivatives ($d/dx$) and indefinite integrals ($\int f(x) dx$) directly from the input bar.

### 🎨 Custom Dark GUI & Visualization
- **Modern Dark Palette:** Clean interface built using custom color variables, hover tooltips, and status updates.
- **Multi-Function Management:** Add, hide, show, toggle derivatives, or delete multiple stacked functions with unique visual color coding.
- **Viewport & Axis Control:** Auto-scaling $Y$-axis, manual domain/range input, square aspect ratio locking, and quick viewport resets.
- **Interactive Navigation:** Matplotlib toolbar support (pan, zoom, reset) and real-time cursor coordinate tracking ($x, y$).

---

## 🛠️ Architecture & Tech Stack

| Component | Technology | Description |
| :--- | :--- | :--- |
| **GUI Framework** | `Tkinter` (Python standard library) | Native windowing, custom dark theme, widget layouts |
| **Plotting Engine** | `Matplotlib` (`TkAgg` backend) | Interactive figure rendering and canvas navigation |
| **Symbolic Math** | `SymPy` | Symbolic parsing, differentiation, and integration |
| **Numerical Computation**| `NumPy` | Vectorized evaluation across 2,000 domain data points |
| **Parser / Lexer** | Custom Regex & Recursive Descent | Tokenizes and normalizes mathematical syntax |

---

## 🚀 Quick Start

### Prerequisites

Python 3.10 or higher is recommended. Install the required dependencies using `pip`:

```bash
pip install numpy sympy matplotlib