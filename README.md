# quatplot

Interactive 3D plots for Jupyter with quaternion-based view controls — no gimbal lock.

Quaternion math is powered by [ganja.js](https://github.com/enkimute/ganja.js)
(`Algebra(0,2)` ≅ ℍ) when its CDN is reachable, with a built-in quaternion
fallback otherwise. Rendering is plain HTML5 canvas inside an `<iframe srcdoc>`,
so plots are self-contained: multiple plots per notebook can't collide on IDs,
CSS, or globals, and no notebook-side JavaScript is required.

## Features

- Surface plots from 2D mesh data (colored fill and/or wireframe, toggleable)
- 3D scatter plots with colored markers and hover coordinate tooltips
- Optional color bar for scatter values
- Isometric ⇄ perspective projection toggle
- Plotly-style mouse controls: drag to rotate, wheel to zoom, shift-drag / right-drag to pan
- Top view button (straight down, x right / y up — isometric top view reads as a 2D plot)
- Auto-scaled, labeled axes with nice-number tick marks
- Light / dark themes; Viridis, Plasma, Jet, Coolwarm, Grayscale colormaps
- PNG screenshot button
- Resizable plots in Jupyter: drag the bottom-right corner
- Standalone HTML export

## Install

```bash
pip install git+https://github.com/drscotthawley/quatplot.git
```

(or `uv pip install ...`; PyPI release may follow).

## Usage

```python
import numpy as np
import quatplot as qp

x = np.linspace(-3, 3, 41)
y = np.linspace(-3, 3, 41)
X, Y = np.meshgrid(x, y)
Z = np.sin(X) * np.cos(Y)

qp.plot(qp.Surface(x, y, Z))                       # in a notebook cell

qp.plot(qp.Scatter3(px, py, pz, v=values),         # scatter, colored by v
        colorbar=True, cmap="Plasma")

qp.plot(qp.Surface(x, y, Z), qp.Scatter3(px, py, pz),  # overlay traces
        theme="light", proj="perspective", height=600)

qp.save("plot.html", qp.Surface(x, y, Z))          # standalone HTML file
```

`plot()` options: `theme` ("dark"/"light"), `proj` ("iso"/"perspective"),
`cmap`, `fill`, `wire`, `colorbar`, `height` (initial px; plots are
corner-drag resizable afterward).

See [examples/quatplot_demo.ipynb](examples/quatplot_demo.ipynb) for a working tour.

## Development

```bash
git clone https://github.com/drscotthawley/quatplot.git
cd quatplot
pip install -e .
```

The entire library is one file, `quatplot.py`; the HTML/JS template embedded in
it is the single source of truth for both notebook rendering and HTML export.

## License

MIT — see [LICENSE](LICENSE).
