"""PROTOTYPE: self-contained Plotly views of the VTS1 saturation landscape."""

from pathlib import Path

import numpy as np
import plotly.graph_objects as go


ROOT = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parents[5] / "mRNA_RBP" / "prototypes" / "pairwise_saturation_landscape"
LIBRARY = ROOT / "mRNA_RBP/runs/deepsquid/vts1/high/instance_00/type3.npz"
WT_PATH = ROOT / "mRNA_RBP/runs/deepsquid/vts1/high/instance_00/wt_seq.txt"
OUTPUT = HERE / "interactive_activity_landscape.html"
ALPHABET = np.asarray(list("ACGU"))
SCORE_KEY = "scores_deepsquid_vts1"


def load_points():
    wt = WT_PATH.read_text().strip()
    wt_ids = np.asarray([np.flatnonzero(ALPHABET == base)[0] for base in wt])
    with np.load(LIBRARY) as data:
        sequences = data["nuc_ids"]
        scores = data[SCORE_KEY].astype(float)

    x, y, hover, orders = [], [], [], []
    for sequence, score in zip(sequences, scores):
        changed = np.flatnonzero(sequence != wt_ids)
        states = [4 * int(pos) + int(sequence[pos]) + 1 for pos in changed]
        mutations = [
            f"{wt[pos]}{pos + 1}{ALPHABET[sequence[pos]]}" for pos in changed
        ]
        x.append(states[0])
        y.append(states[-1])
        orders.append(len(changed))
        hover.append(
            f"Mutation: {' + '.join(mutations)}"
            f"<br>Activity: {score:.4f}"
            f"<br>Sequence: {''.join(ALPHABET[sequence])}"
        )
    return np.asarray(x), np.asarray(y), scores, np.asarray(hover), np.asarray(orders)


def axes_2d():
    ticks = np.arange(0, 41, 5) * 4 + 2.5
    labels = [str(value) for value in np.arange(1, 42, 5)]
    common = dict(
        range=[0, 165],
        tickmode="array",
        tickvals=ticks,
        ticktext=labels,
        gridcolor="#e7e7e7",
        zeroline=False,
    )
    return common


def make_2d(x, y, scores, hover, orders):
    doubles = orders == 2
    singles = ~doubles
    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=x[doubles], y=y[doubles], mode="markers", name="Double mutants",
        text=hover[doubles], hovertemplate="%{text}<extra></extra>",
        marker=dict(size=5, color=scores[doubles], colorscale="Viridis",
                    cmin=scores.min(), cmax=scores.max(),
                    colorbar=dict(title="Activity score")),
    ))
    fig.add_trace(go.Scattergl(
        x=x[singles], y=y[singles], mode="markers", name="Single mutants",
        text=hover[singles], hovertemplate="%{text}<extra></extra>",
        marker=dict(size=8, color=scores[singles], colorscale="Viridis",
                    cmin=scores.min(), cmax=scores.max(),
                    line=dict(color="white", width=0.5), showscale=False),
    ))
    axis = axes_2d()
    fig.update_layout(
        title=dict(text="2D mutation map", x=0.5), template="plotly_white",
        height=820, margin=dict(l=80, r=80, t=80, b=80),
        xaxis=dict(**axis, title="First mutation: position × nucleotide (A C G U)"),
        yaxis=dict(**axis, title="Second mutation: position × nucleotide (A C G U)",
                   scaleanchor="x", scaleratio=1),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.06),
    )
    return fig


def make_3d(x, y, scores, hover, orders):
    doubles = orders == 2
    singles = ~doubles
    fig = go.Figure()
    for mask, name, size, scale in [
        (doubles, "Double mutants", 3, True),
        (singles, "Single mutants", 5, False),
    ]:
        fig.add_trace(go.Scatter3d(
            x=x[mask], y=y[mask], z=scores[mask], mode="markers", name=name,
            text=hover[mask], hovertemplate="%{text}<extra></extra>",
            marker=dict(size=size, color=scores[mask], colorscale="Viridis",
                        cmin=scores.min(), cmax=scores.max(), opacity=0.82,
                        showscale=scale,
                        colorbar=dict(title="Activity score") if scale else None),
        ))
    ticks = np.arange(0, 41, 10) * 4 + 2.5
    labels = [str(value) for value in np.arange(1, 42, 10)]
    axis = dict(tickmode="array", tickvals=ticks, ticktext=labels,
                range=[0, 165], gridcolor="#d8d8d8", zeroline=False)
    fig.update_layout(
        title=dict(text="3D activity landscape", x=0.5), template="plotly_white",
        height=820, margin=dict(l=10, r=20, t=80, b=20),
        scene=dict(
            xaxis=dict(**axis, title="First mutation position × nucleotide"),
            yaxis=dict(**axis, title="Second mutation position × nucleotide"),
            zaxis=dict(title="Activity score"),
            camera=dict(eye=dict(x=1.55, y=-1.65, z=1.05)),
        ),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.04),
    )
    return fig


def write_page(two_d, three_d):
    plot_2d = two_d.to_html(full_html=False, include_plotlyjs=True,
                            config={"displaylogo": False, "scrollZoom": True})
    plot_3d = three_d.to_html(full_html=False, include_plotlyjs=False,
                             config={"displaylogo": False, "scrollZoom": True})
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Interactive pairwise saturated mutagenesis landscape</title>
<style>
body {{ margin: 0; color: #222; font: 16px system-ui, sans-serif; background: #fafafa; }}
header {{ text-align: center; padding: 22px 16px 2px; }}
h1 {{ margin: 0 0 6px; font-size: 25px; }}
p {{ margin: 0; color: #666; }}
.tabs {{ display: flex; justify-content: center; gap: 8px; padding: 16px 0 2px; }}
button {{ border: 1px solid #bbb; border-radius: 999px; padding: 8px 16px; background: white; cursor: pointer; }}
button.active {{ color: white; background: #222; border-color: #222; }}
.panel {{ display: none; max-width: 1180px; margin: auto; }}
.panel.active {{ display: block; }}
</style></head><body>
<header><h1>Pairwise saturated mutagenesis landscape — VTS1</h1>
<p>7,503 measured sequences · hover for mutations and activity · drag to zoom or rotate</p></header>
<nav class="tabs"><button class="active" data-panel="map">2D map</button>
<button data-panel="landscape">3D landscape</button></nav>
<main><section id="map" class="panel active">{plot_2d}</section>
<section id="landscape" class="panel">{plot_3d}</section></main>
<script>
document.querySelectorAll('button[data-panel]').forEach(button => button.onclick = () => {{
  document.querySelectorAll('button[data-panel], .panel').forEach(x => x.classList.remove('active'));
  button.classList.add('active');
  const panel = document.getElementById(button.dataset.panel); panel.classList.add('active');
  panel.querySelectorAll('.plotly-graph-div').forEach(Plotly.Plots.resize);
}});
</script></body></html>"""
    OUTPUT.write_text(page)


def main():
    points = load_points()
    write_page(make_2d(*points), make_3d(*points))
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
