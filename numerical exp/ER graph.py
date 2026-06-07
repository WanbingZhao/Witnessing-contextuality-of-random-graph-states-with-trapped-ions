import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import networkx as nx
import numpy as np

# -----------------------------------------------------------------------------
# PRL-like Erdős-Rényi graph-state schematic with all edges in red tones.
# -----------------------------------------------------------------------------

N = 10
P_EDGE = 0.30
GRAPH_SEED = 7
SUPPORT = (0, 4, 6, 8)  # g_1 g_5 g_7 g_9
OUT_BASENAME = "er_graph_state_publication_prl_red_edges"

COLORS = {
    "complete_edge": "#D0D0D0",   # faint grey for the complete-graph backbone
    "sample_edge": "#C62828",     # strong red for the sampled ER edges
    "node_edge": "#243447",
    "node_fill": "#FFFFFF",
    "support_ring": "#B85833",
    "support_fill": "#FFF6F0",
    "pauli_I": "#6E7A89",
    "pauli_X": "#355F94",
    "pauli_Y": "#A54E6C",
    "pauli_Z": "#3D8D3A",
    "text": "#223344",
}

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.unicode_minus": False,
})

GRAPH_CENTER = np.array([0.0, 0.08])


def er_graph(n: int, p: float, seed: int):
    return nx.erdos_renyi_graph(n, p, seed=seed)


def circle_positions(n: int, radius: float = 0.98, start_deg: float = 90.0):
    pos = {}
    for i in range(n):
        theta = math.radians(start_deg - i * 360.0 / n)
        pos[i] = GRAPH_CENTER + np.array([radius * math.cos(theta), radius * math.sin(theta)])
    return pos


def local_pauli_letters(G, support):
    A = nx.to_numpy_array(G, dtype=int)
    x = np.zeros(G.number_of_nodes(), dtype=int)
    for i in support:
        x[i] = 1
    z = (A @ x) % 2

    letters = []
    for xi, zi in zip(x, z):
        if xi == 0 and zi == 0:
            letters.append("I")
        elif xi == 1 and zi == 0:
            letters.append("X")
        elif xi == 0 and zi == 1:
            letters.append("Z")
        else:
            letters.append("Y")
    return letters


def sign_bit(G, support):
    x = np.zeros(G.number_of_nodes(), dtype=int)
    for i in support:
        x[i] = 1
    c = 0
    for i in range(G.number_of_nodes()):
        if x[i] == 0:
            continue
        neigh = list(G.neighbors(i))
        for a in range(len(neigh)):
            for b in range(a + 1, len(neigh)):
                j, k = neigh[a], neigh[b]
                c ^= int(x[j] & x[k])
    return c


def draw_complete_backbone(ax, pos, lw=0.85, alpha=0.34):
    n = len(pos)
    for i in range(n):
        for j in range(i + 1, n):
            ax.plot(
                [pos[i][0], pos[j][0]],
                [pos[i][1], pos[j][1]],
                color=COLORS["complete_edge"],
                lw=lw,
                alpha=alpha,
                zorder=1,
            )


def draw_sample_edges(ax, G, pos, lw=3.0):
    for u, v in G.edges():
        uu, vv = sorted((u, v))
        ax.plot(
            [pos[uu][0], pos[vv][0]],
            [pos[uu][1], pos[vv][1]],
            color=COLORS["sample_edge"],
            lw=lw,
            alpha=0.97,
            zorder=2,
            solid_capstyle="round",
        )


def draw_nodes(ax, pos, support):
    support = set(support)
    for i, (x, y) in pos.items():
        if i in support:
            outer = Circle(
                (x, y),
                0.090,
                facecolor=COLORS["support_fill"],
                edgecolor=COLORS["support_ring"],
                lw=2.0,
                zorder=4,
            )
            ax.add_patch(outer)
            inner = Circle((x, y), 0.064, facecolor=COLORS["node_fill"], edgecolor=COLORS["node_edge"], lw=1.35, zorder=5)
        else:
            inner = Circle((x, y), 0.068, facecolor=COLORS["node_fill"], edgecolor=COLORS["node_edge"], lw=1.35, zorder=4)
        ax.add_patch(inner)
        ax.text(
            x,
            y,
            f"{i + 1}",
            ha="center",
            va="center",
            fontsize=11.0,
            color=COLORS["text"],
            zorder=6,
        )


def draw_pauli_labels(ax, pos, letters):
    for i, letter in enumerate(letters):
        x, y = pos[i]
        r = np.array([x, y], dtype=float) - GRAPH_CENTER
        norm = np.linalg.norm(r)
        unit = np.array([0.0, 0.0]) if norm == 0 else r / norm
        tx, ty = x + 0.17 * unit[0], y + 0.17 * unit[1]
        ax.text(
            tx,
            ty,
            letter,
            ha="center",
            va="center",
            fontsize=10.8,
            color=COLORS[f"pauli_{letter}"],
            fontweight="bold",
            zorder=7,
        )


def draw_bottom_formula(ax, G, support, letters):
    support_1 = [i + 1 for i in support]
    c = sign_bit(G, support)
    sign = "-" if c == 1 else "+"
    spaced_letters = r"\,".join(letters)
    line1 = rf"$g_{{{support_1[0]}}}g_{{{support_1[1]}}}g_{{{support_1[2]}}}g_{{{support_1[3]}}}={sign}\,{spaced_letters}$"

    ax.text(
        0.0,
        -1.32,
        line1,
        ha="center",
        va="center",
        fontsize=12.6,
        color=COLORS["text"],
        zorder=10,
    )


def main():
    outdir = Path(__file__).resolve().parent
    G = er_graph(N, P_EDGE, GRAPH_SEED)
    pos = circle_positions(N)
    letters = local_pauli_letters(G, SUPPORT)

    fig, ax = plt.subplots(figsize=(6.0, 5.55), facecolor="white")
    ax.set_xlim(-1.30, 1.30)
    ax.set_ylim(-1.42, 1.24)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_complete_backbone(ax, pos)
    draw_sample_edges(ax, G, pos)
    draw_nodes(ax, pos, SUPPORT)
    draw_pauli_labels(ax, pos, letters)
    draw_bottom_formula(ax, G, SUPPORT, letters)

    fig.savefig(outdir / f"{OUT_BASENAME}.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(outdir / f"{OUT_BASENAME}.png", bbox_inches="tight", pad_inches=0.02, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
