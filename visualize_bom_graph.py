import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent


def _anchor_and_radius2_subgraph(G: nx.DiGraph) -> tuple[nx.DiGraph, str, nx.DiGraph]:
    """Pick assembly anchor; radius-2 ego on R = G.reverse(copy=True). Fallback if out-degree hub is leaf in R."""
    degrees_out = dict(G.out_degree())
    top_out = max(degrees_out, key=degrees_out.get)
    R = G.reverse(copy=True)
    sub_nodes = set(nx.ego_graph(R, top_out, radius=2).nodes())
    if len(sub_nodes) >= 2:
        top_node = top_out
        print(f"Anchor (out-degree): {top_out}  out_degree={degrees_out[top_out]}")
    else:
        degrees_in = dict(G.in_degree())
        top_node = max(degrees_in, key=degrees_in.get)
        sub_nodes = set(nx.ego_graph(R, top_node, radius=2).nodes())
        print(f"Anchor (in-degree fallback): {top_node}  in_degree={degrees_in[top_node]}")
        print(f"(out-degree pick {top_out} had no downstream in reversed graph)")
    sub2 = G.subgraph(sub_nodes).copy()
    return sub2, top_node, R


def _tiered_positions(sub2: nx.DiGraph, R: nx.DiGraph, top_node: str) -> dict:
    tier_map = nx.single_source_shortest_path_length(R, top_node, cutoff=2)
    for node in sub2.nodes():
        sub2.nodes[node]["layer"] = int(tier_map.get(node, 0))
    return nx.multipartite_layout(sub2, subset_key="layer", align="horizontal", scale=3.0)


def _draw_tiered_bom(ax, sub2: nx.DiGraph, pos: dict, sub2_colors: list, title: str) -> None:
    nx.draw_networkx(
        sub2,
        pos=pos,
        ax=ax,
        node_color=sub2_colors,
        node_size=120,
        font_size=0,
        arrows=True,
        arrowsize=8,
        edge_color="#B4B2A9",
        width=0.5,
        with_labels=False,
    )
    legend_elements = [
        Patch(facecolor="#E24B4A", label="A-part (critical)"),
        Patch(facecolor="#EF9F27", label="B-part (moderate)"),
        Patch(facecolor="#1D9E75", label="C-part (commodity)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.axis("off")


# Load graph and catalog
with open(ROOT / "outputs/bom_graph.gpickle", "rb") as f:
    G = pickle.load(f)

pc = pd.read_csv(ROOT / "outputs/part_catalog.csv")
crit_map = pc.set_index("part_id")["criticality_class"].to_dict()

color_map = {"A": "#E24B4A", "B": "#EF9F27", "C": "#1D9E75"}

sub2, top_node, R = _anchor_and_radius2_subgraph(G)
sub2_colors = [color_map.get(crit_map.get(n, "C"), "#888780") for n in sub2.nodes()]
pos = _tiered_positions(sub2, R, top_node)

print(f"Radius-2 subgraph: {sub2.number_of_nodes()} nodes, {sub2.number_of_edges()} edges")

fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Left: degree distribution
in_degrees = [d for _, d in G.in_degree()]
axes[0].hist(in_degrees, bins=40, color="#378ADD", edgecolor="white", linewidth=0.5)
axes[0].set_title("BOM in-degree distribution (full graph)", fontsize=13)
axes[0].set_xlabel("In-degree (assemblies consuming this part)")
axes[0].set_ylabel("Count")
axes[0].axvline(
    np.mean(in_degrees),
    color="#E24B4A",
    linestyle="--",
    label=f"Mean: {np.mean(in_degrees):.1f}",
)
axes[0].legend()

# Right: tiered multipartite layout (BFS depth in R from top assembly)
_draw_tiered_bom(
    axes[1],
    sub2,
    pos,
    sub2_colors,
    title=(
        f"BOM assembly hierarchy (top assembly, radius=2)\n"
        f"{sub2.number_of_nodes()} nodes, {sub2.number_of_edges()} edges\n"
        f"Left = top assembly | Right = components"
    ),
)

plt.suptitle("Bill of Materials graph structure", fontsize=15)
plt.tight_layout()
plt.savefig(ROOT / "outputs/bom_graph_visualization_v2.png", dpi=150, bbox_inches="tight")
plt.close()
print("Wrote", ROOT / "outputs/bom_graph_visualization_v2.png")

# Standalone tiered figure (same layout, publication-friendly size)
fig2, ax2 = plt.subplots(figsize=(14, 8))
_draw_tiered_bom(
    ax2,
    sub2,
    pos,
    sub2_colors,
    title=(
        f"BOM assembly hierarchy (top assembly, radius=2)\n"
        f"{sub2.number_of_nodes()} nodes, {sub2.number_of_edges()} edges\n"
        f"Left = top assembly | Right = components"
    ),
)
plt.tight_layout()
plt.savefig(ROOT / "outputs/bom_subgraph_tiered.png", dpi=150, bbox_inches="tight")
plt.close()
print("Wrote", ROOT / "outputs/bom_subgraph_tiered.png")
