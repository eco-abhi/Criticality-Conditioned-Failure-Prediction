"""
bom_graph.py — Tiered automotive BOM DAG generator.

Edge direction: component -> assembly (a part points to what consumes it).
Tier 0 = raw / purchased components (leaf nodes, out-degree > 0, in-degree 0).
Tier K = top-level assemblies (root nodes, in-degree > 0, out-degree = 0 or small).

Algorithm
---------
1. Assign every part to a tier based on criticality and a random depth draw.
2. Build edges strictly upward: parts in tier T feed assemblies in tier T+1.
3. Guarantee every non-root part has at least one consuming assembly.

No public dataset gives real BOM depth/fan-out structure for this domain (investigated; see
data_dictionary.md's BOM citation and scripts/check_bom_benchmark.py) -- this graph is entirely
synthetic, literature-anchored (CIRP Annals), with a real-data face-validity scale check only.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import pandas as pd


def generate_bom_dag(
    part_ids: List[str],
    depth_min: int,
    depth_max: int,
    fanout_min: int,
    fanout_max: int,
    criticality: pd.Series,
    rng: np.random.Generator,
    latent: Optional[pd.Series] = None,
) -> nx.DiGraph:
    """
    Build a tiered DAG where edges run component -> assembly.

    Parameters
    ----------
    part_ids   : ordered list of part_id strings (length N).
    depth_min  : minimum number of BOM tiers.
    depth_max  : maximum number of BOM tiers.
    fanout_min : minimum components consumed per assembly node.
    fanout_max : maximum components consumed per assembly node.
    criticality: pd.Series indexed by part_id with values in {A, B, C} (stored on nodes; labels).
    rng        : numpy Generator for reproducibility.
    latent     : optional continuous score per part_id; when provided, **tier placement** uses
                 latent tertiles mapped to low/mid/high tier biases (same machinery as A/B/C),
                 so graph topology is driven by latent similarity to tier, not by ABC class.

    Returns
    -------
    nx.DiGraph with node attribute 'criticality_class'.
    """
    n = len(part_ids)
    n_tiers = int(rng.integers(depth_min, depth_max + 1))

    # ------------------------------------------------------------------
    # 1. Assign tiers
    #    A-parts bias toward upper tiers (assemblies / subassemblies).
    #    C-parts bias toward lower tiers (raw / purchased components).
    # ------------------------------------------------------------------
    tier_of: Dict[str, int] = {}
    crit_dict = criticality.to_dict()

    tier_probs = {
        "A": _tier_weights(n_tiers, bias="high"),
        "B": _tier_weights(n_tiers, bias="mid"),
        "C": _tier_weights(n_tiers, bias="low"),
    }

    latent_tier_key: Dict[str, str] = {}
    if latent is not None:
        lv = np.array([float(latent.get(pid, np.nan)) for pid in part_ids], dtype=float)
        if not np.isfinite(lv).all():
            raise ValueError("latent series has non-finite values for BOM generation")
        q1, q2 = np.quantile(lv, [1.0 / 3.0, 2.0 / 3.0])
        for pid, v in zip(part_ids, lv):
            if v <= q1:
                latent_tier_key[pid] = "C"
            elif v <= q2:
                latent_tier_key[pid] = "B"
            else:
                latent_tier_key[pid] = "A"

    for pid in part_ids:
        if latent is not None:
            cls = latent_tier_key.get(pid, "C")
        else:
            cls = crit_dict.get(pid, "C")
        probs = tier_probs[cls]
        tier_of[pid] = int(rng.choice(n_tiers, p=probs))

    # Ensure at least one node per tier
    tiers: List[List[str]] = [[] for _ in range(n_tiers)]
    for pid, t in tier_of.items():
        tiers[t].append(pid)

    for t in range(n_tiers):
        if len(tiers[t]) == 0:
            # Steal a random part from the largest tier
            largest = max(range(n_tiers), key=lambda x: len(tiers[x]))
            pid = tiers[largest].pop(
                int(rng.integers(len(tiers[largest])))
            )
            tiers[t].append(pid)
            tier_of[pid] = t

    # Force middle tiers to have meaningful population (intermediate assemblies).
    min_mid_size = max(2, n // (n_tiers * 3))
    for t in range(1, n_tiers - 1):
        while len(tiers[t]) < min_mid_size:
            largest = max(
                (i for i in range(n_tiers) if i != t and len(tiers[i]) > min_mid_size),
                key=lambda x: len(tiers[x]),
                default=None,
            )
            if largest is None:
                break
            pid = tiers[largest].pop(int(rng.integers(len(tiers[largest]))))
            tiers[t].append(pid)
            tier_of[pid] = t

    # Cap extreme tiers so the DAG is not leaf-dominated or root-dominated.
    # Targets: keep in-degree-0 (tier-0 leaves) and out-degree-0 (top assemblies) from
    # swamping the graph when random tier draws are skewed despite the middle-tier pass.
    max_t0 = max(2, n // 6)
    while len(tiers[0]) > max_t0 and n_tiers >= 2:
        pid = tiers[0].pop(int(rng.integers(len(tiers[0]))))
        tiers[1].append(pid)
        tier_of[pid] = 1

    top_t = n_tiers - 1
    max_top = max(2, n // 26)
    while len(tiers[top_t]) > max_top and n_tiers >= 3:
        dest = n_tiers - 2
        pid = tiers[top_t].pop(int(rng.integers(len(tiers[top_t]))))
        tiers[dest].append(pid)
        tier_of[pid] = dest

    # ------------------------------------------------------------------
    # 2. Build edges: tier T -> tier T+1
    #    Each assembly node in tier T+1 consumes fanout_min..fanout_max
    #    components from tier T.
    #    Each component in tier T must be consumed by >= 1 assembly.
    # ------------------------------------------------------------------
    G = nx.DiGraph()
    G.add_nodes_from(part_ids)
    nx.set_node_attributes(G, crit_dict, "criticality_class")

    for t in range(n_tiers - 1):
        components = tiers[t]       # feed upward
        assemblies = tiers[t + 1]   # consume from below

        if not components or not assemblies:
            continue

        # Track which components have been consumed
        consumed: set[str] = set()

        # Each assembly picks fanout components from the tier below
        for asm in assemblies:
            k = int(rng.integers(fanout_min, fanout_max + 1))
            k = min(k, len(components))
            chosen = rng.choice(components, size=k, replace=False).tolist()
            for comp in chosen:
                G.add_edge(comp, asm)
                consumed.add(comp)

        # Guarantee every component is consumed by at least one assembly
        unconsumed = [c for c in components if c not in consumed]
        for comp in unconsumed:
            asm = assemblies[int(rng.integers(len(assemblies)))]
            G.add_edge(comp, asm)

    # ------------------------------------------------------------------
    # 3. Safety: remove any cycles introduced by the random wiring
    #    (should not happen with strict tier ordering, but guard anyway)
    # ------------------------------------------------------------------
    if not nx.is_directed_acyclic_graph(G):
        edges_to_remove = list(nx.find_cycle(G))
        G.remove_edges_from(edges_to_remove)

    assert nx.is_directed_acyclic_graph(G), "BOM graph is not a DAG after cycle removal."
    assert G.number_of_nodes() == n, "Node count mismatch."

    return G


def compute_bom_position_features(
    G: nx.DiGraph,
    crit_dict: Dict[str, str],
    latent_by_part: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """
    Compute per-node BOM position features.

    Features
    --------
    bom_in_degree                   : number of assemblies directly consuming this part.
    bom_out_degree                  : number of components this part directly consumes.
    bom_longest_downstream_path     : longest path from this node to any root assembly.
    bom_n_downstream_A_assemblies   : count of A-class nodes reachable downstream.
    bom_criticality_propagation_score : weighted cascade exposure score.
    latent_by_part : optional latent scores; when set, cascade weights and the downstream
        ``bom_n_downstream_A_assemblies`` counter use **latent tertiles** (high-latent ≡ top third),
        not ABC labels, so BOM summaries are not pure functions of ``criticality_class``.
    """
    weight = {"A": 3.0, "B": 1.5, "C": 0.5}
    topo = list(nx.topological_sort(G))

    lat_vals: Dict[str, float] = {}
    q33 = 0.0
    q66 = 0.0
    if latent_by_part is not None:
        lat_vals = {n: float(latent_by_part.get(n, 0.0)) for n in G.nodes()}
        arr = np.array(list(lat_vals.values()), dtype=float)
        q33, q66 = (float(x) for x in np.quantile(arr, [1.0 / 3.0, 2.0 / 3.0]))

    def _node_weight(node: str) -> float:
        if latent_by_part is not None:
            v = lat_vals[node]
            if v <= q33:
                return 0.5
            if v <= q66:
                return 1.5
            return 3.0
        return weight.get(crit_dict.get(node, "C"), 0.5)

    def _node_is_high_signal(node: str) -> int:
        if latent_by_part is not None:
            return int(lat_vals[node] >= q66)
        return int(crit_dict.get(node, "C") == "A")

    # Initialize per-node accumulators
    longest = {n: 0 for n in G.nodes()}
    n_downstream_A = {n: 0 for n in G.nodes()}
    cascade_score = {n: 0.0 for n in G.nodes()}

    # Single forward pass in topological order
    for node in topo:
        node_weight = _node_weight(node)
        node_is_A = _node_is_high_signal(node)
        for successor in G.successors(node):
            # Propagate longest path
            if longest[node] + 1 > longest[successor]:
                longest[successor] = longest[node] + 1
            # Propagate cascade score and A-count
            cascade_score[successor] += cascade_score[node] + node_weight
            n_downstream_A[successor] += n_downstream_A[node] + node_is_A

    records = [
        {
            "part_id": node,
            "bom_in_degree": G.in_degree(node),
            "bom_out_degree": G.out_degree(node),
            "bom_longest_downstream_path": longest[node],
            "bom_n_downstream_A_assemblies": n_downstream_A[node],
            "bom_criticality_propagation_score": cascade_score[node],
        }
        for node in G.nodes()
    ]
    return pd.DataFrame(records).set_index("part_id")


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _tier_weights(n_tiers: int, bias: str) -> np.ndarray:
    """
    Softmax-normalized weights over tiers.
    bias='low'  -> C-parts: strongly toward tier 0 and 1.
    bias='mid'  -> B-parts: middle tiers.
    bias='high' -> A-parts: strongly toward top tiers.
    """
    x = np.arange(n_tiers, dtype=float)
    if bias == "low":
        logits = -x * 2.5
    elif bias == "high":
        logits = x * 2.5
    else:
        mid = (n_tiers - 1) / 2.0
        logits = -((x - mid) ** 2) * 0.8
    e = np.exp(logits - logits.max())
    return e / e.sum()