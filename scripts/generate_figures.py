#!/usr/bin/env python3
"""Regenerate the curated, cross-experiment figures from stored results only."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FIGSIZE = (12, 6.75)
DPI = 160
COMMON_Y = (1.28, 2.10)
COLORS = {
    "candidate": "#356AA0",
    "best": "#E69F00",
    "baseline": "#4D4D4D",
    "root": "#B8BDC7",
    "improvement": "#2EAD64",
    "regression": "#D9534F",
    "failed": "#8C8C8C",
    "unknown": "#F2F2F2",
    "edge": "#59636E",
}

EXPERIMENTS = [
    ("00_linear_search", "Linear Search"),
    ("01_uniform_random_tree", "Uniform Random Tree Search"),
    ("02_uniform_random_dag", "Uniform Random DAG Search"),
    ("03_credit_guided_parent_dag", "Credit-Guided Parent DAG"),
    ("04_merge_retry_credit_dag", "Merge-Retry Credit-Guided DAG"),
    ("05_outcome_aware_proposal_memory_dag", "Outcome-Aware Proposal-Memory DAG"),
    ("06_outcome_blind_proposal_memory_dag", "Outcome-Blind Proposal-Memory DAG"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def val(node):
    result = node.get("result") or {}
    value = result.get("val_bpb") if isinstance(result, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def linear_series(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    baseline = float(rows[0]["val_bpb"])
    values = [float(row["val_bpb"]) for row in rows[1:31]]
    assert len(values) == 30
    return baseline, values


def dag_series(path: Path):
    state = json.loads(path.read_text())
    baseline_node = next(node for node in state["nodes"] if node.get("mode") == "baseline")
    valid = [
        node for node in state["nodes"]
        if node.get("mode") != "baseline" and node.get("status") == "finished" and val(node) is not None
    ]
    assert len(valid) == 30, (path, len(valid))
    return val(baseline_node), [val(node) for node in valid], state, valid


def plot_performance(directory: Path, title: str, baseline: float, values: list[float]):
    x = np.arange(1, 31)
    best = np.minimum.accumulate(values)
    local_low = min(values + [baseline])
    local_high = max(values + [baseline])
    pad = max((local_high - local_low) * 0.10, 0.008)

    for filename, ylim, subtitle in (
        ("performance_trajectory_common_scale.png", COMMON_Y, "common scale"),
        ("performance_trajectory_local_scale.png", (local_low - pad, local_high + pad), "local scale"),
        ("performance_trajectory.png", COMMON_Y, "common scale"),
    ):
        fig, ax = plt.subplots(figsize=FIGSIZE)
        ax.plot(x, values, color=COLORS["candidate"], marker="o", markersize=4.5,
                linewidth=1.25, label="Candidate val_bpb", zorder=3)
        ax.plot(x, best, color=COLORS["best"], linewidth=2.5,
                label="Candidate best-so-far", zorder=4)
        ax.axhline(baseline, color=COLORS["baseline"], linestyle="--", linewidth=1.8,
                   label=f"Baseline/root ({baseline:.6f})", zorder=2)
        ax.set_xlim(0.5, 30.5)
        ax.set_ylim(*ylim)
        ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
        ax.set_xlabel("Candidate / iteration (1–30)")
        ax.set_ylabel("Validation BPB (lower is better)")
        ax.set_title(f"{title} — Performance Trajectory ({subtitle})")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="best", frameon=True)
        fig.tight_layout()
        fig.savefig(directory / "figures" / filename, bbox_inches="tight")
        plt.close(fig)


def reference_value(node, by_id):
    parents = node.get("parents", [])
    parent_values = [val(by_id[parent]) for parent in parents if parent in by_id]
    parent_values = [value for value in parent_values if value is not None]
    return min(parent_values) if parent_values else None


def node_class(node, by_id):
    if node.get("mode") == "baseline":
        return "root"
    if node.get("status") == "failed":
        return "failed"
    current = val(node)
    reference = reference_value(node, by_id)
    if current is None or reference is None:
        return "unknown"
    return "improvement" if current < reference else "regression"


def depths(nodes):
    by_id = {node["id"]: node for node in nodes}
    memo = {}

    def depth(node_id, active=frozenset()):
        if node_id in memo:
            return memo[node_id]
        if node_id in active:
            raise ValueError(f"cycle detected at {node_id}")
        parents = [parent for parent in by_id[node_id].get("parents", []) if parent in by_id]
        answer = 0 if not parents else 1 + max(depth(parent, active | {node_id}) for parent in parents)
        memo[node_id] = answer
        return answer

    return {node_id: depth(node_id) for node_id in by_id}


def topology_position(nodes):
    depth = depths(nodes)
    layers = defaultdict(list)
    order = {node["id"]: index for index, node in enumerate(nodes)}
    for node in nodes:
        layers[depth[node["id"]]].append(node["id"])
    positions = {}
    max_size = max(len(layer) for layer in layers.values())
    for level, ids in layers.items():
        ids.sort(key=lambda node_id: order[node_id])
        ys = np.linspace((max_size - 1) / 2, -(max_size - 1) / 2, len(ids))
        for node_id, y in zip(ids, ys):
            positions[node_id] = (level, float(y))
    return positions, depth


def draw_topology(directory: Path, title: str, state: dict, success: bool):
    nodes = [
        node for node in state["nodes"]
        if node.get("mode") == "baseline"
        or node.get("status") == "failed"
        or (node.get("status") == "finished" and val(node) is not None)
    ]
    by_id = {node["id"]: node for node in nodes}
    graph = nx.DiGraph()
    graph.add_nodes_from(by_id)
    for node in nodes:
        for parent in node.get("parents", []):
            if parent in by_id:
                graph.add_edge(parent, node["id"])
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError(f"non-DAG topology in {directory}")
    pos, depth = topology_position(nodes)

    fig, ax = plt.subplots(figsize=(16, 10))
    nx.draw_networkx_edges(
        graph, pos, ax=ax, edge_color=COLORS["edge"], width=1.15, alpha=0.75,
        arrows=True, arrowstyle="-|>", arrowsize=14, node_size=650,
        connectionstyle="arc3,rad=0.025", min_source_margin=8, min_target_margin=10,
    )
    groups = {
        "root": ("D", COLORS["root"]),
        "expand": ("o", COLORS["candidate"]),
        "merge": ("h", "#7A5AA6"),
        "failed": ("X", COLORS["failed"]),
        "unknown": ("o", COLORS["unknown"]),
    }
    if success:
        groups = {
            "root": ("D", COLORS["root"]),
            "improvement": ("o", COLORS["improvement"]),
            "regression": ("s", COLORS["regression"]),
            "failed": ("X", COLORS["failed"]),
            "unknown": ("o", COLORS["unknown"]),
        }
        category = {node_id: node_class(node, by_id) for node_id, node in by_id.items()}
    else:
        category = {
            node_id: ("root" if node.get("mode") == "baseline" else
                      "failed" if node.get("status") == "failed" else
                      "merge" if len(node.get("parents", [])) == 2 else
                      "expand" if len(node.get("parents", [])) == 1 else "unknown")
            for node_id, node in by_id.items()
        }
    for kind, (shape, color) in groups.items():
        ids = [node_id for node_id, value in category.items() if value == kind]
        if not ids:
            continue
        nx.draw_networkx_nodes(
            graph, pos, nodelist=ids, node_shape=shape, node_color=color,
            edgecolors="#30343B", linewidths=1.25, node_size=650, ax=ax,
            label=kind.replace("_", " ").title(),
        )
    labels = {node_id: ("root" if node.get("mode") == "baseline" else node_id.replace("exp_", ""))
              for node_id, node in by_id.items()}
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=7.5,
                            font_color="white", font_weight="bold", ax=ax)
    ax.set_title(f"{title} — {'Local-Success Topology' if success else 'Search Topology'}")
    ax.set_xlabel("Search depth / generation  →")
    ax.set_yticks([])
    ax.legend(loc="upper right", frameon=True, ncol=3)
    max_depth = max(depth.values())
    y_values = [point[1] for point in pos.values()]
    ax.set_xlim(-0.85, max_depth + 0.85)
    ax.set_ylim(min(y_values) - 0.9, max(y_values) + 0.9)
    fig.tight_layout()
    filename = "search_topology_success.png" if success else "search_topology.png"
    fig.savefig(directory / "figures" / filename, bbox_inches="tight")
    plt.close(fig)
    return graph, nodes, depth


def plot_parent_delta(directory: Path, title: str, state: dict, valid: list[dict]):
    by_id = {node["id"]: node for node in state["nodes"]}
    deltas = []
    operations = []
    for node in valid:
        reference = reference_value(node, by_id)
        assert reference is not None
        deltas.append(val(node) - reference)
        operations.append("Merge" if len(node.get("parents", [])) == 2 else "Expand")
    x = np.arange(1, 31)
    colors = [COLORS["improvement"] if delta < 0 else COLORS["regression"] for delta in deltas]
    markers = ["D" if op == "Merge" else "o" for op in operations]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.vlines(x, 0, deltas, color=colors, linewidth=1.4, alpha=0.8)
    for xi, delta, color, marker in zip(x, deltas, colors, markers):
        ax.scatter(xi, delta, color=color, edgecolor="#30343B", marker=marker,
                   s=48, linewidth=0.7, zorder=3)
    ax.axhline(0, color=COLORS["baseline"], linewidth=1.4)
    ax.set_xlim(0.5, 30.5)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.set_xlabel("Candidate (1–30)")
    ax.set_ylabel("Child BPB − reference parent BPB")
    ax.set_title(f"{title} — Parent-Relative Delta")
    ax.grid(True, axis="y", alpha=0.25)
    ax.text(0.99, 0.02, "○ Expand   ◇ Merge\nnegative = local improvement",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
            bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.9})
    fig.tight_layout()
    fig.savefig(directory / "figures" / "parent_relative_delta.png", bbox_inches="tight")
    plt.close(fig)


def audit_row(directory: Path, title: str, state: dict, graph: nx.DiGraph, plotted_nodes: list[dict]):
    source_nodes = {node["id"]: node for node in state["nodes"]}
    valid = [node for node in source_nodes.values()
             if node.get("mode") != "baseline" and node.get("status") == "finished" and val(node) is not None]
    failed = [node for node in source_nodes.values() if node.get("status") == "failed"]
    expected_nodes = {node["id"] for node in plotted_nodes}
    source_edges = {
        (parent, node["id"])
        for node in plotted_nodes for parent in node.get("parents", []) if parent in expected_nodes
    }
    plotted_edges = set(graph.edges())
    expand_edges = sum(len(node.get("parents", [])) for node in valid if len(node.get("parents", [])) == 1)
    merge_edges = sum(len(node.get("parents", [])) for node in valid if len(node.get("parents", [])) == 2)
    checks = {
        "node set exact": set(graph.nodes()) == expected_nodes,
        "edge set exact": plotted_edges == source_edges,
        "all 30 valid nodes": len(valid) == 30 and {n["id"] for n in valid} <= set(graph.nodes()),
        "no invented edge": plotted_edges <= source_edges,
        "all source edges plotted": source_edges <= plotted_edges,
    }
    return {
        "directory": directory.name, "title": title, "nodes": len(graph.nodes()),
        "valid": len(valid), "expand_edges": expand_edges, "merge_edges": merge_edges,
        "failed": len(failed), "edges": len(graph.edges()), "checks": checks,
    }


def write_audit(rows):
    lines = [
        "# Figure topology audit", "",
        "Generated from the curated `results/dag.json` files. Both topology figures for each experiment use the same audited node and edge sets. Parent-relative success affects styling only; it never changes topology.", "",
        "| Experiment | Plotted nodes | Valid candidates | Expand edges | Merge edges | Failed nodes | Total edges | Result |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        passed = all(row["checks"].values())
        lines.append(
            f"| {row['title']} | {row['nodes']} | {row['valid']} | {row['expand_edges']} | "
            f"{row['merge_edges']} | {row['failed']} | {row['edges']} | {'PASS' if passed else 'FAIL'} |"
        )
    lines += ["", "## Checks applied", ""]
    for row in rows:
        lines.append(f"### {row['title']}")
        lines.append("")
        for name, passed in row["checks"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — {name}")
        lines.append("")
    lines += [
        "## Counting convention", "",
        "`Expand edges` and `Merge edges` count incoming edges of the 30 valid candidates only. Failed/interrupted nodes are plotted in gray and their source-recorded edges are included in `Total edges`. Root/baseline is included in plotted-node counts but not in valid-candidate counts.", "",
        "Success colors use only local comparisons: Expand child versus its parent; Merge child versus the lower-BPB parent. Lower BPB is better.",
    ]
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "figure_audit.md").write_text("\n".join(lines) + "\n")


def main():
    audit_rows = []
    for name, title in EXPERIMENTS:
        directory = ROOT / "experiments" / name
        figures = directory / "figures"
        figures.mkdir(exist_ok=True)
        if name == "00_linear_search":
            baseline, values = linear_series(directory / "results" / "full_40_iterations.tsv")
            plot_performance(directory, title, baseline, values)
            continue
        baseline, values, state, valid = dag_series(directory / "results" / "dag.json")
        plot_performance(directory, title, baseline, values)
        graph, nodes, _ = draw_topology(directory, title, state, success=False)
        success_graph, success_nodes, _ = draw_topology(directory, title, state, success=True)
        assert set(graph.nodes()) == set(success_graph.nodes())
        assert set(graph.edges()) == set(success_graph.edges())
        assert {n["id"] for n in nodes} == {n["id"] for n in success_nodes}
        plot_parent_delta(directory, title, state, valid)
        audit_rows.append(audit_row(directory, title, state, graph, nodes))
    write_audit(audit_rows)


if __name__ == "__main__":
    main()
