#!/usr/bin/env python3
"""Regenerate the four curated supplementary figures from stored results only.

The assertions in this script are part of the figure audit: generation stops if
the retained node/edge set, taxonomy, or category counts differ from the
previously audited analysis.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DPI = 160
COLORS = {
    "candidate": "#356AA0",
    "root": "#B8BDC7",
    "improvement": "#2EAD64",
    "regression": "#D9534F",
    "edge": "#59636E",
    "aware_root": "#E69F00",
    "baseline_win": "#56B4E9",
    "baseline_miss": "#D9DEE5",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 15,
    "axes.labelsize": 11,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def node_value(node: dict) -> float | None:
    result = node.get("result") or {}
    value = result.get("val_bpb") if isinstance(result, dict) else None
    return float(value) if isinstance(value, (int, float)) else None


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def depth_layout(graph: nx.DiGraph, order: dict[str, int]) -> dict[str, tuple[float, float]]:
    depth: dict[str, int] = {}
    for node_id in nx.topological_sort(graph):
        parents = list(graph.predecessors(node_id))
        depth[node_id] = 0 if not parents else 1 + max(depth[parent] for parent in parents)
    layers: dict[int, list[str]] = defaultdict(list)
    for node_id, level in depth.items():
        layers[level].append(node_id)
    positions: dict[str, tuple[float, float]] = {}
    for level, node_ids in layers.items():
        node_ids.sort(key=lambda node_id: order[node_id])
        ys = np.linspace(1.0, -1.0, len(node_ids)) if len(node_ids) > 1 else np.array([0.0])
        for node_id, y in zip(node_ids, ys):
            positions[node_id] = (float(level), float(y))
    return positions


def generate_outcome_aware_lineage() -> None:
    exp = ROOT / "experiments/05_outcome_aware_proposal_memory_dag"
    state = json.loads((exp / "results/dag.json").read_text())
    all_nodes = {node["id"]: node for node in state["nodes"]}
    full = nx.DiGraph()
    full.add_nodes_from(all_nodes)
    for node in state["nodes"]:
        for parent in node.get("parents", []):
            full.add_edge(parent, node["id"])

    seed = "exp_000006"
    retained = {seed} | nx.descendants(full, seed)
    for node_id in list(retained):
        retained |= nx.ancestors(full, node_id)
    graph = full.subgraph(retained).copy()

    expected_nodes = {
        "exp_000000", "exp_000005", "exp_000006", "exp_000007", "exp_000008",
        "exp_000009", "exp_000010", "exp_000011", "exp_000013", "exp_000015",
        "exp_000016", "exp_000017", "exp_000019", "exp_000020", "exp_000021",
        "exp_000022", "exp_000023", "exp_000024", "exp_000025", "exp_000026",
        "exp_000027", "exp_000028", "exp_000029", "exp_000030",
    }
    assert set(graph.nodes) == expected_nodes
    assert graph.number_of_nodes() == 24
    assert graph.number_of_edges() == 32
    assert all(edge in full.edges for edge in graph.edges)

    audit_rows = read_tsv(exp / "figures/proposal_diversity_audit.tsv")
    mechanisms = {row["candidate_id"]: row["canonical_mechanism_family"] for row in audit_rows}
    assert all(node_id == "exp_000000" or node_id in mechanisms for node_id in graph.nodes)

    baseline = node_value(all_nodes["exp_000000"])
    assert baseline == 1.885642
    beat_baseline = {
        node_id: node_value(all_nodes[node_id]) < baseline
        for node_id in graph.nodes if node_id != "exp_000000"
    }
    beat_best_parent: dict[str, bool] = {}
    for node_id in graph.nodes:
        if node_id == "exp_000000":
            continue
        parents = all_nodes[node_id].get("parents", [])
        parent_values = [node_value(all_nodes[parent]) for parent in parents]
        assert all(value is not None for value in parent_values)
        beat_best_parent[node_id] = node_value(all_nodes[node_id]) < min(parent_values)

    order = {node["id"]: index for index, node in enumerate(state["nodes"])}
    pos = depth_layout(graph, order)
    fig, ax = plt.subplots(figsize=(18, 10.5))
    nx.draw_networkx_edges(
        graph, pos, ax=ax, edge_color=COLORS["edge"], width=1.25, alpha=0.8,
        arrows=True, arrowstyle="-|>", arrowsize=15, node_size=2100,
        connectionstyle="arc3,rad=0.025", min_source_margin=10, min_target_margin=12,
    )

    for node_id in graph.nodes:
        node = all_nodes[node_id]
        if node_id == "exp_000000":
            shape, face, edge, width = "s", COLORS["root"], "#30343B", 1.5
        else:
            shape = "D" if len(node.get("parents", [])) == 2 else "o"
            face = COLORS["baseline_win"] if beat_baseline[node_id] else COLORS["baseline_miss"]
            if node_id in {"exp_000005", "exp_000006"}:
                face = COLORS["aware_root"]
            edge = COLORS["improvement"] if beat_best_parent[node_id] else COLORS["regression"]
            width = 3.0 if beat_best_parent[node_id] else 2.0
        nx.draw_networkx_nodes(
            graph, pos, nodelist=[node_id], node_shape=shape, node_color=face,
            edgecolors=edge, linewidths=width, node_size=2100, ax=ax,
        )

    labels = {}
    for node_id in graph.nodes:
        if node_id == "exp_000000":
            labels[node_id] = f"root\n{baseline:.6f}\nbaseline"
        else:
            labels[node_id] = (
                f"{node_id.replace('exp_', '')}\n{node_value(all_nodes[node_id]):.6f}\n"
                f"{mechanisms[node_id]}"
            )
    nx.draw_networkx_labels(graph, pos, labels=labels, font_size=6.4, ax=ax)

    legend = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS["root"],
               markeredgecolor="#30343B", markersize=10, label="Recorded root baseline"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["aware_root"],
               markeredgecolor="#30343B", markersize=10, label="Sibling lineage roots (000005 / 000006)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["baseline_win"],
               markeredgecolor="#30343B", markersize=10, label="Below recorded root baseline"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS["baseline_miss"],
               markeredgecolor="#30343B", markersize=10, label="Not below recorded root baseline"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=COLORS["improvement"], markeredgewidth=2.5, markersize=10,
               label="Improved over best parent"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor=COLORS["regression"], markeredgewidth=2.0, markersize=10,
               label="No improvement over best parent"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor="white",
               markeredgecolor="#30343B", markersize=9, label="Merge (two parents)"),
    ]
    ax.legend(handles=legend, loc="upper left", frameon=True, ncol=2)
    ax.set_title("Outcome-Aware Proposal-Memory DAG — Productive Success Lineage")
    ax.set_xlabel("Search depth / generation  →")
    ax.set_yticks([])
    xs = [point[0] for point in pos.values()]
    ax.set_xlim(min(xs) - 0.7, max(xs) + 0.7)
    ax.set_ylim(-1.35, 1.35)
    fig.tight_layout()
    fig.savefig(exp / "figures/outcome_aware_success_lineage.png", bbox_inches="tight")
    plt.close(fig)


def horizontal_distribution(path: Path, title: str, rows: list[tuple[str, int]], color: str) -> None:
    labels = [label for label, _ in rows]
    counts = [count for _, count in rows]
    height = max(6.75, 0.42 * len(rows) + 1.7)
    fig, ax = plt.subplots(figsize=(12, height))
    bars = ax.barh(labels[::-1], counts[::-1], color=color, edgecolor="#30343B", linewidth=0.6)
    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=9)
    ax.set_xlabel("Candidate count")
    ax.set_ylabel("Mechanism family")
    ax.set_title(title)
    ax.set_xlim(0, max(counts) + 0.8)
    ax.set_xticks(range(0, max(counts) + 1))
    ax.grid(True, axis="x", alpha=0.22)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def generate_outcome_aware_distribution() -> None:
    exp = ROOT / "experiments/05_outcome_aware_proposal_memory_dag"
    rows = read_tsv(exp / "figures/mechanism_families.tsv")
    values = [(row["family"], int(row["count"])) for row in rows]
    assert len(values) == 18
    assert sum(count for _, count in values) == 30
    assert values[:6] == [
        ("cross-component interaction", 6), ("optimizer", 3), ("optimization", 3),
        ("optimization schedule", 2), ("attention normalization", 2),
        ("output calibration", 2),
    ]
    assert Counter(count for _, count in values) == Counter({1: 12, 2: 3, 3: 2, 6: 1})
    horizontal_distribution(
        exp / "figures/outcome_aware_mechanism_family_distribution.png",
        "Outcome-Aware Proposal-Memory DAG — Mechanism-Family Distribution",
        values, COLORS["candidate"],
    )


def generate_outcome_blind_discovery() -> None:
    exp = ROOT / "experiments/06_outcome_blind_proposal_memory_dag"
    summary = json.loads((exp / "results/summary.json").read_text())
    counts = summary["success_origin_counts"]
    expected = {
        "inherited_success": 19,
        "new_mechanism_local_improvement": 6,
        "independent_discovery": 2,
        "local_improvement_on_successful_lineage": 1,
    }
    assert counts == expected
    rows = read_tsv(exp / "results/candidates.tsv")
    observed = Counter(row["success_origin"] for row in rows)
    assert {key: observed[key] for key in expected} == expected
    assert observed["not_beat_recorded_baseline"] == 2

    labels = [
        "Independent\ndiscovery",
        "Inherited\nsuccess",
        "Local improvement on\nsuccessful lineage",
        "New-mechanism\nlocal improvement",
    ]
    values = [2, 19, 1, 6]
    colors = ["#E69F00", "#8C8C8C", COLORS["candidate"], "#009E73"]
    fig, ax = plt.subplots(figsize=(12, 6.75))
    bars = ax.bar(labels, values, color=colors, edgecolor="#30343B", linewidth=0.8)
    ax.bar_label(bars, labels=[str(value) for value in values], padding=4, fontsize=11)
    ax.set_ylabel("Candidates below the recorded root baseline")
    ax.set_title("Outcome-Blind Proposal-Memory DAG — Discovery vs Inheritance")
    ax.set_ylim(0, 20.5)
    ax.grid(True, axis="y", alpha=0.22)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(exp / "figures/outcome_blind_discovery_vs_inheritance.png", bbox_inches="tight")
    plt.close(fig)


def generate_outcome_blind_distribution() -> None:
    exp = ROOT / "experiments/06_outcome_blind_proposal_memory_dag"
    rows = read_tsv(exp / "results/candidates.tsv")
    counts = Counter(row["mechanism_family"] for row in rows)
    order = list(dict.fromkeys(row["mechanism_family"] for row in rows))
    assert len(counts) == 21
    assert sum(counts.values()) == 30
    # Match the retained analysis ordering for tied families without changing taxonomy.
    retained_order = [
        "cosine LR warmdown", "gradient clipping", "RoPE frequency base",
        "weight-decay endpoint retention", "MLP activation", "output-logit softcap",
        "value-embedding gate scaling", "LR endpoint retention",
        "Muon orthogonalization depth", "value-embedding normalization",
        "residual branch initialization", "embedding bypass initialization",
        "embedding bypass bounding", "optimizer second-moment timescale",
        "positional-activation interaction", "residual normalization", "momentum ramp",
        "LR warmup", "Muon gradient centralization", "attention-logit temperature",
        "weight-decay schedule shape",
    ]
    assert set(retained_order) == set(counts)
    values = [(family, counts[family]) for family in retained_order]
    assert [count for _, count in values] == [3] + [2] * 7 + [1] * 13
    horizontal_distribution(
        exp / "figures/outcome_blind_mechanism_family_distribution.png",
        "Outcome-Blind Proposal-Memory DAG — Mechanism-Family Distribution",
        values, "#7A5AA6",
    )


def main() -> None:
    generate_outcome_aware_lineage()
    generate_outcome_aware_distribution()
    generate_outcome_blind_discovery()
    generate_outcome_blind_distribution()
    print("Supplementary figure audit passed: 4 figures generated from retained source data.")


if __name__ == "__main__":
    main()
