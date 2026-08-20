"""Construct a simple topology graph from a signature skeleton."""

from __future__ import annotations

import cv2
import numpy as np

from .skeleton import count_skeleton_neighbours


def build_skeleton_graph(skeleton: np.ndarray) -> dict[str, object]:
    """Convert endpoint/junction regions and connecting strokes into a graph."""

    binary = np.asarray(skeleton, dtype=bool)
    neighbours = count_skeleton_neighbours(binary)
    node_candidate = binary & ((neighbours == 1) | (neighbours >= 3))
    node_count, node_labels, _, centroids = cv2.connectedComponentsWithStats(
        node_candidate.astype(np.uint8),
        connectivity=8,
    )
    nodes = []
    for label in range(1, node_count):
        region = node_labels == label
        node_type = "junction" if np.any(neighbours[region] >= 3) else "endpoint"
        x, y = centroids[label]
        nodes.append(
            {
                "id": label - 1,
                "type": node_type,
                "xy": [round(float(x), 3), round(float(y), 3)],
                "pixel_count": int(region.sum()),
            }
        )

    stroke_mask = binary & ~node_candidate
    segment_count, segment_labels = cv2.connectedComponents(
        stroke_mask.astype(np.uint8),
        connectivity=8,
    )
    edges = []
    kernel = np.ones((3, 3), dtype=np.uint8)
    for segment_label in range(1, segment_count):
        segment = segment_labels == segment_label
        dilated = cv2.dilate(
            segment.astype(np.uint8),
            kernel,
            iterations=1,
        ).astype(bool)
        touching_labels = sorted(
            int(label)
            for label in np.unique(node_labels[dilated])
            if label > 0
        )
        if len(touching_labels) < 2:
            continue
        for first_index in range(len(touching_labels) - 1):
            for second_index in range(first_index + 1, len(touching_labels)):
                edges.append(
                    {
                        "source": touching_labels[first_index] - 1,
                        "target": touching_labels[second_index] - 1,
                        "segment_pixel_count": int(segment.sum()),
                    }
                )
    unique_edges = {
        (
            min(edge["source"], edge["target"]),
            max(edge["source"], edge["target"]),
        ): edge
        for edge in edges
    }
    edges = list(unique_edges.values())
    degrees = {node["id"]: 0 for node in nodes}
    for edge in edges:
        degrees[edge["source"]] += 1
        degrees[edge["target"]] += 1
    return {
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
        "endpoint_node_count": sum(
            node["type"] == "endpoint" for node in nodes
        ),
        "junction_node_count": sum(
            node["type"] == "junction" for node in nodes
        ),
        "degree_sequence": sorted(degrees.values(), reverse=True),
    }


def graph_distance(
    first_graph: dict[str, object],
    second_graph: dict[str, object],
) -> float:
    """Compare compact graph statistics on a zero-to-one scale."""

    fields = (
        "node_count",
        "edge_count",
        "endpoint_node_count",
        "junction_node_count",
    )
    differences = []
    for field in fields:
        first = float(first_graph[field])
        second = float(second_graph[field])
        differences.append(
            abs(first - second) / max(first, second, 1.0)
        )
    first_degrees = np.asarray(
        first_graph["degree_sequence"],
        dtype=float,
    )
    second_degrees = np.asarray(
        second_graph["degree_sequence"],
        dtype=float,
    )
    length = max(len(first_degrees), len(second_degrees), 1)
    first_degrees = np.pad(first_degrees, (0, length - len(first_degrees)))
    second_degrees = np.pad(second_degrees, (0, length - len(second_degrees)))
    degree_difference = float(
        np.mean(
            np.abs(first_degrees - second_degrees)
            / np.maximum(np.maximum(first_degrees, second_degrees), 1.0)
        )
    )
    differences.append(degree_difference)
    return round(float(np.mean(differences)), 8)
