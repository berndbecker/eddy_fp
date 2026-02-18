import json
import numpy as np
import matplotlib.pyplot as plt

from core.fingerprints import load_fingerprints_json, _stack_metrics
from core.cluster import embed_features

def main():
    # Load fingerprints from JSON/JSONL
    fps = load_fingerprints_json("run_fp.jsonl")

    # Extract labels for coloring
    labels = [fp.get("label", "unknown") for fp in fps]
    label_set = sorted(set(labels))
    label_to_int = {l: i for i, l in enumerate(label_set)}
    color_vals = [label_to_int[l] for l in labels]

    # Stack metrics and embed
    keys = ["linearity", "planarity", "scattering", "circularity", "nnz"]
    M = _stack_metrics(fps, keys)
    emb = embed_features(M)

    # Plot embedded features colored by label
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(emb[:, 0], emb[:, 1], c=color_vals, cmap="tab10", s=40)
    handles = [plt.Line2D([], [], marker="o", color="w", markerfacecolor=scatter.cmap(i), label=label, markersize=8)
               for i, label in enumerate(label_set)]
    plt.legend(handles=handles, title="Label")
    plt.xlabel("Embedding 1")
    plt.ylabel("Embedding 2")
    plt.title("Embedded Features by Eddy Label")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
