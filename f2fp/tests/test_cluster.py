import numpy as np
from f2fp.core.fingerprints import build_fingerprint, to_dict, load_metrics_matrix, _stack_metrics, plot_features_from_json
from f2fp.core.cluster import embed_features, cluster_labels

def test_cluster_pipeline():
    fps = [to_dict(build_fingerprint(np.random.randn(500,3))) for _ in range(20)]
    M = _stack_metrics(fps, ["linearity","planarity","scattering","circularity","nnz"])
    E = embed_features(M, 2, 42)
    labels = cluster_labels(M, 5, 42)
    assert E.shape == (20,2) and labels.shape==(20,)
    return E, labels

def main():
    keys = ["linearity", "planarity", "scattering", "circularity", "nnz"]
    M = load_metrics_matrix("run_fp.jsonl", keys)
    
    emb = embed_features(M)
    labels = cluster_labels(emb)
    print(E.shape, labels)
    plot_features_from_json("run_fp.jsonl", keys)

E, labels = test_cluster_pipeline()
main()
