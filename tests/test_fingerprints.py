import numpy as np
from f2fp.core.fingerprints import build_fingerprint

def test_fingerprint_metrics_present():
    X = np.random.randn(2000,3)
    fp = build_fingerprint(X)
    for k in ["linearity","planarity","scattering","circularity","nnz"]:
        assert k in fp.metrics
