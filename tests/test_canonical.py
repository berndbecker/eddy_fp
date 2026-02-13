import numpy as np
from f2fp.core.canonical import pca_align, sparse_grid_12

def test_canonical_grid_shape():
    X = np.random.randn(500,3)
    Xc, pose = pca_align(X)
    grid = sparse_grid_12(Xc)
    assert grid['nx']==12 and grid['nnz']>0
