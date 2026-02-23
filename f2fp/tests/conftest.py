import os, pytest, numpy as np
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

@pytest.fixture(autouse=True)
def _seed(): np.random.seed(42); yield
