from f2fp.viz.geovista_safe import screenshot_ring
from pathlib import Path

tmp_path="./"

def test_screenshot(tmp_path: Path):
    out = Path(tmp_path + "ring.png")
    screenshot_ring(-30,15,5,out)
    print(" saved to ", out)
    assert out.exists() and out.stat().st_size > 10_000

test_screenshot(tmp_path)
