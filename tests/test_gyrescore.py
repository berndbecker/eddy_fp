from f2fp.core.gyrescore import gyrescore, gyreclass

def test_gyrescore_range():
    s = gyrescore(dict(circularity=1, planarity=0.8, scattering=0.05))
    assert 0.0 <= s <= 1.0
    assert gyreclass(dict(circularity=1, planarity=0.8, scattering=0.05)) == "ring-like"
