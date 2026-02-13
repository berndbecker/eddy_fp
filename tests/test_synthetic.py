from f2fp.core.synthetic import SynthConfig, ring_points, curtain_points

def test_ring_and_curtain_nonempty():
    cfg = SynthConfig()
    lr, ar = ring_points(cfg); lc, ac = curtain_points(cfg)
    assert lr.size > 0 and ar.size > 0 and lc.size > 0 and ac.size > 0
