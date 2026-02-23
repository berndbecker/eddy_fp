#/bin/bash
cd $HOME/prog/dop/eddy_fp/f2fp/tests
python test_synthetic.py
python test_synthetic_generators.py
python test_build_fingerprints.py
python test_fingerprints.py
python test_cluster.py
python conftest.py
python test_gyrescore.py
python test_canonical.py
python test_viz_verify.py
