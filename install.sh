rm -rf build
rm -rf dist
rm -rf *.egg-info
pip uninstall -y asym_gemm
python3 setup.py install