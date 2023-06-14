# reinstall local packages with no deps & edition mode
pip install --no-deps -e /src/ia2

jupyter-lab --ip=0.0.0.0 --allow-root --no-browser