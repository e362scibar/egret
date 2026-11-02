# Notebook-style parity scripts (Jupytext percent format) run their cases at import time.
# They are meant to be run as notebooks or scripts, so keep pytest from collecting them.
collect_ignore_glob = ["test_*_parity.py"]
