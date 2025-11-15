#!/usr/bin/env python3
"""Sync the `project.version` in pyproject.toml with the package __version__.

Usage:
  python scripts/sync_version.py --version-file egret/version.py --pyproject pyproject.toml

This script requires the `toml` package to be installed in the environment.
"""
import argparse
import re
import sys
from pathlib import Path


def get_version_from_file(path: Path) -> str:
    text = path.read_text(encoding='utf-8')
    m = re.search(r"__version__\s*=\s*['\"]([^'\"]+)['\"]", text)
    if not m:
        raise SystemExit(f"Could not find __version__ in {path}")
    return m.group(1)


def load_pyproject(path: Path):
    import toml

    return toml.load(path)


def write_pyproject(path: Path, data):
    import toml

    path.write_text(toml.dumps(data), encoding='utf-8')


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument('--version-file', required=True)
    p.add_argument('--pyproject', required=True)
    args = p.parse_args(argv)

    vfile = Path(args.version_file)
    pyproject = Path(args.pyproject)

    if not vfile.exists():
        raise SystemExit(f"version file not found: {vfile}")
    if not pyproject.exists():
        raise SystemExit(f"pyproject.toml not found: {pyproject}")

    version = get_version_from_file(vfile)
    print(f"Package version: {version}")

    data = load_pyproject(pyproject)
    if 'project' not in data or not isinstance(data['project'], dict):
        data['project'] = {}

    old = data['project'].get('version')
    if old == version:
        print(f"pyproject.toml already has version={version}")
        return 0

    data['project']['version'] = version
    write_pyproject(pyproject, data)
    print(f"Updated {pyproject} project.version: {old} -> {version}")

    # verify
    data2 = load_pyproject(pyproject)
    if data2.get('project', {}).get('version') != version:
        raise SystemExit("Failed to verify pyproject.toml version update")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
