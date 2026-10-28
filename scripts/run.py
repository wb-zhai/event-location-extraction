#!/usr/bin/env python3
"""Run a script by its current path or its original location.

    python scripts/run.py --list
    python scripts/run.py global_v4_postprocess.py --help
"""
import argparse
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def resolve_script(name):
    paths = json.loads((ROOT / 'configs/script_paths.json').read_text())
    target = (ROOT / paths.get(name.replace('\\', '/'), name)).resolve()
    if not target.is_relative_to(ROOT) or not target.is_file() or target.suffix not in {'.py', '.sh'}:
        raise ValueError('Expected an existing Python/shell script inside this repository')
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('script', nargs='?')
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.list:
        paths = json.loads((ROOT / 'configs/script_paths.json').read_text())
        for old, new in sorted(paths.items()):
            print(f'{old} -> {new}')
        return 0
    if not args.script:
        parser.error('Specify a script or --list')
    try:
        target = resolve_script(args.script)
    except ValueError as exc:
        parser.error(str(exc))
    os.chdir(ROOT)
    if target.suffix == '.sh':
        return subprocess.call(['bash', str(target), *args.arguments], cwd=ROOT)
    sys.path[:0] = [str(target.parent), str(ROOT)]
    sys.argv = [str(target), *args.arguments]
    runpy.run_path(str(target), run_name='__main__')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
