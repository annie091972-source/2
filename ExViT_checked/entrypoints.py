"""Compatibility entrypoints with the new unified command-line options."""
import sys
from pathlib import Path

def train(dataset):
    from run_checked import main
    if '--dataset' not in sys.argv:
        sys.argv.extend(['--dataset', dataset])
    if '--root' not in sys.argv:
        root = './dataset/houston2013' if dataset == 'Houston' else './dataset/Trento'
        sys.argv.extend(['--root', root])
    main()

def plot(dataset):
    import argparse
    from run_checked import draw
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('log/checked') / dataset)
    parser.add_argument('--mat', type=Path)
    args = parser.parse_args()
    paths = [args.mat] if args.mat else sorted(args.root.rglob('result.mat'))
    if not paths:
        parser.error('No result.mat found; check --root or supply --mat')
    for path in paths:
        draw(path)
        print('PNG saved:', path.parent)
