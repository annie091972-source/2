"""Generate exactly the same five 5% GT splits as run_checked.py."""
import argparse
import numpy as np
from pathlib import Path
from scipy.io import loadmat, savemat
from run_checked import split

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('dataset/Trento'))
    p.add_argument('--out', type=Path, default=Path('log/Trento_splits_5percent'))
    p.add_argument('--seeds', type=int, nargs='+', default=[21,22,80,443,445])
    args = p.parse_args()
    if args.out.exists():
        p.error('Output exists: choose a new --out to avoid overwriting')
    gt = loadmat(args.root/'GT_Trento.mat')['GT_Trento']
    if not np.array_equal(np.unique(gt[gt>0]), np.arange(1,7)):
        p.error('Trento labels must be 1..6')
    args.out.mkdir(parents=True)
    for seed in args.seeds:
        tr,te = split(gt,6,seed)
        tr_mask,te_mask = np.zeros_like(gt),np.zeros_like(gt)
        tr_mask[tr[:,0],tr[:,1]] = gt[tr[:,0],tr[:,1]]
        te_mask[te[:,0],te[:,1]] = gt[te[:,0],te[:,1]]
        savemat(args.out/f'Trento_seed_{seed}_5percent.mat', {
            'TRLabel':tr_mask,'TSLabel':te_mask,'train_pos':tr,
            'test_pos':te,'coordinate_base':0,'seed':seed,'train_ratio':.05
        })
        print(seed, 'train:',len(tr),'test:',len(te))
