"""Export run metrics, class accuracy, and normalized confusion matrices."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from scipy.io import loadmat

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('log/checked'))
    args=p.parse_args()
    paths=sorted(args.root.rglob('metrics.json'))
    if not paths: p.error('No metrics.json found; check --root')
    fields=['dataset','seed','selected_epoch','selection','OA','AA','Kappa',
            'train_time_s','periodic_test_time_s','selected_single_test_time_s',
            'map_inference_time_s','wall_time_s','peak_torch_allocated_MiB',
            'train_gpu_util_mean_percent','train_gpu_util_peak_percent']
    with (args.root/'results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for path in paths:
            record=json.loads(path.read_text(encoding='utf-8'))
            row={key:record.get(key) for key in fields}
            row['dataset']=path.parent.parent.name
            stats=record['GPU']['phase_stats']['train']
            row['train_gpu_util_mean_percent']=stats['mean_util_percent']
            row['train_gpu_util_peak_percent']=stats['peak_util_percent']
            writer.writerow(row)
            with (path.parent/'class_accuracy.csv').open('w',newline='',encoding='utf-8-sig') as cf:
                cw=csv.writer(cf);cw.writerow(['class_id','accuracy_percent'])
                cw.writerows(enumerate(record['class_acc'],1))
            cm=loadmat(path.parent/'result.mat')['confusion_matrix'].astype(float)
            norm=np.divide(cm,cm.sum(1,keepdims=True),out=np.zeros_like(cm),where=cm.sum(1,keepdims=True)>0)
            np.savetxt(path.parent/'confusion_matrix_row_normalized.csv',norm,delimiter=',',fmt='%.8f')
    print('Saved',args.root/'results.csv')

if __name__=='__main__':main()
