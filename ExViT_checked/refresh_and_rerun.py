"""Update quota fix, remove only the specified experiment logs, rerun H then T."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

REVISION='395d38ee4b07711e14c12e0837faff1d0d7786e3'
SOURCE='https://raw.githubusercontent.com/annie091972-source/2/'+REVISION+'/ExViT_checked/'
EXPECTED_HASHES={'run_checked.py': 'deef0f94776f068d7ce2b64b16225a14afe6cb18da4c3645d144b3fd65a9d02b', 'audit_results.py': 'e03bb8aeae3c8e0f6c64224dd03dbb358a02e7f0be16d574f7fc769bffd9df90', 'gen_trento_split.py': '89753f2a262a286edb693defc954575badb7f3c1b46e1bb54537da698451c154', 'README.md': '37b0f91f98016cb4debbdfd4b33bab525c0e9f969171799dedf985a36dc4277f'}
TRAINERS={'run_houston_5percent.py','run_trento_5percent.py','run_checked.py','MViT_demo_h2013.py'}

def running_processes(root,names):
    found=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:
            if (entry/'cwd').resolve()!=root:continue
            parts=(entry/'cmdline').read_bytes().split(b'\0')
            if any(Path(p.decode(errors='replace')).name in names for p in parts[:4]):
                found.append(int(entry.name))
        except (OSError,RuntimeError):continue
    return found

def alive(pid):
    try:
        status=Path(f'/proc/{pid}/status').read_text()
        return not any(line.startswith('State:') and '\tZ' in line for line in status.splitlines())
    except OSError:return False

def stop_training(root):
    if running_processes(root,{'refresh_and_rerun.py'}):
        raise RuntimeError('Another rerun launcher is already running. Do not start two launchers.')
    pids=running_processes(root,TRAINERS)
    for pid in pids:
        print('Stopping old project training PID',pid,flush=True)
        try:os.kill(pid,signal.SIGINT)
        except ProcessLookupError:pass
    deadline=time.monotonic()+20
    while any(alive(p) for p in pids) and time.monotonic()<deadline:time.sleep(.2)
    for pid in pids:
        if alive(pid):
            try:os.kill(pid,signal.SIGTERM)
            except ProcessLookupError:pass
    deadline=time.monotonic()+5
    while any(alive(p) for p in pids) and time.monotonic()<deadline:time.sleep(.2)
    if any(alive(p) for p in pids):raise RuntimeError('Old training did not stop; logs not deleted')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path('/workspace/ExViT-1-rerun'))
    args=parser.parse_args()
    root=args.root.resolve()
    for name in ['MViT_pytorch_upload.py','run_checked.py','run_houston_5percent.py','run_trento_5percent.py']:
        if not (root/name).is_file():raise FileNotFoundError(root/name)
    for name in ['dataset/houston2013','dataset/Trento']:
        if not (root/name).is_dir():raise FileNotFoundError(root/name)
    log=root/'log'
    if log.is_symlink():raise RuntimeError('Refusing cleanup through symlinked log directory')
    for name in ['allGT_5percent','preflight']:
        target=log/name
        if target.is_symlink() or (target.exists() and not target.is_dir()):
            raise RuntimeError('Unexpected log path: '+str(target))
    downloads={}
    for name,expected in EXPECTED_HASHES.items():
        with urllib.request.urlopen(SOURCE+name,timeout=60) as response:body=response.read()
        if hashlib.sha256(body).hexdigest()!=expected:raise RuntimeError('Download checksum mismatch: '+name)
        if name.endswith('.py'):compile(body.decode('utf-8'),name,'exec')
        downloads[name]=body
        print('Verified update:',name,flush=True)
    stop_training(root)
    for name,body in downloads.items():
        temporary=root/(name+'.quota_update')
        temporary.write_bytes(body);os.replace(temporary,root/name)
    os.chdir(root);sys.path.insert(0,str(root))
    # Verify actual data, quotas and NVML before removing existing results.
    from run_checked import load_data,stratified_train_counts,GPU
    import torch
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; old logs not deleted')
    for dataset,relative in [('Houston','dataset/houston2013'),('Trento','dataset/Trento')]:
        x,gt,band,k=load_data(root/relative,dataset)
        quota=stratified_train_counts(gt,k)
        print(dataset,'new train:',int(quota.sum()),'per-class:',quota.tolist(),flush=True)
        del x,gt
    probe=GPU(torch.device('cuda:0'),root/'.rerun_gpu_probe.csv')
    if probe.handle is not None:probe.nv.nvmlShutdown()
    for name in ['allGT_5percent','preflight']:
        target=log/name
        if target.exists():shutil.rmtree(target);print('Deleted old logs:',target,flush=True)
    log.mkdir(exist_ok=True)
    for dataset,script,relative in [('Houston','run_houston_5percent.py','dataset/houston2013'),('Trento','run_trento_5percent.py','dataset/Trento')]:
        console=log/('console_'+dataset+'.txt')
        command=[sys.executable,'-u',script,'--root',relative,'--out','log/allGT_5percent']
        print('Starting',dataset,'seeds 21/22/80/443/445, 500 epochs each; console:',console,flush=True)
        with console.open('w',encoding='utf-8') as stream:
            result=subprocess.run(command,cwd=root,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:raise RuntimeError(dataset+' failed; inspect '+str(console))
        print(dataset,'finished',flush=True)
    print('Both datasets finished. Results: log/allGT_5percent',flush=True)

if __name__=='__main__':main()
