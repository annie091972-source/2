import argparse, csv, hashlib, json, platform, random, threading, time
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat, savemat
from PIL import Image
from MViT_pytorch_upload import MViT

COLORS = np.array([[0,0,0],[0,160,0],[120,210,70],[0,210,180],[0,100,0],
 [180,110,50],[0,90,255],[230,50,50],[180,0,0],[160,160,160],[255,190,0],
 [130,70,180],[240,120,180],[100,180,240],[255,100,0],[240,240,0]], np.uint8)

def draw(path):
    d = loadmat(path)
    for name in ('gt_map','test_pred_map','labeled_pred_map','full_pred_map'):
        if name in d:
            Image.fromarray(COLORS[d[name].astype(int)]).save(Path(path).with_name(name+'.png'))
    error = (d['test_pred_map'] != d['gt_map']) & (d['test_mask'] > 0)
    Image.fromarray(error.astype(np.uint8)*255).save(Path(path).with_name('test_error_map.png'))

def read(root, name, key):
    p = root/name
    d = loadmat(p)
    if key not in d:
        raise ValueError(f'{p}: missing {key}; keys={list(d)}')
    return d[key]

def load_data(root, dataset):
    if dataset == 'Houston':
        hs = read(root,'HSI.mat','HSI')
        lidar = read(root,'LiDAR.mat','LiDAR')
        tr = read(root,'TRLabel.mat','TRLabel')
        te = read(root,'TSLabel.mat','TSLabel')
        if tr.shape != te.shape or np.any((tr > 0) & (te > 0)):
            raise ValueError('Houston TRLabel/TSLabel overlap or shape mismatch')
        gt = np.maximum(tr,te)
    else:
        hs = read(root,'HSI_Trento.mat','HSI_Trento')
        lidar = read(root,'Lidar_Trento.mat','Lidar_Trento')
        gt = read(root,'GT_Trento.mat','GT_Trento')
    if lidar.ndim == 2:
        lidar = lidar[...,None]
    if hs.ndim != 3 or lidar.ndim != 3 or gt.ndim != 2:
        raise ValueError('Expected HSI/LiDAR HWC and GT HW')
    if hs.shape[:2] != gt.shape or lidar.shape[:2] != gt.shape:
        raise ValueError('HSI/LiDAR/GT spatial shapes differ')
    if not all(np.isfinite(x).all() for x in (hs,lidar,gt)):
        raise ValueError('NaN/Inf in data')
    k = 15 if dataset == 'Houston' else 6
    if not np.array_equal(np.unique(gt[gt > 0]),np.arange(1,k+1)) or np.any(gt < 0):
        raise ValueError('GT must contain contiguous labels 1..K and background 0')
    hs = hs.astype(np.float32)
    lo,hi = hs.min((0,1)),hs.max((0,1))
    hs = (hs-lo)/np.where(hi>lo,hi-lo,1)
    # Preserve uploaded implementation: global bandwise HSI normalization; raw LiDAR.
    x = np.concatenate((hs,lidar.astype(np.float32)),axis=2)
    return x,gt.astype(np.int16),hs.shape[2],k

def split(gt,k,seed):
    rng = np.random.default_rng(seed)
    tr,te = [],[]
    for c in range(1,k+1):
        pos = np.argwhere(gt==c)
        pos = pos[rng.permutation(len(pos))]
        n = int(len(pos)*0.05)
        if n < 1 or n >= len(pos):
            raise ValueError(f'class {c}: cannot form nonempty 5%/95% split')
        tr.append(pos[:n]); te.append(pos[n:])
    return np.concatenate(tr),np.concatenate(te)

class Patches(torch.utils.data.Dataset):
    def __init__(self,padded,gt,pos,p,augment=False):
        self.x,self.gt,self.pos,self.p,self.augment = padded,gt,pos,p,augment
    def __len__(self): return len(self.pos)
    def __getitem__(self,i):
        r,c = self.pos[i]
        x = self.x[r:r+self.p,c:c+self.p].transpose(2,0,1).copy()
        if self.augment:
            if random.random() > .5: x = x[:,:,::-1]
            if random.random() > .5: x = x[:,::-1,:]
        return torch.from_numpy(x.copy()),int(self.gt[r,c])-1

def metrics(y,p,k):
    cm = np.bincount(y*k+p,minlength=k*k).reshape(k,k)
    ca = np.diag(cm)/cm.sum(1)
    oa = np.trace(cm)/cm.sum()
    pe = np.dot(cm.sum(0).astype(float),cm.sum(1))/float(cm.sum())**2
    return {'OA':float(oa*100),'AA':float(ca.mean()*100),
            'Kappa':float((oa-pe)/(1-pe))},ca*100,cm

class GPU:
    def __init__(self,device,path):
        self.device,self.path,self.rows,self.phase,self.error = device,path,[],'train',None
        self.stop_event = threading.Event()
        self.handle = None
        if device.type == 'cuda':
            import pynvml
            self.nv = pynvml
            pynvml.nvmlInit()
            # UUID matches the actual torch device, including CUDA_VISIBLE_DEVICES remapping.
            uuid = str(torch.cuda.get_device_properties(device).uuid)
            self.handle = pynvml.nvmlDeviceGetHandleByUUID(uuid)
            pynvml.nvmlDeviceGetUtilizationRates(self.handle)
        self.thread = threading.Thread(target=self.collect,daemon=True)
    def collect(self):
        with self.path.open('w',newline='',encoding='utf-8') as f:
            writer = csv.writer(f); writer.writerow(['unix_time','phase','gpu_util_percent','device_memory_MiB'])
            while not self.stop_event.is_set():
                if self.handle is not None:
                    try:
                        u = self.nv.nvmlDeviceGetUtilizationRates(self.handle).gpu
                        m = self.nv.nvmlDeviceGetMemoryInfo(self.handle).used/1024**2
                        row = [time.time(),self.phase,u,m]
                        self.rows.append(row); writer.writerow(row); f.flush()
                    except Exception as e: self.error = str(e)
                self.stop_event.wait(1)
    def start(self): self.thread.start()
    def close(self):
        self.stop_event.set(); self.thread.join()
        if self.handle is not None: self.nv.nvmlShutdown()
        stats = {}
        for phase in ('train','test','map'):
            rows = [r for r in self.rows if r[1]==phase]
            stats[phase] = {'samples':len(rows),'mean_util_percent':float(np.mean([r[2] for r in rows])) if rows else None,
                            'peak_util_percent':max([r[2] for r in rows],default=None)}
        return {'phase_stats':stats,'error':self.error,'scope':'whole device, not process',
                'peak_device_memory_MiB':max([r[3] for r in self.rows],default=None)}

def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize(device)

def epoch(model,loader,band,device,k,optimizer=None):
    model.train(optimizer is not None)
    sync(device); start = time.perf_counter()
    ys,ps,total = [],[],0.
    with torch.set_grad_enabled(optimizer is not None):
        for x,y in loader:
            x,y = x.to(device),y.to(device)
            if optimizer is not None: optimizer.zero_grad(set_to_none=True)
            z = model(x[:,:band],x[:,band:])
            loss = torch.nn.functional.cross_entropy(z,y)
            if optimizer is not None:
                loss.backward(); optimizer.step()
            total += loss.item()*len(y)
            ys.append(y.cpu().numpy()); ps.append(z.argmax(1).detach().cpu().numpy())
    sync(device)
    elapsed = time.perf_counter()-start
    y,p = np.concatenate(ys),np.concatenate(ps)
    m,ca,cm = metrics(y,p,k)
    return m,ca,cm,y,p,total/len(y),elapsed

def infer_map(model,padded,gt,band,k,device,args):
    pos = np.argwhere(np.ones(gt.shape,bool) if args.full_map else gt>0)
    loader = torch.utils.data.DataLoader(Patches(padded,gt,pos,args.patch),batch_size=args.batch)
    model.eval(); out = np.zeros_like(gt)
    offset = 0
    with torch.no_grad():
        for x,_ in loader:
            x=x.to(device)
            p=model(x[:,:band],x[:,band:]).argmax(1).cpu().numpy()+1
            rc=pos[offset:offset+len(p)]
            out[rc[:,0],rc[:,1]]=p
            offset+=len(p)
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['Houston','Trento'])
    parser.add_argument('--root',type=Path)
    parser.add_argument('--out',type=Path,default=Path('log/checked'))
    parser.add_argument('--gpu',type=int,default=0,help='logical torch device index; -1 for CPU')
    parser.add_argument('--seeds',type=int,nargs='+',default=[21,22,80,443,445])
    parser.add_argument('--epochs',type=int,default=500)
    parser.add_argument('--batch',type=int,default=64)
    parser.add_argument('--patch',type=int,default=13)
    parser.add_argument('--test-freq',type=int,default=10)
    parser.add_argument('--lr',type=float,default=3e-4)
    parser.add_argument('--weight-decay',type=float)
    parser.add_argument('--dropout',type=float)
    parser.add_argument('--augment',action=argparse.BooleanOptionalAction,default=None)
    parser.add_argument('--select',choices=['last','test_peak'],default='last')
    parser.add_argument('--full-map',action='store_true')
    parser.add_argument('--preflight',action='store_true')
    parser.add_argument('--draw',type=Path,help='render result.mat; no training')
    args=parser.parse_args()
    if args.draw:
        draw(args.draw); return
    if args.dataset is None or args.root is None: parser.error('--dataset and --root required')
    if args.patch<3 or args.patch%2!=1 or min(args.epochs,args.batch,args.test_freq)<1:
        parser.error('patch must be odd >=3; epochs/batch/test-freq positive')
    h=args.dataset=='Houston'
    if args.weight_decay is None: args.weight_decay=7.5e-4 if h else 1e-4
    if args.dropout is None: args.dropout=.15 if h else .2
    if args.augment is None: args.augment=h
    device=torch.device('cpu' if args.gpu<0 else f'cuda:{args.gpu}')
    if device.type=='cuda' and (not torch.cuda.is_available() or args.gpu>=torch.cuda.device_count()):
        raise RuntimeError('Requested CUDA device unavailable; use --gpu -1 for CPU')
    x,gt,band,k=load_data(args.root,args.dataset)
    padded=np.pad(x,((args.patch//2,)*2,(args.patch//2,)*2,(0,0)),mode='symmetric')
    root=args.out/args.dataset
    if root.exists(): raise FileExistsError(f'{root} exists; choose a new --out to avoid overwriting')
    root.mkdir(parents=True)
    config={a:str(v) if isinstance(v,Path) else v for a,v in vars(args).items()}
    config.update(protocol='all-GT per-class floor(5%); remainder test',python=platform.python_version(),
                  torch=str(torch.__version__),cuda=torch.version.cuda,shape=list(x.shape),classes=k,
                  coordinate_base=0,label_map_base=1,metrics_OA_AA_unit='percent',
                  model_sha256=hashlib.sha256(Path(__file__).with_name('MViT_pytorch_upload.py').read_bytes()).hexdigest(),
                  gpu_name=torch.cuda.get_device_name(device) if device.type=='cuda' else None)
    (root/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    splits={s:split(gt,k,s) for s in args.seeds}
    for s,(tr,te) in splits.items():
        print(args.dataset,s,'train',len(tr),'test',len(te))
        print('per-class train',np.bincount(gt[tr[:,0],tr[:,1]],minlength=k+1)[1:].tolist())
        savemat(root/f'split_{s}.mat',{'train_pos':tr,'test_pos':te,'gt_map':gt,'coordinate_base':0})
    probe=GPU(device,root/'preflight_gpu.csv'); probe.start(); probe.close()
    if args.preflight:
        print('Data/split/GPU preflight passed. Use another --out for training.'); return
    records=[]
    for seed,(tr,te) in splits.items():
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if device.type=='cuda':
            torch.cuda.manual_seed_all(seed); torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
        folder=root/f'seed_{seed}'; folder.mkdir()
        train=torch.utils.data.DataLoader(Patches(padded,gt,tr,args.patch,args.augment),batch_size=args.batch,shuffle=True)
        test=torch.utils.data.DataLoader(Patches(padded,gt,te,args.patch),batch_size=args.batch,shuffle=False)
        model=MViT(args.patch,[band,x.shape[2]-band],k,64,6,4,32,dropout=args.dropout,emb_dropout=args.dropout).to(device)
        opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
        sch=torch.optim.lr_scheduler.StepLR(opt,20,.9)
        monitor=GPU(device,folder/'gpu_usage.csv'); monitor.start()
        train_time=test_time=0.; best=-float('inf'); chosen_epoch=0
        start=time.perf_counter()
        try:
            with (folder/'epochs.csv').open('w',newline='',encoding='utf-8') as f:
                w=csv.writer(f); w.writerow(['epoch','lr','train_loss','train_OA','test_OA','test_AA','test_Kappa','train_s','test_s'])
                for e in range(1,args.epochs+1):
                    lr=opt.param_groups[0]['lr']; monitor.phase='train'
                    tm,_,_,_,_,loss,tt=epoch(model,train,band,device,k,opt); train_time+=tt; sch.step()
                    em={}; et=0.
                    if e==1 or e%args.test_freq==0 or e==args.epochs:
                        monitor.phase='test'
                        em,_,_,_,_,_,et=epoch(model,test,band,device,k); test_time+=et
                        print(seed,e,em,flush=True)
                        if args.select=='test_peak' and em['OA']>best:
                            best=em['OA']; chosen_epoch=e; torch.save(model.state_dict(),folder/'selected.pt')
                    w.writerow([e,lr,loss,tm['OA'],em.get('OA'),em.get('AA'),em.get('Kappa'),tt,et]); f.flush()
            torch.save(model.state_dict(),folder/'last.pt')
            if args.select=='last':
                chosen_epoch=args.epochs; torch.save(model.state_dict(),folder/'selected.pt')
            model.load_state_dict(torch.load(folder/'selected.pt',map_location=device,weights_only=True))
            monitor.phase='test'
            m,ca,cm,y,p,_,single_test_time=epoch(model,test,band,device,k)
            pred=np.zeros_like(gt); pred[te[:,0],te[:,1]]=p+1
            mask=np.zeros_like(gt); mask[te[:,0],te[:,1]]=1
            monitor.phase='map'; sync(device); t=time.perf_counter()
            mapping=infer_map(model,padded,gt,band,k,device,args); sync(device)
            map_time=time.perf_counter()-t
            result={'gt_map':gt,'test_pred_map':pred,'test_mask':mask,'train_pos':tr,'test_pos':te,
                    'y_true':y,'y_pred':p,'coordinate_base':0,'vector_label_base':0,'map_label_base':1,
                    'confusion_matrix':cm,'class_acc':ca,'selected_epoch':chosen_epoch,'seed':seed,**m}
            result['full_pred_map' if args.full_map else 'labeled_pred_map']=mapping
            savemat(folder/'result.mat',result,do_compression=True); draw(folder/'result.mat')
        finally:
            gpu=monitor.close()
        peak=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None
        record={'seed':seed,'selected_epoch':chosen_epoch,**m,'class_acc':ca.tolist(),
                'train_time_s':train_time,'periodic_test_time_s':test_time,'selected_single_test_time_s':single_test_time,
                'map_inference_time_s':map_time,'wall_time_s':time.perf_counter()-start,'GPU':gpu,
                'peak_torch_allocated_MiB':peak,'selection':args.select}
        (folder/'metrics.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        records.append(record)
        summary={'runs':records,'std_ddof':1,'mean_std':{key:{'mean':float(np.mean([r[key] for r in records])),
                  'std':float(np.std([r[key] for r in records],ddof=1)) if len(records)>1 else None}
                  for key in ('OA','AA','Kappa')}}
        (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        del model,opt,train,test
        if device.type=='cuda': torch.cuda.empty_cache()

if __name__=='__main__': main()
