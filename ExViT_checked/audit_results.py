"""Audit saved 5% results; optionally reload weights and repeat test inference."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.io import loadmat
from run_checked import load_data, split, Patches, MViT

def independently_score(y,p,k):
    cm=np.zeros((k,k),dtype=np.int64)
    for truth in range(k):
        for prediction in range(k):
            cm[truth,prediction]=np.count_nonzero((y==truth)&(p==prediction))
    ca=np.array([np.mean(p[y==c]==c) for c in range(k)])*100
    oa=float(np.mean(y==p))
    pe=sum(float(np.mean(y==c))*float(np.mean(p==c)) for c in range(k))
    return {'OA':oa*100,'AA':float(ca.mean()),'Kappa':(oa-pe)/(1-pe)},ca,cm

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--dataset',choices=['Houston','Trento'],required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--log',type=Path,required=True,help='dataset-level log directory')
    parser.add_argument('--seed',type=int,default=21)
    parser.add_argument('--gpu',type=int,default=0)
    parser.add_argument('--recheck-model',action='store_true')
    args=parser.parse_args()
    config=json.loads((args.log/'config.json').read_text(encoding='utf-8'))
    folder=args.log/f'seed_{args.seed}'
    saved=loadmat(args.log/f'split_{args.seed}.mat')
    x,gt,band,k=load_data(args.root,args.dataset)
    tr,te=saved['train_pos'].astype(int),saved['test_pos'].astype(int)
    checks={}
    def check(name,ok):
        checks[name]=bool(ok)
        print(('PASS ' if ok else 'FAIL ')+name,flush=True)
    check('dataset matches logged dataset',config['dataset']==args.dataset)
    check('saved GT equals dataset GT',np.array_equal(saved['gt_map'],gt))
    for name,pos in [('train',tr),('test',te)]:
        valid=pos.ndim==2 and pos.shape[1]==2 and np.all(pos>=0) and np.all(pos[:,0]<gt.shape[0]) and np.all(pos[:,1]<gt.shape[1])
        check(name+' coordinates valid',valid)
        if not valid: raise ValueError('Invalid coordinates: '+name)
    train_ids=np.ravel_multi_index(tr.T,gt.shape)
    test_ids=np.ravel_multi_index(te.T,gt.shape)
    check('no duplicate training coordinates',len(np.unique(train_ids))==len(tr))
    check('no duplicate test coordinates',len(np.unique(test_ids))==len(te))
    check('no train/test center overlap',np.intersect1d(train_ids,test_ids).size==0)
    check('train and test cover exactly all labeled GT',np.array_equal(np.sort(np.concatenate([train_ids,test_ids])),np.flatnonzero(gt.ravel()>0)))
    counts=np.bincount(gt.ravel(),minlength=k+1)[1:]
    actual=np.bincount(gt[tr[:,0],tr[:,1]],minlength=k+1)[1:]
    check('per-class floor(5%) training counts',np.array_equal(actual,np.floor(counts*.05).astype(int)))
    expected_tr,expected_te=split(gt,k,args.seed)
    check('saved split reproducible for this seed',np.array_equal(tr,expected_tr) and np.array_equal(te,expected_te))
    model_path=Path(__file__).with_name('MViT_pytorch_upload.py')
    check('model source matches logged hash',hashlib.sha256(model_path.read_bytes()).hexdigest()==config['model_sha256'])
    result_path=folder/'result.mat'
    report={'seed':args.seed,'checks':checks,'protocol':'unchanged all-GT 5%','model_recheck_requested':args.recheck_model}
    if result_path.exists():
        result=loadmat(result_path)
        y=result['y_true'].ravel().astype(int)
        pred=result['y_pred'].ravel().astype(int)
        check('prediction vector length equals test count',len(y)==len(te) and len(pred)==len(te))
        if len(y)!=len(te) or len(pred)!=len(te):raise ValueError('Prediction length mismatch')
        check('saved test coordinates match split',np.array_equal(result['test_pos'],te))
        check('saved training coordinates match split',np.array_equal(result['train_pos'],tr))
        check('saved ground truth matches test coordinates',np.array_equal(y,gt[te[:,0],te[:,1]]-1))
        check('predictions in valid class range',np.all((pred>=0)&(pred<k)))
        check('result GT matches dataset GT',np.array_equal(result['gt_map'],gt))
        mask=np.zeros_like(gt);mask[te[:,0],te[:,1]]=1
        check('saved test mask includes only test coordinates',np.array_equal(mask,result['test_mask']))
        image=np.zeros_like(gt);image[te[:,0],te[:,1]]=pred+1
        check('saved test prediction image matches prediction vector',np.array_equal(image,result['test_pred_map']))
        m,ca,cm=independently_score(y,pred,k)
        report['independently_recomputed_metrics']=m
        check('confusion matrix matches independent calculation',np.array_equal(cm,result['confusion_matrix']))
        check('class accuracy matches independent calculation',np.allclose(ca,result['class_acc'].ravel(),atol=1e-8,rtol=0))
        for key,value in m.items():check('MAT '+key+' matches independent calculation',np.isclose(value,result[key].item(),atol=1e-8,rtol=0))
        if (folder/'metrics.json').exists():
            metrics=json.loads((folder/'metrics.json').read_text(encoding='utf-8'))
            for key,value in m.items():check('JSON '+key+' matches independent calculation',np.isclose(value,metrics[key],atol=1e-8,rtol=0))
        print('Independent metrics:',m,flush=True)
    else:
        print('Result MAT not yet saved; split audit only. Training may have been interrupted before export.',flush=True)
        report['result_status']='not_saved'
    if args.recheck_model:
        checkpoint=folder/'selected.pt'
        if not checkpoint.exists():raise FileNotFoundError('No selected.pt; cannot rerun saved model')
        device=torch.device('cpu' if args.gpu<0 else f'cuda:{args.gpu}')
        p=int(config['patch']);r=p//2
        padded=np.pad(x,((r,r),(r,r),(0,0)),mode='symmetric')
        dataset=Patches(padded,gt,te,p)
        # Changing labels must not change any input image tensor.
        fake_gt=np.where(gt>0,gt%k+1,0).astype(gt.dtype)
        fake=Patches(padded,fake_gt,te,p)
        check('patch input independent of labels (sampled)',all(torch.equal(dataset[i][0],fake[i][0]) for i in np.linspace(0,len(te)-1,min(32,len(te)),dtype=int)))
        loader=torch.utils.data.DataLoader(dataset,batch_size=int(config['batch']),shuffle=False)
        model=MViT(p,[band,x.shape[2]-band],k,64,6,4,32,dropout=config['dropout'],emb_dropout=config['dropout']).to(device)
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True))
        model.eval();outputs=[]
        with torch.no_grad():
            for inputs,_ in loader:
                inputs=inputs.to(device)
                outputs.append(model(inputs[:,:band],inputs[:,band:]).argmax(1).cpu().numpy())
        fresh=np.concatenate(outputs)
        truth=gt[te[:,0],te[:,1]]-1
        fresh_metrics,_,_=independently_score(truth,fresh,k)
        report['reloaded_model_metrics']=fresh_metrics
        print('Reloaded model metrics:',fresh_metrics,flush=True)
        if result_path.exists():
            different=int(np.count_nonzero(fresh!=pred))
            report['reloaded_prediction_mismatches']=different
            check('reloaded model predictions equal saved predictions',different==0)
            print('Prediction mismatches:',different,'/',len(te),flush=True)
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/'audit_report.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Audit report:',path,flush=True)
    if not all(checks.values()):raise SystemExit(2)

if __name__=='__main__':main()
