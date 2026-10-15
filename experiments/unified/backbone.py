# -*- coding: utf-8 -*-
"""Fresh backbone fits on the frozen outer/fold splits. No TEST during fitting."""
from common import *
import argparse,random,importlib.util
sys.path.insert(0,str(BASE))
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset,DataLoader
torch.set_num_threads(4)
DEVICE=torch.device('cuda')

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True

class Windows(Dataset):
    def __init__(self,d,starts,scaled):
        self.d=d;self.starts=np.asarray(starts);self.values=scaled
        self.seq_len=d['L'];self.pred_len=d['H'];self.label_len=48
    def __len__(self):return len(self.starts)
    def __getitem__(self,index):
        s=int(self.starts[index]);end=s+self.seq_len
        return (index,self.values[s:end,None].astype(np.float32),
                self.values[end-48:end+self.pred_len,None].astype(np.float32),
                np.zeros((self.seq_len,4),np.float32),np.zeros((48+self.pred_len,4),np.float32))

def module_from_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def make_model(backbone,l,h):
    from paer_core.backbone_config import build_args
    args=build_args(backbone,l,h)
    folder={'PatchTST':'PatchTST/PatchTST_supervised','iTransformer':'iTransformer','RAFT':'RAFT'}[backbone]
    sys.path.insert(0,str(EXTERNAL/folder))
    if backbone=='PatchTST':
        from models.PatchTST import Model
    elif backbone=='iTransformer':
        from model.iTransformer import Model
    else:
        from models.RAFT import Model
    model=Model(args)
    return model.float().to(DEVICE)

def prepare_raft(model,train,sets,cache_dir,cache_identity):
    """Keep upstream retrieval arithmetic; use bounded batches and no worker processes."""
    cache_dir.mkdir(parents=True,exist_ok=True)
    idpath=cache_dir/'identity.json'
    if idpath.exists():assert read(idpath)==cache_identity
    else:dump(idpath,cache_identity)
    pending={k:v for k,v in sets.items() if not (cache_dir/(k+'.pt')).exists()}
    if pending:
        model.rt.prepare_dataset(train)
        for key,ds in pending.items():
            pieces=[];loader=DataLoader(ds,batch_size=64,shuffle=False,num_workers=0,drop_last=False)
            with torch.no_grad():
                for j,(index,x,*_) in enumerate(loader):
                    p=model.rt.retrieve(x.to(DEVICE),index,train=(key=='train')).cpu()
                    assert torch.isfinite(p).all(),'RAFT retrieved nonfinite values'
                    pieces.append(p)
                    if j%50==0:print('RAFT RETRIEVAL',key,j,len(loader),flush=True)
            torch.save(torch.cat(pieces,dim=1),cache_dir/(key+'.pt'))
    model.retrieval_dict={k:torch.load(cache_dir/(k+'.pt'),map_location='cpu',weights_only=True) for k in sets}
    del model.rt;torch.cuda.empty_cache()

def call(model,bb,batch,h,mode):
    index,x,y,xm,ym=batch;x=x.to(DEVICE)
    if bb=='PatchTST':out=model(x)
    elif bb=='iTransformer':
        # The decoder receives only already-observed label_len context and zero future.
        dec=torch.cat([y[:,:48,:],torch.zeros_like(y[:,-h:,:])],1).to(DEVICE)
        out=model(x,xm.to(DEVICE),dec,ym.to(DEVICE))
    else:out=model(x,index,mode=mode)
    return out[:,-h:,:],y[:,-h:,:].to(DEVICE)

def evaluate(model,bb,ds,batch_size,mode,return_predictions=False):
    model.eval();total=0.;count=0;pieces=[]
    with torch.no_grad():
        for batch in DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=0,drop_last=False):
            pred,y=call(model,bb,batch,ds.pred_len,mode)
            assert pred.shape==y.shape and torch.isfinite(pred).all()
            total+=float(torch.square(pred-y).double().sum());count+=y.numel()
            if return_predictions:pieces.append(pred.cpu().numpy()[:,:,0])
    return (total/count,np.concatenate(pieces)) if return_predictions else total/count

def fit_or_predict(bb,d,fold,seed,stage):
    path=ART/'backbones'/bb/d['ds']/(f'fold{fold}' if fold else f'final/seed{seed}')
    path.mkdir(parents=True,exist_ok=True)
    spec=fold_spec(d,fold) if fold else final_spec(d)
    train_starts=spec['bank'] if fold else d['starts']['train']
    select_starts=spec['fit'] if fold else spec['val']
    assert train_starts[-1]+d['H']<=select_starts[0]
    keys=['cal','eval'] if fold else (['test'] if stage=='test' else ['val'])
    signature=dict(protocol_hash=digest(PLAN),input=identity(d),backbone=bb,fold=fold,seed=seed,
                   train_starts=train_starts.tolist(),selection_starts=select_starts.tolist())
    if (path/'identity.json').exists():assert read(path/'identity.json')==signature
    else:dump(path/'identity.json',signature)
    target=path/('test_predictions.npz' if stage=='test' else 'predictions.npz')
    marker=path/('test_completed.json' if stage=='test' else 'completed.json')
    if marker.exists():
        assert read(marker)['protocol_hash']==digest(PLAN) and target.exists();return
    scaler=StandardScaler()
    # Scale only rows whose labels belong to this fit, never future CAL/EVAL.
    first=int(train_starts[0]);end=int(train_starts[-1])+d['L']+d['H']
    scaler.fit(d['series'][first:end,None].astype(np.float64))
    scaled=scaler.transform(d['series'][:,None].astype(np.float64))[:,0]
    dump(path/'scaler.json',{'fit_row_interval':[first,end],'mean':scaler.mean_.tolist(),'scale':scaler.scale_.tolist()})
    train=Windows(d,train_starts,scaled);selection=Windows(d,select_starts,scaled)
    sets={k:Windows(d,spec[k],scaled) for k in keys}
    seed_all(seed);model=make_model(bb,d['L'],d['H'])
    batch_size=128 if bb=='PatchTST' else 32
    checkpoint=path/'checkpoint.pt'
    if bb=='RAFT':
        allsets={'train':train,'selection':selection,**sets}
        ci=dict(backbone=bb,dataset=d['ds'],fold=fold,input=identity(d),train=train_starts.tolist(),selection=select_starts.tolist(),protocol=digest(PLAN))
        # Retrieval has no fitted parameters and is shared between random seeds.
        cache=ART/'raft_retrieval'/d['ds']/(f'fold{fold}' if fold else 'final')
        prepare_raft(model,train,allsets,cache,ci)
    if stage=='test':assert checkpoint.exists() and (path/'completed.json').exists(),'Fit/VAL checkpoint must be frozen before TEST'
    if not checkpoint.exists():
        assert stage!='test'
        opt=torch.optim.Adam(model.parameters(),lr=1e-4)
        generator=torch.Generator().manual_seed(seed)
        loader=DataLoader(train,batch_size=batch_size,shuffle=True,num_workers=0,drop_last=False,generator=generator)
        best=float('inf');history=[];epochs=PLAN['backbone_epochs']
        for epoch in range(1,epochs+1):
            start=time.monotonic();model.train();total=0.;n=0
            for batch in loader:
                pred,y=call(model,bb,batch,d['H'],'train');loss=torch.nn.functional.mse_loss(pred,y)
                assert torch.isfinite(loss),'Nonfinite training loss'
                opt.zero_grad();loss.backward();opt.step();total+=float(loss.detach())*len(pred);n+=len(pred)
            val=evaluate(model,bb,selection,batch_size,'selection')
            if val<best:
                best=val;torch.save(model.state_dict(),path/'best_in_progress.pt');best_epoch=epoch
            lr=(1e-4 if epoch<3 else 1e-4*.9**(epoch-3)) if bb=='PatchTST' else 1e-4*.5**(epoch-1)
            for group in opt.param_groups:group['lr']=lr
            row=dict(epoch=epoch,train_loss=total/n,selection_loss=val,best_loss=best,seconds=time.monotonic()-start)
            history.append(row);dump(path/'training_history.json',history)
            print('EPOCH',bb,d['ds'],fold,seed,row,flush=True)
        (path/'best_in_progress.pt').replace(checkpoint)
        dump(path/'training_complete.json',dict(epochs=epochs,best_epoch=best_epoch,best_selection_mse=best,checkpoint_sha256=sha(checkpoint)))
    model.load_state_dict(torch.load(checkpoint,map_location=DEVICE,weights_only=True))
    arrays={}
    for k,ds in sets.items():
        _,p=evaluate(model,bb,ds,batch_size,k,True)
        p=scaler.inverse_transform(p.reshape(-1,1)).reshape(p.shape).astype(np.float32)
        _,truth=windows(d['series'],spec[k],d['L'],d['H'])
        assert p.shape==truth.shape and np.isfinite(p).all()
        arrays['yc_'+k]=p;arrays['Y_'+k]=truth;arrays[k+'_starts']=spec[k]
    np.savez_compressed(target,**arrays)
    meta=dict(protocol_hash=digest(PLAN),input=identity(d),backbone=bb,dataset=d['ds'],fold=fold,seed=seed,
              fresh_fit=True,checkpoint_sha256=sha(checkpoint),prediction_sha256=sha(target),source_code_sha256=sha(Path(__file__)))
    dump(marker,meta);print('BACKBONE COMPLETE',bb,d['ds'],fold,seed,stage,flush=True)

def chronos(d,fold,stage):
    from chronos import Chronos2Pipeline
    path=ART/'backbones'/'Chronos-2'/d['ds']/(f'fold{fold}' if fold else 'final/shared')
    path.mkdir(parents=True,exist_ok=True)
    spec=fold_spec(d,fold) if fold else final_spec(d)
    keys=['cal','eval'] if fold else (['test'] if stage=='test' else ['val'])
    target=path/('test_predictions.npz' if stage=='test' else 'predictions.npz')
    marker=path/('test_completed.json' if stage=='test' else 'completed.json')
    if marker.exists():assert read(marker)['protocol_hash']==digest(PLAN);return
    pipe=Chronos2Pipeline.from_pretrained(str(CHRONOS),device_map='cuda')
    from paer_core.forecasting import run_chronos
    arrays={}
    for k in keys:
        x,y=windows(d['series'],spec[k],d['L'],d['H'])
        print('CHRONOS INFERENCE',d['ds'],fold,k,len(x),flush=True)
        p=run_chronos(pipe,list(x),d['H']);assert p.shape==y.shape and np.isfinite(p).all()
        arrays['yc_'+k]=p;arrays['Y_'+k]=y;arrays[k+'_starts']=spec[k]
    np.savez_compressed(target,**arrays)
    dump(marker,dict(protocol_hash=digest(PLAN),input=identity(d),model_snapshot=str(CHRONOS),model_config_hash=sha(CHRONOS/'config.json'),fresh_inference=True,fresh_fit=False,prediction_sha256=sha(target)))
    print('BACKBONE COMPLETE Chronos-2',d['ds'],fold,stage,flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--backbone',required=True)
    ap.add_argument('--fold',type=int,default=0);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--stage',choices=['val','test'],default='val')
    a=ap.parse_args();assert_frozen();assert torch.cuda.is_available()
    d=data(a.dataset)
    if a.backbone=='Chronos-2':chronos(d,a.fold,a.stage)
    else:fit_or_predict(a.backbone,d,a.fold,a.seed,a.stage)
if __name__=='__main__':main()
