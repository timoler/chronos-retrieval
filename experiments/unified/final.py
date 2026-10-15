# -*- coding: utf-8 -*-
"""Train matched retrievers and freeze all per-backbone VAL choices before TEST."""
from common import *
from runtime import *
from reward_core import load_data,semantic,new_model
import argparse
from sentence_transformers import SentenceTransformer

def pool_map(d,backbone):
    pools={'NoExpertMLP':[],'Raw':d['raw']}
    for arm,obj in [('MemoryReward','memory'),('FusionReward','fusion'),('FusionReward_NoCritic','fusion_no_critic'),('FusionReward_Independent','fusion_independent')]:
        p=ART/'pools'/d['ds']/obj/('shared' if obj=='memory' else backbone)
        assert read(p/'completed.json')['protocol_hash']==digest(PLAN)
        pools[arm]=read(p/'final_pool.json')
    return pools

def prepare(d,seed,pool,split,encoder):
    spec=final_spec(d);path=ART/'memories'/d['ds']/f'seed{seed}'/digest(semantic(pool))[:24]
    path.mkdir(parents=True,exist_ok=True)
    signature=dict(protocol_hash=digest(PLAN),data=identity(d),seed=seed,pool=semantic(pool),source=sha(Path(__file__)))
    if (path/'identity.json').exists():assert read(path/'identity.json')==signature
    else:dump(path/'identity.json',signature)
    predpath=path/(split+'.npz')
    if predpath.exists():
        with np.load(predpath) as z:return z['transferred'],path
    X={};Y={};R={};ts={}
    for k in ['bank','fit',split]:
        X[k],Y[k]=windows(d['series'],spec[k],d['L'],d['H']);R[k]=v2.residualize(X[k]);ts[k]=[d['dates'][int(s)+d['L']] for s in spec[k]]
    hist=np.concatenate([X['bank'],X['fit']]);stats=v2.compute_training_stats(hist)
    evidence={'feature_distributions':v1._raw_feature_quantiles(hist),'temporal_coverage':v1._temporal_coverage(d['dates'].iloc[:int(spec['fit'][-1])+d['L']])}
    compiled=semantic(pool);emb=np.zeros((8,384),np.float32);acts={k:np.zeros((len(x),8),np.float32) for k,x in X.items()}
    v1.CONTEXT=d['L'];v1.HORIZON=d['H']
    for e in compiled:
        e['compiled']=v1.compile_condition_v6(e,v1.FEATURE_REGISTRY,evidence,v1.COMPILER_CONFIG)
        e['retrieval_spec']=v1.derive_retrieval_spec(e,v1.FEATURE_REGISTRY)
    if compiled:
        encoded=encoder.encode([e['claim'] for e in compiled],normalize_embeddings=True,convert_to_numpy=True).astype(np.float32)
        for j,e in enumerate(compiled):emb[d['ids'].index(e['id'])]=encoded[j]
        for k in X:
            aa=v1.build_activation_matrix(compiled,X[k],ts[k],stats)
            for j,e in enumerate(compiled):acts[k][:,d['ids'].index(e['id'])]=aa[:,j]
    checkpoint=path/'model.pt';model=new_model(d['L'],emb,seed)
    if checkpoint.exists():model.load_state_dict(torch.load(checkpoint,map_location=DEVICE,weights_only=True))
    else:
        assert split=='val','No new fitting after global VAL freeze'
        print('TRAIN FINAL RETRIEVER',d['ds'],seed,len(pool),path.name,flush=True)
        v2p.train_arm(model,X['bank'],Y['bank'],X['fit'],Y['fit'],R['bank'],R['fit'],acts['bank'],acts['fit'],True,DEVICE,seed)
        torch.save(model.state_dict(),checkpoint);np.save(path/'embeddings.npy',emb);dump(path/'compiled_pool.json',compiled)
    plain,transfer,index=RETRIEVE(model,X[split],X['bank'],Y['bank'],R[split],R['bank'],acts[split],acts['bank'],'full',5,DEVICE)
    assert np.isfinite(transfer).all()
    np.savez_compressed(predpath,prediction=plain,transferred=transfer,indices=index,bank_starts=spec['bank'],query_starts=spec[split],truth=Y[split])
    np.savez_compressed(path/(split+'_activations.npz'),**acts)
    complete(path,dict(seed=seed,pool_size=len(pool),checkpoint_sha256=sha(checkpoint),last_prediction_split=split))
    del model;gc.collect();torch.cuda.empty_cache()
    return transfer,path

def shape(d,seed,split,encoder):
    _,zero_path=prepare(d,seed,[],split,encoder)
    path=ART/'memories'/d['ds']/'Shape';path.mkdir(parents=True,exist_ok=True)
    target=path/(split+'.npz')
    if target.exists():
        with np.load(target) as z:return z['transferred'],path
    spec=final_spec(d);x,y=windows(d['series'],spec[split],d['L'],d['H']);xb,yb=windows(d['series'],spec['bank'],d['L'],d['H'])
    model=new_model(d['L'],np.zeros((8,384),np.float32),seed)
    plain,transfer,index=RETRIEVE(model,x,xb,yb,v2.residualize(x),v2.residualize(xb),np.zeros((len(x),8),np.float32),np.zeros((len(xb),8),np.float32),'shape',5,DEVICE)
    np.savez_compressed(target,prediction=plain,transferred=transfer,indices=index,bank_starts=spec['bank'],query_starts=spec[split],truth=y)
    del model;torch.cuda.empty_cache()
    return transfer,path

def run(ds,seed,split):
    assert_frozen();d=load_data(ds)
    if split=='test':
        for p,h in read(ART/'READOUT_FROZEN.json')['val_files'].items():assert sha(p)==h,'VAL choice changed after freeze'
    encoder=SentenceTransformer(str(EMBED),device='cuda',local_files_only=True)
    _,truth=windows(d['series'],d['starts'][split],d['L'],d['H']);result_rows=[]
    memory_cache={}
    shape_pred,shape_path=shape(d,seed,split,encoder)
    for bb in PLAN['backbones']:
        sd=ART/'final'/ds/bb/f'seed{seed}';sd.mkdir(parents=True,exist_ok=True)
        base=ART/'backbones'/bb/ds/'final'/('shared' if bb=='Chronos-2' else f'seed{seed}')
        bp=base/('predictions.npz' if split=='val' else 'test_predictions.npz')
        with np.load(bp) as z:
            b=z['yc_'+split].astype(np.float64)
            assert np.array_equal(z['Y_'+split],truth) and np.array_equal(z[split+'_starts'],d['starts'][split])
        pools=pool_map(d,bb);preds={'Backbone':b,'Shape':shape_pred};sources={'Backbone':str(bp),'Shape':str(shape_path/(split+'.npz'))}
        for arm,pool in pools.items():
            ph=digest(semantic(pool))
            if ph not in memory_cache:memory_cache[ph]=prepare(d,seed,pool,split,encoder)
            p,mp=memory_cache[ph];preds[arm]=p;sources[arm]=str(mp/(split+'.npz'))
        if split=='val':
            choices={}
            for arm,p in preds.items():
                l,grid=select(b,p,truth)
                choices[arm]=dict(lambda_value=l,grid=grid,val_memory_mse=mse(p,truth),val_fusion_mse=mse((1-l)*b+l*p,truth),
                    memory_path=sources[arm],memory_sha256=sha(sources[arm]),backbone_path=str(bp),backbone_sha256=sha(bp),
                    pool_size=len(pools[arm]) if arm in pools else None,pool_hash=digest(semantic(pools[arm])) if arm in pools else None)
            frozen=sd/'VAL_CHOICES.json'
            if frozen.exists():assert read(frozen)==choices
            else:dump(frozen,choices)
            print('VAL FROZEN',ds,bb,seed,{a:r['lambda_value'] for a,r in choices.items()},flush=True)
        else:
            choices=read(sd/'VAL_CHOICES.json');rows=[]
            for arm,p in preds.items():
                c=choices[arm];l=c['lambda_value'];fusion=(1-l)*b+l*p
                if arm in pools:assert c['pool_hash']==digest(semantic(pools[arm]))
                row=dict(dataset=ds,backbone=bb,seed=seed,arm=arm,lambda_value=l,pool_size=c['pool_size'],
                    backbone_mse=mse(b,truth),backbone_mae=mae(b,truth),memory_mse=mse(p,truth),memory_mae=mae(p,truth),
                    fusion_mse=mse(fusion,truth),fusion_mae=mae(fusion,truth),val_fusion_mse=c['val_fusion_mse'],
                    memory_path=sources[arm],memory_sha256=sha(sources[arm]),backbone_path=str(bp),backbone_sha256=sha(bp),
                    val_choices_path=str(sd/'VAL_CHOICES.json'),val_choices_sha256=sha(sd/'VAL_CHOICES.json'),n=len(truth),
                    protocol_hash=digest(PLAN),empty_pool_alias=(arm in pools and not pools[arm]))
                row['ri_pct']=100*(row['backbone_mse']-row['fusion_mse'])/row['backbone_mse'];rows.append(row)
            dump(sd/'TEST_METRICS.json',rows);result_rows.extend(rows)
            print('TEST COMPLETE',ds,bb,seed,flush=True)
    if split=='test':
        dest=ART/'final'/ds/f'seed{seed}_metrics.json';dump(dest,result_rows)
    complete(ART/'final'/ds/f'{split}_seed{seed}',dict(dataset=ds,seed=seed,split=split))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--split',choices=['val','test'],required=True)
    a=ap.parse_args();run(a.dataset,a.seed,a.split)
