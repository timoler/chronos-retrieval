"""Read-only numerical audit of frozen PAER artifacts; writes only to --out.

No training, inference, model selection on TEST, or source-artifact mutation.
Requires numpy and pandas. Does not import experiment modules.
"""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument('--repo', required=True)
ap.add_argument('--out', required=True)
ap.add_argument('--paired-workspace')
args = ap.parse_args()
repo, out = Path(args.repo), Path(args.out)
out.mkdir(parents=True, exist_ok=True)
pkg = repo/'results/deepseek_matrix_delivery'
checks, inputs, metrics, summaries, protocols, pairs, checkpoints = [], {}, [], [], [], [], []
seeds = [42,123,2026]
datasets = ['etth1','etth2','exchange_rate','traffic','weather','solar','electricity']
backbones = ['Chronos-2','PatchTST','iTransformer','RAFT']
grid = [0,.05,.1,.15,.2,.3,.5,.7,1]

def digest(p):
    p=Path(p)
    if str(p) in inputs:return inputs[str(p)]['sha256']
    if not p.exists(): return None
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    inputs[str(p)]={'sha256':h.hexdigest(),'bytes':p.stat().st_size}
    return h.hexdigest()

def check(name, ok, detail=''):
    checks.append({'check':name,'status':'PASS' if bool(ok) else 'FAIL','detail':str(detail)})

def close(name,a,b,rtol=1e-8,atol=1e-10):
    aa,bb=np.asarray(a),np.asarray(b)
    ok=aa.shape==bb.shape and np.allclose(aa,bb,rtol=rtol,atol=atol,equal_nan=False)
    delta=float(np.max(np.abs(aa-bb))) if aa.shape==bb.shape and aa.size else None
    check(name,ok,f'max_abs_diff={delta}; rtol={rtol}; atol={atol}')

def js(p):
    digest(p)
    return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def rows(p):
    digest(p)
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def arr(p,key=None):
    digest(p)
    if key is None:return np.asarray(np.load(p,allow_pickle=False),dtype=np.float64)
    with np.load(p,allow_pickle=False) as z:return np.asarray(z[key],dtype=np.float64)

def mse(y,p):
    assert y.shape==p.shape, (y.shape,p.shape)
    assert np.isfinite(y).all() and np.isfinite(p).all()
    return float(np.mean((y-p)**2))

def select(y,b,m):
    losses=[mse(y,(1-l)*b+l*m) for l in grid]
    return grid[int(np.argmin(losses))]

def mem(ds,s,split):
    ext=pkg/'external_memories'
    if ds in ['etth1','weather']:return arr(ext/f'{ds}_PAER_{split}_retrieval_s{s}.npz','transferred')
    if ds=='etth2':return arr(ext/f'etth2_align_{split}_s{s}.npy')
    if ds=='traffic':return arr(ext/f'paer_align_{split}_s{s}.npy')
    return arr(pkg/f'mem/{ds}/paer_align_{split}_s{s}.npy')

manifest=js(pkg/'PACKAGE_MANIFEST.json')
for rel,record in manifest['files'].items():
    check('package_hash:'+rel,digest(pkg/rel)==record['sha256'])
for rel,record in manifest['provenance'].items():
    if not isinstance(record,dict) or 'src' not in record or 'sha256' not in record:
        continue
    src=Path(record['src'])
    check('source_hash:'+rel,digest(src)==record['sha256'],src)

saved_summary={(r['dataset'],r['backbone']):r for r in rows(pkg/'summary.csv')}
old_summary={r['dataset']:r for r in rows(repo/'results/paper_completion/phase1_external_baselines/fusion_reaudit_20261011/fusion_summary.csv')}
internal_ri=[]
for ds in datasets:
    d=repo/'results/multidataset'/ds
    cfg=js(d/'config.json'); L,H=int(cfg['context']),int(cfg['horizon'])
    digest(d/'series.csv'); values=pd.read_csv(d/'series.csv')['OT'].to_numpy(np.float32)
    digest(d/'window_indices.npz')
    with np.load(d/'window_indices.npz') as z: starts={k:z[k].copy() for k in ['train','val','test']}
    cache=pkg/f'chronos2_preds/{ds}.npz'; digest(cache)
    with np.load(cache) as z:
        native={k:z[k].copy() for k in z.files}
        cached={k:v.astype(np.float64) for k,v in native.items()}
    labels={}
    for split in ['train','val','test']:
        st=starts[split]; lo,hi=cfg['split'][split]
        expected=np.arange(lo,hi-L-H+1)
        close(f'{ds}:{split}:formal_starts',st,expected,rtol=0,atol=0)
        check(f'{ds}:{split}:within_partition',st.min()>=lo and st.max()+L+H<=hi)
        if split!='train':
            labels[split]=values[st[:,None]+L+np.arange(H)].astype(np.float64)
            close(f'{ds}:{split}:cached_labels',cached['Y_'+split],labels[split],0,0)
            close(f'{ds}:{split}:cached_starts',cached[split+'_starts'],st,0,0)
    check(f'{ds}:outer_train_val_isolation',starts['train'].max()+L+H<=starts['val'].min())
    check(f'{ds}:outer_val_test_isolation',starts['val'].max()+L+H<=starts['test'].min())
    cut=int(len(starts['train'])*.8); bank=starts['train'][:cut]
    # Count training labels unavailable at the earliest fit/query forecast origin.
    boundary_overlap=int(np.sum(bank+L+H-1>=starts['train'][cut]+L))
    protocols.append({'dataset':ds,'context':L,'horizon':H,'n_val':len(starts['val']),
                      'n_test':len(starts['test']),'bank_future_unavailable_at_first_fit':boundary_overlap,
                      'test_stride':int(np.min(np.diff(starts['test'])))})
    frozen=js(repo/f'results/reward_feedback_critic_v2_final_test/{ds}/result.json')
    rawmean=frozen['summary']['raw']['mean_mse']; paermean=frozen['summary']['v2']['mean_mse']
    internal_ri.append((rawmean-paermean)/rawmean*100)
    old_ms=[]
    for s in seeds:
        actual=mse(labels['test'],cached[f'mem_test_{s}'])
        ref=frozen['summary']['v2']['seeds_mse'][str(s)]
        close(f'{ds}:{s}:legacy_frozen_mse',actual,ref,rtol=1e-6,atol=1e-12)
        lam=select(labels['val'],cached['yc_val'],cached[f'mem_val_{s}'])
        close(f'{ds}:{s}:legacy_lambda',lam,float(old_summary[ds][f'lam_{s}']),0,0)
        # Legacy verifier multiplied the original float32 Chronos array before
        # promoting the sum. Preserve that operation order for legacy CSV checks.
        old_ms.append(mse(labels['test'],(1-lam)*native['yc_test']+lam*native[f'mem_test_{s}']))
    close(f'{ds}:legacy_summary',np.mean(old_ms),float(old_summary[ds]['fusion_new_mse_mean']),1e-8,5.1e-7)
    for bb in backbones:
        saved={int(r['seed']):r for r in rows(pkg/f'per_seed_results_{ds}_{bb}.csv')}
        group=[]
        for s in seeds:
            mv,mt=mem(ds,s,'val'),mem(ds,s,'test')
            if bb=='Chronos-2':bv,bt=cached['yc_val'],cached['yc_test']
            else:
                bv=arr(pkg/f'backbone/{ds}/{bb.lower()}_{ds}_s{s}_val.npy')
                bt=arr(pkg/f'backbone/{ds}/{bb.lower()}_{ds}_s{s}_test.npy')
            check(f'{ds}:{bb}:{s}:shape_finite',all(a.shape==labels[sp].shape and np.isfinite(a).all() for a,sp in [(bv,'val'),(bt,'test'),(mv,'val'),(mt,'test')]))
            lam=select(labels['val'],bv,mv); fused=(1-lam)*bt+lam*mt
            rec={'dataset':ds,'backbone':bb,'seed':s,'lambda':lam,'base_val_mse':mse(labels['val'],bv),'base_test_mse':mse(labels['test'],bt),'mem_test_mse':mse(labels['test'],mt),'fused_test_mse':mse(labels['test'],fused)}
            rec['ri_pct']=(rec['base_test_mse']-rec['fused_test_mse'])/rec['base_test_mse']*100
            for k in ['lambda','base_val_mse','base_test_mse','mem_test_mse','fused_test_mse','ri_pct']:
                close(f'{ds}:{bb}:{s}:csv:{k}',rec[k],float(saved[s][k]))
            metrics.append(rec);group.append(rec)
        bm=np.mean([r['base_test_mse'] for r in group]); fm=np.mean([r['fused_test_mse'] for r in group])
        summ={'dataset':ds,'backbone':bb,'base_mean_mse':float(bm),'base_std_ddof1':float(np.std([r['base_test_mse'] for r in group],ddof=1)),'fused_mean_mse':float(fm),'fused_std_ddof1':float(np.std([r['fused_test_mse'] for r in group],ddof=1)),'ri_pct':float((bm-fm)/bm*100),'lambda_42_123_2026':'/'.join(str(r['lambda']) for r in group)}
        for k in ['base_mean_mse','base_std_ddof1','fused_mean_mse','fused_std_ddof1','ri_pct']:close(f'{ds}:{bb}:summary:{k}',summ[k],float(saved_summary[ds,bb][k]))
        summaries.append(summ)
    # Inventory every checkpoint selected by the current inference scripts.
    # A present hash records current provenance, not proof of past execution.
    external=repo.parent/'_external_repos'
    for s in seeds:
        settings={
            'PatchTST':('PatchTST/PatchTST_supervised',f'batch_{ds}_s{s}_PatchTST_chronos_retrieval_ftS_sl{L}_ll48_pl{H}_dm128_nh16_el3_dl1_df256_fc1_ebtimeF_dtTrue_test_0'),
            'iTransformer':('iTransformer',f'batch_{ds}_s{s}_iTransformer_chronos_retrieval_S_ft{L}_sl48_ll{H}_pl512_dm8_nh3_el1_dl512_df1_fctimeF_ebTrue_dttest_projection_0'),
            'RAFT':('RAFT',f'long_term_forecast_batch_{ds}_s{s}_RAFT_chronos_retrieval_ftS_sl{L}_ll48_pl{H}_dm512_nh8_el2_dl1_df2048_expand2_dc4_fc1_ebtimeF_dtTrue_test_0')}
        for bb,(folder,setting) in settings.items():
            cp=external/folder/'checkpoints'/setting/'checkpoint.pth'
            sha=digest(cp)
            check(f'{ds}:{bb}:{s}:checkpoint_present',sha is not None)
            checkpoints.append({'dataset':ds,'backbone':bb,'seed':s,'path':str(cp),'sha256':sha,'binding':'current inference script path; no model forward rerun'})
    print('Audited',ds,flush=True)

if args.paired_workspace:
    cw=Path(args.paired_workspace)
    for ds in ['etth1','weather','traffic']:
        d=repo/'results/multidataset'/ds
        cfg=js(d/'config.json'); L,H=cfg['context'],cfg['horizon']
        values=pd.read_csv(d/'series.csv')['OT'].to_numpy(np.float32)
        with np.load(d/'window_indices.npz') as z:formal={k:z[k].copy() for k in ['train','val','test']}
        c=pkg/f'chronos2_preds/{ds}.npz'
        for s in seeds:
            source=cw/'work'/('pilot_artifacts' if s==42 else f'replicate_artifacts/seed{s}')/ds
            for arm in ['Raw','PAER']:
                pred={}
                for split in ['val','test']:
                    p=source/f'{arm}_{split}_retrieval.npz'; digest(p)
                    with np.load(p) as z:rr={k:z[k].copy() for k in z.files}
                    close(f'paired:{ds}:{s}:{arm}:{split}:starts',rr['query_starts'],formal[split],0,0)
                    selected=rr['bank_starts'][rr['indices']]
                    check(f'paired:{ds}:{s}:{arm}:{split}:historical',np.all(selected+L+H-1<rr['query_starts'][:,None]+L))
                    raw=values[selected[:,:,None]+L+np.arange(H)].mean(axis=1)
                    qmean=values[rr['query_starts'][:,None]+np.arange(L)].mean(axis=1)
                    cmean=values[selected[:,:,None]+np.arange(L)].mean(axis=2).mean(axis=1)
                    transferred=raw+qmean[:,None]-cmean[:,None]
                    close(f'paired:{ds}:{s}:{arm}:{split}:raw_reconstruction',raw,rr['prediction'],1e-6,1e-6)
                    close(f'paired:{ds}:{s}:{arm}:{split}:aligned_reconstruction',transferred,rr['transferred'],1e-6,1e-5)
                    pred[split]=rr
                yv,yt=arr(c,'Y_val'),arr(c,'Y_test'); bv,bt=arr(c,'yc_val'),arr(c,'yc_test')
                for agg,key in [('original','prediction'),('aligned','transferred')]:
                    mv=pred['val'][key].astype(np.float64); mt=pred['test'][key].astype(np.float64)
                    lam=select(yv,bv,mv)
                    pairs.append({'dataset':ds,'seed':s,'arm':arm,'aggregation':agg,'lambda':lam,'memory_test_mse':mse(yt,mt),'fusion_test_mse':mse(yt,(1-lam)*bt+lam*mt)})

def save(name,data):
    if not data:return
    with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
for folder in ['PatchTST/PatchTST_supervised','iTransformer','RAFT']:
    for file in ['data_provider/chronos_loader.py','data_provider/data_factory.py']:
        digest(repo.parent/'_external_repos'/folder/file)
traffic=pd.read_csv(repo/'data/traffic.csv')
formal_traffic=pd.read_csv(repo/'results/multidataset/traffic/series.csv')['OT'].to_numpy()
close('traffic:target_is_Sensor_000',formal_traffic,traffic['Sensor_000'].to_numpy(),0,0)
digest(repo/'data/traffic.csv')
save('checks.csv',checks);save('matrix_per_seed.csv',metrics);save('matrix_summary.csv',summaries);save('protocol_checks.csv',protocols);save('raw_paer_pairs.csv',pairs);save('checkpoint_inventory.csv',checkpoints)
summary={'checks':len(checks),'passed':sum(r['status']=='PASS' for r in checks),'failed':sum(r['status']=='FAIL' for r in checks),'matrix_rows':len(metrics),'matrix_groups':len(summaries),'positive_groups':sum(r['ri_pct']>0 for r in summaries),'internal_mean_ri_pct':float(np.mean(internal_ri)),'backbone_mean_ri_pct':{b:float(np.mean([r['ri_pct'] for r in summaries if r['backbone']==b])) for b in backbones},'limitations':['Cache-level verification; no GPU inference or training rerun.','Chronology of protocol choice is not established by cache arithmetic.','Matrix has PAER only; does not isolate expert optimization contribution.','No independent-window significance claims.']}
(out/'audit_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
(out/'input_hashes.json').write_text(json.dumps(inputs,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
for r in checks:
    if r['status']=='FAIL': print(r)
