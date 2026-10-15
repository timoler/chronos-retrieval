# -*- coding: utf-8 -*-
"""Independent saved-array verification and full 672-row readout."""
import os,json,hashlib,csv
from pathlib import Path
import numpy as np
import pandas as pd

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def score(p,y):
    e=np.asarray(p,dtype=np.float64)-np.asarray(y,dtype=np.float64)
    return float(np.dot(e.ravel(),e.ravel())/e.size),float(np.abs(e).sum()/e.size)
def writecsv(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def main():
    config=read(os.environ['PAER_RUN_CONFIG']);plan=config['protocol'];root=Path(config['run_root']);out=Path(config['report_root'])
    repo=Path(__file__).resolve().parents[2];inputs=Path(config['input_repo'])/'data/processed'
    frozen=read(root/'READOUT_FROZEN.json')
    for p,h in frozen['val_files'].items():assert sha(p)==h
    rows=[];checks=[];traces=[];cache={}
    def arrays(path):
        if path not in cache:
            with np.load(path) as z:cache[path]={k:z[k] for k in z.files}
        return cache[path]
    for ds in plan['datasets']:
        dd=inputs/ds;cfg=read(dd/'config.json');series=pd.read_csv(dd/'series.csv').OT.to_numpy(np.float32)
        with np.load(dd/'window_indices.npz') as z:starts={k:z[k] for k in ['val','test']}
        labels={k:np.stack([series[s+cfg['context']:s+cfg['context']+cfg['horizon']] for s in starts[k]]) for k in starts}
        for seed in plan['seeds']:
            records=read(root/'final'/ds/f'seed{seed}_metrics.json')
            assert len(records)==len(plan['backbones'])*len(plan['core_arms'])
            for r in records:
                choice=read(r['val_choices_path'])[r['arm']];l=float(choice['lambda_value'])
                assert sha(r['val_choices_path'])==r['val_choices_sha256']
                for record in [choice,r]:
                    for field in ['memory','backbone']:
                        assert sha(record[field+'_path'])==record[field+'_sha256']
                memories={};bases={}
                for split,mp,bp in [('val',choice['memory_path'],choice['backbone_path']),('test',r['memory_path'],r['backbone_path'])]:
                    b=arrays(bp);assert np.array_equal(b[split+'_starts'],starts[split]) and np.array_equal(b['Y_'+split],labels[split])
                    bases[split]=b['yc_'+split].astype(np.float64)
                    m=arrays(mp)
                    if r['arm']=='Backbone':memories[split]=m['yc_'+split].astype(np.float64)
                    else:
                        assert np.array_equal(m['query_starts'],starts[split]) and np.array_equal(m['truth'],labels[split])
                        idx=m['indices'];bs=m['bank_starts'];assert idx.min()>=0 and idx.max()<len(bs)
                        assert bs.max()+cfg['context']+cfg['horizon']<=starts[split].min()+cfg['context']
                        # Reconstruct mean-transfer predictions using only saved selected candidates.
                        qb=np.stack([series[s:s+cfg['context']] for s in starts[split]])
                        bx=np.stack([series[s:s+cfg['context']] for s in bs]);by=np.stack([series[s+cfg['context']:s+cfg['context']+cfg['horizon']] for s in bs])
                        reconstructed=by[idx].mean(axis=1)+qb.mean(axis=1)[:,None]-bx[idx].mean(axis=2).mean(axis=1)[:,None]
                        assert np.array_equal(reconstructed,m['transferred']),f'Top5 reconstruction {ds}/{seed}/{r["arm"]}'
                        memories[split]=m['transferred'].astype(np.float64)
                vg=[score((1-g)*bases['val']+g*memories['val'],labels['val'])[0] for g in plan['lambda_grid']]
                selected=0. if np.array_equal(bases['val'],memories['val']) else plan['lambda_grid'][int(np.argmin(vg))]
                assert selected==l
                fusion=(1-l)*bases['test']+l*memories['test'];metrics=score(fusion,labels['test'])
                for name,value in zip(['fusion_mse','fusion_mae'],metrics):
                    old=float(r[name]);passed=bool(np.isclose(value,old,rtol=5e-10,atol=1e-12))
                    checks.append(dict(dataset=ds,backbone=r['backbone'],seed=seed,arm=r['arm'],metric=name,reported=old,recomputed=value,abs_diff=abs(value-old),status='PASS' if passed else 'FAIL'))
                    assert passed
                rows.append(r)
                traces.append(dict(dataset=ds,backbone=r['backbone'],seed=seed,arm=r['arm'],lambda_value=l,mse=metrics[0],mae=metrics[1],
                    backbone_path=r['backbone_path'],backbone_sha256=sha(r['backbone_path']),memory_path=r['memory_path'],memory_sha256=sha(r['memory_path']),
                    series_sha256=sha(dd/'series.csv'),indices_sha256=sha(dd/'window_indices.npz'),val_choices_path=r['val_choices_path'],val_choices_sha256=sha(r['val_choices_path']),evidence='fresh_run_and_independent_array_recalculation'))
            cache.clear()
    assert len(rows)==672
    df=pd.DataFrame(rows);groups=[]
    for (ds,bb,arm),g in df.groupby(['dataset','backbone','arm']):
        assert sorted(g.seed.tolist())==[42,123,2026]
        base=g.backbone_mse.mean();mean=g.fusion_mse.mean()
        groups.append(dict(dataset=ds,backbone=bb,arm=arm,n_seeds=3,mse_mean=mean,mse_std_ddof1=g.fusion_mse.std(ddof=1),mse_std_ddof0=g.fusion_mse.std(ddof=0),
            mae_mean=g.fusion_mae.mean(),mae_std_ddof1=g.fusion_mae.std(ddof=1),mae_std_ddof0=g.fusion_mae.std(ddof=0),base_mse_mean=base,ri_pct=100*(base-mean)/base))
    for name,data in [('PER_SEED.csv',rows),('SUMMARY.csv',groups),('INDEPENDENT_CHECKS.csv',checks),('RESULT_TRACE.csv',traces)]:writecsv(out/name,data)
    public=repo/'docs/unified/results';public.mkdir(parents=True,exist_ok=True)
    publicrows=[{k:v for k,v in r.items() if not k.endswith('_path')} for r in rows]
    writecsv(public/'PER_SEED.csv',publicrows);writecsv(public/'SUMMARY.csv',groups);writecsv(public/'INDEPENDENT_CHECKS.csv',checks)
    done=dict(protocol_hash=frozen['protocol_hash'],status='PASS',rows=len(rows),groups=len(groups),metric_checks=len(checks),scope='fresh fits/inference plus independent cache, labels, Top5 and VAL-grid verification; no significance or blind-test claim')
    (root/'VERIFIED_COMPLETE.json').write_text(json.dumps(done,indent=2),encoding='utf-8');(out/'VERIFIED_COMPLETE.json').write_text(json.dumps(done,indent=2),encoding='utf-8')
    print(json.dumps(done),flush=True)
if __name__=='__main__':main()
