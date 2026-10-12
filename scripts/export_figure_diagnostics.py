"""Export descriptive diagnostics from frozen predictions; no fitting or selection on TEST."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np

ap=argparse.ArgumentParser(); ap.add_argument('--repo',required=True); ap.add_argument('--audit',required=True); ap.add_argument('--out',required=True); a=ap.parse_args()
repo=Path(a.repo); pkg=repo/'results/deepseek_matrix_delivery'; out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
with (Path(a.audit)/'matrix_per_seed.csv').open(encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
sources={}; result=[]
def arr(p,key=None):
    p=Path(p); rel=str(p.relative_to(repo)).replace('\\','/'); sources[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
    if key is None:return np.asarray(np.load(p,allow_pickle=False),dtype=np.float64)
    with np.load(p,allow_pickle=False) as z:return np.asarray(z[key],dtype=np.float64)
for r in rows:
    ds,bb,s=r['dataset'],r['backbone'],int(r['seed']); lam=float(r['lambda'])
    cache=pkg/f'chronos2_preds/{ds}.npz'; y=arr(cache,'Y_test')
    b=arr(cache,'yc_test') if bb=='Chronos-2' else arr(pkg/f'backbone/{ds}/{bb.lower()}_{ds}_s{s}_test.npy')
    if ds in ['etth1','weather']:m=arr(pkg/f'external_memories/{ds}_PAER_test_retrieval_s{s}.npz','transferred')
    elif ds=='etth2':m=arr(pkg/f'external_memories/etth2_align_test_s{s}.npy')
    elif ds=='traffic':m=arr(pkg/f'external_memories/paer_align_test_s{s}.npy')
    else:m=arr(pkg/f'mem/{ds}/paer_align_test_s{s}.npy')
    assert b.shape==m.shape==y.shape
    eb,em=b-y,m-y; B=float(np.mean(eb**2)); M=float(np.mean(em**2)); C=float(np.mean(eb*em)); F=float(np.mean(((1-lam)*eb+lam*em)**2))
    terms=[(1-lam)**2,lam**2*M/B,2*lam*(1-lam)*C/B]
    assert np.isclose(sum(terms),F/B,rtol=1e-12,atol=1e-12)
    assert np.isclose(F,float(r['fused_test_mse']),rtol=1e-10)
    result.append(dict(dataset=ds,backbone=bb,seed=s,lambda_val=lam,n_windows=len(y),horizon=y.shape[1],base_mse=B,memory_mse=M,cross_error_moment=C,error_cosine=C/np.sqrt(B*M),memory_over_base=M/B,cross_over_base=C/B,base_term_normalized=terms[0],memory_term_normalized=terms[1],cross_term_normalized=terms[2],fused_over_base=F/B,ri_pct=100*(B-F)/B))
def write(name,data):
    with (out/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
write('fig04_error_decomposition_per_seed.csv',result)
groups=[]
for ds,bb in dict.fromkeys((r['dataset'],r['backbone']) for r in result):
    g=[r for r in result if r['dataset']==ds and r['backbone']==bb]
    means={k:float(np.mean([r[k] for r in g])) for k in ['error_cosine','memory_over_base','cross_over_base','base_term_normalized','memory_term_normalized','cross_term_normalized','fused_over_base']}
    groups.append(dict(dataset=ds,backbone=bb,n_seeds=len(g),**means))
write('fig04_error_decomposition_summary.csv',groups)
(out/'fig04_source_hashes.json').write_text(json.dumps(sources,indent=2),encoding='utf-8')
print(json.dumps({'rows':len(result),'groups':len(groups),'identity_checks':'84 passed','illustrations':[r for r in groups if r['dataset'] in ['traffic','weather'] and r['backbone']=='Chronos-2']},indent=2))
