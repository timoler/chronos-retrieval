# -*- coding: utf-8 -*-
"""Frozen run identity and chronological splits, shared by every backbone."""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import sys,json,hashlib,time,copy,gc
from pathlib import Path
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[2]
CONFIG_PATH=Path(os.environ['PAER_RUN_CONFIG']).resolve()
CONFIG=json.loads(CONFIG_PATH.read_text(encoding='utf-8-sig'))
PLAN=CONFIG['protocol'];ART=Path(CONFIG['run_root']);OUT=Path(CONFIG['report_root'])
ROOT=Path(CONFIG['input_repo']);DATA_ROOT=ROOT/'data/processed'
EXTERNAL=Path(CONFIG['external_root']);EMBED=Path(CONFIG['embedding_snapshot']);CHRONOS=Path(CONFIG['chronos_snapshot'])
MODEL=PLAN['llm_model'];SEEDS=PLAN['seeds'];DSLIST=PLAN['datasets'];GRID=PLAN['lambda_grid'];SLOTS=8

def dump(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
    # Windows readers/security scanners may briefly deny atomic replacement.
    # Keep the old valid file and retry; permanent errors still stop the run.
    for attempt in range(12):
        try:
            tmp.replace(p)
            return
        except PermissionError:
            if attempt==11:raise
            time.sleep(min(.05*2**attempt,.5))
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):h.update(c)
    return h.hexdigest()
def mse(p,y):return float(np.mean((np.asarray(p,np.float64)-np.asarray(y,np.float64))**2))
def mae(p,y):return float(np.mean(np.abs(np.asarray(p,np.float64)-np.asarray(y,np.float64))))
def select(c,m,y):
    if np.array_equal(c,m):return 0.,[mse(c,y)]*len(GRID)
    values=[mse((1-l)*c+l*m,y) for l in GRID]
    return GRID[int(np.argmin(values))],values
def available(earlier,later,h):
    earlier=np.asarray(earlier,dtype=np.int64);later=np.asarray(later,dtype=np.int64)
    assert len(earlier) and len(later)
    a=earlier[earlier+h<=later[0]]
    assert len(a) and a[-1]+h<=later[0]
    return a
def data(ds):
    dd=DATA_ROOT/ds;cfg=read(dd/'config.json');df=pd.read_csv(dd/'series.csv')
    with np.load(dd/'window_indices.npz') as z:starts={k:z[k].astype(np.int64) for k in ['train','val','test']}
    return dict(ds=ds,dd=dd,cfg=cfg,series=df.OT.to_numpy(np.float32),dates=pd.to_datetime(df.date),starts=starts,L=int(cfg['context']),H=int(cfg['horizon']))
def fold_spec(d,fold):
    t=d['starts']['train'];h=d['H'];edges=np.linspace(len(t)//2,len(t),5).astype(int)
    prefix=t[:edges[fold-1]];block=t[edges[fold-1]:edges[fold]];mid=len(block)//2
    ev=block[mid::h];cal=available(block[:mid],ev,h)[::h]
    prefix=available(prefix,cal,h);cut=int(len(prefix)*.8)
    fit=prefix[cut:];bank=available(prefix[:cut],fit,h)
    return dict(bank=bank,fit=fit,cal=cal,eval=ev)
def final_spec(d):
    train=available(d['starts']['train'],d['starts']['val'],d['H']);cut=int(len(train)*.8)
    return dict(bank=available(train[:cut],train[cut:],d['H']),fit=train[cut:],val=d['starts']['val'],test=d['starts']['test'])
def windows(series,starts,l,h):
    x=np.stack([series[s:s+l] for s in starts]);y=np.stack([series[s+l:s+l+h] for s in starts])
    return x.astype(np.float32),y.astype(np.float32)
def identity(d):
    return {n:sha(d['dd']/n) for n in ['series.csv','window_indices.npz','config.json']}
def assert_frozen():
    assert read(ART/'protocol.json')==PLAN,'Protocol changed: allocate a new run directory'
    for rel,value in read(ART/'code_manifest.json').items():
        assert sha(BASE/rel)==value,'Frozen code changed: '+rel
    for rel,value in read(ART/'input_bytes_manifest.json').items():
        assert sha(ROOT/rel)==value,'Frozen input changed: '+rel
    for rel,value in read(ART/'external_code_manifest.json').items():
        assert sha(EXTERNAL/rel)==value,'Frozen external architecture changed: '+rel
def complete(path,extra):
    dump(Path(path)/'completed.json',dict(protocol_hash=digest(PLAN),finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**extra))

