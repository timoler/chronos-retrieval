# -*- coding: utf-8 -*-
"""CPU-only extraction equivalence checks. This does not run any experiment."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import ast
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time
sys.dont_write_bytecode = True

REPOSITORY = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference-root', type=Path, required=True)
parser.add_argument('--work-dir', type=Path, required=True)
arguments = parser.parse_args()
SOURCE = arguments.reference_root.resolve()
HERE = arguments.work_dir.resolve()
if HERE.exists():
    raise RuntimeError('Verification work directory must be new')
if SOURCE in HERE.parents or REPOSITORY in HERE.parents or HERE in SOURCE.parents or HERE in REPOSITORY.parents:
    raise RuntimeError('Verification output must be separate from both source checkouts')
HERE.mkdir(parents=True)
REFERENCE = HERE / 'reference_sources'
DERIVATION = json.loads((REPOSITORY/'paer_core/DERIVATION.json').read_text(encoding='utf-8'))
checks = []

def check(name, success, **extra):
    if not success:
        raise AssertionError(name)
    checks.append(dict(check=name,status='PASS',**extra))

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

# Import-time output-directory creation is confined to this scratch copy.
legacy = sorted(set(r['source'] for r in DERIVATION['definitions']))
for p in (SOURCE/'priors').glob('*.py'):
    legacy.append(p.relative_to(SOURCE).as_posix())
for p in (SOURCE/'retrieval').glob('*.py'):
    legacy.append(p.relative_to(SOURCE).as_posix())
for rel in sorted(set(legacy)):
    target = REFERENCE/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(SOURCE/rel,target)
sys.path[:0] = [str(REPOSITORY),str(REFERENCE/'experiments/external_baselines'),str(REFERENCE/'experiments'),str(REFERENCE)]

for rec in DERIVATION['definitions']:
    check('source_hash:'+rec['source'],digest(SOURCE/rec['source'])==rec['source_sha256'])
    target=ast.parse((REPOSITORY/rec['output']).read_text(encoding='utf-8'))
    candidates=list(ast.walk(target))
    match=[n for n in candidates if hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()==rec['definition_ast_sha256']]
    check('definition_ast:'+rec['source']+':'+str(rec['start_line']),bool(match))

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
check('cuda_disabled',not torch.cuda.is_available())
import run_ours_fusion_v2 as old
from paer_core import forecasting as new
from paer_core import backbone_config
from paer_core import json_parser
old_v1=old.v1;new_v1=new.v1
old_train=old.v2p;new_train=new.v2p
device=torch.device('cpu')
rng=np.random.default_rng(314159)
x=rng.normal(size=(24,64)).astype(np.float32)
dates=pd.Series(pd.date_range('2024-01-01',periods=len(x),freq='h'))

check('feature_registry',old_v1.FEATURE_REGISTRY==new_v1.FEATURE_REGISTRY)
check('compiler_config',old_v1.COMPILER_CONFIG==new_v1.COMPILER_CONFIG)
stats_old=old.compute_training_stats(x);stats_new=new.compute_training_stats(x)
check('training_statistics',stats_old==stats_new)
eo={'feature_distributions':old_v1._raw_feature_quantiles(x),'temporal_coverage':old_v1._temporal_coverage(dates)}
en={'feature_distributions':new_v1._raw_feature_quantiles(x),'temporal_coverage':new_v1._temporal_coverage(dates)}
check('compiler_train_evidence',eo==en)
check('catalog_prompt',old_v1._catalog_text()==new_v1._catalog_text())
check('schema_prompt',old_v1._expert_schema_text()==new_v1._expert_schema_text())

for ds in ['etth1','etth2','traffic','exchange_rate','weather','solar','electricity']:
    path=SOURCE/'results/reward_feedback_critic_v2_full_cross_dataset'/ds/'raw_experts.json'
    pool=json.loads(path.read_text(encoding='utf-8'))
    check('expert_order:'+ds,old_v1._sorted_experts(pool)==new_v1._sorted_experts(pool))
    po=[];pn=[]
    for expert in pool:
        co=old_v1.compile_condition_v6(expert,old_v1.FEATURE_REGISTRY,eo,old_v1.COMPILER_CONFIG)
        cn=new_v1.compile_condition_v6(expert,new_v1.FEATURE_REGISTRY,en,new_v1.COMPILER_CONFIG)
        check('compiled:'+ds+':'+expert['id'],co==cn)
        check('retrieval_spec:'+ds+':'+expert['id'],old_v1.derive_retrieval_spec(expert,old_v1.FEATURE_REGISTRY)==new_v1.derive_retrieval_spec(expert,new_v1.FEATURE_REGISTRY))
        po.append(dict(expert,compiled=co));pn.append(dict(expert,compiled=cn))
    ao=old_v1.build_activation_matrix(po,x,list(dates),stats_old)
    an=new_v1.build_activation_matrix(pn,x,list(dates),stats_new)
    check('activations:'+ds,np.array_equal(ao,an),max_abs_error=float(np.max(np.abs(ao-an))))

for feature,qualifier in [('recent_level','high'),('recent_level','large'),('recent_volatility','low'),
                          ('hour_of_day','night'),('recent_level','medium'),('future_target','high'),
                          ('hour_of_day','undefined')]:
    expert={'condition_spec':{'conditions':[{'feature':feature,'qualifier':qualifier}]}}
    check('compiler_edge:'+feature+':'+qualifier,
          old_v1.compile_condition_v6(expert,old_v1.FEATURE_REGISTRY,eo,old_v1.COMPILER_CONFIG)==
          new_v1.compile_condition_v6(expert,new_v1.FEATURE_REGISTRY,en,new_v1.COMPILER_CONFIG))

check('residual_shape',np.array_equal(old.residualize(x),new.residualize(x)))
series=np.arange(200,dtype=np.float32)**1.1
wo=old.build_windows(series,[0,5,30],64,8);wn=new.build_windows(series,[0,5,30],64,8)
check('windows',all(np.array_equal(a,b) for a,b in zip(wo,wn)))

emb=rng.normal(size=(8,384)).astype(np.float32)
act=rng.uniform(size=(24,8)).astype(np.float32)
with torch.no_grad():
    for seed in [42,123,2026]:
        so=old._init_state(seed,64,384,8,device);sn=new._init_state(seed,64,384,8,device)
        check('state_dict:'+str(seed),so.keys()==sn.keys() and all(torch.equal(so[k],sn[k]) for k in so))
        mo=old.ConditionalExpertMLPRetriever(64,384,8);mn=new.ConditionalExpertMLPRetriever(64,384,8)
        mo.load_state_dict(so);mn.load_state_dict(sn)
        mo.set_expert_embeddings(emb);mn.set_expert_embeddings(emb)
        z_old=mo.encode(torch.from_numpy(x),torch.from_numpy(act));z_new=mn.encode(torch.from_numpy(x),torch.from_numpy(act))
        check('encoder_and_expert_mlp:'+str(seed),torch.equal(z_old,z_new))
        do=mo.delta(z_old[:5],z_old[5:]);dn=mn.delta(z_new[:5],z_new[5:])
        check('pair_scores:'+str(seed),torch.equal(do,dn))
        shape=torch.from_numpy(old.residualize(x[:5])@old.residualize(x[5:]).T/64)
        for mode in ['full','shape','expert']:
            check('fused_scores:'+str(seed)+':'+mode,torch.equal(mo.combine(do,shape,mode),mn.combine(dn,shape,mode)))

# Two optimizer steps on a tiny synthetic fixture; not full training or a dataset result.
yo=rng.normal(size=(24,8)).astype(np.float32)
old_train.EPOCHS=2;new_train.EPOCHS=2
for model,training in [(mo,old_train),(mn,new_train)]:
    model.load_state_dict(so);model.set_expert_embeddings(emb)
    training.train_arm(model,x[5:],yo[5:],x[:5],yo[:5],old.residualize(x[5:]),old.residualize(x[:5]),act[5:],act[:5],True,device,42)
check('two_cpu_optimizer_steps',all(torch.equal(mo.state_dict()[k],mn.state_dict()[k]) for k in mo.state_dict()))

spec=importlib.util.spec_from_file_location('fixture_retrieval',SOURCE/'experiments/unified/unified_retrieval.py')
retrieval=importlib.util.module_from_spec(spec);spec.loader.exec_module(retrieval)
for mode in ['full','shape','expert']:
    ao=retrieval.retrieve_capture(mo,x[:5],x[5:],yo[5:],old.residualize(x[:5]),old.residualize(x[5:]),act[:5],act[5:],mode,5,device)
    an=retrieval.retrieve_capture(mn,x[:5],x[5:],yo[5:],new.residualize(x[:5]),new.residualize(x[5:]),act[:5],act[5:],mode,5,device)
    check('top5_and_predictions:'+mode,all(np.array_equal(a,b) for a,b in zip(ao,an)))

for coverage in [None,0.,0.005,0.01,0.7]:
    for pfr,mean,median in [(1.,.2,.1),(.75,-.1,.1),(.5,.1,-.1),(.25,-.1,-.1),(0.,.1,.1)]:
        st=dict(positive_fold_ratio=pfr,mean_pool_reward=mean,median_pool_reward=median)
        check('reward_rule:'+repr((coverage,pfr,mean,median)),old_train.v2_reward_rule(st,coverage)==new_train.v2_reward_rule(st,coverage))

def outcome(fn,*args):
    try:return ('OK',fn(*args))
    except Exception as exc:return (type(exc).__name__,str(exc))
for text in ['{"experts":[{"id":"E1"}]}','```json\n{"experts":[]}\n```','[{"id":"E1"}]',
             'prefix {"experts":[1,2]} trailing','{"experts":[1,2','invalid']:
    check('json_parser:'+text,outcome(old_train.av2._list_from,text,'experts')==outcome(new_train.av2._list_from,text,'experts'))

for bb,oldfile,batch in [('PatchTST','infer_patchtst_matrix.py',128),('iTransformer','infer_itransformer_matrix.py',32),('RAFT','infer_raft_matrix.py',32)]:
    t=ast.parse((SOURCE/'experiments/external_baselines'/oldfile).read_text(encoding='utf-8-sig'))
    node=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='build_args')
    ns={'argparse':argparse,'BATCH':batch};exec(compile(ast.Module(body=[node],type_ignores=[]),oldfile,'exec'),ns)
    for context,horizon in [(64,8),(512,96),(96,24)]:
        check('backbone_config:'+bb+':'+str((context,horizon)),vars(ns['build_args'](context,horizon))==vars(backbone_config.build_args(bb,context,horizon)))

class FakeChronos:
    def predict_quantiles(self,inputs,prediction_length,batch_size):
        check('chronos_batch',batch_size==256)
        return None,[torch.full((1,prediction_length),float(v.mean())) for v in inputs]
check('chronos_prediction_adapter',np.array_equal(old.run_chronos(FakeChronos(),x,8),new.run_chronos(FakeChronos(),x,8)))
new.FROZEN_ROOT=SOURCE/'results/reward_feedback_critic_v2_full_cross_dataset'
for ds in ['etth1','etth2','traffic','exchange_rate','weather','solar','electricity']:
    check('raw8_io_adapter:'+ds,new.load_raw_pool(ds)==json.loads((new.FROZEN_ROOT/ds/'raw_experts.json').read_text(encoding='utf-8')))

result=dict(status='PASS',checks=checks,count=len(checks),device='cpu',torch_version=torch.__version__,
            gpu_used=False,full_training_performed=False,experiment_results_generated=False,
            checked_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            limitations=['Synthetic CPU equivalence does not constitute a new full experiment or independent GPU reproduction.',
                         'The frozen run snapshot remains unchanged; packaging relocation has its own source derivation.'])
(HERE/'EQUIVALENCE_CHECKS.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False))
