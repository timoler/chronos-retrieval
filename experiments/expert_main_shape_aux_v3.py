#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V3 — Expert-main Retrieval, Shape as Auxiliary Evidence

1) Shape-only: score = shape_score
2) Expert-only: score = gamma * MLP([e_q,e_i,|e_q-e_i|])
3) Expert-main + Shape-aux: score = gamma*MLP(...) + beta*shape_score

Supports expert feature ablation via --expert-cols and shuffled control via --shuffle-expert.
Default feature order:
0 hour_sin, 1 hour_cos, 2 doy_sin, 3 doy_cos, 4 recent_level, 5 recent_change
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

FEATURE_NAMES=["hour_sin","hour_cos","doy_sin","doy_cos","recent_level","recent_change"]

def mse(y,p):
    y=np.asarray(y,dtype=np.float64); p=np.asarray(p,dtype=np.float64)
    return float(np.mean((y-p)**2))

def mae(y,p):
    y=np.asarray(y,dtype=np.float64); p=np.asarray(p,dtype=np.float64)
    return float(np.mean(np.abs(y-p)))

def improve(old,new):
    return float((old-new)/old*100.0) if old!=0 else float('nan')

def parse_cols(text,total_dim):
    if not text: return list(range(total_dim))
    cols=[int(x.strip()) for x in text.split(',')]
    if min(cols)<0 or max(cols)>=total_dim: raise ValueError(f'expert-cols 越界: {cols}, dim={total_dim}')
    return cols

def fit_scaler(*xs):
    x=np.concatenate(xs,axis=0).astype(np.float32)
    mu=x.mean(0,keepdims=True); sd=x.std(0,keepdims=True); sd[sd<1e-6]=1.0
    return mu,sd

def scale(x,mu,sd): return ((x.astype(np.float32)-mu)/sd).astype(np.float32)

def target_series(ctx):
    ctx=np.asarray(ctx,dtype=np.float32)
    if ctx.ndim==2: return ctx
    if ctx.ndim==3: return ctx[...,-1]
    raise ValueError(ctx.shape)

def context_mean(ctx): return target_series(ctx).mean(axis=1)

class ExpertMLP(nn.Module):
    def __init__(self,dim,hidden=32):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim*3,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))
        self.logit_gamma=nn.Parameter(torch.tensor(-2.0))
    def delta(self,q,c):
        B,M=q.shape; K=c.shape[0]
        qq=q[:,None,:].expand(B,K,M); cc=c[None,:,:].expand(B,K,M)
        return self.net(torch.cat([qq,cc,torch.abs(qq-cc)],dim=-1)).squeeze(-1)
    def score(self,q,c): return torch.sigmoid(self.logit_gamma)*self.delta(q,c)

class ExpertMainShapeAux(nn.Module):
    def __init__(self,dim,hidden=32,beta_init=0.2):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim*3,hidden),nn.ReLU(),nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,1))
        self.logit_gamma=nn.Parameter(torch.tensor(-2.0))
        beta_init=float(np.clip(beta_init,1e-4,1-1e-4))
        self.logit_beta=nn.Parameter(torch.tensor(np.log(beta_init/(1-beta_init)),dtype=torch.float32))
    def delta(self,q,c):
        B,M=q.shape; K=c.shape[0]
        qq=q[:,None,:].expand(B,K,M); cc=c[None,:,:].expand(B,K,M)
        return self.net(torch.cat([qq,cc,torch.abs(qq-cc)],dim=-1)).squeeze(-1)
    def score(self,shape,q,c):
        return torch.sigmoid(self.logit_gamma)*self.delta(q,c)+torch.sigmoid(self.logit_beta)*shape

def train(model,mode,b_shape,a_e,b_e,a_ctx,a_fut,b_ctx,b_fut,epochs,batch_size,candidate_size,lr,temp,device,seed):
    rng=np.random.default_rng(seed); opt=torch.optim.Adam(model.parameters(),lr=lr)
    n_q,n_mem=b_shape.shape; a_t=torch.from_numpy(a_e).to(device); b_t=torch.from_numpy(b_e).to(device)
    hist=[]
    for ep in range(1,epochs+1):
        order=rng.permutation(n_q); losses=[]
        for st in range(0,n_q,batch_size):
            qidx=order[st:st+batch_size]; cidx=rng.choice(n_mem,size=min(candidate_size,n_mem),replace=False)
            q=b_t[qidx]; c=a_t[cidx]
            shape=torch.from_numpy(b_shape[np.ix_(qidx,cidx)].astype(np.float32)).to(device)
            s=model.score(q,c) if mode=='expert' else model.score(shape,q,c)
            w=torch.softmax(s/temp,dim=1)
            q_mu=context_mean(b_ctx[qidx]).astype(np.float32); c_mu=context_mean(a_ctx[cidx]).astype(np.float32)
            transferred=a_fut[cidx][None,:,:].astype(np.float32)+q_mu[:,None,None]-c_mu[None,:,None]
            pred=torch.sum(w.unsqueeze(-1)*torch.from_numpy(transferred).to(device),dim=1)
            y=torch.from_numpy(b_fut[qidx].astype(np.float32)).to(device)
            loss=F.mse_loss(pred,y)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
            losses.append(float(loss.detach().cpu()))
        row={'epoch':ep,'loss':float(np.mean(losses)),'gamma':float(torch.sigmoid(model.logit_gamma).detach().cpu())}
        if hasattr(model,'logit_beta'): row['beta']=float(torch.sigmoid(model.logit_beta).detach().cpu())
        hist.append(row)
        if ep==1 or ep==epochs or ep%max(1,epochs//10)==0:
            extra=f" beta={row['beta']:.4f}" if 'beta' in row else ''
            print(f"[{mode}] epoch={ep:03d} loss={row['loss']:.6f} gamma={row['gamma']:.4f}{extra}")
    return hist

@torch.no_grad()
def build_scores(model,mode,base_shape,q_e,m_e,qb,cb,device):
    model.eval(); N,K=base_shape.shape; out=np.empty((N,K),dtype=np.float32)
    qt=torch.from_numpy(q_e.astype(np.float32)); mt=torch.from_numpy(m_e.astype(np.float32))
    for qs in range(0,N,qb):
        qe=min(qs+qb,N); q=qt[qs:qe].to(device)
        for cs in range(0,K,cb):
            ce=min(cs+cb,K); c=mt[cs:ce].to(device)
            shape=torch.from_numpy(base_shape[qs:qe,cs:ce].astype(np.float32)).to(device)
            s=model.score(q,c) if mode=='expert' else model.score(shape,q,c)
            out[qs:qe,cs:ce]=s.cpu().numpy()
    info={'gamma':float(torch.sigmoid(model.logit_gamma).cpu())}
    if hasattr(model,'logit_beta'): info['beta']=float(torch.sigmoid(model.logit_beta).cpu())
    return out,info

def softmax_np(x):
    x=x-np.max(x,axis=1,keepdims=True); e=np.exp(x); return e/np.sum(e,axis=1,keepdims=True)

def retrieve(scores,mem_ctx,mem_fut,q_ctx,topk,temp,batch_size):
    N,K=scores.shape; H=mem_fut.shape[1]; pred=np.empty((N,H),dtype=np.float32)
    for st in range(0,N,batch_size):
        ed=min(st+batch_size,N); s=scores[st:ed]; k=min(topk,K)
        idx=np.argpartition(-s,kth=k-1,axis=1)[:,:k]; row=np.arange(ed-st)[:,None]; ss=s[row,idx]
        order=np.argsort(-ss,axis=1); idx=idx[row,order]; ss=ss[row,order]; w=softmax_np(ss/temp)
        q_mu=context_mean(q_ctx[st:ed])
        for r in range(ed-st):
            ids=idx[r]; c_mu=context_mean(mem_ctx[ids]); transferred=mem_fut[ids].astype(np.float32)+q_mu[r]-c_mu[:,None]
            pred[st+r]=np.sum(w[r,:,None]*transferred,axis=0)
    return pred

def topk_idx(scores,k):
    k=min(k,scores.shape[1]); return np.argpartition(-scores,kth=k-1,axis=1)[:,:k]

def overlap(a,b):
    return float(np.mean([len(set(a[i].tolist()) & set(b[i].tolist()))/len(a[i]) for i in range(len(a))]))

def best_lambda(y,chronos,memory,lambdas):
    rows=[]
    for lam in lambdas:
        p=(1-lam)*chronos+lam*memory; rows.append({'lambda':float(lam),'mse':mse(y,p),'mae':mae(y,p)})
    return rows,min(rows,key=lambda x:(x['mse'],x['lambda']))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--bundle',type=Path,required=True)
    ap.add_argument('--out-dir',type=Path,default=Path('results/expert_main_shape_aux_v3'))
    ap.add_argument('--expert-cols',type=str,default=None)
    ap.add_argument('--shuffle-expert',action='store_true')
    ap.add_argument('--epochs',type=int,default=30); ap.add_argument('--hidden',type=int,default=32)
    ap.add_argument('--batch-size',type=int,default=32); ap.add_argument('--candidate-size',type=int,default=256)
    ap.add_argument('--lr',type=float,default=1e-3); ap.add_argument('--topk',type=int,default=5)
    ap.add_argument('--softmax-temperature',type=float,default=1.0); ap.add_argument('--beta-init',type=float,default=0.2)
    ap.add_argument('--score-query-batch',type=int,default=32); ap.add_argument('--score-candidate-batch',type=int,default=2048)
    ap.add_argument('--seed',type=int,default=42); ap.add_argument('--device',type=str,default='auto')
    ap.add_argument('--lambdas',nargs='+',type=float,default=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6])
    args=ap.parse_args()
    if 'test' in str(args.bundle).lower(): raise RuntimeError('当前脚本禁止访问 TEST bundle。')
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device))
    print('device:',device)
    with np.load(args.bundle,allow_pickle=False) as f: A={k:f[k] for k in f.files}
    req=['train_a_context','train_a_future','train_a_expert','train_b_context','train_b_future','train_b_expert','val_context','val_future','val_expert','train_b_shape_scores','val_shape_scores']
    miss=[k for k in req if k not in A]
    if miss: raise KeyError(f'bundle 缺字段: {miss}')
    a_ctx=A['train_a_context']; a_fut=A['train_a_future'].astype(np.float32); a_exp=A['train_a_expert']
    b_ctx=A['train_b_context']; b_fut=A['train_b_future'].astype(np.float32); b_exp=A['train_b_expert']
    v_ctx=A['val_context']; v_fut=A['val_future'].astype(np.float32); v_exp=A['val_expert']
    b_shape=A['train_b_shape_scores'].astype(np.float32); v_shape=A['val_shape_scores'].astype(np.float32)
    cols=parse_cols(args.expert_cols,a_exp.shape[1]); names=[FEATURE_NAMES[c] if c<len(FEATURE_NAMES) else f'feature_{c}' for c in cols]
    a_exp=a_exp[:,cols]; b_exp=b_exp[:,cols]; v_exp=v_exp[:,cols]
    if args.shuffle_expert:
        rng=np.random.default_rng(args.seed+999); a_exp=a_exp[rng.permutation(len(a_exp))]; b_exp=b_exp[rng.permutation(len(b_exp))]; v_exp=v_exp[rng.permutation(len(v_exp))]
        print('shuffled expert control ON')
    mu,sd=fit_scaler(a_exp,b_exp); a_e=scale(a_exp,mu,sd); b_e=scale(b_exp,mu,sd); v_e=scale(v_exp,mu,sd)
    mem_ctx=np.concatenate([a_ctx,b_ctx],axis=0); mem_fut=np.concatenate([a_fut,b_fut],axis=0); mem_e=np.concatenate([a_e,b_e],axis=0)
    torch.manual_seed(args.seed)
    expert_model=ExpertMLP(len(cols),args.hidden).to(device); hybrid_model=ExpertMainShapeAux(len(cols),args.hidden,args.beta_init).to(device)
    eh=train(expert_model,'expert',b_shape,a_e,b_e,a_ctx,a_fut,b_ctx,b_fut,args.epochs,args.batch_size,args.candidate_size,args.lr,args.softmax_temperature,device,args.seed)
    hh=train(hybrid_model,'hybrid',b_shape,a_e,b_e,a_ctx,a_fut,b_ctx,b_fut,args.epochs,args.batch_size,args.candidate_size,args.lr,args.softmax_temperature,device,args.seed+1)
    e_scores,e_info=build_scores(expert_model,'expert',v_shape,v_e,mem_e,args.score_query_batch,args.score_candidate_batch,device)
    h_scores,h_info=build_scores(hybrid_model,'hybrid',v_shape,v_e,mem_e,args.score_query_batch,args.score_candidate_batch,device)
    s_pred=retrieve(v_shape,mem_ctx,mem_fut,v_ctx,args.topk,args.softmax_temperature,args.batch_size)
    e_pred=retrieve(e_scores,mem_ctx,mem_fut,v_ctx,args.topk,args.softmax_temperature,args.batch_size)
    h_pred=retrieve(h_scores,mem_ctx,mem_fut,v_ctx,args.topk,args.softmax_temperature,args.batch_size)
    sm,em,hm=mse(v_fut,s_pred),mse(v_fut,e_pred),mse(v_fut,h_pred); sma,ema,hma=mae(v_fut,s_pred),mae(v_fut,e_pred),mae(v_fut,h_pred)
    si,ei,hi=topk_idx(v_shape,args.topk),topk_idx(e_scores,args.topk),topk_idx(h_scores,args.topk)
    report={'experiment':'expert_main_shape_aux_v3','test_accessed':False,'expert_columns':cols,'expert_feature_names':names,'shuffle_expert':bool(args.shuffle_expert),'memory_only':{'shape':{'mse':sm,'mae':sma},'expert_only':{'mse':em,'mae':ema,**e_info},'expert_main_shape_aux':{'mse':hm,'mae':hma,**h_info},'expert_vs_shape_pct':improve(sm,em),'hybrid_vs_expert_pct':improve(em,hm)},'topk_overlap':{'shape_vs_expert':overlap(si,ei),'shape_vs_hybrid':overlap(si,hi),'expert_vs_hybrid':overlap(ei,hi)},'training':{'expert_only':eh,'hybrid':hh}}
    if 'chronos_val_pred' in A:
        c=A['chronos_val_pred'].astype(np.float32); cm=mse(v_fut,c); sg,sb=best_lambda(v_fut,c,s_pred,args.lambdas); eg,eb=best_lambda(v_fut,c,e_pred,args.lambdas); hg,hb=best_lambda(v_fut,c,h_pred,args.lambdas)
        report['chronos_fusion']={'chronos':{'mse':cm},'shape':{'grid':sg,'best':sb},'expert_only':{'grid':eg,'best':eb},'expert_main_shape_aux':{'grid':hg,'best':hb,'vs_expert_pct':improve(eb['mse'],hb['mse']),'vs_shape_pct':improve(sb['mse'],hb['mse'])}}
    args.out_dir.mkdir(parents=True,exist_ok=True)
    torch.save({'expert_only_state':expert_model.state_dict(),'hybrid_state':hybrid_model.state_dict(),'expert_mu':mu,'expert_sd':sd,'expert_columns':cols,'expert_feature_names':names},args.out_dir/'models.pt')
    np.savez_compressed(args.out_dir/'val_predictions.npz',y_true=v_fut,shape_memory_pred=s_pred,expert_memory_pred=e_pred,expert_main_shape_aux_memory_pred=h_pred)
    (args.out_dir/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('='*80); print('V3 Expert-main / Shape-aux'); print('features:',names); print('shuffle:',args.shuffle_expert)
    print(f'Shape Memory        {sm:.6f}'); print(f'Expert-only Memory  {em:.6f}'); print(f'Expert+Shape Memory {hm:.6f}'); print(f'Hybrid vs Expert    {improve(em,hm):+.4f}%')
    print(f'Expert gamma        {e_info["gamma"]:.4f}'); print(f'Hybrid gamma        {h_info["gamma"]:.4f}'); print(f'Shape beta          {h_info["beta"]:.4f}'); print(f'TopK Expert/Hybrid  {report["topk_overlap"]["expert_vs_hybrid"]*100:.2f}%')
    if 'chronos_fusion' in report:
        cf=report['chronos_fusion']; print(f'Chronos             {cf["chronos"]["mse"]:.6f}'); print(f'Best + Shape        {cf["shape"]["best"]["mse"]:.6f}'); print(f'Best + Expert       {cf["expert_only"]["best"]["mse"]:.6f}'); print(f'Best + Expert+Shape {cf["expert_main_shape_aux"]["best"]["mse"]:.6f}')
    print('outputs:',args.out_dir); print('='*80)

if __name__=='__main__': main()
