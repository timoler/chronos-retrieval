#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V2 Ablation: Shape vs Expert-only vs Shape+Expert

公平比较：
1) Shape-only:
   score = score_shape

2) Expert-only:
   score = gamma_e * MLP([e_q, e_i, |e_q-e_i|])

3) Shape+Expert:
   score = score_shape + gamma_h * MLP([e_q, e_i, |e_q-e_i|])

约束：
- TRAIN_A = memory
- TRAIN_B = 训练 MLP
- VAL = 评估
- TEST 完全不访问
- Shape score 直接读取冻结原管线导出的 train_b_shape_scores / val_shape_scores
- Top-K / softmax / mean-anchor 一致
"""

from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def mse(y_true, y_pred):
    y_true=np.asarray(y_true,dtype=np.float64); y_pred=np.asarray(y_pred,dtype=np.float64)
    return float(np.mean((y_true-y_pred)**2))

def mae(y_true, y_pred):
    y_true=np.asarray(y_true,dtype=np.float64); y_pred=np.asarray(y_pred,dtype=np.float64)
    return float(np.mean(np.abs(y_true-y_pred)))

def improve(old,new):
    return float((old-new)/old*100.0) if old!=0 else float("nan")

def fit_scaler(*xs):
    x=np.concatenate(xs,axis=0).astype(np.float32)
    mu=x.mean(axis=0,keepdims=True); sd=x.std(axis=0,keepdims=True)
    sd[sd<1e-6]=1.0
    return mu,sd

def apply_scaler(x,mu,sd):
    return ((x.astype(np.float32)-mu)/sd).astype(np.float32)

def target_series(ctx):
    ctx=np.asarray(ctx,dtype=np.float32)
    if ctx.ndim==2: return ctx
    if ctx.ndim==3: return ctx[...,-1]
    raise ValueError(f"Unsupported context shape: {ctx.shape}")

def context_mean(ctx):
    return target_series(ctx).mean(axis=1)

class ExpertScoreMLP(nn.Module):
    def __init__(self, expert_dim, hidden=32):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(expert_dim*3,hidden),
            nn.ReLU(),
            nn.Linear(hidden,hidden),
            nn.ReLU(),
            nn.Linear(hidden,1),
        )
        self.logit_gamma=nn.Parameter(torch.tensor(-2.0))

    def delta(self,q_e,c_e):
        B,M=q_e.shape; K=c_e.shape[0]
        q=q_e[:,None,:].expand(B,K,M)
        c=c_e[None,:,:].expand(B,K,M)
        x=torch.cat([q,c,torch.abs(q-c)],dim=-1)
        return self.net(x).squeeze(-1)

    def expert_score(self,q_e,c_e):
        d=self.delta(q_e,c_e)
        gamma=torch.sigmoid(self.logit_gamma)
        return gamma*d,gamma

    def hybrid_score(self,shape_score,q_e,c_e):
        d=self.delta(q_e,c_e)
        gamma=torch.sigmoid(self.logit_gamma)
        return shape_score+gamma*d,gamma

def train_model(model,mode,train_b_shape_scores,train_a_expert,train_b_expert,
                train_a_context,train_a_future,train_b_context,train_b_future,
                epochs,batch_size,candidate_size,lr,temp,device,seed):
    rng=np.random.default_rng(seed)
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    n_q,n_mem=train_b_shape_scores.shape
    a_e=torch.from_numpy(train_a_expert).to(device)
    b_e=torch.from_numpy(train_b_expert).to(device)
    hist=[]
    for epoch in range(1,epochs+1):
        order=rng.permutation(n_q); losses=[]
        for st in range(0,n_q,batch_size):
            q_idx=order[st:st+batch_size]
            if len(q_idx)==0: continue
            k=min(candidate_size,n_mem)
            c_idx=rng.choice(n_mem,size=k,replace=False)
            q=b_e[q_idx]; c=a_e[c_idx]
            base=torch.from_numpy(train_b_shape_scores[np.ix_(q_idx,c_idx)].astype(np.float32)).to(device)
            if mode=="expert_only":
                score,gamma=model.expert_score(q,c)
            else:
                score,gamma=model.hybrid_score(base,q,c)
            w=torch.softmax(score/temp,dim=1)
            q_mu=context_mean(train_b_context[q_idx]).astype(np.float32)
            c_mu=context_mean(train_a_context[c_idx]).astype(np.float32)
            transferred=(train_a_future[c_idx][None,:,:].astype(np.float32)
                         +q_mu[:,None,None]-c_mu[None,:,None])
            fut=torch.from_numpy(transferred).to(device)
            pred=torch.sum(w.unsqueeze(-1)*fut,dim=1)
            y=torch.from_numpy(train_b_future[q_idx].astype(np.float32)).to(device)
            loss=F.mse_loss(pred,y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row={"epoch":epoch,"loss":float(np.mean(losses)),
             "gamma":float(torch.sigmoid(model.logit_gamma).detach().cpu())}
        hist.append(row)
        if epoch==1 or epoch==epochs or epoch%max(1,epochs//10)==0:
            print(f"[{mode}] epoch={epoch:03d} loss={row['loss']:.6f} gamma={row['gamma']:.4f}")
    return hist

@torch.no_grad()
def build_score_matrix(model,mode,base_scores,query_expert,memory_expert,q_batch,c_batch,device):
    model.eval()
    N,K=base_scores.shape
    out=np.empty((N,K),dtype=np.float32)
    q_all=torch.from_numpy(query_expert.astype(np.float32))
    c_all=torch.from_numpy(memory_expert.astype(np.float32))
    t0=time.perf_counter()
    for qs in range(0,N,q_batch):
        qe=min(qs+q_batch,N); q=q_all[qs:qe].to(device)
        for cs in range(0,K,c_batch):
            ce=min(cs+c_batch,K); c=c_all[cs:ce].to(device)
            base=torch.from_numpy(base_scores[qs:qe,cs:ce].astype(np.float32)).to(device)
            if mode=="expert_only":
                s,gamma=model.expert_score(q,c)
            else:
                s,gamma=model.hybrid_score(base,q,c)
            out[qs:qe,cs:ce]=s.cpu().numpy()
    gamma=float(torch.sigmoid(model.logit_gamma).cpu())
    return out,gamma,(time.perf_counter()-t0)/N*1000.0

def softmax_np(x,axis=-1):
    x=x-np.max(x,axis=axis,keepdims=True)
    e=np.exp(x)
    return e/np.sum(e,axis=axis,keepdims=True)

def retrieve_from_scores(scores,mem_ctx,mem_fut,q_ctx,topk,temp,batch_size):
    N,Kall=scores.shape; H=mem_fut.shape[1]
    out=np.empty((N,H),dtype=np.float32)
    t0=time.perf_counter()
    for st in range(0,N,batch_size):
        ed=min(st+batch_size,N); s=scores[st:ed]
        k=min(topk,Kall)
        idx=np.argpartition(-s,kth=k-1,axis=1)[:,:k]
        row=np.arange(ed-st)[:,None]
        ss=s[row,idx]
        order=np.argsort(-ss,axis=1)
        idx=idx[row,order]; ss=ss[row,order]
        w=softmax_np(ss/temp,axis=1)
        q_mu=context_mean(q_ctx[st:ed])
        for r in range(ed-st):
            ids=idx[r]; c_mu=context_mean(mem_ctx[ids])
            transferred=mem_fut[ids].astype(np.float32)+q_mu[r]-c_mu[:,None]
            out[st+r]=np.sum(w[r,:,None]*transferred,axis=0)
    return out,(time.perf_counter()-t0)/N*1000.0

def topk_indices(scores,k):
    k=min(k,scores.shape[1])
    return np.argpartition(-scores,kth=k-1,axis=1)[:,:k]

def topk_overlap(a,b):
    return float(np.mean([len(set(a[i])&set(b[i]))/len(a[i]) for i in range(len(a))]))

def best_lambda(y,chronos,memory,lambdas):
    rows=[]
    for lam in lambdas:
        pred=(1-lam)*chronos+lam*memory
        rows.append({"lambda":float(lam),"mse":mse(y,pred),"mae":mae(y,pred)})
    return rows,min(rows,key=lambda x:(x["mse"],x["lambda"]))

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--bundle",type=Path,required=True)
    p.add_argument("--out-dir",type=Path,default=Path("results/expert_only_ablation_v2"))
    p.add_argument("--epochs",type=int,default=30)
    p.add_argument("--hidden",type=int,default=32)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--candidate-size",type=int,default=256)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--topk",type=int,default=5)
    p.add_argument("--softmax-temperature",type=float,default=1.0)
    p.add_argument("--score-query-batch",type=int,default=32)
    p.add_argument("--score-candidate-batch",type=int,default=2048)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--device",type=str,default="auto")
    p.add_argument("--lambdas",nargs="+",type=float,
                   default=[0,0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6])
    args=p.parse_args()

    if "test" in str(args.bundle).lower():
        raise RuntimeError("拒绝运行：当前实验禁止访问 TEST。")

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device("cuda" if args.device=="auto" and torch.cuda.is_available()
                        else ("cpu" if args.device=="auto" else args.device))
    print("device:",device)

    with np.load(args.bundle,allow_pickle=False) as f:
        A={k:f[k] for k in f.files}

    required=["train_a_context","train_a_future","train_a_expert",
              "train_b_context","train_b_future","train_b_expert",
              "val_context","val_future","val_expert",
              "train_b_shape_scores","val_shape_scores"]
    miss=[k for k in required if k not in A]
    if miss: raise KeyError(f"bundle 缺少字段: {miss}")

    a_ctx=A["train_a_context"]; a_fut=A["train_a_future"].astype(np.float32); a_exp=A["train_a_expert"]
    b_ctx=A["train_b_context"]; b_fut=A["train_b_future"].astype(np.float32); b_exp=A["train_b_expert"]
    v_ctx=A["val_context"]; v_fut=A["val_future"].astype(np.float32); v_exp=A["val_expert"]
    b_shape=A["train_b_shape_scores"].astype(np.float32); v_shape=A["val_shape_scores"].astype(np.float32)

    mu,sd=fit_scaler(a_exp,b_exp)
    a_e=apply_scaler(a_exp,mu,sd); b_e=apply_scaler(b_exp,mu,sd); v_e=apply_scaler(v_exp,mu,sd)
    mem_ctx=np.concatenate([a_ctx,b_ctx],axis=0)
    mem_fut=np.concatenate([a_fut,b_fut],axis=0)
    mem_e=np.concatenate([a_e,b_e],axis=0)

    expert_model=ExpertScoreMLP(a_e.shape[1],args.hidden).to(device)
    hybrid_model=ExpertScoreMLP(a_e.shape[1],args.hidden).to(device)

    expert_hist=train_model(expert_model,"expert_only",b_shape,a_e,b_e,a_ctx,a_fut,b_ctx,b_fut,
                            args.epochs,args.batch_size,args.candidate_size,args.lr,
                            args.softmax_temperature,device,args.seed)
    hybrid_hist=train_model(hybrid_model,"hybrid",b_shape,a_e,b_e,a_ctx,a_fut,b_ctx,b_fut,
                            args.epochs,args.batch_size,args.candidate_size,args.lr,
                            args.softmax_temperature,device,args.seed+1)

    expert_scores,gamma_e,ms_e_score=build_score_matrix(
        expert_model,"expert_only",v_shape,v_e,mem_e,
        args.score_query_batch,args.score_candidate_batch,device)
    hybrid_scores,gamma_h,ms_h_score=build_score_matrix(
        hybrid_model,"hybrid",v_shape,v_e,mem_e,
        args.score_query_batch,args.score_candidate_batch,device)

    shape_pred,ms_shape=retrieve_from_scores(v_shape,mem_ctx,mem_fut,v_ctx,args.topk,args.softmax_temperature,args.batch_size)
    expert_pred,ms_exp=retrieve_from_scores(expert_scores,mem_ctx,mem_fut,v_ctx,args.topk,args.softmax_temperature,args.batch_size)
    hybrid_pred,ms_hyb=retrieve_from_scores(hybrid_scores,mem_ctx,mem_fut,v_ctx,args.topk,args.softmax_temperature,args.batch_size)

    sm,em,hm=mse(v_fut,shape_pred),mse(v_fut,expert_pred),mse(v_fut,hybrid_pred)
    sma,ema,hma=mae(v_fut,shape_pred),mae(v_fut,expert_pred),mae(v_fut,hybrid_pred)

    si=topk_indices(v_shape,args.topk); ei=topk_indices(expert_scores,args.topk); hi=topk_indices(hybrid_scores,args.topk)

    report={
        "experiment":"expert_only_ablation_v2",
        "test_accessed":False,
        "temperature":args.softmax_temperature,
        "memory_only":{
            "shape":{"mse":sm,"mae":sma},
            "expert_only":{"mse":em,"mae":ema,"gamma":gamma_e},
            "shape_plus_expert":{"mse":hm,"mae":hma,"gamma":gamma_h},
            "expert_only_vs_shape_pct":improve(sm,em),
            "hybrid_vs_shape_pct":improve(sm,hm),
            "hybrid_vs_expert_only_pct":improve(em,hm),
        },
        "topk_overlap":{
            "shape_vs_expert_only":topk_overlap(si,ei),
            "shape_vs_hybrid":topk_overlap(si,hi),
            "expert_only_vs_hybrid":topk_overlap(ei,hi),
        },
        "timing_ms_per_query":{
            "shape":ms_shape,
            "expert_only":ms_e_score+ms_exp,
            "hybrid":ms_h_score+ms_hyb,
        },
        "training":{"expert_only":expert_hist,"hybrid":hybrid_hist},
    }

    if "chronos_val_pred" in A:
        chronos=A["chronos_val_pred"].astype(np.float32)
        c_mse=mse(v_fut,chronos)
        sg,sb=best_lambda(v_fut,chronos,shape_pred,args.lambdas)
        eg,eb=best_lambda(v_fut,chronos,expert_pred,args.lambdas)
        hg,hb=best_lambda(v_fut,chronos,hybrid_pred,args.lambdas)
        same=[]
        for lam in args.lambdas:
            same.append({
                "lambda":float(lam),
                "shape_mse":mse(v_fut,(1-lam)*chronos+lam*shape_pred),
                "expert_only_mse":mse(v_fut,(1-lam)*chronos+lam*expert_pred),
                "hybrid_mse":mse(v_fut,(1-lam)*chronos+lam*hybrid_pred),
            })
        report["chronos_fusion"]={
            "chronos":{"mse":c_mse},
            "shape":{"grid":sg,"best":sb,"improvement_vs_chronos_pct":improve(c_mse,sb["mse"])},
            "expert_only":{"grid":eg,"best":eb,"improvement_vs_chronos_pct":improve(c_mse,eb["mse"])},
            "hybrid":{"grid":hg,"best":hb,
                      "improvement_vs_chronos_pct":improve(c_mse,hb["mse"]),
                      "improvement_vs_expert_only_pct":improve(eb["mse"],hb["mse"]),
                      "improvement_vs_shape_pct":improve(sb["mse"],hb["mse"])},
            "same_lambda":same,
        }

    args.out_dir.mkdir(parents=True,exist_ok=True)
    torch.save({"expert_only_state":expert_model.state_dict(),
                "hybrid_state":hybrid_model.state_dict(),
                "expert_mu":mu,"expert_sd":sd},
               args.out_dir/"models.pt")
    np.savez_compressed(args.out_dir/"val_predictions.npz",
                        y_true=v_fut,shape_memory_pred=shape_pred,
                        expert_only_memory_pred=expert_pred,
                        hybrid_memory_pred=hybrid_pred)
    (args.out_dir/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")

    md=[]
    md.append("# V2 Ablation — Shape vs Expert-only vs Shape+Expert")
    md.append("")
    md.append("| Method | MSE | MAE |")
    md.append("|---|---:|---:|")
    md.append(f"| Shape | {sm:.6f} | {sma:.6f} |")
    md.append(f"| Expert-only | {em:.6f} | {ema:.6f} |")
    md.append(f"| Shape+Expert | {hm:.6f} | {hma:.6f} |")
    md.append("")
    md.append(f"Expert-only vs Shape: **{improve(sm,em):+.4f}%**")
    md.append(f"Hybrid vs Shape: **{improve(sm,hm):+.4f}%**")
    md.append(f"Hybrid vs Expert-only: **{improve(em,hm):+.4f}%**")
    md.append("")
    md.append(f"Gamma expert-only: **{gamma_e:.4f}**")
    md.append(f"Gamma hybrid: **{gamma_h:.4f}**")
    md.append(f"Top-K overlap Shape/Expert-only: **{report['topk_overlap']['shape_vs_expert_only']*100:.2f}%**")
    md.append(f"Top-K overlap Shape/Hybrid: **{report['topk_overlap']['shape_vs_hybrid']*100:.2f}%**")
    md.append(f"Top-K overlap Expert-only/Hybrid: **{report['topk_overlap']['expert_only_vs_hybrid']*100:.2f}%**")
    if "chronos_fusion" in report:
        cf=report["chronos_fusion"]
        md.append("")
        md.append(f"Chronos MSE: **{cf['chronos']['mse']:.6f}**")
        md.append(f"Best + Shape: **{cf['shape']['best']['mse']:.6f}**, lambda={cf['shape']['best']['lambda']}")
        md.append(f"Best + Expert-only: **{cf['expert_only']['best']['mse']:.6f}**, lambda={cf['expert_only']['best']['lambda']}")
        md.append(f"Best + Shape+Expert: **{cf['hybrid']['best']['mse']:.6f}**, lambda={cf['hybrid']['best']['lambda']}")
    (args.out_dir/"report.md").write_text("\n".join(md),encoding="utf-8")

    print("="*80)
    print(f"Shape Memory       : {sm:.6f}")
    print(f"Expert-only Memory : {em:.6f}")
    print(f"Shape+Expert       : {hm:.6f}")
    print(f"Hybrid vs Expert   : {improve(em,hm):+.4f}%")
    print(f"Gamma expert-only  : {gamma_e:.4f}")
    print(f"Gamma hybrid       : {gamma_h:.4f}")
    if "chronos_fusion" in report:
        cf=report["chronos_fusion"]
        print(f"Chronos             : {cf['chronos']['mse']:.6f}")
        print(f"Best + Shape        : {cf['shape']['best']['mse']:.6f}")
        print(f"Best + Expert-only  : {cf['expert_only']['best']['mse']:.6f}")
        print(f"Best + Shape+Expert : {cf['hybrid']['best']['mse']:.6f}")
    print("outputs:",args.out_dir)
    print("="*80)

if __name__=="__main__":
    main()
