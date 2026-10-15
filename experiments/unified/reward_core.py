# -*- coding: utf-8 -*-
from common import *
from runtime import *

def semantic(pool):
    keys=['id','mechanism','scope','activation_condition','failure_condition','observables','retrieval_action','expected_effect','reason','condition_spec','replaced']
    result=[]
    for e in v1._sorted_experts(copy.deepcopy(pool)):
        d={k:e[k] for k in keys if k in e}
        d['claim']=e.get('mechanism',e.get('claim',e.get('reason','')))
        d['observable_features']=list(e.get('observables',e.get('observable_features',[])))
        result.append(d)
    return result

def load_data(ds):
    dd=DATA_ROOT/ds;cfg=read(dd/'config.json')
    df=pd.read_csv(dd/'series.csv');series=df.OT.to_numpy(np.float32);dates=pd.to_datetime(df.date)
    z=np.load(dd/'window_indices.npz');starts={k:z[k].astype(np.int64) for k in ['train','val','test']}
    pool=semantic(v2.load_raw_pool(ds));ids=[e['id'] for e in pool]
    assert len(pool)==8 and len(set(ids))==8
    return dict(ds=ds,dd=dd,cfg=cfg,series=series,dates=dates,starts=starts,raw=pool,ids=ids,L=cfg['context'],H=cfg['horizon'])

def available(earlier,later,H):
    """Allow only continuations ending at/before first later forecast origin."""
    assert len(earlier) and len(later)
    a=np.asarray(earlier)[np.asarray(earlier)+H<=int(later[0])]
    assert len(a)>0 and int(a[-1])+H<=int(later[0])
    return a

def fold_spec(data,fold):
    from common import fold_spec as shared_spec
    return shared_spec(data,fold)

def final_spec(data):
    train=available(data['starts']['train'],data['starts']['val'],data['H'])
    cut=int(len(train)*.8);fit=train[cut:];bank=available(train[:cut],fit,data['H'])
    return dict(bank=bank,fit=fit,val=data['starts']['val'],test=data['starts']['test'])


class Fold:
    def __init__(self,data,number,encoder,backbone):
        self.data=data;self.number=number;self.encoder=encoder
        self.path=ART/data['ds']/f'fold{number}';self.spec=fold_spec(data,number)
        self.X={};self.Y={};self.R={};self.ts={}
        for k,a in self.spec.items():
            self.X[k],self.Y[k]=v2.build_windows(data['series'],a,data['L'],data['H'])
            self.R[k]=v2.residualize(self.X[k]);self.ts[k]=[data['dates'][int(s)+data['L']] for s in a]
        hist=np.concatenate([self.X['bank'],self.X['fit']])
        self.stats=v2.compute_training_stats(hist)
        self.evidence={'feature_distributions':v1._raw_feature_quantiles(hist),'temporal_coverage':v1._temporal_coverage(data['dates'].iloc[:int(self.spec['fit'][-1])+data['L']])}
        self.backbone=backbone
        bp=ART/'backbones'/backbone/data['ds']/f'fold{number}'
        self.c=np.load(bp/'predictions.npz')
        assert read(bp/'completed.json')['protocol_hash']==digest(PLAN)
        for k in ['cal','eval']:
            assert np.array_equal(self.c[f'{k}_starts'],self.spec[k])
            assert np.array_equal(self.c[f'Y_{k}'],self.Y[k])
        self.signature=digest({'plan':PLAN,'data':read(ART/'data_provenance.json')[data['ds']],'spec':{k:a.tolist() for k,a in self.spec.items()},'runner':sha(Path(__file__)),'common':sha(Path(__file__).with_name('common.py')),'retrieval':sha(Path(__file__).with_name('unified_retrieval.py'))})
    def compile(self,pool):
        v1.CONTEXT=self.data['L'];v1.HORIZON=self.data['H']
        pool=semantic(pool)
        for e in pool:
            e['compiled']=v1.compile_condition_v6(e,v1.FEATURE_REGISTRY,self.evidence,v1.COMPILER_CONFIG)
            e['retrieval_spec']=v1.derive_retrieval_spec(e,v1.FEATURE_REGISTRY)
        return pool
    def arrays(self,pool):
        compiled=self.compile(pool);ids=self.data['ids']
        emb=np.zeros((8,384),np.float32);acts={k:np.zeros((len(x),8),np.float32) for k,x in self.X.items()}
        if compiled:
            e=self.encoder.encode([p['claim'] for p in compiled],normalize_embeddings=True,convert_to_numpy=True).astype(np.float32)
            for k in acts:
                a=v1.build_activation_matrix(compiled,self.X[k],self.ts[k],self.stats)
                for j,p in enumerate(compiled):acts[k][:,ids.index(p['id'])]=a[:,j]
            for j,p in enumerate(compiled):emb[ids.index(p['id'])]=e[j]
        return compiled,emb,acts
    def predict(self,pool):
        key=digest({'signature':self.signature,'pool':semantic(pool),'seed':42})[:24]
        p=self.path/'retrievers'/key;result=p/'predictions.npz'
        if result.exists():
            meta=read(p/'metadata.json');assert meta['key']==key and meta['signature']==self.signature
            return {k:v for k,v in np.load(result).items()}
        p.mkdir(parents=True,exist_ok=True)
        compiled,emb,acts=self.arrays(pool)
        m=new_model(self.data['L'],emb,42)
        print('TRAIN REWARD',self.data['ds'],'fold',self.number,'experts',[e['id'] for e in pool],flush=True)
        v2p.train_arm(m,self.X['bank'],self.Y['bank'],self.X['fit'],self.Y['fit'],self.R['bank'],self.R['fit'],acts['bank'],acts['fit'],True,DEVICE,42)
        torch.save(m.state_dict(),p/'model.pt');np.save(p/'embeddings.npy',emb)
        np.savez_compressed(p/'activations.npz',**acts)
        preds={}
        for k in ['cal','eval']:
            plain,transferred,indices=RETRIEVE(m,self.X[k],self.X['bank'],self.Y['bank'],self.R[k],self.R['bank'],acts[k],acts['bank'],'full',5,DEVICE)
            preds[k]=transferred.astype(np.float64)
            np.savez_compressed(p/f'{k}_retrieval.npz',indices=indices,bank_starts=self.spec['bank'],query_starts=self.spec[k],prediction=plain,transferred=transferred)
        cov=np.mean(np.concatenate([acts['bank'],acts['fit']])>=.5,axis=0)
        preds['coverage']=cov
        np.savez_compressed(result,**preds)
        dump(p/'metadata.json',{'key':key,'signature':self.signature,'pool':semantic(pool),'compiled':compiled,'fold':self.number,'seed':42})
        del m;torch.cuda.empty_cache()
        return preds

def new_model(context,emb,seed):
    state=v2._init_state(seed,context,384,8,DEVICE);v2p.seed_everything(seed)
    m=v2.ConditionalExpertMLPRetriever(context=context,expert_dim=384,n_experts=8,d_enc=v2.D_ENC,d_exp=v2.D_EXP,d_z=v2.D_Z,hidden=v2.HIDDEN).to(DEVICE)
    m.load_state_dict(state);m.set_expert_embeddings(emb)
    return m


def score_pool(data,folds,pool,objective,stage,path):
    result=path/'reward.json';signature=digest({'plan':PLAN,'pool':semantic(pool),'objective':objective,'folds':[f.signature for f in folds],'backbone':folds[0].backbone})
    if result.exists():
        saved=read(result);assert saved['signature']==signature
        return saved['records']
    points=[];per={e['id']:[] for e in pool};cov={e['id']:[] for e in pool}
    for fold in folds:
        independent=objective=='fusion_independent'
        full=fold.predict([] if independent else pool);cf=fold.c['yc_cal'].astype(np.float64);ce=fold.c['yc_eval'].astype(np.float64)
        lf,gf=select(cf,full['cal'],fold.Y['cal'])
        full_fusion=mse((1-lf)*ce+lf*full['eval'],fold.Y['eval'])
        full_memory=mse(full['eval'],fold.Y['eval'])
        for e in pool:
            if independent:
                full=fold.predict([e]);minus=fold.predict([])
                lf,gf=select(cf,full['cal'],fold.Y['cal'])
                full_fusion=mse((1-lf)*ce+lf*full['eval'],fold.Y['eval'])
                full_memory=mse(full['eval'],fold.Y['eval'])
            else:
                minus=fold.predict([p for p in pool if p['id']!=e['id']])
            lm,gm=select(cf,minus['cal'],fold.Y['cal'])
            minus_fusion=mse((1-lm)*ce+lm*minus['eval'],fold.Y['eval'])
            minus_memory=mse(minus['eval'],fold.Y['eval'])
            fixed=mse((1-lf)*ce+lf*minus['eval'],fold.Y['eval'])
            mr=(minus_memory-full_memory)/max(minus_memory,1e-15)
            fr=(minus_fusion-full_fusion)/max(minus_fusion,1e-15)
            reward=mr if objective=='memory' else fr
            per[e['id']].append(reward);cov[e['id']].append(float(full['coverage'][data['ids'].index(e['id'])]))
            row={'dataset':data['ds'],'objective':objective,'stage':stage,'fold':fold.number,'expert_id':e['id'],'full_lambda':lf,'minus_lambda':lm,'chronos_eval_mse':mse(ce,fold.Y['eval']),'full_memory_mse':full_memory,'minus_memory_mse':minus_memory,'full_fusion_mse':full_fusion,'minus_fusion_mse':minus_fusion,'minus_fusion_fixed_full_lambda_mse':fixed,'memory_reward':mr,'fusion_reward':fr,'chosen_reward':reward,'n_cal':len(cf),'n_eval':len(ce)}
            points.append(row)
    records=[]
    for e in pool:
        r=np.asarray(per[e['id']]);st={'fold_rewards':r.tolist(),'mean_pool_reward':float(r.mean()),'median_pool_reward':float(np.median(r)),'positive_fold_ratio':float((r>0).mean()),'worst_pool_reward':float(r.min())}
        coverage=float(np.mean(cov[e['id']]));reward,verdict=v2p.v2_reward_rule(st,coverage)
        records.append({'expert':semantic([e])[0],'pool_stats':st,'coverage':coverage,'reward':reward,'verdict':verdict,'pool_size':len(pool),'single_effect':None,'synergy_flag':None})
    dump(result,{'signature':signature,'records':records,'pool':semantic(pool),'objective':objective,'stage':stage})
    pd.DataFrame(points).to_csv(path/'fold_effects.csv',index=False,encoding='utf-8-sig')
    print('REWARD DONE',data['ds'],objective,stage,[(r['expert']['id'],r['reward'],round(r['pool_stats']['mean_pool_reward'],5)) for r in records],flush=True)
    return records

def critic(records,objective,stage,backbone):
    definition=('the standalone mean-transferred Top5 MEMORY MSE' if objective=='memory' else f'the FINAL {backbone} plus mean-transferred Top5 MEMORY FUSION MSE; FULL and MINUS lambda are separately CAL-selected and frozen on EVAL')
    system=('You are a rigorous critic for an expert-conditioned retrieval forecaster. '
      'The supplied marginal reward compares '+definition+'. The evaluation design supplied below defines the comparison; do not reinterpret singleton comparisons as pool interactions. '
      'KEEP experts remain in the full pool during all later revisions. All numeric evidence is TRAIN-only. '
      'Respect the supplied verdict and do not recompute data, infer VAL/TEST outcomes, invent patterns, or claim significance. '
      'single_effect and synergy_flag are unknown: do not assert proven synergy. Output strict JSON only.')
    system+=(' This ablation compares each SINGLETON expert against the zero-expert retriever; it does not measure marginal pool contribution.' if objective=='fusion_independent' else ' FULL versus MINUS is a leave-one-out pool comparison.')
    evidence=[{'expert':r['expert'],'reward_stats':r['pool_stats'],'coverage':r['coverage'],'reward':r['reward'],'verdict':r['verdict'],'pool_size':r['pool_size']} for r in records]
    user='OBJECTIVE: '+definition+'\nEVIDENCE:\n'+json.dumps(evidence,ensure_ascii=False,indent=2)+'\nReturn {"critiques":[{"expert_id":"...","verdict":"same as given","pool_contribution":"positive|negative|neutral","synergy_interpretation":"unverified","diagnosis":"2-3 sentences citing supplied numbers","revision_instruction":"TRAIN-verifiable refinement for failed experts"}]}. Include every expert ID once.'
    result,_=v1._call_llm_list(system,user,'reward_alignment_'+objective+'_'+stage+'_critic','critiques',MODEL)
    assert len(result)==len(records) and {r.get('expert_id') for r in result}=={r['expert']['id'] for r in records}
    return {r['expert_id']:r for r in result}

def revision(records,objective,stage,backbone):
    definition='standalone memory accuracy' if objective=='memory' else f'extra reduction in final {backbone}-plus-memory fusion error'
    comparison=('SINGLETON expert versus the zero-expert retriever' if objective=='fusion_independent' else 'FULL pool versus MINUS one expert')
    system=('You revise conditional retrieval experts after a TRAIN-only review using '+comparison+'. '
      'Improve each failed expert\'s contribution under this comparison to '+definition+'. KEEP experts are retained unchanged. '
      'Do not merely add AND conditions or invent numeric data patterns. You may replace an unsupported mechanism using only supplied TRAIN evidence and the feature catalog; mark replaced=true. '
      'Do not infer or use external VAL/TEST performance. Return strict JSON only.')
    feedback=[{'expert':r['expert'],'pool_reward_evidence':r['pool_stats'],'critic':r['critic'],'verdict':r['verdict']} for r in records]
    user=v1._catalog_text()+'\n'+v1._expert_schema_text()+'\nOBJECTIVE: '+definition+'\nFEEDBACK:\n'+json.dumps(feedback,ensure_ascii=False,indent=2)+'\nReturn {"experts":[...]}, the same IDs, all required expert fields plus replaced:boolean. No extra IDs.'
    result,_=v1._call_llm_list(system,user,'reward_alignment_'+objective+'_'+stage+'_revision','experts',MODEL)
    assert len(result)==len(records) and {e['id'] for e in result}=={r['expert']['id'] for r in records}
    return semantic(result)

def optimize(data,folds,objective):
    backbone=folds[0].backbone
    path=ART/'pools'/data['ds']/objective/('shared' if objective=='memory' else backbone);final=path/'final_pool.json'
    path.mkdir(parents=True,exist_ok=True)
    if final.exists():return read(final)
    pool=copy.deepcopy(data['raw']);history=[]
    for rnd in range(3):
        stage=f'round{rnd}';p=path/stage;p.mkdir(parents=True,exist_ok=True)
        records=score_pool(data,folds,pool,objective,stage,p)
        if (p/'critiques.json').exists():cr=read(p/'critiques.json')
        elif objective=='fusion_no_critic':
            cr={r['expert']['id']:{'expert_id':r['expert']['id'],'verdict':r['verdict'],'critic_omitted':True,'revision_instruction':'Use only the supplied numeric TRAIN reward evidence; no separate Critic analysis was performed.'} for r in records}
            dump(p/'critiques.json',cr)
        else:cr=critic(records,objective,data['ds']+'_'+backbone+'_'+stage,backbone);dump(p/'critiques.json',cr)
        for r in records:r['critic']=cr[r['expert']['id']];r['critic']['verdict']=r['verdict']
        dump(p/'records_with_critic.json',records);history.extend([dict(stage=stage,expert_id=r['expert']['id'],reward=r['reward'],verdict=r['verdict'],coverage=r['coverage'],**r['pool_stats']) for r in records])
        failed=[r for r in records if r['reward']<=0]
        if rnd==2 or not failed:break
        if (p/'revisions.json').exists():revised=read(p/'revisions.json')
        else:revised=revision(failed,objective,data['ds']+'_'+backbone+'_'+stage,backbone);dump(p/'revisions.json',revised)
        updates={e['id']:e for e in revised}
        # KEEP experts stay; every slot corresponds to its original lineage.
        pool=semantic([updates.get(e['id'],e) for e in pool])
        assert {e['id'] for e in pool}==set(data['ids'])
    pool=semantic([r['expert'] for r in records if r['reward']>0])
    for step in range(9):
        if not pool:
            dump(path/'final_audit.json',{'empty':True,'method':'NoExpertMLP','all_retained_positive':True});break
        stage=f'final_audit{step}';p=path/stage;p.mkdir(parents=True,exist_ok=True)
        checked=score_pool(data,folds,pool,objective,stage,p)
        history.extend([dict(stage=stage,expert_id=r['expert']['id'],reward=r['reward'],verdict=r['verdict'],coverage=r['coverage'],**r['pool_stats']) for r in checked])
        failed=[r for r in checked if r['reward']<=0]
        if not failed:
            dump(path/'final_audit.json',{'empty':False,'all_retained_positive':True,'records':checked});break
        assert step<8
        # Fixed deterministic removal rule; no VAL or TEST data involved.
        worst=sorted(failed,key=lambda r:(r['reward'],r['pool_stats']['mean_pool_reward'],r['expert']['id']))[0]
        pool=semantic([e for e in pool if e['id']!=worst['expert']['id']])
    dump(final,pool)
    pd.DataFrame(history).to_csv(path/'history.csv',index=False,encoding='utf-8-sig')
    print('FINAL POOL FROZEN',data['ds'],objective,[e['id'] for e in pool],flush=True)
    return pool

