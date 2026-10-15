# -*- coding: utf-8 -*-
"""Single-GPU resumable queue; reports real completed jobs, never inherited scores."""
from common import *
import argparse,subprocess,shutil,csv

def jobs():
    result=[]
    order=['traffic','exchange_rate','etth1','etth2','weather','solar','electricity']
    for ds in order:
        for fold in range(1,5):
            for bb in ['PatchTST','iTransformer','RAFT','Chronos-2']:
                result.append(dict(id=f'fold-{bb}-{ds}-{fold}',phase='backbones',script='backbone.py',args=['--dataset',ds,'--backbone',bb,'--fold',str(fold),'--seed','42'],marker=str(ART/'backbones'/bb/ds/f'fold{fold}/completed.json')))
        for bb in PLAN['backbones']:
            for seed in ([42] if bb=='Chronos-2' else SEEDS):
                marker=ART/'backbones'/bb/ds/'final'/('shared' if bb=='Chronos-2' else f'seed{seed}')/'completed.json'
                result.append(dict(id=f'base-val-{bb}-{ds}-{seed}',phase='backbones',script='backbone.py',args=['--dataset',ds,'--backbone',bb,'--seed',str(seed)],marker=str(marker)))
    for ds in order:
        for objective in ['memory','fusion','fusion_no_critic','fusion_independent']:
            for bb in (['Chronos-2'] if objective=='memory' else PLAN['backbones']):
                marker=ART/'pools'/ds/objective/('shared' if objective=='memory' else bb)/'completed.json'
                result.append(dict(id=f'pool-{objective}-{bb}-{ds}',phase='pools',script='pools.py',args=['--dataset',ds,'--backbone',bb,'--objective',objective],marker=str(marker)))
    for ds in order:
        for seed in SEEDS:
            result.append(dict(id=f'val-{ds}-{seed}',phase='val',script='final.py',args=['--dataset',ds,'--seed',str(seed),'--split','val'],marker=str(ART/'final'/ds/f'val_seed{seed}/completed.json')))
    result.append(dict(id='freeze-all-val',phase='freeze',script=None,args=[],marker=str(ART/'READOUT_FROZEN.json')))
    for ds in order:
        for bb in PLAN['backbones']:
            for seed in ([42] if bb=='Chronos-2' else SEEDS):
                marker=ART/'backbones'/bb/ds/'final'/('shared' if bb=='Chronos-2' else f'seed{seed}')/'test_completed.json'
                result.append(dict(id=f'base-test-{bb}-{ds}-{seed}',phase='test',script='backbone.py',args=['--dataset',ds,'--backbone',bb,'--seed',str(seed),'--stage','test'],marker=str(marker)))
        for seed in SEEDS:
            result.append(dict(id=f'test-{ds}-{seed}',phase='test',script='final.py',args=['--dataset',ds,'--seed',str(seed),'--split','test'],marker=str(ART/'final'/ds/f'test_seed{seed}/completed.json')))
    result.append(dict(id='independent-verification',phase='verify',script='verify.py',args=[],marker=str(ART/'VERIFIED_COMPLETE.json')))
    return result

def is_done(job):
    p=Path(job['marker'])
    if not p.exists():return False
    assert read(p)['protocol_hash']==digest(PLAN),'Marker has wrong protocol: '+str(p)
    return True

def status(alljobs,state,current=None,pid=None,error=None):
    counts={}
    for j in alljobs:
        c=counts.setdefault(j['phase'],{'completed':0,'total':0});c['total']+=1;c['completed']+=int(is_done(j))
    s=dict(protocol=PLAN['version'],protocol_hash=digest(PLAN),state=state,current_job=current,worker_pid=pid,
           updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),phases=counts,
           target_test_rows=672,test_readout_started=(ART/'READOUT_FROZEN.json').exists(),error=error,
           note='Counts are pipeline jobs; shared deterministic predictions are not independent training runs.')
    for p in [ART/'STATUS.json',OUT/'STATUS.json',BASE/'docs/unified/STATUS.json']:dump(p,s)
    from publish import publish
    phase=next((j['phase'] for j in alljobs if j['id']==current),None)
    publish(BASE,ART,OUT,state,phase)

def freeze():
    files={}
    for ds in DSLIST:
        for bb in PLAN['backbones']:
            for seed in SEEDS:
                p=ART/'final'/ds/bb/f'seed{seed}/VAL_CHOICES.json';choices=read(p)
                assert set(choices)==set(PLAN['core_arms'])
                files[str(p)]=sha(p)
                for choice in choices.values():
                    for field in ['memory','backbone']:
                        source=Path(choice[field+'_path'])
                        assert sha(source)==choice[field+'_sha256']
                        files[str(source)]=sha(source)
                        for name in ['checkpoint.pt','model.pt','embeddings.npy','compiled_pool.json','training_complete.json']:
                            artifact=source.parent/name
                            if artifact.exists():files[str(artifact)]=sha(artifact)
    for p in (ART/'pools').rglob('final_pool.json'):files[str(p)]=sha(p)
    dump(ART/'READOUT_FROZEN.json',dict(protocol_hash=digest(PLAN),val_files=files,frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['all','backbones'],default='all');a=ap.parse_args()
    assert_frozen();alljobs=jobs();dump(ART/'JOBS.json',alljobs)
    lock=ART/'runner.lock'
    if lock.exists():raise RuntimeError('Queue lock exists: inspect the saved controller PID before restarting')
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY);os.write(fd,str(os.getpid()).encode());os.close(fd)
    dump(ART/'controller.json',dict(pid=os.getpid(),command='experiments/unified/runner.py --phase '+a.phase))
    current=None
    try:
        for job in alljobs:
            if a.phase=='backbones' and job['phase']!='backbones':continue
            if is_done(job):continue
            current=job['id']
            if shutil.disk_usage(ART).free<3*1024**3:raise RuntimeError('Less than 3GiB free; stopping before artifact writes')
            if job['phase']=='pools' and not (ART/'LLM_NETWORK_READY.json').exists():
                status(alljobs,'WAITING_FOR_LLM_NETWORK',current,error='LLM transport not enabled; local backbone phase finished')
                return
            if job['phase']=='freeze':freeze();continue
            if job['phase']=='test':assert (ART/'READOUT_FROZEN.json').exists()
            log=ART/'logs'/(job['id']+'.log');log.parent.mkdir(exist_ok=True)
            command=[sys.executable,'-B','-X','utf8',str(Path(__file__).with_name(job['script'])),*job['args']]
            with log.open('a',encoding='utf-8') as f:
                proc=subprocess.Popen(command,stdout=f,stderr=subprocess.STDOUT,cwd=BASE,env=os.environ.copy(),creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                status(alljobs,'RUNNING',current,proc.pid)
                while proc.poll() is None:
                    time.sleep(20);status(alljobs,'RUNNING',current,proc.pid)
                assert proc.returncode==0,f'{current} exited {proc.returncode}; inspect {log}'
            assert is_done(job),'Worker exited without valid completion marker: '+current
            status(alljobs,'RUNNING',current)
        status(alljobs,'COMPLETE' if a.phase=='all' else 'BACKBONES_COMPLETE')
    except Exception as e:
        status(alljobs,'FAILED',current,error=str(e));raise
    finally:
        lock.unlink(missing_ok=True)
if __name__=='__main__':main()
