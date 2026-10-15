# -*- coding: utf-8 -*-
from common import *
import argparse
from reward_core import Fold,load_data,optimize
from sentence_transformers import SentenceTransformer

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True);ap.add_argument('--backbone',required=True)
    ap.add_argument('--objective',required=True,choices=['memory','fusion','fusion_no_critic','fusion_independent'])
    a=ap.parse_args();assert_frozen()
    d=load_data(a.dataset)
    encoder=SentenceTransformer(str(EMBED),device='cuda',local_files_only=True)
    folds=[Fold(d,i,encoder,a.backbone) for i in range(1,5)]
    pool=optimize(d,folds,a.objective)
    path=ART/'pools'/a.dataset/a.objective/('shared' if a.objective=='memory' else a.backbone)
    complete(path,dict(dataset=a.dataset,backbone=a.backbone,objective=a.objective,pool_size=len(pool),pool_sha256=sha(path/'final_pool.json')))
    print('POOL COMPLETE',a.dataset,a.backbone,a.objective,len(pool),flush=True)
if __name__=='__main__':main()

