# -*- coding: utf-8 -*-
"""Audited PAER components extracted into the unified library."""
from common import *
import torch
sys.dont_write_bytecode=True
sys.path.insert(0,str(BASE))
from paer_core import forecasting as v2
from paer_core import model as cm
v1=v2.v1;v2p=v2.v2p;av2=v2p.av2
v2.FROZEN_ROOT=ROOT/'data/experts'
v2p.EPOCHS=PLAN['retriever_epochs']
from llm_adapter import CodexTextAdapter
av2._call_llm=CodexTextAdapter(ART/'llm_calls',MODEL).call
av2._default_model=lambda:MODEL
DEVICE=torch.device('cuda')
torch.set_num_threads(4)
assert torch.cuda.is_available(),'The frozen protocol requires CUDA; do not silently switch to CPU'
from unified_retrieval import retrieve_capture as RETRIEVE

