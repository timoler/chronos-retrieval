# -*- coding: utf-8 -*-

"""Architecture arguments extracted unchanged from the audited backbone adapters."""

import argparse

# Source: experiments/external_baselines/infer_patchtst_matrix.py:29-41
def _make_patchtst_builder():
    BATCH = 128
    def build_args(seq_len, pred_len):
        a = argparse.Namespace()
        a.enc_in = 1; a.dec_in = 1; a.c_out = 1
        a.seq_len = seq_len; a.label_len = 48; a.pred_len = pred_len
        a.e_layers = 3; a.d_model = 128; a.n_heads = 16; a.d_ff = 256
        a.dropout = 0.2; a.fc_dropout = 0.2; a.head_dropout = 0.0
        a.patch_len = 16; a.stride = 8; a.padding_patch = "end"; a.individual = 0
        a.revin = 1; a.affine = 0; a.subtract_last = 0
        a.decomposition = 0; a.kernel_size = 25
        a.features = "S"; a.target = "OT"; a.freq = "h"; a.data = "chronos_retrieval"
        a.batch_size = BATCH; a.num_workers = 0; a.gpu = 0; a.use_gpu = True
        a.use_multi_gpu = False; a.inverse = True
        return a
    return build_args

build_patchtst_args = _make_patchtst_builder()

# Source: experiments/external_baselines/infer_itransformer_matrix.py:29-40
def _make_itransformer_builder():
    BATCH = 32
    def build_args(seq_len, pred_len):
        a = argparse.Namespace()
        a.seq_len = seq_len; a.label_len = 48; a.pred_len = pred_len
        a.enc_in = 1; a.dec_in = 1; a.c_out = 1
        a.d_model = 512; a.n_heads = 8; a.e_layers = 3; a.d_layers = 1; a.d_ff = 512
        a.factor = 1; a.embed = "timeF"; a.freq = "h"; a.dropout = 0.1
        a.activation = "gelu"; a.output_attention = False
        a.features = "S"; a.target = "OT"; a.data = "chronos_retrieval"
        a.distil = True; a.use_amp = False; a.inverse = True
        a.class_strategy = "projection"; a.use_norm = True
        a.gpu = 0; a.use_gpu = True; a.use_multi_gpu = False
        return a
    return build_args

build_itransformer_args = _make_itransformer_builder()

# Source: experiments/external_baselines/infer_raft_matrix.py:29-45
def _make_raft_builder():
    BATCH = 32
    def build_args(seq_len, pred_len):
        a = argparse.Namespace()
        a.task_name = "long_term_forecast"
        a.model = "RAFT"
        a.data = "chronos_retrieval"
        a.root_path = ""   # 运行时填
        a.data_path = "series.csv"
        a.features = "S"; a.target = "OT"; a.freq = "h"
        a.seq_len = seq_len; a.label_len = 48; a.pred_len = pred_len
        a.enc_in = 1; a.dec_in = 1; a.c_out = 1
        a.n_period = 3; a.topm = 20
        a.d_model = 512; a.n_heads = 8; a.e_layers = 2; a.d_layers = 1; a.d_ff = 2048
        a.moving_avg = 25; a.factor = 1; a.embed = "timeF"; a.dropout = 0.1
        a.activation = "gelu"; a.output_attention = False; a.distil = True
        a.batch_size = BATCH; a.num_workers = 0; a.gpu = 0; a.use_gpu = True; a.use_multi_gpu = False
        a.inverse = True; a.use_amp = False; a.seasonal_patterns = "M"; a.timeenc = 0
        return a
    return build_args

build_raft_args = _make_raft_builder()

BUILDERS = {'PatchTST': build_patchtst_args, 'iTransformer': build_itransformer_args, 'RAFT': build_raft_args}

def build_args(backbone, seq_len, pred_len):
    return BUILDERS[backbone](seq_len, pred_len)
