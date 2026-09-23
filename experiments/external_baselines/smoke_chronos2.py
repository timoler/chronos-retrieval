# -*- coding: utf-8 -*-
"""Chronos-2 loading smoke test (no formal results touched).

Loads amazon/chronos-2 in the CURRENT environment and runs a tiny
predict_quantiles batch. If load fails, prints the full traceback so the
historically-working chronos/transformers/torch versions can be identified.
"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch

print("=" * 64)
print("Environment versions:")
print(f"  python        : {sys.version.split()[0]}")
print(f"  torch         : {torch.__version__}")
print(f"  cuda          : {torch.cuda.is_available()}"
      f" ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'})")
try:
    import chronos
    print(f"  chronos       : {getattr(chronos, '__version__', 'unknown')}")
    print(f"  chronos file  : {chronos.__file__}")
except Exception as e:
    print(f"  chronos       : import error {e!r}")
try:
    import transformers
    print(f"  transformers  : {transformers.__version__}")
except Exception as e:
    print(f"  transformers  : import error {e!r}")
print("=" * 64)

MODEL = "amazon/chronos-2"

try:
    from chronos import Chronos2Pipeline
    print(f"\n[1] Chronos2Pipeline import ... OK")
    print(f"[2] from_pretrained({MODEL}, device_map='cuda') ...")
    t0 = time.time()
    pipe = Chronos2Pipeline.from_pretrained(MODEL, device_map="cuda")
    print(f"    loaded in {time.time() - t0:.1f}s")
except Exception:
    print(f"\n>>> FAILED to load {MODEL}:")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    print(f"\n[3] predict_quantiles on a tiny batch (context=512, H=96) ...")
    rng = np.random.default_rng(0)
    ctx = rng.normal(size=512).astype(np.float32)
    inputs = [torch.tensor(ctx, dtype=torch.float32)]
    t0 = time.time()
    _, means = pipe.predict_quantiles(inputs, prediction_length=96, batch_size=256)
    print(f"    predict_quantiles OK in {time.time() - t0:.1f}s")
    m = means[0].squeeze(0)
    print(f"    mean forecast shape: {tuple(m.shape)}")
    print(f"    mean[:5] = {[f'{v:.4f}' for v in m[:5].numpy()]}")
    print("\n>>> SMOKE TEST PASSED")
except Exception:
    print(f"\n>>> FAILED predict_quantiles:")
    import traceback
    traceback.print_exc()
    sys.exit(1)
