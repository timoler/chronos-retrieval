"""Command line interface to the frozen PAER method."""
import argparse
import json
from pathlib import Path
import os
import tempfile


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def output_dir(path):
    path.parent.mkdir(exist_ok=True, parents=True)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output directory must be empty: {path}")
    return path


def save_json(path, data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def parser():
    arg = argparse.ArgumentParser(prog="python -m paer")
    sub = arg.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="Run the built-in synthetic protocol and retrieval checks")
    for name, helptext in [
        ("evaluate","Compute full-pool and leave-one-expert-out four-fold contributions"),
        ("refine","Evaluate contributions and accept saved revision proposals"),
        ("deploy","Fit the frozen pool on TRAIN, select optional mixture weight on VAL, predict TEST"),
    ]:
        p = sub.add_parser(name,help=helptext)
        p.add_argument("--series",required=True,type=Path,help="Prepared chronological date,OT CSV")
        p.add_argument("--protocol",required=True,type=Path,help="Task JSON with context, horizon and split")
        p.add_argument("--pool",required=True,type=Path,help="JSON list of expert objects")
        p.add_argument("--embeddings",type=Path,help="Directory of normalized, per-text .npy files")
        p.add_argument("--encoder",type=Path,help="Optional local all-MiniLM-L6-v2 model directory")
        p.add_argument("--device",default="auto",choices=("auto","cpu","cuda"))
        p.add_argument("--out",required=True,type=Path,help="New or empty directory")
        if name=="refine":
            p.add_argument("--proposals",required=True,type=Path,help='JSON mapping "1", "2" to complete proposals')
        if name=="deploy":
            p.add_argument("--seed",type=int,default=42)
            p.add_argument("--base-val",type=Path,help="VAL .npz with starts and prediction")
            p.add_argument("--base-test",type=Path,help="TEST .npz with starts and prediction")
    return arg


def load_base(path, starts, shape):
    import numpy as np
    with np.load(path, allow_pickle=False) as data:
        np.testing.assert_array_equal(data["starts"], starts)
        values = np.asarray(data["prediction"],dtype=np.float64)
    if values.shape!=shape or not np.isfinite(values).all():
        raise ValueError(f"Expected base prediction {shape} in original target units: {path}")
    return values


def deploy(args, series, protocol, pool, vectors, device, out):
    import numpy as np
    import torch
    from .retrieval import fit, predict, select_weight
    weight=None
    if bool(args.base_val) != bool(args.base_test):
        raise ValueError("Supply both --base-val and --base-test to evaluate a fusion.")
    metrics={}
    for split in ("val","test"):
        first,last=protocol["split"][split]
        starts=np.arange(first,last-series.context-series.horizon+1,dtype=np.int64)
        stage=series.stage(series.train_end,last,starts)
        if split=="val":
            model=fit(stage,pool,vectors,args.seed,device)
            torch.save(model.state_dict(),out/"retriever.pt")
        pred=predict(model,stage,pool)
        arrays={k:pred[k] for k in ("starts","indices","raw","aligned")}
        arrays["labels"]=stage["Y"]["query"]
        if args.base_val:
            base=load_base(getattr(args,"base_"+split),starts,arrays["labels"].shape)
            if split=="val":
                weight=select_weight(arrays["labels"],base,pred["aligned"])
                save_json(out/"weight.json",{"lambda":weight,"selected_on":"val","metric":"MSE"})
            arrays["base"]=base
            arrays["fused"]=(1-weight)*base+weight*pred["aligned"]
        for key in ("raw","aligned","base","fused"):
            if key in arrays:
                diff=np.asarray(arrays[key],np.float64)-arrays["labels"]
                metrics[f"{split}_{key}"]={"mse":float(np.mean(diff**2)),"mae":float(np.mean(abs(diff)))}
        np.savez_compressed(out/f"{split}.npz",**arrays)
    save_json(out/"metrics.json",metrics)


def main(argv=None):
    args=parser().parse_args(argv)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import torch
    torch.set_num_threads(4)
    if args.command=="check":
        from .check import check
        return check()
    import pandas as pd
    from .data import Series,compile_pool
    from .refine import Evaluator,refine
    from .retrieval import TextVectors
    device="cuda" if args.device=="auto" and torch.cuda.is_available() else args.device
    if device=="auto":
        device="cpu"
    spec=read_json(args.protocol)
    frame=pd.read_csv(args.series,usecols=["date","OT"])
    train,val,test=(spec["split"][part] for part in ("train","val","test"))
    if not (train[0]==0 < train[1]==val[0] < val[1]==test[0] < test[1]<=len(frame)):
        raise ValueError("Expect consecutive TRAIN, VAL, TEST half-open row ranges")
    calendar=spec.get("calendar") != "disabled" and "exchange" not in str(spec.get("dataset","")).lower()
    series=Series(frame.OT.to_numpy(),pd.to_datetime(frame.date),int(spec["context"]),
                  int(spec["horizon"]),train[1],calendar)
    pool=compile_pool(read_json(args.pool),calendar)
    if pool and not (args.embeddings or args.encoder):
        raise ValueError("Specify --embeddings or --encoder for a nonempty knowledge pool")
    vectors=TextVectors(args.embeddings,args.encoder)
    out=output_dir(args.out)
    # Commit files only once the command has produced a complete set of results.
    with tempfile.TemporaryDirectory(prefix="paer-",dir=out.parent) as tmp:
        staging=Path(tmp)
        if args.command=="deploy":
            deploy(args,series,spec,pool,vectors,device,staging)
        else:
            evaluator=Evaluator(series,vectors,device)
            if args.command=="evaluate":
                save_json(staging/"evaluation.json",evaluator.evaluate(pool))
            else:
                final,rounds=refine(pool,read_json(args.proposals),evaluator,calendar)
                save_json(staging/"final_pool.json",final)
                save_json(staging/"rounds.json",rounds)
        if out.exists():
            out.rmdir()
        staging.rename(out)
    print(f"Wrote {args.command} results to {out}")


if __name__=="__main__":
    main()
