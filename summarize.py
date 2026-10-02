"""여러 시드의 평가 결과(JSON)를 평균 ± 표준편차로 요약한다.
사용 예: python summarize.py ./checkpoints/seed*/eval_test_crops.json
"""
import json, sys, glob, numpy as np
files = sorted(f for pat in sys.argv[1:] for f in glob.glob(pat))
keys = ["closed_accuracy", "closed_macro_f1", "AUROC_off", "AUPR_off", "full_weighted_f1"]
paper = {"closed_accuracy": 0.9928, "closed_macro_f1": 0.9922, "AUROC_off": 0.9319, "AUPR_off": 0.8057, "full_weighted_f1": 0.91}
rs = [json.load(open(f)) for f in files]
print(f"{len(rs)} runs: {files}")
print(f"{'metric':18s} {'mean':>8s} {'std':>8s} {'paper':>8s}")
for k in keys:
    v = np.array([r[k] for r in rs if k in r])
    if len(v):
        print(f"{k:18s} {v.mean():8.4f} {v.std(ddof=1) if len(v)>1 else 0:8.4f} {paper[k]:8.4f}")
