"""라벨이 있는 폴더(예: test_crops)로 학습된 모델을 평가한다.

폴더 구조: <data_root>/<클래스명>/*.png, OOD 클래스 폴더(기본 OffTarget) 포함 가능.
출력:
  - Closed-set (OffTarget 제외): accuracy, macro F1          -> 논문 Table 1의 accuracy / macro F1
  - Off-Target 탐지: AUROC_off, AUPR_off (cosine similarity)  -> 논문 Table 1
  - 전체 분류 (cosine 임계값 p 적용): 클래스별 F1, weighted F1, confusion matrix -> 논문 Fig. 3 / S53

사용 예:
    python evaluate.py --ckpt_dir ./checkpoints/seed0 --data_root ./si_dataset/shape/test_crops --gpu 0
"""
import argparse, glob, json, os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, classification_report)

from src.model import Classifier, percentile_threshold
from src.data import UnlabeledDataset, get_eval_transform


def latest(ckpt_dir, prefix, ext):
    fs = sorted(glob.glob(os.path.join(ckpt_dir, f"{prefix}_*.{ext}")))
    if not fs:
        raise FileNotFoundError(f"{ckpt_dir}에 {prefix}_*.{ext} 파일이 없습니다")
    return fs[-1]


p = argparse.ArgumentParser()
p.add_argument("--ckpt_dir", required=True)
p.add_argument("--data_root", required=True)
p.add_argument("--ood_class", default="OffTarget")
p.add_argument("--percentile", type=float, default=7.0, help="논문 기본값 p=7")
p.add_argument("--batch_size", type=int, default=128)
p.add_argument("--gpu", type=int, default=None)
p.add_argument("--train_root", default=None,
               help="학습 이미지 폴더(예: ./data/seed0/train). 주면 클래스별 임계값·유사도 분석을 추가로 수행")
p.add_argument("--calib_root", default=None,
               help="학습에 안 쓴 라벨 폴더(예: work/seed0/calib). 주면 이 폴더의 in-class 유사도로 임계값을 보정")
p.add_argument("--out", default=None, help="결과 JSON 경로 (기본: ckpt_dir/eval_<폴더명>.json)")
a = p.parse_args()

device = torch.device(f"cuda:{a.gpu}" if a.gpu is not None and torch.cuda.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))
model_path = latest(a.ckpt_dir, "model", "pth")
classes = [l.strip() for l in open(latest(a.ckpt_dir, "classes", "txt")) if l.strip()]
proto = torch.load(latest(a.ckpt_dir, "prototypes", "pt"), map_location="cpu")
prototypes, train_sim = proto["prototypes"], proto["train_sim"]
print(f"device={device}  model={os.path.basename(model_path)}  classes={classes}")

model = Classifier(num_classes=len(classes))
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.to(device).eval()

# 정답 라벨 수집
folders = sorted(d for d in os.listdir(a.data_root) if os.path.isdir(os.path.join(a.data_root, d)))
unknown = [f for f in folders if f not in classes and f != a.ood_class]
if unknown:
    raise ValueError(f"학습 클래스에 없는 폴더: {unknown}")
paths, y = [], []
for f in folders:
    fs = sorted(glob.glob(os.path.join(a.data_root, f, "*.png")))
    paths += fs
    y += [classes.index(f) if f != a.ood_class else -1] * len(fs)
y = np.array(y)

loader = DataLoader(UnlabeledDataset(paths, get_eval_transform(224)),
                    batch_size=a.batch_size, shuffle=False, num_workers=0)
logits_all, sim_all = [], []
protos = prototypes.to(device)
with torch.no_grad():
    for imgs, _ in loader:
        imgs = imgs.to(device)
        feats = model.extract_features(imgs)
        logits_all.append(model.mlp(feats).cpu())
        sim_all.append((torch.nn.functional.normalize(feats, dim=1) @ protos.T).max(1).values.cpu())
logits = torch.cat(logits_all).numpy()
max_sim = torch.cat(sim_all).numpy()
pred = logits.argmax(1)

res = {"n_images": int(len(y)), "n_in_class": int((y >= 0).sum()), "n_off_target": int((y < 0).sum())}

# 1) Closed-set (OffTarget 제외)
m = y >= 0
res["closed_accuracy"] = accuracy_score(y[m], pred[m])
res["closed_macro_f1"] = f1_score(y[m], pred[m], average="macro")

# 2) Off-Target 탐지 (유사도가 낮을수록 Off-Target) - threshold-free
if (y < 0).any():
    is_off = (y < 0).astype(int)
    res["AUROC_off"] = roc_auc_score(is_off, -max_sim)
    res["AUPR_off"] = average_precision_score(is_off, -max_sim)

# 3) 전체 분류: cosine 유사도 < p-th percentile 이면 OffTarget
thr = percentile_threshold(train_sim, a.percentile)
full_pred = np.where(max_sim < thr, -1, pred)
names = classes + ([a.ood_class] if (y < 0).any() else [])
lab = list(range(len(classes))) + ([-1] if (y < 0).any() else [])
res["threshold"] = {"percentile": a.percentile, "cosine": thr}
res["full_weighted_f1"] = f1_score(y, full_pred, labels=lab, average="weighted")
res["full_per_class_f1"] = dict(zip(names, f1_score(y, full_pred, labels=lab, average=None).round(4).tolist()))
cm = confusion_matrix(y, full_pred, labels=lab)
res["confusion_matrix"] = {"labels(rows=true, cols=pred)": names, "matrix": cm.tolist()}

print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}, indent=2, ensure_ascii=False))
print(classification_report(y, full_pred, labels=lab, target_names=names, digits=4, zero_division=0))

# 4) 추가 분석: 클래스별 유사도 분포, 퍼센타일 스윕, 클래스별 임계값
sweep = {}
names_idx = dict(zip(lab, names))
def full_f1(fp):
    return f1_score(y, fp, labels=lab, average="weighted")
def off_f1(fp):
    return f1_score((y < 0).astype(int), (fp < 0).astype(int)) if (y < 0).any() else float("nan")

print("\n[테스트 이미지의 최대 cosine 유사도 (정답 클래스별)]")
for l in lab:
    v = max_sim[y == l]
    print(f"  {names_idx[l]:11s} n={len(v):4d}  median={np.median(v):.3f}  p10={np.percentile(v,10):.3f}  "
          f"< thr({thr:.3f}) 비율={np.mean(v < thr):.2%}")

print("\n[퍼센타일 스윕: 전역 임계값]")
for pp in [1, 2, 3, 5, 7, 10]:
    t = percentile_threshold(train_sim, pp)
    fp = np.where(max_sim < t, -1, pred)
    sweep[f"global_p{pp}"] = {"thr": t, "weighted_f1": full_f1(fp), "offtarget_f1": off_f1(fp)}
    print(f"  p={pp:2d} thr={t:.3f}  weighted F1={full_f1(fp):.4f}  Off-Target F1={off_f1(fp):.4f}")

if a.train_root:
    tr_paths, tr_y = [], []
    for ci, c in enumerate(classes):
        fs = sorted(glob.glob(os.path.join(a.train_root, c, "*.png")))
        tr_paths += fs; tr_y += [ci] * len(fs)
    tr_y = np.array(tr_y)
    tl = DataLoader(UnlabeledDataset(tr_paths, get_eval_transform(224)), batch_size=a.batch_size, num_workers=0)
    ts = []
    with torch.no_grad():
        for imgs, _ in tl:
            f = torch.nn.functional.normalize(model.extract_features(imgs.to(device)), dim=1)
            ts.append((f @ protos.T).max(1).values.cpu())
    ts = torch.cat(ts).numpy()
    print("\n[학습 이미지의 최대 cosine 유사도 (클래스별)]")
    for ci, c in enumerate(classes):
        v = ts[tr_y == ci]
        print(f"  {c:11s} n={len(v):3d}  median={np.median(v):.3f}  min={v.min():.3f}")
    print("\n[퍼센타일 스윕: 클래스별 임계값 (예측 클래스의 임계값 적용)]")
    for pp in [1, 2, 3, 5, 7, 10]:
        cthr = np.array([np.percentile(ts[tr_y == ci], pp) for ci in range(len(classes))])
        fp = np.where(max_sim < cthr[pred], -1, pred)
        sweep[f"perclass_p{pp}"] = {"thr": cthr.round(4).tolist(), "weighted_f1": full_f1(fp), "offtarget_f1": off_f1(fp)}
        print(f"  p={pp:2d} thr={np.round(cthr,3)}  weighted F1={full_f1(fp):.4f}  Off-Target F1={off_f1(fp):.4f}")
# 5) 오라클 임계값: 테스트 정답으로 고른 최적 임계값 (진단 전용, 보고용 수치 아님)
grid = np.unique(np.round(max_sim, 3))
best = max(((full_f1(np.where(max_sim < t, -1, pred)), t) for t in grid), key=lambda x: x[0])
sweep["oracle"] = {"thr": float(best[1]), "weighted_f1": float(best[0])}
print(f"\n[오라클 임계값 (진단용)] thr={best[1]:.3f}  weighted F1={best[0]:.4f}"
      f"  <- 점수 순위가 좋아도 임계값 선택이 문제인지 판단")

# 6) 검증 세트 보정 임계값: 학습에 안 쓴 in-class 이미지의 유사도 p-th 퍼센타일
if a.calib_root:
    cp = []
    for c in classes:
        cp += sorted(glob.glob(os.path.join(a.calib_root, c, "*.png")))
    cl = DataLoader(UnlabeledDataset(cp, get_eval_transform(224)), batch_size=a.batch_size, num_workers=0)
    cs = []
    with torch.no_grad():
        for imgs, _ in cl:
            f = torch.nn.functional.normalize(model.extract_features(imgs.to(device)), dim=1)
            cs.append((f @ protos.T).max(1).values.cpu())
    cs = torch.cat(cs).numpy()
    print(f"\n[검증 세트 보정 임계값] calib in-class n={len(cs)}")
    for pp in [1, 2, 3, 5, 7, 10]:
        t = float(np.percentile(cs, pp))
        fp = np.where(max_sim < t, -1, pred)
        sweep[f"calib_p{pp}"] = {"thr": t, "weighted_f1": full_f1(fp), "offtarget_f1": off_f1(fp)}
        print(f"  p={pp:2d} thr={t:.3f}  weighted F1={full_f1(fp):.4f}  Off-Target F1={off_f1(fp):.4f}")
res["sweep"] = sweep

out = a.out or os.path.join(a.ckpt_dir, f"eval_{os.path.basename(os.path.normpath(a.data_root))}.json")
json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
print(f"saved -> {out}")
