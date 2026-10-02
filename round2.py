"""논문의 2라운드 학습을 시뮬레이션한다. 사람 대신 검증 세트의 정답 라벨이 '사용자' 역할을 한다.

흐름 (시드마다):
  1) 1라운드: val_crops에서 클래스당 k1장(기본 10)을 무작위로 뽑아 학습 -> ckpt_r1
  2) 풀(pool) = 나머지 val_crops 전체 (Off-Target 포함, 정답은 '사용자'만 앎)
     1라운드 모델로 풀을 분류하고, cosine 유사도 < p-th 퍼센타일인 이미지를 '저신뢰(Off-Target 후보)'로 표시
  3) 2라운드: 저신뢰 이미지 중 정답이 실제 클래스인 것을 클래스당 k2장(기본 10) 골라 추가 -> 재학습 -> ckpt_r2
  4) 대조군(control): 1라운드 k1장 + 풀에서 '무작위' k2장(저신뢰 여부 무관) -> 학습 -> ckpt_ctrl
     => r2 vs ctrl 비교로 '애매한 샘플을 고른 효과'와 '라벨 수가 늘어난 효과'를 분리
  5) 각 모델을 test_crops로 평가 (evaluate.py 호출), 남은 풀은 calib 폴더로 저장(임계값 보정 실험용)

사용 예:
  python round2.py --seeds 0 --gpu 0 --batch_size 32
  python round2.py --seeds 0 1 2 --gpu 0 --batch_size 32 --skip_control
"""
import argparse, glob, os, random, shutil, subprocess, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model import Classifier, cosine_ood_scores, percentile_threshold
from src.data import UnlabeledDataset, get_eval_transform

p = argparse.ArgumentParser()
p.add_argument("--src", default="./si_dataset/shape/val_crops")
p.add_argument("--test", default="./si_dataset/shape/test_crops")
p.add_argument("--work", default="./work")
p.add_argument("--seeds", type=int, nargs="+", default=[0])
p.add_argument("--k1", type=int, default=10, help="1라운드 클래스당 라벨 수")
p.add_argument("--k2", type=int, default=10, help="2라운드 클래스당 추가 라벨 수")
p.add_argument("--percentile", type=float, default=7.0)
p.add_argument("--pick", choices=["random", "hardest"], default="random",
               help="저신뢰 샘플 중 고르는 방식: 무작위 / 유사도가 가장 낮은 것")
p.add_argument("--ood_class", default="OffTarget")
p.add_argument("--epochs", type=int, default=13)
p.add_argument("--batch_size", type=int, default=32)
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--skip_control", action="store_true")
a = p.parse_args()

PY = sys.executable
classes_all = sorted(d for d in os.listdir(a.src) if os.path.isdir(os.path.join(a.src, d)))
classes = [c for c in classes_all if c != a.ood_class]


def copy_set(items, root):
    """items: list of (path, class) -> root/class/"""
    if os.path.exists(root):
        shutil.rmtree(root)
    for f, c in items:
        d = os.path.join(root, c); os.makedirs(d, exist_ok=True); shutil.copy2(f, d)


def run(cmd):
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def train(data_root, out_dir):
    run([PY, "train.py", "--data_root", data_root, "--epochs", str(a.epochs), "--gpu", str(a.gpu),
         "--batch_size", str(a.batch_size), "--output_dir", out_dir])


def evaluate(ckpt, train_root, calib_root):
    run([PY, "evaluate.py", "--ckpt_dir", ckpt, "--data_root", a.test, "--train_root", train_root,
         "--calib_root", calib_root, "--percentile", str(a.percentile), "--gpu", str(a.gpu)])


def latest(d, pat):
    return sorted(glob.glob(os.path.join(d, pat)))[-1]


def pool_scores(ckpt, paths):
    dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")
    names = [l.strip() for l in open(latest(ckpt, "classes_*.txt")) if l.strip()]
    m = Classifier(num_classes=len(names))
    m.load_state_dict(torch.load(latest(ckpt, "model_*.pth"), map_location="cpu"))
    m.to(dev).eval()
    pr = torch.load(latest(ckpt, "prototypes_*.pt"), map_location="cpu")
    thr = percentile_threshold(pr["train_sim"], a.percentile)
    sims = []
    dl = DataLoader(UnlabeledDataset(paths, get_eval_transform(224)), batch_size=64, num_workers=0)
    with torch.no_grad():
        for x, _ in dl:
            s, _ = cosine_ood_scores(m, x.to(dev), pr["prototypes"], dev)
            sims.append(s.cpu())
    return torch.cat(sims).numpy(), thr


for seed in a.seeds:
    rng = random.Random(seed)
    W = os.path.join(a.work, f"seed{seed}")
    os.makedirs(W, exist_ok=True)
    files = {c: sorted(glob.glob(os.path.join(a.src, c, "*.png"))) for c in classes_all}

    # ---- 1라운드
    r1 = [(f, c) for c in classes for f in rng.sample(files[c], a.k1)]
    r1_set = {f for f, _ in r1}
    pool = [(f, c) for c in classes_all for f in files[c] if f not in r1_set]
    copy_set(r1, os.path.join(W, "r1_train"))
    train(os.path.join(W, "r1_train"), os.path.join(W, "ckpt_r1"))

    # ---- 풀 분류 & 저신뢰 표시
    sims, thr = pool_scores(os.path.join(W, "ckpt_r1"), [f for f, _ in pool])
    low = sims < thr
    log = {"seed": seed, "threshold_r1": thr, "pool_size": len(pool), "flagged": int(low.sum()),
           "flagged_by_true_class": {c: int(sum(1 for (f, cc), l in zip(pool, low) if l and cc == c))
                                     for c in classes_all}}

    # ---- 2라운드: 저신뢰 중 실제 클래스인 것을 클래스당 k2장
    r2_add = []
    for c in classes:
        cand = [(f, s) for (f, cc), s, l in zip(pool, sims, low) if l and cc == c]
        if a.pick == "hardest":
            cand.sort(key=lambda t: t[1])
            chosen = cand[:a.k2]
        else:
            chosen = rng.sample(cand, min(a.k2, len(cand)))
        if len(chosen) < a.k2:
            print(f"[경고] {c}: 저신뢰 후보가 {len(cand)}개뿐이라 {len(chosen)}장만 추가")
        r2_add += [(f, c) for f, _ in chosen]
    log["r2_added"] = {c: sum(1 for _, cc in r2_add if cc == c) for c in classes}

    # ---- 대조군: 풀에서 무작위 k2장 (저신뢰 여부 무관, 실제 클래스만)
    ctrl_add = []
    for c in classes:
        cand = [f for f, cc in pool if cc == c]
        ctrl_add += [(f, c) for f in rng.sample(cand, a.k2)]

    # ---- 보정용 폴더: 풀에서 r2/대조군에 쓴 이미지를 모두 제외 (Off-Target 포함)
    used = {f for f, _ in r2_add} | {f for f, _ in ctrl_add}
    copy_set([(f, c) for f, c in pool if f not in used], os.path.join(W, "calib"))

    copy_set(r1 + r2_add, os.path.join(W, "r2_train"))
    train(os.path.join(W, "r2_train"), os.path.join(W, "ckpt_r2"))
    if not a.skip_control:
        copy_set(r1 + ctrl_add, os.path.join(W, "ctrl_train"))
        train(os.path.join(W, "ctrl_train"), os.path.join(W, "ckpt_ctrl"))

    json.dump(log, open(os.path.join(W, "round2_log.json"), "w"), indent=2, ensure_ascii=False)
    print(json.dumps(log, indent=2, ensure_ascii=False))

    # ---- 평가
    calib = os.path.join(W, "calib")
    evaluate(os.path.join(W, "ckpt_r1"), os.path.join(W, "r1_train"), calib)
    evaluate(os.path.join(W, "ckpt_r2"), os.path.join(W, "r2_train"), calib)
    if not a.skip_control:
        evaluate(os.path.join(W, "ckpt_ctrl"), os.path.join(W, "ctrl_train"), calib)

print("\n요약:  python summarize.py \"./work/seed*/ckpt_r1/eval_test_crops.json\"  (r2, ctrl도 동일)")
