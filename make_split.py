"""Validation 크롭에서 클래스당 k장을 떼어 학습 세트를 만들고, 나머지를 검증 세트로 둔다.

사용 예:
    python make_split.py --src ./si_dataset/shape/val_crops --out ./data --k 20 --seeds 0 1 2
결과:
    data/seed0/train/{Circle,DoubleLoop,Lemniscate}/   (클래스당 k장)
    data/seed0/val/{Circle,DoubleLoop,Lemniscate,OffTarget}/   (학습에 쓴 이미지 제외)
"""
import argparse, os, random, shutil, glob

p = argparse.ArgumentParser()
p.add_argument("--src", default="./si_dataset/shape/val_crops")
p.add_argument("--out", default="./data")
p.add_argument("--k", type=int, default=20, help="클래스당 학습 이미지 수")
p.add_argument("--seeds", type=int, nargs="+", default=[0])
p.add_argument("--ood_class", default="OffTarget", help="학습에서 제외할 클래스 폴더명")
a = p.parse_args()

classes = sorted(d for d in os.listdir(a.src) if os.path.isdir(os.path.join(a.src, d)))
for seed in a.seeds:
    rng = random.Random(seed)
    root = os.path.join(a.out, f"seed{seed}")
    if os.path.exists(root):
        shutil.rmtree(root)
    for c in classes:
        files = sorted(glob.glob(os.path.join(a.src, c, "*.png")))
        train = set(rng.sample(files, a.k)) if c != a.ood_class else set()
        for f in files:
            split = "train" if f in train else "val"
            dst = os.path.join(root, split, c)
            os.makedirs(dst, exist_ok=True)
            shutil.copy2(f, dst)
    n = {s: {c: len(glob.glob(os.path.join(root, s, c, "*.png"))) for c in classes}
         for s in ["train", "val"]}
    print(f"[seed {seed}] train={n['train']}  val={n['val']}")
