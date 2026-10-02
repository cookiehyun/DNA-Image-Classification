"""Off-Target 이미지에 자연스러운 하위 종류가 있는지 확인한다 (군집화 + 시각화).

하는 일
  1) 모델로 Off-Target 이미지와 정상 클래스 이미지(비교용 일부)의 특징을 뽑는다
  2) Off-Target 특징만 KMeans로 k=2..10 군집화하고 silhouette 점수를 비교한다
  3) 군집마다: 크기, 가장 닮은 정상 클래스, 그 클래스와의 유사도를 표로 출력
  4) 군집마다 이미지 몽타주(PNG)와 전체 2D 지도(t-SNE, PNG)를 저장 -> 눈으로 보고 이름 붙이기
  5) 이미지별 군집 번호를 CSV로 저장 (나중에 하위 클래스 라벨로 사용)

특징은 두 가지로 볼 수 있다
  --ckpt_dir 지정: 학습된 모델 특징 (3클래스 구분에 맞춰진 특징)
  --ckpt_dir 생략: 학습 전 ImageNet 특징 (Off-Target의 다양성이 더 잘 보존될 수 있음)

사용 예
  python explore_offtarget.py --ckpt_dir ./checkpoints/seed0 --out ./explore/finetuned
  python explore_offtarget.py --out ./explore/imagenet
"""
import argparse, glob, os, csv, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]  # Windows 한글 폰트
plt.rcParams["axes.unicode_minus"] = False

from src.model import Classifier
from src.data import UnlabeledDataset, get_eval_transform

p = argparse.ArgumentParser()
p.add_argument("--data_root", default="./si_dataset/shape/val_crops")
p.add_argument("--ckpt_dir", default=None, help="생략하면 ImageNet 사전학습 특징 사용")
p.add_argument("--ood_class", default="OffTarget")
p.add_argument("--n_ref", type=int, default=80, help="비교용 정상 클래스 이미지 수 (클래스당)")
p.add_argument("--k_min", type=int, default=2)
p.add_argument("--k_max", type=int, default=10)
p.add_argument("--k", type=int, default=None, help="군집 수 직접 지정 (생략 시 silhouette 최고값)")
p.add_argument("--montage_n", type=int, default=48, help="군집당 몽타주 이미지 수")
p.add_argument("--out", default="./explore/run")
p.add_argument("--gpu", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()
os.makedirs(a.out, exist_ok=True)
rng = random.Random(a.seed)
dev = torch.device(f"cuda:{a.gpu}" if torch.cuda.is_available() else "cpu")

# ---- 모델
if a.ckpt_dir:
    names = [l.strip() for l in open(sorted(glob.glob(os.path.join(a.ckpt_dir, "classes_*.txt")))[-1]) if l.strip()]
    model = Classifier(num_classes=len(names))
    model.load_state_dict(torch.load(sorted(glob.glob(os.path.join(a.ckpt_dir, "model_*.pth")))[-1], map_location="cpu"))
    feat_name = f"finetuned ({a.ckpt_dir})"
else:
    model = Classifier(num_classes=3)  # MLP는 쓰지 않음, ImageNet 백본만 사용
    feat_name = "ImageNet (학습 전)"
model.to(dev).eval()
print(f"특징: {feat_name}  device={dev}")

# ---- 데이터
classes = sorted(d for d in os.listdir(a.data_root) if os.path.isdir(os.path.join(a.data_root, d)) and d != a.ood_class)
off_paths = sorted(glob.glob(os.path.join(a.data_root, a.ood_class, "*.png")))
ref_paths, ref_y = [], []
for ci, c in enumerate(classes):
    fs = sorted(glob.glob(os.path.join(a.data_root, c, "*.png")))
    fs = rng.sample(fs, min(a.n_ref, len(fs)))
    ref_paths += fs; ref_y += [ci] * len(fs)
ref_y = np.array(ref_y)


def feats(paths):
    dl = DataLoader(UnlabeledDataset(paths, get_eval_transform(224)), batch_size=64, num_workers=0)
    out = []
    with torch.no_grad():
        for x, _ in dl:
            out.append(torch.nn.functional.normalize(model.extract_features(x.to(dev)), dim=1).cpu())
    return torch.cat(out).numpy()


Fo, Fr = feats(off_paths), feats(ref_paths)
protos = np.stack([Fr[ref_y == i].mean(0) for i in range(len(classes))])
protos /= np.linalg.norm(protos, axis=1, keepdims=True)
print(f"Off-Target {len(off_paths)}장, 비교용 정상 {len(ref_paths)}장 ({classes})")

# ---- 군집 수 탐색
print("\n[군집 수별 silhouette 점수] (높을수록 군집이 잘 나뉨; 0.1 미만이면 뚜렷한 군집이 없다는 뜻)")
scores = {}
for k in range(a.k_min, a.k_max + 1):
    lab = KMeans(k, n_init=10, random_state=a.seed).fit_predict(Fo)
    scores[k] = silhouette_score(Fo, lab, metric="cosine")
    print(f"  k={k:2d}  silhouette={scores[k]:.3f}")
k = a.k or max(scores, key=scores.get)
lab = KMeans(k, n_init=10, random_state=a.seed).fit_predict(Fo)
print(f"-> k={k} 사용")

# ---- 군집별 요약
sim = Fo @ protos.T
ref_self = (Fr @ protos.T).max(1)
print(f"\n참고: 정상 이미지의 최대 유사도 중앙값 = {np.median(ref_self):.3f}")
print(f"\n[군집별 요약]  (정상 클래스와 유사도가 정상 이미지 수준에 가까울수록 헷갈리는 군집)")
print(f"  {'군집':4s} {'개수':>4s}  {'가장 닮은 클래스(비율)':28s} {'최대 유사도 중앙값':>14s}  {'평균 크기(w×h)':>14s}")
rows = []
for c in range(k):
    m = lab == c
    near = sim[m].argmax(1)
    top = np.bincount(near, minlength=len(classes))
    top_c = top.argmax()
    sizes = np.array([Image.open(f).size for f in np.array(off_paths)[m]])
    desc = f"{classes[top_c]} ({top[top_c] / m.sum():.0%})"
    print(f"  {c:4d} {m.sum():4d}  {desc:28s} {np.median(sim[m].max(1)):14.3f}  {sizes[:,0].mean():6.0f}×{sizes[:,1].mean():<6.0f}")
    rows.append((c, int(m.sum()), desc))

# ---- 군집별 몽타주
T = 96
for c in range(k):
    fs = [f for f, l in zip(off_paths, lab) if l == c]
    fs = rng.sample(fs, min(a.montage_n, len(fs)))
    cols = 8; rws = (len(fs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * (T + 4), rws * (T + 4)), "white")
    for i, f in enumerate(fs):
        im = Image.open(f).convert("RGB"); im.thumbnail((T, T))
        canvas.paste(im, ((i % cols) * (T + 4), (i // cols) * (T + 4)))
    canvas.save(os.path.join(a.out, f"cluster_{c:02d}_n{int((lab == c).sum())}.png"))

# 비교용: 정상 클래스 몽타주도 한 장씩
for ci, cname in enumerate(classes):
    fs = [f for f, y in zip(ref_paths, ref_y) if y == ci][:24]
    canvas = Image.new("RGB", (8 * (T + 4), 3 * (T + 4)), "white")
    for i, f in enumerate(fs):
        im = Image.open(f).convert("RGB"); im.thumbnail((T, T))
        canvas.paste(im, ((i % 8) * (T + 4), (i // 8) * (T + 4)))
    canvas.save(os.path.join(a.out, f"ref_{cname}.png"))

# ---- 2D 지도 (t-SNE)
X = np.vstack([Fo, Fr])
Z = TSNE(2, metric="cosine", init="pca", perplexity=30, random_state=a.seed).fit_transform(X)
Zo, Zr = Z[:len(Fo)], Z[len(Fo):]
plt.figure(figsize=(9, 7))
for ci, cname in enumerate(classes):
    plt.scatter(*Zr[ref_y == ci].T, s=18, marker="x", label=f"[정상] {cname}")
cmap = plt.get_cmap("tab10")
for c in range(k):
    plt.scatter(*Zo[lab == c].T, s=14, color=cmap(c % 10), alpha=0.7, label=f"Off-Target 군집 {c}")
plt.legend(fontsize=8, loc="best"); plt.title(f"t-SNE: {feat_name}")
plt.tight_layout(); plt.savefig(os.path.join(a.out, "tsne.png"), dpi=150); plt.close()

# ---- CSV
with open(os.path.join(a.out, "clusters.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["path", "cluster", "nearest_class", "max_sim"])
    for pth, l, s in zip(off_paths, lab, sim):
        w.writerow([pth, int(l), classes[int(s.argmax())], round(float(s.max()), 4)])
print(f"\n저장 위치: {a.out}  (cluster_XX.png, ref_*.png, tsne.png, clusters.csv)")
