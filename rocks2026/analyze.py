"""Make it right.

Every number here is on rocks (or rock pairs) the model was never fitted on.
Same-image pairs are excluded by construction: the 30x30 human matrix has a
diagonal and it is never read. The 2016 flaw is the headline test, not a footnote.

Inputs: features.npz (from embed.py), resources/nosofsky.txt (30x30, 80 subjects),
resources/transformedratings.txt (360 rocks x 15 rated physical dimensions),
loader.vals (which 30 of the 360 the 30x30 matrix refers to).
Outputs: rocks2026/results.json, rocks2026/fig_*.png, printed summary.
"""
import os, json, numpy as np
from numpy.linalg import lstsq
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.model_selection import GroupKFold, LeaveOneOut
from PIL import Image
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, '..')
F = np.load(os.path.join(HERE, 'features.npz'))
X = np.concatenate([F['cls'], F['mean']], 1); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
names, cats, paths = list(F['names']), list(F['cats']), list(F['paths'])
catset = sorted(set(cats)); cat_idx = np.array([catset.index(c) for c in cats])

# the 30 rocks behind the 30x30 matrix (1-based in the 2016 loader), in sorted-glob order which matches the 2016 loader's order
vals = [6, 14, 34, 39, 59, 69, 73, 95, 102, 119, 127, 144, 151, 164, 172, 188, 201, 213, 218, 237, 250, 263, 267, 284, 300, 309, 314, 336, 348, 352]
t30 = [v - 1 for v in vals]
S = np.loadtxt(os.path.join(ROOT, 'resources/nosofsky.txt'), skiprows=2, usecols=range(1, 31))   # 1..9.99 similarity
R = np.loadtxt(os.path.join(ROOT, 'resources/transformedratings.txt'))[:, 3:]                     # 360 x 15 human-rated dims
res = {}

def thumb(p, px=64):
    im = Image.open(p).convert('RGBA'); bg = Image.new('RGBA', im.size, (255,255,255,255))
    im = Image.alpha_composite(bg, im).convert('RGB'); im.thumbnail((px, px)); return im

# ---------- Test A: predict human similarity for pairs of rocks the model never saw ----------
iu = np.triu_indices(30, 1)                       # 435 distinct pairs; the diagonal is never read
hum = S[iu]
X30 = X[t30]
# A1 zero-shot: cosine similarity in the network's space, no fitting at all
Xn = X30 / np.linalg.norm(X30, axis=1, keepdims=True)
cos = (Xn @ Xn.T)[iu]
res['A1_zero_shot'] = {'pearson': pearsonr(cos, hum)[0], 'spearman': spearmanr(cos, hum)[0], 'n_pairs': 435}
# A1b same thing in raw pixel space (what the 2016 code fed the network: squared pixel differences)
pix = np.stack([np.asarray(thumb(p, 32).convert('L').resize((32, 32)), dtype=float).ravel() for p in [paths[i] for i in t30]])
pix = (pix - pix.mean(1, keepdims=True)); pixn = pix / np.linalg.norm(pix, axis=1, keepdims=True)
res['A1b_pixel_baseline'] = {'pearson': pearsonr((pixn @ pixn.T)[iu], hum)[0]}
# A2 learned metric, leave-one-ROCK-out: hold out every pair containing rock k, fit on the rest, predict the held-out pairs
D = np.abs(X30[iu[0]] - X30[iu[1]])            # pair features 435 x 768
pred = np.zeros(435)
for k in range(30):
    test = (iu[0] == k) | (iu[1] == k)
    m = Ridge(alpha=3000.0).fit(D[~test], hum[~test])
    pred[test] = m.predict(D[test])              # each pair is predicted twice (once per rock); keep the mean
# (average the two predictions per pair)
pred2 = np.zeros(435); cnt = np.zeros(435)
for k in range(30):
    test = (iu[0] == k) | (iu[1] == k)
    m = Ridge(alpha=3000.0).fit(D[~test], hum[~test]); pred2[test] += m.predict(D[test]); cnt[test] += 1
pred = pred2 / cnt
res['A2_learned_metric_held_out_rock'] = {'pearson': pearsonr(pred, hum)[0], 'spearman': spearmanr(pred, hum)[0]}
# A3 the 2016 flaw, reproduced on purpose: include same-image pairs (similarity 9.99) in the evaluation and watch r jump
diag = np.arange(30)
D_all = np.concatenate([D, np.zeros((30, D.shape[1]))]); hum_all = np.concatenate([hum, S[diag, diag]]); pred_all = np.concatenate([pred, np.full(30, 9.99)])
res['A3_with_same_image_pairs_the_2016_flaw'] = {'pearson': pearsonr(pred_all, hum_all)[0], 'note': 'identical images predicted identical; r inflates. This is the number the other professor caught.'}

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(hum, pred, s=14, alpha=.6, color='#2b6cb0')
ax.plot([hum.min(), hum.max()], [hum.min(), hum.max()], '--', color='#999', lw=1)
ax.set_xlabel('human similarity, 80 subjects (1 = nothing alike, 10 = same)'); ax.set_ylabel('predicted, rock held out of the fit')
ax.set_title(f"435 pairs of 30 rocks, every pair predicted with one of its rocks unseen\nr = {res['A2_learned_metric_held_out_rock']['pearson']:.2f}")
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_A_heldout_similarity.png'), dpi=130); plt.close(fig)

# ---------- Test B: which physical descriptions can be read off the network's space, on rock TYPES it never saw ----------
gkf = GroupKFold(n_splits=10)
B = []
for d in range(R.shape[1]):
    y = R[:, d]; p = np.zeros(360)
    for tr, te in gkf.split(X, y, groups=cat_idx):        # whole categories held out
        p[te] = Ridge(alpha=300.0).fit(X[tr], y[tr]).predict(X[te])
    B.append({'dim': d + 1, 'r_heldout_category': float(pearsonr(p, y)[0]), 'binary': bool(y.max() <= 1.0)})
res['B_attributes_from_features'] = B

# simple image statistics, to put names on what we can name without a human
def stats(p):
    im = np.asarray(thumb(p, 96).convert('RGB'), dtype=float) / 255
    mask = im.mean(2) < 0.98                              # off-white = rock
    px = im[mask] if mask.sum() > 50 else im.reshape(-1, 3)
    mx, mn = px.max(1), px.min(1)
    g = np.asarray(thumb(p, 96).convert('L'), dtype=float)
    return [px.mean(), (mx - mn).mean(), g[mask].std() if mask.sum() > 50 else g.std()]
ST = np.array([stats(p) for p in paths])                  # lightness, saturation, local contrast
stat_names = ['mean lightness', 'mean saturation', 'grey-level std (texture)']
auto = {}
for d in range(R.shape[1]):
    auto[d + 1] = {stat_names[j]: float(pearsonr(ST[:, j], R[:, d])[0]) for j in range(3)}
res['B_dims_vs_image_stats'] = auto

# ---------- Test C: Sanders & Nosofsky 2020, without training a network: MDS coords predicted for a held-out rock ----------
dis = (10 - S); np.fill_diagonal(dis, 0)
mds = MDS(n_components=8, dissimilarity='precomputed', random_state=0, n_init=8, max_iter=2000)
Y = mds.fit_transform(dis)                                # 30 x 8 psychological space from humans only
res['C_mds_stress'] = float(mds.stress_)
C = []
for d in range(8):
    p = np.zeros(30)
    for k in range(30):
        tr = np.arange(30) != k
        p[k] = Ridge(alpha=1000.0).fit(X30[tr], Y[tr, d]).predict(X30[[k]])[0]
    C.append({'mds_dim': d + 1, 'r_leave_one_rock_out': float(pearsonr(p, Y[:, d])[0]), 'variance': float(Y[:, d].var())})
res['C_mds_dims_from_features'] = C
# what each MDS dim correlates with among the 15 rated dims (on the same 30 rocks)
R30 = R[t30]
res['C_mds_dim_vs_rated'] = [{'mds_dim': d + 1, 'best_rated_dim': int(np.argmax(np.abs([pearsonr(Y[:, d], R30[:, j])[0] for j in range(15)])) + 1),
                              'r': float(max([pearsonr(Y[:, d], R30[:, j])[0] for j in range(15)], key=abs))} for d in range(8)]

fig, ax = plt.subplots(figsize=(11, 8.5))
for i, k in enumerate(t30):
    ab = AnnotationBbox(OffsetImage(thumb(paths[k], 70), zoom=1), (Y[i, 0], Y[i, 1]), frameon=False); ax.add_artist(ab)
    ax.annotate(cats[k].split('_', 1)[1], (Y[i, 0], Y[i, 1]), xytext=(0, -42), textcoords='offset points', ha='center', fontsize=7, color='#444')
ax.set_xlim(Y[:, 0].min() - 1.2, Y[:, 0].max() + 1.2); ax.set_ylim(Y[:, 1].min() - 1.2, Y[:, 1].max() + 1.2)
ax.set_title('the 30 rocks placed by 80 people (multidimensional scaling, first two of eight dimensions)'); ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig(os.path.join(HERE, 'fig_C_human_mds_map.png'), dpi=120); plt.close(fig)

# ---------- D: characteristics that fall out of the order, unsupervised, and then get named ----------
pca = PCA(n_components=8, random_state=0).fit(X); P = pca.transform(X)
res['D_pca_explained'] = [float(v) for v in pca.explained_variance_ratio_]
Dn = []
for c in range(6):
    order = np.argsort(P[:, c]); lo, hi = order[:10], order[-10:][::-1]
    rs = [pearsonr(P[:, c], R[:, j])[0] for j in range(15)]
    j = int(np.argmax(np.abs(rs)))
    st = {stat_names[k]: float(pearsonr(P[:, c], ST[:, k])[0]) for k in range(3)}
    Dn.append({'pc': c + 1, 'best_rated_dim': j + 1, 'r_with_rated': float(rs[j]), 'image_stats': st,
               'low_end': [names[i] for i in lo], 'high_end': [names[i] for i in hi]})
    sheet = Image.new('RGB', (10 * 100, 2 * 100 + 30), 'white')
    for r_, idxs in enumerate([lo, hi]):
        for c_, i in enumerate(idxs):
            sheet.paste(thumb(paths[i], 96), (c_ * 100 + 2, r_ * 100 + 30 + 2))
    from PIL import ImageDraw
    ImageDraw.Draw(sheet).text((6, 8), f'network axis {c+1}: top row = one end, bottom row = the other end. no human label used to make this order.', fill='black')
    sheet.save(os.path.join(HERE, f'fig_D_axis{c+1}.png'))
res['D_axes'] = Dn

# ---------- E: the fair version of what 2016 tried: same-category vs different-category, pairs of unseen rocks ----------
# For every pair of the 360 (64,620 pairs, no diagonal), is cosine similarity higher for same-category pairs? AUC-style number.
Xa = X / np.linalg.norm(X, axis=1, keepdims=True); CS = Xa @ Xa.T
iu360 = np.triu_indices(360, 1); same = (cat_idx[iu360[0]] == cat_idx[iu360[1]]); cs = CS[iu360]
from sklearn.metrics import roc_auc_score
res['E_same_category_auc_zero_shot'] = float(roc_auc_score(same, cs))
# and 1-nearest-neighbour category accuracy, leave-one-out (the classifier nobody trained)
np.fill_diagonal(CS, -1); nn = CS.argmax(1); res['E_1nn_category_accuracy_leave_one_out'] = float((cat_idx[nn] == cat_idx).mean())
res['E_chance'] = 1 / 30

json.dump(res, open(os.path.join(HERE, 'results.json'), 'w'), indent=1, default=float)
print(json.dumps({k: v for k, v in res.items() if not k.startswith('D_axes')}, indent=1, default=float))
