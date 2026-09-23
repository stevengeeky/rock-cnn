"""demo: hand it a rock, it hands back the rocks the network thinks look like it, and
where that rock sits on the axes that fell out of the network's own order.

    python3 rocks2026/demo.py                 # a random rock
    python3 rocks2026/demo.py Obsidian_04     # by name (any unique substring)
Writes rocks2026/demo.png. Never compares a rock with its own picture.
"""
import sys, os, random, numpy as np
from PIL import Image, ImageDraw
from sklearn.decomposition import PCA
HERE = os.path.dirname(os.path.abspath(__file__))
F = np.load(os.path.join(HERE, 'features.npz')); names, cats, paths = list(F['names']), list(F['cats']), list(F['paths'])
X = np.concatenate([F['cls'], F['mean']], 1); X = (X - X.mean(0)) / (X.std(0) + 1e-6)
Xn = X / np.linalg.norm(X, axis=1, keepdims=True); CS = Xn @ Xn.T; np.fill_diagonal(CS, -9)
P = PCA(n_components=4, random_state=0).fit_transform(X)
axes = [('grain: speckled/coarse  <->  smooth/glassy', -1), ('grain: mottled/mixed  <->  uniform slab', -1), ('lightness: black/shiny  <->  pale/matte', 1), ('organisation: jumbled fragments  <->  banded/layered', 1)]
q = [i for i, n in enumerate(names) if len(sys.argv) > 1 and sys.argv[1].lower() in n.lower()]
q = q[0] if q else random.randrange(360)
nn = np.argsort(-CS[q])[:6]
def thumb(p, px=140):
    im = Image.open(p).convert('RGBA'); bg = Image.new('RGBA', im.size, (255,255,255,255)); im = Image.alpha_composite(bg, im).convert('RGB'); im.thumbnail((px, px)); return im
W = 7 * 150 + 20; sheet = Image.new('RGB', (W, 150 + 40 + 4 * 22 + 20), 'white'); d = ImageDraw.Draw(sheet)
sheet.paste(thumb(paths[q]), (10, 30)); d.text((10, 8), f'{names[q]}   (this rock)', fill='black')
d.text((170, 8), 'the six rocks the network puts nearest, never having seen a label. green = same category', fill='black')
for k, i in enumerate(nn):
    x = 170 + k * 150; sheet.paste(thumb(paths[i]), (x, 30)); same = cats[i] == cats[q]
    d.text((x, 175), f'{names[i]}  {CS[q, i]:.2f}', fill=(0, 120, 0) if same else (150, 0, 0))
y = 200
for k, (label, sign) in enumerate(axes):
    pct = (P[:, k] * sign < P[q, k] * sign).mean() * 100
    bar = int(pct / 100 * 400); d.text((10, y), label, fill='black'); d.rectangle([420, y + 3, 820, y + 13], outline='#999'); d.rectangle([420, y + 3, 420 + bar, y + 13], fill='#2b6cb0'); d.text((830, y), f'{pct:3.0f}th of 360', fill='black'); y += 22
sheet.save(os.path.join(HERE, 'demo.png'))
print(names[q], '->', ', '.join(f'{names[i]}({"same" if cats[i]==cats[q] else "other"})' for i in nn))
