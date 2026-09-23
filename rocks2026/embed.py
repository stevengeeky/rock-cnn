"""Embed every rock in collection/ with a pretrained self-supervised vision model.

No training on rocks, no labels, no human ratings touched here. The network has
never seen a rock label. Whatever structure comes out is what a general-purpose
eye already carries. facebook/dino-vits16: ViT-S/16, DINO self-supervised on
ImageNet-1k images (no labels), 21.8M params. Output: rocks2026/features.npz
"""
import os, sys, glob, time, numpy as np, torch
from PIL import Image
from transformers import ViTModel, ViTFeatureExtractor

root = os.path.join(os.path.dirname(__file__), '..', 'collection')
paths = sorted(glob.glob(os.path.join(root, '*', '*', '*.png')))
assert len(paths) == 360, len(paths)
names = [os.path.splitext(os.path.basename(p))[0] for p in paths]   # e.g. I_Andesite_01
cats  = [n.rsplit('_', 1)[0] for n in names]                          # e.g. I_Andesite

torch.set_num_threads(8)
model = ViTModel.from_pretrained('facebook/dino-vits16').eval()
proc  = ViTFeatureExtractor.from_pretrained('facebook/dino-vits16')

def load(p):
    im = Image.open(p).convert('RGBA')
    bg = Image.new('RGBA', im.size, (255, 255, 255, 255))   # rocks are cut out on transparent; put them on white like the rating study
    return Image.alpha_composite(bg, im).convert('RGB')

cls, mean = [], []
t = time.time()
with torch.no_grad():
    for i in range(0, 360, 12):
        ims = [load(p) for p in paths[i:i+12]]
        out = model(**proc(images=ims, return_tensors='pt')).last_hidden_state   # [b, 197, 384]
        cls.append(out[:, 0].numpy()); mean.append(out[:, 1:].mean(1).numpy())
        print(f'{i+12}/360  {time.time()-t:.0f}s', flush=True)
cls, mean = np.concatenate(cls), np.concatenate(mean)
np.savez(os.path.join(os.path.dirname(__file__), 'features.npz'), cls=cls, mean=mean, names=np.array(names), cats=np.array(cats), paths=np.array(paths))
print('saved', cls.shape, mean.shape)
