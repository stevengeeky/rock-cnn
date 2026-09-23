# rock-cnn, ten years on

Summer 2016 asked one question: can a neural network's view of 360 rock photographs be
made to give up the physical dimensions people use when they say two rocks look alike, so
that those dimensions can be named? This folder answers it, on this laptop, in one evening,
with the rule the 2016 code broke made into the headline test: **every number below is
scored on rocks or rock pairs the model was never fitted on, and a rock is never compared
with its own picture.**

Method. `embed.py` runs every rock through a self-supervised vision transformer
(DINO ViT-S/16, trained on ImageNet photographs without labels). It has never seen a rock
label, a similarity rating, or a category name. `analyze.py` then asks the human data the
questions, always with something held out. Rocks are pasted on white, as in the rating study.
No network is trained anywhere in this folder.

Data (all in `resources/`, all from Nosofsky's lab, 2016): the 30 x 30 similarity matrix,
80 subjects, one representative rock per category; 360 x 15 mean ratings of physical
dimensions; the 360 cut-out photographs.

## A. Does the network's space predict how similar people say two rocks look?

435 distinct pairs of the 30 representative rocks. The diagonal is never read.

| how the pair is scored | r with the 80 subjects |
|---|---|
| squared pixel differences (what 2016 fed its net) | 0.13 |
| cosine in the network's space, nothing fitted | 0.51 |
| a linear metric fitted on the other 29 rocks, this rock unseen | **0.76** |
| the same metric with same-image pairs added to the evaluation | 0.87 |

The last row is the 2016 flaw reproduced on purpose. Adding pairs of a rock with itself,
which any method predicts perfectly, lifts the number by 0.11 without the method getting
any better. That is the gap the other professor saw. The honest number is 0.76.

`fig_A_heldout_similarity.png`

## B. Which physical descriptions can be read off the network's space?

For each of the 15 human-rated dimensions, a ridge regression from the network's features,
scored with **whole rock categories held out** (ten folds, three categories per fold), so
the rock being scored comes from a type the fit never saw.

| rated dimension (named from the extreme rocks) | r, held-out categories |
|---|---|
| lightness of colour | 0.92 |
| average grain size | 0.89 |
| organisation, layered vs random | 0.88 |
| variability of colour | 0.88 |
| physical layers present | 0.88 |
| roughness | 0.86 |
| fragment size | 0.86 |
| visible grain present | 0.84 |
| fragments present | 0.84 |
| shininess | 0.83 |
| flat cleavage | 0.73 |
| holes / porous | 0.68 |
| veins / bands | 0.62 |
| conchoidal fracture | 0.60 |
| fragment roundness | 0.29 |

Fourteen of fifteen at r >= 0.60 without the network ever being told any of these words.
Lightness is also 0.93 correlated with plain mean pixel brightness, so that one is cheap.
Grain, organisation, layers, roughness and shininess are not readable from brightness,
saturation or contrast (all |r| < 0.45); the network carries them as structure.

## C. Sanders & Nosofsky 2020, without training anything

Multidimensional scaling of the 30 x 30 matrix into eight dimensions (the algorithm Rob
liked, the one you couldn't name tonight). Then each MDS coordinate predicted from the
network's features for a held-out rock.

| MDS dimension | r, leave-one-rock-out | what it tracks among the rated dims |
|---|---|---|
| 1 | 0.71 | organisation (-0.59) |
| 2 | 0.78 | grain size (0.75) |
| 3 | 0.39 | roughness (0.63) |
| 4 | 0.68 | lightness (-0.68) |
| 5 | 0.71 | lightness (-0.41) |
| 6 | 0.48 | organisation (-0.43) |
| 7 | 0.63 | holes (-0.43) |
| 8 | 0.72 | fragment roundness (-0.39) |

Their 2020 paper trained an ensemble of CNNs on the MDS coordinates. Here, seven of eight
coordinates come out of a frozen general-purpose net at r >= 0.48 with thirty data points.

`fig_C_human_mds_map.png` is the 30 rocks placed by the 80 people, first two dimensions.

## D. Characteristics that fall out of the order, and then get named

This was the actual hope. Take the network's rock-space with no human data at all, find its
principal axes, and look at the ten rocks at each end.

| network axis | what the eye sees at the two ends | rated dim it best matches |
|---|---|---|
| 1 | coarse speckled granite-like  vs  smooth, glassy, shiny | visible grain (-0.54) |
| 2 | mottled mixed-grain  vs  uniform flat slabs | grain size (-0.53) |
| 3 | black and shiny  vs  pale and matte | lightness (0.65) |
| 4 | jumbled fragments, breccia  vs  banded, layered, foliated | organisation (0.72) |

`fig_D_axis1.png` through `fig_D_axis6.png` are the contact sheets. The order was made
by the network; the names were put on afterwards by looking, which is exactly the
procedure Rob described: group them, see what they share, then say the word.

## E. The classifier nobody trained

Nearest neighbour in the network's space, leave-one-out over all 360 rocks, puts a rock
in its correct one of 30 categories 45% of the time (chance 3%), and ranks a same-category
pair above a different-category pair 83% of the time. No training, no labels.

## What this is not

Not a replication of the paper's numbers, which used more subjects and the 360 x 360
sparse matrix this repo never had. Not tested on anything but rocks; the pipeline takes any
folder of images and any ratings file, and that generalisation is the next thing to try.
The dimension names in B are read off the extreme rocks, not from the lab's codebook.

## Files

`embed.py` (5 min on the CPU), `analyze.py` (1 min), `features.npz`, `results.json`,
`fig_A_heldout_similarity.png`, `fig_C_human_mds_map.png`, `fig_D_axis1..6.png`.
