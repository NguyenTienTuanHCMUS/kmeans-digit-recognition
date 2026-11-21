# Handwritten digit recognition with K-means

**English** | [Tiếng Việt](README.vi.md)

This small project started as an idea that came up while learning **K-means clustering**: can K-means, an **unsupervised** algorithm that has no idea what a "digit" is, be used to recognize the digits a person draws?

It can, in two ways that this project tries one after the other:

| Approach | What K-means clusters | Correct on the MNIST test set |
|---|---|---:|
| 1. Whole-image templates | whole 28×28 images, so each centroid is a template digit | 92.15% (400 clusters) |
| 2. Feature learning | small 6×6 patches, so each centroid is a basic stroke | **98.54%** (400 strokes) |

Both interfaces (desktop and web) now use **approach 2**.

**Try it in your browser:** https://nguyentientuanhcmus.github.io/kmeans-digit-recognition/web/?lang=en (the page has a Tiếng Việt / English switch)

## Contents

- [Project layout](#project-layout)
- [Setup and running](#setup-and-running)
- [Training data](#training-data)
- [Approach 1: whole-image templates](#approach-1-whole-image-templates)
- [Approach 2: K-means as feature learning](#approach-2-k-means-as-feature-learning)
- [Comparing the two approaches](#comparing-the-two-approaches)
- [Normalizing the drawing: a small detail that matters](#normalizing-the-drawing-a-small-detail-that-matters)
- [Two interfaces](#two-interfaces)
- [Limitations](#limitations)
- [Credits and licenses](#credits-and-licenses)

## Project layout

```
├── train_kmeans.py      Approach 1: K-means on whole images, labels clusters, saves clusters.npz
├── train_features.py    Approach 2: K-means on 6×6 patches + logistic regression, saves features_k<K>.npz
├── digit_app.py         Tkinter interface with a 28×28 drawing board (uses approach 2)
├── export_web.py        Converts features_k400.npz into web/model.js for the web version
├── clusters.npz         Trained result of approach 1 (K = 400)
├── features_k400.npz    Trained result of approach 2 (400 strokes), used by both interfaces
├── requirements.txt
├── README.md            English README (this file)
├── README.vi.md         Vietnamese README
├── index.html           Redirects to web/ (for GitHub Pages)
├── MNIST/raw/*.gz       MNIST data (compressed; the programs read it directly)
└── web/
    ├── index.html       The browser version
    ├── model.js         The 400-stroke model, about 182 KB
    └── sakuga/          Sakuga VFX interface components (MIT)
```

## Setup and running

You need Python 3 (tested with 3.14):

```bash
pip install -r requirements.txt
```

NumPy and Pillow are enough to run the interfaces. scikit-learn is only needed to train approach 2.

**Training** (optional, since the trained models are already in the repo):

```bash
python train_kmeans.py --k 400       # approach 1, writes clusters.npz
python train_features.py --k 400     # approach 2, writes features_k400.npz and dictionary_k400.png
```

`train_features.py` also saves a picture of the learned stroke dictionary. A run takes from about half a minute (K = 50) to a few minutes on a CPU; the slowest part is describing 70,000 images with the dictionary.

**Desktop interface (Tkinter):**

```bash
python digit_app.py                              # uses features_k400.npz
python digit_app.py --model features_k200.npz    # or another trained dictionary
```

**Web interface:** use the [GitHub Pages version](https://nguyentientuanhcmus.github.io/kmeans-digit-recognition/web/?lang=en), or open `web/index.html` directly in a browser. No server and no internet connection needed. After retraining, run this to update the web version's model:

```bash
python export_web.py                             # or --model features_k200.npz
```

## Training data

- Both approaches train on the **60,000 MNIST training images**.
- The **10,000 test images** are kept apart and only used to measure accuracy. Every accuracy figure in this README is measured on them, unless it says otherwise.
- Approach 2 learns its stroke dictionary from **100,000 random 6×6 patches** cut from the training images (near-empty background patches are skipped). It then trains its classifier on all 60,000 training images.
- The drawings people make in the interfaces are never used for training.

## Approach 1: whole-image templates

### How it works

1. **Clustering (no labels).** Each 28×28 image is a point in 784-dimensional space. K-means splits the 60,000 points into K clusters. Each cluster's **centroid** is the average of its images and looks like a slightly blurry digit.
2. **Labeling the clusters.** Each cluster takes the label of the digit that appears most often in it (a majority vote).
3. **Recognition.** Find the nearest centroid (Euclidean distance); its label is the answer.

K-means itself stays unsupervised: it finds the "writing styles" on its own, and the labels only **name** what it found.

### The K-means implementation

K-means is written from scratch with NumPy (no scikit-learn), so every step is visible:

- **k-means++ initialization.** Each next starting centroid is picked with probability proportional to the squared distance to the nearest one already chosen. This spreads them apart, so the algorithm converges faster and more reliably.
- **Fast distances.** Uses `‖x − c‖² = ‖x‖² − 2·x·c + ‖c‖²`, so the `x·c` term for every (image, centroid) pair is a single 60,000 × 784 × K matrix multiplication. NumPy hands it to an optimized BLAS library: on the development machine that multiplication takes about 30 ms for K = 200.
- **Two alternating steps.** *Assignment:* each image joins the nearest centroid. *Update:* each centroid becomes the average of its images.
- **Empty clusters** get their centroid moved to the image currently farthest from its own centroid.
- **Stopping rule:** inertia drops by less than 0.01% in one iteration, or 100 iterations. With K = 200 it converges after 33 iterations, in about 12 seconds.

### Number of clusters vs. accuracy

Same random seed, only K changing:

| K (clusters) | Correct on the test set | |
|---:|---:|:---|
| 10 | 62.81% | `██████` |
| 20 | 71.17% | `███████████` |
| 50 | 81.17% | `████████████████` |
| 100 | 87.37% | `███████████████████` |
| 200 | 90.77% | `████████████████████` |
| 400 | **92.15%** | `█████████████████████` |

- **More clusters help, with diminishing returns:** 10 → 50 gains almost 20 points, 200 → 400 only about 1.4.
- **No clear overfitting:** with K = 200, 90.04% on the training set and 90.77% on the test set.
- **Harder digits get more clusters.** With K = 200, the clusters labeled as each digit are:

  | Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | Clusters | 23 | 10 | 30 | 23 | 17 | 21 | 21 | 15 | 20 | 20 |

- **The extreme case:** with K = 60,000 (one cluster per image) K-means just memorizes the data and becomes 1-nearest-neighbor, about 97% on MNIST according to the results table on Yann LeCun's MNIST page, but every prediction then compares against 60,000 images.

### Why not 10 clusters for 10 digits?

Because K-means only groups images whose pixels look alike, and two 4s in different styles can differ more than a 4 and a 9. With K = 10:

| Cluster | Label given | What is actually inside |
|---|---|---|
| 9 | 4 | 2,835 **4**s, 2,602 **9**s, 1,091 7s |
| 5 | 9 | 2,703 **9**s, 2,586 **4**s, 491 7s |
| 2 | 3 | 3,819 3s, 1,756 **5**s, 1,010 8s |
| 3 | 6 | 1,707 6s, 1,278 **5**s, 1,217 0s |
| 4 | 6 | 3,721 6s, 140 0s, 129 2s |

4s and 9s are mixed almost evenly, two clusters are labeled 6 and **none is labeled 5**, so the program can never answer 5: only 62.81% correct. Every digit has several writing styles, and a centroid that has to average all of them looks like none. More clusters give each style its own centroid, but each style still needs a template of its own. That limit is what approach 2 gets around.

## Approach 2: K-means as feature learning

Instead of learning whole digits, K-means learns an **alphabet of strokes**, and digits are described as combinations of strokes. The idea follows Coates, Lee & Ng (2011).

### How it works

1. **Cut patches.** 100,000 random 6×6 patches from the training images, skipping near-empty ones.
2. **Normalize and whiten.** Each patch has its mean brightness subtracted and is divided by its standard deviation, then all patches are whitened (ZCA) so they differ by shape rather than by brightness. Without this the centroids come out as blurry blobs.
3. **K-means on the patches** (the same implementation as approach 1). Each centroid is now a **basic stroke**: horizontal, vertical and diagonal lines at different angles, curves, stroke ends, corners. These K centroids are the **stroke dictionary**.
4. **Describe each image.** Slide a 6×6 window over all 23 × 23 positions of the image. At each position, measure how well the patch matches each stroke with the "triangle" encoding: `max(0, mean distance − distance to stroke k)`.
5. **Pool by region.** Sum those values in each quarter of the image. Each image becomes a vector of 4 × K numbers ("lots of curve in the top left, a vertical stroke in the bottom right..."). Pooling also makes it tolerant of drawings shifted by a few pixels.
6. **Classify.** A logistic regression (scikit-learn) is trained on those vectors with the labels, and outputs a probability for each digit.

```
Approach 1:  image 28×28 ──► compare with K digit templates ──► label of the nearest one

Approach 2:  image 28×28 ──► 6×6 patches ──► compare with K strokes ──► pool by quarter
                                                                         │
                                       vector of 4K numbers ──► logistic regression ──► digit
```

### Dictionary size vs. accuracy

| Strokes (K) | Correct on the training set | Correct on the test set |
|---:|---:|---:|
| 50 | 98.55% | 97.98% |
| 100 | 99.24% | 98.28% |
| 200 | 99.74% | 98.28% |
| 400 | 99.90% | **98.54%** |

- **50 strokes already beat 400 digit templates** (97.98% vs. 92.15%). Strokes are shared by every digit and every writing style, so there is no need for a template per style.
- **Diminishing returns again:** 100 → 400 strokes adds only about 0.3 points.
- **Some overfitting appears:** with 400 strokes the model almost memorizes the training set (99.90%) but gets 98.54% on the test set. Approach 1 shows no such gap because it is much simpler.
- K-means stays unsupervised here as well: it decides *which strokes exist*. The labels are used by the classifier to learn *which strokes, where, make which digit*. This is close to what the first layer of a convolutional neural network does, except that its filters are found by K-means instead of by gradient descent.

## Comparing the two approaches

| | Approach 1: templates | Approach 2: strokes |
|---|---|---|
| Correct on the test set | 92.15% | **98.54%** |
| What a centroid looks like | a whole (blurry) digit | a 6×6 stroke |
| How labels are used | to name each cluster | to train a classifier on the stroke features |
| Work per prediction | compare with 400 templates | compare 529 patches with 400 strokes, then classify |
| Easy to explain an answer | yes: "it looks most like this template" | less so: "these strokes appear in these places" |

## Normalizing the drawing: a small detail that matters

MNIST images were made with a fixed procedure: the digit is scaled to fit a 20×20 box (keeping its aspect ratio), then placed in a 28×28 image so that its **center of mass** sits in the middle. A person's drawing can be small, large or off to one side, so both interfaces normalize it the same way before recognizing it:

1. Crop tightly around the strokes.
2. Scale to fit a 20×20 box, keeping the aspect ratio.
3. Place it in a 28×28 image and shift it so the center of mass is at **(14, 14)**.

Step 3 taught a lesson during approach 1. At first the center of mass was placed at (13.5, 13.5), the geometric center of the grid in pixel-index terms. That half-pixel offset alone dropped accuracy on 2,000 test images from 87.65% to 82.6%. Switching to (14, 14) made it match exactly: the whole test set run through the normalization still gave 90.78%, versus 90.77% on the original images (K = 200). For an algorithm that compares images pixel by pixel, half a pixel matters.

## Two interfaces

Both interfaces use approach 2 with the 400-stroke dictionary.

### Desktop: Tkinter (`digit_app.py`)

- A 28×28 drawing board. Left mouse button draws, right button erases. The brush is dark in the middle and fades at the edges, like MNIST strokes.
- Let go of the mouse to get the answer with its confidence, the normalized image, the most used stroke and a probability bar for each digit. About 3 ms per drawing. The interface text is in Vietnamese.

### Web: `web/index.html`

- Plain HTML, CSS and JavaScript. Everything is computed in the browser, and it works offline.
- Two languages: a **Tiếng Việt / English** switch at the top. On the first visit the page follows the browser's language, then remembers the choice; `?lang=en` or `?lang=vi` in the link opens a language directly.
- Recognizes while you draw (can be turned off), with a mouse, a pen or touch.
- Shows the probability of each digit, the 5 strokes your drawing uses most (the strokes that best match the most inked spots), and the whole 400-stroke dictionary with the 20 most used strokes outlined.
- The interface uses [Sakuga VFX](https://github.com/finktheartist/sakuga-vfx): buttons, tabs, a slider, a switch and a tooltip that react like anime (spring squash, smoke, impact bursts, ドン! lettering). The answer box is a custom component in the same style that fires an effect whenever the answer changes on release. Effects switch off when the device has reduced motion turned on.
- **Checked:** on the first 1,000 test images the JavaScript version gives the same answer as the Python version for all 1,000, with probabilities within 0.0000005 of each other. One prediction takes about 30 ms in the browser, since every drawing compares 529 patches with 400 strokes.

Keyboard shortcuts: `B` to draw, `E` to erase, `C` to clear the board.

## Limitations

- **Still not state of the art.** Convolutional neural networks score above 99% on MNIST. This project is about understanding K-means, not about reaching the highest accuracy.
- **Drawings differ from MNIST.** The model was trained on scanned handwriting, so very thin, very thick, rotated or unusually styled drawings can still be misread.
- **Slower on the web than approach 1** (about 30 ms instead of well under 1 ms per prediction). Drawing stays smooth, but a slow phone may stutter with "Recognize while drawing" on.
- The web version has only been tried in Chromium-based browsers. According to its author, Sakuga VFX has not been tried in Safari or Firefox yet.

## Credits and licenses

- **MNIST data** by Yann LeCun, Corinna Cortes and Christopher J.C. Burges, released under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
- **Approach 2** follows Adam Coates, Honglak Lee and Andrew Y. Ng, "An Analysis of Single-Layer Networks in Unsupervised Feature Learning" (AISTATS 2011).
- **Sakuga VFX** by Justin Finkelman, MIT license. A copy of the license is in [`web/sakuga/LICENSE`](web/sakuga/LICENSE).
