# Handwritten digit recognition with K-means

**English** | [Tiếng Việt](README.vi.md)

This small project started as an idea that came up while learning **K-means clustering**: if K-means groups the 60,000 MNIST digit images into clusters, and each cluster is then given a name, can those clusters tell which digit a person draws?

They can, and fairly well. With 400 clusters the program gets **92.15%** of the MNIST test images right, even though K-means is an **unsupervised** algorithm and has no idea what a "digit" is.

**Try it in your browser:** https://nguyentientuanhcmus.github.io/kmeans-digit-recognition/web/?lang=en (the page has a Tiếng Việt / English switch)

## Contents

- [The idea](#the-idea)
- [Project layout](#project-layout)
- [Setup and running](#setup-and-running)
- [The algorithm in detail](#the-algorithm-in-detail)
- [Number of clusters vs. accuracy](#number-of-clusters-vs-accuracy)
- [Why not 10 clusters for 10 digits?](#why-not-10-clusters-for-10-digits)
- [Normalizing the drawing: a small detail that matters](#normalizing-the-drawing-a-small-detail-that-matters)
- [Two interfaces](#two-interfaces)
- [Limitations](#limitations)
- [Credits and licenses](#credits-and-licenses)

## The idea

The whole pipeline has 3 steps:

1. **Clustering (no labels).** Each 28×28 image is a point in 784-dimensional space. K-means splits the 60,000 points into K clusters. Each cluster has a **centroid**: the average of every image in it, which looks like a slightly blurry digit.
2. **Labeling the clusters.** Only now are the training labels used: each cluster takes the label of the digit that appears most often in it (a majority vote).
3. **Recognition.** For a new drawing, find the nearest centroid (Euclidean distance). That cluster's label is the answer.

So K-means itself stays unsupervised: it finds the "writing styles" on its own. The labels are only used to **name** what K-means found.

## Project layout

```
├── train_kmeans.py      Program 1: runs K-means, labels the clusters, saves clusters.npz
├── digit_app.py         Program 2: Tkinter interface with a 28×28 drawing board
├── export_web.py        Converts clusters.npz into web/clusters.js for the web version
├── clusters.npz         Pre-trained result (K = 400)
├── requirements.txt
├── README.md            English README (this file)
├── README.vi.md         Vietnamese README
├── index.html           Redirects to web/ (for GitHub Pages)
├── MNIST/raw/*.gz       MNIST data (compressed; the program reads it directly)
├── web/
│   ├── index.html       The browser version
│   ├── clusters.js      400 quantized centroids, about 414 KB
│   └── sakuga/          Sakuga VFX interface components (MIT)
```

## Setup and running

You need Python 3 (tested with 3.14) and two libraries:

```bash
pip install -r requirements.txt
```

**Training** (optional, since the repo already includes `clusters.npz` with K = 400):

```bash
python train_kmeans.py            # K = 200 by default
python train_kmeans.py --k 400    # more clusters, more accurate
```

With K = 200, training takes about 12 seconds on a CPU and converges after 33 iterations.

**Desktop interface (Tkinter):**

```bash
python digit_app.py
```

**Web interface:** use the [GitHub Pages version](https://nguyentientuanhcmus.github.io/kmeans-digit-recognition/web/?lang=en), or open `web/index.html` directly in a browser. No server and no internet connection needed. After retraining, run this to update the web version's data:

```bash
python export_web.py
```

## The algorithm in detail

K-means is implemented from scratch with NumPy (no scikit-learn), so every step is visible:

- **k-means++ initialization.** The first centroid is picked at random. Each next centroid is picked with probability proportional to the squared distance to the nearest centroid chosen so far. This spreads the starting centroids apart, so the algorithm converges faster and more reliably than with a purely random start.
- **Fast distances.** Uses `‖x − c‖² = ‖x‖² − 2·x·c + ‖c‖²`. The `x·c` term is computed for every (image, centroid) pair at once with a single 60,000 × 784 × K matrix multiplication.
- **Two alternating steps.** *Assignment:* each image joins the cluster with the nearest centroid. *Update:* each centroid becomes the average of the images in its cluster.
- **Empty clusters.** If a cluster ends up with no images, its centroid is moved to the image that is currently farthest from its own centroid.
- **Stopping rule.** Stop when the inertia (sum of squared distances) drops by less than 0.01% in one iteration, or after 100 iterations.

## Number of clusters vs. accuracy

Results on the **10,000 MNIST test images**, with the same random seed and only K changing:

| K (clusters) | Correct on the test set | |
|---:|---:|:---|
| 10 | 62.81% | `██████` |
| 20 | 71.17% | `███████████` |
| 50 | 81.17% | `████████████████` |
| 100 | 87.37% | `███████████████████` |
| 200 | 90.77% | `████████████████████` |
| 400 | **92.15%** | `█████████████████████` |

What this shows:

- **More clusters means higher accuracy, with diminishing returns.** Going from 10 to 50 clusters gains almost 20 percentage points; going from 200 to 400 gains only about 1.4.
- **No clear overfitting.** With K = 200, accuracy is 90.04% on the training set and 90.77% on the test set, almost the same.
- **Cluster purity.** With K = 200, on average 89.9% of the images in a cluster are the same digit.
- **Harder digits get more clusters.** With K = 200, the number of clusters labeled as each digit is:

  | Digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
  |---|---|---|---|---|---|---|---|---|---|---|
  | Clusters | 23 | 10 | 30 | 23 | 17 | 21 | 21 | 15 | 20 | 20 |

  The digit 2 has the most writing styles (with or without a loop at the bottom, curved or angular), so it takes 30 clusters. The digit 1 is simple and needs only 10.

- **The extreme case.** With K = 60,000 (every image its own cluster), K-means no longer "learns" anything; it simply **memorizes the whole dataset**. At that point it is the 1-nearest-neighbor algorithm (1-NN), which reaches about 97% on MNIST according to the results table on Yann LeCun's MNIST page. But each prediction then compares against 60,000 images instead of a few hundred centroids. K = 200 to 400 is a good balance between accuracy, speed and file size.

## Why not 10 clusters for 10 digits?

Because **K-means knows nothing about "digits"**. It only groups images whose pixels look alike. Two 4s written in different styles can differ more, pixel by pixel, than a 4 and a 9.

Here is what some clusters actually contain when K = 10:

| Cluster | Label given | What is actually inside |
|---|---|---|
| 9 | 4 | 2,835 **4**s, 2,602 **9**s, 1,091 7s |
| 5 | 9 | 2,703 **9**s, 2,586 **4**s, 491 7s |
| 2 | 3 | 3,819 3s, 1,756 **5**s, 1,010 8s |
| 3 | 6 | 1,707 6s, 1,278 **5**s, 1,217 0s |
| 4 | 6 | 3,721 6s, 140 0s, 129 2s |

- **4s and 9s are mixed almost evenly**, because their pixels are very similar.
- **Two clusters are both labeled 6**, while **no cluster is labeled 5**. So the program can never answer 5.
- Result: only **62.81%** correct.

The deeper reason is that **every digit has several writing styles**: a 1 can be straight or slanted, a 7 can have a crossbar or not, a 4 can be open or closed at the top. A centroid is the **average** of its cluster. If one centroid has to stand for every style of 4, it becomes a blurry shape that looks like none of them. With more clusters, each style gets its own centroid. You can see this in the "centroids K-means found" section of the web version: each row is a digit, each tile a writing style.

## Normalizing the drawing: a small detail that matters

MNIST images were made with a fixed procedure: the digit is scaled to fit a 20×20 box (keeping its aspect ratio), then placed in a 28×28 image so that its **center of mass** sits in the middle. A person's drawing can be small, large or off to one side, so both interfaces normalize it the same way before comparing it with the centroids:

1. Crop tightly around the strokes.
2. Scale to fit a 20×20 box, keeping the aspect ratio.
3. Place it in a 28×28 image and shift it so the center of mass is at **(14, 14)**.

Step 3 taught an interesting lesson. At first the center of mass was placed at (13.5, 13.5), the geometric center of the 28×28 grid in pixel-index terms. That half-pixel offset alone dropped accuracy on 2,000 test images from 87.65% to 82.6%. Switching to (14, 14) made it match exactly: running the whole test set through the normalization still gives 90.78%, versus 90.77% on the original images (K = 200). For an algorithm that compares images pixel by pixel, like K-means here, half a pixel matters.

## Two interfaces

### Desktop: Tkinter (`digit_app.py`)

- A 28×28 drawing board. Left mouse button draws, right button erases. The brush is dark in the middle and fades at the edges, like MNIST strokes.
- Let go of the mouse to get the answer, along with the normalized image, the nearest centroid and the 5 nearest clusters. The interface text is in Vietnamese.

### Web: `web/index.html`

- Plain HTML, CSS and JavaScript. Everything is computed in the browser, and it works offline.
- Two languages: a **Tiếng Việt / English** switch at the top of the page. On the first visit the page follows the browser's language, then remembers the choice. A language can be opened directly with `?lang=en` or `?lang=vi` in the link.
- Recognizes while you draw (can be turned off), and works with a mouse, a pen or touch.
- Distance bars for each digit 0–9: see which digit your drawing is closest to, and which one is the runner-up.
- A gallery of every centroid, grouped by digit. The closest cluster is highlighted.
- The interface uses [Sakuga VFX](https://github.com/finktheartist/sakuga-vfx): buttons, tabs, a slider, a switch and a tooltip that react like anime (spring squash, smoke, impact bursts, ドン! lettering). The answer box is a custom component in the same style that fires an effect whenever the answer changes on release. Effects switch off when the device has reduced motion turned on.
- **Checked:** the JavaScript recognizer gives the same answer as the Python version on all 1,000 of 1,000 test images tried, at about 0.3 ms per image. Centroids are quantized to 1 byte per pixel to keep the file small. On the 10,000 test images this changes only 3 predictions (92.15% becomes 92.14%).

Keyboard shortcuts: `B` to draw, `E` to erase, `C` to clear the board.

## Limitations

- **Pixel-by-pixel comparison.** Euclidean distance does not understand shape. Digits drawn very thin, very thick, rotated, or in an unusual style (for example a 1 with a base, or a 7 with a crossbar) can be misread.
- **Telling 4 from 9 and 3 from 5** is still a weak spot, as the K = 10 breakdown above shows.
- **Not a "real" classifier.** Supervised methods (SVMs, convolutional neural networks) score above 99% on MNIST. This project is about understanding K-means, not about reaching the highest accuracy.
- The web version has only been tried in Chromium-based browsers. According to its author, Sakuga VFX has not been tried in Safari or Firefox yet.

## Credits and licenses

- **MNIST data** by Yann LeCun, Corinna Cortes and Christopher J.C. Burges, released under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
- **Sakuga VFX** by Justin Finkelman, MIT license. A copy of the license is in [`web/sakuga/LICENSE`](web/sakuga/LICENSE).
