# 🔍 Vision Transformer (ViT) — CIFAR-10 Image Classifier

A from-scratch implementation of the **Vision Transformer (ViT)** architecture using TensorFlow/Keras, trained on the CIFAR-10 dataset. Based on the paper [*An Image is Worth 16x16 Words*](https://arxiv.org/abs/2010.11929) by Dosovitskiy et al.

---

## 📌 Overview

This project implements a ViT that:
- Splits images into fixed-size **patches**
- Linearly projects and **positionally encodes** each patch
- Passes the sequence through stacked **Transformer encoder blocks**
- Uses an **MLP classification head** to predict one of 10 CIFAR-10 classes

---

## 🗂️ Project Structure

```
.
├── ViT.py               # Full model definition, training, and evaluation
├── tmp/
│   └── checkpoint.weights.h5   # Best model weights (auto-saved during training)
└── README.md
```

---

## ⚙️ Architecture

| Component            | Details                              |
|----------------------|--------------------------------------|
| Input shape          | 32 × 32 × 3 (resized to 72 × 72)    |
| Patch size           | 6 × 6                                |
| Number of patches    | 144                                  |
| Projection dim       | 64                                   |
| Transformer layers   | 8                                    |
| Attention heads      | 4                                    |
| Transformer MLP units| [128, 64]                            |
| Classifier MLP units | [2048, 1024]                         |
| Output classes       | 10                                   |

---

## 🧪 Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) — 60,000 color images across 10 classes:

`airplane · automobile · bird · cat · deer · dog · frog · horse · ship · truck`

| Split      | Samples |
|------------|---------|
| Training   | 50,000  |
| Test       | 10,000  |

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/your-username/vit-cifar10.git
cd vit-cifar10
```

### 2. Install dependencies

```bash
pip install tensorflow numpy matplotlib
```

> Python 3.8+ and TensorFlow 2.x recommended.

### 3. Run training

```bash
python ViT.py
```

This will:
- Download and preprocess CIFAR-10 automatically
- Apply data augmentation (flip, rotation, zoom, normalization)
- Train for 5 epochs with checkpointing
- Evaluate and print test accuracy

---

## 📊 Training Configuration

| Hyperparameter   | Value   |
|------------------|---------|
| Learning rate    | 0.001   |
| Weight decay     | 0.0001  |
| Batch size       | 256     |
| Epochs           | 5       |
| Optimizer        | AdamW   |
| Loss             | Sparse Categorical Crossentropy |

---

## 📈 Sample Output

```
x_train shape: (50000, 32, 32, 3) - y_train shape: (50000, 1)
x_test shape: (10000, 32, 32, 3)  - y_test shape: (10000, 1)

Epoch 1/5 - loss: 2.1234 - accuracy: 0.2341 - val_accuracy: 0.3012
...
Test accuracy: ~55–60%
Top 5 test accuracy: ~90–95%
```

> ⚠️ ViT models generally require large datasets and longer training to reach peak accuracy. Fine-tuning from pretrained weights or training for more epochs will significantly improve results.

---

## 🧠 Key Concepts

- **Patch Embedding** — Images are split into non-overlapping patches and linearly projected into a latent space.
- **Positional Encoding** — Learned position embeddings are added to preserve spatial order.
- **Multi-Head Self-Attention** — Each patch attends to every other patch globally.
- **MLP Head** — Final representation is flattened and passed through dense layers for classification.

---

## 📄 Reference

```bibtex
@article{dosovitskiy2020image,
  title={An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale},
  author={Dosovitskiy, Alexey and others},
  journal={arXiv preprint arXiv:2010.11929},
  year={2020}
}
```

---

## 🪪 License

This project is open-source under the [MIT License](LICENSE).
