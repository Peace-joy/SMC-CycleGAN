
# SMC-CycleGAN: Self-Attention and Multi-Scale Enhanced CycleGAN for Style Transfer

This repository contains the official PyTorch implementation of **SMC-CycleGAN**, an enhanced framework for cross-domain image style transfer. Our method introduces two key components — a **Self-Attention Feature Module (SAFM)** and a **Multi-Scale Feature Module (MSFM)** — to improve feature learning, structure retention, and translation fidelity in high-resolution artistic image transfer.

> 📝 *This code is directly related to our manuscript currently submitted to* ***The Visual Computer***. *If you find this work helpful, please consider citing our paper.*

---

## 📁 Project Structure

```
├── SAFM.py              # Self-Attention Feature Module
├── base_model.py        # Base class for models
├── cycle_gan_model.py   # Modified CycleGAN with SAFM and MSFM
├── networks.py          # Generator and discriminator architectures
├── self.py              # Multi-Scale Feature Module
├── train.py             # Training entry point
├── test.py              # Testing script
├── test_model.py        # Model evaluation utilities
├── environment.yml      # Conda environment configuration
├── requirements.txt     # pip dependencies
├── .gitignore
├── .replit
├── LICENSE
└── README.md
```
  
---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/<your_username>/SMC-CycleGAN.git
cd SMC-CycleGAN
```

### 2. Create Environment

Using conda:

```bash
conda env create -f environment.yml
conda activate smc-cycle
```

Or use `pip`:

```bash
pip install -r requirements.txt
```

---


## 🗂️ Datasets

You can download the public datasets used in this project from the following sources:

1. [WikiArt Dataset](https://www.wikiart.org/)
2. [Kaggle Dataset](https://www.kaggle.com/)
3. [Art2Photo on AI Studio](https://aistudio.baidu.com/global/search?keyword=art2photo&tab=ALL)
4. [MS-COCO Dataset](https://paperswithcode.com/dataset/coco)
|-datasets
  |-photos2image
    |-trainA
    |-trainB
    |-testA
    |-testB
---

## 🚀 Usage

### ✅ Train the Model

```bash
python train.py --dataroot ./datasets/photos2image --name transfer --model cycle_gan --direction BtoA
```

Training configuration (epochs, batch size, learning rate, dataset path) can be set directly in `train.py`.

### 🧪 Test the Model

```bash
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/photos2image --name transfer --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot ./datasets/photos2image/testA --name transfer --model test --no_dropout
```

Evaluation mode will generate style-transferred images for the test dataset.

---

## 🧠 Core Contributions

- **Self-Attention**: Enhances global contextual awareness using self-attention after the encoder.
- **Multi-Scale Feature Extraction**: Extracts features at multiple scales to retain both low- and high-frequency information.
- **CycleGAN Backbone**: Built upon CycleGAN with modular improvements for better domain translation.

---

## 📄 Paper

> "This code is directly related to our manuscript currently submitted to *The Visual Computer*. If you find this work helpful, please consider citing our paper."


---

## 📜 License

This project is licensed under the terms of the MIT license. See `LICENSE` for details.
