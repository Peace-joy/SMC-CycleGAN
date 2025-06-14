
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

## 🚀 Usage

### ✅ Train the Model

```bash
python train.py
```

Training configuration (epochs, batch size, learning rate, dataset path) can be set directly in `train.py`.

### 🧪 Test the Model

```bash
python test.py
```

Evaluation mode will generate style-transferred images for the test dataset.

---

## 🧠 Core Contributions

- **SAFM**: Enhances global contextual awareness using self-attention after the encoder.
- **MSFM**: Extracts features at multiple scales to retain both low- and high-frequency information.
- **CycleGAN Backbone**: Built upon CycleGAN with modular improvements for better domain translation.

---

## 📄 Paper

> "This code is directly related to our manuscript currently submitted to *The Visual Computer*. If you find this work helpful, please consider citing our paper."


---

## 📜 License

This project is licensed under the terms of the MIT license. See `LICENSE` for details.
