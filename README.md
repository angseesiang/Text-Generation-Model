# ✨ Text Generation Model (GPT-2)

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-red)](#)
[![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-green)](#)

This repository contains a **GPT-2 based text generation model** with simple training, saving, loading, and generation utilities.  
It also includes **unit tests** to verify functionality.

---

## 📖 Contents

- `text_generation_model.py` – Main implementation of the text generation model  
- `test_text_generation_model.py` – Unit tests for training, saving, and generation  
- `requirements.txt` – Project dependencies  
- `readme.md` – Documentation and usage guide  

---

## 🚀 How to Use

### 1) Clone this repository

```bash
git clone https://github.com/angseesiang/Text_Generation_Model.git
cd Text_Generation_Model
```

### 2) Create and activate a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate   # On Linux / macOS
venv\Scripts\activate      # On Windows
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Run unit tests

```bash
python -m unittest test_text_generation_model.py
```

If everything is configured correctly, you should see test results confirming the model works.

### 5) Use the model

Run the model with a sample prompt:

```bash
python text_generation_model.py --prompt "Once upon a time"
```

This will generate text based on the GPT-2 model.

---

## 🛠️ Requirements

- Python 3.9+
- PyTorch
- Hugging Face Transformers
- Datasets
- tf-keras

All dependencies are listed in `requirements.txt`.

---

## 📌 Notes

- The model is based on GPT-2 (`gpt2`) and uses Hugging Face Transformers.  
- Text generation supports **greedy decoding** (default) and **sampling** with `--sample`, `--temperature`, `--top_p`, `--top_k`.  
- Models can be saved and reloaded with `save_model(path)` and `load_model(path)`.  

---

## 📜 License

This project is for **educational purposes only**.
