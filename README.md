# Char-Level Transformer Language Model

A minimal, from-scratch implementation of a decoder-only Transformer (GPT-style) trained to generate text character by character.

The model learns to predict the next character in a sequence, trained here on the Tiny Shakespeare dataset.

## Features

- Self-attention (`Head`) and multi-head attention (`MultiHeadAttention`) implemented from scratch
- Transformer blocks with residual connections, LayerNorm, and a feed-forward MLP
- Character-level tokenizer (simple stoi/itos mapping, no BPE)
- Train/validation split with periodic loss evaluation
- Autoregressive text generation

## Requirements

- Python 3.8+
- [PyTorch](https://pytorch.org/) (CUDA optional, falls back to CPU automatically)

Install PyTorch:
```bash
pip install torch
```

## Dataset

This script trains on the Tiny Shakespeare dataset. Download it into the same directory as the script:

```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## Usage

Once `input.txt` is present, run:

```bash
python model.py
```

The script will:
1. Load and tokenize the text at the character level
2. Split data into 90% train / 10% validation
3. Train the model for `max_iters` steps, printing train/val loss every `eval_interval` steps
4. Generate 500 new characters of text from the trained model and print them to the console

## Hyperparameters

| Parameter       | Value | Description                              |
|-----------------|-------|-------------------------------------------|
| `batch_size`    | 32    | Sequences processed in parallel           |
| `block_size`    | 8     | Max context length for predictions        |
| `max_iters`     | 5000  | Total training iterations                 |
| `eval_interval` | 500   | Steps between loss evaluations            |
| `learning_rate` | 1e-3  | AdamW learning rate                       |
| `eval_iters`    | 200   | Batches averaged per loss estimate        |
| `n_embd`        | 32    | Embedding dimension                       |
| `n_head`        | 6     | Number of attention heads per block       |
| `n_layer`       | 6     | Number of Transformer blocks              |
| `dropout`       | 0.2   | Dropout rate                              |

Feel free to tune these at the top of the script — larger `n_embd`, `n_layer`, and `block_size` values generally produce more coherent (but slower to train) generations.

## Project Structure

```
.
├── model.py       # Model definition, training loop, and generation
├── input.txt      # Training corpus (Tiny Shakespeare, downloaded separately)
└── README.md
```