# 🎙️ Speech-to-Text from Scratch

An end-to-end **speech-to-text** system built entirely from scratch using PyTorch — no pre-trained ASR models, no high-level libraries. Every component (FFT, mel filterbanks, acoustic encoder, transformer decoder) is implemented manually.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Preparing Training Data](#preparing-training-data)
- [Training](#training)
- [Testing & Inference](#testing--inference)
- [Model Checkpoints](#model-checkpoints)
- [Configuration](#configuration)

---

## Overview

This project implements a complete automatic speech recognition (ASR) pipeline:

1. **Audio Preprocessing** — Raw `.wav` audio is converted to log-mel spectrograms using a hand-written FFT and mel filterbank (no `torchaudio` / `librosa`).
2. **Acoustic Encoder** — A convolutional downsampler compresses the mel spectrogram in time, followed by a stack of Transformer encoder layers with sinusoidal positional encoding.
3. **Transformer Decoder** — An autoregressive decoder with causal self-attention and encoder cross-attention generates character-level tokens.
4. **Custom Tokenizer** — A character-level BPE tokenizer (A–Z, space, apostrophe) built with HuggingFace `tokenizers`.

The model is trained with teacher forcing and cross-entropy loss, and supports both **greedy** and **beam-search** decoding at inference time.

---

## Architecture

```
Audio (.wav)
    │
    ▼
┌──────────────────────────┐
│  Fast Fourier Transform  │   ← hand-written STFT (512-pt FFT, 25 ms window, 10 ms hop)
└──────────────────────────┘
    │  (T, 257) magnitudes
    ▼
┌──────────────────────────┐
│    Mel Filterbank (80)   │   ← 80-channel log-mel spectrogram
└──────────────────────────┘
    │  (T, 80)
    ▼
┌──────────────────────────┐
│ Convolutional Downsampler│   ← residual Conv1D blocks with strides [4, 4, 6]
└──────────────────────────┘
    │  (T', 512)
    ▼
┌──────────────────────────┐
│  Transformer Encoder ×6  │   ← multi-head self-attention + FFN + sinusoidal PE
└──────────────────────────┘
    │  encoder_out (T', 512)
    ▼
┌──────────────────────────┐
│  Transformer Decoder ×6  │   ← causal self-attn + cross-attn + FFN
└──────────────────────────┘
    │  logits (seq_len, 29)
    ▼
  Predicted Text
```

| Hyperparameter | Value |
|---|---|
| `d_model` | 512 |
| `n_heads` | 8 |
| `ff_dim` | 2048 |
| `encoder layers` | 6 |
| `decoder layers` | 6 |
| `vocab_size` | 29 (`[PAD]` + A–Z + space + `'`) |
| `dropout` | 0.1 |

---

## Project Structure

```
SpeechToText/
├── audio_preprocessing/           # Audio feature extraction
│   ├── fast_fourier_transformation.py   # Hand-written STFT
│   ├── mel_spectrogram.py               # Mel filterbank + log compression
│   ├── custom_tokenizer.py              # Character-level BPE tokenizer
│   ├── dataset.py                       # SpeechDataset (PyTorch-compatible)
│   └── test_steps.py                    # Unit test for the preprocessing pipeline
│
├── acoustic_encoder/              # Encoder network
│   ├── downsampler.py                   # Conv1D downsampling blocks
│   ├── transformer_with_attn.py         # Sinusoidal PE + Transformer encoder block
│   └── full_encoder.py                  # Full acoustic encoder (downsampler + transformer)
│
├── decoder/                       # Decoder network
│   └── decoder_transformer.py           # Transformer decoder with cross-attention
│
├── training_data/                 # Dataset directory
│   ├── audio/                           # .wav audio files
│   └── speech.json                      # Transcription metadata (JSONL)
│
├── checkpoints/                   # Saved model checkpoints (every 5 epochs)
├── train.py                       # Training pipeline
├── test.py                        # Inference & evaluation (WER)
├── main.py                        # Entry point (placeholder)
├── tokenizer.json                 # Saved tokenizer vocabulary
├── model_final.pt                 # Final trained model weights
├── requirements.txt               # Python dependencies
├── pyproject.toml                 # Project metadata
└── .gitignore
```

---

## Setup & Installation

### Prerequisites

- **Python 3.13+**
- **pip** (or **uv** — this project includes a `uv.lock`)
- A CUDA-capable GPU is recommended but not required (CPU training is supported)

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/speech-to-text-from-scratch.git
cd speech-to-text-from-scratch
```

### 2. Create a virtual environment

```bash
# Using venv
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS / Linux)
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** If you are using **uv**, you can alternatively run:
> ```bash
> uv sync
> ```

---

## Preparing Training Data

The training data should be placed in the `training_data/` directory:

```
training_data/
├── audio/          # WAV files (16 kHz mono recommended)
│   ├── 0001.wav
│   ├── 0002.wav
│   └── ...
└── speech.json     # JSONL file with transcription metadata
```

Each line in `speech.json` should be a JSON object with the following fields:

```json
{"audio_path": "audio/0001.wav", "text": "hello world"}
```

| Field | Description |
|---|---|
| `audio_path` | Relative path to the audio file (from `training_data/`) |
| `text` | Ground-truth transcription |

---

## Training

Run the training pipeline:

```bash
python train.py
```

### What happens during training:

1. All `.wav` files are loaded and converted to log-mel spectrograms
2. The `SpeechToText` model (encoder + decoder) is initialized
3. Training runs for **50 epochs** with:
   - **Adam optimizer** (`lr=1e-4`)
   - **StepLR scheduler** (halves LR every 3 epochs)
   - **Gradient clipping** (max norm = 1.0)
4. Checkpoints are saved every **5 epochs** to `checkpoints/`
5. Final model is saved as `model_final.pt`

### Training output example:

```
================================================================
SPEECH-TO-TEXT TRAINING PIPELINE
================================================================
Device: cuda
Vocab size: 29
Model: d_model=512, n_heads=8, enc_layers=6, dec_layers=6

Step 1: Loading and preprocessing audio data...
Found 1296 audio files
...
Epoch 1 completed | Avg Loss: 3.2451 | LR: 0.000050
```

---

## Testing & Inference

### Evaluate on training samples

```bash
python test.py
```

This will randomly sample 5 examples from the training data and show greedy + beam-search transcriptions alongside the reference text and WER (Word Error Rate).

### Transcribe custom audio files

```bash
python test.py --audio path/to/audio1.wav path/to/audio2.wav
```

### Transcribe with reference text (to compute WER)

```bash
python test.py --audio path/to/audio.wav --reference "THE EXPECTED TRANSCRIPTION"
```

### All CLI options

| Flag | Default | Description |
|---|---|---|
| `--checkpoint`, `-c` | `model_final.pt` | Path to the model checkpoint |
| `--audio`, `-a` | *(none)* | One or more `.wav` files to transcribe |
| `--reference`, `-r` | *(none)* | Reference text(s) for WER computation |
| `--n-samples`, `-n` | `5` | Number of training samples to evaluate (if no `--audio`) |
| `--beam-size`, `-b` | `5` | Beam width for beam-search decoding |

---

## Model Checkpoints

Checkpoints are saved during training in `checkpoints/`:

| File | Description |
|---|---|
| `model_epoch_5.pt` | Checkpoint at epoch 5 (dict with `model_state_dict`, `optimizer_state_dict`, `epoch`, `loss`) |
| `model_epoch_10.pt` | Checkpoint at epoch 10 |
| ... | Every 5 epochs |
| `model_final.pt` | Final model weights only (`state_dict`) |

To resume or test from a specific checkpoint:

```bash
python test.py --checkpoint checkpoints/model_epoch_50.pt
```

---

## Configuration

Key hyperparameters are defined at the top of `train.py` in the `main()` function:

```python
vocab_size    = 29
n_mels        = 80
d_model       = 512
n_heads       = 8
ff_dim        = 2048
enc_layers    = 6
dec_layers    = 6
dropout       = 0.1
learning_rate = 1e-4
num_epochs    = 50
batch_size    = 4
```

Audio preprocessing parameters in `FastFourierTransformation`:

```python
sample_rate = 16000   # 16 kHz
n_fft       = 512     # 32 ms FFT window
hop_length  = 160     # 10 ms stride
win_length  = 400     # 25 ms analysis window
```

---

## License

This project is for educational and research purposes.
