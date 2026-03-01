import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import numpy as np
import torch
import soundfile as sf

from audio_preprocessing.custom_tokenizer import create_tokenizer
from audio_preprocessing.fast_fourier_transformation import FastFourierTransformation
from audio_preprocessing.mel_spectrogram import MelSpectrogram
from train import SpeechToText   # model class only — we use our own decode functions


# ─────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_audio(path: str):
    """Load a WAV file → mono float32 numpy array + sample rate."""
    samples, sr = sf.read(path)
    if samples.ndim > 1:
        samples = np.mean(samples, axis=1)
    return samples.astype(np.float32), sr


def audio_to_mel(audio_np: np.ndarray, n_mels: int = 80) -> torch.Tensor:
    """
    STFT → mel filterbank.
    Returns FloatTensor of shape [1, n_mels, T]  (ready to feed the encoder).
    """
    fft     = FastFourierTransformation(sample_rate=16000, n_fft=512, hop_length=160, win_length=400)
    mel_obj = MelSpectrogram(n_fft=512)

    magnitudes = fft.apply_fft(audio_np)              # (T, 257)
    log_mel    = mel_obj.build_mel_spectrogram(magnitudes)  # (T, 80)

    return torch.tensor(log_mel, dtype=torch.float32).T.unsqueeze(0)  # (1, 80, T)


# ─────────────────────────────────────────────────────────────────────────────
# Decode functions  (self-contained — do NOT call train.py's methods)
# Vocab: [PAD]=0, A-Z=1-26, space=27, apostrophe=28  — no BOS/EOS token.
# Strategy: seed the decoder with PAD (0); stop when the model predicts PAD
# again after at least one real character has been emitted.
# ─────────────────────────────────────────────────────────────────────────────

PAD_ID = 0

@torch.no_grad()
def greedy_decode(model, mel: torch.Tensor, tokenizer, max_len: int = 150, seed_id: int = PAD_ID) -> str:
    model.eval()
    device      = mel.device
    encoder_out = model.encoder(mel)                       # (1, T', d_model)

    tokens = torch.tensor([[seed_id]], device=device)
    result = [seed_id]

    for _ in range(max_len):
        logits   = model.decoder(tokens, encoder_out)      # (1, cur_len, vocab)
        next_id  = int(logits[:, -1, :].argmax(dim=-1))

        if next_id == PAD_ID and len(result) > 1:
            break   # first PAD after real output = end of sequence
            
        # simple stuttering prevention: if last 4 tokens are identical, stop
        if len(result) > 4 and result[-1] == next_id and result[-2] == next_id and result[-3] == next_id:
            break

        result.append(next_id)
        tokens = torch.cat(
            [tokens, torch.tensor([[next_id]], device=device)], dim=1
        )

    valid = [t for t in result if t != PAD_ID]
    return tokenizer.decode(valid) if valid else ""


@torch.no_grad()
def beam_search_decode(model, mel: torch.Tensor, tokenizer,
                       beam_size: int = 5, max_len: int = 150) -> str:
    model.eval()
    device      = mel.device
    encoder_out = model.encoder(mel)                       # (1, T', d_model)

    # Since there's no BOS token trained, seed with *all* possible characters
    # to let the model decide the best start token.
    beams     = [(0.0, [i]) for i in range(1, 29)]
    completed = []

    for _ in range(max_len):
        candidates = []

        for score, seq in beams:
            # End if PAD is generated after real tokens
            if seq[-1] == PAD_ID and len(seq) > 1:
                completed.append((score, seq))
                continue
            
            # Repetition / stutter penalty: if repeating same sequence of words, stop it
            if len(seq) > 12:
                # e.g., " THE THE THE"
                if seq[-4:] == seq[-8:-4]:
                    completed.append((score, seq))
                    continue

            tokens_t  = torch.tensor([seq], device=device)
            logits    = model.decoder(tokens_t, encoder_out)
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

            topk_vals, topk_ids = log_probs[0].topk(beam_size)
            for val, tid in zip(topk_vals, topk_ids):
                candidates.append((score + val.item(), seq + [tid.item()]))

        if not candidates:
            break

        beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]

    completed += beams
    # Length-normalised score; strip PAD from final output
    best_seq = max(completed, key=lambda x: x[0] / max(len(x[1]), 1))[1]
    valid    = [t for t in best_seq if t != PAD_ID]
    return tokenizer.decode(valid) if valid else ""


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_model(checkpoint_path: str, device: torch.device):
    tokenizer = create_tokenizer()

    model = SpeechToText(
        vocab_size=29, n_mels=80, d_model=512, n_heads=8,
        ff_dim=2048, enc_layers=6, dec_layers=6, dropout=0.1,
    ).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint — epoch {ckpt.get('epoch','?')}, "
              f"loss {ckpt.get('loss','?')}")
    else:
        model.load_state_dict(ckpt)
        print("  Loaded raw state-dict.")

    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────────────────────
# WER
# ─────────────────────────────────────────────────────────────────────────────

def word_error_rate(reference: str, hypothesis: str) -> float:
    ref = reference.upper().split()
    hyp = hypothesis.upper().split()
    r, h = len(ref), len(hyp)
    dp = list(range(h + 1))
    for i in range(1, r + 1):
        new = [i] + [0] * h
        for j in range(1, h + 1):
            if ref[i-1] == hyp[j-1]:
                new[j] = dp[j-1]
            else:
                new[j] = 1 + min(dp[j], new[j-1], dp[j-1])
        dp = new
    return dp[h] / max(r, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Test routines
# ─────────────────────────────────────────────────────────────────────────────

def test_single_file(model, tokenizer, audio_path: str,
                     reference, device, beam_size: int = 5):
    print(f"\n{'─'*60}")
    print(f"Audio : {os.path.basename(audio_path)}")

    audio_np, sr = load_audio(audio_path)
    dur = len(audio_np)/sr
    print(f"  Duration : {dur:.2f}s  |  Sample rate : {sr} Hz")

    mel = audio_to_mel(audio_np).to(device)          # (1, 80, T)
    
    # Calc max len based on a generous 25 chars/sec
    max_len = max(int(dur * 25), 30)

    # For greedy, if we have a reference, supply its first letter so it has a valid seed
    seed_id = PAD_ID
    if reference:
        enc = tokenizer.encode(reference.upper()).ids
        if enc: seed_id = enc[0]

    greedy_text = greedy_decode(model, mel, tokenizer, max_len=max_len, seed_id=seed_id)
    beam_text   = beam_search_decode(model, mel, tokenizer, beam_size=beam_size, max_len=max_len)

    print(f"  Greedy  : {greedy_text!r}")
    print(f"  Beam({beam_size}) : {beam_text!r}")

    if reference:
        ref = reference.upper()
        print(f"  Ref     : {ref!r}")
        print(f"  WER (greedy): {word_error_rate(ref, greedy_text):.2%}  "
              f"WER (beam): {word_error_rate(ref, beam_text):.2%}")

    return greedy_text, beam_text


def test_from_training_data(model, tokenizer, device, n_samples: int = 5, beam_size: int = 5):
    import pandas as pd

    base      = os.path.dirname(__file__)
    json_path = os.path.join(base, "training_data", "speech.json")

    if not os.path.exists(json_path):
        print("speech.json not found — skipping training-data test.")
        return

    df      = pd.read_json(json_path, lines=True)
    samples = df.sample(min(n_samples, len(df)), random_state=42)

    wers = []
    for _, row in samples.iterrows():
        path = os.path.join(base, "training_data",
                            row["audio_path"].replace("/", os.sep))
        ref  = row["text"]
        if not os.path.exists(path):
            print(f"  Skipping missing: {path}")
            continue
        _, beam_text = test_single_file(model, tokenizer, path, ref, device, beam_size)
        wers.append(word_error_rate(ref.upper(), beam_text))

    if wers:
        print(f"\n{'─'*60}")
        print(f"Average WER over {len(wers)} samples: {sum(wers)/len(wers):.2%}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Test a trained SpeechToText model")
    parser.add_argument("--checkpoint", "-c",
                        default=os.path.join(os.path.dirname(__file__), "model_final.pt"),
                        help="Path to .pt checkpoint (default: model_final.pt)")
    parser.add_argument("--audio", "-a", nargs="*",
                        help="WAV file(s) to transcribe")
    parser.add_argument("--reference", "-r", nargs="*",
                        help="Reference transcription(s), one per --audio file")
    parser.add_argument("--n-samples", "-n", type=int, default=5,
                        help="Samples from training data if --audio omitted (default: 5)")
    parser.add_argument("--beam-size", "-b", type=int, default=5,
                        help="Beam width (default: 5)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  SPEECH-TO-TEXT — MODEL TEST")
    print("=" * 60)
    print(f"  Device     : {device}")
    print(f"  Checkpoint : {args.checkpoint}")

    if not os.path.exists(args.checkpoint):
        print(f"\nERROR: checkpoint not found — {args.checkpoint}")
        return

    print("\nLoading model...")
    model, tokenizer = load_model(args.checkpoint, device)
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")

    if args.audio:
        refs = args.reference or []
        for i, path in enumerate(args.audio):
            ref = refs[i] if i < len(refs) else None
            test_single_file(model, tokenizer, path, ref, device, args.beam_size)
    else:
        print(f"\nNo --audio files given — sampling {args.n_samples} from training data...\n")
        test_from_training_data(model, tokenizer, device, args.n_samples, args.beam_size)

    print(f"\n{'='*60}")
    print("  Test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
