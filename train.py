import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import torch
import torch.nn as nn
import pandas as pd

from audio_preprocessing.custom_tokenizer import create_tokenizer
from audio_preprocessing.dataset import SpeechDataset
from audio_preprocessing.fast_fourier_transformation import FastFourierTransformation
from audio_preprocessing.mel_spectrogram import MelSpectrogram

from acoustic_encoder.full_encoder import AcousticEncoder
from decoder.decoder_transformer import TransformerDecoderBlock, make_causal_mask, TransformerDecoder

PAD_TOKEN_ID = 0


def load_and_apply_preprocessing_to_audio():
    audio_dir = os.path.join(os.path.dirname(__file__), 'training_data', 'audio')

    if not os.path.exists(audio_dir):
        print(f"Audio directory not found: {audio_dir}")
        return None, None, []

    fft = FastFourierTransformation(sample_rate=16000, n_fft=512, hop_length=160, win_length=400)
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])
    print(f"\nFound {len(audio_files)} audio files in {audio_dir}")
    print("Applying FFT to all audio files...\n")

    def collate_fn(batch):
        audios = [item['audio'] for item in batch]
        audios_padded = torch.nn.utils.rnn.pad_sequence(audios, batch_first=True)
        texts = [item['text'] for item in batch]
        input_ids = [torch.tensor(item['input_ids']) for item in batch]
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True)
        return {'audio': audios_padded, 'text': texts, 'input_ids': input_ids_padded}

    json_path = os.path.join(os.path.dirname(__file__), 'training_data', 'speech.json')
    if not os.path.exists(json_path):
        print(f"speech.json not found: {json_path}")
        return None, None, []

    df = pd.read_json(json_path, lines=True)
    audio_bytes = []
    for _, row in df.iterrows():
        rel = row.get('audio_path')
        audio_file = os.path.join(os.path.dirname(__file__), 'training_data', rel.replace('/', os.sep))
        try:
            with open(audio_file, 'rb') as f:
                audio_bytes.append(f.read())
        except Exception as e:
            print(f"Warning: could not read {audio_file}: {e}")
            audio_bytes.append(b'')

    df['audio'] = audio_bytes
    df['transcription'] = df['text']

    tokenizer = create_tokenizer()
    speech_dataset = SpeechDataset(df, tokenizer=tokenizer)

    # FIX 1: dataloader is now returned so main() can use it
    dataloader = torch.utils.data.DataLoader(speech_dataset, batch_size=1, collate_fn=collate_fn)

    mel_outputs = []
    processed_count = 0

    for idx, batch in enumerate(dataloader):
        try:
            audio = batch['audio']
            audio_numpy = audio[0].numpy()
            magnitudes = fft.apply_fft(audio_numpy)   # (num_frames, 257)
            mel_spectrogram = MelSpectrogram(n_fft=512)
            mel_output = mel_spectrogram.build_mel_spectrogram(magnitudes)  # (num_frames, 80)
            mel_outputs.append(mel_output)
            processed_count += 1
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(audio_files)} files")
        except Exception as e:
            print(f"Error processing {batch['text']}: {str(e)}")
            continue

    print(f"\nSuccessfully processed {processed_count}/{len(audio_files)} audio files")
    return speech_dataset, dataloader, mel_outputs  # FIX 1: return dataloader


class SpeechToText(nn.Module):
    def __init__(self, vocab_size, n_mels=80, d_model=512, n_heads=8,
                 ff_dim=2048, enc_layers=6, dec_layers=6, dropout=0.1):
        super().__init__()
        self.encoder = AcousticEncoder(n_mels=n_mels, d_model=d_model, n_heads=n_heads,
                                       ff_dim=ff_dim, n_layers=enc_layers)
        self.decoder = TransformerDecoder(vocab_size, d_model, n_heads, ff_dim, dec_layers)
        self.criterion = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN_ID)

    def forward(self, mel, tokens):
        # mel:    [batch, n_mels, T]
        # tokens: [batch, tgt_len]
        encoder_out = self.encoder(mel)
        logits = self.decoder(tokens, encoder_out)
        return logits

    @staticmethod
    def train_step(model, mel, transcript_ids, optimizer):
        model.train()
        decoder_input  = transcript_ids[:, :-1]
        decoder_target = transcript_ids[:, 1:]

        logits = model(mel, decoder_input)

        loss = model.criterion(
            logits.reshape(-1, logits.size(-1)),
            decoder_target.reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        return loss.item()

    # FIX 2: @staticmethod must be outermost decorator (written first)
    @staticmethod
    @torch.no_grad()
    def greedy_decode(model, mel, tokenizer, max_len=200):
        model.eval()
        encoder_out = model.encoder(mel)

        bos_id = getattr(tokenizer, 'bos_token_id', 1)
        eos_id = getattr(tokenizer, 'eos_token_id', 2)
        tokens = torch.tensor([[bos_id]], device=mel.device)

        for _ in range(max_len):
            logits = model.decoder(tokens, encoder_out)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tokens = torch.cat([tokens, next_token], dim=1)
            if next_token.item() == eos_id:
                break

        return tokenizer.decode(tokens[0].tolist())

    @staticmethod
    @torch.no_grad()
    def beam_search(model, mel, tokenizer, beam_size=5, max_len=200):
        model.eval()
        device = mel.device
        encoder_out = model.encoder(mel)

        eos_id = getattr(tokenizer, 'eos_token_id', 2)
        beams = [(0.0, [tokenizer.bos_token_id])]
        completed = []

        for _ in range(max_len):
            candidates = []

            for score, seq in beams:
                if seq[-1] == eos_id:
                    completed.append((score, seq))
                    continue

                tokens = torch.tensor([seq], device=device)
                logits = model.decoder(tokens, encoder_out)
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)

                topk_probs, topk_ids = log_probs[0].topk(beam_size)
                for prob, tok_id in zip(topk_probs, topk_ids):
                    candidates.append((score + prob.item(), seq + [tok_id.item()]))

            if not candidates:
                break

            beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_size]

        completed += beams
        best = max(completed, key=lambda x: x[0] / len(x[1]))
        return tokenizer.decode(best[1])


def main():
    print("=" * 80)
    print("SPEECH-TO-TEXT TRAINING PIPELINE")
    print("=" * 80)

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
    device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nDevice: {device}")
    print(f"Vocab size: {vocab_size}")
    print(f"Model: d_model={d_model}, n_heads={n_heads}, enc_layers={enc_layers}, dec_layers={dec_layers}")
    print(f"Hyperparameters: lr={learning_rate}, epochs={num_epochs}, batch_size={batch_size}\n")

    print("Step 1: Loading and preprocessing audio data...")
    # FIX 1: unpack the returned dataloader
    speech_dataset, dataloader, mel_outputs = load_and_apply_preprocessing_to_audio()
    print(f"Loaded dataset with {len(speech_dataset) if speech_dataset else 0} examples, "
          f"produced {len(mel_outputs)} mel outputs")

    if speech_dataset is None or len(speech_dataset) == 0:
        print("ERROR: No audio data loaded. Exiting.")
        return

    print("\nStep 2: Initializing Speech-to-Text model...")
    model = SpeechToText(
        vocab_size=vocab_size, n_mels=n_mels, d_model=d_model, n_heads=n_heads,
        ff_dim=ff_dim, enc_layers=enc_layers, dec_layers=dec_layers, dropout=dropout
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params:,} parameters")

    print("\nStep 3: Setting up optimizer...")
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    print("\nStep 4: Starting training loop...")
    print("-" * 80)

    # FIX 3: FFT instantiated once outside all loops
    fft = FastFourierTransformation()

    total_loss = 0
    batches_processed = 0

    for epoch in range(num_epochs):
        epoch_loss  = 0
        num_batches = 0
        skipped     = 0
        errors      = 0

        for batch_idx, batch in enumerate(dataloader):
            try:
                audio_batch     = batch['audio']
                input_ids_batch = batch['input_ids'].to(device)

                # n_fft=512 matches FastFourierTransformation — one shared instance is enough
                mel_spec_obj = MelSpectrogram(n_fft=512)
                mel_batch = []
                for audio in audio_batch:
                    audio_np   = audio.numpy()
                    magnitudes = fft.apply_fft(audio_np)   # (num_frames, 257)
                    mel = mel_spec_obj.build_mel_spectrogram(magnitudes)  # (num_frames, 80)
                    mel_batch.append(torch.tensor(mel, dtype=torch.float32))

                mel_padded = torch.nn.utils.rnn.pad_sequence(mel_batch, batch_first=True)

                if mel_padded.dim() == 2:
                    mel_padded = mel_padded.unsqueeze(0)

                mel_padded = mel_padded.permute(0, 2, 1).to(device)  # [batch, n_mels, T]

                if mel_padded.shape[2] < 2:
                    skipped += 1
                    continue

                loss = SpeechToText.train_step(model, mel_padded, input_ids_batch, optimizer)

                epoch_loss        += loss
                num_batches       += 1
                batches_processed += 1

                if (batch_idx + 1) % 100 == 0:
                    print(f"Epoch {epoch+1}/{num_epochs} | "
                          f"Batch {batch_idx+1}/{len(dataloader)} | "
                          f"Loss: {epoch_loss/num_batches:.4f}")

            except Exception as e:
                errors += 1
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue

        # FIX: scheduler.step() must come AFTER optimizer.step() (inside train_step)
        if num_batches > 0:
            scheduler.step()

        avg_epoch_loss = epoch_loss / max(num_batches, 1)
        print(f"Epoch {epoch+1} completed | "
              f"Avg Loss: {avg_epoch_loss:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Trained: {num_batches} | Skipped: {skipped} | Errors: {errors}")

        if num_batches == 0:
            print("  WARNING: No batches trained this epoch - check audio files and FFT output shapes.")

        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(
                os.path.dirname(__file__), 'checkpoints', f'model_epoch_{epoch+1}.pt'
            )
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save({
                'epoch':                epoch + 1,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':                 avg_epoch_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    print("\n" + "-" * 80)
    print("Training complete!")
    final_model_path = os.path.join(os.path.dirname(__file__), 'model_final.pt')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
