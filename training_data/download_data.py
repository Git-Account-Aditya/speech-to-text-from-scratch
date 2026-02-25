from huggingface_hub import hf_hub_download
import pandas as pd
import soundfile as sf
import numpy as np
import io
import os
import json


def load_stt_dataset():
    try:
        # Download the parquet file directly — completely bypasses torchcodec/FFmpeg
        print("Downloading dataset...")
        parquet_path = hf_hub_download(
            repo_id="m-aliabbas/idrak_timit_subsample1",
            filename="data/train-00000-of-00001-aeb35d2d506d38bf.parquet",
            repo_type="dataset"
        )

        # Read with pandas
        df = pd.read_parquet(parquet_path)

        print(f"Dataset columns: {list(df.columns)}")
        print(f"Dataset shape: {df.shape}")

        # Print first record structure to understand the data
        first_row = df.iloc[0]
        for col in df.columns:
            val = first_row[col]
            if isinstance(val, (bytes, bytearray)):
                print(f"  {col}: <bytes, {len(val)} bytes>")
            elif isinstance(val, dict):
                print(f"  {col}: dict with keys {list(val.keys())}")
            else:
                print(f"  {col}: {type(val).__name__} = {repr(val)[:100]}")

        # Create directories
        script_dir = os.path.dirname(os.path.abspath(__file__))
        audio_dir = os.path.join(script_dir, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        output_path = os.path.join(script_dir, "speech.json")

        records = []
        for idx, row in df.iterrows():
            audio_data = row.get("audio") or row.get("Audio")
            text = None
            # Try common text column names
            for text_col in ["text", "Text", "sentence", "Sentence", "transcription", "Transcription", "label"]:
                if text_col in row.index:
                    text = row[text_col]
                    break
            # If no known text column, grab any non-audio string column
            if text is None:
                for col in df.columns:
                    if col.lower() != "audio" and isinstance(row[col], str):
                        text = row[col]
                        break

            # Extract raw audio bytes
            if isinstance(audio_data, dict) and "bytes" in audio_data:
                audio_bytes = audio_data["bytes"]
            elif isinstance(audio_data, (bytes, bytearray)):
                audio_bytes = audio_data
            else:
                print(f"  Skipping record {idx}: unexpected audio format ({type(audio_data)})")
                continue

            # Save audio file to disk
            audio_filename = f"audio_{idx:04d}.wav"
            audio_path = os.path.join(audio_dir, audio_filename)
            with open(audio_path, "wb") as audio_file:
                audio_file.write(audio_bytes)

            records.append({
                "audio_path": f"audio/{audio_filename}",
                "text": text if text else ""
            })

            if (idx + 1) % 200 == 0:
                print(f"  Processed {idx + 1}/{len(df)} records...")

        # Save metadata JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            for record in records:
                f.write(json.dumps(record) + '\n')

        print(f'\nDataset downloaded successfully!')
        print(f'  Audio files: {len(records)} saved to {audio_dir}')
        print(f'  Metadata: {output_path}')

    except Exception as e:
        print(f'Error occurred while downloading dataset: {e}')
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    load_stt_dataset()
