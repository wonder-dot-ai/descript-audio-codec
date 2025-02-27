import torch
import sys
import time
import pandas as pd
import os

from tqdm import tqdm
from pathlib import Path
from audiotools import AudioSignal
from torch.utils.data import Dataset, DataLoader
from dac.utils import load_model


class AudioDataset(Dataset):
    """
    A dataset that loads audio (using AudioSignal and ffmpeg under the hood)
    and returns the raw waveforms (float32 Tensors). This is CPU work.
    """

    def __init__(self, input_dir, whisper_dir):
        input_dir = Path(input_dir)
        whisper_dir = Path(whisper_dir)

        input_files = []
        for ext in ["*.mp3", "*.wav", "*.mp4", "*.flac"]:
            input_files.extend(list(input_dir.glob(ext)))

        whisper_files = list(whisper_dir.glob("*.txt"))
        allowed_stems = {f.stem for f in whisper_files}
        self.audio_files = sorted(
            [file for file in input_files if file.stem in allowed_stems]
        )

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        try:
            audio_path = self.audio_files[idx]

            signal = AudioSignal.load_from_file_with_ffmpeg(str(audio_path))
            signal = signal.to_mono()
        except Exception as e:
            print(f"Error loading audio file {audio_path}: {e}")
            return str(audio_path), None

        return str(audio_path), signal


def audio_collate_fn(batch):
    """
    Custom collate function that returns lists so we can handle variable
    lengths easily in compress_batch_signals.
    """
    filter_batch = [b for b in batch if b[1] is not None]
    paths, waves = zip(*filter_batch)
    return list(paths), list(waves)


def create_dataloader(
    input_dir,
    whisper_dir,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
):
    dataset = AudioDataset(input_dir, whisper_dir)
    # shuffle=False for deterministic encoding
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=audio_collate_fn,
    )
    return dataloader


@torch.inference_mode()
@torch.no_grad()
def encode_with_dataloader(
    input_dir: str,
    whisper_dir: str,
    output_dir: str,
    generator,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    max_duration_s: float,
    normalize_db: float,
):
    """
    Scans input_dir for audio files, uses a DataLoader to load them in parallel,
    compresses them in batches, and saves .bin output in output_dir.

    Parameters
    ----------
    input_dir : str
        Directory (or single file) containing input audio.
    output_dir : str
        Where to write .bin files.
    generator : DAC model
        The loaded DAC model (generator).
    batch_size : int
        Number of audio files per batch.
    num_workers : int
        DataLoader CPU workers.
    prefetch_factor : int
        DataLoader prefetch factor per worker.
    max_duration_s : float
        Used by the compress logic to pad up to X seconds.
    normalize_db : float
        Loudness normalization target in dB.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize lists to track successes and failures
    success_results = []
    failure_results = []

    print("Creating Dataloader...")
    dataloader_start = time.time()

    # Create the DataLoader
    dataloader = create_dataloader(
        input_dir=input_dir,
        whisper_dir=whisper_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    dataloader_end = time.time()
    print(f"Dataloader created in {dataloader_end - dataloader_start:.2f} seconds")

    total_processed = 0
    total_encoding_time = 0.0
    pbar = tqdm(dataloader, desc="Encoding in batches", unit="batch")

    for paths, waves in pbar:
        batch_start_time = time.time()

        try:
            # Batch compress
            codes_list = generator.compress_batch(
                audio_signals=waves,
                max_duration_s=max_duration_s,
                normalize_db=normalize_db,
            )

            batch_time = time.time() - batch_start_time
            total_encoding_time += batch_time

            # Save each file in the batch
            for path_str, codes in zip(paths, codes_list):
                try:
                    rel_path = Path(path_str).relative_to(input_dir)
                    out_path = (output_dir / rel_path).with_suffix(".bin")
                    out_path.parent.mkdir(parents=True, exist_ok=True)

                    with open(out_path, "wb") as f:
                        f.write(codes.tobytes())

                    # Record successful encoding
                    success_results.append(
                        {
                            "filename": os.path.basename(path_str),
                            "encoding_time": batch_time
                            / len(paths),  # Approximate per-file time
                            "output_path": str(out_path),
                        }
                    )

                    total_processed += 1

                except Exception as e:
                    print(f"Error saving encoded file {path_str}: {str(e)}")
                    failure_results.append(
                        {
                            "filename": os.path.basename(path_str),
                            "error_type": "file_save_error",
                            "error_message": str(e),
                        }
                    )

        except Exception as e:
            # Collect batch processing error
            print(f"Error processing batch: {str(e)}")
            for path_str in paths:
                failure_results.append(
                    {
                        "filename": os.path.basename(path_str),
                        "error_type": "batch_processing_error",
                        "error_message": str(e),
                    }
                )

        # Update progress bar with additional information
        pbar.set_postfix(
            {
                "processed": total_processed,
                "batch_time": (
                    f"{batch_time:.2f}s" if "batch_time" in locals() else "error"
                ),
                "success_rate": f"{(len(success_results)/(len(success_results) + len(failure_results))*100):.1f}%",
            }
        )

    # Save results to CSV files
    if success_results:
        success_df = pd.DataFrame(success_results)
        success_df.to_csv(os.path.join(output_dir, "dac_success.csv"), index=False)
    else:
        # Create empty success CSV with headers
        pd.DataFrame(columns=["filename", "encoding_time", "output_path"]).to_csv(
            os.path.join(output_dir, "dac_success.csv"), index=False
        )

    if failure_results:
        failure_df = pd.DataFrame(failure_results)
        failure_df.to_csv(os.path.join(output_dir, "dac_failure.csv"), index=False)
    else:
        # Create empty failure CSV with headers
        pd.DataFrame(columns=["filename", "error_type", "error_message"]).to_csv(
            os.path.join(output_dir, "dac_failure.csv"), index=False
        )

    print(f"\nEncoding Summary:")
    print(f"Total files processed successfully: {len(success_results)}")
    print(f"Total files failed: {len(failure_results)}")
    print(f"Total encoding time: {total_encoding_time:.2f} seconds")
    if len(success_results) > 0:
        print(
            f"Average time per successful file: {total_encoding_time/len(success_results):.2f} seconds"
        )
    print(
        f"Success rate: {(len(success_results)/(len(success_results) + len(failure_results))*100):.1f}%"
    )

    return len(success_results), total_encoding_time


if __name__ == "__main__":
    """
    Main entry point.
    Usage:
        python 5_dac.py <start> <end>

    This script:
    1. Uses <start> and <end> to create paths for audio segments and their
       corresponding output directories.
    2. Loads a DAC model from disk.
    3. Runs the encoding process on the audio segment directory.
    4. Saves encoded .bin files in the output directory.
    """
    if len(sys.argv) < 3:
        print("Usage: python 5_dac.py <start> <end>")
        sys.exit(1)

    start = int(sys.argv[1])
    end = int(sys.argv[2])

    # Build folder/file paths
    base = f"{start}to{end}"
    input_dir = f"./{base}/{start}to{end}_segments"
    output_dir = f"./{base}/{start}to{end}_dac"
    whisper_dir = f"./{base}/{start}to{end}_transcriptions"

    # Model configuration
    MODEL_BITRATE = "8kbps"
    MODEL_TYPE = "44khz"
    MAX_DURATION = 30.0
    BATCH_SIZE = 12
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 2
    NORMALIZE_DB = -16

    # Start timing the entire process
    start_time = time.time()

    # Load model with progress indication
    print("Loading DAC model...")
    model_load_start = time.time()
    generator = load_model(
        model_type=MODEL_TYPE,
        model_bitrate=MODEL_BITRATE,
    )
    generator.to(torch.device("cuda"))
    generator.eval()
    model_load_end = time.time()
    print(f"Model loaded in {model_load_end - model_load_start:.2f} seconds")

    # Process the directory of audio segments
    total_processed, total_encoding_time = encode_with_dataloader(
        input_dir=input_dir,
        whisper_dir=whisper_dir,
        output_dir=output_dir,
        generator=generator,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        max_duration_s=MAX_DURATION,
        normalize_db=NORMALIZE_DB,
    )

    # End timing
    end_time = time.time()
    total_time = end_time - start_time

    print("\nExecution Summary:")
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Model load time: {model_load_end - model_load_start:.2f} seconds")
    print(f"Pure encoding time: {total_encoding_time:.2f} seconds")
    if total_processed > 0:
        print(
            f"Average time per file (including overhead): {total_time/total_processed:.2f} seconds"
        )
    print(f"\nResults saved in:")
    print(f"- {output_dir}/dac_success.csv")
    print(f"- {output_dir}/dac_failure.csv")
