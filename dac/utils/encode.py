import torch
import argbind
import warnings

from tqdm import tqdm
from pathlib import Path
from audiotools import AudioSignal
from audiotools.core import util
from torch.utils.data import Dataset, DataLoader
from dac.utils import load_model


warnings.filterwarnings("ignore", category=UserWarning)


class AudioDataset(Dataset):
    """
    A dataset that loads audio (using AudioSignal and ffmpeg under the hood)
    and returns the raw waveforms (float32 Tensors). This is CPU work.
    """

    def __init__(self, input_dir):
        """
        Parameters
        ----------
        input_dir : str or Path
            Directory containing audio files, or a single file path.
        """
        input_dir = Path(input_dir)
        if input_dir.is_file():
            # Single file
            self.audio_files = [input_dir]
        else:
            # Directory: gather recursively
            self.audio_files = util.find_audio(input_dir)

        self.audio_files = sorted(self.audio_files)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]

        signal = AudioSignal.load_from_file_with_ffmpeg(str(audio_path))
        signal = signal.to_mono()

        return str(audio_path), signal


def audio_collate_fn(batch):
    """
    Custom collate function that returns lists so we can handle variable
    lengths easily in compress_batch_signals.
    """
    paths, waves = zip(*batch)
    return list(paths), list(waves)


def create_dataloader(
    input_dir,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
):
    dataset = AudioDataset(input_dir)
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


def encode_with_dataloader(
    input_dir: str,
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

    # Create the DataLoader
    dataloader = create_dataloader(
        input_dir=input_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    total_processed = 0
    pbar = tqdm(dataloader, desc="Encoding in batches", unit="batch")

    for paths, waves in pbar:
        # Batch compress
        codes_list = generator.compress_batch(
            audio_signals=waves,
            max_duration_s=max_duration_s,
            normalize_db=normalize_db,
        )

        # Save each file in the batch
        for path_str, codes in zip(paths, codes_list):
            rel_path = Path(path_str).relative_to(input_dir)
            out_path = (output_dir / rel_path).with_suffix(".bin")
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "wb") as f:
                f.write(codes.tobytes())

        total_processed += len(paths)
        pbar.set_postfix({"processed": total_processed})

    print(f"Done. Processed {total_processed} files.")


@argbind.bind(group="encode", positional=True, without_prefix=True)
@torch.inference_mode()
@torch.no_grad()
def encode(
    input: str,
    output: str = "",
    weights_path: str = "",
    model_tag: str = "latest",
    model_bitrate: str = "8kbps",
    device: str = "cuda",
    model_type: str = "44khz",
    win_duration: float = 30.0,
    batch_size: int = 12,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    normalize_db: float = -16,
):
    """
    Encode audio files in input path to raw binary (.bin) of codebook indices,
    but now in batched mode with DataLoader prefetch.

    Parameters
    ----------
    input : str
        Path to input audio file or directory
    output : str, optional
        Path to output directory. If input is a directory, the folder structure
        is replicated under output.
    weights_path : str, optional
        If provided, load model weights from here. Otherwise, downloads by tag.
    model_tag : str, optional
        Tag of the model to use, by default "latest". Ignored if weights_path is set.
    model_bitrate: str
        "8kbps" or "16kbps". Defaults to "8kbps".
    device : str, optional
        'cuda' or 'cpu'. Defaults to 'cuda'.
    model_type : str, optional
        '44khz', '24khz', or '16khz'. Defaults to '44khz'.
    win_duration : float, optional
        We'll pad up to `win_duration` seconds for single-pass encoding. Defaults to 30.0.
    batch_size : int, optional
        Number of audio files per batch in the DataLoader.
    num_workers : int, optional
        Number of CPU workers for loading audio.
    prefetch_factor : int, optional
        DataLoader prefetch factor per worker.
    normalize_db : float, optional
        Loudness normalization target in dB. Defaults to -16.
    """
    # Load model (CPU or GPU)
    generator = load_model(
        model_type=model_type,
        model_bitrate=model_bitrate,
        tag=model_tag,
        load_path=weights_path,
    )
    generator.to(device)
    generator.eval()

    if output == "":
        output = "./encoded_output"  # default fallback

    # Use the new data loader approach
    encode_with_dataloader(
        input_dir=input,
        output_dir=output,
        generator=generator,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        max_duration_s=win_duration,
        normalize_db=normalize_db,
    )


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        encode()
