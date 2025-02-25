import math
import warnings
from pathlib import Path

import argbind
import numpy as np
import torch
from audiotools import AudioSignal
from audiotools.core import util
from tqdm import tqdm

from dac.utils import load_model

warnings.filterwarnings("ignore", category=UserWarning)


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
):
    """
    Encode audio files in input path to raw binary (.bin) of codebook indices.

    Parameters
    ----------
    input : str
        Path to input audio file or directory
    output : str, optional
        Path to output directory. If input is a directory, the folder structure
        is replicated under output. Defaults to "".
    weights_path : str, optional
        If provided, load model weights from here. Otherwise, downloads by tag.
    model_tag : str, optional
        Tag of the model to use, by default "latest". Ignored if weights_path is set.
    model_bitrate: str
        "8kbps" or "16kbps". Defaults to "8kbps".
    n_quantizers : int, optional
        Number of quantizers to use, by default None (use all).
    device : str, optional
        'cuda' or 'cpu'. Defaults to 'cuda'.
    model_type : str, optional
        '44khz', '24khz', or '16khz'. Defaults to '44khz'.
    win_duration : float, optional
        We pass this to the codec to pad up to `win_duration` seconds for single-pass
        encoding. Defaults to 30.0.
    verbose : bool, optional
        If True, enable verbose logging in the codec. Defaults to False.
    """
    generator = load_model(
        model_type=model_type,
        model_bitrate=model_bitrate,
        tag=model_tag,
        load_path=weights_path,
    )
    generator.to(device)
    generator.eval()

    # Gather audio files
    input = Path(input)
    audio_files = util.find_audio(input)

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(len(audio_files)), desc="Encoding files"):
        audio_path = audio_files[i]
        signal = AudioSignal(audio_path)

        # Encode audio -> raw code indices (np.int16 array)
        codes = generator.compress(
            audio_path_or_signal=signal,
            max_duration_s=win_duration,
        )

        # Build output path, replicate folder structure if needed
        relative_path = audio_path.relative_to(input)
        out_path = (output / relative_path).with_suffix(".bin")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Write raw binary data
        with open(out_path, "wb") as f:
            f.write(codes.tobytes())


if __name__ == "__main__":
    args = argbind.parse_args()
    with argbind.scope(args):
        encode()
