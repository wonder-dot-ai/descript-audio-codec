import math
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import torch
import tqdm
from audiotools import AudioSignal
from torch import nn

SUPPORTED_VERSIONS = ["1.0.1"]


@dataclass
class DACFile:
    codes: torch.Tensor

    # Metadata
    chunk_length: int
    original_length: int
    input_db: float
    channels: int
    sample_rate: int
    padding: bool
    dac_version: str

    def save(self, path):
        artifacts = {
            "codes": self.codes.numpy().astype(np.uint16),
            "metadata": {
                "input_db": self.input_db.numpy().astype(np.float32),
                "original_length": self.original_length,
                "sample_rate": self.sample_rate,
                "chunk_length": self.chunk_length,
                "channels": self.channels,
                "padding": self.padding,
                "dac_version": SUPPORTED_VERSIONS[-1],
            },
        }
        path = Path(path).with_suffix(".dac")
        with open(path, "wb") as f:
            np.save(f, artifacts)
        return path

    def save_bin(self, path):
        """
        Saves just the raw codebook indices as int16 as flattened binary file.
        """
        codes_np = self.codes.cpu().numpy().astype(np.int16)
        path = Path(path).with_suffix(".bin")

        # Save as raw binary (.bin) file
        with open(path, "wb") as f:
            f.write(codes_np.tobytes())
        return path

    @classmethod
    def load(cls, path):
        artifacts = np.load(path, allow_pickle=True)[()]
        codes = torch.from_numpy(artifacts["codes"].astype(int))
        if artifacts["metadata"].get("dac_version", None) not in SUPPORTED_VERSIONS:
            raise RuntimeError(
                f"Given file {path} can't be loaded with this version of descript-audio-codec."
            )
        return cls(codes=codes, **artifacts["metadata"])


class CodecMixin:
    @property
    def padding(self):
        if not hasattr(self, "_padding"):
            self._padding = True
        return self._padding

    @padding.setter
    def padding(self, value):
        assert isinstance(value, bool)

        layers = [
            l for l in self.modules() if isinstance(l, (nn.Conv1d, nn.ConvTranspose1d))
        ]

        for layer in layers:
            if value:
                if hasattr(layer, "original_padding"):
                    layer.padding = layer.original_padding
            else:
                layer.original_padding = layer.padding
                layer.padding = tuple(0 for _ in range(len(layer.padding)))

        self._padding = value

    def get_delay(self):
        # Any number works here, delay is invariant to input length
        l_out = self.get_output_length(0)
        L = l_out

        layers = []
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                layers.append(layer)

        for layer in reversed(layers):
            d = layer.dilation[0]
            k = layer.kernel_size[0]
            s = layer.stride[0]

            if isinstance(layer, nn.ConvTranspose1d):
                L = ((L - d * (k - 1) - 1) / s) + 1
            elif isinstance(layer, nn.Conv1d):
                L = (L - 1) * s + d * (k - 1) + 1

            L = math.ceil(L)

        l_in = L

        return (l_in - l_out) // 2

    def get_output_length(self, input_length):
        L = input_length
        # Calculate output length
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                d = layer.dilation[0]
                k = layer.kernel_size[0]
                s = layer.stride[0]

                if isinstance(layer, nn.Conv1d):
                    L = ((L - d * (k - 1) - 1) / s) + 1
                elif isinstance(layer, nn.ConvTranspose1d):
                    L = (L - 1) * s + d * (k - 1) + 1

                L = math.floor(L)
        return L

    @torch.no_grad()
    def compress(
        self,
        audio_path_or_signal: Union[str, Path, AudioSignal],
        max_duration_s: float = 30.0,
        normalize_db: float = -16,
    ) -> np.ndarray:
        """
        One-pass encode with optional padding up to `max_duration_s`.
        Returns raw codebook indices as an np.int16 array [n_codebooks, T_codes].
        """
        self.eval()

        # 1. Load or use existing signal
        if isinstance(audio_path_or_signal, (str, Path)):
            audio_signal = AudioSignal.load_from_file_with_ffmpeg(
                str(audio_path_or_signal)
            ).to_mono()
        else:
            audio_signal = audio_path_or_signal.to_mono()

        # 2. Move to device & resample
        audio_signal = audio_signal.clone().to(self.device)
        audio_signal.resample(self.sample_rate)

        original_length = audio_signal.signal_length
        max_samples = int(self.sample_rate * max_duration_s)

        # 3. If shorter than max, pad on the end
        if original_length < max_samples:
            pad_amount = max_samples - original_length
            audio_signal.zero_pad(0, pad_amount)

        # 4. Loudness normalization
        if normalize_db is not None:
            audio_signal.normalize(normalize_db)
        audio_signal.ensure_max_of_audio()

        # 5. Flatten batch + channels
        nb, nac, nt = audio_signal.audio_data.shape
        audio_signal.audio_data = audio_signal.audio_data.reshape(nb * nac, 1, nt)

        # 6. Model preprocess (e.g. pad to hop-multiple)
        audio_data = audio_signal.audio_data
        audio_data = self.preprocess(audio_data, self.sample_rate)

        # 7. Forward pass -> codes
        #    codes shape: [B=1, n_codebooks, T_code]
        _, codes, _, _, _ = self.encode(audio_data, n_quantizers=None)

        # 8. Only keep code frames for the original unpadded portion
        # Calculate the ratio between encoded frames and input samples
        input_length = audio_data.shape[-1]
        encoded_length = codes.shape[-1]
        codes_ratio = encoded_length / input_length
        needed_frames = math.ceil(original_length * codes_ratio)

        # Ensure we don't exceed the actual number of frames
        needed_frames = min(needed_frames, encoded_length)
        codes = codes[..., :needed_frames]  # shape => [1, n_codebooks, needed_frames]

        # 9. Remove batch dimension => [n_codebooks, needed_frames]
        codes = codes.squeeze(0)

        # 10. Convert to CPU np.int16 array
        result = codes.cpu().numpy().astype(np.int16)
        return result

    @torch.no_grad()
    def decompress(
        self,
        obj: Union[str, Path, DACFile],
        verbose: bool = False,
    ) -> AudioSignal:
        """Reconstruct audio from a given .dac file

        Parameters
        ----------
        obj : Union[str, Path, DACFile]
            .dac file location or corresponding DACFile object.
        verbose : bool, optional
            Prints progress if True, by default False

        Returns
        -------
        AudioSignal
            Object with the reconstructed audio
        """
        self.eval()
        if isinstance(obj, (str, Path)):
            obj = DACFile.load(obj)

        original_padding = self.padding
        self.padding = obj.padding

        range_fn = range if not verbose else tqdm.trange
        codes = obj.codes
        original_device = codes.device
        chunk_length = obj.chunk_length
        recons = []

        for i in range_fn(0, codes.shape[-1], chunk_length):
            c = codes[..., i : i + chunk_length].to(self.device)
            z = self.quantizer.from_codes(c)[0]
            r = self.decode(z)
            recons.append(r.to(original_device))

        recons = torch.cat(recons, dim=-1)
        recons = AudioSignal(recons, self.sample_rate)

        resample_fn = recons.resample
        loudness_fn = recons.loudness

        # If audio is > 10 minutes long, use the ffmpeg versions
        if recons.signal_duration >= 10 * 60 * 60:
            resample_fn = recons.ffmpeg_resample
            loudness_fn = recons.ffmpeg_loudness

        recons.normalize(obj.input_db)
        resample_fn(obj.sample_rate)
        recons = recons[..., : obj.original_length]
        loudness_fn()
        recons.audio_data = recons.audio_data.reshape(
            -1, obj.channels, obj.original_length
        )

        self.padding = original_padding
        return recons
