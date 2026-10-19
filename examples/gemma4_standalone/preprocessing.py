"""Gemma 4 input preprocessing in plain PyTorch/NumPy — no `transformers`.

Ported from
  transformers/models/gemma4/image_processing_pil_gemma4.py
  transformers/models/gemma4/feature_extraction_gemma4.py
  transformers/models/gemma4/processing_gemma4.py
  transformers/tokenization_utils_fast.py  (only the tiny slice we need)
  transformers/audio_utils.py             (mel_filter_bank / window_function)

Three independent pieces, all verified numerically against the reference:

    Gemma4ImageProcessor   PIL image -> (pixel_values, image_position_ids)
    Gemma4Tokenizer        text -> ids (uses the Rust `tokenizers` lib on tokenizer.json)
    Gemma4FeatureExtractor waveform -> log-mel (input_features, input_features_mask)

and a thin `Gemma4Processor` that stitches them into model-ready tensors.

Runtime deps: torch, numpy, pillow, tokenizers, safetensors. No transformers.
"""

from __future__ import annotations

import json
import math
import os

import numpy as np
import torch
from PIL import Image

__all__ = ["Gemma4ImageProcessor", "Gemma4FeatureExtractor", "Gemma4Tokenizer", "Gemma4Processor"]

SUPPORTED_SOFT_TOKENS = (70, 140, 280, 560, 1120)


# ===========================================================================
# image
# ===========================================================================
def get_aspect_ratio_preserving_size(height, width, patch_size, max_patches, pooling_kernel_size):
    total_px = height * width
    target_px = max_patches * (patch_size**2)
    factor = math.sqrt(target_px / total_px)
    ideal_height = factor * height
    ideal_width = factor * width
    side_mult = pooling_kernel_size * patch_size

    target_height = int(math.floor(ideal_height / side_mult)) * side_mult
    target_width = int(math.floor(ideal_width / side_mult)) * side_mult

    if target_height == 0 and target_width == 0:
        raise ValueError(f"resize to 0x0 image; must be divisible by {side_mult}")

    max_side_length = (max_patches // pooling_kernel_size**2) * side_mult
    if target_height == 0:
        target_height = side_mult
        target_width = min(int(math.floor(width / height)) * side_mult, max_side_length)
    elif target_width == 0:
        target_width = side_mult
        target_height = min(int(math.floor(height / width)) * side_mult, max_side_length)

    if target_height * target_width > target_px:
        raise ValueError(f"[{height}x{width}] -> [{target_height}x{target_width}] exceeds {max_patches} patches")
    return target_height, target_width


def convert_image_to_patches(image: np.ndarray, patch_size: int) -> np.ndarray:
    """(C,H,W) -> (num_patches, patch_size*patch_size*C), row-major over the patch grid."""
    num_channels, image_height, image_width = image.shape
    ph, pw = image_height // patch_size, image_width // patch_size
    patched = image.reshape(num_channels, ph, patch_size, pw, patch_size)
    patched = patched.transpose(1, 3, 2, 4, 0)
    return patched.reshape(ph * pw, -1)


def pad_along_first_dim(image, positions, target_length):
    pad = target_length - image.shape[0]
    if pad > 0:
        image = np.pad(image, [(0, pad)] + [(0, 0)] * (image.ndim - 1), mode="constant", constant_values=0)
        positions = np.pad(positions, [(0, pad), (0, 0)], mode="constant", constant_values=-1)
    return image, positions


class Gemma4ImageProcessor:
    """Resize (aspect preserving, bicubic) -> [0,1] -> patchify -> pad.

    Note the model-side quirk: `pixel_values` is `(B, max_patches, 3*P*P)` — already
    patchified, not an image tensor. The patch embedding is a plain Linear.
    """

    def __init__(self, patch_size=16, max_soft_tokens=280, pooling_kernel_size=3,
                 rescale_factor=1 / 255, do_rescale=True, do_normalize=False,
                 image_mean=(0.0, 0.0, 0.0), image_std=(1.0, 1.0, 1.0), resample="bicubic"):
        if max_soft_tokens not in SUPPORTED_SOFT_TOKENS:
            raise ValueError(f"max_soft_tokens must be one of {SUPPORTED_SOFT_TOKENS}")
        self.patch_size = patch_size
        self.max_soft_tokens = max_soft_tokens
        self.pooling_kernel_size = pooling_kernel_size
        self.rescale_factor = rescale_factor
        self.do_rescale = do_rescale
        self.do_normalize = do_normalize
        self.image_mean = np.array(image_mean, dtype=np.float32)
        self.image_std = np.array(image_std, dtype=np.float32)
        self.resample = {"bicubic": Image.BICUBIC, "bilinear": Image.BILINEAR, "nearest": Image.NEAREST}[resample]

    @classmethod
    def from_processor_config(cls, config: dict) -> "Gemma4ImageProcessor":
        cfg = config.get("image_processor", config)
        resample = cfg.get("resample", 3)
        # PIL enum: 0=nearest 1=lanczos 2=bilinear 3=bicubic 4=box 5=hamming
        resample = {0: "nearest", 1: "bicubic", 2: "bilinear", 3: "bicubic"}.get(resample, "bicubic")
        return cls(
            patch_size=cfg.get("patch_size", 16),
            max_soft_tokens=cfg.get("max_soft_tokens", 280),
            pooling_kernel_size=cfg.get("pooling_kernel_size", 3),
            rescale_factor=cfg.get("rescale_factor", 1 / 255),
            do_rescale=cfg.get("do_rescale", True),
            do_normalize=cfg.get("do_normalize", False),
            image_mean=cfg.get("image_mean", (0.0, 0.0, 0.0)),
            image_std=cfg.get("image_std", (1.0, 1.0, 1.0)),
            resample=resample,
        )

    @property
    def max_patches(self) -> int:
        return self.max_soft_tokens * self.pooling_kernel_size**2

    def _to_numpy_chw(self, image) -> np.ndarray:
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            arr = np.asarray(image, dtype=np.uint8)
        else:
            arr = np.asarray(image)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            if arr.shape[0] in (1, 3) and arr.shape[0] < arr.shape[-1]:
                arr = arr.transpose(1, 2, 0)
        return arr  # HWC

    def _resize(self, arr_hwc: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        if arr_hwc.shape[0] == target_h and arr_hwc.shape[1] == target_w:
            return arr_hwc
        img = Image.fromarray(arr_hwc)
        img = img.resize((target_w, target_h), resample=self.resample)
        return np.asarray(img)

    def preprocess(self, images) -> dict:
        if isinstance(images, (list, tuple)):
            images = list(images)
        else:
            images = [images]

        all_pixels, all_positions, soft_tokens = [], [], []
        for image in images:
            arr = self._to_numpy_chw(image)
            height, width = arr.shape[0], arr.shape[1]
            target_h, target_w = get_aspect_ratio_preserving_size(
                height, width, self.patch_size, self.max_patches, self.pooling_kernel_size
            )
            arr = self._resize(arr, target_h, target_w)

            chw = arr.astype(np.float32).transpose(2, 0, 1)  # (C,H,W)
            if self.do_rescale:
                chw = chw * self.rescale_factor
            if self.do_normalize:
                chw = (chw - self.image_mean[:, None, None]) / self.image_std[:, None, None]

            patches = convert_image_to_patches(chw, self.patch_size)
            soft_tokens.append(patches.shape[0] // self.pooling_kernel_size**2)

            patch_h = chw.shape[-2] // self.patch_size
            patch_w = chw.shape[-1] // self.patch_size
            grid_x, grid_y = np.meshgrid(np.arange(patch_w), np.arange(patch_h), indexing="xy")
            real_positions = np.stack([grid_x, grid_y], axis=-1).reshape(patches.shape[0], 2)

            patches, positions = pad_along_first_dim(patches, real_positions, self.max_patches)
            all_pixels.append(patches)
            all_positions.append(positions)

        return {
            "pixel_values": torch.from_numpy(np.stack(all_pixels, axis=0)),
            "image_position_ids": torch.from_numpy(np.stack(all_positions, axis=0)).long(),
            "num_soft_tokens_per_image": soft_tokens,
        }


# ===========================================================================
# audio
# ===========================================================================
def hertz_to_mel(freq, mel_scale="htk"):
    if mel_scale == "htk":
        return 2595.0 * np.log10(1.0 + (freq / 700.0))
    if mel_scale == "kaldi":
        return 1127.0 * np.log(1.0 + (freq / 700.0))
    raise NotImplementedError(mel_scale)


def mel_to_hertz(mels, mel_scale="htk"):
    if mel_scale == "htk":
        return 700.0 * (np.power(10, mels / 2595.0) - 1.0)
    if mel_scale == "kaldi":
        return 700.0 * (np.exp(mels / 1127.0) - 1.0)
    raise NotImplementedError(mel_scale)


def mel_filter_bank(num_frequency_bins, num_mel_filters, min_frequency, max_frequency,
                    sampling_rate, norm=None, mel_scale="htk"):
    mel_min = hertz_to_mel(min_frequency, mel_scale)
    mel_max = hertz_to_mel(max_frequency, mel_scale)
    mel_freqs = np.linspace(mel_min, mel_max, num_mel_filters + 2)
    filter_freqs = mel_to_hertz(mel_freqs, mel_scale)
    fft_freqs = np.linspace(0, sampling_rate // 2, num_frequency_bins)

    filter_diff = np.diff(filter_freqs)
    slopes = np.expand_dims(filter_freqs, 0) - np.expand_dims(fft_freqs, 1)
    down_slopes = -slopes[:, :-2] / filter_diff[:-1]
    up_slopes = slopes[:, 2:] / filter_diff[1:]
    mel_filters = np.maximum(np.zeros(1), np.minimum(down_slopes, up_slopes))

    if norm == "slaney":
        enorm = 2.0 / (filter_freqs[2 : num_mel_filters + 2] - filter_freqs[:num_mel_filters])
        mel_filters *= np.expand_dims(enorm, 0)
    return mel_filters


def hann_window(window_length: int, periodic: bool = True) -> np.ndarray:
    length = window_length + 1 if periodic else window_length
    return np.hanning(length)[:window_length]


class Gemma4FeatureExtractor:
    """USM-style log-mel frontend: 20 ms frame, 10 ms hop, semicausal padding, htk mel."""

    def __init__(self, feature_size=128, sampling_rate=16_000, frame_length_ms=20.0, hop_length_ms=10.0,
                 min_frequency=0.0, max_frequency=8000.0, preemphasis=0.0, preemphasis_htk_flavor=True,
                 fft_overdrive=False, dither=0.0, input_scale_factor=1.0, mel_floor=1e-3,
                 per_bin_mean=None, per_bin_stddev=None, max_length=480_000, pad_to_multiple_of=128):
        self.feature_size = feature_size
        self.sampling_rate = sampling_rate
        self.preemphasis = preemphasis
        self.preemphasis_htk_flavor = preemphasis_htk_flavor
        self.dither = dither
        self.input_scale_factor = input_scale_factor
        self.mel_floor = mel_floor
        self.frame_length = int(round(sampling_rate * frame_length_ms / 1000.0))
        self.hop_length = int(round(sampling_rate * hop_length_ms / 1000.0))
        fft_length = 2 ** math.ceil(math.log2(self.frame_length))
        self.fft_length = fft_length * 2 if fft_overdrive else fft_length
        self.window = hann_window(self.frame_length, periodic=True).astype(np.float32)
        self.mel_filters = mel_filter_bank(
            num_frequency_bins=self.fft_length // 2 + 1,
            num_mel_filters=feature_size,
            min_frequency=min_frequency,
            max_frequency=max_frequency,
            sampling_rate=sampling_rate,
            norm=None,
            mel_scale="htk",
        )
        self.per_bin_mean = None if per_bin_mean is None else np.array(per_bin_mean).reshape(1, 1, feature_size)
        self.per_bin_stddev = None if per_bin_stddev is None else np.array(per_bin_stddev).reshape(1, 1, feature_size)
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of

    @classmethod
    def from_processor_config(cls, config: dict) -> "Gemma4FeatureExtractor":
        cfg = config.get("feature_extractor", config)
        return cls(
            feature_size=cfg.get("feature_size", 128),
            sampling_rate=cfg.get("sampling_rate", 16_000),
            min_frequency=cfg.get("min_frequency", 0.0),
            max_frequency=cfg.get("max_frequency", 8000.0),
            preemphasis=cfg.get("preemphasis", 0.0),
            preemphasis_htk_flavor=cfg.get("preemphasis_htk_flavor", True),
            fft_overdrive=cfg.get("fft_overdrive", False),
            dither=cfg.get("dither", 0.0),
            input_scale_factor=cfg.get("input_scale_factor", 1.0),
            mel_floor=cfg.get("mel_floor", 1e-3),
            per_bin_mean=cfg.get("per_bin_mean"),
            per_bin_stddev=cfg.get("per_bin_stddev"),
        )

    def _spectrogram(self, waveform: np.ndarray, attention_mask: np.ndarray):
        if waveform.ndim == 1:
            waveform = waveform[None]
        if self.dither > 0.0:
            waveform = waveform + self.dither * np.random.randn(*waveform.shape).astype(waveform.dtype)
        if self.input_scale_factor != 1.0:
            waveform = waveform * self.input_scale_factor

        pad_left = self.frame_length // 2
        waveform = np.pad(waveform, ((0, 0), (pad_left, 0)), mode="constant")
        attention_mask = np.pad(attention_mask, (pad_left, 0), mode="constant", constant_values=0)

        frame_size = self.frame_length + 1
        num_frames = (waveform.shape[1] - frame_size) // self.hop_length + 1
        idx = np.arange(num_frames)[:, None] * self.hop_length + np.arange(frame_size)[None, :]
        frames_to_process = waveform[:, idx]  # (1, T, frame_size)

        if self.preemphasis > 0.0:
            if self.preemphasis_htk_flavor:
                first = frames_to_process[..., :1] * (1.0 - self.preemphasis)
                rest = frames_to_process[..., 1:-1] - self.preemphasis * frames_to_process[..., :-2]
                frames = np.concatenate([first, rest], axis=-1)
            else:
                frames = frames_to_process[..., 1:] - self.preemphasis * frames_to_process[..., :-1]
        else:
            frames = frames_to_process[..., :-1]

        frames = frames * self.window
        stft = np.fft.rfft(frames, n=self.fft_length, axis=-1)
        mel_spec = np.abs(stft) @ self.mel_filters
        log_mel = np.log(mel_spec + self.mel_floor)
        if self.per_bin_mean is not None:
            log_mel = log_mel - self.per_bin_mean
        if self.per_bin_stddev is not None:
            log_mel = log_mel / self.per_bin_stddev

        log_mel = log_mel.squeeze(0)
        frame_end_indices = np.arange(log_mel.shape[0]) * self.hop_length + frame_size - 1
        mask = attention_mask[frame_end_indices].astype(bool)
        return log_mel, mask

    def __call__(self, waveforms, return_tensors="pt"):
        single = isinstance(waveforms, np.ndarray) and waveforms.ndim == 1
        if single:
            waveforms = [waveforms]
        waveforms = [np.asarray(w, dtype=np.float64) for w in waveforms]

        # pad (right) to the longest, to a multiple of 128, truncate at max_length
        lengths = [min(len(w), self.max_length) for w in waveforms]
        waveforms = [w[: self.max_length] for w in waveforms]
        target = max(lengths)
        if self.pad_to_multiple_of:
            target = int(math.ceil(target / self.pad_to_multiple_of) * self.pad_to_multiple_of)
        padded, masks = [], []
        for w, length in zip(waveforms, lengths):
            m = np.zeros(target, dtype=np.float64)
            m[:length] = 1.0
            padded.append(np.pad(w, (0, target - len(w)), mode="constant"))
            masks.append(m.astype(np.int64))

        features, feature_masks = [], []
        for w, m in zip(padded, masks):
            spec, mask = self._spectrogram(w, m)
            features.append(spec.astype(np.float32) * mask[..., None])
            feature_masks.append(mask)

        out = {
            "input_features": torch.from_numpy(np.stack(features)),
            "input_features_mask": torch.from_numpy(np.stack(feature_masks)),
        }
        if return_tensors is None:
            out = {k: v.numpy() for k, v in out.items()}
        return out


# ===========================================================================
# tokenizer
# ===========================================================================
class Gemma4Tokenizer:
    """Fast-tokenizer wrapper around `tokenizers.Tokenizer` (Rust) + tokenizer.json.

    What `transformers` adds on top of tokenizer.json is: the special-token table,
    padding/truncation plumbing, and decode-with-skip-special-tokens. All of that
    is reimplemented here; BPE itself stays in the Rust library.
    """

    def __init__(self, tokenizer_json: str, tokenizer_config: dict | None = None):
        from tokenizers import Tokenizer

        if not os.path.exists(tokenizer_json):
            raise FileNotFoundError(tokenizer_json)
        self._tok = Tokenizer.from_file(tokenizer_json)
        cfg = tokenizer_config or {}
        self.cfg = cfg

        self.bos_token = cfg.get("bos_token", "<bos>")
        self.eos_token = cfg.get("eos_token", "<eos>")
        self.pad_token = cfg.get("pad_token", "<pad>")
        self.unk_token = cfg.get("unk_token", "<unk>")
        self.image_token = cfg.get("image_token", "<|image|>")
        self.boi_token = cfg.get("boi_token", "<|image>")
        self.eoi_token = cfg.get("eoi_token", "<image|>")
        self.audio_token = cfg.get("audio_token", "<|audio|>")
        self.boa_token = cfg.get("boa_token", "<|audio>")
        self.eoa_token = cfg.get("eoa_token", "<audio|>")
        self.padding_side = cfg.get("padding_side", "left")
        self.add_bos_token = cfg.get("add_bos_token", True)

        self.special_tokens = {k: v for k, v in cfg.items() if k.endswith("_token") and isinstance(v, str)}
        self.special_tokens.update({
            "image_token": "<|image|>", "video_token": "<|video|>", "audio_token": "<|audio|>",
        })
        self._token_to_id = {}
        for tok in self.special_tokens.values():
            tid = self._tok.token_to_id(tok)
            if tid is not None:
                self._token_to_id[tok] = tid
        self.bos_token_id = self._token_to_id.get(self.bos_token, 2)
        self.eos_token_id = self._token_to_id.get(self.eos_token, 1)
        self.pad_token_id = self._token_to_id.get(self.pad_token, 0)

        for attr in ("image", "video", "audio", "boi", "eoi", "boa", "eoa"):
            key = f"{attr}_token"
            tok = self.special_tokens.get(key) or getattr(self, key, None)
            setattr(self, f"{attr}_token_id", self._token_to_id.get(tok) if tok else None)

    # --- basic api ---
    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size(with_added_tokens=True)

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self._tok.token_to_id(token)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        ids = list(self._tok.encode(text, add_special_tokens=False).ids)
        if add_special_tokens and self.add_bos_token:
            ids = [self.bos_token_id] + ids
        return ids

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        ids = [int(i) for i in ids]
        if skip_special_tokens:
            special = set(self._token_to_id.values())
            ids = [i for i in ids if i not in special]
        return self._tok.decode(ids, skip_special_tokens=False)

    def batch_encode(self, texts, add_special_tokens=True, padding=True, pad_to_multiple_of=None):
        batch = [self.encode(t, add_special_tokens) for t in texts]
        if not padding:
            return {"input_ids": batch, "attention_mask": [[1] * len(b) for b in batch]}
        max_len = max(len(b) for b in batch)
        if pad_to_multiple_of:
            max_len = int(math.ceil(max_len / pad_to_multiple_of) * pad_to_multiple_of)
        input_ids, attention_mask = [], []
        for b in batch:
            pad_len = max_len - len(b)
            if self.padding_side == "left":
                input_ids.append([self.pad_token_id] * pad_len + b)
                attention_mask.append([0] * pad_len + [1] * len(b))
            else:
                input_ids.append(b + [self.pad_token_id] * pad_len)
                attention_mask.append([1] * len(b) + [0] * pad_len)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    # --- chat formatting ---
    # NOTE: google/gemma-4-E2B ships **no** chat template (tokenizer_config.json has no
    # `chat_template`, tokenizer.json has none either). This follows the Gemma 3
    # convention (`<bos><|turn>role\n...<turn|>\n`); verify against your own prompt
    # format if you need bit-exact parity for a chat-templated SFT dataset.
    def apply_chat_template(self, messages, add_generation_prompt: bool = True) -> str:
        out = [self.bos_token] if self.add_bos_token else []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if isinstance(content, list):
                content = "".join(part.get("text", self.image_token) if part.get("type") == "text" else self.image_token
                                  for part in content)
            out.append(f"<|turn>{role}\n{content}<turn|>\n")
        if add_generation_prompt:
            out.append("<|turn>model\n")
        return "".join(out)


# ===========================================================================
# processor
# ===========================================================================
class Gemma4Processor:
    """Ties tokenizer + image + audio together, including the soft-token expansion."""

    def __init__(self, tokenizer: Gemma4Tokenizer, image_processor: Gemma4ImageProcessor | None = None,
                 feature_extractor: Gemma4FeatureExtractor | None = None,
                 image_seq_length: int = 280, audio_seq_length: int = 750):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.feature_extractor = feature_extractor
        self.image_seq_length = image_seq_length
        self.audio_seq_length = audio_seq_length

    @classmethod
    def from_pretrained_dir(cls, model_dir: str) -> "Gemma4Processor":
        with open(os.path.join(model_dir, "processor_config.json"), "r", encoding="utf-8") as f:
            proc_cfg = json.load(f)
        with open(os.path.join(model_dir, "tokenizer_config.json"), "r", encoding="utf-8") as f:
            tok_cfg = json.load(f)
        return cls(
            tokenizer=Gemma4Tokenizer(os.path.join(model_dir, "tokenizer.json"), tok_cfg),
            image_processor=Gemma4ImageProcessor.from_processor_config(proc_cfg),
            feature_extractor=Gemma4FeatureExtractor.from_processor_config(proc_cfg),
            image_seq_length=proc_cfg.get("image_seq_length", 280),
            audio_seq_length=proc_cfg.get("audio_seq_length", 750),
        )

    def expand_image_tokens(self, num_soft_tokens: int) -> str:
        """`<|image>` + `<|image|>` * n_soft + `<image|>`  (exactly what the reference does)."""
        return f"{self.tokenizer.boi_token}{self.tokenizer.image_token * num_soft_tokens}{self.tokenizer.eoi_token}"

    def expand_audio_tokens(self, n_soft_tokens: int) -> str:
        return f"{self.tokenizer.boa_token}{self.tokenizer.audio_token * n_soft_tokens}{self.tokenizer.eoa_token}"

    def __call__(self, text: str | list[str] | None = None, images=None, audio=None,
                 return_tensors: str = "pt") -> dict:
        batch_text = [text] if isinstance(text, str) else list(text or [""])
        out: dict = {}

        if images is not None:
            img = self.image_processor(images) if callable(getattr(self.image_processor, "__call__", None)) \
                else self.image_processor.preprocess(images)
            out.update(img)
            replacement = self.expand_image_tokens(int(np.mean(img["num_soft_tokens_per_image"])))
            batch_text = [t.replace(self.tokenizer.image_token, replacement, 1) for t in batch_text]

        if audio is not None and self.feature_extractor is not None:
            aud = self.feature_extractor(audio)
            out.update(aud)
            n_soft = int(out["input_features_mask"][0].sum())
            batch_text = [t.replace(self.tokenizer.audio_token, self.expand_audio_tokens(n_soft), 1) for t in batch_text]

        out.update(self.tokenizer.batch_encode(batch_text, add_special_tokens=True, padding=True))
        return out
