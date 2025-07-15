from typing import List, Tuple

import numpy as np
import librosa
import torch
import torch.nn.functional as F
from s3tokenizer.utils import padding
from s3tokenizer.model_v2 import S3TokenizerV2, ModelConfig
from .config import S3_SR, S3_HOP, S3_TOKEN_RATE

class S3Tokenizer(S3TokenizerV2):
    ignore_state_dict_missing = ("_mel_filters", "window")

    def __init__(self, name: str = "speech_tokenizer_v2_25hz", config: ModelConfig = ModelConfig()):
        super().__init__(name)
        self.n_fft = 400
        mel_filters = librosa.filters.mel(sr=S3_SR, n_fft=self.n_fft, n_mels=config.n_mels)
        self.register_buffer("_mel_filters", torch.FloatTensor(mel_filters))
        self.register_buffer("window", torch.hann_window(self.n_fft))

    def pad(self, wavs, sr) -> List[torch.Tensor]:
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            n_tokens = np.ceil((wav.shape[1] / sr) * S3_TOKEN_RATE)
            intended_len = int(n_tokens * (sr / S3_TOKEN_RATE))
            wav = F.pad(wav, (0, intended_len - wav.shape[-1]), mode="constant", value=0)
            processed_wavs.append(wav)
        return processed_wavs

    def _prepare_audio(self, wavs):
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            processed_wavs.append(wav)
        return processed_wavs

    @torch.no_grad()
    def forward(self, wavs: torch.Tensor, accelerator=None, max_len: int = None) -> Tuple[torch.Tensor, torch.LongTensor]:
        processed_wavs = self._prepare_audio(wavs)
        mels = []
        for wav in processed_wavs:
            wav = wav.to(self.device)
            mel = self.log_mel_spectrogram(wav)
            if max_len is not None:
                mel = mel[..., :max_len * 4]
            mels.append(mel.squeeze(0))
        mels, mel_lens = padding(mels)
        tokenizer = self if accelerator is None else accelerator.unwrap_model(self)
        tokens, token_lens = tokenizer.quantize(mels, mel_lens.to(self.device))
        return tokens.long().detach(), token_lens.long().detach()

    def log_mel_spectrogram(self, audio: torch.Tensor, padding: int = 0):
        if not torch.is_tensor(audio):
            audio = torch.from_numpy(audio)
        audio = audio.to(self.device)
        if padding > 0:
            audio = F.pad(audio, (0, padding))
        stft = torch.stft(audio, self.n_fft, S3_HOP, window=self.window.to(self.device), return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        mel_spec = self._mel_filters.to(self.device) @ magnitudes
        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        return (log_spec + 4.0) / 4.0
