import os
import librosa
import torch
import numpy as np
from torch.utils.data import Dataset

# This dataset uses pairs of padded text + mel spectograms.

ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.!? '-\";:"
char_to_id = {c: i+1 for i, c in enumerate(ALPHABET)}
UNK_TOKEN = 0
VOCAB_SIZE = len(char_to_id) + 1
def text_to_sequence(text):
    return torch.tensor(
        [char_to_id.get(ch, UNK_TOKEN) for ch in text],
        dtype=torch.long
    )


class LJSpeechDataset(Dataset):
    def __init__(self,
                 data_dir,
                 sample_rate=22050,
                 n_mels=80,
                 use_normalized_text=True):
        super().__init__()
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.use_normalized_text = use_normalized_text

        metadata_file = os.path.join(self.data_dir, "metadata.csv")
        if not os.path.isfile(metadata_file):
            raise FileNotFoundError(f"metadata.csv not found in {self.data_dir}")

        self.metadata = []
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('|')
                if len(parts) != 3:
                    continue
                wav_id, raw_text, norm_text = parts
                text = norm_text if self.use_normalized_text else raw_text
                wav_path = os.path.join(self.data_dir, "wavs", f"{wav_id}.wav")
                self.metadata.append((wav_path, text))

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        wav_path, text_str = self.metadata[idx]
        text_tensor = text_to_sequence(text_str)
        audio, sr = librosa.load(wav_path, sr=self.sample_rate)
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            fmin=0,
            fmax=self.sample_rate // 2
        )
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        mel_tensor = torch.tensor(mel_spec, dtype=torch.float)
        return text_tensor, mel_tensor


def ljspeech_collate_fn(batch):
    texts, mels = zip(*batch)

    text_lengths = [t.size(0) for t in texts]
    max_text_len = max(text_lengths)
    padded_texts = []
    for t in texts:
        pad_size = max_text_len - t.size(0)
        if pad_size > 0:
            t = torch.cat([t, torch.zeros(pad_size, dtype=torch.long)])
        padded_texts.append(t.unsqueeze(0))
    padded_texts = torch.cat(padded_texts, dim=0)

    mel_lengths = [m.size(1) for m in mels]
    max_mel_len = max(mel_lengths)
    padded_mels = []
    for m in mels:
        pad_size = max_mel_len - m.size(1)
        if pad_size > 0:
            pad_tensor = torch.zeros((m.size(0), pad_size), dtype=m.dtype)
            m = torch.cat([m, pad_tensor], dim=1)
        padded_mels.append(m.unsqueeze(0))
    padded_mels = torch.cat(padded_mels, dim=0)

    return padded_texts, text_lengths, padded_mels, mel_lengths
