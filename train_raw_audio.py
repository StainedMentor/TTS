import os
import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
# ClearML
from clearml import Task, Dataset as ClearML_Dataset

PROJECT_NAME = "TTS Project"
TASK_NAME = "Train Dummy TTS (Direct Waveform)"

task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)
logger = task.get_logger()

device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# --------------------------------------------------------------------------------
# 2. BASIC TEXT -> ID MAPPING
# --------------------------------------------------------------------------------
# We'll define a small set of characters. Unknown chars -> 0.
ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,.!? '-\";:"
char_to_id = {c: i+1 for i, c in enumerate(ALPHABET)}  # 1-based indexing
UNK_TOKEN = 0
VOCAB_SIZE = len(char_to_id) + 1  # +1 for UNK

def text_to_sequence(text):
    """Naive character-level tokenizer."""
    return torch.tensor([char_to_id.get(ch, UNK_TOKEN) for ch in text], dtype=torch.long)

# --------------------------------------------------------------------------------
# 3. LJSPEECH DATASET - TEXT + RAW AUDIO
# --------------------------------------------------------------------------------
class LJSpeechWaveDataset(Dataset):
    """
    Loads text + raw audio from LJSpeech.
    Crops (or zero-pads) each audio to a fixed max duration.
    """
    def __init__(self,
                 data_dir,
                 sample_rate=22050,
                 max_duration=1.0,
                 use_normalized_text=True):
        super().__init__()
        self.data_dir = data_dir
        self.sample_rate = sample_rate
        self.max_duration = max_duration
        self.use_normalized_text = use_normalized_text

        metadata_path = os.path.join(data_dir, "metadata.csv")
        if not os.path.isfile(metadata_path):
            raise FileNotFoundError("metadata.csv not found in %s" % data_dir)

        with open(metadata_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        self.entries = []
        for line in lines:
            parts = line.strip().split("|")
            if len(parts) != 3:
                continue
            wav_id, raw_text, norm_text = parts
            text_str = norm_text if self.use_normalized_text else raw_text
            wav_path = os.path.join(data_dir, "wavs", f"{wav_id}.wav")
            self.entries.append((wav_path, text_str))

        # Max number of samples for each audio
        self.max_samples = int(self.max_duration * self.sample_rate)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        wav_path, text_str = self.entries[idx]
        # Text -> IDs
        text_ids = text_to_sequence(text_str)

        # Load audio
        audio, sr = librosa.load(wav_path, sr=self.sample_rate)
        # Crop or pad to max_samples
        if len(audio) > self.max_samples:
            audio = audio[:self.max_samples]
        else:
            # zero-pad
            pad_len = self.max_samples - len(audio)
            audio = np.pad(audio, (0, pad_len), mode="constant")

        # Convert to torch
        audio_tensor = torch.from_numpy(audio).float()  # shape: (max_samples,)
        return text_ids, audio_tensor

def rawaudio_collate_fn(batch):

    texts, audios = zip(*batch)

    text_lengths = [t.size(0) for t in texts]
    max_text_len = max(text_lengths)
    padded_texts = []
    for t in texts:
        pad_amt = max_text_len - t.size(0)
        if pad_amt > 0:
            t = torch.cat([t, torch.zeros(pad_amt, dtype=torch.long)])
        padded_texts.append(t.unsqueeze(0))
    padded_texts = torch.cat(padded_texts, dim=0)

    audios = torch.stack(audios, dim=0)

    return padded_texts, text_lengths, audios

class DummyRawAudioTTS(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64, hidden_dim=128, num_samples=22050):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_samples)

    def forward(self, text_padded, text_lengths):
        embedded = self.embedding(text_padded)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, text_lengths, batch_first=True, enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        last_hidden = h[-1]
        audio_out = self.fc(last_hidden)
        return audio_out

def main():
    # Hyperparams
    sample_rate = 22050
    max_duration = 1.0
    num_samples = int(sample_rate * max_duration)
    batch_size = 4
    num_epochs = 200
    learning_rate = 1e-4

    task.connect_configuration({
        "sample_rate": sample_rate,
        "max_duration": max_duration,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate
    })

    dataset_path = ClearML_Dataset.get(dataset_name="LJSpeech-1.1", dataset_project=PROJECT_NAME).get_local_copy()
    print(f"LJSpeech dataset path: {dataset_path}")

    train_dataset = LJSpeechWaveDataset(
        data_dir=dataset_path,
        sample_rate=sample_rate,
        max_duration=max_duration,
        use_normalized_text=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=rawaudio_collate_fn,
        num_workers=1
    )

    model = DummyRawAudioTTS(num_samples=num_samples).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (text_padded, text_lengths, audios) in enumerate(train_loader):
            if batch_idx == 2:
                break

            text_padded = text_padded.to(device)
            audios = audios.to(device)

            predicted_audio = model(text_padded, text_lengths)
            loss = criterion(predicted_audio, audios)*1000
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"[Epoch {epoch}/{num_epochs}] Loss: {avg_loss:.4f}")
        logger.report_scalar("Training Loss", "loss", iteration=epoch, value=avg_loss)

        model.eval()
        with torch.no_grad():
            sample_pred = predicted_audio[0]
        sample_pred_np = sample_pred.detach().cpu().numpy()

        out_wav = f"output/epoch_{epoch}_pred.wav"
        sf.write(out_wav, sample_pred_np, sample_rate)
        task.upload_artifact(name=f"Audio_Epoch_{epoch}", artifact_object=out_wav)

    # Save final model checkpoint
    os.makedirs("checkpoints", exist_ok=True)
    ckpt_path = "checkpoints/dummy_rawaudio_tts.pth"
    torch.save(model.state_dict(), ckpt_path)
    task.upload_artifact(name="final_model_checkpoint", artifact_object=ckpt_path)

    print("Training complete. Check ClearML for logs & artifacts.")

if __name__ == "__main__":
    main()
