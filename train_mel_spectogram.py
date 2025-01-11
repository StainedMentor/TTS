import os

import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torch.nn as nn
import torch.optim as optim
# ClearML imports
from clearml import Task, Dataset as ClearML_Dataset
from torch.utils.data import DataLoader

from dataset import LJSpeechDataset, ljspeech_collate_fn, VOCAB_SIZE

TASK_NAME = "Train Dummy TTS Model"
PROJECT_NAME = "TTS Project"

task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)
logger = task.get_logger()

# Hyperparameters
learning_rate = 1e-4
num_epochs = 2
batch_size = 40
mel_bins = 80
max_frames = 200

task.connect_configuration(
    {
        "learning_rate": learning_rate,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "mel_bins": mel_bins,
        "mel_frames": max_frames,
    }
)

class DummyTTSModel(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64, mel_bins=80):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, mel_bins * 200)

    def forward(self, text_padded, text_lengths):

        embedded = self.embedding(text_padded)

        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            lengths=text_lengths,
            batch_first=True,
            enforce_sorted=False
        )
        _, (h, _) = self.lstm(packed)
        last_hidden = h[-1]
        out = self.fc(last_hidden)
        out = out.view(-1, 80, 200)
        return out



dataset_path = ClearML_Dataset.get(
    dataset_name="LJSpeech-1.1",
    dataset_project="TTS Project"
).get_local_copy()

ljspeech_dataset = LJSpeechDataset(
    data_dir=dataset_path,
    use_normalized_text=True,
    sample_rate=22050,
    n_mels=80
)

train_loader = DataLoader(
    ljspeech_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=ljspeech_collate_fn,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")
print(f"Using device: {device}")

model = DummyTTSModel(vocab_size=VOCAB_SIZE, embed_dim=64, mel_bins=mel_bins).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def plot_spectrogram(mel_spectrogram):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(mel_spectrogram, aspect="auto", origin="lower", cmap="viridis")
    ax.set_title("Predicted Mel-Spectrogram")
    ax.set_xlabel("Frames")
    ax.set_ylabel("Mel Bins")
    plt.tight_layout()
    return fig


def load_waveglow_vocoder(device="cpu"):
    waveglow = torch.hub.load(
        'nvidia/DeepLearningExamples:torchhub',
        'nvidia_waveglow',
        model_math='fp32',
        pretrained=True,
        force_reload=True
    )

    waveglow = waveglow.to(device)
    waveglow.eval()
    return waveglow

def vocoder_infer(mel, vocoder, device="cpu", denoise=False):

    with torch.no_grad():
        audio = vocoder.infer(mel, sigma=0.666)
    audio = audio.cpu().numpy()[0]
    return audio

def main():
    waveglow = load_waveglow_vocoder("cpu")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch_idx, (text_padded, text_lengths, padded_mels, mel_lengths) in enumerate(train_loader):
            print(batch_idx)
            if batch_idx == 5: # Temporary
                break
            text_padded = text_padded.to(device)
            batch_size_now = padded_mels.size(0)
            mel_out = torch.zeros((batch_size_now, mel_bins, max_frames), dtype=padded_mels.dtype, device=device)

            for i in range(batch_size_now):
                length = min(padded_mels[i].size(1), max_frames)
                mel_out[i, :, :length] = padded_mels[i, :, :length].to(device)

            predicted_mels = model(text_padded, text_lengths)
            loss = criterion(predicted_mels, mel_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch}/{num_epochs}], Loss: {avg_loss:.4f}")
        logger.report_scalar("Training Loss", "loss", iteration=epoch, value=avg_loss)

        # Log one predicted mel + audio
        model.eval()
        with torch.no_grad():
            sample_mel = predicted_mels[0].cpu().numpy()

        fig = plot_spectrogram(sample_mel)
        logger.report_matplotlib_figure(
            title="Predicted Mel-Spectrogram",
            series="train_spectrograms",
            iteration=epoch,
            figure=fig
        )
        plt.close(fig)

        audio_wg = vocoder_infer(sample_mel, waveglow, device="cpu")
        audio_path = f"generated_epoch_{epoch}.wav"
        sf.write(audio_path, audio_wg, 22050)
        # Upload audio as ClearML artifact
        task.upload_artifact(name=f"Audio_Epoch_{epoch}", artifact_object=audio_path)

    # SAVE CHECKPOINT
    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/dummy_tts_model.pth"
    torch.save(model.state_dict(), checkpoint_path)

    task.upload_artifact(name="final_model_checkpoint", artifact_object=checkpoint_path)

    print("Training complete. Model saved and artifacts logged to ClearML.")


if __name__ == '__main__':
    main()
