import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, f1_score


class TextDataset(Dataset):
    def __init__(self, sequences, labels, texts=None):
        self.sequences = sequences
        self.labels = labels
        self.texts = texts

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        if self.texts is not None:
            return (
                torch.tensor(self.sequences[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.float),
                self.texts[idx]
            )
        else:
            return (
                torch.tensor(self.sequences[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.float)
            )


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.embedding(x)
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n[-1])
        return out.squeeze(1)


def split_datasets(csv_path, train_frac=0.8, val_frac=0.1, seed=42):
    """
    CSVファイルを読み込み、訓練、検証、テストの3つのデータフレームに分割する関数
    """
    
    df = pd.read_csv(csv_path)
    df["text"] = df["text"].str.replace("\n", " ")
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    train_size = int(train_frac * len(df))
    val_size = int(val_frac * len(df))

    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size + val_size].copy()
    test_df = df.iloc[train_size + val_size:].copy()

    return train_df, val_df, test_df


def encode(df, vocab, max_length=100):
    sequences = [
        pad_sequence(tokens_to_ids(tokens, vocab), max_length)
        for tokens in df["tokens"]
    ]
    labels = df["class"].values
    texts = df["text"].values
    return sequences, labels, texts


def text_to_tokens(text):
    remove_chars = ['.', ',', '!', '?', '"', "'", ":"]
    for char in remove_chars:
        text = text.replace(char, '')
    text = text.lower()
    tokens = text.split()
    tokens = [
        word for word in tokens
        if not word.startswith('http') and not word.startswith('@')
    ]
    return tokens


def build_vocab(token_lists, min_freq=1):
    counter = Counter()
    for tokens in token_lists:
        counter.update(tokens)

    vocab = {"<PAD>": 0, "<OOV>": 1}
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = len(vocab)

    return vocab


def tokens_to_ids(tokens, vocab):
    return [vocab.get(token, vocab["<OOV>"]) for token in tokens]


def pad_sequence(seq, max_length):
    if len(seq) >= max_length:
        return seq[:max_length]
    return seq + [0] * (max_length - len(seq))


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device).float()

        optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        preds = (torch.sigmoid(logits) >= 0.5).long()
        correct += (preds == y.long()).sum().item()
        total += y.size(0)

    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device).float()
            logits = model(x)
            loss = criterion(logits, y)

            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == y.long()).sum().item()
            total += y.size(0)
            total_loss += loss.item()

    return total_loss / len(loader), correct / total


def save_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies, save_path="learning_curves.png"):
    """
    学習曲線を保存する関数
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Train Accuracy')
    plt.plot(epochs, val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)


def train_with_validation(model, vocab, train_loader, val_loader, optimizer, criterion, device, epochs=20, save_path="best_model.pt"):
    """
    モデルの訓練と検証を行い、最良モデルを保存する関数
    """
    best_val_loss = float("inf")

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        print(f"Epoch {epoch+1}/{epochs} ", f"- train_loss: {train_loss:.4f} ", f"- train_acc: {train_acc:.4f} ", f"- val_loss: {val_loss:.4f} ", f"- val_acc: {val_acc:.4f}")
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict(), "vocab": vocab}, save_path)
            print("Best model saved")
    
    # 学習曲線の保存
    save_learning_curves(train_losses, val_losses, train_accuracies, val_accuracies, save_path="learning_curves.png")


def train_hate_speech_classifier(train_df, val_df, best_model_path="best_model.pt"):
    """
    モデルを訓練し、最良モデルを保存する関数
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # トークン化
    train_df["tokens"] = train_df["text"].apply(text_to_tokens)
    val_df["tokens"] = val_df["text"].apply(text_to_tokens)

    # 訓練データから語彙を構築
    vocab = build_vocab(train_df["tokens"])
    print("Vocab size:", len(vocab))

    # エンコードとパディングを行いデータセットを準備
    train_data = encode(train_df, vocab=vocab)
    val_data = encode(val_df, vocab=vocab)

    # データローダーの作成
    train_dataset = TextDataset(*train_data)
    val_dataset = TextDataset(*val_data)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # モデル構築
    model = LSTMClassifier(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # モデルの訓練と検証
    train_with_validation(model, vocab, train_loader, val_loader, optimizer, criterion, device, epochs=10, save_path=best_model_path)


def evaluate_and_save_test_results(model_path, test_df, output_path="test_predictions.csv"):
    """
    モデルを読み込み、テストデータで評価し、結果をCSVに保存する関数
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    
    # モデルと語彙の読み込み
    checkpoint = torch.load(model_path, map_location=device)
    vocab = checkpoint["vocab"]
    model = LSTMClassifier(len(vocab))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # 前処理
    test_df["tokens"] = test_df["text"].apply(text_to_tokens)
    test_data = encode(test_df, vocab=vocab)
    test_dataset = TextDataset(*test_data)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 予測
    model.eval()
    all_texts = []
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for x, y, texts in test_loader:
            x = x.to(device)

            logits = model(x)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()

            all_texts.extend(texts)
            all_labels.extend(y.numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 評価指標の出力
    cm = confusion_matrix(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))

    print("Test F1:", f1)

    # 結果をCSVに保存
    df_out = pd.DataFrame({"text": all_texts, "true_label": all_labels, "pred_label": all_preds, "probability": all_probs,})
    df_out["correct"] = df_out["true_label"] == df_out["pred_label"]
    df_out.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Predictions saved to {output_path}")


def main():
    # 乱数シードの設定
    torch.manual_seed(42)
    np.random.seed(42)

    # データセットを読み込み3分割する
    csv_path = "data.csv"
    train_df, val_df, test_df = split_datasets(csv_path)

    # モデルを学習・保存
    model_path = "best_model.pt"
    train_hate_speech_classifier(train_df, val_df, best_model_path=model_path)
    
    # モデルを読み込み、テストデータで評価
    evaluate_and_save_test_results(model_path, test_df, output_path="test_predictions.csv")


if __name__ == "__main__":
    main()
