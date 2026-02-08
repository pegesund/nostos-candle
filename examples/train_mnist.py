"""Train LSTM on MNIST digit classification.

Treats each 28x28 image as a sequence of 28 rows (28 time steps, 28-dim input).
Model: LSTM(28, 128) -> linear(128, 10) on last hidden state.

Loads data from safetensors (created by mnist_prepare.py).
"""
import torch
import torch.nn as nn
from safetensors.torch import load_file

DATA_PATH = "/tmp/nostos-candle/data/mnist.safetensors"


class MnistLSTM(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=128, num_classes=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, num_classes, bias=False)

    def forward(self, x):
        # x: [batch, 28, 28]
        hidden, _ = self.lstm(x)       # [batch, 28, 128]
        last = hidden[:, -1, :]        # [batch, 128]
        logits = self.linear(last)     # [batch, 10]
        return logits


def main():
    torch.manual_seed(42)

    # Load data
    print("Loading MNIST from safetensors...")
    data = load_file(DATA_PATH)
    train_images = data["train_images"]    # [60000, 28, 28]
    train_labels = data["train_labels"].long()  # [60000]
    test_images = data["test_images"]      # [10000, 28, 28]
    test_labels = data["test_labels"].long()    # [10000]

    print(f"  Train: {train_images.shape}, Test: {test_images.shape}")

    # Model
    model = MnistLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    batch_size = 128
    num_epochs = 3
    n_train = train_images.shape[0]

    print(f"\nTraining LSTM on MNIST:")
    print(f"  Model: LSTM(28, 128) -> linear(128, 10)")
    print(f"  Optimizer: Adam(lr=0.001)")
    print(f"  Batch size: {batch_size}, Epochs: {num_epochs}")
    print()

    for epoch in range(num_epochs):
        # Shuffle training data
        perm = torch.randperm(n_train)
        train_images = train_images[perm]
        train_labels = train_labels[perm]

        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            images = train_images[start:end]
            labels = train_labels[start:end]

            logits = model(images)
            loss = nn.functional.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        print(f"  Epoch {epoch + 1}/{num_epochs} - avg loss: {avg_loss:.4f}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    model.eval()
    with torch.no_grad():
        # Process in batches to save memory
        correct = 0
        total = 0
        for start in range(0, test_images.shape[0], 256):
            end = min(start + 256, test_images.shape[0])
            logits = model(test_images[start:end])
            preds = logits.argmax(dim=1)
            correct += (preds == test_labels[start:end]).sum().item()
            total += end - start

    accuracy = correct / total * 100
    print(f"  Test accuracy: {correct}/{total} = {accuracy:.1f}%")


if __name__ == "__main__":
    main()
