"""Train an LSTM to predict the sum of a 3-number sequence.
Input: [batch, 3, 1] random values
Target: [batch, 1] sum of inputs
Model: LSTM(1, 16) -> linear(16, 1) on last hidden state

This is the Python equivalent of train_lstm.nos for verification.
"""
import torch
import torch.nn as nn

class SumLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, batch_first=True)
        self.linear = nn.Linear(16, 1, bias=False)

    def forward(self, x):
        # x: [batch, 3, 1] -> hidden: [batch, 3, 16]
        hidden, _ = self.lstm(x)
        # Take last time step: [batch, 16]
        last = hidden[:, -1, :]
        # Project: [batch, 16] -> [batch, 1]
        return self.linear(last)


def make_batch(batch_size):
    raw = torch.randn(batch_size, 3)
    targets = raw.sum(dim=1, keepdim=True)  # [batch, 1]
    inputs = raw.unsqueeze(-1)  # [batch, 3, 1]
    return inputs, targets


def main():
    torch.manual_seed(42)
    model = SumLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    batch_size = 64
    num_steps = 500

    print("Training LSTM to predict sum of 3 numbers:")
    print("  Model: LSTM(1, 16) -> linear(16, 1)")
    print("  Optimizer: Adam(lr=0.01)")
    print(f"  Batch size: {batch_size}")
    print()

    for step in range(num_steps):
        inputs, targets = make_batch(batch_size)
        pred = model(inputs)
        loss = nn.functional.mse_loss(pred, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"  step {step} loss: {loss.item()}")

    # Test on known examples
    print()
    print("Test: sum of [1.0, 2.0, 3.0] should be 6.0")
    test_input = torch.tensor([[[1.0], [2.0], [3.0]]])
    with torch.no_grad():
        pred = model(test_input)
    print(f"  Prediction: {pred.tolist()}")

    print()
    print("Test: sum of [-1.0, 0.5, 2.5] should be 2.0")
    test_input2 = torch.tensor([[[-1.0], [0.5], [2.5]]])
    with torch.no_grad():
        pred2 = model(test_input2)
    print(f"  Prediction: {pred2.tolist()}")


if __name__ == "__main__":
    main()
