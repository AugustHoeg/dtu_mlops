import os
import torch
import typer
from data import corrupt_mnist
from model import MyAwesomeModel

app = typer.Typer()

@app.command()
def train(lr: float = 1e-3, epochs: int = 50) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(lr)

    model = MyAwesomeModel(hidden_feats=[784, 512, 256, 128, 64, 10], p_drop=0.2).cuda()
    train_set, _ = corrupt_mnist()

    path = "s1_development_environment/exercise_files/final_exercise"
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    #statistics = {"train_loss": 0, "validation_loss": 0, "accuracy": 0}

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for images, labels in train_loader:
            images = images.cuda() # to GPU
            labels = labels.cuda() # to GPU

            # Zero gradients
            optimizer.zero_grad()
            # Forward pass
            output = model(images)
            # Compute loss
            loss = torch.nn.functional.nll_loss(output, labels)
            total_loss += loss.item()
            # Backward pass
            loss.backward()
            # Update weights
            optimizer.step()
        else:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss/len(train_loader)}")

    torch.save(model.state_dict(), os.path.join(path, "corruptmnist_model.pt"))

@app.command()
def evaluate(model_checkpoint: str) -> None:
    """Evaluate a trained model."""
    print("Evaluating like my life depends on it")
    path = "s1_development_environment/exercise_files/final_exercise"
    model_checkpoint = os.path.join(path, model_checkpoint)
    print(model_checkpoint)

    model = MyAwesomeModel(hidden_feats=[784, 512, 256, 128, 64, 10], p_drop=0.2).cuda()
    model.load_state_dict(torch.load(model_checkpoint))

    _, test_set = corrupt_mnist()
    test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=64)

    model.eval()
    correct, total = 0, 0
    for img, target in test_dataloader:
        img, target = img.cuda(), target.cuda()
        y_pred = model(img)
        correct += (y_pred.argmax(dim=1) == target).float().sum().item()
        total += target.size(0)
    print(f"Test accuracy: {correct / total}")


if __name__ == "__main__":
    app()
