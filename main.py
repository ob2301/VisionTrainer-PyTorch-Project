import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
from models import get_model
from utils import train_one_epoch, evaluate

def main(model_name="resnet18", epochs=5, lr=1e-3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_data = datasets.CIFAR10("data", train=True, download=True, transform=transform)
    test_data  = datasets.CIFAR10("data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=128)

    model = get_model(model_name, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}/{epochs} — Loss: {loss:.4f} | Accuracy: {acc*100:.2f}%")

    torch.save(model.state_dict(), f"{model_name}_cifar10.pth")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main("vit_b_16", epochs=3)
