import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from dataset import FaceLandmarkDataset, get_transforms
from model import FaceLandmarkModel
import os


def train_model(csv_file, root_dir, num_epochs=10, batch_size=32, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = FaceLandmarkDataset(csv_file, root_dir, transform=get_transforms())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = FaceLandmarkModel()
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for batch in dataloader:
            images = batch['image'].to(device)
            landmarks = batch['landmarks'].view(batch['landmarks'].size(0), -1).to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, landmarks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(dataloader):.4f}")

    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/face_landmark_model.pth")
    print("Model saved to saved_models/face_landmark_model.pth")


if __name__ == "__main__":
    train_model(csv_file="data/landmarks.csv", root_dir="data/images")
