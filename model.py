import torch.nn as nn
import torchvision.models as models


class FaceLandmarkModel(nn.Module):
    def __init__(self, num_landmarks=68):
        super(FaceLandmarkModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_landmarks * 2)

    def forward(self, x):
        return self.resnet(x)
