import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# This is a PyTorch implementation of the FaceNet model for face recognition.
class FaceNetBackbone(nn.Module):
    def __init__(self, embedding_size=128, pretrained=False):
        super(FaceNetBackbone, self).__init__()

        base_model = models.resnet50(pretrained=pretrained)
        modules = list(base_model.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)

        self.embedding = nn.Linear(base_model.fc.in_features, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.embedding(x)
        x = self.bn(x)
        x = F.normalize(x, p=2, dim=1)
        return x

# Load the FaceNet model and the pretrained weights.
def load_facenet_model(model_path: str, device=None) -> FaceNetBackbone:
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceNetBackbone(embedding_size=128, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
