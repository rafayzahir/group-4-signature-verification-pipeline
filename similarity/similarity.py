import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _load_model():
    global _model
    if _model is None:
        weights = models.ResNet50_Weights.DEFAULT
        m = models.resnet50(weights=weights)
        m.fc = nn.Identity()  # remove classifier, keep embeddings
        m.eval()
        _model = m.to(_device)
    return _model


def get_embedding(image_path):
    img = Image.open(image_path).convert("RGB")
    x = _transform(img).unsqueeze(0).to(_device)
    with torch.no_grad():
        emb = _load_model()(x)
    return emb.squeeze(0)


def compute_similarity(image_path_1, image_path_2):
    """Takes two signature image paths, returns a score from 0 to 100."""
    e1 = get_embedding(image_path_1)
    e2 = get_embedding(image_path_2)
    cos = torch.nn.functional.cosine_similarity(e1, e2, dim=0).item()
    score = max(0.0, min(1.0, cos)) * 100
    return round(score, 2)