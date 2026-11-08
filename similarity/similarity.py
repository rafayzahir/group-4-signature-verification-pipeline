import os

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None

# Trained weights (fine-tuned on signature pairs). Not stored in git.
_DEFAULT_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "similarity_resnet50_v3.pth")
WEIGHTS_PATH = os.environ.get("SIMILARITY_WEIGHTS", _DEFAULT_WEIGHTS)

# Score at or above this counts as a match (chosen on held-out writers)
THRESHOLD = 60

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def _load_model():
    global _model
    if _model is None:
        if not os.path.exists(WEIGHTS_PATH):
            raise FileNotFoundError(
                "Trained weights not found at " + WEIGHTS_PATH +
                ". Put similarity_resnet50_v3.pth next to similarity.py "
                "or set the SIMILARITY_WEIGHTS environment variable."
            )
        m = models.resnet50(weights=None)
        m.fc = nn.Identity()  # remove classifier, keep embeddings
        state = torch.load(WEIGHTS_PATH, map_location=_device, weights_only=True)
        # training saved the model with a "backbone." prefix
        state = {k.replace("backbone.", "", 1): v for k, v in state.items()}
        m.load_state_dict(state)
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