
import os
import io
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image



def model_fn(model_dir):
    model = models.resnet50(pretrained=False)

#     for param in model.parameters():
#         param.requires_grad = False

    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 32),
        nn.ReLU(inplace=True),
        nn.Linear(num_features, 5)
    )

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    return model


def input_fn(request_body, content_type):
    image = Image.open(io.BytesIO(request_body))

    transformation = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()])

    return transformation(image).unsqueeze(0)

