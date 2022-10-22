#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 02:00:56 2022

@author: Nathan
"""

import io
from torchvision import transforms
from transformers import ViTModel
from transformers import ViTFeatureExtractor
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from PIL import Image

#load the model
class ViTForImageClassification(nn.Module):
    def __init__(self, num_labels=2):
        super(ViTForImageClassification, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels):
        outputs = self.vit(pixel_values=pixel_values)
        output = self.dropout(outputs.last_hidden_state[:,0])
        logits = self.classifier(output)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if loss is not None:
            return logits, loss.item()
        else:
            return logits, None

model = ViTForImageClassification()
MODEL_PATH = "app/model_dict.pt"
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()


feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')

# image to tensor
def transform_image(byte_array):
    transform = transforms.Compose([
    transforms.Resize((224, 224), interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
    #convert byte input into readable image
    image = Image.open(io.BytesIO(byte_array))
    return transform(image)

# make prediction
def get_prediction(img_tensor):
    #need y to be a long and be in brackets or it will result in error
    y = torch.LongTensor([1])
    with torch.no_grad():
        pixels = torch.tensor(np.stack(feature_extractor(img_tensor)['pixel_values'], axis=0))
        #need a y for this run, don't know why, must be a long type
        prediction,loss = model(pixels,y)
        predicted_class = np.argmax(prediction)
        if predicted_class.item()== 1:
            return "Wild"
        else:
            return "Farm"