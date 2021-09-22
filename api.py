# upload image 
# save image 
# function to make predictions 
from flask import Flask
from flask import render_template
from flask import request

import numpy as np
import pandas as pd
import albumentations as A
import os

import efficientnet_pytorch
import pretrainedmodels
import torch
import torch.nn as nn
from torch.nn import functional as F
from wtfml.data_loaders import image

from wtfml.utils import EarlyStopping
from wtfml.engine import Engine
from wtfml.data_loaders.image import ClassificationLoader

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
DEVICE = 'cuda'
MODEL = None

class EfficientNet(nn.Module):
    def __init__(self):
        super(EfficientNet, self).__init__()
        self.base_model = efficientnet_pytorch.EfficientNet.from_pretrained(
            'efficientnet-b4'
        )
        self.base_model._fc = nn.Linear(
            in_features=1792, 
            out_features=1, 
            bias=True
        )
        
    def forward(self, image, targets):
        out = self.base_model(image)
        # out = F.sigmoid(self.base_model(image))
        loss = nn.BCEWithLogitsLoss()(out, targets.view(-1, 1).type_as(out))
        # loss = 0
        return out, loss

def predict(image_path,model):
    
    
    #model_path = "f'/kaggle/working/model_fold{fold}'"
    #model_path = '/kaggle/working/model_fold0_epoch0.bin'
    # model_path = './model_fold_0.bin'
        
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    test_aug = A.Compose(
        [
            A.Normalize(mean, std, max_pixel_value=255.0, always_apply=True,p=1.0)
        ]
    )
    
    test_images = [image_path]
    test_targets = [0]
    
    test_dataset = ClassificationLoader(
        image_paths = test_images,
        targets= test_targets,
        resize = None,
        augmentations = test_aug
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers=0
    )
    #Earlier defined class for model

    
    predictions_op = Engine.predict(
        test_loader,
        model,
        DEVICE
    )
    # predictions_op = torch.sigmoid(predictions_op)
    print(predictions_op)
    l = np.vstack((predictions_op)).ravel()
    l = torch.Tensor(l)
    print(l)
    an = torch.sigmoid(l)
    print(an)
    np_arr = an.cpu().detach().numpy()
    print(np_arr)

    # return np.vstack((predictions_op)).ravel()
    return np_arr


@app.route('/', methods=["GET","POST"])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image'] #name given in html should be same
        if image_file:
            image_location = os.path.join(UPLOAD_FOLDER,image_file.filename)
            image_file.save(image_location)
            pred = predict(image_location,MODEL)[0]
            # print(pred)
            return render_template('index.html',prediction = pred,image_loc = image_file.filename)

    return render_template('index.html',prediction = 0,image_loc = None) #jinja2

if __name__ == '__main__':
    MODEL = EfficientNet()
    MODEL.load_state_dict(torch.load('weights\model_fold_0.bin'))
    MODEL.to(DEVICE)
    app.run(port = 1200,debug=True)