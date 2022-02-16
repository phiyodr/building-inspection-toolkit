import numpy as np
import torch
from torch import nn
from torchvision import models

from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import MemoryEfficientSwish
import time

from PIL import Image
import matplotlib.pyplot as plt
import json
from pathlib import Path
import sys

# Dict to find the suiting EfficientNet model according to the resolution of the input-images:
efnet_dict = {'b0': 224, 'b1': 240, 'b2': 260, 'b3': 300,
              'b4': 380, 'b5': 456, 'b6': 528, 'b7': 600
              }


class DaclNet(nn.Module):
    def __init__(self, base_name, resolution, hidden_layers, num_class, drop_prob=0.2, freeze_base=True):
        ''' 
        Builds a network separated into a base model and classifier with arbitrary hidden layers.
        
        Attributes
        ---------
        base_name: string, basemodel for the NN
        resolution: resolution of the input-images, example: 224, 240...(look efnet_dic), Only needed for EfficientNet
        hidden_layers: list of integers, the sizes of the hidden layers
        drop_prob: float, dropout probability
        freeze_base: boolean, choose if you want to freeze the parameters of the base model
        num_class: integer, size of the output layer according to the number of classes

        Example
        ---------
        model = Network(base_name='efficientnet', resolution=224, hidden_layers=[32,16], num_class=6, drop_prob=0.2, freeze_base=True)
        '''
        super(DaclNet, self).__init__()
        self.base_name = base_name
        self.resolution = resolution
        self.hidden_layers = hidden_layers
        self.freeze_base = freeze_base

        if self.base_name == 'mobilenet':
            base = models.mobilenet_v3_large(pretrained=True) 
            modules = list(base.children())[:-1] 
            self.base = nn.Sequential(*modules)
            # for pytorch model:
            if hidden_layers:
                self.classifier = nn.ModuleList([nn.Linear(base.classifier[0].in_features, self.hidden_layers[0])]) 
            else:
                self.classifier = nn.Linear(base.classifier[0].in_features, num_class)

            self.activation = nn.Hardswish()

        elif self.base_name == 'resnet':
            base = models.resnet50(pretrained=True) 
            modules = list(base.children())[:-1]
            self.base = nn.Sequential(*modules)
            if self.hidden_layers:
                self.classifier = nn.ModuleList([nn.Linear(base.fc.in_features, self.hidden_layers[0])])
            else:
                self.classifier = nn.Linear(base.fc.in_features, num_class)   
            self.activation = nn.ELU() # Eliminates dying RELU problem according to: https://tungmphung.com/elu-activation-a-comprehensive-analysis/

        elif self.base_name == 'efficientnet':   
            # Implementing Effnet the same way like the others didn't work, because omitting the last module also removes last batchnorm, avg-pooling   
            for ver in efnet_dict:
                if efnet_dict[ver] == self.resolution:
                    self.version = ver
                    full_name = self.base_name+'-'+ver
            self.base = EfficientNet.from_pretrained(model_name=full_name) 
            if self.hidden_layers:
                self.classifier = nn.ModuleList([nn.Linear(self.base._fc.in_features, self.hidden_layers[0])])
            else:
                self.classifier = nn.Linear(self.base._fc.in_features, num_class)   
            self.activation = MemoryEfficientSwish()
        elif self.base_name == 'mobilenetv2':
            base = models.mobilenet.mobilenet_v2(pretrained=True)
            print(base)
            modules = list(base.children())[:-1]
            self.base = nn.Sequential(*modules)
            if hidden_layers:
                # Input features = depth of the last BatchNorm layer = input features of first layer of original classifier:
                self.classifier = nn.ModuleList([nn.Linear(base.classifier[1].in_features, self.hidden_layers[0])]) 
            else:
                self.classifier = nn.Linear(base.classifier[1].in_features, num_class)
            self.activation = nn.ReLU()
 

        else:
            raise NotImplementedError    
        
        # freeze the base
        if self.freeze_base:
            for param in self.base.parameters(): 
                param.requires_grad_(False)
        
        self.dropout = nn.Dropout(p=drop_prob, inplace=True)

        # classifier
        # Add a variable number of more hidden layers
        if self.hidden_layers:
            layer_sizes = zip(self.hidden_layers[:-1], self.hidden_layers[1:])     # The default baseV3Large model has one hidden layer with 1280 nodes
            self.classifier.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
            # add output layer to classifier
            self.classifier.append(nn.Linear(self.hidden_layers[-1], num_class))
        else:
            pass
        
    def forward(self, input_batch):
        ''' 
        Performs the feed-forward process for the input batch and return the logits

        Arguments
        ---------
        input_batch: torch.Tensor, Multidimensional array holding elements of datatype: torch.float32, 
                     it's shape is: [1, 3, 224, 224] according to N x C x H x W,
                     The input batch carries all pixel values from the images inside teh batch
        Note
        ---------
        Every model uses 2d-Average-Pooling with output_size=1 after the feature extraction or rather before flattening.
        The pooling layer of ResNet50 and MobileNetV3 was kept in the squential -> Doesn't have to be called in forward!
        EffNet had to be implemented with the AdaptiveAvgpool2d in this forward function because of missing pooling when
        calling: "effnet.extract_features(input_batch)"
        Also mobilenetV2 needs the manually added pooling layer.

        Returns
        ---------
        logits: torch.Tensor, shape: [1, num_class], datatype of elements: float
        '''
        # Check if model is one that needs Pooling layer
        if self.base_name in ['efficientnet', 'mobilenetv2']:
            if self.base_name == 'efficientnet':
                x = self.base.extract_features(input_batch)
            else:
                # For MobileNetV2:
                x= self.base(input_batch)
            pool = nn.AdaptiveAvgPool2d(1)
            x = pool(x)
        else:
            # For any other model don't additionally apply pooling:
            x = self.base(input_batch)
        
        x = self.dropout(x)         # Originally only in EfficientNet a Dropout is aplied after last bottleneck, in others not!  
        x = x.view(x.size(0), -1)   # Or: x.flatten(start_dim=1)
        if self.hidden_layers:    
            for i,each in enumerate(self.classifier):
                if i < len(self.classifier)-1:
                    x = self.activation(each(x))
                    x = self.dropout(x)
                else:
                    logits = each(x)
                    break
        else:
            logits = self.classifier(x)

        return logits


def preprocess_img(img):
    if isinstance(img, str):
        img = Image.open(img)
    img = img.resize((224,224))
    img_np = np.array(img)

    img_np = (img_np / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    img_np = img_np.transpose(2, 0, 1)
    img = torch.from_numpy(img_np)
    img = img.unsqueeze_(0)
    img = img.type(torch.FloatTensor)
    return img


def _print_prediction_bar(prediction_probability, label):
    assert (prediction_probability>=0.0) and (prediction_probability<=1.0)
    bar_size = 40
    bar = '█' * int(bar_size * prediction_probability)
    bar = bar + '' * int(bar_size * (1-prediction_probability))

    sys.stdout.write(f"{label.ljust(20)} [{bar:{bar_size}s}] {prediction_probability*100:>6.2f}% \n")
    sys.stdout.flush()

def make_prediction(model, img, metadata, print_predictions=True, preprocess_image=True):
    # Read image if is is a string
    if isinstance(img, str):
        img = Image.open(img)
    # Preprocess image
    if preprocess_image:
        img = preprocess_img(img)

    model.eval()
    tic = time.perf_counter()
    with torch.no_grad():
        logits = model(img)
    probabilities = torch.sigmoid(logits).numpy()[0]
    predictions = probabilities > 0.5
    toc = time.perf_counter()

    if print_predictions:
        n_classes = len(metadata["id2labels"])
        for i in range(n_classes):
            label_name = metadata["id2labels"][str(i)]
            _print_prediction_bar(probabilities[i], label_name)
        print(f"Inference time (CPU): {(toc - tic)*1000:0.2f} ms")

    return probabilities, predictions

if __name__ == "__main__":
    from bikit.utils import load_model, get_metadata
    img_path = "/home/philipp/Documents/MyLocalProjects/dacl_project/bridge-inspection-toolkit/bikit/data/11_001990.jpg"
    model_name = "MCDS_MobileNetV3Large"

    model, metadata = load_model(model_name)
    make_prediction(model, img_path, metadata)