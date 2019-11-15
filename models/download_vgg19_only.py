import torch
from os import path
from sys import version_info
from collections import OrderedDict
from torch.utils.model_zoo import load_url

# Download the VGG-19 model and fix the layer names
print("Downloading the VGG-19 model")
sd = load_url("https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg19-d01eb7cb.pth")
map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
sd = OrderedDict([(map[k] if k in map else k,v) for k,v in sd.items()])
torch.save(sd, path.join("models", "vgg19-d01eb7cb.pth"))

print("VGG-19 downloaded")
