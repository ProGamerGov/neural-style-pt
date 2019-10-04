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

# Download the VGG-16 model and fix the layer names
print("Downloading the VGG-16 model")
sd = load_url("https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth")
map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
sd = OrderedDict([(map[k] if k in map else k,v) for k,v in sd.items()])
torch.save(sd, path.join("models", "vgg16-00b39a1b.pth"))

# Download the NIN model
print("Downloading the NIN model")
if version_info[0] < 3:
    import urllib
    urllib.URLopener().retrieve("https://raw.githubusercontent.com/ProGamerGov/pytorch-nin/master/nin_imagenet.pth", path.join("models", "nin_imagenet.pth"))
else: 
    import urllib.request
    urllib.request.urlretrieve("https://raw.githubusercontent.com/ProGamerGov/pytorch-nin/master/nin_imagenet.pth", path.join("models", "nin_imagenet.pth"))

print("All models have been successfully downloaded")
