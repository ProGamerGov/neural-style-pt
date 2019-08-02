#!/usr/bin/env python
import torch
import torch.nn as nn


class VGG(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )


class NIN(nn.Module):
    def __init__(self, pooling):
        super(NIN, self).__init__()
        if pooling == 'max':
            pool2d = nn.MaxPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True)
        elif pooling == 'avg':
            pool2d = nn.AvgPool2d((3, 3),(2, 2),(0, 0),ceil_mode=True)

        self.features = nn.Sequential(
            nn.Conv2d(3,96,(11, 11),(4, 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,96,(1, 1)),
            nn.ReLU(inplace=True),
            pool2d,
            nn.Conv2d(96,256,(5, 5),(1, 1),(2, 2)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,256,(1, 1)),
            nn.ReLU(inplace=True),
            pool2d,
            nn.Conv2d(256,384,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,384,(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(384,384,(1, 1)),
            nn.ReLU(inplace=True),
            pool2d,
            nn.Dropout(0.5),
            nn.Conv2d(384,1024,(3, 3),(1, 1),(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024,1024,(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024,1000,(1, 1)),
            nn.ReLU(inplace=True),
            nn.AvgPool2d((6, 6),(1, 1),(0, 0),ceil_mode=True),
            nn.Softmax(),
        )



def buildSequential(channel_list, pooling):
    layers = []
    in_channels = 3
    if pooling == 'max':
        pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
    elif pooling == 'avg':
        pool2d = nn.AvgPool2d(kernel_size=2, stride=2)
    else: 
        raise ValueError("Unrecognized pooling parameter")
    for c in channel_list:
        if c == 'P':
            layers += [pool2d]
        else:
            conv2d = nn.Conv2d(in_channels, c, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = c
    return nn.Sequential(*layers)


channel_list = {
'VGG-16': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 'P', 512, 512, 512, 'P', 512, 512, 512, 'P'],
'VGG-19': [64, 64, 'P', 128, 128, 'P', 256, 256, 256, 256, 'P', 512, 512, 512, 512, 'P', 512, 512, 512, 512, 'P'],
}

nin_dict = {
'C': ['conv1', 'cccp1', 'cccp2', 'conv2', 'cccp3', 'cccp4', 'conv3', 'cccp5', 'cccp6', 'conv4-1024', 'cccp7-1024', 'cccp8-1024'], 
'R': ['relu0', 'relu1', 'relu2', 'relu3', 'relu5', 'relu6', 'relu7', 'relu8', 'relu9', 'relu10', 'relu11', 'relu12'],
'P': ['pool1', 'pool2', 'pool3', 'pool4'],
'D': ['drop'],
}
vgg16_dict = {
'C': ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3'],
'R': ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu3_3', 'relu4_1', 'relu4_2', 'relu4_3', 'relu5_1', 'relu5_2', 'relu5_3'],
'P': ['pool1', 'pool2', 'pool3', 'pool4', 'pool5'],
}
vgg19_dict = {
'C': ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4'],
'R': ['relu1_1', 'relu1_2', 'relu2_1', 'relu2_2', 'relu3_1', 'relu3_2', 'relu3_3', 'relu3_4', 'relu4_1', 'relu4_2', 'relu4_3', 'relu4_4', 'relu5_1', 'relu5_2', 'relu5_3', 'relu5_4'],
'P': ['pool1', 'pool2', 'pool3', 'pool4', 'pool5'],
}


def modelSelector(model_file, pooling):
    if "vgg" in model_file:
        if "19" in model_file:
            print("VGG-19 Architecture Detected")
            cnn, layerList = VGG(buildSequential(channel_list['VGG-19'], pooling)), vgg19_dict
        elif "16" in model_file:
            print("VGG-16 Architecture Detected")
            cnn, layerList = VGG(buildSequential(channel_list['VGG-16'], pooling)), vgg16_dict
        else:
            raise ValueError("VGG architecture not recognized.")    
    elif "nin" in model_file:
        print("NIN Architecture Detected")
        cnn, layerList = NIN(pooling), nin_dict
    elif model_file == "":
        print("")
        print("Please run 'neural-style -download_models' if you have not done so already.")
        print("Then select a model with: '-model_file path/to/model'.")
        raise ValueError("No model selected.")
    else:
        raise ValueError("Model architecture not recognized.")
    return cnn, layerList

# Print like Torch7/loadcaffe
def print_loadcaffe(cnn, layerList): 
    c = 0
    for l in list(cnn):
         if "Conv2d" in str(l):
             in_c, out_c, ks  = str(l.in_channels), str(l.out_channels), str(l.kernel_size)
             print(layerList['C'][c] +": " +  (out_c + " " + in_c + " " + ks).replace(")",'').replace("(",'').replace(",",'') )
             c+=1
         if c == len(layerList['C']):
             break

# Load the model, and configure pooling layer type
def loadCaffemodel(model_file, pooling, use_gpu):
    cnn, layerList = modelSelector(str(model_file).lower(), pooling)
    cnn.load_state_dict(torch.load(model_file))
    print("Successfully loaded " + str(model_file))

    # Maybe convert the model to cuda now, to avoid later issues
    if use_gpu > -1:
        cnn = cnn.cuda()
    cnn = cnn.features 

    print_loadcaffe(cnn, layerList)

    return cnn, layerList




# Download pretrained models
def downloadModels():
    import os, sys
    from collections import OrderedDict
    from torch.utils.model_zoo import load_url

    current_path = os.getcwd()
    
    # Download the VGG-19 model and fix the layer names
    sd = load_url("https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg19-d01eb7cb.pth")
    map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
    sd = OrderedDict([(map[k] if k in map else k,v) for k,v in sd.items()])
    torch.save(sd, current_path + "/vgg19-d01eb7cb.pth")

    # Download the VGG-16 model and fix the layer names
    sd = load_url("https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth")
    map = {'classifier.1.weight':u'classifier.0.weight', 'classifier.1.bias':u'classifier.0.bias', 'classifier.4.weight':u'classifier.3.weight', 'classifier.4.bias':u'classifier.3.bias'}
    sd = OrderedDict([(map[k] if k in map else k,v) for k,v in sd.items()])
    torch.save(sd, current_path + "/vgg16-00b39a1b.pth")

    # Download the NIN model
    if sys.version_info[0] < 3:
        import urllib
        urllib.URLopener().retrieve("https://raw.githubusercontent.com/ProGamerGov/pytorch-nin/master/nin_imagenet.pth", current_path + "/nin_imagenet.pth")
    else: 
        import urllib.request
        urllib.request.urlretrieve("https://raw.githubusercontent.com/ProGamerGov/pytorch-nin/master/nin_imagenet.pth", current_path + "/nin_imagenet.pth")
        
        
    sys.exit()