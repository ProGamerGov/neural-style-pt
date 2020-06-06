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


class VGG_SOD(nn.Module):
    def __init__(self, features, num_classes=100):
        super(VGG_SOD, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 100),
        )


class VGG_FCN32S(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG_FCN32S, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Conv2d(512,4096,(7, 7)),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv2d(4096,4096,(1, 1)),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )


class VGG_PRUNED(nn.Module):
    def __init__(self, features, num_classes=1000):
        super(VGG_PRUNED, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.5),
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



class ModelParallel(nn.Module):
    def __init__(self, net, device_ids, device_splits):
        super(ModelParallel, self).__init__()
        self.device_list = self.name_devices(device_ids.split(','))
        self.chunks = self.chunks_to_devices(self.split_net(net, device_splits.split(',')))

    def name_devices(self, input_list):
        device_list = []
        for i, device in enumerate(input_list):
            if str(device).lower() != 'c':
                device_list.append("cuda:" + str(device))
            else:
                device_list.append("cpu")
        return device_list

    def split_net(self, net, device_splits):
        chunks, cur_chunk = [], nn.Sequential()
        for i, l in enumerate(net):
            cur_chunk.add_module(str(i), net[i])
            if str(i) in device_splits and device_splits != '':
                del device_splits[0]
                chunks.append(cur_chunk)
                cur_chunk = nn.Sequential()
        chunks.append(cur_chunk)
        return chunks

    def chunks_to_devices(self, chunks):
        for i, chunk in enumerate(chunks):
            chunk.to(self.device_list[i])
        return chunks

    def c(self, input, i):
        if input.type() == 'torch.FloatTensor' and 'cuda' in self.device_list[i]:
            input = input.type('torch.cuda.FloatTensor')
        elif input.type() == 'torch.cuda.FloatTensor' and 'cpu' in self.device_list[i]:
            input = input.type('torch.FloatTensor')
        return input

    def forward(self, input):
        for i, chunk in enumerate(self.chunks):
            if i < len(self.chunks) -1:
                input = self.c(chunk(self.c(input, i).to(self.device_list[i])), i+1).to(self.device_list[i+1])
            else:
                input = chunk(input)
        return input



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
'VGG-16p': [24, 22, 'P', 41, 51, 'P', 108, 89, 111, 'P', 184, 276, 228, 'P', 512, 512, 512, 'P'],
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


def modelSelector(model_file, pooling, verbose=True):
    vgg_list = ["fcn32s", "pruning", "sod", "vgg"]
    if any(name in model_file for name in vgg_list):
        if "pruning" in model_file:
            if verbose: print("VGG-16 Architecture Detected")
            if verbose: print("Using The Channel Pruning Model")
            cnn, layerList = VGG_PRUNED(buildSequential(channel_list['VGG-16p'], pooling)), vgg16_dict
        elif "fcn32s" in model_file:
            if verbose: print("VGG-16 Architecture Detected")
            if verbose: print("Using the fcn32s-heavy-pascal Model")
            cnn, layerList = VGG_FCN32S(buildSequential(channel_list['VGG-16'], pooling)), vgg16_dict
        elif "sod" in model_file:
            if verbose: print("VGG-16 Architecture Detected")
            if verbose: print("Using The SOD Fintune Model")
            cnn, layerList = VGG_SOD(buildSequential(channel_list['VGG-16'], pooling)), vgg16_dict
        elif "19" in model_file:
            if verbose: print("VGG-19 Architecture Detected")
            cnn, layerList = VGG(buildSequential(channel_list['VGG-19'], pooling)), vgg19_dict
        elif "16" in model_file:
            if verbose: print("VGG-16 Architecture Detected")
            cnn, layerList = VGG(buildSequential(channel_list['VGG-16'], pooling)), vgg16_dict
        else:
            raise ValueError("VGG architecture not recognized.")
    elif "nin" in model_file:
        if verbose: print("NIN Architecture Detected")
        cnn, layerList = NIN(pooling), nin_dict
    else:
        raise ValueError("Model architecture not recognized.")
    return cnn, layerList


# Print like Torch7/loadcaffe
def print_loadcaffe(cnn, layerList, verbose=True):
    c = 0
    for l in list(cnn):
         if "Conv2d" in str(l):
             in_c, out_c, ks  = str(l.in_channels), str(l.out_channels), str(l.kernel_size)
             if verbose: print(layerList['C'][c] +": " +  (out_c + " " + in_c + " " + ks).replace(")",'').replace("(",'').replace(",",'') )
             c+=1
         if c == len(layerList['C']):
             break


# Load the model, and configure pooling layer type
def loadCaffemodel(model_file, pooling, use_gpu, disable_check, verbose=True):
    cnn, layerList = modelSelector(str(model_file).lower(), pooling, verbose)

    cnn.load_state_dict(torch.load(model_file), strict=(not disable_check))
    if verbose:
        print("Successfully loaded " + str(model_file))

    # Maybe convert the model to cuda now, to avoid later issues
    if "c" not in str(use_gpu).lower() or "c" not in str(use_gpu[0]).lower():
        cnn = cnn.cuda()
    cnn = cnn.features

    print_loadcaffe(cnn, layerList, verbose)

    return cnn, layerList
