import copy
import torch
import torch.nn as nn
import torch.optim as optim
from CaffeLoader import loadCaffemodel



######################################################
# StyleNet model

class StyleNet(torch.nn.Module):
        
    def __init__(self, params, dtype, multidevice, backward_device):
        super(StyleNet, self).__init__()

        self.content_seg = "this is temp"
        self.tv_weight, self.content_weight, self.style_weight = 1e-3, 5e0, 1e2
        self.dtype, self.multidevice, self.backward_device = dtype, multidevice, backward_device
        self.content_losses, self.style_losses, self.tv_losses = [], [], []

        content_layers = params.content_layers.split(',')
        style_layers = params.style_layers.split(',')

        next_content_idx, next_style_idx, c, r = 1, 1, 0, 0
        cnn, layerList = loadCaffemodel(params.model_file, params.pooling, params.gpu, params.disable_check)
        
        net = nn.Sequential()
        
        tv_mod = TVLoss(self.tv_weight).type(dtype)
        net.add_module(str(len(net)), tv_mod)
        self.tv_losses.append(tv_mod)

        for i, layer in enumerate(list(cnn), 1):
            if next_content_idx <= len(content_layers) or next_style_idx <= len(style_layers):

                if isinstance(layer, nn.Conv2d):
                    net.add_module(str(len(net)), layer)

                    if layerList['C'][c] in content_layers:
                        print("Setting up content layer " + str(i) + ": " + str(layerList['C'][c]))
                        loss_module = ContentLoss(self.content_weight)
                        net.add_module(str(len(net)), loss_module)
                        self.content_losses.append(loss_module)

                    if layerList['C'][c] in style_layers:
                        print("Setting up style layer " + str(i) + ": " + str(layerList['C'][c]))
                        if self.content_seg != None:
                            loss_module = MaskedStyleLoss(self.style_weight)
                        else:
                            loss_module = StyleLoss(self.style_weight)
                        net.add_module(str(len(net)), loss_module)
                        self.style_losses.append(loss_module)
                    
                    c += 1

                if isinstance(layer, nn.ReLU):
                    net.add_module(str(len(net)), layer)

                    if layerList['R'][r] in content_layers:
                        print("Setting up content layer " + str(i) + ": " + str(layerList['R'][r]))
                        loss_module = ContentLoss(self.content_weight)
                        net.add_module(str(len(net)), loss_module)
                        self.content_losses.append(loss_module)
                        next_content_idx += 1

                    if layerList['R'][r] in style_layers:
                        print("Setting up style layer " + str(i) + ": " + str(layerList['R'][r]))
                        if self.content_seg != None:
                            loss_module = MaskedStyleLoss(self.style_weight)
                        else:
                            loss_module = StyleLoss(self.style_weight)
                        net.add_module(str(len(net)), loss_module)
                        self.style_losses.append(loss_module)
                        next_style_idx += 1
                    
                    r += 1

                if isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                    net.add_module(str(len(net)), layer)
        
        self.net = net
        print(self.net)
        
        # Freeze the network to prevent unnecessary gradient calculations
        for param in self.net.parameters():
            param.requires_grad = False

        # Setup multidevice
        if self.multidevice:
            self.setup_multi_device(params.gpu, params.multidevice_strategy)

    def setup_multi_device(self, gpu, multidevice_strategy):
        from CaffeLoader import ModelParallel
        self.multidevice = True
        assert len(gpu.split(',')) - 1 == len(multidevice_strategy.split(',')), \
            "The number of -multidevice_strategy layer indices minus 1, must be equal to the number of -gpu devices."
        self.net = ModelParallel(self.net, gpu, multidevice_strategy)
        
    def set_content_weight(self, content_weight):
        if self.content_weight == content_weight:
            return
        self.content_weight = content_weight
        for layer in self.net:
            if isinstance(layer, ContentLoss):
                layer.weight = self.content_weight

    def set_style_weight(self, style_weight):
        if self.style_weight == style_weight:
            return
        self.style_weight = style_weight
        for layer in self.net:
            if isinstance(layer, MaskedStyleLoss):
                layer.weight = self.style_weight
            elif isinstance(layer, StyleLoss):
                layer.weight = self.style_weight
        
    def set_tv_weight(self, tv_weight):
        if self.tv_weight == tv_weight:
            return
        self.tv_weight = tv_weight
        for layer in self.net:
            if isinstance(layer, TVLoss):
                layer.weight = self.tv_weight
 
    def setup_masks(self, content_masks, style_masks):
        self.content_masks, self.style_masks = copy.deepcopy(content_masks), copy.deepcopy(style_masks)
        self.num_styles = len(self.style_masks)
        for layer in self.net:
            if isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                for k in range(self.num_styles):
                    h, w = self.content_masks[k].shape
                    h, w = int(h/2), int(w/2)
                    self.content_masks[k] = torch.nn.functional.interpolate(
                        self.content_masks[k].repeat(1,1,1,1), mode='bilinear', size=(h, w))[0][0]
                for j in range(self.num_styles):
                    for k in range(self.num_styles):
                        h, w = self.style_masks[j][k].shape
                        h, w = int(h/2), int(w/2)
                        self.style_masks[j][k] = torch.nn.functional.interpolate(
                            self.style_masks[j][k].repeat(1,1,1,1), mode='bilinear', size=(h, w))[0][0]
                    self.style_masks[j] = copy.deepcopy(self.style_masks[j])

            elif isinstance(layer, nn.Conv2d):
                sap = nn.AvgPool2d(kernel_size=(3,3), stride=(1, 1), padding=(1,1))
                for k in range(self.num_styles):
                    self.content_masks[k] = sap(self.content_masks[k].repeat(1,1,1))[0].clone()
                for j in range(self.num_styles):
                    for k in range(self.num_styles):
                        self.style_masks[j][k] = sap(self.style_masks[j][k].repeat(1,1,1))[0].clone()
                    self.style_masks[j] = copy.deepcopy(self.style_masks[j])

            if isinstance(layer, MaskedStyleLoss):
                layer.set_masks(self.content_masks, self.style_masks)
                                
    def capture_content(self, content_image):
        for l in self.content_losses:
            l.mode = 'capture'
        for l in self.style_losses:
            l.mode = 'none'
        print("Capturing content targets")
        self.forward(content_image, loss_mode=False)

    def capture_style(self, style_images, style_blend_weights):
        for l in self.content_losses:
            l.mode = 'none'
        for l in self.style_losses:
            l.mode = 'capture'
        for i, style_image in enumerate(style_images):
            print("Capturing style target " + str(i+1))
            for l in self.style_losses:
                l.blend_weight = style_blend_weights[i]
            self.forward(style_image, loss_mode=False)

    def forward(self, x, loss_mode=True):
        if loss_mode == True:
            for l in self.content_losses:
                l.mode = 'loss'
            for l in self.style_losses:
                l.mode = 'loss'
        self.net(x)
        
    def get_loss(self):
        loss = 0
        for mod in self.content_losses:
            loss += mod.loss.to(self.backward_device)
        for mod in self.style_losses:
            loss += mod.loss.to(self.backward_device)
        if self.tv_weight > 0:
            for mod in self.tv_losses:
                loss += mod.loss.to(self.backward_device)
        return loss



######################################################
# Content Loss module

class ContentLoss(nn.Module):

    def __init__(self, strength):
        super(ContentLoss, self).__init__()
        self.strength = strength
        self.crit = nn.MSELoss()
        self.mode = 'none'

    def forward(self, input):
        if self.mode == 'loss':
            self.loss = self.crit(input, self.target) * self.strength
        elif self.mode == 'capture':
            self.target = input.detach()
        return input



######################################################
# Stlye Loss module (no masks)

class StyleLoss(nn.Module):

    def __init__(self, strength):
        super(StyleLoss, self).__init__()
        self.target = torch.Tensor()
        self.strength = strength
        #self.gram = GramMatrix()
        self.crit = nn.MSELoss()
        self.mode = 'none'
        self.blend_weight = None

    def forward(self, input):
        self.G = self.gram(input)
        self.G = self.G.div(input.nelement())
        if self.mode == 'capture':
            if self.blend_weight == None:
                self.target = self.G.detach()
            elif self.target.nelement() == 0:
                self.target = self.G.detach().mul(self.blend_weight)
            else:
                self.target = self.target.add(self.blend_weight, self.G.detach())
        elif self.mode == 'loss':
            self.loss = self.strength * self.crit(self.G, self.target)
        return input

    def gram(self, input):
        B, C, H, W = input.size()
        x_flat = input.view(C, H * W)
        return torch.mm(x_flat, x_flat.t())




######################################################
# Stlye Loss module (with masks)

class MaskedStyleLoss(nn.Module):

    def __init__(self, strength):
        super(MaskedStyleLoss, self).__init__()
        self.strength = strength
        #self.gram = GramMatrix()
        self.crit = nn.MSELoss()
        self.mode = 'none'
        self.blend_weight = None
        
    def set_masks(self, content_masks, style_masks):
        self.content_masks = copy.deepcopy(content_masks)
        self.style_masks = copy.deepcopy(style_masks)
        self.target_grams = []
        self.masked_grams = []
        self.masked_features = []
        self.num_styles = len(self.style_masks)
        self.capture_count = 0

    def forward(self, input):
        if self.mode == 'capture':
            masks = self.style_masks[self.capture_count]
            self.capture_count += 1
        elif self.mode == 'loss':
            masks = self.content_masks
            self.style_masks = None
        if self.mode != 'none':
            loss = 0
            for j in range(self.num_styles):
                l_mask_ori = masks[j].clone()
                l_mask = l_mask_ori.repeat(1,1,1).expand(input.size())
                l_mean = l_mask_ori.mean()
                masked_feature = l_mask.mul(input)
                masked_gram = self.gram(masked_feature).clone()
                if l_mean > 0:
                    masked_gram = masked_gram.div(input.nelement() * l_mean)
                if self.mode == 'capture':
                    if j >= len(self.target_grams):
                        self.target_grams.append(masked_gram.detach().mul(self.blend_weight))
                        self.masked_grams.append(self.target_grams[j].clone())
                        self.masked_features.append(masked_feature)
                    else:
                        self.target_grams[j] += masked_gram.detach().mul(self.blend_weight)
                elif self.mode == 'loss':
                    self.masked_grams[j] = masked_gram
                    self.masked_features[j] = masked_feature
                    loss += self.crit(self.masked_grams[j], self.target_grams[j]) * l_mean * self.strength
            self.loss = loss
        return input

    def gram(self, input):
        B, C, H, W = input.size()
        x_flat = input.view(C, H * W)
        return torch.mm(x_flat, x_flat.t())


######################################################
# TV regularization

class TVLoss(nn.Module):

    def __init__(self, strength):
        super(TVLoss, self).__init__()
        self.strength = strength

    def forward(self, input):
        self.x_diff = input[:,:,1:,:] - input[:,:,:-1,:]
        self.y_diff = input[:,:,:,1:] - input[:,:,:,:-1]
        self.loss = self.strength * (torch.sum(torch.abs(self.x_diff)) + torch.sum(torch.abs(self.y_diff)))
        return input



# class GramMatrix(nn.Module):

#     def forward(self, input):
#         B, C, H, W = input.size()
#         x_flat = input.view(C, H * W)
#         return torch.mm(x_flat, x_flat.t())



######################################################
# Optimizer

def setup_optimizer(img, params, num_iterations):
    if params.optimizer == 'lbfgs':
        print("Running optimization with L-BFGS")
        optim_state = {
            'max_iter': num_iterations,
            'tolerance_change': -1,
            'tolerance_grad': -1,
        }
        if params.lbfgs_num_correction != 100:
            optim_state['history_size'] = params.lbfgs_num_correction
        optimizer = optim.LBFGS([img], **optim_state)
        loopVal = 1
    elif params.optimizer == 'adam':
        print("Running optimization with ADAM")
        optimizer = optim.Adam([img], lr = params.learning_rate)
        loopVal = num_iterations - 1
    return optimizer, loopVal



######################################################
# GPU config

def setup_gpu(params):
    def setup_cuda():
        if 'cudnn' in params.backend:
            torch.backends.cudnn.enabled = True
            if params.cudnn_autotune:
                torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.enabled = False
    def setup_cpu():
        if 'mkl' in params.backend and 'mkldnn' not in params.backend:
            torch.backends.mkl.enabled = True
        elif 'mkldnn' in params.backend:
            raise ValueError("MKL-DNN is not supported yet.")
        elif 'openmp' in params.backend:
            torch.backends.openmp.enabled = True
    multidevice = False
    if "," in str(params.gpu):
        devices = params.gpu.split(',')
        multidevice = True
        if 'c' in str(devices[0]).lower():
            backward_device = "cpu"
            setup_cuda(), setup_cpu()
        else:
            backward_device = "cuda:" + devices[0]
            setup_cuda()
        dtype = torch.FloatTensor
    elif "c" not in str(params.gpu).lower():
        setup_cuda()
        dtype, backward_device = torch.cuda.FloatTensor, "cuda:" + str(params.gpu)
    else:
        setup_cpu()
        dtype, backward_device = torch.FloatTensor, "cpu"
    return dtype, multidevice, backward_device

