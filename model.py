import copy
import torch
import torch.utils.cpp_extension
import torch.nn as nn
import torch.optim as optim
from CaffeLoader import loadCaffemodel

#cpp = torch.utils.cpp_extension.load(name="histogram", sources=["histogram.cpp", "histogram.cu"])

######################################################
# StyleNet model

class StyleNet(torch.nn.Module):
        
    def __init__(self, params, dtype, multidevice, backward_device):
        super(StyleNet, self).__init__()
        self.params = params
        self.content_masks_orig = None
        self.style_masks_orig = None
        self.tv_weight, self.content_weight, self.style_weight, self.hist_weight, self.style_stat = 1e-3, 5e0, 1e2, 0, 'gram'
        self.dtype, self.multidevice, self.backward_device = dtype, multidevice, backward_device
        self.content_losses, self.style_losses, self.hist_losses, self.tv_losses = [], [], [], []

        content_layers = params.content_layers.split(',')
        style_layers = params.style_layers.split(',')
        hist_layers = params.hist_layers.split(',')

        next_content_idx, next_style_idx, next_hist_idx, c, r = 1, 1, 1, 0, 0
        cnn, layerList = loadCaffemodel(params.model_file, params.pooling, params.gpu, params.disable_check)
        
        net = nn.Sequential()
        
        if self.tv_weight > 0:
            tv_mod = TVLoss(self.tv_weight).type(self.dtype)
            net.add_module(str(len(net)), tv_mod)
            self.tv_losses.append(tv_mod)

        for i, layer in enumerate(list(cnn), 1):
            if next_content_idx <= len(content_layers) or next_style_idx <= len(style_layers) or next_hist_idx <= len(hist_layers):

                if isinstance(layer, nn.Conv2d):
                    net.add_module(str(len(net)), layer)

                    if layerList['C'][c] in content_layers:
                        print("Setting up content layer " + str(i) + ": " + str(layerList['C'][c]))
                        loss_module = ContentLoss(self.content_weight)
                        net.add_module(str(len(net)), loss_module)
                        self.content_losses.append(loss_module)

                    if layerList['C'][c] in style_layers:
                        print("Setting up style layer " + str(i) + ": " + str(layerList['C'][c]))
                        loss_module = MaskedStyleLoss(self.style_weight)
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
                        loss_module = MaskedStyleLoss(self.style_weight)
                        net.add_module(str(len(net)), loss_module)
                        self.style_losses.append(loss_module)
                        next_style_idx += 1

                    if layerList['R'][r] in hist_layers:
                        print("Setting up histogram layer " + str(i) + ": " + str(layerList['R'][r]))
                        loss_module = MaskedHistLoss(self.hist_weight)
                        net.add_module(str(len(net)), loss_module)
                        self.hist_losses.append(loss_module)
                        next_hist_idx += 1
                    
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
            self.__setup_multi_device(params.gpu, params.multidevice_strategy)


    def __setup_multi_device(self, gpu, multidevice_strategy):
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
                layer.strength = self.content_weight


    def set_style_layer(self, idx_layer, style_stat, style_weight):
        style_layers = [layer for layer in self.net if isinstance(layer, MaskedStyleLoss)]
        style_layers[idx_layer].strength = style_weight
        style_layers[idx_layer].set_statistic(style_stat)


    def set_hist_layer(self, idx_layer, hist_weight):
        hist_layers = [layer for layer in self.net if isinstance(layer, MaskedHistLoss)]
        hist_layers[idx_layer].strength = hist_weight


    def set_style_weight(self, style_weight):
        if self.style_weight == style_weight:
            return
        self.style_weight = style_weight
        for layer in self.net:
            if isinstance(layer, MaskedStyleLoss):
                layer.strength = self.style_weight
        

    def set_hist_weight(self, hist_weight):
        if self.hist_weight == hist_weight:
            return
        self.hist_weight = hist_weight
        for layer in self.net:
            if isinstance(layer, MaskedHistLoss):
                layer.strength = self.hist_weight


    def set_tv_weight(self, tv_weight):
        if self.tv_weight == tv_weight:
            return
        self.tv_weight = tv_weight
        for layer in self.net:
            if isinstance(layer, TVLoss):
                layer.strength = self.tv_weight
 

    def set_style_statistic(self, style_stat):
        if self.style_stat == style_stat:
            return
        self.style_stat = style_stat
        for layer in self.net:
            if isinstance(layer, MaskedStyleLoss):
                layer.set_statistic(self.style_stat)


    def get_content_weight(self):
        return self.content_weight
    
    def get_style_weight(self):
        return self.style_weight
    
    def get_hist_weight(self):
        return self.hist_weight
    
    def get_tv_weight(self):
        return self.tv_weight
    
    def get_style_statistic(self):
        return self.style_stat


    def capture(self, content_image, style_images, style_blend_weights=None, content_masks=None, style_masks=None):
        style_images = [style_images] if type(style_images) != list else style_images
        self.content_masks = copy.deepcopy(content_masks)
        self.style_masks = copy.deepcopy(style_masks)
        self.num_styles = len(style_images)
        self.__setup_style_masks__(style_images)
        self.__setup_layer_masks__()
        self.__capture_content__(content_image.type(self.dtype))
        self.__capture_style__(style_images, style_blend_weights)         


    def __setup_style_masks__(self, style_images):            
        if self.style_masks == None:
            self.style_masks  = [torch.ones(style_images[i].shape).type(self.dtype) 
                                 for i in range(self.num_styles)]
        style_masks = []
        for i in range(self.num_styles):
            tmp_table = []
            for j in range(self.num_styles):
                style_seg_image = self.style_masks[i][0][0]
                if i == j:
                    style_mask_i_j = style_seg_image.type(self.dtype)
                else:                
                    style_mask_i_j = torch.zeros(style_seg_image.shape).type(self.dtype)
                tmp_table.append(style_mask_i_j)
            style_masks.append(tmp_table)
        self.style_masks = style_masks


    def __setup_layer_masks__(self):
        if self.content_masks is not None:
            for c, content_mask in enumerate(self.content_masks):
                self.content_masks[c] = torch.mean(content_mask.type(self.dtype), axis=1)[0]

        for layer in self.net:
            if isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AvgPool2d):
                if self.content_masks != None:
                    for k in range(self.num_styles):
                        h, w = self.content_masks[k].shape
                        h, w = int(h/2), int(w/2)
                        self.content_masks[k] = torch.nn.functional.interpolate(
                            self.content_masks[k].repeat(1,1,1,1), mode='bilinear', size=(h, w))[0][0]
                if self.style_masks != None:
                    for j in range(self.num_styles):
                        for k in range(self.num_styles):
                            h, w = self.style_masks[j][k].shape
                            h, w = int(h/2), int(w/2)
                            self.style_masks[j][k] = torch.nn.functional.interpolate(
                                self.style_masks[j][k].repeat(1,1,1,1), mode='bilinear', size=(h, w))[0][0]
                        self.style_masks[j] = copy.deepcopy(self.style_masks[j])

            elif isinstance(layer, nn.Conv2d):
                sap = nn.AvgPool2d(kernel_size=(3,3), stride=(1, 1), padding=(1,1))
                if self.content_masks != None:
                    for k in range(self.num_styles):
                        self.content_masks[k] = sap(self.content_masks[k].repeat(1,1,1))[0].clone()
                if self.style_masks != None:
                    for j in range(self.num_styles):
                        for k in range(self.num_styles):
                            self.style_masks[j][k] = sap(self.style_masks[j][k].repeat(1,1,1))[0].clone()
                        self.style_masks[j] = copy.deepcopy(self.style_masks[j])

            if isinstance(layer, MaskedStyleLoss) or isinstance(layer, MaskedHistLoss):
                layer.set_masks(self.content_masks, self.style_masks)


    def __capture_content__(self, content_image):
        for l in self.content_losses:
            l.mode = 'capture'
        for l in self.style_losses:
            l.mode = 'none'
        for i in self.hist_losses:
            i.mode = 'none'
        print("Capturing content targets")
        self.forward(content_image.type(self.dtype), loss_mode=False)


    def __capture_style__(self, style_images, style_blend_weights):
        if style_blend_weights == None:
            style_blend_weights = [1.0 / len(style_images) for image in style_images]
        for l in self.content_losses:
            l.mode = 'none'
        for l in self.style_losses:
            l.mode = 'capture'
        for l in self.hist_losses:
            l.mode = 'capture'
        for i, style_image in enumerate(style_images):
            print("Capturing style target " + str(i+1))
            for l in self.style_losses:
                l.blend_weight = style_blend_weights[i]
            self.forward(style_image.type(self.dtype), loss_mode=False)


    def forward(self, x, loss_mode=True):
        if loss_mode:
            for l in self.content_losses:
                l.mode = 'loss'
            for l in self.style_losses:
                l.mode = 'loss'
            for l in self.hist_losses:
                l.mode = 'loss'
        self.net(x)
        

    def get_loss(self):
        loss = 0
        if self.content_weight > 0:
            for mod in self.content_losses:
                loss += mod.loss.to(self.backward_device)
        for mod in self.style_losses:
            if mod.strength > 0:
                loss += mod.loss.to(self.backward_device)
        for mod in self.hist_losses:
            if mod.strength > 0:
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
# Modules for style loss

class GramMatrix(nn.Module):

    def forward(self, input):
        _, C, H, W = input.size()
        x_flat = input.view(C, H * W)
        return torch.mm(x_flat, x_flat.t())


class CovarianceMatrix(nn.Module):

    def forward(self, input):
        _, C, H, W = input.size()
        x_flat = input.view(C, H * W)
        x_flat = x_flat - x_flat.mean(1).unsqueeze(1)
        return torch.mm(x_flat, x_flat.t())


######################################################
# Histogram matching function (unused at the moment)

# Define a module to match histograms
class MatchHistogram(nn.Module):
    def __init__(self, eps=1e-5, mode='pca'):
        super(MatchHistogram, self).__init__()
        self.eps = eps or 1e-5
        self.mode = mode or 'pca'
        self.dim_val = 3
                
    def get_histogram(self, tensor):
        m = tensor.mean(0).mean(0)
        h = (tensor - m).permute(2,0,1).reshape(tensor.size(2),-1)     
        if h.is_cuda:
            ch = torch.mm(h, h.T) / h.shape[1] + self.eps * torch.eye(h.shape[0], device=h.get_device())
        else:
            ch = torch.mm(h, h.T) / h.shape[1] + self.eps * torch.eye(h.shape[0])        
        return m, h, ch
        
    def convert_tensor(self, tensor):
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0).permute(2, 1, 0)
            self.dim_val = 4
        elif tensor.dim() == 3 and self.dim_val != 4:       
            tensor = tensor.permute(2, 1, 0)        
        elif tensor.dim() == 3 and self.dim_val == 4:
            tensor = tensor.permute(2, 1, 0).unsqueeze(0)
        return tensor

    def nan2zero(self, tensor):
        tensor[tensor != tensor] = 0
        return tensor

    def chol(self, t, c, s):
        chol_t, chol_s = torch.cholesky(c), torch.cholesky(s)
        return torch.mm(torch.mm(chol_s, torch.inverse(chol_t)), t)

    def sym(self, t, c, s):         
        p = self.pca(t, c)    
        psp = torch.mm(torch.mm(p, s), p)   
        eval_psp, evec_psp = torch.symeig(psp, eigenvectors=True, upper=True)
        e = self.nan2zero(torch.sqrt(torch.diagflat(eval_psp)))        
        evec_mm = torch.mm(torch.mm(evec_psp, e), evec_psp.T)
        return torch.mm(torch.mm(torch.mm(torch.inverse(p), evec_mm), torch.inverse(p)), t)
    
    def pca(self, t, c): 
        eval_t, evec_t = torch.symeig(c, eigenvectors=True, upper=True)
        e = self.nan2zero(torch.sqrt(torch.diagflat(eval_t)))
        return torch.mm(torch.mm(evec_t, e), evec_t.T)

    def match(self, target_tensor, source_tensor):               
        source_tensor = self.convert_tensor(source_tensor)   
        target_tensor = self.convert_tensor(target_tensor)      

        _, t, ct = self.get_histogram(target_tensor) 
        ms, s, cs = self.get_histogram(source_tensor) 
    
        if self.mode == 'pca':
            mt = torch.mm(torch.mm(self.pca(s, cs), torch.inverse(self.pca(t, ct))), t)
        elif self.mode == 'sym':
            mt = self.sym(t, ct, cs)
        elif self.mode == 'chol':
            mt = self.chol(t, ct, cs)
        
        matched_tensor = mt.reshape(*target_tensor.permute(2,0,1).shape).permute(1,2,0)
        matched_tensor += ms  
        return self.convert_tensor(matched_tensor)
        
    def forward(self, input, source_tensor):     
        return self.match(input, source_tensor)



######################################################
# Style Loss module (Gram/Covariance loss with masks)

class MaskedStyleLoss(nn.Module):

    def __init__(self, strength):
        super(MaskedStyleLoss, self).__init__()
        self.strength = strength
        self.crit = nn.MSELoss()
        self.mode = 'none'
        self.blend_weight = None
        self.set_statistic('gram')
        self.set_masks(None, None)

    def set_statistic(self, style_stat):
        if style_stat == 'gram':
            self.gram = GramMatrix()
        elif style_stat == 'covariance':
            self.gram = CovarianceMatrix()

    def set_masks(self, content_masks, style_masks):
        self.content_masks = copy.deepcopy(content_masks)
        self.style_masks = copy.deepcopy(style_masks)
        self.target_grams = []
        self.masked_grams = []
        self.masked_features = []
        self.capture_count = 0

    def forward(self, input):
        if self.mode == 'capture':
            if self.style_masks != None:
                masks = self.style_masks[self.capture_count]
            else:
                masks = None
            self.capture_count += 1
        elif self.mode == 'loss':
            masks = self.content_masks
            self.style_masks = None
        if self.mode != 'none':
            if self.strength == 0:
                self.loss = 0
                return input
            loss = 0
            for j in range(self.capture_count):
                if masks != None:
                    l_mask_ori = masks[j].clone()
                    l_mask = l_mask_ori.repeat(1,1,1).expand(input.size())
                    l_mean = l_mask_ori.mean()
                    masked_feature = l_mask.mul(input)
                    masked_gram = self.gram(masked_feature).clone()
                    if l_mean > 0:
                        masked_gram = masked_gram.div(input.nelement() * l_mean)
                else:
                    l_mean = 1.0
                    masked_feature = input
                    masked_gram = self.gram(input).clone()
                    masked_gram = masked_gram.div(input.nelement())
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



######################################################
# Style Loss module (histogram loss with masks)

class MaskedHistLoss_old(nn.Module):

    def __init__(self, strength):
        super(MaskedHistLoss, self).__init__()
        self.strength = strength
        self.crit = nn.MSELoss()
        self.mode = 'none'
        self.blend_weight = 1.0
        self.set_masks(None, None)

    def set_masks(self, content_masks, style_masks):
        self.content_masks = copy.deepcopy(content_masks)
        self.style_masks = copy.deepcopy(style_masks)
        self.target_hists = []
        self.target_maxs = []
        self.target_mins = []
        self.capture_count = 0

    def minmax(self, input):
        return torch.min(input[0].view(input.shape[1], -1), 1)[0].data.clone(), \
        torch.max(input[0].view(input.shape[1], -1), 1)[0].data.clone()
		
    def calcHist(self, input, target, min_val, max_val):
        res = input.data.clone() 
        cpp.matchHistogram(res, target.clone())
        for c in range(res.size(0)):
            res[c].mul_(max_val[c] - min_val[c]) 
            res[c].add_(min_val[c])                      
        return res.data.unsqueeze(0)
		
    def forward(self, input):
        if self.mode == 'capture':
            if self.style_masks != None:
                masks = self.style_masks[self.capture_count]
            else:
                masks = None
            self.capture_count += 1
        elif self.mode == 'loss':
            masks = self.content_masks
            self.style_masks = None
        if self.mode != 'none':
            if self.strength == 0:
                self.loss = 0
                return input  
            loss = 0
            for j in range(self.capture_count):
                if masks != None:
                    l_mask_ori = masks[j].clone()
                    l_mask = l_mask_ori.repeat(1,1,1).expand(input.size())
                    masked_feature = l_mask.mul(input)
                else:
                    masked_feature = input
                target_min, target_max = self.minmax(masked_feature)
                target_hist = cpp.computeHistogram(masked_feature[0], 256)
                if self.mode == 'capture':		
                    if j >= len(self.target_hists):
                        self.target_mins.append(target_min)
                        self.target_maxs.append(target_max)
                        self.target_hists.append(target_hist.mul(self.blend_weight))
                    else:
                        self.target_hists[j] += target_hist.mul(self.blend_weight)
                        self.target_mins[j] = torch.min(self.target_mins[j], target_min)
                        self.target_maxs[j] = torch.max(self.target_maxs[j], target_max)
                elif self.mode == 'loss':
                    target = self.calcHist(masked_feature[0], self.target_hists[j], self.target_mins[j], self.target_maxs[j])
                    loss += 0.01 * self.strength * self.crit(masked_feature, target)
            self.loss = loss
        return input



class MaskedHistLoss(nn.Module):

    def __init__(self, strength):
        super(MaskedHistLoss, self).__init__()
        self.strength = strength
        self.crit = nn.MSELoss()
        self.mode = 'none'
        self.blend_weight = 1.0
        self.set_masks(None, None)

    def double_mean(self, tensor):
        tensor = tensor.squeeze(0).permute(2, 1, 0)
        return tensor.mean(0).mean(0)

    def set_masks(self, content_masks, style_masks):
        self.content_masks = copy.deepcopy(content_masks)
        self.style_masks = copy.deepcopy(style_masks)
        self.targets = []
        self.capture_count = 0
		
    def forward(self, input):
        if self.mode == 'capture':
            if self.style_masks != None:
                masks = self.style_masks[self.capture_count]
            else:
                masks = None
            self.capture_count += 1
        elif self.mode == 'loss':
            masks = self.content_masks
            self.style_masks = None
        if self.mode != 'none':
            if self.strength == 0:
                self.loss = 0
                return input  
            loss = 0
            for j in range(self.capture_count):
                if masks != None:
                    l_mask_ori = masks[j].clone()
                    l_mask = l_mask_ori.repeat(1,1,1).expand(input.size())
                    masked_feature = l_mask.mul(input)
                else:
                    masked_feature = input
                if self.mode == 'capture':		
                    target = self.double_mean(masked_feature.detach())      
                    if j >= len(self.targets):
                        self.targets.append(target.mul(self.blend_weight))
                    else:
                        self.targets[j] += target.mul(self.blend_weight)
                elif self.mode == 'loss':
                    loss += self.strength * self.crit(self.double_mean(masked_feature.clone()), self.targets[j])
            self.loss = loss
        return input



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

