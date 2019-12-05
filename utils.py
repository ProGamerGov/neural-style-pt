import os
import torch
import torchvision.transforms as transforms
from PIL import Image

Image.MAX_IMAGE_PIXELS = 1000000000 # Support gigapixel images


class StylenetArgs:
    def __init__(self):
        self.gpu = 'c'
        self.optimizer = 'lbfgs'
        self.learning_rate = 1e0
        self.lbfgs_num_correction = 100
        self.pooling = 'max'
        self.model_file = 'models/vgg19-d01eb7cb.pth'
        self.disable_check = False
        self.backend = 'nn'
        self.cudnn_autotune = False
        self.content_layers = 'relu4_2'
        self.style_layers = 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1'
        self.multidevice_strategy = '4,7,29'


# Preprocess an image before passing it to a model.
# We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
# and subtract the mean pixel.
def preprocess(image_name, image_size, to_normalize=True):
    image = Image.open(image_name).convert('RGB')
    if type(image_size) is not tuple:
        image_size = tuple([int((float(image_size) / max(image.size))*x) for x in (image.height, image.width)])
    Loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    rgb2bgr = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
    if to_normalize:
        Normalize = transforms.Compose([transforms.Normalize(mean=[103.939, 116.779, 123.68], std=[1,1,1])])
        tensor = Normalize(rgb2bgr(Loader(image) * 256)).unsqueeze(0)
    else:
        tensor = rgb2bgr(Loader(image)).unsqueeze(0)
    return tensor


#  Undo the above preprocessing.
def deprocess(output_tensor):
    Normalize = transforms.Compose([transforms.Normalize(mean=[-103.939, -116.779, -123.68], std=[1,1,1])])
    bgr2rgb = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
    output_tensor = bgr2rgb(Normalize(output_tensor.squeeze(0).cpu())) / 256
    output_tensor.clamp_(0, 1)
    Image2PIL = transforms.ToPILImage()
    image = Image2PIL(output_tensor.cpu())
    return image


# Combine the Y channel of the generated image and the UV/CbCr channels of the
# content image to perform color-independent style transfer.
def original_colors(content, generated):
    content_channels = list(content.convert('YCbCr').split())
    generated_channels = list(generated.convert('YCbCr').split())
    content_channels[0] = generated_channels[0]
    return Image.merge('YCbCr', content_channels).convert('RGB')


def get_style_image_paths(style_image_input):
    style_image_list = []
    for path in style_image_input:
        if os.path.isdir(path):
            images = (image + "/" + file for file in os.listdir(image) 
                      if os.path.splitext(file)[1].lower() in [".jpg", ".jpeg", ".png", ".tiff"])
            style_image_list.extend(images)
        else:
            style_image_list.append(path)
    return style_image_list


def maybe_print(net, t, print_iter, num_iterations, loss):
    if print_iter > 0 and t % print_iter == 0:
        print("Iteration " + str(t) + " / "+ str(num_iterations))
        for i, loss_module in enumerate(net.content_losses):
            print("  Content " + str(i+1) + " loss: " + str(loss_module.loss.item()))
        for i, loss_module in enumerate(net.style_losses):
            print("  Style " + str(i+1) + " loss: " + str(loss_module.loss.item()))
        print("  Total loss: " + str(loss.item()))


def maybe_save(img, t, save_iter, num_iterations, orig_colors, output_path):
    should_save = save_iter > 0 and t % save_iter == 0
    should_save = should_save or t == num_iterations
    if not should_save:
        return
    output_filename, file_extension = os.path.splitext(output_path)
    if t == num_iterations:
        filename = output_filename + str(file_extension)
    else:
        filename = str(output_filename) + "_" + str(t) + str(file_extension)
    disp = deprocess(img.clone())
    if orig_colors == 1:   # color-independent style transfer
        disp = original_colors(deprocess(content_image.clone()), disp)  ## this doesn't work yet
    disp.save(str(filename))

    