import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import IPython
from decimal import Decimal


def resize_tensor(image, image_size, mode='bicubic', align_corners=True):
    assert isinstance(image_size, tuple), 'Error: image_size must be a tuple'
    _, _, h1, w1 = image.shape
    if (w1, h1) == image_size:
        return image
    return torch.nn.functional.interpolate(
        image, 
        tuple(reversed(image_size)), 
        mode=mode, 
        align_corners=align_corners
    )


def random_tensor(w, h, c=3):
    tensor = torch.randn(c, h, w).mul(0.001).unsqueeze(0)
    return tensor


def random_tensor_like(base_image):
    w, h = base_image.size
    return random_tensor(w, h, 3)


def preprocess(image, image_size=None, to_normalize=True):
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    if image_size is None:
        image_size = (image.height, image.width)
    elif type(image_size) is not tuple:
        image_size = tuple([int((float(image_size) / max(image.size)) * x) for x in (image.height, image.width)])
    Loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    rgb2bgr = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
    if to_normalize:
        Normalize = transforms.Compose([transforms.Normalize(mean=[103.939, 116.779, 123.68], std=[1,1,1])])
        tensor = Normalize(rgb2bgr(Loader(image) * 256)).unsqueeze(0)
    else:
        tensor = rgb2bgr(Loader(image)).unsqueeze(0)
    return tensor


def deprocess(tensor):
    Normalize = transforms.Compose([transforms.Normalize(mean=[-103.939, -116.779, -123.68], std=[1,1,1])])
    bgr2rgb = transforms.Compose([transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])])])
    tensor = bgr2rgb(Normalize(tensor.squeeze(0).cpu())) / 256
    tensor.clamp_(0, 1)
    Image2PIL = transforms.ToPILImage()
    image = Image2PIL(tensor.cpu())
    return image


def original_colors(content, generated):
    #content, generated = deprocess(content.clone()), deprocess(generated.clone())    
    content_channels = list(content.convert('YCbCr').split())
    generated_channels = list(generated.convert('YCbCr').split())
    content_channels[0] = generated_channels[0]
    return Image.merge('YCbCr', content_channels).convert('RGB')


def get_style_image_paths(style_image_input):
    style_image_list = []
    for path in style_image_input:
        if os.path.isdir(path):
            images = (os.path.join(path, file) for file in os.listdir(path) 
                      if os.path.splitext(file)[1].lower() in [".jpg", ".jpeg", ".png", ".tiff"])
            style_image_list.extend(images)
        else:
            style_image_list.append(path)
    return style_image_list


def maybe_update(net, t, update_iter, num_iterations, loss):
    if update_iter != None and t % update_iter == 0:
        IPython.display.clear_output()
        print('Iteration %d/%d: '%(t, num_iterations))
        if net.content_weight > 0:
            print('  Content loss = %s' % ', '.join(['%.1e' % Decimal(module.loss.item()) for module in net.content_losses]))
        print('  Style loss = %s' % ', '.join(['%.1e' % Decimal(module.loss.item()) for module in net.style_losses if module.strength > 0]))
        print('  Histogram loss = %s' % ', '.join(['%.1e' % Decimal(module.loss.item()) for module in net.hist_losses if module.strength > 0]))
        if net.tv_weight > 0:
            print('  TV loss = %s' % ', '.join(['%.1e' % Decimal(module.loss.item()) for module in net.tv_losses]))
        print('  Total loss = %.2e' % Decimal(loss.item()))

        
def maybe_save_preview(img, t, save_iter, num_iterations, output_path):
    should_save = save_iter > 0 and t % save_iter == 0
    if not should_save:
        return
    output_filename, file_extension = os.path.splitext(output_path)
    #output_filename = output_filename.replace('results', 'results/preview')
    filename = '%s_%04d%s' % (output_filename, t, file_extension)
    save(deprocess(img), filename)
