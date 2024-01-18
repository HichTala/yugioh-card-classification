import io
import imageio
from random import normalvariate

import numpy as np
from PIL import Image, ImageEnhance
from torch import add
from torchvision import transforms


def jpeg_bur(im, q):
    buf = io.BytesIO()
    imageio.imwrite(buf, im, format='jpg', quality=q)
    s = buf.getbuffer()
    return imageio.imread(s, format='jpg')


def train_data_transforms(img):
    width, height = img.size
    gaussian_noise = np.random.normal(0, 1, (height, width, 3))
    noisy_image_np = np.clip(np.array(img) + gaussian_noise, 0, 255).astype(np.uint8)

    img_compressed = jpeg_bur(noisy_image_np, 10)

    img = Image.fromarray(img_compressed)

    img = transforms.RandomResizedCrop(50, scale=(0.65, 1.0), antialias=True)(img)
    degrees = normalvariate(mu=0, sigma=1)
    img = transforms.RandomRotation(degrees=abs(degrees))(img)
    img = transforms.Resize((224, 224), antialias=True)(img)

    # gamma = normalvariate(mu=1.25, sigma=0.4)
    # gamma = max(gamma, 0.15)
    # gamma = min(gamma, 1.55)
    # gain = normalvariate(mu=0.75, sigma=0.15)
    # gain = max(gain, 0.7)
    # gain = min(gain, 1.6)
    sat_factor = normalvariate(mu=0.5, sigma=0.15)
    sat_factor = max(sat_factor, 0.35)
    # hue_factor = normalvariate(mu=0, sigma=0.025)
    contrast_factor = normalvariate(mu=0.3, sigma=0.15)
    contrast_factor = max(contrast_factor, 0.35)
    blue_factor = normalvariate(mu=2.1, sigma=0.05)
    green_factor = normalvariate(mu=1.9, sigma=0.05)
    red_factor = normalvariate(mu=1.8, sigma=0.05)

    # img = transforms.functional.adjust_gamma(img, gain=gain)
    img = transforms.functional.adjust_saturation(img, saturation_factor=sat_factor)
    img = transforms.GaussianBlur(kernel_size=9, sigma=1)(img)
    # img = transforms.functional.adjust_hue(img, hue_factor=hue_factor)
    # # img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    img = transforms.functional.adjust_contrast(img, contrast_factor=contrast_factor)

    img = np.array(img)
    img[:, :, 0] = np.clip(img[:, :, 0] * red_factor, 0, 255).astype(np.uint8)
    img[:, :, 1] = np.clip(img[:, :, 1] * green_factor, 0, 255).astype(np.uint8)
    img[:, :, 2] = np.clip(img[:, :, 2] * blue_factor, 0, 255).astype(np.uint8)

    img = Image.fromarray(img)

    return img


def final_data_transforms(img):
    img = transforms.Resize((200, 200), antialias=True)(img)

    img = transforms.functional.adjust_gamma(img, gamma=0.75, gain=1.15)
    img = transforms.functional.adjust_saturation(img, saturation_factor=0.8)
    # img = T.GaussianBlur(kernel_size=(5, 9), sigma=0.5)(img)
    img = transforms.functional.adjust_hue(img, hue_factor=0)

    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

    return img


def image_transforms(img):
    img = transforms.Resize((200, 200), antialias=True)(img)

    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

    return img


def image_transforms_no_tensor(img):
    img = transforms.Resize((224, 224), antialias=True)(img)

    return img
