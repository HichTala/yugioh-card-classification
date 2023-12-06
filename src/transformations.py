from random import normalvariate

from torchvision import transforms


def train_data_transforms(img):
    img = transforms.RandomResizedCrop(60, antialias=True)(img)
    img = transforms.RandomRotation(degrees=40)(img)
    img = transforms.Resize((224, 224), antialias=True)(img)

    gamma = normalvariate(mu=1.25, sigma=0.4)
    gamma = max(gamma, 0.15)
    gamma = min(gamma, 1.55)
    gain = normalvariate(mu=0.75, sigma=0.15)
    gain = max(gain, 0.7)
    gain = min(gain, 1.6)
    sat_factor = normalvariate(mu=0.7, sigma=0.15)
    sat_factor = max(sat_factor, 0.35)
    hue_factor = normalvariate(mu=0, sigma=0.025)
    contrast_factor = normalvariate(mu=0.5, sigma=0.15)
    contrast_factor = max(contrast_factor, 0.35)

    img = transforms.functional.adjust_gamma(img, gamma=gamma, gain=gain)
    img = transforms.functional.adjust_saturation(img, saturation_factor=sat_factor)
    # img = T.GaussianBlur(kernel_size=5, sigma=1)(img)
    img = transforms.functional.adjust_hue(img, hue_factor=hue_factor)
    # img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
    img = transforms.functional.adjust_contrast(img, contrast_factor=contrast_factor)

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
