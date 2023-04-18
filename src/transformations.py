from random import normalvariate

from torchvision import transforms as T


def train_data_transforms(img):
    # img = T.Resize(60, antialias=True)(img)
    img = T.Resize((50, 50), antialias=True)(img)

    gamma = normalvariate(mu=0.75, sigma=0.2)
    gamma = max(gamma, 0.15)
    gain = normalvariate(mu=1.15, sigma=0.15)
    gain = max(gain, 0.7)
    gain = min(gain, 1.6)
    sat_factor = normalvariate(mu=0.8, sigma=0.15)
    sat_factor = max(sat_factor, 0.35)
    hue_factor = normalvariate(mu=0, sigma=0.025)

    img = T.functional.adjust_gamma(img, gamma=gamma, gain=gain)
    img = T.functional.adjust_saturation(img, saturation_factor=sat_factor)
    # img = T.GaussianBlur(kernel_size=5, sigma=1)(img)
    img = T.functional.adjust_hue(img, hue_factor=hue_factor)
    img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

    return img


def final_data_transforms(img):
    img = T.Resize((50, 50), antialias=True)(img)

    img = T.functional.adjust_gamma(img, gamma=0.75, gain=1.15)
    img = T.functional.adjust_saturation(img, saturation_factor=0.8)
    # img = T.GaussianBlur(kernel_size=(5, 9), sigma=0.5)(img)
    img = T.functional.adjust_hue(img, hue_factor=0)

    img = T.ToTensor()(img)
    img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

    return img


def image_transforms(img):
    img = T.Resize((50, 50), antialias=True)(img)

    img = T.ToTensor()(img)
    img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

    return img


def image_transforms_no_tensor(img):
    img = T.Resize((98, 65), antialias=True)(img)

    return img
