from random import normalvariate

from torchvision import transforms as T


def train_data_transforms(img):
    img = T.Resize(50, antialias=True)(img)
    img = T.Resize((75, 50), antialias=True)(img)

    gamma = normalvariate(mu=1, sigma=0.3)
    gamma = gamma if gamma >= 0 else 0
    gain = normalvariate(mu=1, sigma=1)
    sat_factor = normalvariate(mu=0.7, sigma=0.1)
    hue_factor = normalvariate(mu=0, sigma=0.03)

    img = T.functional.adjust_gamma(img, gamma=gamma, gain=gain)
    img = T.functional.adjust_saturation(img, saturation_factor=sat_factor)
    img = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(img)
    img = T.functional.adjust_hue(img, hue_factor=hue_factor)
    img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

    return img


def final_data_transforms(img):
    img = T.Resize(50, antialias=True)(img)
    img = T.Resize((75, 50), antialias=True)(img)

    img = T.functional.adjust_gamma(img, gamma=1, gain=1)
    img = T.functional.adjust_saturation(img, saturation_factor=0.7)
    img = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(img)
    img = T.functional.adjust_hue(img, hue_factor=0)

    img = T.ToTensor()(img)
    img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

    return img


def image_transforms(img):
    img = T.Resize((300, 200), antialias=True)(img)

    img = T.ToTensor()(img)
    img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

    return img
