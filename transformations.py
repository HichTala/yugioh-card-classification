from random import uniform, normalvariate

from torchvision import transforms as T


def data_transforms(img):
    img = data_transforms_no_tensor(img)

    img = T.ToTensor()(img)
    img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

    return img


def image_transforms(img):
    img = T.Resize((300, 200))(img)

    img = T.ToTensor()(img)
    img = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

    return img


def data_transforms_no_tensor(img):
    img = T.Resize(50)(img)
    img = T.Resize((300, 200))(img)

    gamma = normalvariate(mu=1, sigma=0.5)
    gain = normalvariate(mu=1, sigma=2)
    sat_factor = normalvariate(mu=0.7, sigma=0.3)
    hue_factor = normalvariate(mu=0, sigma=0.1)

    img = T.functional.adjust_gamma(img, gamma=gamma, gain=gain)
    img = T.functional.adjust_saturation(img, saturation_factor=sat_factor)
    img = T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(img)
    img = T.functional.adjust_hue(img, hue_factor=hue_factor)

    return img


def image_transforms_no_tensor(img):
    img = T.Resize((300, 200))(img)

    return img
