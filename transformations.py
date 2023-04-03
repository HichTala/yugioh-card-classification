from torchvision import transforms


def data_transforms(img):
    img = transforms.Resize(50)(img)
    img = transforms.Resize((300, 200))(img)

    img = transforms.functional.adjust_gamma(img, 0.7, 1.3)
    img = transforms.functional.adjust_saturation(img, 0.7)
    img = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(img)
    img = transforms.functional.adjust_hue(img, -0.044444)

    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

    return img


def image_transforms(img):
    img = transforms.Resize((300, 200))(img)

    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)

    return img


def data_transforms_no_tensor(img):
    img = transforms.Resize(50)(img)
    img = transforms.Resize((300, 200))(img)

    img = transforms.functional.adjust_gamma(img, 0.7, 1.3)
    img = transforms.functional.adjust_saturation(img, 0.7)
    img = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(img)
    img = transforms.functional.adjust_hue(img, -0.044444)

    return img


def image_transforms_no_tensor(img):
    img = transforms.Resize((300, 200))(img)

    return img
