from PIL import Image
from torchvision import transforms

data_transforms = transforms.Compose([
    transforms.Resize(50),
    transforms.Resize((300, 200),
                      )])

img0 = Image.open('251194600.jpg')
img1 = Image.open('xyz.jpg')

img0 = data_transforms(img0)
img1 = data_transforms(img1)

img0 = transforms.functional.adjust_gamma(img0, 0.7, 1.3)
img0 = transforms.functional.adjust_saturation(img0, 0.7)
img0 = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))(img0)
img0 = transforms.functional.adjust_hue(img0, -0.044444)

img0.show()
# img1.show()
