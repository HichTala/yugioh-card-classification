def art_cropper(img):
    width, height = img.size
    return img.crop((int(0.115 * width), int(0.18 * height), int(0.89 * width), int(0.71 * height)))
