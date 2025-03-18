def art_cropper(img):
    width, height = img.size
    return img.crop((32, 71, 236, 275))
