def art_cropper(img):
    width, height = img.size
    return img.crop((6, 30, width - 6, height - 65))
