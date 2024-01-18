def art_cropper(img):
    width, height = img.size
    return img.crop((6, 6, width - 6, height - 65))
