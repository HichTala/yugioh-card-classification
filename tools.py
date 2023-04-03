import cv2


def calculate_SIFT_score(similarScore, numOfPts):
    return similarScore - (numOfPts ** 3) / 10000


def calculate_HOG_points(orb, img0, img1):
    k0, desc0 = orb.detectAndCompute(img0, None)

    k1, desc1 = orb.detectAndCompute(img1, None)

    if desc1 is None:
        return 0

    # match descriptors and sort them in the order of their distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc0, desc1)

    numOfMatches = len(matches)

    return numOfMatches


def art_cropper(img):
    width, height = img.size
    return img.crop((int(0.2 * width), int(0.2 * height), int(0.8 * width), int(0.7 * height)))