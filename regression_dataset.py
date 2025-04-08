from tqdm import tqdm

from datasets import load_dataset
import cv2
import numpy as np
import math
import json
import random
import os
from PIL import Image


def get_transform(image):
    rotx = random.randint(0, 60) * math.pi / 180
    roty = random.randint(0, 60) * math.pi / 180
    rotz = random.randint(0, 360) * math.pi / 180

    cx, cy, cz = math.cos(rotx), math.cos(roty), math.cos(rotz)
    sx, sy, sz = math.sin(rotx), math.sin(roty), math.sin(rotz)

    rotation = [[cz * cy, cz * sy * sx - sz * cx],
                [sz * cy, sz * sy * sx + cz * cx],
                [-sy, cy * sx]]

    pt = [[-image.shape[1] / 2, -image.shape[0] / 2],
          [image.shape[1] / 2, -image.shape[0] / 2],
          [image.shape[1] / 2, image.shape[0] / 2],
          [-image.shape[1] / 2, image.shape[0] / 2]]

    ptt = []
    for i in range(4):
        pz = pt[i][0] * rotation[2][0] + pt[i][1] * rotation[2][1]
        ptt.append([image.shape[1] / 2 + (pt[i][0] * rotation[0][0] + pt[i][1] * rotation[0][1]) * 50 * image.shape[
            0] / (50 * image.shape[0] + pz),
                    image.shape[0] / 2 + (pt[i][0] * rotation[1][0] + pt[i][1] * rotation[1][1]) * 50 * image.shape[
                        0] / (50 * image.shape[0] + pz)])

    in_pt = np.array([
        [0, 0],
        [image.shape[1], 0],
        [image.shape[1], image.shape[0]],
        [0, image.shape[0]]
    ], np.float32)
    out_pt = np.array(ptt, np.float32)

    new_w = int(max(out_pt[:, 0]) - min(out_pt[:, 0]))
    new_h = int(max(out_pt[:, 1]) - min(out_pt[:, 1]))

    if min(out_pt[:, 0]) < 0:
        out_pt[:, 0] = out_pt[:, 0] - min(out_pt[:, 0])
    if max(out_pt[:, 0]) > new_w:
        out_pt[:, 0] = out_pt[:, 0] - max(out_pt[:, 0]) + new_w
    if min(out_pt[:, 1]) < 0:
        out_pt[:, 1] = out_pt[:, 1] - min(out_pt[:, 1])
    if max(out_pt[:, 1]) > new_h:
        out_pt[:, 1] = out_pt[:, 1] - max(out_pt[:, 1]) + new_h

    return cv2.getPerspectiveTransform(in_pt, out_pt), out_pt


def alphaMerge(small_foreground, background, top, left):
    result = background.copy()

    fg_b, fg_g, fg_r, fg_a = cv2.split(small_foreground)

    fg_a = fg_a / 255.0

    label_rgb = cv2.merge([fg_b * fg_a, fg_g * fg_a, fg_r * fg_a])

    height, width = small_foreground.shape[0], small_foreground.shape[1]
    part_of_bg = result[top:top + height, left:left + width, :]

    bg_b, bg_g, bg_r = cv2.split(part_of_bg)
    part_of_bg = cv2.merge([bg_b * (1 - fg_a), bg_g * (1 - fg_a), bg_r * (1 - fg_a)])

    cv2.add(label_rgb, part_of_bg, part_of_bg)

    result[top:top + height, left:left + width, :] = part_of_bg
    return result


def main():
    # data = load_dataset("benjamin-paine/imagenet-1k-256x256")
    # shuffled_data = data['train'].shuffle()

    with tqdm(total=7000, desc="Generating Dataset", colour='cyan') as pbar:
        # for i, single_data in enumerate(shuffled_data):
        for single_data in os.listdir("datasets/playmate"):
            for i in range(1000):
                pbar.update(1)

                background = Image.open(os.path.join("datasets/playmate", single_data)).convert('RGB')
                # background = single_data["image"].convert('RGB')
                background = np.array(background)[:, :, ::-1].copy()
                background = cv2.resize(background, (640, 640), interpolation=cv2.INTER_LINEAR)

                nb_image = random.randint(1, 5)
                # coords = None
                # out_pt = None

                f = open(f"datasets/yolo/val/labels/{i}.txt", "a+")
                for _ in range(nb_image):
                    random_card = random.choice(os.listdir("datasets/ddraw"))
                    path = os.path.join("datasets/ddraw", random_card)
                    try:
                        path = os.path.join(path, random.choice(os.listdir(path)))
                    except IndexError as e:
                        breakpoint()

                    image = Image.open(path).convert('RGB')
                    image = np.array(image)[:, :, ::-1].copy()

                    width = random.randint(30, 90)
                    height = int(width * image.shape[0] / image.shape[1])
                    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

                    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
                    transform, out_pt = get_transform(image)
                    new_w = int(max(out_pt[:, 0]) - min(out_pt[:, 0]))
                    new_h = int(max(out_pt[:, 1]) - min(out_pt[:, 1]))
                    image = cv2.warpPerspective(image, transform, (new_w, new_h), flags=cv2.INTER_CUBIC,
                                                borderMode=cv2.BORDER_CONSTANT, borderValue=[255, 255, 255, 0])

                    coords = random.randint(0, 640 - image.shape[0]), random.randint(0, 640 - image.shape[1])
                    background = alphaMerge(image, background, coords[0], coords[1])
                    f.write(
                        f"0 {(coords[1] + out_pt[0, 0]) / 640} {(coords[0] + out_pt[0, 1]) / 640} {(coords[1] + out_pt[1, 0]) / 640} {(coords[0] + out_pt[1, 1]) / 640} {(coords[1] + out_pt[2, 0]) / 640} {(coords[0] + out_pt[2, 1]) / 640} {(coords[1] + out_pt[3, 0]) / 640} {(coords[0] + out_pt[3, 1]) / 640}\n")
                f.close()
                cv2.imwrite(f"./datasets/yolo/val/images/{i}.png", background)


if __name__ == "__main__":
    main()
