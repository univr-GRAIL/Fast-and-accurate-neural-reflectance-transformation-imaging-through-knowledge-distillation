import cv2
import numpy as np
from skimage import color
from colormath.color_diff import delta_e_cie2000


def compute_deltaE2000_image(img1, img2, mask=None):
    # Resize and ensure both are RGB float in [0, 1]
    img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) / 255.0
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) / 255.0

    # Convert to Lab
    lab1 = color.rgb2lab(img1_rgb)
    lab2 = color.rgb2lab(img2_rgb)

    # Compute Delta E (2000) pixel-wise
    deltaE = color.deltaE_ciede2000(lab1, lab2)
    if mask is not None:
        deltaE = deltaE[mask.astype(bool)]

    mean_deltaE = np.mean(deltaE)
    return mean_deltaE


def calculate_avg_deltaE(gt_path, est_path, mask_img=None):
    result_array = np.zeros(len(est_path))

    for i in range(len(gt_path)):
        img1 = cv2.imread(gt_path[i])
        img2 = cv2.imread(est_path[i])
        deltaE_map = compute_deltaE2000_image(img1, img2, mask_img)
        result_array[i] = deltaE_map
    avg_deltaE = np.round(np.average(result_array), 4)
    return avg_deltaE


