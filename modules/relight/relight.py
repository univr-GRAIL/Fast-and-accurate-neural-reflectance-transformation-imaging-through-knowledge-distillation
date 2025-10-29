import os
import cv2 as cv
import numpy as np
import torch
def relight(model_path, gt_path, mask=False):
    input_feature_path = model_path + '/coefficient.npy'
    est_path = 'relighted'
    if not os.path.exists(est_path):
        os.makedirs(est_path)

    [h, w] = cv.imread(model_path + '/plane_0.jpg', 0).shape
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    samples = torch.from_numpy(np.load(input_feature_path)).to(device)
    hw = samples.shape[0]
    if mask:
        mask_img = cv.imread(gt_path + '/mask.png', cv.IMREAD_GRAYSCALE)
        _, binary_mask = cv.threshold(mask_img, 127, 255, cv.THRESH_BINARY)
        mask_img = binary_mask.flatten()
        masked_indices = np.squeeze(np.column_stack(np.where(mask_img == 255)))

    decoder = torch.load(model_path + '/decoder.pth', weights_only=False).to(device)
    ld_file = gt_path + '/dirs.lp'
    light_dimension = 2

    def get_lights(ld_file, light_dimension=2):
        with open(ld_file) as f:
            data = f.read()
        data = data.split('\n')
        data = data[
               1:int(data[0]) + 1]  # keep the lines with light directions, remove the first one which is number of samples

        num_lights = len(data)
        ld = np.zeros((num_lights, light_dimension), np.float32)
        for i, dirs in enumerate(data):
            if (len(dirs.split(' ')) == 4):
                sep = ' '
            else:
                sep = '\t'
            s = dirs.split(sep)
            if len(s) == 4:
                ld[i] = [float(s[l]) for l in range(1, light_dimension + 1)]
            else:
                ld[i] = [float(s[l]) for l in range(light_dimension)]
        return ld

    light_dirs = get_lights(ld_file=ld_file, light_dimension=light_dimension)
    for i, l_dir in enumerate(light_dirs):
        lights_list = np.tile(l_dir, reps=hw)
        lights_list = np.reshape(lights_list, (hw, light_dimension))
        light_dir = torch.from_numpy(lights_list).to(device)

        with torch.no_grad():
            zy = torch.cat((samples, light_dir), dim=1).to(device)
            reconst_imgs = decoder(zy.to(device))
            relighted_img = reconst_imgs.cpu().numpy()

            outputs = relighted_img.clip(min=0, max=1)
            if mask:
                relighted_img[masked_indices] = outputs
            else:
                relighted_img = outputs

            outputs = np.reshape(relighted_img, (h, w, 3))
            outputs *= 255
            outputs = outputs.astype('uint8')
            cv.imwrite(est_path + '/relighted' + str(i).zfill(2) + '.jpg', outputs)