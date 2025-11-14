import torch
import lpips
import torchvision.transforms as T
from PIL import Image
import numpy as np
# 1. Load LPIPS model (Alex/VGG)
loss_fn = lpips.LPIPS(net='alex').cuda()  # or 'vgg'

# 2. Load and preprocess images
def preprocess(img_path):
    img = Image.open(img_path).convert('RGB')
    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)  # LPIPS expects [-1, 1] range
    ])
    return transform(img).unsqueeze(0)  # shape: [1, 3, H, W]



# 3. Load binary mask (1 = use pixel, 0 = ignore)
def load_mask(mask_path):
    mask = Image.open(mask_path).convert('L')  # grayscale
    mask = mask.resize((256, 256))  # resize to match image
    mask_tensor = T.ToTensor()(mask)  # [1, H, W] with values in [0, 1]
    mask_tensor = (mask_tensor > 0.5).float()  # binary mask
    return mask_tensor.unsqueeze(0).cuda()  # shape: [1, 1, H, W]


def calculate_LPIPS(gt_path, est_path, mask_img=None):
    result_array = np.zeros(len(est_path))
    for i in range(len(gt_path)):
        img1 = gt_path[i]
        img2 = est_path[i]

        img0 = preprocess(img1).cuda()
        img1 = preprocess(img2).cuda()
        with torch.no_grad():
            dist_map = loss_fn(img0, img1)  # shape: [1, 1, H, W]

        # 5. Compute masked LPIPS value
        lpips_value = dist_map
        if mask_img is not None:
            mask = load_mask(mask_img)
            masked_dist = dist_map * mask
            lpips_value = masked_dist.sum() / mask.sum()
        # print(f"Masked LPIPS: {lpips_value.item():.4f}")
        result_array[i] = lpips_value.item()
    avg_lpips = np.round(np.average(result_array), 4)
    return avg_lpips