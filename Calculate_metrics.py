import os
import glob
import pandas as pd
import numpy as np
from utils.LPIPS_masked import calculate_LPIPS
from utils.deltaE import calculate_avg_deltaE

def list_subfolders(directory):
    return [f.name for f in os.scandir(directory) if f.is_dir()]


# Example usage
parent_folder = "../datasets/RealRTI"
objindex = []
subfolders = list_subfolders(parent_folder)
col_names = (list_subfolders(parent_folder + '/' + subfolders[0]))
if 'train' in col_names:
    col_names.remove('train')
if 'test' in col_names:
    col_names.remove('test')

result_store_ssim = np.zeros((len(subfolders) + 1, len(col_names)))
result_store_psnr = np.zeros((len(subfolders) + 1, len(col_names)))
result_store_deltaE = np.zeros((len(subfolders) + 1, len(col_names)))
result_store_lpips = np.zeros((len(subfolders) + 1, len(col_names)))


for indx, item in enumerate(subfolders):
    subsubfolders = (list_subfolders(parent_folder + '/' + item))
    if 'train' in subsubfolders:
        subsubfolders.remove('train')

    for mindx, m in enumerate(subsubfolders):
        if m != "test":
            est_path = sorted(glob.glob(parent_folder + '/' + item + '/' + m + '/*.jpg'))
            gt_path = sorted(glob.glob(parent_folder + '/' + item + '/test/*.jpg'))

            lpips_val = calculate_LPIPS(gt_path, est_path)
            deltaE_val = calculate_avg_deltaE(gt_path, est_path)

            result_store_lpips[indx,mindx] = round(lpips_val, 3)
            result_store_deltaE[indx,mindx] = round(deltaE_val, 2)

result_store_lpips[-1, :] = np.round(np.average(result_store_lpips[0:len(subfolders), :], axis=0),3)
result_store_deltaE[-1, :] = np.round(np.average(result_store_deltaE[0:len(subfolders), :], axis=0),2)
merged = np.char.add(result_store_lpips.astype(str), ' / ')
merged = np.char.add(merged, result_store_deltaE.astype(str))
index_names = subfolders
index_names.append('Average')

merged_df = pd.DataFrame(merged, columns=col_names, index=index_names)
merged_df.to_csv(parent_folder + f'/LPIPS&DeltaE_result.csv')
# lpips_df = pd.DataFrame(result_store_lpips, columns=col_names, index=index_names)
# lpips_df.to_csv(parent_folder + f'/LPIPS_result.csv')
#
# deltaE_df = pd.DataFrame(result_store_deltaE, columns=col_names, index=index_names)
# deltaE_df.to_csv(parent_folder + f'/deltaE_result.csv')
