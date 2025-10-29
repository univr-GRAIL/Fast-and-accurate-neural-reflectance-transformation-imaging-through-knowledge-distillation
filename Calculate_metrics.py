import os
import glob
import argparse
import pandas as pd
import numpy as np
from modules.utils.LPIPS_masked import calculate_LPIPS
from modules.utils.deltaE import calculate_avg_deltaE


def list_subfolders(directory):
    """Return a list of all subfolders in the given directory."""
    return [f.name for f in os.scandir(directory) if f.is_dir()]


def main():
    # ----------------------------
    # Argument parser setup
    # ----------------------------
    parser = argparse.ArgumentParser(
        description="Compute LPIPS and DeltaE metrics for all subfolders in a parent dataset directory."
    )

    parser.add_argument(
        "--parent_folder",
        type=str,
        default="../datasets/RealRTI",
        help="Path to the parent dataset folder containing subfolders (default: ../datasets/RealRTI)."
    )

    args = parser.parse_args()


    parent_folder = args.parent_folder

    subfolders = list_subfolders(parent_folder)
    if not subfolders:
        raise ValueError(f"No subfolders found in {parent_folder}")

    col_names = list_subfolders(os.path.join(parent_folder, subfolders[0]))
    if "train" in col_names:
        col_names.remove("train")
    if "test" in col_names:
        col_names.remove("test")


    result_store_deltaE = np.zeros((len(subfolders) + 1, len(col_names)))
    result_store_lpips = np.zeros((len(subfolders) + 1, len(col_names)))

    # Loop over all objects
    for indx, item in enumerate(subfolders):
        subsubfolders = list_subfolders(os.path.join(parent_folder, item))
        if "train" in subsubfolders:
            subsubfolders.remove("train")

        for mindx, m in enumerate(subsubfolders):
            if m != "test":
                est_path = sorted(glob.glob(os.path.join(parent_folder, item, m, "*.jpg")))
                gt_path = sorted(glob.glob(os.path.join(parent_folder, item, "test", "*.jpg")))

                if not est_path or not gt_path:
                    print(f"⚠️ Skipping {item}/{m}: missing images.")
                    continue

                lpips_val = calculate_LPIPS(gt_path, est_path)
                deltaE_val = calculate_avg_deltaE(gt_path, est_path)

                result_store_lpips[indx, mindx] = round(lpips_val, 3)
                result_store_deltaE[indx, mindx] = round(deltaE_val, 2)

    # Compute averages
    result_store_lpips[-1, :] = np.round(np.average(result_store_lpips[0:len(subfolders), :], axis=0), 3)
    result_store_deltaE[-1, :] = np.round(np.average(result_store_deltaE[0:len(subfolders), :], axis=0), 2)

    # Merge results (LPIPS / DeltaE)
    merged = np.char.add(result_store_lpips.astype(str), " / ")
    merged = np.char.add(merged, result_store_deltaE.astype(str))

    index_names = subfolders + ["Average"]
    merged_df = pd.DataFrame(merged, columns=col_names, index=index_names)

    # Save CSV
    output_csv = os.path.join(parent_folder, "LPIPS&DeltaE_result.csv")
    merged_df.to_csv(output_csv)
    print(f"Results saved to: {output_csv}")


if __name__ == "__main__":
    main()


