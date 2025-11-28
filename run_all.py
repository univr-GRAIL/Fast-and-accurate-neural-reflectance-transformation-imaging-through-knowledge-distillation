import os
import subprocess
import sys
import argparse
def main():
     parser = argparse.ArgumentParser(description="Run test on all RealRTI datasets.")
     parser.add_argument(
        "--parent_folder",
        type=str,
        default="test_dataset/outputs/Student",
        help="Path to the trained model directores."
    )
     args = parser.parse_args()

    # Define arguments with default values

     for sub in os.listdir(args.parent_folder):
         sub_folder = os.path.join(args.parent_folder, sub)
         if not os.path.isdir(sub_folder):
             continue

         model_file = os.path.join(sub_folder, "outputs")
         light_file = os.path.join(sub_folder, "test", "dirs.lp")

    # Skip if required files do not exist
         if not (os.path.exists(model_file) and os.path.exists(light_file) ):
           print(f"Skipping {sub_folder} — missing files")
           continue

         print(f"Running test for: {sub_folder}")

         subprocess.call([
           sys.executable, "test.py",
            "--model_path", model_file,
            "--light_path", light_file,
           ])
     print("Calculating metrics for all datasets...")
     subprocess.call([
        sys.executable, "calculate_metrics.py",
        "--parent_folder", args.parent_folder ])
if __name__ == "__main__":
    main()

