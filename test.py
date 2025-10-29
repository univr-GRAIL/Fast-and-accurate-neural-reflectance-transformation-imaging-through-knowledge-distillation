import argparse
import numpy as np
from modules.relight.relight import relight

def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Run relighting model on test data.")

    # Define arguments with default values
    parser.add_argument(
        "--model_path",
        type=str,
        default="test_dataset/outputs/Student",
        help="Path to the trained model directory (default: test_dataset/outputs/Student)."
    )

    parser.add_argument(
        "--light_path",
        type=str,
        default="test_dataset/test_fig6",
        help="Path to the test light direction and mask path, if needed (default: test_dataset/test_fig6)."
    )

    parser.add_argument(
        "--masked",
        action="store_true", default=False,
        help="Use mask during relighting (default: False)."
    )

    # Parse arguments
    args = parser.parse_args()

    # Initialize results array
    results = np.zeros((1, 2))

    # Run relight
    relight(args.model_path, args.light_path, mask=args.masked)

if __name__ == "__main__":
    main()
