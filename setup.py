"""
setup.py - One-command setup: install deps + pretrain model.

Usage:
    python setup.py
"""

import subprocess
import sys
import os

def run(cmd):
    print(f"\n{'='*50}")
    print(f"  {cmd}")
    print(f"{'='*50}\n")
    subprocess.check_call(cmd, shell=True)


def main():
    print("\n" + "=" * 55)
    print("  Sign Language Detector - Setup")
    print("=" * 55)

    # Step 1: Install dependencies
    print("\n[1/3] Installing dependencies...\n")
    run(f"{sys.executable} -m pip install -r requirements.txt")

    # Step 2: Verify imports
    print("\n[2/3] Verifying imports...\n")
    try:
        import numpy
        print(f"  numpy:         {numpy.__version__}")
        import cv2
        print(f"  opencv:        {cv2.__version__}")
        import mediapipe
        print(f"  mediapipe:     {mediapipe.__version__}")
        import sklearn
        print(f"  scikit-learn:  {sklearn.__version__}")
        import flask
        print(f"  flask:         {flask.__version__}")
        print("\n  All imports OK!")
    except ImportError as e:
        print(f"\n  MISSING: {e}")
        print("  Try: pip install -r requirements.txt")
        sys.exit(1)

    # Step 3: Pretrain model
    print("\n[3/3] Pre-training baseline model...\n")
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    run(f"{sys.executable} pretrain.py --samples 300")

    print("\n" + "=" * 55)
    print("  Setup complete!")
    print()
    print("  Run the app:")
    print("    python app.py")
    print()
    print("  Then open: http://localhost:5000")
    print()
    print("  To improve accuracy:")
    print("    python collect.py    # record your own signs")
    print("    python train.py      # retrain on real data")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
