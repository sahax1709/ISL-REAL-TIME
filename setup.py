"""
setup.py - Install deps + pretrain baseline model.
Run: python setup.py
"""

import subprocess
import sys
import os


def run(cmd):
    print(f"\n{'='*50}\n  {cmd}\n{'='*50}\n")
    subprocess.check_call(cmd, shell=True)


def main():
    print("\n" + "=" * 55)
    print("  Sign Language Detector - Setup")
    print("=" * 55)

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    print("\n[1/3] Installing dependencies...\n")
    run(f'"{sys.executable}" -m pip install -r requirements.txt')

    print("\n[2/3] Verifying...\n")
    ok = True
    for mod, name in [("numpy","numpy"),("cv2","opencv"),("mediapipe","mediapipe"),("sklearn","scikit-learn"),("flask","flask"),("joblib","joblib")]:
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "ok")
            print(f"  {name:15s} {v}")
        except ImportError:
            print(f"  {name:15s} MISSING")
            ok = False
    if not ok:
        print("\n  Some packages missing. Check pip output above.")
        sys.exit(1)
    print("\n  All imports OK!")

    print("\n[3/3] Pre-training baseline model...\n")
    run(f'"{sys.executable}" pretrain.py --samples 300')

    print("\n" + "=" * 55)
    print("  Setup complete!\n")
    print("  Start the app:")
    print("    python app.py\n")
    print("  Then open: http://localhost:5000\n")
    print("  To improve accuracy:")
    print("    python collect.py")
    print("    python train.py")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
