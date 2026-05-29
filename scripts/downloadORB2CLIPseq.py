"""Download ORB2B PAR-CLIP data from CLIPdb (fly transcriptome-wide CLIP atlas).

The file fly.txt.gz contains PAR-CLIP / PARalyzer peaks for multiple Drosophila RBPs
(including ORB2B / GSE59611) and is hosted by the CLIPdb resource at Tsinghua.

Usage:
    python scripts/downloadORB2CLIPseq.py --output_dir data/CLIPseq
"""
import argparse
import os
import subprocess
import sys

FILES = {
    "fly_clip.txt.gz": (
        "https://cloud.tsinghua.edu.cn/d/be6d966ba7834f44b62b/files/"
        "?p=%2Ffly.txt.gz&dl=1"
    ),
}


def main():
    parser = argparse.ArgumentParser(
        description="Download ORB2B PAR-CLIP data from CLIPdb."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the downloaded files.",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for filename, url in FILES.items():
        dest = os.path.join(args.output_dir, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"  {filename} already exists, skipping download.")
            continue
        print(f"Downloading {filename} ...")
        result = subprocess.run(
            ["curl", "-L", "--retry", "3", "-o", dest, url],
            check=False,
        )
        if result.returncode != 0:
            print(
                f"ERROR: curl exited with code {result.returncode} for {filename}",
                file=sys.stderr,
            )
            sys.exit(result.returncode)
        print(f"  Saved to {dest} ({os.path.getsize(dest) / 1e6:.1f} MB)")

    print("Download complete.")


if __name__ == "__main__":
    main()
