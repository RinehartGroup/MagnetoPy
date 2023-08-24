import argparse
import shutil
from pathlib import Path

from git import Repo

magnetopy_dir = Path.home() / ".magnetopy"


def install_calibration(git_url):
    if not magnetopy_dir.exists():
        magnetopy_dir.mkdir()

    calibration_dir = magnetopy_dir / "calibration"

    if calibration_dir.exists():
        # If the directory already exists, remove it first to ensure a clean installation
        shutil.rmtree(calibration_dir)

    Repo.clone_from(git_url, calibration_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Install calibration data from GitHub repository URL"
    )
    parser.add_argument(
        "git_url", help="URL of the GitHub repository to install calibration data from"
    )

    args = parser.parse_args()
    install_calibration(args.git_url)

    print(f"Calibration data installed successfully in {magnetopy_dir}")


if __name__ == "__main__":
    main()
