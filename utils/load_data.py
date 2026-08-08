import os
import shutil
from pathlib import Path
import kagglehub
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"


def is_dataset_present(target_path: Path) -> bool:
    """Check if target directory exists and contains at least one non-empty file."""
    if not target_path.exists():
        return False
    files = [f for f in target_path.rglob("*") if f.is_file() and f.stat().st_size > 0]
    return len(files) > 0


def download_competition(
    competition_name: str = "home-credit-default-risk",
    target_dir: Path | str | None = None,
    force_download: bool = False,
) -> Path:
    """
    Download a Kaggle competition dataset and save to data/raw/{competition_name}.
    Skips download if data already exists locally unless force_download=True.
    """
    dest_dir = Path(target_dir) if target_dir else DATA_RAW_DIR / competition_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not force_download and is_dataset_present(dest_dir):
        print(f"✓ Competition dataset '{competition_name}' already exists at: {dest_dir}")
        print("  Skipping download (use force_download=True to override).")
        return dest_dir

    print(f"Downloading Kaggle competition '{competition_name}' via kagglehub...")
    try:
        cache_path = Path(kagglehub.competition_download(competition_name))
        print(f"Competition files downloaded to cache: {cache_path}")
    except Exception as err:
        print(f"\n❌ [Kaggle API Error]: Failed to download competition '{competition_name}'.")
        print(f"URL cuộc thi: https://www.kaggle.com/competitions/{competition_name}/rules")
        print("Vui lòng truy cập đường dẫn trên bằng tài khoản Kaggle của bạn và nhấn 'I Understand and Accept' (Chấp nhận điều khoản cuộc thi).")
        raise err

    for item in cache_path.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(cache_path)
            dest_file = dest_dir / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest_file)
            print(f"Saved: {dest_file} ({dest_file.stat().st_size / (1024 * 1024):.2f} MB)")

    print(f"\n✓ Competition dataset successfully saved to: {dest_dir}")
    return dest_dir


def download_dataset(
    dataset_handle: str = "tranhuunhan/vietnam-bank-churn-dataset-2025",
    target_dir: Path | str | None = None,
    force_download: bool = False,
) -> Path:
    """
    Download a Kaggle public dataset and save to data/raw/{dataset_folder_name}.
    Skips download if data already exists locally unless force_download=True.
    """
    folder_name = dataset_handle.split("/")[-1]
    dest_dir = Path(target_dir) if target_dir else DATA_RAW_DIR / folder_name
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not force_download and is_dataset_present(dest_dir):
        print(f"✓ Dataset '{dataset_handle}' already exists at: {dest_dir}")
        print("  Skipping download (use force_download=True to override).")
        return dest_dir

    print(f"Downloading Kaggle dataset '{dataset_handle}' via kagglehub...")
    cache_path = Path(kagglehub.dataset_download(dataset_handle))
    print(f"Dataset downloaded to cache: {cache_path}")

    for item in cache_path.rglob("*"):
        if item.is_file():
            rel_path = item.relative_to(cache_path)
            dest_file = dest_dir / rel_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest_file)
            print(f"Saved: {dest_file} ({dest_file.stat().st_size / (1024 * 1024):.2f} MB)")

    print(f"\n✓ Dataset successfully saved to: {dest_dir}")
    return dest_dir


def load_all_datasets(force_download: bool = False):
    """Load all project datasets into separate subfolders under data/raw/."""
    print("==================================================")
    print("1. Loading Home Credit Default Risk Competition Data")
    print("==================================================")
    download_competition("home-credit-default-risk", force_download=force_download)


if __name__ == "__main__":
    load_all_datasets()
