import shutil
import zipfile
from pathlib import Path

import requests


def download_data():
    """Скачивает и распаковывает датасет Fruits-360 в data/raw."""
    url = (
        "https://github.com/fruits-360/fruits-360-100x100/archive/refs/heads/master.zip"
    )
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    zip_path = data_dir / "fruits.zip"
    print(f"Скачивание {url} ...")
    r = requests.get(url, stream=True)
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    extract_path = data_dir / "temp_extract"
    if extract_path.exists():
        shutil.rmtree(extract_path)
    extract_path.mkdir()

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_path)

    zip_path.unlink()

    items = list(extract_path.iterdir())
    if len(items) != 1 or not items[0].is_dir():
        raise RuntimeError("Неожиданная структура архива")

    source_dir = items[0]

    train_src = source_dir / "Training"
    test_src = source_dir / "Test"

    if not train_src.exists() or not test_src.exists():
        raise RuntimeError("В архиве нет папок Training и Test")

    target_dir = data_dir / "raw"
    if target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir()

    shutil.move(str(train_src), str(target_dir / "Training"))
    shutil.move(str(test_src), str(target_dir / "Test"))

    shutil.rmtree(extract_path)

    print(f"Данные загружены и распакованы в {target_dir}")


if __name__ == "__main__":
    download_data()
