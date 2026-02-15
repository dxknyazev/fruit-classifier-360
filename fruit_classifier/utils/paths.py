from pathlib import Path


def get_data_path() -> Path:
    """Возвращает путь к папке с данными (data/raw)."""
    return Path(__file__).parent.parent.parent / "data" / "raw"
