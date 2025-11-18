# uv run src/mains/timeline.py

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import initialize_main  # noqa: E402
from src.config.config_settings import OUTPUT_PATH, PROCESSED_PATH  # noqa: E402


def collect_date_folders(base_path: Path) -> list[str]:
    if not base_path.exists():
        raise FileNotFoundError(f"PROCESSED_PATH does not exist: {base_path}")

    dates: set[str] = set()
    for entry in base_path.iterdir():
        if not entry.is_dir():
            continue
        try:
            datetime.strptime(entry.name, "%Y-%m-%d")
        except ValueError:
            continue
        dates.add(entry.name)

    return sorted(dates)


def write_timeline(dates: list[str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as txt_file:
        txt_file.write("\n".join(dates))


def main() -> None:
    logger = initialize_main()

    try:
        date_folders = collect_date_folders(PROCESSED_PATH)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        return

    if not date_folders:
        logger.warning(f"No date folders found under {PROCESSED_PATH}")
        return

    # timeline_file = OUTPUT_PATH / "timeline.txt"
    timeline_file = Path("output") / "timeline.txt"
    write_timeline(date_folders, timeline_file)

    logger.info(
        "Wrote %d dates to %s",
        len(date_folders),
        timeline_file,
    )


if __name__ == "__main__":
    main()

