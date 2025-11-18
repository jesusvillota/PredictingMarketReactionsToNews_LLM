# uv run src/mains/delete_daily_folders.py
"""
Examples:
  # Dry-run: show what would be deleted from year 2021
  uv run src/mains/delete_daily_folders.py --path "E:/DUCKDB_PROCESSED_TRADE_DATA_PARQUET" --year 2021

  # Actually delete year 2021 (with confirmation)
  uv run src/mains/delete_daily_folders.py --path "E:/DUCKDB_PROCESSED_TRADE_DATA_PARQUET" --year 2021 --execute

  # Delete multiple selections: year 2021 AND specific date
  uv run src/mains/delete_daily_folders.py --path "E:/DUCKDB_PROCESSED_TRADE_DATA_PARQUET" --year 2021 --date 2018-08-02 --execute

  # Delete specific year-month
  uv run src/mains/delete_daily_folders.py --path "E:/DUCKDB_PROCESSED_TRADE_DATA_PARQUET" --year-month 2015-08 --execute
"""

from pathlib import Path
import sys
import shutil
import argparse
from typing import List

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config.utils import DailyFolderFilter


def get_folder_size(folder: Path) -> int:
    """Calculate total size of a folder in bytes."""
    total = 0
    try:
        for entry in folder.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
    except (PermissionError, OSError):
        pass
    return total


def format_size(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def collect_folders_to_delete(args) -> List[Path]:
    """Collect all folders matching the filter criteria."""
    filter = DailyFolderFilter(args.path)
    folders_to_delete = []
    
    if args.year:
        for year in args.year:
            folders = filter.by_year([year], return_globs=False)
            folders_to_delete.extend(folders)
            print(f"  Year {year}: {len(folders)} folders")
    
    if args.year_month:
        for year_month in args.year_month:
            folders = filter.by_year_month([year_month], return_globs=False)
            folders_to_delete.extend(folders)
            print(f"  Year-month {year_month}: {len(folders)} folders")
    
    if args.date:
        for date in args.date:
            folders = filter.by_year_month_date([date], return_globs=False)
            folders_to_delete.extend(folders)
            print(f"  Date {date}: {len(folders)} folders")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_folders = []
    for folder in folders_to_delete:
        if folder not in seen:
            seen.add(folder)
            unique_folders.append(folder)
    
    return unique_folders


def confirm_deletion() -> bool:
    """Ask user for confirmation before deletion."""
    while True:
        response = input("\nAre you sure you want to DELETE these folders? Type 'yes' to confirm or 'no' to cancel: ").strip().lower()
        if response in ['yes', 'y']:
            return True
        elif response in ['no', 'n']:
            return False
        else:
            print("Please type 'yes' or 'no'")


def delete_folders(folders: List[Path]) -> None:
    """Delete folders and log results."""
    deleted_count = 0
    failed_count = 0
    
    print("\nDeleting folders...")
    for folder in folders:
        try:
            shutil.rmtree(folder)
            deleted_count += 1
            print(f"  ✓ Deleted: {folder}")
        except (PermissionError, OSError) as e:
            failed_count += 1
            print(f"  ✗ Failed to delete {folder}: {e}")
    
    print(f"\n✓ Successfully deleted: {deleted_count} folders")
    if failed_count > 0:
        print(f"✗ Failed to delete: {failed_count} folders")


def main():
    parser = argparse.ArgumentParser(
        description="Delete daily folders (YYYY-MM-DD) using DailyFolderFilter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry-run: show what would be deleted from year 2016
  uv run src/mains/delete_daily_folders.py --path "E:\DUCKDB_PROCESSED_TRADE_DATA_PARQUET" --year 2016

  # Actually delete year 2016 (with confirmation)
  uv run src/mains/delete_daily_folders.py --path "E:\DUCKDB_PROCESSED_TRADE_DATA_PARQUET" --year 2016 --execute

  # Delete multiple selections: year 2016 AND specific date
  uv run src/mains/delete_daily_folders.py --path "E:\DUCKDB_PROCESSED_TRADE_DATA_PARQUET" --year 2016 --date 2018-08-02 --execute

  # Delete specific year-month
  uv run src/mains/delete_daily_folders.py --path "E:\DUCKDB_PROCESSED_TRADE_DATA_PARQUET" --year-month 2015-08 --execute
        """
    )
    
    parser.add_argument(
        '--path',
        type=str,
        required=True,
        help='Path to directory containing YYYY-MM-DD folders'
    )
    
    parser.add_argument(
        '--year',
        type=int,
        nargs='+',
        help='Delete all folders from specified years (e.g., --year 2016 2017)'
    )
    
    parser.add_argument(
        '--year-month',
        type=str,
        nargs='+',
        help='Delete folders from specific year-months (e.g., --year-month 2015-08 2016-01)'
    )
    
    parser.add_argument(
        '--date',
        type=str,
        nargs='+',
        help='Delete specific dates (e.g., --date 2018-08-02 2019-01-15)'
    )
    
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually perform deletion (without this flag, dry-run only)'
    )
    
    args = parser.parse_args()
    
    # Validate path
    path = Path(args.path)
    if not path.exists():
        print(f"✗ Error: Path does not exist: {path}")
        sys.exit(1)
    
    if not path.is_dir():
        print(f"✗ Error: Path is not a directory: {path}")
        sys.exit(1)
    
    # Ensure at least one filter is provided
    if not any([args.year, args.year_month, args.date]):
        print("✗ Error: At least one filter must be specified (--year, --year-month, or --date)")
        parser.print_help()
        sys.exit(1)
    
    print(f"\n📂 Target directory: {path}")
    print(f"{'🔍 DRY-RUN MODE' if not args.execute else '⚠️  EXECUTION MODE'}\n")
    print("Filters applied:")
    
    # Collect folders to delete
    folders_to_delete = collect_folders_to_delete(args)
    
    if not folders_to_delete:
        print("\n✓ No folders match the specified criteria. Nothing to delete.")
        return
    
    # Calculate total size
    print(f"\n📊 Summary:")
    print(f"  Total folders to delete: {len(folders_to_delete)}")
    
    total_size = 0
    for folder in folders_to_delete:
        total_size += get_folder_size(folder)
    
    print(f"  Total size: {format_size(total_size)}")
    
    # Show first few folders as preview
    if len(folders_to_delete) <= 10:
        print("\n  Folders to be deleted:")
        for folder in sorted(folders_to_delete):
            print(f"    - {folder}")
    else:
        print("\n  First 5 folders to be deleted:")
        for folder in sorted(folders_to_delete)[:5]:
            print(f"    - {folder}")
        print(f"    ... and {len(folders_to_delete) - 5} more")
    
    # Execute deletion if requested
    if args.execute:
        if confirm_deletion():
            delete_folders(folders_to_delete)
        else:
            print("\n❌ Deletion cancelled by user.")
    else:
        print("\n💡 This is a dry-run. Use --execute flag to actually delete these folders.")


if __name__ == '__main__':
    main()

