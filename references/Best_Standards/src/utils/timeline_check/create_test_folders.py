import os
from datetime import datetime, timedelta

def create_test_folders(base_path, n=10):
    os.makedirs(base_path, exist_ok=True)
    start_date = datetime(2025, 1, 1)
    for i in range(n):
        folder_name = (start_date + timedelta(days=i)).strftime('%Y-%m-%d.parquet')
        folder_path = os.path.join(base_path, folder_name)
        os.makedirs(folder_path, exist_ok=True)
    print(f"Created {n} test folders in {base_path}")

if __name__ == "__main__":
    create_test_folders("test_folders", n=10)
