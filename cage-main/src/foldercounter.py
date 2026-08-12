import os
'''
# Replace with your directory path
target_directory = "/data/cage/mel_spectograms_counting_128"  # Change this to the directory you want to count folders in

# Count all folders recursively
folder_count = sum(len(dirs) for _, dirs, _ in os.walk(target_directory))

print(f"Total folders (including subfolders): {folder_count}")'''

import os
from pathlib import Path

# 1. Set your path here
target_input = "../data/cage/mel_spectograms_counting_128"

# 2. Convert to absolute path to see exactly where Python is looking
path = Path(target_input).resolve()
print(f"--- DEBUG INFO ---")
print(f"Target Path (Absolute): {path}")
print(f"Does path exist?:        {path.exists()}")
print(f"Is it a directory?:     {path.is_dir()}")

# 3. Test reading the directory contents
print(f"\n--- SCANNING CONTENTS ---")
try:
    all_items = list(path.iterdir())
    print(f"Total items found (files + folders): {len(all_items)}")
    
    if len(all_items) > 0:
        print("\nFirst 5 items found:")
        for item in all_items[:5]:
            item_type = "Folder" if item.is_dir() else "File  "
            print(f" [{item_type}] {item.name}")
    else:
        print("The directory appears completely empty to Python.")
        
except Exception as e:
    print(f"Error accessing directory: {e}")
