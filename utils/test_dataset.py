import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.mmfakebench import MMFakeBenchDataset

print("Testing Dataset Loading...")
try:
    ds = MMFakeBenchDataset("dataset/train.json", "dataset/images/")
    print(f"Total samples: {len(ds)}")
    if len(ds) > 0:
        item = ds[0]
        print(f"First item text: {item['text'][:50]}...")
        print(f"First item label: {item['label']}")
        print(f"First item image path: {item['image_path']}")
        if item['image'] is not None:
             print(f"Image loaded successfully! Size: {item['image'].size}")
        else:
             print("Image failed to load.")
    print("Dataset verification successful!")
except Exception as e:
    print(f"Error: {e}")
