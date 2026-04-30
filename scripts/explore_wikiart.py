"""Quick script to explore WikiArt dataset structure on the training machine.

Run on training machine:
    python scripts/explore_wikiart.py
"""

from pathlib import Path
from collections import Counter

WIKIART_DIR = Path("/transfer/wikidatasets/steubk/wikiart/version")

image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# List top-level folders (likely style or artist categories)
subdirs = sorted([d.name for d in WIKIART_DIR.iterdir() if d.is_dir()])
print(f"Top-level directories ({len(subdirs)}):")
for d in subdirs[:30]:
    count = sum(1 for f in (WIKIART_DIR / d).rglob("*") if f.suffix.lower() in image_exts)
    print(f"  {d}: {count} images")
if len(subdirs) > 30:
    print(f"  ... and {len(subdirs) - 30} more")

# Count total images
total = sum(1 for f in WIKIART_DIR.rglob("*") if f.suffix.lower() in image_exts)
print(f"\nTotal images: {total}")

# Check file extensions
ext_counts = Counter(f.suffix.lower() for f in WIKIART_DIR.rglob("*") if f.is_file())
print(f"\nFile extensions: {dict(ext_counts)}")

# Check if there are any metadata files
meta_files = [f.name for f in WIKIART_DIR.rglob("*") if f.suffix.lower() in {".json", ".csv", ".txt", ".jsonl"}]
if meta_files:
    print(f"\nMetadata files found: {meta_files[:20]}")
