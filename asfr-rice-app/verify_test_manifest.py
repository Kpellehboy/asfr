
import csv
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = (
    PROJECT_ROOT
    / "../../asfr_resume_bundle/manifests/splits/test_manifest.csv"
).resolve()
LOCK_PATH = (
    PROJECT_ROOT
    / "../../asfr_resume_bundle/reports/split_lock.json"
).resolve()
DATASET_ROOT = (
    PROJECT_ROOT / "../../Rice_Leaf_AUG"
).resolve()

def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

with LOCK_PATH.open("r", encoding="utf-8") as file:
    split_lock = json.load(file)

with MANIFEST_PATH.open("r", encoding="utf-8-sig", newline="") as file:
    rows = list(csv.DictReader(file))

manifest_hash = sha256_file(MANIFEST_PATH)
expected_manifest_hash = split_lock["manifest_sha256"]["test"]

print("TEST MANIFEST INTEGRITY")
print("=" * 45)
print("Manifest records:", len(rows))
print("Manifest SHA-256 matches split lock:",
      manifest_hash == expected_manifest_hash)

mismatches = []
missing = []

for index, row in enumerate(rows, start=1):
    relative_path = row["relative_path"]

    # Local folder-name mapping only; manifest remains unchanged.
    relative_path = relative_path.replace(
        "Neck Blast/", "Neck_Blast/", 1
    )

    image_path = DATASET_ROOT / relative_path

    if not image_path.is_file():
        missing.append(relative_path)
        continue

    actual_hash = sha256_file(image_path)
    expected_hash = row["sha256"].strip()

    if actual_hash.lower() != expected_hash.lower():
        mismatches.append(relative_path)

    if index % 250 == 0:
        print(f"Checked {index}/{len(rows)} images")

print("\nRESULT")
print("=" * 45)
print("Missing files:", len(missing))
print("SHA-256 mismatches:", len(mismatches))
print("Manifest hash valid:", manifest_hash == expected_manifest_hash)

if missing:
    print("\nMissing examples:", missing[:5])

if mismatches:
    print("\nMismatch examples:", mismatches[:5])

if (
    not missing
    and not mismatches
    and manifest_hash == expected_manifest_hash
):
    print("\nPASS: Locked test images verified.")
else:
    print("\nFAIL: Do not proceed to gate analysis.")
