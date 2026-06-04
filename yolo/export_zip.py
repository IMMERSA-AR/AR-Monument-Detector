import os
import zipfile


DATASETS = [
    {
        "name":       "obelisk",
        "source_dir": "data/processed/obelisk/augmented",
        "output_zip": "data/processed/obelisk/obelisk_augmented_dataset.zip",
    },
    {
        "name":       "panel",
        "source_dir": "data/processed/panel/augmented",
        "output_zip": "data/processed/panel/panel_augmented_dataset.zip",
    },
]


def create_zip(source_dir, output_zip):
    file_count = 0
    os.makedirs(os.path.dirname(output_zip), exist_ok=True)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, _, filenames in os.walk(source_dir):
            for fname in filenames:
                abs_path = os.path.join(dirpath, fname)
                arcname  = abs_path.replace("\\", "/")   
                zf.write(abs_path, arcname)
                file_count += 1

    size_mb = os.path.getsize(output_zip) / 1e6
    return file_count, size_mb

created = 0
skipped = 0

for ds in DATASETS:
    name       = ds["name"]
    source_dir = ds["source_dir"]
    output_zip = ds["output_zip"]

    if not os.path.isdir(source_dir):
        print(f"Skip {name:8s}, '{source_dir}' not found. " f"Run augment.py first.")
        skipped += 1
        continue

    print(f"Zip {name:8s}, compressing '{source_dir}' ...")
    file_count, size_mb = create_zip(source_dir, output_zip)
    print(f"Created : {output_zip}, Files   : {file_count}, Size: {size_mb:.1f} MB")
    created += 1

print()
print(f"Done. {created} zip(s) created, {skipped} skipped.")