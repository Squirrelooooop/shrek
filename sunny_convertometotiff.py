from pathlib import Path
import tifffile as tiff
from tifffile import TiffWriter
import re

# paths
input_dir = Path("/Users/sunny/Desktop/20260115_flyc_chloron")              # folder with .ome.tif files
output_dir = input_dir / "multipage_tiff"    # output folder
output_dir.mkdir(exist_ok=True)

def ome_sort_key(path):
    name = path.name
    m = re.search(r"_Default_(\d+)\.ome\.tif$", name)
    if m:
        return int(m.group(1))
    elif name.endswith("_Default.ome.tif"):
        return 0
    else:
        return 9999


for subfolder in sorted(p for p in input_dir.iterdir() if p.is_dir()):

    ome_files = sorted(
        subfolder.glob("*.ome.tif"),
        key=ome_sort_key
    )

    if not ome_files:
        continue

    print(f"Processing folder: {subfolder.name}")
   
    out_file = output_dir / f"{subfolder.name}.tif"
    
    page_count = 0

    with TiffWriter(out_file, bigtiff=True) as tif:
        for ome_file in ome_files:
            print(f"  Reading: {ome_file.name}")
            with tiff.TiffFile(ome_file) as src:
                for page in src.pages:
                    tif.write(
                        page.asarray(),
                        photometric="minisblack"
                    )
                    page_count += 1
    print(f"  Saved {out_file.name} with {page_count} pages\n")

print("Done.")
