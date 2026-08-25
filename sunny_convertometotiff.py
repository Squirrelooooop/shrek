from pathlib import Path
import tifffile as tiff
from tifffile import TiffWriter
import re

# paths
input_dir = Path("/Users/sunny/Desktop/20260819_FrodoNeu_v37")              # folder with .ome.tif files
output_dir = input_dir / "multipage_tiff"    # output folder
output_dir.mkdir(exist_ok=True)

OME_SUFFIX_RE = re.compile(r"_(\d+)\.ome\.tiff?$", re.IGNORECASE)

def ome_sort_key(path):
    # MicroManager names the first chunk "<prefix>.ome.tif(f)" and later
    # chunks "<prefix>_1.ome.tif(f)", "<prefix>_2.ome.tif(f)", ... Whatever
    # <prefix> is (Default, Pos0, a series name, ...), the chunk index is
    # always the last "_<N>" right before the extension, so match that
    # generically instead of a hardcoded prefix - a hardcoded prefix left
    # every file with the same sort key (scrambling the stitch order) as
    # soon as the real filenames didn't match it.
    m = OME_SUFFIX_RE.search(path.name)
    return int(m.group(1)) if m else 0


for subfolder in sorted(p for p in input_dir.iterdir() if p.is_dir()):

    ome_files = sorted(
        (p for p in subfolder.iterdir() if re.search(r"\.ome\.tiff?$", p.name, re.IGNORECASE)),
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
