from pathlib import Path
import tifffile as tiff

# paths
input_dir = Path("/Users/sunny/Desktop/Data/20250725_IvanHEK_MscL/all_ome_tiff")              # folder with .ome.tif files
output_dir = input_dir / "multipage_tiff"    # output folder
output_dir.mkdir(exist_ok=True)

for ome_file in input_dir.rglob("*.ome.tif"):
    print(f"Converting: {ome_file.name}")

    with tiff.TiffFile(ome_file) as tif:
        pages = [p.asarray() for p in tif.pages]

    out_name = ome_file.stem.replace(".ome", "") + ".tif"
    out_file = output_dir / out_name

    tiff.imwrite(
        out_file,
        pages,
        photometric="minisblack",
        metadata=None,
        bigtiff=True      # ⭐ THIS FIXES YOUR ERROR
    )

print("Done.")
