import tifffile
import numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter1d

# -------------------- USER SETTINGS --------------------
pIn = Path("/Users/sunny/Desktop/ChlorON_01292026_EQ_PosCtrl/")
pOut = pIn / "ds"
pOut.mkdir(exist_ok=True)

ds = int(input("Enter downsample factor (ds): "))
msPerFrame = float(input("Enter frame duration (ms): "))

# Calculate window size for rolling average (1 second of data)
window_size = int(round(1000 / msPerFrame))

# -------------------- LOA1D FILES --------------------
# Finds both .tif and .tiff
files = [f for f in pIn.glob("*.tif*") if "_snap" not in f.name]
print(f"Found {len(files)} valid files.")

# -------------------- PROCESS FILES --------------------
for file_path in files:
    print(f"\nProcessing: {file_path.name}")

    # Load: tifffile usually returns (Frames, H, W)
    data = tifffile.imread(file_path).astype(np.float32)

    # Standardize to 3D: (Frames, H, W)
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    
    # Ensure time is the first axis (standard for TIFF stacks)
    # If H < Frames, it likely loaded as (H, W, F), so we move F to front
    if data.ndim == 3 and data.shape[1] > data.shape[0] and data.shape[1] > data.shape[2]:
        data = np.moveaxis(data, 1, 0)

    print(f"Shape: {data.shape} (Frames, H, W)")

    # SMOOTHING: Fast rolling mean along the 0th (time) axis
    data = uniform_filter1d(data, size=window_size, axis=0)

    # DOWNSAMPLE: Skip frames
    data_ds = data[::ds, :, :]

    # -------------------- SAVE OUTPUT --------------------
    # Rename: change suffix or replace keywords
    out_name = pOut / file_path.name.replace("merged", "processed")
    print(f"Saving to: {out_name}")

    # Save as uint16 to preserve intensity but save space
    tifffile.imwrite(out_name, data_ds.astype(np.uint16), bigtiff=True)

print("\nAll files processed successfully.")