import tifffile
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F

# -------------------- USER SETTINGS --------------------
pIn = Path("/Users/sunny/Desktop/20260126_IvanHEK_FLYChlorOn/multipage_tiff/")
pOut = pIn / "ds"
pOut.mkdir(exist_ok=True)

ds = int(input("Enter downsample factor (ds): "))
msPerFrame = float(input("Enter frame duration (ms): "))

# Calculate window size for 1-second rolling mean
window_size = int(round(1000 / msPerFrame))

# Choose device (MPS for Apple GPU, else CPU)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(f"Using device: {device}")

# -------------------- FILE LIST --------------------
files = [f for f in pIn.glob("*.tif*") if "_snap" not in f.name]
print(f"Found {len(files)} valid files.")

# -------------------- GPU SMOOTHING FUNCTION --------------------
def smooth_time_torch(data, window_size, chunk=200):
    """
    Smooth a 3D movie along the time axis using GPU (MPS if available).
    
    Parameters
    ----------
    data : np.ndarray
        Input movie of shape (T, H, W)
    window_size : int
        Length of rolling average
    chunk : int
        Number of frames to process at a time (memory-friendly)

    Returns
    -------
    out : np.ndarray
        Smoothed movie, same shape as input
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    
    T, H, W = data.shape
    out = np.empty_like(data, dtype=np.float32)
    
    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, device=device) / window_size

    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        t0 = start
        t1 = end
        
        # Load chunk to GPU
        x = torch.from_numpy(data[t0:t1]).to(device)  # (frames,H,W)
        x = x.permute(1, 2, 0).reshape(-1, 1, x.shape[0])  # (H*W,1,frames)
        
        # Pad on both sides along time to ensure full convolution
        x = F.pad(x, (pad, pad), mode='replicate')
        
        # Convolve along time
        y = F.conv1d(x, kernel, padding=0)
        
        # Reshape back to (frames, H, W)
        y = y.reshape(H, W, -1).permute(2, 0, 1)
        
        # Slice exactly to original chunk length
        out[start:end] = y[:end-start].cpu().numpy()
    
    return out

# -------------------- PROCESS FILES --------------------
for file_path in files:
    print(f"\nProcessing: {file_path.name}")

    # Load multipage TIFF as (T,H,W)
    with tifffile.TiffFile(file_path) as tif:
        data = np.stack([page.asarray() for page in tif.pages], axis=0).astype(np.float32)

    print(f"Loaded: {data.shape} (Frames, H, W)")

    # GPU smoothing
    data = smooth_time_torch(data, window_size)
    print("Smoothing done.")

    # Downsample
    data_ds = data[::ds]
    print(f"Downsampled: {data_ds.shape[0]} frames")

    # Save BigTIFF
    out_name = pOut / file_path.name.replace("merged", "processed")
    tifffile.imwrite(out_name, data_ds.astype(np.uint16), bigtiff=True)
    print(f"Saved: {out_name}\n")

print("\nAll files processed successfully.")