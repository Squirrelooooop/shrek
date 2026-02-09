from pathlib import Path
import tifffile
import numpy as np
import re
import torch
from cellpose import models, core, io, plot
from tqdm import trange
from natsort import natsorted
import pandas as pd
import scipy.io as sio

def ome_sort_key(path: Path):
    name = path.name
    m = re.search(r"_Default_(\d+)\.ome\.tif$", name)
    if m:
        return int(m.group(1))
    elif name.endswith("_Default.ome.tif"):
        return 0
    else:
        return 9999


def ometotiff(input_dir, output_dir=None):
    """
    Merge OME-TIFF files inside each subfolder into one multipage TIFF.

    Parameters
    ----------
    input_dir : str or Path
        Folder containing subfolders with .ome.tif files.
    output_dir : str or Path or None
        Where merged files will be written.
        If None, creates 'multipage_tiff' inside input_dir.
    """

    input_dir = Path(input_dir)

    if output_dir is None:
        output_dir = input_dir / "multipage_tiff"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

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

        with tifffile.TiffWriter(out_file, bigtiff=True) as tif:
            for ome_file in ome_files:
                print(f"  Reading: {ome_file.name}")
                with tifffile.TiffFile(ome_file) as src:
                    for page in src.pages:
                        tif.write(
                            page.asarray(),
                            photometric="minisblack"
                        )
                        page_count += 1

        print(f"  Saved {out_file.name} with {page_count} pages\n")

    #print the output directory
    print(f"All folders processed. Merged files saved to: {output_dir}")

def smooth_time_torch(data, window_size, chunk=200, device=None):
    """
    Smooth a 3D movie along the time axis using GPU/CPU.
    
    Parameters
    ----------
    data : np.ndarray
        Input movie of shape (T, H, W)
    window_size : int
        Length of rolling average
    chunk : int
        Number of frames to process at a time (memory-friendly)
    device : torch.device or None
        Device to run convolution on (MPS, CUDA, or CPU). If None, defaults to CPU.
        
    Returns
    -------
    out : np.ndarray
        Smoothed movie, same shape as input
    """
    if device is None:
        device = torch.device("cpu")

    T, H, W = data.shape
    out = np.empty_like(data, dtype=np.float32)
    
    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, device=device) / window_size

    print(f"Smoothing on device: {device}")

    for start in range(0, T, chunk):
        end = min(start + chunk, T)
        t0 = start
        t1 = end
        
        # Load chunk to device
        x = torch.from_numpy(data[t0:t1]).to(device).float()          # (frames,H,W)
        x = x.permute(1, 2, 0).reshape(-1, 1, x.shape[0])           # (H*W,1,frames)
        
        # Pad on both sides along time
        x = torch.nn.functional.pad(x, (pad, pad), mode='replicate')
        
        # Convolve along time
        y = torch.nn.functional.conv1d(x, kernel, padding=0)
        
        # Reshape back to (frames,H,W)
        y = y.reshape(H, W, -1).permute(2, 0, 1)
        
        # Slice exactly to original chunk length
        out[start:end] = y[:end-start].cpu().numpy()
    
    return out


def downsample(input_dir, msPerFrame, ds_factor = 10, output_dir=None):
    """
    Downsample multipage TIFF files by averaging every ds_factor frames.
    
    input_dir: the folder containing the multipage TIFF files to be downsampled
    msPerFrame: what is the original frame rate (in ms per frame) of the TIFF files? This is needed to calculate the new frame rate after downsampling.
    ds_factor: the factor by which to downsample (default 10 means average every 10 frames)
    output_dir: where to save the downsampled files. If None, creates 'ds' inside input_dir.
    """
    pIn = Path(input_dir)
    if output_dir is None:
        output_dir = pIn/ "ds"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(exist_ok=True)

    # Calculate window size for 1-second rolling mean
    window_size = int(round(1000 / msPerFrame))

    # -------------------- DEVICE SELECTION --------------------
    if torch.backends.mps.is_available():        # Mac Apple GPU
        device = "mps"
        print("Using Apple GPU (MPS)")
    elif torch.cuda.is_available():             # Windows/Linux NVIDIA GPU
        device = "cuda"
        print("Using CUDA GPU")
    else:                                       # fallback to CPU
        device = "cpu"
        print("Using CPU")

    torch_device = torch.device(device)

    # -------------------- FILE LIST --------------------
    files = [f for f in pIn.glob("*.tif*") if "_snap" not in f.name]
    print(f"Found {len(files)} valid files.")   

    # -------------------- PROCESS FILES --------------------
    for file_path in files:
        print(f"\nProcessing: {file_path.name}")

        # Load multipage TIFF as (T,H,W)
        with tifffile.TiffFile(file_path) as tif:
            data = np.stack([page.asarray() for page in tif.pages], axis=0).astype(np.float32)

        print(f"Loaded: {data.shape} (Frames, H, W)")

        # GPU smoothing
        data = smooth_time_torch(data, window_size, device=torch_device)
        print("Smoothing done.")

        # Downsample
        data_ds = data[::ds_factor]
        print(f"Downsampled: {data_ds.shape[0]} frames")

        # Save BigTIFF
        out_name = output_dir / file_path.name.replace("merged", "processed")
        tifffile.imwrite(out_name, data_ds.astype(np.uint16), bigtiff=True)
        print(f"Saved: {out_name}\n")

    #print the output directory
    print(f"\nAll files processed successfully. Downsampled files saved to: {output_dir}")

def tiff_to_mask(input_dir, snap_dir):
    """
    Convert multipage TIFF files to masks using Cellpose and count cells in each frame.
    
    input_dir: folder containing the multipage TIFF files
    snap_dir: folder containing the corresponding snapshot TIFF files for cell counting
    """

    io.logger_setup()
    if torch.backends.mps.is_available():   # Mac Apple GPU
        device = "mps"
        print("Using Apple GPU (MPS) for Cellpose")
    elif torch.cuda.is_available():          # Windows/Linux NVIDIA GPU
        device = "cuda"
        print("Using CUDA GPU for Cellpose")
    else:                                    # fallback to CPU
        device = "cpu"
        print("Using CPU for Cellpose")

    torch_device = torch.device(device)
    model = models.CellposeModel(device=torch_device)
    dir = Path(input_dir)
    snap = Path(snap_dir)
    output_csv = dir / 'cell_counts.csv'
    image_ext = ".tif"
    masks_ext = "_masks.tif"

    files = natsorted([f for f in dir.glob("*"+image_ext) if "_masks" not in f.name and "_flows" not in f.name])
    results = []

    for f in files:
        print(f.name)
        img = tifffile.imread(f)

        print(f"dimentions", img.ndim)
        if img.ndim == 3:
            img = np.max(img, axis=0)
            #save max projection as new tiff
            max_proj_path = dir / f"{f.stem}_maxproj{image_ext}"
            tifffile.imwrite(max_proj_path, img)
            print("Saved max projection:", max_proj_path)
        img_tensor = torch.from_numpy(img).to(torch.float32).to(torch_device)
        masks, flows, styles = model.eval(img_tensor, normalize={"tile_norm_blocksize": 256})
        
        cell_ids = np.unique(masks)
        cell_ids = cell_ids[cell_ids != 0] #remove background

        # Determine snapshot file
        prefix = f.stem.split('_')[0]
        snap_file = snap / f"{prefix}_snap{image_ext}"

        cell_ids_in_snap = []
        if snap_file.exists():
            snap_img = tifffile.imread(snap_file)
            snap_tensor = torch.from_numpy(snap_img).to(torch.float32).to(torch_device)
            snap_masks, _, _ = model.eval(snap_tensor, normalize={"tile_norm_blocksize": 256})
            snap_binary = snap_masks > 0

            for cid in cell_ids:
                cell_mask = (masks == cid)
                if np.any(cell_mask & snap_binary):
                    cell_ids_in_snap.append(cid)

        print(f"Cells present in snapshot: {cell_ids_in_snap}")
  
        mask_path = dir / f"{f.stem}{masks_ext}"
        tifffile.imwrite(mask_path, masks.astype(np.uint16))
        print(f"Saved labeled mask: {mask_path}")
        H, W = masks.shape
        masks_3d = np.zeros((H, W, len(cell_ids)), dtype=np.uint8)
        for i, cid in enumerate(cell_ids):
            masks_3d[:, :, i] = (masks == cid).astype(np.uint8)
        
        mat_path = dir / f"{f.stem}_masks_3d.mat"
        sio.savemat(mat_path, {"masks_3d": masks_3d})
        print(f"Saved 3D mask .mat: {mat_path}")
        results.append({
            'file': f.name,
            'total_cells': len(cell_ids),
            'cells_in_snapshot': len(cell_ids_in_snap),
            'cell_ids_in_snapshot': ','.join(map(str, cell_ids_in_snap))
        })
    
        bg_mask = (masks == 0)
        mat_bg_path = dir / f"{f.stem}_bg.mat"
        sio.savemat(mat_bg_path, {"bg": bg_mask})
        print(f"Saved MATLAB background mask: {mat_bg_path}")

    # save results to csv
    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Done! Cell counts saved to {output_csv}. All masks saved in {dir}.")


# allows BOTH importing AND running directly
if __name__ == "__main__":
    merge_ome_to_multipage("/Users/sunny/Desktop/extra")