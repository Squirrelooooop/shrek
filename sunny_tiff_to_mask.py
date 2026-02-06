import numpy as np
import tifffile
from cellpose import models, core, io, plot
from pathlib import Path
from tqdm import trange
from natsort import natsorted
import pandas as pd
import scipy.io as sio
import torch

io.logger_setup() # run this to get printing of progres
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

# *** change to your google drive folder path ***
dir = Path("/Users/sunny/Desktop/20260126_IvanHEK_FLYChlorOn/multipage_tiff/ds/motion_corrected/")
snap = Path("/Users/sunny/Desktop/20260126_IvanHEK_FLYChlorOn/multipage_tiff/")
output_csv = dir / 'cell_counts.csv' 

image_ext = ".tif"
masks_ext = "_masks.tif"

# list all files
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
  print("Saved labeled mask:", mask_path)

  H, W = masks.shape
  masks_3d = np.zeros((H, W, len(cell_ids)), dtype=np.uint8)
  for i, cid in enumerate(cell_ids):
      masks_3d[:, :, i] = (masks == cid).astype(np.uint8)
  mat_path = dir / f"{f.stem}_masks_3d.mat"
  sio.savemat(mat_path, {"masks_3d": masks_3d})
  print("Saved 3D mask .mat:", mat_path)
  results.append({
        'file': f.name,
        'total_cells': len(cell_ids),
        'cells_in_snapshot': len(cell_ids_in_snap),
        'cell_ids_in_snapshot': ','.join(map(str, cell_ids_in_snap))
    })
    
  bg_mask = (masks == 0)
  mat_bg_path = dir / f"{f.stem}_bg.mat"
  sio.savemat(mat_bg_path, {"bg": bg_mask})
  print("Saved MATLAB background mask:", mat_bg_path)

# save results to csv
pd.DataFrame(results).to_csv(output_csv, index=False)
print('Done! Cell counts saved to', output_csv)