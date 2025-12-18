import numpy as np
import tifffile
from cellpose import models, core, io, plot
from pathlib import Path
from tqdm import trange
from natsort import natsorted
import pandas as pd
import scipy.io as sio

io.logger_setup() # run this to get printing of progress
model = models.CellposeModel()

# *** change to your google drive folder path ***
dir = Path("/Users/sunny/Desktop/Kelly_data/")
output_csv = dir / 'cell_counts.csv' 

image_ext = ".tif"
masks_ext = "_masks.tif"

# list all files
files = natsorted([f for f in dir.glob("*"+image_ext) if "_masks" not in f.name and "_flows" not in f.name])

results = []

for f in files:
  print(f.name)
  img = tifffile.imread(f)[0]
  masks, flows, styles = model.eval(img, normalize={"tile_norm_blocksize": 256})

  num_cells = len(np.unique(masks)) - 1  # subtract 1 for background
  results.append({'file': f.name, 'count': num_cells})
  
  mask_path = dir / f"{f.stem}{masks_ext}"
  tifffile.imwrite(mask_path, masks.astype(np.uint16))
  print("Saved labeled mask:", mask_path)

  cell_ids = np.unique(masks)
  cell_ids = cell_ids[cell_ids != 0]  # ignore background

  H, W = masks.shape
  masks_3d = np.zeros((H, W, len(cell_ids)), dtype=np.uint8)

  for i, cid in enumerate(cell_ids):
    masks_3d[:, :, i] = (masks == cid).astype(np.uint8)

  mat_path = dir / f"{f.stem}_masks_3d.mat"
  sio.savemat(mat_path, {"masks_3d": masks_3d})
  print("Saved 3D mask .mat:", mat_path)


# save results to csv
pd.DataFrame(results).to_csv(output_csv, index=False)
print('Done! Cell counts saved to', output_csv)