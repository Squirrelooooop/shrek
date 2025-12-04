import numpy as np
import tifffile
from cellpose import models, core, io, plot
from pathlib import Path
from tqdm import trange
from natsort import natsorted
import pandas as pd
import torch

io.logger_setup() # run this to get printing of progress
model = models.CellposeModel()

# *** change to your google drive folder path ***
dir = Path("/Users/sunny/Desktop/Data/20250522_HEK_Sono3/")
output_csv = dir / 'cell_counts.csv' 

image_ext = ".tif"
masks_ext = "_masks.tif"

if torch.backends.mps.is_available():
    device = "mps"
    print("Using Apple GPU (MPS) for Cellpose")
else:
    device = "cpu"
    print("Using CPU for Cellpose")

torch_device = torch.device(device)

# list all files
files = natsorted([f for f in dir.glob("*"+image_ext) if "_masks" not in f.name and "_flows" not in f.name])

results = []

for f in files:
  print(f.name)
  img = tifffile.imread(f)[0]
  img_tensor = torch.from_numpy(img).to(torch_device)
  masks, flows, styles = model.eval(img, normalize={"tile_norm_blocksize": 256})
  num_cells = len(np.unique(masks)) - 1  # subtract 1 for background
  results.append({'file': f.name, 'count': num_cells})
  
  print("saving masks")
  mask_path = dir / f"{f.stem}{masks_ext}"
  tifffile.imwrite(mask_path, masks.astype(np.uint16))

# save results to csv
pd.DataFrame(results).to_csv(output_csv, index=False)
print('Done! Cell counts saved to', output_csv)