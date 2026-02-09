import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tifffile as tiff
import caiman as cm
from caiman.motion_correction import MotionCorrect
import multiprocessing

def main():
    # ------------------------------
    # PARAMETERS
    # ------------------------------

    input_dir = Path("/Users/sunny/Desktop/extra/multipage_tiff/ds")
    output_dir = input_dir / "motion_corrected"
    output_dir.mkdir(exist_ok=True)

    shift_dir = output_dir / "shift"
    shift_dir.mkdir(exist_ok=True)  # make folder if it doesn't exist

    template_dir = output_dir / "template"
    template_dir.mkdir(exist_ok=True)  # make folder if it doesn't exist


    max_shifts = (60, 60)       # maximum rigid shift (pixels)
    strides = (48, 48)          # patch size stride for pw-rigid
    overlaps = (24, 24)         # patch overlap
    max_deviation_rigid = 3     # maximum deviation per patch
    pw_rigid = True            # rigid or piecewise-rigid
    shifts_opencv = True        # use bicubic interpolation
    border_nan = 'copy'         # replicate border values
    nonneg_movie = True         # ensure movie values stay >=0

    # ------------------------------
    # SETUP CLUSTER
    # ------------------------------
    n_cores = multiprocessing.cpu_count()      # total cores
    n_processes = max(1, n_cores - 2)          # leave 2 cores free
    print(f"Using {n_processes} processes for motion correction.")

    # pick a safe number of processes
    c, dview, _ = cm.cluster.setup_cluster(
        backend='multiprocessing',
        n_processes=n_processes,
        single_thread=False
    )
    print(f"Starting cluster with {n_processes} processes...")

    # ------------------------------
    # PROCESS EACH TIFF
    # ------------------------------

    tiff_files = sorted(f for f in input_dir.glob("*.tif") if "_snap" not in f.name)


    for fpath in tiff_files:
        print(f"Processing {fpath.name}...")

        fnames = [str(fpath)]
        mc = MotionCorrect(fnames, dview=dview,
                           max_shifts=max_shifts,
                           strides=strides,
                           overlaps=overlaps,
                           max_deviation_rigid=max_deviation_rigid,
                           shifts_opencv=shifts_opencv,
                           pw_rigid=pw_rigid,
                           border_nan=border_nan,
                           nonneg_movie=nonneg_movie)

        # Run motion correction
        mc.motion_correct(save_movie=True)

        # Load corrected movie
        m_corr = cm.load(mc.mmap_file)

        # Save corrected TIFF
        corrected_path = output_dir / f"{fpath.stem}_mc.tif"
        tiff.imwrite(str(corrected_path), m_corr.astype(np.float32))

        # Save shift plot
        shift_plot_path = shift_dir / f"{fpath.stem}_shifts.png"
        plt.figure(figsize=(10,5))
        plt.plot(mc.shifts_rig)
        plt.xlabel("Frame")
        plt.ylabel("Pixels")
        plt.legend(['x shifts','y shifts'])
        plt.title(f"Rigid shifts for {fpath.name}")
        plt.tight_layout()
        plt.savefig(shift_plot_path)
        plt.close()

        # Save correction template
        template_path = template_dir / f"{fpath.stem}_template.png"
        plt.figure(figsize=(8,8))
        plt.imshow(mc.total_template_rig, cmap='gray')
        plt.title(f"Template for {fpath.name}")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(template_path)
        plt.close()

        print(f"Finished {fpath.name}: corrected TIFF, shifts, template saved.")

    print("All files processed.")

    # ------------------------------
    # STOP CLUSTER
    # ------------------------------
    cm.stop_server(dview=dview)
    print("Cluster stopped.")


if __name__ == "__main__":
    main()
