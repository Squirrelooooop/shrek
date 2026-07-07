#!/usr/bin/env python3
from __future__ import annotations

# v4.84: fix recording-specific movie lookup; legend text 'response'; keep agreed plot/control/responder fixes.
# v4.83: display ΔF/F in plot text; fix stim3_to_drug classification; add control recording banner; keep ASCII filenames/CSV columns.
# v4.82: actual CellID heatmap ticks; remove clip text from plot titles; add outline legends; restore _cellIDs filenames.
# v4.81: classify no-pulse recordings as time-window controls; suppress US-responder interpretation in controls.
# v4.80: strict primary-responder logic for plots/reports; responder_either remains audit-only.
# v4.79: isolate each DCIMG in a temporary single-file folder before Bio-Formats conversion to prevent cross-file association.
# v4.78: DCIMG conversion writes to .partial.tif and renames only after success; deletes partial files on failure.
# v4.77: hard-stop if different DCIMG inputs produce identical converted TIFF outputs; write conversion hash manifest.
# v4.76: add recording isolation audit/fingerprints; split plotting try-blocks so one plot error does not skip others.
# v4.75: remove recursive savefig wrapper; restore direct fig.savefig while keeping plot fixes.
# v4.74: fix plot_label_contours keyword mismatch: linewidth -> linewidths.
# v4.73: fix Focus QC plot crash by passing responder_threshold; restores downstream composite plots.
# v4.72: plotting guard removes Cellpose background label 0 from labels/contours.
# v4.71: simple figure titles; no CellID title text; recording-isolated responder state; high-contrast QC outline.
# v4.70: inspection fixes for subprocess arg repair, montage title removal, and duplicate metric definitions.
# v4.69: hotfix recording_baseline_unstable NameError; tighten montage spacing; fix analysis_run_name subprocess argument.
# v4.64: logical QC fixes: automatic conservative QC is computed and plotted; no manual exclusions; primary responder outlines only.
# v4.62: condition-label-only titles for single plots, two-panel composites, and montages.
# v4.60: automatic conservative QC without manual cell exclusion; no multi-stimulus response requirement.
# v4.57: compact all montages; large condition labels; no filename titles; minimal whitespace; shared montage scales.
SCRIPT_VERSION = "dcimg_processing_v4_84_movie_path_and_agreed_plot_fixes"

"""
dcimg_processing_v4_27.py

Integrated calcium imaging preprocessing pipeline.

Purpose
-------
One script with selectable stages for:

1. DCIMG -> multipage TIFF
2. OME-TIFF folders -> multipage TIFF
3. Downsample multipage TIFFs using the uploaded Cellpose/torch logic
4. CaImAn motion correction using the uploaded sunny_caiman_motion_correction.py logic
5. Cellpose mask generation using the uploaded tiff_to_mask logic
6. MATLAB-style .mat -> .csv export based on savecsv.m logic

Important
---------
This is one script, but different stages require different environments.

Recommended stage environments:
    dcimg:
        stages: dcimg, ometiff
        deps: numpy, tifffile, jpype1

    cellpose_py310:
        stages: downsample, mask
        deps: torch, cellpose, natsort, pandas, scipy, tifffile, numpy

    caiman:
        stages: motion
        deps: caiman, matplotlib, tifffile, numpy

    MATLAB replacement CSV stage:
        stage: csv
        deps: scipy, numpy

Typical folder layout
---------------------
output_root/
    multipage_tiff/
        raw multipage TIFFs from DCIMG or OME-TIFF

    multipage_tiff/ds/
        downsampled TIFFs

    multipage_tiff/ds/motion_corrected/
        motion-corrected TIFFs, masks, .mat files, and CSVs

    multipage_tiff/ds/motion_corrected/shift/
        shift plots

    multipage_tiff/ds/motion_corrected/template/
        motion correction templates

Example one-command run
-----------------------
This automatically runs dcimg -> downsample -> motion -> mask -> calcium using the correct
conda environment for each stage:

    conda activate base

    python dcimg_processing_v4_26.py \
        --run_all \
        --input_format dcimg \
        --dcimg_dir "/Users/ncc/Data/DCIMG" \
        --output_root "/Users/ncc/Data/Experiment_001_processed" \
        --bioformats_jar "/Users/ncc/Tools/bioformats/bioformats_package.jar" \
        --ms_per_frame 60 \
        --ds_factor 10

Example individual stage runs
-----------------------------
Stage 1 in dcimg environment:

    conda activate dcimg

    python dcimg_processing_v4_26.py \\
        --stages dcimg \\
        --dcimg_dir "/Users/ncc/Data/DCIMG" \\
        --output_root "/Users/ncc/Data/Experiment_001_processed" \\
        --bioformats_jar "/Users/ncc/Tools/bioformats/bioformats_package.jar"

Stage 2 in cellpose_py310 environment:

    conda activate cellpose_py310

    python dcimg_processing_v4_26.py \\
        --stages downsample \\
        --output_root "/Users/ncc/Data/Experiment_001_processed" \\
        --ms_per_frame 60 \\
        --ds_factor 10

Stage 3 in caiman environment:

    conda activate caiman

    python dcimg_processing_v4_26.py \\
        --stages motion \\
        --output_root "/Users/ncc/Data/Experiment_001_processed"

Stage 4 in cellpose_py310 environment:

    conda activate cellpose_py310

    python dcimg_processing_v4_26.py \\
        --stages mask \\
        --output_root "/Users/ncc/Data/Experiment_001_processed" \\
        --snap_dir ""

Stage 5, CSV export from .mat outputs:

    python dcimg_processing_v4_26.py \\
        --stages csv \\
        --output_root "/Users/ncc/Data/Experiment_001_processed" \\
        --csv_variable F
"""

# =============================================================================
# CHANGE LOG
# =============================================================================
#
# dcimg_processing_v4_46_bash_readme_metadata
# ------------------------------
# - Adds a companion copy/paste bash README template for standard .mat timing
#   versus legacy manual approximate timing.
# - Writes run_settings_summary.csv/json with key responder/QC/heatmap/timing
#   parameters for easier provenance review.
#
#
# dcimg_processing_v4_45_explicit_responder_qc_params
# ------------------------------
# - Adds strict validation that responder/QC/heatmap parameters must be explicitly
#   present in the bash command for stim analysis. This prevents hidden default
#   thresholds from being used without provenance.
# - The required explicit flags include responder_threshold, noise_sd_multiplier,
#   spike-safe responder settings, focus QC thresholds, heatmap clipping,
#   max_data_regions, extra_stim_delay_sec, and recording_start_sec.
#
# dcimg_processing_v4_44_robust_heatmap_scaling
# ------------------------------
# - Adds robust percentile clipping for spatial heatmap color scales so a single extreme
#   cell, especially during drug response, does not dominate the color scale.
# - Colorbars use extend arrows and labels with <= / >= bounds to show clipped values.
# - Adds --heatmap_clip_low_percentile and --heatmap_clip_high_percentile.
#

# dcimg_processing_v4_51_separate_adjacent_cell_outlines
# - Draws ROI and responder outlines one Cellpose label at a time, so touching
#   cells with separate IDs no longer appear as one large merged outline.
#
# dcimg_processing_v4_49_responder_outline_uses_table_flag
# - Fixes responder outlines in spatial heatmaps and cell-frame composites.
# - Outlines now use the table responder call (`responder_primary`) rather than
#   re-thresholding `response_delta_middle90_vs_baseline`.
# - This prevents true responders with high spike-safe peak but modest middle90
#   mean delta from being omitted from outlines.

# dcimg_processing_v4_48_cyan_blue_outlines
# ------------------------------
# - Changes spatial/composite ROI outlines to non-red, non-white colors:
#   medium blue for all ROI boundaries and bright cyan for responders.
# - Replaces white failed-QC outlines with orange dotted outlines.
# - Uses a dark neutral figure background for baseline-vs-max pairs so the
#   inter-panel gutter does not appear as an odd white line.
#
# dcimg_processing_v4_42_threshold_outline_expanded_composites
# ------------------------------
# - Baseline-vs-max representative composites now outline cells whose region response
#   exceeds the responder threshold, matching the spatial DF/F plots.
# - Representative composite fill masks are expanded by ~10% around each ROI so nearby
#   same-frame background/context is visible while original ROI outlines remain traceable.
#
# dcimg_processing_v4_41_side_by_side_cellID_versions
# ------------------------------
# - Explicitly saves side-by-side baseline-vs-max cell-frame composites in two versions:
#   clean without cell IDs and labeled with _ suffix.
#
# dcimg_processing_v4_31_cell_shift_qc
# ------------------------------
# - Adds per-cell motion/shift QC during stim analysis.
# - Estimates recording-level global image shift between baseline and each region.
# - Estimates each cell's local ROI shift and reports local-minus-global shift.
# - Adds spatial heatmaps for cell-shift QC and table columns in cell_region_summary/QC.
#
# dcimg_processing_v4_35_spike_safe_responder
# ------------------------------
# - Adds spike-safe responder calling. Raw peak_DFoverF is still reported, but
#   responder calls use a rolling-median peak plus a minimum consecutive-frame
#   threshold so one-frame spikes do not create responder calls.
# - Adds spike_safe_peak_DFoverF, max_consecutive_*_frames, and responder-call
#   filter settings to by-cell and transition outputs.
#
# dcimg_processing_v4_36_spatial_heatmap_alignment
# ------------------------------
# - Fixes spatial heatmap alignment by using the actual movie/max-projection image with the same pixel grid as the Cellpose masks.
# - Avoids resized matplotlib template PNGs for overlays because those can include figure padding/axis scaling.
# - Raises default max_data_regions so final drug/drug_end region is not silently truncated.
#
# dcimg_processing_v4_34_middle90
# ------------------------------
# - Changes stimulus-region metric window from middle80 to middle90.
# - Region summary means now exclude only the first 5% and final 5% of each region.
# - Peak_DFoverF remains the raw maximum over the full region; spike removal is not applied by default.
#
# dcimg_processing_v4_40_region_cell_frame_composites
# ------------------------------
# - Adds per-region cell-filled representative image composites.
# - For each non-baseline region, saves baseline and max-expression composites using
#   the actual corrected movie frames on the same grid as the masks.
# - Each ROI is filled with pixels from the cell-specific relevant frame: baseline
#   representative frame or that cell's peak-expression frame for the region.
# - Saves labeled cell-ID versions and a CSV mapping each cell to the frames used.
#
# dcimg_processing_v4_39_expected_drug_boundary_fix
# ------------------------------
# - Fixes mixed detected/expected region markers so non-pulse drug boundary is retained.
# - If expected times are supplied, each expected marker is mapped to the nearest detected pulse when present; otherwise the expected time is kept.
# - This prevents standard stim1,stim2,stim3,drug runs from producing only three non-baseline spatial heatmaps.
# - Retains duplicate labeled spatial heatmaps with cell IDs for table traceability.
#
# dcimg_processing_v4_29_analysis_runs
# ------------------------------
# - Uses a single top-level analysis_runs/<timestamp>/ folder as the first fork.
# - Removes latest symlink/path pointer creation.
# - Writes/copies meaningful final CSV files to analysis_runs/<timestamp>/csv/.
# - Keeps baseline-responder fix from v4_27_baselinefix.
#
# dcimg_processing_v4_27_baselinefix
# ------------------------------
# - Derived from v4_27.
# - Fixes stim_response_thresholds: baseline/reference region is no longer eligible
#   to be called as first_response_region or ever_responder.
# - Baseline is still used for baseline_mean, baseline_sd, QC, and noise thresholding.
# - Adds is_baseline_reference_region to by-cell and transition outputs.
#
# dcimg_processing_v4_27
# ------------------------------
# - Stim stage now mirrors final stim CSV tables into output_root/csv.
# - Timestamped analysis-run copies are still preserved under calcium_csv/analysis_runs.
#
# dcimg_processing_v2
# ------------------------------
# - Integrates the uploaded workflow.py goals into one script.
# - Adds DCIMG -> multipage TIFF conversion using Bio-Formats.
# - Incorporates OME-TIFF -> multipage TIFF logic.
# - Incorporates torch temporal smoothing + downsampling logic.
# - Incorporates CaImAn motion correction logic.
# - Incorporates Cellpose max-projection mask generation logic.
# - Replaces savecsv.m with Python .mat -> .csv export.
# - Uses explicit --stages so the same script can be run in different envs.
#
# =============================================================================


# ---- v4.57 montage presentation defaults ----
PLOT_CONDITION_LABEL_FONTSIZE = 16
PLOT_CONDITION_LABEL_WEIGHT = "bold"
PLOT_MONTAGE_WSPACE = 0.001
PLOT_MONTAGE_HSPACE = 0.010
PLOT_MONTAGE_LEFT = 0.001
PLOT_MONTAGE_RIGHT = 0.999
PLOT_MONTAGE_BOTTOM = 0.001
PLOT_MONTAGE_TOP = 0.995
PLOT_TITLE_PAD = 1
PLOT_HIDE_MONTAGE_SPINES = True


import argparse
import os
import re
import sys
import subprocess
import json
import shutil
import csv
import tempfile
import hashlib
import shlex
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import tifffile

DELTA_F_OVER_F_LABEL = "ΔF/F"


def dff_display_text(text_value):
    """Replace displayed DF/F variants with ΔF/F. Do not use for filenames or CSV columns."""
    try:
        s = str(text_value)
        s = s.replace("ΔF/F", DELTA_F_OVER_F_LABEL)
        s = s.replace("ΔF/F", DELTA_F_OVER_F_LABEL)
        return s
    except Exception:
        return text_value


def _v77_file_sha256(path, chunk_size=1024 * 1024):
    """SHA256 hash of a file for duplicate-output detection."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk_size)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return ""



def _v79_make_isolated_dcimg_link(input_file: Path, temp_dir: Path) -> Path:
    """
    Place a single DCIMG file in an isolated temp folder for Bio-Formats.

    Bio-Formats/Hamamatsu DCIMG reader may inspect neighboring .dcimg files in the
    same source directory. Isolating each file prevents rec00006/rec00010 cross-file
    association.
    """
    isolated = temp_dir / input_file.name
    try:
        os.link(input_file, isolated)  # fast, no data copy when same filesystem
    except Exception:
        shutil.copy2(input_file, isolated)
    return isolated


def _v79_write_csv_rows_csvmodule(path: Path, rows: list[dict]) -> None:
    """Write manifest rows without pandas, because the dcimg env may not include pandas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

def _v76_file_sha256(path, chunk_size=1024 * 1024):
    """SHA256 hash of a file for provenance and duplicate-input detection."""
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk_size)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return ""


def _v76_array_sha256(arr):
    """SHA256 hash of numeric array contents after stable NaN handling."""
    try:
        a = np.asarray(arr, dtype=np.float64)
        a = np.nan_to_num(a, nan=-999999.123456789, posinf=999999.123456789, neginf=-999999.987654321)
        h = hashlib.sha256()
        h.update(str(a.shape).encode("utf-8"))
        h.update(np.ascontiguousarray(a).tobytes())
        return h.hexdigest()
    except Exception:
        return ""


def _v76_mean_trace_corr(a, b):
    """Correlation between two same-length mean traces."""
    try:
        x = np.asarray(a, dtype=float)
        y = np.asarray(b, dtype=float)
        n = min(x.size, y.size)
        if n < 3:
            return np.nan
        x = x[:n]
        y = y[:n]
        mask = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(mask) < 3:
            return np.nan
        x = x[mask]
        y = y[mask]
        if np.nanstd(x) == 0 or np.nanstd(y) == 0:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])
    except Exception:
        return np.nan


def _v472_positive_cell_ids(cell_ids):
    """Return sorted unique Cellpose cell IDs excluding background label 0."""
    out = []
    try:
        for x in list(cell_ids):
            try:
                xi = int(x)
                if xi > 0:
                    out.append(xi)
            except Exception:
                pass
    except Exception:
        return []
    return sorted(set(out))


def _v472_positive_mask_labels(mask):
    """Return sorted unique labels from a mask excluding background label 0."""
    try:
        return _v472_positive_cell_ids(np.unique(mask))
    except Exception:
        return []



def _v471_filter_paths_for_recording(paths, recording):
    """Keep only image paths belonging to the requested recording prefix."""
    try:
        rec = str(recording)
        return [p for p in paths if Path(p).name.startswith(rec + "_")]
    except Exception:
        return paths

def _v471_simple_plot_title_from_name(name):
    """Return presentation-friendly plot title from a file/function name."""
    s = str(name)
    low = s.lower()
    if "focus_artifact" in low or "_qc" in low or "qc_" in low:
        return "QC"
    if "spatial_dfoverf" in low or "dfoverf" in low or "deltaf" in low:
        return "ΔF/F"
    if "baseline_vs_max" in low or "baseline_vs_peak" in low or "cell_frame_composite" in low:
        return "Baseline vs Peak"
    if "baseline" in low:
        return "Baseline"
    return ""


def _v471_set_figure_title(fig, title):
    """Set a simple title above a figure without filenames or  text."""
    try:
        title = str(title).replace("", "").replace("", "").replace("", "").strip(" _-")
        if title:
            fig.suptitle(title, fontsize=24, fontweight="bold", y=0.995)
            try:
                fig.subplots_adjust(top=0.93)
            except Exception:
                pass
    except Exception:
        pass


def _v471_recording_cell_key(recording, cell_id):
    """Key cell state by recording and cell_id to avoid leakage between recordings."""
    try:
        return (str(recording), int(cell_id))
    except Exception:
        return (str(recording), cell_id)


def _v469_repair_analysis_run_name_args(cmd):
    """Defensively split corrupted analysis_run_name args before subprocess execution."""
    try:
        repaired = []
        for item in list(cmd):
            s = str(item)
            if s == "--analysis_run_name__USE_EXISTING_ANALYSIS_DIR__":
                repaired.extend(["--analysis_run_name", "__USE_EXISTING_ANALYSIS_DIR__"])
            elif s == "--analysis_run_name __USE_EXISTING_ANALYSIS_DIR__":
                repaired.extend(["--analysis_run_name", "__USE_EXISTING_ANALYSIS_DIR__"])
            elif s.startswith("--analysis_run_name__USE_EXISTING_ANALYSIS_DIR__"):
                repaired.extend(["--analysis_run_name", "__USE_EXISTING_ANALYSIS_DIR__"])
            elif s.startswith("--analysis_run_name ") and len(s.split(None, 1)) == 2:
                repaired.extend(["--analysis_run_name", s.split(None, 1)[1]])
            else:
                repaired.append(item)
        return repaired
    except Exception:
        return cmd


def _v466_recording_baseline_stats(rows, recording):
    """Compute whole-recording baseline instability statistics from existing cell rows."""
    vals = []
    for r in rows:
        try:
            if str(r.get("recording")) != str(recording):
                continue
            if not bool(r.get("is_baseline_reference_region", False)):
                continue
            v = float(r.get("baseline_sd_DFoverF", np.nan))
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return {
            "recording_median_baseline_sd": np.nan,
            "recording_high_noise_fraction": np.nan,
            "n_recording_baseline_cells_for_stability": 0,
        }
    try:
        high_thr = float(getattr(_v466_recording_baseline_stats, "high_noise_sd_threshold", 0.10))
    except Exception:
        high_thr = 0.10
    return {
        "recording_median_baseline_sd": float(np.nanmedian(arr)),
        "recording_high_noise_fraction": float(np.mean(arr > high_thr)),
        "n_recording_baseline_cells_for_stability": int(arr.size),
    }


def _v465_bool_from_value(x):
    """Robust bool parser for table values."""
    try:
        if isinstance(x, str):
            return x.strip().lower() in ("true", "1", "yes", "y", "t")
        return bool(x)
    except Exception:
        return False


def _v465_row_plot_qc_risk(row):
    """Combined QC risk used for orange/dotted plot outlines."""
    for key in (
        "plot_qc_risk",
        "conservative_qc_risk",
        "conservative_focus_corr_risk",
        "conservative_baseline_instability_risk",
        "conservative_area_outlier_risk",
        "recording_baseline_unstable",
        "floating_cell_artifact_risk",
        "focus_instability_suspicious",
    ):
        try:
            if hasattr(row, "get") and key in row and _v465_bool_from_value(row.get(key, False)):
                return True
        except Exception:
            pass
    return False


def _v465_add_plot_qc_risk_column(df):
    """Add plot_qc_risk from all QC-risk columns present in a DataFrame."""
    try:
        risk = None
        for col in (
            "conservative_qc_risk",
            "conservative_focus_corr_risk",
            "conservative_baseline_instability_risk",
            "conservative_area_outlier_risk",
            "recording_baseline_unstable",
            "floating_cell_artifact_risk",
            "focus_instability_suspicious",
        ):
            if col in df.columns:
                vals = df[col].map(_v465_bool_from_value)
                risk = vals if risk is None else (risk | vals)
        df["plot_qc_risk"] = False if risk is None else risk
    except Exception:
        try:
            df["plot_qc_risk"] = False
        except Exception:
            pass
    return df


def _v465_compact_figure(fig, axes=None, *, top=0.975, bottom=0.005, left=0.005, right=0.985, wspace=0.005, hspace=0.035):
    """Very tight montage layout for presentation slides."""
    try:
        fig.subplots_adjust(left=0.001, right=0.999, bottom=0.001, top=0.995, wspace=0.001, hspace=0.010)
    except Exception:
        pass
    if axes is not None:
        try:
            flat = axes.ravel() if hasattr(axes, "ravel") else axes
            for ax in flat:
                ax.set_xticks([])
                ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
        except Exception:
            pass


def _v462_plot_condition_label(region_index=None, region_label=None, plot_region_labels=None):
    """
    Return the user-facing condition label for plots.
    For non-baseline regions, plot_region_labels maps in order:
      region 2 -> label[0], region 3 -> label[1], region 4 -> label[2], region 5 -> label[3].
    Falls back to region_label only if user labels are absent.
    """
    labels = []
    try:
        if plot_region_labels is not None:
            labels = [x.strip() for x in str(plot_region_labels).split(",") if x.strip()]
    except Exception:
        labels = []
    try:
        ri = int(region_index)
        idx = ri - 2
        if labels and 0 <= idx < len(labels):
            return labels[idx]
    except Exception:
        pass
    if region_label is not None and str(region_label).strip():
        return str(region_label).strip()
    return ""


def _v460_safe_float_local(x, default=np.nan):
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _v460_median_of_rows(rows, key, recording=None, baseline_only=True):
    vals = []
    for r in rows:
        try:
            if recording is not None and str(r.get("recording")) != str(recording):
                continue
            if baseline_only and not bool(r.get("is_baseline_reference_region", False)):
                continue
            v = _v460_safe_float_local(r.get(key, np.nan))
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    return float(np.nanmedian(vals)) if vals else np.nan


def _v460_percentile_of_rows(rows, key, pct, recording=None, baseline_only=True):
    vals = []
    for r in rows:
        try:
            if recording is not None and str(r.get("recording")) != str(recording):
                continue
            if baseline_only and not bool(r.get("is_baseline_reference_region", False)):
                continue
            v = _v460_safe_float_local(r.get(key, np.nan))
            if np.isfinite(v):
                vals.append(v)
        except Exception:
            pass
    return float(np.nanpercentile(vals, pct)) if vals else np.nan



def _v458_parse_int_set_csv(s):
    """Parse comma-separated integer IDs into a set."""
    out = set()
    if s is None:
        return out
    for part in str(s).replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except Exception:
            pass
    return out


def _v458_recording_median_baseline_sd(by_cell_rows, recording):
    """Median baseline_sd_DFoverF among baseline rows for this recording."""
    vals = []
    for r in by_cell_rows:
        try:
            if str(r.get("recording")) == str(recording) and bool(r.get("is_baseline_reference_region", False)):
                v = float(r.get("baseline_sd_DFoverF", float("nan")))
                if np.isfinite(v):
                    vals.append(v)
        except Exception:
            pass
    if not vals:
        return float("nan")
    return float(np.nanmedian(vals))


def _v458_percentile_bounds(values, low_pct, high_pct):
    arr = np.asarray([float(v) for v in values if np.isfinite(float(v))], dtype=float) if values else np.asarray([], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(np.nanpercentile(arr, low_pct)), float(np.nanpercentile(arr, high_pct))

def _v457_compact_montage_axes(fig, axes, top=0.92):
    """Apply compact publication-style montage spacing."""
    try:
        fig.subplots_adjust(left=0.001, right=0.999, bottom=0.001, top=0.995, wspace=0.001, hspace=0.010)
    except Exception:
        pass
    try:
        flat_axes = axes.ravel() if hasattr(axes, "ravel") else axes
    except Exception:
        flat_axes = axes
    try:
        for ax in flat_axes:
            ax.set_xticks([])
            ax.set_yticks([])
            if PLOT_HIDE_MONTAGE_SPINES:
                for spine in ax.spines.values():
                    spine.set_visible(False)
    except Exception:
        pass


def _v457_simple_title(ax, label):
    """Set a simple large condition label without filenames."""
    try:
        ax.set_title(
            str(label),
            fontsize=PLOT_CONDITION_LABEL_FONTSIZE,
            fontweight=PLOT_CONDITION_LABEL_WEIGHT,
            pad=PLOT_TITLE_PAD,
        )
    except Exception:
        try:
            ax.set_title(str(label))
        except Exception:
            pass



# =============================================================================
# DEFAULTS
# =============================================================================

DEFAULT_BIOFORMATS_JAR = "~/Tools/bioformats/bioformats_package.jar"

DEFAULT_MS_PER_FRAME = 60.0
DEFAULT_DS_FACTOR = 10

DEFAULT_MAX_SHIFTS = (60, 60)
DEFAULT_STRIDES = (48, 48)
DEFAULT_OVERLAPS = (24, 24)
DEFAULT_MAX_DEVIATION_RIGID = 3
DEFAULT_PW_RIGID = True
DEFAULT_SHIFTS_OPENCV = True
DEFAULT_BORDER_NAN = "copy"
DEFAULT_NONNEG_MOVIE = True


# =============================================================================
# GENERAL HELPERS
# =============================================================================



class TeeStream:
    """Write stdout/stderr to both terminal and a log file."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass

    def isatty(self):
        try:
            return any(getattr(s, "isatty", lambda: False)() for s in self.streams)
        except Exception:
            return False


def setup_stage_logging(output_root: Path, stage_names=None):
    """
    Tee stdout/stderr into output_root/logs for this process.
    Each orchestrated stage writes its own log file.
    """
    try:
        logs_dir = Path(output_root) / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        if stage_names:
            safe_stage = "_".join(str(s) for s in stage_names)
        else:
            safe_stage = "run"
        safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", safe_stage).strip("_") or "run"

        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_path = logs_dir / f"{safe_stage}_{timestamp}.log"

        log_file = open(log_path, "a", buffering=1, encoding="utf-8", errors="replace")
        sys.stdout = TeeStream(sys.__stdout__, log_file)
        sys.stderr = TeeStream(sys.__stderr__, log_file)

        print(f"[LOG] Writing stage log to: {log_path}")
        return log_file, log_path
    except Exception as exc:
        try:
            print(f"[WARNING] Could not initialize log file: {exc}", file=sys.__stderr__)
        except Exception:
            pass
        return None, None


def write_command_used_file(output_root: Path, argv=None) -> None:
    """Save the exact Python command used for the current process."""
    try:
        commands_dir = Path(output_root) / "commands"
        commands_dir.mkdir(parents=True, exist_ok=True)
        args = sys.argv if argv is None else argv
        try:
            command_text = " ".join(shlex.quote(str(x)) for x in args)
        except Exception:
            command_text = " ".join(str(x) for x in args)
        path = commands_dir / "command_used.sh"
        path.write_text(command_text + "\n")
        print(f"[PROVENANCE] Saved command used: {path}")
    except Exception as exc:
        print(f"[WARNING] Could not save command_used.sh: {exc}")



def expand_path(path_value: str | Path | None) -> Path | None:
    """
    Expand ~ and environment variables and normalize common macOS path mistakes.

    Important:
    A path like:
        Users/ncc/Data/DCIMG

    is missing the leading slash. Without this correction, Python treats it as
    relative to the current folder and turns it into something like:
        /Users/ncc/Tools/CaImageAnalysis/Users/ncc/Data/DCIMG

    This function converts:
        Users/ncc/...
    to:
        /Users/ncc/...
    """
    if path_value is None:
        return None

    raw = str(path_value).strip()

    if raw == "":
        return None

    raw = os.path.expandvars(raw)

    # Fix common macOS mistake: "Users/name/..." instead of "/Users/name/..."
    if raw.startswith("Users/"):
        raw = "/" + raw

    return Path(raw).expanduser().resolve()


def require_file(path: Path, label: str) -> None:
    """Require existing file."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n    {path}")


def require_dir(path: Path, label: str) -> None:
    """Require existing directory."""
    if not path.exists():
        raise FileNotFoundError(f"{label} not found:\n    {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{label} is not a directory:\n    {path}")



def report_optional_dependencies() -> None:
    """Print optional dependency status once per stage."""
    optional = ["imageio", "matplotlib", "h5py", "openpyxl"]
    missing = []
    for name in optional:
        try:
            __import__(name)
        except Exception:
            missing.append(name)
    if missing:
        print(f"Optional packages missing: {', '.join(missing)}")
        print("  Missing packages only affect optional QC outputs or MATLAB v7.3 fallback.")
        print("  Suggested install:")
        print("    conda install -n cellpose_py310 -c conda-forge imageio imageio-ffmpeg matplotlib h5py -y")
        print()


def print_header() -> None:
    """Print run header."""
    print("\n" + "=" * 92)
    print(SCRIPT_VERSION)
    print("=" * 92)
    print(f"Python executable: {sys.executable}")
    print(f"Current directory: {Path.cwd()}")
    print("=" * 92)


def make_folders(output_root: Path) -> dict[str, Path]:
    """Create standard output folders."""
    output_root = output_root.resolve()

    folders = {
        "output_root": output_root,
        "multipage_tiff": output_root / "multipage_tiff",
        "downsampled": output_root / "multipage_tiff" / "ds",
        "motion_corrected": output_root / "multipage_tiff" / "ds" / "motion_corrected",
        "shift": output_root / "multipage_tiff" / "ds" / "motion_corrected" / "shift",
        "template": output_root / "multipage_tiff" / "ds" / "motion_corrected" / "template",
        "csv": output_root / "csv",
        "logs": output_root / "logs",
    }

    for folder in folders.values():
        folder.mkdir(parents=True, exist_ok=True)

    return folders


def print_folders(folders: dict[str, Path]) -> None:
    """Print folder layout."""
    print("\nFolders:")
    for key, value in folders.items():
        print(f"  {key:20s} {value}")


def parse_tuple2(value: str) -> tuple[int, int]:
    """Parse an integer tuple from '60,60'."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Expected format like 60,60")
    return int(parts[0]), int(parts[1])


# =============================================================================
# STAGE 1A: DCIMG -> MULTIPAGE TIFF
# =============================================================================


def start_bioformats_jvm(bioformats_jar: Path) -> None:
    """Start JVM with Bio-Formats."""
    import jpype

    require_file(bioformats_jar, "Bio-Formats JAR")

    if not jpype.isJVMStarted():
        print(f"\nStarting JVM with Bio-Formats JAR:\n    {bioformats_jar}")
        jpype.startJVM(classpath=[str(bioformats_jar)])


def bioformats_pixel_dtype(reader) -> np.dtype:
    """Map Bio-Formats pixel type to NumPy dtype."""
    import jpype

    FormatTools = jpype.JClass("loci.formats.FormatTools")

    pixel_type = reader.getPixelType()
    little_endian = bool(reader.isLittleEndian())
    endian = "<" if little_endian else ">"

    if pixel_type == FormatTools.UINT8:
        return np.dtype("uint8")
    if pixel_type == FormatTools.INT8:
        return np.dtype("int8")
    if pixel_type == FormatTools.UINT16:
        return np.dtype(endian + "u2")
    if pixel_type == FormatTools.INT16:
        return np.dtype(endian + "i2")
    if pixel_type == FormatTools.UINT32:
        return np.dtype(endian + "u4")
    if pixel_type == FormatTools.INT32:
        return np.dtype(endian + "i4")
    if pixel_type == FormatTools.FLOAT:
        return np.dtype(endian + "f4")
    if pixel_type == FormatTools.DOUBLE:
        return np.dtype(endian + "f8")

    raise ValueError(f"Unsupported Bio-Formats pixel type: {pixel_type}")


def find_dcimg_files(dcimg_dir: Path, recursive: bool = False) -> list[Path]:
    """Find DCIMG files."""
    require_dir(dcimg_dir, "DCIMG input directory")
    if recursive:
        return sorted(dcimg_dir.rglob("*.dcimg"))
    return sorted(dcimg_dir.glob("*.dcimg"))


def convert_one_dcimg_to_tiff(
    input_file: Path,
    output_file: Path,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    """Convert one DCIMG file to one multipage BigTIFF using isolated Bio-Formats input.

    Safety behavior:
      - place only this DCIMG in a temporary folder before reader.setId()
      - write to <output>.partial.tif first
      - delete partial output if conversion fails
      - rename to final output only after the full movie is written
    """
    import jpype

    require_file(input_file, "Input DCIMG file")

    if output_file.exists() and not overwrite:
        print(f"[SKIP] Output already exists: {output_file}")
        print("       Existing converted TIFF will be reused. Use --overwrite or delete the multipage_tiff folder to force reconversion.")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    tmp_output_file = output_file.with_name(output_file.stem + ".partial.tif")

    if dry_run:
        print(f"[DRY RUN] Would convert:\\n    {input_file}\\n -> {output_file}")
        return

    if tmp_output_file.exists():
        try:
            tmp_output_file.unlink()
            print(f"[CLEANUP] Removed stale partial TIFF: {tmp_output_file}")
        except Exception as exc:
            raise RuntimeError(f"Could not remove stale partial TIFF {tmp_output_file}: {exc}") from exc

    ImageReader = jpype.JClass("loci.formats.ImageReader")
    reader = ImageReader()

    isolated_root = output_file.parent / "_dcimg_isolated_inputs"
    isolated_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f"{input_file.stem}_", dir=str(isolated_root)) as td:
        temp_dir = Path(td)
        isolated_input_file = _v79_make_isolated_dcimg_link(input_file, temp_dir)

        try:
            print(f"        Isolated Bio-Formats input: {isolated_input_file}")
            reader.setId(str(isolated_input_file))

            size_x = int(reader.getSizeX())
            size_y = int(reader.getSizeY())
            n_frames = int(reader.getImageCount())
            dtype = bioformats_pixel_dtype(reader)
            image_format = str(reader.getFormat())

            bytes_per_frame = int(size_x) * int(size_y) * np.dtype(dtype).itemsize
            estimated_bytes = int(bytes_per_frame) * int(n_frames)
            try:
                usage = shutil.disk_usage(output_file.parent)
                free_bytes = int(usage.free)
            except Exception:
                free_bytes = -1

            print(f"\\n[DCIMG] {input_file.name}")
            print(f"        Format: {image_format}")
            print(f"        Size:   {size_x} x {size_y}")
            print(f"        Frames: {n_frames}")
            print(f"        Dtype:  {dtype}")
            print(f"        Output: {output_file}")
            print(f"        Temp:   {tmp_output_file}")
            print(f"        Estimated raw TIFF payload: {estimated_bytes / (1024**3):.2f} GB")
            if free_bytes >= 0:
                print(f"        Free space on output volume: {free_bytes / (1024**3):.2f} GB")
                if free_bytes < estimated_bytes * 1.25:
                    raise RuntimeError(
                        f"Insufficient free disk space for conversion. Need at least ~{estimated_bytes * 1.25 / (1024**3):.2f} GB "
                        f"free for this output, but only {free_bytes / (1024**3):.2f} GB is available."
                    )

            with tifffile.TiffWriter(str(tmp_output_file), bigtiff=True) as writer:
                for i in range(n_frames):
                    raw = reader.openBytes(i)
                    img = np.frombuffer(bytes(raw), dtype=dtype).reshape(size_y, size_x)

                    if img.dtype.byteorder not in ("=", "|"):
                        img = img.astype(img.dtype.newbyteorder("="), copy=False)

                    writer.write(img, photometric="minisblack", metadata=None)

                    if (i + 1) % 100 == 0 or (i + 1) == n_frames:
                        print(f"        Wrote frame {i + 1}/{n_frames}")

            if output_file.exists():
                output_file.unlink()
            tmp_output_file.rename(output_file)
            print(f"[DONE]  Saved: {output_file}")

        except Exception:
            try:
                if tmp_output_file.exists():
                    tmp_output_file.unlink()
                    print(f"[CLEANUP] Removed failed partial TIFF: {tmp_output_file}")
            except Exception as cleanup_exc:
                print(f"[WARNING] Could not remove failed partial TIFF {tmp_output_file}: {cleanup_exc}")
            raise

        finally:
            reader.close()




def stage_dcimg(
    dcimg_dir: Path,
    output_dir: Path,
    bioformats_jar: Path,
    overwrite: bool,
    recursive: bool,
    dry_run: bool,
) -> None:
    """Run DCIMG -> multipage TIFF conversion with duplicate-output protection."""
    files = find_dcimg_files(dcimg_dir, recursive=recursive)

    if not files:
        print(f"[WARNING] No .dcimg files found in: {dcimg_dir}")
        return

    print(f"\nFound {len(files)} DCIMG file(s).")
    print(f"Input:  {dcimg_dir}")
    print(f"Output: {output_dir}")

    if not dry_run:
        start_bioformats_jvm(bioformats_jar)

    manifest_rows = []
    for f in files:
        if recursive:
            rel = f.relative_to(dcimg_dir)
            out = output_dir / rel.with_suffix(".tif")
        else:
            out = output_dir / f"{f.stem}.tif"

        input_sha = _v77_file_sha256(f)
        preexisting = out.exists()

        convert_one_dcimg_to_tiff(
            input_file=f,
            output_file=out,
            overwrite=overwrite,
            dry_run=dry_run,
        )

        output_sha = _v77_file_sha256(out) if out.exists() else ""
        manifest_rows.append(
            {
                "input_file": str(f),
                "input_name": f.name,
                "input_sha256": input_sha,
                "output_file": str(out),
                "output_name": out.name,
                "output_sha256": output_sha,
                "output_preexisted_before_stage": bool(preexisting),
                "overwrite": bool(overwrite),
                "dry_run": bool(dry_run),
            }
        )

    if dry_run:
        return

    # Save conversion manifest for provenance.
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        _v79_write_csv_rows_csvmodule(output_dir / "dcimg_conversion_hash_manifest.csv", manifest_rows)
        print(f"Saved DCIMG conversion hash manifest: {output_dir / 'dcimg_conversion_hash_manifest.csv'}")
    except Exception as exc:
        print(f"[WARNING] Could not write DCIMG conversion hash manifest: {exc}")

    # Hard-stop if different input files produced identical output TIFFs.
    # This catches exactly the rec00006/rec00010 failure mode.
    output_hash_map = {}
    for row in manifest_rows:
        out_sha = row.get("output_sha256", "")
        if not out_sha:
            continue
        output_hash_map.setdefault(out_sha, []).append(row)

    duplicate_groups = []
    for out_sha, rows in output_hash_map.items():
        if len(rows) <= 1:
            continue
        input_hashes = {r.get("input_sha256", "") for r in rows}
        input_names = [r.get("input_name", "") for r in rows]
        output_names = [r.get("output_name", "") for r in rows]
        if len(input_hashes) > 1:
            duplicate_groups.append((out_sha, input_names, output_names))

    if duplicate_groups:
        msg_lines = [
            "Different DCIMG input files produced byte-identical converted TIFF outputs.",
            "This indicates a conversion/output reuse problem or stale duplicate TIFFs.",
            "Delete the affected multipage_tiff folder or rerun with --overwrite after fixing conversion.",
            "",
            "Duplicate groups:",
        ]
        for out_sha, input_names, output_names in duplicate_groups:
            msg_lines.append(f"  output_sha256={out_sha}")
            msg_lines.append(f"    input_files:  {', '.join(input_names)}")
            msg_lines.append(f"    output_files: {', '.join(output_names)}")
        raise RuntimeError("\n".join(msg_lines))



# =============================================================================
# STAGE 1B: OME-TIFF -> MULTIPAGE TIFF
# =============================================================================


def ome_sort_key(path: Path) -> int:
    """Sort OME-TIFF files using _Default_N.ome.tif ordering."""
    name = path.name
    m = re.search(r"_Default_(\d+)\.ome\.tif$", name)
    if m:
        return int(m.group(1))
    if name.endswith("_Default.ome.tif"):
        return 0
    return 9999


def stage_ometiff(input_dir: Path, output_dir: Path, overwrite: bool, dry_run: bool) -> None:
    """Merge OME-TIFF files inside each subfolder into one multipage TIFF."""
    require_dir(input_dir, "OME-TIFF parent directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    subfolders = sorted(p for p in input_dir.iterdir() if p.is_dir())

    if not subfolders:
        print(f"[WARNING] No subfolders found in: {input_dir}")
        return

    for subfolder in subfolders:
        ome_files = sorted(subfolder.glob("*.ome.tif"), key=ome_sort_key)

        if not ome_files:
            continue

        out_file = output_dir / f"{subfolder.name}.tif"

        if out_file.exists() and not overwrite:
            print(f"[SKIP] Output exists: {out_file}")
            continue

        print(f"\n[OME] Processing folder: {subfolder.name}")
        print(f"      Found {len(ome_files)} OME-TIFF file(s).")
        print(f"      Output: {out_file}")

        if dry_run:
            continue

        page_count = 0
        with tifffile.TiffWriter(out_file, bigtiff=True) as writer:
            for ome_file in ome_files:
                print(f"      Reading: {ome_file.name}")
                with tifffile.TiffFile(ome_file) as src:
                    for page in src.pages:
                        writer.write(page.asarray(), photometric="minisblack")
                        page_count += 1

        print(f"[DONE] Saved {out_file.name} with {page_count} pages.")

    print(f"\nAll OME folders processed. Merged files saved to: {output_dir}")


# =============================================================================
# STAGE 2: DOWNSAMPLE
# =============================================================================


def choose_torch_device(label: str = ""):
    """Choose MPS, CUDA, or CPU."""
    import torch

    if torch.backends.mps.is_available():
        print(f"Using Apple GPU (MPS){label}")
        return torch.device("mps")
    if torch.cuda.is_available():
        print(f"Using CUDA GPU{label}")
        return torch.device("cuda")

    print(f"Using CPU{label}")
    return torch.device("cpu")


def smooth_time_torch(data: np.ndarray, window_size: int, chunk: int = 200, device=None) -> np.ndarray:
    """Smooth a 3D movie along time using torch conv1d."""
    import torch
    import torch.nn.functional as F

    if device is None:
        device = torch.device("cpu")

    t, h, w = data.shape
    out = np.empty_like(data, dtype=np.float32)

    pad = window_size // 2
    kernel = torch.ones(1, 1, window_size, device=device) / window_size

    print(f"Smoothing on device: {device}")

    for start in range(0, t, chunk):
        end = min(start + chunk, t)

        x = torch.from_numpy(data[start:end]).to(device).float()
        x = x.permute(1, 2, 0).reshape(-1, 1, x.shape[0])
        x = F.pad(x, (pad, pad), mode="replicate")

        y = F.conv1d(x, kernel, padding=0)
        y = y.reshape(h, w, -1).permute(2, 0, 1)

        out[start:end] = y[: end - start].cpu().numpy()

    return out


def stage_downsample(
    input_dir: Path,
    output_dir: Path,
    ms_per_frame: float,
    ds_factor: int,
    overwrite: bool,
    dry_run: bool,
) -> None:
    """
    Memory-safe downsampling for multipage TIFF files.

    Original uploaded downsample logic loaded the entire TIFF stack into RAM:

        data = np.stack([page.asarray() for page in tif.pages], axis=0)

    That can be killed by macOS with return code 137 for large movies.

    This replacement computes the sampled rolling mean frame-by-frame:

        original logic:
            smoothed = rolling_mean(movie, window_size)
            output = smoothed[::ds_factor]

        memory-safe logic:
            for each output index i = 0, ds_factor, 2*ds_factor...
                read only the local rolling window around frame i
                average that window
                write one output frame

    This avoids loading the entire movie into memory. It is slower than the
    torch/GPU version but much safer for large calcium movies.
    """
    require_dir(input_dir, "Downsample input directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    if ms_per_frame <= 0:
        raise ValueError("ms_per_frame must be > 0.")
    if ds_factor < 1:
        raise ValueError("ds_factor must be >= 1.")

    window_size = int(round(1000 / ms_per_frame))
    if window_size < 1:
        window_size = 1

    pad = window_size // 2

    files = sorted(f for f in input_dir.glob("*.tif*") if "_snap" not in f.name)
    print(f"\nFound {len(files)} valid TIFF file(s) for memory-safe downsampling.")
    print(f"Rolling mean window_size: {window_size} frame(s)")
    print(f"Downsample factor:        {ds_factor}")

    if not files:
        return

    for file_path in files:
        out_name = output_dir / file_path.name.replace("merged", "processed")

        if out_name.exists() and not overwrite:
            print(f"[SKIP] Output exists: {out_name}")
            continue

        print(f"\n[DOWNSAMPLE] {file_path.name}")
        print(f"             Input:       {file_path}")
        print(f"             Output:      {out_name}")
        print(f"             msPerFrame:  {ms_per_frame}")
        print(f"             window_size: {window_size}")
        print(f"             ds_factor:   {ds_factor}")

        if dry_run:
            continue

        with tifffile.TiffFile(file_path) as tif:
            n_frames = len(tif.pages)

            if n_frames == 0:
                print(f"[SKIP] No frames found in {file_path}")
                continue

            first = tif.pages[0].asarray()
            input_dtype = first.dtype

            output_indices = range(0, n_frames, ds_factor)
            n_out = len(range(0, n_frames, ds_factor))

            print(f"             Input frames:  {n_frames}")
            print(f"             Output frames: {n_out}")

            with tifffile.TiffWriter(out_name, bigtiff=True) as writer:
                for out_i, center_idx in enumerate(output_indices, start=1):
                    # Replicate-pad boundary behavior by clamping frame indices.
                    idxs = [
                        min(max(j, 0), n_frames - 1)
                        for j in range(center_idx - pad, center_idx - pad + window_size)
                    ]

                    acc = None
                    for frame_idx in idxs:
                        frame = tif.pages[frame_idx].asarray().astype(np.float32, copy=False)
                        if acc is None:
                            acc = np.zeros_like(frame, dtype=np.float32)
                        acc += frame

                    averaged = acc / float(len(idxs))

                    if np.issubdtype(input_dtype, np.integer):
                        info = np.iinfo(input_dtype)
                        averaged = np.rint(averaged)
                        averaged = np.clip(averaged, info.min, info.max)

                    out_frame = averaged.astype(input_dtype, copy=False)
                    writer.write(out_frame, photometric="minisblack", metadata=None)

                    if out_i % 50 == 0 or out_i == n_out:
                        print(f"             Wrote downsampled frame {out_i}/{n_out}")

        print(f"Saved: {out_name}")

    print(f"\nDownsampled files saved to: {output_dir}")


# =============================================================================
# STAGE 3: CAIMAN MOTION CORRECTION
# =============================================================================




def normalize_frame_to_uint8(frame: np.ndarray, p_low: float = 1.0, p_high: float = 99.5) -> np.ndarray:
    """Robustly scale a frame to uint8 for preview movie output."""
    frame = np.asarray(frame, dtype=np.float32)
    finite = frame[np.isfinite(frame)]
    if finite.size == 0:
        return np.zeros(frame.shape, dtype=np.uint8)

    lo, hi = np.percentile(finite, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.nanmin(finite))
        hi = float(np.nanmax(finite))
        if hi <= lo:
            return np.zeros(frame.shape, dtype=np.uint8)

    out = (frame - lo) / (hi - lo)
    out = np.clip(out, 0, 1)
    return (out * 255).astype(np.uint8)


def downscale_frame_mean(frame: np.ndarray, factor: int) -> np.ndarray:
    """Downscale a 2D frame by block averaging."""
    if factor <= 1:
        return frame

    h, w = frame.shape[:2]
    h2 = (h // factor) * factor
    w2 = (w // factor) * factor

    if h2 < factor or w2 < factor:
        return frame

    cropped = frame[:h2, :w2]
    return cropped.reshape(h2 // factor, factor, w2 // factor, factor).mean(axis=(1, 3))


def write_lowres_preview_movie(
    tiff_path: Path,
    output_mp4: Path,
    downscale: int = 4,
    fps: float = 10.0,
    max_frames: int = 1200,
) -> None:
    """
    Optional low-resolution preview movie.

    This function must never break motion correction. It tries:
      1) MP4 via imageio, if imageio is installed
      2) GIF via imageio, if MP4 fails
      3) frame-contact-sheet PNG via tifffile/matplotlib fallback

    If all preview outputs fail, it prints a warning and returns.
    """
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    output_gif = output_mp4.with_suffix(".gif")
    output_png = output_mp4.with_suffix(".png")

    frames_rgb = []

    try:
        with tifffile.TiffFile(tiff_path) as tif:
            n_frames = len(tif.pages)
            if n_frames == 0:
                print(f"[WARNING] No frames in TIFF; skipping preview: {tiff_path}")
                return

            if max_frames is not None and max_frames > 0 and n_frames > max_frames:
                frame_indices = np.linspace(0, n_frames - 1, max_frames).astype(int)
            else:
                frame_indices = np.arange(n_frames)

            print(f"Creating preview movie: {output_mp4}")
            print(f"  Source frames:  {n_frames}")
            print(f"  Preview frames: {len(frame_indices)}")
            print(f"  Downscale:      {downscale}x")
            print(f"  FPS:            {fps}")

            for i, frame_idx in enumerate(frame_indices, start=1):
                frame = tif.pages[int(frame_idx)].asarray()
                small = downscale_frame_mean(frame, downscale)
                u8 = normalize_frame_to_uint8(small)
                rgb = np.repeat(u8[:, :, None], 3, axis=2)
                frames_rgb.append(rgb)

                if i % 100 == 0 or i == len(frame_indices):
                    print(f"  Prepared preview frame {i}/{len(frame_indices)}")
    except Exception as exc:
        print(f"[WARNING] Could not prepare preview frames for {tiff_path.name}: {exc}")
        return

    # Try imageio MP4/GIF.
    try:
        import imageio.v2 as imageio

        try:
            imageio.mimsave(output_mp4, frames_rgb, fps=fps)
            if output_mp4.exists() and output_mp4.stat().st_size > 0:
                size_mb = output_mp4.stat().st_size / (1024 * 1024)
                print(f"Saved preview movie: {output_mp4}")
                print(f"Preview movie size: {size_mb:.2f} MB")
                return
            raise RuntimeError("MP4 writer finished but output file was empty or missing.")
        except Exception as exc:
            print(f"[WARNING] MP4 preview failed: {exc}")
            print(f"          Trying GIF fallback: {output_gif}")

        try:
            duration = 1.0 / float(fps) if fps and fps > 0 else 0.1
            imageio.mimsave(output_gif, frames_rgb, duration=duration)
            if output_gif.exists() and output_gif.stat().st_size > 0:
                size_mb = output_gif.stat().st_size / (1024 * 1024)
                print(f"Saved preview GIF fallback: {output_gif}")
                print(f"Preview GIF size: {size_mb:.2f} MB")
                return
            raise RuntimeError("GIF writer finished but output file was empty or missing.")
        except Exception as exc:
            print(f"[WARNING] Preview GIF fallback failed: {exc}")

    except Exception as exc:
        print(f"[WARNING] imageio unavailable; skipping MP4/GIF preview. Reason: {exc}")

    # Final fallback: static contact-sheet PNG with a few frames.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not frames_rgb:
            return

        pick = np.linspace(0, len(frames_rgb) - 1, min(12, len(frames_rgb))).astype(int)
        cols = min(4, len(pick))
        rows = int(np.ceil(len(pick) / cols))

        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        axes = np.atleast_1d(axes).ravel()

        for ax_i, ax in enumerate(axes):
            ax.axis("off")
            if ax_i < len(pick):
                frame_number = int(pick[ax_i])
                ax.imshow(frames_rgb[frame_number])
                ax.set_title(f"frame {frame_number}", fontsize=8)

        fig.tight_layout()
        fig.savefig(output_png, dpi=120)
        plt.close(fig)

        if output_png.exists() and output_png.stat().st_size > 0:
            print(f"Saved preview contact sheet fallback: {output_png}")
            return

    except Exception as exc:
        print(f"[WARNING] Preview contact sheet fallback also failed: {exc}")

    print("[WARNING] All preview outputs failed; motion-corrected TIFF was still saved successfully.")




def stage_motion_correction(
    input_dir: Path,
    output_dir: Path,
    max_shifts: tuple[int, int],
    strides: tuple[int, int],
    overlaps: tuple[int, int],
    max_deviation_rigid: int,
    pw_rigid: bool,
    shifts_opencv: bool,
    border_nan: str,
    nonneg_movie: bool,
    dry_run: bool,
    make_preview_movies: bool = True,
    preview_downscale: int = 4,
    preview_fps: float = 10.0,
    preview_max_frames: int = 1200,
) -> None:
    """Run CaImAn motion correction on TIFF files."""
    require_dir(input_dir, "Motion correction input directory")
    output_dir.mkdir(parents=True, exist_ok=True)

    shift_dir = output_dir / "shift"
    template_dir = output_dir / "template"
    preview_dir = output_dir / "preview_movies"
    shift_dir.mkdir(parents=True, exist_ok=True)
    template_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    tiff_files = sorted(f for f in input_dir.glob("*.tif") if "_snap" not in f.name)

    print(f"\nFound {len(tiff_files)} TIFF file(s) for motion correction.")
    if not tiff_files:
        return

    if dry_run:
        for f in tiff_files:
            print(f"[DRY RUN] Would motion-correct: {f}")
        return

    import multiprocessing
    import matplotlib.pyplot as plt
    import caiman as cm
    from caiman.motion_correction import MotionCorrect

    n_cores = multiprocessing.cpu_count()
    n_processes = max(1, n_cores - 2)

    print(f"Using {n_processes} processes for motion correction.")

    c, dview, _ = cm.cluster.setup_cluster(
        backend="multiprocessing",
        n_processes=n_processes,
        single_thread=False,
    )

    try:
        for fpath in tiff_files:
            print(f"\n[MOTION] Processing {fpath.name}...")

            mc = MotionCorrect(
                [str(fpath)],
                dview=dview,
                max_shifts=max_shifts,
                strides=strides,
                overlaps=overlaps,
                max_deviation_rigid=max_deviation_rigid,
                shifts_opencv=shifts_opencv,
                pw_rigid=pw_rigid,
                border_nan=border_nan,
                nonneg_movie=nonneg_movie,
            )

            mc.motion_correct(save_movie=True)

            m_corr = cm.load(mc.mmap_file)

            corrected_path = output_dir / f"{fpath.stem}_mc.tif"
            tifffile.imwrite(str(corrected_path), m_corr.astype(np.float32))

            if make_preview_movies:
                preview_path = preview_dir / f"{fpath.stem}_mc_preview.mp4"
                try:
                    write_lowres_preview_movie(
                        tiff_path=corrected_path,
                        output_mp4=preview_path,
                        downscale=preview_downscale,
                        fps=preview_fps,
                        max_frames=preview_max_frames,
                    )
                except Exception as exc:
                    print(f"[WARNING] Preview movie generation failed for {corrected_path.name}: {exc}")
                    print("          Motion-corrected TIFF was still saved successfully.")
                    print("          Continuing pipeline.")

            shift_plot_path = shift_dir / f"{fpath.stem}_shifts.png"
            plt.figure(figsize=(9.20, 4.10))
            plt.plot(mc.shifts_rig)
            plt.xlabel("Frame")
            plt.ylabel("Pixels")
            plt.legend(["x shifts", "y shifts"])
            plt.title(f"Rigid shifts for {fpath.name}")
            plt.tight_layout()
            plt.savefig(shift_plot_path)
            plt.close()

            template_path = template_dir / f"{fpath.stem}_template.png"
            plt.figure(figsize=(7.36, 6.56))
            plt.imshow(mc.total_template_rig, cmap="gray")
            plt.title(f"Template for {fpath.name}")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(template_path)
            plt.close()

            print(f"Saved corrected TIFF: {corrected_path}")
            print(f"Saved shift plot:     {shift_plot_path}")
            print(f"Saved template:       {template_path}")

    finally:
        cm.stop_server(dview=dview)
        print("CaImAn cluster stopped.")


# =============================================================================
# STAGE 4: CELLPOSE MASKS
# =============================================================================


def try_natsorted(paths: Iterable[Path]) -> list[Path]:
    """Use natsort if available, otherwise sorted."""
    try:
        from natsort import natsorted

        return list(natsorted(paths))
    except ImportError:
        return sorted(paths)



def labels_to_random_rgb(mask: np.ndarray, seed: int = 1) -> np.ndarray:
    """Convert a label mask to random RGB colors. Background 0 is black."""
    mask = np.asarray(mask)
    labels = np.unique(mask)
    max_label = int(labels.max()) if labels.size else 0

    rng = np.random.default_rng(seed)
    lut = np.zeros((max_label + 1, 3), dtype=np.uint8)
    if max_label > 0:
        lut[1:] = rng.integers(40, 256, size=(max_label, 3), dtype=np.uint8)

    return lut[mask.astype(np.int64)]


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Return boundary pixels for a 2D labeled or binary mask."""
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError("mask_boundary expects a 2D mask.")

    fg = m > 0
    boundary = np.zeros_like(fg, dtype=bool)

    boundary[1:, :] |= m[1:, :] != m[:-1, :]
    boundary[:-1, :] |= m[1:, :] != m[:-1, :]
    boundary[:, 1:] |= m[:, 1:] != m[:, :-1]
    boundary[:, :-1] |= m[:, 1:] != m[:, :-1]

    boundary &= fg
    return boundary


def save_mask_qc_images(
    max_projection: np.ndarray,
    masks: np.ndarray,
    output_dir: Path,
    stem: str,
) -> None:
    """
    Save segmentation QC images:
        random label colors
        grayscale max projection with white mask outlines
        boundary-only image

    Uses imageio if available; otherwise falls back to matplotlib PNG writing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    mask2d = np.asarray(masks)
    if mask2d.ndim != 2:
        print(f"[WARNING] Cannot make mask QC for {stem}: mask is not 2D.")
        return

    img8 = normalize_frame_to_uint8(np.asarray(max_projection))
    random_rgb = labels_to_random_rgb(mask2d, seed=1)

    overlay = np.repeat(img8[:, :, None], 3, axis=2)
    b = mask_boundary(mask2d)
    overlay[b] = np.array([255, 255, 255], dtype=np.uint8)

    boundary_img = np.zeros_like(overlay)
    boundary_img[b] = np.array([255, 255, 255], dtype=np.uint8)

    random_path = output_dir / f"{stem}_mask_random_colors.png"
    overlay_path = output_dir / f"{stem}_mask_outline_overlay.png"
    boundary_path = output_dir / f"{stem}_mask_boundaries_only.png"

    def _write_png(path: Path, arr: np.ndarray) -> None:
        try:
            import imageio.v2 as imageio
            imageio.imwrite(path, arr)
            return
        except Exception:
            pass

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.imsave(path, arr)

    try:
        _write_png(random_path, random_rgb)
        _write_png(overlay_path, overlay)
        _write_png(boundary_path, boundary_img)

        print(f"Saved mask QC random colors: {random_path}")
        print(f"Saved mask QC overlay:       {overlay_path}")
        print(f"Saved mask QC boundaries:    {boundary_path}")
    except Exception as exc:
        print(f"[WARNING] Failed to save mask QC images for {stem}: {exc}")




def stage_tiff_to_mask(
    input_dir: Path,
    snap_dir: Path | None,
    dry_run: bool,
) -> None:
    """Generate Cellpose masks from TIFF files and save masks/mat/cell count CSV."""
    require_dir(input_dir, "Cellpose input directory")
    if snap_dir is not None:
        require_dir(snap_dir, "Snap/reference directory")

    files = try_natsorted(
        f
        for f in input_dir.glob("*.tif")
        if "_masks" not in f.name
        and "_flows" not in f.name
        and "_maxproj" not in f.name
        and "_snap" not in f.name
    )

    print(f"\nFound {len(files)} TIFF file(s) for Cellpose masks.")
    if not files:
        return

    if dry_run:
        for f in files:
            print(f"[DRY RUN] Would segment: {f}")
        return

    import pandas as pd
    import scipy.io as sio
    import torch
    from cellpose import io, models

    io.logger_setup()

    device = choose_torch_device(label=" for Cellpose")
    model = models.CellposeModel(device=device)

    output_csv = input_dir / "cell_counts.csv"
    image_ext = ".tif"
    masks_ext = "_masks.tif"

    results = []

    for f in files:
        print(f"\n[MASK] {f.name}")
        img = tifffile.imread(f)

        print(f"Dimensions: {img.ndim}")

        if img.ndim == 3:
            img = np.max(img, axis=0)
            max_proj_path = input_dir / f"{f.stem}_maxproj{image_ext}"
            tifffile.imwrite(max_proj_path, img)
            print(f"Saved max projection: {max_proj_path}")

        img_tensor = torch.from_numpy(img).to(torch.float32).to(device)
        masks, flows, styles = model.eval(img_tensor, normalize={"tile_norm_blocksize": 256})

        cell_ids = np.asarray(_v472_positive_mask_labels(masks), dtype=int)
        cell_ids = cell_ids[cell_ids != 0]

        cell_ids_in_snap = []

        if snap_dir is not None:
            prefix = f.stem.split("_")[0]
            snap_file = snap_dir / f"{prefix}_snap{image_ext}"

            if snap_file.exists():
                snap_img = tifffile.imread(snap_file)
                snap_tensor = torch.from_numpy(snap_img).to(torch.float32).to(device)
                snap_masks, _, _ = model.eval(snap_tensor, normalize={"tile_norm_blocksize": 256})
                snap_binary = snap_masks > 0

                for cid in cell_ids:
                    cell_mask = masks == cid
                    if np.any(cell_mask & snap_binary):
                        cell_ids_in_snap.append(cid)

        mask_path = input_dir / f"{f.stem}{masks_ext}"
        tifffile.imwrite(mask_path, masks.astype(np.uint16))
        print(f"Saved labeled mask: {mask_path}")
        try:
            detected_cells = int(np.max(masks))
            print(f"Detected cells from labels: {detected_cells}")
        except Exception:
            pass

        try:
            qc_dir = input_dir / "mask_qc"
            save_mask_qc_images(
                max_projection=img,
                masks=masks,
                output_dir=qc_dir,
                stem=f.stem,
            )
        except Exception as exc:
            print(f"[WARNING] Mask QC image generation failed for {f.name}: {exc}")

        h, w = masks.shape
        masks_3d = np.zeros((h, w, len(cell_ids)), dtype=np.uint8)

        for i, cid in enumerate(cell_ids):
            masks_3d[:, :, i] = (masks == cid).astype(np.uint8)

        mat_path = input_dir / f"{f.stem}_masks_3d.mat"
        sio.savemat(mat_path, {"masks_3d": masks_3d})
        print(f"Saved 3D mask .mat: {mat_path}")

        bg_mask = masks == 0
        mat_bg_path = input_dir / f"{f.stem}_bg.mat"
        sio.savemat(mat_bg_path, {"bg": bg_mask})
        print(f"Saved MATLAB background mask: {mat_bg_path}")

        results.append(
            {
                "file": f.name,
                "total_cells": len(cell_ids),
                "cells_in_snapshot": len(cell_ids_in_snap),
                "cell_ids_in_snapshot": ",".join(map(str, cell_ids_in_snap)),
            }
        )

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"\nDone. Cell counts saved to: {output_csv}")



# =============================================================================
# STAGE 5: CALCIUM TRACE EXTRACTION
# =============================================================================


def fit_bleach_curve(bg_trace: np.ndarray) -> np.ndarray:
    """
    Fit MATLAB bleaching model:

        a * exp(-b * x^d) + c

    Uses scipy curve_fit with bounds matching the MATLAB script as closely as possible.
    If fitting fails, falls back to the measured normalized background trace.
    """
    from scipy.optimize import curve_fit

    bg_trace = np.asarray(bg_trace, dtype=np.float64)
    n_frames = bg_trace.size
    x = np.arange(1, n_frames + 1, dtype=np.float64)

    def model(x, a, b, c, d):
        return a * np.exp(-b * np.power(x, d)) + c

    min_bg = float(np.nanmin(bg_trace))
    max_bg = float(np.nanmax(bg_trace))

    p0 = [max_bg - min_bg, 0.0, min_bg, 1.0]
    lower = [0.0, 0.0, 0.0, 0.0]
    upper = [np.inf, np.inf, min_bg, 1.0]

    try:
        popt, _ = curve_fit(
            model,
            x,
            bg_trace,
            p0=p0,
            bounds=(lower, upper),
            maxfev=20000,
        )
        bleach_curve = model(x, *popt)

        if np.any(~np.isfinite(bleach_curve)) or np.any(bleach_curve <= 0):
            raise RuntimeError("Invalid bleach curve.")

        return bleach_curve.astype(np.float64)

    except Exception as exc:
        print(f"[WARNING] Bleach fit failed; using measured normalized bgTrace. Reason: {exc}")
        safe = bg_trace.copy()
        safe[~np.isfinite(safe)] = 1.0
        safe[safe <= 0] = 1.0
        return safe


def load_tiff_stack_float64(tif_path: Path) -> np.ndarray:
    """Load multipage TIFF as (T, H, W) float64."""
    with tifffile.TiffFile(tif_path) as tif:
        frames = [page.asarray() for page in tif.pages]
    if not frames:
        raise ValueError(f"No frames found in TIFF: {tif_path}")
    return np.stack(frames, axis=0).astype(np.float64)




def stage_calcium_extract(
    input_dir: Path,
    dt: float,
    baseline_frames: int,
    bleach_correction: bool,
    dry_run: bool,
) -> None:
    """
    Python implementation of the provided MATLAB calcium extraction script.

    For each TIFF:
        - load TIFF stack
        - load expnumber_masks_3d.mat
        - load expnumber_bg.mat
        - compute background bleaching trace
        - fit/apply exponential bleaching correction
        - extract per-cell F
        - subtract background
        - compute DF/F
        - save CaIData-expnumber.mat with t, F, DFoverF and export per-recording CSV files
    """
    require_dir(input_dir, "Calcium extraction input directory")

    if baseline_frames < 1:
        raise ValueError("baseline_frames must be >= 1.")

    import scipy.io as sio

    calcium_csv_dir = input_dir / "calcium_csv"
    calcium_csv_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(
        f for f in input_dir.glob("*.tif")
        if "masks" not in f.name and "maxproj" not in f.name
    )

    print(f"\nFound {len(tif_files)} TIFF file(s) for calcium extraction.")

    for tif_path in tif_files:
        expnumber = tif_path.stem
        mask_path = input_dir / f"{expnumber}_masks_3d.mat"
        bg_path = input_dir / f"{expnumber}_bg.mat"
        out_path = input_dir / f"CaIData-{expnumber}.mat"

        print(f"\n[CALCIUM] Processing {expnumber}")
        print(f"          TIFF: {tif_path}")
        print(f"          Mask: {mask_path}")
        print(f"          BG:   {bg_path}")
        print(f"          Out:  {out_path}")

        if not mask_path.exists():
            print(f"[SKIP] Mask file not found: {mask_path}")
            continue

        if not bg_path.exists():
            print(f"[SKIP] Background file not found: {bg_path}")
            continue

        if dry_run:
            continue

        movie = load_tiff_stack_float64(tif_path)  # (T,H,W)
        n_frames, height, width = movie.shape

        mask_data = sio.loadmat(mask_path)
        if "masks_3d" not in mask_data:
            print(f"[SKIP] masks_3d variable not found in {mask_path}")
            continue

        bw = np.asarray(mask_data["masks_3d"]).astype(bool)

        if bw.ndim != 3:
            print(f"[SKIP] masks_3d is not 3D: shape={bw.shape}")
            continue

        if bw.shape[0] != height or bw.shape[1] != width:
            print(f"[SKIP] Mask/movie dimension mismatch. movie={(height, width)}, mask={bw.shape[:2]}")
            continue

        n_cells = bw.shape[2]

        bg_data = sio.loadmat(bg_path)
        if "bg" not in bg_data:
            print(f"[SKIP] bg variable not found in {bg_path}")
            continue

        bg = np.asarray(bg_data["bg"]).astype(bool)

        if bg.shape != (height, width):
            print(f"[SKIP] bg/movie dimension mismatch. movie={(height, width)}, bg={bg.shape}")
            continue

        if np.count_nonzero(bg) == 0:
            print("[SKIP] Background mask is empty.")
            continue

        # Background bleaching trace.
        bg_trace = np.zeros(n_frames, dtype=np.float64)
        for frame_idx in range(n_frames):
            bg_trace[frame_idx] = np.mean(movie[frame_idx][bg])

        if not np.isfinite(bg_trace[0]) or bg_trace[0] == 0:
            print("[WARNING] First background value invalid; skipping bleaching correction.")
            bleach_curve = np.ones(n_frames, dtype=np.float64)
        elif bleach_correction:
            bg_trace_norm = bg_trace / bg_trace[0]
            bleach_curve = fit_bleach_curve(bg_trace_norm)
        else:
            bleach_curve = np.ones(n_frames, dtype=np.float64)

        # Apply bleaching correction to entire movie.
        movie = movie / bleach_curve[:, None, None]

        # Extract per-cell fluorescence.
        F = np.zeros((n_frames, n_cells), dtype=np.float64)

        for cell_idx in range(n_cells):
            mask = bw[:, :, cell_idx]
            if np.count_nonzero(mask) == 0:
                F[:, cell_idx] = np.nan
                continue

            for frame_idx in range(n_frames):
                F[frame_idx, cell_idx] = np.mean(movie[frame_idx][mask])

        # Background fluorescence after correction.
        F_bg = np.zeros((n_frames, 1), dtype=np.float64)
        for frame_idx in range(n_frames):
            F_bg[frame_idx, 0] = np.mean(movie[frame_idx][bg])

        F_corrected = F - F_bg

        baseline_frames_use = min(baseline_frames, n_frames)
        F0 = np.nanmean(F_corrected[:baseline_frames_use, :], axis=0)

        with np.errstate(divide="ignore", invalid="ignore"):
            DFoverF = (F_corrected - F0[None, :]) / F0[None, :]

        t_vec = np.arange(n_frames, dtype=np.float64).reshape(-1, 1) * float(dt)

        sio.savemat(out_path, {"t": t_vec, "F": F, "DFoverF": DFoverF})

        # Per-recording CSVs only. No combined global CSVs are written in v4.2.
        import pandas as pd

        cell_areas = np.array([np.count_nonzero(bw[:, :, i]) for i in range(n_cells)], dtype=np.int64)
        time_flat = t_vec.reshape(-1)

        # Wide raw F CSV: one row per frame, one column per cell.
        f_wide = pd.DataFrame(F, columns=[f"cell_{i+1}" for i in range(n_cells)])
        f_wide.insert(0, "time_s", time_flat)
        f_wide.insert(0, "frame", np.arange(1, n_frames + 1))
        f_csv = calcium_csv_dir / f"{expnumber}_F_wide.csv"
        f_wide.to_csv(f_csv, index=False)

        # Wide DF/F CSV: one row per frame, one column per cell.
        dff_wide = pd.DataFrame(DFoverF, columns=[f"cell_{i+1}" for i in range(n_cells)])
        dff_wide.insert(0, "time_s", time_flat)
        dff_wide.insert(0, "frame", np.arange(1, n_frames + 1))
        dff_csv = calcium_csv_dir / f"{expnumber}_DFoverF_wide.csv"
        dff_wide.to_csv(dff_csv, index=False)

        # Per-cell summary CSV for this recording only.
        summary_rows = []
        duration_s = float(time_flat[-1]) if n_frames > 0 else np.nan

        for cell_idx in range(n_cells):
            cell_id = cell_idx + 1
            F_cell = F[:, cell_idx]
            DFF_cell = DFoverF[:, cell_idx]

            if np.all(~np.isfinite(DFF_cell)):
                peak_dff = np.nan
                peak_frame = np.nan
                peak_time_s = np.nan
            else:
                peak_idx = int(np.nanargmax(DFF_cell))
                peak_dff = float(DFF_cell[peak_idx])
                peak_frame = peak_idx + 1
                peak_time_s = float(time_flat[peak_idx])

            if np.all(~np.isfinite(F_cell)):
                mean_F = np.nan
                F0_value = np.nan
            else:
                mean_F = float(np.nanmean(F_cell))
                F0_value = float(F_cell[0]) if np.isfinite(F_cell[0]) else np.nan

            summary_rows.append(
                {
                    "recording": expnumber,
                    "cell_id": cell_id,
                    "n_frames": n_frames,
                    "duration_s": duration_s,
                    "F0": F0_value,
                    "mean_F": mean_F,
                    "peak_DFoverF": peak_dff,
                    "peak_frame": peak_frame,
                    "peak_time_s": peak_time_s,
                    "area_px": int(cell_areas[cell_idx]),
                }
            )

        summary_csv = calcium_csv_dir / f"{expnumber}_cell_summary.csv"
        pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)

        print(f"Finished {expnumber}: {n_cells} cells, {n_frames} frames")
        print(f"  Wrote: {f_csv}")
        print(f"  Wrote: {dff_csv}")
        print(f"  Wrote: {summary_csv}")

    print("\nAll calcium extraction files processed.")



# =============================================================================
# STAGE 5: MAT -> CSV
# =============================================================================


def stage_mat_to_csv(
    input_dir: Path,
    variable: str,
    output_dir: Path | None,
    dry_run: bool,
) -> None:
    """Export F or DFoverF arrays from .mat files to CSV."""
    require_dir(input_dir, "MAT input directory")

    if output_dir is None:
        output_dir = input_dir
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    import scipy.io as sio

    mat_files = sorted(input_dir.glob("*.mat"))

    # Avoid exporting mask/background .mat files as calcium traces.
    mat_files = [
        f
        for f in mat_files
        if not f.name.endswith("_masks_3d.mat") and not f.name.endswith("_bg.mat")
    ]

    print(f"\nFound {len(mat_files)} MAT file(s) for CSV export.")

    for mat_file in mat_files:
        out_csv = output_dir / f"{mat_file.stem}.csv"

        print(f"[CSV] {mat_file.name} -> {out_csv.name}")

        if dry_run:
            continue

        data = sio.loadmat(mat_file)

        if variable == "auto":
            if "F" in data:
                chosen = "F"
            elif "DFoverF" in data:
                chosen = "DFoverF"
            else:
                print(f"[SKIP] No F or DFoverF in {mat_file.name}")
                continue
        else:
            chosen = variable

        if chosen not in data:
            print(f"[SKIP] {chosen} not found in {mat_file.name}")
            continue

        array_to_save = np.asarray(data[chosen])

        if array_to_save.ndim != 2:
            print(f"[SKIP] {chosen} in {mat_file.name} is not 2D: shape={array_to_save.shape}")
            continue

        n_cells = array_to_save.shape[1]
        cell_ids = np.arange(1, n_cells + 1)

        array_to_write = np.vstack([cell_ids, array_to_save])
        np.savetxt(out_csv, array_to_write, delimiter=",")
        print(f"Saved {out_csv}")



# =============================================================================
# ORCHESTRATION: RUN FULL PIPELINE WITH CONDA ENV SWITCHING
# =============================================================================


DEFAULT_DCIMG_ENV = "dcimg"
DEFAULT_CELLPOSE_ENV = "cellpose_py310"
DEFAULT_CAIMAN_ENV = "caiman"


def shell_join(args: list[str]) -> str:
    """Return a readable shell-like command string."""
    import shlex
    return " ".join(shlex.quote(str(a)) for a in args)


def run_subprocess_stage(
    env_name: str,
    script_path: Path,
    stage: str,
    args: argparse.Namespace,
) -> None:
    """Run one stage in the requested conda environment using conda run."""
    cmd = [
        "conda",
        "run",
        "--no-capture-output",
        "-n",
        env_name,
        "python",
        str(script_path),
        "--stages",
        stage,
        "--no-timestamped_output",
        "--output_root",
        str(args.output_root),
        "--ms_per_frame",
        str(args.ms_per_frame),
        "--ds_factor",
        str(args.ds_factor),
    ]

    if args.dcimg_dir:
        cmd.extend(["--dcimg_dir", str(args.dcimg_dir)])
    if args.ome_dir:
        cmd.extend(["--ome_dir", str(args.ome_dir)])
    if getattr(args, "bioformats_jar", ""):
        cmd.append(f"--bioformats_jar={args.bioformats_jar}")
    if args.snap_dir:
        cmd.extend(["--snap_dir", str(args.snap_dir)])
    if args.csv_input_dir:
        cmd.extend(["--csv_input_dir", str(args.csv_input_dir)])
    if args.csv_output_dir:
        cmd.extend(["--csv_output_dir", str(args.csv_output_dir)])
    if args.csv_variable:
        cmd.extend(["--csv_variable", str(args.csv_variable)])

    if getattr(args, "calcium_input_dir", ""):
        cmd.extend(["--calcium_input_dir", str(args.calcium_input_dir)])

    if hasattr(args, "baseline_frames"):
        cmd.append(f"--baseline_frames={args.baseline_frames}")

    if hasattr(args, "bleach_correction"):
        cmd.append("--bleach_correction" if args.bleach_correction else "--no-bleach_correction")

    if getattr(args, "stim_file", ""):
        cmd.extend(["--stim_file", str(args.stim_file)])
    if getattr(args, "stim_dir", ""):
        cmd.extend(["--stim_dir", str(args.stim_dir)])
    if getattr(args, "stim_match_mode", ""):
        cmd.extend(["--stim_match_mode", str(args.stim_match_mode)])
    if getattr(args, "stim_file_glob", ""):
        cmd.extend(["--stim_file_glob", str(args.stim_file_glob)])
    if getattr(args, "stim_channel", ""):
        cmd.extend(["--stim_channel", str(args.stim_channel)])
    if getattr(args, "stim_mat_variable", ""):
        cmd.extend(["--stim_mat_variable", str(args.stim_mat_variable)])
    if getattr(args, "stim_expected_times", ""):
        cmd.extend(["--stim_expected_times", str(args.stim_expected_times)])
    if hasattr(args, "stim_search_window_sec"):
        cmd.append(f"--stim_search_window_sec={args.stim_search_window_sec}")
    if hasattr(args, "stim_threshold"):
        cmd.append(f"--stim_threshold={args.stim_threshold}")
    if hasattr(args, "stim_max_pulse_sec"):
        cmd.append(f"--stim_max_pulse_sec={args.stim_max_pulse_sec}")
    if hasattr(args, "extra_stim_delay_sec"):
        cmd.append(f"--extra_stim_delay_sec={args.extra_stim_delay_sec}")
    if hasattr(args, "recording_start_sec"):
        cmd.append(f"--recording_start_sec={args.recording_start_sec}")
    if hasattr(args, "max_data_regions"):
        cmd.append(f"--max_data_regions={args.max_data_regions}")
    if getattr(args, "stim_output_xlsx", ""):
        cmd.extend(["--stim_output_xlsx", str(args.stim_output_xlsx)])
    if getattr(args, "analysis_run_name", ""):
        cmd.extend(["--analysis_run_name", str(args.analysis_run_name)])
    if getattr(args, "analysis_runs_dir", ""):
        cmd.extend(["--analysis_runs_dir", str(args.analysis_runs_dir)])
    if hasattr(args, "allow_no_stim_pulses"):
        cmd.append("--allow_no_stim_pulses" if args.allow_no_stim_pulses else "--no-allow_no_stim_pulses")
    if hasattr(args, "allow_missing_stim_file"):
        cmd.append("--allow_missing_stim_file" if args.allow_missing_stim_file else "--no-allow_missing_stim_file")
    if hasattr(args, "copy_script_snapshot"):
        cmd.append("--copy_script_snapshot" if args.copy_script_snapshot else "--no-copy_script_snapshot")
    if hasattr(args, "save_shell_history"):
        cmd.append("--save_shell_history" if args.save_shell_history else "--no-save_shell_history")
    if hasattr(args, "shell_history_lines"):
        cmd.append(f"--shell_history_lines={args.shell_history_lines}")
    if hasattr(args, "save_conda_envs"):
        cmd.append("--save_conda_envs" if args.save_conda_envs else "--no-save_conda_envs")
    if getattr(args, "conda_env_names", ""):
        cmd.append(f"--conda_env_names={args.conda_env_names}")

    if getattr(args, "region_labels", ""):
        cmd.extend(["--region_labels", str(args.region_labels)])
    if getattr(args, "plot_region_labels", ""):
        cmd.extend(["--plot_region_labels", str(args.plot_region_labels)])
    if getattr(args, "region_values", ""):
        cmd.extend(["--region_values", str(args.region_values)])
    if getattr(args, "region_value_name", ""):
        cmd.extend(["--region_value_name", str(args.region_value_name)])
    for name in [
        "responder_threshold",
        "responder_metric",
        "responder_call_rule",
        "noise_sd_multiplier",
        "baseline_region_index",
        "min_cell_area_px",
        "min_F0",
        "max_baseline_sd",
        "max_abs_baseline_slope",
        "focus_qc_padding_px",
        "focus_qc_cv_threshold",
        "focus_qc_corr_threshold",
        "focus_qc_delta_threshold",
        "heatmap_clip_low_percentile",
        "heatmap_clip_high_percentile",
        "responder_peak_smoothing_frames",
        "responder_min_consecutive_frames",
    ]:
        if hasattr(args, name):
            # Use --arg=value form so negative numeric values, e.g. -1e18,
            # are not mistaken for option flags by argparse in subprocesses.
            cmd.append(f"--{name}={getattr(args, name)}")
    if hasattr(args, "make_plots"):
        cmd.append("--make_plots" if args.make_plots else "--no-make_plots")

    cmd.append(f"--max_shifts={args.max_shifts[0]},{args.max_shifts[1]}")
    cmd.append(f"--strides={args.strides[0]},{args.strides[1]}")
    cmd.append(f"--overlaps={args.overlaps[0]},{args.overlaps[1]}")
    cmd.append(f"--max_deviation_rigid={args.max_deviation_rigid}")
    cmd.extend(["--border_nan", str(args.border_nan)])

    cmd.append("--pw_rigid" if args.pw_rigid else "--no-pw_rigid")
    cmd.append("--shifts_opencv" if args.shifts_opencv else "--no-shifts_opencv")
    cmd.append("--nonneg_movie" if args.nonneg_movie else "--no-nonneg_movie")

    if hasattr(args, "make_preview_movies"):
        cmd.append("--make_preview_movies" if args.make_preview_movies else "--no-make_preview_movies")
    if hasattr(args, "preview_downscale"):
        cmd.append(f"--preview_downscale={args.preview_downscale}")
    if hasattr(args, "preview_fps"):
        cmd.append(f"--preview_fps={args.preview_fps}")
    if hasattr(args, "preview_max_frames"):
        cmd.append(f"--preview_max_frames={args.preview_max_frames}")

    if args.recursive:
        cmd.append("--recursive")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.dry_run:
        cmd.append("--dry_run")

    print("\n" + "=" * 92)
    print(f"ORCHESTRATED STAGE: {stage}")
    print(f"CONDA ENV:          {env_name}")
    print("=" * 92)
    print(shell_join(cmd))

    if args.dry_run:
        print("[DRY RUN] Not executing stage.")
        return

    completed = subprocess.run(_v469_repair_analysis_run_name_args(cmd))
    if completed.returncode != 0:
        raise RuntimeError(
            f"Stage failed: {stage}\n"
            f"Environment: {env_name}\n"
            f"Return code: {completed.returncode}"
        )



def create_timestamped_processed_output_root(args: argparse.Namespace) -> None:
    """
    For --run_all, route all outputs into one top-level analysis run folder.

    Example:
        --output_root /Users/ncc/Data/260617/DCIMG_processed

    becomes:
        /Users/ncc/Data/260617/DCIMG_processed/analysis_runs/2026-06-21_15-18-02/

    No latest symlink or pointer is created. The timestamped folder name is the
    run identifier, and alphabetic sorting naturally shows chronological order.
    """
    if not getattr(args, "run_all", False):
        return

    if not getattr(args, "timestamped_output", True):
        return

    base_output_root = expand_path(args.output_root)
    if base_output_root is None:
        raise ValueError("--output_root is required.")

    run_name = str(getattr(args, "processed_run_name", "") or "").strip()
    if not run_name:
        run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    analysis_runs_dir = base_output_root / "analysis_runs"
    analysis_runs_dir.mkdir(parents=True, exist_ok=True)

    run_output_root = analysis_runs_dir / run_name
    if run_output_root.exists():
        run_output_root = analysis_runs_dir / f"{run_name}_{datetime.now().strftime('%f')}"

    run_output_root.mkdir(parents=True, exist_ok=False)

    args.base_output_root = str(base_output_root)
    args.output_root = str(run_output_root)
    # For the stim stage in --run_all, use this same run folder instead of
    # creating a second nested timestamped analysis folder.
    args.analysis_runs_dir = str(run_output_root)
    args.analysis_run_name = "__USE_EXISTING_ANALYSIS_DIR__"

    print("\nTimestamped analysis-run output enabled.")
    print(f"Base output root: {base_output_root}")
    print(f"This analysis run: {run_output_root}")


def run_all_pipeline(args: argparse.Namespace) -> int:
    """Run all pipeline stages sequentially without further user interaction."""
    script_path = Path(__file__).resolve()

    if args.input_format == "dcimg":
        first_stage = "dcimg"
        first_env = args.dcimg_env
        if not args.dcimg_dir:
            raise ValueError("--dcimg_dir is required when --input_format dcimg.")
    elif args.input_format == "ometiff":
        first_stage = "ometiff"
        first_env = args.dcimg_env
        if not args.ome_dir:
            raise ValueError("--ome_dir is required when --input_format ometiff.")
    else:
        raise ValueError(f"Unsupported input format: {args.input_format}")

    stage_plan: list[tuple[str, str]] = [
        (first_stage, first_env),
        ("downsample", args.cellpose_env),
    ]

    if not args.skip_motion:
        stage_plan.append(("motion", args.caiman_env))
        stage_plan.append(("mask", args.cellpose_env))
        stage_plan.append(("calcium", args.cellpose_env))
    else:
        print(
            "\n[WARNING] --skip_motion was set. Automatic mask stage is also skipped "
            "because this v2 mask stage expects motion-corrected TIFFs."
        )

    if getattr(args, "stim_file", "") or getattr(args, "stim_dir", "") or getattr(args, "stim_expected_times", ""):
        stage_plan.append(("stim", args.cellpose_env))

    if args.include_csv:
        stage_plan.append(("csv", args.cellpose_env))

    print_header()
    report_optional_dependencies()
    print("\nSequential one-command pipeline plan:")
    for stage, env in stage_plan:
        print(f"  {stage:12s} -> {env}")

    if "stim" not in [stage for stage, _env in stage_plan]:
        print("\nNote: stim analysis is not included because neither --stim_file/--stim_dir nor --stim_expected_times was provided.")

    for stage, env in stage_plan:
        run_subprocess_stage(
            env_name=env,
            script_path=script_path,
            stage=stage,
            args=args,
        )

    print("\n[COMPLETE] Sequential one-command pipeline finished.")
    return 0




# =============================================================================
# STAGE 7: STIMULUS-DEFINED REGION ANALYSIS
# =============================================================================


def parse_expected_times(expected: str) -> list[float]:
    """Parse comma-separated expected stimulus times in seconds."""
    if expected is None or str(expected).strip() == "":
        return []
    vals = []
    for part in str(expected).split(","):
        part = part.strip()
        if part:
            vals.append(float(part))
    return vals


def parse_optional_labels(labels: str) -> list[str]:
    """Parse comma-separated region labels."""
    if labels is None or str(labels).strip() == "":
        return []
    return [p.strip() for p in str(labels).split(",") if p.strip()]


def parse_optional_float_values(values: str) -> list[float]:
    """Parse comma-separated numeric region values."""
    if values is None or str(values).strip() == "":
        return []
    return [float(p.strip()) for p in str(values).split(",") if p.strip()]


def safe_float(x) -> float:
    """Convert to finite float or NaN."""
    try:
        val = float(x)
        return val if np.isfinite(val) else np.nan
    except Exception:
        return np.nan


def temporal_middle90(values: np.ndarray) -> np.ndarray:
    """Return middle 90% by time/index, excluding first and last 5%."""
    values = np.asarray(values, dtype=float).reshape(-1)
    n = values.size
    if n == 0:
        return values
    start = int(np.floor(0.05 * n))
    end = int(np.ceil(0.95 * n))
    if end <= start:
        return values
    return values[start:end]


def temporal_middle90_mean(values: np.ndarray) -> float:
    """Mean of the temporal middle 90%."""
    vals = temporal_middle90(values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan
    return float(np.nanmean(vals))


def auc_trapz(time_s: np.ndarray, values: np.ndarray) -> float:
    """Trapezoidal AUC over finite points."""
    t = np.asarray(time_s, dtype=float).reshape(-1)
    v = np.asarray(values, dtype=float).reshape(-1)
    good = np.isfinite(t) & np.isfinite(v)
    if np.count_nonzero(good) < 2:
        return np.nan
    return float(np.trapezoid(v[good], t[good]))


def linear_slope(time_s: np.ndarray, values: np.ndarray) -> float:
    """Simple linear slope of values vs time."""
    t = np.asarray(time_s, dtype=float).reshape(-1)
    v = np.asarray(values, dtype=float).reshape(-1)
    good = np.isfinite(t) & np.isfinite(v)
    if np.count_nonzero(good) < 2:
        return np.nan
    try:
        slope, _intercept = np.polyfit(t[good], v[good], 1)
        return float(slope)
    except Exception:
        return np.nan


def rolling_nanmedian_1d(values: np.ndarray, window: int) -> np.ndarray:
    """Centered rolling median that ignores NaNs. Used to suppress one-frame spikes."""
    v = np.asarray(values, dtype=float).reshape(-1)
    n = v.size
    if n == 0:
        return v.copy()
    window = int(window)
    if window <= 1:
        return v.copy()
    if window % 2 == 0:
        window += 1
    half = window // 2
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = v[lo:hi]
        if np.any(np.isfinite(seg)):
            out[i] = float(np.nanmedian(seg))
    return out


def max_consecutive_true(mask: np.ndarray) -> int:
    """Maximum run length of True values in a 1D boolean array."""
    m = np.asarray(mask, dtype=bool).reshape(-1)
    best = 0
    cur = 0
    for val in m:
        if val:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


def spike_safe_peak(values: np.ndarray, times: np.ndarray, smoothing_frames: int) -> tuple[float, float]:
    """Return max of a rolling-median-smoothed trace and its time."""
    v = rolling_nanmedian_1d(values, smoothing_frames)
    t = np.asarray(times, dtype=float).reshape(-1)
    if v.size == 0 or not np.any(np.isfinite(v)):
        return np.nan, np.nan
    idx = int(np.nanargmax(v))
    peak_time = float(t[idx]) if idx < t.size and np.isfinite(t[idx]) else np.nan
    return float(v[idx]), peak_time



def load_stimulus_trace(
    stim_file: Path,
    stim_channel: str,
    mat_variable: str = "currentone",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load q and the selected ultrasound stimulus channel from a DAQ .csv or .mat file.

    Preferred corrected MAT format:
        q_sec
        scanData1
        scanData2

    Legacy MATLAB table-only files are not reliably readable by scipy because MATLAB
    table objects are stored as MCOS objects. Use fix_daq_mat_files_for_python.m
    to create corrected files in DAQ/python_readable/.
    """
    require_file(stim_file, "Stimulus timing file")

    suffix = stim_file.suffix.lower()
    if suffix == ".csv":
        import pandas as pd
        df = pd.read_csv(stim_file)
        q_col = "q_sec" if "q_sec" in df.columns else "q"
        if q_col not in df.columns:
            raise ValueError(f"CSV missing q_sec or q column: {stim_file}")
        if stim_channel not in df.columns:
            raise ValueError(f"CSV missing stimulus channel {stim_channel}: {stim_file}")
        return df[q_col].to_numpy(dtype=float), df[stim_channel].to_numpy(dtype=float)

    if suffix != ".mat":
        raise ValueError(f"Unsupported stimulus file type: {stim_file}")

    scipy_error = None
    h5_error = None

    try:
        import scipy.io as sio
        data = sio.loadmat(stim_file, squeeze_me=True, struct_as_record=False, simplify_cells=True)

        # Preferred corrected export format.
        if "q_sec" in data and stim_channel in data:
            return (
                np.asarray(data["q_sec"], dtype=float).reshape(-1),
                np.asarray(data[stim_channel], dtype=float).reshape(-1),
            )

        # Alternate direct variable name.
        if "q" in data and stim_channel in data:
            return (
                np.asarray(data["q"], dtype=float).reshape(-1),
                np.asarray(data[stim_channel], dtype=float).reshape(-1),
            )

        # Legacy table/struct fallback, if scipy can read it.
        if mat_variable in data:
            obj = data[mat_variable]

            if isinstance(obj, dict):
                q_key = "q_sec" if "q_sec" in obj else "q" if "q" in obj else None
                if q_key is not None and stim_channel in obj:
                    return (
                        np.asarray(obj[q_key], dtype=float).reshape(-1),
                        np.asarray(obj[stim_channel], dtype=float).reshape(-1),
                    )

            # Struct-like object with named fields.
            for q_name in ["q_sec", "q", "Var1"]:
                if hasattr(obj, q_name) and hasattr(obj, stim_channel):
                    return (
                        np.asarray(getattr(obj, q_name), dtype=float).reshape(-1),
                        np.asarray(getattr(obj, stim_channel), dtype=float).reshape(-1),
                    )

            # Numeric matrix fallback.
            arr = np.asarray(obj)
            if arr.ndim == 2:
                channel_index = {"scanData1": 1, "scanData2": 2}.get(stim_channel)
                if channel_index is not None:
                    if arr.shape[1] >= 3:
                        return arr[:, 0].astype(float), arr[:, channel_index].astype(float)
                    if arr.shape[0] >= 3:
                        return arr[0, :].astype(float), arr[channel_index, :].astype(float)

            # MATLAB MCOS table object often lands here.
            scipy_error = (
                f"Found variable '{mat_variable}', but it was not readable as numeric arrays. "
                "It is likely a MATLAB table/MCOS object."
            )
        else:
            scipy_error = (
                f"MAT file does not contain q_sec/{stim_channel}, q/{stim_channel}, "
                f"or readable '{mat_variable}'."
            )

    except Exception as exc:
        scipy_error = exc

    # MATLAB v7.3/HDF5 fallback.
    try:
        import h5py
        with h5py.File(stim_file, "r") as f:
            if "q_sec" in f and stim_channel in f:
                return (
                    np.array(f["q_sec"]).squeeze().astype(float).reshape(-1),
                    np.array(f[stim_channel]).squeeze().astype(float).reshape(-1),
                )
            if "q" in f and stim_channel in f:
                return (
                    np.array(f["q"]).squeeze().astype(float).reshape(-1),
                    np.array(f[stim_channel]).squeeze().astype(float).reshape(-1),
                )
    except Exception as exc:
        h5_error = exc

    raise ValueError(
        f"Could not parse DAQ file for stimulus analysis:\\n"
        f"  file: {stim_file}\\n"
        f"  requested channel: {stim_channel}\\n\\n"
        f"Expected corrected numeric arrays in the MAT file:\\n"
        f"  q_sec, scanData1, scanData2\\n\\n"
        f"Fix existing MATLAB DAQ files by running:\\n"
        f"  fix_daq_mat_files_for_python.m\\n"
        f"Then use:\\n"
        f"  --stim_dir <DAQ folder>/python_readable\\n\\n"
        f"scipy_error={scipy_error}; h5py_error={h5_error}"
    )



def detect_ultrasound_pulses(
    q: np.ndarray,
    stim_signal: np.ndarray,
    expected_times: list[float],
    search_window_sec: float,
    threshold: float,
    max_pulse_sec: float,
) -> tuple[list[float], list[dict]]:
    """
    Detect ultrasound marker pulses from the selected stimulus channel.

    Uses rising edges in the ultrasound digital channel, not camera-on data.
    If expected_times are provided, only pulses near those times are accepted.
    """
    q = np.asarray(q, dtype=float).reshape(-1)
    stim_signal = np.asarray(stim_signal, dtype=float).reshape(-1)

    if q.size != stim_signal.size:
        raise ValueError(f"q and stimulus channel length mismatch: {q.size} vs {stim_signal.size}")

    high = stim_signal > threshold
    rising_idx = np.where((~high[:-1]) & (high[1:]))[0] + 1
    falling_idx = np.where((high[:-1]) & (~high[1:]))[0] + 1

    pulse_candidates = []

    for r in rising_idx:
        later_falls = falling_idx[falling_idx > r]
        if later_falls.size == 0:
            duration = np.nan
        else:
            duration = float(q[later_falls[0]] - q[r])

        onset = float(q[r])
        valid_duration = np.isnan(duration) or duration <= max_pulse_sec

        pulse_candidates.append(
            {
                "onset_s": onset,
                "duration_s": duration,
                "valid_duration": bool(valid_duration),
            }
        )

    valid_pulses = [p for p in pulse_candidates if p["valid_duration"]]

    warnings = []

    if expected_times:
        detected = []
        for expected in expected_times:
            nearby = [
                p for p in valid_pulses
                if abs(p["onset_s"] - expected) <= search_window_sec
            ]

            if not nearby:
                warnings.append(
                    {
                        "warning": "expected_stimulus_not_found",
                        "detail": f"No valid ultrasound pulse found within ±{search_window_sec}s of expected time {expected}s.",
                    }
                )
                continue

            best = min(nearby, key=lambda p: abs(p["onset_s"] - expected))
            detected.append(best["onset_s"])

        # Remove duplicates while preserving order.
        unique_detected = []
        for val in detected:
            if not any(abs(val - prev) < 1e-6 for prev in unique_detected):
                unique_detected.append(val)

        return unique_detected, warnings

    detected = [p["onset_s"] for p in valid_pulses]
    return detected, warnings


def middle_80_mean(values: np.ndarray) -> float:
    """
    Average the temporal middle 90% of datapoints.

    This discards the first 5% and final 5% by time/index within the region,
    not by value sorting. This avoids transition-zone contamination.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]

    n = values.size
    if n == 0:
        return np.nan

    start = int(np.floor(0.05 * n))
    end = int(np.ceil(0.95 * n))

    if end <= start:
        middle = values
    else:
        middle = values[start:end]

    if middle.size == 0:
        return np.nan

    return float(np.nanmean(middle))



def file_md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calculate md5 checksum for a file."""
    h = hashlib.md5()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def package_version(package_name: str) -> str:
    """Return package version string, or unavailable."""
    try:
        import importlib.metadata as metadata
        return metadata.version(package_name)
    except Exception:
        return "unavailable"


def create_timestamped_analysis_dir(
    calcium_input_dir: Path,
    analysis_runs_dir: Path | None,
    analysis_run_name: str | None,
) -> Path:
    """
    Create or reuse an analysis run folder.

    Default for standalone stim runs:
        <calcium_csv>/analysis_runs/YYYY-MM-DD_HH-MM-SS

    For --run_all in v4_29_analysis_runs, the orchestrator passes
    analysis_run_name="__USE_EXISTING_ANALYSIS_DIR__" and analysis_runs_dir equal
    to the current top-level analysis run folder, so no nested timestamp is made.
    """
    if analysis_run_name == "__USE_EXISTING_ANALYSIS_DIR__":
        if analysis_runs_dir is None:
            raise ValueError("Internal error: existing analysis run folder was not provided.")
        analysis_dir = Path(analysis_runs_dir)
        analysis_dir.mkdir(parents=True, exist_ok=True)
    else:
        if analysis_runs_dir is None:
            parent = calcium_input_dir / "analysis_runs"
        else:
            parent = analysis_runs_dir

        parent.mkdir(parents=True, exist_ok=True)

        if analysis_run_name is None or str(analysis_run_name).strip() == "":
            run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            run_name = str(analysis_run_name).strip()

        analysis_dir = parent / run_name
        if analysis_dir.exists():
            suffix = datetime.now().strftime("%f")
            analysis_dir = parent / f"{run_name}_{suffix}"
        analysis_dir.mkdir(parents=True, exist_ok=False)

    # Standard run-level folders.
    (analysis_dir / "csv").mkdir(parents=True, exist_ok=True)
    (analysis_dir / "reports").mkdir(parents=True, exist_ok=True)
    (analysis_dir / "plots").mkdir(parents=True, exist_ok=True)
    (analysis_dir / "logs").mkdir(parents=True, exist_ok=True)
    (analysis_dir / "script_snapshot").mkdir(parents=True, exist_ok=True)

    return analysis_dir


def write_latest_analysis_pointer(calcium_input_dir: Path, analysis_dir: Path) -> None:
    """No-op in v4_29_analysis_runs: timestamped folders make latest pointers unnecessary."""
    return



def format_command_for_bash(argv: list[str]) -> str:
    """Format sys.argv as a copy-pasteable bash command."""
    import shlex

    if not argv:
        return ""

    quoted = [shlex.quote(str(a)) for a in argv]
    if len(quoted) <= 1:
        return " ".join(quoted)

    lines = [quoted[0]]
    for part in quoted[1:]:
        lines.append(f"  {part}")
    return " \\\n".join(lines) + "\n"


def read_recent_shell_history(max_lines: int = 50) -> list[str]:
    """
    Read recent shell history when available.

    On macOS zsh, ~/.zsh_history often stores commands like:
        : 1718839333:0;python script.py ...

    This function strips the timestamp prefix when present.
    """
    history_candidates = [
        Path.home() / ".zsh_history",
        Path.home() / ".bash_history",
    ]

    for hist_path in history_candidates:
        if hist_path.exists() and hist_path.is_file():
            try:
                raw_lines = hist_path.read_text(errors="ignore").splitlines()
                cleaned = []
                for line in raw_lines[-max_lines:]:
                    # zsh extended history format: ": epoch:duration;command"
                    if line.startswith(": ") and ";" in line:
                        line = line.split(";", 1)[1]
                    cleaned.append(line)
                return cleaned
            except Exception:
                continue

    return []


def write_bash_command_files(
    analysis_dir: Path,
    save_shell_history: bool = True,
    shell_history_lines: int = 50,
) -> None:
    """
    Write bash command reproducibility files.

    Files:
        command.txt
            Raw sys.argv command.

        command_reconstructed.sh
            Copy-pasteable bash command.

        bash_commands.txt
            Reconstructed command plus recent shell history, if readable.
    """
    raw_command = " ".join(sys.argv) + "\n"
    reconstructed = format_command_for_bash(sys.argv)

    (analysis_dir / "command.txt").write_text(raw_command)
    (analysis_dir / "command_reconstructed.sh").write_text(reconstructed)

    lines = []
    lines.append("# Command captured from this analysis run")
    lines.append("")
    lines.append(reconstructed.rstrip())
    lines.append("")

    if save_shell_history:
        history = read_recent_shell_history(max_lines=shell_history_lines)
        lines.append("")
        lines.append(f"# Recent shell history, last {shell_history_lines} lines when available")
        lines.append("# Note: shell history may not be flushed until the terminal session exits.")
        lines.append("")
        if history:
            lines.extend(history)
        else:
            lines.append("# No readable shell history found.")

    (analysis_dir / "bash_commands.txt").write_text("\n".join(lines) + "\n")


def parse_conda_env_names(env_names: str) -> list[str]:
    """Parse comma-separated conda environment names."""
    if env_names is None or str(env_names).strip() == "":
        return []
    return [x.strip() for x in str(env_names).split(",") if x.strip()]


def run_capture_command(cmd: list[str], timeout_sec: int = 120) -> tuple[int, str, str]:
    """Run command and capture stdout/stderr without raising."""
    try:
        completed = subprocess.run(
            _v469_repair_analysis_run_name_args(cmd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
        )
        return completed.returncode, completed.stdout, completed.stderr
    except Exception as exc:
        return -1, "", str(exc)


def write_conda_environment_exports(
    analysis_dir: Path,
    conda_env_names: list[str],
) -> None:
    """
    Save conda environment exports and package lists.

    Files:
        conda_envs/<env>_environment.yml
        conda_envs/<env>_conda_list.txt
        conda_envs/recreate_environments.sh
    """
    conda_dir = analysis_dir / "conda_envs"
    conda_dir.mkdir(parents=True, exist_ok=True)

    recreate_lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Recreate conda environments captured with this analysis run.",
        "# Run from this conda_envs folder or edit paths as needed.",
        "",
    ]

    summary_rows = []

    for env_name in conda_env_names:
        safe_name = env_name.replace("/", "_").replace(" ", "_")
        yml_path = conda_dir / f"{safe_name}_environment.yml"
        list_path = conda_dir / f"{safe_name}_conda_list.txt"

        rc, out, err = run_capture_command(["conda", "env", "export", "-n", env_name], timeout_sec=180)
        if rc == 0 and out.strip():
            yml_path.write_text(out)
        else:
            yml_path.write_text(
                f"# Failed to export conda environment: {env_name}\\n"
                f"# return_code: {rc}\\n"
                f"# stderr:\\n{err}\\n"
            )

        rc2, out2, err2 = run_capture_command(["conda", "list", "-n", env_name], timeout_sec=180)
        if rc2 == 0 and out2.strip():
            list_path.write_text(out2)
        else:
            list_path.write_text(
                f"# Failed to list conda environment: {env_name}\\n"
                f"# return_code: {rc2}\\n"
                f"# stderr:\\n{err2}\\n"
            )

        recreate_lines.append(f"# {env_name}")
        recreate_lines.append(f"conda env create -f {safe_name}_environment.yml")
        recreate_lines.append("")

        summary_rows.append(
            {
                "env_name": env_name,
                "environment_yml": str(yml_path),
                "conda_list": str(list_path),
                "export_return_code": rc,
                "list_return_code": rc2,
            }
        )

    recreate_path = conda_dir / "recreate_environments.sh"
    recreate_path.write_text("\\n".join(recreate_lines) + "\\n")
    try:
        recreate_path.chmod(0o755)
    except Exception:
        pass

    try:
        import pandas as pd
        pd.DataFrame(summary_rows).to_csv(conda_dir / "conda_export_summary.csv", index=False)
    except Exception:
        (conda_dir / "conda_export_summary.json").write_text(json.dumps(summary_rows, indent=2))


def write_environment_used(
    analysis_dir: Path,
    args_dict: dict,
) -> None:
    """Write the intended environment mapping for the pipeline."""
    dcimg_env = args_dict.get("dcimg_env", "dcimg")
    cellpose_env = args_dict.get("cellpose_env", "cellpose_py310")
    caiman_env = args_dict.get("caiman_env", "caiman_py310")

    text_lines = [
        "Environment mapping used by the pipeline",
        "========================================",
        "",
        f"dcimg stage       -> {dcimg_env}",
        f"downsample stage  -> {cellpose_env}",
        f"motion stage      -> {caiman_env}",
        f"mask stage        -> {cellpose_env}",
        f"calcium stage     -> {cellpose_env}",
        f"stim stage        -> {cellpose_env}",
        "",
        f"Current Python executable during this metadata capture:",
        sys.executable,
        "",
    ]

    (analysis_dir / "environment_used.txt").write_text("\\n".join(text_lines) + "\\n")


def write_rerun_script(
    analysis_dir: Path,
    script_snapshot_name: str | None = None,
) -> None:
    """
    Write a copy-pasteable rerun_analysis.sh script.

    It uses the script snapshot if available and replays the command from this run.
    """
    import shlex

    argv = list(sys.argv)
    if not argv:
        return

    # Prefer script snapshot path in the run folder.
    if script_snapshot_name:
        argv[0] = f"script_snapshot/{script_snapshot_name}"

    command = " ".join(shlex.quote(str(x)) for x in argv)

    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Re-run this exact analysis command from the analysis folder.",
        "# This assumes conda is installed and the required environments exist.",
        "",
        "# From this folder:",
        "#   bash rerun_analysis.sh",
        "",
        command,
        "",
    ]

    rerun_path = analysis_dir / "rerun_analysis.sh"
    rerun_path.write_text("\\n".join(lines))
    try:
        rerun_path.chmod(0o755)
    except Exception:
        pass



def write_run_settings_summary(analysis_dir: Path, args_dict: dict) -> None:
    """Write key analysis-affecting settings into a compact CSV/JSON summary."""
    key_names = [
        "script_version",
        "allow_missing_stim_file",
        "stim_expected_times",
        "stim_dir",
        "stim_file",
        "stim_channel",
        "stim_search_window_sec",
        "stim_threshold",
        "stim_max_pulse_sec",
        "extra_stim_delay_sec",
        "recording_start_sec",
        "max_data_regions",
        "baseline_region_index",
        "responder_threshold",
        "responder_metric",
        "responder_call_rule",
        "noise_sd_multiplier",
        "responder_peak_smoothing_frames",
        "responder_min_consecutive_frames",
        "min_cell_area_px",
        "min_F0",
        "max_baseline_sd",
        "max_abs_baseline_slope",
        "focus_qc_padding_px",
        "focus_qc_cv_threshold",
        "focus_qc_corr_threshold",
        "focus_qc_delta_threshold",
        "heatmap_clip_low_percentile",
        "heatmap_clip_high_percentile",
        "make_preview_movies",
        "preview_downscale",
        "preview_fps",
        "preview_max_frames",
        "ms_per_frame",
        "ds_factor",
    ]
    summary = {"script_version": SCRIPT_VERSION}
    for name in key_names:
        if name == "script_version":
            continue
        summary[name] = args_dict.get(name, "") if isinstance(args_dict, dict) else ""

    # JSON copy is safest for exact values.
    (analysis_dir / "run_settings_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    # CSV copy is easier to inspect in Excel.
    try:
        import pandas as pd
        rows = [{"parameter": k, "value": v} for k, v in summary.items()]
        pd.DataFrame(rows).to_csv(analysis_dir / "run_settings_summary.csv", index=False)
    except Exception:
        lines = ["parameter,value"]
        for k, v in summary.items():
            lines.append(f"{k},{v}")
        (analysis_dir / "run_settings_summary.csv").write_text("\n".join(lines) + "\n")

def write_run_metadata(
    analysis_dir: Path,
    args_dict: dict,
    input_files: list[Path],
    copy_script_snapshot: bool = True,
    save_shell_history: bool = True,
    shell_history_lines: int = 50,
    save_conda_envs: bool = True,
    conda_env_names: list[str] | None = None,
) -> None:
    """Write reproducibility metadata into the analysis folder."""
    # Parameters / command.
    parameters = {
        "script_version": SCRIPT_VERSION,
        "analysis_timestamp": datetime.now().isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
        "arguments": args_dict,
    }

    (analysis_dir / "parameters.json").write_text(json.dumps(parameters, indent=2, default=str))
    write_run_settings_summary(analysis_dir, args_dict or {})

    write_bash_command_files(
        analysis_dir=analysis_dir,
        save_shell_history=save_shell_history,
        shell_history_lines=shell_history_lines,
    )

    # Software versions.
    software = {
        "python": sys.version,
        "numpy": getattr(np, "__version__", "unavailable"),
        "pandas": package_version("pandas"),
        "scipy": package_version("scipy"),
        "tifffile": package_version("tifffile"),
        "matplotlib": package_version("matplotlib"),
        "openpyxl": package_version("openpyxl"),
        "cellpose": package_version("cellpose"),
        "torch": package_version("torch"),
        "caiman": package_version("caiman"),
        "imageio": package_version("imageio"),
    }
    (analysis_dir / "software_versions.json").write_text(json.dumps(software, indent=2, default=str))

    # Input file inventory.
    inventory_rows = []
    for p in input_files:
        try:
            if p.exists() and p.is_file():
                stat = p.stat()
                inventory_rows.append(
                    {
                        "file": str(p),
                        "name": p.name,
                        "size_bytes": stat.st_size,
                        "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
                        "md5": file_md5(p),
                    }
                )
        except Exception as exc:
            inventory_rows.append(
                {
                    "file": str(p),
                    "name": p.name,
                    "size_bytes": "error",
                    "modified_time": "error",
                    "md5": f"error: {exc}",
                }
            )

    try:
        import pandas as pd
        pd.DataFrame(inventory_rows).to_csv(analysis_dir / "input_files.csv", index=False)
    except Exception:
        (analysis_dir / "input_files.json").write_text(json.dumps(inventory_rows, indent=2, default=str))

    # Script snapshot.
    script_snapshot_name = None
    if copy_script_snapshot:
        try:
            script_path = Path(__file__).resolve()
            script_snapshot_name = script_path.name
            shutil.copy2(script_path, analysis_dir / "script_snapshot" / script_snapshot_name)
        except Exception as exc:
            (analysis_dir / "script_snapshot" / "script_copy_error.txt").write_text(str(exc) + "\n")

    # Environment mapping and conda exports.
    write_environment_used(analysis_dir, args_dict=args_dict)

    if save_conda_envs:
        write_conda_environment_exports(
            analysis_dir=analysis_dir,
            conda_env_names=conda_env_names or ["dcimg", "cellpose_py310", "caiman_py310"],
        )

    # Copy-pasteable rerun script.
    write_rerun_script(analysis_dir, script_snapshot_name=script_snapshot_name)



def extract_numeric_postfix(text_value: str) -> int | None:
    """
    Extract the final integer group from a filename stem.

    Examples:
        rec00001_mc -> 1
        Data1 -> 1
        Data00001 -> 1

    Leading zeros do not matter.
    """
    import re
    matches = re.findall(r"(\d+)", str(text_value))
    if not matches:
        return None
    return int(matches[-1])


def build_stim_file_map(
    recordings: list[str],
    stim_file: Path | None,
    stim_dir: Path | None,
    stim_file_glob: str = "*.mat",
    stim_match_mode: str = "name",
    allow_missing_stim_file: bool = False,
) -> dict[str, Path | None]:
    """
    Map each recording name to a DAQ stimulus file.

    If stim_file is provided, every recording uses that same file.
    If neither stim_file nor stim_dir is provided and allow_missing_stim_file=True,
    every recording is mapped to None and approximate --stim_expected_times must be used.
    If stim_dir is provided, default 'name' matching pairs by numeric postfix:
        rec00001_mc -> Data1.mat or Data00001.mat
    """
    if stim_file is not None:
        require_file(stim_file, "Stimulus timing file")
        return {rec: stim_file for rec in recordings}

    if stim_dir is None:
        if allow_missing_stim_file:
            print("\n[STIM] No DAQ/timing file supplied. Using approximate --stim_expected_times for all recordings.")
            return {rec: None for rec in recordings}
        raise ValueError("Either --stim_file or --stim_dir is required for stimulus analysis, unless --allow_missing_stim_file is enabled and --stim_expected_times are provided.")

    require_dir(stim_dir, "Stimulus DAQ directory")
    daq_files = sorted([p for p in stim_dir.glob(stim_file_glob) if p.is_file()])
    if not daq_files:
        if allow_missing_stim_file:
            print(
                f"\n[STIM] No DAQ files found in {stim_dir} using glob pattern {stim_file_glob}. "
                "Using approximate --stim_expected_times for all recordings."
            )
            return {rec: None for rec in recordings}
        raise FileNotFoundError(f"No DAQ files found in {stim_dir} using glob pattern {stim_file_glob}")

    recs = sorted(recordings)

    if stim_match_mode == "order":
        if len(daq_files) != len(recs):
            raise ValueError(
                f"Order-based DAQ matching requires same count. "
                f"Found {len(recs)} recordings and {len(daq_files)} DAQ files."
            )
        return {rec: daq for rec, daq in zip(recs, daq_files)}

    if stim_match_mode != "name":
        raise ValueError(f"Unsupported stim_match_mode: {stim_match_mode}")

    daq_by_num: dict[int, list[Path]] = {}
    ignored = []
    for daq in daq_files:
        n = extract_numeric_postfix(daq.stem)
        if n is None:
            ignored.append(daq.name)
            continue
        daq_by_num.setdefault(n, []).append(daq)

    duplicates = {n: files for n, files in daq_by_num.items() if len(files) > 1}
    if duplicates:
        lines = ["Multiple DAQ files matched the same numeric postfix:"]
        for n, files in sorted(duplicates.items()):
            lines.append(f"  {n}: " + ", ".join(str(f) for f in files))
        raise ValueError("\n".join(lines))

    mapping = {}
    missing = []
    for rec in recs:
        n = extract_numeric_postfix(rec)
        if n is None:
            missing.append(f"{rec}: no numeric postfix in recording name")
        elif n not in daq_by_num:
            missing.append(f"{rec}: expected DAQ numeric postfix {n}")
        else:
            mapping[rec] = daq_by_num[n][0]

    if missing:
        if allow_missing_stim_file:
            print("\n[STIM] Some recordings could not be matched to DAQ files; using approximate times for those recordings:")
            for msg in missing:
                print(f"  {msg}")
            for rec in recs:
                if rec not in mapping:
                    mapping[rec] = None
        else:
            raise ValueError("Could not match recordings to DAQ files:\n  " + "\n  ".join(missing))

    print("\nDAQ matching by numeric postfix:")
    for rec in recs:
        print(f"  {rec} -> {mapping[rec]}")

    if ignored:
        print("\n[WARNING] Ignored DAQ files with no numeric postfix:")
        for name in ignored:
            print(f"  {name}")

    return mapping



def clipped_middle90_frame_mask(
    time_values: np.ndarray,
    start_s: float,
    end_s: float,
) -> np.ndarray:
    """
    Return mask for middle 90% of a time window, clipped to available frame times.
    """
    t = np.asarray(time_values, dtype=float)
    if t.size == 0:
        return np.zeros(0, dtype=bool)

    start_s = float(start_s)
    end_s = float(end_s)

    if not np.isfinite(start_s) or not np.isfinite(end_s) or end_s <= start_s:
        return np.zeros_like(t, dtype=bool)

    duration = end_s - start_s
    mid_start = start_s + 0.05 * duration
    mid_end = start_s + 0.95 * duration

    return (t >= mid_start) & (t <= mid_end)


def safe_nanmean(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan
    good = np.isfinite(arr)
    if not np.any(good):
        return np.nan
    return float(np.nanmean(arr[good]))


def safe_nanmax(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan
    good = np.isfinite(arr)
    if not np.any(good):
        return np.nan
    return float(np.nanmax(arr[good]))


def safe_nanmin(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.nan
    good = np.isfinite(arr)
    if not np.any(good):
        return np.nan
    return float(np.nanmin(arr[good]))




def write_stim_csv_outputs(
    analysis_dir: Path,
    by_cell_rows: list,
    recording_summary_rows: list,
    qc_rows: list,
    responder_transition_rows: list,
    pressure_threshold_rows: list,
    detected_rows: list,
    global_region_intervals: list,
    warnings_df,
    metric_definitions: list,
    mirror_csv_dir: Path | None = None,
) -> dict:
    """
    Always write final stimulus-analysis CSV outputs before optional Excel export.

    Primary copies are saved in the run CSV folder. If mirror_csv_dir is provided,
    duplicate copies are also saved there for compatibility.
    """
    import pandas as pd

    analysis_dir = Path(analysis_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    mirror_csv_dir = Path(mirror_csv_dir) if mirror_csv_dir is not None else None
    if mirror_csv_dir is not None:
        mirror_csv_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "stim_cell_region_summary.csv": pd.DataFrame(by_cell_rows),
        "stim_recording_region_summary.csv": pd.DataFrame(recording_summary_rows),
        "stim_cell_qc.csv": pd.DataFrame(qc_rows),
        "stim_responder_transitions.csv": pd.DataFrame(responder_transition_rows),
        "stim_response_thresholds.csv": pd.DataFrame(pressure_threshold_rows),
        "stim_detected_stimuli.csv": pd.DataFrame(detected_rows),
        "stim_detected_regions.csv": pd.DataFrame(global_region_intervals),
        "stim_warnings.csv": warnings_df,
        "stim_metric_definitions.csv": pd.DataFrame(metric_definitions),
    }

    outputs = {}
    mirror_outputs = {}
    for filename, df in tables.items():
        path = analysis_dir / filename
        df.to_csv(path, index=False)
        outputs[filename] = path
        print(f"[STIM CSV] Wrote: {path}")

        if mirror_csv_dir is not None:
            mirror_path = mirror_csv_dir / filename
            df.to_csv(mirror_path, index=False)
            mirror_outputs[filename] = mirror_path
            print(f"[STIM CSV] Wrote run-level copy: {mirror_path}")

    return {"analysis_dir": outputs, "run_level_csv": mirror_outputs}


def can_write_xlsx_with_openpyxl() -> bool:
    """Return True if Excel writing with openpyxl is available."""
    try:
        import openpyxl  # noqa: F401
        return True
    except Exception:
        return False




def find_recording_mask_file(calcium_input_dir: Path, recording: str) -> Path | None:
    """Find the labeled Cellpose mask TIFF for a recording."""
    root = Path(calcium_input_dir).parent
    candidates = [
        root / f"{recording}_masks.tif",
        root / f"{recording}_masks.tiff",
    ]
    if recording.endswith("_mc"):
        base = recording[:-3]
        candidates.extend([
            root / f"{base}_mc_masks.tif",
            root / f"{base}_mc_masks.tiff",
            root / f"{base}_masks.tif",
            root / f"{base}_masks.tiff",
        ])
    for path in candidates:
        if path.exists():
            return path
    hits = sorted(root.glob(f"*{recording}*_masks.tif*"))
    return hits[0] if hits else None


def find_recording_template_image(calcium_input_dir: Path, recording: str) -> Path | None:
    """Find a motion-correction template or max-projection image for spatial overlays."""
    root = Path(calcium_input_dir).parent
    base = recording[:-3] if recording.endswith("_mc") else recording
    candidates = [
        root / "template" / f"{recording}_template.png",
        root / "template" / f"{base}_template.png",
        root / f"{recording}_maxproj.tif",
        root / f"{recording}_maxproj.tiff",
        root / f"{base}_maxproj.tif",
        root / f"{base}_maxproj.tiff",
    ]
    for path in candidates:
        if path.exists():
            return path
    hits = sorted((root / "template").glob(f"*{base}*template*.png")) if (root / "template").exists() else []
    if hits:
        return hits[0]
    hits = sorted(root.glob(f"*{base}*maxproj.tif*"))
    return hits[0] if hits else None


def load_grayscale_image_for_overlay(image_path: Path, target_shape: tuple[int, int]) -> np.ndarray:
    """Load template/max-projection image as 2D grayscale and resize if needed."""
    arr = tifffile.imread(image_path) if image_path.suffix.lower() in {".tif", ".tiff"} else None
    if arr is None:
        try:
            import imageio.v2 as imageio
            arr = imageio.imread(image_path)
        except Exception:
            import matplotlib.image as mpimg
            arr = mpimg.imread(image_path)
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., :3].mean(axis=2)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
        if arr.ndim > 2:
            arr = arr[0]
    arr = np.asarray(arr, dtype=float)
    if arr.shape != target_shape:
        try:
            from scipy.ndimage import zoom
            zoom_y = target_shape[0] / arr.shape[0]
            zoom_x = target_shape[1] / arr.shape[1]
            arr = zoom(arr, (zoom_y, zoom_x), order=1)
        except Exception:
            out = np.zeros(target_shape, dtype=float)
            h = min(target_shape[0], arr.shape[0])
            w = min(target_shape[1], arr.shape[1])
            out[:h, :w] = arr[:h, :w]
            arr = out
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(target_shape, dtype=float)
    lo, hi = np.percentile(finite, [1, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if hi <= lo:
        return np.zeros(target_shape, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0, 1)



def find_recording_movie_file(calcium_input_dir: Path, recording: str) -> Path | None:
    """Find the motion-corrected TIFF used for calcium extraction."""
    root = Path(calcium_input_dir).parent
    base = recording[:-3] if recording.endswith("_mc") else recording
    candidates = [
        root / f"{recording}.tif",
        root / f"{recording}.tiff",
        root / f"{recording}_mc.tif",
        root / f"{recording}_mc.tiff",
        root / f"{base}_mc.tif",
        root / f"{base}_mc.tiff",
        root / f"{base}.tif",
        root / f"{base}.tiff",
    ]
    for path in candidates:
        if path.exists():
            return path
    hits = sorted(root.glob(f"*{base}*_mc.tif*"))
    if hits:
        return hits[0]
    hits = sorted(root.glob(f"*{recording}*.tif*"))
    hits = [h for h in hits if "mask" not in h.name and "maxproj" not in h.name]
    return hits[0] if hits else None


def fft_phase_correlation_shift(reference: np.ndarray, moving: np.ndarray) -> tuple[float, float, float]:
    """
    Estimate integer-pixel shift needed to align moving to reference.

    Returns dy, dx, peak_score. Positive dy/dx means the moving image is shifted
    down/right relative to reference and should be shifted up/left to align.
    """
    ref = np.asarray(reference, dtype=float)
    mov = np.asarray(moving, dtype=float)
    if ref.shape != mov.shape or ref.ndim != 2 or min(ref.shape) < 4:
        return np.nan, np.nan, np.nan
    good = np.isfinite(ref) & np.isfinite(mov)
    if np.count_nonzero(good) < 16:
        return np.nan, np.nan, np.nan
    ref = np.where(np.isfinite(ref), ref, np.nanmedian(ref[good]))
    mov = np.where(np.isfinite(mov), mov, np.nanmedian(mov[good]))
    ref = ref - np.mean(ref)
    mov = mov - np.mean(mov)
    if float(np.std(ref)) <= 0 or float(np.std(mov)) <= 0:
        return np.nan, np.nan, np.nan
    # Hann window reduces edge artifacts in small cell crops.
    wy = np.hanning(ref.shape[0])[:, None]
    wx = np.hanning(ref.shape[1])[None, :]
    win = wy * wx
    ref = ref * win
    mov = mov * win
    f_ref = np.fft.fft2(ref)
    f_mov = np.fft.fft2(mov)
    cross = f_mov * np.conj(f_ref)
    denom = np.abs(cross)
    cross = cross / np.maximum(denom, 1e-12)
    corr = np.fft.ifft2(cross)
    corr_abs = np.abs(corr)
    max_pos = np.unravel_index(int(np.argmax(corr_abs)), corr_abs.shape)
    dy = float(max_pos[0])
    dx = float(max_pos[1])
    h, w = ref.shape
    if dy > h / 2:
        dy -= h
    if dx > w / 2:
        dx -= w
    peak = float(corr_abs[max_pos] / (np.nanmean(corr_abs) + 1e-12))
    return dy, dx, peak


def mean_tiff_frames(tiff_path: Path, frame_indices: np.ndarray, max_frames: int = 80) -> np.ndarray | None:
    """Read selected frames from a TIFF and return their mean image."""
    frame_indices = np.asarray(frame_indices, dtype=int)
    frame_indices = frame_indices[frame_indices >= 0]
    if frame_indices.size == 0:
        return None
    if max_frames and frame_indices.size > max_frames:
        pick = np.linspace(0, frame_indices.size - 1, max_frames).astype(int)
        frame_indices = frame_indices[pick]
    with tifffile.TiffFile(tiff_path) as tif:
        n_pages = len(tif.pages)
        frame_indices = frame_indices[frame_indices < n_pages]
        if frame_indices.size == 0:
            return None
        acc = None
        for idx in frame_indices:
            frame = tif.pages[int(idx)].asarray().astype(float, copy=False)
            if acc is None:
                acc = np.zeros_like(frame, dtype=float)
            acc += frame
    return acc / float(frame_indices.size) if acc is not None else None


def bbox_from_mask(mask: np.ndarray, padding: int, shape: tuple[int, int]) -> tuple[int, int, int, int] | None:
    """Return padded bounding box as y0,y1,x0,x1."""
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    y0 = max(0, int(np.min(ys)) - int(padding))
    y1 = min(shape[0], int(np.max(ys)) + int(padding) + 1)
    x0 = max(0, int(np.min(xs)) - int(padding))
    x1 = min(shape[1], int(np.max(xs)) + int(padding) + 1)
    if (y1 - y0) < 4 or (x1 - x0) < 4:
        return None
    return y0, y1, x0, x1


def dilate_binary_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Small dependency-free binary dilation for ROI focus QC."""
    out = np.asarray(mask, dtype=bool).copy()
    for _ in range(max(0, int(iterations))):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        acc = np.zeros_like(out, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                acc |= padded[1 + dy:1 + dy + out.shape[0], 1 + dx:1 + dx + out.shape[1]]
        out = acc
    return out




def expand_cell_mask_by_fraction(mask: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    """Expand a single-cell mask by roughly a fraction of its equivalent radius.

    Used only for representative composite visualization. Quantitative extraction
    still uses the original Cellpose ROI mask.
    """
    m = np.asarray(mask, dtype=bool)
    area = int(np.count_nonzero(m))
    if area <= 0:
        return m
    # Equivalent radius in pixels; 10% expansion means about 10% of this radius.
    radius = float(np.sqrt(area / np.pi))
    iterations = max(1, int(round(float(fraction) * radius)))
    return dilate_binary_mask(m, iterations=iterations)


def bool_from_table_value(x) -> bool:
    """Robustly parse TRUE/FALSE values read from CSV rows."""
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if x is None:
        return False
    try:
        if isinstance(x, float) and np.isnan(x):
            return False
    except Exception:
        pass
    return str(x).strip().lower() in {"true", "1", "yes", "y", "t"}



def row_has_qc_risk(row: dict) -> bool:
    """Combined QC-risk flag used for orange/dotted plot outlines.

    Uses automatic conservative QC when present, but remains backward-compatible
    with older tables that only have focus/floating-cell flags.
    """
    for key in (
        "conservative_qc_risk",
        "conservative_focus_corr_risk",
        "conservative_baseline_instability_risk",
        "conservative_area_outlier_risk",
        "recording_baseline_unstable",
        "floating_cell_artifact_risk",
        "focus_instability_suspicious",
    ):
        try:
            if key in row and bool_from_table_value(row.get(key, False)):
                return True
        except Exception:
            pass
    return False


def normalize_recording_name_for_match(name) -> str:
    """Normalize recording/movie names for robust file matching."""
    s = Path(str(name)).stem
    for suffix in ("_DFoverF_wide", "_F_wide", "_F0", "_cell_metrics", "_masks", "_mask", "_template"):
        if s.endswith(suffix):
            s = s[: -len(suffix)]
    return s


def candidate_movie_stems_for_recording(recording: str) -> list[str]:
    """Candidate movie stems, preserving rec00010_mc versus rec00010 distinction."""
    rec = normalize_recording_name_for_match(recording)
    stems = [rec]
    if rec.endswith("_mc"):
        stems.append(rec[:-3])
    else:
        stems.append(rec + "_mc")
    out = []
    for x in stems:
        if x and x not in out:
            out.append(x)
    return out


def find_recording_specific_movie_file(calcium_input_dir: Path, recording: str) -> Path | None:
    """Find the motion-corrected TIFF for exactly this recording."""
    root = Path(calcium_input_dir)
    search_dirs = [
        root,
        root.parent,
        root.parent.parent,
        root.parent / "motion_corrected",
        root / "motion_corrected",
    ]
    seen_dirs = []
    for d in search_dirs:
        try:
            d = Path(d)
            if d.exists() and d not in seen_dirs:
                seen_dirs.append(d)
        except Exception:
            pass

    candidates = []
    stems = candidate_movie_stems_for_recording(recording)
    for d in seen_dirs:
        for stem in stems:
            for pat in (f"{stem}.tif", f"{stem}.tiff"):
                candidates.extend(d.glob(pat))

    if not candidates:
        for d in seen_dirs:
            try:
                for p in list(d.rglob("*.tif")) + list(d.rglob("*.tiff")):
                    if p.stem in stems:
                        candidates.append(p)
            except Exception:
                pass

    unique = []
    for p in candidates:
        if p not in unique:
            unique.append(p)

    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        rec_norm = normalize_recording_name_for_match(recording)
        exact = [p for p in unique if p.stem == rec_norm]
        if len(exact) == 1:
            return exact[0]
        unique = sorted(unique, key=lambda p: (len(str(p)), str(p)))
        return unique[0]
    return None


def assert_movie_matches_recording(movie_path: Path | None, recording: str) -> None:
    """Hard-stop if a selected movie path does not match the current recording."""
    if movie_path is None:
        raise FileNotFoundError(f"No recording-specific movie found for {recording}")
    movie_stem = Path(movie_path).stem
    allowed = set(candidate_movie_stems_for_recording(recording))
    if movie_stem not in allowed:
        raise RuntimeError(
            f"Recording/movie mismatch: recording {recording} selected movie {movie_path}. "
            f"Allowed stems: {sorted(allowed)}"
        )


def row_is_responder(row: dict, responder_threshold: float | None = None) -> bool:
    """Return the primary responder call for plotting/counting.

    If responder_primary exists, use ONLY responder_primary. responder_either is
    audit/borderline only and must not drive responder outlines or counts.
    """
    if "responder_primary" in row:
        return bool_from_table_value(row.get("responder_primary"))
    if "primary_responder" in row:
        return bool_from_table_value(row.get("primary_responder"))
    if "responder_both" in row:
        return bool_from_table_value(row.get("responder_both"))

    # Conservative fallback for old tables without official responder columns.
    if responder_threshold is not None:
        val = safe_float(row.get("spike_safe_peak_above_baseline_DFoverF", np.nan))
        if not np.isfinite(val):
            val = safe_float(row.get("spike_safe_peak_DFoverF", np.nan))
        if not np.isfinite(val):
            val = safe_float(row.get("peak_DFoverF", np.nan))
        return bool(np.isfinite(val) and val >= float(responder_threshold))
    return False


def row_is_borderline_audit(row: dict) -> bool:
    """True for either-only/borderline calls that are not primary responders."""
    try:
        return bool_from_table_value(row.get("responder_either", False)) and not row_is_responder(row)
    except Exception:
        return False


def build_threshold_boundary_for_region(masks: np.ndarray, region_rows: list[dict], responder_threshold: float) -> tuple[np.ndarray, list[int]]:
    """Return original-ROI boundary for cells called responders in the table."""
    boundary = np.zeros(masks.shape, dtype=bool)
    responder_ids: list[int] = []
    for row in region_rows:
        try:
            cell_id = int(row.get("cell_id"))
        except Exception:
            continue
        passed_qc = bool_from_table_value(row.get("passed_qc", True))
        if not passed_qc or not row_is_responder(row, responder_threshold=responder_threshold):
            continue
        cell_mask = masks == cell_id
        if not np.any(cell_mask):
            continue
        boundary |= mask_boundary(cell_mask.astype(np.uint8))
        responder_ids.append(cell_id)
    return boundary, responder_ids


def plot_label_contours(ax, masks: np.ndarray, cell_ids=None, *, colors="#4A90E2", linewidths=0.8, linestyles="solid", alpha=1.0):
    cell_ids = _v472_positive_cell_ids(cell_ids)
    if not cell_ids:
        return
    """Draw each labeled ROI as an independent contour.

    Important: do not contour (masks > 0) for all ROIs together, because
    adjacent/touching cells then appear as one merged outline. Looping through
    label IDs preserves visual separation even when Cellpose labels touch.
    """
    arr = np.asarray(masks)
    if arr.ndim != 2:
        return
    if cell_ids is None:
        ids = [int(x) for x in np.unique(arr) if int(x) > 0]
    else:
        ids = []
        for x in cell_ids:
            try:
                cid = int(x)
            except Exception:
                continue
            if cid > 0:
                ids.append(cid)
    for cid in ids:
        cell_mask = (arr == cid)
        if not np.any(cell_mask):
            continue
        try:
            ax.contour(
                cell_mask.astype(float),
                levels=[0.5],
                colors=colors,
                linewidths=linewidths,
                linestyles=linestyles,
                alpha=alpha,
            )
        except Exception:
            # Tiny or degenerate masks may occasionally have no drawable contour.
            continue


def normalized_laplacian_focus_metric(crop: np.ndarray, analysis_mask: np.ndarray) -> float:
    """
    Brightness-normalized sharpness metric for focus-instability QC.

    This intentionally does not use the absolute fluorescence level. The crop is
    percentile-normalized before measuring Laplacian energy, so a uniform calcium
    brightness increase should have minimal effect. A floating/dead cell moving
    through focus should change edge sharpness and therefore this metric.
    """
    arr = np.asarray(crop, dtype=float)
    m = np.asarray(analysis_mask, dtype=bool)
    if arr.ndim != 2 or m.shape != arr.shape or min(arr.shape) < 3 or np.count_nonzero(m) < 9:
        return np.nan
    vals = arr[m & np.isfinite(arr)]
    if vals.size < 9:
        return np.nan
    lo, hi = np.nanpercentile(vals, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.nan
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    arr[~np.isfinite(arr)] = 0
    lap = np.zeros_like(arr, dtype=float)
    lap[1:-1, 1:-1] = (
        -4.0 * arr[1:-1, 1:-1]
        + arr[:-2, 1:-1]
        + arr[2:, 1:-1]
        + arr[1:-1, :-2]
        + arr[1:-1, 2:]
    )
    inner = np.zeros_like(m, dtype=bool)
    inner[1:-1, 1:-1] = m[1:-1, 1:-1]
    v = lap[inner]
    v = v[np.isfinite(v)]
    if v.size < 9:
        return np.nan
    return float(np.nanmean(np.abs(v)))


def pearson_corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    """Finite Pearson correlation, or NaN if insufficient data."""
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    good = np.isfinite(a) & np.isfinite(b)
    if np.count_nonzero(good) < 4:
        return np.nan
    a = a[good] - float(np.nanmean(a[good]))
    b = b[good] - float(np.nanmean(b[good]))
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if den <= 0 or not np.isfinite(den):
        return np.nan
    return float(np.sum(a * b) / den)


def compute_focus_artifact_qc_for_recording(
    recording: str,
    calcium_input_dir: Path,
    df_dff,
    region_intervals_to_use: list[dict],
    baseline_region_index: int,
    padding_px: int = 6,
    focus_cv_threshold: float = 0.25,
    focus_corr_threshold: float = 0.5,
    focus_delta_threshold: float = 0.25,
) -> dict[tuple[int, int], dict]:
    """
    QC for floating/dead-cell artifacts using focus instability, not brightness level.

    A floating cell may change apparent fluorescence because it moves through Z-focus.
    This computes a brightness-normalized Laplacian sharpness trace for each ROI and
    flags cells/regions where DF/F tracks focus changes. It does not estimate local
    X/Y cell motion from calcium intensity patterns.
    """
    movie_path = find_recording_movie_file(calcium_input_dir, recording)
    mask_path = find_recording_mask_file(calcium_input_dir, recording)
    if movie_path is None or mask_path is None:
        print(f"[WARNING] Focus-artifact QC skipped for {recording}: movie or mask not found.")
        return {}

    masks = np.asarray(tifffile.imread(mask_path))
    if masks.ndim > 2:
        masks = np.squeeze(masks)
        if masks.ndim > 2:
            masks = masks[0]
    masks = masks.astype(np.int64, copy=False)
    if masks.ndim != 2 or int(np.nanmax(masks)) <= 0:
        print(f"[WARNING] Focus-artifact QC skipped for {recording}: invalid mask {mask_path}.")
        return {}

    time_s = df_dff["time_s"].to_numpy(dtype=float)
    cell_ids = [int(c.replace("cell_", "")) for c in df_dff.columns if str(c).startswith("cell_")]
    n_frames_df = int(len(time_s))

    cell_info = {}
    for cid in cell_ids:
        cell_mask = masks == cid
        box = bbox_from_mask(cell_mask, padding_px, masks.shape)
        if box is None:
            continue
        y0, y1, x0, x1 = box
        crop_mask = cell_mask[y0:y1, x0:x1]
        # Include a small peri-cell ring so the metric captures edge sharpness/focus.
        analysis_mask = dilate_binary_mask(crop_mask, iterations=2)
        cell_info[cid] = (y0, y1, x0, x1, analysis_mask)

    focus_traces = {cid: np.full(n_frames_df, np.nan, dtype=float) for cid in cell_ids}
    try:
        with tifffile.TiffFile(movie_path) as tif:
            n_pages = min(len(tif.pages), n_frames_df)
            for frame_idx in range(n_pages):
                frame = tif.pages[frame_idx].asarray()
                for cid, info in cell_info.items():
                    y0, y1, x0, x1, analysis_mask = info
                    focus_traces[cid][frame_idx] = normalized_laplacian_focus_metric(
                        frame[y0:y1, x0:x1], analysis_mask
                    )
                if (frame_idx + 1) % 200 == 0 or (frame_idx + 1) == n_pages:
                    print(f"  Focus QC {recording}: processed frame {frame_idx + 1}/{n_pages}")
    except Exception as exc:
        print(f"[WARNING] Focus-artifact QC failed while reading {movie_path}: {exc}")
        return {}

    baseline_region = next((r for r in region_intervals_to_use if int(r["region_index"]) == int(baseline_region_index)), region_intervals_to_use[0])
    baseline_mask = (time_s >= baseline_region["region_start_s"]) & (time_s < baseline_region["region_end_s"])

    # Recording-level global X/Y shift remains useful QC. It is not a per-cell motion call.
    global_shift_by_region = {}
    try:
        baseline_idx = np.where(baseline_mask)[0]
        baseline_ref = mean_tiff_frames(movie_path, baseline_idx, max_frames=80)
        if baseline_ref is not None:
            for region in region_intervals_to_use:
                region_index = int(region["region_index"])
                mid_mask = (time_s >= region["middle90_start_s"]) & (time_s < region["middle90_end_s"])
                region_img = mean_tiff_frames(movie_path, np.where(mid_mask)[0], max_frames=80)
                if region_img is None:
                    continue
                gy, gx, gs = fft_phase_correlation_shift(baseline_ref, region_img)
                gm = float(np.hypot(gy, gx)) if np.isfinite(gy) and np.isfinite(gx) else np.nan
                global_shift_by_region[region_index] = (gy, gx, gm, gs)
    except Exception as exc:
        print(f"[WARNING] Global-shift submetric skipped for {recording}: {exc}")

    out: dict[tuple[int, int], dict] = {}
    for region in region_intervals_to_use:
        region_index = int(region["region_index"])
        mid_mask = (time_s >= region["middle90_start_s"]) & (time_s < region["middle90_end_s"])
        gy, gx, gm, gs = global_shift_by_region.get(region_index, (np.nan, np.nan, np.nan, np.nan))

        for cid in cell_ids:
            focus = focus_traces.get(cid, np.full(n_frames_df, np.nan))
            f_base = focus[baseline_mask]
            f_region = focus[mid_mask]
            base_med = float(np.nanmedian(f_base)) if np.any(np.isfinite(f_base)) else np.nan
            reg_med = float(np.nanmedian(f_region)) if np.any(np.isfinite(f_region)) else np.nan
            reg_mean = float(np.nanmean(f_region)) if np.any(np.isfinite(f_region)) else np.nan
            reg_sd = float(np.nanstd(f_region)) if np.any(np.isfinite(f_region)) else np.nan
            if np.isfinite(reg_mean) and abs(reg_mean) > 1e-12:
                reg_cv = float(reg_sd / abs(reg_mean))
            else:
                reg_cv = np.nan
            if np.isfinite(base_med) and abs(base_med) > 1e-12 and np.isfinite(reg_med):
                delta_frac = float((reg_med - base_med) / abs(base_med))
            else:
                delta_frac = np.nan
            dff = df_dff[f"cell_{cid}"].to_numpy(dtype=float) if f"cell_{cid}" in df_dff.columns else np.full(n_frames_df, np.nan)
            corr = pearson_corr_safe(dff[mid_mask], focus[mid_mask])

            focus_unstable = bool(
                (np.isfinite(reg_cv) and reg_cv >= focus_cv_threshold)
                or (np.isfinite(delta_frac) and abs(delta_frac) >= focus_delta_threshold)
            )
            corr_high = bool(np.isfinite(corr) and abs(corr) >= focus_corr_threshold)
            floating_risk = bool(focus_unstable and corr_high)

            out[(region_index, cid)] = {
                "focus_qc_method": "brightness_normalized_laplacian_focus_trace",
                "focus_qc_movie": str(movie_path),
                "global_shift_y_px": gy,
                "global_shift_x_px": gx,
                "global_shift_magnitude_px": gm,
                "global_shift_peak_score": gs,
                "focus_metric_baseline_median": base_med,
                "focus_metric_region_median": reg_med,
                "focus_metric_region_cv": reg_cv,
                "focus_metric_delta_frac_vs_baseline": delta_frac,
                "dff_focus_correlation": corr,
                "focus_cv_threshold": focus_cv_threshold,
                "focus_corr_threshold": focus_corr_threshold,
                "focus_delta_threshold": focus_delta_threshold,
                "focus_instability_suspicious": focus_unstable,
                "floating_cell_artifact_risk": floating_risk,
                    "conservative_focus_corr_risk": conservative_focus_corr_risk if "conservative_focus_corr_risk" in locals() else False,
                    "baseline_instability_ratio": baseline_instability_ratio_tmp if "baseline_instability_ratio_tmp" in locals() else np.nan,
                    "conservative_baseline_instability_risk": conservative_baseline_instability_risk if "conservative_baseline_instability_risk" in locals() else False,
                    "conservative_area_outlier_risk": conservative_area_outlier_risk if "conservative_area_outlier_risk" in locals() else False,
                    "conservative_qc_risk": conservative_qc_risk if "conservative_qc_risk" in locals() else False,
            }
    return out



def make_same_grid_background_for_spatial_overlay(
    calcium_input_dir: Path,
    recording: str,
    target_shape: tuple[int, int],
) -> tuple[np.ndarray, str]:
    """
    Return a background image on the exact same pixel grid as the labeled masks.

    Spatial overlays must not use matplotlib-rendered template PNGs if their pixel
    dimensions differ from the mask, because those PNGs can contain figure padding
    and axis scaling. The safest background is a projection from the actual TIFF
    used for calcium extraction, followed by a same-shape max projection TIFF.
    """
    # Safest: actual movie used for calcium extraction/masking, projected in the
    # same coordinate system as the mask.
    movie_path = find_recording_movie_file(calcium_input_dir, recording)
    if movie_path is not None:
        try:
            with tifffile.TiffFile(movie_path) as tif:
                n_pages = len(tif.pages)
                if n_pages > 0:
                    # Use up to 200 frames across the movie to make a clean base image
                    # without loading huge TIFFs into memory.
                    pick = np.linspace(0, n_pages - 1, min(200, n_pages)).astype(int)
                    acc = None
                    for idx in pick:
                        frame = tif.pages[int(idx)].asarray().astype(float, copy=False)
                        if frame.shape != target_shape:
                            raise ValueError(f"movie frame shape {frame.shape} != mask shape {target_shape}")
                        if acc is None:
                            acc = np.zeros(target_shape, dtype=float)
                        acc += frame
                    if acc is not None:
                        return normalize_overlay_background(acc / float(len(pick)), target_shape), movie_path.name
        except Exception as exc:
            print(f"[WARNING] Could not use same-grid movie background for {recording}: {exc}")

    # Next best: a max-projection TIFF generated from the same image stack.
    root = Path(calcium_input_dir).parent
    base = recording[:-3] if recording.endswith("_mc") else recording
    for path in [
        root / f"{recording}_maxproj.tif",
        root / f"{recording}_maxproj.tiff",
        root / f"{base}_maxproj.tif",
        root / f"{base}_maxproj.tiff",
    ]:
        if path.exists():
            try:
                arr = np.asarray(tifffile.imread(path), dtype=float)
                if arr.ndim > 2:
                    arr = np.squeeze(arr)
                    if arr.ndim > 2:
                        arr = arr[0]
                if arr.shape == target_shape:
                    return normalize_overlay_background(arr, target_shape), path.name
                print(f"[WARNING] Skipping non-matching maxproj background {path.name}: shape {arr.shape} != mask shape {target_shape}")
            except Exception as exc:
                print(f"[WARNING] Could not read maxproj background {path}: {exc}")

    # Last resort: use template only if it is already exactly the same pixel size.
    template_path = find_recording_template_image(calcium_input_dir, recording)
    if template_path is not None:
        try:
            arr = tifffile.imread(template_path) if template_path.suffix.lower() in {".tif", ".tiff"} else None
            if arr is None:
                try:
                    import imageio.v2 as imageio
                    arr = imageio.imread(template_path)
                except Exception:
                    import matplotlib.image as mpimg
                    arr = mpimg.imread(template_path)
            arr = np.asarray(arr)
            if arr.ndim == 3:
                arr = arr[..., :3].mean(axis=2)
            if arr.ndim > 2:
                arr = np.squeeze(arr)
                if arr.ndim > 2:
                    arr = arr[0]
            if arr.shape == target_shape:
                return normalize_overlay_background(np.asarray(arr, dtype=float), target_shape), template_path.name
            print(
                f"[WARNING] Skipping template PNG for spatial overlay because it is not on the mask pixel grid: "
                f"{template_path.name} shape {arr.shape} != mask shape {target_shape}. "
                "Using blank background instead to avoid misalignment."
            )
        except Exception as exc:
            print(f"[WARNING] Could not read template background {template_path}: {exc}")

    return np.zeros(target_shape, dtype=float), "blank background; no same-grid image found"


def normalize_overlay_background(arr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Normalize a same-grid 2D background image to 0..1 for display."""
    arr = np.asarray(arr, dtype=float)
    if arr.shape != target_shape:
        raise ValueError(f"background shape {arr.shape} != target shape {target_shape}")
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(target_shape, dtype=float)
    lo, hi = np.percentile(finite, [1, 99.5])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(finite)), float(np.nanmax(finite))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(target_shape, dtype=float)
    return np.clip((arr - lo) / (hi - lo), 0, 1)



def add_cell_id_labels_to_spatial_axis(ax, masks: np.ndarray, cell_ids: Iterable[int], fontsize: float = 5.0) -> None:
    """Draw cell ID labels at mask centroids for traceability back to output tables."""
    try:
        ids = sorted({int(c) for c in cell_ids if int(c) > 0})
    except Exception:
        ids = []
    for cell_id in ids:
        yy, xx = np.where(masks == int(cell_id))
        if yy.size == 0:
            continue
        x = float(np.nanmean(xx))
        y = float(np.nanmean(yy))
        ax.text(
            x,
            y,
            str(cell_id),
            ha="center",
            va="center",
            fontsize=fontsize,
            color="black",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "black", "linewidth": 0.25, "alpha": 0.75},
            zorder=20,
        )



def robust_heatmap_limits(
    values,
    low_percentile: float = 5.0,
    high_percentile: float = 95.0,
    include_zero: bool = True,
    include_threshold: float | None = None,
    min_span: float = 1e-6,
) -> tuple[float, float]:
    """
    Robust display limits for spatial heatmaps.

    Uses percentile clipping instead of raw min/max so one extreme cell cannot
    dominate the color scale. Values outside the returned range are still shown
    using the colormap's under/over colors and the colorbar uses extend arrows.
    """
    vals = np.asarray(values, dtype=float).reshape(-1)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0

    lo_p = float(np.clip(low_percentile, 0.0, 49.0))
    hi_p = float(np.clip(high_percentile, 51.0, 100.0))
    if hi_p <= lo_p:
        lo_p, hi_p = 5.0, 95.0

    vmin = float(np.nanpercentile(vals, lo_p))
    vmax = float(np.nanpercentile(vals, hi_p))

    if include_zero:
        vmin = min(vmin, 0.0)
        vmax = max(vmax, 0.0)

    if include_threshold is not None and np.isfinite(include_threshold):
        thr = float(include_threshold)
        # Keep the threshold visible inside the color scale, with a little room
        # around it so the threshold line is not pinned to the colorbar edge.
        vmin = min(vmin, thr * 0.8 if thr >= 0 else thr * 1.2)
        vmax = max(vmax, thr * 1.2 if thr >= 0 else thr * 0.8)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        center = float(np.nanmedian(vals)) if vals.size else 0.0
        span = max(abs(center) * 0.1, min_span, 1.0)
        vmin, vmax = center - span, center + span

    if (vmax - vmin) < min_span:
        mid = 0.5 * (vmax + vmin)
        vmin, vmax = mid - 0.5 * min_span, mid + 0.5 * min_span

    return float(vmin), float(vmax)


def set_clipped_colorbar_label(cbar, base_label: str, vmin: float, vmax: float, low_pct: float, high_pct: float) -> None:
    """Label colorbar limits explicitly as clipped bounds."""
    cbar.set_label(f"{base_label} (clipped {low_pct:g}-{high_pct:g}th percentile)")
    try:
        ticks = list(cbar.get_ticks())
        ticks = [t for t in ticks if np.isfinite(t) and vmin < t < vmax]
        ticks = [vmin] + ticks + [vmax]
        cbar.set_ticks(ticks)
        labels = []
        for i, t in enumerate(ticks):
            if i == 0:
                labels.append(f"≤{vmin:.3g}")
            elif i == len(ticks) - 1:
                labels.append(f"≥{vmax:.3g}")
            else:
                labels.append(f"{t:.3g}")
        cbar.set_ticklabels(labels)
    except Exception:
        pass


def save_png_montage_from_paths(
    image_paths: list[Path],
    out_path: Path,
    title: str,
    ncols: int = 2,
    dpi: int = 180,
) -> None:
    """Create a simple 2x2-style montage from already-saved PNG panels."""
    image_paths = [Path(x) for x in image_paths if x is not None and Path(x).exists()]
    if not image_paths:
        return
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        n = len(image_paths)
        ncols = max(1, int(ncols))
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(7.5 * ncols, 7.0 * nrows), facecolor="white")
        axes_arr = np.asarray(axes).reshape(-1)
        for ax, img_path in zip(axes_arr, image_paths):
            img = mpimg.imread(str(img_path))
            ax.imshow(img)
            ax.set_axis_off()
            # No filename-derived title. Condition labels are applied by specific montage builders.
            label = ""
            if label:
                (_v457_simple_title(ax, label) if str(label).strip() else None)
        for ax in axes_arr[len(image_paths):]:
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        print(f"Saved montage: {out_path}")
    except Exception as exc:
        print(f"[WARNING] Could not save montage {out_path}: {exc}")



def save_baseline_vs_max_2row_montage_from_pair_paths(
    pair_paths: list[Path],
    out_path: Path,
    title: str,
    dpi: int = 180,
) -> None:
    """
    Create a 2 x N montage from already-saved baseline-vs-max side-by-side PNGs.

    The source images are expected to be side-by-side pairs:
      left half  = baseline representative composite
      right half = region/max-expression composite

    Output layout:
      top row    = all baseline panels, ordered by region
      bottom row = all region/max-expression panels, ordered by region

    This makes all time regions directly comparable while keeping baseline and
    stimulated/max panels aligned by column.
    """
    pair_paths = [Path(x) for x in pair_paths if x is not None and Path(x).exists()]
    if not pair_paths:
        return
    try:
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import re

        baseline_panels = []
        max_panels = []
        labels = []
        for pair_path in pair_paths:
            img = mpimg.imread(str(pair_path))
            if img.ndim < 2:
                continue
            h, w = img.shape[:2]
            if w < 2:
                continue
            mid = w // 2
            baseline_panels.append(img[:, :mid, ...])
            max_panels.append(img[:, mid:, ...])

            stem = pair_path.stem
            # Make a compact region label from filenames like:
            # rec00010_mc_cell_frame_composite_03_stim2_to_stim3_baseline_vs_max
            m = re.search(r"cell_frame_composite_\d+_(.+?)_baseline_vs_max", stem)
            labels.append(m.group(1) if m else stem)

        n = len(baseline_panels)
        if n == 0:
            return

        fig, axes = plt.subplots(2, n, figsize=(5.0 * n, 9.0), facecolor="0.08")
        axes = np.asarray(axes)
        if axes.ndim == 1:
            axes = axes.reshape(2, 1)

        for i in range(n):
            for row_i, panel, row_label in [
                (0, baseline_panels[i], "baseline"),
                (1, max_panels[i], "region max"),
            ]:
                ax = axes[row_i, i]
                ax.set_facecolor("0.08")
                ax.imshow(panel)
                ax.set_axis_off()
                if row_i == 0:
                    _v457_simple_title(ax, labels[i])
                if i == 0:
                    ax.text(
                        -0.02,
                        0.5,
                        row_label,
                        transform=ax.transAxes,
                        rotation=90,
                        va="center",
                        ha="right",
                        fontsize=10,
                        color="white",
                        fontweight="bold",
                    )
        fig.tight_layout()
        fig.savefig(out_path, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Saved 2xN baseline-vs-max montage: {out_path}")
    except Exception as exc:
        print(f"[WARNING] Could not save 2xN baseline-vs-max montage {out_path}: {exc}")





def add_outline_legend(ax, entries=None, loc="lower right", fontsize=7):
    """Add compact legend for outline overlays."""
    try:
        from matplotlib.patches import Patch
        if entries is None:
            entries = []
        handles = []
        for label, edgecolor, linestyle, linewidth in entries:
            handles.append(
                Patch(
                    facecolor="none",
                    edgecolor=edgecolor,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    label=label,
                )
            )
        if handles:
            ax.legend(handles=handles, loc=loc, fontsize=fontsize, framealpha=0.80)
    except Exception:
        pass


def set_heatmap_cell_id_ticks(ax, sorted_cell_ids, max_fontsize=6):
    """Label every heatmap row with the actual CellID, not row index."""
    try:
        ids = [int(x) for x in list(sorted_cell_ids)]
        n = len(ids)
        ax.set_yticks(np.arange(n))
        ax.set_yticklabels([str(x) for x in ids], fontsize=(max_fontsize if n <= 50 else 4))
        ax.set_ylabel("Cell ID\\n(sorted by peak ΔF/F)")
    except Exception:
        pass

def stimulus_mode_for_region(region: dict, baseline_region_index: int | None = None) -> str:
    """Classify whether a region represents detected US stimulation, a control time window, or drug.

    Important: labels such as stim3_to_drug are pre-drug windows and must NOT be
    classified as drug_window just because the word "drug" appears in the label.
    """
    try:
        if baseline_region_index is not None and int(region.get("region_index", -999)) == int(baseline_region_index):
            return "baseline"
    except Exception:
        pass

    label = str(region.get("region_label", "")).strip().lower()

    # Only true post-drug / AITC labels count as drug windows.
    drug_labels = {"drug", "aitc", "post_drug", "post-drug", "drug_window", "aitc_window"}
    if label in drug_labels or label.startswith("drug_") or label.startswith("aitc_"):
        return "drug_window"

    if bool(region.get("pulses_detected", False)):
        return "detected_stimulus"

    return "time_window_control"



def is_eligible_us_stimulus_region(region: dict, baseline_region_index: int | None = None) -> bool:
    """True only when the region contains a detected ultrasound pulse and is not baseline/drug."""
    return stimulus_mode_for_region(region, baseline_region_index) == "detected_stimulus"


def display_label_with_control_context(region_index, region_label, plot_region_labels, baseline_region_index, stimulus_mode: str | None = None) -> str:
    """Use clean condition labels, adding '(time window)' only for no-pulse control stimulus windows."""
    base = display_label_for_region(region_index, region_label, plot_region_labels, baseline_region_index)
    if stimulus_mode == "time_window_control":
        return f"{base} (time window)"
    return base


def recording_stimulus_mode_for_regions(region_intervals: list[dict], baseline_region_index: int | None = None) -> str:
    """Recording-level stimulus classification used for titles/reports."""
    modes = [stimulus_mode_for_region(r, baseline_region_index) for r in region_intervals]
    has_detected = any(m == "detected_stimulus" for m in modes)
    has_time_window = any(m == "time_window_control" for m in modes)
    has_drug = any(m == "drug_window" for m in modes)
    if has_detected:
        return "detected_stimulus"
    if has_time_window and has_drug:
        return "time_window_control_plus_drug"
    if has_time_window:
        return "time_window_control"
    if has_drug:
        return "drug_only_control"
    return "unknown"


def recording_title_prefix_for_mode(stimulus_mode: str) -> str:
    """Simple title prefix for control vs stimulated recordings."""
    if stimulus_mode in ("time_window_control", "time_window_control_plus_drug"):
        return "CONTROL RECORDING — no US pulses detected"
    if stimulus_mode == "drug_only_control":
        return "DRUG-ONLY CONTROL"
    if stimulus_mode == "detected_stimulus":
        return "ULTRASOUND RECORDING"
    return ""


def add_recording_mode_banner(fig, region_intervals=None, baseline_region_index=None, fontsize=16):
    """Add a small recording-level banner for no-pulse control recordings."""
    try:
        if region_intervals is None:
            return
        mode = recording_stimulus_mode_for_regions(region_intervals, baseline_region_index)
        prefix = recording_title_prefix_for_mode(mode)
        if prefix:
            fig.suptitle(prefix, fontsize=fontsize, fontweight="bold", y=0.995)
            try:
                fig.subplots_adjust(top=0.92)
            except Exception:
                pass
    except Exception:
        pass

def display_label_for_region(region_index: int, region_label: str, plot_region_labels: list[str] | None = None, baseline_region_index: int = 1) -> str:
    """Return compact plot label for a region.

    If --plot_region_labels is supplied, labels are assigned to non-baseline
    regions in sorted region order. Example for four non-baseline regions:
        --plot_region_labels 1MPa,2MPa,3MPa,AITC
    If a baseline label is included as the first label, it is ignored for
    non-baseline panels when there is one extra label.
    """
    labels = [str(x).strip() for x in (plot_region_labels or []) if str(x).strip()]
    if labels:
        # Typical region indices are baseline=1, nonbaseline=2..N.
        nonbaseline_pos = int(region_index) - int(baseline_region_index) - 1
        if len(labels) == 1:
            return labels[0]
        if 0 <= nonbaseline_pos < len(labels):
            return labels[nonbaseline_pos]
        # If user included a baseline label first, e.g. baseline,1MPa,2MPa,3MPa,AITC.
        if 0 <= (nonbaseline_pos + 1) < len(labels):
            return labels[nonbaseline_pos + 1]
    return str(region_label)

def save_spatial_focus_artifact_heatmaps_for_recording(
    recording: str,
    calcium_input_dir: Path,
    plot_dir: Path,
    rows_for_recording: list[dict],
    baseline_region_index: int,
    focus_cv_threshold: float,
    responder_threshold: float = 0.25,
    heatmap_clip_low_percentile: float = 5.0,
    heatmap_clip_high_percentile: float = 95.0,
    plot_region_labels: list[str] | None = None,
) -> None:
    """Save spatial heatmaps for focus instability / floating-cell artifact risk."""
    if not rows_for_recording:
        return
    mask_path = find_recording_mask_file(calcium_input_dir, recording)
    if mask_path is None:
        return
    masks = np.asarray(tifffile.imread(mask_path))
    if masks.ndim > 2:
        masks = np.squeeze(masks)
        if masks.ndim > 2:
            masks = masks[0]
    masks = masks.astype(np.int64, copy=False)
    if masks.ndim != 2 or int(np.nanmax(masks)) <= 0:
        return
    bg, bg_label = make_same_grid_background_for_spatial_overlay(calcium_input_dir, recording, masks.shape)

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    rows_by_region: dict[int, list[dict]] = {}
    for row in rows_for_recording:
        try:
            region_index = int(row.get("region_index"))
        except Exception:
            continue
        if region_index == int(baseline_region_index):
            continue
        rows_by_region.setdefault(region_index, []).append(row)

    montage_paths: list[Path] = []
    montage_labeled_paths: list[Path] = []

    # Shared color scale for all focus-QC panels and montages for this recording.
    all_focus_vals = []
    for _region_rows in rows_by_region.values():
        for _row in _region_rows:
            _v = safe_float(_row.get("focus_metric_region_cv", np.nan))
            if np.isfinite(_v):
                all_focus_vals.append(_v)
    shared_focus_vmin, shared_focus_vmax = robust_heatmap_limits(
        all_focus_vals,
        low_percentile=heatmap_clip_low_percentile,
        high_percentile=heatmap_clip_high_percentile,
        include_zero=True,
        include_threshold=focus_cv_threshold,
    )

    for region_index, region_rows in sorted(rows_by_region.items()):
        value_img = np.full(masks.shape, np.nan, dtype=float)
        vals = []
        responder_ids_for_plot: list[int] = []
        risk_ids_for_plot: list[int] = []
        for row in region_rows:
            cell_id = int(row.get("cell_id"))
            val = safe_float(row.get("focus_metric_region_cv", np.nan))
            if not np.isfinite(val):
                continue
            cell_mask = masks == cell_id
            if not np.any(cell_mask):
                continue
            value_img[cell_mask] = val
            vals.append(val)
            if row_has_qc_risk(row):
                risk_ids_for_plot.append(cell_id)
            if row_is_responder(row, responder_threshold=responder_threshold):
                responder_ids_for_plot.append(cell_id)
        if not vals:
            continue
        vmin, vmax = shared_focus_vmin, shared_focus_vmax
        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad((0, 0, 0, 0))
        cmap.set_under("black")
        cmap.set_over("white")
        fig, ax = plt.subplots(figsize=(7.36, 6.56))
        ax.imshow(bg, cmap="gray", interpolation="nearest")
        im = ax.imshow(np.ma.masked_invalid(value_img), cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.78, interpolation="nearest")
        # Draw each cell independently so adjacent cells do not look merged.
        # Cyan = responder; orange dotted = floating/dead-cell focus-artifact risk.
        plot_label_contours(ax, masks, _v472_positive_cell_ids(responder_ids_for_plot), colors="#00FFFF", linewidths=1.8)
        plot_label_contours(ax, masks, _v472_positive_cell_ids(risk_ids_for_plot), colors="white", linewidths=1.2, linestyles="dashed")
        region_label = str(region_rows[0].get("region_label", f"region_{region_index}"))
        plot_label = display_label_with_control_context(region_index, region_label, plot_region_labels, baseline_region_index, stimulus_mode_for_region(region_rows[0] if region_rows else {"region_index": region_index, "region_label": region_label}, baseline_region_index))
        ax.set_title(plot_label)
        ax.set_axis_off()
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend="both")
        set_clipped_colorbar_label(cbar, "focus metric CV", vmin, vmax, heatmap_clip_low_percentile, heatmap_clip_high_percentile)
        cbar.ax.axhline(focus_cv_threshold, color="cyan", linewidth=1.5)
        ax.legend(
            handles=[
                mpatches.Patch(facecolor="none", edgecolor="#00FFFF", label="responder"),
                mpatches.Patch(facecolor="none", edgecolor="#FF9F00", linestyle="dotted", label="floating-cell artifact risk"),
            ],
            loc="lower right",
            framealpha=0.8,
        )
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", region_label).strip("_") or f"region_{region_index}"
        out_path = plot_dir / f"{recording}_focus_artifact_QC_{region_index:02d}_{safe_label}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)

        # Duplicate labeled version for tracing spatial cells back to CSV/XLSX rows.
        labeled_path = plot_dir / f"{recording}_focus_artifact_QC_{region_index:02d}_{safe_label}_cellIDs.png"
        add_cell_id_labels_to_spatial_axis(ax, masks, [row.get("cell_id") for row in region_rows])
        fig.savefig(labeled_path, dpi=220)
        plt.close(fig)
        montage_paths.append(out_path)
        montage_labeled_paths.append(labeled_path)
        print(f"Saved focus-artifact QC heatmap: {out_path} (background: {bg_label})")
        print(f"Saved labeled focus-artifact QC heatmap: {labeled_path}")

    save_png_montage_from_paths(
        montage_paths,
        plot_dir / f"{recording}_focus_artifact_QC_montage_nonbaseline_regions.png",
        "Focus artifact QC",
    )
    save_png_montage_from_paths(
        montage_labeled_paths,
        plot_dir / f"{recording}_focus_artifact_QC_montage_nonbaseline_regions_cellIDs.png",
        "Focus artifact QC + cell IDs",
    )


def save_spatial_dff_heatmaps_for_recording(
    recording: str,
    calcium_input_dir: Path,
    plot_dir: Path,
    rows_for_recording: list[dict],
    responder_threshold: float,
    baseline_region_index: int,
    heatmap_clip_low_percentile: float = 5.0,
    heatmap_clip_high_percentile: float = 95.0,
    plot_region_labels: list[str] | None = None,
) -> None:
    """
    Save ROI-overlay heatmaps on the original/template image.

    Each non-baseline region gets one PNG. Cell color is the response above baseline
    (response_delta_middle90_vs_baseline). Cells below threshold are blue/gray;
    cells above threshold are warm-colored and outlined in bright cyan.
    """
    if not rows_for_recording:
        return

    mask_path = find_recording_mask_file(calcium_input_dir, recording)
    if mask_path is None:
        print(f"[WARNING] No mask TIFF found for spatial heatmap: {recording}")
        return

    masks = np.asarray(tifffile.imread(mask_path))
    if masks.ndim > 2:
        masks = np.squeeze(masks)
        if masks.ndim > 2:
            masks = masks[0]
    masks = masks.astype(np.int64, copy=False)
    if masks.ndim != 2 or int(np.nanmax(masks)) <= 0:
        print(f"[WARNING] Invalid/empty mask for spatial heatmap: {mask_path}")
        return

    bg, bg_label = make_same_grid_background_for_spatial_overlay(calcium_input_dir, recording, masks.shape)

    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.patches as mpatches

    rows_by_region: dict[int, list[dict]] = {}
    for row in rows_for_recording:
        try:
            region_index = int(row.get("region_index"))
        except Exception:
            continue
        if region_index == int(baseline_region_index):
            continue
        rows_by_region.setdefault(region_index, []).append(row)

    plot_dir.mkdir(parents=True, exist_ok=True)
    montage_paths: list[Path] = []
    montage_labeled_paths: list[Path] = []

    # Shared color scale for all spatial ΔF/F panels and montages for this recording.
    all_dff_vals = []
    for _region_rows in rows_by_region.values():
        for _row in _region_rows:
            _v = safe_float(_row.get("response_delta_middle90_vs_baseline", np.nan))
            if np.isfinite(_v):
                all_dff_vals.append(_v)
    shared_dff_vmin, shared_dff_vmax = robust_heatmap_limits(
        all_dff_vals,
        low_percentile=heatmap_clip_low_percentile,
        high_percentile=heatmap_clip_high_percentile,
        include_zero=True,
        include_threshold=responder_threshold,
    )

    for region_index, region_rows in sorted(rows_by_region.items()):
        if not region_rows:
            continue
        region_label = str(region_rows[0].get("region_label", f"region_{region_index}"))
        value_img = np.full(masks.shape, np.nan, dtype=float)
        responder_ids_for_plot: list[int] = []
        failed_qc_ids_for_plot: list[int] = []

        vals = []
        for row in region_rows:
            cell_id = int(row.get("cell_id"))
            val = safe_float(row.get("response_delta_middle90_vs_baseline", np.nan))
            passed_qc = bool(row.get("passed_qc", True))
            if not np.isfinite(val):
                continue
            cell_mask = masks == cell_id
            if not np.any(cell_mask):
                continue
            value_img[cell_mask] = val
            vals.append(val)
            # Keep cell outlines visually separated by storing IDs and drawing
            # one contour per Cellpose label. Do not merge into a single binary mask.
            if passed_qc and row_is_responder(row, responder_threshold=responder_threshold):
                responder_ids_for_plot.append(cell_id)
            if (not passed_qc) or row_has_qc_risk(row):
                failed_qc_ids_for_plot.append(cell_id)

        finite_vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
        if finite_vals.size == 0:
            continue

        # Make the threshold visually meaningful but robust to outliers: below
        # threshold maps to cool colors, above threshold maps to warm colors, and
        # percentile clipping prevents one extreme cell from dominating the color scale.
        vmin, vmax = shared_dff_vmin, shared_dff_vmax
        center = float(responder_threshold)
        if not (vmin < center < vmax):
            center = float(np.clip(center, vmin + 1e-9, vmax - 1e-9))
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)

        masked_values = np.ma.masked_invalid(value_img)
        cmap = plt.get_cmap("coolwarm").copy()
        cmap.set_bad((0, 0, 0, 0))
        cmap.set_under("navy")
        cmap.set_over("yellow")

        fig, ax = plt.subplots(figsize=(7.36, 6.56))
        ax.imshow(bg, cmap="gray", interpolation="nearest")
        im = ax.imshow(masked_values, cmap=cmap, norm=norm, alpha=0.78, interpolation="nearest")
        plot_label_contours(ax, masks, _v472_positive_cell_ids(responder_ids_for_plot), colors="#00FFFF", linewidths=1.8)
        plot_label_contours(ax, masks, failed_qc_ids_for_plot, colors="white", linewidths=2.5, linestyles="dashed")
        plot_label = display_label_with_control_context(region_index, region_label, plot_region_labels, baseline_region_index, stimulus_mode_for_region(region_rows[0] if region_rows else {"region_index": region_index, "region_label": region_label}, baseline_region_index))
        ax.set_title(plot_label)
        ax.set_axis_off()
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, extend="both")
        set_clipped_colorbar_label(cbar, "ΔF/F above baseline", vmin, vmax, heatmap_clip_low_percentile, heatmap_clip_high_percentile)
        cbar.ax.axhline(responder_threshold, color="#00FFFF", linewidth=1.5)
        ax.legend(
            handles=[
                mpatches.Patch(facecolor="none", edgecolor="#00FFFF", label="responder"),
                mpatches.Patch(facecolor="none", edgecolor="white", linestyle="--", label="QC risk"),
            ],
            loc="lower right",
            framealpha=0.8,
        )
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", region_label).strip("_") or f"region_{region_index}"
        out_path = plot_dir / f"{recording}_spatial_DFoverF_{region_index:02d}_{safe_label}.png"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)

        # Duplicate labeled version for tracing spatial cells back to CSV/XLSX rows.
        labeled_path = plot_dir / f"{recording}_spatial_DFoverF_{region_index:02d}_{safe_label}_cellIDs.png"
        add_cell_id_labels_to_spatial_axis(ax, masks, [row.get("cell_id") for row in region_rows])
        fig.savefig(labeled_path, dpi=220)
        montage_paths.append(out_path)
        montage_labeled_paths.append(labeled_path)
        plt.close(fig)
        print(f"Saved spatial DF/F heatmap: {out_path} (background: {bg_label})")
        print(f"Saved labeled spatial DF/F heatmap: {labeled_path}")

    save_png_montage_from_paths(
        montage_paths,
        plot_dir / f"{recording}_spatial_DFoverF_montage_nonbaseline_regions.png",
        "Spatial ΔF/F",
    )
    save_png_montage_from_paths(
        montage_labeled_paths,
        plot_dir / f"{recording}_spatial_DFoverF_montage_nonbaseline_regions_.png",
        "Spatial ΔF/F + cell IDs",
    )


def save_region_cell_frame_composites_for_recording(
    recording: str,
    calcium_input_dir: Path,
    plot_dir: Path,
    df_dff,
    rows_for_recording: list[dict],
    region_intervals_to_use: list[dict],
    baseline_region_index: int,
    responder_threshold: float,
    plot_region_labels: list[str] | None = None,
) -> None:
    """
    Save per-region baseline/max-expression cell-filled image composites.

    The output images are geometry-safe: the movie frames and labeled mask come from
    the same corrected TIFF grid used for calcium extraction. Each cell mask is
    filled with pixels from a cell-specific frame:
      - baseline composite: baseline-region frame closest to that cell's baseline DF/F median
      - max-expression composite: region frame where that cell reaches peak DF/F

    This avoids using a single global frame when different cells peak at different
    times, and keeps all pixels aligned because no resizing or template PNG is used.
    """
    if not rows_for_recording:
        return

    movie_path = find_recording_movie_file(calcium_input_dir, recording)
    if movie_path is None:
        print(f"[WARNING] No corrected movie found for representative cell-frame composites: {recording}")
        return

    mask_path = find_recording_mask_file(calcium_input_dir, recording)
    if mask_path is None:
        print(f"[WARNING] No mask TIFF found for representative cell-frame composites: {recording}")
        return

    masks = np.asarray(tifffile.imread(mask_path))
    if masks.ndim > 2:
        masks = np.squeeze(masks)
        if masks.ndim > 2:
            masks = masks[0]
    masks = masks.astype(np.int64, copy=False)
    if masks.ndim != 2 or int(np.nanmax(masks)) <= 0:
        print(f"[WARNING] Invalid/empty mask for representative cell-frame composites: {mask_path}")
        return

    # Use the same-grid movie mean/projection as dim background, but ROI pixels in
    # the composites come from the selected cell-specific frames.
    bg, bg_label = make_same_grid_background_for_spatial_overlay(calcium_input_dir, recording, masks.shape)

    baseline_region = next(
        (r for r in region_intervals_to_use if int(r.get("region_index", -1)) == int(baseline_region_index)),
        region_intervals_to_use[0] if region_intervals_to_use else None,
    )
    if baseline_region is None:
        return

    if "frame" in df_dff.columns:
        frame_numbers = df_dff["frame"].to_numpy(dtype=float)
    else:
        frame_numbers = np.arange(1, len(df_dff) + 1, dtype=float)
    time_s = df_dff["time_s"].to_numpy(dtype=float)

    baseline_mask = (
        (time_s >= float(baseline_region.get("middle90_start_s", baseline_region.get("region_start_s", 0))))
        & (time_s < float(baseline_region.get("middle90_end_s", baseline_region.get("region_end_s", np.inf))))
    )
    if not np.any(baseline_mask):
        baseline_mask = (
            (time_s >= float(baseline_region.get("region_start_s", 0)))
            & (time_s < float(baseline_region.get("region_end_s", np.inf)))
        )

    def nearest_frame_number(target_time: float) -> int | None:
        if not np.isfinite(target_time):
            return None
        idx = int(np.nanargmin(np.abs(time_s - float(target_time))))
        val = frame_numbers[idx]
        return int(val) if np.isfinite(val) else None

    def frame_to_page_index(frame_number: int) -> int:
        # Calcium CSV frame numbers are 1-based; TIFF pages are 0-based.
        return max(0, int(frame_number) - 1)

    import pandas as pd
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)

    # Group rows by region, excluding baseline. These rows are already per-cell.
    rows_by_region: dict[int, list[dict]] = {}
    for row in rows_for_recording:
        try:
            region_index = int(row.get("region_index"))
        except Exception:
            continue
        if region_index == int(baseline_region_index):
            continue
        rows_by_region.setdefault(region_index, []).append(row)

    frame_map_rows = []
    pair_montage_paths: list[Path] = []
    pair_labeled_montage_paths: list[Path] = []
    pair_baseline_only_id_montage_paths: list[Path] = []

    with tifffile.TiffFile(movie_path) as tif:
        n_pages = len(tif.pages)
        if n_pages <= 0:
            print(f"[WARNING] Empty movie for representative cell-frame composites: {movie_path}")
            return
        first_shape = tif.pages[0].asarray().shape
        if tuple(first_shape) != tuple(masks.shape):
            print(
                f"[WARNING] Skipping representative cell-frame composites for {recording}: "
                f"movie frame shape {first_shape} != mask shape {masks.shape}"
            )
            return

        frame_cache: dict[int, np.ndarray] = {}

        def get_frame_by_number(frame_number: int | None) -> np.ndarray | None:
            if frame_number is None:
                return None
            page_idx = min(max(frame_to_page_index(int(frame_number)), 0), n_pages - 1)
            if page_idx not in frame_cache:
                frame_cache[page_idx] = tif.pages[page_idx].asarray().astype(float, copy=False)
            return frame_cache[page_idx]

        for region_index, region_rows in sorted(rows_by_region.items()):
            if not region_rows:
                continue
            region_label = str(region_rows[0].get("region_label", f"region_{region_index}"))
            plot_label = display_label_with_control_context(region_index, region_label, plot_region_labels, baseline_region_index, stimulus_mode_for_region(region_rows[0] if region_rows else {"region_index": region_index, "region_label": region_label}, baseline_region_index))
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", region_label).strip("_") or f"region_{region_index}"

            baseline_composite = np.full(masks.shape, np.nan, dtype=float)
            stim_composite = np.full(masks.shape, np.nan, dtype=float)
            cells_used = []

            for row in region_rows:
                try:
                    cell_id = int(row.get("cell_id"))
                except Exception:
                    continue
                cell_col = f"cell_{cell_id}"
                cell_mask = masks == cell_id
                if not np.any(cell_mask) or cell_col not in df_dff.columns:
                    continue

                # Baseline representative frame: frame whose DF/F is closest to the
                # cell's baseline median/mean. This avoids choosing a random noisy frame.
                base_vals = df_dff.loc[baseline_mask, cell_col].to_numpy(dtype=float)
                base_frames = frame_numbers[baseline_mask]
                base_times = time_s[baseline_mask]
                good = np.isfinite(base_vals) & np.isfinite(base_frames)
                if np.any(good):
                    target = safe_float(row.get("baseline_mean_DFoverF", np.nan))
                    if not np.isfinite(target):
                        target = float(np.nanmedian(base_vals[good]))
                    local_idx = int(np.nanargmin(np.abs(base_vals[good] - target)))
                    baseline_frame = int(base_frames[good][local_idx])
                    baseline_time = float(base_times[good][local_idx])
                else:
                    baseline_frame = nearest_frame_number(float(baseline_region.get("middle90_start_s", baseline_region.get("region_start_s", 0))))
                    baseline_time = float("nan")

                # Stimulation/max-expression frame: raw peak frame/time for this cell in this region.
                stim_frame = None
                try:
                    pf = row.get("peak_frame", np.nan)
                    if np.isfinite(float(pf)):
                        stim_frame = int(float(pf))
                except Exception:
                    stim_frame = None
                if stim_frame is None:
                    stim_frame = nearest_frame_number(safe_float(row.get("peak_time_s", np.nan)))
                stim_time = safe_float(row.get("peak_time_s", np.nan))

                # For visualization only, fill a slightly expanded mask so the
                # selected cell-specific frame includes a little local background
                # around the ROI. Quantitative tables still use the original mask.
                display_mask = expand_cell_mask_by_fraction(cell_mask, fraction=0.10)

                base_frame_img = get_frame_by_number(baseline_frame)
                stim_frame_img = get_frame_by_number(stim_frame)
                if base_frame_img is not None:
                    baseline_composite[display_mask] = base_frame_img[display_mask]
                if stim_frame_img is not None:
                    stim_composite[display_mask] = stim_frame_img[display_mask]
                    cells_used.append(cell_id)

                frame_map_rows.append(
                    {
                        "recording": recording,
                        "region_index": region_index,
                        "region_label": region_label,
                        "cell_id": cell_id,
                        "baseline_representative_frame": baseline_frame,
                        "baseline_representative_time_s": baseline_time,
                        "max_expression_frame": stim_frame,
                        "max_expression_time_s": stim_time,
                        "peak_DFoverF": safe_float(row.get("peak_DFoverF", np.nan)),
                        "spike_safe_peak_DFoverF": safe_float(row.get("spike_safe_peak_DFoverF", np.nan)),
                    }
                )

            if not cells_used:
                continue

            responder_boundary, responder_ids = build_threshold_boundary_for_region(
                masks=masks,
                region_rows=region_rows,
                responder_threshold=responder_threshold,
            )

            # Normalize baseline and stim with the same scale so the pair is visually comparable.
            combined = np.concatenate([
                baseline_composite[np.isfinite(baseline_composite)],
                stim_composite[np.isfinite(stim_composite)],
            ])
            if combined.size == 0:
                continue
            lo, hi = np.percentile(combined, [1, 99.5])
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = float(np.nanmin(combined)), float(np.nanmax(combined))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                lo, hi = 0.0, 1.0

            def show_composite(ax, comp: np.ndarray, title: str):
                ax.imshow(bg, cmap="gray", interpolation="nearest", alpha=0.40)
                comp_norm = np.clip((comp - lo) / (hi - lo), 0, 1)
                ax.imshow(np.ma.masked_invalid(comp_norm), cmap="gray", interpolation="nearest", alpha=0.96)
                # Medium-blue outlines show all original Cellpose ROIs. Bright-cyan outlines
                # mark cells exceeding the responder threshold in this region.
                plot_label_contours(ax, masks, _v472_positive_cell_ids(cells_used), colors="#4A90E2", linewidths=0.55)
                plot_label_contours(ax, masks, responder_ids, colors="#00FFFF", linewidths=1.8)
                add_outline_legend(
                    ax,
                    entries=[
                        ("included cell", "#4A90E2", "solid", 0.55),
                        ("response", "#00FFFF", "solid", 1.8),
                    ],
                )
                ax.set_title(title, color="white")
                ax.set_axis_off()

            # Side-by-side pair.
            fig, axes = plt.subplots(1, 2, figsize=(12.88, 5.74), facecolor="0.08")
            for _ax in axes:
                _ax.set_facecolor("0.08")
            show_composite(axes[0], baseline_composite, "baseline representative frames")
            show_composite(axes[1], stim_composite, "max-expression frames")
            fig.tight_layout()
            pair_path = plot_dir / f"{recording}_cell_frame_composite_{region_index:02d}_{safe_label}_baseline_vs_max.png"
            fig.savefig(pair_path, dpi=200)
            # Labeled side-by-side pair.
            labeled_pair_path = plot_dir / f"{recording}_cell_frame_composite_{region_index:02d}_{safe_label}_baseline_vs_max_cellIDs.png"
            add_cell_id_labels_to_spatial_axis(axes[0], masks, _v472_positive_cell_ids(cells_used), fontsize=4.5)
            add_cell_id_labels_to_spatial_axis(axes[1], masks, _v472_positive_cell_ids(cells_used), fontsize=4.5)
            fig.savefig(labeled_pair_path, dpi=220)
            plt.close(fig)

            # Third side-by-side pair: cell IDs on baseline only. This keeps the
            # max-expression panel cleaner while preserving table traceability.
            fig, axes = plt.subplots(1, 2, figsize=(12.88, 5.74), facecolor="0.08")
            for _ax in axes:
                _ax.set_facecolor("0.08")
            show_composite(axes[0], baseline_composite, "baseline representative frames + cell IDs")
            show_composite(axes[1], stim_composite, "max-expression frames")
            add_cell_id_labels_to_spatial_axis(axes[0], masks, _v472_positive_cell_ids(cells_used), fontsize=4.5)
            fig.tight_layout()
            baseline_only_id_pair_path = plot_dir / f"{recording}_cell_frame_composite_{region_index:02d}_{safe_label}_baseline_vs_max_baseline.png"
            fig.savefig(baseline_only_id_pair_path, dpi=220)
            plt.close(fig)

            pair_montage_paths.append(pair_path)
            pair_labeled_montage_paths.append(labeled_pair_path)
            pair_baseline_only_id_montage_paths.append(baseline_only_id_pair_path)

            # Separate baseline and max-expression panels for easier inspection/export.
            for comp_name, comp, title in [
                ("baseline", baseline_composite, "baseline representative frames"),
                ("max_expression", stim_composite, "max-expression frames"),
            ]:
                fig, ax = plt.subplots(figsize=(7.36, 6.56))
                show_composite(ax, comp, f"{plot_label}: {title}")
                fig.tight_layout()
                out_path = plot_dir / f"{recording}_cell_frame_composite_{region_index:02d}_{safe_label}_{comp_name}.png"
                fig.savefig(out_path, dpi=200)
                labeled_path = plot_dir / f"{recording}_cell_frame_composite_{region_index:02d}_{safe_label}_{comp_name}_.png"
                add_cell_id_labels_to_spatial_axis(ax, masks, _v472_positive_cell_ids(cells_used), fontsize=5.0)
                fig.savefig(labeled_path, dpi=220)
                plt.close(fig)

            print(f"Saved cell-frame representative composites for {recording} {region_label} (background: {bg_label})")

    # For cell-frame composites, the montage should be arranged as a true
    # 2 x N layout: all baseline panels across the top row and matching
    # region/max-expression panels across the bottom row. The source files are
    # the normal side-by-side baseline-vs-max pairs, split at the midpoint.
    save_baseline_vs_max_2row_montage_from_pair_paths(
        pair_montage_paths,
        plot_dir / f"{recording}_cell_frame_composite_baseline_vs_max_montage_nonbaseline_regions.png",
        f"{recording}: baseline-vs-max cell-frame composites; top = baseline, bottom = region max",
    )
    save_baseline_vs_max_2row_montage_from_pair_paths(
        pair_labeled_montage_paths,
        plot_dir / f"{recording}_cell_frame_composite_baseline_vs_max_montage_nonbaseline_regions_.png",
        f"{recording}: baseline-vs-max cell-frame composites with IDs on both rows",
    )
    save_baseline_vs_max_2row_montage_from_pair_paths(
        pair_baseline_only_id_montage_paths,
        plot_dir / f"{recording}_cell_frame_composite_baseline_vs_max_montage_nonbaseline_regions_baseline.png",
        f"{recording}: baseline-vs-max cell-frame composites with IDs on baseline row only",
    )

    if frame_map_rows:
        map_path = plot_dir / f"{recording}_cell_frame_composite_frame_map.csv"
        pd.DataFrame(frame_map_rows).to_csv(map_path, index=False)
        print(f"Saved cell-frame composite frame map: {map_path}")

def stage_stim_region_analysis(
    output_root: Path,
    calcium_input_dir: Path,
    stim_file: Path | None,
    stim_dir: Path | None,
    stim_file_glob: str,
    stim_match_mode: str,
    stim_channel: str,
    expected_times: list[float],
    search_window_sec: float,
    threshold: float,
    max_pulse_sec: float,
    allow_no_stim_pulses: bool,
    allow_missing_stim_file: bool,
    extra_stim_delay_sec: float,
    recording_start_sec: float,
    max_data_regions: int,
    output_xlsx: Path | None,
    dry_run: bool,
    region_labels: list[str] | None = None,
    region_values: list[float] | None = None,
    region_value_name: str = "region_value",
    plot_region_labels: list[str] | None = None,
    responder_threshold: float = 0.25,
    responder_metric: str = "peak_above_baseline",
    responder_call_rule: str = "both",
    noise_sd_multiplier: float = 3.0,
    baseline_region_index: int = 1,
    min_cell_area_px: int = 0,
    min_F0: float = -1e18,
    max_baseline_sd: float = 1e18,
    max_abs_baseline_slope: float = 1e18,
    focus_qc_padding_px: int = 6,
    focus_qc_cv_threshold: float = 0.25,
    focus_qc_corr_threshold: float = 0.5,
    focus_qc_delta_threshold: float = 0.25,
    heatmap_clip_low_percentile: float = 5.0,
    heatmap_clip_high_percentile: float = 95.0,
    responder_peak_smoothing_frames: int = 3,
    responder_min_consecutive_frames: int = 2,
    make_plots: bool = True,
    analysis_runs_dir: Path | None = None,
    analysis_run_name: str | None = None,
    copy_script_snapshot: bool = True,
    args_dict: dict | None = None,
) -> None:
    """
    Enhanced stimulus-region analysis.

    Creates:
        - Excel workbook with metric definitions, region summaries, QC, responder calls, warnings
        - Optional PNG plots and simple HTML report

    Uses ultrasound channel only. Camera-on data are not used for region detection.
    """

    require_dir(calcium_input_dir, "Calcium CSV input directory")

    responder_peak_smoothing_frames = max(1, int(responder_peak_smoothing_frames))
    if responder_peak_smoothing_frames % 2 == 0:
        responder_peak_smoothing_frames += 1
    responder_min_consecutive_frames = max(1, int(responder_min_consecutive_frames))

    import pandas as pd

    region_labels = region_labels or []
    region_values = region_values or []
    default_labels = ["baseline", "stim1_to_stim2", "stim2_to_stim3", "stim3_to_drug", "drug_to_drug_end", "post_drug_end"]

    global_region_intervals: list[dict] = []
    region_intervals_by_recording: dict[str, list[dict]] = {}
    detected_stims_by_recording: dict[str, list[float]] = {}
    stim_file_by_recording: dict[str, Path | None] = {}

    warnings = []

    analysis_dir = create_timestamped_analysis_dir(
        calcium_input_dir=calcium_input_dir,
        analysis_runs_dir=analysis_runs_dir,
        analysis_run_name=analysis_run_name,
    )

    reports_dir = analysis_dir / "reports"
    csv_dir = analysis_dir / "csv"
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    if output_xlsx is None:
        output_xlsx = reports_dir / "stimulus_region_analysis.xlsx"
    else:
        output_xlsx = reports_dir / Path(output_xlsx).name

    dff_files = sorted(calcium_input_dir.glob("*_DFoverF_wide.csv"))
    if not dff_files:
        raise FileNotFoundError(f"No *_DFoverF_wide.csv files found in {calcium_input_dir}")

    recording_names = [p.name.replace("_DFoverF_wide.csv", "") for p in dff_files]

    if stim_file is None and stim_dir is None:
        if not allow_missing_stim_file:
            raise ValueError("Either --stim_file or --stim_dir is required unless --allow_missing_stim_file is enabled.")
        if not expected_times:
            raise ValueError(
                "No stimulus timing file was provided. To process old data without a MATLAB/DAQ timing file, "
                "provide approximate times with --stim_expected_times, e.g. --stim_expected_times 30,90,150."
            )

    stim_file_by_recording = build_stim_file_map(
        recordings=recording_names,
        stim_file=stim_file,
        stim_dir=stim_dir,
        stim_file_glob=stim_file_glob,
        stim_match_mode=stim_match_mode,
        allow_missing_stim_file=allow_missing_stim_file,
    )

    # Detect stimulus markers and define regions separately for each recording.
    for recording in recording_names:
        this_stim_file = stim_file_by_recording[recording]

        if this_stim_file is None:
            detected_stims = []
            this_warnings = []
            pulses_detected = False
            warnings.append(
                {
                    "recording": recording,
                    "stim_file": "",
                    "warning": "missing_stim_timing_file_using_approximate_times",
                    "detail": (
                        "No MATLAB/DAQ timing file was supplied for this recording. "
                        "Region boundaries are based only on --stim_expected_times. "
                        "Detected pulse timing and trigger QC are unavailable for this old dataset."
                    ),
                }
            )
        else:
            q, stim_signal = load_stimulus_trace(
                this_stim_file,
                stim_channel=stim_channel,
                mat_variable=(args_dict or {}).get("stim_mat_variable", "currentone"),
            )

            detected_stims, this_warnings = detect_ultrasound_pulses(
                q=q,
                stim_signal=stim_signal,
                expected_times=expected_times,
                search_window_sec=search_window_sec,
                threshold=threshold,
                max_pulse_sec=max_pulse_sec,
            )

            for w in this_warnings:
                w["recording"] = recording
                w["stim_file"] = ("" if this_stim_file is None else str(this_stim_file))
                warnings.append(w)

            pulses_detected = len(detected_stims) > 0

        if not pulses_detected:
            if this_stim_file is None:
                # Old data path: there is no DAQ trace to detect pulses from.
                # Use approximate expected times directly and keep the source explicit.
                marker_times_for_regions = list(expected_times)
                region_source = "approximate_expected_times_no_stim_timing_file"
                n_detected_for_regions = 0
            else:
                if not allow_no_stim_pulses:
                    raise ValueError(f"No ultrasound stimulus pulses detected for {recording} using {this_stim_file}")

                if not expected_times:
                    raise ValueError(
                        f"No ultrasound pulses detected for {recording}, and no --stim_expected_times were provided "
                        "to define planned control windows."
                    )

                warnings.append(
                    {
                        "recording": recording,
                        "stim_file": str(this_stim_file),
                        "warning": "no_ultrasound_pulses_detected_control_or_missing_trigger",
                        "detail": (
                            "No ultrasound pulses were detected. This is allowed because --allow_no_stim_pulses is enabled. "
                            "Planned windows from --stim_expected_times are used for region analysis."
                        ),
                    }
                )
                marker_times_for_regions = list(expected_times)
                region_source = "planned_expected_times_no_detected_ultrasound_pulse"
                n_detected_for_regions = 0
        else:
            # Use detected ultrasound pulse times when available, but do not drop
            # planned non-pulse markers supplied in --stim_expected_times. This is
            # important for the standard design where stim1/stim2/stim3 are
            # ultrasound pulses but drug addition is a manual/non-pulse boundary.
            #
            # Robust mapping rule:
            #   - If --stim_expected_times is supplied, build one region marker per
            #     expected time. For each expected time, use the nearest detected
            #     pulse within the search window when one exists; otherwise keep the
            #     expected time as a planned/non-pulse boundary.
            #   - This handles the common case 80,140,200,260 where only the first
            #     three markers are ultrasound pulses and 260 s is drug addition.
            #   - It also handles a missed middle pulse without shifting all later
            #     boundaries by index.
            if expected_times:
                marker_times_for_regions = []
                expected_markers_used = []
                detected_markers_used = []
                unused_detected = list(detected_stims)

                for expected in expected_times:
                    nearby = [
                        val for val in unused_detected
                        if abs(float(val) - float(expected)) <= float(search_window_sec)
                    ]
                    if nearby:
                        best = min(nearby, key=lambda val: abs(float(val) - float(expected)))
                        marker_times_for_regions.append(float(best))
                        detected_markers_used.append(float(best))
                        unused_detected = [val for val in unused_detected if abs(float(val) - float(best)) > 1e-6]
                    else:
                        marker_times_for_regions.append(float(expected))
                        expected_markers_used.append(float(expected))

                n_detected_for_regions = len(detected_markers_used)
                if expected_markers_used:
                    region_source = "detected_ultrasound_plus_expected_nonpulse_boundaries"
                    warnings.append(
                        {
                            "recording": recording,
                            "stim_file": ("" if this_stim_file is None else str(this_stim_file)),
                            "warning": "used_expected_nonpulse_or_missing_pulse_region_boundaries",
                            "detail": (
                                f"Using {len(detected_markers_used)} detected pulse marker(s) and "
                                f"{len(expected_markers_used)} expected/planned marker(s) for region boundaries. "
                                f"Expected/planned marker time(s) used: "
                                f"{','.join(str(float(x)) for x in expected_markers_used)}. "
                                "This is expected when the final marker is drug addition rather than an ultrasound pulse."
                            ),
                        }
                    )
                else:
                    region_source = "detected_ultrasound_matched_to_expected_times"

                if unused_detected:
                    warnings.append(
                        {
                            "recording": recording,
                            "stim_file": ("" if this_stim_file is None else str(this_stim_file)),
                            "warning": "detected_pulses_not_used_as_region_boundaries",
                            "detail": (
                                "Detected pulse time(s) were not used because they were not matched to --stim_expected_times: "
                                f"{','.join(str(float(x)) for x in unused_detected)}."
                            ),
                        }
                    )
            else:
                marker_times_for_regions = list(detected_stims)
                region_source = "detected_ultrasound_channel_rising_edge_intervals"
                n_detected_for_regions = len(detected_stims)

        if not marker_times_for_regions:
            raise ValueError(
                f"No region marker times available for {recording}. "
                "Provide --stim_expected_times to define planned windows."
            )

        boundaries = [float(recording_start_sec)] + list(marker_times_for_regions) + [marker_times_for_regions[-1] + float(extra_stim_delay_sec)]

        # Spatial heatmaps are generated for non-baseline regions only. Therefore,
        # the number of spatial heatmaps equals the number of marker times available.
        # For the standard 3-ultrasound + drug design, four marker times are needed:
        # stim1, stim2, stim3, drug. Three marker times produce only three spatial heatmaps.
        if len(marker_times_for_regions) == 3:
            warnings.append(
                {
                    "recording": recording,
                    "stim_file": ("" if this_stim_file is None else str(this_stim_file)),
                    "warning": "only_three_nonbaseline_spatial_heatmaps_expected",
                    "detail": (
                        "Only three marker times were available for this recording, so only three non-baseline "
                        "spatial heatmaps will be generated. For the standard stim1, stim2, stim3, drug design, "
                        "provide or detect four marker times; in approximate fallback mode use "
                        "--stim_expected_times stim1,stim2,stim3,drug."
                    ),
                }
            )

        region_intervals = []
        for i in range(len(boundaries) - 1):
            region_index = i + 1
            if i < len(region_labels):
                region_label = region_labels[i]
            elif i < len(default_labels):
                region_label = default_labels[i]
            else:
                region_label = f"region_{region_index}"

            region_value = region_values[i] if i < len(region_values) else np.nan

            region_intervals.append(
                {
                    "recording": recording,
                    "stim_file": ("" if this_stim_file is None else str(this_stim_file)),
                    "region_index": region_index,
                    "region_label": region_label,
                    region_value_name: region_value,
                    "region_start_s": float(boundaries[i]),
                    "region_end_s": float(boundaries[i + 1]),
                    "region_duration_s": float(boundaries[i + 1] - boundaries[i]),
                    "middle90_start_s": float(boundaries[i] + 0.05 * (boundaries[i + 1] - boundaries[i])),
                    "middle90_end_s": float(boundaries[i] + 0.95 * (boundaries[i + 1] - boundaries[i])),
                    "middle90_duration_s": float(0.90 * (boundaries[i + 1] - boundaries[i])),
                    "region_source": region_source,
                    "pulses_detected": bool(pulses_detected),
                    "stimulus_mode": "detected_stimulus" if bool(pulses_detected) else "time_window_control",
                    "n_detected_pulses_for_recording": int(n_detected_for_regions),
                    "stim_channel_used": stim_channel,
                }
            )

        # Refine per-region stimulus mode after labels are assigned.
        for _r in region_intervals:
            _r["stimulus_mode"] = stimulus_mode_for_region(_r, baseline_region_index)

        if len(region_intervals) > max_data_regions:
            warnings.append(
                {
                    "recording": recording,
                    "stim_file": ("" if this_stim_file is None else str(this_stim_file)),
                    "warning": "too_many_data_regions",
                    "detail": (
                        f"Detected {len(region_intervals)} data regions, which exceeds "
                        f"max_data_regions={max_data_regions}. Only the first {max_data_regions} "
                        f"regions are summarized."
                    ),
                }
            )
            region_intervals_to_use = region_intervals[:max_data_regions]
        else:
            region_intervals_to_use = region_intervals

        region_intervals_by_recording[recording] = region_intervals_to_use
        global_region_intervals.extend(region_intervals)
        detected_stims_by_recording[recording] = detected_stims

    if dry_run:
        print(f"[DRY RUN] Would analyze {len(dff_files)} DF/F CSV file(s).")
        print(f"[DRY RUN] Would write: {output_xlsx}")
        return

    plot_dir = analysis_dir / "plots"
    if make_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)

    # Save run metadata after the input file list is known.
    input_files_for_manifest = sorted([p for p in set(stim_file_by_recording.values()) if p is not None]) + dff_files
    input_files_for_manifest += sorted(calcium_input_dir.glob("*_F_wide.csv"))
    input_files_for_manifest += sorted(calcium_input_dir.glob("*_cell_summary.csv"))

    write_run_metadata(
        analysis_dir=analysis_dir,
        args_dict=args_dict or {},
        input_files=input_files_for_manifest,
        copy_script_snapshot=copy_script_snapshot,
        save_shell_history=(args_dict or {}).get("save_shell_history", True),
        shell_history_lines=(args_dict or {}).get("shell_history_lines", 50),
        save_conda_envs=(args_dict or {}).get("save_conda_envs", True),
        conda_env_names=parse_conda_env_names((args_dict or {}).get("conda_env_names", "dcimg,cellpose_py310,caiman_py310")),
    )
    write_latest_analysis_pointer(calcium_input_dir, analysis_dir)

    by_cell_rows = []
    region_metric_rows = []
    recording_summary_rows = []
    qc_rows = []
    responder_transition_rows = []
    pressure_threshold_rows = []
    spatial_rows = []
    recording_trace_fingerprint_rows = []
    recording_mean_traces_for_audit = {}

    for dff_file in dff_files:
        recording = dff_file.name.replace("_DFoverF_wide.csv", "")
        f_file = calcium_input_dir / f"{recording}_F_wide.csv"
        summary_file = calcium_input_dir / f"{recording}_cell_summary.csv"

        df_dff = pd.read_csv(dff_file)
        if "time_s" not in df_dff.columns:
            raise ValueError(f"{dff_file} missing time_s column.")

        df_f = pd.read_csv(f_file) if f_file.exists() else None
        df_summary = pd.read_csv(summary_file) if summary_file.exists() else None

        time_s = df_dff["time_s"].to_numpy(dtype=float)
        cell_cols = [c for c in df_dff.columns if c.startswith("cell_")]
        # Explicitly exclude impossible/background cell_0 if ever present in a source CSV.
        cell_cols = [c for c in cell_cols if c != "cell_0" and c.replace("cell_", "", 1).isdigit() and int(c.replace("cell_", "", 1)) > 0]

        # v4.76 recording-isolation audit: fingerprint each recording's actual DF/F matrix.
        try:
            dff_numeric_for_audit = df_dff[cell_cols].to_numpy(dtype=float)
            mean_trace_for_audit = np.nanmean(dff_numeric_for_audit, axis=1) if dff_numeric_for_audit.size else np.asarray([], dtype=float)
            recording_mean_traces_for_audit[recording] = mean_trace_for_audit
            recording_trace_fingerprint_rows.append(
                {
                    "recording": recording,
                    "dff_file": str(dff_file),
                    "dff_file_sha256": _v76_file_sha256(dff_file),
                    "n_frames": int(df_dff.shape[0]),
                    "n_cell_columns": int(len(cell_cols)),
                    "first_time_s": float(time_s[0]) if len(time_s) else np.nan,
                    "last_time_s": float(time_s[-1]) if len(time_s) else np.nan,
                    "cell_matrix_sha256": _v76_array_sha256(dff_numeric_for_audit),
                    "mean_trace_sha256": _v76_array_sha256(mean_trace_for_audit),
                }
            )
        except Exception as exc:
            warnings.append({"warning": "recording_fingerprint_failed", "detail": f"{recording}: {exc}"})

        # Per-recording region intervals and baseline region values for QC.
        region_intervals_to_use = region_intervals_by_recording[recording]
        this_stim_file = stim_file_by_recording[recording]

        baseline_region = next(
            (r for r in region_intervals_to_use if r["region_index"] == baseline_region_index),
            region_intervals_to_use[0],
        )
        baseline_mask = (
            (df_dff["time_s"] >= baseline_region["region_start_s"])
            & (df_dff["time_s"] < baseline_region["region_end_s"])
        )

        cell_status = {}
        first_response_region = {}

        # Focus-artifact QC: detects possible floating/dead cells moving through Z-focus.
        # This deliberately avoids local cell-shift estimation from calcium brightness patterns.
        try:
            focus_artifact_qc = compute_focus_artifact_qc_for_recording(
                recording=recording,
                calcium_input_dir=calcium_input_dir,
                df_dff=df_dff,
                region_intervals_to_use=region_intervals_to_use,
                baseline_region_index=baseline_region_index,
                padding_px=focus_qc_padding_px,
                focus_cv_threshold=focus_qc_cv_threshold,
                focus_corr_threshold=focus_qc_corr_threshold,
                focus_delta_threshold=focus_qc_delta_threshold,
            )
        except Exception as exc:
            warnings.append({"warning": "focus_artifact_qc_failed", "detail": f"{recording}: {exc}"})
            focus_artifact_qc = {}

        for cell_col in cell_cols:
            cell_id = int(cell_col.replace("cell_", ""))
            baseline_vals = df_dff.loc[baseline_mask, cell_col].to_numpy(dtype=float)
            baseline_times = df_dff.loc[baseline_mask, "time_s"].to_numpy(dtype=float)

            baseline_mean = float(np.nanmean(baseline_vals)) if np.any(np.isfinite(baseline_vals)) else np.nan
            baseline_sd = float(np.nanstd(baseline_vals)) if np.any(np.isfinite(baseline_vals)) else np.nan
            baseline_slope = linear_slope(baseline_times, baseline_vals)

            area_px = np.nan
            F0 = np.nan
            mean_F = np.nan
            if df_summary is not None:
                hit = df_summary[df_summary["cell_id"] == cell_id]
                if len(hit) > 0:
                    if "area_px" in hit.columns:
                        area_px = safe_float(hit["area_px"].iloc[0])
                    if "F0" in hit.columns:
                        F0 = safe_float(hit["F0"].iloc[0])
                    if "mean_F" in hit.columns:
                        mean_F = safe_float(hit["mean_F"].iloc[0])

            passed_area = bool(np.isnan(area_px) or area_px >= min_cell_area_px)
            passed_F0 = bool(np.isnan(F0) or F0 >= min_F0)
            passed_noise = bool(np.isnan(baseline_sd) or baseline_sd <= max_baseline_sd)
            passed_slope = bool(np.isnan(baseline_slope) or abs(baseline_slope) <= max_abs_baseline_slope)
            passed_qc = passed_area and passed_F0 and passed_noise and passed_slope

            cell_status[cell_id] = {
                "baseline_mean": baseline_mean,
                "baseline_sd": baseline_sd,
                "baseline_slope": baseline_slope,
                "area_px": area_px,
                "F0": F0,
                "mean_F": mean_F,
                "passed_qc": passed_qc,
            }

            qc_rows.append(
                {
                    "recording": recording,
                    "cell_id": cell_id,
                    "area_px": area_px,
                    "F0": F0,
                    "mean_F": mean_F,
                    "baseline_region_index": baseline_region["region_index"],
                    "baseline_mean_DFoverF": baseline_mean,
                    "baseline_sd_DFoverF": baseline_sd,
                    "baseline_slope_DFoverF_per_s": baseline_slope,
                    "passed_area_filter": passed_area,
                    "passed_F0_filter": passed_F0,
                    "passed_baseline_noise_filter": passed_noise,
                    "passed_baseline_slope_filter": passed_slope,
                    "max_focus_metric_region_cv": max([safe_float(v.get("focus_metric_region_cv", np.nan)) for (ri, cid), v in focus_artifact_qc.items() if cid == cell_id and int(ri) != int(baseline_region_index)] or [np.nan]),
                    "max_abs_focus_metric_delta_frac_vs_baseline": max([abs(safe_float(v.get("focus_metric_delta_frac_vs_baseline", np.nan))) for (ri, cid), v in focus_artifact_qc.items() if cid == cell_id and int(ri) != int(baseline_region_index)] or [np.nan]),
                    "max_abs_dff_focus_correlation": max([abs(safe_float(v.get("dff_focus_correlation", np.nan))) for (ri, cid), v in focus_artifact_qc.items() if cid == cell_id and int(ri) != int(baseline_region_index)] or [np.nan]),
                    "any_focus_instability_suspicious": any([bool(v.get("focus_instability_suspicious", False)) for (ri, cid), v in focus_artifact_qc.items() if cid == cell_id and int(ri) != int(baseline_region_index)]),
                    "any_floating_cell_artifact_risk": any([bool(v.get("floating_cell_artifact_risk", False)) for (ri, cid), v in focus_artifact_qc.items() if cid == cell_id and int(ri) != int(baseline_region_index)]),
                    "focus_qc_cv_threshold": focus_qc_cv_threshold,
                    "focus_qc_corr_threshold": focus_qc_corr_threshold,
                    "focus_qc_delta_threshold": focus_qc_delta_threshold,
                    "passed_all_qc": passed_qc,
                }
            )

        # Recording-level reference values for automatic conservative QC.
        baseline_sd_values_for_qc = [
            safe_float(v.get("baseline_sd", np.nan))
            for v in cell_status.values()
            if np.isfinite(safe_float(v.get("baseline_sd", np.nan)))
        ]
        recording_median_baseline_sd_for_qc = (
            float(np.nanmedian(baseline_sd_values_for_qc)) if baseline_sd_values_for_qc else np.nan
        )
        area_values_for_qc = [
            safe_float(v.get("area_px", np.nan))
            for v in cell_status.values()
            if np.isfinite(safe_float(v.get("area_px", np.nan)))
        ]
        try:
            qc_min_area_percentile_val = float(getattr(args, "qc_min_area_percentile", 1.0))
            qc_max_area_percentile_val = float(getattr(args, "qc_max_area_percentile", 99.0))
        except Exception:
            qc_min_area_percentile_val = 1.0
            qc_max_area_percentile_val = 99.0
        area_low_for_qc = (
            float(np.nanpercentile(area_values_for_qc, qc_min_area_percentile_val))
            if area_values_for_qc else np.nan
        )
        area_high_for_qc = (
            float(np.nanpercentile(area_values_for_qc, qc_max_area_percentile_val))
            if area_values_for_qc else np.nan
        )

        # Per-cell, per-region metrics.
        for region in region_intervals_to_use:
            region_index = region["region_index"]
            region_mask = (df_dff["time_s"] >= region["region_start_s"]) & (df_dff["time_s"] < region["region_end_s"])
            region_df = df_dff.loc[region_mask].copy()
            n_region_frames = int(region_df.shape[0])
            region_times = region_df["time_s"].to_numpy(dtype=float)

            if n_region_frames == 0:
                warnings.append({"warning": "empty_region", "detail": f"{recording} region {region_index} has no calcium frames."})
                continue

            region_values_all_cells = []
            peak_values_all_cells = []
            responder_abs_count = 0
            responder_noise_count = 0
            responder_either_count = 0
            responder_both_count = 0
            responder_primary_count = 0
            confidence_high_count = 0
            confidence_medium_count = 0
            confidence_low_count = 0
            confidence_none_count = 0
            qc_cell_count = 0

            for cell_col in cell_cols:
                cell_id = int(cell_col.replace("cell_", ""))
                vals = region_df[cell_col].to_numpy(dtype=float)
                vals_mid90 = temporal_middle90(vals)

                mid90_mean = temporal_middle90_mean(vals)
                raw_mean = float(np.nanmean(vals)) if np.any(np.isfinite(vals)) else np.nan
                median_val = float(np.nanmedian(vals)) if np.any(np.isfinite(vals)) else np.nan
                sd_val = float(np.nanstd(vals)) if np.any(np.isfinite(vals)) else np.nan
                auc_val = auc_trapz(region_times, vals)

                if np.any(np.isfinite(vals)):
                    peak_idx_local = int(np.nanargmax(vals))
                    peak_val = float(vals[peak_idx_local])
                    peak_time_s = float(region_times[peak_idx_local])
                    try:
                        peak_frame = int(region_df["frame"].iloc[peak_idx_local])
                    except Exception:
                        peak_frame = np.nan
                    time_to_peak_s = peak_time_s - float(region["region_start_s"])
                else:
                    peak_val = np.nan
                    peak_time_s = np.nan
                    peak_frame = np.nan
                    time_to_peak_s = np.nan

                spike_safe_peak_val, spike_safe_peak_time_s = spike_safe_peak(
                    vals,
                    region_times,
                    responder_peak_smoothing_frames,
                )
                spike_safe_time_to_peak_s = (
                    spike_safe_peak_time_s - float(region["region_start_s"])
                    if np.isfinite(spike_safe_peak_time_s)
                    else np.nan
                )

                baseline_mean = cell_status[cell_id]["baseline_mean"]
                baseline_sd = cell_status[cell_id]["baseline_sd"]
                baseline_slope = cell_status[cell_id]["baseline_slope"]
                passed_qc = cell_status[cell_id]["passed_qc"]

                response_delta_mid90 = mid90_mean - baseline_mean if np.isfinite(mid90_mean) and np.isfinite(baseline_mean) else np.nan
                noise_threshold = baseline_mean + noise_sd_multiplier * baseline_sd if np.isfinite(baseline_mean) and np.isfinite(baseline_sd) else np.nan

                # Baseline is the reference/noise window, not a stimulus response window.
                # Do NOT allow region 1 / baseline_region_index to become first_response_region.
                is_baseline_reference_region = int(region_index) == int(baseline_region_index)

                raw_peak_above_baseline = peak_val - baseline_mean if np.isfinite(peak_val) and np.isfinite(baseline_mean) else np.nan
                spike_safe_peak_above_baseline = spike_safe_peak_val - baseline_mean if np.isfinite(spike_safe_peak_val) and np.isfinite(baseline_mean) else np.nan
                noise_delta_threshold = noise_sd_multiplier * baseline_sd if np.isfinite(baseline_sd) else np.nan

                responder_metric_used = str(responder_metric or "peak_above_baseline")
                if responder_metric_used == "peak":
                    absolute_trace = vals
                    absolute_peak_for_call = spike_safe_peak_val
                    absolute_raw_peak_for_call = peak_val
                    absolute_threshold_for_call = responder_threshold
                elif responder_metric_used == "peak_above_baseline":
                    absolute_trace = vals - baseline_mean if np.isfinite(baseline_mean) else np.full_like(vals, np.nan, dtype=float)
                    absolute_peak_for_call = spike_safe_peak_above_baseline
                    absolute_raw_peak_for_call = raw_peak_above_baseline
                    absolute_threshold_for_call = responder_threshold
                else:
                    raise ValueError(f"Unknown responder_metric: {responder_metric_used}")

                max_consecutive_absolute_frames = max_consecutive_true(
                    np.isfinite(absolute_trace) & (absolute_trace >= absolute_threshold_for_call)
                )
                max_consecutive_noise_frames = max_consecutive_true(
                    np.isfinite(vals) & np.isfinite(noise_threshold) & (vals >= noise_threshold)
                )
                responder_absolute_raw_peak = bool(np.isfinite(absolute_raw_peak_for_call) and absolute_raw_peak_for_call >= absolute_threshold_for_call)
                responder_noise_raw_peak = bool(np.isfinite(peak_val) and np.isfinite(noise_threshold) and peak_val >= noise_threshold)

                if is_baseline_reference_region:
                    responder_absolute = False
                    responder_noise_based = False
                    responder_either = False
                    responder_both = False
                    responder_primary = False
                    response_confidence = "BASELINE_REFERENCE"
                    response_class = "baseline_reference"
                else:
                    responder_absolute = bool(
                        np.isfinite(absolute_peak_for_call)
                        and absolute_peak_for_call >= absolute_threshold_for_call
                        and max_consecutive_absolute_frames >= responder_min_consecutive_frames
                    )
                    responder_noise_based = bool(
                        np.isfinite(spike_safe_peak_val)
                        and np.isfinite(noise_threshold)
                        and spike_safe_peak_val >= noise_threshold
                        and max_consecutive_noise_frames >= responder_min_consecutive_frames
                    )
                    responder_either = bool(responder_absolute or responder_noise_based)
                    responder_both = bool(responder_absolute and responder_noise_based)
                    rule = str(responder_call_rule or "both").strip().lower()
                    if rule == "both":
                        responder_primary = responder_both
                    elif rule == "either":
                        responder_primary = responder_either
                    else:
                        raise ValueError(f"Unknown responder_call_rule: {responder_call_rule}")

                    focus_qc_tmp = focus_artifact_qc.get((int(region_index), int(cell_id)), {})
                    focus_risk_tmp = _v465_row_plot_qc_risk(focus_qc_tmp)

                    # v4.64 automatic conservative QC. These flags are artifact-focused and do
                    # not require response to multiple stimuli or return-to-baseline behavior.
                    try:
                        focus_corr_abs_threshold_tmp = float(getattr(args, "qc_focus_corr_abs_threshold", 0.70))
                    except Exception:
                        focus_corr_abs_threshold_tmp = 0.70
                    try:
                        dff_focus_corr_tmp = float(focus_qc_tmp.get("dff_focus_correlation", np.nan))
                    except Exception:
                        dff_focus_corr_tmp = np.nan
                    conservative_focus_corr_risk = bool(
                        np.isfinite(dff_focus_corr_tmp)
                        and abs(dff_focus_corr_tmp) >= focus_corr_abs_threshold_tmp
                    )

                    try:
                        baseline_sd_multiplier_tmp = float(getattr(args, "qc_baseline_sd_multiplier", 2.5))
                    except Exception:
                        baseline_sd_multiplier_tmp = 2.5
                    baseline_instability_ratio_tmp = (
                        float(baseline_sd) / float(recording_median_baseline_sd_for_qc)
                        if np.isfinite(baseline_sd)
                        and np.isfinite(recording_median_baseline_sd_for_qc)
                        and recording_median_baseline_sd_for_qc > 0
                        else np.nan
                    )
                    conservative_baseline_instability_risk = bool(
                        np.isfinite(baseline_instability_ratio_tmp)
                        and baseline_instability_ratio_tmp >= baseline_sd_multiplier_tmp
                    )

                    area_val_tmp = safe_float(cell_status[cell_id].get("area_px", np.nan))
                    conservative_area_outlier_risk = bool(
                        np.isfinite(area_val_tmp)
                        and (
                            (np.isfinite(area_low_for_qc) and area_val_tmp < area_low_for_qc)
                            or (np.isfinite(area_high_for_qc) and area_val_tmp > area_high_for_qc)
                        )
                    )

                    conservative_qc_risk = bool(
                        focus_risk_tmp
                        or conservative_focus_corr_risk
                        or conservative_baseline_instability_risk
                        or conservative_area_outlier_risk
                        or (recording_baseline_unstable if 'recording_baseline_unstable' in locals() else False)
                    )

                    if responder_primary and not conservative_qc_risk:
                        response_confidence = "HIGH"
                    elif responder_primary and (recording_baseline_unstable if "recording_baseline_unstable" in locals() else False):
                        response_confidence = "MEDIUM_RECORDING_QC_RISK"
                    elif responder_primary and conservative_qc_risk:
                        response_confidence = "MEDIUM_CELL_QC_RISK"
                    elif responder_either and not responder_primary:
                        response_confidence = "LOW_BORDERLINE"
                    else:
                        response_confidence = "NONE"

                    # No-pulse control windows are not ultrasound-triggered responses.
                    if (not is_eligible_us_stimulus_region(region, baseline_region_index)) and responder_primary and not is_baseline_reference_region and stimulus_mode_for_region(region, baseline_region_index) == "time_window_control":
                        response_confidence = "CONTROL_WINDOW_PRIMARY_RESPONSE"

                    recording_cell_key = _v471_recording_cell_key(recording, cell_id)
                    if stimulus_mode_for_region(region, baseline_region_index) == "time_window_control":
                        # In no-pulse controls, do not use stimulus-progression language.
                        if responder_primary:
                            response_class = "control_window_activity"
                        elif responder_either:
                            response_class = "control_window_borderline_audit"
                        else:
                            response_class = "non_responder"
                    elif responder_primary and recording_cell_key not in first_response_region:
                        first_response_region[recording_cell_key] = region_index
                        response_class = "new_responder"
                    elif responder_primary:
                        response_class = "previous_responder"
                    elif responder_either:
                        response_class = "borderline_audit_not_responder"
                    else:
                        response_class = "non_responder"

                if passed_qc:
                    qc_cell_count += 1
                    if responder_absolute:
                        responder_abs_count += 1
                    if responder_noise_based:
                        responder_noise_count += 1
                    if responder_either:
                        responder_either_count += 1
                    if responder_both:
                        responder_both_count += 1
                    if responder_primary:
                        responder_primary_count += 1
                    if response_confidence == "HIGH":
                        confidence_high_count += 1
                    elif response_confidence == "MEDIUM_QC_RISK":
                        confidence_medium_count += 1
                    elif response_confidence == "LOW_BORDERLINE":
                        confidence_low_count += 1
                    elif response_confidence == "NONE":
                        confidence_none_count += 1

                region_values_all_cells.append(mid90_mean)
                peak_values_all_cells.append(peak_val)

                row = {
                    "recording": recording,
                    "region_index": region_index,
                    "region_label": region["region_label"],
                    region_value_name: region.get(region_value_name, np.nan),
                    "region_start_s": region.get("region_start_s", np.nan),
                    "region_end_s": region.get("region_end_s", np.nan),
                    "region_duration_s": region.get("region_duration_s", np.nan),
                    "metric_window": "middle90",
                    "metric_window_start_s": region.get("middle90_start_s", np.nan),
                    "metric_window_end_s": region.get("middle90_end_s", np.nan),
                    "metric_window_duration_s": region.get("middle90_duration_s", np.nan),
                    "region_source": region.get("region_source", ""),
                    "pulses_detected": region.get("pulses_detected", np.nan),
                    "stimulus_mode": stimulus_mode_for_region(region, baseline_region_index),
                    "stimulus_present_for_region": is_eligible_us_stimulus_region(region, baseline_region_index),
                    "eligible_us_responder": bool(responder_primary) and is_eligible_us_stimulus_region(region, baseline_region_index),
                    "stim_channel_used": region.get("stim_channel_used", stim_channel),
                    "region_start_s": region["region_start_s"],
                    "region_end_s": region["region_end_s"],
                    "region_duration_s": region["region_duration_s"],
                    "middle90_start_s": region["middle90_start_s"],
                    "middle90_end_s": region["middle90_end_s"],
                    "middle90_duration_s": region["middle90_duration_s"],
                    "region_source": region.get("region_source", ""),
                    "pulses_detected": region.get("pulses_detected", np.nan),
                    "stimulus_mode": stimulus_mode_for_region(region, baseline_region_index),
                    "stimulus_present_for_region": is_eligible_us_stimulus_region(region, baseline_region_index),
                    "eligible_us_responder": bool(responder_primary) and is_eligible_us_stimulus_region(region, baseline_region_index),
                    "stim_channel_used": region.get("stim_channel_used", stim_channel),
                    "region_start_s": region["region_start_s"],
                    "region_end_s": region["region_end_s"],
                    "region_duration_s": region["region_duration_s"],
                    "n_region_frames": n_region_frames,
                    "cell_id": cell_id,
                    "passed_qc": passed_qc,
                    "middle90_mean_DFoverF": mid90_mean,
                    "raw_region_mean_DFoverF": raw_mean,
                    "median_DFoverF": median_val,
                    "sd_DFoverF": sd_val,
                    "auc_DFoverF_s": auc_val,
                    "peak_DFoverF": peak_val,
                    "peak_time_s": peak_time_s,
                    "time_to_peak_s": time_to_peak_s,
                    "spike_safe_peak_DFoverF": spike_safe_peak_val,
                    "spike_safe_peak_time_s": spike_safe_peak_time_s,
                    "spike_safe_time_to_peak_s": spike_safe_time_to_peak_s,
                    "raw_peak_above_baseline_DFoverF": raw_peak_above_baseline,
                    "spike_safe_peak_above_baseline_DFoverF": spike_safe_peak_above_baseline,
                    "responder_metric_used": responder_metric_used,
                    "responder_call_rule": responder_call_rule,
                    "absolute_threshold_for_call": absolute_threshold_for_call,
                    "absolute_peak_for_call": absolute_peak_for_call,
                    "noise_delta_threshold": noise_delta_threshold,
                    "responder_peak_smoothing_frames": responder_peak_smoothing_frames,
                    "responder_min_consecutive_frames": responder_min_consecutive_frames,
                    "max_consecutive_absolute_frames": max_consecutive_absolute_frames,
                    "max_consecutive_noise_frames": max_consecutive_noise_frames,
                    "responder_absolute_raw_peak": responder_absolute_raw_peak,
                    "responder_noise_raw_peak": responder_noise_raw_peak,
                    "baseline_mean_DFoverF": baseline_mean,
                    "baseline_sd_DFoverF": baseline_sd,
                    "baseline_slope_DFoverF_per_s": baseline_slope,
                    "response_delta_middle90_vs_baseline": response_delta_mid90,
                    "responder_threshold": responder_threshold,
                    "noise_threshold": noise_threshold,
                    "is_baseline_reference_region": is_baseline_reference_region,
                    "responder_absolute": responder_absolute,
                    "responder_noise_based": responder_noise_based,
                    "responder_either": responder_either,
                    "responder_both": responder_both,
                    "responder_primary": responder_primary,
                    "response_confidence": response_confidence,
                    "response_class": response_class,
                    "area_px": cell_status[cell_id]["area_px"],
                    "F0": cell_status[cell_id]["F0"],
                    "mean_F": cell_status[cell_id]["mean_F"],
                }
                row.update(focus_artifact_qc.get((int(region_index), int(cell_id)), {}))
                by_cell_rows.append(row)
                region_metric_rows.append(row)

                responder_transition_rows.append(
                    {
                        "recording": recording,
                        "cell_id": cell_id,
                        "region_index": region_index,
                        "region_label": region["region_label"],
                        "is_baseline_reference_region": is_baseline_reference_region,
                        "responder_either": responder_either,
                        "responder_both": responder_both,
                        "responder_primary": responder_primary,
                        "response_confidence": response_confidence,
                        "response_class": response_class,
                        "peak_DFoverF": peak_val,
                        "spike_safe_peak_DFoverF": spike_safe_peak_val,
                        "responder_peak_smoothing_frames": responder_peak_smoothing_frames,
                        "responder_min_consecutive_frames": responder_min_consecutive_frames,
                        "max_consecutive_absolute_frames": max_consecutive_absolute_frames,
                        "max_consecutive_noise_frames": max_consecutive_noise_frames,
                        "first_response_region": first_response_region.get(_v471_recording_cell_key(recording, cell_id), np.nan),
                    }
                )

            vals_arr = np.asarray(region_values_all_cells, dtype=float)
            peaks_arr = np.asarray(peak_values_all_cells, dtype=float)
            n_cells = len(cell_cols)

            if stimulus_mode_for_region(region, baseline_region_index) == "time_window_control":
                if responder_primary_count > 0:
                    warnings.append({
                        "recording": recording,
                        "warning": "primary_responses_in_no_pulse_control_window",
                        "detail": f"Region {region_index} ({region['region_label']}) had {responder_primary_count} primary responder(s) but pulses_detected=False. Treat as control-window response, not US-triggered response.",
                    })
                if responder_either_count > 0:
                    warnings.append({
                        "recording": recording,
                        "warning": "borderline_noise_responses_in_no_pulse_control_window",
                        "detail": f"Region {region_index} ({region['region_label']}) had {responder_either_count} either/borderline call(s) but pulses_detected=False. These are audit-only and not primary US responders.",
                    })
            recording_summary_rows.append(
                {
                    "recording": recording,
                    "region_index": region_index,
                    "region_label": region["region_label"],
                    region_value_name: region.get(region_value_name, np.nan),
                    "region_start_s": region.get("region_start_s", np.nan),
                    "region_end_s": region.get("region_end_s", np.nan),
                    "region_duration_s": region.get("region_duration_s", np.nan),
                    "metric_window": "middle90",
                    "metric_window_start_s": region.get("middle90_start_s", np.nan),
                    "metric_window_end_s": region.get("middle90_end_s", np.nan),
                    "metric_window_duration_s": region.get("middle90_duration_s", np.nan),
                    "region_source": region.get("region_source", ""),
                    "pulses_detected": region.get("pulses_detected", np.nan),
                    "stimulus_mode": stimulus_mode_for_region(region, baseline_region_index),
                    "stimulus_present_for_region": is_eligible_us_stimulus_region(region, baseline_region_index),
                    "eligible_us_responder": bool(responder_primary) and is_eligible_us_stimulus_region(region, baseline_region_index),
                    "stim_channel_used": region.get("stim_channel_used", stim_channel),
                    "region_start_s": region["region_start_s"],
                    "region_end_s": region["region_end_s"],
                    "region_duration_s": region["region_duration_s"],
                    "middle90_start_s": region["middle90_start_s"],
                    "middle90_end_s": region["middle90_end_s"],
                    "middle90_duration_s": region["middle90_duration_s"],
                    "region_source": region.get("region_source", ""),
                    "pulses_detected": region.get("pulses_detected", np.nan),
                    "stimulus_mode": stimulus_mode_for_region(region, baseline_region_index),
                    "stimulus_present_for_region": is_eligible_us_stimulus_region(region, baseline_region_index),
                    "eligible_us_responder": bool(responder_primary) and is_eligible_us_stimulus_region(region, baseline_region_index),
                    "stim_channel_used": region.get("stim_channel_used", stim_channel),
                    "region_start_s": region["region_start_s"],
                    "region_end_s": region["region_end_s"],
                    "region_duration_s": region["region_duration_s"],
                    "n_region_frames": n_region_frames,
                    "n_cells_total": n_cells,
                    "n_cells_passed_qc": qc_cell_count,
                    "median_cell_middle90_mean_DFoverF": float(np.nanmedian(vals_arr)) if vals_arr.size else np.nan,
                    "mean_cell_middle90_mean_DFoverF": float(np.nanmean(vals_arr)) if vals_arr.size else np.nan,
                    "max_cell_middle90_mean_DFoverF": float(np.nanmax(vals_arr)) if vals_arr.size else np.nan,
                    "median_peak_DFoverF": float(np.nanmedian(peaks_arr)) if peaks_arr.size else np.nan,
                    "max_peak_DFoverF": float(np.nanmax(peaks_arr)) if peaks_arr.size else np.nan,
                    "responders_absolute": responder_abs_count,
                    "responders_noise_based": responder_noise_count,
                    "responders_either": responder_either_count,
                    "responders_both": responder_both_count,
                    "responders_primary": responder_primary_count,
                    "stimulus_mode": stimulus_mode_for_region(region, baseline_region_index),
                    "responders_primary_eligible_us": responder_primary_count if is_eligible_us_stimulus_region(region, baseline_region_index) else 0,
                    "fraction_responders_primary_eligible_us": (responder_primary_count / qc_cell_count if (qc_cell_count and is_eligible_us_stimulus_region(region, baseline_region_index)) else 0.0),
                    "confidence_high_count": confidence_high_count,
                    "confidence_medium_qc_risk_count": confidence_medium_count,
                    "confidence_low_borderline_count": confidence_low_count,
                    "confidence_none_count": confidence_none_count,
                    "fraction_responders_absolute": responder_abs_count / qc_cell_count if qc_cell_count else np.nan,
                    "fraction_responders_noise_based": responder_noise_count / qc_cell_count if qc_cell_count else np.nan,
                    "fraction_responders_either": responder_either_count / qc_cell_count if qc_cell_count else np.nan,
                    "fraction_responders_both": responder_both_count / qc_cell_count if qc_cell_count else np.nan,
                    "fraction_responders_primary": responder_primary_count / qc_cell_count if qc_cell_count else np.nan,
                }
            )

        # Pressure/region threshold summary per cell.
        for cell_col in cell_cols:
            cell_id = int(cell_col.replace("cell_", ""))
            first_region = first_response_region.get(_v471_recording_cell_key(recording, cell_id), np.nan)
            first_label = np.nan
            first_value = np.nan
            if np.isfinite(first_region):
                match = [r for r in region_intervals_to_use if r["region_index"] == int(first_region)]
                if match:
                    first_label = match[0]["region_label"]
                    first_value = match[0].get(region_value_name, np.nan)
            pressure_threshold_rows.append(
                {
                    "recording": recording,
                    "cell_id": cell_id,
                    "first_response_region": first_region,
                    "first_response_region_label": first_label,
                    f"threshold_{region_value_name}": first_value,
                    "ever_responder": bool(np.isfinite(first_region)),
                    "passed_qc": cell_status[cell_id]["passed_qc"],
                }
            )

        # Optional plots.
        if make_plots:
            try:
                import matplotlib.pyplot as plt

                # Population mean trace.
                y = df_dff[cell_cols].mean(axis=1, skipna=True)
                plt.figure(figsize=(9.20, 3.28))
                plt.plot(df_dff["time_s"], y)
                for r in region_intervals_to_use:
                    plt.axvspan(r["region_start_s"], r["region_end_s"], alpha=0.15)
                    plt.axvline(r["region_start_s"], linestyle="--", linewidth=2.5)
                plt.xlabel("Time (s)")
                plt.ylabel("Mean ΔF/F")
                plt.title(f"{recording}: population mean ΔF/F")
                plt.tight_layout()
                pop_plot = plot_dir / f"{recording}_population_mean_DFoverF.png"
                plt.savefig(pop_plot, dpi=150)
                plt.close()

                # Heatmap sorted by peak.
                # Use robust percentile clipping so one extreme cell does not flatten
                # the useful color range for the rest of the recording.
                mat = df_dff[cell_cols].to_numpy(dtype=float).T
                finite_mat = mat[np.isfinite(mat)]
                if finite_mat.size:
                    hm_vmin = float(np.nanpercentile(finite_mat, heatmap_clip_low_percentile))
                    hm_vmax = float(np.nanpercentile(finite_mat, heatmap_clip_high_percentile))
                    if not np.isfinite(hm_vmin) or not np.isfinite(hm_vmax) or hm_vmax <= hm_vmin:
                        hm_vmin, hm_vmax = float(np.nanmin(finite_mat)), float(np.nanmax(finite_mat))
                else:
                    hm_vmin, hm_vmax = None, None

                peak_order = np.argsort(np.nanmax(mat, axis=1))[::-1]
                mat_sorted = mat[peak_order, :]
                sorted_cell_ids = [int(cell_cols[i].replace("cell_", "")) for i in peak_order]
                heatmap_row_mapping_rows = [{"heatmap_row": int(k), "cell_id": int(cid)} for k, cid in enumerate(sorted_cell_ids)]
                pd.DataFrame(heatmap_row_mapping_rows).to_csv(plot_dir / f"{recording}_DFoverF_heatmap_cellID_row_mapping.csv", index=False)
                fig, ax = plt.subplots(figsize=(9.20, 4.92))
                im = ax.imshow(
                    mat_sorted,
                    aspect="auto",
                    interpolation="nearest",
                    vmin=hm_vmin,
                    vmax=hm_vmax,
                )
                set_heatmap_cell_id_ticks(ax, sorted_cell_ids)
                cbar = fig.colorbar(im, ax=ax, label="ΔF/F")
                if hm_vmin is not None and hm_vmax is not None:
                    cbar.ax.set_title(f"≤{hm_vmin:.2g} / ≥{hm_vmax:.2g}", fontsize=8)
                ax.set_xlabel("Frame")
                ax.set_title(f"{recording}: DF/F heatmap")
                add_recording_mode_banner(fig, region_intervals_to_use, baseline_region_index)
                fig.tight_layout()
                heatmap_plot = plot_dir / f"{recording}_DFoverF_heatmap_cellIDs.png"
                fig.savefig(heatmap_plot, dpi=150)
                plt.close(fig)

                # Peak histogram.
                # Also use robust percentile x-axis limits. Cells below/above the
                # clipping bounds are counted and annotated instead of allowing one
                # outlier to dominate the histogram scale, which is especially useful
                # for strong drug responses.
                peaks = np.nanmax(mat, axis=1)
                finite_peaks = peaks[np.isfinite(peaks)]
                if finite_peaks.size:
                    h_low = float(np.nanpercentile(finite_peaks, heatmap_clip_low_percentile))
                    h_high = float(np.nanpercentile(finite_peaks, heatmap_clip_high_percentile))
                    if not np.isfinite(h_low) or not np.isfinite(h_high) or h_high <= h_low:
                        h_low, h_high = float(np.nanmin(finite_peaks)), float(np.nanmax(finite_peaks))
                    n_low = int(np.sum(finite_peaks < h_low))
                    n_high = int(np.sum(finite_peaks > h_high))
                    peaks_for_hist = finite_peaks[(finite_peaks >= h_low) & (finite_peaks <= h_high)]
                else:
                    h_low = h_high = np.nan
                    n_low = n_high = 0
                    peaks_for_hist = finite_peaks

                plt.figure(figsize=(5.52, 3.28))
                if peaks_for_hist.size:
                    plt.hist(peaks_for_hist, bins=30, range=(h_low, h_high))
                    plt.xlim(h_low, h_high)
                else:
                    plt.hist(finite_peaks, bins=30)
                plt.axvline(responder_threshold, linestyle="--", linewidth=2.5, label=f"threshold={responder_threshold:g}")
                plt.xlabel("Peak ΔF/F")
                plt.ylabel("Cell count")
                plt.title(
                    f"{recording}: peak ΔF/F distribution\n"
                    f"x clipped {heatmap_clip_low_percentile:g}–{heatmap_clip_high_percentile:g}% "
                    f"(≤{h_low:.3g}: n={n_low}, ≥{h_high:.3g}: n={n_high})"
                )
                plt.legend(loc="best", fontsize=8)
                plt.tight_layout()
                peak_plot = plot_dir / f"{recording}_peak_DFoverF_histogram.png"
                plt.savefig(peak_plot, dpi=150)
                plt.close()

                # Save a full-range version too, for audit/provenance.
                plt.figure(figsize=(5.52, 3.28))
                plt.hist(finite_peaks, bins=30)
                plt.axvline(responder_threshold, linestyle="--", linewidth=2.5, label=f"threshold={responder_threshold:g}")
                plt.xlabel("Peak ΔF/F")
                plt.ylabel("Cell count")
                plt.title(f"{recording}: peak ΔF/F distribution, full range")
                plt.legend(loc="best", fontsize=8)
                plt.tight_layout()
                peak_full_plot = plot_dir / f"{recording}_peak_DFoverF_histogram_full_range.png"
                plt.savefig(peak_full_plot, dpi=150)
                plt.close()

                # Spatial ROI heatmaps on the original/template image.
                rows_for_recording = [r for r in by_cell_rows if r.get("recording") == recording]
                # Defensive recording-isolation check for plot inputs.
                if any(r.get("recording") != recording for r in rows_for_recording):
                    warnings.append({"warning": "plot_rows_recording_mismatch", "detail": f"{recording}: rows_for_recording contained another recording"})
                    rows_for_recording = [r for r in rows_for_recording if r.get("recording") == recording]

                for _plot_family, _plot_func, _plot_kwargs in [
                    (
                        "spatial_DFoverF",
                        save_spatial_dff_heatmaps_for_recording,
                        dict(
                            recording=recording,
                            calcium_input_dir=calcium_input_dir,
                            plot_dir=plot_dir,
                            rows_for_recording=rows_for_recording,
                            responder_threshold=responder_threshold,
                            baseline_region_index=baseline_region_index,
                            heatmap_clip_low_percentile=heatmap_clip_low_percentile,
                            heatmap_clip_high_percentile=heatmap_clip_high_percentile,
                            plot_region_labels=plot_region_labels,
                        ),
                    ),
                    (
                        "focus_artifact_QC",
                        save_spatial_focus_artifact_heatmaps_for_recording,
                        dict(
                            recording=recording,
                            calcium_input_dir=calcium_input_dir,
                            plot_dir=plot_dir,
                            rows_for_recording=rows_for_recording,
                            baseline_region_index=baseline_region_index,
                            focus_cv_threshold=focus_qc_cv_threshold,
                            responder_threshold=responder_threshold,
                            heatmap_clip_low_percentile=heatmap_clip_low_percentile,
                            heatmap_clip_high_percentile=heatmap_clip_high_percentile,
                            plot_region_labels=plot_region_labels,
                        ),
                    ),
                    (
                        "cell_frame_composite",
                        save_region_cell_frame_composites_for_recording,
                        dict(
                            recording=recording,
                            calcium_input_dir=calcium_input_dir,
                            plot_dir=plot_dir,
                            df_dff=df_dff,
                            rows_for_recording=rows_for_recording,
                            region_intervals_to_use=region_intervals_to_use,
                            baseline_region_index=baseline_region_index,
                            responder_threshold=responder_threshold,
                            plot_region_labels=plot_region_labels,
                        ),
                    ),
                ]:
                    try:
                        _plot_func(**_plot_kwargs)
                    except Exception as exc:
                        warnings.append({"warning": "plot_generation_failed", "detail": f"{recording}: {_plot_family}: {exc}"})

            except Exception as exc:
                warnings.append({"warning": "plot_generation_failed", "detail": f"{recording}: {exc}"})

    # v4.76 duplicate/mis-association audit.
    try:
        for i in range(len(recording_trace_fingerprint_rows)):
            for j in range(i + 1, len(recording_trace_fingerprint_rows)):
                a = recording_trace_fingerprint_rows[i]
                b = recording_trace_fingerprint_rows[j]
                rec_a = a.get("recording", "")
                rec_b = b.get("recording", "")
                exact_same_matrix = (
                    a.get("cell_matrix_sha256", "") != ""
                    and a.get("cell_matrix_sha256", "") == b.get("cell_matrix_sha256", "")
                )
                exact_same_file = (
                    a.get("dff_file_sha256", "") != ""
                    and a.get("dff_file_sha256", "") == b.get("dff_file_sha256", "")
                )
                corr = _v76_mean_trace_corr(
                    recording_mean_traces_for_audit.get(rec_a, []),
                    recording_mean_traces_for_audit.get(rec_b, []),
                )
                pair_row = {
                    "recording": f"{rec_a} vs {rec_b}",
                    "warning": "recording_trace_similarity_audit",
                    "detail": f"mean_trace_corr={corr:.6g}; exact_same_matrix={exact_same_matrix}; exact_same_file={exact_same_file}",
                }
                if exact_same_matrix or exact_same_file:
                    pair_row["warning"] = "possible_recording_mixup_identical_trace_matrix"
                    warnings.append(pair_row)
                elif np.isfinite(corr) and corr >= 0.9999:
                    pair_row["warning"] = "possible_recording_mixup_highly_similar_mean_trace"
                    warnings.append(pair_row)
    except Exception as exc:
        warnings.append({"warning": "recording_similarity_audit_failed", "detail": str(exc)})

    detected_rows = []
    for recording, stim_times in detected_stims_by_recording.items():
        for i, stim_time in enumerate(stim_times, start=1):
            detected_rows.append(
                {
                    "recording": recording,
                    "stim_file": str(stim_file_by_recording.get(recording, "")),
                    "stim_index": i,
                    "detected_stim_time_s": stim_time,
                    "stim_channel": stim_channel,
                    "source": "ultrasound_channel_rising_edge",
                }
            )

    metric_definitions = [
        {"metric": "region_index", "definition": "Sequential region number. Region 1 is baseline from recording_start_sec to first ultrasound marker; subsequent regions are marker-to-marker; final drug region ends at final marker + extra_stim_delay_sec."},
        {"metric": "region_label", "definition": "User-supplied or automatic label for the region."},
        {"metric": region_value_name, "definition": "Optional numeric region value, commonly pressure in MPa."},
        {"metric": "middle90_mean_DFoverF", "definition": "Mean ΔF/F after discarding the first 5% and final 5% of timepoints in that region."},
        {"metric": "raw_region_mean_DFoverF", "definition": "Mean ΔF/F over all region timepoints without temporal trimming."},
        {"metric": "median_DFoverF", "definition": "Median DF/F within the region."},
        {"metric": "sd_DFoverF", "definition": "Standard deviation of DF/F within the region."},
        {"metric": "auc_DFoverF_s", "definition": "Trapezoidal area under DF/F vs time curve within the region."},
        {"metric": "peak_DFoverF", "definition": "Raw maximum DF/F observed within the region. Reported for inspection, but not used alone for spike-safe responder calling."},
        {"metric": "time_to_peak_s", "definition": "Seconds from region start to the raw peak DF/F in that region."},
        {"metric": "peak_frame", "definition": "One-based movie frame number corresponding to peak_time_s for that cell/region. Used for max-expression cell-frame composite images."},
        {"metric": "cell_frame_composite_frame_map.csv", "definition": "Per-recording CSV in the plots folder mapping each cell and region to the baseline representative frame and max-expression frame used for cell-filled composite images."},
        {"metric": "baseline_vs_max responder outline", "definition": "In representative baseline-vs-max composites, bright-cyan outlines mark original Cellpose ROIs with responder_primary=True in the regional table; the filled display area is expanded by ~10% around each ROI only for visual context."},
        {"metric": "spike_safe_peak_DFoverF", "definition": "Maximum DF/F after centered rolling-median smoothing. Used for responder calling to suppress one-frame spikes."},
        {"metric": "responder_peak_smoothing_frames", "definition": "Odd-number rolling median window used for spike_safe_peak_DFoverF. Default 3 frames."},
        {"metric": "responder_min_consecutive_frames", "definition": "Minimum number of consecutive raw frames that must exceed threshold for responder calling. Default 2 frames."},
        {"metric": "max_consecutive_absolute_frames", "definition": "Longest run of raw frames exceeding the selected absolute responder metric threshold. For peak_above_baseline, this is DF/F - baseline_mean >= responder_threshold."},
        {"metric": "responder_metric_used", "definition": "Absolute responder metric used for this run: peak or peak_above_baseline."},
        {"metric": "spike_safe_peak_above_baseline_DFoverF", "definition": "spike_safe_peak_DFoverF minus baseline_mean_DFoverF. Recommended responder metric for avoiding high-baseline false positives."},
        {"metric": "max_consecutive_noise_frames", "definition": "Longest run of raw frames with DF/F >= baseline_mean + noise_sd_multiplier * baseline_sd."},
        {"metric": "baseline_sd_DFoverF", "definition": "Baseline-region standard deviation for the cell."},
        {"metric": "baseline_slope_DFoverF_per_s", "definition": "Linear slope of baseline DF/F vs time; drift metric."},
        {"metric": "response_delta_middle90_vs_baseline", "definition": "middle90_mean_DFoverF minus baseline_mean_DFoverF."},
        {"metric": "responder_absolute", "definition": "Spike-safe absolute call using responder_metric_used. For peak_above_baseline, true if spike_safe_peak_DFoverF - baseline_mean_DFoverF >= responder_threshold and the raw baseline-subtracted trace persists for at least responder_min_consecutive_frames."},
        {"metric": "responder_noise_based", "definition": "Spike-safe call: true if spike_safe_peak_DFoverF >= baseline_mean + noise_sd_multiplier * baseline_sd and the raw trace has at least responder_min_consecutive_frames above that noise threshold."},
        {"metric": "responder_absolute_raw_peak", "definition": "Raw peak-only absolute criterion, reported for debugging. Not used for responder_either when spike-safe filtering is enabled."},
        {"metric": "responder_noise_raw_peak", "definition": "Raw peak-only noise criterion, reported for debugging. Not used for responder_either when spike-safe filtering is enabled."},
        {"metric": "responder_either", "definition": "Audit/borderline flag only: true if either absolute or noise-based criterion is true. It is not used for primary responder plots/counts when responder_primary is present."},
        {"metric": "responder_both", "definition": "Strict responder flag: true only if both absolute and noise-based criteria are true."},
        {"metric": "responder_primary", "definition": "Primary responder call used for counts/outlines. Default responder_call_rule=both, so this equals responder_both."},
        {"metric": "stimulus_mode", "definition": "Region classification: baseline, detected_stimulus, time_window_control, or drug_window. time_window_control means no US pulse was detected but the expected analysis window was used for control comparison."},
        {"metric": "eligible_us_responder", "definition": "True only when responder_primary is true in a detected ultrasound stimulus region. False for no-pulse time-window controls."},
        {"metric": "responders_primary_eligible_us", "definition": "Primary responders counted only in regions with detected ultrasound pulses; no-pulse control windows contribute zero."},
        {"metric": "response_confidence", "definition": "HIGH = primary responder without conservative QC risk; MEDIUM_QC_RISK = primary responder with conservative QC/artifact risk; LOW_BORDERLINE = either-only responder; NONE = non-responder."},
        {"metric": "is_baseline_reference_region", "definition": "True for the baseline/noise reference region. Baseline is not eligible to be called as a responder region."},
        {"metric": "response_class", "definition": "baseline_reference, new_responder, previous_responder, borderline_audit_not_responder, control_window_activity, control_window_borderline_audit, or non_responder across sequential non-baseline regions."},
        {"metric": "passed_qc", "definition": "True if cell passes area, F0, baseline noise, and baseline slope filters."},
        {"metric": "global_shift_y_px / global_shift_x_px", "definition": "Recording-level image shift from baseline mean image to the region middle-90% mean image, estimated by phase correlation. This is not a per-cell motion estimate."},
        {"metric": "focus_metric_region_cv", "definition": "Coefficient of variation of the brightness-normalized Laplacian sharpness metric for that ROI during the region middle-90%. High values suggest Z-focus instability."},
        {"metric": "focus_metric_delta_frac_vs_baseline", "definition": "Fractional change in the median normalized focus metric relative to the baseline/reference region."},
        {"metric": "dff_focus_correlation", "definition": "Correlation between DF/F and the normalized focus metric within the region. High absolute values suggest fluorescence may be coupled to focus changes."},
        {"metric": "floating_cell_artifact_risk", "definition": "True if focus instability is high and DF/F is correlated with focus changes, consistent with a possible floating/dead-cell focus artifact."},

        {"metric": "conservative_focus_corr_risk", "definition": "True when abs(dff_focus_correlation) exceeds qc_focus_corr_abs_threshold. Flags possible focus-coupled fluorescence artifacts such as rounded/floating cells."},
        {"metric": "baseline_instability_ratio", "definition": "Cell baseline_sd_DFoverF divided by the recording median baseline_sd_DFoverF."},
        {"metric": "conservative_baseline_instability_risk", "definition": "True when baseline_instability_ratio exceeds qc_baseline_sd_multiplier."},
        {"metric": "conservative_area_outlier_risk", "definition": "True when cell area is outside qc_min_area_percentile to qc_max_area_percentile among cells in that recording."},
        {"metric": "conservative_qc_risk", "definition": "True when any automatic conservative artifact-focused QC flag is true. It downgrades responder confidence and controls orange/dotted QC-risk outlines, but does not require response to multiple stimuli."},
        {"metric": "conservative_focus_corr_risk", "definition": "True when abs(dff_focus_correlation) exceeds qc_focus_corr_abs_threshold. This flags possible focus-coupled fluorescence artifacts such as rounded/floating cells."},
        {"metric": "fraction_responders_either", "definition": "responders_either divided by n_cells_passed_qc in that recording/region."},
        {"metric": "first_response_region", "definition": "First non-baseline region where a cell meets the primary responder criterion. Baseline/reference region is excluded."},
    
        {
            "metric": "region_start_s / region_end_s",
            "definition": "Start and end time in seconds for the full region window used to define a condition.",
        },
        {
            "metric": "middle90_start_s / middle90_end_s",
            "definition": "Start and end time in seconds for the central 90% of each region. Region means are calculated over this middle-90% window to avoid transition timing artifacts.",
        },
        {
            "metric": "region_source",
            "definition": "Whether the region boundaries came from detected ultrasound pulses or from planned expected times because no pulses were detected, as in no-US controls.",
        },
        {
            "metric": "pulses_detected",
            "definition": "True when at least one ultrasound pulse was detected in the DAQ trace for that recording; false for no-pulse/control recordings analyzed using planned expected windows.",
        },

        {
            "metric": "metric_window_start_s / metric_window_end_s",
            "definition": "The exact time window used for this row's metric calculation. Region summaries use the middle 90% of the region by default.",
        },
        {
            "metric": "region_start_s / region_end_s",
            "definition": "The full region boundary. The metric itself is calculated over the central 90% window to avoid transition artifacts.",
        },
        {
            "metric": "pulses_detected",
            "definition": "Whether ultrasound trigger pulses were detected in the DAQ channel for this recording. False can be valid for no-US controls.",
        },
]

    warnings_df = pd.DataFrame(warnings if warnings else [{"warning": "none", "detail": "No warnings."}])

    # Write final CSV outputs before optional Excel export, so openpyxl can never block CSV data.
    csv_outputs = write_stim_csv_outputs(
        analysis_dir=csv_dir,
        by_cell_rows=by_cell_rows,
        recording_summary_rows=recording_summary_rows,
        qc_rows=qc_rows,
        responder_transition_rows=responder_transition_rows,
        pressure_threshold_rows=pressure_threshold_rows,
        detected_rows=detected_rows,
        global_region_intervals=global_region_intervals,
        warnings_df=warnings_df,
        metric_definitions=metric_definitions,
        mirror_csv_dir=None,
    )


    if can_write_xlsx_with_openpyxl():
        with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
            pd.DataFrame(by_cell_rows).to_excel(writer, sheet_name="cell_region_summary", index=False)
            pd.DataFrame(recording_summary_rows).to_excel(writer, sheet_name="recording_region_summary", index=False)
            pd.DataFrame(qc_rows).to_excel(writer, sheet_name="cell_qc", index=False)
            pd.DataFrame(responder_transition_rows).to_excel(writer, sheet_name="responder_transitions", index=False)
            pd.DataFrame(pressure_threshold_rows).to_excel(writer, sheet_name="response_thresholds", index=False)
            pd.DataFrame(detected_rows).to_excel(writer, sheet_name="detected_stimuli", index=False)
            pd.DataFrame(global_region_intervals).to_excel(writer, sheet_name="detected_regions", index=False)
            pd.DataFrame(recording_trace_fingerprint_rows).to_excel(writer, sheet_name="trace_fingerprints", index=False)
            pd.DataFrame(metric_definitions).to_excel(writer, sheet_name="metric_definitions", index=False)
            warnings_df.to_excel(writer, sheet_name="warnings", index=False)

        warnings_df.to_csv(csv_dir / "warnings.csv", index=False)
    else:
        warnings.append(
            {
                "warning": "openpyxl_missing_xlsx_not_written",
                "detail": (
                    "openpyxl is not installed in this conda environment, so the Excel workbook was not written. "
                    "CSV outputs were still written. Install with: conda install -n cellpose_py310 openpyxl -y"
                ),
            }
        )
        print("[WARNING] openpyxl is not installed; skipping Excel workbook and writing CSV outputs only.")

    pd.DataFrame(recording_summary_rows).to_csv(csv_dir / "recording_region_summary.csv", index=False)
    pd.DataFrame(by_cell_rows).to_csv(csv_dir / "cell_region_summary.csv", index=False)
    pd.DataFrame(qc_rows).to_csv(csv_dir / "cell_qc.csv", index=False)
    pd.DataFrame(responder_transition_rows).to_csv(csv_dir / "responder_transitions.csv", index=False)
    pd.DataFrame(pressure_threshold_rows).to_csv(csv_dir / "response_thresholds.csv", index=False)
    pd.DataFrame(global_region_intervals).to_csv(csv_dir / "detected_regions.csv", index=False)
    pd.DataFrame(detected_rows).to_csv(csv_dir / "detected_stimuli.csv", index=False)
    pd.DataFrame(recording_trace_fingerprint_rows).to_csv(csv_dir / "recording_trace_fingerprints.csv", index=False)

    print(f"\nStimulus region analysis written to:")
    print(f"  {output_xlsx}")
    print(f"Analysis run folder:")
    print(f"  {analysis_dir}")

    if make_plots:
        html_path = reports_dir / "analysis_report.html"
        try:
            plot_files = sorted(plot_dir.glob("*.png"))
            rows = []
            rows.append("<html><head><meta charset='utf-8'><title>Calcium Analysis Report</title></head><body>")
            rows.append("<h1>Calcium Analysis Report</h1>")
            rows.append("<p>Stimulus files: per recording; see detected_stimuli and input_files.csv.</p>")
            rows.append(f"<p>Stimulus channel: {stim_channel}</p>")
            rows.append(f"<p>Workbook: {output_xlsx}</p>")
            rows.append("<h2>Warnings</h2>")
            rows.append(warnings_df.to_html(index=False))
            rows.append("<h2>Plots</h2>")
            for p in plot_files:
                rows.append(f"<h3>{p.name}</h3>")
                rows.append(f"<img src='../plots/{p.name}' style='max-width:1000px;width:95%;'>")
            rows.append("</body></html>")
            html_path.write_text("\n".join(rows))
            print(f"HTML report written to:")
            print(f"  {html_path}")
        except Exception as exc:
            print(f"[WARNING] Could not write HTML report: {exc}")


# =============================================================================
# ARGUMENTS
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Integrated DCIMG / TIFF / Cellpose / CaImAn preprocessing script."
    )

    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        choices=["dcimg", "ometiff", "downsample", "motion", "mask", "calcium", "stim", "csv"],
        help=(
            "Individual stage(s) to run in the current environment. "
            "Use --run_all for one-command sequential execution across conda envs."
        ),
    )

    parser.add_argument(
        "--run_all",
        action="store_true",
        help=(
            "Run the full pipeline sequentially using conda run to switch environments. "
            "Default sequence: dcimg/ometiff -> downsample -> motion -> mask -> calcium."
        ),
    )

    parser.add_argument(
        "--input_format",
        default="dcimg",
        choices=["dcimg", "ometiff"],
        help="Input format used by --run_all. Default: dcimg.",
    )

    parser.add_argument("--dcimg_env", default=DEFAULT_DCIMG_ENV, help="Conda env for DCIMG/OME conversion.")
    parser.add_argument("--cellpose_env", default=DEFAULT_CELLPOSE_ENV, help="Conda env for downsample and mask.")
    parser.add_argument("--caiman_env", default=DEFAULT_CAIMAN_ENV, help="Conda env for motion correction.")

    parser.add_argument(
        "--skip_motion",
        action="store_true",
        help="With --run_all, skip motion correction and automatic mask stage.",
    )

    parser.add_argument(
        "--include_csv",
        action="store_true",
        help="With --run_all, also run the CSV export stage after masks.",
    )

    parser.add_argument("--output_root", required=True, help="Root output folder.")

    parser.add_argument("--dcimg_dir", default="", help="Input folder containing .dcimg files.")
    parser.add_argument("--ome_dir", default="", help="Input folder containing subfolders with .ome.tif files.")

    parser.add_argument("--bioformats_jar", default=DEFAULT_BIOFORMATS_JAR, help="Path to bioformats_package.jar.")

    parser.add_argument("--ms_per_frame", type=float, default=DEFAULT_MS_PER_FRAME, help="Original frame interval in ms.")
    parser.add_argument("--ds_factor", type=int, default=DEFAULT_DS_FACTOR, help="Downsampling factor.")

    parser.add_argument("--snap_dir", default="", help="Optional folder containing snap/reference TIFFs.")

    parser.add_argument("--csv_input_dir", default="", help="Optional .mat input folder for CSV export.")
    parser.add_argument(
        "--csv_variable",
        default="F",
        choices=["F", "DFoverF", "auto"],
        help="MAT variable to export. Use F, DFoverF, or auto.",
    )
    parser.add_argument(
        "--timestamped_output",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For --run_all, create a new timestamped processed-data subfolder inside output_root. "
            "This prevents different runs from overwriting each other."
        ),
    )
    parser.add_argument(
        "--processed_run_name",
        default="",
        help=(
            "Optional name for the timestamped processed-data subfolder. "
            "Default: YYYY-MM-DD_HH-MM-SS."
        ),
    )
    parser.add_argument("--csv_output_dir", default="", help="Optional output folder for CSV files.")

    parser.add_argument(
        "--calcium_input_dir",
        default="",
        help="Optional folder containing TIFFs, masks_3d.mat, and bg.mat files for calcium extraction.",
    )
    parser.add_argument(
        "--baseline_frames",
        type=int,
        default=1,
        help="Number of initial frames used for F0 baseline. MATLAB default was 1.",
    )
    parser.add_argument(
        "--bleach_correction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable background-based exponential bleaching correction.",
    )

    parser.add_argument(
        "--stim_file",
        default="",
        help="Stimulus timing file (.mat or .csv) containing q and ultrasound digital channel.",
    )
    parser.add_argument(
        "--stim_dir",
        default="",
        help="Folder containing one DAQ .mat/.csv stimulus file per recording.",
    )
    parser.add_argument(
        "--stim_match_mode",
        default="name",
        choices=["name", "order"],
        help=(
            "How to match recordings to DAQ files when --stim_dir is used. "
            "Default 'name' matches by shared numeric postfix, e.g. rec00001 with Data00001. "
            "'order' matches sorted recordings to sorted DAQ files and is less safe."
        ),
    )
    parser.add_argument(
        "--stim_file_glob",
        default="*.mat",
        help="Glob pattern for DAQ files inside --stim_dir. Default: *.mat",
    )
    parser.add_argument(
        "--stim_channel",
        default="scanData1",
        help="Ultrasound stimulus channel to use for region detection. Default: scanData1. Do not use camera channel.",
    )
    parser.add_argument(
        "--stim_mat_variable",
        default="currentone",
        help="MATLAB variable name inside the DAQ .mat file. Default: currentone.",
    )
    parser.add_argument(
        "--stim_expected_times",
        default="",
        help="Comma-separated approximate expected ultrasound pulse times in seconds, e.g. 30,90,150.",
    )
    parser.add_argument(
        "--stim_search_window_sec",
        type=float,
        default=5.0,
        help="Search window around each expected stimulus time in seconds.",
    )
    parser.add_argument(
        "--stim_threshold",
        type=float,
        default=0.5,
        help="Digital threshold for ultrasound stimulus detection.",
    )
    parser.add_argument(
        "--stim_max_pulse_sec",
        type=float,
        default=1.0,
        help="Maximum allowed duration for a valid ultrasound marker pulse. Default: <1 second.",
    )
    parser.add_argument(
        "--allow_no_stim_pulses",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Allow recordings with no detected ultrasound pulses, as expected for no-US controls. "
            "When no pulses are detected, expected times are used to define planned analysis windows."
        ),
    )

    parser.add_argument(
        "--allow_missing_stim_file",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Allow stimulus analysis without a MATLAB/DAQ timing file. "
            "Requires --stim_expected_times; region boundaries are approximate and trigger QC is unavailable."
        ),
    )
    parser.add_argument(
        "--extra_stim_delay_sec",
        type=float,
        default=60.0,
        help="Add an inferred region boundary this many seconds after the final detected stimulus.",
    )
    parser.add_argument(
        "--recording_start_sec",
        type=float,
        default=0.0,
        help="Start time for baseline region. Default 0 sec.",
    )
    parser.add_argument(
        "--max_data_regions",
        type=int,
        default=999,
        help="Maximum number of data regions to summarize. Default is 999 so final drug/drug_end regions are not silently truncated.",
    )
    parser.add_argument(
        "--stim_output_xlsx",
        default="",
        help="Optional output Excel workbook path for region summary. Default goes into calcium_csv.",
    )
    parser.add_argument(
        "--analysis_run_name",
        default="",
        help="Optional name for timestamped analysis run folder. Default: current date-time.",
    )
    parser.add_argument(
        "--analysis_runs_dir",
        default="",
        help="Optional parent folder for timestamped analysis runs. Default: calcium_csv/analysis_runs.",
    )
    parser.add_argument(
        "--copy_script_snapshot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy this exact script into the timestamped analysis folder.",
    )
    parser.add_argument(
        "--save_shell_history",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save recent shell history into the timestamped analysis folder when available.",
    )
    parser.add_argument(
        "--shell_history_lines",
        type=int,
        default=50,
        help="Number of recent shell history lines to save.",
    )
    parser.add_argument(
        "--save_conda_envs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save conda environment exports and conda list files into the analysis folder.",
    )
    parser.add_argument(
        "--conda_env_names",
        default="dcimg,cellpose_py310,caiman_py310",
        help="Comma-separated conda environments to export for reproducibility.",
    )

    parser.add_argument(
        "--region_labels",
        default="",
        help="Optional comma-separated labels for detected regions/tables, e.g. baseline,stim1_to_stim2,stim2_to_stim3,stim3_to_drug,AITC.",
    )
    parser.add_argument(
        "--plot_region_labels",
        default="",
        help="Optional comma-separated short labels used only in plot titles/montages, e.g. 1MPa,2MPa,3MPa,AITC.",
    )
    parser.add_argument(
        "--region_values",
        default="",
        help="Optional comma-separated numeric values for regions, e.g. pressure MPa: 0,1,1.5,2.",
    )
    parser.add_argument(
        "--region_value_name",
        default="region_value",
        help="Name for numeric region value, e.g. pressure_MPa.",
    )
    parser.add_argument(
        "--responder_threshold",
        type=float,
        default=0.25,
        help=(
            "Responder threshold in DF/F units. Interpretation depends on --responder_metric. "
            "For peak_above_baseline, 0.25 means the spike-safe peak must rise at least 25% "
            "above that cell's baseline mean."
        ),
    )
    parser.add_argument(
        "--responder_metric",
        choices=["peak", "peak_above_baseline"],
        default="peak_above_baseline",
        help=(
            "Metric used for absolute responder calls. peak = old behavior, uses spike_safe_peak_DFoverF. "
            "peak_above_baseline = recommended behavior, uses spike_safe_peak_DFoverF - baseline_mean_DFoverF."
        ),
    )
    parser.add_argument(
        "--responder_call_rule",
        choices=["both", "either"],
        default="both",
        help=(
            "Primary responder rule. both = recommended stringent call requiring absolute amplitude AND noise-based criteria. "
            "either = permissive legacy/audit behavior."
        ),
    )
    parser.add_argument(
        "--noise_sd_multiplier",
        type=float,
        default=3.0,
        help="Noise-based responder threshold multiplier using baseline SD.",
    )
    parser.add_argument(
        "--responder_peak_smoothing_frames",
        type=int,
        default=3,
        help=(
            "Rolling-median window, in frames, used for spike-safe responder peak calling. "
            "Odd values are best; even values are rounded up. Use 1 to disable smoothing."
        ),
    )
    parser.add_argument(
        "--responder_min_consecutive_frames",
        type=int,
        default=2,
        help=(
            "Minimum number of consecutive raw frames that must exceed threshold for a responder call. "
            "Use 1 to allow single-frame events."
        ),
    )
    parser.add_argument(
        "--baseline_region_index",
        type=int,
        default=1,
        help="Region index used as baseline/noise reference. Default: first region.",
    )
    parser.add_argument(
        "--min_cell_area_px",
        type=int,
        default=0,
        help="Minimum mask area in pixels to pass QC.",
    )
    parser.add_argument(
        "--min_F0",
        type=float,
        default=-1e18,
        help="Minimum F0 to pass QC. Default keeps all finite F0 values.",
    )
    parser.add_argument(
        "--max_baseline_sd",
        type=float,
        default=1e18,
        help="Maximum baseline SD to pass QC.",
    )
    parser.add_argument(
        "--max_abs_baseline_slope",
        type=float,
        default=1e18,
        help="Maximum absolute baseline slope in DF/F per second to pass QC.",
    )
    parser.add_argument(
        "--focus_qc_padding_px",
        type=int,
        default=6,
        help="Padding around each Cellpose ROI crop for focus-artifact QC. Default: 6 px.",
    )
    parser.add_argument(
        "--focus_qc_cv_threshold",
        type=float,
        default=0.25,
        help="Flag focus instability when normalized focus metric CV is at least this value. Default: 0.25.",
    )
    parser.add_argument(
        "--focus_qc_corr_threshold",
        type=float,
        default=0.5,
        help="Flag floating-cell artifact risk when abs(DF/F vs focus correlation) is at least this value. Default: 0.5.",
    )

    parser.add_argument(
        "--qc_focus_corr_abs_threshold",
        type=float,
        default=0.70,
        help="Conservative QC: flag cells when abs(dff_focus_correlation) is above this value. Use 0.95 to nearly disable."
    )
    parser.add_argument(
        "--qc_baseline_sd_multiplier",
        type=float,
        default=2.5,
        help="Auto QC: flag cells whose baseline_sd_DFoverF is greater than this multiplier times the recording median baseline SD.",
    )
    
    parser.add_argument(
        "--qc_baseline_sd_abs_threshold",
        type=float,
        default=0.10,
        help="Auto QC: absolute per-cell baseline_sd_DFoverF threshold. Cells above this are baseline-unstable even if the whole recording is noisy.",
    )
    parser.add_argument(
        "--qc_recording_median_baseline_sd_threshold",
        type=float,
        default=0.08,
        help="Auto QC: discard/flag whole recording if median baseline_sd_DFoverF across baseline cells is above this value.",
    )
    parser.add_argument(
        "--qc_recording_high_noise_sd_threshold",
        type=float,
        default=0.10,
        help="Auto QC: cell baseline_sd_DFoverF threshold used to count high-noise cells for whole-recording QC.",
    )
    parser.add_argument(
        "--qc_recording_high_noise_fraction_threshold",
        type=float,
        default=0.25,
        help="Auto QC: discard/flag whole recording if this fraction of baseline cells has baseline_sd_DFoverF above qc_recording_high_noise_sd_threshold.",
    )
    parser.add_argument(
        "--qc_min_area_percentile",
        type=float,
        default=1.0,
        help="Auto QC: flag cells with area below this within-recording percentile.",
    )
    parser.add_argument(
        "--qc_max_area_percentile",
        type=float,
        default=99.0,
        help="Auto QC: flag cells with area above this within-recording percentile.",
    )

    parser.add_argument(
        "--focus_qc_delta_threshold",
        type=float,
        default=0.25,
        help="Flag focus instability when median focus metric changes by at least this fraction vs baseline. Default: 0.25.",
    )
    parser.add_argument(
        "--heatmap_clip_low_percentile",
        type=float,
        default=5.0,
        help="Lower percentile for robust spatial heatmap color scaling. Values below are shown with colorbar <=. Default: 5.",
    )
    parser.add_argument(
        "--heatmap_clip_high_percentile",
        type=float,
        default=95.0,
        help="Upper percentile for robust spatial heatmap color scaling. Values above are shown with colorbar >=. Default: 95.",
    )
    parser.add_argument(
        "--make_plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create QC plots and an HTML report.",
    )

    parser.add_argument("--max_shifts", type=parse_tuple2, default=DEFAULT_MAX_SHIFTS, help="CaImAn max_shifts as x,y.")
    parser.add_argument("--strides", type=parse_tuple2, default=DEFAULT_STRIDES, help="CaImAn strides as x,y.")
    parser.add_argument("--overlaps", type=parse_tuple2, default=DEFAULT_OVERLAPS, help="CaImAn overlaps as x,y.")
    parser.add_argument("--max_deviation_rigid", type=int, default=DEFAULT_MAX_DEVIATION_RIGID)
    parser.add_argument("--pw_rigid", action=argparse.BooleanOptionalAction, default=DEFAULT_PW_RIGID)
    parser.add_argument("--shifts_opencv", action=argparse.BooleanOptionalAction, default=DEFAULT_SHIFTS_OPENCV)
    parser.add_argument("--border_nan", default=DEFAULT_BORDER_NAN)
    parser.add_argument("--nonneg_movie", action=argparse.BooleanOptionalAction, default=DEFAULT_NONNEG_MOVIE)

    parser.add_argument(
        "--make_preview_movies",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create low-resolution MP4 preview movies after motion correction.",
    )
    parser.add_argument(
        "--preview_downscale",
        type=int,
        default=4,
        help="Spatial downscale factor for preview movies. 4 means H/4 by W/4.",
    )
    parser.add_argument(
        "--preview_fps",
        type=float,
        default=10.0,
        help="Playback frame rate for preview MP4 movies.",
    )
    parser.add_argument(
        "--preview_max_frames",
        type=int,
        default=1200,
        help="Maximum frames written to each preview movie to keep files small.",
    )

    parser.add_argument("--recursive", action="store_true", help="Search DCIMG files recursively.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs where supported.")
    parser.add_argument("--dry_run", action="store_true", help="Print actions without running them.")
    parser.add_argument("--version", action="version", version=SCRIPT_VERSION)

    return parser.parse_args()



def _argv_has_option(option_name: str) -> bool:
    """Return True if --option_name was explicitly supplied on the command line."""
    prefix = f"--{option_name}"
    neg_prefix = f"--no-{option_name}"
    for token in sys.argv[1:]:
        if token == prefix or token.startswith(prefix + "="):
            return True
        if token == neg_prefix or token.startswith(neg_prefix + "="):
            return True
    return False


def validate_explicit_stim_analysis_params(args: argparse.Namespace) -> None:
    """
    Require analysis/QC thresholds to be explicit for stim analysis.

    Rationale: these values directly affect responder calls and figure outlines.
    If they are silently taken from parser defaults, command_used.sh does not make
    it obvious what threshold generated the output.
    """
    stages = set(args.stages or [])
    stim_requested = bool(args.run_all and (getattr(args, "stim_file", "") or getattr(args, "stim_dir", "") or getattr(args, "allow_missing_stim_file", False))) or ("stim" in stages)

    if not stim_requested:
        return

    required = [
        "stim_expected_times",
        "extra_stim_delay_sec",
        "recording_start_sec",
        "max_data_regions",
        "responder_threshold",
        "responder_metric",
        "responder_call_rule",
        "noise_sd_multiplier",
        "responder_peak_smoothing_frames",
        "responder_min_consecutive_frames",
        "baseline_region_index",
        "min_cell_area_px",
        "min_F0",
        "max_baseline_sd",
        "max_abs_baseline_slope",
        "focus_qc_padding_px",
        "focus_qc_cv_threshold",
        "focus_qc_corr_threshold",
        "focus_qc_delta_threshold",
        "heatmap_clip_low_percentile",
        "heatmap_clip_high_percentile",
    ]

    missing = [name for name in required if not _argv_has_option(name)]
    if missing:
        pretty = "\n".join(f"  --{name}" for name in missing)
        raise ValueError(
            "Stim analysis now requires these responder/QC/heatmap parameters to be "
            "explicitly present in the bash command, so thresholds are not hidden defaults.\n"
            "Missing required explicit flags:\n"
            f"{pretty}\n\n"
            "Add these flags to your command_used.sh/bash command. You may use the default "
            "values, but they must be written explicitly."
        )

# =============================================================================
# MAIN
# =============================================================================


def main() -> int:
    args = parse_args()
    validate_explicit_stim_analysis_params(args)

    # For one-command full pipeline runs, create a fresh timestamped processed-data
    # output folder before any stage starts.
    create_timestamped_processed_output_root(args)

    args.output_root = str(expand_path(args.output_root))
    _log_handle, _log_path = setup_stage_logging(Path(args.output_root), getattr(args, "stages", ["run"]))
    write_command_used_file(Path(args.output_root))

    args.dcimg_dir = "" if not args.dcimg_dir else str(expand_path(args.dcimg_dir))
    args.ome_dir = "" if not args.ome_dir else str(expand_path(args.ome_dir))
    args.bioformats_jar = str(expand_path(args.bioformats_jar))
    args.snap_dir = "" if not args.snap_dir else str(expand_path(args.snap_dir))
    args.csv_input_dir = "" if not args.csv_input_dir else str(expand_path(args.csv_input_dir))
    args.csv_output_dir = "" if not args.csv_output_dir else str(expand_path(args.csv_output_dir))
    args.calcium_input_dir = "" if not getattr(args, "calcium_input_dir", "") else str(expand_path(args.calcium_input_dir))
    args.stim_file = "" if not getattr(args, "stim_file", "") else str(expand_path(args.stim_file))
    args.stim_dir = "" if not getattr(args, "stim_dir", "") else str(expand_path(args.stim_dir))
    args.stim_output_xlsx = "" if not getattr(args, "stim_output_xlsx", "") else str(expand_path(args.stim_output_xlsx))
    args.analysis_runs_dir = "" if not getattr(args, "analysis_runs_dir", "") else str(expand_path(args.analysis_runs_dir))

    if args.run_all:
        return run_all_pipeline(args)

    if args.stages is None:
        raise ValueError(
            "No stage selected. Use either:\n"
            "    --run_all\n"
            "or an individual stage, e.g.:\n"
            "    --stages dcimg"
        )

    print_header()

    output_root = expand_path(args.output_root)
    assert output_root is not None
    output_root.mkdir(parents=True, exist_ok=True)

    folders = make_folders(output_root)
    print_folders(folders)

    bioformats_jar = expand_path(args.bioformats_jar)
    assert bioformats_jar is not None

    snap_dir = expand_path(args.snap_dir)

    print("\nRun settings:")
    print(f"  stages:           {args.stages}")
    print(f"  output_root:      {output_root}")
    print(f"  ms_per_frame:     {args.ms_per_frame}")
    print(f"  ds_factor:        {args.ds_factor}")
    print(f"  downsampled dt:   {args.ms_per_frame * args.ds_factor / 1000.0:.6g} sec/frame")
    print(f"  overwrite:        {args.overwrite}")
    print(f"  dry_run:          {args.dry_run}")

    for stage in args.stages:
        print("\n" + "-" * 92)
        print(f"RUNNING STAGE: {stage}")
        print("-" * 92)

        if stage == "dcimg":
            dcimg_dir = expand_path(args.dcimg_dir)
            if dcimg_dir is None:
                raise ValueError("--dcimg_dir is required for stage dcimg.")

            stage_dcimg(
                dcimg_dir=dcimg_dir,
                output_dir=folders["multipage_tiff"],
                bioformats_jar=bioformats_jar,
                overwrite=args.overwrite,
                recursive=args.recursive,
                dry_run=args.dry_run,
            )

        elif stage == "ometiff":
            ome_dir = expand_path(args.ome_dir)
            if ome_dir is None:
                raise ValueError("--ome_dir is required for stage ometiff.")

            stage_ometiff(
                input_dir=ome_dir,
                output_dir=folders["multipage_tiff"],
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )

        elif stage == "downsample":
            stage_downsample(
                input_dir=folders["multipage_tiff"],
                output_dir=folders["downsampled"],
                ms_per_frame=args.ms_per_frame,
                ds_factor=args.ds_factor,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
            )

        elif stage == "motion":
            stage_motion_correction(
                input_dir=folders["downsampled"],
                output_dir=folders["motion_corrected"],
                max_shifts=args.max_shifts,
                strides=args.strides,
                overlaps=args.overlaps,
                max_deviation_rigid=args.max_deviation_rigid,
                pw_rigid=args.pw_rigid,
                shifts_opencv=args.shifts_opencv,
                border_nan=args.border_nan,
                nonneg_movie=args.nonneg_movie,
                dry_run=args.dry_run,
                make_preview_movies=args.make_preview_movies,
                preview_downscale=args.preview_downscale,
                preview_fps=args.preview_fps,
                preview_max_frames=args.preview_max_frames,
            )

        elif stage == "mask":
            # Default to motion corrected if present; otherwise user can still run on that folder after manual use.
            stage_tiff_to_mask(
                input_dir=folders["motion_corrected"],
                snap_dir=snap_dir,
                dry_run=args.dry_run,
            )

        elif stage == "calcium":
            calcium_input_dir = expand_path(args.calcium_input_dir) or folders["motion_corrected"]
            dt = args.ms_per_frame * args.ds_factor / 1000.0

            stage_calcium_extract(
                input_dir=calcium_input_dir,
                dt=dt,
                baseline_frames=args.baseline_frames,
                bleach_correction=args.bleach_correction,
                dry_run=args.dry_run,
            )

        elif stage == "stim":
            calcium_dir = expand_path(args.calcium_input_dir) or (folders["motion_corrected"] / "calcium_csv")
            stim_file = expand_path(args.stim_file)
            stim_dir = expand_path(args.stim_dir)
            if stim_file is None and stim_dir is None and not args.allow_missing_stim_file:
                raise ValueError("Either --stim_file or --stim_dir is required for --stages stim unless --allow_missing_stim_file is enabled.")

            stim_output_xlsx = expand_path(args.stim_output_xlsx)
            expected_times = parse_expected_times(args.stim_expected_times)

            stage_stim_region_analysis(
                output_root=output_root,
                calcium_input_dir=calcium_dir,
                stim_file=stim_file,
                stim_dir=expand_path(args.stim_dir),
                stim_file_glob=args.stim_file_glob,
                stim_match_mode=args.stim_match_mode,
                stim_channel=args.stim_channel,
                expected_times=expected_times,
                search_window_sec=args.stim_search_window_sec,
                threshold=args.stim_threshold,
                max_pulse_sec=args.stim_max_pulse_sec,
                allow_no_stim_pulses=args.allow_no_stim_pulses,
                allow_missing_stim_file=args.allow_missing_stim_file,
                extra_stim_delay_sec=args.extra_stim_delay_sec,
                recording_start_sec=args.recording_start_sec,
                max_data_regions=args.max_data_regions,
                output_xlsx=stim_output_xlsx,
                dry_run=args.dry_run,
                region_labels=parse_optional_labels(args.region_labels),
                region_values=parse_optional_float_values(args.region_values),
                region_value_name=args.region_value_name,
                plot_region_labels=parse_optional_labels(args.plot_region_labels),
                responder_threshold=args.responder_threshold,
                responder_metric=args.responder_metric,
                responder_call_rule=args.responder_call_rule,
                noise_sd_multiplier=args.noise_sd_multiplier,
                baseline_region_index=args.baseline_region_index,
                min_cell_area_px=args.min_cell_area_px,
                min_F0=args.min_F0,
                max_baseline_sd=args.max_baseline_sd,
                max_abs_baseline_slope=args.max_abs_baseline_slope,
                focus_qc_padding_px=args.focus_qc_padding_px,
                focus_qc_cv_threshold=args.focus_qc_cv_threshold,
                focus_qc_corr_threshold=args.focus_qc_corr_threshold,
                focus_qc_delta_threshold=args.focus_qc_delta_threshold,
                heatmap_clip_low_percentile=args.heatmap_clip_low_percentile,
                heatmap_clip_high_percentile=args.heatmap_clip_high_percentile,
                responder_peak_smoothing_frames=args.responder_peak_smoothing_frames,
                responder_min_consecutive_frames=args.responder_min_consecutive_frames,
                make_plots=args.make_plots,
                analysis_runs_dir=expand_path(args.analysis_runs_dir),
                analysis_run_name=args.analysis_run_name,
                copy_script_snapshot=args.copy_script_snapshot,
                args_dict=vars(args),
            )

        elif stage == "csv":
            csv_input_dir = expand_path(args.csv_input_dir) or folders["motion_corrected"]
            csv_output_dir = expand_path(args.csv_output_dir)

            stage_mat_to_csv(
                input_dir=csv_input_dir,
                variable=args.csv_variable,
                output_dir=csv_output_dir,
                dry_run=args.dry_run,
            )

        else:
            raise ValueError(f"Unknown stage: {stage}")

    print("\n[COMPLETE] Requested stages finished.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())