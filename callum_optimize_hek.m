clear; clc;

% Folder containing TIFF files and masks
folder = '/Users/sunny/Desktop/20260115_flyc_chloron/multipage_tiff/ds/motion_corrected/';

% List all TIFF files in the folder (ignore masks)
filelist = dir(fullfile(folder, '*.tif'));
files = {filelist.name};
files = files(~contains(files,["masks","maxproj"]));          % remove mask files if any
expnumbers = erase(files, '.tif');                % base filenames without extension

%% Loop through each experiment
for ifil = 1:length(expnumbers)
    expnumber = expnumbers{ifil};
    fprintf('Processing %s...\n', expnumber);

    %% --- Load TIFF stack ---
    tifpath = fullfile(folder, [expnumber, '.tif']);
    if ~exist(tifpath,'file')
        warning('TIFF file not found: %s', tifpath);
        continue;
    end

    info = imfinfo(tifpath);
    nFrames = numel(info);
    thisPage = zeros(info(1).Height, info(1).Width, nFrames, 'like', imread(tifpath,1));

    for k = 1:nFrames
        thisPage(:,:,k) = imread(tifpath, k);
    end

    %% --- Load 3D mask ---
    maskpath = fullfile(folder, [expnumber, '_masks_3d.mat']);
    if ~exist(maskpath,'file')
        warning('Mask file not found: %s', maskpath);
        continue;
    end
    load(maskpath, 'masks_3d');  % ensure variable is masks_3d
    BW = imbinarize(masks_3d);
    nCells = size(BW,3);

    %% --- Extract fluorescence per cell ---
    F = zeros(nFrames, nCells);

    for icell = 1:nCells
        mask = BW(:,:,icell);
        for iframe = 1:nFrames
            slice = thisPage(:,:,iframe);       % <- FIX: assign frame to temp variable
            F(iframe, icell) = mean(slice(mask),'all');
        end
    end

    %% --- Compute ΔF/F ---
    numBaselineFrames = 100; 
    F0 = mean(F(1:numBaselineFrames, :), 1);  % 1 = compute mean along rows (frames)

    DFoverF = (F - F0) ./ F0;

    %% --- Time vector ---
    dt = 0.6;                      % frame interval (s)
    t = (0:nFrames-1)' * dt;

    %% --- Save results ---
    save(fullfile(folder, ['CaIData-', expnumber, '.mat']), 't', 'F', 'DFoverF');

    fprintf('Finished %s: %d cells, %d frames\n', expnumber, nCells, nFrames);
end

fprintf('All experiments processed.\n');
