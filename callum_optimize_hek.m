clear; clc;

% Folder containing TIFF files and masks
folder = '/Users/sunny/Desktop/ChlorON_01292026_EQ_PosCtrl/ds/motion_corrected/';

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
    
    %% --- Load background mask --
    bgpath = fullfile(folder, [expnumber, '_bg.mat']);
    if ~exist(bgpath, 'file')
        warning('Background file not found: %s', bgpath);
        continue;
    end
    load(bgpath, 'bg');  % ensure variable is bg

    %% --- Process background fluorescence ---
    F_bg = zeros(nFrames, 1);

    for iframe = 1:nFrames
        frame = thisPage(:, :, iframe);
        F_bg(iframe) = mean(frame(bg), 'all');
    end

    %% --- Compute ΔF/F ---
    numBaselineFrames = 160; 
    F_corrected = F - F_bg; % normalize by background 
    F0 = mean(F_corrected(1:numBaselineFrames, :), 1);  % 1 = compute mean along rows (frames)

    DFoverF = (F_corrected - F0) ./ F0;

    %% --- Time vector ---
    dt = 1.8;                      % frame interval (s)
    t = (0:nFrames-1)' * dt;

    %% --- Save results ---
    save(fullfile(folder, ['CaIData-', expnumber, '.mat']), 't', 'F', 'DFoverF');

    fprintf('Finished %s: %d cells, %d frames\n', expnumber, nCells, nFrames);
end

fprintf('All experiments processed.\n');
