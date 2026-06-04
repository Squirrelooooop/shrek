clear; clc;

% Folder containing TIFF files and masks
folder = '/Users/sunny/Desktop/20260427_ARPA_Demo/multipage_tiff/ds/motion_corrected/';

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
    thisPage = zeros(info(1).Height, info(1).Width, nFrames, 'double');

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

    %% --- Load background mask --
    bgpath = fullfile(folder, [expnumber, '_bg.mat']);
    if ~exist(bgpath, 'file')
        warning('Background file not found: %s', bgpath);
        continue;
    end
    load(bgpath, 'bg');  % ensure variable is bg
    
    %% --- GLOBAL BLEACHING CORRECTION (background-based) ---
    % 1) compute background bleaching trace
    bgTrace = zeros(1, nFrames);
    for t = 1:nFrames
        frame = thisPage(:,:,t);
        bgTrace(t) = mean(frame(bg), 'all');
    end
    bgTrace = bgTrace / bgTrace(1);   % normalize

    % 2) fit exponential decay
    x = 1:nFrames;
    myFitType = fittype(@(a,b,c,d,x) a*exp(-b*x.^d) + c);

    myFit = fit(x', bgTrace', myFitType, ...
        'Lower', [0,0,0,0], ...
        'Upper', [inf,inf,min(bgTrace),1], ...
        'StartPoint', [max(bgTrace)-min(bgTrace), 0, min(bgTrace), 1]);

    bleachCurve = reshape(myFit(x), [1 nFrames]);

    % 3) apply bleaching correction to entire movie
    datIn = reshape(thisPage, [], nFrames);
    datIn = datIn ./ bleachCurve;
    thisPage = reshape(datIn, size(thisPage));
    
    %% --- Extract fluorescence per cell (bleach-corrected movie) ---
    F = zeros(nFrames, nCells);
    for icell = 1:nCells
        mask = BW(:,:,icell);
        for iframe = 1:nFrames
            slice = thisPage(:,:,iframe);
            F(iframe, icell) = mean(slice(mask),'all');
        end
    end

    %% --- Process background fluorescence ---
    F_bg = zeros(nFrames, 1);

    for iframe = 1:nFrames
        frame = thisPage(:, :, iframe);
        F_bg(iframe) = mean(frame(bg), 'all');
    end

    %% --- Compute ΔF/F ---
    numBaselineFrames = 1; 
    F_corrected = F - F_bg; % normalize by background 
    F0 = mean(F_corrected(1:numBaselineFrames, :), 1);  % 1 = compute mean along rows (frames)

    DFoverF = (F_corrected - F0) ./ F0;

    %% --- Time vector ---
    dt = 0.6;                      % frame interval (s)
    t = (0:nFrames-1)' * dt;

    %% --- Save results ---
    save(fullfile(folder, ['CaIData-', expnumber, '.mat']), 't', 'F', 'DFoverF');

    fprintf('Finished %s: %d cells, %d frames\n', expnumber, nCells, nFrames);
end

fprintf('All experiments processed.\n');
