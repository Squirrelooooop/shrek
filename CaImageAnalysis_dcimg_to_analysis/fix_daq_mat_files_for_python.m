%% fix_daq_mat_files_for_python.m
% Batch-correct DAQ .mat files so Python can read q_sec, scanData1, scanData2.
%
% Problem:
%   The DAQ files currently save `currentone` as a MATLAB table object.
%   scipy.io.loadmat cannot reliably read MATLAB table/MCOS objects.
%
% This script:
%   1. Scans a DAQ folder for .mat files.
%   2. Loads each file.
%   3. Extracts q, scanData1, scanData2 from currentone.
%   4. Saves a corrected copy containing numeric arrays:
%        q_sec
%        scanData1
%        scanData2
%      plus the original currentone table when available.
%
% Recommended:
%   Run this once on the DAQ directory before the Python stim stage.

clear; clc;

%% ---------------- USER SETTINGS ----------------
% Choose DAQ folder interactively. Cancel to use current folder.
selectedFolder = uigetdir(pwd, 'Select folder containing DAQ .mat files');

if isequal(selectedFolder, 0)
    folder = pwd;
else
    folder = selectedFolder;
end

% Output mode:
%   true  = create corrected files in subfolder "python_readable"
%   false = update original .mat files in place by appending q_sec/scanData1/scanData2
makeCorrectedCopies = true;

% Corrected copy suffix, used only if makeCorrectedCopies = true.
correctedSuffix = '';

% If true, skips files that already contain q_sec, scanData1, scanData2.
skipAlreadyCorrected = true;

%% ---------------- SETUP ----------------
files = dir(fullfile(folder, '*.mat'));

if isempty(files)
    error('No .mat files found in: %s', folder);
end

if makeCorrectedCopies
    outFolder = fullfile(folder, 'python_readable');
    if ~exist(outFolder, 'dir')
        mkdir(outFolder);
    end
else
    outFolder = folder;
end

fprintf('\nDAQ folder: %s\n', folder);
fprintf('Output folder: %s\n', outFolder);
fprintf('Found %d .mat file(s).\n\n', numel(files));

summary = table( ...
    strings(0,1), strings(0,1), strings(0,1), ...
    zeros(0,1), zeros(0,1), strings(0,1), ...
    'VariableNames', {'input_file','output_file','status','n_samples','duration_sec','message'} ...
);

%% ---------------- PROCESS FILES ----------------
for i = 1:numel(files)
    inPath = fullfile(folder, files(i).name);
    [~, baseName, ~] = fileparts(files(i).name);

    fprintf('Processing %s...\n', files(i).name);

    try
        vars = whos('-file', inPath);
        varNames = string({vars.name});

        hasDirectArrays = all(ismember(["q_sec","scanData1","scanData2"], varNames));

        if hasDirectArrays && skipAlreadyCorrected
            fprintf('  Already has q_sec, scanData1, scanData2. Skipping.\n');
            summary = [summary; {string(files(i).name), "", "skipped", NaN, NaN, "already corrected"}]; %#ok<AGROW>
            continue;
        end

        S = load(inPath);

        q_sec = [];
        scanData1 = [];
        scanData2 = [];

        %% Case 1: already has direct arrays
        if isfield(S, 'q_sec') && isfield(S, 'scanData1') && isfield(S, 'scanData2')
            q_sec = S.q_sec;
            scanData1 = S.scanData1;
            scanData2 = S.scanData2;

        %% Case 2: has q instead of q_sec
        elseif isfield(S, 'q') && isfield(S, 'scanData1') && isfield(S, 'scanData2')
            q_sec = S.q;
            scanData1 = S.scanData1;
            scanData2 = S.scanData2;

        %% Case 3: currentone table
        elseif isfield(S, 'currentone')
            T = S.currentone;

            if istable(T)
                names = string(T.Properties.VariableNames);

                % Prefer explicit variable names if present.
                if all(ismember(["q","scanData1","scanData2"], names))
                    q_sec = T.q;
                    scanData1 = T.scanData1;
                    scanData2 = T.scanData2;

                % Your MATLAB table was created without explicit names:
                % currentone = table(q', scanData1', scanData2');
                % so the columns may be Var1, Var2, Var3.
                elseif width(T) >= 3
                    q_sec = T{:,1};
                    scanData1 = T{:,2};
                    scanData2 = T{:,3};
                else
                    error('currentone table has fewer than 3 columns.');
                end

            elseif isnumeric(T) && size(T,2) >= 3
                q_sec = T(:,1);
                scanData1 = T(:,2);
                scanData2 = T(:,3);

            elseif isnumeric(T) && size(T,1) >= 3
                q_sec = T(1,:)';
                scanData1 = T(2,:)';
                scanData2 = T(3,:)';

            else
                error('currentone exists but is not a readable table or numeric array.');
            end

        else
            error('No readable currentone, q_sec/scanData1/scanData2, or q/scanData1/scanData2 found.');
        end

        %% Normalize shape
        q_sec = double(q_sec(:));
        scanData1 = double(scanData1(:));
        scanData2 = double(scanData2(:));

        n = numel(q_sec);

        if numel(scanData1) ~= n || numel(scanData2) ~= n
            error('Length mismatch: q_sec=%d, scanData1=%d, scanData2=%d.', ...
                numel(q_sec), numel(scanData1), numel(scanData2));
        end

        if n == 0
            error('Extracted arrays are empty.');
        end

        duration_sec = q_sec(end) - q_sec(1);

        %% Save
        if makeCorrectedCopies
            outPath = fullfile(outFolder, [baseName correctedSuffix '.mat']);

            % Preserve original currentone when available.
            if isfield(S, 'currentone')
                currentone = S.currentone; %#ok<NASGU>
                save(outPath, 'q_sec', 'scanData1', 'scanData2', 'currentone');
            else
                save(outPath, 'q_sec', 'scanData1', 'scanData2');
            end
        else
            outPath = inPath;

            % Append without deleting original variables.
            save(outPath, 'q_sec', 'scanData1', 'scanData2', '-append');
        end

        fprintf('  Saved: %s\n', outPath);
        fprintf('  Samples: %d, duration: %.3f sec\n', n, duration_sec);

        summary = [summary; {string(files(i).name), string(outPath), "ok", n, duration_sec, ""}]; %#ok<AGROW>

    catch ME
        fprintf(2, '  ERROR: %s\n', ME.message);
        summary = [summary; {string(files(i).name), "", "error", NaN, NaN, string(ME.message)}]; %#ok<AGROW>
    end
end

%% ---------------- SAVE SUMMARY ----------------
summaryPath = fullfile(outFolder, 'daq_python_export_summary.csv');
writetable(summary, summaryPath);

fprintf('\nDone.\n');
fprintf('Summary written to:\n  %s\n', summaryPath);

if makeCorrectedCopies
    fprintf('\nUse this folder for --stim_dir in Python:\n  %s\n', outFolder);
else
    fprintf('\nOriginal files were updated in place.\n');
end
