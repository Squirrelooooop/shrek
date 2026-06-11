%% mat_to_csv.m
% For each .mat file in a folder, finds the longest continuous detection
% window of var3 (scanData2) with no gap of zeros longer than 1 second,
% and saves a CSV with columns: time, var2, var3

% ============================================================
%  CHANGE THESE PATHS EACH TIME
folder     = '/Volumes/mnlscdata/ARPA/CalciumImaging/20260527_JesusHEK3_MscL_TRPA1/raw';   % input folder
out_folder = '/Volumes/mnlscdata/ARPA/CalciumImaging/20260527_JesusHEK3_MscL_TRPA1/output';  % output folder
% ============================================================

period      = 0.005;          % seconds per sample
max_gap_s   = 1.0;            % maximum allowed zero gap in seconds
max_gap_n   = round(max_gap_s / period);  % in samples (200)

% Create output folder if it doesn't exist
if ~exist(out_folder, 'dir')
    mkdir(out_folder);
    fprintf('Created output folder: %s\n', out_folder);
end


files = dir(fullfile(folder, '*.mat'));
fprintf('Found %d .mat file(s) in %s\n\n', numel(files), folder);

for i = 1:numel(files)
    fpath = fullfile(files(i).folder, files(i).name);
    [~, fname, ~] = fileparts(files(i).name);
    fprintf('Processing: %s\n', files(i).name);

    try
        data = load(fpath);

        % Get the table variable (should be 'currentone')
        varnames = fieldnames(data);
        if ~ismember('currentone', varnames)
            fprintf('  ERROR: ''currentone'' not found. Variables: %s\n', ...
                strjoin(varnames, ', '));
            continue
        end

        t_data = data.currentone;

        % Extract columns (Var1=time, Var2=scanData1, Var3=scanData2)
        var2 = t_data.Var2;
        var3 = t_data.Var3;

        % ---- Find longest continuous window ----
        n = numel(var3);
        best_start = 1;
        best_len   = 0;
        win_start  = 1;
        zero_run   = 0;
        zero_run_start = -1;

        k = 1;
        while k <= n
            if var3(k) == 0
                if zero_run == 0
                    zero_run_start = k;
                end
                zero_run = zero_run + 1;

                if zero_run > max_gap_n
                    % Window broken — record if best
                    len = zero_run_start - win_start;
                    if len > best_len
                        best_len   = len;
                        best_start = win_start;
                    end
                    % Restart window after start of bad zero run
                    win_start = zero_run_start + 1;
                    k = win_start;
                    zero_run = 0;
                    zero_run_start = -1;
                    continue
                end
            else
                zero_run = 0;
                zero_run_start = -1;
            end
            k = k + 1;
        end

        % Check final window
        len = n - win_start + 1;
        if len > best_len
            best_len   = len;
            best_start = win_start;
        end

        if best_len == 0
            fprintf('  WARNING: No valid window found, skipping.\n');
            continue
        end

        best_end = best_start + best_len - 1;

        % ---- Trim so window starts and ends with a 1 ----
        while best_start <= best_end && var3(best_start) ~= 1
            best_start = best_start + 1;
        end
        while best_end >= best_start && var3(best_end) ~= 1
            best_end = best_end - 1;
        end
        best_len = best_end - best_start + 1;

        if best_len <= 0
            fprintf('  WARNING: No valid window with var3=1 at edges, skipping.\n');
            continue
        end

        % ---- Slice and build output ----
        var2_slice = var2(best_start:best_end);
        var3_slice = var3(best_start:best_end);
        time_col   = (0:best_len-1)' * period;

        % ---- Write CSV to output folder ----
        out_path = fullfile(out_folder, [fname '.csv']);
        T = table(time_col, var2_slice, var3_slice, ...
            'VariableNames', {'time', 'var2', 'var3'});
        writetable(T, out_path);

        fprintf('  -> %d samples (%.1fs) written to %s.csv\n', ...
            best_len, best_len * period, fname);

    catch ME
        fprintf('  ERROR: %s\n', ME.message);
    end
end

fprintf('\nDone.\n');
