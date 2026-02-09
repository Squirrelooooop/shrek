matFiles = dir(fullfile('/Users/sunny/Desktop/extra/multipage_tiff/ds/motion_corrected/', '*.mat'));

for k = 1:length(matFiles)
    % Load the .mat file
    data = load(fullfile(matFiles(k).folder, matFiles(k).name));
    
    % Choose the variable to save (example: 'F')
    if isfield(data, 'F')
        arrayToSave = data.F;
    elseif isfield(data, 'DFoverF')
        arrayToSave = data.DFoverF;
    else
        warning('No recognized variable in %s', matFiles(k).name);
        continue;
    end
    
    % Create cell ID vector (1..nCells)
    nCells = size(arrayToSave, 2);
    cellIDs = 1:nCells;

    % Combine IDs as first row for CSV
    arrayToWrite = [cellIDs; data.DFoverF];  % first row = cell IDs

    % Write to CSV (same folder as .mat)
    csvFilename = fullfile(matFiles(k).folder, strrep(matFiles(k).name, '.mat', '.csv'));
    writematrix(arrayToWrite, csvFilename);

    fprintf('Saved %s\n', csvFilename);
end
