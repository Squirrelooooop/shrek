matFiles = dir(fullfile('/Users/sunny/Desktop/Data/20250725_IvanHEK_MscL/all_ome_tiff/multipage_tiff/ds/', '*.mat'));

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
    
    % Write to CSV (same folder as .mat)
    csvFilename = fullfile( ...
        matFiles(k).folder, ...
        strrep(matFiles(k).name, '.mat', '.csv') ...
    );
    
    writematrix(data.DFoverF, csvFilename);
    fprintf('Saved %s\n', csvFilename);
end
