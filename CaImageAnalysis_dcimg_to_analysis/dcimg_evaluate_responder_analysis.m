%% DCIMG responder analysis from stimulus_region_analysis.xlsx
% Recreates the main goals of 20260608_analysis1.Rmd using the newer
% DCIMG analysis workbook instead of raw trace CSV files.
%
% Inputs expected in the current folder:
%   stimulus_region_analysis.xlsx
%   JesusHEK6_TRPA1_TRP_lookup.csv
%
% Main outputs:
%   dcimg_matlab_analysis_output/responder_by_recording_period.csv
%   dcimg_matlab_analysis_output/responder_summary_by_condition.csv
%   dcimg_matlab_analysis_output/new_previous_summary_by_condition.csv
%   dcimg_matlab_analysis_output/cell_region_joined.csv
%   dcimg_matlab_analysis_output/*.png
%
% Notes:
%   - Uses region_index to define periods:
%       1 baseline, 2 1MPa, 3 1.5MPa, 4 2MPa, 5 drug
%   - Uses passed_qc cells only.
%   - Does NOT globally filter to eligible_us_responder, because that would
%     remove no-US control periods. This matches the R notebook logic better.
%   - Default responder call uses pipeline responder_primary. Set
%     USE_PIPELINE_RESPONDER = false to mimic the original R cutoff of
%     peak_DFoverF > 0.25.

clear; clc;

%% ----------------------- User settings ----------------------------------
analysisFile = "stimulus_region_analysis.xlsx";
lookupFile   = "JesusHEK6_TRPA1_TRP_lookup.csv";
outDir       = "dcimg_matlab_analysis_output";

USE_PIPELINE_RESPONDER = true;     % true: responder_primary; false: peak_DFoverF > cutoff
R_STYLE_PEAK_CUTOFF    = 0.25;

% Special denominator from R notebook:
% If AITC == true and transfection == "D21", denominator is number of drug
% responders in that recording, otherwise denominator is total QC-passing cells.
USE_R_D21_AITC_DENOMINATOR = true;
SPECIAL_DENOM_TRANSFECTION = "D21";

periodOrder  = ["baseline", "1MPa", "1.5MPa", "2MPa", "drug"];
periodByIdx  = containers.Map({1,2,3,4,5}, cellstr(periodOrder));

if ~isfolder(outDir)
    mkdir(outDir);
end

%% ----------------------- Load tables ------------------------------------
cellRegion = readtable(analysisFile, "Sheet", "cell_region_summary", "VariableNamingRule", "preserve");
qc         = readtable(analysisFile, "Sheet", "cell_qc",              "VariableNamingRule", "preserve");
lookup     = readtable(lookupFile,   "VariableNamingRule", "preserve");

cellRegion = normalizeColumnTypes(cellRegion);
qc         = normalizeColumnTypes(qc);
lookup     = normalizeColumnTypes(lookup);

%% ----------------------- Identify key columns ---------------------------
recColCell   = findVar(cellRegion, ["recording"]);
cellIdCol    = findVar(cellRegion, ["cell_id", "cellID", "cell"]);
regIdxCol    = findVar(cellRegion, ["region_index"]);
regLabelCol  = findVar(cellRegion, ["region_label"]);
passedColCR  = findVar(cellRegion, ["passed_qc"]);
peakCol      = findVar(cellRegion, ["peak_DFoverF", "max_dff", "dff_peak"]);
respCol      = findVar(cellRegion, ["responder_primary", "is_responder"]);

recColQC     = findVar(qc, ["recording"]);
cellIdColQC  = findVar(qc, ["cell_id", "cellID", "cell"]);
passedColQC  = findVar(qc, ["passed_all_qc", "passed_qc"]);

lookupRecCol = findVar(lookup, ["rec#", "rec_num", "recording", "file"]);
transCol     = findVar(lookup, ["transfection"]);
usCol        = findVar(lookup, ["US"]);
aitcCol      = findVar(lookup, ["AITC"]);
notesCol     = findVar(lookup, ["Notes", "notes"], false);

%% ----------------------- Add recording numbers --------------------------
cellRegion.rec_num_for_join = recNumberFromRecording(cellRegion.(recColCell));
qc.rec_num_for_join         = recNumberFromRecording(qc.(recColQC));

if strcmp(lookupRecCol, "recording")
    lookup.rec_num_for_join = recNumberFromRecording(lookup.(lookupRecCol));
else
    lookup.rec_num_for_join = double(lookup.(lookupRecCol));
end

% Remove lookup rows without a recording number.
lookup = lookup(~isnan(lookup.rec_num_for_join), :);

%% ----------------------- Join lookup metadata ---------------------------
T = outerjoin(cellRegion, lookup, ...
    "LeftKeys", "rec_num_for_join", ...
    "RightKeys", "rec_num_for_join", ...
    "MergeKeys", true, ...
    "Type", "left");

% Remove recordings marked for exclusion in lookup Notes, matching R logic.
if notesCol ~= ""
    % After outerjoin, the name may be preserved or suffixed. Find again.
    notesColJoined = findVar(T, [notesCol, "Notes", "notes"], false);
    if notesColJoined ~= ""
        keep = ismissingStringOrEmpty(T.(notesColJoined));
        T = T(keep, :);
    end
end

%% ----------------------- Keep QC-passing cells --------------------------
qcGood = qc(toLogical(qc.(passedColQC)), :);
qcGood = unique(qcGood(:, {"rec_num_for_join", cellIdColQC}), "rows");
qcGood.Properties.VariableNames{2} = "cell_id_for_join";
T.cell_id_for_join = double(T.(cellIdCol));

T = innerjoin(T, qcGood, "Keys", {"rec_num_for_join", "cell_id_for_join"});

% Keep only rows that also passed the per-region QC flag, if present.
T = T(toLogical(T.(passedColCR)), :);

%% ----------------------- Add R-style period labels -----------------------
regionIndex = double(T.(regIdxCol));
period = strings(height(T), 1);
for i = 1:height(T)
    if isKey(periodByIdx, regionIndex(i))
        period(i) = string(periodByIdx(regionIndex(i)));
    else
        period(i) = string(T.(regLabelCol)(i));
    end
end
T.period = categorical(period, periodOrder, "Ordinal", true);

%% ----------------------- Clean condition labels -------------------------
% After join, lookup columns should be present under their original names.
transColJoined = findVar(T, [transCol, "transfection"]);
usColJoined    = findVar(T, [usCol, "US"]);
aitcColJoined  = findVar(T, [aitcCol, "AITC"]);

T.transfection_clean = string(T.(transColJoined));
T.US_clean           = normalizeTF(T.(usColJoined));
T.AITC_clean         = normalizeTF(T.(aitcColJoined));

%% ----------------------- Responder calls --------------------------------
if USE_PIPELINE_RESPONDER
    T.is_responder = toLogical(T.(respCol));
    responderDescription = "pipeline responder_primary";
else
    T.is_responder = double(T.(peakCol)) > R_STYLE_PEAK_CUTOFF;
    responderDescription = "peak_DFoverF > " + string(R_STYLE_PEAK_CUTOFF);
end

%% ----------------------- Per-recording responder stats ------------------
% One row per recording x period x transfection x US x AITC.
[G, recNumG, periodG, transG, usG, aitcG] = findgroups( ...
    T.rec_num_for_join, T.period, T.transfection_clean, T.US_clean, T.AITC_clean);

nResponder = splitapply(@(x) sum(x, "omitnan"), double(T.is_responder), G);

% Total QC-passing cells per recording.
cellKey = unique(T(:, {"rec_num_for_join", "cell_id_for_join"}), "rows");
[Gc, recNumCells] = findgroups(cellKey.rec_num_for_join);
fileTotalCells = splitapply(@numel, cellKey.cell_id_for_join, Gc);
fileTotalTable = table(recNumCells, fileTotalCells, ...
    "VariableNames", {"rec_num_for_join", "file_total_cells"});

% Number of drug responders per recording for the special D21/AITC denominator.
drugRows = T(T.period == "drug", :);
[Gd, recNumDrug] = findgroups(drugRows.rec_num_for_join);
fileDrugResponders = splitapply(@(x) sum(x, "omitnan"), double(drugRows.is_responder), Gd);
drugRespTable = table(recNumDrug, fileDrugResponders, ...
    "VariableNames", {"rec_num_for_join", "file_aitc_resps"});

responderByRecording = table(recNumG, periodG, transG, usG, aitcG, nResponder, ...
    "VariableNames", {"rec_num", "period", "transfection", "US", "AITC", "n_responder"});
responderByRecording = outerjoin(responderByRecording, fileTotalTable, ...
    "LeftKeys", "rec_num", "RightKeys", "rec_num_for_join", "MergeKeys", false, "Type", "left");
responderByRecording = removevars(responderByRecording, "rec_num_for_join");
responderByRecording = outerjoin(responderByRecording, drugRespTable, ...
    "LeftKeys", "rec_num", "RightKeys", "rec_num_for_join", "MergeKeys", false, "Type", "left");
responderByRecording = removevars(responderByRecording, "rec_num_for_join");
responderByRecording.file_aitc_resps(isnan(responderByRecording.file_aitc_resps)) = 0;

useSpecialDenom = USE_R_D21_AITC_DENOMINATOR & ...
    responderByRecording.AITC == "true" & ...
    responderByRecording.transfection == SPECIAL_DENOM_TRANSFECTION;

responderByRecording.total = responderByRecording.file_total_cells;
responderByRecording.total(useSpecialDenom) = responderByRecording.file_aitc_resps(useSpecialDenom);
responderByRecording.prop_responder = zeros(height(responderByRecording), 1);
validDenom = responderByRecording.total > 0;
responderByRecording.prop_responder(validDenom) = ...
    responderByRecording.n_responder(validDenom) ./ responderByRecording.total(validDenom);
responderByRecording.percent_responders = 100 * responderByRecording.prop_responder;

%% ----------------------- New vs previous responders ---------------------
T = sortrows(T, {"rec_num_for_join", "cell_id_for_join", "period"});
[Gcell, ~, ~] = findgroups(T.rec_num_for_join, T.cell_id_for_join);
previousResponder = false(height(T), 1);
newResponder      = false(height(T), 1);

for g = 1:max(Gcell)
    idx = find(Gcell == g);
    already = false;
    for j = 1:numel(idx)
        previousResponder(idx(j)) = T.is_responder(idx(j)) && already;
        newResponder(idx(j))      = T.is_responder(idx(j)) && ~already;
        if T.is_responder(idx(j))
            already = true;
        end
    end
end
T.previous_responder = previousResponder;
T.new_responder      = newResponder;

[Gnp, recNP, periodNP, transNP, usNP, aitcNP] = findgroups( ...
    T.rec_num_for_join, T.period, T.transfection_clean, T.US_clean, T.AITC_clean);
newN  = splitapply(@(x) sum(x, "omitnan"), double(T.new_responder), Gnp);
prevN = splitapply(@(x) sum(x, "omitnan"), double(T.previous_responder), Gnp);
newPrevByRecording = table(recNP, periodNP, transNP, usNP, aitcNP, newN, prevN, ...
    "VariableNames", {"rec_num", "period", "transfection", "US", "AITC", "New_responders", "Previous_responders"});

responderByRecording = outerjoin(responderByRecording, newPrevByRecording, ...
    "Keys", {"rec_num", "period", "transfection", "US", "AITC"}, ...
    "MergeKeys", true, "Type", "left");
responderByRecording.New_responders(isnan(responderByRecording.New_responders)) = 0;
responderByRecording.Previous_responders(isnan(responderByRecording.Previous_responders)) = 0;

%% ----------------------- Condition summaries ----------------------------
[Gsum, periodS, transS, usS, aitcS] = findgroups( ...
    responderByRecording.period, responderByRecording.transfection, responderByRecording.US, responderByRecording.AITC);

summaryByCondition = table(periodS, transS, usS, aitcS, ...
    splitapply(@numel, responderByRecording.percent_responders, Gsum), ...
    splitapply(@meanOmitNaN, responderByRecording.percent_responders, Gsum), ...
    splitapply(@stdOmitNaN, responderByRecording.percent_responders, Gsum), ...
    splitapply(@sum, responderByRecording.n_responder, Gsum), ...
    splitapply(@sum, responderByRecording.total, Gsum), ...
    "VariableNames", {"period", "transfection", "US", "AITC", "n_recordings", ...
    "mean_percent_responders", "sd_percent_responders", "sum_n_responder", "sum_total"});

summaryByCondition.sem_percent_responders = ...
    summaryByCondition.sd_percent_responders ./ sqrt(summaryByCondition.n_recordings);

[Gnp2, periodNPS, transNPS, usNPS, aitcNPS] = findgroups( ...
    responderByRecording.period, responderByRecording.transfection, responderByRecording.US, responderByRecording.AITC);

newPreviousSummary = table(periodNPS, transNPS, usNPS, aitcNPS, ...
    splitapply(@sum, responderByRecording.n_responder, Gnp2), ...
    splitapply(@sum, responderByRecording.New_responders, Gnp2), ...
    splitapply(@sum, responderByRecording.Previous_responders, Gnp2), ...
    splitapply(@sum, responderByRecording.total, Gnp2), ...
    "VariableNames", {"period", "transfection", "US", "AITC", ...
    "sum_n_responder", "sum_n_new_resp", "sum_n_prev_resp", "sum_total"});

newPreviousSummary.mean_percent_responders = safePercent(newPreviousSummary.sum_n_responder, newPreviousSummary.sum_total);
newPreviousSummary.mean_new_resp           = safePercent(newPreviousSummary.sum_n_new_resp,  newPreviousSummary.sum_total);
newPreviousSummary.mean_prev_resp          = safePercent(newPreviousSummary.sum_n_prev_resp, newPreviousSummary.sum_total);

%% ----------------------- Export tables ----------------------------------
writetable(T, fullfile(outDir, "cell_region_joined.csv"));
writetable(responderByRecording, fullfile(outDir, "responder_by_recording_period.csv"));
writetable(summaryByCondition, fullfile(outDir, "responder_summary_by_condition.csv"));
writetable(newPreviousSummary, fullfile(outDir, "new_previous_summary_by_condition.csv"));

%% ----------------------- Plots ------------------------------------------
makeMaxDffPlot(T, peakCol, outDir);
makePercentResponderPlot(responderByRecording, outDir);
makeNResponderPlot(responderByRecording, outDir);
makeNewPreviousPlot(newPreviousSummary, outDir);

%% ----------------------- Console summary --------------------------------
fprintf("\nDone. Responder call: %s\n", responderDescription);
fprintf("Output folder: %s\n", outDir);
disp(summaryByCondition);

%% ========================================================================
% Local helper functions
%% ========================================================================
function T = normalizeColumnTypes(T)
    for i = 1:width(T)
        if iscellstr(T.(i)) || ischar(T.(i))
            T.(i) = string(T.(i));
        end
    end
end

function name = findVar(T, candidates, required)
    if nargin < 3
        required = true;
    end
    names = string(T.Properties.VariableNames);
    candidates = string(candidates);
    name = "";
    for c = candidates
        idx = find(strcmpi(names, c), 1);
        if ~isempty(idx)
            name = names(idx);
            return;
        end
    end
    % Also allow MATLAB-renamed versions such as rec_ for rec#.
    cleanNames = lower(regexprep(names, "[^a-zA-Z0-9]", ""));
    for c = candidates
        cleanC = lower(regexprep(c, "[^a-zA-Z0-9]", ""));
        idx = find(cleanNames == cleanC, 1);
        if ~isempty(idx)
            name = names(idx);
            return;
        end
    end
    if required
        error("Could not find required column. Tried: %s", strjoin(candidates, ", "));
    end
end

function recNum = recNumberFromRecording(x)
    sx = string(x);
    recNum = nan(numel(sx), 1);
    for i = 1:numel(sx)
        tok = regexp(sx(i), "rec0*(\d+)", "tokens", "once");
        if isempty(tok)
            tok = regexp(sx(i), "(\d+)", "tokens", "once");
        end
        if ~isempty(tok)
            recNum(i) = str2double(tok{1});
        end
    end
end

function tf = normalizeTF(x)
    if islogical(x)
        tf = strings(numel(x), 1);
        tf(x) = "true";
        tf(~x) = "false";
        return;
    end
    if isnumeric(x)
        tf = strings(numel(x), 1);
        tf(x ~= 0 & ~isnan(x)) = "true";
        tf(x == 0) = "false";
        tf(isnan(x)) = "";
        return;
    end
    sx = lower(strtrim(string(x)));
    tf = strings(numel(sx), 1);
    tf(ismember(sx, ["t", "true", "1", "yes", "y"])) = "true";
    tf(ismember(sx, ["f", "false", "0", "no", "n"])) = "false";
end

function tf = toLogical(x)
    if islogical(x)
        tf = x;
    elseif isnumeric(x)
        tf = x ~= 0 & ~isnan(x);
    else
        sx = lower(strtrim(string(x)));
        tf = ismember(sx, ["true", "t", "1", "yes", "y"]);
    end
end

function keep = ismissingStringOrEmpty(x)
    if isnumeric(x)
        keep = isnan(x);
    else
        sx = string(x);
        keep = ismissing(sx) | strlength(strtrim(sx)) == 0;
    end
end

function y = meanOmitNaN(x)
    y = mean(x, "omitnan");
end

function y = stdOmitNaN(x)
    y = std(x, "omitnan");
end

function pct = safePercent(num, den)
    pct = zeros(size(num));
    ok = den > 0;
    pct(ok) = 100 .* num(ok) ./ den(ok);
end

function makeMaxDffPlot(T, peakCol, outDir)
    transfections = unique(T.transfection_clean, "stable");
    transfections(transfections == "") = [];
    f = figure("Color", "w", "Position", [100 100 1400 400]);
    tiledlayout(1, max(1, numel(transfections)), "TileSpacing", "compact");
    for i = 1:numel(transfections)
        nexttile;
        idx = T.transfection_clean == transfections(i);
        boxchart(T.period(idx), double(T.(peakCol)(idx)), "GroupByColor", T.US_clean(idx));
        hold on;
        swarmchart(T.period(idx), double(T.(peakCol)(idx)), 8, T.US_clean(idx), "filled", "MarkerFaceAlpha", 0.25);
        title(transfections(i));
        ylabel("Max \DeltaF/F");
        xlabel("");
        xtickangle(45);
        grid on;
    end
    sgtitle("Max \DeltaF/F by period and transfection");
    exportgraphics(f, fullfile(outDir, "max_dff_by_period.png"), "Resolution", 300);
    close(f);
end

function makePercentResponderPlot(R, outDir)
    transfections = unique(R.transfection, "stable");
    transfections(transfections == "") = [];
    f = figure("Color", "w", "Position", [100 100 1400 400]);
    tiledlayout(1, max(1, numel(transfections)), "TileSpacing", "compact");
    for i = 1:numel(transfections)
        nexttile;
        idx = R.transfection == transfections(i);
        boxchart(R.period(idx), R.percent_responders(idx), "GroupByColor", R.US(idx));
        hold on;
        swarmchart(R.period(idx), R.percent_responders(idx), 35, R.US(idx), "filled", "MarkerFaceAlpha", 0.35);
        ylim([0 110]);
        title(transfections(i));
        ylabel("% responders");
        xlabel("");
        xtickangle(45);
        grid on;
    end
    sgtitle("Responder cutoff / pipeline responder call by period");
    exportgraphics(f, fullfile(outDir, "percent_responders_by_period.png"), "Resolution", 300);
    close(f);
end

function makeNResponderPlot(R, outDir)
    transfections = unique(R.transfection, "stable");
    transfections(transfections == "") = [];
    f = figure("Color", "w", "Position", [100 100 1400 400]);
    tiledlayout(1, max(1, numel(transfections)), "TileSpacing", "compact");
    for i = 1:numel(transfections)
        nexttile;
        idx = R.transfection == transfections(i);
        boxchart(R.period(idx), R.n_responder(idx), "GroupByColor", R.US(idx));
        hold on;
        swarmchart(R.period(idx), R.n_responder(idx), 35, R.US(idx), "filled", "MarkerFaceAlpha", 0.35);
        title(transfections(i));
        ylabel("# responder cells");
        xlabel("");
        xtickangle(45);
        grid on;
    end
    sgtitle("Number of responding cells per recording and period");
    exportgraphics(f, fullfile(outDir, "n_responders_by_period.png"), "Resolution", 300);
    close(f);
end

function makeNewPreviousPlot(S, outDir)
    transfections = unique(S.transfection, "stable");
    transfections(transfections == "") = [];
    f = figure("Color", "w", "Position", [100 100 1400 450]);
    tiledlayout(1, max(1, numel(transfections)), "TileSpacing", "compact");
    for i = 1:numel(transfections)
        nexttile;
        idx = S.transfection == transfections(i);
        Ssub = sortrows(S(idx, :), {"US", "period"});
        x = categorical(strcat(string(Ssub.period), " / US=", string(Ssub.US)));
        x = reordercats(x, string(x));
        bar(x, [Ssub.mean_new_resp, Ssub.mean_prev_resp], "stacked");
        title(transfections(i));
        ylabel("% of denominator");
        xlabel("");
        xtickangle(45);
        legend({"New responder", "Previous responder"}, "Location", "best");
        grid on;
    end
    sgtitle("New vs previous responders");
    exportgraphics(f, fullfile(outDir, "new_vs_previous_responders.png"), "Resolution", 300);
    close(f);
end
