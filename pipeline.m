addpath('util/treeClass/')
addpath('util/colormaps/')
addpath('util/')

% Input options
measure = "Max_Max_Rate_Gbps";
signalsDir = "./data/signals";
decompDir = "./data/decompositions";
resultDir = "./data/events";

% Create directories if they don't exist
if ~isfolder(signalsDir)
    mkdir(signalsDir);
end
if ~isfolder(decompDir)
    mkdir(decompDir);
end
if ~isfolder(resultDir)
    mkdir(resultDir);
end

% Preprocess data
preprocessing(SignalsDir=signalsDir);

% Extract trend, seasonality, residuals and events
% 1. Standard decomposition (full data)
decomps = analyseSignals(fullfile(signalsDir, measure + ".mat"), DecompDir=decompDir);

% Generate correlation matrix
correlation(decomps.(measure).Residuals, "OutputFile", "residuals");
correlation(decomps.(measure).Trends, "OutputFile", "trends");
clear decomps;

% 2. Decompositions at specified dates
% Define cutoff dates for causal decompositions
cutoffDates = [
    datetime(2024, 9, 30);
    datetime(2024, 10, 31);
    datetime(2025, 3, 31);
    datetime(2025, 4, 30);
];

% Perform causal decomposition at each cutoff date
for cutoffDate = cutoffDates'
    dateStr = string(cutoffDate, 'yyyy-MM-dd');
    causalDecompDir = fullfile(decompDir, "cutoff_" + dateStr);
    fprintf('Decomposing up to %s ...\n', dateStr);
    analyseSignals(fullfile(signalsDir, measure + ".mat"), DecompDir=causalDecompDir, CutoffDate=cutoffDate);
end

% Generate the indicator function for Serie A matches
footballGamesNoTz();

% Generate the indicator function for Champions League matches
championsLeagueGamesNoTz();