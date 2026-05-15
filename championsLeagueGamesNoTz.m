% championsLeagueGamesNoTz Builds a 15-minute wall-clock activity matrix of Champions League matches
% https://fixturedownload.com/results/champions-league-2025
%
% Input:
%   saveFilePath (optional) - string specifying full path to save the output .mat file
%                            Default: './data/cleaned/champions_matches.mat'
%
% Output:
%   champions_matches - Cell array of size [numSlots x 2], column 1 = timestamps (string),
%                      column 2 = count of simultaneous matches in each time slot
%
% The function reads match fixtures from CSV files for years 2023-2025,
% converts kickoff times to Europe/Rome timezone, constructs 15-min slots,
% and marks which slots have ongoing matches.

function champions_matches = championsLeagueGamesNoTz(saveFilePath)
    arguments
            saveFilePath string = "./data/cleaned/champions_matches.mat"
    end

    if nargin < 1
        saveFilePath = './data/cleaned/champions_matches.mat';
    end

    [folderPath, ~, ~] = fileparts(saveFilePath);
    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
    end

    %% --- Parameters ---
    fmt = 'dd-MMM-yyyy HH:mm:ss';
    rome_tz = 'Europe/Rome';
    % Local file paths
    f23 = 'data/raw/champions-league-2023-UTC.csv';
    f24 = 'data/raw/champions-league-2024-UTC.csv';
    f25 = 'data/raw/champions-league-2025-UTC.csv';
    
    % Italian teams that participate in Champions League
    italianTeams = ["Juventus", "Inter", "Milan", "Napoli", "Roma", "Lazio", ...
                    "Atalanta", "Bologna", "Fiorentina", "AS Roma", "AC Milan", "Inter Milan"];

    %% --- Wall-clock grid ---
    t0_wc = datetime(2024,1,1,0,0,0,'TimeZone','');    % naive (no tz)
    % Will extend to max date in data after reading CSVs

    %% --- Read and combine all fixture CSVs ---
    T23 = readtable(f23,'TextType','string');
    T24 = readtable(f24,'TextType','string');
    T25 = readtable(f25,'TextType','string');

    % Normalize column names (spaces -> underscores)
    T23.Properties.VariableNames = matlab.lang.makeValidName(T23.Properties.VariableNames, 'ReplacementStyle', 'underscore');
    T24.Properties.VariableNames = matlab.lang.makeValidName(T24.Properties.VariableNames, 'ReplacementStyle', 'underscore');
    T25.Properties.VariableNames = matlab.lang.makeValidName(T25.Properties.VariableNames, 'ReplacementStyle', 'underscore');

    % Keep only columns that exist in all files (2023 has an extra Group column)
    commonVars = intersect(T23.Properties.VariableNames, T24.Properties.VariableNames, 'stable');
    commonVars = intersect(commonVars, T25.Properties.VariableNames, 'stable');
    T23 = T23(:, commonVars);
    T24 = T24(:, commonVars);
    T25 = T25(:, commonVars);

    T = [T23; T24; T25];

    %% --- Mark matches with Italian teams (but keep all matches) ---
    hasItalianTeam = ismember(T.HomeTeam, italianTeams) | ismember(T.AwayTeam, italianTeams);
    fprintf('Total Champions League matches: %d\n', height(T));
    fprintf('Matches with Italian teams: %d\n', sum(hasItalianTeam));
    
    %% --- Parse kickoff times and convert to Rome local time ---
    hasDate = ~ismissing(T.Date) & T.Date ~= "";
    dateFormat = 'dd/MM/yyyy HH:mm'; % adapt if seconds present
    utcTimes = datetime(T.Date(hasDate),'InputFormat',dateFormat,'TimeZone','UTC');
    
    % Extend wall-clock grid to include all matches in data
    maxDate = max(utcTimes);
    maxDate.TimeZone = '';
    t1_wc = maxDate + hours(3);  % extend beyond last match
    t_wc  = (t0_wc:minutes(15):t1_wc)';
    
    koRome   = utcTimes;
    koRome.TimeZone = rome_tz;

    % Define match window (kickoff -15min to +2 hours +15min), tz-aware datetimes
    matchStart_tz = koRome;
    matchEnd_tz   = koRome + hours(2) + minutes(15);

    % Convert to wall-clock (timezone-naive)
    matchStart = matchStart_tz;
    matchStart.TimeZone = '';
    matchEnd   = matchEnd_tz;
    matchEnd.TimeZone   = '';

    % Prune out-of-range matches
    keep = matchEnd   >= t0_wc & matchStart <= (t1_wc + minutes(15));
    matchStart = matchStart(keep);
    matchEnd   = matchEnd(keep);

    %% --- Mark intervals that overlap matches (all wall-clock/naive) ---
    slotStart = t_wc;
    slotEnd   = t_wc + minutes(15);

    % Logical matrix: rows=time slots, columns=matches; true if overlap
    overlapMat = (slotStart >= matchStart') & (slotStart < matchEnd') ...
              | (slotEnd   >  matchStart') & (slotEnd   <= matchEnd') ...
              | (slotStart <= matchStart') & (slotEnd   >= matchEnd');

    % Count ONLY matches with Italian teams (put 0 for non-Italian matches)
    italianTeamMatches_keep = hasItalianTeam(keep);  % Apply same 'keep' filter
    overlapMat_italian = overlapMat;
    overlapMat_italian(:, ~italianTeamMatches_keep) = false;  % Zero out non-Italian matches
    
    % Count matches ongoing in each 15-minute slot
    matchCount = sum(overlapMat_italian, 2);

    %% --- Build output ---
    timeStrings = string(t_wc, fmt);
    champions_matches = [cellstr(timeStrings), num2cell(matchCount)];

    fprintf('Total 15-min slots (wall-clock): %d\n', numel(t_wc));
    fprintf('Slots with Italian team Champions League matches: %d\n', sum(matchCount > 0));
    fprintf('Max simultaneous Italian team matches in a slot: %d\n', max(matchCount));

    % Save activity matrix with match counts
    save(saveFilePath, "champions_matches");
    % Write with headers
    csvData = [{'Time', 'Indicator'}; champions_matches];
    writecell(csvData, folderPath + "/champions_matches.csv");

    %% --- Helper for debugging DST gap rows (optional) ---
    gapIdx =  startsWith(timeStrings, "31-Mar-2024 02:") | ...
                   startsWith(timeStrings, "30-Mar-2025 02:") ;
    disp(champions_matches(gapIdx,:));
end
