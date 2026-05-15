% footballGamesNoTz Builds a 15-minute wall-clock activity matrix of Serie A matches
% https://fixturedownload.com/results/serie-a-2025
%
% Input:
%   saveFilePath (optional) - string specifying full path to save the output .mat file
%                            Default: './data/cleaned/matches.mat'
%
% Output:
%   matches - Cell array of size [numSlots x 2], column 1 = timestamps (string),
%             column 2 = count of simultaneous matches in each time slot
%
% The function reads match fixtures from CSV files for years 2023-2025,
% converts kickoff times to Europe/Rome timezone, constructs 15-min slots,
% and marks which slots have ongoing matches.

function matches = footballGamesNoTz(saveFilePath)
    arguments
            saveFilePath string = "./data/cleaned/matches.mat"
    end

    if nargin < 1
        saveFilePath = './data/cleaned/matches.mat';
    end

    [folderPath, ~, ~] = fileparts(saveFilePath);
    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
    end

    %% --- Parameters ---
    fmt = 'dd-MMM-yyyy HH:mm:ss';
    rome_tz = 'Europe/Rome';
    % Local file paths
    f23 = 'data/raw/serie-a-2023-UTC.csv';
    f24 = 'data/raw/serie-a-2024-UTC.csv';
    f25 = 'data/raw/serie-a-2025-UTC.csv';

    %% --- Wall-clock grid ---
    t0_wc = datetime(2024,1,1,0,0,0,'TimeZone','');    % naive (no tz)
    % Will extend to max date in data after reading CSVs

    %% --- Read and combine all fixture CSVs ---
    T23 = readtable(f23,'TextType','string');
    T24 = readtable(f24,'TextType','string');
    T25 = readtable(f25,'TextType','string');
    T = [T23; T24; T25];

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

    % Count matches ongoing in each 15-minute slot
    matchCount = sum(overlapMat, 2);

    %% --- Build output ---
    timeStrings = string(t_wc, fmt);
    matches = [cellstr(timeStrings), num2cell(matchCount)];

    fprintf('Total 15-min slots (wall-clock): %d\n', numel(t_wc));
    fprintf('Slots with matches: %d\n', sum(matchCount > 0));
    fprintf('Max simultaneous matches in a slot: %d\n', max(matchCount));

    % Save activity matrix with match counts
    save(saveFilePath, "matches");
    % Write with headers
    csvData = [{'Time', 'Indicator'}; matches];
    writecell(csvData, folderPath + "/matches.csv");

    %% --- Helper for debugging DST gap rows (optional) ---
    gapIdx =  startsWith(timeStrings, "31-Mar-2024 02:") | ...
                   startsWith(timeStrings, "30-Mar-2025 02:") ;
    disp(matches(gapIdx,:));
end