function result = decomposeSignal(signal, selector, options)
    arguments
        signal timetable
        selector (1,:) {mustBeNonempty, mustBeVariableSelector(selector, signal)} = 1:width(signal)
        options.Lag double = 96*28
        options.NumFreqs uint32 = 10
    end

    numFreqs = options.NumFreqs;
    lag = options.Lag;

    signal = signal(:, selector);

    varNames = string(signal.Properties.VariableNames);
    numVars = numel(varNames);

    trends = cell(1, numVars);
    harmonics = cell(1, numVars);
    residuals = cell(1, numVars);

    if isempty(gcp('nocreate'))
        num_workers = 5;                   % Set to the maximum number of workers allowed
        c = parcluster('Processes');        % Get the cluster object for the 'Threads' profile
        c.NumWorkers = num_workers;         % Set the maximum number of workers to 20
        saveProfile(c);                     % Save the change to the profile
        parpool("Processes", num_workers);  % Initialize a pool of threads
    end

    parfor i = 1:width(signal)
        [trends{i}, harmonics{i}, residuals{i}] = ...
            trenddecomp(signal{:,i}, "ssa", lag, NumSeasonal=numFreqs);
    end

    result.Trends = timetable(signal.Time, trends{:}, VariableNames=varNames);
    result.Seasonals = timetable(signal.Time, harmonics{:}, VariableNames=varNames);
    result.Residuals = timetable(signal.Time, residuals{:}, VariableNames=varNames);
end
