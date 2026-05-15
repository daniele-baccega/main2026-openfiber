% Convert link utilizations into rates (excluding links that have no capacity
% information), fix inconsistencies among maximum values and choose the link
% with the highest utilization as a representative of the tunnel.
% Perform further filtering to remove tunnels that have undesirable properties,
% like too many missing values or very low average utilization. 
% 
% Input options:
%   - DataFolder: path to a directory that contains excel files with tunnel
%       data. DataFolder and all its subdirectories will be scanned recursively
%       for data files matching DataPattern
%   - DataPattern: the pattern that identifies data files. The wildcard * will
%             match any number of characters
%   - MainCapacityFile: path to the main excel file containing capacity information
%   - CapacityFolder: path to a directory that contains additional excel files
%       with capcity data. CapacityFolder and all its subdirectories will be 
%       scanned recursively for capcity files matching CapacityPattern
%   - CapacityPattern: the pattern that identifies additional capacity files.
%       The wildcard * will match any number of characters
%   - DataMatFile: Path to a .mat file containing tunnel data to load instead of
%       the excel files in DataFolder
%   - CapacityMatFile: Path to a .mat file containing capacity data to load instead of
%       MainCapacityFile and the excel files in CapacityFolder
%             
% Filtering thresholds (NB: the last three options are fractions, not percentages):
%   - MaxOverlap: filter out any tunnel that has at least MaxOverlap 15-minute slots
%       with more than one active sub-link
%   - Zero: utilization value below which a data point should be considered missing
%       during the filtering phase
%   - MaxMissingValues: filter out any tunnel where the fraction of 15-minute slots
%       where its maximum utilization is zero or NaN is at least MaxMissingValues
%   - MinAvgUtilization: filter out any tunnel which overall average utilization
%       is less than or equal to MinAvgUtilization
%   - MinMaxUtilization: filter out any tunnel which overall maximum utilization
%       is less than or equal to MinMaxUtilization
%
% Output options:
%   - RawDataFile: path to a .mat file where the raw tunnel data will be stored
%       after parsing all the excel files. The name of the variable will be
%       dataTable.
%   - RawCapacityFile: path to a .mat file where the raw capacity data will be stored
%       after parsing all the excel files. The name of the variable will be
%       capacityTable
%   - CleanedDataFile: path to a .mat file where the cleaned data will be stored.
%       The name of the variable will be tunnelData. 
%   - FilteredDataFile: path to a .mat file where the filtered data will be stored.
%       The name of the variable will be tunnelData. 
%

function tunnelData = preprocessing(inArgs, thresholds, outArgs)
    arguments
        inArgs.DataFolder string {mustBeFolder} = "./data/raw"
        inArgs.DataPattern string {mustBeNonzeroLengthText} = "Table_Tunnel w*.xls*"
        inArgs.MainCapacityFile string {mustBeFile} = "./data/raw/Capacity.xlsx"
        inArgs.CapacityFolder string {mustBeNonzeroLengthText} = "./data/raw/Capacity_Files"
        inArgs.CapacityPattern string {mustBeNonzeroLengthText} = "Capacity_31_*"
        inArgs.DataMatFile string {mustBeFile}
        inArgs.CapacityMatFile string {mustBeFile}
        inArgs.PlotTunnels logical = true
        thresholds.MaxOverlap uint32 = 150
        thresholds.Zero double {mustBeBetween(thresholds.Zero, 0, 1)} = 0
        thresholds.MaxMissingValues double {mustBeBetween(thresholds.MaxMissingValues, 0, 1)} = 0.3
        thresholds.MinAvgUtilization double {mustBeBetween(thresholds.MinAvgUtilization, 0, 1)} = 0.01
        thresholds.MinMaxUtilization double {mustBeBetween(thresholds.MinMaxUtilization, 0, 1)} = 0.1
        outArgs.RawDataFile string {mustBeNonzeroLengthText}
        outArgs.RawCapacityFile string {mustBeNonzeroLengthText}
        outArgs.CleanedDataFile string {mustBeNonzeroLengthText}
        outArgs.FilteredDataFile string {mustBeNonzeroLengthText} = "./data/cleaned/tunnelData.mat"
        outArgs.SignalsDir string {mustBeNonzeroLengthText} = "./data/signals"
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Load data from excel files  %
    % or from existing .mat files %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    if ~isfield(inArgs, "DataMatFile")
        dataTable = loadData(inArgs.DataFolder, inArgs.DataPattern);

        if isfield(outArgs, "RawDataFile")
            save(outArgs.RawDataFile, "dataTable", "-v7.3");
        end
    else
        load(inArgs.DataMatFile, "dataTable");
    end

    if ~isfield(inArgs, "CapacityMatFile")
        capacityTable = ...
            loadCapacity(inArgs.MainCapacityFile, inArgs.CapacityFolder, inArgs.CapacityPattern);

        if isfield(outArgs, "RawCapacityFile")
            save(outArgs.RawCapacityFile, "capacityTable", "-v7.3");
        end
    else
        load(inArgs.CapacityMatFile, "capacityTable");
    end


    %%%%%%%%%%%%%%%%%%%%%
    % Clean tunnel data %
    %%%%%%%%%%%%%%%%%%%%%

    % Remove links that have no capacity information or no data
    capacityTable = capacityTable(capacityTable.Capacity_Gbps > 0, :);

    dataTable = dataTable(ismember(dataTable.Link_ID, capacityTable.Link_ID), :);
    capacityTable = capacityTable(ismember(capacityTable.Link_ID, dataTable.Link_ID), :);

    dataRange = min(dataTable.Date_Time):minutes(15):max(dataTable.Date_Time);

    % Calculate the capacity of the remaining links in every 15 minute
    % slot of dataRange
    capacityTable = computeLinkCapacity(capacityTable, dataRange);
    
    % Replace zeros with NaNs
    dataTable = standardizeMissing(dataTable, 0);

    % Handle missing values in Max_Utilization fields by replacing
    % them with the corresponding average value
    rxMaxIsNan= isnan(dataTable.Rx_Max_Utilization);
    dataTable.Rx_Max_Utilization(rxMaxIsNan) = dataTable.Rx_Avg_Utilization(rxMaxIsNan);

    txMaxIsNan= isnan(dataTable.Tx_Max_Utilization);
    dataTable.Tx_Max_Utilization(txMaxIsNan) = dataTable.Tx_Avg_Utilization(txMaxIsNan);

    % Update max values
    dataTable.Max_Avg_Utilization = ...
        max(dataTable.Rx_Avg_Utilization, dataTable.Tx_Avg_Utilization);
    dataTable.Max_Max_Utilization = ...
        max(dataTable.Rx_Max_Utilization, dataTable.Tx_Max_Utilization);

    % Match utilization data with capacity information and create a complete
    % data matrix for each link
    dataTable = matchCapacity(dataTable, capacityTable, dataRange);

    % In each 15 minute slot, choose the link with the highest utilization
    % as a representative of the tunnel and count the number of slots
    % where multiple links are active at the same time 
    dataTable.Tunnel_Name = extractBefore(dataTable.Link_ID, "__");
    
    % Keep only first 5 characters (XX_YY format: province code + first id)
    dataTable.Tunnel_Name = extractBefore(dataTable.Tunnel_Name, 6);
    
    dataTable = removevars(dataTable, "Link_ID");

    dataTable = rowfun( ...
                @mergeLinks, ...
                dataTable, ...
                OutputVariableNames=["Overlaps", "Measurements"], ...
                GroupingVariables="Tunnel_Name" ...
    );

    dataTable = removevars(dataTable, "GroupCount");

    lowOverlapTunnels = dataTable.Tunnel_Name(dataTable.Overlaps < thresholds.MaxOverlap);

    % Convert measurement matrices to timetables
    dataTable = rowfun( ...
                    @(x,y) toTimetable(x,y,dataRange), ...
                    dataTable(:, [1,3]), ...
                    ExtractCellContents=true ...
    );
    dataTable = horzcat(dataTable{:,1}{:});

    tunnelData = timetable2struct(dataTable);

    if isfield(outArgs, "CleanedDataFile")
        save(outArgs.CleanedDataFile, "tunnelData", "-v7.3");
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Filter out tunnels that don't meet criteria and plot them %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    baseFigDir = './figures';
    if ~exist(baseFigDir, 'dir')
        mkdir(baseFigDir);
    end

    if inArgs.PlotTunnels
        tunnelNames = dataTable.Properties.VariableNames;
        for i = 1:numel(tunnelNames)
            tunnelName = char(tunnelNames(i));
            subTable = dataTable(:, i);
    
            if(~ismember(tunnelName, lowOverlapTunnels))
                plotTunnel(tunnelName, subTable, "max_overlap", baseFigDir, "Max_Avg_Rate_Gbps")
                plotTunnel(tunnelName, subTable, "max_overlap", baseFigDir, "Max_Max_Rate_Gbps")
            end
        end
    end

    % Remove tunnels where the overlap exceeds the threshold
    dataTable = dataTable(:, lowOverlapTunnels);

    % Manually remove tunnels by name pattern (we hide tunnel names for privacy reasons)
    manualExclude = contains(dataTable.Properties.VariableNames, ["Tunnel1", "Tunnel2", "Tunnel3"]);

    if inArgs.PlotTunnels && any(manualExclude)
        manualNames = dataTable.Properties.VariableNames(manualExclude);
        for i = 1:numel(manualNames)
            tunnelName = char(manualNames(i));
            subTable = dataTable(:, manualNames(i));

            plotTunnel(tunnelName, subTable, "manual_exclude", baseFigDir, "Max_Avg_Rate_Gbps")
            plotTunnel(tunnelName, subTable, "manual_exclude", baseFigDir, "Max_Max_Rate_Gbps")
        end
    end
    dataTable = dataTable(:, ~manualExclude);

    % Remove tunnels where the number of 15-minutes timeslots where Max_Max_Rate_Gbps
    % is zero or missing is >= thresholds.MaxMissingValues OR the maximum utilization
    % is <= thresholds.MinMaxUtilization OR the average utilization
    % is <= thresholds.MinAvgUtilization
    nTunnels = width(dataTable);
    compliantTunnels = false(1, nTunnels);
    reasons = strings(1, nTunnels);

    for i = 1:nTunnels
        maxUtil = dataTable{:, i}.Max_Max_Rate_Gbps ./ dataTable{:, i}.Capacity_Gbps;
        [compliantTunnels(i), reasons(i)] = isTunnelCompliant(maxUtil, thresholds);
    end

    if inArgs.PlotTunnels
        tunnelNames = dataTable.Properties.VariableNames;
        for i = 1:numel(tunnelNames)
            tunnelName = char(tunnelNames(i));
            subTable = dataTable(:, i);
    
            plotTunnel(tunnelName, subTable, reasons{i}, baseFigDir, "Max_Avg_Rate_Gbps")
            plotTunnel(tunnelName, subTable, reasons{i}, baseFigDir, "Max_Max_Rate_Gbps")
        end
    end

    dataTable = dataTable(:, compliantTunnels);
     
    [folderPath, ~, ~] = fileparts(outArgs.FilteredDataFile);
    if ~exist(folderPath, 'dir')
        mkdir(folderPath);
    end

    tunnelData = timetable2struct(dataTable);
    save(outArgs.FilteredDataFile, "tunnelData", "-v7.3");
    [folderPath, ~, ~] = fileparts(outArgs.FilteredDataFile);
    saveCsvs(folderPath, tunnelData)

    signal = tunnelData.Max_Max_Rate_Gbps;
    save(fullfile(outArgs.SignalsDir, "Max_Max_Rate_Gbps.mat"), "signal", "-v7.3");
    
    signal = tunnelData.Max_Avg_Rate_Gbps;
    save(fullfile(outArgs.SignalsDir, "Max_Avg_Rate_Gbps.mat"), "signal", "-v7.3");
end


%%%%%%%%%%%%%
% Functions %
%%%%%%%%%%%%%

function out = loadCapacity(mainFile, targetDir, pattern)
    % Load main capacity file
    variables = ["WDM_NE_NAME", "WDM_INTERFACE_NAME", "Daily", "Capacity_Gbps_"];

    if ~isempty(mainFile)
        opts = detectImportOptions(mainFile);
        % Enforce data types
        opts = setvartype(opts, variables(1:2), "string");
        opts = setvartype(opts, variables(3), "datetime");
        opts = setvartype(opts, variables(4), "double");
        % Establish display format for dates
        opts = setvaropts(opts, variables(3), DatetimeFormat="dd/MM/yyyy");

        % Only read desired variables
        opts.SelectedVariableNames = variables;

        mainCapacityData = readtable(mainFile, opts);

        % Make tunnel names valid matlab variable names
        mainCapacityData.WDM_NE_NAME = replace(mainCapacityData.WDM_NE_NAME, "-", "_");

        % Merge columns WDM_NE_NAME and WDM_INTERFACE_NAME to simplify future manipulation
        mainCapacityData.Link_ID = ...
            mainCapacityData.WDM_NE_NAME + "__" + mainCapacityData.WDM_INTERFACE_NAME;

        % Rename and reorder columns for consistency
        mainCapacityTable(:, ["Date", "Link_ID", "Capacity_Gbps"]) = ...
            mainCapacityData(:, [variables(3), "Link_ID", variables(4)]);

        % Remove duplicate rows (Same Link_ID and Date)
        [links, idx] = unique(mainCapacityTable(:, ["Date", "Link_ID"]));
        mainCapacityTable = mainCapacityTable(idx, :);
    end

    if ~isempty(targetDir)
        % Load additional files from targetDir
        filePattern = fullfile(targetDir, "**/" + pattern);
        files = dir(filePattern);

        % Read all files matching the pattern and merge the contents in a single table
        numFiles = length(files);
        toBeMerged = cell(1, numFiles);
        for i = 1:numFiles
            filename = string(files(i).name);
            path = fullfile(files(i).folder, filename);
            
            opts = detectImportOptions(path);
            % Enforce data types
            opts = setvartype(opts, variables(1:2), "string");
            opts = setvartype(opts, variables(4), "double");
            % Only read desired variables
            opts.SelectedVariableNames = variables([1,2,4]);

            partialData = readtable(path, opts);

            rawDate = extractBetween(filename, "Capacity_31_", ".xlsx");
            Date = datetime(extractAfter(rawDate, "_"), InputFormat="ddMMuuuu", Format="dd/MM/uuuu");
            Date = repelem(Date, height(partialData),1);
            partialData.Date = Date;
            
            toBeMerged{i} = partialData;
        end
        extraCapacityData = vertcat(toBeMerged{:});

        % Make tunnel names valid matlab variable names
        extraCapacityData.WDM_NE_NAME = replace(extraCapacityData.WDM_NE_NAME, "-", "_");

        % Merge columns WDM_NE_NAME and WDM_INTERFACE_NAME to simplify future manipulation
        extraCapacityData.Link_ID = ...
            extraCapacityData.WDM_NE_NAME + "__" + extraCapacityData.WDM_INTERFACE_NAME;

        % Rename and reorder columns for consistency
        extraCapacityTable(:, ["Date", "Link_ID", "Capacity_Gbps"]) = ...
            extraCapacityData(:, ["Date", "Link_ID", variables(4)]);

        % Remove duplicate rows (Same Link_ID and Date)
        [extraLinks, idx] = unique(extraCapacityTable(:, ["Date", "Link_ID"]));
        extraCapacityTable = extraCapacityTable(idx, :);
    end

    if isempty(mainFile)
        out = extraCapacityTable;
    elseif isempty(targetDir)
        out = mainCapacityTable;
    else
        % Merge the two tables giving precedence to the values in extraCapacityTable
        [~, mainIdx, extraIdx] = intersect(links, extraLinks);

        mainCapacityTable(mainIdx, :) = extraCapacityTable(extraIdx, :);

        out = vertcat(mainCapacityTable, setdiff(extraCapacityTable, extraCapacityTable(extraIdx,:)));
    end
end


function out = loadData(targetDir, pattern)
    filePattern = fullfile(targetDir, "**/" + pattern);
    files = dir(filePattern);

    variables = [
        "WDM_NE_NAME", "WDM_INTERFACE_NAME", "DateTime" ...
        "Ethernet_RxBandwidthUtilization", "Ethernet_RxBandwidthUtilizationMAX" ...
        "Ethernet_TxBandwidthUtilization", "Ethernet_TxBandwidthUtilizationMAX" ...
        "EthernetBandwidthUtilization_AVERAGE_", "EthernetBandwidthUtilizationMAX_MAX_"
    ];

    if isempty(gcp('nocreate'))
        num_workers = 5;                   % Set to the maximum number of workers allowed
        c = parcluster('Processes');        % Get the cluster object for the 'Threads' profile
        c.NumWorkers = num_workers;         % Set the maximum number of workers to 20
        saveProfile(c);                     % Save the change to the profile
        parpool("Processes", num_workers);  % Initialize a pool of threads
    end

    % Read all files matching the file pattern and merge the contents in a single table
    numFiles = length(files);
    toBeMerged = cell(1, numFiles);
    vars = parallel.pool.Constant(variables);
    parfor i = 1:numFiles
        path = fullfile(files(i).folder, files(i).name);
        
        opts = detectImportOptions(path, Sheet=1);
        % Enforce data types
        opts = setvartype(opts, vars.Value(1:2), "string");
        opts = setvartype(opts, vars.Value(3), "datetime");
        opts = setvartype(opts, vars.Value(4:end), "double");
        % Establish display format for dates
        opts = setvaropts(opts, vars.Value(3), DatetimeFormat="dd/MM/yyyy HH:mm");
        % Treat null values as missing
        opts = setvaropts(opts, vars.Value(4:end), TreatAsMissing=["null", "NULL"]);
        % Only read desired variables
        opts.SelectedVariableNames = vars.Value;
        
        toBeMerged{i} = readtable(path, opts);
    end
    mergedData = vertcat(toBeMerged{:});

    % Make tunnel names valid matlab variable names
    mergedData.WDM_NE_NAME = replace(mergedData.WDM_NE_NAME, "-", "_");

    % Merge columns WDM_NE_NAME and WDM_INTERFACE_NAME to simplify future manipulation
    mergedData.Link_ID = mergedData.WDM_NE_NAME + "__" + mergedData.WDM_INTERFACE_NAME; 

    % Remove duplicate measurements (Same Link_ID and DateTime)
    [~, idx] = unique(mergedData(:, ["Link_ID", "DateTime"]));
    mergedData = mergedData(idx, :);

    measures = [
        "Rx_Avg_Utilization", "Rx_Max_Utilization" ...
        "Tx_Avg_Utilization", "Tx_Max_Utilization" ...
        "Max_Avg_Utilization", "Max_Max_Utilization"
    ];

    % Rename and reorder columns for consistency
    out(:, ["Date_Time", "Link_ID", measures]) = ...
        mergedData(:, [variables(3), "Link_ID", variables(4:end)]);
end

function out = computeLinkCapacity(capacityTable, dataRange)
    [startDate, endDate] = bounds(capacityTable.Date);
    superSet = min(startDate, dataRange(1)):minutes(15):max(endDate, dataRange(end));
  
    timeTables = groupsummary( ...
                    capacityTable, ...
                    "Link_ID", ...
                    @(x,y,z) {retime(timetable(x, z, VariableNames=y(1)), superSet)}, ...
                    {1,2,3} ...
    );

    out = horzcat(timeTables{:,3}{:});

    % Infer capacity when an actual value is missing
    % giving precedence to older values
    out = fillmissing(out, "previous");
    out = fillmissing(out, "next");
    out = out(timerange(dataRange(1), dataRange(end), "closed"), :);
end

function out = matchCapacity(dataTable, capacityTable, dataRange)
    opt.range = dataRange;
    opt.capacity = capacityTable;

    matchValues = @(date, linkID, rxAvg, rxMax, txAvg, txMax, maxAvg, maxMax) ...
        matchValuesImpl(date, linkID, rxAvg, rxMax, txAvg, txMax, maxAvg, maxMax, opt);

    linkMeasurements = ...
        groupsummary(dataTable, "Link_ID", matchValues, num2cell(1:width(dataTable)));

    linkMeasurements = removevars(linkMeasurements, "GroupCount");

    out = renamevars(linkMeasurements, 2, "Measurements");
end

function out = matchValuesImpl(date, linkID, rxAvg, rxMax, txAvg, txMax, maxAvg, maxMax, opt)
    tmpTable = retime(timetable(date, rxAvg, rxMax, txAvg, txMax, maxAvg, maxMax), opt.range);
    out = {[opt.capacity.(linkID(1)) tmpTable{:,:}]};
end

function [overlaps, measurements] = mergeLinks(linkData)
    numLinks = length(linkData);
    if numLinks > 1
        allData = vertcat(linkData{:});

        maxAvg = reshape(allData(:, 6), [], numLinks);

        % Count the overlaps
        overlaps = nnz(sum(maxAvg > 0, 2) > 1);

        [~, choice] = max(maxAvg, [], 2, "linear");

        mat = allData(choice, :);
    else
        overlaps = 0;
        mat = linkData{1};
    end

    mat(:, 2:end) = (mat(:, 2:end) ./ 100) .* mat(:,1);

    measurements = {mat};
end

function ttbl = toTimetable(tunnelName, measurements, dataRange)
    measures = [
        "Capacity_Gbps" ...
        "Rx_Avg_Rate_Gbps", "Rx_Max_Rate_Gbps" ...
        "Tx_Avg_Rate_Gbps", "Tx_Max_Rate_Gbps" ...
        "Max_Avg_Rate_Gbps", "Max_Max_Rate_Gbps"
    ];
    ttbl = array2timetable(measurements, RowTimes=dataRange, VariableNames=measures);
    ttbl = mergevars(ttbl, 1:width(ttbl), NewVariableName=tunnelName, MergeAsTable=true);
    ttbl = {ttbl};
end

function [res, reason] = isTunnelCompliant(maxUtil, thresholds)
    % Initialize as compliant, no reason
    res = true;
    reason = "valid";
    
    % Identify missing or zero-utilization values
    isMissing = isnan(maxUtil) | maxUtil <= thresholds.Zero;

    % Check proportion of missing/zero values
    if nnz(isMissing) / height(maxUtil) >= thresholds.MaxMissingValues
        res = false;
        reason = "high_nan_percent";
        return;
    end

    % Check max and mean utilization thresholds
    if max(maxUtil) <= thresholds.MinMaxUtilization
        res = false;
        reason = "low_max_value";
        return;
    end

    if mean(maxUtil, "omitmissing") <= thresholds.MinAvgUtilization
        res = false;
        reason = "low_usage";
        return;
    end

    % Check for sustained low-value periods (below 1% of capacity)
    lowThreshold = 0.01; % 1% of capacity
    isLow = maxUtil < lowThreshold;
    
    % Find consecutive runs of low values
    % Pad with false at start and end to ensure proper transition detection
    isLow_padded = [false; isLow(:); false];
    changes = diff(double(isLow_padded));
    startIdx = find(changes == 1);
    endIdx = find(changes == -1) - 1;
    
    % Check if any low period is substantial (more than 1% of data)
    minLowPeriodLength = ceil(length(maxUtil) * 0.01);
    
    for i = 1:length(startIdx)
        periodLength = endIdx(i) - startIdx(i) + 1;
        if periodLength >= minLowPeriodLength
            res = false;
            reason = "low_values_period";
            return;
        end
    end
end


function out = timetable2struct(TimeTable)
    varNames = string(TimeTable.(1).Properties.VariableNames);

    out = struct();
    for i = 1:length(varNames)
        subTable = varfun( ...
            @(x) x.(varNames(i)), ...
            TimeTable ...
        );

        subTable = renamevars( ...
            subTable, ...
            1:width(subTable), ...
            extractAfter(subTable.Properties.VariableNames, "Fun_") ...
        );

        out.(varNames(i)) = subTable;
    end
end

function plotTunnel(tunnelName, subTable, reason, baseFigDir, measure)
    fullSubdirPath = fullfile(baseFigDir, measure, reason);
    if ~exist(fullSubdirPath, 'dir')
        mkdir(fullSubdirPath);
    end

    % Plotting
    fig = figure('Visible', 'off');
    plot(subTable.Time, subTable.(tunnelName).(measure), 'o')
    hold on;
    plot(subTable.Time, subTable.(tunnelName).Capacity_Gbps, 'r--', 'LineWidth', 2)
    hold off;
    xlabel('Date/Time')
    ylabel(measure)
    title(['Tunnel: ' strrep(tunnelName, '_', '-')], 'Interpreter', 'none')
    grid on
    ylim([0, max(max(subTable.(tunnelName).Capacity_Gbps), max(subTable.(tunnelName).(measure))) + 1]);

    % Save figure
    saveas(fig, fullfile(fullSubdirPath, [tunnelName '.png']))
    close(fig)
end

function saveCsvs(Directory, tunnelData)
    fields = fieldnames(tunnelData);

    for k = 1:numel(fields)
        fieldName = fields{k};
        value = tunnelData.(fieldName);
        writetimetable(value, Directory + "/" + fieldName + '.csv');
    end
end