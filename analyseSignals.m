function [decomps, decompSums] = analyseSignals(signalFiles, outArgs)
    arguments
        signalFiles (1,:) string {mustBeFile} = "./data/signals/Max_Avg_Rate_Gbps.mat"
        outArgs.DecompDir (1,1) string = "./data/decompositions"
        outArgs.CutoffDate datetime = datetime.empty  % Optional cutoff date for causal decomposition
    end

    [~, measures] = fileparts(signalFiles);

    if isfield(outArgs, "DecompDir") && ~isempty(outArgs.DecompDir)
        outDir = outArgs.DecompDir;
        if ~isfolder(outDir)
            mkdir(outDir)
        end
        csvDir = fullfile(outDir, "csv");
        if ~isfolder(csvDir)
            mkdir(csvDir);
        end
    end

    decomps = struct();
    decompSums = struct();
    for i = 1:numel(measures)
        load(signalFiles(i), "signal");
        
        % If cutoff date is provided, truncate signal to data up to that date (causal)
        if ~isempty(outArgs.CutoffDate)
            signal = signal(signal.Time <= outArgs.CutoffDate, :);
            if height(signal) == 0
                warning("No data available before cutoff date %s for %s", outArgs.CutoffDate, measures(i));
                continue;
            end
        end

        holes = array2timetable( ...
            isnan(signal{:,:}), ...
            RowTimes=signal.Time, ...
            VariableNames=signal.Properties.VariableNames ...
        );

        signal = array2timetable( ...
            fillgaps(signal{:,:}), ...
            RowTimes=signal.Time, ...
            VariableNames=signal.Properties.VariableNames ...
        );
        decomp = decomposeSignal(signal);
        decomp.Holes = holes;

        decomps.(measures(i)) = decomp;

        totalSignal = timetable(signal.Time, sum(signal{:,:}, 2), VariableNames="Sum");
        decompSum = decomposeSignal(totalSignal);

        decompSums.(measures(i)) = decompSum;

        if exist("outDir", "var")
            save(fullfile(outDir, measures(i) + "_Decomp.mat"), "decomp", "-v7.3");
            saveCsvs(fullfile(csvDir, measures(i)), decomp);

            decomp = decompSum;
            save(fullfile(outDir, measures(i) + "_Sum_Decomp.mat"), "decomp", "-v7.3");
            saveCsvs(fullfile(csvDir, measures(i) + "_Sum"), decomp);
        end
    end
end

function saveCsvs(outputFile, dataStruct)
    fields = string(fieldnames(dataStruct));
    for i = 1:numel(fields)
        writetimetable(dataStruct.(fields(i)), outputFile + "_" + fields(i) + ".csv");
    end
end
