% correlation Computes Spearman correlation matrix, performs hierarchical
% clustering with automatic optimal cut detection, and visualizes results.
%
% Inputs:
%   signals - timetable with variables to correlate
%   selector - (optional) variable selection indices or names (default: all)
%   outArgs.OutputFile - base name suffix for output files (default: "residuals")
%   outArgs.NumClusters - desired number of clusters (0 = auto-detect, default: 0)
%   outArgs.PlotDendrogram - plot dendrogram visualization (default: true)
%
% Outputs:
%   corrTable - correlation matrix as table with reordered rows/columns
%   clusterInfo - structure with cluster assignments and metrics
%
% Example:
%   [corrTable, info] = correlation(signals);
%   [corrTable, info] = correlation(signals, 1:50, OutputFile="filtered");

function [corrTable, clusterInfo] = correlation(signals, selector, outArgs)
    arguments
        signals timetable
        selector (1,:) {mustBeNonempty, mustBeVariableSelector(selector, signals)} = 1:width(signals)
        outArgs.OutputFile (1,1) string = "residuals"
        outArgs.NumClusters (1,1) double = 0
        outArgs.PlotDendrogram (1,1) logical = true
    end

    signals = signals(:, selector);
    
    labels = signals.Properties.VariableNames;
    
    % Extract base name from OutputFile (remove extension) to use as suffix
    [~, baseName, ~] = fileparts(outArgs.OutputFile);

    % Correlation matrix of z-scored variables, pairwise rows
    coeff = corr(zscore(signals{:,:}), 'Type', 'Spearman', 'Rows', 'pairwise');
    
    % Handle NaN values in correlation matrix
    coeff(isnan(coeff)) = 0;
    coeff(eye(size(coeff))==1) = 1;
    
    % Get clustered variable ordering and linkage matrix
    [order, Z] = getClusterOrder(coeff);
    
    % Find optimal cut
    [bestCut, metrics] = findOptimalCut(Z, "elbow");
    
    % Use provided numClusters or fall back to automatic detection
    numClusters = outArgs.NumClusters;
    if numClusters <= 0 || numClusters > size(coeff, 1) - 1
        numClusters = bestCut;
    end
    
    % Get cluster assignments
    clusterAssignments = getClusterAssignments(Z, numClusters);
    
    % Create output table
    corrTable = array2table(coeff(order,order), 'VariableNames', labels(order), 'RowNames', labels(order));
    
    % For visualization, create display matrix with NaN on diagonal
    corrDataViz = coeff(order,order);
    
    % Get min/max values from off-diagonal elements for color scaling
    offDiagMask = ~eye(size(corrDataViz), 'logical');
    offDiagValues = corrDataViz(offDiagMask);
    offDiagValues = offDiagValues(~isnan(offDiagValues));
    
    if ~isempty(offDiagValues)
        cMin = min(offDiagValues);
        cMax = max(offDiagValues);
    else
        cMin = 0;
        cMax = 1;
    end
    
    % Set diagonal to NaN for visualization
    corrDataViz(eye(size(corrDataViz))=='1') = NaN;

    % Plot reordered correlation matrix
    fig = figure('Visible', 'off', 'Position', [100, 100, 4500, 2000]);  % rectangular figure
    ax = axes;
    imagesc(corrDataViz);
    axis tight;              % do NOT use axis square/image
    daspect([0.5 1 1]);      % controls rectangle aspect (tune 0.3, 0.5, etc.)
    set(gca, 'XTick', 1:size(corrDataViz,2));
    set(gca, 'YTick', 1:size(corrDataViz,1));
    
    % Use city names for label visualization
    set(gca, 'XTickLabel', {});
    set(gca, 'YTickLabel', {});
    set(gca, 'FontSize', 35);
    ax.XAxis.FontSize = 38;   % larger x tick labels
    ax.YAxis.FontSize = 22;   % smaller y tick labels
    xtickangle(90);  % Rotate x-axis labels 90 degrees
    colormap(slanCM("gist_rainbow", 40));
    cb = colorbar('northoutside');
    cb.Position(4) = cb.Position(4) + 0.01;
    cb.Position(2) = cb.Position(2) + 0.05; 
    cb.Position(1) = cb.Position(1) + 0.013; 
    cb.FontSize = 50;
    clim([cMin, cMax]);
    
    % Minimize margins and padding
    ax.Position = [0.12, 0.12, 0.82, 0.80];
    
    % Add grey rectangles on the diagonal
    hold on;
    n = size(corrDataViz, 1);
    for i = 1:n
        rectangle('Position', [i-0.5, i-0.5, 1, 1], 'FaceColor', [0.5 0.5 0.5], 'EdgeColor', 'none');
    end
    
    % Add horizontal and vertical lines separating clusters
    orderedClusterAssignments = clusterAssignments(order);
    clusterBoundaries = find(diff(orderedClusterAssignments) ~= 0);
    for boundary = clusterBoundaries
        % Vertical line at cluster boundary
        line([boundary+0.5, boundary+0.5], [0.5, n+0.5], 'Color', 'black', 'LineWidth', 5);
        % Horizontal line at cluster boundary
        line([0.5, n+0.5], [boundary+0.5, boundary+0.5], 'Color', 'black', 'LineWidth', 5);
    end
    hold off;
    
    saveas(fig, "figures/correlation_" + baseName + ".png")
    close(fig)
    
    % Store cluster info
    clusterInfo.numClusters = numClusters;
    clusterInfo.bestCutAutomatic = bestCut;
    clusterInfo.clusterAssignments = clusterAssignments;
    clusterInfo.labels = labels;
    clusterInfo.metrics = metrics;
    
    % Plot dendrogram if requested
    if outArgs.PlotDendrogram
        cityLabels = arrayfun(@(x) getCityName(x), labels, UniformOutput=false);
        plotDendrogram(Z, numClusters, cityLabels, baseName);
    end
    
    % Print clustering summary
    fprintf('\n=== Hierarchical Clustering Summary ===\n');
    fprintf('Number of variables: %d\n', size(coeff, 1));
    fprintf('Automatic best cut (elbow): %d clusters\n', bestCut);
    fprintf('Using: %d clusters\n', numClusters);
    fprintf('Largest distance jump: %.4f at index %d\n', metrics.distanceDiffs(metrics.elbowIdx), metrics.elbowIdx);
    fprintf('\nCluster assignments:\n');
    for c = 1:numClusters
        clustMembers = find(clusterAssignments == c);
        clusterCityLabels = arrayfun(@(x) getCityName(x), labels(clustMembers), UniformOutput=true);
        fprintf('  Cluster %d: %s\n', c, strjoin(clusterCityLabels, ', '));
    end
    fprintf('=====================================\n\n');
end

function [order, Z] = getClusterOrder(data)
    D     = pdist(data, 'spearman');           % Statistics and ML Toolbox
    Z     = linkage(D, "average");
    order = optimalleaforder(Z, D);
end

function [bestCut, metrics] = findOptimalCut(Z, method)
    % Find optimal cut level for hierarchical clustering
    %
    % Inputs:
    %   Z - Linkage matrix from hierarchical clustering
    %   method - 'elbow' (default), 'inconsistency', or 'all'
    %
    % Outputs:
    %   bestCut - Suggested number of clusters
    %   metrics - Structure with distances, inconsistency coefficients, etc.
    
    arguments
        Z (:, 3) {mustBeNumeric}
        method string = "elbow"
    end
    
    n = size(Z, 1) + 1;  % Number of observations
    distances = Z(:, 3);  % Merge distances
    
    % Elbow method: find largest distance jump
    diffs = diff(distances);
    [~, elbowIdx] = max(diffs);
    elbowClusters = n - elbowIdx;
    
    % Inconsistency coefficient
    inconsist = inconsistent(Z, 2);
    inconsistCoeff = inconsist(:, 4);  % Standard deviation of distances
    
    % Second derivative to find inflection points
    d2 = diff(inconsistCoeff);
    
    metrics.distances = distances;
    metrics.distanceDiffs = diffs;
    metrics.elbowIdx = elbowIdx;
    metrics.elbowClusters = elbowClusters;
    metrics.inconsistencyCoeff = inconsistCoeff;
    metrics.secondDerivative = d2;
    
    if method == "elbow"
        bestCut = elbowClusters;
    elseif method == "inconsistency"
        % Find peak in second derivative
        [~, peakIdx] = max(abs(d2));
        bestCut = n - peakIdx;
    else  % "all"
        bestCut = elbowClusters;  % Default to elbow
    end
    
    bestCut = max(2, min(bestCut, n-1));  % Ensure 2 <= bestCut <= n-1
end

function plotDendrogram(Z, numClusters, labels, baseName)
    % Plot dendrogram with suggested cut line
    %
    % Inputs:
    %   Z - Linkage matrix
    %   numClusters - Number of clusters to highlight
    %   labels - (optional) City names for leaves
    %   baseName - (optional) Base name suffix for output file
    
    arguments
        Z (:, 3) {mustBeNumeric}
        numClusters (1,1) double = 5
        labels (1, :) string = string.empty
        baseName (1,1) string = ""
    end
    
    % Calculate cut height from linkage matrix
    n = size(Z, 1) + 1;
    distances = Z(:, 3);
    cutHeight = distances(n - numClusters);
    
    fig = figure('Visible', 'off', 'Position', [100, 100, 1400, 700]);
    
    % Plot dendrogram with labels if provided
    if ~isempty(labels)
        dendrogram(Z, 0, 'Labels', labels);
    else
        dendrogram(Z, 0);
    end
    
    % Add cut line
    hold on;
    line(xlim, [cutHeight, cutHeight], 'Color', 'red', 'LineWidth', 2, ...
        'LineStyle', '--', 'DisplayName', sprintf('Cut @ %d clusters', numClusters));
    hold off;
    
    xlabel('Tunnel');
    ylabel('Distance (1 - Spearman correlation)');
    title(sprintf('Hierarchical Clustering Dendrogram (Cut at %d clusters)', numClusters));
    grid on;
    
    % Save with suffix if baseName provided
    if strlength(baseName) > 0
        saveas(fig, 'figures/dendrogram_' + baseName + '.png');
    else
        saveas(fig, 'figures/dendrogram.png');
    end
    close(fig);
end

function clusterAssignments = getClusterAssignments(Z, numClusters)
    % Get cluster assignments based on cut level
    %
    % Inputs:
    %   Z - Linkage matrix
    %   numClusters - Number of clusters desired
    %
    % Outputs:
    %   clusterAssignments - Vector of cluster IDs for each observation
    
    clusterAssignments = cluster(Z, 'MaxClust', numClusters);
end

function cityName = getCityName(label)
    % Convert province abbreviation to city name
    label = string(label);
    labelChar = char(label);
    
    % Extract province code (first 2-3 characters before underscore)
    underscorePos = find(labelChar == '_', 1);
    
    if isempty(underscorePos)
        provinceCode = label;
    else
        endPos = min(3, underscorePos - 1);
        if endPos < 1
            provinceCode = string(labelChar(1));
        else
            provinceCode = string(labelChar(1:endPos));
        end
    end
    
    % Map Italian province codes to city names (only cities in the dataset)
    % We hide city names for privacy reasons
    cityMap = containers.Map(...
        {'C1', 'C2', 'C3'}, {'City1', 'City2', 'City3'});
    
    % Get city name or return province code if not found
    if isKey(cityMap, char(provinceCode))
        cityName = string(cityMap(char(provinceCode)));
    else
        cityName = provinceCode;
    end
end