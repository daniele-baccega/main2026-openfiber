function mustBeVariableSelector(selector, table)
    numVars = width(table);
    msgID = "Selector: ";

    if isstring(selector) || iscellstr(selector)
        variables = table.Properties.VariableNames;
        selector = string(selector);
        missing = ~ismember(selector, variables);
        if any(missing)
            firstIdx = find(missing, 1);
            error(msgID + """" + selector(firstIdx) + """ is not a valid column name");
        end
    elseif isnumeric(selector)
        [first, last] = bounds(selector);
        if nnz(rem(selector, 1)) > 0 || first <= 0
            error(msgID + "column indices must be positive integers");
        elseif last > numVars
            error(msgID + "index " + last + " is out of bounds");
        end
    elseif islogical(selector)
        if ~any(selector)
            error(msgID + "logical array has no true values");
        elseif find(selector, 1, "last") > numVars
            error(msgID + "logical array has true values beyond the columns bounds");
        end
    else
        error(msgID + "input must be a logical array, an array of positive integers or " + ...
              "a string or cell array of column names");
    end
end