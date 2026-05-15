function mustBeRowSelector(selector, table)
    numRows = height(table);
    msgID = "Selector: ";

    if isnumeric(selector)
        [first, last] = bounds(selector);
        if nnz(mod(selector, 1)) > 0 || first <= 0
            error(msgID + "row indices must be positive integers");
        elseif last > numRows
            error(msgID + "index " + last + " is out of bounds");
        end
    elseif islogical(selector)
        if ~any(selector)
            error(msgID + "logical array has no true values");
        elseif find(selector, 1, "last") > numRows
            error(msgID + "logical array has true values beyond the row bounds");
        end
    else
        error(msgID + "input must be a logical array or an array of positive integers");
    end
end