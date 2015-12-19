function [data, label] = parse_data(filename, flag)
% Import the data and parse into matrix format.
% Usage:
%       filename - Path to the file to be imported
%       flag - Boolean. 1: training data, 0: test data.

[n, t, ~] = xlsread(filename);

if flag
    % Wine Quality
    label   =   n(:,end);
    
    % Retrieve wine type and convert it to Boolean
    % 1 - White; 0 - Red
    label   =   [label, strcmp(t(2:end, end), 'White')];
    
    % Standardize the data
    data    =   zscore(n(:,1:end-1));
else
    % Standardize the data
    data    =   zscore(n(:,2:end));
    
    label   =   [];
end

end
