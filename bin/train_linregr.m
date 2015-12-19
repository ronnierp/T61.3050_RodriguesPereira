function linregr_err = train_linregr(data, label)
% Training routine for the Linear Regression Algorithm
% Usage:
%       data  - Matrix containing observations in rows and variables in
%               columns;
%       label - Column vector corresponding to the observation label

[obs, ~]    =       size(data);

% Generate vector with random integers in the range of [1, obs]
idx         =       randperm(obs);

% k-fold
n           =       10;
r           =       1:obs/n:obs;

% Error-score
fold_score  =   zeros(1, n);
    
for i = 1:n
    % Union of all k's
    x_train     =       data;
    y_train     =       label;

    % Remove the current k subset
    x_train(idx(r(i):r(i) + obs/n - 1), :) = [];
    y_train(idx(r(i):r(i) + obs/n - 1), :) = [];

    % Assign the current subset as testing sample
    x_test      =       data(idx(r(i):r(i) + obs/n - 1), :);
    y_test      =       label(idx(r(i):r(i) + obs/n - 1), :);

    % Predict the labels
    qual        =       linear_regression(x_train, y_train(:,1), x_test);
    
    % Calculate the Regression F-score
    fold_score(i) =   error_score(qual, y_test(:,1), 1);
end

% The error associated with the model
linregr_err       =       mean(fold_score);

end