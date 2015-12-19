function bayes_err = train_bayes(data, label)
% Training routine for the Naive Bayes Classifier
% Usage:
%       data  - Matrix containing observations in rows and variables in
%                 columns;
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
    type        =       bayes(x_train, y_train(:,2), x_test);
        
    % Calculate the Classification F-score
    fold_score(1, i) =   error_score(type, y_test(:,2), 1);
    
end

% The error associated with the model
bayes_err       =       mean(fold_score, 2);

end