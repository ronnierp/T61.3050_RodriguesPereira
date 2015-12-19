function [label] = linear_regression(x_train, y_train, x_test)
% Linear Regression
% Usage:
%       x_train - Matrix containing observations in rows and variables in
%                 columns;
%       y_train - Column vector corresponding to the observation label
%       x_test  - Matrix of unlabeled data, containing observations in
%                 rows and variables in columns;

% Dimensionality
[ntrain, ~] =       size(x_train);
[ntest, ~]  =       size(x_test);

% Include one vector in the independent variable
x_train     =       [ones(ntrain, 1), x_train];
x_test      =       [ones(ntest, 1), x_test];

% Optimum solution (Minimizes the error, Wo is in W)
W           =       (x_train'*x_train)\x_train'*y_train;

% Apply the regression on the test set
label       =       x_test*W;
label       =       round(label);

end