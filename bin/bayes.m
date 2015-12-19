function [label] = bayes(x_train, y_train, x_test)
% Naive Bayes Classifier
% Usage:
%       x_train - Matrix containing observations in rows and variables in
%                 columns;
%       y_train - Column vector corresponding to the observation label
%       x_test  - Matrix of unlabeled data, containing observations in
%                 rows and variables in columns;

% Dimensionality
[ntrain, ~]     =       size(x_train);
[ntest, ~]      =       size(x_test);
label           =       zeros(ntest, 1);

% Priors
n       =       sum([y_train == 1, y_train == 0]);
pclass  =       n/ntrain;

% Covariance Matrix
S1      =       cov(x_train(y_train == 1, :), 1);
S0      =       cov(x_train(y_train == 0, :), 1);

% Discriminant
g1 = -0.5*diag(x_test/S1*x_test');
g1 = g1 + (log(pclass(1)) - 0.5*log(det(S1)))*ones(ntest,1);

g2 = -0.5*diag(x_test/S0*x_test');
g2 = g2 + (log(pclass(2)) - 0.5*log(det(S0)))*ones(ntest,1);

% Assign labels (Prob. White > Prob. Red)
label(g1 > g2) = 1;

end