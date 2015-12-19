function [label] = logisticreg(x_train, y_train, x_test)
% Logistic Regression Classification Algorithm
% Usage:
%       x_train - Matrix containing observations in rows and variables in
%                 columns;
%       y_train - Column vector corresponding to the observation label
%       x_test  - Matrix of unlabeled data, containing observations in
%                 rows and variables in columns;

% Include column vector to account for Wo in W
x_train         =       [ones(length(x_train), 1), x_train];
x_test          =       [ones(length(x_test), 1), x_test];

[ntrain, d]     =       size(x_train);
[ntest, ~]      =       size(x_test);

% Initial guess for W
w       =       zeros(d, 1);

% Iteration step size
alpha   =       0.0001;

% Initiate iterations
lp_diff =       200;	% Stopping criteria (log-likelihood diff)
lp      =       [0, 0];
while lp_diff > 1e-3
    % Prior Estimates
    p       =       1./(1 + exp(-x_train*w));

    % Log-likelihood
    lp(1)   =       sum(y_train.*log(p) + (ones(ntrain, 1) - y_train).*log(ones(ntrain, 1) - p));

    % Gradient
    gradL   =       repmat((y_train - p), 1, d).*x_train;
    gradL   =       -sum(gradL);

    % Iteration step
    w       =       w - alpha*gradL';

    % Stopping Criteria
    lp_diff =       abs(lp(1) - lp(2));
    lp(2)   =       lp(1);
end

% Recalculate the Posterior
posterior   =       1./(1 + exp(-x_test*w));

% Assign labels if Posterior(r = 1| x) > 0.5
label       =       zeros(ntest, 1);
label(posterior >= 0.5) = 1;

end