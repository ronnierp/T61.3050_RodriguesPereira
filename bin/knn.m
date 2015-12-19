function [knn_label] = knn(x_train, y_train, x_test, k, classify)
% k-Nearest Neighbor algorithm
% Usage:
%       x_train - Matrix containing observations in rows and variables in
%                 columns;
%       y_train - Column vector corresponding to the observation label
%       x_test  - Matrix of unlabeled data, containing observations in
%                 rows and variables in columns;
%       k - Hyperparameter denoting the number of neighbors
%       classify - Boolean. 1 - Classification; 0 - Regression

% Calculate the distance between observation and training data
k_nn        =       kdist(x_test, x_train);

% Perform classification
if classify
    knn_label      =       kclassification(y_train, k_nn);
   
% Perform regression
else
    knn_label       =       kregression(y_train, k_nn);
end

    function k_n = kdist(dtest, dtrain)
        no          =       length(dtest);     % Number of Observations
        nd          =       length(dtrain);    % Number of Training Samples
        
        k_n         =       zeros(no, k);       % Allocate matrix (n, k)
        
        for i = 1:no
            % Euclidean Distance
            d           =       sum((repmat(dtest(i,:), nd, 1) - dtrain).^2, 2);
            
            % Sort Ascending (Nearest neighbors are closer to x)
            [~, idx]    =       sort(d);
            
            k_n(i,:)    =       idx(1:k)';
        end
    end

    function klabel = kclassification(label, k_idx)
        
        klabel      =       zeros(length(k_idx), 1);
        
        [r, ~]      =       size(k_idx);
        
        for i = 1:r
            vote    =   sum(label(k_idx(i,:)));
            
            % Check the majority vote
            if vote >= k/2
                klabel(i)   =   1;      % White Wine
            else
                klabel(i)   =   0;      % Red Wine
            end
        end
        
    end

    function klabel = kregression(label, k_idx)
        klabel      =       zeros(length(k_idx), 1);
        
        [r, ~]      =       size(k_idx);
        
        for i = 1:r
            % Calculate the mean of the neighbors
            vote    =   mean(label(k_idx(i,:)));
            
            % Round to the nearest integer
            klabel(i) = round(vote);
            
        end
    end
end