function k_err = train_knn(data, label, pca_flag)
% Training routine for k-Nearest Neighbors Algorithm
% Usage:
%       data     - Matrix containing observations in rows and variables in
%                  columns;
%       label    - Matrix corresponding to the observation labels
%                  (1st column = quality; 2nd column = type)
%       pca_flag - Lower dimensional dataset indicator;
%                  Boolean: 1 - True; 0 - False

[obs, ~]    =       size(data);

% Generate vector with random integers in the range of [1, obs]
idx         =       randperm(obs);

% k-fold
n           =       10;
r           =       1:obs/n:obs;

% Number of neighbors
k_range     =       linspace(1, 10, 10);
ks          =       length(k_range);

% Error-score
k_err           =       zeros(2, ks);

for k = 1:ks
    
    fold_score  =   zeros(2, n);
    
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
        if pca_flag
            qual        =       knn(x_train, y_train(:,1), x_test, k_range(k), 0);
        else
            type        =       knn(x_train, y_train(:,2), x_test, k_range(k), 1);
            qual        =       knn(x_train, y_train(:,1), x_test, k_range(k), 0);
            
            % Calculate the Classification F-score
            fold_score(1, i) =   error_score(type, y_test(:,2), 1);
        end
        
        % Calculate the Regression F-score
        fold_score(2, i) =   error_score(qual, y_test(:,1), 1);
        
    end
    
    k_err(:, k)         =       mean(fold_score, 2);
    
end

if pca_flag
    
    figure(1)
    plot(k_range, k_err(2,:), 'o-')
    title('k-NN Regression (CV: 10-fold, PCA)')
    xlabel('Number of k-NN')
    ylabel('F-score')
    
    print('../results/k_PCA_selection.png', '-dpng', '-r300')
else
    
    figure(1)
    subplot(2,1,1)
    plot(k_range, k_err(1,:), 'o-')
    title('k-NN Classification (CV: 10-fold)')
    xlabel('Number of k-NN')
    ylabel('F-Score')

    subplot(2,1,2)
    plot(k_range, k_err(2,:), 'o-')
    title('k-NN Regression (CV: 10-fold)')
    xlabel('Number of k-NN')
    ylabel('F-Score')
    
    print('../results/k_selection.png', '-dpng', '-r300')
end

close(intersect(findall(0,'type','figure'),1))

end