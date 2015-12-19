function tree_err = train_forest(data, label, pca_flag)
% Training routine for the Random Forest regressor
% Usage:
%       data     - Matrix containing observations in rows and variables in
%                  columns;
%       label    - Column vector corresponding to the observation label
%       pca_flag - Lower dimensional dataset indicator;
%                  Boolean: 1 - True; 0 - False

[obs, ~]    =       size(data);

% Generate vector with random integers in the range of [1, obs]
idx         =       randperm(obs);

% k-fold
n           =       10;
r           =       1:obs/n:obs;

% Number of neighbors
if pca_flag
    tree_range  =       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50];
else
    tree_range  =       [10, 50, 100, 200, 300, 400, 500];
end
ntrees      =       length(tree_range);

% Error-score
tree_err    =       zeros(1, ntrees);

for k = 1:ntrees
    
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
        if pca_flag
            qual        =       knn(x_train, y_train(:,1), x_test, tree_range(k), 0);
        else
            tr_tree     =       TreeBagger(tree_range(k), x_train, y_train, 'method', 'regression');
            qual        =       predict(tr_tree, x_test);
            qual        =       round(qual);
            
        end
        
        % Calculate the Regression F-score
        fold_score(i) =   error_score(qual, y_test, 1);
        
    end
    
    tree_err(:, k)         =       mean(fold_score);
    
end

figure(1)
plot(tree_range, tree_err, 'o-')
xlabel('Number of Trees')
ylabel('F-score')
    
if pca_flag
    
    title('Random Forest (CV: 10-fold, PCA)')
    print('../results/forest_PCA_selection.png', '-dpng', '-r300')
else
    
    title('Random Forest (CV: 10-fold)')
    print('../results/forest_selection.png', '-dpng', '-r300')
end

close(intersect(findall(0,'type','figure'),1))

end