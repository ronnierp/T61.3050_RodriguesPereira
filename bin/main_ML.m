function [] = main_ML()
% Machine Learning Course Project
% Classification of wine type and Regression of wine quality.
% Implemented methods:
%       -   Classification
%               k-Nearest Neighbors
%               Bayes Classifier
%               Logistic Regression
%       -   Regression
%               k-Nearest Neighbors
%               Linear Regression
%               Random Forest

%%
%-------------Train Algorithms and Evaluate Validation Error---------------
%---Load the Training Data
[data, label]   =       parse_data('../data/training_classification_regression_2015.csv', 1);

%---Evaluate parameters and F-Score (10-Fold Cross Validation)
%-----k-NN Classifier/Regressor
knn_err         =       train_knn(data, label, 0);

%-----Bayes Classifier/Regressor
bayes_err       =       train_bayes(data, label);

%-----Linear Regression
linregr_err     =      train_linregr(data, label);

%-----Logistic Regression Classifier
logreg_err      =       train_logreg(data, label);

%-----Random Forest (Regression)
forest_err      =       train_forest(data, label(:,1), 0);

%---PCA Analysis
% The data presents correlation across variables
%null_covariance =       any(any(cov(data, 1) == 0));

% Matrix eigendecomposition
w = PCA_analysis(data, 1);

pca_data = data*w;

%---Evaluate parameters and F-Score (10-Fold Cross Validation)
%-----k-NN Classifier/Regressor
knn_pca_err         =       train_knn(pca_data, label, 1);

%-----Linear Regression
linregr_pca_err     =       train_linregr(pca_data, label);

%-----Random Forest (Regression)
forest_pca_err      =       train_forest(pca_data, label(:,1), 1);

save('../results/validation_error_summary.mat')
%--------------------------------------------------------------------------

%%
%------------------------Evaluate the Test Set-----------------------------
%---Load Data and Assign Parameters
%---Parameters
knn_clas        =   4;
knn_regr        =   1;% or 4;
knn_pca_regr    =   1;% or 3;

ntrees          =   200;
ntrees_pca      =   1;% or 5;

%---Load Data
[tdata, ~]      =   parse_data('../data/challenge_public_test_classification_regression_2015.csv', 0);

%---Matrix eigendecomposition
w = PCA_analysis(data, 0);
pca_data = data*w;
pca_tdata = tdata*w;

%---Classification
%------k-NN
knn_type            =       knn(data, label(:,2), tdata, knn_clas, 1);
write_label(knn_type, 'classification_knn', 1)

%-----Bayes
bayes_type          =       bayes(data, label(:,2), tdata);
write_label(bayes_type, 'classification_bayes', 1)

%-----Logistic Regression
lg_type             =       logisticreg(data, label(:,2), tdata);
write_label(lg_type, 'classification_logreg', 1)

%---Regression
%-----k-NN
knn_qual            =       knn(data, label(:,1), tdata, knn_regr, 0);
write_label(knn_qual, 'regression_knn', 0)

knn_pca_qual        =       knn(pca_data, label(:,1), pca_tdata, ...
                            knn_pca_regr, 0);
write_label(knn_pca_qual, 'regression_knn_pca', 0)

%-----Linear Regression
linregr_qual        =       linear_regression(data, label(:,1), tdata);
write_label(linregr_qual, 'regression_linear', 0)

linregr_pca_qual    =       linear_regression(pca_data, label(:,1), ...
                                                pca_tdata);
write_label(linregr_pca_qual, 'regression_linear_pca', 0)

%-----Random Forest
qual_tree           =       TreeBagger(ntrees, data, label(:,1), ...
                                        'method', 'regression');
forest_qual         =       predict(qual_tree, tdata);
forest_qual         =       round(forest_qual);
write_label(forest_qual, 'regression_random_forest', 0)

qual_pca_tree       =       TreeBagger(ntrees_pca, pca_data, ...
                            label(:,1), 'method', 'regression');
forest_pca_qual     =       predict(qual_pca_tree, pca_tdata);
forest_pca_qual     =       round(forest_pca_qual);
write_label(forest_pca_qual, 'regression_random_forest_pca', 0)
%--------------------------------------------------------------------------
end