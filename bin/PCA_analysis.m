function [w] = PCA_analysis(data, plot_fig)
% Principal Component Analysis
% Usage:
%       data - Matrix containing observations in rows and variables in
%              columns;
%       plot_fig - Boolean. 1: save plot to results folder, 0: otherwise.

% Sample size and Original dimension
[~, d]      =       size(data);
neigenv     =       linspace(1, d, d);

% Evaluate covariance matrix with centered data
S           =       cov(data, 1);

% Calculate Eigenvectors and Eigenvalues
[evc, evl]  =       eig(S);

% Retrieve eigenvalues from the diagonal
evl         =       diag(evl);

% Order the eigenvalues in descending order
[evl_sorted, evl_idx] = sort(evl, 'descend');

% Calculate the Proportion of Variance expected
PoV         =       cumsum(evl_sorted)./sum(evl_sorted);

% Select eigenvectors that render PoV >= 0.9
ncomponents =       neigenv(PoV >= 0.9);

% The selected Principal Components
principal_components = evl_idx(1:ncomponents(1))

% The new space is formed by the selected eigenvectors
w           =       evc(:, evl_idx(1:ncomponents(1)));

if plot_fig
    figure(1)
    subplot(2,1,1)
    plot(linspace(1, d, d), PoV, '-o')
    xlabel('Number of Eigenvectors')
    ylabel('PoV')
    title('Proportion of Variance')

    subplot(2,1,2)
    plot(linspace(1, d, d), evl_sorted, '-o')
    xlabel('Number of Eigenvectors')
    ylabel('$$\hat{\lambda}$$', 'Interpreter', 'Latex')
    title('Scree plot')

    print('../results/PCA_components.png', '-dpng', '-r300')

    close(intersect(findall(0,'type','figure'),1))
end

end