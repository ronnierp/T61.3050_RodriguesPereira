function score = error_score(ypred, ytrue, score_type)

% Evaluate the F-score or the Sum of Squred Errors
switch score_type
    case 1
        idx     =       unique(ytrue);
        [n, ~]  =       size(idx);
        
        TP = 0;     FP = 0;     FN = 0;
        for i = 1:n
            % True Positives
            TP      =       TP + sum(ypred(ytrue == idx(i)) == idx(i));

            % False Positives
            FP      =       FP + sum(ypred(ytrue ~= idx(i)) == idx(i));

            % False Negatives
            FN      =       FN + sum(ypred(ytrue == idx(i)) ~= idx(i));
        end

        precision   =   TP/(TP + FP);

        recall      =   TP/(TP + FN);

        % Calculate F-score (avoid zero division)
        if precision + recall > 0
            score       =   2*precision*recall/(precision + recall);
        else
            score       =   0;
        end
        
    case 2
        % True Positives
        TP      =       sum(ypred(ytrue == 1) == 1);
        
        % True Negatives
        TN      =       sum(ypred(ytrue == 0) == 0);
        
        % Error = 1 - Accuracy
        score = 1 - (TP + TN)/length(ytrue);
        
    case 3
        % Calculate the Mean Squared Errors
        score       =       mean((ypred - ytrue).^2);
end

end