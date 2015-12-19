function [] = write_label(label, filename, class_flag)
% Write the labels according to the required format (Id, Type/Quality)
% Usage:
%       label - Column vector corresponding to the observation label
%       filename - Name of the output file without extension (filname.csv)
%       class_flag - Boolean: 1 - Classification (Type);
%                             0 - Regression (Quality)

txt_file = fopen(strcat('../results/', filename, '.csv'), 'w');

if class_flag
    % Classification Results
    fprintf(txt_file, 'id,type\n');

    for i = 1:1000
        if label(i)
            wine_type = 'White';
        else
            wine_type = 'Red';
        end

        if i == 1000
            fprintf(txt_file, '%d,%s', i, wine_type);
        else
            fprintf(txt_file, '%d,%s\n', i, wine_type);
        end
    end
else
    % Regression Results
    fprintf(txt_file, 'id,quality\n');

    for i = 1:1000
        if i == 1000
            fprintf(txt_file, '%d,%d', i, label(i));
        else
            fprintf(txt_file, '%d,%d\n', i, label(i));
        end
    end
end

fclose(txt_file);

end