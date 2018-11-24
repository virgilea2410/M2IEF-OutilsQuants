function [correlation_matrix, pairwise_pvalues] = Pairwise_Pvalues(dataset, rf_rate)

nb_fund = size(dataset, 2);

% Pairwise Correlation Matrix and Pairwise Two-Sample One-Tailes Test SRa <
% SRb
% between funds

correlation_matrix = corr(table2array(dataset));
pairwise_pvalues = zeros(nb_fund, nb_fund);
for i_row=1:nb_fund
    for i_col=1:nb_fund
        fund_row = table2array(dataset(:, i_row));
        col_row = table2array(dataset(:, i_col));
        [~, ~, ~, ~, test_results] = EstimateSharpes(fund_row, col_row, 0, 0, rf_rate);
        two_sample_LT_pvalue_biased = test_results(1,3);
        two_sample_LT_pvalue_unbiased = test_results(2,3);
        pairwise_pvalues(i_row, i_col) = two_sample_LT_pvalue_biased;
    end
end

correlation_matrix = array2table(correlation_matrix, ...
    'VariableNames', dataset.Properties.VariableNames, ...
    'RowNames', dataset.Properties.VariableNames); 
pairwise_pvalues = array2table(pairwise_pvalues, ...
    'VariableNames', dataset.Properties.VariableNames, ...
    'RowNames', dataset.Properties.VariableNames);

end