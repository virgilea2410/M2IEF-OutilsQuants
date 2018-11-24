function [fund_ranking] = Rank_Funds(dataset, rf_rate)

% Funds Rankings using Bias-Corrected sharpe ratio and One-Sample
% One-Tailes Test

nb_fund = size(dataset, 2);

fund_ranking = zeros(nb_fund, 10);
for i_fund=1:nb_fund
    curr_fund = table2array(dataset(:, i_fund));
    
    [descriptive_stats ...
        , sharpe_stats ...
        , one_sample_test_stats ...
        ,~, test_results] = EstimateSharpes(curr_fund, curr_fund, 0, 0, 0.001);
    
    est_sharpe_biased = sharpe_stats(2,1);
    est_sharpe_unbiased = sharpe_stats(3,1);
    LT_pval = test_results(2,1);
    RT_pval = test_results(4,1);
    est_mean = descriptive_stats(1,1);
    est_vol = descriptive_stats(2,1);
    est_skew = descriptive_stats(3,1);
    est_kurto = descriptive_stats(4,1);
    est_med = descriptive_stats(5,1);
    
    fund_ranking(i_fund, 1) = 0;
    fund_ranking(i_fund, 2) = est_sharpe_biased;
    fund_ranking(i_fund, 3) = est_sharpe_unbiased;
    fund_ranking(i_fund, 4) = LT_pval;
    fund_ranking(i_fund, 5) = est_mean;
    %fund_ranking(i_fund, 6) = table2array(rf_rate(1, 1));
    fund_ranking(i_fund, 6) = rf_rate;
    fund_ranking(i_fund, 7) = est_vol;
    fund_ranking(i_fund, 8) = est_med;
    fund_ranking(i_fund, 9) = est_skew;
    fund_ranking(i_fund, 10) = est_kurto;
end

fund_ranking = sortrows(fund_ranking, 2, 'descend');
fund_ranking(:,1) = 1:size(fund_ranking, 1);
fund_ranking = array2table(fund_ranking, ...
    'VariableNames', {'Rank', 'SharpeBiased', 'SharpeUnbiased', 'PValue_SharpeSup0', ...
                        'EstimatedMean', 'RiskFreeRate', 'EstimatedVolatility', ...
                        'EstimatedMedian', 'EstimatedSkew', 'EstimatedKurto'}, ...
    'RowNames', dataset.Properties.VariableNames);

end