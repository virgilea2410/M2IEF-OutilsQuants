%% II) The statistial distribution of the Sharpe Ratio

mean_level = 1;
tested_sharpe = 1;
std_level = 1;
corr_level = 0.5;
rf_rate = 0;
return_simul = 50;
sharpes_simul = 5000;

normal_sharpes = zeros(sharpes_simul, 1);
laplacian_sharpes = zeros(sharpes_simul, 1);
real_world_sharpes = zeros(sharpes_simul, 1);

normal_sharpes_diff = zeros(sharpes_simul, 1);
laplacian_sharpes_diff = zeros(sharpes_simul, 1);
real_world_sharpes_diff = zeros(sharpes_simul, 1);

% Simulating 3 Distributions of Sharpe Ratios and Sharpes Ratios Difference : 
%   - With Normal underlying returns
%   - With Laplacian underlying returns
%   - With APD (with "real world" parameters) underlying returns

for i_simul=1:sharpes_simul
    normal_returns = generate_normal_return(mean_level, std_level, return_simul, corr_level);
    [~, normal_sharpe_stats, normal_one_sample, normal_two_sample_test_stats] = ...
        EstimateSharpes(normal_returns(:, 1),normal_returns(:, 2), tested_sharpe, tested_sharpe, rf_rate);
    normal_sharpes(i_simul, 1) = normal_sharpe_stats(2, 1);
    %normal_sharpes(i_simul, 1) = normal_one_sample(1, 1);
    normal_sharpes_diff(i_simul, 1) = normal_two_sample_test_stats(1, 1);
    %normal_sharpes_diff(i_simul, 1) = normal_two_sample_test_stats(6, 1);
    
    laplacian_returns = generate_laplacian_return(mean_level, std_level, return_simul, corr_level);
    [~, laplacian_sharpe_stats, laplacian_one_sample, laplacian_two_sample_test_stats] = ...
        EstimateSharpes(laplacian_returns(:, 1),laplacian_returns(:, 2), tested_sharpe, tested_sharpe, rf_rate);
    laplacian_sharpes(i_simul, 1) = laplacian_sharpe_stats(2, 1);
    %laplacian_sharpes(i_simul, 1) = laplacian_one_sample(1, 1);
    laplacian_sharpes_diff(i_simul, 1) = laplacian_two_sample_test_stats(1, 1);
    %laplacian_sharpes_diff(i_simul, 1) = laplacian_two_sample_test_stats(6, 1);
    
    %real_world_returns = generate_APD_return(0.7, 1.35, return_simul, 0.5, mean_level - 0.05, true);
    real_world_returns = generate_APD_return(0.7, 1.35, return_simul, corr_level, mean_level, mean_level, true);
    [~, real_world_sharpe_stats, real_world_one_sample, real_world_two_sample_test_stats] = ...
        EstimateSharpes(real_world_returns(:, 1),real_world_returns(:, 2), tested_sharpe, tested_sharpe, rf_rate);
    real_world_sharpes(i_simul, 1) = real_world_sharpe_stats(2, 1);
    %real_world_sharpes(i_simul, 1) = real_world_one_sample(1, 1);
    real_world_sharpes_diff(i_simul, 1) = real_world_two_sample_test_stats(1, 1);
    %real_world_sharpes_diff(i_simul, 1) = real_world_two_sample_test_stats(6, 1);
end

% Normalizing the sharpe ratios
normal_sharpes = (normal_sharpes-tested_sharpe) * sqrt(return_simul);% * normal_sharpe_stats(4, 1) %* sqrt(return_simul);
laplacian_sharpes = (laplacian_sharpes-tested_sharpe) * sqrt(return_simul);% * laplacian_sharpe_stats(4, 1) %* sqrt(return_simul);
real_world_sharpes = (real_world_sharpes-tested_sharpe) * sqrt(return_simul);% * real_world_sharpe_stats(4, 1) %* sqrt(return_simul);
% % 
normal_sharpes_diff = (normal_sharpes_diff-(tested_sharpe-tested_sharpe)) * sqrt(return_simul);% * normal_two_sample_test_stats(3, 1);%* sqrt(return_simul);
laplacian_sharpes_diff = (laplacian_sharpes_diff-(tested_sharpe-tested_sharpe)) * sqrt(return_simul);% * laplacian_two_sample_test_stats(3, 1);%* sqrt(return_simul);
real_world_sharpes_diff = (real_world_sharpes_diff-(tested_sharpe-tested_sharpe)) * sqrt(return_simul);% * real_world_two_sample_test_stats(3, 1);%* sqrt(return_simul);

% Plotting Sharpe Ratios distribution for each type of returns
figure()
hold on;
[f, xi] = ksdensity(normal_sharpes);
plot(xi, f, 'r-');
[f, xi] = ksdensity(laplacian_sharpes);
plot(xi+2.5, f, 'g-');
[f, xi] = ksdensity(real_world_sharpes);
plot(xi-1.5, f, 'b-');
%plot(xi, f, 'b-');
legend({'Normal Returns', 'Laplacian Returns', 'APD Returns (alpha = 0.7, lambda = 1.35)'})
xlabel('Sharpe Ratio Value');
ylabel('Probability');
hold off;

%% III) A Two Sample statistic for comparing Sharpe Ratios

% Plotting Sharpe Ratios Differences distribution for each type of returns
figure()
hold on;
[f, xi] = ksdensity(normal_sharpes_diff);
plot(xi, f, 'r-');
[f, xi] = ksdensity(laplacian_sharpes_diff);
plot(xi, f, 'g-');
[f, xi] = ksdensity(real_world_sharpes_diff);
plot(xi, f, 'b-');
legend({'Normal Returns', 'Laplacian Returns', 'APD Returns (alpha = 0.7, lambda = 1.35)'})
xlabel('Sharpe Ratio Difference Value');
ylabel('Probability');
hold off;

%% IV) Simulation Study 

% Plotting "Real World" Returns Shape

one_simulated_return = generate_APD_return(0.7, 1.35, 10000, 0.6, 0, 0, false);
figure();
ksdensity(one_simulated_return(:, 1));
title("Standardized Asymmetric Power Distribution (APD) of Returns With 'Real World' Parameters");
ylabel('Standardize Return');
xlabel('Probability');
legend({'"Real World" APD Returns (Alpha = 0.7, Lambda = 1.35)'});

% Simulation Study 

rf_rate = 0.001;

ci_col = 1;
simulated_returns = [];
%range_alpha = [0.1, 0.3, 0.5, 0.7, 0.9];
range_alpha = [0.7];
%range_lambda = [1, 1.25, 1.5, 1.75, 2];
range_lambda = [1.35];
range_sample_size = [15, 30, 50, 100, 300];
%range_correl = [0, 0.25, 0.5, 0.75];
range_correl = [0.5];
range_sharpe_1 = [0, 0.2, 0.3, 0.4, 1, 3];
range_sharpe_2 = [0, 0.1, 0.2, 0.5; ...
                    0.2, 0.4, -1, -1; ...
                    0.3, 0.6, -1, -1; ...
                    0.4, 0.8, -1, -1; ...
                    1, 1.5, -1, -1; ...
                    3, 3.5, -1, -1];
max_simul = 1000;

nb_alphas = size(range_alpha, 2);
nb_lambdas = size(range_lambda, 2);
nb_sample_sizes = size(range_sample_size, 2);
nb_correl = size(range_correl, 2);
nb_sharpe_1 = size(range_sharpe_1, 2);

%AllResults = zeros(5000, 1);
AllScenarResults_H0 = cell(nb_alphas*nb_lambdas*nb_sample_sizes*nb_correl*nb_sharpe_1, 1);
AllScenarResults_H1 = cell(nb_alphas*nb_lambdas*nb_sample_sizes*nb_correl*nb_sharpe_1, 1);
AllSimulModels_H0 = cell(max_simul, 1);
AllSimulModels_H1 = cell(max_simul, 1);
OneSimulModel_H0 = {'Mean', []; ...
                'Vol', []; ...
                'Skew', []; ...
                'Kurto', []; ...
                'SharpeBias', []; ...
                'BiasedSharpe', []; ...
                'UnbiasedSharpe', []; ...
                'StdBiasedSharpe', []; ...
                'StdUnbiasedSharpe', []; ...
                'OneSampleStatBiased', []; ...
                'OneSampleStatUnbiased', []; ...
                'LT_one_sample_PvalBiased', []; ...
                'LT_one_sample_PvalUnbiased', []; ...
                'RT_one_sample_PvalBiased', []; ...
                'RT_one_sample_PvalUnbiased', []; ...
                'TwoT_one_sample_PvalBiased', []; ...
                'TwoT_one_sample_PvalUnbiased', []; ...
                'OneT_one_sample_Confidence_Boundary_Biased', []; ...
                'OneT_one_sample_Confidence_Boundary_Unbiased', []; ...
                'TwoT_one_sample_Confidence_Boundary_Biased', []; ...
                'TwoT_one_sample_Confidence_Boundary_Unbiased', []; ...
                'TwoSampleStatBiased', []; ...
                'TwoSampleStatUnbiased', []; ...
                'LT_two_sample_PvalBiased', []; ...
                'LT_two_sample_PvalUnbiased', []; ...
                'RT_two_sample_PvalBiased', []; ...
                'RT_two_sample_PvalUnbiased', []; ...
                'TwoT_two_sample_PvalBiased', []; ...
                'TwoT_two_sample_PvalUnbiased', []; ...
                'TwoT_two_sample_Confidence_Boundary_Biased', []; ...
                'TwoT_two_sample_Confidence_Boundary_Unbiased', []};

OneSimulModel_H1 = {'Mean', []; ...
                'Vol', []; ...
                'Skew', []; ...
                'Kurto', []; ...
                'SharpeBias', []; ...
                'BiasedSharpe', []; ...
                'UnbiasedSharpe', []; ...
                'StdBiasedSharpe', []; ...
                'StdUnbiasedSharpe', []; ...
                'OneSampleStatBiased', []; ...
                'OneSampleStatUnbiased', []; ...
                'LT_one_sample_PvalBiased', []; ...
                'LT_one_sample_PvalUnbiased', []; ...
                'RT_one_sample_PvalBiased', []; ...
                'RT_one_sample_PvalUnbiased', []; ...
                'TwoT_one_sample_PvalBiased', []; ...
                'TwoT_one_sample_PvalUnbiased', []; ...
                'OneT_one_sample_Confidence_Boundary_Biased', []; ...
                'OneT_one_sample_Confidence_Boundary_Unbiased', []; ...
                'TwoT_one_sample_Confidence_Boundary_Biased', []; ...
                'TwoT_one_sample_Confidence_Boundary_Unbiased', []; ...
                'TwoSampleStatBiased', []; ...
                'TwoSampleStatUnbiased', []; ...
                'LT_two_sample_PvalBiased', []; ...
                'LT_two_sample_PvalUnbiased', []; ...
                'RT_two_sample_PvalBiased', []; ...
                'RT_two_sample_PvalUnbiased', []; ...
                'TwoT_two_sample_PvalBiased', []; ...
                'TwoT_two_sample_PvalUnbiased', []; ...
                'TwoT_two_sample_Confidence_Boundary_Biased', []; ...
                'TwoT_two_sample_Confidence_Boundary_Unbiased', []};
            
OneScenarResult_H0 = {'Alpha', []; ...
                    'Lambda', []; ...
                    'Correl', []; ...
                    'c_Sharpe_1', []; ...
                    'c_Sharpe_2', []; ...
                    'SampleSize', []; ...
                    'AllSimulModels', []; ...
                    'LT_one_sample_RejectionRateBiased', []; ...
                    'LT_one_sample_RejectionRateUnbiased', []; ...
                    'RT_one_sample_RejectionRateBiased', []; ...
                    'RT_one_sample_RejectionRateUnbiased', []; ...
                    'TwoT_one_sample_RejectionRateBiased', []; ...
                    'TwoT_one_sample_RejectionRateUnbiased', []; ...
                    'LT_two_sample_RejectionRateBiased', []; ...
                    'LT_two_sample_RejectionRateUnbiased', []; ...
                    'RT_two_sample_RejectionRateBiased', []; ...
                    'RT_two_sample_RejectionRateUnbiased', []; ...
                    'TwoT_two_sample_RejectionRateBiased', []; ...
                    'TwoT_two_sample_RejectionRateUnbiased', []};

OneScenarResult_H1 = {'Alpha', []; ...
                    'Lambda', []; ...
                    'Correl', []; ...
                    'c_Sharpe_1', []; ...
                    'c_Sharpe_2', []; ...
                    'SampleSize', []; ...
                    'AllSimulModels', []; ...
                    'LT_one_sample_RejectionRateBiased', []; ...
                    'LT_one_sample_RejectionRateUnbiased', []; ...
                    'RT_one_sample_RejectionRateBiased', []; ...
                    'RT_one_sample_RejectionRateUnbiased', []; ...
                    'TwoT_one_sample_RejectionRateBiased', []; ...
                    'TwoT_one_sample_RejectionRateUnbiased', []; ...
                    'LT_two_sample_RejectionRateBiased', []; ...
                    'LT_two_sample_RejectionRateUnbiased', []; ...
                    'RT_two_sample_RejectionRateBiased', []; ...
                    'RT_two_sample_RejectionRateUnbiased', []; ...
                    'TwoT_two_sample_RejectionRateBiased', []; ...
                    'TwoT_two_sample_RejectionRateUnbiased', []};
                
% Iterating over all Scenarios Parameters : Alpha, Lambda, Correlation,
% Sample_Sizes and Sharpe A & B level for Parametrics Tests
i_scenar = 1;
for i_alpha = range_alpha
    for i_lambda = range_lambda
        for i_correl = range_correl
            for i_sharpe_1 = range_sharpe_1
                for i_sharpe_2 = range_sharpe_2(find(range_sharpe_1 == i_sharpe_1), :)
                    if i_sharpe_2 >= 0
                        for i_sample_size = range_sample_size
                            OneScenarResult_H0(1,2) = {i_alpha};
                            OneScenarResult_H0(2,2) = {i_lambda};
                            OneScenarResult_H0(3,2) = {i_correl};
                            OneScenarResult_H0(4,2) = {i_sharpe_1};
                            OneScenarResult_H0(5,2) = {i_sharpe_2}; 
                            OneScenarResult_H0(6,2) = {i_sample_size};
                            
                            OneScenarResult_H1(1,2) = {i_alpha};
                            OneScenarResult_H1(2,2) = {i_lambda};
                            OneScenarResult_H1(3,2) = {i_correl};
                            OneScenarResult_H1(4,2) = {i_sharpe_1};
                            OneScenarResult_H1(5,2) = {i_sharpe_2}; 
                            OneScenarResult_H1(6,2) = {i_sample_size};
                            
                            test_one_sample_LT_decision_biased_H0 = zeros(max_simul, 1);
                            test_one_sample_LT_decision_unbiased_H0 = zeros(max_simul, 1);
                            test_one_sample_RT_decision_biased_H0 = zeros(max_simul, 1);
                            test_one_sample_RT_decision_unbiased_H0 = zeros(max_simul, 1);
                            test_one_sample_2T_decision_biased_H0 = zeros(max_simul, 1);
                            test_one_sample_2T_decision_unbiased_H0 = zeros(max_simul, 1);
                            
                            test_one_sample_LT_decision_biased_H1 = zeros(max_simul, 1);
                            test_one_sample_LT_decision_unbiased_H1 = zeros(max_simul, 1);
                            test_one_sample_RT_decision_biased_H1 = zeros(max_simul, 1);
                            test_one_sample_RT_decision_unbiased_H1 = zeros(max_simul, 1);
                            test_one_sample_2T_decision_biased_H1 = zeros(max_simul, 1);
                            test_one_sample_2T_decision_unbiased_H1 = zeros(max_simul, 1);
                            
                            test_two_sample_LT_decision_biased_H0 = zeros(max_simul, 1);
                            test_two_sample_LT_decision_unbiased_H0 = zeros(max_simul, 1);
                            test_two_sample_RT_decision_biased_H0 = zeros(max_simul, 1);
                            test_two_sample_RT_decision_unbiased_H0 = zeros(max_simul, 1);
                            test_two_sample_2T_decision_biased_H0 = zeros(max_simul, 1);
                            test_two_sample_2T_decision_unbiased_H0 = zeros(max_simul, 1);
                            
                            test_two_sample_LT_decision_biased_H1 = zeros(max_simul, 1);
                            test_two_sample_LT_decision_unbiased_H1 = zeros(max_simul, 1);
                            test_two_sample_RT_decision_biased_H1 = zeros(max_simul, 1);
                            test_two_sample_RT_decision_unbiased_H1 = zeros(max_simul, 1);
                            test_two_sample_2T_decision_biased_H1 = zeros(max_simul, 1);
                            test_two_sample_2T_decision_unbiased_H1 = zeros(max_simul, 1);
                            
                            for i_simul=1:max_simul
                                epsilon_1 = 0.5 * i_sharpe_1 * (i_sharpe_1 > 0) + 0.5 * (i_sharpe_1 == 0);
                                epsilon_2 = 0.5 * i_sharpe_2 * (i_sharpe_2 > 0) + 0.5 * (i_sharpe_2 == 0);
                                %epsilon_1 = 0.3 * i_sharpe_1 * (i_sharpe_1 > 0) + 0.3 * (i_sharpe_1 == 0);
                                %epsilon_2 = 0.3 * i_sharpe_2 * (i_sharpe_2 > 0) + 0.3 * (i_sharpe_2 == 0);
                                one_simulated_return_H0 = generate_APD_return(i_alpha, i_lambda, i_sample_size, i_correl, i_sharpe_1 - epsilon_1, i_sharpe_2 - epsilon_2, true);
                                one_simulated_return_H1 = generate_APD_return(i_alpha, i_lambda, i_sample_size, i_correl, i_sharpe_1 + epsilon_1, i_sharpe_2 + epsilon_2, true); %H1

                                % 2 Returns Vectors
                                returns_a_H0 = one_simulated_return_H0(:, 1);
                                returns_b_H0 = one_simulated_return_H0(:, 2);
                                returns_a_H1 = one_simulated_return_H1(:, 1); %H1
                                returns_b_H1 = one_simulated_return_H1(:, 2); %H1
                                
                                % Test 02/06 : Scaling des données pour
                                % qu'elle ait la moyenne testée
                                %returns_a = (returns_a - mean(returns_a)) / var(returns_a);
                                %returns_b = (returns_b - mean(returns_b)) / var(returns_b);
                                
                                % Sharpe Calucations and Parametrics Tests
                                [descriptive_stats_H0, ...
                                    sharpe_stats_H0, ...
                                    one_sample_test_stats_H0, ...
                                    two_sample_test_stats_H0, ...
                                    test_results_H0] ...
                                = EstimateSharpes(returns_a_H0, returns_b_H0, i_sharpe_1, i_sharpe_2, rf_rate);
                                [descriptive_stats_H1, ...
                                    sharpe_stats_H1, ...
                                    one_sample_test_stats_H1, ...
                                    two_sample_test_stats_H1, ...
                                    test_results_H1] ...
                                = EstimateSharpes(returns_a_H1, returns_b_H1, i_sharpe_1, i_sharpe_2, rf_rate); %H1

                                % Getting Results of Sharpe Tests
                                est_mean_a_H0 = descriptive_stats_H0(1,1);
                                est_vol_a_H0 = descriptive_stats_H0(2,1);
                                est_skew_a_H0 = descriptive_stats_H0(3,1);
                                est_kurto_a_H0 = descriptive_stats_H0(4,1);
                                est_med_a_H0 = descriptive_stats_H0(5,1);
                                
                                est_mean_a_H1 = descriptive_stats_H1(1,1); %H1
                                est_vol_a_H1 = descriptive_stats_H1(2,1); %H1
                                est_skew_a_H1 = descriptive_stats_H1(3,1); %H1
                                est_kurto_a_H1 = descriptive_stats_H1(4,1); %H1
                                est_med_a_H1 = descriptive_stats_H1(5,1); %H1
                                
                                confidence_boundary_one_sample_2T_biased_a_H0 = one_sample_test_stats_H0(3, 1);
                                confidence_boundary_one_sample_2T_unbiased_a_H0 = one_sample_test_stats_H0(4, 1);
                                confidence_boundary_one_sample_1T_biased_a_H0 = one_sample_test_stats_H0(5, 1);
                                confidence_boundary_one_sample_1T_unbiased_a_H0 = one_sample_test_stats_H0(6, 1);

                                confidence_boundary_one_sample_2T_biased_a_H1 = one_sample_test_stats_H1(3, 1); %H1
                                confidence_boundary_one_sample_2T_unbiased_a_H1 = one_sample_test_stats_H1(4, 1); %H1
                                confidence_boundary_one_sample_1T_biased_a_H1 = one_sample_test_stats_H1(5, 1); %H1
                                confidence_boundary_one_sample_1T_unbiased_a_H1 = one_sample_test_stats_H1(6, 1); %H1

                                confidence_boundary_two_sample_2T_biased_a_H0 = two_sample_test_stats_H0(7, 1);
                                confidence_boundary_two_sample_2T_unbiased_a_H0 = two_sample_test_stats_H0(8, 1);

                                confidence_boundary_two_sample_2T_biased_a_H1 = two_sample_test_stats_H1(7, 1); %H1
                                confidence_boundary_two_sample_2T_unbiased_a_H1 = two_sample_test_stats_H1(8, 1); %H1 %H1
                                
                                est_sharpe_bias_a_H0 = sharpe_stats_H0(1, 1);
                                est_sharpe_biased_a_H0 = sharpe_stats_H0(2, 1);
                                est_sharpe_unbiased_a_H0 = sharpe_stats_H0(3, 1);
                                std_est_sharpe_biased_a_H0 = sharpe_stats_H0(4, 1);
                                std_est_sharpe_unbiased_a_H0 = sharpe_stats_H0(5, 1);

                                est_sharpe_bias_a_H1 = sharpe_stats_H1(1, 1); %H1
                                est_sharpe_biased_a_H1 = sharpe_stats_H1(2, 1); %H1
                                est_sharpe_unbiased_a_H1 = sharpe_stats_H1(3, 1); %H1
                                std_est_sharpe_biased_a_H1 = sharpe_stats_H1(4, 1); %H1
                                std_est_sharpe_unbiased_a_H1 = sharpe_stats_H1(5, 1); %H1

                                one_sample_stat_biased_a_H0 = one_sample_test_stats_H0(1,1);
                                one_sample_stat_unbiased_a_H0 = one_sample_test_stats_H0(2,1);

                                one_sample_stat_biased_a_H1 = one_sample_test_stats_H1(1,1); %H1
                                one_sample_stat_unbiased_a_H1 = one_sample_test_stats_H1(2,1); %H1
                                
                                two_sample_stat_biased_a_H0 = two_sample_test_stats_H0(5,1);
                                two_sample_stat_unbiased_a_H0 = two_sample_test_stats_H0(6,1);

                                two_sample_stat_biased_a_H1 = two_sample_test_stats_H1(5,1); %H1
                                two_sample_stat_unbiased_a_H1 = two_sample_test_stats_H1(6,1); %H1

                                one_sample_LT_pvalue_biased_a_H0 = test_results_H0(1,1);
                                one_sample_LT_pvalue_unbiased_a_H0 = test_results_H0(2,1);
                                one_sample_RT_pvalue_biased_a_H0 = test_results_H0(3,1);
                                one_sample_RT_pvalue_unbiased_a_H0 = test_results_H0(4,1);
                                one_sample_2T_pvalue_biased_a_H0 = test_results_H0(5,1);
                                one_sample_2T_pvalue_unbiased_a_H0 = test_results_H0(6,1);
                                
                                one_sample_LT_pvalue_biased_a_H1 = test_results_H1(1,1); %H1
                                one_sample_LT_pvalue_unbiased_a_H1 = test_results_H1(2,1); %H1
                                one_sample_RT_pvalue_biased_a_H1 = test_results_H1(3,1); %H1
                                one_sample_RT_pvalue_unbiased_a_H1 = test_results_H1(4,1); %H1
                                one_sample_2T_pvalue_biased_a_H1 = test_results_H1(5,1); %H1
                                one_sample_2T_pvalue_unbiased_a_H1 = test_results_H1(6,1); %H1
                                
                                two_sample_LT_pvalue_biased_a_H0 = test_results_H0(1,3);
                                two_sample_LT_pvalue_unbiased_a_H0 = test_results_H0(2,3);
                                two_sample_RT_pvalue_biased_a_H0 = test_results_H0(3,3);
                                two_sample_RT_pvalue_unbiased_a_H0 = test_results_H0(4,3);
                                two_sample_2T_pvalue_biased_a_H0 = test_results_H0(5,3);
                                two_sample_2T_pvalue_unbiased_a_H0 = test_results_H0(6,3);
                                
                                two_sample_LT_pvalue_biased_a_H1 = test_results_H1(1,3); %H1
                                two_sample_LT_pvalue_unbiased_a_H1 = test_results_H1(2,3); %H1
                                two_sample_RT_pvalue_biased_a_H1 = test_results_H1(3,3); %H1
                                two_sample_RT_pvalue_unbiased_a_H1 = test_results_H1(4,3); %H1
                                two_sample_2T_pvalue_biased_a_H1 = test_results_H1(5,3); %H1
                                two_sample_2T_pvalue_unbiased_a_H1 = test_results_H1(6,3); %H1
                                
                                % Two-Sided Test Results (H0 : SR ~= i_sharpe_1)
                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                
                                % One Sample Stats
                                
                                if one_sample_2T_pvalue_biased_a_H0 < 0.05
                                    %Rejet de H0
                                   test_one_sample_2T_decision_biased_H0(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_one_sample_2T_decision_biased_H0(i_simul,1) = 0;    
                                end
                                
                                if one_sample_2T_pvalue_biased_a_H1 < 0.05
                                    %Rejet de H0
                                   test_one_sample_2T_decision_biased_H1(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_one_sample_2T_decision_biased_H1(i_simul,1) = 0;    
                                end
                                    
                                if one_sample_2T_pvalue_unbiased_a_H0 < 0.05
                                    %Rejet de H0
                                   test_one_sample_2T_decision_unbiased_H0(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_one_sample_2T_decision_unbiased_H0(i_simul,1) = 0;    
                                end
                                
                                if one_sample_2T_pvalue_unbiased_a_H1 < 0.05
                                    %Rejet de H0
                                    test_one_sample_2T_decision_unbiased_H1(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_one_sample_2T_decision_unbiased_H1(i_simul,1) = 0;    
                                end
                                
                                % Two Sample Stats
                                
                                    %Rejet de H0
                                if two_sample_2T_pvalue_biased_a_H0 < 0.05
                                   test_two_sample_2T_decision_biased_H0(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_two_sample_2T_decision_biased_H0(i_simul,1) = 0;    
                                end
                                
                                if two_sample_2T_pvalue_biased_a_H1 < 0.05
                                    %Rejet de H0
                                   test_two_sample_2T_decision_biased_H1(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_two_sample_2T_decision_biased_H1(i_simul,1) = 0;    
                                end
                                    
                                if two_sample_2T_pvalue_unbiased_a_H0 < 0.05
                                    %Rejet de H0
                                   test_two_sample_2T_decision_unbiased_H0(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_two_sample_2T_decision_unbiased_H0(i_simul,1) = 0;    
                                end
                                
                                if two_sample_2T_pvalue_unbiased_a_H1 < 0.05
                                    %Rejet de H0
                                    test_two_sample_2T_decision_unbiased_H1(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_two_sample_2T_decision_unbiased_H1(i_simul,1) = 0;    
                                end
                                                                  
                                % Left-Sided Test Results (H0 : SR > i_sharpe_1)
                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                
                                % One Sample Stats
                                
                                if one_sample_LT_pvalue_biased_a_H0 < 0.05
                                    %Rejet de H0
                                    test_one_sample_LT_decision_biased_H0(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_one_sample_LT_decision_biased_H0(i_simul,1) = 0;    
                                end

                                if one_sample_LT_pvalue_unbiased_a_H0 < 0.05
                                    %Rejet de H0
                                    test_one_sample_LT_decision_unbiased_H0(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_one_sample_LT_decision_unbiased_H0(i_simul,1) = 0;
                                end
                         
                                if one_sample_LT_pvalue_biased_a_H1 < 0.05
                                   %Rejet de H0
                                    test_one_sample_LT_decision_biased_H1(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_one_sample_LT_decision_biased_H1(i_simul,1) = 0;
                                end

                                if one_sample_LT_pvalue_unbiased_a_H1 < 0.05
                                   %Rejet de H0
                                    test_one_sample_LT_decision_unbiased_H1(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_one_sample_LT_decision_unbiased_H1(i_simul,1) = 0;
                                end
                                
                                % Two Sample Stats
                                
                                if two_sample_LT_pvalue_biased_a_H0 < 0.05
                                    %Rejet de H0
                                    test_two_sample_LT_decision_biased_H0(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_two_sample_LT_decision_biased_H0(i_simul,1) = 0;    
                                end

                                if two_sample_LT_pvalue_unbiased_a_H0 < 0.05
                                    %Rejet de H0
                                    test_two_sample_LT_decision_unbiased_H0(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_two_sample_LT_decision_unbiased_H0(i_simul,1) = 0;
                                end
                         
                                if two_sample_LT_pvalue_biased_a_H1 < 0.05
                                   %Rejet de H0
                                    test_two_sample_LT_decision_biased_H1(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_two_sample_LT_decision_biased_H1(i_simul,1) = 0;
                                end

                                if two_sample_LT_pvalue_unbiased_a_H1 < 0.05
                                   %Rejet de H0
                                    test_two_sample_LT_decision_unbiased_H1(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_two_sample_LT_decision_unbiased_H1(i_simul,1) = 0;
                                end
                                                                                         
                                % Right-Sided Test Results (H0 : SR < i_sharpe_1)
                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                
                                % One Sample Stats
                                
                                if one_sample_RT_pvalue_biased_a_H0 < 0.05
                                    %Rejet de H0
                                    test_one_sample_RT_decision_biased_H0(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_one_sample_RT_decision_biased_H0(i_simul,1) = 0;
                                end

                                if one_sample_RT_pvalue_unbiased_a_H0 < 0.05
                                    %Rejet de H0
                                    test_one_sample_RT_decision_unbiased_H0(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_one_sample_RT_decision_unbiased_H0(i_simul,1) = 0;
                                end
                     
                                if one_sample_RT_pvalue_biased_a_H1 < 0.05
                                    %Rejet de H0
                                    test_one_sample_RT_decision_biased_H1(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_one_sample_RT_decision_biased_H1(i_simul,1) = 0;
                                end

                                if one_sample_RT_pvalue_unbiased_a_H1 < 0.05
                                    %Rejet de H0
                                    test_one_sample_RT_decision_unbiased_H1(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_one_sample_RT_decision_unbiased_H1(i_simul,1) = 0;
                                end
                                
                                % Two Sample Stats
                                
                                if two_sample_RT_pvalue_biased_a_H0 < 0.05
                                    %Rejet de H0
                                    test_two_sample_RT_decision_biased_H0(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_two_sample_RT_decision_biased_H0(i_simul,1) = 0;
                                end

                                if two_sample_RT_pvalue_unbiased_a_H0 < 0.05
                                    %Rejet de H0
                                    test_two_sample_RT_decision_unbiased_H0(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_two_sample_RT_decision_unbiased_H0(i_simul,1) = 0;
                                end
                     
                                if two_sample_RT_pvalue_biased_a_H1 < 0.05
                                    %Rejet de H0
                                    test_two_sample_RT_decision_biased_H1(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_two_sample_RT_decision_biased_H1(i_simul,1) = 0;
                                end

                                if two_sample_RT_pvalue_unbiased_a_H1 < 0.05
                                    %Rejet de H0
                                    test_two_sample_RT_decision_unbiased_H1(i_simul,1) = 1;
                                else
                                    %Non-rejet de H0
                                    test_two_sample_RT_decision_unbiased_H1(i_simul,1) = 0;
                                end
                                                                             
                                % One scenari model summary
                                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                                
                                OneSimulModel_H0(1,2) = {est_mean_a_H0};
                                OneSimulModel_H0(2,2) = {est_vol_a_H0};
                                OneSimulModel_H0(3,2) = {est_skew_a_H0};
                                OneSimulModel_H0(4,2) = {est_kurto_a_H0};
                                OneSimulModel_H0(5,2) = {est_sharpe_bias_a_H0};
                                OneSimulModel_H0(6,2) = {est_sharpe_biased_a_H0};
                                OneSimulModel_H0(7,2) = {est_sharpe_unbiased_a_H0};
                                OneSimulModel_H0(8,2) = {std_est_sharpe_biased_a_H0};
                                OneSimulModel_H0(9,2) = {std_est_sharpe_unbiased_a_H0};
                                OneSimulModel_H0(10,2) = {one_sample_stat_biased_a_H0};
                                OneSimulModel_H0(11,2) = {one_sample_stat_unbiased_a_H0};
                                OneSimulModel_H0(12,2) = {one_sample_LT_pvalue_biased_a_H0};
                                OneSimulModel_H0(13,2) = {one_sample_LT_pvalue_unbiased_a_H0};
                                OneSimulModel_H0(14,2) = {one_sample_RT_pvalue_biased_a_H0};
                                OneSimulModel_H0(15,2) = {one_sample_RT_pvalue_unbiased_a_H0};                            
                                OneSimulModel_H0(16,2) = {one_sample_2T_pvalue_biased_a_H0};
                                OneSimulModel_H0(17,2) = {one_sample_2T_pvalue_unbiased_a_H0};
                                OneSimulModel_H0(18,2) = {confidence_boundary_one_sample_1T_biased_a_H0};
                                OneSimulModel_H0(19,2) = {confidence_boundary_one_sample_1T_unbiased_a_H0};                            
                                OneSimulModel_H0(20,2) = {confidence_boundary_one_sample_2T_biased_a_H0};
                                OneSimulModel_H0(21,2) = {confidence_boundary_one_sample_2T_unbiased_a_H0};
                                OneSimulModel_H0(22,2) = {two_sample_stat_biased_a_H0};
                                OneSimulModel_H0(23,2) = {two_sample_stat_unbiased_a_H0};
                                OneSimulModel_H0(24,2) = {two_sample_LT_pvalue_biased_a_H0};
                                OneSimulModel_H0(25,2) = {two_sample_LT_pvalue_unbiased_a_H0};
                                OneSimulModel_H0(26,2) = {two_sample_RT_pvalue_biased_a_H0};
                                OneSimulModel_H0(27,2) = {two_sample_RT_pvalue_unbiased_a_H0};                            
                                OneSimulModel_H0(28,2) = {two_sample_2T_pvalue_biased_a_H0};
                                OneSimulModel_H0(29,2) = {two_sample_2T_pvalue_unbiased_a_H0};                           
                                OneSimulModel_H0(30,2) = {confidence_boundary_two_sample_2T_biased_a_H0};
                                OneSimulModel_H0(31,2) = {confidence_boundary_two_sample_2T_unbiased_a_H0};

                                OneSimulModel_H1(1,2) = {est_mean_a_H1};
                                OneSimulModel_H1(2,2) = {est_vol_a_H1};
                                OneSimulModel_H1(3,2) = {est_skew_a_H1};
                                OneSimulModel_H1(4,2) = {est_kurto_a_H1};
                                OneSimulModel_H1(5,2) = {est_sharpe_bias_a_H1};
                                OneSimulModel_H1(6,2) = {est_sharpe_biased_a_H1};
                                OneSimulModel_H1(7,2) = {est_sharpe_unbiased_a_H1};
                                OneSimulModel_H1(8,2) = {std_est_sharpe_biased_a_H1};
                                OneSimulModel_H1(9,2) = {std_est_sharpe_unbiased_a_H1};
                                OneSimulModel_H1(10,2) = {one_sample_stat_biased_a_H1};
                                OneSimulModel_H1(11,2) = {one_sample_stat_unbiased_a_H1};
                                OneSimulModel_H1(12,2) = {one_sample_LT_pvalue_biased_a_H1};
                                OneSimulModel_H1(13,2) = {one_sample_LT_pvalue_unbiased_a_H1};
                                OneSimulModel_H1(14,2) = {one_sample_RT_pvalue_biased_a_H1};
                                OneSimulModel_H1(15,2) = {one_sample_RT_pvalue_unbiased_a_H1};                            
                                OneSimulModel_H1(16,2) = {one_sample_2T_pvalue_biased_a_H1};
                                OneSimulModel_H1(17,2) = {one_sample_2T_pvalue_unbiased_a_H1};
                                OneSimulModel_H1(18,2) = {confidence_boundary_one_sample_1T_biased_a_H1};
                                OneSimulModel_H1(19,2) = {confidence_boundary_one_sample_1T_unbiased_a_H1};                            
                                OneSimulModel_H1(20,2) = {confidence_boundary_one_sample_2T_biased_a_H1};
                                OneSimulModel_H1(21,2) = {confidence_boundary_one_sample_2T_unbiased_a_H1};
                                OneSimulModel_H1(22,2) = {two_sample_stat_biased_a_H1};
                                OneSimulModel_H1(23,2) = {two_sample_stat_unbiased_a_H1};
                                OneSimulModel_H1(24,2) = {two_sample_LT_pvalue_biased_a_H1};
                                OneSimulModel_H1(25,2) = {two_sample_LT_pvalue_unbiased_a_H1};
                                OneSimulModel_H1(26,2) = {two_sample_RT_pvalue_biased_a_H1};
                                OneSimulModel_H1(27,2) = {two_sample_RT_pvalue_unbiased_a_H1};                            
                                OneSimulModel_H1(28,2) = {two_sample_2T_pvalue_biased_a_H1};
                                OneSimulModel_H1(29,2) = {two_sample_2T_pvalue_unbiased_a_H1};                           
                                OneSimulModel_H1(30,2) = {confidence_boundary_two_sample_2T_biased_a_H1};
                                OneSimulModel_H1(31,2) = {confidence_boundary_two_sample_2T_unbiased_a_H1};
                                
                                AllSimulModels_H0(i_simul) = ...
                                    {cell2table(OneSimulModel_H0(:, 2)', 'VariableName', OneSimulModel_H0(:, 1))};
                                
                                AllSimulModels_H1(i_simul) = ...
                                    {cell2table(OneSimulModel_H1(:, 2)', 'VariableName', OneSimulModel_H1(:, 1))};
                            end

                            % One-Sample Two-Tailed Tests Results : Alpha and Beta Errors, Test Power
                            rejection_rate_one_sample_2T_biased_H0 = sum(test_one_sample_2T_decision_biased_H0 == 1) / size(test_one_sample_2T_decision_biased_H0,1);
                            rejection_rate_one_sample_2T_unbiased_H0 = sum(test_one_sample_2T_decision_unbiased_H0 == 1) / size(test_one_sample_2T_decision_unbiased_H0,1);
                            rejection_rate_one_sample_2T_biased_H1 = sum(test_one_sample_2T_decision_biased_H1 == 1) / size(test_one_sample_2T_decision_biased_H1,1);
                            rejection_rate_one_sample_2T_unbiased_H1 = sum(test_one_sample_2T_decision_unbiased_H1 == 1) / size(test_one_sample_2T_decision_unbiased_H1,1);
                            
                            % One-Sample Left-Tailed Tests Results : Alpha and Beta Errors, Test Power
                            rejection_rate_one_sample_LT_biased_H0 = sum(test_one_sample_LT_decision_biased_H0 == 1) / size(test_one_sample_LT_decision_biased_H0,1);
                            rejection_rate_one_sample_LT_unbiased_H0 = sum(test_one_sample_LT_decision_unbiased_H0 == 1) / size(test_one_sample_LT_decision_unbiased_H0,1);
                            rejection_rate_one_sample_LT_biased_H1 = sum(test_one_sample_LT_decision_biased_H1 == 1) / size(test_one_sample_LT_decision_biased_H1,1);
                            rejection_rate_one_sample_LT_unbiased_H1 = sum(test_one_sample_LT_decision_unbiased_H1 == 1) / size(test_one_sample_LT_decision_unbiased_H1,1);
                            
                            % One-Sample Right-Tailed Tests Results : Alpha and Beta Errors, Test Power
                            rejection_rate_one_sample_RT_biased_H0 = sum(test_one_sample_RT_decision_biased_H0 == 1) / size(test_one_sample_RT_decision_biased_H0,1);
                            rejection_rate_one_sample_RT_unbiased_H0 = sum(test_one_sample_RT_decision_unbiased_H0 == 1) / size(test_one_sample_RT_decision_unbiased_H0,1);
                            rejection_rate_one_sample_RT_biased_H1 = sum(test_one_sample_RT_decision_biased_H1 == 1) / size(test_one_sample_RT_decision_biased_H1,1);
                            rejection_rate_one_sample_RT_unbiased_H1 = sum(test_one_sample_RT_decision_unbiased_H1 == 1) / size(test_one_sample_RT_decision_unbiased_H1,1);
                            
                            % Two-Sample Two-Tailed Tests Results : Alpha and Beta Errors, Test Power
                            rejection_rate_two_sample_2T_biased_H0 = sum(test_two_sample_2T_decision_biased_H0 == 1) / size(test_two_sample_2T_decision_biased_H0,1);
                            rejection_rate_two_sample_2T_unbiased_H0 = sum(test_two_sample_2T_decision_unbiased_H0 == 1) / size(test_two_sample_2T_decision_unbiased_H0,1);
                            rejection_rate_two_sample_2T_biased_H1 = sum(test_two_sample_2T_decision_biased_H1 == 1) / size(test_two_sample_2T_decision_biased_H1,1);
                            rejection_rate_two_sample_2T_unbiased_H1 = sum(test_two_sample_2T_decision_unbiased_H1 == 1) / size(test_two_sample_2T_decision_unbiased_H1,1);
                            
                            % Two-Sample Left-Tailed Tests Results : Alpha and Beta Errors, Test Power
                            rejection_rate_two_sample_LT_biased_H0 = sum(test_two_sample_LT_decision_biased_H0 == 1) / size(test_two_sample_LT_decision_biased_H0,1);
                            rejection_rate_two_sample_LT_unbiased_H0 = sum(test_two_sample_LT_decision_unbiased_H0 == 1) / size(test_two_sample_LT_decision_unbiased_H0,1);
                            rejection_rate_two_sample_LT_biased_H1 = sum(test_two_sample_LT_decision_biased_H1 == 1) / size(test_two_sample_LT_decision_biased_H1,1);
                            rejection_rate_two_sample_LT_unbiased_H1 = sum(test_two_sample_LT_decision_unbiased_H1 == 1) / size(test_two_sample_LT_decision_unbiased_H1,1);
                            
                            % Two-Sample Right-Tailed Tests Results : Alpha and Beta Errors, Test Power
                            rejection_rate_two_sample_RT_biased_H0 = sum(test_two_sample_RT_decision_biased_H0 == 1) / size(test_two_sample_RT_decision_biased_H0,1);
                            rejection_rate_two_sample_RT_unbiased_H0 = sum(test_two_sample_RT_decision_unbiased_H0 == 1) / size(test_two_sample_RT_decision_unbiased_H0,1);
                            rejection_rate_two_sample_RT_biased_H1 = sum(test_two_sample_RT_decision_biased_H1 == 1) / size(test_two_sample_RT_decision_biased_H1,1);
                            rejection_rate_two_sample_RT_unbiased_H1 = sum(test_two_sample_RT_decision_unbiased_H1 == 1) / size(test_two_sample_RT_decision_unbiased_H1,1);
                            
                            OneScenarResult_H0(7,2) = {AllSimulModels_H0};
                            OneScenarResult_H0(8,2) = {rejection_rate_one_sample_LT_biased_H0};
                            OneScenarResult_H0(9,2) = {rejection_rate_one_sample_LT_unbiased_H0};
                            OneScenarResult_H0(10,2) = {rejection_rate_one_sample_RT_biased_H0};
                            OneScenarResult_H0(11,2) = {rejection_rate_one_sample_RT_unbiased_H0};
                            OneScenarResult_H0(12,2) = {rejection_rate_one_sample_2T_biased_H0};
                            OneScenarResult_H0(13,2) = {rejection_rate_one_sample_2T_unbiased_H0};
                            OneScenarResult_H0(14,2) = {rejection_rate_two_sample_LT_biased_H0};
                            OneScenarResult_H0(15,2) = {rejection_rate_two_sample_LT_unbiased_H0};
                            OneScenarResult_H0(16,2) = {rejection_rate_two_sample_RT_biased_H0};
                            OneScenarResult_H0(17,2) = {rejection_rate_two_sample_RT_unbiased_H0};
                            OneScenarResult_H0(18,2) = {rejection_rate_two_sample_2T_biased_H0};
                            OneScenarResult_H0(19,2) = {rejection_rate_two_sample_2T_unbiased_H0};

                            OneScenarResult_H1(7,2) = {AllSimulModels_H1};
                            OneScenarResult_H1(8,2) = {rejection_rate_one_sample_LT_biased_H1};
                            OneScenarResult_H1(9,2) = {rejection_rate_one_sample_LT_unbiased_H1};
                            OneScenarResult_H1(10,2) = {rejection_rate_one_sample_RT_biased_H1};
                            OneScenarResult_H1(11,2) = {rejection_rate_one_sample_RT_unbiased_H1};
                            OneScenarResult_H1(12,2) = {rejection_rate_one_sample_2T_biased_H1};
                            OneScenarResult_H1(13,2) = {rejection_rate_one_sample_2T_unbiased_H1};
                            OneScenarResult_H1(14,2) = {rejection_rate_two_sample_LT_biased_H1};
                            OneScenarResult_H1(15,2) = {rejection_rate_two_sample_LT_unbiased_H1};
                            OneScenarResult_H1(16,2) = {rejection_rate_two_sample_RT_biased_H1};
                            OneScenarResult_H1(17,2) = {rejection_rate_two_sample_RT_unbiased_H1};
                            OneScenarResult_H1(18,2) = {rejection_rate_two_sample_2T_biased_H1};
                            OneScenarResult_H1(19,2) = {rejection_rate_two_sample_2T_unbiased_H1};
                            
                            AllScenarResults_H0(i_scenar) = ...
                                {cell2table(OneScenarResult_H0(:, 2)', 'VariableName', OneScenarResult_H0(:, 1))};
                            AllScenarResults_H1(i_scenar) = ...
                                {cell2table(OneScenarResult_H1(:, 2)', 'VariableName', OneScenarResult_H1(:, 1))};
                            i_scenar = i_scenar + 1;
                        end
                    end
                end
            end
        end
    end
end
%% Plotting Alpha Errors VS Sample Size // Test Power VS Sample Size

% Alpha Error is the Rejection Rate under H0 
% Test Power = 1 - Beta Error, which is the Rejection Rate under Ha 

% Creating vector to plots (Sample size -> Rejection Rate Biased & Unbiased)
% for each level of threshold c for SRa
prev_sharpe_1 = AllScenarResults_H0{1}.c_Sharpe_1;
prev_sharpe_2 = AllScenarResults_H0{1}.c_Sharpe_2;
all_plots_one_sample_alpha_biased = cell(size(range_sharpe_1, 1), 1);
all_plots_one_sample_alpha_unbiased = cell(size(range_sharpe_1, 1), 1);
all_plots_one_sample_power_biased = cell(size(range_sharpe_1, 1), 1);
all_plots_one_sample_power_unbiased = cell(size(range_sharpe_1, 1), 1);
all_plots_two_sample_alpha_biased = cell(size(range_sharpe_1, 1), 1);
all_plots_two_sample_alpha_unbiased = cell(size(range_sharpe_1, 1), 1);
all_plots_two_sample_power_biased = cell(size(range_sharpe_1, 1), 1);
all_plots_two_sample_power_unbiased = cell(size(range_sharpe_1, 1), 1);
sharpes = cell(size(range_sharpe_1, 2), 2);
X_sample_size = [];
Y_one_sample_alpha_error_biased = [];
Y_one_sample_alpha_error_unbiased = [];
Y_one_sample_test_power_biased = [];
Y_one_sample_test_power_unbiased = [];
Y_two_sample_alpha_error_biased = [];
Y_two_sample_alpha_error_unbiased = [];
Y_two_sample_test_power_biased = [];
Y_two_sample_test_power_unbiased = [];
i_one_period = 1;
i_one_sample_size = 1;
for i_row=1:size(AllScenarResults_H0, 1)
    oneScenario_H0 = AllScenarResults_H0{i_row};
    oneScenario_H1 = AllScenarResults_H0{i_row};
    one_sharpe_1 = oneScenario_H0.c_Sharpe_1;
    one_sharpe_2 = oneScenario_H0.c_Sharpe_2;
    if one_sharpe_1 == prev_sharpe_1 && one_sharpe_2 == prev_sharpe_2
        X_sample_size(i_one_sample_size) = oneScenario_H0.SampleSize;
        Y_one_sample_alpha_error_biased(i_one_sample_size) = oneScenario_H0.RT_one_sample_RejectionRateBiased;
        Y_one_sample_alpha_error_unbiased(i_one_sample_size) = oneScenario_H0.RT_one_sample_RejectionRateUnbiased;
        Y_one_sample_test_power_biased(i_one_sample_size) = oneScenario_H1.LT_one_sample_RejectionRateBiased;
        Y_one_sample_test_power_unbiased(i_one_sample_size) = oneScenario_H1.LT_one_sample_RejectionRateUnbiased;
        Y_two_sample_alpha_error_biased(i_one_sample_size) = oneScenario_H0.RT_two_sample_RejectionRateBiased;
        Y_two_sample_alpha_error_unbiased(i_one_sample_size) = oneScenario_H0.RT_two_sample_RejectionRateUnbiased;
        Y_two_sample_test_power_biased(i_one_sample_size) = oneScenario_H1.LT_two_sample_RejectionRateBiased;
        Y_two_sample_test_power_unbiased(i_one_sample_size) = oneScenario_H1.LT_two_sample_RejectionRateUnbiased;
        i_one_sample_size = i_one_sample_size + 1;
    else
        i_one_sample_size = 1;
        prev_sharpe_1 = one_sharpe_1;
        prev_sharpe_2 = one_sharpe_2;
        all_plots_one_sample_alpha_biased(i_one_period, 1) = {[X_sample_size', Y_one_sample_alpha_error_biased']};
        all_plots_one_sample_alpha_unbiased(i_one_period, 1) = {[X_sample_size', Y_one_sample_alpha_error_unbiased']};
        all_plots_one_sample_power_biased(i_one_period, 1) = {[X_sample_size', Y_one_sample_test_power_biased']};
        all_plots_one_sample_power_unbiased(i_one_period, 1) = {[X_sample_size', Y_one_sample_test_power_unbiased']};
        all_plots_two_sample_alpha_biased(i_one_period, 1) = {[X_sample_size', Y_two_sample_alpha_error_biased']};
        all_plots_two_sample_alpha_unbiased(i_one_period, 1) = {[X_sample_size', Y_two_sample_alpha_error_unbiased']};
        all_plots_two_sample_power_biased(i_one_period, 1) = {[X_sample_size', Y_two_sample_test_power_biased']};
        all_plots_two_sample_power_unbiased(i_one_period, 1) = {[X_sample_size', Y_two_sample_test_power_unbiased']};
        sharpes(i_one_period, 1) = {AllScenarResults_H0{i_row - 1}.c_Sharpe_1};
        sharpes(i_one_period, 2) = {AllScenarResults_H0{i_row - 1}.c_Sharpe_2};
        i_one_period = i_one_period + 1;
        X_sample_size = [];
        Y_one_sample_alpha_error_biased = [];
        Y_one_sample_alpha_error_unbiased = [];
        Y_one_sample_test_power_biased = [];
        Y_one_sample_test_power_unbiased = [];
        Y_two_sample_alpha_error_biased = [];
        Y_two_sample_alpha_error_unbiased = [];
        Y_two_sample_test_power_biased = [];
        Y_two_sample_test_power_unbiased = [];
        
        X_sample_size(i_one_sample_size) = oneScenario_H0.SampleSize;
        Y_one_sample_alpha_error_biased(i_one_sample_size) = oneScenario_H0.RT_one_sample_RejectionRateBiased;
        Y_one_sample_alpha_error_unbiased(i_one_sample_size) = oneScenario_H0.RT_one_sample_RejectionRateUnbiased;
        Y_one_sample_test_power_biased(i_one_sample_size) = oneScenario_H1.LT_one_sample_RejectionRateBiased;
        Y_one_sample_test_power_unbiased(i_one_sample_size) = oneScenario_H1.LT_one_sample_RejectionRateBiased;
        Y_two_sample_alpha_error_biased(i_one_sample_size) = oneScenario_H0.RT_two_sample_RejectionRateBiased;
        Y_two_sample_alpha_error_unbiased(i_one_sample_size) = oneScenario_H0.RT_two_sample_RejectionRateUnbiased;
        Y_two_sample_test_power_biased(i_one_sample_size) = oneScenario_H1.LT_two_sample_RejectionRateBiased;
        Y_two_sample_test_power_unbiased(i_one_sample_size) = oneScenario_H1.LT_two_sample_RejectionRateUnbiased;
        i_one_sample_size = i_one_sample_size + 1;
    end
    
    if i_row == size(AllScenarResults_H0, 1)
        all_plots_one_sample_alpha_biased(i_one_period, 1) = {[X_sample_size', Y_one_sample_alpha_error_biased']};
        all_plots_one_sample_alpha_unbiased(i_one_period, 1) = {[X_sample_size', Y_one_sample_alpha_error_unbiased']};
        all_plots_one_sample_power_biased(i_one_period, 1) = {[X_sample_size', Y_one_sample_test_power_biased']};
        all_plots_one_sample_power_unbiased(i_one_period, 1) = {[X_sample_size', Y_one_sample_test_power_unbiased']};
        all_plots_two_sample_alpha_biased(i_one_period, 1) = {[X_sample_size', Y_two_sample_alpha_error_biased']};
        all_plots_two_sample_alpha_unbiased(i_one_period, 1) = {[X_sample_size', Y_two_sample_alpha_error_unbiased']};
        all_plots_two_sample_power_biased(i_one_period, 1) = {[X_sample_size', Y_two_sample_test_power_biased']};
        all_plots_two_sample_power_unbiased(i_one_period, 1) = {[X_sample_size', Y_two_sample_test_power_unbiased']};
        sharpes(i_one_period, 1) = {AllScenarResults_H0{i_row - 1}.c_Sharpe_1};
        sharpes(i_one_period, 2) = {AllScenarResults_H0{i_row - 1}.c_Sharpe_2};
    end
end

% Plotting One Sample Biased Alpha Error
%lines = {'-', ':', '-.', ':'};
lines = {'-', '-', '-', '-', '-', '-'};
colors = {'c', 'r', 'g', 'b', 'm', 'k'};
markers = {'o', '+', '*', '.', 'x', 's'};
legends = cell(size(range_sharpe_1, 1), 1);
n_graphs = 1;
fig = figure;
hold on;
for i_plots=1:size(all_plots_one_sample_alpha_biased, 1)    
    if sharpes{i_plots, 1} == sharpes{i_plots, 2}
        one_plot = all_plots_one_sample_alpha_biased{i_plots, 1};
        one_sharpe_threshold = sharpes{i_plots, 1};
        
        random_linecolor = colors(n_graphs);
        random_linestyle = lines(n_graphs);
        random_linemarker = markers(n_graphs);
        
        plot(one_plot(:, 1), one_plot(:, 2), ...
            string(random_linecolor) + string(random_linestyle) + string(random_linemarker));
        %legend('SR = c = ' + string(one_sharpe_threshold));
        legends(n_graphs, 1) = {'c = ' + string(one_sharpe_threshold)};
        
        n_graphs = n_graphs + 1;
    end
end
title('Esimated One Sample Alpha Error for Test SR < c (Bias Not Corrected)');
xlabel('Sample Size');
ylabel('Estimated Alpha Error');
legend(legends);
hold off;

% Plotting One Sample Unbiased Alpha Error 
lines = {'-', '-', '-', '-', '-', '-'};
colors = {'c', 'r', 'g', 'b', 'm', 'k'};
markers = {'o', '+', '*', '.', 'x', 's'};
legends = cell(size(range_sharpe_1, 1), 1);
n_graphs = 1;
fig = figure;
hold on;
for i_plots=1:size(all_plots_one_sample_alpha_unbiased, 1)    
    if sharpes{i_plots, 1} == sharpes{i_plots, 2}
        one_plot = all_plots_one_sample_alpha_unbiased{i_plots, 1};
        one_sharpe_threshold = sharpes{i_plots, 1};
        
        random_linecolor = colors(n_graphs);
        random_linestyle = lines(n_graphs);
        random_linemarker = markers(n_graphs);
        
        plot(one_plot(:, 1), one_plot(:, 2), ...
            string(random_linecolor) + string(random_linestyle) + string(random_linemarker));
        %legend('SR = c = ' + string(one_sharpe_threshold));
        legends(n_graphs, 1) = {'c = ' + string(one_sharpe_threshold)};
        
        n_graphs = n_graphs + 1;
    end
end
title('Estimated One Sample Alpha Error for Test SR < c (Bias Corrected)');
xlabel('Sample Size');
ylabel('Estimated Alpha Error');
legend(legends);
hold off;

% Plotting One Sample Biased Test Power 
lines = {'-', '-', '-', '-', '-', '-'};
colors = {'c', 'r', 'g', 'b', 'm', 'k'};
markers = {'o', '+', '*', '.', 'x', 's'};
legends = cell(size(range_sharpe_1, 1), 1);
n_graphs = 1;
fig = figure;
hold on;
for i_plots=1:size(all_plots_one_sample_power_biased, 1)    
    if sharpes{i_plots, 1} == sharpes{i_plots, 2}
        one_plot = all_plots_one_sample_power_biased{i_plots, 1};
        one_sharpe_threshold = sharpes{i_plots, 1};
        
        random_linecolor = colors(n_graphs);
        random_linestyle = lines(n_graphs);
        random_linemarker = markers(n_graphs);
        
        plot(one_plot(:, 1), one_plot(:, 2), ...
            string(random_linecolor) + string(random_linestyle) + string(random_linemarker));
        %legend('SR = c = ' + string(one_sharpe_threshold));
        legends(n_graphs, 1) = {'c = ' + string(one_sharpe_threshold)};
        
        n_graphs = n_graphs + 1;
    end
end
title('Estimated One Sample Test Power for Test SR < c (Bias Not Corrected)');
xlabel('Sample Size');
ylabel('(1 - Beta error)');
legend(legends);
hold off;

% Plotting One Sample Unbiased Test Power 
lines = {'-', '-', '-', '-', '-', '-'};
colors = {'c', 'r', 'g', 'b', 'm', 'k'};
markers = {'o', '+', '*', '.', 'x', 's'};
legends = cell(size(range_sharpe_1, 1), 1);
n_graphs = 1;
fig = figure;
hold on;
for i_plots=1:size(all_plots_one_sample_power_unbiased, 1)    
    if sharpes{i_plots, 1} == sharpes{i_plots, 2}
        one_plot = all_plots_one_sample_power_unbiased{i_plots, 1};
        one_sharpe_threshold = sharpes{i_plots, 1};
        
        random_linecolor = colors(n_graphs);
        random_linestyle = lines(n_graphs);
        random_linemarker = markers(n_graphs);
        
        plot(one_plot(:, 1), one_plot(:, 2), ...
            string(random_linecolor) + string(random_linestyle) + string(random_linemarker));
        %legend('SR = c = ' + string(one_sharpe_threshold));
        legends(n_graphs, 1) = {'c = ' + string(one_sharpe_threshold)};
        
        n_graphs = n_graphs + 1;
    end
end
title('Estimated One Sample Test Power for Test SR < c (Bias Corrected)');
xlabel('Sample Size');
ylabel('(1 - Beta error)');
legend(legends);
hold off;

% Plotting Two Sample Biased Alpha Error
%lines = {'-', ':', '-.', ':'};
lines = {'-', '-', '-', '-', '-', '-'};
colors = {'c', 'r', 'g', 'b', 'm', 'k'};
markers = {'o', '+', '*', '.', 'x', 's'};
legends = cell(size(range_sharpe_1, 1), 1);
n_graphs = 1;
fig = figure;
hold on;
for i_plots=1:size(all_plots_one_sample_alpha_biased, 1)    
    if sharpes{i_plots, 1} == sharpes{i_plots, 2}
        one_plot = all_plots_one_sample_alpha_biased{i_plots, 1};
        one_sharpe_threshold = sharpes{i_plots, 1};
        
        random_linecolor = colors(n_graphs);
        random_linestyle = lines(n_graphs);
        random_linemarker = markers(n_graphs);
        
        plot(one_plot(:, 1), one_plot(:, 2), ...
            string(random_linecolor) + string(random_linestyle) + string(random_linemarker));
        %legend('SR = c = ' + string(one_sharpe_threshold));
        legends(n_graphs, 1) = {'c = ' + string(one_sharpe_threshold)};
        
        n_graphs = n_graphs + 1;
    end
end
title('Esimated Two Sample Alpha Error for Test SR < c (Bias Not Corrected)');
xlabel('Sample Size');
ylabel('Estimated Alpha Error');
legend(legends);
hold off;

% Plotting Two Sample Unbiased Alpha Error 
lines = {'-', '-', '-', '-', '-', '-'};
colors = {'c', 'r', 'g', 'b', 'm', 'k'};
markers = {'o', '+', '*', '.', 'x', 's'};
legends = cell(size(range_sharpe_1, 1), 1);
n_graphs = 1;
fig = figure;
hold on;
for i_plots=1:size(all_plots_one_sample_alpha_unbiased, 1)    
    if sharpes{i_plots, 1} == sharpes{i_plots, 2}
        one_plot = all_plots_one_sample_alpha_unbiased{i_plots, 1};
        one_sharpe_threshold = sharpes{i_plots, 1};
        
        random_linecolor = colors(n_graphs);
        random_linestyle = lines(n_graphs);
        random_linemarker = markers(n_graphs);
        
        plot(one_plot(:, 1), one_plot(:, 2), ...
            string(random_linecolor) + string(random_linestyle) + string(random_linemarker));
        %legend('SR = c = ' + string(one_sharpe_threshold));
        legends(n_graphs, 1) = {'c = ' + string(one_sharpe_threshold)};
        
        n_graphs = n_graphs + 1;
    end
end
title('Estimated Two Sample Alpha Error for Test SR < c (Bias Corrected)');
xlabel('Sample Size');
ylabel('Estimated Alpha Error');
legend(legends);
hold off;

% Plotting Two Sample Biased Test Power 
lines = {'-', '-', '-', '-', '-', '-'};
colors = {'c', 'r', 'g', 'b', 'm', 'k'};
markers = {'o', '+', '*', '.', 'x', 's'};
legends = cell(size(range_sharpe_1, 1), 1);
n_graphs = 1;
fig = figure;
hold on;
for i_plots=1:size(all_plots_one_sample_power_biased, 1)    
    if sharpes{i_plots, 1} == sharpes{i_plots, 2}
        one_plot = all_plots_one_sample_power_biased{i_plots, 1};
        one_sharpe_threshold = sharpes{i_plots, 1};
        
        random_linecolor = colors(n_graphs);
        random_linestyle = lines(n_graphs);
        random_linemarker = markers(n_graphs);
        
        plot(one_plot(:, 1), one_plot(:, 2), ...
            string(random_linecolor) + string(random_linestyle) + string(random_linemarker));
        %legend('SR = c = ' + string(one_sharpe_threshold));
        legends(n_graphs, 1) = {'c = ' + string(one_sharpe_threshold)};
        
        n_graphs = n_graphs + 1;
    end
end
title('Estimated Two Sample Test Power for Test SR < c (Bias Not Corrected)');
xlabel('Sample Size');
ylabel('(1 - Beta error)');
legend(legends);
hold off;

% Plotting Two Sample Unbiased Test Power 
lines = {'-', '-', '-', '-', '-', '-'};
colors = {'c', 'r', 'g', 'b', 'm', 'k'};
markers = {'o', '+', '*', '.', 'x', 's'};
legends = cell(size(range_sharpe_1, 1), 1);
n_graphs = 1;
fig = figure;
hold on;
for i_plots=1:size(all_plots_one_sample_power_unbiased, 1)    
    if sharpes{i_plots, 1} == sharpes{i_plots, 2}
        one_plot = all_plots_one_sample_power_unbiased{i_plots, 1};
        one_sharpe_threshold = sharpes{i_plots, 1};
        
        random_linecolor = colors(n_graphs);
        random_linestyle = lines(n_graphs);
        random_linemarker = markers(n_graphs);
        
        plot(one_plot(:, 1), one_plot(:, 2), ...
            string(random_linecolor) + string(random_linestyle) + string(random_linemarker));
        %legend('SR = c = ' + string(one_sharpe_threshold));
        legends(n_graphs, 1) = {'c = ' + string(one_sharpe_threshold)};
        
        n_graphs = n_graphs + 1;
    end
end
title('Estimated Two Sample Test Power for Test SR < c (Bias Corrected)');
xlabel('Sample Size');
ylabel('(1 - Beta error)');
legend(legends);
hold off;

%% V) Sharpe Ratio Comparisons of Actual Mutual Fund Returns

%funds_data = xlsread('DataFunds.xlsx', 'LargeGrowth_5Y_Monthly');
%funds_data = readtable('DataFunds.xlsx', 'Sheet', 'LargeGrowth_5Y_Monthly');
funds_data = readtable('dataopc.xlsm', 'Sheet', 'DataReturn');

dates = funds_data(:, 1);
% rf_rate = funds_data(:, 2);
% dataset = funds_data(:,3:end);
dataset = funds_data(:, 2:end);
rf_rate = 0.00052;

[correlation_matrix, pairwise_pvalues] = Pairwise_Pvalues(dataset, rf_rate);

disp('------------------------------------');
disp('All Pairwise Correlation of Funds :');
disp(correlation_matrix);
disp('------------------------------------');

disp('------------------------------------');
disp('All Pairwise Two-Sample Test P_Values of Funds :');
disp(pairwise_pvalues);
disp('------------------------------------');

fund_ranking = Rank_Funds(dataset, rf_rate);

disp('------------------------------------');
disp('Funds Ranking');
disp(fund_ranking);
disp('------------------------------------');

%% Ouverture : Trading Idea Implementation

% Pair Trading With Two-Sample Stats
fund_data = readtable('basedehc.xlsx');
fund_data = fund_data(:, 1:end-1);
fund_price = array2table(ret2price(table2array(fund_data), 100), ...
    'VariableNames', fund_data.Properties.VariableNames);
benchmark = -sum(1/size(fund_data, 2) * table2array(fund_price(1, :))) ...
            + sum(1/size(fund_data, 2) * table2array(fund_price(end, :)));
rf_rate = 0.001;
nb_max_traded_fund = 1;
nb_funds = size(fund_data, 2);
window = 30;
cash = 0;
prev_cash = 0;
open_position = false;
short_fund = "";
long_fund = "";
trades_payoff = zeros(round(size(fund_data, 1)/30), 1);
trades_returns = zeros(round(size(fund_data, 1)/30), 1);
position_price = 0;
position_payoff = 0;
max_drawdown = 0;
i_trades = 1;
for i_row=window:window:size(fund_data, 1)
    rolling_dataset = fund_data(i_row-window+1:i_row, :);
    [~, pairwise_pvalues] = Pairwise_Pvalues(rolling_dataset, rf_rate);
    vect_pvals = sort(reshape(table2array(pairwise_pvalues), [nb_funds*nb_funds, 1]));
    vect_pvals = vect_pvals(vect_pvals ~= -1);
    best_pvals = vect_pvals(1:nb_max_traded_fund, 1);
    
    fund_to_short = cell(nb_max_traded_fund, 1);
    fund_to_buy = cell(nb_max_traded_fund, 1);
    
    for i_pval=1:size(best_pvals, 1)
        
        if open_position
            %cash = cash - table2array(fund_price(i_row + 1, short_fund)) + table2array(fund_price(i_row + 1, long_fund));
            position_payoff = - table2array(fund_price(i_row + 1, short_fund)) + table2array(fund_price(i_row + 1, long_fund));
            cash = cash + position_payoff;
            %trades_payoff(i_trades, 1) = cash - prev_cash;
            trades_payoff(i_trades, 1) = position_payoff - position_price;
            trades_returns(i_trades, 1) = trades_payoff(i_trades, 1) / ((price_fund_to_short+price_fund_to_buy)/2);
            position_payoff = 0;
            position_price = 0;
            short_fund = "";
            long_fund = "";
            open_position = false;
            prev_cash = cash;
            i_trades = i_trades + 1;
            
            if cash < max_drawdown
               max_drawdown = cash; 
            end
        end
        
        [best_funds_index_row, best_funds_index_col] ...
            = find(table2array(pairwise_pvalues) == best_pvals(i_pval, 1));
        fund_to_buy(i_pval, 1) = pairwise_pvalues.Properties.VariableNames(best_funds_index_col);
        fund_to_short(i_pval, 1) = pairwise_pvalues.Properties.RowNames(best_funds_index_row);
        price_fund_to_short = table2array(fund_price(i_row + 1, fund_to_short(i_pval, 1)));
        price_fund_to_buy = table2array(fund_price(i_row + 1, fund_to_buy(i_pval, 1)));
        
        %cash = cash + price_fund_to_short - price_fund_to_buy;
        position_price = price_fund_to_short - price_fund_to_buy;
        cash = cash + position_price;
        open_position = true;
        short_fund = fund_to_short(i_pval, 1);
        long_fund = fund_to_buy(i_pval, 1);
    end
end

        if open_position
            %cash = cash - table2array(fund_price(i_row + 1, short_fund)) + table2array(fund_price(i_row + 1, long_fund));
            position_payoff = - table2array(fund_price(i_row + 1, short_fund)) + table2array(fund_price(i_row + 1, long_fund));
            cash = cash + position_payoff;
            %trades_payoff(i_trades, 1) = cash - prev_cash;
            trades_payoff(i_trades, 1) = position_payoff - position_price;
            trades_returns(i_trades, 1) = trades_payoff(i_trades, 1) / ((price_fund_to_short+price_fund_to_buy)/2);
            position_payoff = 0;
            position_price = 0;
            short_fund = "";
            long_fund = "";
            open_position = false;
            prev_cash = cash;
            i_trades = i_trades + 1;
            
            if cash < max_drawdown
               max_drawdown = cash; 
            end
        end

disp("--------------------------------------");
disp("Final Cash-Out of the strategy : " + cash);
disp("ROI of the strategy : " + sum(trades_returns));
disp("ROI of the benchmark : " + benchmark);
disp("Average trade payoff : " + mean(trades_payoff));
disp("Average trade return : " + mean(trades_returns));
disp("Standard Error of trades payoffs : " + std(trades_payoff));
disp("Standard Error of trades returns : " + std(trades_returns));
disp("Worst trade payoff : " + min(trades_payoff));
disp("Worst trade returns : " + min(trades_returns));
disp("Best trade payoff : " + max(trades_payoff));
disp("Best trade returns : " + max(trades_returns));
disp("Max Drawdown of the strategy : " + max_drawdown);
disp("--------------------------------------");

% Allocation With One-Sample Stats Ranking
rf_rate = 0.001;
nb_max_traded_fund = 1;
nb_funds = size(fund_data, 2);
window = 30;
cash = 0;
prev_cash = 0;
open_position = false;
short_fund = "";
long_fund = "";
trades_payoff = zeros(round(size(fund_data, 1)/30), 1);
trades_returns = zeros(round(size(fund_data, 1)/30), 1);
position_price = 0;
position_payoff = 0;
max_drawdown = 0;
i_trades = 1;
for i_row=window:window:size(fund_data, 1)
    rolling_dataset = fund_data(i_row-window+1:i_row, :);
    fund_ranking = Rank_Funds(rolling_dataset, rf_rate);
    
    best_fund = fund_ranking(1,:);
    fund_to_buy = best_fund.Properties.RowNames;
    
    if open_position
        %cash = cash + table2array(fund_price(i_row + 1, long_fund));
        position_payoff = + table2array(fund_price(i_row + 1, long_fund));
        cash = cash + position_payoff;
        %trades_payoff(i_trades, 1) = cash - prev_cash;
        trades_payoff(i_trades, 1) = position_payoff - position_price;
        trades_returns(i_trades, 1) = trades_payoff(i_trades, 1) / price_fund_to_buy;
        position_payoff = 0;
        position_price = 0;
        long_fund = "";
        open_position = false;
        prev_cash = cash;
        i_trades = i_trades + 1;

        if cash < max_drawdown
           max_drawdown = cash; 
        end
    end

    price_fund_to_buy = table2array(fund_price(i_row + 1, fund_to_buy));

    %cash = cash - price_fund_to_buy;
    position_price = price_fund_to_buy;
    cash = cash + position_price;
    open_position = true;
    long_fund = fund_to_buy;
end

if open_position
    %cash = cash + table2array(fund_price(i_row + 1, long_fund));
    position_payoff = + table2array(fund_price(i_row + 1, long_fund));
    cash = cash + position_payoff;
    %trades_payoff(i_trades, 1) = cash - prev_cash;
    trades_payoff(i_trades, 1) = position_payoff - position_price;
    trades_returns(i_trades, 1) = trades_payoff(i_trades, 1) /price_fund_to_buy;
    position_payoff = 0;
    position_price = 0;
    long_fund = "";
    open_position = false;
    prev_cash = cash;
    i_trades = i_trades + 1;

    if cash < max_drawdown
       max_drawdown = cash; 
    end
end

disp("--------------------------------------");
disp("Final Cash-Out of the strategy : " + cash);
disp("ROI of the strategy : " + sum(trades_returns));
disp("ROI of the benchmark : " + benchmark);
disp("Average trade payoff : " + mean(trades_payoff));
disp("Average trade return : " + mean(trades_returns));
disp("Standard Error of trades payoffs : " + std(trades_payoff));
disp("Standard Error of trades returns : " + std(trades_returns));
disp("Worst trade payoff : " + min(trades_payoff));
disp("Worst trade returns : " + min(trades_returns));
disp("Best trade payoff : " + max(trades_payoff));
disp("Best trade returns : " + max(trades_returns));
disp("Max Drawdown of the strategy : " + max_drawdown);
disp("--------------------------------------");