function [descriptive_stats, sharpe_stats, one_sample_test_stats, ...
    two_sample_test_stats, test_results] ...
    = EstimateSharpes(returns_a, returns_b, tested_sharpe_a, tested_sharpe_b, rf_rate)

sample_size = size(returns_a, 1);

% One-Sample Tests (For Returns A & B)
est_mean_a = mean(returns_a);
est_vol_a = std(returns_a);
est_skew_a = skewness(returns_a);
est_kurto_a = kurtosis(returns_a);
est_median_a = median(returns_a);

est_mean_b = mean(returns_b);
est_vol_b = std(returns_b);
est_skew_b = skewness(returns_b);
est_kurto_b = kurtosis(returns_b);
est_median_b = median(returns_b);

est_sharpe_bias_a = (1 + 1/4 * (est_kurto_a-1)/sample_size);
est_sharpe_biased_a = (est_mean_a-rf_rate)/est_vol_a;
est_sharpe_unbiased_a = est_sharpe_biased_a/est_sharpe_bias_a;
std_est_sharpe_biased_a = sqrt((1+est_sharpe_biased_a^2/4*(est_kurto_a-1)-est_sharpe_biased_a*(est_skew_a))/(sample_size-1));
std_est_sharpe_unbiased_a = sqrt((1+est_sharpe_unbiased_a^2/4*(est_kurto_a-1)-est_sharpe_unbiased_a*(est_skew_a))/(sample_size-1));

est_sharpe_bias_b = (1 + 1/4 * (est_kurto_b-1)/sample_size);
est_sharpe_biased_b = (est_mean_b-rf_rate)/est_vol_b;
est_sharpe_unbiased_b = est_sharpe_biased_b/est_sharpe_bias_b;
std_est_sharpe_biased_b = sqrt((1+est_sharpe_biased_b^2/4*(est_kurto_b-1)-est_sharpe_biased_b*(est_skew_b))/(sample_size-1));
std_est_sharpe_unbiased_b = sqrt((1+est_sharpe_unbiased_b^2/4*(est_kurto_b-1)-est_sharpe_unbiased_b*(est_skew_b))/(sample_size-1));

one_sample_stat_biased_a = (est_sharpe_biased_a-tested_sharpe_a)/std_est_sharpe_biased_a;
one_sample_stat_unbiased_a = (est_sharpe_unbiased_a-tested_sharpe_a)/std_est_sharpe_unbiased_a;
% Test 05/06
%one_sample_stat_biased_a = (est_sharpe_biased_a-tested_sharpe_a)/std_est_sharpe_biased_a;
%one_sample_stat_unbiased_a = (est_sharpe_unbiased_a-tested_sharpe_a)/std_est_sharpe_unbiased_a;

confidence_boundary_1T_biased_a = norminv(0.95) * std_est_sharpe_biased_a;
confidence_boundary_1T_unbiased_a = norminv(0.95) * std_est_sharpe_unbiased_a;
confidence_boundary_2T_biased_a = norminv(0.975) * std_est_sharpe_biased_a;
confidence_boundary_2T_unbiased_a = norminv(0.975) * std_est_sharpe_unbiased_a;

one_sample_stat_biased_b = (est_sharpe_biased_b-tested_sharpe_a)/std_est_sharpe_biased_b;
one_sample_stat_unbiased_b = (est_sharpe_unbiased_b-tested_sharpe_a)/std_est_sharpe_unbiased_b;
confidence_boundary_1T_biased_b = norminv(0.95) * std_est_sharpe_biased_b;
confidence_boundary_1T_unbiased_b = norminv(0.95) * std_est_sharpe_unbiased_b;
confidence_boundary_2T_biased_b = norminv(0.975) * std_est_sharpe_biased_b;
confidence_boundary_2T_unbiased_b = norminv(0.975) * std_est_sharpe_unbiased_b;

% Two-Sample Tests
est_sharpe_biased_diff = est_sharpe_biased_a - est_sharpe_biased_b;
est_sharpe_unbiased_diff = est_sharpe_unbiased_a - est_sharpe_unbiased_b;
s_0a_1b = sum(returns_a.^0.*returns_b.^1);
s_1a_0b = sum(returns_a.^1.*returns_b.^0);
s_1a_1b = sum(returns_a.^1.*returns_b.^1);

s_0a_2b = sum(returns_a.^0.*returns_b.^2);
s_2a_0b = sum(returns_a.^2.*returns_b.^0);
s_1a_2b = sum(returns_a.^1.*returns_b.^2);
s_2a_1b = sum(returns_a.^2.*returns_b.^1);
s_2a_2b = sum(returns_a.^2.*returns_b.^2);

s_0b_1a = sum(returns_b.^0.*returns_a.^1);
s_1b_0a = sum(returns_b.^1.*returns_a.^0);
s_1b_1a = sum(returns_b.^1.*returns_a.^1);

s_0b_2a = sum(returns_b.^0.*returns_a.^2);
s_2b_0a = sum(returns_b.^2.*returns_a.^0);
s_1b_2a = sum(returns_b.^1.*returns_a.^2);
s_2b_1a = sum(returns_b.^2.*returns_a.^1);
s_2b_2a = sum(returns_b.^2.*returns_a.^2);

mu_1a_2b = (2 * s_0a_1b^2 * s_1a_0b ...
    - sample_size * s_0a_2b * s_1a_0b ...
    - 2 * s_0a_1b * s_1a_1b ...
    + sample_size^2 * s_1a_2b) ...
    / (sample_size * (sample_size-1) * (sample_size-2));
mu_1b_2a = (2 * s_0b_1a^2 * s_1b_0a ...
    - sample_size * s_0b_2a * s_1b_0a ...
    - 2 * s_0b_1a * s_1b_1a ...
    + sample_size^2 * s_1b_2a) ...
    / (sample_size * (sample_size-1) * (sample_size-2));
mu_2a_2b = (-3 * s_0a_1b^2 * s_1a_0b^2 ...
    + sample_size * s_0a_2b * s_1a_0b^2 ...
    + 4 * sample_size * s_0a_1b * s_1a_0b * s_1a_1b ...
    - 2 * (2*sample_size - 3) * s_1a_1b^2 ...
    - 2 * (sample_size^2 - 2*sample_size + 3) * s_1a_0b * s_1a_2b ...
    + s_0a_1b^2 * s_2a_0b ...
    - (2*sample_size -3) * s_0a_2b * s_2a_0b ...
    - 2 * (sample_size^2 - 2*sample_size + 3) * s_0a_1b * s_2a_1b ...
    + sample_size * (sample_size^2 - 2*sample_size + 3) * s_2a_2b) ... 
    / ((sample_size-3)*(sample_size-2)*(sample_size-1)*sample_size);

std_est_sharpe_biased_diff = 1 + est_sharpe_biased_a^2/4 * (est_kurto_a-1) ...
    - est_sharpe_biased_a * est_skew_a ...
    + 1 + est_sharpe_biased_b^2/4 * (est_kurto_b-1) ...
    - est_sharpe_biased_b * est_skew_b ...
    - 2 * (corr(returns_a, returns_b) ...
    + est_sharpe_biased_a*est_sharpe_biased_b/4*(mu_2a_2b/(est_vol_a^2*est_vol_b^2) - 1) ...
    - 1/2 * est_sharpe_biased_a * mu_1b_2a/(est_vol_b*est_vol_a^2) ...
    - 1/2 * est_sharpe_biased_b * mu_1a_2b/(est_vol_a*est_vol_b^2));
std_est_sharpe_biased_diff = sqrt(std_est_sharpe_biased_diff/(sample_size-1));
std_est_sharpe_unbiased_diff = 1 + est_sharpe_unbiased_a^2/4 * (est_kurto_a-1) ...
    - est_sharpe_unbiased_a * est_skew_a ...
    + 1 + est_sharpe_unbiased_b^2/4 * (est_kurto_b-1) ...
    - est_sharpe_unbiased_b * est_skew_b ...
    - 2 * (corr(returns_a, returns_b) ...
    + est_sharpe_unbiased_a*est_sharpe_unbiased_b/4*(mu_2a_2b/(est_vol_a^2*est_vol_b^2) - 1) ...
    - 1/2 * est_sharpe_unbiased_a * mu_1b_2a/(est_vol_b*est_vol_a^2) ...
    - 1/2 * est_sharpe_unbiased_b * mu_1a_2b/(est_vol_a*est_vol_b^2));
std_est_sharpe_unbiased_diff = sqrt(std_est_sharpe_unbiased_diff/(sample_size-1));

two_sample_stat_biased = (est_sharpe_biased_diff - (tested_sharpe_a - tested_sharpe_b)) / std_est_sharpe_biased_diff;
two_sample_stat_unbiased = (est_sharpe_unbiased_diff - (tested_sharpe_a - tested_sharpe_b)) / std_est_sharpe_unbiased_diff;
confidence_boundary_1T_biased_diff = norminv(0.95) * std_est_sharpe_biased_diff;
confidence_boundary_1T_unbiased_diff = norminv(0.95) * std_est_sharpe_unbiased_diff;
confidence_boundary_2T_biased_diff = norminv(0.975) * std_est_sharpe_biased_diff;
confidence_boundary_2T_unbiased_diff = norminv(0.975) * std_est_sharpe_unbiased_diff;

% Hypothesis Tests
one_sample_LT_pvalue_biased_a = normcdf(one_sample_stat_biased_a, 0, 1);
one_sample_LT_pvalue_unbiased_a = normcdf(one_sample_stat_unbiased_a, 0, 1);
one_sample_RT_pvalue_biased_a = 1-normcdf(one_sample_stat_biased_a, 0, 1);
one_sample_RT_pvalue_unbiased_a = 1-normcdf(one_sample_stat_unbiased_a, 0, 1);
one_sample_2T_pvalue_biased_a = 2* (1-normcdf(abs(one_sample_stat_biased_a), 0, 1));
one_sample_2T_pvalue_unbiased_a = 2* (1-normcdf(abs(one_sample_stat_unbiased_a), 0,1));

one_sample_LT_pvalue_biased_b = normcdf(one_sample_stat_biased_b, 0, 1);
one_sample_LT_pvalue_unbiased_b = normcdf(one_sample_stat_unbiased_b, 0, 1);
one_sample_RT_pvalue_biased_b = 1-normcdf(one_sample_stat_biased_b, 0, 1);
one_sample_RT_pvalue_unbiased_b = 1-normcdf(one_sample_stat_unbiased_b, 0, 1);
one_sample_2T_pvalue_biased_b = 2* (1-normcdf(abs(one_sample_stat_biased_b), 0, 1));
one_sample_2T_pvalue_unbiased_b = 2* (1-normcdf(abs(one_sample_stat_unbiased_b), 0,1));

if two_sample_stat_biased ~= 0 && imag(two_sample_stat_biased) == 0
    two_sample_LT_pvalue_biased = normcdf(two_sample_stat_biased, 0, 1);
    two_sample_LT_pvalue_unbiased = normcdf(two_sample_stat_unbiased, 0, 1);
    two_sample_RT_pvalue_biased = 1-normcdf(two_sample_stat_biased, 0, 1);
    two_sample_RT_pvalue_unbiased = 1-normcdf(two_sample_stat_unbiased, 0, 1);
    two_sample_2T_pvalue_biased = 2* (1-normcdf(abs(two_sample_stat_biased), 0, 1));
    two_sample_2T_pvalue_unbiased = 2* (1-normcdf(abs(two_sample_stat_unbiased), 0,1));
else
    two_sample_LT_pvalue_biased = -1;
    two_sample_LT_pvalue_unbiased = -1;
    two_sample_RT_pvalue_biased = -1;
    two_sample_RT_pvalue_unbiased = -1;
    two_sample_2T_pvalue_biased = -1;
    two_sample_2T_pvalue_unbiased = -1;
end
    
% Results Outputs
descriptive_stats_a = [est_mean_a ...
                        ; est_vol_a ...
                        ; est_skew_a ...
                        ; est_kurto_a ...
                        ; est_median_a];
descriptive_stats_b = [est_mean_b ...
                        ; est_vol_b ...
                        ; est_skew_b ...
                        ; est_kurto_b; ...
                        est_median_b];
descriptive_stats = [descriptive_stats_a, descriptive_stats_b];

sharpe_stats_a = [est_sharpe_bias_a ...
                    ; est_sharpe_biased_a ...
                    ; est_sharpe_unbiased_a ...
                    ; std_est_sharpe_biased_a ...
                    ; std_est_sharpe_unbiased_a];
sharpe_stats_b = [est_sharpe_bias_b ...
                    ; est_sharpe_biased_b ...
                    ; est_sharpe_unbiased_b ...
                    ; std_est_sharpe_biased_b ...
                    ; std_est_sharpe_unbiased_b];
sharpe_stats = [sharpe_stats_a, sharpe_stats_b];

one_sample_test_stats_a = [one_sample_stat_biased_a ...
                            ; one_sample_stat_unbiased_a ...
                            ; confidence_boundary_2T_biased_a ...
                            ; confidence_boundary_2T_unbiased_a ...
                            ; confidence_boundary_1T_biased_a ...
                            ; confidence_boundary_1T_unbiased_a];
one_sample_test_stats_b = [one_sample_stat_biased_b ...
                            ; one_sample_stat_unbiased_b ...
                            ; confidence_boundary_2T_biased_b ...
                            ; confidence_boundary_2T_unbiased_b ...
                            ; confidence_boundary_1T_biased_b ...
                            ; confidence_boundary_1T_unbiased_b];
one_sample_test_stats = [one_sample_test_stats_a, one_sample_test_stats_b];

two_sample_test_stats = [est_sharpe_biased_diff ...
                        ; est_sharpe_unbiased_diff ...
                        ; std_est_sharpe_biased_diff ...
                        ; std_est_sharpe_unbiased_diff ...
                        ; two_sample_stat_biased ...
                        ; two_sample_stat_unbiased ...
                        ; confidence_boundary_2T_biased_diff ...
                        ; confidence_boundary_2T_unbiased_diff];

test_results_a = [one_sample_LT_pvalue_biased_a; ...
    one_sample_LT_pvalue_unbiased_a; ...
    one_sample_RT_pvalue_biased_a; ...
    one_sample_RT_pvalue_unbiased_a; ...
    one_sample_2T_pvalue_biased_a; ...
    one_sample_2T_pvalue_unbiased_a];
test_results_b = [one_sample_LT_pvalue_biased_b; ...
    one_sample_LT_pvalue_unbiased_b; ...s
    one_sample_RT_pvalue_biased_b; ...
    one_sample_RT_pvalue_unbiased_b; ...
    one_sample_2T_pvalue_biased_b; ...
    one_sample_2T_pvalue_unbiased_b];
test_results_diff = [two_sample_LT_pvalue_biased; ...
    two_sample_LT_pvalue_unbiased; ...
    two_sample_RT_pvalue_biased; ...
    two_sample_RT_pvalue_unbiased; ...
    two_sample_2T_pvalue_biased; ...
    two_sample_2T_pvalue_unbiased];
test_results = [test_results_a, test_results_b, test_results_diff];


end