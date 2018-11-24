function [simulated_returns] = generate_normal_return(mu, sigma, sample_size, desired_correlation)
%Function simulating two normally distributed vector of random variables
% - mu : Mean of the distribution
% - sigma : Standard Error of the distribution 
% - sample_size : number of observations per vector
% - desired_correlation : desired level of correlation betwen the two
%                           vector of random variables

empirical_corr = -2;

while empirical_corr > desired_correlation + 0.05 || empirical_corr < desired_correlation - 0.05
    random_proba_matrix = copularnd('gaussian', desired_correlation, sample_size);

    simulated_returns = zeros(sample_size, 2);
    for i_sample=1:sample_size
       %random_proba = randi([0, 100])/100;
       random_proba1 = random_proba_matrix(i_sample, 1);
       random_proba2 = random_proba_matrix(i_sample, 2);
       one_simulated_return1 = norminv(random_proba1, mu, sigma);
       one_simulated_return2 = norminv(random_proba2, mu, sigma);
       simulated_returns(i_sample, 1) = one_simulated_return1;
       simulated_returns(i_sample, 2) = one_simulated_return2;
    end

    empirical_corr = corr(simulated_returns(:, 1), simulated_returns(:, 2));
end

end