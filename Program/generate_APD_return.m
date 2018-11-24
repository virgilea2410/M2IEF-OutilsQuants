function [simulated_returns] = generate_APD_return(alpha, lambda, sample_size, desired_correlation, desired_mean_1, desired_mean_2, normalize)

% To fit Normal Distributionn : alpha = 0.5 ; lambda = 2;
% To fit Laplaciann Distribution : alpha = 0.5 ; lambda = 1;
% To fit "real world" returns ditribution : alpha = 0.7 ; lambda = 1.35

empirical_corr = -2;

while empirical_corr > desired_correlation + 0.05 || empirical_corr < desired_correlation - 0.05

    random_proba_matrix = copularnd('gaussian', desired_correlation, sample_size);

    [real_mean, real_var, ~, ~] = APD_moment(alpha, lambda);

    simulated_returns = zeros(sample_size, 1);
    for i_sample=1:sample_size
       %random_proba = randi([0, 100])/100;
       random_proba1 = random_proba_matrix(i_sample, 1);
       random_proba2 = random_proba_matrix(i_sample, 2);
       delta = (2 * alpha^lambda * (1-alpha)^lambda) / (alpha^lambda + (1-alpha)^lambda);

       if random_proba1 <= alpha
           incomplete_gamma_function1 = gammaincinv(1-random_proba1/alpha, 1/lambda);
           APDinv1 = -(alpha^lambda/(delta*sqrt(lambda)))^(1/lambda) * incomplete_gamma_function1^(1/lambda);
       else
           incomplete_gamma_function1 = gammaincinv(1-(1-random_proba1)/(1-alpha), 1/lambda);
           APDinv1 = ((1-alpha)^lambda/(delta*sqrt(lambda)))^(1/lambda) * incomplete_gamma_function1^(1/lambda);
       end

       if random_proba2 <= alpha
           incomplete_gamma_function2 = gammaincinv(1-random_proba2/alpha, 1/lambda);
           APDinv2 = -(alpha^lambda/(delta*sqrt(lambda)))^(1/lambda) * incomplete_gamma_function2^(1/lambda);
       else
           incomplete_gamma_function2 = gammaincinv(1-(1-random_proba2)/(1-alpha), 1/lambda);
           APDinv2 = ((1-alpha)^lambda/(delta*sqrt(lambda)))^(1/lambda) * incomplete_gamma_function2^(1/lambda);
       end

       one_simulated_return1 = APDinv1;
       one_simulated_return2 = APDinv2;

       simulated_returns(i_sample, 1) = one_simulated_return1; 
       simulated_returns(i_sample, 2) = one_simulated_return2; 
    end
    
    if normalize
        simulated_returns(:, 1) = (simulated_returns(:, 1) - real_mean) / sqrt(real_var)  + desired_mean_1;
        simulated_returns(:, 2) = (simulated_returns(:, 2) - real_mean) / sqrt(real_var) + desired_mean_2;
    else
        simulated_returns(:, 1) = simulated_returns(:, 1) + desired_mean_1;
        simulated_returns(:, 2) = simulated_returns(:, 2) + desired_mean_2;
    end

    empirical_corr = corr(simulated_returns(:, 1), simulated_returns(:, 2));
end

end