function [mean, var, skew, kurt] = APD_moment(alpha, lambda)

mean = 0;
var = 0;
skew = 0;
kurt = 0;

original_alpha = alpha;
original_lambda = lambda;
do_interpolate = 0;

lambdas = [1, 1.25, 1.5, 1.75, 2];
alphas = [0.1, 0.3, 0.5, 0.7, 0.9];

[nearest_lambdas, index_lambda] = min(abs(lambda - lambdas));
[nearest_alphas, index_alpha] = min(abs(alpha - alphas));
alpha = alphas(index_alpha);
lambda = lambdas(index_lambda);

if alpha ~= original_alpha || lambda ~= original_lambda
    do_interpolate = 1;
end

delta = (2 * original_alpha^original_lambda * (1-original_alpha)^original_lambda) / ...
    (original_alpha^original_lambda + (1-original_alpha)^original_lambda);
mean = gamma(2/original_lambda)/gamma(1/original_lambda) * (1-2*alpha) * delta^(-1/original_lambda);
var = (gamma(3/original_lambda)*gamma(1/original_lambda)*(1-3*alpha+3*alpha^2) ...
    - gamma(2/original_lambda)^2 * (1-2*alpha)^2) ...
    /(gamma(1/original_lambda)^2) ...
    * delta^(-2/lambda);

if ~do_interpolate
    if alpha == 0.1 || alpha == 0.9
        if lambda == 1
            skew = -2.2311 * (alpha == 0.9) + 2.2311 * (alpha == 0.1);
            kurt = 6.6485;
        elseif lambda == 1.25
            skew = -1.9870 * (alpha == 0.9) + 1.9870 * (alpha == 0.1);
            kurt = 5.0165;
        elseif lambda == 1.5
            skew = -1.8415 * (alpha == 0.9) + 1.8415 * (alpha == 0.1);
            kurt = 4.1686;
        elseif lambda == 1.75
            skew = -1.7457 * (alpha == 0.9) + 1.7457 * (alpha == 0.1);
            kurt = 3.6595;
        elseif lambda == 2
            skew = -1.6784 * (alpha == 0.9) + 1.6784 * (alpha == 0.1);
            kurt = 3.3243;
        end
    elseif alpha == 0.3 || alpha == 0.7 
        if lambda == 1
            skew = -2.1867 * (alpha == 0.7) + 2.1867 * (alpha == 0.3);
            kurt = 7.4726;
        elseif lambda == 1.25
            skew = -1.9474 * (alpha == 0.7) + 1.9474 * (alpha == 0.3);
            kurt = 5.6383;
        elseif lambda == 1.5
            skew = -1.8048 * (alpha == 0.7) + 1.8048 * (alpha == 0.3);
            kurt = 4.6853;
        elseif lambda == 1.75
            skew = -1.7109 * (alpha == 0.7) + 1.7109 * (alpha == 0.3);
            kurt = 4.1131;
        elseif lambda == 2
            skew = -1.6450 * (alpha == 0.7) + 1.6450 * (alpha == 0.3);
            kurt = 3.7363;
        end
    elseif alpha == 0.5
        if lambda == 1
            skew = 0;
            kurt = 6;
        elseif lambda == 1.25
            skew = 0;
            kurt = 4.5272;
        elseif lambda == 1.5
            skew = 0;
            kurt = 3.7620;
        elseif lambda == 1.75
            skew = 0;
            kurt = 3.3026;
        elseif lambda == 2
            skew = 0;
            kurt = 3;
        end
    end
else
    [X_alphas, Y_lambdas] = meshgrid(alphas, lambdas);
    skew_values = [2.2311, 1.9870, 1.8415, 1.7457, 1.6784; ...
                    2.1867, 1.9474, 1.8048, 1.7109, 1.6450; ...
                    0, 0, 0, 0, 0; ...
                    -2.1867, -1.9474, -1.8048, -1.7109, -1.6450; ...
                    -2.2311, -1.9870, -1.8415, -1.7457, -1.6784]';
    kurt_values = [6.6485, 5.0165, 4.1686, 3.6595, 3.3243; ...
                    7.4726, 5.6383, 4.6853, 4.1131, 3.7363; ...
                    6, 4.5272, 3.7620, 3.3026, 3; ...
                    7.4726, 5.6383, 4.6853, 4.1131, 3.7363; ...
                    6.6485, 5.0165, 4.1686, 3.6595, 3.3243]';
    skew = interp2(X_alphas, Y_lambdas, skew_values, original_alpha, original_lambda);
    kurt = interp2(X_alphas, Y_lambdas, kurt_values, original_alpha, original_lambda);
end

end