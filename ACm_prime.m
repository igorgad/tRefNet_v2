
% Auto Correntropy
function z = ACm_prime (x, y, m, sigma)
	N = length(x);
    
    S = 0;
    c = 0;
    
    li = max(1,m);
    if m <= 0
        lm = N+m-1;
    else
        lm = N;
    end
    
    for n = li:lm
        
        S = S + Gaussian_prime (x(n), y(n - m + 1), sigma);
        c = c + 1;
    end

    z = (1/c) * S;
end

function z = Gaussian_prime(x, y, sigma)
    z = ( -(x-y) / ((sigma^3)*sqrt(2*pi)) ) * exp((-(x-y).^2) / (2*sigma^2));
end