
% Auto Correntropy
function z = ACm_prime (x, y, m, sigma)
	N = length(x);
    
    S = 0;
    c = 0;
    for n = m:N
        if (n < 1 || n-m+1 > N)
            continue;
        end
        
        S = S + Gaussian_prime (x(n), y(n - m + 1), sigma);
        c = c + 1;
    end

    z = (1/c) * S;
end

function z = Gaussian_prime(x, y, sigma)
    z = ( -(x-y) / ((sigma^3)*sqrt(2*pi)) ) * exp((-(x-y).^2) / (2*sigma^2));
end