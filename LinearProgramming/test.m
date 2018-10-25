m = 6;
n = 4;

count = 0;
tests = 10000;

tic
for i = 1:tests
    A = randn(m,n);
    b = randn(m,1);
    c = randn(n, 1);
    
    cvx_begin quiet
        variable x(n)
        minimize(c'*x);
        subject to
            A*x <= b;
            x >= 0;
    cvx_end
    if ~isnan(x) & ~isinf(cvx_optval)
        count = count + 1;
    end
end
toc
