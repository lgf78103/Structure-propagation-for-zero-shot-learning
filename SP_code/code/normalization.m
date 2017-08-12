function [Xn, m, v, mx] = normalization(X, m, v, mx)

n = size(X, 1);

if(nargin == 1)
	X1 = X;
	m = sum(X1) / n;
	X1 = X1 - repmat(m,[n 1]);
	v = sqrt(sum(X1.^2) / n);	
	v(1) = 1;
	X1 = X1 ./ repmat(v,[n 1]);
	mx = max(max(X1));
end

Xn = (X - repmat(m,[n 1])) ./ repmat(v,[n 1]);
Xn = Xn / mx;

