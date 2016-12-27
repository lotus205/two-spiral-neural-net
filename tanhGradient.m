function g = tanhGradient(z)
%SIGMOIDGRADIENT returns the gradient of the tanh function
%evaluated at z
%   g = TANHGRADIENT(z) computes the gradient of the tanh function
%   evaluated at z. This should work regardless if z is a matrix or a
%   vector. In particular, if z is a vector or matrix, it should return
%   the gradient for each element.

g = zeros(size(z));
g = 1 - tanh(z).^2;

end

