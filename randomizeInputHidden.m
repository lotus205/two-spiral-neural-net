function W = randomizeInputHidden( L_in, L_out )
%RANDINITIALIZEWEIGHTS Randomly initialize the weights of a layer with L_in
%incoming connections and L_out outgoing connections

% W = ones(L_out, 1 + L_in);
epsilon = 0.12;
 W = 7 + rand(L_out, 1 + L_in) * 2 * epsilon - epsilon;


end

