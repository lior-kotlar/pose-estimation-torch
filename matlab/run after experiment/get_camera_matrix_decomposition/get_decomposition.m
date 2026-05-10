M = [[ 9.68045143e+03 -1.43381171e+04  1.68684775e+04  6.94329642e+02]; 
    [ 1.14436022e+04 -1.26282807e+04 -1.69192172e+04  3.55061841e+02]; 
    [-3.19872457e+00 -3.41590913e+00 -1.57985145e-01  1.00000000e+00]]

[K, Rc_w, Pc, pp, pv] = DecomposeCamera(M);

[xyz,T,ypr,Uo,Vo,Z, T3] = DLTcameraPosition(M(1:end-1));


pinvM = pinv(M);
K = K / K(end);
M_prime = K * [Rc_w, pv];

t_prime = M_prime(:, end);
t_prime = t_prime(1:end-1) / t_prime(end);
