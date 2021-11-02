function [P3, P1, P0] = coefficients_S(pi, f_pi, gamma, theta2, theta1, sigmaD, r, p12, p21, pi2, Gamma_pi)
    
    h = (theta2 - theta1) / sigmaD * pi * (1 - pi);

    P3 = h^2 / 2;
    P1 = gamma * r * Gamma_pi * h^2 + gamma * sigmaD * h - (p12 + p21) * (pi2 - pi) + f_pi * h^2;
    P0 = gamma * r * Gamma_pi^2 * h^2 + 2 * gamma * Gamma_pi * sigmaD * h + f_pi * Gamma_pi * h^2 + f_pi / r * sigmaD * h;

end