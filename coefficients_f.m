function [Q3, Q1, Q0] = coefficients_f(pi, gamma, theta2, theta1, sigmaD, r, delta, p12, p21, pi2, Gamma_pi)
    
    h = (theta2 - theta1) / sigmaD * pi * (1 - pi);
    
    Q3 = h^2 / 2;
    Q1 = gamma * sigmaD * h + r * gamma * Gamma_pi * h^2 - (p12 + p21) * (pi2 - pi);
    Q0 = (gamma * r)^2 / 2 * Gamma_pi^2 * h^2 + gamma^2 * r * Gamma_pi * sigmaD * h + r * log(delta);

end