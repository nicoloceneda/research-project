function dx = ode_f(pi, x, gamma, theta2, theta1, sigmaD, r, delta, p12, p21, pi2, Gamma_pi)
    
    [Q3, Q1, Q0] = coefficients_f(pi, gamma, theta2, theta1, sigmaD, r, delta, p12, p21, pi2, Gamma_pi);

    dx = [x(2)
          x(2)^2 + x(2) * Q1 / Q3 + x(1) * r / Q3 + Q0 / Q3];
end