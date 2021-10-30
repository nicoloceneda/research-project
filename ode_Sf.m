function dy = ode_Sf(pi, y, gamma, theta2, theta1, sigmaD, r, delta, p12, p21, pi2, Gamma_pi)
    
    h = (theta2 - theta1) / sigmaD * pi * (1 - pi);

    Q3 = h^2 / 2;
    Q1 = gamma * sigmaD * h + r * gamma * Gamma_pi * h^2 - (p12 + p21) * (pi2 - pi);
    Q0 = (gamma * r)^2 / 2 * Gamma_pi^2 * h^2 + gamma^2 * r * Gamma_pi * sigmaD * h + r * log(delta);

    P3 = Q3;
    P1 = gamma * r * Gamma_pi * h^2 + gamma * sigmaD * h - (p12 + p21) * (pi2 - pi) + y(2) * h^2;
    P0 = gamma * r * Gamma_pi^2 * h^2 + 2 * gamma * Gamma_pi * sigmaD * h + y(2) * Gamma_pi * h^2 + y(2) / r * sigmaD * h;

    dy = [y(2)
          y(2)^2 + y(2) * Q1 / Q3 + y(1) * r / Q3 + Q0 / Q3
          y(4)
          y(4) * P1 / P3 + y(3) * r / P3 + P0 / P3];
end