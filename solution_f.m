%% Parameters

clear;
clc;
warning ('off','all');

% Parameters

theta2 = 0.0025;
theta1 = - 0.0150;
sigmaD = 0.0240;
r = 0.0041;
delta = 1;
p12 = 0.1000;
p21 = 0.0167;

pi2 = p12 / (p12 + p21);
Gamma_pi = (theta2 - theta1) / (r * (r + p12 + p21));

%% Phase diagram for f(pi)

% Set gamma to draw the phase diagram
gamma = 4;

% x2
x2 = linspace(-2, 2);

% x1 for pi = 0.00
psi_0 = p12 / r * x2;

% x1 for pi = 0.05
[Q3, Q1, Q0] = coefficients_f(0.05, gamma, theta2, theta1, sigmaD, r, delta, p12, p21, pi2, Gamma_pi);
psi_1 = - Q3 / r * x2.^2 - Q1 / r * x2 - Q0 / r;

% x1 for pi = 0.50
[Q3, Q1, Q0] = coefficients_f(0.5, gamma, theta2, theta1, sigmaD, r, delta, p12, p21, pi2, Gamma_pi);
psi_2 = - Q3 / r * x2.^2 - Q1 / r * x2 - Q0 / r;

% x1 for pi = 1.00
psi_3 = - p21 / r * x2;

% Plot 
figure;
plot(x2, psi_0);
hold on;
plot(x2, psi_1);
plot(x2, psi_2);
plot(x2, psi_3);
hold off;
xlabel('x_2=f^\prime(\pi)')
ylabel('x_1=f(\pi)')
legend('\pi=0.00', '\pi=0.05', '\pi=0.50', '\pi=1.00', 'Location', 'northwest');
title('Phase diagram for f(\pi)');
set(gca, 'XAxisLocation', 'origin', 'YAxisLocation', 'origin');
grid on;

% Clear the value of gamma
clear gamma

%% Solution of the ODE of f(pi)

% Pi range
eps = 0.001;
pi_f = 0.950;
n = (pi_f - eps) * 1000;
pi_range = linspace(eps, pi_f, n);

% Store numerical solutions
f = nan(n,3);
f_pr = nan(n,3);

%Loop over all values of gamma
i = 1;

for gamma = [1,2,4]
    
    % Numerical solution for x2_hat
    [Q3, Q1, Q0] = coefficients_f(0.5, gamma, theta2, theta1, sigmaD, r, delta, p12, p21, pi2, Gamma_pi);
    syms x2
    eq_1 = p12 / r * x2;
    eq_2 = - Q3 / r * x2.^2 - Q1 / r * x2 - Q0 / r;
    intersect = vpasolve(eq_1 == eq_2, x2);

    % Value of x2_eps_str in [x2_hat, 0]
    x2_hat = intersect(2);
    x2_eps_str = (x2_hat + 0) / 2;

    % Initial condition
    x1_init = x2_eps_str * p12 / r;
    x2_init = x2_eps_str;
    x0 = [double(x1_init) double(x2_init)];

    % Model
    model = @(pi, x) ode_f(pi, x, gamma, theta2, theta1, sigmaD, r, delta, p12, p21, pi2, Gamma_pi);
    
    % Solution
    [pi, x] = ode15s(model, pi_range, x0);
    f(1:size(x,1),i) = x(:,1);
    f_pr(1:size(x,1),i) = x(:,2);
    i=i+1;

end

%% Plot solutions

% Normalize to the same starting value
f_plot = nan(size(f,1),size(f,2));
f_plot(:,1) = f(:,1) - f(1,1) + f(1,3);
f_plot(:,2) = f(:,2) - f(1,2) + f(1,3);
f_plot(:,3) = f(:,3);

% Plot the figure
figure;
plot(pi_range, f_plot(:,1));
hold on;
plot(pi_range, f_plot(:,2));
plot(pi_range, f_plot(:,3));
hold off;
xlabel('\pi')
ylabel('f(\pi)')
legend('\gamma=1', '\gamma=2', '\gamma=4', 'Location', 'northwest');
title('Solutions to the ODE of f(\pi)');
grid on;