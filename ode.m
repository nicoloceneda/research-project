%% Parameters

clear clc
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

% pi = 0

x2 = linspace(-2, 2);
x1 = p12 / r * x2;

% pi = small

[Q3_1, Q1_1, Q0_1] = coefficients_f(0.05, gamma, theta2, theta1, sigmaD, r, delta, p12, p21, pi2, Gamma_pi);
psi_1 = - Q3_1 / r * x2.^2 - Q1_1 / r * x2 - Q0_1 / r;

% pi = 0.5

[Q3_2, Q1_2, Q0_2] = coefficients_f(0.5, gamma, theta2, theta1, sigmaD, r, delta, p12, p21, pi2, Gamma_pi);
psi_2 = - Q3_2 / r * x2.^2 - Q1_2 / r * x2 - Q0_2 / r;

% Plot 

figure;
plot(x2, x1);
hold on;
plot(x2, psi_1);
plot(x2, psi_2);
hold off;
legend('x_1', '\psi(0.05)', '\psi(0.5)')
set(gca, 'XAxisLocation', 'origin', 'YAxisLocation', 'origin');

%% Phase diagram for S(pi)

% pi = 0

x4 = linspace(-2, 2);
x3 = p12 / r * x4;

% pi = small

[P3_1, P1_1, P0_1] = coefficients_S(0.05, gamma, theta2, theta1, sigmaD, r, delta, p12, p21, pi2, Gamma_pi);
psi_1 = - P1_1 / r * x4 - P0_1 / r;

% pi = 0.5

[P3_1, P1_1, P0_1] = coefficients_S(0.5, gamma, theta2, theta1, sigmaD, r, delta, p12, p21, pi2, Gamma_pi);
psi_2 = - P1_1 / r * x4 - P0_1 / r;

% Plot 

figure;
plot(x4, x3);
hold on;
plot(x4, psi_1);
plot(x4, psi_2);
hold off;
legend('x_4', '\psi(0.05)', '\psi(0.5)')
set(gca, 'XAxisLocation', 'origin', 'YAxisLocation', 'origin');

%% Finding y2_hat

% Numerical solution to limit and parabola

syms x2
eq_left = p12 / r * x2;
eq_right = - Q3_2 / r * x2.^2 - Q1_2 / r * x2 - Q0_2 / r;
sol = vpasolve(eq_left == eq_right, x2);

% y2_epsilon_str in [y2_hat, 0]

y2_hat = sol(2);
y2_epsilon_str = (y2_hat + 0) / 2;

%% Model

n = 994;
f = nan(n,3);
S = nan(n,3);

epsilon = 0.001;

pi_range = linspace(epsilon, 0.985, n);

% Gamma = 1

gamma = 1;

y0 = [0 -300 -105 -300]; % [-0.1 -1 -105 -5];

model = @(pi, y) ode_Sf(pi, y, gamma, theta2, theta1, sigmaD, r, delta, p12, p21, pi2, Gamma_pi);
[pi, y] = ode15s(model, pi_range, y0);
f(1:size(y,1),1) = y(:,1);
S(1:size(y,1),1) = y(:,3);

% Gamma = 2

gamma = 2;

y0 = [-0.0001 -300 -105 -300]; % [-0.1 -1 -105 -5];

model = @(pi, y) ode_Sf(pi, y, gamma, theta2, theta1, sigmaD, r, delta, p12, p21, pi2, Gamma_pi);
[pi, y] = ode15s(model, pi_range, y0);
f(1:size(y,1),2) = y(:,1);
S(1:size(y,1),2) = y(:,3);

% Gamma = 4

gamma = 4;

y2_epsilon = y2_epsilon_str;
y1_epsilon = y2_epsilon_str * p12 / r;

y0 = [-2.0775357902067921536254859771245 -0.085178967398478478298644925062104 -105 -30]; % [-0.015 -300 -105 -30]; %[y1_epsilon y2_epsilon -105 -30]; %

model = @(pi, y) ode_Sf(pi, y, gamma, theta2, theta1, sigmaD, r, delta, p12, p21, pi2, Gamma_pi);
[pi, y] = ode15s(model, pi_range, y0);
f(1:size(y,1),3) = y(:,1);
S(1:size(y,1),3) = y(:,3);

% Plot

figure;
plot(pi_range, f(:,1));
hold on;
plot(pi_range, f(:,2));
plot(pi_range, f(:,3));
hold off;
legend('\gamma=1', '\gamma=2', '\gamma=4')
xlabel('\pi');
ylabel('f(\pi)');
grid on;

figure;
plot(pi_range, S(:,1));
hold on;
plot(pi_range, S(:,2));
plot(pi_range, S(:,3));
hold off;
legend('\gamma=1', '\gamma=2', '\gamma=4')
xlabel('\pi');
ylabel('S(\pi)');
grid on;