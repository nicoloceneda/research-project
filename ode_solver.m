%% ODE for f(pi)

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

Gamma_D = 1 / r;
Gamma_pi = (theta2 - theta1) / (r * (r + p12 + p21));
Gamma_1 = theta1 / r^2 + (theta2 - theta1) * p12 / (r^2 * (r + p12 + p21));

% Model, range, initial condition

Y = nan(985,3);
i=1;

pi_range = linspace(0.005, 0.990, 985);
y0 = [-200 10];

for gamma = [1,2,4]
    model = @(pi, y) ode_f(pi, y, gamma, theta2, theta1, sigmaD, r, delta, p12, p21, pi2, Gamma_pi);
    [pi, y] = ode45(model, pi_range, y0);
    Y(1:size(y,1),i) = y(:,1);
    i=i+1;
end

% Plot

figure;
plot(pi_range, Y(:,1));
hold on;
plot(pi_range, Y(:,2));
plot(pi_range, Y(:,3));
hold off;
xlabel('\pi');
ylabel('f(\pi)');
grid on;

%% ODE for S(pi) and f(pi)

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

Gamma_D = 1 / r;
Gamma_pi = (theta2 - theta1) / (r * (r + p12 + p21));
Gamma_1 = theta1 / r^2 + (theta2 - theta1) * p12 / (r^2 * (r + p12 + p21));

% Model, range, initial condition

f = nan(985,3);
S = nan(985,3);
i=1;

pi_range = linspace(0.005, 0.990, 985);
y0 = [-200 10 -200 10];

for gamma = [1,2,4]
    model = @(pi, y) ode_Sf(pi, y, gamma, theta2, theta1, sigmaD, r, delta, p12, p21, pi2, Gamma_pi);
    [pi, y] = ode45(model, pi_range, y0);
    f(1:size(y,1),i) = y(:,1);
    S(1:size(y,1),i) = y(:,3);
    i=i+1;
end

% Plot

figure;
plot(pi_range, f(:,1));
hold on;
plot(pi_range, f(:,2));
plot(pi_range, f(:,3));
hold off;
xlabel('\pi');
ylabel('f(\pi)');
grid on;

figure;
plot(pi_range, S(:,1));
hold on;
plot(pi_range, S(:,2));
plot(pi_range, S(:,3));
hold off;
xlabel('\pi');
ylabel('S(\pi)');
grid on;

