%% MTE544 Assignment 1
%  Script for part 1 calculations
%
%  Multiply the transformation matrices to compute the transform 
%  from frame E to frame s
%
%   G_E2 : E -> 2
%   G_21 : 2 -> 1
%   G_1B : 1 -> B
%   G_Bs : B -> s

clear; clc;

%% System Variables
syms gamma theta alpha l2 l1 lB x y;

%% 1a)Matrices
%
G_E2 = [0 1 l2; -1 0 0; 0 0 1];
G_21 = [cos(gamma) -sin(gamma) l1; sin(gamma) cos(gamma) 0; 0 0 1];
G_1B = [cos(alpha) -sin(alpha) lB; sin(alpha) cos(alpha) 0; 0 0 1];
G_Bs = [cos(theta) -sin(theta) x; sin(theta) cos(theta) y; 0 0 1];

%% --- Compose the Transformation Matrix E -> s ---
G_Es = G_Bs * G_1B * G_21 * G_E2;
disp('Composed transformation matrix G_Es (E to s):');
disp(G_Es);

%% --- Simplify and display ---
G_Es_simp = simplify(G_Es);
disp('Simplified transformation matrix G_Es (E to s):');
disp(G_Es_simp);

%% 1b) Point Transformation
% Points of interest
P_E1 = [0;0;1];
P_E2 = [0.5;1;1];
% Assign values to variables
theta_val = -pi/2; %(rad)
alpha_val = pi/6; %(rad)
gamma_val = pi/4; %(rad)
l1_val=1; % (m)
l2_val=1; % (m)
lB_val=0.5; % (m)
x_val=0.8; % (m)
y_val=0.8; % (m)
% Swap symbolic variabels for values
G_Es_num = subs(G_Es_simp, ...
    [theta, gamma, alpha, l1, l2, lB, x, y], ...
    [theta_val, gamma_val, alpha_val, l1_val, l2_val, lB_val, x_val, y_val]);

% Calculate points 1 & 2 in frame s
P_s1 = double(G_Es_num * P_E1)
P_s2 = double(G_Es_num * P_E2)

%% 1c) Inverse Matrices for Reverse Transformation
q_B = [1;2;1];

% Find transformation matrix inverse
G_E2_inv = simplify(inv(G_E2));
G_21_inv = simplify(inv(G_21));
G_1B_inv = simplify(inv(G_1B));

% Transformation matrix for B to E
G_BE = G_E2_inv * G_21_inv * G_1B_inv;
G_BE_simp = simplify(G_BE);
display(G_BE_simp);

% Numerical G_BE
G_BE_num = subs(G_BE_simp, ...
    [gamma, alpha, l1, l2, lB], ...
    [gamma_val, alpha_val, l1_val, l2_val, lB_val]);
display(G_BE_num);

% Calculate point q transformation
q_E = double(G_BE_num * q_B)
