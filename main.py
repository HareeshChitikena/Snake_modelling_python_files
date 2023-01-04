import numpy as np
clear('all')
## Physical properties of snake robot - phy_properties structure
#N- total number of links
# l=0.07m, m= 1 kg,
phy_properties.l = 0.14 / 2

phy_properties.m = 1

phy_properties.N = 10
phy_properties.g = 9.8

phy_properties.c_n = 10
phy_properties.c_t = 1
## Parameters for the sinusoidal reference signal
# snake parameter structure
# alpha - a - amplitude of the link,
# w - angular frequency of periodic motion
# theta- th- link angle
# t - time
# delta - d - phase difference.
# i - link number
snake_parameters.alpha = 30 * np.pi / 180

snake_parameters.freq_w = 50 * np.pi / 180

snake_parameters.delta = 40 * np.pi / 180

#N = 5
snake_parameters.joint_offset = 0 * np.pi / 180
#snake_parameters.heading_angle = 10; # degree
#link1_theta = 45* pi/180 # in radians
# snake_parameters.T = linspace(0,80,300)';
snake_parameters.Tspan = np.arange(0,80+0.1,0.1)
t = 0
# @function: phi_angle_animation:- Getting reference signal  i.e. joint reference angles
# phi_angle_animation_plot(snake_parameters);

#[phir, phir_d, phir_dd] = phi_angle_animation_plotV2(snake_parameters); # calculating reference phi at t=0

## Control input to the system
# Define where the snake is located...
# Global coordinate system (X_0, Y_0)
initial.X_0 = 0
initial.Y_0 = 0
# initial position of first link starting position
initial.starting_point_l_1 = np.array([[0],[0]])
initial.x1_i = initial.X_0 + initial.starting_point_l_1(1)
initial.y1_i = initial.Y_0 + initial.starting_point_l_1(2)

# link1 Angle with the Global coordinate system
#initial.theta1_i = 0; # link 1 theta  # I think this is not needed because we find theta1 based on phi angles and theta_N

# Initial conditions for phi, phi_d (rad/s), phi_dd (rad/s^2) @ zero second
# qa_phi = [phi1, phi2, phi3, phi4]  row matrix in radians
initial.qa_phi,initial.qa_phi_d,initial.qa_phi_dd = deal(np.array([0,0,0,0,0,0,0,0,0]) * np.pi / 180,np.zeros((1,phy_properties.N - 1)),np.zeros((1,phy_properties.N - 1)))
# Theta and theta_dot value for the first or head link
theta_n = 0.0 * np.pi / 180

theta_n_d = 0.0

# initial phi and phi_dot values # column vector
initial.phi_bar = np.array([[np.transpose(initial.qa_phi)],[theta_n]])

initial.phi_bar_d = np.array([[np.transpose(initial.qa_phi_d)],[theta_n_d]])
# initial thetas and theta_dots based on initial conditions
# theta is column vector
H = - 1 * triu(np.ones((phy_properties.N,phy_properties.N)))
H[:,end()] = - 1 * H(:,end())
initial.theta = H * initial.phi_bar

initial.theta_d = H * initial.phi_bar_d

# Heading angle:
initial.theta_bar = mean(initial.theta)

initial
# initial positions of CMs of each link
# link 1 CM @ t=0 sec
x1g_t0 = initial.x1_i + phy_properties.l * np.cos(initial.theta(1))

y1g_t0 = initial.y1_i + phy_properties.l * np.sin(initial.theta(1))

# link 2 CM @ t=0 sec
x2g_t0 = x1g_t0 + phy_properties.l * (np.cos(initial.theta(1)) + np.cos(initial.theta(2)))

y2g_t0 = y1g_t0 + phy_properties.l * (np.sin(initial.theta(1)) + np.sin(initial.theta(2)))

# link 3 CM @ t=0 sec
x3g_t0 = x2g_t0 + phy_properties.l * (np.cos(initial.theta(2)) + np.cos(initial.theta(3)))

y3g_t0 = y2g_t0 + phy_properties.l * (np.sin(initial.theta(2)) + np.sin(initial.theta(3)))

# link 4 CM @ t=0 sec
x4g_t0 = x3g_t0 + phy_properties.l * (np.cos(initial.theta(3)) + np.cos(initial.theta(4)))

y4g_t0 = y3g_t0 + phy_properties.l * (np.sin(initial.theta(3)) + np.sin(initial.theta(4)))
# link 5 CM @ t=0 sec
x5g_t0 = x4g_t0 + phy_properties.l * (np.cos(initial.theta(4)) + np.cos(initial.theta(5)))

y5g_t0 = y4g_t0 + phy_properties.l * (np.sin(initial.theta(4)) + np.sin(initial.theta(5)))

# link 6 CM @ t=0 sec
x6g_t0 = x5g_t0 + phy_properties.l * (np.cos(initial.theta(5)) + np.cos(initial.theta(6)))

y6g_t0 = y5g_t0 + phy_properties.l * (np.sin(initial.theta(5)) + np.sin(initial.theta(6)))

# link 7 CM @ t=0 sec
x7g_t0 = x6g_t0 + phy_properties.l * (np.cos(initial.theta(6)) + np.cos(initial.theta(7)))

y7g_t0 = y6g_t0 + phy_properties.l * (np.sin(initial.theta(6)) + np.sin(initial.theta(7)))

# link 8 CM @ t=0 sec
x8g_t0 = x7g_t0 + phy_properties.l * (np.cos(initial.theta(7)) + np.cos(initial.theta(8)))

y8g_t0 = y7g_t0 + phy_properties.l * (np.sin(initial.theta(7)) + np.sin(initial.theta(8)))

# link 9 CM @ t=0 sec
x9g_t0 = x8g_t0 + phy_properties.l * (np.cos(initial.theta(8)) + np.cos(initial.theta(9)))

y9g_t0 = y8g_t0 + phy_properties.l * (np.sin(initial.theta(8)) + np.sin(initial.theta(9)))

# link 10 CM @ t=0 sec
x10g_t0 = x9g_t0 + phy_properties.l * (np.cos(initial.theta(9)) + np.cos(initial.theta(10)))

y10g_t0 = y9g_t0 + phy_properties.l * (np.sin(initial.theta(9)) + np.sin(initial.theta(10)))

# initial CMs of the links
initial.X = np.array([[x1g_t0],[x2g_t0],[x3g_t0],[x4g_t0],[x5g_t0],[x6g_t0],[x7g_t0],[x8g_t0],[x9g_t0],[x10g_t0]])

initial.Y = np.array([[y1g_t0],[y2g_t0],[y3g_t0],[y4g_t0],[y5g_t0],[y6g_t0],[y7g_t0],[y8g_t0],[y9g_t0],[y10g_t0]])

initial.X_d = np.zeros((phy_properties.N,1))

initial.Y_d = np.zeros((phy_properties.N,1))

# px and py initial values
initial.px_t0 = mean(initial.X)

initial.py_t0 = mean(initial.Y)

initial.P_t0 = np.array([[initial.px_t0],[initial.py_t0]])

##
# px and py initial velocities
initial.px_d_t0 = 0

initial.py_d_t0 = 0

initial.P_d_t0 = np.array([[initial.px_d_t0],[initial.py_d_t0]])

# qu = [theta_n, px, py] ;
initial.qu,initial.qu_d,initial.qu_dd = deal(np.array([theta_n,initial.px_t0,initial.py_t0]),np.array([theta_n_d,initial.px_d_t0,initial.py_d_t0]),np.array([0,0,0]))

## sending the initial values and reference signal directly to the model

# Control_gains............................................
control_gains.kp = 2
control_gains.kd = 1.5
control_gains.ki = 0
current = initial
Tm,qTm = Snake_dynamic_modelV2(phy_properties,snake_parameters,control_gains,current)
