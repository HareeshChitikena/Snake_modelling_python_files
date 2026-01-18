
function [T, qT]= Snake_dynamic_modelV2(phy_properties_f, snake_parameters_f, control_gains_f, current_f)
% Interpolation with ODE45 for the reference signal date: 18/10/2022
%% Steps to follow
% - from u_bar we are getting torque required for the joint @ t=0 sec
% - from current_qa and current_qu... (phi, thetaN, px, py) and its derivatives
% - write x_dot and x, @ t= 0
% - Find x @ for t= 1 step by integrating x_dot, @t0
% - write x_dot and x, @ t= 1 step
% - Find x @ for t= 2 step by integrating x_dot, @t1
% [Tm, qTm] = Snake_dynamic_model(phy_properties, snake_parameters, control_gains, current)
%% Physical properties of the snake robot @ data provided in main file 
% ph_pro= [l, m, N]
% T_qa_ti;   % R^4x895
% T_qu_ti;  % R^3x895
global q l m J Nf A D e E H V K ub qDot T_ref_phis_ti T_ubs_ti T_qa_ti T_qu_ti Tt ...
    XY_ode_T qa_ddot qu_ddot
% Total reference phis @ ti
%control_gains_f phy_properties_f
lf = phy_properties_f.l; % link length in meters
l = lf;
mf = phy_properties_f.m; % link weight in kgs
m = mf;
Nf = phy_properties_f.N; % number of links
J = (mf*(l)^2)/3;  % kgm2
N= Nf;

%% Constant matrices
A = eye(N-1,N) + [zeros(N-1,1), eye(N-1,N-1)];  % R^N-1xN
D = eye(N-1,N) + [zeros(N-1,1), -eye(N-1,N-1)]; % R^N-1xN
e = ones(N,1);   % R^Nx1
E = [e zeros(N,1); zeros(N,1) e];  % R^2Nx2
%H = triu(ones(N)); % R^NxN
H = -1*triu(ones(N))
H(:,end) = -1* H(:,end) % modified to change the direction of snake movement. It makes theta_i = phi_i - phi_i+1 or otherwise. 
V = A'*inv(D*D')*A; % R^NxN
% K1 = A'/(D*D')
% K2 = A'*inv(D*D')
% K3 = A'/inv(D*D')
% V2 = K3*A
K = A'*inv(D*D')*D;   % R^NxN

%% friction model
g_f = phy_properties_f.g; % m/s2
c_n_f = phy_properties_f.c_n; %
c_t_f = phy_properties_f.c_t;
%F_rv_all_aniso:: R^2Nx1 col
%  F_rv_all_aniso = -1*[c_t_f*C_theta^2+c_n_f*S_theta^2, (c_t_f-c_n_f)*S_theta*C_theta; ... 
%     (c_t_f-c_n_f)*S_theta*C_theta, c_t_f*S_theta^2+c_n_f*C_theta^2]*[X_d_f; Y_d_f]
%  
 %% Model of the snake robot 
 

 % qu = [theta_n, px, py] row matrix
 x1 = current_f.qa_phi';  % in radians
 x2 = current_f.qu';  % x2 is column matrix  % in radians (thetaN) and meters (px and py)
 x3 = current_f.qa_phi_d'; % in radians/sec
 x4 = current_f.qu_d';  % in radians/sec and meters/sec
 xaf = [x1;x2;x3;x4]; % R^2N+6x1

 %xaf_d = [x3; x4; qa_ddot; qu_ddot]
 T_ref_phis_ti = [];
 T_qa_ti = [];  % [phi1; phi2; ...; phiN] 
 T_qu_ti = [];
 T_ubs_ti = []; % [] % column matrix
 XY_ode_T = [];
 p_moving = [];
 Forward_total_pro_force_t = [];
 Forward_total_pro_force_t2 = [];
 error_dynamics_t = [];
 thetaN_ode = [];

q = xaf;  % column xaf = [x1;x2;x3;x4] % radians   R^2N+4x1

qa_ddot = current_f.qa_phi_dd';  % qa_ddot is column matrix [0;0;0;0] R^N-1x1
qu_ddot = current_f.qu_dd'; % qu_ddot is column matrix [0;0;0] R^3x1

% Getting reference time and phi, phid, phidd signals for sending into the ode for
% interpolation
snake_parameters_f.Tspan;  % is a row matrix 1x21 
tspan = [0, snake_parameters_f.Tspan(end)];   % [0, 10]
[phir, phir_d, phir_dd] = phi_angle_animation_plotV2(snake_parameters_f);  % phir size R^21x4 with columns [phir1,phir2,phir3,...phirN] 
phi_reference = [phir, phir_d, phir_dd]; % phi_reference size R^21x12


%% Invoking ode45
options = odeset('AbsTol',1e-6,'RelTol',1e-6);
[T, qT] = ode45(@ (t, q) fun(t, q , snake_parameters_f.Tspan, phir, phir_d, phir_dd), tspan, q, options);

qT; % R^1765x14
T;
Tt;       % R^1x2863
T_ref_phis_ti;   % R^895x4 % T_ref_phis_ti: for every row ti to phi column angles
T_ubs_ti;  % R^4x895
T_qa_ti;   % R^4x895
T_qu_ti;  % R^3x895

% figure(13)
% plot(Tt, thetaN_ode)
% hold on
% title('Theta_N inside ode')
% xlabel('t')
% ylabel('Theta N')
% %legend({'Fx from inside ode','Fx from loop'},'Location','southwest');
% 
% 
% figure(1)
% subplot(2,1,1)
% plot(snake_parameters_f.Tspan,phir(:,1)*180/pi, '--g', 'LineWidth',1)
% hold on
% plot(T, qT(:,1)*180/pi, ':r', 'LineWidth',1)
% xlabel('$T_t~(sec)$','Interpreter','latex');
% ylabel('$T_{\phi_1}~{deg}$','Interpreter','latex');
% legend({'reference with ode values','\phi_1 inside ode values'},'Location','southwest');
% title('$T_{ref_\phi}~and~T_{qa1_t}~vs~T_t$','Interpreter', 'latex');
% % Plotting reference phi1 in degrees
% % qT- [phi1, phi2,..., phiN-1, thetaN, px, py, phi1d,...,phiN-1d, thetaNd,
% % pxd, pyd]
% subplot(2,1,2)
% plot(T, qT(:,1)*180/pi, '--b','LineWidth',1)
% hold on
% plot(Tt, T_qa_ti(1,:)*180/pi, ':r', 'LineWidth',1)
% xlabel('$T~(sec)$','Interpreter','latex');
% ylabel('$qT_{\phi_1}~(deg)$','Interpreter','latex');
% legend({'\phi_1 with return ode values','\phi_1 ode inside values'},'Location','southwest');
% title('$qT_{phi1}~vs~T$', 'Interpreter','latex');
% 
% figure(2)
% hold on
% subplot(2,2,1)
% plot(T, qT(:,1)*180/pi, '--b','LineWidth',1)
% hold on
% plot(snake_parameters_f.Tspan,phir(:,1)*180/pi, '--g', 'LineWidth',1)
% title('$\phi_1~vs~T$', 'Interpreter','latex');
% xlabel('$T~(sec)$','Interpreter','latex');
% ylabel('$\phi_1~(deg)$','Interpreter','latex');
% legend({'\phi_1','\phi_1 ref'},'Location','southwest');
% hold on
% 
% subplot(2,2,2)
% plot(T, qT(:,2)*180/pi, '--b','LineWidth',1)
% hold on
% plot(snake_parameters_f.Tspan,phir(:,2)*180/pi, '--g', 'LineWidth',1)
% title('$\phi_2~vs~T$', 'Interpreter','latex');
% xlabel('$T~(sec)$','Interpreter','latex');
% ylabel('$\phi_2~(deg)$','Interpreter','latex');
% legend({'\phi_2','\phi_2 ref'},'Location','southwest');
% hold on
% 
% subplot(2,2,3)
% plot(T, qT(:,3)*180/pi, '--b','LineWidth',1)
% hold on
% plot(snake_parameters_f.Tspan,phir(:,3)*180/pi, '--g', 'LineWidth',1)
% title('$\phi_3~vs~T$', 'Interpreter','latex');
% xlabel('$T~(sec)$','Interpreter','latex');
% ylabel('$\phi_3~(deg)$','Interpreter','latex');
% legend({'\phi_3','\phi_3 ref'},'Location','southwest');
% hold on
% 
% subplot(2,2,4)
% plot(T, qT(:,4)*180/pi, '--b','LineWidth',1)
% hold on
% plot(snake_parameters_f.Tspan,phir(:,4)*180/pi, '--g', 'LineWidth',1)
% title('$\phi_4~vs~T$', 'Interpreter','latex');
% xlabel('$T~(sec)$','Interpreter','latex');
% ylabel('$\phi_4~(deg)$','Interpreter','latex');
% legend({'\phi_4','\phi_4 ref'},'Location','southwest');

figure(3)
plot(T, qT(:, N+1))
hold on
plot(T, qT(:,N+2))
hold on
plot(T, qT(:,N))
title('snake CM movement along x and y axis, m = 1, for 80 sec');
xlabel('t')
ylabel('x and y and $\theta_N$','Interpreter','latex')
grid on
grid minor
legend({'X','Y', '$\theta_N$'},'Location','southwest', 'Interpreter','latex');

figure(4)
plot(qT(:, N+1), qT(:, N+2))
title('snake CM movement along x and y axis')
grid on
grid minor
xlabel('X')
ylabel('Y')
%% 

%     figure(13)
%     plot(Tt, error_dynamics_t(1,:), Tt, error_dynamics_t(2,:), Tt, error_dynamics_t(3,:), Tt, error_dynamics_t(4,:))
%     title('Error dynamics graph')
%     xlabel('t')
%     ylabel('error dynamics phis ')
%     legend({'$\phi_1$','$\phi_2$','$\phi_3$','$\phi_4$'},'Location','northeast');

qDot2 = [];
ub_T = [];
F_propulsion_T = [];
error_dynamics_T = [];
X_t = [];
Y_t = [];
phi_avg_heading_t = [];
ref_signal_phi_mean_t = [];
qa_ddot = current_f.qa_phi_dd';  % qa_ddot is column matrix [0;0;0;0] R^N-1x1
qu_ddot = current_f.qu_dd'; % qu_ddot is column matrix [0;0;0] R^3x1
[phir, phir_d, phir_dd] = phi_angle_animationV2(snake_parameters_f, T); % phir size R^21x4 with columns [phir1,phir2,phir3,...phirN] 
for i = 1: length(T)
    [qDot, F_propulsion, ub, error_dynamics, X, Y, phi_avg_heading, ref_signal_phi_mean, theta]  = fun2(T(i), qT(i,:)', phir(i,1:N-1), phir_d(i,1:N-1), phir_dd(i,1:N-1)); 
    % F_propulsion is coming from viscous forces
    qDot2  = [qDot2, qDot];
    ub_T = [ub_T, ub];
    F_propulsion_T = [F_propulsion_T, F_propulsion];
    error_dynamics_T =[error_dynamics_T, error_dynamics];
    
    X_head = X(end) + l * cos(theta(end));
    Y_head = Y(end) + l* sin(theta(end));
    X_t = [X_t, X_head];
    Y_t = [Y_t, Y_head];
    phi_avg_heading_t = [phi_avg_heading_t, phi_avg_heading];
    ref_signal_phi_mean_t = [ref_signal_phi_mean_t, ref_signal_phi_mean];
end 

%% 
figure(5)
% plot(Tt, Forward_total_pro_force_t)
% hold on
plot(T, F_propulsion_T)
title('Total Forward force w.r.to time with friction inside ode and from loop')
xlabel('t')
ylabel('Total forward force')
grid on
grid minor
legend({'Fx from inside ode','Fx from loop'},'Location','southwest');
% 
% figure(6)
% plot(T,error_dynamics_T)
% title('Error dynamics graph')
% xlabel('t')
% ylabel('error dynamics')
% grid on
% grid minor
% legend({'$\phi_1$','$\phi_2$','$\phi_3$','$\phi_4$'},'Location','northeast');
% 
% figure(7)
% plot(T,ub_T)
% title('control input vs time', 'FontSize', 24)
% xlabel('t')
% ylabel('control input (u)', 'FontSize', 22)
% grid on
% grid minor
% legend({'\phi_1','\phi_2','\phi_3','\phi_4'},'Location','northeast');
% 
% figure(8)
% plot(X_t, Y_t)
% title('Trace of the head of the snake at w = 50 deg/sec', 'FontSize', 24)
% xlabel('X (m)', 'FontSize', 22)
% ylabel('Y (m)', 'FontSize', 22)
% grid on
% grid minor
% %legend({'\phi_1','\phi_2','\phi_3','\phi_4'},'Location','northeast');
% 
% figure(9)
% subplot(2,1,1)
% plot(T,phi_avg_heading_t*180/pi)
% title('phi avg heading_t vs time', 'FontSize', 24)
% grid on
% grid minor
% xlabel('T (sec)', 'FontSize',22)
% ylabel('Phi avg heading_t ', 'FontSize',22)
% subplot(2,1,2)
% plot(T,ref_signal_phi_mean_t*180/pi)
% title('ref phi avg heading_t vs time', 'FontSize',22)
% grid on
% grid minor
% xlabel('T (sec)', 'FontSize',22)
% ylabel('Ref. Phi avg.', 'FontSize',22)

%[T, qT] = ode45(@ (t, q) fun(t, q , snake_parameters_f.Tspan, phir, phir_d, phir_dd), tspan, q, options);

%fun(t, q , tData, phir, phir_d, phir_dd)
    function qDot = fun(t,q, tData, phir, phir_d, phir_dd)
       
       phirr = interp1(tData, phir,t);  % phirr size R^1xN-1 ... [phi1, phi2,...phiN]
       phirr_d = interp1(tData, phir_d, t);  %  size R^1xN-1 ... [phi1, phi2,...phiN]
       phirr_dd = interp1(tData, phir_dd, t);  %  size R^1xN-1 ... [phi1, phi2,...phiN]

       % Initial values
       % q= R^2N+4x1
       x1 = q(1:N-1,:);   % x1 is column % q is column with [phi;phid;phidd;] ... qa_phi = [phi1,...phiN-1]'  % radians
       x2 = q(N:N+2,:);   % qu = [theta_n, px, py]'  % radians
       x3 = q(N+3:2*N+1,:); % qa_phi_dot = [phi1d,...phiN-1d]' % in radians/s
       x4 = q(2*N+2:2*N+4,:); % qu_dot = [theta_nd, pxd, pyd]' % in radians/s
       q_d = [x3; x4; qa_ddot; qu_ddot]; % q_d size R^14x1
       px_dd = qu_ddot(2);
       t;
       thetaN_ode = [thetaN_ode, x2(1)];
       phi_bar_ode = [x1;x2(1)]; % radians % R^Nx1
       phi_bar_d_ode = [x3;x4(1)]; % radians % R^Nx1
      
      %% Reference sinusoidal wave
     
       ref_signal_ode.phi = phirr'; % in radians % ref_signal_ode.phi is R^4x1 at time t.... phi1 to phiN are rows with time 1 columns
       ref_signal_ode.phi_d = phirr_d'; % in radians/s
       ref_signal_ode.phi_dd = phirr_dd'; % in radians/s2
       
       current_ode.qa_phi= x1; % radians R^4xt.... phi1 to phiN are rows with time columns
       current_ode.qu = x2; 
       current_ode.qa_phi_d= x3;
       current_ode.qu_d = x4;
       
       theta_ode = H* phi_bar_ode; % R^Nx1
       theta_d_ode = H* phi_bar_d_ode; % R^Nx1
           
       theta_bar = mean(theta_ode);

       sine_btheta = sin(theta_ode);  % in radians/s
       cose_btheta = cos(theta_ode);  % R^N % in radians/s
       S_theta = diag(sine_btheta); % R^N*N
       C_theta = diag(cose_btheta); % R^N*N
       sgn_theta = sign(theta_ode); % R^Nx1
       theta_d_sq = theta_d_ode.^2; % R^Nx1
       sine_N = sin(theta_ode); 
       cose_N = cos(theta_ode);

       p_x = x2(2);  % x2 = q(N:N+2,:) % qu = [theta_n, px, py]'
       p_y = x2(3);
       p = [p_x; p_y]; % in meters P is R^2x1

       p_moving= [p_moving, p]; % center of mass positions while moving forward  R^2xt
       
       %x4 = q(2*N+2:2*N+4,:) % qu_dot = [theta_nd, pxd, pyd]'
       p_x_d = x4(2);
       p_y_d = x4(3);
       p_d = [p_x_d; p_y_d]; % in meter/sec
        
       % tangential forward velocity 
       v_t_bar = p_x_d*cos(theta_bar)+p_y_d*cos(theta_bar); % m/s
        

       l;
       K;
       t;
       X_ode = -l*K'*cose_N + e*p(1); % R^5x1
       Y_ode = -l*K'*sine_N + e*p(2);
  
       % Define p_x_d and p_y_d 
        
       X_d_ode = l*K'*S_theta*theta_d_ode + e.*p_d(1);  % R^Nx1
       Y_d_ode = -l*K'*C_theta*theta_d_ode + e.*p_d(2);   % R^Nx1

       mu_t = 1;
       mu_n = 9;

      %% friction model - Viscous friction      
      g_f = phy_properties_f.g;
      c_n_f = phy_properties_f.c_n;
      c_t_f = phy_properties_f.c_t;
%       if t<40
%           c_n_f = c_n_f(1)
%       else 
%           c_n_f = c_n_f(2)
%       end

      %F_rv_all_aniso:: R^2Nx1 % in m/s
      F_rv_all_aniso = -1*[c_t_f*C_theta^2+c_n_f*S_theta^2, (c_t_f-c_n_f)*S_theta*C_theta; ... 
      (c_t_f-c_n_f)*S_theta*C_theta, c_t_f*S_theta^2+c_n_f*C_theta^2]*[X_d_ode; Y_d_ode]; % X_d_f how to get these values

      % Friction model - Coulomb friction model
       
      %F_rv_all_ani_coulomb = -1*m*g_f*[mu_t.*C_theta, -1*mu_n.*S_theta;...
      %                                  mu_t.*S_theta, mu_n.*C_theta]*sgn([C_theta, S_theta; ...
       %                                                                    -S_theta, C_theta]*[X_d_ode;Y_d_ode])


      % propulsion force calculation
       F_x_theta_i = c_t_f*cose_btheta.^2 + c_n_f * sine_btheta.^2; % R^Nx1
       tem = sine_btheta.*cose_btheta;  % R^Nx1
       F_y_theta_i = (c_t_f - c_n_f).*tem; % R^Nx1

       F_prop_i = (-1.*F_x_theta_i .* X_d_ode) - (F_y_theta_i .* Y_d_ode);
       F_prop_total = sum(F_prop_i);

       Forward_total_pro_force_t = [Forward_total_pro_force_t, F_prop_total];
       
       % propulsion calculation with px_dd
       Forward_total_pro_force_t2 =  [Forward_total_pro_force_t2, px_dd*N*m];
        
      %% Model of the snake robot
      m;
      M_theta = J*eye(N)+(m*l^2).*S_theta*V*S_theta+(m*l^2).*C_theta*V*C_theta; % R^NxN

      phi_bar_ode; % R^Nx1
      sine_phi_bar_ode = sin(phi_bar_ode);  % R^Nx1
      cose_phi_bar_ode = cos(phi_bar_ode);   % R^Nx1
      S_phi_bar_ode = diag(sine_phi_bar_ode); % R^NxN
      C_phi_bar_ode = diag(cose_phi_bar_ode);   % R^NxN
      M_theta_phib = J*eye(N)+(m*l^2).*S_phi_bar_ode*V*S_phi_bar_ode+(m*l^2).*C_phi_bar_ode*V*C_phi_bar_ode;  % R^NxN
      W_phib= ((m*l^2).*S_phi_bar_ode*V*C_phi_bar_ode)-((m*l^2).*C_phi_bar_ode*V*S_phi_bar_ode);  % R^NxN
      
      

      %W = (m*l^2).*S_theta*V*C_theta-(m*l^2).*C_theta*V*S_theta; % R^NxN
      
      % Mb_phib = [H'*M_theta*H, zeros(N,2); zeros(2,N), N*(m*eye(2))];
      Mb_phib = [H'*M_theta_phib*H, zeros(N,2); zeros(2,N), N*(m*eye(2))];   % R^N+2xN+2

      % Wb_phib_phibdot = [H'*W*diag(theta_d_ode)*theta_d_ode; zeros(2,1)]; 
      Wb_phib_phibdot = [H'*W_phib*diag(H*phi_bar_d_ode)*H*phi_bar_d_ode; zeros(2,1)]; %R^N+2x1

      Gb_phib = [-l*H'*S_theta*K, l*H'*C_theta*K; -e', zeros(1,N); zeros(1,N), -e'];  % R^12x20
 
      M11b = Mb_phib(1:N-1,1:N-1); % R^N-1xN-1
      M12b = Mb_phib(1:N-1,N:N+2); % R^N-1x3
      W1b = Wb_phib_phibdot(1:N-1); % R^N-1x1
      G1b = Gb_phib(1:N-1,:);  % R^N-1x2N
      
      M21b = Mb_phib(N:N+2,1:N-1); % R^3xN-1
      M22b = Mb_phib(N:N+2,N:N+2); % R^3x3
      W2b = Wb_phib_phibdot(N:N+2); % R^3x1
      G2b = Gb_phib(N:N+2,:); % R^3x2N
      B_bar = [eye(N-1); zeros(3,N-1)];

      t;
      fR = F_rv_all_aniso; %R^2Nx1
      A_qphi_qphid = -1*inv(M22b)*(W2b+G2b*fR);  % R^3x1
      B_qa = -1*inv(M22b)*M21b;  % R^3xN-1 
      current_ode.qa_phi_dd = q_d(N+3:2*N+1,:); %x3 = q(N+3:2*N+1,:)

      %% Control input u_bar
      
       ub_ode = control_snake(current_ode.qa_phi,current_ode.qa_phi_d, current_ode.qa_phi_dd, ...
                  ref_signal_ode.phi, ref_signal_ode.phi_d, ref_signal_ode.phi_dd, control_gains_f);
       
       error_dynamics = (ref_signal_ode.phi_dd - current_ode.qa_phi_dd) + ... 
           control_gains_f.kd * (ref_signal_ode.phi_d - current_ode.qa_phi_d) + ... 
           control_gains_f.kp * (ref_signal_ode.phi - current_ode.qa_phi) ;

       error_dynamics_t = [error_dynamics_t, error_dynamics];

%      ub_ode = zeros(N-1,1)  % using to turn off the controller
      qa_ddot = ub_ode;
      qu_ddot = A_qphi_qphid + B_qa*ub_ode;
      
      xaf = [x1;x2;x3;x4]; % R^2N+4x1
          
      xaf_d = [x3; x4; qa_ddot; qu_ddot];
      q = xaf;  % xaf = [x1;x2;x3;x4];
      qDot = xaf_d;  % xaf_d = [x3; x4; qa_ddot; qu_ddot]

      % Need to change this... append vertical or horizontal ?
      Tt = [Tt, t]; % in horizontal direction 
      r = [phirr; phirr_d; phirr_dd]; % phir = [phi1, phi2, phi3, phi4]
      T_ref_phis_ti = [T_ref_phis_ti; phirr]; % T_ref_phis_ti== for every row ti to phi column angles
      T_qa_ti = [T_qa_ti, x1];  % rows = [phi1; phi2; ...; phiN] x columns = t
      T_qu_ti = [T_qu_ti, x2]; % % qu = [theta_n; px; py]
      T_ubs_ti = [T_ubs_ti, ub_ode]; % ubs in rows and time columns  R^4x39

    end 
    
    function [qDot, F_prop_total, ub_ode, error_dynamics, X_ode, Y_ode, phi_avg_heading, ref_signal_phi_mean, theta_ode] = fun2(t,q, phirr, phirr_d, phirr_dd)
 
       % Initial values
       % q= R^2N+4x1
       x1 = q(1:N-1);   % x1 is column % q is column with [phi;phid;phidd;] ... qa_phi = [phi1,...phiN-1]'  % radians
       x2 = q(N:N+2);   % qu = [theta_n, px, py]'  % radians
       x3 = q(N+3:2*N+1); % qa_phi_dot = [phi1d,...phiN-1d]' % in radians/s
       x4 = q(2*N+2:2*N+4); % qu_dot = [theta_nd, pxd, pyd]' % in radians/s
       q_d = [x3; x4; qa_ddot; qu_ddot]; % q_d size R^14x1   % qa_ddot and qu_ddot has to be stored from previous step
       px_dd = qu_ddot(2);
       t;
       phi_bar_ode = [x1;x2(1)]; % radians % R^Nx1
       phi_bar_d_ode = [x3;x4(1)]; % radians % R^Nx1
       phi_avg_heading = mean(x1);
      
      %% Reference sinusoidal wave
     
       ref_signal_ode.phi = phirr'; % in radians % ref_signal_ode.phi is R^4x1 at time t.... phi1 to phiN are rows with time 1 columns
       ref_signal_ode.phi_d = phirr_d'; % in radians/s
       ref_signal_ode.phi_dd = phirr_dd'; % in radians/s2
       
       ref_signal_phi_mean= mean(ref_signal_ode.phi);

       current_ode.qa_phi= x1; % radians R^4xt.... phi1 to phiN are rows with time columns
       current_ode.qu = x2; 
       current_ode.qa_phi_d= x3;
       current_ode.qu_d = x4;
       
       theta_ode = H* phi_bar_ode; % R^Nx1
       theta_d_ode = H* phi_bar_d_ode; % R^Nx1
           
       theta_bar = mean(theta_ode);

       sine_btheta = sin(theta_ode);  % in radians/s
       cose_btheta = cos(theta_ode);  % R^N % in radians/s
       S_theta = diag(sine_btheta); % R^N*N
       C_theta = diag(cose_btheta); % R^N*N
       sgn_theta = sign(theta_ode); % R^Nx1
       theta_d_sq = theta_d_ode.^2; % R^Nx1
       sine_N = sin(theta_ode); 
       cose_N = cos(theta_ode);

       p_x = x2(2);  % x2 = q(N:N+2,:) % qu = [theta_n, px, py]'
       p_y = x2(3);
       p = [p_x; p_y]; % in meters P is R^2x1

       p_moving= [p_moving, p]; % center of mass positions while moving forward  R^2xt
       
       %x4 = q(2*N+2:2*N+4,:) % qu_dot = [theta_nd, pxd, pyd]'
       p_x_d = x4(2);
       p_y_d = x4(3);
       p_d = [p_x_d; p_y_d]; % in meter/sec
        
       % tangential forward velocity 
       v_t_bar = p_x_d*cos(theta_bar)+p_y_d*cos(theta_bar); % m/s
        

       l;
       K;
       X_ode = -l*K'*cose_N + e*p(1); % R^5x1
       Y_ode = -l*K'*sine_N + e*p(2);
  
       % Define p_x_d and p_y_d 
        
       X_d_ode = l*K'*S_theta*theta_d_ode + e.*p_d(1);  % R^Nx1
       Y_d_ode = -l*K'*C_theta*theta_d_ode + e.*p_d(2);   % R^Nx1
      

       mu_t = 1;
       mu_n = 9;

      %% friction model - Viscous friction      
      g_f = phy_properties_f.g;
      c_n_f = phy_properties_f.c_n;
      c_t_f = phy_properties_f.c_t;
      % friction is changed
%       if t<40
%           c_n_f = c_n_f(1)
%       else 
%           c_n_f = c_n_f(2)
%       end
      %F_rv_all_aniso:: R^2Nx1 % in m/s
      F_rv_all_aniso = -1*[c_t_f*C_theta^2+c_n_f*S_theta^2, (c_t_f-c_n_f)*S_theta*C_theta; ... 
      (c_t_f-c_n_f)*S_theta*C_theta, c_t_f*S_theta^2+c_n_f*C_theta^2]*[X_d_ode; Y_d_ode]; % X_d_f how to get these values

      % Friction model - Coulomb friction model
       
      %F_rv_all_ani_coulomb = -1*m*g_f*[mu_t.*C_theta, -1*mu_n.*S_theta;...
      %                                  mu_t.*S_theta, mu_n.*C_theta]*sgn([C_theta, S_theta; ...
       %                                                                    -S_theta, C_theta]*[X_d_ode;Y_d_ode])


      % propulsion force calculation
       F_x_theta_i = c_t_f*cose_btheta.^2 + c_n_f * sine_btheta.^2; % R^Nx1
       tem = sine_btheta.*cose_btheta;  % R^Nx1
       F_y_theta_i = (c_t_f - c_n_f).*tem; % R^Nx1

       F_prop_i = (-1.*F_x_theta_i .* X_d_ode) - (F_y_theta_i .* Y_d_ode);
       
       F_prop_total = sum(F_prop_i);
 
       F_prop_t_px_dd = px_dd*N*m;  % Q: Need to figure out how to use this in the beginning or end...

   
      %% Model of the snake robot
      M_theta = J*eye(N)+(m*l^2).*S_theta*V*S_theta+(m*l^2).*C_theta*V*C_theta; % R^NxN

      phi_bar_ode; % R^Nx1
      sine_phi_bar_ode = sin(phi_bar_ode);  % R^Nx1
      cose_phi_bar_ode = cos(phi_bar_ode);   % R^Nx1
      S_phi_bar_ode = diag(sine_phi_bar_ode); % R^NxN
      C_phi_bar_ode = diag(cose_phi_bar_ode);   % R^NxN
      M_theta_phib = J*eye(N)+(m*l^2).*S_phi_bar_ode*V*S_phi_bar_ode+(m*l^2).*C_phi_bar_ode*V*C_phi_bar_ode;  % R^NxN
      W_phib= (m*l^2).*S_phi_bar_ode*V*C_phi_bar_ode-(m*l^2).*C_phi_bar_ode*V*S_phi_bar_ode;  % R^NxN
      
      %W = (m*l^2).*S_theta*V*C_theta-(m*l^2).*C_theta*V*S_theta; % R^NxN
      
      % Mb_phib = [H'*M_theta*H, zeros(N,2); zeros(2,N), N*(m*eye(2))];
      Mb_phib = [H'*M_theta_phib*H, zeros(N,2); zeros(2,N), N*(m*eye(2))];   % R^N+2xN+2

      % Wb_phib_phibdot = [H'*W*diag(theta_d_ode)*theta_d_ode; zeros(2,1)]; 
      Wb_phib_phibdot = [H'*W_phib*diag(H*phi_bar_d_ode)*H*phi_bar_d_ode; zeros(2,1)]; %R^N+2x1

      Gb_phib = [-l*H'*S_theta*K, l*H'*C_theta*K; -e', zeros(1,N); zeros(1,N), -e'];  % R^12x20
 
      M11b = Mb_phib(1:N-1,1:N-1); % R^N-1xN-1
      M12b = Mb_phib(1:N-1,N:N+2); % R^N-1x3
      W1b = Wb_phib_phibdot(1:N-1); % R^N-1x1
      G1b = Gb_phib(1:N-1,:);  % R^N-1x2N
      
      M21b = Mb_phib(N:N+2,1:N-1); % R^3xN-1
      M22b = Mb_phib(N:N+2,N:N+2); % R^3x3
      W2b = Wb_phib_phibdot(N:N+2); % R^3x1
      G2b = Gb_phib(N:N+2,:); % R^3x2N
      B_bar = [eye(N-1); zeros(3,N-1)];

      t;
      fR = F_rv_all_aniso; %R^2Nx1
      A_qphi_qphid = -1*inv(M22b)*(W2b+G2b*fR);  % R^3x1
      B_qa = -1*inv(M22b)*M21b;  % R^3xN-1 
      current_ode.qa_phi_dd = q_d(N+3:2*N+1,:); %x3 = q(N+3:2*N+1,:)

      %% Control input u_bar
      % new set of control inputs, enables us to rewrite 2.41a&b as
      %  ub = [u1b, u2b, u3b]' 
%        if t == 0 
%              ub_ode = u_bar_f' % ub - column vector with R^N-1
%        else 
      ub_ode = control_snake(current_ode.qa_phi,current_ode.qa_phi_d, current_ode.qa_phi_dd, ...
                  ref_signal_ode.phi, ref_signal_ode.phi_d, ref_signal_ode.phi_dd, control_gains_f);
       
      error_dynamics = (ref_signal_ode.phi_dd - current_ode.qa_phi_dd) + ... 
           control_gains_f.kd * (ref_signal_ode.phi_d - current_ode.qa_phi_d) + ... 
           control_gains_f.kp * (ref_signal_ode.phi - current_ode.qa_phi);

%      ub_ode = zeros(N-1,1)  % using to turn off the controller
      qa_ddot = ub_ode;
      qu_ddot = A_qphi_qphid + B_qa*ub_ode;
      
      xaf = [x1;x2;x3;x4]; % R^2N+4x1
          
      xaf_d = [x3; x4; qa_ddot; qu_ddot];
      q = xaf;  % xaf = [x1;x2;x3;x4];
      qDot = xaf_d;  % xaf_d = [x3; x4; qa_ddot; qu_ddot]

  end 


end
