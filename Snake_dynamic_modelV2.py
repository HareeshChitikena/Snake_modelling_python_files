
import numpy as np
import matplotlib.pyplot as plt
    
def Snake_dynamic_modelV2(phy_properties_f = None,snake_parameters_f = None,control_gains_f = None,current_f = None): 
    # Interpolation with ODE45 for the reference signal date: 18/10/2022
## Steps to follow
# - from u_bar we are getting torque required for the joint @ t=0 sec
# - from current_qa and current_qu... (phi, thetaN, px, py) and its derivatives
# - write x_dot and x, @ t= 0
# - Find x @ for t= 1 step by integrating x_dot, @t0
# - write x_dot and x, @ t= 1 step
# - Find x @ for t= 2 step by integrating x_dot, @t1
# [Tm, qTm] = Snake_dynamic_model(phy_properties, snake_parameters, control_gains, current)
## Physical properties of the snake robot @ data provided in main file
# ph_pro= [l, m, N]
# T_qa_ti;   # R^4x895
# T_qu_ti;  # R^3x895
    global q,l,m,J,Nf,A,D,e,E,H,V,K,ub,qDot,T_ref_phis_ti,T_ubs_ti,T_qa_ti,T_qu_ti,Tt,XY_ode_T,qa_ddot,qu_ddot
    # Total reference phis @ ti
#control_gains_f phy_properties_f
    lf = phy_properties_f.l
    
    l = lf
    mf = phy_properties_f.m
    
    m = mf
    Nf = phy_properties_f.N
    
    J = (mf * (l) ** 2) / 3
    
    N = Nf
    ## Constant matrices
    A = np.eye(N - 1,N) + np.array([np.zeros((N - 1,1)),np.eye(N - 1,N - 1)])
    
    D = np.eye(N - 1,N) + np.array([np.zeros((N - 1,1)),- np.eye(N - 1,N - 1)])
    
    e = np.ones((N,1))
    
    E = np.array([[e,np.zeros((N,1))],[np.zeros((N,1)),e]])
    
    #H = triu(ones(N)); # R^NxN
    H = - 1 * triu(np.ones((N,N)))
    H[:,end()] = - 1 * H(:,end())
    V = np.transpose(A) * inv(D * np.transpose(D)) * A
    
    # K1 = A'/(D*D')
# K2 = A'*inv(D*D')
# K3 = A'/inv(D*D')
# V2 = K3*A
    K = np.transpose(A) * inv(D * np.transpose(D)) * D
    
    ## friction model
    g_f = phy_properties_f.g
    
    c_n_f = phy_properties_f.c_n
    
    c_t_f = phy_properties_f.c_t
    #F_rv_all_aniso:: R^2Nx1 col
#  F_rv_all_aniso = -1*[c_t_f*C_theta^2+c_n_f*S_theta^2, (c_t_f-c_n_f)*S_theta*C_theta; ...
#     (c_t_f-c_n_f)*S_theta*C_theta, c_t_f*S_theta^2+c_n_f*C_theta^2]*[X_d_f; Y_d_f]
    
    ## Model of the snake robot
    
    # qu = [theta_n, px, py] row matrix
    x1 = np.transpose(current_f.qa_phi)
    
    x2 = np.transpose(current_f.qu)
    
    x3 = np.transpose(current_f.qa_phi_d)
    
    x4 = np.transpose(current_f.qu_d)
    
    xaf = np.array([[x1],[x2],[x3],[x4]])
    
    #xaf_d = [x3; x4; qa_ddot; qu_ddot]
    T_ref_phis_ti = []
    T_qa_ti = []
    
    T_qu_ti = []
    T_ubs_ti = []
    
    XY_ode_T = []
    p_moving = []
    Forward_total_pro_force_t = []
    Forward_total_pro_force_t2 = []
    error_dynamics_t = []
    thetaN_ode = []
    q = xaf
    
    qa_ddot = np.transpose(current_f.qa_phi_dd)
    
    qu_ddot = np.transpose(current_f.qu_dd)
    
    # Getting reference time and phi, phid, phidd signals for sending into the ode for
# interpolation
    snake_parameters_f.Tspan
    
    tspan = np.array([0,snake_parameters_f.Tspan(end())])
    
    phir,phir_d,phir_dd = phi_angle_animation_plotV2(snake_parameters_f)
    
    phi_reference = np.array([phir,phir_d,phir_dd])
    
    ## Invoking ode45
    options = odeset('AbsTol',1e-06,'RelTol',1e-06)
    T,qT = ode45(lambda t = None,q = None: fun(t,q,snake_parameters_f.Tspan,phir,phir_d,phir_dd),tspan,q,options)
    qT
    
    T
    Tt
    
    T_ref_phis_ti
    
    T_ubs_ti
    
    T_qa_ti
    
    T_qu_ti
    
    plt.figure(13)
    plt.plot(Tt,thetaN_ode)
    hold('on')
    plt.title('Theta_N inside ode')
    plt.xlabel('t')
    plt.ylabel('Theta N')
    #legend({'Fx from inside ode','Fx from loop'},'Location','southwest');
    
    plt.figure(1)
    subplot(2,1,1)
    plt.plot(snake_parameters_f.Tspan,phir(:,1) * 180 / np.pi,'--g','LineWidth',1)
    hold('on')
    plt.plot(T,qT(:,1) * 180 / np.pi,':r','LineWidth',1)
    plt.xlabel('$T_t~(sec)$','Interpreter','latex')
    plt.ylabel('$T_{\phi_1}~{deg}$','Interpreter','latex')
    plt.legend(np.array(['reference with ode values','\phi_1 inside ode values']),'Location','southwest')
    plt.title('$T_{ref_\phi}~and~T_{qa1_t}~vs~T_t$','Interpreter','latex')
    # Plotting reference phi1 in degrees
# qT- [phi1, phi2,..., phiN-1, thetaN, px, py, phi1d,...,phiN-1d, thetaNd,
# pxd, pyd]
    subplot(2,1,2)
    plt.plot(T,qT(:,1) * 180 / np.pi,'--b','LineWidth',1)
    hold('on')
    plt.plot(Tt,T_qa_ti(1,:) * 180 / np.pi,':r','LineWidth',1)
    plt.xlabel('$T~(sec)$','Interpreter','latex')
    plt.ylabel('$qT_{\phi_1}~(deg)$','Interpreter','latex')
    plt.legend(np.array(['\phi_1 with return ode values','\phi_1 ode inside values']),'Location','southwest')
    plt.title('$qT_{phi1}~vs~T$','Interpreter','latex')
    plt.figure(2)
    hold('on')
    subplot(2,2,1)
    plt.plot(T,qT(:,1) * 180 / np.pi,'--b','LineWidth',1)
    hold('on')
    plt.plot(snake_parameters_f.Tspan,phir(:,1) * 180 / np.pi,'--g','LineWidth',1)
    plt.title('$\phi_1~vs~T$','Interpreter','latex')
    plt.xlabel('$T~(sec)$','Interpreter','latex')
    plt.ylabel('$\phi_1~(deg)$','Interpreter','latex')
    plt.legend(np.array(['\phi_1','\phi_1 ref']),'Location','southwest')
    hold('on')
    subplot(2,2,2)
    plt.plot(T,qT(:,2) * 180 / np.pi,'--b','LineWidth',1)
    hold('on')
    plt.plot(snake_parameters_f.Tspan,phir(:,2) * 180 / np.pi,'--g','LineWidth',1)
    plt.title('$\phi_2~vs~T$','Interpreter','latex')
    plt.xlabel('$T~(sec)$','Interpreter','latex')
    plt.ylabel('$\phi_2~(deg)$','Interpreter','latex')
    plt.legend(np.array(['\phi_2','\phi_2 ref']),'Location','southwest')
    hold('on')
    subplot(2,2,3)
    plt.plot(T,qT(:,3) * 180 / np.pi,'--b','LineWidth',1)
    hold('on')
    plt.plot(snake_parameters_f.Tspan,phir(:,3) * 180 / np.pi,'--g','LineWidth',1)
    plt.title('$\phi_3~vs~T$','Interpreter','latex')
    plt.xlabel('$T~(sec)$','Interpreter','latex')
    plt.ylabel('$\phi_3~(deg)$','Interpreter','latex')
    plt.legend(np.array(['\phi_3','\phi_3 ref']),'Location','southwest')
    hold('on')
    subplot(2,2,4)
    plt.plot(T,qT(:,4) * 180 / np.pi,'--b','LineWidth',1)
    hold('on')
    plt.plot(snake_parameters_f.Tspan,phir(:,4) * 180 / np.pi,'--g','LineWidth',1)
    plt.title('$\phi_4~vs~T$','Interpreter','latex')
    plt.xlabel('$T~(sec)$','Interpreter','latex')
    plt.ylabel('$\phi_4~(deg)$','Interpreter','latex')
    plt.legend(np.array(['\phi_4','\phi_4 ref']),'Location','southwest')
    plt.figure(3)
    plt.plot(T,qT(:,N + 1))
    hold('on')
    plt.plot(T,qT(:,N + 2))
    hold('on')
    plt.plot(T,qT(:,N))
    plt.title('snake CM movement along x and y axis, m = 1, for 80 sec')
    plt.xlabel('t')
    plt.ylabel('x and y and $\theta_N$','Interpreter','latex')
    grid('on')
    grid('minor')
    plt.legend(np.array(['X','Y','$\theta_N$']),'Location','southwest','Interpreter','latex')
    plt.figure(4)
    plt.plot(qT(:,N + 1),qT(:,N + 2))
    plt.title('snake CM movement along x and y axis')
    grid('on')
    grid('minor')
    plt.xlabel('X')
    plt.ylabel('Y')
    ##
    
    #     figure(13)
#     plot(Tt, error_dynamics_t(1,:), Tt, error_dynamics_t(2,:), Tt, error_dynamics_t(3,:), Tt, error_dynamics_t(4,:))
#     title('Error dynamics graph')
#     xlabel('t')
#     ylabel('error dynamics phis ')
#     legend({'$\phi_1$','$\phi_2$','$\phi_3$','$\phi_4$'},'Location','northeast');
    
    qDot2 = []
    ub_T = []
    F_propulsion_T = []
    error_dynamics_T = []
    X_t = []
    Y_t = []
    phi_avg_heading_t = []
    ref_signal_phi_mean_t = []
    qa_ddot = np.transpose(current_f.qa_phi_dd)
    
    qu_ddot = np.transpose(current_f.qu_dd)
    
    phir,phir_d,phir_dd = phi_angle_animationV2(snake_parameters_f,T)
    
    for i in np.arange(1,len(T)+1).reshape(-1):
        qDot,F_propulsion,ub,error_dynamics,X,Y,phi_avg_heading,ref_signal_phi_mean,theta = fun2(T(i),np.transpose(qT(i,:)),phir(i,np.arange(1,N - 1+1)),phir_d(i,np.arange(1,N - 1+1)),phir_dd(i,np.arange(1,N - 1+1)))
        # F_propulsion is coming from viscous forces
        qDot2 = np.array([qDot2,qDot])
        ub_T = np.array([ub_T,ub])
        F_propulsion_T = np.array([F_propulsion_T,F_propulsion])
        error_dynamics_T = np.array([error_dynamics_T,error_dynamics])
        X_head = X(end()) + l * np.cos(theta(end()))
        Y_head = Y(end()) + l * np.sin(theta(end()))
        X_t = np.array([X_t,X_head])
        Y_t = np.array([Y_t,Y_head])
        phi_avg_heading_t = np.array([phi_avg_heading_t,phi_avg_heading])
        ref_signal_phi_mean_t = np.array([ref_signal_phi_mean_t,ref_signal_phi_mean])
    
    ##
    plt.figure(5)
    plt.plot(Tt,Forward_total_pro_force_t)
    hold('on')
    plt.plot(T,F_propulsion_T)
    plt.title('Total Forward force w.r.to time with friction inside ode and from loop')
    plt.xlabel('t')
    plt.ylabel('Total forward force')
    grid('on')
    grid('minor')
    plt.legend(np.array(['Fx from inside ode','Fx from loop']),'Location','southwest')
    plt.figure(6)
    plt.plot(T,error_dynamics_T)
    plt.title('Error dynamics graph')
    plt.xlabel('t')
    plt.ylabel('error dynamics')
    grid('on')
    grid('minor')
    plt.legend(np.array(['$\phi_1$','$\phi_2$','$\phi_3$','$\phi_4$']),'Location','northeast')
    plt.figure(7)
    plt.plot(T,ub_T)
    plt.title('control input vs time','FontSize',24)
    plt.xlabel('t')
    plt.ylabel('control input (u)','FontSize',22)
    grid('on')
    grid('minor')
    plt.legend(np.array(['\phi_1','\phi_2','\phi_3','\phi_4']),'Location','northeast')
    plt.figure(8)
    plt.plot(X_t,Y_t)
    plt.title('Trace of the head of the snake at w = 50 deg/sec','FontSize',24)
    plt.xlabel('X (m)','FontSize',22)
    plt.ylabel('Y (m)','FontSize',22)
    grid('on')
    grid('minor')
    #legend({'\phi_1','\phi_2','\phi_3','\phi_4'},'Location','northeast');
    
    plt.figure(9)
    subplot(2,1,1)
    plt.plot(T,phi_avg_heading_t * 180 / np.pi)
    plt.title('phi avg heading_t vs time','FontSize',24)
    grid('on')
    grid('minor')
    plt.xlabel('T (sec)','FontSize',22)
    plt.ylabel('Phi avg heading_t ','FontSize',22)
    subplot(2,1,2)
    plt.plot(T,ref_signal_phi_mean_t * 180 / np.pi)
    plt.title('ref phi avg heading_t vs time','FontSize',22)
    grid('on')
    grid('minor')
    plt.xlabel('T (sec)','FontSize',22)
    plt.ylabel('Ref. Phi avg.','FontSize',22)
    #fun(t, q , tData, phir, phir_d, phir_dd)
    
def fun(t = None,q = None,tData = None,phir = None,phir_d = None,phir_dd = None): 
    phirr = interp1(tData,phir,t)
    
    phirr_d = interp1(tData,phir_d,t)
    
    phirr_dd = interp1(tData,phir_dd,t)
    
    # Initial values
# q= R^2N+4x1
    x1 = q(np.arange(1,N - 1+1),:)
    
    x2 = q(np.arange(N,N + 2+1),:)
    
    x3 = q(np.arange(N + 3,2 * N + 1+1),:)
    
    x4 = q(np.arange(2 * N + 2,2 * N + 4+1),:)
    
    q_d = np.array([[x3],[x4],[qa_ddot],[qu_ddot]])
    
    px_dd = qu_ddot(2)
    t
    thetaN_ode = np.array([thetaN_ode,x2(1)])
    phi_bar_ode = np.array([[x1],[x2(1)]])
    
    phi_bar_d_ode = np.array([[x3],[x4(1)]])
    
    ## Reference sinusoidal wave
    
    ref_signal_ode.phi = np.transpose(phirr)
    
    ref_signal_ode.phi_d = np.transpose(phirr_d)
    
    ref_signal_ode.phi_dd = np.transpose(phirr_dd)
    
    current_ode.qa_phi = x1
    
    current_ode.qu = x2
    current_ode.qa_phi_d = x3
    current_ode.qu_d = x4
    theta_ode = H * phi_bar_ode
    
    theta_d_ode = H * phi_bar_d_ode
    
    theta_bar = mean(theta_ode)
    sine_btheta = np.sin(theta_ode)
    
    cose_btheta = np.cos(theta_ode)
    
    S_theta = diag(sine_btheta)
    
    C_theta = diag(cose_btheta)
    
    sgn_theta = np.sign(theta_ode)
    
    theta_d_sq = theta_d_ode ** 2
    
    sine_N = np.sin(theta_ode)
    cose_N = np.cos(theta_ode)
    p_x = x2(2)
    
    p_y = x2(3)
    p = np.array([[p_x],[p_y]])
    
    p_moving = np.array([p_moving,p])
    
    #x4 = q(2*N+2:2*N+4,:) # qu_dot = [theta_nd, pxd, pyd]'
    p_x_d = x4(2)
    p_y_d = x4(3)
    p_d = np.array([[p_x_d],[p_y_d]])
    
    # tangential forward velocity
    v_t_bar = p_x_d * np.cos(theta_bar) + p_y_d * np.cos(theta_bar)
    
    l
    K
    X_ode = - l * np.transpose(K) * cose_N + e * p(1)
    
    Y_ode = - l * np.transpose(K) * sine_N + e * p(2)
    # Define p_x_d and p_y_d
    
    X_d_ode = l * np.transpose(K) * S_theta * theta_d_ode + np.multiply(e,p_d(1))
    
    Y_d_ode = - l * np.transpose(K) * C_theta * theta_d_ode + np.multiply(e,p_d(2))
    
    mu_t = 1
    mu_n = 9
    ## friction model - Viscous friction
    g_f = phy_properties_f.g
    c_n_f = phy_properties_f.c_n
    c_t_f = phy_properties_f.c_t
    #F_rv_all_aniso:: R^2Nx1 # in m/s
    F_rv_all_aniso = - 1 * np.array([[c_t_f * C_theta ** 2 + c_n_f * S_theta ** 2,(c_t_f - c_n_f) * S_theta * C_theta],[(c_t_f - c_n_f) * S_theta * C_theta,c_t_f * S_theta ** 2 + c_n_f * C_theta ** 2]]) * np.array([[X_d_ode],[Y_d_ode]])
    
    # Friction model - Coulomb friction model
    
    #F_rv_all_ani_coulomb = -1*m*g_f*[mu_t.*C_theta, -1*mu_n.*S_theta;...
#                                  mu_t.*S_theta, mu_n.*C_theta]*sgn([C_theta, S_theta; ...
#                                                                    -S_theta, C_theta]*[X_d_ode;Y_d_ode])
    
    # propulsion force calculation
    F_x_theta_i = c_t_f * cose_btheta ** 2 + c_n_f * sine_btheta ** 2
    
    tem = np.multiply(sine_btheta,cose_btheta)
    
    F_y_theta_i = np.multiply((c_t_f - c_n_f),tem)
    
    F_prop_i = (np.multiply(- 1.0 * F_x_theta_i,X_d_ode)) - (np.multiply(F_y_theta_i,Y_d_ode))
    F_prop_total = sum(F_prop_i)
    Forward_total_pro_force_t = np.array([Forward_total_pro_force_t,F_prop_total])
    # propulsion calculation with px_dd
    Forward_total_pro_force_t2 = np.array([Forward_total_pro_force_t2,px_dd * N * m])
    ## Model of the snake robot
    m
    M_theta = J * np.eye(N) + np.multiply((m * l ** 2),S_theta) * V * S_theta + np.multiply((m * l ** 2),C_theta) * V * C_theta
    
    phi_bar_ode
    
    sine_phi_bar_ode = np.sin(phi_bar_ode)
    
    cose_phi_bar_ode = np.cos(phi_bar_ode)
    
    S_phi_bar_ode = diag(sine_phi_bar_ode)
    
    C_phi_bar_ode = diag(cose_phi_bar_ode)
    
    M_theta_phib = J * np.eye(N) + np.multiply((m * l ** 2),S_phi_bar_ode) * V * S_phi_bar_ode + np.multiply((m * l ** 2),C_phi_bar_ode) * V * C_phi_bar_ode
    
    W_phib = (np.multiply((m * l ** 2),S_phi_bar_ode) * V * C_phi_bar_ode) - (np.multiply((m * l ** 2),C_phi_bar_ode) * V * S_phi_bar_ode)
    
    #W = (m*l^2).*S_theta*V*C_theta-(m*l^2).*C_theta*V*S_theta; # R^NxN
    
    # Mb_phib = [H'*M_theta*H, zeros(N,2); zeros(2,N), N*(m*eye(2))];
    Mb_phib = np.array([[np.transpose(H) * M_theta_phib * H,np.zeros((N,2))],[np.zeros((2,N)),N * (m * np.eye(2))]])
    
    # Wb_phib_phibdot = [H'*W*diag(theta_d_ode)*theta_d_ode; zeros(2,1)];
    Wb_phib_phibdot = np.array([[np.transpose(H) * W_phib * diag(H * phi_bar_d_ode) * H * phi_bar_d_ode],[np.zeros((2,1))]])
    
    Gb_phib = np.array([[- l * np.transpose(H) * S_theta * K,l * np.transpose(H) * C_theta * K],[- np.transpose(e),np.zeros((1,N))],[np.zeros((1,N)),- np.transpose(e)]])
    
    M11b = Mb_phib(np.arange(1,N - 1+1),np.arange(1,N - 1+1))
    
    M12b = Mb_phib(np.arange(1,N - 1+1),np.arange(N,N + 2+1))
    
    W1b = Wb_phib_phibdot(np.arange(1,N - 1+1))
    
    G1b = Gb_phib(np.arange(1,N - 1+1),:)
    
    M21b = Mb_phib(np.arange(N,N + 2+1),np.arange(1,N - 1+1))
    
    M22b = Mb_phib(np.arange(N,N + 2+1),np.arange(N,N + 2+1))
    
    W2b = Wb_phib_phibdot(np.arange(N,N + 2+1))
    
    G2b = Gb_phib(np.arange(N,N + 2+1),:)
    
    B_bar = np.array([[np.eye(N - 1)],[np.zeros((3,N - 1))]])
    t
    fR = F_rv_all_aniso
    
    A_qphi_qphid = - 1 * inv(M22b) * (W2b + G2b * fR)
    
    B_qa = - 1 * inv(M22b) * M21b
    
    current_ode.qa_phi_dd = q_d(np.arange(N + 3,2 * N + 1+1),:)
    
    ## Control input u_bar
    ub_ode = control_snake(current_ode.qa_phi,current_ode.qa_phi_d,current_ode.qa_phi_dd,ref_signal_ode.phi,ref_signal_ode.phi_d,ref_signal_ode.phi_dd,control_gains_f)
    error_dynamics = (ref_signal_ode.phi_dd - current_ode.qa_phi_dd) + control_gains_f.kd * (ref_signal_ode.phi_d - current_ode.qa_phi_d) + control_gains_f.kp * (ref_signal_ode.phi - current_ode.qa_phi)
    error_dynamics_t = np.array([error_dynamics_t,error_dynamics])
    #      ub_ode = zeros(N-1,1)  # using to turn off the controller
    qa_ddot = ub_ode
    qu_ddot = A_qphi_qphid + B_qa * ub_ode
    xaf = np.array([[x1],[x2],[x3],[x4]])
    
    xaf_d = np.array([[x3],[x4],[qa_ddot],[qu_ddot]])
    q = xaf
    
    qDot = xaf_d
    
    # Need to change this... append vertical or horizontal ?
    Tt = np.array([Tt,t])
    
    r = np.array([[phirr],[phirr_d],[phirr_dd]])
    
    T_ref_phis_ti = np.array([[T_ref_phis_ti],[phirr]])
    
    T_qa_ti = np.array([T_qa_ti,x1])
    
    T_qu_ti = np.array([T_qu_ti,x2])
    
    T_ubs_ti = np.array([T_ubs_ti,ub_ode])
    
    return qDot
    
    
def fun2(t = None,q = None,phirr = None,phirr_d = None,phirr_dd = None): 
    # Initial values
# q= R^2N+4x1
    x1 = q(np.arange(1,N - 1+1))
    
    x2 = q(np.arange(N,N + 2+1))
    
    x3 = q(np.arange(N + 3,2 * N + 1+1))
    
    x4 = q(np.arange(2 * N + 2,2 * N + 4+1))
    
    q_d = np.array([[x3],[x4],[qa_ddot],[qu_ddot]])
    
    px_dd = qu_ddot(2)
    t
    phi_bar_ode = np.array([[x1],[x2(1)]])
    
    phi_bar_d_ode = np.array([[x3],[x4(1)]])
    
    phi_avg_heading = mean(x1)
    ## Reference sinusoidal wave
    
    ref_signal_ode.phi = np.transpose(phirr)
    
    ref_signal_ode.phi_d = np.transpose(phirr_d)
    
    ref_signal_ode.phi_dd = np.transpose(phirr_dd)
    
    ref_signal_phi_mean = mean(ref_signal_ode.phi)
    current_ode.qa_phi = x1
    
    current_ode.qu = x2
    current_ode.qa_phi_d = x3
    current_ode.qu_d = x4
    theta_ode = H * phi_bar_ode
    
    theta_d_ode = H * phi_bar_d_ode
    
    theta_bar = mean(theta_ode)
    sine_btheta = np.sin(theta_ode)
    
    cose_btheta = np.cos(theta_ode)
    
    S_theta = diag(sine_btheta)
    
    C_theta = diag(cose_btheta)
    
    sgn_theta = np.sign(theta_ode)
    
    theta_d_sq = theta_d_ode ** 2
    
    sine_N = np.sin(theta_ode)
    cose_N = np.cos(theta_ode)
    p_x = x2(2)
    
    p_y = x2(3)
    p = np.array([[p_x],[p_y]])
    
    p_moving = np.array([p_moving,p])
    
    #x4 = q(2*N+2:2*N+4,:) # qu_dot = [theta_nd, pxd, pyd]'
    p_x_d = x4(2)
    p_y_d = x4(3)
    p_d = np.array([[p_x_d],[p_y_d]])
    
    # tangential forward velocity
    v_t_bar = p_x_d * np.cos(theta_bar) + p_y_d * np.cos(theta_bar)
    
    l
    K
    X_ode = - l * np.transpose(K) * cose_N + e * p(1)
    
    Y_ode = - l * np.transpose(K) * sine_N + e * p(2)
    # Define p_x_d and p_y_d
    
    X_d_ode = l * np.transpose(K) * S_theta * theta_d_ode + np.multiply(e,p_d(1))
    
    Y_d_ode = - l * np.transpose(K) * C_theta * theta_d_ode + np.multiply(e,p_d(2))
    
    mu_t = 1
    mu_n = 9
    ## friction model - Viscous friction
    g_f = phy_properties_f.g
    c_n_f = phy_properties_f.c_n
    c_t_f = phy_properties_f.c_t
    #F_rv_all_aniso:: R^2Nx1 # in m/s
    F_rv_all_aniso = - 1 * np.array([[c_t_f * C_theta ** 2 + c_n_f * S_theta ** 2,(c_t_f - c_n_f) * S_theta * C_theta],[(c_t_f - c_n_f) * S_theta * C_theta,c_t_f * S_theta ** 2 + c_n_f * C_theta ** 2]]) * np.array([[X_d_ode],[Y_d_ode]])
    
    # Friction model - Coulomb friction model
    
    #F_rv_all_ani_coulomb = -1*m*g_f*[mu_t.*C_theta, -1*mu_n.*S_theta;...
#                                  mu_t.*S_theta, mu_n.*C_theta]*sgn([C_theta, S_theta; ...
#                                                                    -S_theta, C_theta]*[X_d_ode;Y_d_ode])
    
    # propulsion force calculation
    F_x_theta_i = c_t_f * cose_btheta ** 2 + c_n_f * sine_btheta ** 2
    
    tem = np.multiply(sine_btheta,cose_btheta)
    
    F_y_theta_i = np.multiply((c_t_f - c_n_f),tem)
    
    F_prop_i = (np.multiply(- 1.0 * F_x_theta_i,X_d_ode)) - (np.multiply(F_y_theta_i,Y_d_ode))
    F_prop_total = sum(F_prop_i)
    F_prop_t_px_dd = px_dd * N * m
    
    ## Model of the snake robot
    M_theta = J * np.eye(N) + np.multiply((m * l ** 2),S_theta) * V * S_theta + np.multiply((m * l ** 2),C_theta) * V * C_theta
    
    phi_bar_ode
    
    sine_phi_bar_ode = np.sin(phi_bar_ode)
    
    cose_phi_bar_ode = np.cos(phi_bar_ode)
    
    S_phi_bar_ode = diag(sine_phi_bar_ode)
    
    C_phi_bar_ode = diag(cose_phi_bar_ode)
    
    M_theta_phib = J * np.eye(N) + np.multiply((m * l ** 2),S_phi_bar_ode) * V * S_phi_bar_ode + np.multiply((m * l ** 2),C_phi_bar_ode) * V * C_phi_bar_ode
    
    W_phib = np.multiply((m * l ** 2),S_phi_bar_ode) * V * C_phi_bar_ode - np.multiply((m * l ** 2),C_phi_bar_ode) * V * S_phi_bar_ode
    
    #W = (m*l^2).*S_theta*V*C_theta-(m*l^2).*C_theta*V*S_theta; # R^NxN
    
    # Mb_phib = [H'*M_theta*H, zeros(N,2); zeros(2,N), N*(m*eye(2))];
    Mb_phib = np.array([[np.transpose(H) * M_theta_phib * H,np.zeros((N,2))],[np.zeros((2,N)),N * (m * np.eye(2))]])
    
    # Wb_phib_phibdot = [H'*W*diag(theta_d_ode)*theta_d_ode; zeros(2,1)];
    Wb_phib_phibdot = np.array([[np.transpose(H) * W_phib * diag(H * phi_bar_d_ode) * H * phi_bar_d_ode],[np.zeros((2,1))]])
    
    Gb_phib = np.array([[- l * np.transpose(H) * S_theta * K,l * np.transpose(H) * C_theta * K],[- np.transpose(e),np.zeros((1,N))],[np.zeros((1,N)),- np.transpose(e)]])
    
    M11b = Mb_phib(np.arange(1,N - 1+1),np.arange(1,N - 1+1))
    
    M12b = Mb_phib(np.arange(1,N - 1+1),np.arange(N,N + 2+1))
    
    W1b = Wb_phib_phibdot(np.arange(1,N - 1+1))
    
    G1b = Gb_phib(np.arange(1,N - 1+1),:)
    
    M21b = Mb_phib(np.arange(N,N + 2+1),np.arange(1,N - 1+1))
    
    M22b = Mb_phib(np.arange(N,N + 2+1),np.arange(N,N + 2+1))
    
    W2b = Wb_phib_phibdot(np.arange(N,N + 2+1))
    
    G2b = Gb_phib(np.arange(N,N + 2+1),:)
    
    B_bar = np.array([[np.eye(N - 1)],[np.zeros((3,N - 1))]])
    t
    fR = F_rv_all_aniso
    
    A_qphi_qphid = - 1 * inv(M22b) * (W2b + G2b * fR)
    
    B_qa = - 1 * inv(M22b) * M21b
    
    current_ode.qa_phi_dd = q_d(np.arange(N + 3,2 * N + 1+1),:)
    
    ## Control input u_bar
# new set of control inputs, enables us to rewrite 2.41a&b as
#  ub = [u1b, u2b, u3b]'
#        if t == 0
#              ub_ode = u_bar_f' # ub - column vector with R^N-1
#        else
    ub_ode = control_snake(current_ode.qa_phi,current_ode.qa_phi_d,current_ode.qa_phi_dd,ref_signal_ode.phi,ref_signal_ode.phi_d,ref_signal_ode.phi_dd,control_gains_f)
    error_dynamics = (ref_signal_ode.phi_dd - current_ode.qa_phi_dd) + control_gains_f.kd * (ref_signal_ode.phi_d - current_ode.qa_phi_d) + control_gains_f.kp * (ref_signal_ode.phi - current_ode.qa_phi)
    #      ub_ode = zeros(N-1,1)  # using to turn off the controller
    qa_ddot = ub_ode
    qu_ddot = A_qphi_qphid + B_qa * ub_ode
    xaf = np.array([[x1],[x2],[x3],[x4]])
    
    xaf_d = np.array([[x3],[x4],[qa_ddot],[qu_ddot]])
    q = xaf
    
    qDot = xaf_d
    
    return qDot,F_prop_total,ub_ode,error_dynamics,X_ode,Y_ode,phi_avg_heading,ref_signal_phi_mean,theta_ode
    
    return qDot,F_prop_total,ub_ode,error_dynamics,X_ode,Y_ode,phi_avg_heading,ref_signal_phi_mean,theta_ode
    
    return T,qT