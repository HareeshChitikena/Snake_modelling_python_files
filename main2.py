# Snake robot modeling and control Ref: snake robotics
# solve ivp: https://pythonnumericalmethods.berkeley.edu/notebooks/chapter22.06-Python-ODE-Solvers.html
# import necessary libraries
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import cm
from numpy.linalg import multi_dot

from control_snake import control_snake

# D = = multi_dot([A, B, C, D])

global qa_ddot, qu_ddot, q
print('Code for snake model and control')


class snake_initiate:
    def __init__(self):
        # Creating dictionary for phy_properties

        self.phy_properties = {
            "l": 0.14 / 2,  # length in meters
            "m": 1,  # mass in kgs
            "N": 10,  # number of links
            "g": 9.8,  # gravity in m/s^2
            "c_n": 10,  # viscous normal friction coefficient
            "c_t": 1,  # viscous transfers friction coefficient
            "mu_n": 0,  # coulomb normal friction coefficient
            "mu_t": 0  # coulomb transfers friction coefficient
        }
        # Creating dictionary snake sinusoidal motion parameters
        self.snake_parameters = {
            "alpha": 30 * np.pi / 180,
            "freq_w": 50 * np.pi / 180,
            "delta": 40 * np.pi / 180,
            "joint_offset": 0 * np.pi / 180,
            "T_span": np.arange(0, 10 + 0.1, 0.1)  # (start, end (not included), increment)
        }
        # Global coordinate system (X_0, Y_0)
        self.X_0 = np.array([0.0])
        self.Y_0 = np.array([0.0])
        self.starting_point_l_1 = np.array([0, 0])  # starting point of link 1 end
        # Initial conditions for phi, phi_d (rad/s), phi_dd (rad/s^2) @ zero second
        # qa_phi = [phi1, phi2, phi3, phi4]  row matrix in radians
        self.qa_phi = np.array(np.zeros([self.phy_properties['N'] - 1, 1]) * np.pi / 180)   # 9 X 1
        self.qa_phi[0] = 20 * np.pi / 180
        self.qa_phi_d = np.zeros([self.phy_properties['N'] - 1, 1])
        self.qa_phi_dd = np.zeros([self.phy_properties['N'] - 1, 1])
        # Theta and theta_dot value for the first or head link
        self.theta_n = np.mat('0.0') * np.pi / 180

        # kk1 = np.mat('0 3; 4 2')
        # print(kk1[2, 3])
        # self.theta_n_d = np.array(np.mat(0.0))
        self.theta_n_d = np.mat('0')
        self.theta_n_dd = np.mat('0.0')
        # initial phi and phi_dot values # row vector for now -- in initial dict it is vertical Reshaped
        self.phi_bar = np.append(self.qa_phi, self.theta_n, 0)
        self.phi_bar_d = np.append(self.qa_phi_d, self.theta_n, 0)
        # initial thetas and theta_dots based on initial conditions
        # H = np.triu(np.ones((phy_properties['N'], phy_properties['N'])))
        self.H = -1 * np.triu(np.ones((self.phy_properties['N'], self.phy_properties['N'])))
        self.H[:, -1] = -1 * self.H[:, -1]
        # print(self.H)
        self.theta = np.matmul(self.H, self.phi_bar)
        self.theta_d = np.matmul(self.H, self.phi_bar_d)
        self.theta_bar = np.mean(self.theta)
        # print(theta[0])
        # print(phi_bar)
        # print('reshape:')
        # print(phi_bar.reshape(-1,1))
        self.x1_i = self.X_0 + self.starting_point_l_1[0]
        self.y1_i = self.Y_0 + self.starting_point_l_1[1]
        self.x1g_t0 = self.x1_i + self.phy_properties['l'] * np.cos(self.theta[0])
        self.y1g_t0 = self.y1_i + self.phy_properties['l'] * np.sin(self.theta[0])
        self.xg_t0 = self.x1g_t0
        self.yg_t0 = self.y1g_t0
        # Defining lambda function for calculating initial c.g. positions of the links
        xig_t0 = lambda xim1_t0, theta1, theta2: np.array(
            xim1_t0 + self.phy_properties['l'] * (np.cos(theta1) + np.cos(theta2)))
        yig_t0 = lambda yim1_t0, theta1, theta2: np.array(
            yim1_t0 + self.phy_properties['l'] * (np.sin(theta1) + np.sin(theta2)))

        for i in range(1, len(self.theta)):
            x = self.xg_t0[i - 1]
            y = self.yg_t0[i - 1]
            t1 = self.theta[i - 1]
            t2 = self.theta[i]
            self.xg_t0 = np.append([self.xg_t0], [xig_t0(x, t1[0], t2[0])])
            self.yg_t0 = np.append([self.yg_t0], [yig_t0(y, t1[0], t2[0])])

        self.xg_t0_d = np.zeros(self.phy_properties['N'] - 1)
        self.yg_t0_d = np.zeros(self.phy_properties['N'] - 1)
        self.xg_t0_dd = np.zeros(self.phy_properties['N'] - 1)
        self.yg_t0_dd = np.zeros(self.phy_properties['N'] - 1)
        self.px_t0 = np.mean(self.xg_t0)
        self.py_t0 = np.mean(self.yg_t0)
        self.p_t0 = np.mat(np.append(self.px_t0, self.py_t0)).reshape(-1, 1)
        self.px_t0_d = np.mean(self.xg_t0_d)
        self.py_t0_d = np.mean(self.yg_t0_d)
        self.p_t0_d = np.mat(np.append(self.px_t0_d, self.py_t0_d)).reshape(-1, 1)
        self.px_t0_dd = np.mean(self.xg_t0_dd)
        self.py_t0_dd = np.mean(self.yg_t0_dd)
        self.p_t0_dd = np.mat(np.append(self.px_t0_dd, self.py_t0_dd)).reshape(-1, 1)

        # self.qu = np.mat('self.theta_n, self.px_t0, self.py_t0')
        self.qu = np.append(self.theta_n, self.p_t0, 0)
        print(self.qu[1])
        self.qu_d = np.append(self.theta_n_d, self.p_t0_d, 0)
        self.qu_dd = np.append(self.theta_n_dd, self.p_t0_dd, 0)
        self.initial_list = [self.x1_i, self.y1_i, self.qa_phi, self.qa_phi_d,
                             self.qa_phi_dd, self.theta_n, self.theta_n_d, self.phi_bar, self.phi_bar_d,
                             self.theta, self.theta_d, self.theta_bar, self.xg_t0, self.yg_t0, self.xg_t0_d,
                             self.yg_t0_d, self.px_t0, self.py_t0, self.px_t0_d, self.py_t0_d, self.qu, self.qu_d,
                             self.qu_dd]
        # initializing dictionary
        self.initial = {'x1_i': '', 'y1_i': '', 'qa_phi': '', 'qa_phi_d': '', 'qa_phi_dd': '', 'theta_n': '',
                        "theta_n_d": "", "phi_bar": "", "phi_bar_d": "", "theta": "", "theta_d": "",
                        "theta_bar": "", "xg_t0": "", "yg_t0": " ", "xg_t0_d": "", "yg_t0_d": "", "px_t0": "",
                        "py_t0": "",
                        "px_t0_d": "", "py_t0_d": "", "qu": "", "qu_d": "", "qu_dd": ""}
        self.x = list(self.initial.keys())
        for i in range(0, len(self.x)):
            self.initial[self.x[i]] = self.initial_list[i]
        # print("The initial position, velocity, and acceleration values for snake:" + str(self.initial))
        self.snake_control_gains = {
            "kp": 2,
            "kd": 1.5,
            "ki": 0
        }
        print("Control gains for snake:" + str(self.snake_control_gains))

    def snake_para(self):
        return self.phy_properties, self.initial, self.snake_parameters, self.snake_control_gains

class plotting:
    @staticmethod
    def plotting(x, y):
        import matplotlib.font_manager as fm
        # Create some data to plot
        # https://towardsdatascience.com/an-introduction-to-making-scientific-publication-plots-with-python-ea19dfa7f51e
        # https://github.com/venkatesannaveen/python-science-tutorial
        x = [1, 2, 3, 4, 5]
        y = [2, 4, 6, 8, 10]

        # Create a figure and axis object
        fig, ax = plt.subplots()
        # Plot the data
        ax.plot(x, y)

        # Add axis labels and a title
        ax.set_xlabel(r'$\mathregular{\lambda}$ (nm)', labelpad=10)
        ax.set_ylabel('Y-axis', labelpad=10)
        ax.set_title('Scientific Plot')
        # Edit the font, font size, and axes width
        mpl.rcParams['font.family'] = 'Avenir'
        plt.rcParams['font.size'] = 18
        plt.rcParams['axes.linewidth'] = 2
        # Generate 2 colors from the 'tab10' colormap
        colors = cm.get_cmap('tab10', 2)
        # Create figure object and store it in a variable called 'fig'
        fig = plt.figure(figsize=(3, 3))  # in inches with default 6.4, 4.8
        # Add axes object to our figure that takes up entire figure
        ax = fig.add_axes([0, 0, 1, 1])
        # Add two axes objects to create a paneled figure
        ax1 = fig.add_axes([0, 0, 1, 0.4])
        ax2 = fig.add_axes([0, 0.6, 1, 0.4])
        ax.xaxis.set_tick_params(which='major', size=10, width=2, direction='in', top='on')
        ax.plot(x, y, linewidth=2, color=colors(0), label='Sample 1')
        # Set the axis limits
        ax.set_xlim(370, 930)
        ax.set_ylim(-0.2, 2.2)
        # Add legend to plot
        ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=False, fontsize=16)

        # Show the plot
        plt.show()

    @staticmethod
    def snake_plotting(time, angle):
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        font_names = [f.name for f in fm.fontManager.ttflist]
        import matplotlib.pylab as pylab
        from pylab import cm
        # params = {'legend.fontsize': 'x-large',
        #           'figure.figsize': (15, 5),
        #           'axes.labelsize': 'x-large',
        #           'axes.titlesize': 'x-large',
        #           'xtick.labelsize': 'x-large',
        #           'ytick.labelsize': 'x-large'}
        # pylab.rcParams.update(params)

        import matplotlib.font_manager as fm
        # Create figure and add axes object
        # fig = plt.figure(figsize=(6, 3))
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_axes([0, 0, 3, 1.5])
        # Edit the major and minor ticks of the x and y axes
        ax.xaxis.set_tick_params(which='major', size=10, width=1, direction='in')
        ax.xaxis.set_tick_params(which='minor', size=7, width=0.5, direction='in')
        ax.yaxis.set_tick_params(which='major', size=10, width=1, direction='in', right='off')
        ax.yaxis.set_tick_params(which='minor', size=7, width=0.5, direction='in', right='off')
        colors = cm.get_cmap('tab10', 10)
        # mpl.rcParams['font.family'] = 'Avenir'
        plt.rcParams['font.size'] = 18
        plt.rcParams['axes.linewidth'] = 1
        # Plot and show our data
        for i in range(1, 11):
            print(len(angle['phi' + str(i)]))
            ax.plot(time, angle['phi' + str(i)], linewidth=1, color=colors(i - 1),
                    label='$\mathregular{\phi}$' + str(i))

        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))
        ax.set_xlabel('Time (sec)', labelpad=10, fontsize=18)
        ax.set_ylabel('Phi (rad)', labelpad=10, fontsize=18)
        # Set the axis limits
        ax.set_xlim(-0.1, 80)
        ax.set_ylim(-0.8, 0.8)
        # Add legend to plot
        ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=True, fontsize=12, ncol=5)
        plt.title("Reference angle graph", fontsize=20)
        # Save figure
        plt.savefig('Reference_phi_plot.png', dpi=400, transparent=False, bbox_inches='tight')
        plt.show()
        # Edit the font, font size, and axes width

    def snake_plot_q_and_ref(phir, T_span, phic, T):
        # plotting.snake_plot_q_and_ref(phir, T_span, x1[:][0], T)
        import matplotlib as mpl
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
        font_names = [f.name for f in fm.fontManager.ttflist]
        import matplotlib.pylab as pylab
        from pylab import cm
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_axes([0, 0, 3, 1.5])
        # Edit the major and minor ticks of the x and y axes
        ax.xaxis.set_tick_params(which='major', size=10, width=1, direction='in')
        ax.xaxis.set_tick_params(which='minor', size=7, width=0.5, direction='in')
        ax.yaxis.set_tick_params(which='major', size=10, width=1, direction='in', right='off')
        ax.yaxis.set_tick_params(which='minor', size=7, width=0.5, direction='in', right='off')
        colors = cm.get_cmap('tab10', 10)
        ax.plot(T_span, phir,  linewidth=1,
                label='$\mathregular{\phi1_reference}$')
        ax.plot(T, phic, linewidth=1,
                label='$\mathregular{\phi1}$')

        ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(10))
        ax.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.1))
        ax.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))
        ax.set_xlabel('Time (sec)', labelpad=10, fontsize=18)
        ax.set_ylabel('Phi (rad)', labelpad=10, fontsize=18)
        # Set the axis limits
        ax.set_xlim(-0.1, 80)
        ax.set_ylim(-0.8, 0.8)
        # Add legend to plot
        ax.legend(bbox_to_anchor=(1, 1), loc=1, frameon=True, fontsize=12, ncol=5)
        plt.title("Reference angle graph", fontsize=20)
        # Save figure
        plt.savefig('Reference_and_model_phi1_vs_time.png', dpi=400, transparent=False, bbox_inches='tight')
        plt.show()


class snake_dynamic_model:
    def __init__(self, phy_properties, initial_values, snake_parameters, control_gains):
        self.snake_parameters = snake_parameters
        self.ll = phy_properties["l"]
        self.m = phy_properties["m"]
        self.N = phy_properties["N"]
        self.J = (self.m * self.ll ** 2) / 3
        # self.dynamic_out()
        self.A = np.eye(self.N - 1, self.N) + np.append(np.zeros(self.N - 1).reshape(self.N - 1, 1),
                                                        np.eye(self.N - 1, self.N - 1), axis=1)
        self.D = np.eye(self.N - 1, self.N) + np.append(np.zeros(self.N - 1).reshape(self.N - 1, 1),
                                                        -1 * np.eye(self.N - 1, self.N - 1), axis=1)
        self.e = np.ones((self.N, 1))
        self.E = np.array([[self.e, np.zeros((self.N, 1))], [np.zeros((self.N, 1)), self.e]])
        self.H = -1 * np.triu(np.ones((self.N, self.N)))
        self.H[:, -1] = -1 * self.H[:, -1]
        self.V = np.matmul(np.transpose(self.A),
                           np.matmul(np.linalg.inv(np.matmul(self.D, np.transpose(self.D))), self.A))
        self.K = np.matmul(np.transpose(self.A),
                           np.matmul(np.linalg.inv(np.matmul(self.D, np.transpose(self.D))), self.D))

        self.g = phy_properties["g"]
        self.c_n = phy_properties["c_n"]
        self.c_t = phy_properties["c_t"]

        self.x1 = np.transpose(initial_values["qa_phi"]).reshape(-1, 1)  # in radians
        self.x2 = initial_values["qu"]  # x2 is column matrix  % in radians (thetaN) and meters (px and py)
        self.x3 = np.transpose(initial_values["qa_phi_d"]).reshape(-1, 1)  # in radians/sec
        self.x4 = initial_values["qu_d"]  # in radians/sec and meters/sec

        self.xaf = np.append(self.x1, np.append(self.x2, np.append(self.x3, self.x4, 0), 0), 0)

        self.T_ref_phis_ti = np.empty((0, 0))
        self.T_qa_ti = np.empty((0, 0))
        self.T_qu_ti = np.empty((0, 0))
        self.T_ubs_ti = np.empty((0, 0))
        self.XY_ode_T = np.empty((0, 0))
        self.forward_total_pro_force_t = np.empty((0, 0))
        self.thetaN_ode = np.empty((0, 0))
        self.q = self.xaf
        self.qa_ddot = np.transpose(initial_values["qa_phi_dd"]).reshape(-1, 1)
        self.qu_ddot = np.transpose(initial_values["qu_dd"]).reshape(-1, 1)
        self.T_span = snake_parameters["T_span"]

        self.k_p = control_gains["kp"]
        self.k_d = control_gains["kd"]
        self.k_i = control_gains["ki"]

        jr_angle, jr_angle_d, jr_angle_dd, T_span = self.phi_angle_generator(snake_parameters)
        jr_array = np.array([jr_angle, jr_angle_d, jr_angle_dd])
        print(jr_array[0])
        # plotting.snake_plotting(self.T_span, jr_angle)
        # snake_plotting(self.T_span, jr_angle)
        tt = 1
        # jr_angle_t, jr_angle_d_t, jr_angle_dd_t = self.phi_angle_generator_at_t(snake_parameters, tt)
        # jr_array = np.array([jr_angle, jr_angle_d, jr_angle_dd])
        # print(j_angle)
        # print(j_angle_t)

        self.t, self.y = self.ode_solver2(self.q)
        print('time values')
        print(self.t)
        print('q values')
        print(self.y)

    def dynamic_out(self):
        return self.t, self.y

    def phi_angle_generator(self, snake_parameters):
        a = snake_parameters["alpha"]
        w = snake_parameters["freq_w"]
        d = snake_parameters["delta"]
        phi0 = snake_parameters["joint_offset"]
        T_span = snake_parameters["T_span"]
        # print(phi0)
        j_angle = {}
        j_angle_d = {}
        j_angle_dd = {}
        for i in range(0, self.N):
            # print(self.N)
            temp_JA = np.array([])
            temp_JA_d = np.array([])
            temp_JA_dd = np.array([])
            for j in range(0, len(T_span)):
                print(T_span[j])

                if 20 <= T_span[j] <= 30:
                    phi0 = np.deg2rad(5)
                    temp_JA = np.append([temp_JA], [a * np.sin(w * T_span[j] + i * d) + phi0])
                    temp_JA_d = np.append([temp_JA_d], [a * w * np.cos(w * T_span[j] + i * d)])
                    temp_JA_dd = np.append([temp_JA_dd], [-1 * a * w ** 2 * np.sin(w * T_span[j] + i * d)])
                elif 50 <= T_span[j] <= 60:
                    phi0 = np.deg2rad(-10)
                    temp_JA = np.append([temp_JA], [a * np.sin(w * T_span[j] + i * d) + phi0])
                    temp_JA_d = np.append([temp_JA_d], [a * w * np.cos(w * T_span[j] + i * d)])
                    temp_JA_dd = np.append([temp_JA_dd], [-1 * a * w ** 2 * np.sin(w * T_span[j] + i * d)])
                else:
                    phi0 = 0
                    temp_JA = np.append([temp_JA], [a * np.sin(w * T_span[j] + i * d) + phi0])
                    temp_JA_d = np.append([temp_JA_d], [a * w * np.cos(w * T_span[j] + i * d)])
                    temp_JA_dd = np.append([temp_JA_dd], [-1 * a * w ** 2 * np.sin(w * T_span[j] + i * d)])
            j_angle["phi" + str(i + 1)] = temp_JA
            j_angle_d["phi_d" + str(i + 1)] = temp_JA_d
            j_angle_dd["phi_dd" + str(i + 1)] = temp_JA_dd
        # print(j_angle['phi2'])
        return j_angle, j_angle_d, j_angle_dd, T_span

    def phi_angle_generator_at_t(self, snake_parameters, t):
        global j_angle_ta, j_angle_d_ta, j_angle_dd_ta
        a = snake_parameters["alpha"]
        w = snake_parameters["freq_w"]
        d = snake_parameters["delta"]
        phi0 = snake_parameters["joint_offset"]
        T_span = snake_parameters["T_span"]
        # print(phi0)
        j_angle_t = {}
        j_angle_d_t = {}
        j_angle_dd_t = {}
        j_angle_ta = np.array([])
        j_angle_d_ta = np.array([])
        j_angle_dd_ta = np.array([])
        for i in range(0, self.N - 1):
            # print(self.N)
            temp_JA = np.array([])
            temp_JA_d = np.array([])
            temp_JA_dd = np.array([])
            if 20 <= t <= 30:
                phi0 = np.deg2rad(5)
                temp_JA = np.append(temp_JA, a * np.sin(w * t + i * d) + phi0)
                temp_JA_d = np.append(temp_JA_d, a * w * np.cos(w * t + i * d))
                temp_JA_dd = np.append(temp_JA_dd, -1 * a * w ** 2 * np.sin(w * t + i * d))
            elif 50 <= t <= 60:
                phi0 = np.deg2rad(-10)
                temp_JA = np.append(temp_JA, a * np.sin(w * t + i * d) + phi0)
                temp_JA_d = np.append(temp_JA_d, a * w * np.cos(w * t + i * d))
                temp_JA_dd = np.append(temp_JA_dd, -1 * a * w ** 2 * np.sin(w * t + i * d))
            else:
                phi0 = 0
                temp_JA = np.append(temp_JA, a * np.sin(w * t + i * d) + phi0)
                temp_JA_d = np.append(temp_JA_d, a * w * np.cos(w * t + i * d))
                temp_JA_dd = np.append(temp_JA_dd, -1 * a * w ** 2 * np.sin(w * t + i * d))

            j_angle_t["phi" + str(i + 1)] = temp_JA
            j_angle_d_t["phi_d" + str(i + 1)] = temp_JA_d
            j_angle_dd_t["phi_dd" + str(i + 1)] = temp_JA_dd
            j_angle_ta = np.append(j_angle_ta, temp_JA)
            j_angle_d_ta = np.append(j_angle_d_ta, temp_JA_d)
            j_angle_dd_ta = np.append(j_angle_dd_ta, temp_JA_dd)

        # print(j_angle_t['phi2'])
        return j_angle_ta, j_angle_d_ta, j_angle_dd_ta

    def control_snake(self, jc_array, jr_array):  # jc - joint current, jr- joint reference
        phir_dd = jr_array[2]
        phi_dd = jc_array[2]
        phir_d = jr_array[1]
        phi_d = jc_array[1]
        phir = jr_array[0]
        phi = jc_array[0]
        u_bar = phir_dd + self.k_d * (phir_d - phi_d) + self.k_p * (phir - phi)
        return u_bar

    def ode_solver2(self, q):
        global q1, qa_ddot, qu_ddot
        qa_ddot = self.qa_ddot
        qu_ddot = self.qu_ddot

        def snake_state_space_model(t, y):
            global qa_ddot, qu_ddot, q1
            # print(t)
            # q = y
            # print(qa_ddot)
            # print(qu_ddot)
            q_ode = y
            # q_ode = q1
            # print(q_ode[1][1])
            #     # print(jr_array)
            # print(self.N)
            x1 = np.array(q_ode[0:self.N - 1]).reshape(-1, 1)  # [phi1, phi2,...phiN]
            x2 = np.array(q_ode[self.N - 1: self.N + 2]).reshape(-1, 1)  # qu = [theta_n, px, py]'
            # print(x2[1])
            x3 = np.array(q_ode[self.N + 2:2 * self.N + 1]).reshape(-1, 1)  # qa_phi_dot = [phi1d,...phiN-1d]'
            x4 = np.array(q_ode[2 * self.N + 1:2 * self.N + 4]).reshape(-1, 1)  # qu_dot = [theta_nd, pxd, pyd]'
            q_d = np.append(x3, np.append(x4, np.append(qa_ddot, qu_ddot, 0), 0), 0)

            # print(q_d)
            phi_bar = np.append(x1, x2[0]).reshape((-1, 1))
            # phi_bar2 = np.array(x1, x2[0])
            phi_bar_d = np.append(x3, x4[0]).reshape(-1, 1)

            jr_angle_t, jr_angle_d_t, jr_angle_dd_t = self.phi_angle_generator_at_t(self.snake_parameters, t)
            jr_array_t = np.array(
                [jr_angle_t.reshape(-1, 1), jr_angle_d_t.reshape(-1, 1), jr_angle_dd_t.reshape(-1, 1)], dtype=object)
            # print(jr_array_t)
            current_qa_phi = x1
            current_qu = x2
            current_qa_phi_d = x3
            current_qu_d = x4
            jc_array_t = np.array([x1, x3, qa_ddot])
            # jc_array_t2 = np.append(x1, np.append(x3, qa_ddot, 0), 0)
            # print(jc_array_t)
            theta_ode = np.matmul(self.H, phi_bar)
            theta_d_ode = np.matmul(self.H, phi_bar_d)

            theta_bar = np.mean(theta_ode)
            sine_btheta = np.sin(theta_ode)
            cose_btheta = np.cos(theta_ode)
            S_theta = np.zeros((self.N, self.N))
            C_theta = np.zeros((self.N, self.N))
            np.fill_diagonal(S_theta, sine_btheta)
            np.fill_diagonal(C_theta, cose_btheta)

            p_x = x2[1]
            p_y = x2[2]
            p = np.array([p_x, p_y])
            p_x_d = x4[1]
            p_y_d = x4[2]
            p_d = np.array([p_x_d, p_y_d])
            #
            X_ode = -self.ll * np.matmul(np.transpose(self.K), cose_btheta) + self.e * p[0]
            Y_ode = -self.ll * np.matmul(np.transpose(self.K), sine_btheta) + self.e * p[1]
            #
            X_d_ode = -self.ll * (np.matmul(np.matmul(np.transpose(self.K), S_theta), theta_d_ode)) + self.e * p_d[0]
            Y_d_ode = -self.ll * (np.matmul(np.matmul(np.transpose(self.K), C_theta), theta_d_ode)) + self.e * p_d[1]
            #
            kk1 = np.append([(self.c_t * np.square(C_theta) + self.c_n * np.square(S_theta))],
                            [(self.c_t - self.c_n) * np.matmul(S_theta, C_theta)], 2)
            kk2 = np.append([(self.c_t - self.c_n) * np.matmul(S_theta, C_theta)],
                            [self.c_t * np.square(S_theta) + self.c_n * np.square(C_theta)], 2)  # [A, B]
            kk = np.append(kk1, kk2, 1)
            kk = -1 * kk
            F_rv_all_aniso = np.matmul(kk, np.append(X_d_ode, Y_d_ode, 0))
            # F_rv_all_aniso = np.matmul(-1 * np.array([[(self.c_t * np.square(C_theta) + self.c_n * np.square(S_theta)), (self.c_t - self.c_n) * np.matmul(S_theta, C_theta)],
            # [(self.c_t - self.c_n) * np.matmul(S_theta, C_theta), self.c_t * np.square(S_theta) + self.c_n * np.square(C_theta)]]), np.append(X_d_ode, Y_d_ode, 0))

            # Model of the snake robot

            M_theta = self.J * np.eye(self.N) + (self.m * self.ll ** 2) * np.matmul(S_theta,
                                                                                    np.matmul(self.V, S_theta)) + (
                              self.m * self.ll ** 2) * np.matmul(C_theta, np.matmul(self.V, C_theta))
            sine_phi_bar_ode = np.sin(phi_bar)
            cose_phi_bar_ode = np.cos(phi_bar)

            S_phi_bar_ode = np.zeros((self.N, self.N))
            C_phi_bar_ode = np.zeros((self.N, self.N))
            np.fill_diagonal(S_phi_bar_ode, sine_phi_bar_ode)
            np.fill_diagonal(C_phi_bar_ode, cose_phi_bar_ode)

            M_theta_phib = self.J * np.eye(self.N) + (
                    self.m * self.ll ** 2) * np.matmul(S_phi_bar_ode, np.matmul(self.V, S_phi_bar_ode)) + (
                                   self.m * self.ll ** 2) * np.matmul(C_phi_bar_ode,
                                                                      np.matmul(self.V, C_phi_bar_ode))  # R^NxN
            W_phib = (self.m * self.ll ** 2) * np.matmul(S_phi_bar_ode, np.matmul(self.V, C_phi_bar_ode)) - (
                    self.m * self.ll ** 2) * np.matmul(C_phi_bar_ode, np.matmul(self.V, S_phi_bar_ode))

            kk1 = np.append([np.matmul(np.transpose(self.H), np.matmul(M_theta_phib, self.H))],
                            [np.zeros([self.N, 2])], 2)  # 10 x 12
            kk2 = np.append([np.zeros([2, self.N])],
                            [self.N * self.m * np.eye(2)], 2)  # [A, B]
            Mb_phib = np.append(kk1, kk2, 1)
            k0 = np.zeros((self.N, self.N))
            np.fill_diagonal(k0, np.matmul(self.H, phi_bar_d))
            # D = = multi_dot([A, B, C, D])
            kk1 = multi_dot([np.transpose(self.H), W_phib, k0, self.H, phi_bar_d])  # 10x1
            # kk1 = np.matmul(np.matmul(np.transpose(self.H), np.matmul(W_phib, np.matmul(k0, np.matmul(self.H, phi_bar_d)))))
            Wb_phib_phibdot = np.append(kk1, np.zeros([2, 1]), 0)

            kk1 = np.append([-self.ll * multi_dot([np.transpose(self.H), S_theta, self.K])],
                            [self.ll * multi_dot([np.transpose(self.H), C_theta, self.K])],
                            2)  # -l*H'*S_theta*K, l*H'*C_theta*K  10 X 20

            kk2 = np.append([-1 * np.transpose(self.e)],
                            [np.zeros([1, self.N])], 2)  # 1 X 20

            kk3 = np.append([np.zeros([1, self.N])], [-1 * np.transpose(self.e)], 2)  # 1 X 20

            kk4 = np.append(kk1, kk2, 1)
            Gb_phib = np.append(kk4, kk3, 1)

            M11b = Mb_phib[0][0:self.N - 1, 0:self.N - 1]  # Mb_phib[0][0][1:4] KKK2= Mb_phib[0][0:9,0:9]
            M12b = Mb_phib[0][0:self.N - 1, self.N - 1:self.N + 2]  # (1:N - 1, N: N + 2)
            W1b = Wb_phib_phibdot[0: self.N - 1]  # (1:N - 1)
            G1b = Gb_phib[0][0:self.N - 1][:]
            #
            M21b = Mb_phib[0][self.N - 1:self.N + 2, 0:self.N - 1]  # (N:N + 2, 1: N - 1);
            M22b = Mb_phib[0][self.N - 1:self.N + 2, self.N - 1:self.N + 2]  # (N:N + 2, N: N + 2);
            W2b = Wb_phib_phibdot[self.N - 1: self.N + 2]  # (N:N + 2);
            G2b = Gb_phib[0][self.N - 1:self.N + 2][:]  # (N:N + 2,:);
            B_bar = np.append([np.eye(self.N - 1)], [np.zeros([3, self.N - 1])], 1)  # [eye(N-1); zeros(3,N-1)];
            #
            px_dd = qu_ddot[1]
            fR = F_rv_all_aniso

            A_qphi_qphid = -1 * np.matmul(np.linalg.inv(M22b), (W2b + np.matmul(G2b, fR)))
            A_qphi_qphid = A_qphi_qphid.reshape((-1, 1))
            B_qa = -1 * np.matmul(np.linalg.inv(M22b), M21b)
            current_qa_phi_d = x3
            ub_ode = self.control_snake(jc_array_t, jr_array_t).reshape(-1, 1)
            #
            qa_ddot = ub_ode
            kkk1 = np.matmul(B_qa, ub_ode).reshape(-1, 1)
            qu_ddot = A_qphi_qphid + np.matmul(B_qa, ub_ode)
            xaf = np.append(x1, np.append(x2, np.append(x3, x4, 0), 0), 0)
            # xaf = np.array([x1, x2, x3, x4], dtype=object)
            xaf_d = np.append(x3, np.append(x4, np.append(qa_ddot, qu_ddot, 0), 0), 0)
            q = xaf
            q = np.array(q.transpose())
            qDot = xaf_d
            qDot = np.array(qDot.transpose())
            y = qDot[0]
            q1 = y
            return y

        t_span = [0, self.T_span[-1]]
        # t_span = [0, 1]  # time span for the integration
        y0 = np.array(q.transpose())  # initial value array with 4 arrays
        # https://stackoverflow.com/questions/61745988/how-to-pass-an-array-to-scipy-integrate-solve-ivp-using-args
        # https://stackoverflow.com/questions/69442104/passing-matrices-as-input-in-scipy-integrate-solve-ivp-python
        # y0 = [1]
        q1 = y0[0]
        sol = solve_ivp(snake_state_space_model, t_span, y0[0], t_eval=np.linspace(0, 10, 101))

        print(sol.t)  # time points at which the solution was computed
        print(sol.y)  # solution values at the time points
        return sol.t, sol.y


snake = snake_initiate()
phy_properties2, initial_values2, snake_parameters2, control_gains2 = snake.snake_para()
dynamic = snake_dynamic_model(phy_properties2, initial_values2, snake_parameters2, control_gains2)
phir, phird, phirdd, T_span = dynamic.phi_angle_generator(snake_parameters2)

T, q = dynamic.dynamic_out()
print(q)
N = phy_properties2['N']
# T_ref = range(0, len(snake_parameters2['T_span']))
# x1 = np.array(q[0:N - 1]).reshape(-1, 1)
x1 = q[0]
# Need to address the plotting.
phir1 = phir['phi1']
plotting.snake_plot_q_and_ref(phir1, T_span, x1, T)

print(T)

