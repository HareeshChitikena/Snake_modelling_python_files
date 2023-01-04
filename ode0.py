# from scipy.integrate import solve_ivp
# import numpy as np
#
# def f(t, y):
#     print(t)
#     print(y)
#     print(t - y)
#     return t - y
#
# t_span = [0, 1]  # time span for the integration
# y0 = [1]  # initial value
#
# # solve the IVP
# sol = solve_ivp(f, t_span, y0, t_eval=np.linspace(0, 1, 11))
#
# print(sol.t)  # time points at which the solution was computed
# print(sol.y)  # solution values at the time points


import numpy as np
from scipy.integrate import solve_ivp

# Define the system of differential equations
def dydt(t, y):
    print('y')
    print(y)
    dy1dt = y[1]
    print('dy1dt')
    print(dy1dt)
    dy2dt = -y[0] - y[1]
    print('dy2dt')
    print(dy2dt)
    return [dy1dt, dy2dt]

# Set the initial values for the dependent variables
y0 = np.array([1.0, 2.0])

# Set the time range over which to solve the differential equations
t_range = (0.0, 10.0)

# Solve the differential equations
solution = solve_ivp(dydt, t_range, y0)
