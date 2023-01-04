# Control file
# Control needs the current values of the state variable 'x' after every
# iteration
# Number of

    
def control_snake(phi = None,phi_d = None,phi_dd = None,phir = None,phir_d = None,phir_dd = None,control_gains_f = None): 
    #PD control
# ub_ode = control_snake(current_ode.qa_phi,current_ode.qa_phi_d, current_ode.qa_phi_dd, ...
#ref_signal_ode.phi, ref_signal_ode.phi_d, ref_signal_ode.phi_dd, control_gains_f)
    kd = control_gains_f.kd
    kp = control_gains_f.kp
    ki = control_gains_f.ki
    # in radiuns
# phi = current_f.qa_phi
# phi_d = current_f.qa_phi_d
# phi_dd = current_f.qa_phi_dd
    
    # phir = ref_signal_f.phi
# phir_d = ref_signal_f.phi_d
# phir_dd = ref_signal_f.phi_dd
# phir =
# u_bar: is a column vector with R^N-1
#u_bar = ki*(phir_dd-phi_dd) + kd* (phir_d - phi_d) + kp* (phir - phi)
    u_bar = (phir_dd) + kd * (phir_d - phi_d) + kp * (phir - phi)
    return u_bar
    
    return u_bar