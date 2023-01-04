import numpy as np
    
def phi_angle_animationV2(snake_para1 = None,t = None): 
    # 17-06: t added as parameter
# To get the values for the entire span just remove t and run.
#snake_para = [alpha, freq_w, delta, joint_offset, t]
#snake_para
    print(snake_para1)
    #fprintf(snake_para1)
    print('I am in phi_angle_animation function')
    a = snake_para1.alpha
    
    w = snake_para1.freq_w
    
    d = snake_para1.delta
    
    phi00 = snake_para1.joint_offset
    
    #heading_angle = snake_para1.heading_angle #cell2mat({snake_para1.heading_angle})
    
    phir_1,phir_2,phir_3,phir_4,phir_5,phir_6,phir_7,phir_8,phir_9 = deal([],[],[],[],[],[],[],[],[])
    phir_1_d,phir_2_d,phir_3_d,phir_4_d,phir_5_d,phir_6_d,phir_7_d,phir_8_d,phir_9_d = deal([],[],[],[],[],[],[],[],[])
    phir_1_dd,phir_2_dd,phir_3_dd,phir_4_dd,phir_5_dd,phir_6_dd,phir_7_dd,phir_8_dd,phir_9_dd = deal([],[],[],[],[],[],[],[],[])
    for i in np.arange(1,len(t)+1).reshape(-1):
        t(i)
        if t(i) >= 20 and t(i) <= 30:
            phi0 = deg2rad(5)
        else:
            if t(i) >= 50 and t(i) <= 60:
                phi0 = deg2rad(- 10)
            else:
                phi0 = deg2rad(0)
        phi0
        phir_1i,phir_2i,phir_3i,phir_4i,phir_5i,phir_6i,phir_7i,phir_8i,phir_9i = deal(a * np.sin(w * t(i)) + phi0,a * np.sin(w * t(i) + d) + phi0,a * np.sin(w * t(i) + 2 * d) + phi0,a * np.sin(w * t(i) + 3 * d) + phi0,a * np.sin(w * t(i) + 4 * d) + phi0,a * np.sin(w * t(i) + 5 * d) + phi0,a * np.sin(w * t(i) + 6 * d) + phi0,a * np.sin(w * t(i) + 7 * d) + phi0,a * np.sin(w * t(i) + 8 * d) + phi0)
        phir_1,phir_2,phir_3,phir_4,phir_5,phir_6,phir_7,phir_8,phir_9 = deal(np.array([phir_1,phir_1i]),np.array([phir_2,phir_2i]),np.array([phir_3,phir_3i]),np.array([phir_4,phir_4i]),np.array([phir_5,phir_5i]),np.array([phir_6,phir_6i]),np.array([phir_7,phir_7i]),np.array([phir_8,phir_8i]),np.array([phir_9,phir_9i]))
        # Joint angular velocities in radiuns/sec
        phir_1_di,phir_2_di,phir_3_di,phir_4_di,phir_5_di,phir_6_di,phir_7_di,phir_8_di,phir_9_di = deal(a * w * np.cos(w * t(i)),a * w * np.cos(w * t(i) + d),a * w * np.cos(w * t(i) + 2 * d),a * w * np.cos(w * t(i) + 3 * d),a * w * np.cos(w * t(i) + 4 * d),a * w * np.cos(w * t(i) + 5 * d),a * w * np.cos(w * t(i) + 6 * d),a * w * np.cos(w * t(i) + 7 * d),a * w * np.cos(w * t(i) + 8 * d))
        phir_1_d,phir_2_d,phir_3_d,phir_4_d,phir_5_d,phir_6_d,phir_7_d,phir_8_d,phir_9_d = deal(np.array([phir_1_d,phir_1_di]),np.array([phir_2_d,phir_2_di]),np.array([phir_3_d,phir_3_di]),np.array([phir_4_d,phir_4_di]),np.array([phir_5_d,phir_5_di]),np.array([phir_6_d,phir_6_di]),np.array([phir_7_d,phir_7_di]),np.array([phir_8_d,phir_8_di]),np.array([phir_9_d,phir_9_di]))
        # Joint angular velocities in radiuns/sec^2
        phir_1_ddi,phir_2_ddi,phir_3_ddi,phir_4_ddi,phir_5_ddi,phir_6_ddi,phir_7_ddi,phir_8_ddi,phir_9_ddi = deal(- a * w ** 2 * np.sin(w * t(i)),- a * w ** 2 * np.sin(w * t(i) + d),- a * w ** 2 * np.sin(w * t(i) + 2 * d),- a * w ** 2 * np.sin(w * t(i) + 3 * d),- a * w ** 2 * np.sin(w * t(i) + 4 * d),- a * w ** 2 * np.sin(w * t(i) + 5 * d),- a * w ** 2 * np.sin(w * t(i) + 6 * d),- a * w ** 2 * np.sin(w * t(i) + 7 * d),- a * w ** 2 * np.sin(w * t(i) + 8 * d))
        phir_1_dd,phir_2_dd,phir_3_dd,phir_4_dd,phir_5_dd,phir_6_dd,phir_7_dd,phir_8_dd,phir_9_dd = deal(np.array([phir_1_dd,phir_1_ddi]),np.array([phir_2_dd,phir_2_ddi]),np.array([phir_3_dd,phir_3_ddi]),np.array([phir_4_dd,phir_4_ddi]),np.array([phir_5_dd,phir_5_ddi]),np.array([phir_6_dd,phir_6_ddi]),np.array([phir_7_dd,phir_7_ddi]),np.array([phir_8_dd,phir_8_ddi]),np.array([phir_9_dd,phir_9_ddi]))
    
    T = t
    phir_r = np.array([np.transpose(phir_1),np.transpose(phir_2),np.transpose(phir_3),np.transpose(phir_4),np.transpose(phir_5),np.transpose(phir_6),np.transpose(phir_7),np.transpose(phir_8),np.transpose(phir_9)])
    
    phir = np.array([np.transpose(phir_1),np.transpose(phir_2),np.transpose(phir_3),np.transpose(phir_4),np.transpose(phir_5),np.transpose(phir_6),np.transpose(phir_7),np.transpose(phir_8),np.transpose(phir_9)]) * 180 / np.pi
    
    phir_d = np.array([np.transpose(phir_1_d),np.transpose(phir_2_d),np.transpose(phir_3_d),np.transpose(phir_4_d),np.transpose(phir_5_d),np.transpose(phir_6_d),np.transpose(phir_7_d),np.transpose(phir_8_d),np.transpose(phir_9_d)])
    phir_dd = np.array([np.transpose(phir_1_dd),np.transpose(phir_2_dd),np.transpose(phir_3_dd),np.transpose(phir_4_dd),np.transpose(phir_5_dd),np.transpose(phir_6_dd),np.transpose(phir_7_dd),np.transpose(phir_8_dd),np.transpose(phir_9_dd)])
    return phir_r,phir_d,phir_dd
    
    return phir_r,phir_d,phir_dd