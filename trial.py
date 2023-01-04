import numpy as np
p ={}

a = 30*np.pi/180
w  = 50*np.pi/180
d = 40*np.pi/180
T_span= np.arange(0, 10 + 0.1, 0.5)
for i in range(3):
    temp_JA = np.array([])
    for j in range(0, len(T_span)):
        print(T_span[j])
        if 2 <= T_span[j] <= 3:
            phi0 = np.deg2rad(5)
            k = a * np.sin(w * T_span[j] + i * d) + phi0
            temp_JA = np.append([temp_JA], [k])
        elif 5 <= T_span[j] <= 6:
            phi0 = np.deg2rad(-10)
            k = a * np.sin(w * T_span[j] + i * d) + phi0
            temp_JA = np.append([temp_JA], [k])
        else:
            phi0 = 0
            k = a * np.sin(w * T_span[j] + i * d) + phi0
            temp_JA = np.append([temp_JA], [k])
    p["phi"+str(i)] = temp_JA

print(p)


