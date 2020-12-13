# uniform content loss + adaptive threshold + per_class_input + recursive G
# improvement upon cqf37
from __future__ import division
import numpy as np

print("\n\n\n")

gammas = [100, 250, 300]

gamma_curve = np.zeros((3, 65536), dtype=np.uint16)
g = [[]*6]*3#65536 should be removed? put 6 instead? check g[j] below
for j in range(3):
    #g[j] = [1/(gammas[j]), 4.500000, 0.081243, 0.018054, 0.099297, 0.517181]
    #g[j] = [1/gammas[j], 1.099297 * (np.power(0.018054, 1.0 /gammas[j]) - 0.099297)/0.018054, 
     #       0.081243, 0.018054, 0.099297, 0.517181]
    encoding_gamma = 1 / gammas[j]
    a = np.power(0.018054, 1.0 / encoding_gamma)
    print("a: %.10f" % (a))
    b = 0.018054 / encoding_gamma
    print("b: %.10f" % (b))
    nu = -(a - b)/(a - b - 1)
    print(nu)
    eta = 1 + nu
    alpha = eta / encoding_gamma
    print(alpha)
    g[j] = [1/encoding_gamma, alpha, alpha * 0.018054, 0.018054, nu, 0.517181]
    for i in range(65536):
        if (i/65535 < g[j][3]):#forgot j here
            #gamma_curve[j][i] = int(i/g[j][1])
            #To add noise,
            gamma_curve[j][i] = int(i * g[j][1])
        else:
            gamma_curve[j][i] = int(65535 * (np.power(i/65535., g[j][0])))
            #To add noise
            #gamma_curve[j][i] = int(65535 * (np.power((1.0 * i / 65535. + g[j][4]) / (1.0+g[j][4]), 
             #                                         gammas[j]) ))
            #gamma_curve[j][i] = int(65535 * (np.power((1.0 * i / 65535. + g[j][4]) / (1.0+g[j][4]), 
            #                                          1/ g[j][0]) ))
            #np.power is taking a number instead of a numpy array as input
            
print("Gamma curve:")
#print(gamma_curve[:, 0:3])
#print(gamma_curve[:, 990:993])
#print(gamma_curve[:, 10000:10003])
#print(gamma_curve[:, 34000:34003])
#print(gamma_curve[:, 64000:64003])

print(gamma_curve[:, 0:100:10])
print(gamma_curve[:, 0:1000:100])
print(gamma_curve[:, 0:8000:1000])
print("Every 8K")
print(gamma_curve[:, 0:64000:8000])
print("Every 1K")
print(gamma_curve[:, 0:64000:1000])
print(gamma_curve[:, 65536-8000:65536:1000])
print(gamma_curve[:, 65536-1000:65536:125])
print(gamma_curve[:, 65536-100:65536:10])
print(gamma_curve[:, 65536-10:65536:1])