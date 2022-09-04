"""Implementation of NEDP-PINN in Python 
[NEDP-PINN](https://arxiv.org/abs/2208.13483 of our article in arXiv) (A comprehensive numerical study of PINN on
solving neutron diffusion eigenvalue problem) is an PINN algorithm published by
Yu Yang, Helin Gong*, Shiquan Zhang*, Qihong Yang, Zhang Chen, Qiaolin He, Qing Li in 2022.
With this code, a comprehensive numerical study of PINN for solving neutron diffusion
eigenvalue problem (NDEP) is brought out. A very small amount of prior points are used
to improve the accuracy and efficiency of training, which can be obtained from physical
experiments. We design an adaptive optimization procedure for training and also discuss
the parameter dependence in the equations.Numerical results show that PINN with a few
prior data can efficiently solve the problem with an appropriate optimization procedure
for different parameters. The work confirms the possibility of PINN for practical
engineering applications in nuclear reactor physics.

All the computations are carried on NVIDIA A100(80G) and NVIDIA TITAN RTX. 

Author: Yu Yang, Helin Gong*, Shiquan Zhang*, Qihong Yang, Zhang Chen, Qiaolin He, Qing Li.
Yu Yang, Shiquan Zhang, Qihong Yang and Qiaolin He are with School of Mathematics, Sichuan University.
Helin Gong is with ParisTech Elite Institute of Technology, Shanghai Jiao Tong University.
Zhang Chen and Qing Li are with Science and Technology on Reactor System Design Technology Laboratory,
Nuclear Power Institute of China, 610041, Chengdu, China.

Code author: Yu Yang.
Supported by: Qihong Yang.
Reviewed by: Helin Gong, Qiaolin He and Shiquan Zhang.

Copyright
---------
NEDP-PINN is developed in School of Mathematics, Sichuan University, In collaboration with
ParisTech Elite Institute of Technology, Shanghai Jiao Tong University. 
More information about the technique can be found through corresponding authors.
 
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy import special
from scipy.interpolate import griddata

if __name__ == "__main__":
    
    
    #Path to load and save data
    data_path = '....../data/'
    
    #Equation parameters
    sigma_a = 0.45
    sigma_s = 2
    sigma_f = 0.5
    vsigma_f = 2.5
    d = 0.7104
    D = 1/(3*(sigma_s + sigma_a))
    B_g_square = (vsigma_f - sigma_a)/D
    R_e_square = (np.pi**2 + 2.405**2) / B_g_square
    R_e = R_e_square ** (1/2)
    R = R_e - d
    H_e = R_e 
    AC = 1/special.jv(0,0)

    # Analytical solution
    def Exact_u_func(x,y):
        u = AC*special.jv(0,(2.405/R_e)*x)*np.cos((np.pi/H_e)*y)
        return u

    #Predict data
    Case2_X_Pred = scipy.io.loadmat(data_path + 'Case2_X_Pred.mat')
    x_star = Case2_X_Pred['x_pred']
    y_star = Case2_X_Pred['y_pred']
    u_star = Exact_u_func(x_star,y_star)
    
    X_grid = x_star.reshape(200,200)
    Y_grid = y_star.reshape(200,200)
    X_star = np.hstack((x_star,y_star))

    Case2_U_Pred = scipy.io.loadmat(data_path + 'Case2_U_Pred.mat')
    u_pred = Case2_U_Pred['u_pred'].flatten()[:, None]


    U_star = griddata(X_star, u_star.flatten(), (X_grid, Y_grid), method='cubic')
    U_pred = griddata(X_star, u_pred.flatten(), (X_grid, Y_grid), method='cubic')

    # Error
    RE_Linfinity = np.linalg.norm(u_star-u_pred,np.inf) /np.linalg.norm(u_star, np.inf)
    RE_L2 = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    print('RE_Linfinity: %e' % (RE_Linfinity))
    print('RE_L2: %e' % (RE_L2))


    #Plot
    cset = plt.contourf(X_grid, Y_grid , U_star)
    plt.colorbar(cset)
    plt.xlabel('r')
    plt.ylabel('z')
    plt.savefig(data_path + 'Case2_True.png', dpi=300)
    plt.show()

    cset = plt.contourf(X_grid, Y_grid , U_pred)
    plt.colorbar(cset)
    plt.xlabel('r')
    plt.ylabel('z')
    plt.savefig(data_path + 'Case2_Pred.png', dpi=300)
    plt.show()

    cset = plt.contourf(X_grid, Y_grid , np.abs(U_star-U_pred))
    plt.colorbar(cset)
    plt.xlabel('r')
    plt.ylabel('z')
    plt.savefig(data_path + 'Case2_Error.png', dpi=300)
    plt.show()