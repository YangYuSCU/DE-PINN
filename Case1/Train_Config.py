"""Implementation of NEDP-PINN in Python 
[NEDP-PINN](http://web of our article in arXiv) (A comprehensive numerical study of PINN on
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

import scipy.io




class TrainConfig():
    def __init__(self,device,data_path):
        super(TrainConfig, self).__init__()
        
        
        self.device  = device

        #Path to load and save data
        self.data_path = data_path
        
        #Train data
        Case1_X_Res = scipy.io.loadmat(self.data_path + 'Case1_X_Res.mat')
        self.x_res = Case1_X_Res['x_res']
        
        Case1_X_B = scipy.io.loadmat(self.data_path + 'Case1_X_B.mat')
        self.x_b = Case1_X_B['x_b']
        self.u_b = Case1_X_B['u_b']

        #Equation parameters
        Case1_Equation_Parameter = scipy.io.loadmat(self.data_path + 'Case1_Equation_Parameter.mat')
        self.B_g_square = Case1_Equation_Parameter['B_g_square']
        
        
        #Boundary
        Case1_Boundary = scipy.io.loadmat(self.data_path + 'Case1_Boundary.mat')
        self.lb = Case1_Boundary['lb']
        self.ub = Case1_Boundary['ub']
        
        #Net Size
        self.layers = [1, 20, 20, 20, 20, 1]
        
    def Train_Dict(self):
        return  {'lb':self.lb, 
            'ub':self.ub,
            'x_res': self.x_res,
            'x_b': self.x_b,
            'u_b': self.u_b}
    
    def Param_Dict(self):
        return  {'B_g_square':self.B_g_square,
           'layers': self.layers,
           'data_path':self.data_path,
           'device':self.device}
    

    
    



