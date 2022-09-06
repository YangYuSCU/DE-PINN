"""Implementation of DEPINN in Python 
[DEPINN](https://arxiv.org/abs/2208.13483 of our article in arXiv, GitHub: https://github.com/YangYuSCU/DEPINN) (A data-enabled physics-informed neural network 
with comprehensive numerical study on solving neutron diffusion eigenvalue problems) We present a data-enabled physics-informed neural network (DEPINN) with 
comprehensive numerical study for solving industrial scale neutron diffusion eigenvalue problems (NDEPs). In order to achieve an engineering acceptable accuracy 
for complex engineering problems, a very small amount of prior data from physical experiments are suggested to be used, to improve the accuracy and efficiency of 
training. We design an adaptive optimization procedure with Adam and LBFGS to accelerate the convergence in the training stage. We discuss the effect of different 
physical parameters, sampling techniques, loss function allocation and the generalization performance of the proposed DEPINN model for solving complex problem. 
The feasibility of proposed DEPINN model is tested on three typical benchmark problems, from simple geometry to complex geometry, and from mono-energetic equation 
to two-group equations. Numerous numerical results show that DEPINN can efficiently solve NDEPs with an appropriate optimization procedure. The proposed DEPINN 
can be generalized for other input parameter settings once its structure been trained. This work confirms the possibility of DEPINN for practical engineering 
applications in nuclear reactor physics.

All the computations are carried on NVIDIA A100(80G) and NVIDIA TITAN RTX. 

Author: Yu Yang(yuyang123@stu.scu.edu.cn), Helin Gong*(gonghelin@sjtu.edu.cn), Shiquan Zhang*(shiquanzhang@scu.edu.cn), Qihong Yang(yangqh@stu.scu.edu.cn), 
Zhang Chen(chenzhang208@qq.com), Qiaolin He(qlhejenny@scu.edu.cn), Qing Li(liqing_xueshu@163.com).
Yu Yang, Shiquan Zhang, Qihong Yang and Qiaolin He are with School of Mathematics, Sichuan University.
Helin Gong is with ParisTech Elite Institute of Technology, Shanghai Jiao Tong University.
Zhang Chen and Qing Li are with Science and Technology on Reactor System Design Technology Laboratory,
Nuclear Power Institute of China, 610041, Chengdu, China.

Code author: Yu Yang.
Supported by: Qihong Yang.
Reviewed by: Helin Gong, Qiaolin He and Shiquan Zhang.



Copyright
---------
DEPINN is developed in School of Mathematics, Sichuan University (Yu Yang, Shiquan Zhang, Qihong Yang, Qiaolin He), In collaboration with ParisTech Elite Institute 
of Technology, Shanghai Jiao Tong University (Helin Gong). 
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
        Case2_X_Res = scipy.io.loadmat(self.data_path + 'Case2_X_Res.mat')
        self.x_res = Case2_X_Res['x_res']
        self.y_res = Case2_X_Res['y_res']
        
        
        Case2_X_B = scipy.io.loadmat(self.data_path + 'Case2_X_B.mat')
        self.X_b_left = Case2_X_B['X_b_left']
        self.X_b_right = Case2_X_B['X_b_right']
        self.X_b_down = Case2_X_B['X_b_down']
        self.X_b_up = Case2_X_B['X_b_up']
        

        #Equation parameters
        Case2_Equation_Parameter = scipy.io.loadmat(self.data_path + 'Case2_Equation_Parameter.mat')
        self.B_g_square = Case2_Equation_Parameter['B_g_square']
        
        
        #Boundary
        Case2_Boundary = scipy.io.loadmat(self.data_path + 'Case2_Boundary.mat')
        self.lb = Case2_Boundary['lb']
        self.ub = Case2_Boundary['ub']
        
        #Net Size
        self.layers = [2, 40, 40, 40, 40, 40, 40, 40, 40, 1]
        
    def Train_Dict(self):
        return  {'lb':self.lb, 
            'ub':self.ub,
            'x_res': self.x_res,
            'y_res': self.y_res,
            'X_b_down': self.X_b_down,
            'X_b_up': self.X_b_up,
            'X_b_left': self.X_b_left,
            'X_b_right': self.X_b_right}

    def Param_Dict(self):
        return  {'B_g_square':self.B_g_square,
           'layers': self.layers,
           'data_path':self.data_path,
           'device':self.device}
    

    
    



