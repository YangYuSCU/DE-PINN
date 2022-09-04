"""
@author: yangyu
"""
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


import torch
from NN_Config import PhysicsInformedNN
from Train_Config import TrainConfig
import scipy.io

class OptimConfig():
    def __init__(self,device,data_path):
        super(OptimConfig, self).__init__()   
        
        self.device = device
        
        #Path to load and save data
        self.data_path = data_path
        
        train_dict = TrainConfig(self.device,self.data_path).Train_Dict()
        param_dict = TrainConfig(self.device,self.data_path).Param_Dict()

        #Load PINN
        self.model = PhysicsInformedNN(train_dict=train_dict, param_dict=param_dict)
        self.model.to(device)
        
        
    #Optimize and predict
    def OptimAndPredi(self,n_steps_1,n_steps_2):
        Adam_optimizer = torch.optim.Adam(params=self.model.weights + self.model.biases,
                                            lr=1e-3,
                                            betas=(0.9, 0.999),
                                            eps=1e-8,
                                            weight_decay=0,
                                            amsgrad=False)
        self.model.train_Adam(Adam_optimizer, n_steps_1, None)
        
        LBFGS_optimizer = torch.optim.LBFGS(
            params=self.model.weights + self.model.biases,
            lr=1,
            max_iter=n_steps_2,
            tolerance_grad=-1,
            tolerance_change=-1,
            history_size=100,
            line_search_fn=None)
        self.model.train_LBFGS(LBFGS_optimizer, None)    
    

        Case2_X_Pred = scipy.io.loadmat(self.data_path + 'Case2_X_Pred.mat')
        self.x_pred = Case2_X_Pred['x_pred']
        self.y_pred = Case2_X_Pred['y_pred']
        u_pred = self.model.predict(self.x_pred,self.y_pred) 
        scipy.io.savemat(self.data_path +'Case2_U_Pred.mat', {'u_pred':u_pred})
        
        
