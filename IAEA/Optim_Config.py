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
        #Adam
        Adam_optimizer = torch.optim.Adam(params=self.model.weights + self.model.biases+[self.model.lambda_1],
                                            lr=1e-3,
                                            betas=(0.9, 0.999),
                                            eps=1e-8,
                                            weight_decay=0,
                                            amsgrad=False)
        self.model.train_Adam(Adam_optimizer, n_steps_1, None)
        
        #LBFGS
        LBFGS_optimizer = torch.optim.LBFGS(
            params=self.model.weights + self.model.biases +[self.model.lambda_1],
            lr=1,
            max_iter=n_steps_2,
            tolerance_grad=-1,
            tolerance_change=-1,
            history_size=100,
            line_search_fn=None)
        self.model.train_LBFGS(LBFGS_optimizer, None)    
    
        #Prediction
        Data_X_Pred = scipy.io.loadmat(self.data_path + '/2DIBP_X_Pred.mat')
        self.x_pred = Data_X_Pred['x_pred']
        self.y_pred = Data_X_Pred['y_pred']
        u_pred , v_pred = self.model.predict(self.x_pred,self.y_pred) 
        scipy.io.savemat(self.data_path +'/2DIBP_U_Pred.mat', {'u_pred':u_pred , 'v_pred':v_pred})
        
        
