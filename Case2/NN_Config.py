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


import os
import time
import matplotlib
import numpy as np
import torch
from torch import autograd, nn
from torch.autograd import Variable

matplotlib.use('Agg')

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class PhysicsInformedNN(nn.Module):

    def __init__(self, train_dict, param_dict):
        super(PhysicsInformedNN, self).__init__()
        
        #Retrieve data
        B_g_square, self.layers, self.data_path,self.device = self.unzip_param_dict(
            param_dict=param_dict)

        lb, ub, x_res, y_res, X_b_down, X_b_up, X_b_left, X_b_right = self.unzip_train_dict(
            train_dict=train_dict)    

        self.B_g_square = self.data_loader(B_g_square)
        

        self.lb = self.data_loader(lb, requires_grad=False)
        self.ub = self.data_loader(ub, requires_grad=False)


        self.x_res = self.data_loader(x_res)
        self.y_res = self.data_loader(y_res)
        

        self.x_b_down  = self.data_loader(X_b_down[:, 0:1])
        self.y_b_down  = self.data_loader(X_b_down[:, 1:2])
        
        self.x_b_up  = self.data_loader(X_b_up[:, 0:1])
        self.y_b_up  = self.data_loader(X_b_up[:, 1:2])
        
        self.x_b_left  = self.data_loader(X_b_left[:, 0:1])
        self.y_b_left  = self.data_loader(X_b_left[:, 1:2])
        
        self.x_b_right  = self.data_loader(X_b_right[:, 0:1])
        self.y_b_right  = self.data_loader(X_b_right[:, 1:2])
        
        self.weights, self.biases = self.initialize_NN(self.layers)

        self.loss = None
        
        self.loss_fn = torch.nn.MSELoss(reduction='mean')
        
        self.nIter = 0
        self.optimizer = None
        self.optimizer_name = None
        self.scheduler = None

        self.start_time = None

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = Variable(torch.zeros([1, layers[l + 1]],
                                     dtype=torch.float32)).to(self.device)
            b.requires_grad_()
            weights.append(W)
            biases.append(b)
        return weights, biases
    
    def detach(self, data):
        return data.detach().cpu().numpy()
    
    def xavier_init(self, size):
        W = Variable(nn.init.xavier_normal_(torch.empty(size[0], size[1]))).to(
            self.device)
        W.requires_grad_()
        return W

    def data_loader(self, x, requires_grad=True):
        x_tensor = torch.tensor(x,
                                requires_grad=requires_grad,
                                dtype=torch.float32)
        return x_tensor.to(self.device)

    def coor_shift(self, X):
        X_shift = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        return X_shift

    def unzip_train_dict(self, train_dict):
        train_data = (  train_dict['lb'], 
                        train_dict['ub'], 
                        train_dict['x_res'],
                        train_dict['y_res'],
                        train_dict['X_b_down'],
                        train_dict['X_b_up'],
                        train_dict['X_b_left'],
                        train_dict['X_b_right'])
        return train_data

    def unzip_param_dict(self, param_dict):
        param_data = (param_dict['B_g_square'],
                      param_dict['layers'],
                      param_dict['data_path'],
                      param_dict['device'])
        return param_data


    def neural_net(self, x, y, weights, biases):
        num_layers = len(weights) + 1
        X = torch.cat((x, y), 1)
        X = self.coor_shift(X)
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            X = torch.tanh(torch.add(torch.matmul(X, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = torch.add(torch.matmul(X, W), b)
        return Y

    def net_u(self, x, y):
        u = self.neural_net(x, y, self.weights, self.biases)
        return u


    def grad_u(self, u, x):
        u_x = autograd.grad(u.sum(), x, create_graph=True)[0]
        return u_x


    def grad_2_u(self, u, x):
        u_x = self.grad_u(u,x)
        u_xx = autograd.grad(u_x.sum(), x, create_graph=True)[0]
        return u_x,u_xx

    def forward(self, x, y):
        u= self.net_u(x,y)
        return u.detach().cpu().numpy().squeeze()


    def loss_func(self, pred_, true_=None, alpha=1):
        if true_ is None:
            true_ = torch.zeros_like(pred_).to(self.device)
        return alpha * self.loss_fn(pred_, true_)

    def optimize_one_epoch(self):
        if self.start_time is None:
            self.start_time = time.time()

        #Loss function initialization
        self.optimizer.zero_grad()
        self.loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        self.loss.requires_grad_()

        # Loss_res
        loss_res = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        loss_res.requires_grad_()
        x_res = self.x_res
        y_res = self.y_res
        u_res = self.net_u(x_res,y_res)
        u_res_x, u_res_xx= self.grad_2_u(u_res, x_res)
        u_res_y, u_res_yy= self.grad_2_u(u_res, y_res)
        f_u = x_res * (u_res_xx + u_res_yy+ self.B_g_square * u_res ) +  u_res_x 
        loss_res = self.loss_func(f_u)

        # Loss_boundary
        loss_boundary = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        loss_boundary.requires_grad_()
        x_b_down = self.x_b_down
        y_b_down = self.y_b_down
        u_b_down  = self.net_u(x_b_down,y_b_down)
        u_down_y = self.grad_u(u_b_down, y_b_down) 
          
        x_b_up = self.x_b_up
        y_b_up = self.y_b_up
        u_b_up  = self.net_u(x_b_up,y_b_up)    
        
        x_b_left = self.x_b_left
        y_b_left = self.y_b_left
        u_b_left  = self.net_u(x_b_left,y_b_left)
        u_left_x = self.grad_u(u_b_left, x_b_left)
        
        x_b_right = self.x_b_right
        y_b_right = self.y_b_right
        u_b_right  = self.net_u(x_b_right,y_b_right)
         
        loss_boundary = self.loss_func(u_down_y) + self.loss_func(u_left_x) + \
             self.loss_func(u_b_up) +  self.loss_func(u_b_right)

        #Normalization
        loss_nor = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        loss_nor.requires_grad_()
        x_nor = self.lb[:,0:1]
        y_nor = self.lb[:,1:2]
        u_nor  =  self.net_u(x_nor,y_nor) 
        nor_one =  torch.ones_like(u_nor).to(self.device)
        loss_nor = self.loss_func(u_nor,nor_one)
        

        # Weights
        alpha_res = 1
        alpha_boundary = 1
        alpha_nor = 1 
        self.loss = loss_res * alpha_res + loss_boundary * alpha_boundary + alpha_nor * loss_nor
        self.loss.backward()
        self.nIter = self.nIter + 1

        loss_remainder = 10
        if np.remainder(self.nIter, loss_remainder) == 0:
            loss_res = self.detach(loss_res)
            loss_boundary = self.detach(loss_boundary)
            loss = self.detach(self.loss)
            
            log_str = str(self.optimizer_name) + ' Iter ' + str(self.nIter) + ' Loss ' + str(loss) +\
                ' loss_res ' + str(loss_res) + ' loss_boundary ' + str(loss_boundary)
            print(log_str)


            elapsed = time.time() - self.start_time
            print('Iter:', loss_remainder, 'Time: %.4f' % (elapsed))

            self.start_time = time.time()
            
        return self.loss

    def train_LBFGS(self, optimizer, LBFGS_scheduler):
        self.optimizer = optimizer
        self.optimizer_name = 'LBFGS'
        self.scheduler = LBFGS_scheduler

        def closure():
            loss = self.optimize_one_epoch()
            if self.scheduler is not None:
                self.scheduler.step()
            return loss

        self.optimizer.step(closure)

    def train_Adam(self, optimizer, nIter, Adam_scheduler):
        self.optimizer = optimizer
        self.optimizer_name = 'Adam'
        self.scheduler = Adam_scheduler
        for it in range(nIter):
            self.optimize_one_epoch()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step(self.loss)

    def predict(self, x ,y):
        x = self.data_loader(x)
        y = self.data_loader(y)
        with torch.no_grad():
            u= self.forward(x,y)
            return u

