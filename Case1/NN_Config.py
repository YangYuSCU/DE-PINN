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

        lb, ub, x_res, x_b, u_b = self.unzip_train_dict(
            train_dict=train_dict)    

        self.B_g_square = self.data_loader(B_g_square)



        self.lb = self.data_loader(lb, requires_grad=False)
        self.ub = self.data_loader(ub, requires_grad=False)


        self.x_res = self.data_loader(x_res)


        self.x_b = self.data_loader(x_b)
        self.u_b = self.data_loader(u_b, requires_grad=False)

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
                        train_dict['x_b'], 
                        train_dict['u_b'])
        return train_data

    def unzip_param_dict(self, param_dict):
        param_data = (param_dict['B_g_square'],
                      param_dict['layers'],
                      param_dict['data_path'],
                      param_dict['device'])
        return param_data

    def neural_net(self, x, weights, biases):
        num_layers = len(weights) + 1
        X = self.coor_shift(x)
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            X = torch.tanh(torch.add(torch.matmul(X, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = torch.add(torch.matmul(X, W), b)  
        return Y

    def net_u(self, x):
        u = self.neural_net(x, self.weights, self.biases)
        return u

    def grad_u(self, u, x):
        u_x = autograd.grad(u.sum(), x, create_graph=True)[0]
        return u_x

    def grad_2_u(self, u, x):
        u_x = self.grad_u(u,x)
        u_xx = autograd.grad(u_x.sum(), x, create_graph=True)[0]
        return u_x,u_xx



    def forward(self, x):
        u= self.net_u(x)
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
        u_res = self.net_u(x_res)
        u_res_x,u_res_xx = self.grad_2_u(u_res,x_res)
        f_u = u_res_xx * x_res  + self.B_g_square * u_res * x_res + 2*u_res_x
        loss_res = self.loss_func(f_u)

        # Loss_boundary
        loss_boundary = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        loss_boundary.requires_grad_()
        x_b = self.x_b
        u_b  = self.net_u(x_b)
        loss_boundary = self.loss_func(u_b, self.u_b)
            
        # Weights
        alpha_res = 1
        alpha_boundary = 1

        self.loss = loss_res * alpha_res + loss_boundary * alpha_boundary 
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

        

    def predict(self, X_input):
        x = self.data_loader(X_input)
        with torch.no_grad():
            u= self.forward(x)
            return u

