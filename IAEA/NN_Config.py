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
        (lb, ub, x1, y1, x2, y2, x3, y3, x4, y4, yAB, xBC, yCD, xDE, yEF, xFG,
         yGH, xHI, yIJ, xJK, yKL, xLA, xp, yp, up, vp) = self.unzip_train_dict(train_dict)


        (D11, D21, D12, D22, D13, D23, D14,
         D24, Sigma121, Sigma122, Sigma123, Sigma124,
         Sigmaa11, Sigmaa21, Sigmaa12, Sigmaa22,
         Sigmaa13, Sigmaa23, Sigmaa14, Sigmaa24,
         vSigmaf21, vSigmaf22, vSigmaf23, vSigmaf24,
         Bz1, Bz2, self.layers , self.data_path, self.device) = self.unzip_param_dict(param_dict) 



        self.D11 = self.data_loader(D11, requires_grad=False)
        self.D21 = self.data_loader(D21, requires_grad=False)
        self.D12 = self.data_loader(D12, requires_grad=False)
        self.D22 = self.data_loader(D22, requires_grad=False)
        self.D13 = self.data_loader(D13, requires_grad=False)
        self.D23 = self.data_loader(D23, requires_grad=False)
        self.D14 = self.data_loader(D14, requires_grad=False)
        self.D24 = self.data_loader(D24, requires_grad=False)
        self.Sigma121 = self.data_loader(Sigma121, requires_grad=False)
        self.Sigma122 = self.data_loader(Sigma122, requires_grad=False)
        self.Sigma123 = self.data_loader(Sigma123, requires_grad=False)
        self.Sigma124 = self.data_loader(Sigma124, requires_grad=False)
        self.Sigmaa11 = self.data_loader(Sigmaa11, requires_grad=False)
        self.Sigmaa21 = self.data_loader(Sigmaa21, requires_grad=False)
        self.Sigmaa12 = self.data_loader(Sigmaa12, requires_grad=False)
        self.Sigmaa22 = self.data_loader(Sigmaa22, requires_grad=False)
        self.Sigmaa13 = self.data_loader(Sigmaa13, requires_grad=False)
        self.Sigmaa23 = self.data_loader(Sigmaa23, requires_grad=False)
        self.Sigmaa14 = self.data_loader(Sigmaa14, requires_grad=False)
        self.Sigmaa24 = self.data_loader(Sigmaa24, requires_grad=False)
        self.vSigmaf21 = self.data_loader(vSigmaf21, requires_grad=False)
        self.vSigmaf22 = self.data_loader(vSigmaf22, requires_grad=False)
        self.vSigmaf23 = self.data_loader(vSigmaf23, requires_grad=False)
        self.vSigmaf24 = self.data_loader(vSigmaf24, requires_grad=False)
        self.Bz1 = self.data_loader(Bz1, requires_grad=False)
        self.Bz2 = self.data_loader(Bz2, requires_grad=False)



        #Load upper and lower boundaries
        self.lb = self.data_loader(lb, requires_grad=False)
        self.ub = self.data_loader(ub, requires_grad=False)

        #Obtain the coordinates of the 4 regions
        self.x_1 = self.data_loader(x1)
        self.y_1 = self.data_loader(y1)
        self.x_2 = self.data_loader(x2)
        self.y_2 = self.data_loader(y2)
        self.x_3 = self.data_loader(x3)
        self.y_3 = self.data_loader(y3)
        self.x_4 = self.data_loader(x4)
        self.y_4 = self.data_loader(y4)
        
        #Configure boundary points
        X_AB = np.concatenate((0 * yAB + 0, yAB), 1)
        X_BC = np.concatenate((xBC, 0 * xBC + 170), 1)
        X_CD = np.concatenate((0 * yCD + 70, yCD), 1)
        X_DE = np.concatenate((xDE, 0 * xDE + 150), 1)
        X_EF = np.concatenate((0 * yEF + 110, yEF), 1)
        X_FG = np.concatenate((xFG, 0 * xFG + 130), 1)
        X_GH = np.concatenate((0 * yGH + 130, yGH), 1)
        X_HI = np.concatenate((xHI, 0 * xHI + 110), 1)
        X_IJ = np.concatenate((0 * yIJ + 150, yIJ), 1)
        X_JK = np.concatenate((xJK, 0 * xJK + 70), 1)
        X_KL = np.concatenate((0 * yKL + 170, yKL), 1)
        X_LA = np.concatenate((xLA, 0 * xLA + 0), 1)

        self.x_AB = self.data_loader(X_AB[:, 0:1])
        self.y_AB = self.data_loader(X_AB[:, 1:2])
        self.x_BC = self.data_loader(X_BC[:, 0:1])
        self.y_BC = self.data_loader(X_BC[:, 1:2])
        self.x_CD = self.data_loader(X_CD[:, 0:1])
        self.y_CD = self.data_loader(X_CD[:, 1:2])
        self.x_DE = self.data_loader(X_DE[:, 0:1])
        self.y_DE = self.data_loader(X_DE[:, 1:2])
        self.x_EF = self.data_loader(X_EF[:, 0:1])
        self.y_EF = self.data_loader(X_EF[:, 1:2])
        self.x_FG = self.data_loader(X_FG[:, 0:1])
        self.y_FG = self.data_loader(X_FG[:, 1:2])
        self.x_GH = self.data_loader(X_GH[:, 0:1])
        self.y_GH = self.data_loader(X_GH[:, 1:2])
        self.x_HI = self.data_loader(X_HI[:, 0:1])
        self.y_HI = self.data_loader(X_HI[:, 1:2])
        self.x_IJ = self.data_loader(X_IJ[:, 0:1])
        self.y_IJ = self.data_loader(X_IJ[:, 1:2])
        self.x_JK = self.data_loader(X_JK[:, 0:1])
        self.y_JK = self.data_loader(X_JK[:, 1:2])
        self.x_KL = self.data_loader(X_KL[:, 0:1])
        self.y_KL = self.data_loader(X_KL[:, 1:2])
        self.x_LA = self.data_loader(X_LA[:, 0:1])
        self.y_LA = self.data_loader(X_LA[:, 1:2])
        
        # Prior data
        X_p = np.concatenate((xp, yp), 1)
        self.x_p = self.data_loader(X_p[:, 0:1])
        self.y_p = self.data_loader(X_p[:, 1:2])
        self.u_p = self.data_loader(up)
        self.v_p = self.data_loader(vp)

        #The initial value of keff is set to 0.1
        self.lambda_1 = Variable(torch.tensor(0.1)).to(self.device)
        self.lambda_1.requires_grad_()

        self.weights, self.biases = self.initialize_NN(self.layers)

        self.loss = None
        self.loss_total_list = []
        self.loss_fn = torch.nn.MSELoss(reduction='sum')
        
        self.nIter = 0
        self.optimizer = None
        self.optimizer_name = None
        self.scheduler = None

        self.start_time = None

    #Initialize neural network
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
        return data.detach().cpu().numpy().squeeze()
    
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
        train_data = (train_dict['lb'], train_dict['ub'], train_dict['x1'],
                      train_dict['y1'], train_dict['x2'], train_dict['y2'],
                      train_dict['x3'], train_dict['y3'], train_dict['x4'],
                      train_dict['y4'], train_dict['yAB'], train_dict['xBC'],
                      train_dict['yCD'], train_dict['xDE'], train_dict['yEF'],
                      train_dict['xFG'], train_dict['yGH'], train_dict['xHI'],
                      train_dict['yIJ'], train_dict['xJK'], train_dict['yKL'],
                      train_dict['xLA'], train_dict['xp'], train_dict['yp'],
                      train_dict['up'], train_dict['vp'])
        return train_data

    def unzip_param_dict(self, param_dict):
        param_data = (param_dict['D11'], param_dict['D21'], param_dict['D12'],
                      param_dict['D22'], param_dict['D13'], param_dict['D23'],
                      param_dict['D14'], param_dict['D24'],
                      param_dict['Sigma121'], param_dict['Sigma122'],
                      param_dict['Sigma123'], param_dict['Sigma124'],
                      param_dict['Sigmaa11'], param_dict['Sigmaa21'],
                      param_dict['Sigmaa12'], param_dict['Sigmaa22'],
                      param_dict['Sigmaa13'], param_dict['Sigmaa23'],
                      param_dict['Sigmaa14'], param_dict['Sigmaa24'],
                      param_dict['vSigmaf21'], param_dict['vSigmaf22'],
                      param_dict['vSigmaf23'], param_dict['vSigmaf24'],
                      param_dict['Bz1'], param_dict['Bz2'], param_dict['layers'],
                      param_dict['data_path'],param_dict['device'])
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


    #Input x, y ; output u, v
    def net_uv(self, x, y):
        uv = self.neural_net(x, y, self.weights, self.biases)
        u = uv[:, 0:1]
        v = uv[:, 1:2]
        return u, v


    #first-order derivative
    def grad_uv(self, u, v, x, y):
        u_x = autograd.grad(u.sum(), x, create_graph=True)[0]
        u_y = autograd.grad(u.sum(), y, create_graph=True)[0]
        v_x = autograd.grad(v.sum(), x, create_graph=True)[0]
        v_y = autograd.grad(v.sum(), y, create_graph=True)[0]
        return u_x, u_y, v_x, v_y

    #first-order and second-order derivative
    def grad_2_uv(self, u, v, x, y):
        u_x, u_y, v_x, v_y = self.grad_uv(u, v, x, y)
        u_xx = autograd.grad(u_x.sum(), x, create_graph=True)[0]
        u_yy = autograd.grad(u_y.sum(), y, create_graph=True)[0]
        v_xx = autograd.grad(v_x.sum(), x, create_graph=True)[0]
        v_yy = autograd.grad(v_y.sum(), y, create_graph=True)[0]
        return u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy

    def forward(self, x, y):
        u, v = self.net_uv(x, y)
        return self.detach(u), self.detach(v)

    def equation_loss(self, u, v, u_xx, u_yy, v_xx, v_yy, type=1, alpha=1):
        # type = 1, 2, 3, 4 represent different regions
        if 1 == type:
            f_u = -self.D11 * (u_xx + u_yy) + (self.Sigmaa11 + self.Sigma121 + self.D11 * self.Bz1) * u \
                  - (self.lambda_1 * self.vSigmaf21 * v)
            f_v = -self.D21 * (v_xx + v_yy) + (
                self.Sigmaa21 + self.D21 * self.Bz2) * v - (self.Sigma121 * u)
        elif 2 == type:
            f_u = -self.D12 * (u_xx + u_yy) + (self.Sigmaa12 + self.Sigma122 + self.D12 * self.Bz1) * u \
                  - (self.lambda_1 * self.vSigmaf22 * v)
            f_v = -self.D22 * (v_xx + v_yy) + (
                self.Sigmaa22 + self.D22 * self.Bz2) * v - (self.Sigma122 * u)
        elif 3 == type:
            f_u = -self.D13 * (u_xx + u_yy) + (self.Sigmaa13 + self.Sigma123 + self.D13 * self.Bz1) * u \
                  - (self.lambda_1 * self.vSigmaf23 * v)
            f_v = -self.D23 * (v_xx + v_yy) + (
                self.Sigmaa23 + self.D23 * self.Bz2) * v - (self.Sigma123 * u)
        elif 4 == type:
            f_u = -self.D14 * (u_xx + u_yy) + (self.Sigmaa14 + self.Sigma124 + self.D14 * self.Bz1) * u \
                  - (self.lambda_1 * self.vSigmaf24 * v)
            f_v = -self.D24 * (v_xx + v_yy) + (
                self.Sigmaa24 + self.D24 * self.Bz2) * v - (self.Sigma124 * u)
        loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        loss.requires_grad_()
        loss = loss + self.loss_func(f_u, alpha=alpha)
        loss = loss + self.loss_func(f_v, alpha=alpha)
        return loss

    # Loss_res
    def loss_region(self, x, y, type=1, alpha=1):
        u, v = self.net_uv(x, y)
        u_x, u_y, u_xx, u_yy, v_x, v_y, v_xx, v_yy = self.grad_2_uv(u, v, x, y)
        return self.equation_loss(u, v, u_xx, u_yy, v_xx, v_yy, type, alpha=alpha)

    def loss_regions(self, alpha=1):
        loss1 = self.loss_region(self.x_1, self.y_1, type=1, alpha=alpha)
        loss2 = self.loss_region(self.x_2, self.y_2, type=2, alpha=alpha)
        loss3 = self.loss_region(self.x_3, self.y_3, type=3, alpha=alpha)
        loss4 = self.loss_region(self.x_4, self.y_4, type=4, alpha=alpha)
        return loss1, loss2, loss3, loss4

    #Loss_prior
    def loss_prior(self, x_p, y_p, u_p, v_p, alpha=1):
        u, v = self.net_uv(x_p, y_p)
        loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        loss.requires_grad_()
        loss = loss + self.loss_func(u, u_p, alpha=alpha)
        loss = loss + self.loss_func(v, v_p, alpha=alpha)
        return loss

    #Loss_boundary
    def loss_func_boundary(self, x, y, type=0, alpha=1):
        # type=1 indicates the lower boundary
        # type=2 indicates the left boundary
        # type=3 indicates the upper boundary
        # type=2 indicates the right boundary
        u, v = self.net_uv(x, y)
        u_x, u_y, v_x, v_y = self.grad_uv(u, v, x, y)
        loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        loss.requires_grad_()
        if 1 == type:
            loss = loss + self.loss_func(u_y, alpha=alpha)
            loss = loss + self.loss_func(v_y, alpha=alpha)
        elif 2 == type:
            loss = loss + self.loss_func(u_x, alpha=alpha)
            loss = loss + self.loss_func(v_x, alpha=alpha)
        elif 3 == type:
            loss = loss + self.loss_func(u_y + 0.4692 / self.D14 * u, alpha=alpha)
            loss = loss + self.loss_func(v_y + 0.4692 / self.D24 * v, alpha=alpha)
        elif 4 == type:
            loss = loss + self.loss_func(u_x + 0.4692 / self.D14 * u, alpha=alpha)
            loss = loss + self.loss_func(v_x + 0.4692 / self.D24 * v, alpha=alpha)
        return loss

    def loss_boundary(self, alpha=1):
        loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        loss.requires_grad_()
        loss = loss + self.loss_func_boundary(
            self.x_AB, self.y_AB, type=2, alpha=alpha)
 
        loss = loss + self.loss_func_boundary(
            self.x_BC, self.y_BC, type=3, alpha=alpha)

        loss = loss + self.loss_func_boundary(
            self.x_CD, self.y_CD, type=4, alpha=alpha)

        loss = loss + self.loss_func_boundary(
            self.x_DE, self.y_DE, type=3, alpha=alpha)

        loss = loss + self.loss_func_boundary(
            self.x_EF, self.y_EF, type=4, alpha=alpha)

        loss = loss + self.loss_func_boundary(
            self.x_FG, self.y_FG, type=3, alpha=alpha)

        loss = loss + self.loss_func_boundary(
            self.x_GH, self.y_GH, type=4, alpha=alpha)

        loss = loss + self.loss_func_boundary(
            self.x_HI, self.y_HI, type=3, alpha=alpha)
 
        loss = loss + self.loss_func_boundary(
            self.x_IJ, self.y_IJ, type=4, alpha=alpha)
 
        loss = loss + self.loss_func_boundary(
            self.x_JK, self.y_JK, type=3, alpha=alpha)
 
        loss = loss + self.loss_func_boundary(
            self.x_KL, self.y_KL, type=4, alpha=alpha)
  
        loss = loss + self.loss_func_boundary(
            self.x_LA, self.y_LA, type=1, alpha=alpha)
        return loss


    def loss_func(self, pred_, true_=None, alpha=1):
        if true_ is None:
            true_ = torch.zeros_like(pred_).to(self.device)
        return alpha * self.loss_fn(pred_, true_)


    def optimize_one_epoch(self):
        if self.start_time is None:
            self.start_time = time.time()

        # Loss Weights
        alpha_boundary = 1
        alpha_region = 1
        alpha_prior = 10

        self.optimizer.zero_grad()
        self.loss = torch.tensor(0.0, dtype=torch.float32).to(self.device)
        self.loss.requires_grad_()


        loss_boundary = self.loss_boundary(alpha=alpha_boundary)
        self.loss = self.loss + loss_boundary
        loss_boundary = self.detach(loss_boundary) / alpha_boundary


        loss1, loss2, loss3, loss4 = self.loss_regions(alpha=alpha_region)
        loss_regions = loss1 + loss2 + loss3 + loss4
        self.loss = self.loss + loss_regions
        loss_regions = self.detach(loss_regions) / alpha_region


        loss_prior = self.loss_prior(self.x_p,
                                     self.y_p,
                                     self.u_p,
                                     self.v_p,
                                     alpha=alpha_prior)
        self.loss = self.loss + loss_prior
        loss_prior = self.detach(loss_prior) / alpha_prior

        self.loss.backward()
        self.nIter = self.nIter + 1
        loss = self.detach(self.loss)
        self.loss_total_list.append(loss)


        loss_remainder = 10
            
        if np.remainder(self.nIter, loss_remainder) == 0:

            loss = self.detach(self.loss)
            lambda_1_value = self.detach(self.lambda_1)
            # print loss and Keff
            log_str1 = str(self.optimizer_name) + ' Iter:' + \
                str(self.nIter) + ' Loss:' + str(loss) + \
                ' Keff:' + str(1/lambda_1_value)
            print(log_str1)

            #print the values of different losses
            log_str2  = 'Loss of boundary:' + str(loss_boundary) + \
                         ' Loss of regions:' + str(loss_regions) + \
                         ' Loss of prior:' + str(loss_prior)
            print(log_str2)

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
        diff_mean = []
        for it in range(nIter):
            self.optimize_one_epoch()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step(self.loss)
            if (len(self.loss_total_list)-1) % 2000 == 0 and (len(self.loss_total_list)-1) // 2000 >= 1:
                diff = [np.abs(self.loss_total_list[i]-self.loss_total_list[i+1]) for i in range(len(self.loss_total_list)-1)]
                diff_mean.append(np.mean(diff[(2000*((len(self.loss_total_list)-1) // 2000)-2000) :(2000*((len(self.loss_total_list)-1) // 2000))]))
                if len(diff_mean)>=2 and abs(diff_mean[-1]-diff_mean[-2])<0.1:
                    break   

    def predict(self, x ,y):
        x = self.data_loader(x)
        y = self.data_loader(y)
        with torch.no_grad():
            u,v= self.forward(x,y)
            return u,v

