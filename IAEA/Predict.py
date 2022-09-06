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

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.interpolate import griddata


data_path = '....../data'

data_true = scipy.io.loadmat(data_path +'/2DIBPPrior.mat')

data_pred = scipy.io.loadmat(data_path + '/2DIBP_U_Pred_2.mat')

#Get FreeFem prior data
x_true =data_true['x']
y_true =data_true['y']
u_true =data_true['phi1']
v_true =data_true['phi2']

def nptolist(x):
    x=x.reshape((x.shape[0]),)
    x=list(x)
    return x

x_true = nptolist(x_true)
y_true = nptolist(y_true)
u_true = nptolist(u_true)
v_true = nptolist(v_true)

#Create prior dictionary
X_true = list(zip(x_true,y_true))
dictu = dict(zip(X_true, u_true))
dictv = dict(zip(X_true, v_true))


xstar =np.arange(0.0,171.0,1.0)
xstar=xstar.reshape(xstar.shape[0],1)
ystar =np.arange(0.0,171.0,1.0)
ystar=ystar.reshape(ystar.shape[0],1)
X_star, Y_star = np.meshgrid(xstar,ystar)
XY_star = np.hstack((X_star.flatten()[:, None], Y_star.flatten()[:, None]))
Xstar_list =X_star.reshape((X_star.shape[0]*X_star.shape[0]),)
Ystar_list =Y_star.reshape((Y_star.shape[0]*Y_star.shape[0]),)


            
#Search prior dictionary
u_prior = []
v_prior = []
for i in range(len(Xstar_list)):
    if Xstar_list[i] <= Ystar_list[i]:
        Xstar_list[i],Ystar_list[i] =Ystar_list[i],Xstar_list[i]
    if 151<=Xstar_list[i] and Ystar_list[i] >=71:
        u_prior.append(0)
        v_prior.append(0)
    elif 131<=Xstar_list[i]<=150 and Ystar_list[i] >=111:
        u_prior.append(0)
        v_prior.append(0)    
    elif 111<=Xstar_list[i]<=130 and Ystar_list[i] >=131:
        u_prior.append(0)
        v_prior.append(0)
    elif 71<=Xstar_list[i]<=110 and Ystar_list[i] >=151:
        u_prior.append(0)
        v_prior.append(0)
    else:
        u_prior.append(dictu[(Xstar_list[i],Ystar_list[i])])
        v_prior.append(dictv[(Xstar_list[i],Ystar_list[i])]) 
        
        
#Get predicted value
u_pred = data_pred['u_pred']
v_pred = data_pred['v_pred']

u_pred = u_pred.reshape((u_pred.shape[0]*u_pred.shape[1]),)
u_pred =nptolist(u_pred)
v_pred = v_pred.reshape((v_pred.shape[0]*v_pred.shape[1]),)
v_pred =nptolist(v_pred)

XY_pred =  list(zip(Xstar_list,Ystar_list))

#Create predicted dictionary
dictu_pred =dict(zip(XY_pred, u_pred))
dictv_pred =dict(zip(XY_pred, v_pred))
    
#Search predicted dictionary 
u_prediction = []
v_prediction= []
for i in range(len(Xstar_list)):
    if Xstar_list[i] <= Ystar_list[i]:
        Xstar_list[i],Ystar_list[i] =Ystar_list[i],Xstar_list[i]
    if 151<=Xstar_list[i] and Ystar_list[i] >=71:
        u_prediction.append(0)
        v_prediction.append(0)
    elif 131<=Xstar_list[i]<=150 and Ystar_list[i] >=111:
        u_prediction.append(0)
        v_prediction.append(0)    
    elif 111<=Xstar_list[i]<=130 and Ystar_list[i] >=131:
        u_prediction.append(0)
        v_prediction.append(0)
    elif 71<=Xstar_list[i]<=110 and Ystar_list[i] >=151:
        u_prediction.append(0)
        v_prediction.append(0)
    else:
        u_prediction.append(dictu_pred[(Xstar_list[i],Ystar_list[i])])
        v_prediction.append(dictv_pred[(Xstar_list[i],Ystar_list[i])]) 
        

u_prior = np.array(u_prior)
v_prior = np.array(v_prior)
u_prediction = np.array(u_prediction)
v_prediction = np.array(v_prediction)




  

#Plot heatmap
X_star, Y_star = np.meshgrid(xstar,ystar)

U_star = griddata(XY_star, u_prior.flatten(), (X_star, Y_star), method='cubic')
U_pred = griddata(XY_star, u_prediction.flatten(), (X_star, Y_star), method='cubic')

V_star = griddata(XY_star, v_prior.flatten(), (X_star, Y_star), method='cubic')
V_pred = griddata(XY_star, v_prediction.flatten(), (X_star, Y_star), method='cubic')

cset = plt.contourf(X_star, Y_star , U_star)
plt.colorbar(cset)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(data_path + '/U_star.png', dpi=300)
plt.show()

cset = plt.contourf(X_star, Y_star , U_pred)
plt.colorbar(cset)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(data_path + '/U_pred.png', dpi=300)
plt.show()


cset = plt.contourf(X_star, Y_star , np.abs(U_star-U_pred))
plt.colorbar(cset)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(data_path + '/U_error.png', dpi=300)
plt.show()


cset = plt.contourf(X_star, Y_star , V_star)
plt.colorbar(cset)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(data_path + '/V_star.png', dpi=300)
plt.show()

cset = plt.contourf(X_star, Y_star , V_pred)
plt.colorbar(cset)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(data_path + '/V_pred.png', dpi=300)
plt.show()


cset = plt.contourf(X_star, Y_star , np.abs(V_star-V_pred))
plt.colorbar(cset)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig(data_path + '/V_error.png', dpi=300)
plt.show()