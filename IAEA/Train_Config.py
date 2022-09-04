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

import scipy.io

class TrainConfig():
    def __init__(self,device,data_path):
        super(TrainConfig, self).__init__()
        
        
        self.device  = device

        #Path to load and save data
        self.data_path = data_path
        
        #Train data
        data1 = scipy.io.loadmat(data_path + '/2DIBPRegion1.mat')
        data2 = scipy.io.loadmat(data_path + '/2DIBPRegion2.mat')
        data3 = scipy.io.loadmat(data_path + '/2DIBPRegion3.mat')
        data4 = scipy.io.loadmat(data_path + '/2DIBPRegion4.mat')
        
        #Obtain coordinates in each region
        self.y_1 = data1['y']
        self.x_1 = data1['x']
        self.y_2 = data2['y']
        self.x_2 = data2['x']
        self.y_3 = data3['y']
        self.x_3 = data3['x']
        self.y_4 = data4['y']
        self.x_4 = data4['x']
        
        #Training points on the boundary
        data_b = scipy.io.loadmat(data_path +'/2DIBPBoundary.mat')
        self.y_AB = data_b['y_AB']
        self.x_BC = data_b['x_BC']
        self.y_CD = data_b['y_CD']
        self.x_DE = data_b['x_DE']
        self.y_EF = data_b['y_EF']
        self.x_FG = data_b['x_FG']
        self.y_GH = data_b['y_GH']
        self.x_HI = data_b['x_HI']
        self.y_IJ = data_b['y_IJ']
        self.x_JK = data_b['x_JK']
        self.y_KL = data_b['y_KL']
        self.x_LA = data_b['x_LA']
    

        #Equation parameters
        data_Equation_Parameter = scipy.io.loadmat(self.data_path + '/2DIBP_Equation_Parameter.mat')
        self.D11 = data_Equation_Parameter['D11']
        self.D21 = data_Equation_Parameter['D21']
        self.D12 = data_Equation_Parameter['D12']
        self.D22 = data_Equation_Parameter['D22']
        self.D13 = data_Equation_Parameter['D13']
        self.D23 = data_Equation_Parameter['D23']
        self.D14 = data_Equation_Parameter['D14']
        self.D24 = data_Equation_Parameter['D24']
        self.Sigma121 = data_Equation_Parameter['Sigma121']
        self.Sigma122 = data_Equation_Parameter['Sigma122']
        self.Sigma123 = data_Equation_Parameter['Sigma123']
        self.Sigma124 = data_Equation_Parameter['Sigma124']
        self.Sigmaa11 = data_Equation_Parameter['Sigmaa11']
        self.Sigmaa21 = data_Equation_Parameter['Sigmaa21']
        self.Sigmaa12 = data_Equation_Parameter['Sigmaa12']
        self.Sigmaa22 = data_Equation_Parameter['Sigmaa22']
        self.Sigmaa13 = data_Equation_Parameter['Sigmaa13']
        self.Sigmaa23 = data_Equation_Parameter['Sigmaa23']
        self.Sigmaa14 = data_Equation_Parameter['Sigmaa14']
        self.Sigmaa24 = data_Equation_Parameter['Sigmaa24']
        self.vSigmaf21 = data_Equation_Parameter['vSigmaf21']
        self.vSigmaf22 = data_Equation_Parameter['vSigmaf22']
        self.vSigmaf23 = data_Equation_Parameter['vSigmaf23']
        self.vSigmaf24 = data_Equation_Parameter['vSigmaf24']
        self.Bz1 = data_Equation_Parameter['Bz1']
        self.Bz2 = data_Equation_Parameter['Bz2']

        
        #Upper and lower boundaries of coordinates
        data_lbub = scipy.io.loadmat(self.data_path + '/2DIBP_lbub.mat')
        self.lb = data_lbub['lb']
        self.ub = data_lbub['ub']
        
        #Net Size
        self.layers = [2, 80, 80, 80, 80, 80, 80, 80, 80, 2]


        #Load prior data
        data_prior = scipy.io.loadmat(self.data_path + '/2DIBP_PriorTrain.mat')
        self.x_prior = data_prior['x_prior']
        self.y_prior = data_prior['y_prior']
        self.u_prior = data_prior['u_prior']
        self.v_prior = data_prior['v_prior']    

    def Train_Dict(self):
        return  {'lb':self.lb, 
            'ub':self.ub,
            'y1' : self.y_1,
            'x1' : self.x_1,
            'y2' : self.y_2,
            'x2' : self.x_2,
            'y3' : self.y_3,
            'x3' : self.x_3,
            'y4' : self.y_4,
            'x4' : self.x_4,
            'yAB' : self.y_AB,
            'xBC' : self.x_BC,
            'yCD' : self.y_CD,
            'xDE' : self.x_DE,
            'yEF' : self.y_EF,
            'xFG' : self.x_FG,
            'yGH' : self.y_GH,
            'xHI' : self.x_HI,
            'yIJ' : self.y_IJ,
            'xJK' : self.x_JK,
            'yKL' : self.y_KL,
            'xLA' : self.x_LA,
            'xp':self.x_prior,
            'yp':self.y_prior,
            'up':self.u_prior,
            'vp':self.v_prior
            }

    def Param_Dict(self):
            return  { 'D11': self.D11,
            'D21': self.D21,
            'D12': self.D12,
            'D22': self.D22,
            'D13': self.D13,
            'D23': self.D23,
            'D14': self.D14,
            'D24': self.D24,
            'Sigma121': self.Sigma121,
            'Sigma122': self.Sigma122,
            'Sigma123': self.Sigma123,
            'Sigma124': self.Sigma124,
            'Sigmaa11': self.Sigmaa11,
            'Sigmaa21': self.Sigmaa21,
            'Sigmaa12': self.Sigmaa12,
            'Sigmaa22': self.Sigmaa22,
            'Sigmaa13': self.Sigmaa13,
            'Sigmaa23': self.Sigmaa23,
            'Sigmaa14': self.Sigmaa14,
            'Sigmaa24': self.Sigmaa24,
            'vSigmaf21': self.vSigmaf21,
            'vSigmaf22': self.vSigmaf22,
            'vSigmaf23': self.vSigmaf23,
            'vSigmaf24' : self.vSigmaf24,
            'Bz1' : self.Bz1,
            'Bz2' : self.Bz2,
           'layers': self.layers,
           'data_path':self.data_path,
           'device':self.device}
    

    
    



