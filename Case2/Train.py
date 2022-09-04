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
import time
from Optim_Config import OptimConfig


if __name__ == "__main__":
    
    #Use cuda
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    print('using device ' + device)
    device = torch.device(device)
    
    #Path to load and save data
    data_path = '....../data/'
    
    #Set training Epoches for Adam
    N_Adam = 50000
    #Set training Epoches for LBFGS
    N_LBFGS = 50000

    start_time = time.time()
    OptimConfig(device,data_path).OptimAndPredi(N_Adam,N_LBFGS)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % (elapsed))
    
    
    
    