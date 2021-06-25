import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from arguments import get_args

args = get_args()
if args.drop_type == 'Gauss':
    from dropout.Gauss_dropout import GaussDropout as DropoutLinear
elif args.drop_type == 'AddNoise':
    from dropout.AddNoise import Conv2DAddNoise as DropoutConv2d
    from dropout.AddNoise import LinearAddNoise as DropoutLinear
    
if args.approach == 'ucl_vd':
    from bayes_layer import BayesianLinear as BayesianLinear
elif args.approach == 'vcl_vd':    
    from bayes_layer import BayesianLinearVCL_bias as BayesianLinear

class Net(nn.Module):
    def __init__(self, inputsize, taskcla, ratio, unitN = 400, split = False, notMNIST=False):
        super().__init__()

        ncha, size, _= inputsize
        self.notMNIST = notMNIST
        if notMNIST:
            unitN = 150
        self.taskcla = taskcla
        self.split = split

        self.drop1 = DropoutLinear(ncha*size*size, p= args.droprate)
        self.fc1 = BayesianLinear(28*28, unitN, ratio= ratio)
        self.drop2 = DropoutLinear(unitN, p= args.droprate)
        self.fc2 = BayesianLinear(unitN, unitN, ratio= ratio)
      
        # if notMNIST:
        #     self.fc3 = BayesianLinear(unitN, unitN, ratio= ratio)
        #     self.fc4 = BayesianLinear(unitN, unitN, ratio= ratio)
        self.last = torch.nn.ModuleList()
        
        if split:
            for t,n in self.taskcla:
                self.last.append(torch.nn.Linear(unitN, n))       
        else:
            self.drop3 = DropoutLinear(unitN, p= args.droprate)
            self.fc3 = BayesianLinear(unitN, taskcla[0][1], ratio= ratio)
                
        
    def forward(self, x, sample= False, noise= None):
        x = x.view(-1, 28*28)
        kld = []
        if sample: 
            x, kl = self.drop1(x)
            kld.append(kl)
            x = F.relu(self.fc1(x, sample))

            x, kl = self.drop2(x)
            kld.append(kl)
            x = F.relu(self.fc2(x, sample))

            # if self.notMNIST:
            #     x=F.relu(self.fc3(x, sample))
            #     x=F.relu(self.fc4(x, sample))
            
            if self.split:
                y = []
                for t,i in self.taskcla:
                    y.append(self.last[t](x))
            else:
                x, kl = self.drop3(x)
                kld.append(kl)
                x = self.fc3(x, sample)
                y = F.log_softmax(x, dim= 1)

            return y, sum(kld)
        else:
            x, kl = self.drop1(x, noise= noise['drop1.log_alpha'])
            kld.append(kl)
            x = F.relu(self.fc1(x, sample))
            x, kl = self.drop2(x, noise= noise['drop2.log_alpha'])
            kld.append(kl)
            x = F.relu(self.fc2(x, sample))

            # if self.notMNIST:
            #     x = F.relu(self.fc3(x, sample))
            #     x = F.relu(self.fc4(x, sample))
            
            if self.split:
                y = []
                for t, i in self.taskcla:
                    y.append(self.last[t](x))
            else:
                x, kl = self.drop3(x, noise= noise['drop3.log_alpha'])
                kld.append(kl)
                x = self.fc3(x, sample)
                y = F.log_softmax(x, dim= 1)
          
        return y
       