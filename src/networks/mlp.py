import sys
import torch
import torch.nn.functional as F
from arguments import get_args

args = get_args()

class Net(torch.nn.Module):

    def __init__(self,inputsize,taskcla, unitN = 400, split = False, notMNIST = False):
        super(Net,self).__init__()

        ncha, size, _ = inputsize
        self.notMNIST = notMNIST
        if notMNIST:
            unitN = 150
        self.taskcla = taskcla
        self.split = split
        self.relu = torch.nn.ReLU()
        if not args.wo_Dropout:
            self.drop = torch.nn.Dropout(0.5)
        self.fc1 = torch.nn.Linear(ncha*size*size,unitN)
        self.fc2 = torch.nn.Linear(unitN,unitN)
        
        if notMNIST:
            self.fc3 = torch.nn.Linear(unitN,unitN)
            self.fc4 = torch.nn.Linear(unitN,unitN)
        
        if split:
            self.last = torch.nn.ModuleList()
            for t, n in self.taskcla:
                self.last.append(torch.nn.Linear(unitN, n))
        else:
            self.fc3 = torch.nn.Linear(unitN, taskcla[0][1])

    def forward(self,x):
        h= x.view(x.size(0),-1)
        if args.wo_Dropout:
            h= F.relu(self.fc1(h))
            h= F.relu(self.fc2(h))
            if self.notMNIST:
                h= F.relu(self.fc3(h))
                h= F.relu(self.fc4(h))
        else:
            h= self.drop(F.relu(self.fc1(h)))
            h= self.drop(F.relu(self.fc2(h)))
            if self.notMNIST:
                h= self.drop(F.relu(self.fc3(h)))
                h= self.drop(F.relu(self.fc4(h)))
        
        if self.split:
            y = []
            for t,i in self.taskcla:
                y.append(self.last[t](h))
            
        else:
            y = self.fc3(h)
        
        return y

