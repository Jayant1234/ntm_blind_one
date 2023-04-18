import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Parameter
from ntm.utils import _convolve


class Blind_Head(nn.Module):
    def __init__(self, memory, hidden_size):
        super(Blind_Head, self).__init__()
        self.memory = memory
        memory_length, memory_vector_length = memory.get_size()
        self.memory_binary_length= len(bin(memory_vector_length)[3:])# 3: to account for 4 bits for size 16
        # (k : vector, beta: scalar, g: scalar, s: vector, gamma: scalar)
        self.layer_1 = nn.Linear(hidden_size,self.memory_binary_length)
        self.layer_3 = nn.Linear(hidden_size,memory_vector_length)
        #self.layer_3 = nn.Linear(self.memory_binary_length, self.memory_binary_length)
        
        self.layer_2 = nn.Linear(hidden_size, self.memory_binary_length)
        
        for layer in [self.layer_1, self.layer_2]:#, self.layer_3, self.layer_4]:
            nn.init.xavier_uniform_(layer.weight, gain=1.4)
            nn.init.normal_(layer.bias, std=0.01)

        self._initial_state = Parameter(torch.randn(1, self.memory.get_size()[0]) * 1e-5)

    def get_initial_state(self, batch_size):
        # Softmax to ensure weights are normalized
        return F.softmax(self._initial_state, dim=1).repeat(batch_size, 1)

class ReadHead(Blind_Head):

    def forward(self, x, previous_state):
        w = F.relu(self.layer_2(x))# Try sigmoid on it's magnitude next?
        out = (w>0.5).float()
        #print('out:',out)
        key = [[''.join(str(j.item())[0] for j in i) for i in out ]]
        #print("key from readhead:", key)
        memory_read = self.memory.read(key)
        
        return memory_read, w


class WriteHead(Blind_Head):

    def forward(self, x, previous_state):
        #print(x.size())# is it 4 because of batch_size or size of binary vector?
        w = F.relu(self.layer_1(x))# Try sigmoid on it's magnitude next?
        value = F.relu(self.layer_3(x))
        #w = F.relu(self.layer_3(y))
        out = (w>0.5).float()
        #print("out", out)
        pre_keys=out.tolist()
        key=[]
        for item in pre_keys:
            l = [str(int(i)) for i in item]
            key.append(''.join(l))
        #print("key from write head:",key)
        # write to memory (w, memory, e , a)
        self.memory.write(key,value)
        return key
