import torch
from torch import nn
from torch.nn import Parameter

class Blind_Memory(nn.Module):
    def __init__(self, memory_size):
        super(Blind_Memory, self).__init__()
        self._memory_size = memory_size
        # Initialize memory bias
        self.b= [bin(i)[2:].zfill(len(bin(memory_size[1])[3:])) for i in range(0, memory_size[1])]# 3: is used to deal with memory size 16 resulting in 5 bits, we want 4.
        self.initial_state = torch.ones(memory_size) * 1e-6
        self.memory= dict(zip(self.b, self.initial_state.data))
        # Initial read vector is a learnt parameter
        self.initial_read = Parameter(torch.randn(1, self._memory_size[1]) * 0.01)

    def get_size(self):
        return self._memory_size

    def reset(self, batch_size):
        #print("reset was played......")
        #self.memory = dict(zip(self.b, self.initial_state.data.clone()))# lets' see whats without reset
        num_keys = len(self.memory.keys())
        num_values = len(self.memory.values())
        #print("Memory is being reset and dict has keys,values=",num_keys, num_values)
        #print()
        #print(batch_size)
    def get_initial_read(self, batch_size):
        return self.initial_read.clone().repeat(batch_size, 1)

    def read(self,key):
        #print(self.memory.keys())
        my_values=[]
        for k in key[0]:
            #print('keys:',k)
            #print('values',self.memory[k].dim()) 
            my_values.append(self.memory[k])
        read_value= torch.stack(my_values,dim=0)
        print("Read keys and Read value at that location,",key[0],read_value)
        return read_value

    def write(self,key,value):
        i=0
        for k in key:
            #print("k printed is:", k)
            self.memory[k]=value[i].clone().detach() #this does not work?
            print("Written keys and Memory at that location: ",k,self.memory[k])
            i+=i
        
        return self.memory

    def size(self):
        return self._memory_size
