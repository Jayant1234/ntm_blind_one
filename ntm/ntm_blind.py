import torch
from torch import nn
import torch.nn.functional as F
from ntm.controller import Controller
from ntm.blind_memory import Blind_Memory
from ntm.blind_head import ReadHead, WriteHead


class NTM_blind(nn.Module):
    def __init__(self, vector_length, hidden_size, memory_size, lstm_controller=True):
        super(NTM_blind, self).__init__()
        self.controller = Controller(lstm_controller, vector_length + memory_size[1]+ 1, hidden_size)#  (+ memory_size[1]) changed vector_length+1+memory_size to just vector length+1
        self.Memory = Blind_Memory(memory_size)
        self.read_head = ReadHead(self.Memory, hidden_size)
        self.write_head = WriteHead(self.Memory, hidden_size)
        self.fc = nn.Linear(hidden_size , vector_length)#removed + memory_size[1] from hidden size.
        nn.init.xavier_uniform_(self.fc.weight, gain=1)
        nn.init.normal_(self.fc.bias, std=0.01)

    def get_initial_state(self, batch_size=1):
        self.Memory.reset(batch_size)
        controller_state = self.controller.get_initial_state(batch_size)
        read = self.Memory.get_initial_read(batch_size)
        read_head_state = self.read_head.get_initial_state(batch_size)
        write_head_state = self.write_head.get_initial_state(batch_size)
        return (read, read_head_state, write_head_state, controller_state)

    def forward(self, x, previous_state):
        previous_read, previous_read_head_state, previous_write_head_state, previous_controller_state = previous_state
        #previous_controller_state = previous_state #replaces above line
        controller_input = torch.cat([x, previous_read], dim=1)#changes to just x
        #controller_input = x #for baseline
        controller_output, controller_state = self.controller(controller_input, previous_controller_state)
        #Read
        read_head_output, read_head_state = self.read_head(controller_output, previous_read_head_state)
        # Write
        write_head_state = self.write_head(controller_output, previous_write_head_state)
        #print('Controller:', controller_output.size(),'read_head:', read_head_output)
        fc_input = torch.cat((controller_output, read_head_output), dim=1)
        fc_input = controller_output
        state = (read_head_output, read_head_state, write_head_state, controller_state)
        
        #state = controller_state #for baseline
        
        return F.sigmoid(self.fc(fc_input)), state
