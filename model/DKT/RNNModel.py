
import torch
import torch.nn as nn

class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim, device):
        '''
        input_dim: The number of expected features in the input.
        hidden_dim: The number of features in the hidden state of the RNN.
        layer_dim: The number of recurrent layers.
        output_dim: The number of output classes.
        device: The device on which the model will be run (e.g., CPU or GPU).
        '''
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.output_dim = output_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, layer_dim, batch_first=True, nonlinearity='tanh')
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)
        self.sig = nn.Sigmoid()
        self.device = device
        
    

    def forward(self, x):
        '''
        The forward method defines the forward pass of the model. It takes an input tensor x and returns the output tensor res.
        Within the forward method:
        h0 is initialized as a tensor of zeros with the shape (layer_dim, batch_size, hidden_dim) using torch.zeros. This represents the initial hidden state of the RNN.

        '''
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim, device=self.device)
        out, hn = self.rnn(x, h0)
        res = self.sig(self.fc(out))
        return res
