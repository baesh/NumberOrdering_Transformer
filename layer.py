from torch import nn
import torch
import math
import numpy as np

#Fully Connected layer
class FC(nn.Module):
    def __init__(self, in_size, out_size):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(in_size, out_size, bias= False)

    def forward(self, x):
        return self.fc1(x)

#Feed Forward Neural Network layer
class FFNN(nn.Module):
    def __init__(self, input_size):
        super(FFNN, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.ReLU(),
            nn.Linear(input_size, input_size)
        )
    def forward(self, x):
        return self.layer(x)

#Dense layer
class Dense(nn.Module):
    def __init__(self, in_size):
        super(Dense, self).__init__()
        self.fc1 = nn.Linear(in_size, 1)
    def forward(self, x):
        out = torch.transpose(x, 0, 1)
        out = self.fc1(out)
        out = torch.transpose(out, 0 ,1)
        return out

#Attenton layer(not multihead)
class attention(nn.Module):
    def __init__(self, in_size, out_size):
        super(attention, self).__init__()
        self.Wq = FC(in_size, out_size)
        self.Wk = FC(in_size, out_size)
        self.Wv = FC(in_size, out_size)

    # in self_attention_case, input_tensor1 = input_tensor2
    # in decoder's attention case, input_tensor1 is from decoder's previous layer, input_tensor2 is from encoder's output
    def attention_mat(self, input_tensor1, input_tensor2):
        if input_tensor1.shape[0] == input_tensor2.shape[0] and input_tensor1.shape[1] == input_tensor2.shape[1]:
            input_length = input_tensor1.shape[0]
        else:
            print('error in input size')
            exit()

        Q = self.Wq(input_tensor1)
        K = self.Wk(input_tensor2)
        V = self.Wv(input_tensor2)

        QKt = torch.matmul(Q, K.transpose(0, 1))
        soft = nn.Softmax(dim=1)
        attention_mat = torch.matmul(soft(QKt / math.sqrt(input_length)), V)

        return attention_mat
    def forward(self, x, y):
        return self.attention_mat(x,y)

#Multihead Attention layer
class multihead_attention(nn.Module):
    def __init__(self, in_size, out_size, dim):
        super(multihead_attention, self).__init__()
        self.simple_attention = attention(in_size, out_size*dim)
        self.fully_connect = FC(out_size*dim, out_size)
    def multihead(self, input_tensor1, input_tensor2):
        expanded_attention = self.simple_attention(input_tensor1, input_tensor2)
        multihead = self.fully_connect(expanded_attention)
        return multihead
    def forward(self, x, y):
        return self.multihead(x,y)

#Add & Norm layer
class add_and_norm(nn.Module):
    def __init__(self, input_size):
        super(add_and_norm, self).__init__()
        self.norm = nn.LayerNorm(input_size)
    def forward(self, x, y):
        add = x + y
        result = self.norm(add)
        return result

#Encoder block
class encoder(nn.Module):
    def __init__(self, in_size, multihead_times):
        super(encoder, self).__init__()
        self.multihead = multihead_attention(in_size, in_size, multihead_times)
        self.addnorm = add_and_norm(in_size)
        self.ffnn = FFNN(in_size)
    def forward(self, x, encoder_repeat_times):
        out = x
        for i in range(encoder_repeat_times):
            multiheaded = self.multihead(out, out)
            addnorm1 = self.addnorm(multiheaded, out)
            pointwise_ffnn = self.ffnn(addnorm1)
            out = self.addnorm(pointwise_ffnn, addnorm1)
        return out

# ex) boy = [0,0,0,0,1] --> input size = 5
# ex) ' i am a boy' --> 4 words, input_length = 4
#Decoder block
class decoder(nn.Module):
    #in this case, size of inputs of encoder and decoder are same
    def __init__(self, in_size, in_length, encoder_multihead_times, decoder_multihead_times1, decoder_multihead_times2):
        super(decoder, self).__init__()
        self.multihead_self = multihead_attention(in_size, in_size, decoder_multihead_times1)
        self.addnorm = add_and_norm(in_size)
        self.encoded = encoder(in_size, encoder_multihead_times)
        self.multihead = multihead_attention(in_size, in_size, decoder_multihead_times2)
        self.ffnn = FFNN(in_size)
        self.dense = Dense(in_length)
        self.softmax = nn.Softmax(dim=0)

    def input_mask(self, input_ten, show_range, device):
        if show_range >= (input_ten.shape[0]):
            return input_ten
        else:
            output_ten = torch.zeros(input_ten.shape[0], input_ten.shape[1]).to(device)
            for i in range(show_range):
                for j in range(input_ten.shape[1]):
                    output_ten[i][j] = input_ten[i][j]
            return output_ten
    def forward(self, encoder_input, decoder_input, encoder_repeat, decoder_repeat, show_range, device):
        masked = self.input_mask(decoder_input, show_range, device)
        out = masked
        for i in range(decoder_repeat):
            multihead1 = self.multihead_self(out, out)
            addnorm1 = self.addnorm(multihead1, out)
            encoder_output = self.encoded(encoder_input, encoder_repeat)
            multihead2 = self.multihead(addnorm1, encoder_output)
            addnorm2 = self.addnorm(multihead2, addnorm1)
            pointwise_ffnn = self.ffnn(addnorm2)
            out = self.addnorm(pointwise_ffnn, addnorm2)
        out = self.dense(out)
        out = self.softmax(torch.squeeze(out))

        return out

#Positional Encoding layer
def positional_encoding(tensor, device):
    dim = tensor.shape[1]
    len = tensor.shape[0]
    PE = torch.zeros([len, dim]).to(device)
    for i in range(len):
        for j in range(dim):
            if j%2 == 0:
                PE[i][j] = np.sin(i/np.power(10000, (j/dim)))
            else:
                PE[i][j] = np.cos(i / np.power(10000, ((j-1) / dim)))

    return 0.1*PE + tensor

