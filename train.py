import torch
import os
from torch import nn
import random

from functions import data_to_tensor
from functions import test_to_tensor
import layer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#get sample train data and get their shape
encoder_in_ten_sample, decoder_in_ten_sample = data_to_tensor(0)
encoder_in_ten_sample = encoder_in_ten_sample.to(device)
decoder_in_ten_sample = decoder_in_ten_sample.to(device)
in_size = encoder_in_ten_sample.shape[1]
in_length = encoder_in_ten_sample.shape[0]


encoder_multihead_times = 4
decoder_multihead_times1 = 4
decoder_multihead_times2 = 4
encoder_repeat = 2
decoder_repeat = 2
learning_rate = 0.001
decoder = layer.decoder(in_size, in_length, encoder_multihead_times, decoder_multihead_times1, decoder_multihead_times2).to(device)
optim = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

#if there are previous save data, load them
if os.path.isfile('model_state/state.pth'):
    print('continue from previous data')
    decoder.load_state_dict(torch.load('model_state/state.pth', map_location=device))
else:
    print('new_start')
if os.path.isfile('model_state/optim.pth'):
    print('optimizer continued')
    optim.load_state_dict(torch.load('model_state/optim.pth', map_location=device))
else:
    print('new_optimizer')


epoch = 90000
minibatch = 10
data_no = 100000
loss_li = []
accuracy_li = []

for i in range(int(epoch/minibatch)):
    optim.zero_grad()
    loss_sum = 0
    for j in range(minibatch):
        input_no = random.randint(0, data_no-1)
        # get train data
        encoder_in_ten, decoder_in_ten = data_to_tensor(input_no)
        encoder_in_ten = encoder_in_ten.to(device)
        decoder_in_ten = decoder_in_ten.to(device)

        # encoder_in_ten = layer.positional_encoding(encoder_in_ten, device)
        # decoder_in_ten = layer.positional_encoding(decoder_in_ten, device)

        for show_range in range(in_length):
            result = decoder(encoder_in_ten, decoder_in_ten, encoder_repeat, decoder_repeat, show_range, device)
            bce = nn.MSELoss()
            loss = bce(result ,decoder_in_ten[show_range])
            loss_sum = loss_sum + loss


    loss_sum.backward()
    optim.step()

    print(loss_sum)
    #save loss_sum to loss.txt
    loss_file = open('accuracynloss_during_training/loss.txt','a')
    loss_file.write(str(loss_sum.item()))
    loss_file.write('\n')
    loss_file.close()

    #for certain period, test the accuracy of the model
    if i%10 == 0:
        correct = 0
        wrong = 0

        for test_no in range(2000):
            unsorted, answer = test_to_tensor(test_no)
            unsorted = unsorted.to(device)
            answer = answer.to(device)
            in_tensor = torch.zeros(10, 10, dtype=torch.float64).to(device)
            show_range = 0
            for show_range in range(10):
                test_result = decoder(unsorted, in_tensor, encoder_repeat, decoder_repeat, show_range, device).detach()
                sub_ten = torch.zeros(10, dtype=torch.float64).to(device)
                sub_ten[torch.argmax(test_result)] = 1.0
                in_tensor[show_range] = sub_ten

            if torch.equal(in_tensor, answer) == True:
                correct = correct + 1
            else:
                wrong = wrong + 1

        print('correct : ', correct, 'wrong : ', wrong)

        #save the tested result to accuracy.txt
        accuracy_file = open('accuracynloss_during_training/accuracy.txt', 'a')
        accuracy_file.write(str(correct/(correct+wrong)))
        accuracy_file.write('\n')
        accuracy_file.close()

    #save the state of decoder and optimizer
    torch.save(decoder.state_dict(), 'model_state/state.pth')
    torch.save(optim.state_dict(), 'model_state/optim.pth')
