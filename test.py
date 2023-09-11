import torch
from functions import test_to_tensor
import layer


device = torch.device('cpu')
#get sample test data and get their shape
unsorted, answer = test_to_tensor(0)
in_size = unsorted.shape[1]
in_length = unsorted.shape[0]
encoder_multihead_times = 4
decoder_multihead_times1 = 4
decoder_multihead_times2 = 4
encoder_repeat_times = 2
decoder_repeat_times = 2
decoder = layer.decoder(in_size, in_length, encoder_multihead_times, decoder_multihead_times1, decoder_multihead_times2)
#get states of model for testing
decoder.load_state_dict(torch.load('model_state/state.pth'))

correct = 0
wrong = 0

for i in range(2000):
    #get test data
    unsorted, answer = test_to_tensor(i)
    in_tensor = torch.zeros(10,10, dtype= torch.float64)
    show_range = 0
    for show_range in range(10):
        result = decoder(unsorted, in_tensor, encoder_repeat_times, decoder_repeat_times, show_range, device)
        sub_ten = torch.zeros(10, dtype= torch.float64)
        sub_ten[torch.argmax(result)] = 1.0
        in_tensor[show_range] = sub_ten

    if torch.equal(in_tensor, answer) == True:
        correct = correct + 1
    else:
        wrong = wrong + 1

print('correct : ', correct, 'wrong : ', wrong)

