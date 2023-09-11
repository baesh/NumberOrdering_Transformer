import torch

#get train data txt files and convert text to tensor
def data_to_tensor(file_no):
    file1 = open('train_set/unsorted/'+str(file_no)+'_unsorted.txt','r')

    unsorted_li = []
    while True:
        line = file1.readline()
        if not line:
            break
        sub_li = []
        for i in line:
            if i != '\n':
                sub_li.append(int(i))
        unsorted_li.append(sub_li)
    file1.close()

    file2 = open('train_set/sorted/' + str(file_no) + '_sorted.txt', 'r')

    sorted_li = []
    while True:
        line = file2.readline()
        if not line:
            break
        sub_li = []
        for i in line:
            if i != '\n':
                sub_li.append(int(i))
        sorted_li.append(sub_li)
    file1.close()

    return torch.tensor(unsorted_li, dtype=torch.float32), torch.tensor(sorted_li, dtype=torch.float32)

#get test data txt files and convert text to tensor
def test_to_tensor(file_no):
    file1 = open('test_set/unsorted/'+str(file_no)+'_unsorted.txt','r')

    unsorted_li = []
    while True:
        line = file1.readline()
        if not line:
            break
        sub_li = []
        for i in line:
            if i != '\n':
                sub_li.append(int(i))
        unsorted_li.append(sub_li)
    file1.close()

    file2 = open('test_set/sorted/' + str(file_no) + '_sorted.txt', 'r')

    sorted_li = []
    while True:
        line = file2.readline()
        if not line:
            break
        sub_li = []
        for i in line:
            if i != '\n':
                sub_li.append(int(i))
        sorted_li.append(sub_li)
    file1.close()

    return torch.tensor(unsorted_li, dtype=torch.float32), torch.tensor(sorted_li, dtype=torch.float32)
