import random
import copy

def num_to_list(len, num):
    li = []
    for i in range(num-1):
        li.append(0)
    li.append(1)
    for j in range(len-num):
        li.append(0)
    return li


list_len = 10

data_number = 2000

for data_no in range(data_number):
    data = []
    for i in range(list_len):
        data.append(random.randint(1,list_len))


    sorted = copy.deepcopy(data)
    sorted.sort()
    # print(data)
    # print(sorted)
    #

    for i in range(len(data)):
        data[i] = num_to_list(list_len, data[i])
        sorted[i] = num_to_list(list_len, sorted[i])
    #
    # print(data)
    # print(sorted)

    #file1 = open('test_set/unsorted/'+str(data_no)+'_unsorted.txt','w')
    for no in data:
        line = ''
        for i in no:
            line = line + str(i)
        line = line + '\n'
        file1.write(line)
    file1.close()

    #file2 = open('test_set/sorted/'+str(data_no)+'_sorted.txt','w')
    for no in sorted:
        line = ''
        for i in no:
            line = line + str(i)
        line = line + '\n'
        file2.write(line)
    file2.close()
