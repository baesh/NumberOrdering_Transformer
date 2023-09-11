from matplotlib import pyplot as plt

accuracy_file = open('accuracynloss_during_training/accuracy.txt', 'r')
accuracy_list = []
accuracy_lines = accuracy_file.readlines()
accuracy_x_label = []
for i in accuracy_lines:
    accuracy_list.append(float(i.replace('\n', '')))


for i in range(len(accuracy_list)):
    accuracy_x_label.append(i*10)

loss_file = open('accuracynloss_during_training/loss.txt', 'r')
loss_list = []
loss_lines = loss_file.readlines()
loss_x_label = []
for i in loss_lines:
    loss_list.append(float(i.replace('\n', '')))

for i in range(len(loss_list)):
    loss_x_label.append(i)

plt.subplot(2,1,1)
plt.plot(accuracy_x_label, accuracy_list)
plt.ylabel('accuracy')

plt.subplot(2,1,2)
plt.plot(loss_x_label, loss_list)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()