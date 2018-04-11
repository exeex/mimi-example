from loader import get_data
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToPILImage


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(360,1))
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 120)
        self.fc1_drop = nn.Dropout()
        self.fc2 = nn.Linear(120, 10)
        self.upsample = torch.nn.Upsample(scale_factor=2)

    def forward(self, x):
        x = F.sigmoid(self.conv1(x))
        # print(x)
        x = x.permute([0,2,1,3])
        # print(x.size)
        x = F.sigmoid(self.conv2_drop(x))
        x = self.upsample(x)[:,:,::2,:]
        # print(x)
        return F.sigmoid(x)

    # def num_flat_features(self, x):
    #     size = x.size()[1:]  # all dimensions except the batch dimension
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features


net = Net()
# print(net)
#


optimizer = optim.SGD(net.parameters(),lr=0.1)


x, y = get_data()
x = torch.from_numpy(x).float()
y = torch.from_numpy(y).float()
trainset = torch.utils.data.TensorDataset(x, y)
trainloader = torch.utils.data.DataLoader(trainset, batch_size = 5,
                                          shuffle = True, num_workers = 4)




def train(epoch):
    net.train()
    for batch_index, (inputs, labels) in enumerate(trainloader):
        inputs, labels = Variable(inputs), Variable(labels)
        optimizer.zero_grad()
        outputs = net(inputs)

        # print(outputs)
        loss = F.mse_loss(outputs, labels)
        loss.backward()
        optimizer.step()
        print(batch_index)

        params = list(net.parameters())
        global a2
        a2 = params[0].data.numpy()[:, 0, :, 0]

        if batch_index % 400 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_index * len(inputs), len(trainloader.dataset),
                    100. * batch_index / len(trainloader), loss.data[0]))

        if batch_index % 5 == 0:
            o = net(inputs[0:1, :, :, :])
            plt.imshow(o.data.numpy()[0, 0, :, :])
            plt.show()


for epoch in range(10): # loop over the dataset multiple times
    train(epoch)



    # params = list(net.parameters())
    # print(len(params))
    # print(params[0])
    # # print(params[2].size())
    # # print(params[4].size())
    # a = params[0].data.numpy()[:, 0, :, 0]
    # plt.figure()
    # plt.imshow(params[0].data.numpy()[:, 0, :, 0])
    # plt.show()



