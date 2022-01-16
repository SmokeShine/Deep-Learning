import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from data import get_CIFAR10_data
import numpy as np

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu=nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8192, 10)
        np.random.seed(1024)
        self.fc1.weight.data = torch.FloatTensor(1e-3 * np.random.randn(8192, 10)).T
        np.random.seed(1024)
        self.fc1.bias.data = torch.FloatTensor(np.zeros(10))

        np.random.seed(1024)
        self.conv1.weight.data = torch.FloatTensor(1e-3 * np.random.randn(32, 3, 5, 5))
        self.conv1.bias.data = torch.FloatTensor(np.zeros(32))
        
        self.register_hook = False
        self.hook = {'conv1':[], 'relu':[], 'pool':[],'flat':[],'fc1':[]}

    def hook_fn(self, grad, name):
        self.hook[name].append(grad)
    def reset_hook(self):
        self.hook = {'conv1':[], 'relu':[], 'pool':[],'flat':[],'fc1':[]}
    
    def forward(self, x):
        step1 = self.conv1(x)
        step2 = self.relu(step1)
        step3 = self.pool(step2)
        step3_flatten=step3.reshape(len(x),-1)
        step4 = self.fc1(step3_flatten)

        if self.register_hook:
            step1.register_hook(lambda grad: self.hook_fn(grad=grad, name='conv1'))
            step1.retain_grad()

            step2.register_hook(lambda grad: self.hook_fn(grad=grad, name='relu'))
            step2.retain_grad()

            step3.register_hook(lambda grad: self.hook_fn(grad=grad, name='pool'))
            step3.retain_grad()

            step3_flatten.register_hook(lambda grad: self.hook_fn(grad=grad, name='flat'))
            step3.retain_grad()

            step4.register_hook(lambda grad: self.hook_fn(grad=grad, name='fc1'))
            step4.retain_grad()
        return step4


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9,weight_decay=0.001)

root = 'data/cifar-10-batches-py'
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data(root)

# np.random.seed(1)
X,y=X_train[:50], y_train[:50]
batch_size=10
num_epochs=10
N = X.shape[0]
iterations_per_epoch = N // batch_size  # using SGD
epoch = 0
num_iters = num_epochs * iterations_per_epoch
acc_frequency=None
verbose=True
learning_rate_decay=0.95
sample_batches=True

# optimizer.zero_grad()
mask_file = open("mask.pkl", "rb")
import pickle
mask_history=pickle.load(mask_file)
mask_file.close()
loss_history = []
train_acc_history = []
z=0
for it in range(num_iters):
    if sample_batches:
        batch_mask = mask_history[z]
        X_batch = torch.Tensor(X[batch_mask])
        y_batch = torch.Tensor(y[batch_mask])
        z=z+1
    else:
        # no SGD used, full gradient descent
        X_batch = torch.Tensor(X)
        y_batch = torch.Tensor(y)
    
    # zero the parameter gradients
    optimizer.zero_grad()

    net = net.float()
    # forward + backward + optimize
    outputs = net(X_batch.float())
    loss = criterion(outputs, y_batch.long())
    loss.backward()
    # print(net.conv1.weight.grad.sum())
    # print(net.conv1.)
    # print(net.conv1.bias.grad.sum())
    # print(net.fc1.weight.grad.sum())
    # print(net.fc1.bias.grad.sum())
    optimizer.step()
    # print(loss.item())
    # loss_history.append(cost)
    # print(net.conv1.weight.grad[2,2].numpy())
    # every epoch perform an evaluation on the validation set
    first_it = (it == 0)
    epoch_end = (it + 1) % iterations_per_epoch == 0
    acc_check = (acc_frequency is not None and it % acc_frequency == 0)

    if first_it or epoch_end or acc_check:
        if it > 0 and epoch_end:
            # decay the learning rate
            for param_group in optimizer.param_groups:
                lr=param_group['lr'] 
            lr *= learning_rate_decay
            # optimizer.learning_rate *= learning_rate_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            epoch += 1

        # evaluate train accuracy
        if N > 1000:
            print("Never called")
            # train_mask = np.random.choice(N, 1000)
            # X_train_subset = X[train_mask]
            # y_train_subset = y[train_mask]
        else:
            X_train_subset = X
            y_train_subset = y
        with torch.no_grad():
            scores_train = net.forward(torch.Tensor(X_train_subset).float())
        y_pred_train = np.argmax(scores_train.detach().numpy(), axis=1)
        train_acc = np.mean(y_pred_train == y_train_subset)
        train_acc_history.append(train_acc)

        # print progress if needed
        if verbose:
            for param_group in optimizer.param_groups:
                ll=param_group['lr']
            print('Finished epoch %d / %d: cost %f train: %f, lr %e'
                    % (epoch, num_epochs, loss.item() ,train_acc, ll))


print('Finished Training')
print('Finished Training')

