from __future__ import print_function       # 使用py3.0版本的print

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

from torch.optim import lr_scheduler
from torch.autograd import Variable

dim=8
klloss_alpha = 0.2

def squash(x):
    lengths2 = x.pow(2).sum(dim=2)
    lengths = lengths2.sqrt()
    x = x * (lengths2 / (1 + lengths2) / lengths).view(x.size(0), x.size(1), 1)
    return x

def KL_div(mu):
    sigma = torch.ones_like(mu)
    '''KL divergence between N(mu,sigma**2) and N(mu1,sigma1**2)'''
    return 0.5 * (sigma**2 + mu**2 - 1 - torch.log(sigma**2))

class AgreementRouting(nn.Module):
    def __init__(self, input_caps, output_caps, n_iterations):   #input_caps=1152, output_caps=10, r=3
        super(AgreementRouting, self).__init__()
        self.n_iterations = n_iterations
        self.b = nn.Parameter(torch.zeros((input_caps, output_caps)))  #(1152,10)

    def forward(self, u_predict):
        batch_size, input_caps, output_caps, output_dim = u_predict.size()
        #(batch, 1152, 10, 16)
        c = F.softmax(self.b, dim=1)  #dim=1 #(1152,10)
        s = (c.unsqueeze(2) * u_predict).sum(dim=1) #(1152,10,1)*(batch,1152,10,16) (batch,10,16)
        v = squash(s)  #(batch, 10, 16)

        if self.n_iterations > 0:
            b_batch = self.b.expand((batch_size, input_caps, output_caps)) #(batch, 1152, 10)
            for r in range(self.n_iterations):
                v = v.unsqueeze(1)    # 在第1维处增加维度：(batch, 1, 10, 16)
                b_batch = b_batch + (u_predict * v).sum(-1) # (batch, 1152, 10)

                c = F.softmax(b_batch.view(-1, output_caps), dim=1).view(-1, input_caps, output_caps, 1)
                #(batch,1152,10,1)
                s = (c * u_predict).sum(dim=1) #(batch,10,16)
                v = squash(s)

        return v


class CapsLayer(nn.Module):
    def __init__(self, input_caps, input_dim, output_caps, output_dim, routing_module): #1152,8,10,16,routing
        super(CapsLayer, self).__init__()
        self.input_dim = input_dim
        self.input_caps = input_caps
        self.output_dim = output_dim
        self.output_caps = output_caps
        self.weights = nn.Parameter(torch.Tensor(input_caps, input_dim, output_caps * output_dim)) #(1152,8,160)
        self.routing_module = routing_module
        self.reset_parameters()

    def reset_parameters(self):   # 重置参数
        stdv = 1. / math.sqrt(self.input_caps)
        self.weights.data.uniform_(-stdv, stdv)

    def forward(self, caps_output):   # caps_output=(1152,8)
        caps_output = caps_output.unsqueeze(2)  #(1152,8,1)
        u_predict = caps_output.matmul(self.weights)  #(1152,8,160)
        u_predict = u_predict.view(u_predict.size(0), self.input_caps, self.output_caps, self.output_dim) #(1152,10,16)
        v = self.routing_module(u_predict)   #(10,16)
        return v


class PrimaryCapsLayer(nn.Module):
    def __init__(self, input_channels, output_caps, output_dim, kernel_size, stride):
        super(PrimaryCapsLayer, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_caps * output_dim, kernel_size=kernel_size, stride=stride)#(256,256,9,2)
        self.input_channels = input_channels
        self.output_caps = output_caps
        self.output_dim = output_dim

    def forward(self, input):
        out = self.conv(input)
        N, C, H, W = out.size()   #batch,256,6,6
        out = out.view(N, self.output_caps, self.output_dim, H, W) #batch,32,8,6,6

        # will output N x OUT_CAPS x OUT_DIM
        out = out.permute(0, 1, 3, 4, 2).contiguous()  #batch,32,6,6,8
        out = out.view(out.size(0), -1, out.size(4))   #batch,1152,8
        out = squash(out)
        return out


class CapsNet(nn.Module):
    def __init__(self, routing_iterations, n_classes=1, capsuleblock=16, n_dim=dim):
        super(CapsNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=9, stride=1)
        self.primaryCaps = PrimaryCapsLayer(256, capsuleblock, 8, kernel_size=9, stride=2)  # primaryCaps: (1152,8)
        self.num_primaryCaps = capsuleblock * 6 * 6
        routing_module = AgreementRouting(self.num_primaryCaps, n_classes, routing_iterations)
        self.digitCaps = CapsLayer(self.num_primaryCaps, 8, n_classes, n_dim, routing_module)

    def forward(self, input):
        x = self.conv1(input)  #(batch, 256, 10, 10)
        x = F.relu(x)
        x = self.primaryCaps(x)   #(batch,1152,8)
        x = self.digitCaps(x)     #(batch,10,16)
        return x


class ReconstructionNet(nn.Module):
    def __init__(self, n_dim, n_classes=1):
        super(ReconstructionNet, self).__init__()
        self.fc1 = nn.Linear(n_dim, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc3 = nn.Linear(1024, 784)

        # setup the non-linearity
        self.act = nn.ReLU(inplace=True)
        self.act1 = nn.Sigmoid()
        self.n_dim = n_dim
        self.n_classes = n_classes

    def forward(self, x):  # x:batch,16
        #x = x.view(-1, self.n_dim)  #(batch,16)
        klloss = KL_div(x)
        h = x.view(-1, self.n_dim, 1, 1)

        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act1(self.fc3(x))


        return x.view(-1,784), klloss.mean()    #重构(batch,784)

class CapsNetWithReconstruction(nn.Module):
    def __init__(self, capsnet, reconstruction_net):
        super(CapsNetWithReconstruction, self).__init__()
        self.capsnet = capsnet
        self.reconstruction_net = reconstruction_net
        
    def forward(self, x):
        x = self.capsnet(x)   #分类胶囊(batch,10,16)、胶囊模长(batch,10)
        reconstruction, klloss = self.reconstruction_net(x)   #重构样本(batch,784)
        return reconstruction, klloss


if __name__ == '__main__':

    import argparse
    import torch.optim as optim
    from torchvision import datasets, transforms
    from torch.autograd import Variable

    # Training settings
    parser = argparse.ArgumentParser(description='CapsNet with MNIST')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=500, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=300, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--routing_iterations', type=int, default=3)
    parser.add_argument('--with_reconstruction', action='store_true', default=True)
    #parser.add_argument('--n_dim', type=int, default=4)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.Pad(2), transforms.RandomCrop(28),
                           transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)

    model = CapsNet(args.routing_iterations)

    if args.with_reconstruction:
        reconstruction_model = ReconstructionNet(n_dim=dim, n_classes=1)
        reconstruction_alpha = 1   #0.0005
        model = CapsNetWithReconstruction(model, reconstruction_model)

    if args.cuda:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, patience=15, min_lr=1e-6)

    print(model)

    def train(epoch):
        model.train()
        for batch_idx, (data,label) in enumerate(train_loader):  # batch_idx:1, data:(batch,1,28,28)
            if args.cuda:
                data,label = data.cuda(), label.cuda()
            optimizer.zero_grad()
            if args.with_reconstruction:
                output, klloss = model(data)
                reconstruction_loss = F.mse_loss(output, data.view(-1, 784))
                loss = reconstruction_alpha * reconstruction_loss + klloss_alpha * klloss# + margin_loss

            loss.backward()
            optimizer.step()
            if batch_idx % args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.4f}\t recloss:{:.4f}\t kl_loss:{:.4f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item(), 
                           reconstruction_loss.item(), klloss.item()))
        return reconstruction_loss.item(), klloss.item()
    
    recloss_base = 10000
    for epoch in range(1, args.epochs + 1):
        torch.cuda.synchronize()
        start = time.time()
        reconstruction_loss, klloss = train(epoch)
        torch.cuda.synchronize()
        end = time.time()
        print('\n Training this model take {}s each epoch'.format(int(end-start)))
        
        if recloss_base > reconstruction_loss:
            recloss_base = reconstruction_loss
            torch.save(model.state_dict(), './trained_model/FMNIST_{}epoch_dim{}_kl{}_rec{:.4f}_kl{:.4f}.pth'.format(epoch, dim, klloss_alpha,
             reconstruction_loss, klloss))