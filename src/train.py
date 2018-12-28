import argparse
import torch
import torch.optim as optim
from model import CNN_Net
from utils import *
import config

parser = argparse.ArgumentParser(description='Training of AlphaDraughts Zero')

parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')

args = parser.parse_args()
torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available()
logger = build_logger("train")

model = CNN_Net()

if use_cuda:
	model.cuda()
	logger.info("Using GPU")
    print('Using GPU')
else:
	logger.info("Using CPU")
    print('Using CPU')

c = 0.01 # L2 regularization coefficient
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=c)

'''
One step of training:

optimizer.zero_grad()
values, policies = model(data)
cross_entropy = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
mse = torch.nn.MSELoss()
loss = mse(values, target_values) + cross_entropy(policies, target_policies)
loss.backward()
optimizer.step()
'''