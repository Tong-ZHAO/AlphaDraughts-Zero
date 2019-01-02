import argparse
import torch
import torch.optim as optim
from model import CNN_Net
from utils import *
import config
import os, time, datetime

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(dataset):
        if use_cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
		values, policies = model(data)
		cross_entropy = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
		mse = torch.nn.MSELoss()
		loss = mse(values, target_values) + cross_entropy(policies, target_policies)
		loss.backward()
		optimizer.step()

t0 = time.time()

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

checkpoints_directory = "../checkpoints"
if not os.path.exists(log_directory):
	os.makedirs(log_directory)

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

logger.info("Training begins")

try:
	for epoch in range(1, args.epochs + 1):
	    train(epoch)
	    logger.info("model_" + str(epoch))
	    file_path = os.path.join(checkpoints_directory, "model_" + str(epoch) + ".pth")
except KeyboardInterrupt:
	torch.save(model.state_dict(), file_path)
	logger.info("Latest model saved to " + checkpoints_directory)
	logger.info("Keyboard interruption after %.4f s." % (time.time() - t0))
else:
	torch.save(model.state_dict(), file_path)
	logger.info("Latest model saved to " + checkpoints_directory)
	logger.info("Training done in %.4f s." % (time.time() - t0))

logger.info("Training finished at " + datetime.datetime.now())

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