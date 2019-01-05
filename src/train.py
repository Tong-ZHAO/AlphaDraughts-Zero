import argparse
import torch
import torch.optim as optim
from utils import *
import config
import os, time, datetime
from pipeline import Pipeline
from model import CNN_Net

def message(mess):
	logger.info(mess)
	print(mess)

t0 = time.time()

parser = argparse.ArgumentParser(description='Training of AlphaDraughts Zero')

parser.add_argument('--iterations', type=int, default=config.nb_iter_training, metavar='N',
                    help='number of iterations of pipeline training)')
parser.add_argument('--lr', type=float, default=config.lr, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--seed', type=int, default=config.random_seed, metavar='S',
                    help='random seed (default: 42)')

args = parser.parse_args()
torch.manual_seed(args.seed)
use_cuda = torch.cuda.is_available()

logger = build_logger("train", config.file2write)

model = CNN_Net()

if use_cuda:
	model.cuda()
	message("Using GPU")
else:
	message("Using CPU")

c = 10e-4 # L2 regularization coefficient
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=c)
pipeline = Pipeline(model, optimizer, config.dataset_max_size, config.resignation_threshold)

message("Training begins.")

try:
	pipeline.train(args.iterations)
except KeyboardInterrupt:
	message("Keyboard interruption after %.4f s." % (time.time() - t0))
	pipeline.save_model()
else:
	message("Training done in %.4f s." % (time.time() - t0))
	pipeline.save_model()

message("Training finished.")



