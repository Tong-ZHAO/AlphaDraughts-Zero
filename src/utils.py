import logging
import os
from torch.utils.data import Dataset
import numpy as np
import pickle

'''
Example:

from utils import *
import config

logger = build_logger("model", config.file2write)
logger.info(".......")

'''
def build_logger(name, file2write):
	log_directory = "../logs"
	if not os.path.exists(log_directory):
			os.makedirs(log_directory)
	file_path = os.path.join(log_directory, file2write + ".txt")

	logger = logging.getLogger(name)
	logger.setLevel(logging.INFO)
	handler = logging.FileHandler(file_path)
	fmt = "%(asctime)s %(name)s: %(message)s"
	handler.setFormatter(fmt)
	logging.basicConfig(filename=file_path, filemode='w', format=fmt)
	return logger


class Fifo():
	'''
	A fixed-sized FIFO-like container in RAM
	'''
	def __init__(self, max_size):
		self.array = []
		self.max_size = max_size

	@property
	def size(self):
		return len(self.array)

	def __getitem__(self, idx):
		return self.array[idx]

	def clear(self):
		self.array = []

	def __pop(self):
		self.array.pop(0) # FIFO

	def append(self, x):
		self.array.append(x)
		if self.size > self.max_size:
			self.__pop()

	def __str__(self):
		return str(self.array)

	'''
		self.dataset.append(Data(current_state_node, 
								 search_policy,
								 z,
								 current_player))'''
	'''
		for i in range(1, min(game_length + 1, self.dataset.size)):
			buffer[- i].outcome = z * dataset[- i].current_player'''


class Data():
	def __init__(self, curr_node, search_policy, outcome, current_player):
		self.state = np.array([curr_node.map.array,                         
							   curr_node.get_movable_pieces(),              
							   np.ones((8, 8)) * curr_node.player.mark]) # tensor 8x8x3
		self.policy = search_policy # array
		self.outcome = outcome # 1 or -1
		self.current_player = current_player # 1 or -1


class GameDataset(Dataset):
	def __init__(self, max_size, root_dir="../dataset", init_dataset=True):
		self.root_dir = root_dir
		self.max_size = max_size
		self.current_size = 0
		self.cursor = 0

		if not os.path.exists(root_dir):
			os.makedirs(root_dir)

		if init_dataset:
			self.clear()
		

	def __len__(self):
		return self.current_size

	def __getitem__(self, idx):
		file_path = os.path.join(self.root_dir, str(idx) + ".pkl")
		with open(file_path, 'rb') as f:
			return pickle.load(f)

	def append(self, data):
		with open(str(self.cursor) + ".pkl", "wb") as f:
			pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

		if self.cursor == self.max_size - 1:
			self.cursor = 0
		else:
			self.cursor += 1

		self.current_size = min(self.max_size, self.current_size + 1)

	def clear(self):
		if os.path.exists(self.root_dir):
			file_list = [f for f in os.listdir(self.root_dir) if f.endswith(".bak")]
			for f in file_list:
				os.remove(os.path.join(self.root_dir, f))






