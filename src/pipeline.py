from mcts import MCTS, StateNode
from game_mcts import init_game, game_over
from utils import *
import config
import pickle
import os
import torch

class Pipeline():
	'''
	Self-play training pipeline
	'''
	def __init__(self, model, optimizer, dataset_max_size, resignation_threshold):
		self.best_mcts = MCTS(StateNode(None, init_game()), 1) # best player to generate data
		self.dataset = GameDataset(dataset_max_size)
		self.resignation_threshold = resignation_threshold # not used for now
		self.model = model

		self.logger = build_logger("pipeline", config.file2write)
		self.checkpoints_directory = "../checkpoints"
		if not os.path.exists(self.checkpoints_directory):
			os.makedirs(self.checkpoints_directory)

		self.optimizer = optimizer


	def train(self, nb_iter):
		for iter_ in range(1, nb_iter + 1):
			self.message("Pipeline training iteration: (" + \
				str(iter_) + ", " + str(nb_iter) + ")")

			# initialization
			mcts = self.build_new_mcts(self.model)

			if iter_ % config.freq_update_best_mcts == 0:
				self.update_best_mcts(mcts)
			elif iter_ == 1:
				self.best_mcts = mcts
			

			# Self-play
			for _ in range(config.nb_self_play_in_each_iteration):
				self.self_play_one_game()

			# training of ConvNet
			self.message("training of ConvNet")
			train_loader = torch.utils.data.DataLoader(self.dataset,
				batch_size=config.batch_size, shuffle=True, num_workers=1)

			for epoch in config.nb_epochs_per_iteration:
				self.train_net_for_one_epoch(train_loader, epoch, iter_)

			if iter_ % config.freq_iter_checkpoint == 0:
				self.save_model(iter_, nb_iter)



	def train_net_for_one_epoch(train_loader, epoch, iter_):
		self.model.train()
		for batch_idx, (state, target_policy, target_value, dummy) in enumerate(train_loader):
			if torch.cuda.is_available():
				state = state.cuda()
				target_policy = target_policy.reshape((1, - 1)).cuda()
				target_value = target_value.cuda()

			self.optimizer.zero_grad()
			value, policy = self.model(state)
			cross_entropy = torch.nn.CrossEntropyLoss(reduction='elementwise_mean')
			mse = torch.nn.MSELoss()
			loss = mse(value, target_value) + cross_entropy(policy, target_policy)
			loss.backward()
			self.optimizer.step()

			if batch_idx % config.train_net_log_interval == 0:
				mess = "ConvNet train iteraion %d epoch %d (%d / %d) / %d (%.4f)%, loss: %.4f" \
					% (iter_, epoch, batch_idx * len(data), len(train_loader.dataset), \
						100. * batch_idx / len(train_loader), loss.data.item())
				self.message(mess)

	def save_model(self, iter_=None, nb_iter=None):
		file_path_model = os.path.join(self.checkpoints_directory, "latest_model.pth")
		torch.save(self.model.state_dict(), file_path_model)

		if iter_ is not None and nb_iter is not None:
			self.message("Latest model saved as " + file_path_model \
				+ " (Iteration: " + str(iter_) + "/" + str(nb_iter) + ")")
			

	def self_play_one_game(self):
		'''
		self-play to generate dataset
		'''
		buff = [] # buffer of data tuples
		z = 0
		current_state_node = self.best_mcts.root
		while z == 0:
			# move one step
			action_node = current_state_node.best_children(determinstic=False)
			game_state = move_piece(game_state, 
									action_node.x, 
									action_node.y, 
									action_node.action)

			# collect data
			search_policy = current_state_node.get_policy()
			z = game_over(game_state.my_map)
			current_player = current_state_node.player.mark
			buff.append(Data(current_state_node, 
							   search_policy,
							   z,
							   current_player))
			
			# move to next state node
			current_state_node = action_node.out_node
		
		# update the outcome of each data tuple (map, policy, outcome)
		for i in range(len(buff)):
			buff[i].outcome = z * buff[i].current_player

		# save to disk
		for i in range(len(buff)):
			self.dataset.append(buff[i])
		
		self.message("self-play game length = " + str(len(buff)) \
			+ ", current dataset size = " + str(len(self.dataset)))

	def build_new_mcts(self, model):
		game_state = init_game()
		mcts = MCTS(StateNode(None, game_state), 1)

		# Simulation of MCTS
		for _ in range(config.nb_simulations):
			mcts.move_to_leaf(model, config.max_iter_move_to_leaf)
		return mcts


	def update_best_mcts(self, mcts):
		# For now, just keep the most recent for simplicity and training speed
		self.best_mcts = mcts

	def message(self, mess):
		self.logger(mess)
		print(mess)





