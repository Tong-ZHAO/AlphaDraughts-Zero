from mcts import MCTS, StateNode
from game_mcts import init_game, game_over, move_piece
from utils import *
import config
import pickle
import os
import torch
import sys

try:
	import asyncio
except:
	print("Only Python 3.4 or later is supported by asynchronous IO for data generation.")

class Pipeline():
	'''
	Self-play training pipeline
	'''
	def __init__(self, model, optimizer, 
				 dataset_max_size, 
				 resignation_threshold, 
				 vis,
				 asycio_data_generation=False):
		self.best_mcts = MCTS(StateNode(None, init_game()), config.cpuct) # best player to generate data
		self.dataset = GameDataset(dataset_max_size)
		self.resignation_threshold = resignation_threshold # not used for now
		self.model = model
		# Initialize visdom
		self.vis = vis
		self.iter_plot = create_vis_plot(self.vis, 'Iteration', 'Loss', "Avg Loss")
		self.len_plot = create_vis_plot(self.vis, 'Iteration', 'Length', "Avg Self-Play Length")

		self.logger = build_logger("pipeline", config.file2write)
		self.checkpoints_directory = "../checkpoints/2901"
		if not os.path.exists(self.checkpoints_directory):
			os.makedirs(self.checkpoints_directory)

		self.optimizer = optimizer
		self.asycio_data_generation = asycio_data_generation # Python >= 3.4 

		self.epoch_index = 0
		self.play_index = 0
		self.model_index = 0


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
			self.self_play(config.nb_self_play_in_each_iteration)

			# training of ConvNet
			self.message("training of ConvNet")
			train_loader = torch.utils.data.DataLoader(self.dataset,
				batch_size=config.batch_size, shuffle=True, num_workers=1)

			for i, epoch in enumerate(range(config.nb_epochs_per_iteration)):
				self.train_net_for_one_epoch(train_loader, epoch, iter_)

			if iter_ % config.freq_iter_checkpoint == 0:
				self.save_model(iter_, nb_iter)



	def train_net_for_one_epoch(self, train_loader, epoch, iter_):
		self.model.train()
		avg_loss = 0.
		
		for batch_idx, (state, target_policy, target_value, dummy) in enumerate(train_loader):
			state, target_policy, target_value = state.float(), target_policy.float(), target_value.float()
			target_policy = target_policy.reshape((-1, 256))
			if torch.cuda.is_available():
				state = state.cuda()
				target_policy = target_policy.cuda()
				target_value = target_value.cuda()

			self.optimizer.zero_grad()
			value, policy = self.model(state)
			mse = torch.nn.MSELoss()
			loss = mse(value, target_value) + \
				   cross_entropy_continuous_target(policy, target_policy)
			loss.backward()
			avg_loss += float(loss.cpu().detach())
			self.optimizer.step()

			if batch_idx % config.train_net_log_interval == 0:
				mess = "ConvNet train iteraion %d epoch %d (%d / %d) / (%.4f), loss: %.4f" \
					% (iter_, epoch, batch_idx * len(state), len(train_loader.dataset), \
						100. * batch_idx / len(train_loader), loss.data.item())
				self.message(mess)

		avg_loss = avg_loss / len(self.dataset)
		update_vis_plot(self.vis, self.epoch_index, avg_loss, self.iter_plot, "append") 
		self.epoch_index += 1


	def save_model(self, iter_=None, nb_iter=None):
		file_path_model = os.path.join(self.checkpoints_directory, "model_" + str(self.model_index) + ".pth")
		torch.save(self.model.state_dict(), file_path_model)
		self.model_index += 1

		if iter_ is not None and nb_iter is not None:
			self.message("Latest model saved as " + file_path_model \
				+ " (Iteration: " + str(iter_) + "/" + str(nb_iter) + ")")

	def self_play(self, nb_self_play_in_each_iteration):

		if not self.asycio_data_generation:
			for _ in range(nb_self_play_in_each_iteration):
				self.self_play_one_game_sync()
		else:
			loop = asyncio.new_event_loop()
			#tasks = [hello(), hello()]
			tasks = []
			for _ in range(nb_self_play_in_each_iteration):
				tasks.append(self.self_play_one_game_async())
			loop.run_until_complete(asyncio.wait(tasks))
			loop.close()
			

	def self_play_one_game_sync(self):
		'''
		self-play to generate dataset
		'''
		buff = [] # buffer of data tuples
		z = 0
		current_state_node = self.best_mcts.root
		while z == 0:
			# move one step
			if current_state_node.is_leaf():
				action_node = current_state_node.best_children(determinstic = False)
				search_policy = current_state_node.get_policy()
			else:
				action_node, search_policy = self.leaf_simulation(current_state_node)

			# collect data
			z = action_node.out_node.gameover
			current_player = action_node.player.mark
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

		#self.message(buff[-1].state[0])
		
		#self.message("self-play game length = " + str(len(buff)) \
		#	+ ", current dataset size = " + str(len(self.dataset)))

		update_vis_plot(self.vis, self.play_index, len(buff), self.len_plot, "append") 
		self.play_index += 1


	def leaf_simulation(self, in_node):

		my_mcts = MCTS(StateNode(None, in_node.state), config.cpuct)

		for i in range(config.nb_simulations):
			my_mcts.move_to_leaf(self.model)
		
		action_node = my_mcts.root.best_children(determinstic = False)

		return action_node, my_mcts.root.get_policy()
		

	@asyncio.coroutine
	def self_play_one_game_async(self):
		'''
		self-play to generate dataset
		'''
		buff = [] # buffer of data tuples
		z = 0
		current_state_node = self.best_mcts.root
		while z == 0 and not current_state_node.is_leaf():
			# move one step
			action_node = current_state_node.best_children(determinstic = True)
			#game_state = move_piece(current_state_node.state, 
			#						action_node.x, 
			#						action_node.y, 
			#						action_node.action)

			# collect data
			search_policy = current_state_node.get_policy()
			z = action_node.out_node.gameover
			current_player = action_node.player.mark
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
			for data in buff:
				self.dataset.append(data)
			#yield from self.dataset.append(buff[i])
		
		self.message("self-play game length = " + str(len(buff)) \
			+ ", current dataset size = " + str(len(self.dataset)))


	def build_new_mcts(self, model):
		game_state = init_game()
		mcts = MCTS(StateNode(None, game_state), config.cpuct)

		# Simulation of MCTS
		for _ in range(config.nb_simulations):
			#print("Build new mcts")
			mcts.move_to_leaf(model, config.max_iter_move_to_leaf)
		return mcts


	def update_best_mcts(self, mcts):
		# For now, just keep the most recent for simplicity and training speed
		self.best_mcts = mcts

	def message(self, mess):
		self.logger.info(mess)
		print(mess)





