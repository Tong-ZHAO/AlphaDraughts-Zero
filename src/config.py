import datetime

# Log file name
file2write = "logs " + str(datetime.datetime.now())[:16]

# Model hyperparameters
nb_filters_3x3 = 64 # number of filters in each 3x3 convolutional layers, 256 in AlphaGo Zero
value_head_hidden_layer_size = 64 # number of hidden units in the hidden layer of value head
nb_residual_blocks = 19 # number of residual blocks in the neural network architecture

# MCTS parameters
max_iter_move_to_leaf = 300 # one argument for mcts.move_to_leaf()
cpuct = 0.02 # exploration / exploitation

# Training parameters
lr = 0.001
random_seed = 42
batch_size = 32
nb_iter_training = 1000

# Pipeline parameters
nb_epochs_per_iteration = 2 # Number of epochs of training of ConvNet in every iteration of pipeline
dataset_max_size = 5000 # Maximum number of kept self-play records, 500000 in AlphaGo Zero
resignation_threshold = - 0.2 # Resignation if the estimated value is lower than this threshold
nb_simulations = 2000 # Number of simulations for MCTS
nb_self_play_in_each_iteration = 3 # Number of self-play games in each iteration of training, 25000 in AlphaGo Zero
freq_iter_checkpoint = 10 # Frequency of saving checkpoint by iteration, 1000 in AlphaGo Zero's paper
freq_update_best_mcts = 1 # Frequency of updating the best player so far, 1000 in AlphaGo Zero
train_net_log_interval = 10 # Interval (batch_idx) of print


