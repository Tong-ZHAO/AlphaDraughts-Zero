import numpy as np
import game_mcts
import torch

class StateNode:

    def __init__(self, parent, game_state):
        self.parent = parent
        self.actions = []
        self.state = game_state
        self.player = self.state.player
        self.gameover = game_mcts.game_over(self.state.my_map)

    def is_leaf(self):
        return True if len(self.actions) == 0 else False

    def init_children(self, prior):

        prior = prior.detach().cpu().numpy().reshape((8, 8, 4))

        assert(len(self.actions) == 0), "The actions have been initialized!"

        # find all pieces of the current player
        pieces = self.state.pieces
        flag_eat = True if (self.parent and self.parent.action[-1] == False) else False
        # find possible moves for the player
        for i in range(len(pieces)):
            x, y = pieces[i, 0], pieces[i, 1]
            moves = game_mcts.find_possible_pathes(self.state.my_map, self.state.player.mark, x, y, flag_eat)

            # add action
            for move in moves:
                out_state = game_mcts.move_piece(self.state, x, y, move)
                action = ActionNode(self, [x, y], move, prior[x, y, move[0]])
                out_node = StateNode(action, out_state)
                action.set_child(out_node)
                self.actions.append(action)

    def get_movable_pieces(self):

        pieces = self.state.pieces
        # create mask: 1 - movable
        mask = np.zeros((8, 8))
        mask[pieces[:, 0], pieces[:, 1]] = 1

        return mask
                

    def get_children(self):

        return [self.actions[i].out_node for i in len(actions)]

    def best_children(self, determinstic = True):
        Stats_N = np.array([float(action.stats["N"]) for action in self.actions])

        if determinstic:
            ind = np.argmax(Stats_N)
        else: # stochastici
            print(Stats_N)
            ind = np.random.choice(len(Stats_N), p = Stats_N / Stats_N.sum())

        return self.actions[ind]

    def get_policy(self):

        policy = np.zeros((8, 8, 4))

        for action in self.actions:
            policy[action.x, action.y, action.action[0]] = action.stats["N"]

        return policy / policy.sum()

    def get_value(self):

        return self.parent.stats["Q"]




class ActionNode:

    def __init__(self, in_node, coord, action, prior):
        self.player = in_node.player
        self.in_node = in_node
        self.x, self.y = coord
        self.action = action
        self.stats = {'N': 0,     # the number of times the action has been taken
                      'W': 0,     # the total value of the next state
                      'Q': 0,     # the mean value of the next state
                      'P': prior  # the prior proba of selecting the action
                     }
        self.out_node = None

    def set_child(self, child):
        self.out_node = child


class MCTS:

    def __init__(self, root, cpuct):

        self.root = root
        self.num_node = 1
        self.cpuct = cpuct
        self.use_cuda = torch.cuda.is_available()

    def move_to_leaf(self, net, max_iter):

        curr_node = self.root
        actions = []

        while not curr_node.is_leaf():

            if len(actions) > max_iter or curr_node.gameover != 0:
                break

            max_val, max_action = -np.inf, None
            N_sum = sum([curr_node.actions[i].stats['N'] for i in range(len(curr_node.actions))])

            for action in curr_node.actions:

                N, W, Q, P = action.stats['N'], action.stats['W'], action.stats['Q'], action.stats['P'],
                U = Q + self.cpuct * P * np.sqrt(N_sum) / (1 + N)

                if action.stats['Q'] + U > max_val:
                    max_val = action.stats['Q'] + U
                    max_action = action

            actions.append(max_action)
            curr_node = max_action.out_node

        curr_input = [ curr_node.state.my_map.array,                # current game map
                       curr_node.get_movable_pieces(),              # piece mask map
                       np.ones((8, 8)) * curr_node.player.mark]     # player map

        curr_input = torch.from_numpy(np.expand_dims(np.stack(curr_input, axis = 0), axis = 0)).float()

        if self.use_cuda:
            curr_input = curr_input.cuda()

        value, prior = net(curr_input)

        if curr_node.is_leaf():
            curr_node.init_children(prior)
            self.num_node += len(curr_node.actions)

        # update value for all passed actions
        for action in actions:
            action.stats['N'] += 1
            action.stats['W'] += value
            action.stats['Q'] = action.stats['W'] / action.stats['N']