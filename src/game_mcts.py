import copy
import numpy as np


FINISH, NORTHWEST, NORTHEAST, SOUTHWEST, SOUTHEAST  = -1, 0, 1, 2, 3
# List of moves
Moves = [NORTHWEST, NORTHEAST, SOUTHWEST, SOUTHEAST]
# Move steps
coor_matrix = [[-1, -1], # NorthWest
               [-1,  1], # NorthEast
               [ 1, -1], # SouthWest
               [ 1,  1]] # SouthEast


class Map:

    def __init__(self, init_map = None):
        self.map = np.zeros((8, 8)) if init_map is None else init_map

    def __getitem__(self, tup):
        return self.map[tup[0], tup[1]]

    def __setitem__(self, tup, value):
        self.map[tup[0], tup[1]] = value

    def __repr__(self):
        return 'Map\n' + np.array2string(self.map)

    def num_pieces(self, white):
        if white:
            return (self.map > 0).sum()
        else:
            return (self.map < 0).sum()

    def get_player_pieces(self, white):
        if white:
            return np.vstack(np.where(self.map > 0)).T
        else:
            return np.vstack(np.where(self.map < 0)).T

    @property
    def array(self):
        return self.map


class Player:

    def __init__(self, name, map, white = False):

        self.name = name
        self.white = white
        self.mark = 1 if white else -1
        self.init_pieces(map)

    def init_pieces(self, map):

        rows = [5, 6, 7] if self.white else [0, 1 ,2]

        for row in rows:
            cols = [0, 2, 4, 6] if row % 2 == 1 else [1, 3, 5, 7]
            for col in cols:
                map[row, col] = self.mark


class GameState:

    def __init__(self, my_map, player1, player2, pieces = None):

        self.my_map = my_map
        self.player = player1
        self.opponent = player2

        if pieces is not None:
            self.pieces = pieces
            self.flag_eat = True
        else:
            self.pieces = self.my_map.get_player_pieces(self.player.white)
            self.flag_eat = False


def init_game(name_p1 = "white", name_p2 = "black"):

    my_map = Map()
    player1 = Player(name_p1, my_map, True)
    player2 = Player(name_p2, my_map, False)

    return GameState(my_map, player1, player2)


def move_piece(old_state, curr_x, curr_y, move):

    step_x, step_y = coor_matrix[move[0]]
    new_x, new_y = None, None
    my_map = copy.deepcopy(old_state.my_map)

    if move[1] == True:
        new_x, new_y = curr_x + 2 * step_x, curr_y + 2 * step_y
        my_map[curr_x + step_x, curr_y + step_y] = 0
    else:
        new_x, new_y = curr_x + step_x, curr_y + step_y

    my_map[new_x, new_y] = my_map[curr_x, curr_y]
    my_map[curr_x, curr_y] = 0

    val, flag = check_king(my_map, new_x, new_y)
    my_map[new_x, new_y] = val

    

    if move[2] or flag:
        return GameState(my_map, old_state.opponent, old_state.player)
    else:
        pieces = np.array([new_x, new_y])
        return GameState(my_map, old_state.player, old_state.opponent, pieces.reshape((1, -1)))


def check_king(my_map, x, y):

    mark = my_map[x, y]

    if x == 0 and mark == 1:
        return 2, True
    elif x == 7 and mark == -1:
        return -2, True
    else:
        return mark, False


def game_over(map):

    num_white = map.num_pieces(True)
    num_black = map.num_pieces(False)

    # One player has no piece
    if num_white * num_black == 0:
        return 1 if num_black == 0 else -1

    # Check White
    pieces_white = map.get_player_pieces(True)
    flag_white, flag_black = False, False

    for piece in pieces_white:
        pathes = find_possible_pathes(map, 1, piece[0], piece[1], False)
        if len(pathes) > 0:
            flag_white = True
            break

    # Check Black
    pieces_black = map.get_player_pieces(False)

    for piece in pieces_black:
        pathes = find_possible_pathes(map, -1, piece[0], piece[1], False)
        if len(pathes) > 0:
            flag_black = True
            break

    if flag_black and flag_white:
        return 0
    elif flag_white: # white wins
        return 1
    else:            # black wins
        return -1


def find_possible_pathes(map, mark, curr_x, curr_y, flag_eat):

    piece = int(map[curr_x, curr_y])
    assert(mark * piece > 0), "Illegal Position"

    moves = []
    pathes, flags_eat, flags_inv = [], [], []
    if abs(piece) == 2:
        moves = Moves
    elif piece == -1:
        moves = [SOUTHWEST, SOUTHEAST]
    else:
        moves = [NORTHWEST, NORTHEAST]

    for move in moves:
        path = is_possible_path(map, mark, curr_x, curr_y, move, flag_eat)
        if path[0] != FINISH:
            pathes.append(path)

    return pathes


def is_possible_path(map, mark, curr_x, curr_y, move, eat):
    """
    Returns:
        move     : the move
        flag_eat : whether we eat a piece in the move
        flag_inv : whether we change the player
    """

    step_x, step_y = coor_matrix[move]
    new_x , new_y = curr_x + step_x, curr_y + step_y  

    # Handle Cases:
    #    1. Already reach the border
    #    2. The destination is taken by its teammate
    if (new_x < 0) or (new_y < 0) or (new_x > 7) or (new_y > 7) or (map[new_x, new_y] * mark > 0):
        return [FINISH, False, True]
    # Handle Case: Available 1 step
    elif map[new_x, new_y] == 0:
        if eat:
            return [FINISH, False, True]
        else:
            return [move, False, True]
    # Eat opponent's piece
    else:
        jump_x , jump_y = new_x + step_x, new_y + step_y
        # Handle Case: No available place
        if (jump_x < 0) or (jump_y < 0) or (jump_x > 7) or (jump_y > 7) or map[jump_x, jump_y] != 0:
            return [FINISH, False, True]
        # Jump, update the map, and find following possible move
        else:
            return [move, True, False]