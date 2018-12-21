import numpy as np
from enum import Enum


NORTHWEST, NORTHEAST, SOUTHWEST, SOUTHEAST  = 0, 1, 2, 3
# List of moves
Moves = [NORTHWEST, NORTHEAST, SOUTHWEST, SOUTHEAST]
# Move steps
coor_matrix = [[-1, -1], # NorthWest
               [-1,  1], # NorthEast
               [ 1, -1], # SouthWest
               [ 1,  1]] # SouthEast


class Map:

    def __init__(self, init_map = None):
        self.map = np.zeros((8, 8)) if map is None else init_map

    def __getitem(self, x, y):
        return self.map(x, y)

    def __setitem__(self, x, y, value):
        self.map(x, y) = value

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
                map.set_piece(row, col, self.mark)


class Checkers:

    def __init__(self, player_1, player_2):
        # Init Map
        self.my_map = Map()
        # Player 1: White, Player 2: Black
        self.players = [Player(player_1, self.my_map, True), Player(player_2, self.my_map, False)]

    def player(self, index):
        return self.players[index]

    def move_piece(self, curr_x, curr_y, moves, flags):

        assert(len(moves) < 2), "Illegal Path"

        for move, flag in zip(moves[:-1], flags[:-1]):
            step_x, step_y = coor_matrix[move]
            if flag == True:
                new_x, new_y = curr_x + 2 * step_x, curr_y + 2 * step_y
                self.my_map[curr_x + step_x, curr_x + step_y] = 0
            else:
                new_x, new_y = curr_x + step_x, curr_y + step_y
                self.my_map[new_x, new_y] = self.my_map[curr_x, curr_y]
                self.my_map[curr_x, curr_y] = 0
            curr_x, curr_y = new_x, new_y

        self.check_king(curr_x, curr_y)

    def check_king(self, x, y):

        mark = self.my_map(x, y)

        if x == 0 and mark == 1:
            self.my_map(x, y) = 2
        elif x == 7 and mark == -1:
            self.my_map(x, y) = -2

    def is_finish(self)

        num_white = self.my_map.num_pieces(True)
        num_black = self.my_map.num_pieces(False)

        if num_white * num_black == 0:
            return 1 if num_black == 0 else -1

        # Check White
        pieces_white = self.my_map.get_player_pieces(True)

        for piece in pieces_white:
            pathes = find_possible_pathes(self.my_map, 1, piece[0], piece[1])
            if sum([(len(path) > 1) for path in pathes]) == 0:
                return -1

        # Check Black
        pieces_black = self.my_map.get_player_pieces(False)

        for piece in pieces_black:
            pathes = find_possible_pathes(self.my_map, -1, piece[0], piece[1])
            if sum([(len(path) > 1) for path in pathes]) == 0:
                return 1

        return 0


def find_possible_pathes(map, mark, curr_x, curr_y):

    piece = map(curr_x, curr_y) 
    assert(mark * piece > 0), "Illegal Position"

    moves, pathes = [], []
    if abs(piece) == 2:
        moves = Moves
    elif piece == -1:
        moves = [SOUTHWEST, SOUTHEAST]
    else:
        moves = [NORTHWEST, NORTHEAST]

    for move in Move:
        pathes += is_possible_path(map, mark, curr_x, curr_y, move)

    return pathes

def is_possible_path(map, mark, curr_x, curr_y, move):
    """
    Returns two lists:
        pathes: the moves 
        flags : whether we eat a piece in each move
    """

    mark = player.mark
    step_x, step_y = coor_matrix[move]
    new_x , new_y = curr_x + step_x, curr_y + step_y  

    # Handle Cases:
    #    1. Already reach the border
    #    2. The destination is taken by its teammate
    if (new_x < 0) or (new_y < 0) or (new_x > 7) or (new_y > 7) or (map(new_x, new_y) * mark > 0):
        return ["Finish"], [False]
    # Handle Case: Available 1 step
    elif self.map(new_x, new_y) == 0:
        return [move, "Finish"], [False, False]
    # Eat opponent's piece
    else:
        jump_x , jump_y = new_x + step_x, new_y + step_y
        # Handle Case: No available place
        if (jump_x < 0) or (jump_y < 0) or (jump_x > 7) or (jump_y > 7):
            return ["Finish"], [False]
        # Jump, update the map, and find following possible move
        else:
            new_map_array = map.array
            new_map_array[new_x, new_y] = 0
            new_map_array[jump_x, jump_y] = new_map_array[curr_x, curr_y] 
            new_map_array[curr_x, curr_y] = 0
            new_map = Map(new_map_array)

            rec_pathes, rec_flags = find_possible_pathes(new_map, player, jump_x, jump_y) 
            return [[move] + path for path in rec_pathes], [[True] + flag for flag in rec_flags] 