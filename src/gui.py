import pygame as pg
from game_mcts import *
import numpy as np
import sys, os
import config
from model import CNN_Net
import argparse


def change_idx(idx):
    if idx == 0:
        return 1
    else:
        return 0


def blingbling():
    #begin_time = pg.time.get_ticks()
    x_star = []
    y_star = []
    for i in range(0, nb_stars):
        x_star.append(np.random.randint(0,screen_size))
        y_star.append(np.random.randint(0,screen_size))
    while True:
        screen.blit(surface, (0, 0))

        for event in pg.event.get():
            if event.type == pg.QUIT: # quit application
                sys.exit()
            #elif event.type == pg.MOUSEBUTTONDOWN: # quick quit animation
            #    if event.button == 1: 
            #        return
        
        for i in range(len(x_star)):
            y_star[i] += 3
            x_star[i] += np.random.randint(- 2, 2)
            if y_star[i] >= screen_size:
                y_star[i] = 0
            if x_star[i] >= screen_size:
                x_star[i] = 0
            elif x_star[i] < 0:
                x_star[i] = screen_size - 1
        
        for i in range(len(x_star)):
            font_star = font.render("*", True, (np.random.randint(125, 255), 
                                                np.random.randint(125, 255), 
                                                0))
            screen.blit(font_star, (x_star[i], y_star[i]))

        pg.display.flip()
        #if pg.time.get_ticks() - begin_time > blingbling_milliseconds:
        #    return


def draw_background():
    color_idx = 0
    for i in range(0, screen_size, square_size):
        for j in range(0, screen_size, square_size):
            square = (i, j, square_size, square_size)
            pg.draw.rect(surface, square_colors[color_idx], square)
            color_idx = change_idx(color_idx) 
        color_idx = change_idx(color_idx)

    pg.draw.rect(surface, background_color, (screen_size, 0, text_size, screen_size))

def draw_pieces():

    pieces = game.my_map.get_player_pieces(True) # white

    for i in range(len(pieces)):
        y, x = pieces[i] * square_size + piece_radius
        pg.draw.circle(surface, piece_colors[0], (x, y), piece_radius)
        if abs(game.my_map[pieces[i]]) == 2:
            pg.draw.polygon(surface, piece_colors[4], [[x - piece_radius / 2, y],
                                                    [x, y - piece_radius / 2],
                                                    [x + piece_radius / 2, y],
                                                    [x, y + piece_radius / 2]])


    pieces = game.my_map.get_player_pieces(False) # white
    for i in range(len(pieces)):
        y, x = pieces[i] * square_size + piece_radius
        pg.draw.circle(surface, piece_colors[1], (x, y), piece_radius)
        if abs(game.my_map[pieces[i]]) == 2:
            pg.draw.polygon(surface, piece_colors[5], [[x - piece_radius / 2, y],
                                                    [x, y - piece_radius / 2],
                                                    [x + piece_radius / 2, y],
                                                    [x, y + piece_radius / 2]])

    movable_pieces = game.pieces
    for i in range(len(movable_pieces)):
        y, x = movable_pieces[i] * square_size + piece_radius
        pg.draw.circle(surface, piece_colors[2], (x, y), move_radius)

def draw_move(coord, moves):

    for move in moves:
        offset = coor_matrix[move[0]]
        offset_y, offset_x = (coord + offset) * square_size + piece_radius
        pg.draw.circle(surface, piece_colors[3], (offset_x, offset_y), move_radius)

def update_surface():
    draw_background()
    draw_pieces()


def computer_play(net):

    my_mcts = MCTS(StateNode(None, game), config.cpuct)

    for i in range(args.simulation):
        my_mcts.move_to_leaf(net)
    
    action_node = my_mcts.root.best_children(True)

    return action_node.out_node.state



parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="model_21.pth", metavar='C',
                    help="which neural network model checkpoint to use.")
parser.add_argument("--human", type=str, default="white", metavar="H",
                    help='"white" or "black", which side human player plays, white side always goes first.')
parser.add_argument("--simulation", type=int, default=config.nb_simulations, metavar="S",
                    help="number of simulations for MCTS at each time step to choose the action.")
parser.add_argument("--ai", type=int, default=1, metavar="A",
                    help="whether use AI, 1 means using AI, 0 means not using AI.")
args = parser.parse_args()
assert args.human in ["black", "white"], "Please choose whether black or white for --human"
assert args.ai in [0, 1], "Please choose whether 1 or 0 for --ai"
assert args.simulation > 0, "--simulation should be greater than 0"


screen_size, text_size = 400, 0
square_size = int(screen_size / 8)
piece_radius = int(square_size / 2)
move_radius = int(piece_radius / 5)

square_colors = [pg.Color('gray'), pg.Color('salmon')]
piece_colors = [pg.Color('white'), pg.Color('black'), pg.Color('red'), pg.Color('yellow'), pg.Color('black'), pg.Color('white')]
background_color = pg.Color('#8EA2F3')


# init game
game = init_game(name_p1 = "white", name_p2 = "black")
curr_pieces = game.pieces.tolist()
my_row, my_col, possible_moves, possible_pieces = None, None, None, None
use_ai = args.ai

if use_ai:
    import torch
    from mcts import MCTS, StateNode
    model = CNN_Net(use_log=False)
    use_cuda = torch.cuda.is_available()
    file_path = os.path.join("../checkpoints", args.checkpoint)
    if not use_cuda:
        model.load_state_dict(torch.load(file_path, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(file_path))

    if use_cuda:
	    model.cuda()

if args.human == "black":
    mark_ = 1
else:
    mark_ = - 1

# init stars
nb_stars = 50
blingbling_milliseconds = 1000
pg.font.init()
font = pg.font.Font("../fonts/wryh.ttf", 1)


pg.init()
screen = pg.display.set_mode((screen_size + text_size, screen_size))
surface = pg.Surface((screen_size + text_size, screen_size))
pg.display.set_caption('AlphaDraughts Zero')
clock = pg.time.Clock()
font = pg.font.Font(None, 36)

draw_background()
draw_pieces()


running = True
while running:
    for event in pg.event.get():
        #print("Here")
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 1: # left click of mouse
                mouse_y, mouse_x = event.pos
                row = int(mouse_x / square_size)
                col = int(mouse_y / square_size)

                # select piece
                if (len(curr_pieces) > 1 and [row, col] in curr_pieces) or (len(curr_pieces) == 1 and [row, col] in curr_pieces and possible_moves is None):
                    possible_moves = find_possible_pathes(game.my_map, game.player.mark, row, col, False)
                    possible_pieces = [[row + coor_matrix[move[0]][0], col + coor_matrix[move[0]][1]] for move in possible_moves]
                    my_row, my_col = row, col
                # move piece
                elif possible_pieces is not None and [row, col] in possible_pieces:
                    game = move_piece(game, my_row, my_col, possible_moves[possible_pieces.index([row, col])])
                    curr_pieces = game.pieces.tolist()
                    # jump
                    if game.flag_eat == True:
                        my_row, my_col = curr_pieces[0]
                        possible_moves = find_possible_pathes(game.my_map, game.player.mark, my_row, my_col, True)
                        # no further move, change player
                        if len(possible_moves) == 0:
                            game = GameState(game.my_map, game.opponent, game.player, pieces = None)
                            curr_pieces = game.pieces.tolist()
                            my_row, my_col, possible_moves, possible_pieces = None, None, None, None
                        # another jump
                        else:
                            possible_pieces = [[my_row + coor_matrix[move[0]][0], my_col + coor_matrix[move[0]][1]] for move in possible_moves]
                    # clear
                    else:
                        my_row, my_col, possible_moves, possible_pieces = None, None, None, None
        #elif event.type == pg.KEYDOWN:
        #    pressed = pg.key.get_pressed()
        #    if pressed[pg.K_w]:
        #        blingbling()
        
        if use_ai and game.player.mark == mark_ and not game_over(game.my_map):
        
            game = computer_play(model)
            curr_pieces = game.pieces.tolist()
            pg.event.clear()

    if game_over(game.my_map):
        print("Game Over")
        blingbling()
        text = font.render("Game Over", True, pg.Color('white'))
        text_rect = text.get_rect()
        text_x = screen.get_width() / 2 - text_rect.width / 2
        text_y = screen.get_height() / 2 - text_rect.height / 2
        screen.blit(text, [text_x, text_y])
        break
                
    

                    
    update_surface()
    if possible_moves is not None:
        draw_move(np.array([my_row, my_col]), possible_moves)
    screen.blit(surface, (0, 0))
    pg.display.flip()
    clock.tick(50)

pg.quit()