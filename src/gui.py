import pygame as pg

def change_idx(idx):
    if idx == 0:
        return 1
    else:
        return 0

def init_pieces(pieces):

    rows = [5, 6, 7] #[0, 1 ,2]

    for row in rows:
        cols = [0, 2, 4, 6] if row % 2 == 1 else [1, 3, 5, 7]
        for col in cols:
            row_ = piece_radius + row * square_size
            col_ = piece_radius + col * square_size
            pieces[row_, col_] = 0

    rows = [0, 1 ,2]

    for row in rows:
        cols = [0, 2, 4, 6] if row % 2 == 1 else [1, 3, 5, 7]
        for col in cols:
            row_ = piece_radius + row * square_size
            col_ = piece_radius + col * square_size
            pieces[row_, col_] = 1

def draw_background():
    color_idx = 0
    for i in range(0, screen_size, square_size):
        for j in range(0, screen_size, square_size):
            square = (i, j, square_size, square_size)
            pg.draw.rect(surface, square_colors[color_idx], square)
            color_idx = change_idx(color_idx) 
        color_idx = change_idx(color_idx)

def draw_pieces():
    for (mouse_x, mouse_y), color_idx in pieces.items():
        piece_center = (mouse_x, mouse_y)
        pg.draw.circle(surface, piece_colors[color_idx], piece_center, piece_radius)

def update_surface():
    draw_pieces()


screen_size = 400
square_size = int(screen_size / 8)
piece_radius = int(square_size / 2)
square_colors = [pg.Color('gray'), pg.Color('salmon')]
piece_colors = [pg.Color('black'), pg.Color('white')]

pieces = {} # every piece is a dict obj (x location, y location)-->(color index)
init_pieces(pieces)


pg.init()
screen = pg.display.set_mode((screen_size, screen_size))
surface = pg.Surface((screen_size, screen_size))
clock = pg.time.Clock()
draw_background()


running = True
while running:
    for event in pg.event.get():
        if event.type == pg.QUIT:
            running = False
        elif event.type == pg.MOUSEBUTTONDOWN:
            if event.button == 1: # left click of mouse
                mouse_x, mouse_y = event.pos
                print(mouse_x, mouse_y)
                color_idx = 1
                pieces[(mouse_x, mouse_y)] = color_idx

    update_surface()
    screen.blit(surface, (0, 0))
    pg.display.flip()
    clock.tick(50)

pg.quit()