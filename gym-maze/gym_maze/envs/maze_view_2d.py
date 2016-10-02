import pygame
import random
import numpy as np
from time import sleep

class MazeView2D:

    def __init__(self, screen_size=(640, 640), maze_size=(50, 50), caption="Maze2D"):

        pygame.init()
        pygame.display.set_caption(caption)

        self.clock = pygame.time.Clock()
        self.game_over = False

        # maze's configuration parameters
        if not (isinstance(maze_size, (list, tuple)) and len(maze_size) == 2):
            raise ValueError("maze_size must be a tuple: (width, height).")
        self.maze_size = maze_size

        # Create a new screen
        if not (isinstance(screen_size, (list, tuple)) and len(screen_size) == 2):
            raise ValueError("screen_size must be a tuple: (width, height).")
        self.screen = pygame.display.set_mode(screen_size)


        # Create a background
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.background.fill((255, 255, 255))

        # Create a layer for the maze
        self.maze_layer = pygame.Surface(self.screen.get_size()).convert_alpha()
        self.maze_layer.fill((0, 0, 0, 0,))

        # Generate the maze
        self.view_update()
        self.maze = Maze(self.maze_layer, screen_size=screen_size, maze_size=maze_size,
                         surface=self.screen, background=self.background)
        self.maze.generate_maze()

    def controller_update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return

    def view_update(self):
        self.clock.tick(60)
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.maze_layer,(0, 0))
        pygame.display.flip()


class Maze:

    def __init__(self, maze_layer, screen_size=(640, 640), maze_size=(10, 10), surface=None, background=None, ):

        # maze configuration
        self.surface = surface
        self.background = background
        self.screen_size = screen_size
        self.maze_size = maze_size
        self.maze_layer = maze_layer
        self.maze_layer.fill((0, 0, 0, 0))

        # list of all cell locations
        self.maze_cells = np.zeros(self.maze_size, dtype=int)

        # drawing the horizontal lines
        for y in range(self.MAZE_H):
            pygame.draw.line(self.maze_layer, (0, 0, 0, 255), (0, y * self.CELL_H),
                             (self.SCREEN_W , y * self.CELL_H))
        # drawing the vertical lines
        for x in range(self.MAZE_W):
            pygame.draw.line(self.maze_layer, (0, 0, 0, 255), (x * self.CELL_W, 0),
                             (x * self.CELL_W, self.SCREEN_H))
    @property
    def MAZE_W(self):
        return int(self.maze_size[0])

    @property
    def MAZE_H(self):
        return int(self.maze_size[1])

    @property
    def SCREEN_W(self):
        return int(self.screen_size[0])

    @property
    def SCREEN_H(self):
        return int(self.screen_size[1])

    @property
    def CELL_W(self):
        return self.SCREEN_W / self.MAZE_W

    @property
    def CELL_H(self):
        return self.SCREEN_H / self.MAZE_H

    @classmethod
    def get_walls_status(cls, cell):
        walls = { "N" : cell & 0x1,
                  "E" : cell & 0x2,
                  "S" : cell & 0x4,
                  "W" : cell & 0x8,}
        return walls

    @classmethod
    def break_walls(cls, cell, dirs):
        if "N" in dirs:
            cell |= 0x1
        if "E" in dirs:
            cell |= 0x2
        if "S" in dirs:
            cell |= 0x4
        if "W" in dirs:
            cell |= 0x8
        return cell

    @classmethod
    def get_opposite_wall(cls, dirs):

        if not isinstance(dirs, str):
            raise TypeError("dirs must be a str.")

        opposite_dirs = ""

        for dir in dirs:
            if dir == "N":
                opposite_dir = "S"
            elif dir == "S":
                opposite_dir = "N"
            elif dir == "E":
                opposite_dir = "W"
            elif dir == "W":
                opposite_dir = "E"
            else:
                raise ValueError("The only valid directions are (N, S, E, W).")

            opposite_dirs += opposite_dir

        return opposite_dirs

    @classmethod
    def all_walls_intact(cls, cell):
        return cell & 0xF == 0

    def cover_walls(self, x, y, dirs, colour=(0, 0, 255, 15)):

        dx = x * self.CELL_W
        dy = y * self.CELL_H

        if not isinstance(dirs, str):
            raise TypeError("dirs must be a str.")

        for dir in dirs:
            if dir == "S":
                line_head = (dx + 1, dy + self.CELL_H)
                line_tail = (dx + self.CELL_W - 1, dy + self.CELL_H)
            elif dir == "N":
                line_head = (dx + 1, dy)
                line_tail = (dx + self.CELL_W - 1, dy)
            elif dir == "W":
                line_head = (dx, dy + 1)
                line_tail = (dx, dy + self.CELL_H - 1)
            elif dir == "E":
                line_head = (dx + self.CELL_W, dy + 1)
                line_tail = (dx + self.CELL_W, dy + self.CELL_H - 1)
            else:
                raise ValueError("The only valid directions are (N, S, E, W).")

            pygame.draw.line(self.maze_layer, colour, line_head, line_tail)

    def generate_maze(self):

        # Initializing constants variables needed for maze generation
        NUM_CELLS_TOTAL = int(self.MAZE_H*self.MAZE_W)

        COMPASS = { "N" : (0, -1),
                    "E" : (1, 0),
                    "S" : (0, 1),
                    "W": (-1, 0)}

        current_cell = (random.randint(0, self.MAZE_W-1), random.randint(0, self.MAZE_H-1))
        num_cells_visited = 1
        cell_stack = [current_cell]

        # Continue until all cells are visited
        while cell_stack:

            # restart from a cell from the cell stack
            current_cell = cell_stack.pop()

            x0, y0 = current_cell

            # find neighbours of the current cells that actually exist
            neighbours = dict()
            for dir_key, dir_val in COMPASS.items():
                x1 = x0 + dir_val[0]
                y1 = y0 + dir_val[1]
                # if cell is within bounds
                if 0 <= x1 < self.MAZE_W and 0 <= y1 < self.MAZE_H:
                    # if all four walls still exist
                    if self.all_walls_intact(self.maze_cells[x1, y1]):
                        neighbours[dir_key] = (x1, y1)

            # if there is a neighbour
            if neighbours:
                # select a random neighbour
                dir = random.choice(tuple(neighbours.keys()))
                x1, y1 = neighbours[dir]

                # knock down the wall between the current cell and the selected neighbour
                self.maze_cells[x1, y1] = self.break_walls(self.maze_cells[x1, y1], self.get_opposite_wall(dir))

                # draw a "transparent line over the wall
                self.cover_walls(x0, y0, dir)

                # push the current cell location to the stack
                cell_stack.append(current_cell)

                # make the this neighbour cell the current cell
                cell_stack.append((x1, y1))

                # increment the visited cell count
                num_cells_visited += 1


                # animate the maze generation process of surface is passed
                if self.surface and self.background:
                    pygame.draw.circle(self.maze_layer, (255, 0, 0, 255),
                                       (int(x1 * self.CELL_W + self.CELL_W / 2),
                                        int(y1 * self.CELL_H + self.CELL_W / 2)),
                                       int(self.CELL_W / 10))
                    self.surface.blit(self.background, (0, 0))
                    self.surface.blit(self.maze_layer, (0, 0))
                    pygame.display.flip()
                    sleep(0.01)


if __name__ == "__main__":

    maze_view = MazeView2D()

    while not maze_view.game_over:
        maze_view.controller_update()
        maze_view.view_update()
