import pygame
import random
import numpy as np
import os


class MazeView2D:

    def __init__(self, caption="Maze2D"):

        # PyGame configurations
        pygame.init()
        pygame.display.set_caption(caption)
        self.clock = pygame.time.Clock()
        self.game_over = False

        # Load a maze
        self.maze = Maze(Maze.load_maze(os.path.join(os.getcwd(), "maze_samples", "maze2d_001.npy")))
        self.maze_size = self.maze.maze_size
        self.screen = pygame.display.set_mode(self.maze.screen_size)

        # Create a background
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.background.fill((255, 255, 255))

        # Create a layer for the maze
        self.maze_layer = pygame.Surface(self.screen.get_size()).convert_alpha()
        self.maze_layer.fill((0, 0, 0, 0,))

        # show the maze
        self.__draw_maze()
        self.view_update()

    def controller_update(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_over = True
                self.quit_game()

    def view_update(self):

        self.clock.tick(60)
        self.screen.blit(self.background, (0, 0))
        self.screen.blit(self.maze_layer,(0, 0))
        pygame.display.flip()

    def quit_game(self):
        pygame.display.quit()
        pygame.quit()

    def __draw_maze(self):

        # drawing the horizontal lines
        for y in range(self.maze.MAZE_H):
            pygame.draw.line(self.maze_layer, (0, 0, 0, 255), (0, y * self.maze.CELL_H),
                             (self.maze.SCREEN_W, y * self.maze.CELL_H))
        # drawing the vertical lines
        for x in range(self.maze.MAZE_W):
            pygame.draw.line(self.maze_layer, (0, 0, 0, 255), (x * self.maze.CELL_W, 0),
                             (x * self.maze.CELL_W, self.maze.SCREEN_H))

        # breaking the walls
        for x in range(len(self.maze.maze_cells)):
            for y in range (len(self.maze.maze_cells[x])):
                # check the which walls are open in each cell
                walls_status = self.maze.get_walls_status(self.maze.maze_cells[x, y])
                dirs = ""
                for dir, open in walls_status.items():
                    if open:
                        dirs += dir
                self.__cover_walls(x, y, dirs)

    def __cover_walls(self, x, y, dirs, colour=(0, 0, 255, 15)):

        dx = x * self.maze.CELL_W
        dy = y * self.maze.CELL_H

        if not isinstance(dirs, str):
            raise TypeError("dirs must be a str.")

        for dir in dirs:
            if dir == "S":
                line_head = (dx + 1, dy + self.maze.CELL_H)
                line_tail = (dx + self.maze.CELL_W - 1, dy + self.maze.CELL_H)
            elif dir == "N":
                line_head = (dx + 1, dy)
                line_tail = (dx + self.maze.CELL_W - 1, dy)
            elif dir == "W":
                line_head = (dx, dy + 1)
                line_tail = (dx, dy + self.maze.CELL_H - 1)
            elif dir == "E":
                line_head = (dx + self.maze.CELL_W, dy + 1)
                line_tail = (dx + self.maze.CELL_W, dy + self.maze.CELL_H - 1)
            else:
                raise ValueError("The only valid directions are (N, S, E, W).")

            pygame.draw.line(self.maze_layer, colour, line_head, line_tail)

class Maze:

    def __init__(self, maze_cells=None, screen_size=(640,640), maze_size=(10,10)):

        # maze member variables
        self.maze_cells = maze_cells

        # Setting the screen size
        if not (isinstance(screen_size, (list, tuple)) and len(screen_size) == 2):
            raise ValueError("screen_size must be a tuple: (width, height).")
        self.screen_size = screen_size

        # Use existing one if exists
        if self.maze_cells is not None:
            if isinstance(self.maze_cells, (np.ndarray, np.generic)) and len(self.maze_cells.shape) == 2:
                self.maze_size = tuple(maze_cells.shape)
            else:
                raise ValueError("maze_cells must be a 2D NumPy array.")
        # Otherwise, generate a random one
        else:
            # maze's configuration parameters
            if not (isinstance(maze_size, (list, tuple)) and len(maze_size) == 2):
                raise ValueError("maze_size must be a tuple: (width, height).")
            self.maze_size = maze_size


            self._generate_maze()

    def save_maze(self, file_path):

        if not isinstance(file_path, str):
            raise TypeError("Invalid file_path. It must be a str.")

        if not os.path.exists(os.path.dirname(file_path)):
            raise ValueError("Cannot find the directory for %s." % file_path)

        else:
            np.save(file_path, self.maze_cells, allow_pickle=False, fix_imports=True)


    @classmethod
    def load_maze(cls, file_path):

        if not isinstance(file_path, str):
            raise TypeError("Invalid file_path. It must be a str.")

        if not os.path.exists(file_path):
            raise ValueError("Cannot find %s." % file_path)

        else:
            return np.load(file_path, allow_pickle=False, fix_imports=True)

    def _generate_maze(self):

        # list of all cell locations
        self.maze_cells = np.zeros(self.maze_size, dtype=int)

        # Initializing constants and variables needed for maze generation
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
                self.maze_cells[x1, y1] = self.__break_walls(self.maze_cells[x1, y1], self.__get_opposite_wall(dir))

                # push the current cell location to the stack
                cell_stack.append(current_cell)

                # make the this neighbour cell the current cell
                cell_stack.append((x1, y1))

                # increment the visited cell count
                num_cells_visited += 1

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
        walls = {
            "N" : (cell & 0x1) >> 0,
            "E" : (cell & 0x2) >> 1,
            "S" : (cell & 0x4) >> 2,
            "W" : (cell & 0x8) >> 3,
        }
        return walls

    @classmethod
    def all_walls_intact(cls, cell):
        return cell & 0xF == 0

    @classmethod
    def __break_walls(cls, cell, dirs):
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
    def __get_opposite_wall(cls, dirs):

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


if __name__ == "__main__":

    maze_view = MazeView2D()

    while not maze_view.game_over:
        maze_view.view_update()
        maze_view.controller_update()



