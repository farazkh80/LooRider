import pygame
from queue import PriorityQueue
import numpy as np

WIDTH = 800
WIN = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("A* Star Path Finding Algorithm")


GREEN = (69, 139, 116)
BLUE = (154, 192, 205)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
PURPLE = (153, 0, 153)
ORANGE = (255, 165, 0)
GREY = (128, 128, 128)
BROWN = (139, 35, 35)
DEEPPINK = (139, 10, 80)
BLUEVIOLET = (138, 43, 226)
YELLOW = (255, 185, 15)
RED = (200,0,0)

pathCoordinates = []


# Spot attributes
class Spot:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.color = WHITE
        self.neighbors = []
        self.width = width
        self.total_rows = total_rows

    def get_pos(self):
        return self.row, self.col

    def is_closed(self):
        return self.color == YELLOW

    def is_open(self):
        return self.color == PURPLE

    def is_barrier(self):
        return self.color == BLACK

    def is_start(self):
        return self.color == RED

    def is_end(self):
        return self.color == GREEN

    def reset(self):
        self.color = WHITE

    def make_closed(self):
        self.color = YELLOW

    def make_open(self):
        self.color = PURPLE

    def make_barrier(self):
        self.color = BLACK

    def make_start(self):
        self.color = RED

    def make_end(self):
        self.color = GREEN

    def make_path(self):
        self.color = BLUE

    def draw(self, win):
        pygame.draw.rect(
            win, self.color, (self.x, self.y, self.width, self.width))

    def update_neighbors(self, grid):
        self.neighbors = []
        # DOWN
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():
            self.neighbors.append(grid[self.row + 1][self.col])

        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():  # UP
            self.neighbors.append(grid[self.row - 1][self.col])

        # RIGHT
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():
            self.neighbors.append(grid[self.row][self.col + 1])

        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():  # LEFT
            self.neighbors.append(grid[self.row][self.col - 1])

    def __lt__(self, other):
        return False


def h(p1, p2):
    x1, y1 = p1
    x2, y2 = p2

    return abs(x1 - x2) + abs(y1 - y2)


# Draws the path
def reconstruct_path(came_from, current, draw, end):
    end.make_path()
    end_row, end_col = end.get_pos()
    append_the_path_coordinates((end_col, end_row))

    while current in came_from:
        current = came_from[current]
        current.make_path()
        row, col = current.get_pos()
        append_the_path_coordinates((col, row))
        draw()


# Runs the algorithm
def algorithm(draw, grid, start, end):
    count = 0
    # open set to store open nodes
    open_set = PriorityQueue()
    # Add start to the Open Set
    open_set.put((0, count, start))
    came_from = {}
    # set g score as infinity for all of them
    g_score = {spot: float("inf") for row in grid for spot in row}
    # set start node's g score as 0
    g_score[start] = 0
    # set set f score as infinity for all of them
    f_score = {spot: float("inf") for row in grid for spot in row}
    # calculate and set the f score for the start node
    f_score[start] = h(start.get_pos(), end.get_pos())

    # adds the start position to open_set_hash
    open_set_hash = {start}

    while not open_set.empty():

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
        # Choose the node with the smallest f_score
        current = open_set.get()[2]
        # remove current from open_set
        open_set_hash.remove(current)

        # if target is found
        if current == end:
            # trace the shorest path way
            reconstruct_path(came_from, end, draw, end)
            return True

        # loop through neighbours
        for neighbor in current.neighbors:
            temp_g_score = g_score[current] + 1

            if temp_g_score < g_score[neighbor]:
                # add the neighbour to the came from tracker
                came_from[neighbor] = current
                # set the g score
                g_score[neighbor] = temp_g_score
                # set the f score
                f_score[neighbor] = temp_g_score + \
                    h(neighbor.get_pos(), end.get_pos())
                # if the neighbor is not already green
                if neighbor not in open_set_hash:
                    # adds the neighbour to the set of green spots
                    count += 1
                    open_set.put((f_score[neighbor], count, neighbor))
                    open_set_hash.add(neighbor)
                    neighbor.make_open()

        draw()
        # close and make the spot red if passed
        if current != start:
            current.make_closed()

    return False


# Function to make a grid
def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            spot = Spot(i, j, gap, rows)
            grid[i].append(spot)

    return grid

# Draw the grid


def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap))
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width))


# Draw pygame grid with the blocks and the start and end points
def draw(win, grid, rows, width):
    win.fill(WHITE)

    for row in grid:
        for spot in row:
            spot.draw(win)

    draw_grid(win, rows, width)
    pygame.display.update()


# Appending the cells of the shortest path
def append_the_path_coordinates(coordinate):
    pathCoordinates.append(coordinate)


# Function to return the cordinates of the shortest ordered path
def return_the_path_coordinates():
    return pathCoordinates


def find_directions(ordered_path_list):

    print(ordered_path_list)

    counter = 0
    directions = []
    heading = "east"
    reached_the_end = False

    while(reached_the_end == False):

        row1, col1 = ordered_path_list[counter]
        row2, col2 = ordered_path_list[counter+1]

        if heading == "east":
            if(row1 == row2):
                if(col2-col1 == 1):
                    directions.append("forward")
                    if(ordered_path_list[len(ordered_path_list)-1] == ordered_path_list[counter + 1]):
                        reached_the_end = True
                    counter += 1

            elif(col1 == col2):
                if(row2-row1 == 1):
                    directions.append("right")
                    heading = "south"

                elif(row1-row2 == 1):
                    directions.append("left")
                    heading = "north"

        elif heading == "west":
            if(row1 == row2):
                if(col1-col2 == 1):
                    directions.append("forward")
                    if(ordered_path_list[len(ordered_path_list)-1] == ordered_path_list[counter + 1]):
                        reached_the_end = True
                    counter += 1

            elif(col1 == col2):
                if(row2-row1 == 1):
                    directions.append("left")
                    heading = "south"

                elif(row1-row2 == 1):
                    directions.append("right")
                    heading = "north"

        elif heading == "north":
            if(col1 == col2):
                if(row1-row2 == 1):
                    directions.append("forward")
                    if(ordered_path_list[len(ordered_path_list)-1] == ordered_path_list[counter + 1]):
                        reached_the_end = True
                    counter += 1

            elif(row1 == row2):

                if(col2-col1 == 1):
                    directions.append("right")
                    heading = "east"

                elif(col1-col2 == 1):
                    directions.append("left")
                    heading = "west"

        elif heading == "south":
            if(col1 == col2):
                if(row2-row1 == 1):
                    directions.append("forward")
                    if(ordered_path_list[len(ordered_path_list)-1] == ordered_path_list[counter + 1]):
                        reached_the_end = True
                    counter += 1
            elif(row1 == row2):
                if(col2-col1 == 1):
                    directions.append("left")
                    heading = "east"

                elif(col1-col2 == 1):
                    directions.append("right")
                    heading = "west"

    print(directions)
    return directions


# Main function to find the path
def find_the_path(board):
    ROWS = 6

    grid = make_grid(ROWS, WIDTH)

    start = None
    end = None

    run = True
    started = False

    board = board.tolist()

    # intiate start and end points

    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == 3:
                start = grid[i][j]
    start.make_start()
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == 4:
                end = grid[i][j]
    end.make_end()

    # draw walls here
    for i in range(len(board)):
        for j in range(len(board)):
            if board[i][j] == 1:
                grid[i][j].make_barrier()

    draw(WIN, grid, ROWS, WIDTH)

    while run:

        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                run = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and start and end:

                    for row in grid:
                        for spot in row:
                            spot.update_neighbors(grid)
                    if algorithm(lambda: draw(WIN, grid, ROWS, WIDTH), grid, start, end):
                        print("found a path, see the purple line.")
                    else:
                        print("impossible")


    pygame.quit()

    reversed_path_list = return_the_path_coordinates()

    # ordered path of coordinates
    ordered_path_list = reversed_path_list[::-1]

    directions = find_directions(ordered_path_list)
    print(directions)
    return directions

    

