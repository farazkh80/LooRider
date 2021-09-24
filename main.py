# import the necessary packages
from mapprocessor import process_image
from finder import find_the_path
from pytocs import updatePath

def main():
    # process the map image and get the board of coords
    board = process_image('map-pictures/test2.jpg')
    # get the directions for shortest path
    directions = find_the_path(board)
    # update path
    updatePath(directions)

if __name__=="__main__":
    main()



