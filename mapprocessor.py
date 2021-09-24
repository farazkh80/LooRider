# import the necessary packages
from imutils.perspective import four_point_transform
from skimage.segmentation import clear_border
import numpy as np
import imutils
import cv2

# Function to find the grid map in the picture
def find_map(image, debug=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 3)
    # apply threshold
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    thresh = cv2.bitwise_not(thresh)
    if debug:
        cv2.imshow("map Thresh", thresh)
        cv2.imwrite("visuals\\map_thresh.png", thresh)
        cv2.waitKey(0)
    
    # find contours and sort them in order
    cnts = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    mapCnt = None

    for c in cnts:  # loop over the contours
        peri = cv2.arcLength(c, True)  # approximate the contour
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4: # find the biggest contour which is a square
            mapCnt = approx
            break
    if mapCnt is None:
        raise Exception(("NO MAP FOUND"))

    # check to see if we are visualizing the outline of the detected map
    if debug:
        # draw the contour of the map on the image and then display
        # it to our screen for visualization/debugging purposes
        output = image.copy()
        cv2.drawContours(output, [mapCnt], -1, (0, 255, 0), 2)
        cv2.imshow("map Outline", output)
        cv2.imwrite("visuals\\map_outline.png", output)
        cv2.waitKey(0)
    
    # apply a four point perspective transform to both the original
    # image and grayscale image to obtain a top-down bird's eye view
    # of the map
    map = four_point_transform(image, mapCnt.reshape(4, 2))
    warped = four_point_transform(gray, mapCnt.reshape(4, 2))
    # check to see if we are visualizing the perspective transform
    if debug:
        # show the output warped image (again, for debugging purposes)
        cv2.imshow("map Transform", map)
        cv2.imwrite("visuals\\map_transform.png", map)
        cv2.waitKey(0)

    # return a 2-tuple of map in both RGB and grayscale
    return (map, warped)

# Function to check if a cell of a grid is white or black
def is_not_empty(cell):
    # apply automatic thresholding to the cell and then clear any
    # connected borders that touch the border of the cell
    thresh = cv2.threshold(cell, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    thresh = clear_border(thresh)

    # find and order contours in the thresholded cell
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    # if no contours were found than this is an empty cell(white cell)
    if not (len(cnts) == 0):
        # otherwise, find the  contour in the cell and create a
        # mask for the contour
        c = max(cnts, key=cv2.contourArea)
        mask = np.zeros(thresh.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)

        # compute the percentage of masked pixels relative to the total
        # area of the image
        (h, w) = thresh.shape
        percentFilled = cv2.countNonZero(mask) / float(w * h)

        # if less than 2% of the mask is filled then we are looking at
        # noise and can safely ignore the contour
        # it is a black cell
        if percentFilled > 0.02:
            return True

    # otherwise it is a white cell and return None
    return None


def process_image(image_file_path='map-pictures/test2.jpg'):

    image = image_file_path
    image = cv2.imread(image)
    image = imutils.resize(image, width=600)

    # find the map
    (mapImage, warped) = find_map(image, 1)

    # initialize our 6x6  board for corrdinates
    board = np.zeros((6, 6), dtype="int")

    # a map is a 6x6 grid (36 individual cells), so we can
    # infer the location of each cell by dividing the warped image
    # into a 6x6 grid
    stepX = warped.shape[1] // 6
    stepY = warped.shape[0] // 6

    # initialize a list to store the (x, y)-coordinates of each cell
    # location
    cellLocs = []
    counter = 0
    # loop over the grid locations
    for y in range(0, 6):
        # initialize the current list of cell locations
        row = []

        for x in range(0, 6):
            # compute the starting and ending (x, y)-coordinates of the
            # current cell
            startX = x * stepX
            startY = y * stepY
            endX = (x + 1) * stepX
            endY = (y + 1) * stepY

            # add the (x, y)-coordinates to our cell locations list
            row.append((startX, startY, endX, endY))

            # crop the cell from the warped transform image and then
            # extract the cellColor from the cell
            cell = warped[startY:endY, startX:endX]
            image = mapImage[startY:endY, startX:endX]
            cellColor = is_not_empty(cell)

            if cellColor == None:
                board[x, y] = 0

            elif cellColor == True:

                board[x, y] = 1

            # to detect the color of the car which is red
            # use lower upper color boundary masking
            boundaries = [([0, 0, 80], [80, 86, 255])]
            for (lower, upper) in boundaries:
                lower = np.array(lower, dtype = "uint8")
                upper = np.array(upper, dtype = "uint8")
                mask = cv2.inRange(image, lower, upper)
                (h, w) = mask.shape
                percentFilled = cv2.countNonZero(mask) / float(w * h)

                if percentFilled>0.1:
                    board[x,y] = 3 # detect the car

            # to detect the color of the dest which is green
            # use lower upper color boundary masking
            boundaries = [([0, 40, 0], [80,255,80])]
            for (lower, upper) in boundaries:
                lower = np.array(lower, dtype = "uint8")
                upper = np.array(upper, dtype = "uint8")
                mask = cv2.inRange(image, lower, upper)
                (h, w) = mask.shape
                percentFilled = cv2.countNonZero(mask) / float(w * h)

                if percentFilled>0.5:
                    board[x,y] = 4
            
            output = cv2.bitwise_and(image, image, mask = mask)
            cv2.imwrite("visuals\\cells\\cell{}.png".format(counter), np.hstack([image, output]))
            counter+=1

        # add the row to our cell locations
        cellLocs.append(row)


    return board

def main():
    process_image('map-pictures/test2.jpg')

if __name__=="__main__":
    main()