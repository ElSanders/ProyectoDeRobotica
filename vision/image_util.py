import cv2

# Given an image delete everything that is not in the work area.
def WorkingArea(img):
    for x in range(560):
        for y in range(560):
            if x < 20 or x > 360 or y < 39 or y > 517:
                img[x][y] = [255,255,255]

# Transform from pixels to real world.
def ToRealWorld(x,y):
    new_x = -0.001379595588235294 * x + 1.5666544117647059 - 0.01358576962
    new_y = -0.001380753138075314 * y + 1.1338493723849374 - -0.0001348925677
    return new_x, new_y

# Show an image using opencv
def ShowImage(img):
    while(True):
        cv2.imshow('img',img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()