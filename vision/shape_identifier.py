import cv2
from object_identifier import *
import image_util

class ShapeIdentifier(ObjectIdentifier):
    def ProcessImage(self):
        img = cv2.imread(self.img_path_)
        image_util.WorkingArea(img)
        for i in range (560):
            for j in range(560):
                if img[i][j][2] == 255 and img[i][j][0] < 200 and img[i][j][1] < 200:
                    img[i][j][0] = 0
                    img[i][j][1] = 0
        # converting image into grayscale image
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # setting threshold of gray image
        _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        # using a findContours() function
        contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        i = 0

        # list for storing names of shapes
        for contour in contours:

            # here we are ignoring first counter because
            # findcontour function detects whole image as shape
            if i == 0:
                i = 1
                continue

            # cv2.approxPloyDP() function to approximate the shape
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
            
            # using drawContours() function
            cv2.drawContours(img, [contour], 0, (0, 255, 0), 5)

            # finding center point of shape
            M = cv2.moments(contour)
            x = 0.0
            y = 0.0
            if M['m00'] != 0.0:
                x = int(M['m10']/M['m00'])
                y = int(M['m01']/M['m00'])

            # putting shape name at center of each shape
            if len(approx) <= 7:
                # Square
                cv2.putText(img, 'Square', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                self.obs['achieved_goal'][0], self.obs['achieved_goal'][1] = image_util.ToRealWorld(y,x)
            else:
                # Circle
                cv2.putText(img, 'Circle', (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                self.obs['desired_goal'][0], self.obs['desired_goal'][1] = image_util.ToRealWorld(y,x)

    def GetCordinates(self, obs):
        if (self.cont == 0):
            self.cont += 1
            return self.obs
        return obs