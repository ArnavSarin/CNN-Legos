import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

def main():

    LEGO_BLOCKS = {"2x2-" : 351, "1x2-" : 351, "1x1-" : 351, "2x2plate-" : 351, "1x2plate-" :351, "1x1plate-" : 351, "rooftile-" : 351, "peg2-" : 351, "lever-" : 351, "halfbush-" : 351 }
    
    for i in range (0,50):
        for key in LEGO_BLOCKS:
            cv_img = cv2.imread("/Users/arnavsarin/Desktop/CNN/50_IMAGES_TO_RESIZE/" + key +  str(LEGO_BLOCKS.get(key)).zfill(4) + ".png",0)
            print(key + str(i))
            scale_percent = 40
            width = int(cv_img.shape[1] * scale_percent / 100)
            height = int(cv_img.shape[0] * scale_percent / 100)
            dsize = (width, height)
            output = cv2.resize(cv_img, dsize)
            cv2.imwrite("/Users/arnavsarin/Desktop/CNN/80x80_50/" + key + str(LEGO_BLOCKS.get(key)).zfill(4) + ".png", output)
            LEGO_BLOCKS[key] = LEGO_BLOCKS[key] + 1
            
#                pixel_nodes = cv_img.flatten()

if __name__ == "__main__":
    main()
