import os
import cv2 as cv
import cv2.aruco as aruco
import numpy as np
import pandas as pd
from PIL import Image

# variables

IMG_WIDTH = int(712)
IMG_HEIGHT = int(972)

# create list of files
files = []
filepath = []
for file in os.listdir("image-files"):
    if file.endswith(".png"):
        files.append(file)

for file in files:
    file = f"image-files/{file}"
    filepath.append(file)

for file in filepath:
    # load image file
        img = cv.imread(file)
        df = pd.read_csv("gc.csv")
        try:
            # detect aruco markers
            aruco_dictionary = cv.aruco.Dictionary_get(aruco.DICT_4X4_50)
            aruco_params = cv.aruco.DetectorParameters_create()
            corners, ids, _ = cv.aruco.detectMarkers(img, aruco_dictionary, parameters=aruco_params)
            ids = np.concatenate(ids, axis=0).tolist()

            # get the coordinates of the aruco markers in each corner
            aruco_index_top_left= ids.index(0)
            aruco_coords_top_left = corners[aruco_index_top_left]
            point1 = aruco_coords_top_left[0][0]

            aruco_index_top_right= ids.index(1)
            aruco_coords_top_right = corners[aruco_index_top_right]
            point2 = aruco_coords_top_right[0][1]

            aruco_index_bottom_right = ids.index(3)
            aruco_coords_bottom_right = corners[aruco_index_bottom_right]
            point3 = aruco_coords_bottom_right[0][2]

            aruco_index_bottom_left = ids.index(2)
            aruco_coords_bottom_left = corners[aruco_index_bottom_left]
            point4 = aruco_coords_bottom_left[0][3]

            # creater working image
            working_img = np.float32([[point1[0], point1[1]],[point2[0], point2[1]],[point3[0],point3[1]],[point4[0],point4[1]]])
            working_target = np.float32([[0,0],[IMG_WIDTH, 0],[IMG_WIDTH,IMG_HEIGHT],[0,IMG_HEIGHT]])
            transformation_matrix = cv.getPerspectiveTransform(working_img, working_target)
            warped_img = cv.warpPerspective(img, transformation_matrix, (IMG_WIDTH, IMG_HEIGHT))

            # extract columns of bubble sheet
            col_1_img = warped_img[52:52+190 ,90:90+14]
            col_2_img = warped_img[53:53+191 ,109:109+14]
            col_3_img = warped_img[53:53+191 ,128:128+14]
            col_4_img = warped_img[53:53+191 ,148:148+14]
            col_5_img = warped_img[53:53+191 ,167:167+14]
            col_6_img = warped_img[53:53+191 ,186:186+14]
            col_7_img = warped_img[53:53+191 ,205:205+14]
            col_8_img = warped_img[53:53+191 ,224:224+14]

            # aplly thresholding
            col_1_img = cv.cvtColor(col_1_img, cv.COLOR_BGR2GRAY)
            col_1_blur = cv.blur(col_1_img, (3,3))
            col_1_thresh = cv.threshold(col_1_blur, 175, 255, cv.THRESH_BINARY_INV)[1]

            col_2_img = cv.cvtColor(col_2_img, cv.COLOR_BGR2GRAY)
            col_2_blur = cv.blur(col_2_img, (3,3))
            col_2_thresh = cv.threshold(col_2_blur, 175, 255, cv.THRESH_BINARY_INV)[1]

            col_3_img = cv.cvtColor(col_3_img, cv.COLOR_BGR2GRAY)
            col_3_blur = cv.blur(col_3_img, (3,3))
            col_3_thresh = cv.threshold(col_3_blur, 175, 255, cv.THRESH_BINARY_INV)[1]

            col_4_img = cv.cvtColor(col_4_img, cv.COLOR_BGR2GRAY)
            col_4_blur = cv.blur(col_4_img, (3,3))
            col_4_thresh = cv.threshold(col_4_blur, 175, 255, cv.THRESH_BINARY_INV)[1]

            col_5_img = cv.cvtColor(col_5_img, cv.COLOR_BGR2GRAY)
            col_5_blur = cv.blur(col_5_img, (3,3))
            col_5_thresh = cv.threshold(col_5_blur, 175, 255, cv.THRESH_BINARY_INV)[1]

            col_6_img = cv.cvtColor(col_6_img, cv.COLOR_BGR2GRAY)
            col_6_blur = cv.blur(col_6_img, (3,3))
            col_6_thresh = cv.threshold(col_6_blur, 175, 255, cv.THRESH_BINARY_INV)[1]

            col_7_img = cv.cvtColor(col_7_img, cv.COLOR_BGR2GRAY)
            col_7_blur = cv.blur(col_7_img, (3,3))
            col_7_thresh = cv.threshold(col_7_blur, 175, 255, cv.THRESH_BINARY_INV)[1]

            col_8_img = cv.cvtColor(col_8_img, cv.COLOR_BGR2GRAY)
            col_8_blur = cv.blur(col_8_img, (3,3))
            col_8_thresh = cv.threshold(col_8_blur, 175, 255, cv.THRESH_BINARY_INV)[1]
            try:
                # extract student number
                col_vals = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

                col_1_values = []
                col_1_values.append(col_1_thresh[0:14].mean())
                col_1_values.append(col_1_thresh[20:20+14].mean())
                col_1_values.append(col_1_thresh[39:39+14].mean())
                col_1_values.append(col_1_thresh[59:59+14].mean())
                col_1_values.append(col_1_thresh[78:78+14].mean())
                col_1_values.append(col_1_thresh[98:98+14].mean())
                col_1_values.append(col_1_thresh[117:117+14].mean())
                col_1_values.append(col_1_thresh[137:137+14].mean())
                col_1_values.append(col_1_thresh[156:156+14].mean())
                col_1_values.append(col_1_thresh[176:176+14].mean())
                col_1_max = max(col_1_values)
                col_1_index = col_1_values.index(col_1_max)
                digit1 = col_vals[col_1_index]

                col_2_values = []
                col_2_values.append(col_2_thresh[0:14].mean())
                col_2_values.append(col_2_thresh[20:20+14].mean())
                col_2_values.append(col_2_thresh[39:39+14].mean())
                col_2_values.append(col_2_thresh[59:59+14].mean())
                col_2_values.append(col_2_thresh[78:78+14].mean())
                col_2_values.append(col_2_thresh[98:98+14].mean())
                col_2_values.append(col_2_thresh[117:117+14].mean())
                col_2_values.append(col_2_thresh[137:137+14].mean())
                col_2_values.append(col_2_thresh[156:156+14].mean())
                col_2_values.append(col_2_thresh[176:176+14].mean())
                col_2_max = max(col_2_values)
                col_2_index = col_2_values.index(col_2_max)
                digit2 = col_vals[col_2_index]

                col_3_values = []
                col_3_values.append(col_3_thresh[0:14].mean())
                col_3_values.append(col_3_thresh[20:20+14].mean())
                col_3_values.append(col_3_thresh[39:39+14].mean())
                col_3_values.append(col_3_thresh[59:59+14].mean())
                col_3_values.append(col_3_thresh[78:78+14].mean())
                col_3_values.append(col_3_thresh[98:98+14].mean())
                col_3_values.append(col_3_thresh[117:117+14].mean())
                col_3_values.append(col_3_thresh[137:137+14].mean())
                col_3_values.append(col_3_thresh[156:156+14].mean())
                col_3_values.append(col_3_thresh[176:176+14].mean())
                col_3_max = max(col_3_values)
                col_3_index = col_3_values.index(col_3_max)
                digit3 = col_vals[col_3_index]

                col_4_values = []
                col_4_values.append(col_4_thresh[0:14].mean())
                col_4_values.append(col_4_thresh[20:20+14].mean())
                col_4_values.append(col_4_thresh[39:39+14].mean())
                col_4_values.append(col_4_thresh[59:59+14].mean())
                col_4_values.append(col_4_thresh[78:78+14].mean())
                col_4_values.append(col_4_thresh[98:98+14].mean())
                col_4_values.append(col_4_thresh[117:117+14].mean())
                col_4_values.append(col_4_thresh[137:137+14].mean())
                col_4_values.append(col_4_thresh[156:156+14].mean())
                col_4_values.append(col_4_thresh[176:176+14].mean())
                col_4_max = max(col_4_values)
                col_4_index = col_4_values.index(col_4_max)
                digit4 = col_vals[col_4_index]

                col_5_values = []
                col_5_values.append(col_5_thresh[0:14].mean())
                col_5_values.append(col_5_thresh[20:20+14].mean())
                col_5_values.append(col_5_thresh[39:39+14].mean())
                col_5_values.append(col_5_thresh[59:59+14].mean())
                col_5_values.append(col_5_thresh[78:78+14].mean())
                col_5_values.append(col_5_thresh[98:98+14].mean())
                col_5_values.append(col_5_thresh[117:117+14].mean())
                col_5_values.append(col_5_thresh[137:137+14].mean())
                col_5_values.append(col_5_thresh[156:156+14].mean())
                col_5_values.append(col_5_thresh[176:176+14].mean())
                col_5_max = max(col_5_values)
                col_5_index = col_5_values.index(col_5_max)
                digit5 = col_vals[col_5_index]

                col_6_values = []
                col_6_values.append(col_6_thresh[0:14].mean())
                col_6_values.append(col_6_thresh[20:20+14].mean())
                col_6_values.append(col_6_thresh[39:39+14].mean())
                col_6_values.append(col_6_thresh[59:59+14].mean())
                col_6_values.append(col_6_thresh[78:78+14].mean())
                col_6_values.append(col_6_thresh[98:98+14].mean())
                col_6_values.append(col_6_thresh[117:117+14].mean())
                col_6_values.append(col_6_thresh[137:137+14].mean())
                col_6_values.append(col_6_thresh[156:156+14].mean())
                col_6_values.append(col_6_thresh[176:176+14].mean())
                col_6_max = max(col_6_values)
                col_6_index = col_6_values.index(col_6_max)
                digit6 = col_vals[col_6_index]

                col_7_values = []
                col_7_values.append(col_7_thresh[0:14].mean())
                col_7_values.append(col_7_thresh[20:20+14].mean())
                col_7_values.append(col_7_thresh[39:39+14].mean())
                col_7_values.append(col_7_thresh[59:59+14].mean())
                col_7_values.append(col_7_thresh[78:78+14].mean())
                col_7_values.append(col_7_thresh[98:98+14].mean())
                col_7_values.append(col_7_thresh[117:117+14].mean())
                col_7_values.append(col_7_thresh[137:137+14].mean())
                col_7_values.append(col_7_thresh[156:156+14].mean())
                col_7_values.append(col_7_thresh[176:176+14].mean())
                col_7_max = max(col_7_values)
                col_7_index = col_7_values.index(col_7_max)
                digit7 = col_vals[col_7_index]

                col_8_values = []
                col_8_values.append(col_8_thresh[0:14].mean())
                col_8_values.append(col_8_thresh[20:20+14].mean())
                col_8_values.append(col_8_thresh[39:39+14].mean())
                col_8_values.append(col_8_thresh[59:59+14].mean())
                col_8_values.append(col_8_thresh[78:78+14].mean())
                col_8_values.append(col_8_thresh[98:98+14].mean())
                col_8_values.append(col_8_thresh[117:117+14].mean())
                col_8_values.append(col_8_thresh[137:137+14].mean())
                col_8_values.append(col_8_thresh[156:156+14].mean())
                col_8_values.append(col_8_thresh[176:176+14].mean())
                col_8_max = max(col_8_values)
                col_8_index = col_8_values.index(col_8_max)
                digit8 = col_vals[col_8_index]

                snumber = f'u{str(digit1)}{str(digit2)}{str(digit3)}{str(digit4)}{str(digit5)}{str(digit6)}{str(digit7)}{str(digit8)}'

                # get details of student
                row_index = df.index[df["Username"] == snumber].tolist()
                surname = df.iloc[row_index,0].values[0]
                first = df.iloc[row_index,1].values[0]
                new_filename = f'renamed/{surname}-{first}-{snumber}.png'

                cv.imwrite(new_filename, img)
                print("Task succesfully completed")
                os.remove(file)
            except:
                print("No/invalid student number")
                cv.imwrite(file, img)
        except:
            print("Task unsuccesful")



