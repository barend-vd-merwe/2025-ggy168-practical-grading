import streamlit as st
import cv2 as cv
import pandas as pd
import numpy as np
from PIL import Image
import io
import re

st.title("Practical 1 - Coordinate Conversion")
st.info("""
INSTRUCTIONS
- Select the student submission
- Grade by using the appropriate checkboxes
- Download the graded copy and save in a folder (you will be sending these to the senior tutor)
""")


image = st.file_uploader("Select the student submission", type = ["png", "jpg"])

if image is not None:
    # get filename
    filename = image.name
    # get image
    img_bytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    img = cv.imdecode(img_bytes, 1)
    # convert image to gray
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # detect aruco markers
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_250)
    parameters =  cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(dictionary, parameters)
    corners, ids, _ = detector.detectMarkers(img)
    ids = np.concatenate(ids, axis=0).tolist()
    IMG_WIDTH = 712
    IMG_HEIGHT = 972
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

    working_img = np.float32([[point1[0], point1[1]],[point2[0], point2[1]],[point3[0],point3[1]],[point4[0],point4[1]]])
    working_target = np.float32([[0,0],[IMG_WIDTH, 0],[IMG_WIDTH,IMG_HEIGHT],[0,IMG_HEIGHT]])
    transformation_matrix = cv.getPerspectiveTransform(working_img, working_target)
    warped_img = cv.warpPerspective(img, transformation_matrix, (IMG_WIDTH, IMG_HEIGHT))

    #question 1 roi
    q1_region = warped_img[16:16+359,382:382+349]
    q2_region = warped_img[339:339+359,0:0+730]

    st.header("Question 1")
    st.image(q1_region)
    
    num1 = st.number_input(label = "Q1a) Latitude", min_value = 0, max_value = 2)
    num2 = st.number_input(label = "Q1a) Longitude", min_value = 0, max_value = 2)

    num3 = st.number_input(label = "Q1b) Latitude", min_value = 0, max_value = 2)
    num4 = st.number_input(label = "Q1b) Longitude", min_value = 0, max_value = 2)

    num5 = st.number_input(label = "Q1c) Latitude", min_value = 0, max_value = 2)
    num6 = st.number_input(label = "Q1c) Longitude", min_value = 0, max_value = 2)

    num7 = st.number_input(label = "Q1d) Latitude", min_value = 0, max_value = 2)
    num8 = st.number_input(label = "Q1d) Longitude", min_value = 0, max_value = 2)

    st.header("Question 2")
    st.image(q2_region)

    num9 = st.number_input(label = "Q2a) Latitude", min_value = 0, max_value = 2)
    num10 = st.number_input(label = "Q2a) Longitude", min_value = 0, max_value = 2)

    num11 = st.number_input(label = "Q2b) Latitude", min_value = 0, max_value = 2)
    num12 = st.number_input(label = "Q2b) Longitude", min_value = 0, max_value = 2)

    num13 = st.number_input(label = "Q2c) Latitude", min_value = 0, max_value = 2)
    num14 = st.number_input(label = "Q2c) Longitude", min_value = 0, max_value = 2)

    num15 = st.number_input(label = "Q2d) Latitude", min_value = 0, max_value = 2)
    num16 = st.number_input(label = "Q2d) Longitude", min_value = 0, max_value = 2)

    if st.button("Grade"):

        q1 = num1 + num2 + num3 + num4 + num5 + num6 + num7 + num8
        q2 = num9 + num10 + num11 + num12 + num13 + num14 + num15 + num16
        final_grade = q1+ q2
        cv.putText(img=warped_img, text=f'{final_grade}', org=(260+50,37+51),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        
        cv.putText(img=warped_img, text=f'{num1}', org=(625,20+41),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{num2}', org=(625,58+41),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{num3}', org=(625,96+41),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{num4}', org=(625,134+41),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{num5}', org=(625,172+41),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{num6}', org=(625,210+41),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{num7}', org=(625,248+41),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{num8}', org=(625,286+41),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)

        cv.putText(img=warped_img, text=f'{num9}', org=(255,347+41),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{num10}', org=(255,385+41),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{num11}', org=(255,423+41),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{num12}', org=(255,461+41),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{num13}', org=(255,499+41),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{num14}', org=(255,537+41),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{num15}', org=(255,575+41),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{num16}', org=(255,613+41),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        

        

        
        final_rgb = cv.cvtColor(warped_img, cv.COLOR_BGR2RGB)

        st.image(final_rgb)
        
        final_pil = Image.fromarray(final_rgb)
        buffer = io.BytesIO()
        final_pil.save(buffer, format = "PNG")
        st.download_button(label=f"Download {filename}", data=buffer, file_name=filename, mime="image/jpeg")
        









    
    

