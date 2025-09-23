import streamlit as st
import cv2 as cv
import pandas as pd
import numpy as np
from PIL import Image
import io
import re

st.title("Practical 3 - Direction")
st.info("""
INSTRUCTIONS
- Select the student submission
- Grade by using the appropriate fields
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

    st.image(warped_img)

    st.header("Question 1")
    q1 = warped_img[384:384+106,0:712]
    st.image(q1)

    q1a_f = st.number_input("True Forward (AB)", min_value = 0, max_value = 1)
    q1a_b = st.number_input("True Backward (AB)", min_value = 0, max_value = 1)

    q1b_f = st.number_input("True Forward (BC)", min_value = 0, max_value = 1)
    q1b_b = st.number_input("True Backward (BC)", min_value = 0, max_value = 1)

    q1c_f = st.number_input("True Forward (AC)", min_value = 0, max_value = 1)
    q1c_b = st.number_input("True Backward (AC)", min_value = 0, max_value = 1)


    st.header("Question 2")
    q2 = warped_img[491:491+121,0:712]
    st.image(q2)

    q2a = st.number_input("Question 2 (a)", min_value = 0, max_value = 2)
    q2b = st.number_input("Question 2 (b)", min_value = 0, max_value = 2)
    q2c = st.number_input("Question 2 (c)", min_value = 0, max_value = 2)
    q2d = st.number_input("Question 2 (d)", min_value = 0, max_value = 2)

    
    st.header("Question 3")
    q3 = warped_img[610:610+110,0:712]
    st.image(q3)

    q3a_f = st.number_input("Magnetic Forward (AB)", min_value = 0, max_value = 1)
    q3a_b = st.number_input("Magnetic Backward (AB)", min_value = 0, max_value = 1)

    q3b_f = st.number_input("Magnetic Forward (BC)", min_value = 0, max_value = 1)
    q3b_b = st.number_input("Magnetic Backward (BC)", min_value = 0, max_value = 1)

    q3c_f = st.number_input("Magnetic Forward (AC)", min_value = 0, max_value = 1)
    q3c_b = st.number_input("Magnetic Backward (AC)", min_value = 0, max_value = 1)
    

    if st.button("Grade"):
        
        q1_grade = q1a_f + q1a_b + q1b_f + q1b_b + q1c_f + q1c_b
        q2_grade = q2a + q2b + q2c + q2d
        q3_grade = q3a_f + q3a_b + q3b_f + q3b_b + q3c_f + q3c_b
        
        final_grade = q1_grade + q2_grade + q3_grade
        cv.putText(img=warped_img, text=f'{final_grade}', org=(263,31+50),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        #grades
        cv.rectangle(img=warped_img, pt1=(331,19), pt2=(331+66,19+322),
                     color=(255,255,255), thickness=-1)
        cv.putText(img=warped_img, text=f'Q1a : {q1a_f + q1a_b}', org=(331,20+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q1b : {q1b_f + q1b_b}', org=(331,40+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q1c : {q1c_f + q1c_b}', org=(331,60+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        
        cv.putText(img=warped_img, text=f'Q2a : {q2a}', org=(331,80+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q2b : {q2b}', org=(331,100+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q2c : {q2c}', org=(331,120+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q2d : {q2d}', org=(331,140+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)

        cv.putText(img=warped_img, text=f'Q3a : {q3a_f + q3a_b}', org=(331,160+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q3b : {q3b_f + q3b_b}', org=(331,180+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q3c : {q3c_f + q3c_b}', org=(331,200+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)

        # student details
        cv.rectangle(img=warped_img, pt1=(207,6), pt2=(207+474,0+20),
                     color=(255,255,255), thickness=-1)
        cv.putText(img=warped_img, text=f'{filename}', org=(210,0+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        
        
        final_rgb = cv.cvtColor(warped_img, cv.COLOR_BGR2RGB)

        st.image(final_rgb)
        
        final_pil = Image.fromarray(final_rgb)
        buffer = io.BytesIO()
        final_pil.save(buffer, format = "PNG")
        st.download_button(label=f"Download {filename}", data=buffer, file_name=filename, mime="image/jpeg")
        









    
    
