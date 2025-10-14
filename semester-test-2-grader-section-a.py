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

    # create working image
    working_img = np.float32([[point1[0], point1[1]],[point2[0], point2[1]],[point3[0],point3[1]],[point4[0],point4[1]]])
    working_target = np.float32([[0,0],[IMG_WIDTH, 0],[IMG_WIDTH,IMG_HEIGHT],[0,IMG_HEIGHT]])
    transformation_matrix = cv.getPerspectiveTransform(working_img, working_target)
    warped_img = cv.warpPerspective(img, transformation_matrix, (IMG_WIDTH, IMG_HEIGHT))
    cv.rectangle(img = warped_img, pt1 = (109,459), pt2 = (109+19,459+19), color = (0,0,255), thickness = 1)
    cv.rectangle(img = warped_img, pt1 = (277,459), pt2 = (277+19,459+19), color = (0,0,255), thickness = 1)

    st.image(warped_img)

    st.header("Question 1")
    q1_img = warped_img[34:34+319, 369:369+343]
    st.image(q1_img)
        
    q1a = st.number_input("Question 1 (a)", min_value = 0, max_value = 1)
    q1b = st.number_input("Question 1 (b)", min_value = 0, max_value = 1)
    q1c = st.number_input("Question 1 (c)", min_value = 0, max_value = 3)

    st.header("Question 2")
    q2_img = warped_img[328:328+53, 0:0+712]
    st.image(q2_img)
    
    q2a = st.number_input("Question 2 (a)", min_value = 0, max_value = 1)
    q2b = st.number_input("Question 2 (b)", min_value = 0, max_value = 1)
    q2c = st.number_input("Question 2 (c)", min_value = 0, max_value = 1)

    st.header("Question 3")
    q3_img = warped_img[378:378+53,0:712]
    st.image(q3_img)
    
    q3a = st.number_input("Question 3 (a)", min_value = 0, max_value = 1)
    q3b = st.number_input("Question 3 (b)", min_value = 0, max_value = 1)

    st.header("Question 4")
    q4_img = warped_img[430:430+265, 0:712]
    st.image(q4_img)
    q4_point_a = st.number_input("Point (a) Zingg", min_value = 0, max_value = 2)
    q4_point_b = st.number_input("Point (b) Zingg", min_value = 0, max_value = 2)
    q4a = st.number_input("Question 4 (a) label", min_value = 0, max_value = 1)
    q4b = st.number_input("Question 4 (b) label", min_value = 0, max_value = 1)

    st.header("Question 5")
    q5_img = warped_img[487:487+219,317:317+395]
    st.image(q5_img)
    q5a = st.number_input("Question 5 (a)", min_value = 0, max_value = 1)
    q5b = st.number_input("Question 5 (b)", min_value = 0, max_value = 2)

    st.header("Question 6")
    q6_img = warped_img[691:691+103,0:712]
    st.image(q6_img)
    q6a = st.number_input("Question 6 (a)", min_value = 0, max_value = 3)
    q6b = st.number_input("Question 6 (b)", min_value = 0, max_value = 1)

    st.header("Question 7")
    q7_img = warped_img[739:739+159, 0:712]
    st.image(q7_img)
    q7a = st.number_input("Question 7 (a)", min_value = 0, max_value = 1)
    q7b = st.number_input("Question 7 (b)", min_value = 0, max_value = 4)

    st.header("Question 8")
    q8_img = warped_img[889:889+54,0:712]
    st.image(q8_img)
    q8a = st.number_input("Question 8 (a)", min_value = 0, max_value = 1)
    q8b = st.number_input("Question 8 (b)", min_value = 0, max_value = 1)




    
    if st.button("Grade"):
        final_grade = q1a + q1b + q1c + q2a + q2b + q2c + q3a + q3b + q4_point_a + q4_point_b + q4a + q4b + q5a + q5b + q6a + q6b + q7a + q7b + q8a + q8b

        cv.putText(img=warped_img, text=f'{final_grade}', org=(260,37+51),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        # grades
        cv.rectangle(img=warped_img, pt1=(331,19), pt2=(331+66,19+322),
                     color=(255,255,255), thickness=-1)
        cv.putText(img=warped_img, text=f'Q1a : {q1a}', org=(331,20+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q1b : {q1b}', org=(331,40+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q1c  : {q1c}', org=(331,60+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q2a : {q2a}', org=(331,80+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q2b : {q2b}', org=(331,100+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q2c  : {q2c}', org=(331,120+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q3a  : {q3a}', org=(331,140+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q3b  : {q3b}', org=(331,160+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'q4a  : {q4_point_a + q4a}', org=(331,180+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q4b : {q4_point_b + q4b}', org=(331,200+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q5a : {q5a}', org=(331,220+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q5b  : {q5b}', org=(331,240+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q6a : {q6a}', org=(331,260+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q6b: {q6b}', org=(331,280+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q7a: {q7a}', org=(331,300+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q7b: {q7b}', org=(331,320+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q8a: {q8a}', org=(331,340+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q8b: {q8b}', org=(331,360+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)

        
        # student details
        cv.rectangle(img=warped_img, pt1=(210,6), pt2=(210+469,6+20),
                     color=(255,255,255), thickness=-1)
        cv.putText(img=warped_img, text=f'{filename}', org=(210,6+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        
        final_rgb = cv.cvtColor(warped_img, cv.COLOR_BGR2RGB)

        st.image(final_rgb)
        
        final_pil = Image.fromarray(final_rgb)
        buffer = io.BytesIO()
        final_pil.save(buffer, format = "PNG")
        st.download_button(label=f"Download {filename}", data=buffer, file_name=filename, mime="image/jpeg")
        









    
    
