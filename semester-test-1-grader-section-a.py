import streamlit as st
import cv2 as cv
import pandas as pd
import numpy as np
from PIL import Image
import io
import re

st.title("Semester Test 1 Short Questions")
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

    st.image(warped_img)

    st.header("Question 1")
    q1_img = warped_img[58:58+194,328:328+382]
    st.image(q1_img)
        
    q1a = st.number_input("Question 1 (a)", min_value = 0, max_value = 2)
    q1b = st.number_input("Question 1 (b)", min_value = 0, max_value = 3)

    st.header("Question 2")
    q2_img = warped_img[230:230+70,328:328+382]
    st.image(q2_img)
    
    q2 = st.number_input("Question 2", min_value = 0, max_value = 2)

    st.header("Question 3")
    q3_img = warped_img[266:266+228,0:712]
    st.image(q3_img)
    
    q3a = st.number_input("Question 3 (a)", min_value = 0, max_value = 1)
    q3b = st.number_input("Question 3 (b)", min_value = 0, max_value = 2)

    st.header("Question 4")
    q4_img = warped_img[480:480+66,0:257]
    st.image(q4_img)
    q4 = st.number_input("Question 4", min_value = 0, max_value = 2)

    st.header("Question 5")
    q5_img = warped_img[480:480+66,248:248+243]
    st.image(q5_img)
    q5 = st.number_input("Question 5", min_value = 0, max_value = 2)

    st.header("Question 6")
    q6_img = warped_img[480:480+66,479:479+231]
    st.image(q6_img)
    q6 = st.number_input("Question 6", min_value = 0, max_value = 1)

    st.header("Question 7")
    q7_img = warped_img[521:521+66,0:264]
    st.image(q7_img)
    q7 = st.number_input("Question 7", min_value = 0, max_value = 2)

    st.header("Question 8")
    q8_img = warped_img[513:513+228,0:712]
    st.image(q8_img)
    q8a = st.number_input("Question 8 (a)", min_value = 0, max_value = 1)
    q8b = st.number_input("Question 8 (b)", min_value = 0, max_value = 2)

    st.header("Question 9")
    q9_img = warped_img[720:720+58,0:253]
    st.image(q9_img)
    q9 = st.number_input("Question 9", min_value = 0, max_value = 1)

    st.header("Question 10")
    q10_img = warped_img[720:720+58,246:246+234]
    st.image(q10_img)
    q10 = st.number_input("Question 10", min_value = 0, max_value = 2)

    st.header("Question 11")
    q11_img = warped_img[717:717+226,0:712]
    st.image(q11_img)
    q11a = st.number_input("Question 11 (a)", min_value = 0, max_value = 1)
    q11b = st.number_input("Question 11 (b)", min_value = 0, max_value = 2)


    
    if st.button("Grade"):
        final_grade = q1a + q1b + q2 + q3a + q3b + q4 + q5 + q6 + q7 + q8a + q8b + q9 + q10 +q11a + q11b

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
        cv.putText(img=warped_img, text=f'Q2  : {q2}', org=(331,60+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q3a : {q3a}', org=(331,80+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q3b : {q3b}', org=(331,100+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q4  : {q4}', org=(331,120+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q5  : {q5}', org=(331,140+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q6  : {q6}', org=(331,160+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q7  : {q7}', org=(331,180+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q8a : {q8a}', org=(331,200+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q8b : {q8b}', org=(331,220+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q9  : {q9}', org=(331,240+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q10 : {q10}', org=(331,260+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q11a: {q11a}', org=(331,280+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q11b: {q11b}', org=(331,300+20),
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
        









    
    

