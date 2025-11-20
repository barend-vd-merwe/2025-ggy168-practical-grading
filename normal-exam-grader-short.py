import streamlit as st
import cv2 as cv
import pandas as pd
import numpy as np
from PIL import Image
import io
import re

st.title("Normal Exam - Short Questions")
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
    st.write("Answer: 54.71 m ± 0.1 m")
    q1 = warped_img[29:29+85, 382:382+327]
    st.image(q1)
    q1 = st.number_input("Question 1", min_value = 0, max_value = 4)

    st.header("Question 2")
    st.write('''Answer: No, the river is not meandering since the sinuosity if 1.06 and a sinuosity of
    1.5 is needed for a river to be considered meandering.''')
    q2_img = warped_img[92:92+194, 382:382+327]
    st.image(q2_img)
    q2 = st.number_input("Question 2", min_value = 0, max_value = 4)

    st.header("Question 3")
    st.write("Answer: 0.000 000 063 94")
    q3a_img =warped_img[269:269+93, 382:382+327]
    st.image(q3a_img)
    q3a = st.number_input("Question 3a", min_value = 0, max_value = 2)

    st.write('''Answer: Yes, the reynold's number of 8000 is greater than 750. This indicates that the flow is turbulent and
     therefore vertical mixing is taking place.''')
    q3b_img = warped_img[349:349+108, 0:712]
    st.image(q3b_img)
    q3b = st.number_input("Question 3b", min_value = 0, max_value = 3)

    st.header("Question 4")
    st.write("Answer: Convex")
    q4_img = warped_img[442:442+51, 0:215]
    st.image(q4_img)
    q4 = st.number_input("Question 4", min_value = 0, max_value = 2)

    st.header("Question 5")
    st.write("Plan: Concave, Profile: Concave")
    q5_img = warped_img[442:442+51, 210:210+502]
    st.image(q5_img)
    q5 = st.number_input("Question 5", min_value = 0, max_value = 2)

    st.header("Question 6")
    st.write("Answer: 64 499 N")
    q6_img = warped_img[474:474+46, 0:211]
    st.image(q6_img)
    q6 = st.number_input("Question 6", min_value = 0, max_value = 4)

    st.header("Question 7")
    st.write("Answer: 0.74")
    q7a_img = warped_img[474:474+46, 203:203+197]
    st.image(q7a_img)
    q7a = st.number_input("Question 7a", min_value = 0, max_value = 4)

    st.write("Answer: Normal")
    q7b_img = warped_img[474:474+46, 397:397+313]
    st.image(q7b_img)
    q7b = st.number_input("Question 7b", min_value = 0, max_value = 2)

    st.header("Question 8")
    st.write("Answer: Strahler = 4")
    q8_strahler_img = warped_img[553:553+378, 0:353]
    st.image(q8_strahler_img)
    q8a = st.number_input("Question 4 Strahler", min_value = 0, max_value = 4)

    st.write("Answer: Shreve = 14")
    q8_shreve_img = warped_img[553:553+378, 351:351+361]
    st.image(q8_shreve_img)
    q8b = st.number_input("Question 4 Shreve", min_value = 0, max_value = 4)

    
    

    
    

    if st.button("Grade"):
        
        final_grade = q1 + q2 + q3a + q3b + q4 + q5 + q6 + q7a + q7b + q8a + q8b

        cv.putText(img=warped_img, text=f'{final_grade}', org=(260,37+51),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        # grades
        cv.rectangle(img=warped_img, pt1=(331,19), pt2=(331+66,19+322),
                     color=(255,255,255), thickness=-1)
        cv.putText(img=warped_img, text=f'Q1 : {q1}', org=(331,20+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q2 : {q2}', org=(331,40+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q3a  : {q3a}', org=(331,60+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q3b : {q3b}', org=(331,80+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q4 : {q4}', org=(331,100+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q5  : {q5}', org=(331,120+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q6  : {q6}', org=(331,140+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q7a  : {q7a}', org=(331,160+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q7b  : {q7b}', org=(331,180+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q8 : {q8a + q8b}', org=(331,200+20),
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
        









    
    
