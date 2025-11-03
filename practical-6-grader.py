import streamlit as st
import cv2 as cv
import pandas as pd
import numpy as np
from PIL import Image
import io
import re

st.title("Practical 6 - Contours")
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
    cv.line(warped_img, (539,371), (539,371+56), (0,0,255), 2)
    cv.line(warped_img, (364,371+56), (714,371+56), (0,0,255), 2)
    st.image(warped_img)

    st.header("Question 1")
    st.write("To grade this question only consider the following:")
    st.write("1) Does the vertical extnet of the profile approximately equal the height of the vertical line on the memo.")
    st.write("2) Does the horizontal extent of the profile approximately equal the length of the horizontal line.")
    st.write("3) The shape of must approximate that given on the model answer.")
    q1 = warped_img[347:347+222, 0:714]
    st.image(q1)
    q1_height = st.number_input("Height of profile", min_value = 0, max_value = 2)
    q1_width = st.number_input("Width of profile", min_value = 0, max_value = 2)
    q1_shape = st.number_input("Shape pf profile", min_value = 0, max_value = 2)

    st.header("Question 2")
    st.write("Answer: 1:1785 to 1:1787")
    st.write("Answer should be in ratio format. If not, subtract 1 mark")
    q2_img = warped_img[553:553+88,585:585+126]
    st.image(q2_img)
    q2 = st.number_input("Question 2", min_value = 0, max_value = 2)

    st.header("Question 3")
    st.write("Answer: 21.12° (± 2°)")
    st.write("If degree (°) symbol is absent, subtract 1 mark.")
    q3_img =warped_img[619:619+88, 586:586+126]
    st.image(q3_img)
    q3 = st.number_input("Question 3", min_value = 0, max_value = 2)

    st.header("Question 4")
    st.write("Answer: 127.5° (± 4°)")
    st.write("If degree (°) symbol is absent, subtract 1 mark.")
    q4_img = warped_img[674:674+88, 585:585+126]
    st.image(q4_img)
    q4 = st.number_input("Question 4", min_value = 0, max_value = 2)
    

    
    

    if st.button("Grade"):
        
        q1_grade = q1_height + q1_width + q1_shape
        
        
        
        final_grade = q1_grade + q2 + q3 + q4
        cv.putText(img=warped_img, text=f'{final_grade}', org=(263,31+50),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        #grades
        cv.rectangle(img=warped_img, pt1=(331,19), pt2=(331+66,19+322),
                     color=(255,255,255), thickness=-1)
        cv.putText(img=warped_img, text=f'Q1 : {q1_grade}', org=(331,20+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q2 : {q2}', org=(331,40+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q3 : {q3}', org=(331,60+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q4 : {q4}', org=(331,80+20),
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
        









    
    
