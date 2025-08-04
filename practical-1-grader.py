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

    # creater working image
    working_img = np.float32([[point1[0], point1[1]],[point2[0], point2[1]],[point3[0],point3[1]],[point4[0],point4[1]]])
    working_target = np.float32([[0,0],[IMG_WIDTH, 0],[IMG_WIDTH,IMG_HEIGHT],[0,IMG_HEIGHT]])
    transformation_matrix = cv.getPerspectiveTransform(working_img, working_target)
    warped_img = cv.warpPerspective(img, transformation_matrix, (IMG_WIDTH, IMG_HEIGHT))

    cv.rectangle(warped_img, (238,484), (238+23,484+31), (0,0,255), 1, 8, 0)
    cv.rectangle(warped_img, (327,599), (327+32,599+34), (0,0,255), 1, 8, 0)
    cv.rectangle(warped_img, (405,643), (405+33,643+21), (0,0,255), 1, 8, 0)
    cv.rectangle(warped_img, (452,785), (452+24,785+32), (0,0,255), 1, 8, 0)

    st.image(warped_img)

    st.header("Question 1")
    chk1a = st.checkbox("Point A latitude", key="chk1a")
    chk1b = st.checkbox("Point A longitude", key="chk1b")
    chk1c = st.checkbox("Point B latitude", key="chk1c")
    chk1d = st.checkbox("Point B longitude", key="chk1d")
    chk1e = st.checkbox("Point C latitude", key="chk1e")
    chk1f = st.checkbox("Point C longitude", key="chk1f")
    chk1g = st.checkbox("Point D latitude", key="chk1g")
    chk1h = st.checkbox("Point D longitude", key="chk1h")

    st.header("Question 2")
    chk2a = st.checkbox("Point E latitude", key="chk2a")
    chk2b = st.checkbox("Point E longitude", key="chk2b")
    chk2c = st.checkbox("Point F latitude", key="chk2c")
    chk2d = st.checkbox("Point F longitude", key="chk2d")
    chk2e = st.checkbox("Point G latitude", key="chk2e")
    chk2f = st.checkbox("Point G longitude", key="chk2f")
    chk2g = st.checkbox("Point H latitude", key="chk2g")
    chk2h = st.checkbox("Point H longitude", key="chk2h")

    q1_grade = 0
    q2_grade = 0
    if st.button("Grade"):
        # question 1
        if st.session_state.chk1a:
            q1_grade += 1
        if st.session_state.chk1b:
            q1_grade += 1
        if st.session_state.chk1c:
            q1_grade += 1
        if st.session_state.chk1d:
            q1_grade += 1
        if st.session_state.chk1e:
            q1_grade += 1
        if st.session_state.chk1f:
            q1_grade += 1
        if st.session_state.chk1g:
            q1_grade += 1
        if st.session_state.chk1h:
            q1_grade += 1

        # question 2
        if st.session_state.chk2a:
            q2_grade += 1
        if st.session_state.chk2b:
            q2_grade += 1
        if st.session_state.chk2c:
            q2_grade += 1
        if st.session_state.chk2d:
            q2_grade += 1
        if st.session_state.chk2e:
            q2_grade += 1
        if st.session_state.chk2f:
            q2_grade += 1
        if st.session_state.chk2g:
            q2_grade += 1
        if st.session_state.chk2h:
            q2_grade += 1

        final_grade = q1_grade + q2_grade
        cv.putText(img=warped_img, text=f'{final_grade}', org=(260,37+51),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{int(st.session_state.chk1a)}', org=(609+30,36+24),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{int(st.session_state.chk1b)}', org=(609+30,72+24),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{int(st.session_state.chk1c)}', org=(609+30,108+24),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{int(st.session_state.chk1d)}', org=(609+30,144+24),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{int(st.session_state.chk1e)}', org=(609+30,180+24),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{int(st.session_state.chk1f)}', org=(609+30,216+24),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{int(st.session_state.chk1g)}', org=(609+30,252+24),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{int(st.session_state.chk1h)}', org=(609+30,288+24),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)

        cv.putText(img=warped_img, text=f'{int(st.session_state.chk2a) + int(st.session_state.chk2b)}', org=(277,483+30),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{int(st.session_state.chk2c) + int(st.session_state.chk2d)}', org=(372,601+31),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{int(st.session_state.chk2e) + int(st.session_state.chk2f)}', org=(450,644+30),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        cv.putText(img=warped_img, text=f'{int(st.session_state.chk2g) + int(st.session_state.chk2h)}', org=(484,786+30),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)

        
        final_rgb = cv.cvtColor(warped_img, cv.COLOR_BGR2RGB)

        st.image(final_rgb)
        
        final_pil = Image.fromarray(final_rgb)
        buffer = io.BytesIO()
        final_pil.save(buffer, format = "PNG")
        st.download_button(label=f"Download {filename}", data=buffer, file_name=filename, mime="image/jpeg")
        









    
    
