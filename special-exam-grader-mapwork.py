import streamlit as st
import cv2 as cv
import pandas as pd
import numpy as np
from PIL import Image
import io
import re
from streamlit_image_coordinates import streamlit_image_coordinates

st.title("Special exam - Mapwork")
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

    st.header("Question 1a")
    q1a_img = warped_img[331:331+414, 18:18+667]
    
    #st.image(q1a_img)
    coords = streamlit_image_coordinates(q1a_img, key = "image_clicks")
    if coords:
        st.write(f"x: {coords['x']}, y:{coords['y']}")

        dist_calc = pd.DataFrame({
            "Start": [0],
            "End": [0]
            })
        distance_calculator = st.data_editor(dist_calc)
        distance_calculator["Dist"] = distance_calculator["End"] - distance_calculator["Start"]
        st.dataframe(distance_calculator)
    
    q1_horziontal = st.number_input("Horizontal extent (approx: 440)", min_value = 0, max_value = 1)
    q1_vertical = st.number_input("Vertical extent (approx: 61)", min_value = 0, max_value = 1)
    q1_shape = st.number_input("Shape", min_value = 0, max_value = 1)

    st.header("Question 1b")
    st.write("Answer: 1:2857")
    q1b_img = warped_img[734:734+74, 18:18+200]
    st.image(q1b_img)
    q1b = st.number_input("Vertical exaggeration", min_value = 0, max_value = 3)

    st.header("Question 1c")
    st.write("Answer: 29.24°")
    q1c_img = warped_img[734:734+74, 208:208+202]
    st.image(q1c_img)
    q1c = st.number_input("Gradient", min_value = 0, max_value = 3)

    st.header("Question 1d")
    st.write("Answer: 52°")
    q1d_img = warped_img[734:734+74, 402:402+194]
    st.image(q1d_img)
    q1d = st.number_input("Aspect", min_value = 0, max_value = 3)
    

    
    

    if st.button("Grade"):
        
        final_grade = q1_horziontal + q1_vertical + q1_shape + q1b + q1c + q1d
        cv.putText(img=warped_img, text=f'{final_grade}', org=(260,37+51),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
        # grades
        cv.rectangle(img=warped_img, pt1=(331,19), pt2=(331+66,19+322),
                     color=(255,255,255), thickness=-1)
        cv.putText(img=warped_img, text=f'Q6a : {q1_horziontal + q1_vertical + q1_shape}', org=(331,20+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q6b : {q1b}', org=(331,40+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q6c  : {q1c}', org=(331,60+20),
                   fontFace=cv.FONT_HERSHEY_PLAIN,
                   fontScale=1, color=(0, 0, 255), thickness=1)
        cv.putText(img=warped_img, text=f'Q6d : {q1d}', org=(331,80+20),
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
        









    
    
