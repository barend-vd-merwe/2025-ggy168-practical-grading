import streamlit as st
import cv2 as cv
import pandas as pd
import numpy as np
from PIL import Image
import io
import re

st.title("Normal Exam - Mapwork")
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

    st.header("Table")
    st.write('''
        **NOTE** The rows may not be in the same sequence as the memo.
        For each row, check (Using the provided calculator) whether the calculation
        is approximately correct.
        Remember that it may not be entirely correct because of rounding, as
        long as it is correct to the first decimal.
    ''')
    table_img = warped_img[146:146+110, 370:370+342]
    

    st.write('''
        Enter the student's answers in the cells below:
    ''')
    calculator = pd.DataFrame(
        {
            "Length (km)": [0.0,0.0,0.0,0.0],
            "Width (km)": [0.0,0.0,0.0,0.0]
        },
        index = ["Strip 1", "Strip 2", "Strip 3", "Strip 4"]
        )  
        
    student_area = st.data_editor(calculator)
    student_area["Area (km²)"] = student_area["Length (km)"] * student_area["Width (km)"]

    st.image(table_img)

    st.write('''
        Their area calculations should be "approximately" the same as the values indicated below:
    ''')
    st.dataframe(student_area)

    
    
    
    row_1 = st.number_input("Row 1 correct", min_value = 0, max_value = 1)
    row_2 = st.number_input("Row 2 correct", min_value = 0, max_value = 1)
    row_3 = st.number_input("Row 3 correct", min_value = 0, max_value = 1)
    row_4 = st.number_input("Row 4 correct", min_value = 0, max_value = 1)
    


    st.header("Map")
    st.write('''
        Check whther their working is shown on the map. Also, as long as you
        can see four strips that roughly approximate that shown on the memo
        you can award marks.
    ''')
    map_img = warped_img[375:375+521, 70:70+574]
    st.image(map_img)

    work_shown = st.number_input("Work Shown", min_value = 0, max_value = 1)
    strip_1 = st.number_input("Strip 1", min_value = 0, max_value =2)
    strip_2 = st.number_input("Strip 2", min_value = 0, max_value =2)
    strip_3 = st.number_input("Strip 3", min_value = 0, max_value =2)
    strip_4 = st.number_input("Strip 4", min_value = 0, max_value =2)

    
    st.header("Answer")
    st.write('''
        The answer should lie between 3749.51 km² and 5624.27 km² (1 marks). The units should
        be shown as well (km²) (1 marks).
    ''')
    answer_img = warped_img[215:215+55, 379:379+333]
    st.image(answer_img)
    answer_val = st.number_input("Answer Value Correct", min_value = 0, max_value = 1)
    answer_units = st.number_input("Answer Units Correct", min_value = 0, max_value = 1)
    
    

    
    

    if st.button("Grade"):
        
        answer_grade = answer_val + answer_units
        table_grade = row_1 + row_2 + row_3 + row_4
        map_grade = work_shown + strip_1 + strip_2 + strip_3 + strip_4
        final_grade = answer_grade + table_grade + map_grade
        cv.putText(img=warped_img, text=f'{final_grade}', org=(260,37+51),
                   fontFace=cv.FONT_HERSHEY_SIMPLEX,
                   fontScale=1, color=(0, 0, 255), thickness=2)
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
        









    
    
