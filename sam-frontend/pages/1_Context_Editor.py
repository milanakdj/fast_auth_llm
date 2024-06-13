import streamlit as st
from streamlit import session_state as ss 
import os
import json
import pandas as pd
from pathlib import Path
from streamlit_extras.switch_page_button import switch_page


with open("config.json") as user_file:
    file_contents = user_file.read()
    config = json.loads(file_contents)

def click_button():
    ss.clicked = True

if 'filename' not in st.session_state:
    ss.filename = config['context_path']

if 'default' not in st.session_state:
    ss.default = 0

if 'file' not in st.session_state:
    ss.file = json.loads(Path(st.session_state.filename).read_text())

if 'df' not in ss:
    ss.df = pd.DataFrame(st.session_state.file).transpose()

if 'edf' not in ss:
    ss.edf = ss.df.copy()  

def edit_df():
    ss.df = ss.edf.copy()
    st.write(ss.df)

def main():

    if 'clicked' not in st.session_state:
        st.session_state.clicked = False

    fu = st.empty()

    with st.container():
        fu.button("Click to edit config file", on_click=click_button, key="ask_button")

    c1,c2 = st.columns([0.7, 0.3], gap="small")
    
    column_config = {
        'is_default':None
    }

    with c1:
        if st.session_state.clicked:
            fu.empty()
            
            ss.edf = st.data_editor(ss.df, column_config=column_config, use_container_width=True, num_rows='dynamic', key='edited_df')

    with c2:
        checked_row = 0
        if st.session_state.clicked:
            for i, column in enumerate(st.session_state.file):
                if st.session_state.file[column]['is_default']:
                    checked_row = i
                    break
            st.markdown(
            """
            <style>
            .stRadio >div>label{
                height: 2em;
            }
            
            .stRadio >div>label>div>div>p{
                color:transparent;
            }
            </style>
            """, unsafe_allow_html=True)
            vals = [str(i) for i in range(1, len(ss.edf)+1)]
            if checked_row not in range(0, len(ss.edf)):
                checked_row = 0
            genre = st.radio("Is Default", vals, index=checked_row)
            st.session_state.default = int(genre) - 1

    if st.session_state.clicked:
        if st.button("click here to submit"):
            ss.df = ss.edf.copy()
            st.session_state.file = json.loads(ss.df.transpose().to_json())
            for i, row in enumerate(st.session_state.file):
                st.session_state.file[row]['is_default'] = (i == st.session_state.default)
            file_path = Path(st.session_state.filename)
            file_path.write_text(json.dumps(st.session_state.file, indent=4))
            switch_page("samyeee")

main()
