import streamlit as st
import os
import json
import pandas as pd
from pathlib import Path
from streamlit_extras.switch_page_button import switch_page


with open("config.json") as user_file:
    file_contents = user_file.read()
    config = json.loads(file_contents)

def click_button():
    st.session_state.clicked = True

if 'filename' not in st.session_state:
    st.session_state.filename = config['context_path']

if 'default' not in st.session_state:
    st.session_state.default = 0

if 'file' not in st.session_state:
    st.session_state.file = {}


def main():
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False
    fu = st.empty()
    with st.container():
        fu.button("Click to edit config file", on_click=click_button, key="ask_button")
    c1,c2 = st.columns([0.7, 0.3], gap="small")
    st.session_state.file = json.loads(Path(st.session_state.filename).read_text())
    column_config = {
        'is_default':None
    }
    df = pd.DataFrame(st.session_state.file)
    with c1:
        if st.session_state.clicked:
            fu.empty()
            edited_df = st.data_editor(df.transpose(), column_config= column_config)
            print(type(edited_df))
            print(edited_df.transpose().to_json())
    with c2:
        checked_row = 0
        if st.session_state.clicked:
            for i,column in enumerate(st.session_state.file):
                if st.session_state.file[column]['is_default'] == True:
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
            vals = [str(i) for i in range(1, len(st.session_state.file) + 1)]
            genre = st.radio("Is Default",vals, index=checked_row,)
            st.session_state.default = int(genre) -1
            print("\n\n\n", genre, "thihhhhhhhhhhh", len(st.session_state.file))

    if st.button("click here to submit"):
        if st.session_state.clicked:
            for i, row in enumerate(st.session_state.file):
                if i == st.session_state.default:
                    st.session_state.file[row]['is_default'] = True
                else:
                    st.session_state.file[row]['is_default'] = False
            print(st.session_state.file)
            file_path =Path(st.session_state.filename)
            file_path.write_text(json.dumps(st.session_state.file, indent=4))
            switch_page("samyeee")

main()