import streamlit as st
import os
import json
import pandas as pd
from pathlib import Path


with open("config.json") as user_file:
    file_contents = user_file.read()
    config = json.loads(file_contents)

def click_button():
    st.session_state.clicked = True

if 'filename' not in st.session_state:
    st.session_state.filename = config['context_path']


def main():
    if 'clicked' not in st.session_state:
        st.session_state.clicked = False
    fu = st.empty()
    with st.container():
        fu.button("Click to edit config file", on_click=click_button, key="ask_button")

    if st.session_state.clicked:
        fu.empty()
        context_value = json.loads(Path(st.session_state.filename).read_text())
        column_config = {
            'is_default':st.column_config.CheckboxColumn('is_default', default=False, required=True)
        }
        df = pd.DataFrame(context_value)
        edited_df = st.data_editor(df.transpose(), column_config= column_config)
        print(type(edited_df))
        print(edited_df.transpose().to_json())
    
main()