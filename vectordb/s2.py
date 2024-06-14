import streamlit as st

# Include the CSS for styling
st.markdown("""
    <style>
        .dropdown-container-model {
            background-color: red; /* example styling */
            padding: 10px;
            margin: 10px;
        }
        .dropdown-container-sub-model {
            background-color: green; /* example styling */
            padding: 10px;
            margin: 10px;
        }
        .dropdown-container-sub-model.stSelectbox >div  {
            margin-top: 20px; /* Adjust margin as needed */
            width:200px
        }
    </style>
""", unsafe_allow_html=True)

# Use Streamlit components within the container
with st.container():
    st.markdown('<div class="dropdown-container-model">', unsafe_allow_html=True)
    st.markdown('<div class="dropdown-container-sub-model">', unsafe_allow_html=True)
    # Place the selectbox within the green div
    modeloption = st.selectbox("Choose LLM Model", ['chatGPT', 'llama3'], index=0)
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
