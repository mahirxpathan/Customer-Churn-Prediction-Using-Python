
import streamlit as st
import pandas as pd

st.set_page_config(page_title='Test', page_icon='📊', layout='wide')

st.markdown('<div class=main-header>Customer Churn Prediction Dashboard</div>', unsafe_allow_html=True)
st.write('App loaded successfully')

page = st.sidebar.selectbox('Choose a page:', ['Home', 'Test'])

if page == 'Home':
    st.subheader('Welcome')
    st.write('Home page content')
elif page == 'Test':
    st.subheader('Test Page')
    st.write('Test page content')

