import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from PIL import Image

#set title

image = Image.open('images/oyster.png')
st.image(image, width = 800)

def main():
    activities = ['Intro to the Chesapeake Bay Challenge', 'Data Preparation',
    'Data Visualization', 'Total Nitrogen Model', 'About The Team']
    option = st.sidebar.selectbox('Selection Option:', activities)

#Intro
    if option == 'Intro to the Chesapeake Bay Challenge':
        st.title('Intro to the Chesapeake Bay Challenge')
        title_page = """
        <div style="background-color:#33A2FF;padding:1px">
        <h3 style="color:#313F3D;text-align:center;">Intro to the Chesapeake Bay Challenge</h3>
        </div>
        """
        st.markdown(title_page,unsafe_allow_html=True)

        ##########
        #Load DataFrames here for charts

        ##########

        if st.sidebar.checkbox('Pollution in the Chesapeake Bay'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">Pollution in the Chesapeake Bay</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

        if st.sidebar.checkbox('Assesment and Plan of Action'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">Assesment and Plan of Action</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

        title_write = """
        put writing here Jen
        """

        st.markdown(title_write,unsafe_allow_html=True)


#Data Preparation
    elif option == 'Data Preparation':
        st.title('Data Preparation')
        html_temp = """
        <div style="background-color:#33A2FF;padding:1px">
        <h3 style="color:#212F3D;text-align:center;">Data Preparation</h3>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)

        ##########
        #Load DataFrames here for charts

        ##########

        if st.sidebar.checkbox('Land Cover Data Collection and Prep'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">Land Cover Data Collection andd Prep</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

        if st.sidebar.checkbox('Weather and Air condition Data prep'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">Weather and Air condition Data prep</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

        if st.sidebar.checkbox('Water Quality Data prep'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">Water Quality Data prep</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

        explorationwrite_up = """
        Jen write here
        """
        st.markdown(explorationwrite_up, unsafe_allow_html=True)

        image = Image.open('images/oyster2.png')
        st.image(image, width = 800)


#Data Visualization
    elif option == 'Data Visualization':
        st.title('Data Visualization')
        html_temp = """
        <div style="background-color:#33A2FF;padding:1px">
        <h3 style="color:#212F3D;text-align:center;">Data Visualization</h3>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)

        ##########
        #Load DataFrames here for charts

        ##########

        if st.sidebar.checkbox('Water Quality Compounds Visualization'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">Water Quality Compounds Visualization</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

        if st.sidebar.checkbox('Land Cover, HUC12_ relation to Nitrogen'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">Land Cover, HUC12_ relation to Nitrogen</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)


        vizwrite_up = """
        Jen write here
        ```python
        This is how I write code here.
        ```
        """
        st.markdown(vizwrite_up, unsafe_allow_html=True)

        image = Image.open('images/oyster2.png')
        st.image(image, width = 800)

#Nitrogen Modeling
    elif option == 'Total Nitrogen Model':
        st.title('Total Nitrogen Model')
        html_temp = """
        <div style="background-color:#33A2FF;padding:1px">
        <h3 style="color:#212F3D;text-align:center;">Total Nitrogen Model</h3>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)

        ##########
        #Load DataFrames here for charts

        ##########

        if st.sidebar.checkbox('CatBoost Model'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">CatBoost Model</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

        if st.sidebar.checkbox('RandomForest Model'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">RandomForest Model</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

        if st.sidebar.checkbox('xgBoost Model'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">xgBoost Model</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

        if st.sidebar.checkbox('Final Model'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">Final Model</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)


        modelwrite_up = """
        Jen write here
        """
        st.markdown(modelwrite_up, unsafe_allow_html=True)

        image = Image.open('images/oyster2.png')
        st.image(image, width = 800)

    elif option == 'About The Team':
        st.title('Data Preparation')
        html_temp = """
        <div style="background-color:#33A2FF;padding:1px">
        <h3 style="color:#212F3D;text-align:center;">Data Preparation</h3>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)

        about_write = """
        One sentence about each team member
        Link to github repo
        """

        st.markdown(about_write,unsafe_allow_html=True)




if __name__ == '__main__':
    main()
