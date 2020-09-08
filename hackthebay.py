import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from PIL import Image

#set title

image = Image.open('images/oyster.png')
st.image(image, width = 800)

def main():
    activities = ['Intro: About The Challenge', 'Data Preparation',
    'Data Visualization', 'Total Nitrogen Model', 'About The Team']
    option = st.sidebar.selectbox('Selection Option:', activities)

#Intro
    if option == 'Intro: About The Challenge':
        st.title('Introduction: About The Challenge')
        title_page = """
        <div style="background-color:#33A2FF;padding:2px">
        <h3 style="color:#313F3D;text-align:center;">Hack The Bay</h3>
        </div>
        """
        st.markdown(title_page,unsafe_allow_html=True)

        title_write = """

        As part of the eight-week Hack The Bay challenge to model water quality,
        we explored how various land use, weather, and other factors are tied to indicators
        of pollution using data from the Chesapeake Monitoring Cooperative and Chesapeake Bay Program,
        as well as supplementary geospatial datasets. The resulting model provides
        visibility so that useful predictors might inform timely decisions and inspire action
        to improve the health of the Chesapeake Bay and its stakeholders.

        """

        st.markdown(title_write,unsafe_allow_html=True)

        ##########
        #Load DataFrames here for charts

        ##########

        if st.sidebar.checkbox('Pollution in the Chesapeake Bay'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:7px">
            <h4 style="color:#212F3D;text-align:center;">Pollution in the Chesapeake Bay</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

            title_write = """
            Intro of Chesapeake bay problems
            """

            st.markdown(title_write,unsafe_allow_html=True)

        if st.sidebar.checkbox('Assessment and Plan of Action'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:7px">
            <h4 style="color:#212F3D;text-align:center;">Assessment and Plan of Action</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

            title_write = """
            TLDR of model and plan
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

        explorationwrite_up = """
        Jen write here for exploration intro
        """
        st.markdown(explorationwrite_up, unsafe_allow_html=True)

        image = Image.open('images/oyster2.png')
        st.image(image, width = 800)

        ##########
        #Load DataFrames here for charts


        ###Weather


        ###Land LandCover

        ###Water Quality

        ##########

        if st.sidebar.checkbox('Land Cover Data Collection and Prep'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">Land Cover Data Collection andd Prep</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

            explorationwrite_up = """
            Landcover clean up method and collection
            """
            st.markdown(explorationwrite_up, unsafe_allow_html=True)

        if st.sidebar.checkbox('Weather and Air condition Data prep'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">Weather and Air condition Data prep</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

            explorationwrite_up = """
            Air and Weather Quality clean up and collection
            """
            st.markdown(explorationwrite_up, unsafe_allow_html=True)

        if st.sidebar.checkbox('Water Quality Data prep'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">Water Quality Data prep</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

            explorationwrite_up = """
            Given dataset and merging of dataset
            """
            st.markdown(explorationwrite_up, unsafe_allow_html=True)




#Data Visualization
    elif option == 'Data Visualization':
        st.title('Data Visualization')
        html_temp = """
        <div style="background-color:#33A2FF;padding:1px">
        <h3 style="color:#212F3D;text-align:center;">Data Visualization</h3>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)

        vizwrite_up = """
        Viz intro here

        ```python
        This is how I write code here.
        ```
        """
        st.markdown(vizwrite_up, unsafe_allow_html=True)

        image = Image.open('images/oyster2.png')
        st.image(image, width = 800)

        ##########
        #Load DataFrames here for charts

        ###Water Quality
        df_water = pd.read_csv('data/dfnitro.csv', index_col = 0)


        ###LandCover

        ##########

        if st.sidebar.checkbox('Water Quality Compounds Visualization'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">Water Quality Compounds Visualization</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

            vizwrite_up = """
            Water Quality with Chemical Compound write up


            """
            st.markdown(vizwrite_up, unsafe_allow_html=True)

            # fig = px.scatter_matrix(predict_rf,
            #     dimensions=["Actual", "gridRF", "gridRF2"])
            #
            # st.plotly_chart(fig)

        if st.sidebar.checkbox('Land Cover, HUC12_ relation to Nitrogen'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">Land Cover, HUC12_ relation to Nitrogen</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

            vizwrite_up = """
            Land cover visualization here

            """
            st.markdown(vizwrite_up, unsafe_allow_html=True)


#Nitrogen Modeling
    elif option == 'Total Nitrogen Model':
        st.title('Total Nitrogen Model')
        html_temp = """
        <div style="background-color:#33A2FF;padding:1px">
        <h3 style="color:#212F3D;text-align:center;">Total Nitrogen Model</h3>
        </div>
        """
        st.markdown(html_temp,unsafe_allow_html=True)


        modelwrite_up = """
        Jen write intro here
        """
        st.markdown(modelwrite_up, unsafe_allow_html=True)

        image = Image.open('images/oyster2.png')
        st.image(image, width = 800)

        ##########
        #Load DataFrames here for charts
        df_main = pd.read_csv('data/model_data_enc.csv', index_col = 0)
        df_main = df_main.set_index('new_date')

        ###CatBoost

        ###Randomforest
        mi_df = pd.read_csv('data/mi_rf.csv', index_col = 0)
        mi_df = mi_df.reset_index().sort_values(ascending = False, by = '0')
        mi_df = mi_df.rename(columns = {'index':'features', '0': 'mi Stat'})
        feature_rf_rob = pd.read_csv('data/featureimprf_rob.csv', index_col = 0)
        predict_rf = pd.read_csv('data/rfpredictions_df.csv', index_col = 0)
        result_rf_rob = pd.read_csv('data/rf_result_robust.csv', index_col = 0)
        feature_rf_std = pd.read_csv('data/featureimprf_scale.csv', index_col = 0)
        result_rf_std = pd.read_csv('data/rf_result_standard.csv',index_col = 0)

        ###xgBoost
        feature_xgb = pd.read_csv('data/featuresimpxgb.csv', index_col = 0)
        predict_xgb = pd.read_csv('data/predictxgbdf.csv',index_col = 0)
        result_xgb = pd.read_csv('data/xgb_results.csv',index_col = 0)

        ###final

        ##########

        if st.sidebar.checkbox('CatBoost Model'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">CatBoost Model</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

            modelwrite_up = """
            Catboost Write up
            """
            st.markdown(modelwrite_up, unsafe_allow_html=True)

        if st.sidebar.checkbox('RandomForest Model'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">RandomForest Model</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

            modelwrite_up = """
            RandomForest Write up
            """
            st.markdown(modelwrite_up, unsafe_allow_html=True)

            st.write(mi_df)
            st.write(feature_rf_rob)
            st.write(result_rf_rob)
            st.write(predict_rf)

            fig = px.bar(mi_df, mi_df['features'], mi_df['mi Stat'])
            st.plotly_chart(fig)

            fig = px.bar(feature_rf_rob, feature_rf_rob['features'], feature_rf_rob['importance'])
            st.plotly_chart(fig)





        if st.sidebar.checkbox('xgBoost Model'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">xgBoost Model</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

            modelwrite_up = """
            xgBoost Write up
            """
            st.markdown(modelwrite_up, unsafe_allow_html=True)

            st.write(feature_xgb)
            st.write(predict_xgb)
            st.write(result_xgb)

            fig = px.bar(feature_xgb, feature_xgb['features'], feature_xgb['importance'])
            st.plotly_chart(fig)

        if st.sidebar.checkbox('Final Model'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">Final Model</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

            modelwrite_up = """
            Final Model Write up
            """
            st.markdown(modelwrite_up, unsafe_allow_html=True)




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
