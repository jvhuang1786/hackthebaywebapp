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
    'Total Nitrogen Model', 'About The Team']
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
            Jen Section approach to the problem and data cleaning as well with features
            chosen and a little bit of data wrangling
            """

            st.markdown(title_write,unsafe_allow_html=True)

        if st.sidebar.checkbox('Total Nitrogen Model Summary'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:7px">
            <h4 style="color:#212F3D;text-align:center;">Total Nitrogen Model Summary</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

            title_write = """
            TLDR of model and plan
            Why was total nitrogen chosen
            what features were chosen to predict total Nitrogen
            some of the graphs and predictions
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
        Bryans land cover, narr, epa nitro oxide and merge with total nitrogen
        """
        st.markdown(explorationwrite_up, unsafe_allow_html=True)

        image = Image.open('images/oyster2.png')
        st.image(image, width = 800)

        ##########
        #Load DataFrames here for charts


        ###Weather


        ###Land LandCover

        ###Water Quality
        df_water = pd.read_csv('data/dfnitro.csv', index_col = 0)

        ##########

        if st.sidebar.checkbox('Data Prep and Wrangling'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">Data Prep and Wrangling</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

            explorationwrite_up = """
            Landcover clean up method and collection just a few sentences
            and images to add in
            NARR data wrangle
            EPA data wrangle nitrogen oxide
            Merging with water_final to predict nitrogen
            """
            st.markdown(explorationwrite_up, unsafe_allow_html=True)

        if st.sidebar.checkbox('Data Visualizations'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">Data Visualizatons</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

            vizwrite_up = """
            exploration of the chemicals in relation to nitrogen or
            other interesting finds from Tim.
            exploring the relation of other chemicals with nitrogen 

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


        modelwrite_up = """
        There were two models that were used.  xgBoost and Catboost.  Randomforest was also experimented with
        however the results when adding nitrogen oxide was not as good as the other two algorithms.

        RandomForest Results:

        ```python
        r2                       0.6935003647333058
        explained_variance_score 0.6936163490312466
        RMSE                     1.2103501514032144
        ```

        xgBoost Results:
        ```python
        r2                       0.8399586734809285
        explained_variance_score 0.8399821265302834
        RMSE                     0.8746053286890739
        ```
        CatBoost
        ```python
        Berenices model results here
        ```
        """
        st.markdown(modelwrite_up, unsafe_allow_html=True)

        ##########
        #Load DataFrames here for charts
        df_main = pd.read_csv('data/model_data_enc.csv', index_col = 0)
        df_main = df_main.set_index('new_date')

        ###CatBoost

        ###xgBoost
        feature_xgb = pd.read_csv('data/allfeatimp.csv', index_col = 0)
        predict_xgb = pd.read_csv('data/allfeatpredictions_df.csv',index_col = 0)
        result_xgb = pd.read_csv('data/allfeatresults.csv',index_col = 0)

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


        if st.sidebar.checkbox('xgBoost Model'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">xgBoost Model</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

            modelwrite_up = """
            The final features used in the final xgb model were:

               * 'areaacres', 'lc_21', 'lc_31', 'lc_41', 'lc_42', 'lc_43', 'lc_52',
               * lc_71', 'lc_81', 'lc_82', 'lc_90', 'lc_95', 'year', 'week',
               * 'airtemp_narr', 'precip3_narr', 'humidity_narr', 'cl_cover_narr',
               * 'sfc_runoff', 'windspeed_narr', 'wdirection_narr', 'precip48_narr',
               * 'of_dist', 'total Nitrogen Oxide in year'

            A total of 24 features used.

            To arrive to these features and how the features were selected and engineered.

                * A KNN imputer worked best with two nearest neighbors.
                * A Robust Scaler had a slight advantage overall when doing model experimentation with
                xgBoot.
                * Total Nitrogen above 50 was removed as there would most likely be serious Algae Bloom
                and deemed unsafe to collect water.


            Below is the code:

            ```python

            #Preprocessing
            knnimputer = KNNImputer(n_neighbors=2)
            scaler_std = StandardScaler()
            scaler_rob = RobustScaler()

            #StandardScaler worked best before but try RobustScaler
            knn_impute_scale = make_pipeline(knnimputer, scaler_std)
            knn_impute_scale1 = make_pipeline(knnimputer, scaler_rob)

            #robust scaler column transformer
            ct = make_column_transformer(
                (knn_impute_scale1, ['areaacres', 'lc_21', 'lc_31', 'lc_41', 'lc_42', 'lc_43', 'lc_52',
                   'lc_71', 'lc_81', 'lc_82', 'lc_90', 'lc_95', 'year', 'week',
                   'airtemp_narr', 'precip3_narr', 'humidity_narr', 'cl_cover_narr',
                   'sfc_runoff', 'windspeed_narr', 'wdirection_narr', 'precip48_narr',
                   'of_dist', 'total Nitrogen Oxide in year']),
                remainder='passthrough')
            ```

            Correlated groups were also removed from the original set of features.  There were originally 31
            total from the merged dataset and this was reduced down to 24.

            ```python
            #Group correlated features together
            grouped_features_ls = []
            correlated_groups = []

            for feature in corrmat.feature1.unique():
                if feature not in grouped_features_ls:

                    #find all features correlated to a single feature
                    correlated_block = corrmat[corrmat.feature1 == feature]
                    grouped_feature_ls = grouped_features_ls + list(correlated_block.feature2.unique()) + [feature]

                    #append the block of features to the list
                    correlated_groups.append(correlated_block)

            print('found {} correlated groups'.format(len(correlated_groups)))
            print('out of {} total features'.format(X_train.shape[1]))



                          feature1 feature2   corr
            0  lc_24    lc_23   0.9727
            6  lc_24    lc_22   0.9071


               feature1 feature2   corr
            1   lc_23    lc_24   0.9727
            2   lc_23    lc_22   0.9344
            15  lc_23    lc_21   0.7200


              feature1 feature2   corr
            3  lc_22    lc_23   0.9344
            7  lc_22    lc_24   0.9071
            9  lc_22    lc_21   0.8304


              feature1  feature2   corr
            4  of_dist  latitude 0.9268


               feature1 feature2   corr
            5  latitude  of_dist 0.9268


               feature1 feature2   corr
            8   lc_21    lc_22   0.8304
            14  lc_21    lc_23   0.7200


                     feature1       feature2   corr
            10  precip48_narr  precip24_narr 0.8140


                     feature1       feature2   corr
            11  precip24_narr  precip48_narr 0.8140


               feature1 feature2   corr
            12  lc_82    za_mean 0.7942


               feature1 feature2   corr
            13  za_mean  lc_82   0.7942


                 feature1 feature2   corr
            16  longitude  lc_41   0.7025


               feature1   feature2   corr
            17  lc_41    longitude 0.7025
            ```

            Feature importance was then used to remove features that were higly correlated
            and had lower feature importance.

            ```python
            #We can screen using a randomforest second feature group
            features = ['lc_22'] + ['lc_23'] + ['lc_24']  + ['lc_21']
            xgb.fit(X_train[features].fillna(0), y_train);

            #get feature importance attributed by random forest model
            importance = pd.concat([pd.Series(features),
                                   pd.Series(xgb.feature_importances_)], axis = 1)

            importance.columns = ['feature', 'importance']
            importance.sort_values(by = 'importance', ascending = False)


                feature	importance
            3	lc_21	0.3180
            2	lc_24	0.2479
            0	lc_22	0.2239
            1	lc_23	0.2101
            ```
            RandomizedSearchCV was used to do hypertuning to the xgBoost Algo.

            ```python
            #Set Parameters
            params = {}
            params['xgbregressor__max_depth'] = np.arange(0, 1100,10)
            params['xgbregressor__n_estimators'] = np.arange(0, 1000,10)
            params['xgbregressor__max_samples'] = np.arange(0, 1100,10)
            params['xgbregressor__min_child_weight'] = np.arange(1,11,1)
            params['xgbregressor__importance_type'] = ['weight', 'gain', 'cover', 'total_gain', 'total_cover']

            #Grid for hypertuning
            gridRF = RandomizedSearchCV(pipeline, params, cv = 5, n_jobs = -1, random_state = 0, n_iter = 30, scoring = 'r2')

            #Robust Scaler
            gridRF.fit(X_train,y_train);

            #Robust Scaler prediction
            y_pred_final = gridRF.predict(X_test)

            #Robust Metrics
            print(r2_score(y_test, y_pred_final))
            print(explained_variance_score(y_test, y_pred_final))
            print(np.sqrt(mean_squared_error(y_test, y_pred_final)))
            ```

            This gave a r2, explained variance and rmse of:

            ```python

            r2                          0.8399586734809285
            explained_variance_score    0.8399821265302834
            RMSE                        0.8746053286890739
            ```
            Model Visuals below along with SHAP (SHapley Additive exPlanations) is a game theoretic
            approach to explain the output of any machine learning model.
            """
            st.markdown(modelwrite_up, unsafe_allow_html=True)
            st.write('xgBoost feature importance and hypertuning results.')

            st.write(predict_xgb.head(10))
            st.write(result_xgb.head())
            st.write(feature_xgb.head())
            fig = px.bar(feature_xgb, feature_xgb['features'], feature_xgb['importance'])
            st.plotly_chart(fig)
            st.write('Shap: The color represents the feature value (red high, blue low). This reveals higher of_distance higher TN')
            image = Image.open('images/xgb1.png')
            st.image(image, width = 500)

            image = Image.open('images/xgb2.png')
            st.image(image, width = 500)


        if st.sidebar.checkbox('Model Choice and Reasoning'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">Model Choice and Reasoning</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

            modelwrite_up = """
            Which model did we choose xgboost or catboost I have a feeling catboost will be better.
            """
            st.markdown(modelwrite_up, unsafe_allow_html=True)


            image = Image.open('images/shap_value.png')
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
