import streamlit as st
import numpy as np
import pandas as pd
import datetime
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from PIL import Image

#set title

image = Image.open('images/virginia.jpg')
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
            Intro of Chesapeake bay problems
            Jen Section approach to the problem and data cleaning as well with features
            chosen and a little bit of data wrangling

            [Fill in once most other stuff is settled.]
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
        ## CMC/CBP Data Preparation

        We filtered the provided dataset for total nitrogen. Then, we
        aggregated data by date and longitude + latitude, and took a mean of the
        measure value, to obtain one reading for each day and each location
        that samples were taken.        """
        st.markdown(explorationwrite_up, unsafe_allow_html=True)


        ##########
        #Load DataFrames here for charts


        ###Weather


        ###Land LandCover

        ###Water Quality
        # df = pd.read_csv('data/dfnitro.csv', index_col = 0)

        ##########

        if st.sidebar.checkbox('Data Prep and Wrangling'):
            html_temp = """
            <div style="background-color:#33A2FF;padding:1px">
            <h4 style="color:#212F3D;text-align:center;">Data Prep and Wrangling</h4>
            </div>
            """
            st.markdown(html_temp,unsafe_allow_html=True)

            explorationwrite_up = """
            ## Preparing: Land Cover Data

            Using land cover data from the Multi-Resolution Land Characteristics
            Consortium (MRLC) National Land Cover Viewer and the watershed HUC12
            boundaries shapefile (1), we obtained the following features:

            * Land cover % usage for each code in the NLCD legend
            * The mean of the pixel values for each segment
            * The area of each segment in acres

            ## Preparing: Weather Data

            We obtained weather data from North American Regional Reanalysis.

            The features we used were:

            * Air Temperature
            * Humidity
            * Cloud Cover
            * Surface Air Temperature
            * Surface Runoff
            * Wind Components
            * Precipitation

            We then extracted the above data for dates and location (longitude +
            latitude) relevant with each pollutant observation. (2)

            ## Creating: Mean Encoded HUC12

            This feature is a mean of total nitrogen for each HUC code with some
            regularization. We used it instead of label encoding the HUC codes because
            there were more than 300 HUC12 locations, and instead of One Hot Encoding
            because OHE would introduce a large amount of new features to the data.
            Mean encoding introduces the correlation between the categories and the
            target variable in only one additional feature. (3)

            ## Creating: Distance from Outflow

            This is the distance from each sample location (latitude + longitude)
            from the outflow of the Bay (36.995833, -75.959444). We measured distance
            as the geodesic distance between two coordinates in miles. (4)

            ## Creating: NO2 from Point Sources

            This feature represents all NO2 pollutants from correlated point source
            locations in the airshed by year for each HUC12 segment.

            Combining air emissions data for stationary sources from four EPA air
            programs, we transformed and aggregated NO2 emissions by HUC12 code
            and year, and merged it with NO2 emissions by Point Source and year.
            The data was correlated and filtered to only correlations >=.6 and <1.
            All correlated point sources for each HUC12 segment were aggregated by
            year. For those missing values, we used a mean of the HUC12 correlated
            point sources. (5)

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
            We explored the relation of other chemicals with nitrogen and other finds
            """
            st.markdown(vizwrite_up, unsafe_allow_html=True)

            # fig = px.density_mapbox(df, lat='Latitude', lon='Longitude',
            #             z='WATER TEMPERATURE DEG deg c',
            #             radius=5, center=dict(lat=39.5, lon=-76),zoom=5,
            #             mapbox_style='stamen-terrain')
            # st.plotly_chart(fig)

            image = Image.open('images/tempmap.png')
            st.image(image, width = 800)

            st.write("""
            The scatter plots are plotted vs Latitude to show the general vertical
            distribution. Figures are split as reporting levels for each item are
            different. Since we focused on Nitrogen and Phosphorus I wanted to display
            both with comparison to Latitude to highlight the higher values are present
            at different latitudes.

            Dissolved Oxygen and Active Chlorophyll were plotted to examine the
            relation however the lack of Active Chlorophyll makes it difficult to
            draw conclusions from the plot. There is one correlated peak between
            the 2 values (negative on DO and positive on AC) around the 39 latitude.
            Additionally it can be noted that low DO is more of a concern below 41
            degrees.
            """
            )


            image = Image.open('images/DO_Active_Chlorophyll.png')
            st.image(image, width = 500)

            st.write("""
            Returning to the Nitrogen and Phosphorus it could be stated locations
            with higher Phosphorus have lower DO values. Nitrogen does not have the
            same effect.
            """
            )
            image = Image.open('images/Nitrogen_Phosphorus.png')
            st.image(image, width = 500)

            st.write("""
            This histogram is an attempt to show the different effects of water
            temperature and salinity on pH. There isn't a large spike in temperature
            from the data and salinity is dominated by freshwater sources. pH remains
            within acceptable bounds looking at the distribution ~5 - 11.)
            """
            )
            image = Image.open('images/pH_Temp_Salinity.png')
            st.image(image, width = 500)

            st.write("""
            Suspended Solids and Turbidity appear well correlated when plotted vs.
            latitude. Figure was included as a good correlation between two values.
            The figure suffers from the lack of data in Turbidity from about 40 - 42
            degrees.
            """
            )
            image = Image.open('images/Suspended_Solids_Turbidity.png')
            st.image(image, width = 500)

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
        CatBoost Results
        ```python
        r2                       0.8401995362308845
        explaned_variance_score  0.8402718767819098
        RMSE                     0.9249663789834368
        ```
        """
        st.markdown(modelwrite_up, unsafe_allow_html=True)

        ##########
        #Load DataFrames here for charts
        df_main = pd.read_csv('data/model_data_enc.csv', index_col = 0)
        df_main = df_main.set_index('new_date')

        ###CatBoost
        feature_cat = pd.read_csv('data/catboost_feature_importance.csv', index_col = 0)
        predict_cat = pd.read_csv('data/catboost_predictions_df.csv', index_col = 0)

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
            1. Feature selection

                In this notebook, we use the following variables for prediction: 'latitude', 'longitude', 'areaacres', 'za_mean', ('lc_21', 'lc_22', 'lc_23', 'lc_24') combined as lc_2t, 'lc_31', ('lc_41', 'lc_42', 'lc_43') combined as lc_4t, 'lc_52', 'lc_71', 'lc_81', 'lc_82', ('lc_90', 'lc_95') combined as lc_9t, month', 'year', 'week', 'dayofweek', 'hour', 'min', 'quarter', 'airtemp_narr', 'precip3_narr', 'humidity_narr', 'cl_cover_narr', 'sfc_runoff', 'windspeed_narr', 'wdirection_narr', 'precip24_narr', 'precip48_narr', 'of_dist', 'total Nitrogen Oxide in year', and 'date_delta'.
                Date_delta is a numeric variable which capture the time in seconds from the latest record. We could not keep 'new_date' in a datetime format (not supported by Catboost). The reasoning behind creating date_delta is that other time variables (month, year, week, day of week and quarter) are categorical. They can capture a seasonal phenomenon (pollution from industry on weekdays for example) but not a trend over time.
                We removed the following variables: 'new_date' (replaced by datedelta which is numeric), 'huc12', and 'huc12_enc'.
                The dependant variable (target) is the total nitrogen ('tn') in mg/L.

            2. Catboost

                Catboost can deal with missing values internally by giving them the minimal value for that feature (which translates into the guarantee to have a split that separates missing values from all other values). We want to test its capabilities on the non-imputed dataset.
                Furthermore, categorical variables in the dataset don't need to be dummified (that's where the name Cat-boost comes from, it's good with categorical variables).
                Because it's an ensemble method, feature scaling is not necessary.
            """
            st.markdown(modelwrite_up, unsafe_allow_html=True)

            st.write(predict_cat.head(10))
            st.write(feature_cat.head())
            fig = px.bar(feature_cat, feature_cat['Feature Id'], feature_cat['Importances'])
            st.plotly_chart(fig)
            image = Image.open('images/SHAP.png')
            st.image(image, width = 500)

            image = Image.open('images/corr1.png')
            st.image(image, width = 500)

            image = Image.open('images/corr2.png')
            st.image(image, width = 500)



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
        ## Berenice
        about you

        online profile link

        github

        ## Justin
        Into anime, finance computer vision and NLP.

        https://jvhuang1786.github.io

        github

        ## Bryan
        about you

        online profile link

        github

        ## Tim
        about you

        online profile link

        github

        ## Jen
        about you

        online profile link

        github

        """

        st.markdown(about_write,unsafe_allow_html=True)

        st.title('Footnotes')
        st.markdown(html_temp,unsafe_allow_html=True)

        about_write = """

        ## (1) Land cover data

        We downloaded land cover data from the Multi-Resolution Land Characteristics
        Consortium (MRLC) National Land Cover Viewer. A bounding box of the watershed
        (-80.44707800, 36.73004000, -74.83524400, 42.80672000) was used to download
        the relevant 2016 NLCD data in the form of a .tiff raster file. The raster
        file displayed 20 different colors that corresponded with different types
        of land cover. We used this with the watershed HUC12 boundaries shapefile
        provided by the organizers of Hack The Bay.

        With the two files loaded into QGIS, the watershed HUC12 boundaries shapefile
        was used to mask the land cover data and provide the land cover features
        of the .tiff file by HUC12 code. We employed the zonal statistics package
        on the resulting output to determine the mean pixel value of each HUC12
        segment, and a zonal histogram on each segment counted the totals for each
        pixel color/value. We downloaded this layer from QGIS and imported it into
        Python, where the layer features were imported and each HUC12 segment histogram
        was normalized so that each HUC12 segment had a total sum of 1. We then
        merged the information with the existing data on each HUC12 location to
        obtain the features we used.

        [HUC12 boundaries shapefile](https://nrcs.app.box.com/v/huc/file/532373547877)

        [National Land Cover Database 2016 (NLCD2016) Legend](https://www.mrlc.gov/data/legends/national-land-cover-database-2016-nlcd2016-legend)

        ## (2) Weather data

        [North American Regional Reanalysis (NARR)](https://www.ncdc.noaa.gov/data-access/model-data/model-datasets/north-american-regional-reanalysis-narr)

        Data downloaded via: ftp://ftp.cdc.noaa.gov/Datasets/NARR/monolevel

        ## (3) Encoded HUC12

        To address data leakage and introduce regularization, a k-fold method was
        utilized, using the out of fold mean.

        [Target encoding categorical variables (Categorically Speaking)](https://mlbook.explained.ai/catvars.html#target-encoding)

        [Mean Encoding (Geeksforgeeks)](https://www.geeksforgeeks.org/mean-encoding-machine-learning/)

        [Target Encoding (H2O 3.30.1.2 documentation)](https://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-munging/target-encoding.html)

        ## (4) Distance from Outflow

        [Chesapeake Bay outflow location (Wikipedia)](https://en.wikipedia.org/wiki/Chesapeake_Bay)

        ## (5) NO2 from Point Sources

        [Air Emissions Dataset (EPA)](https://echo.epa.gov/tools/data-downloads)
        Combined air emissions data for stationary sources are from four EPA air programs:
        - National Emissions Inventory (NEI)
        - Greenhouse Gas Reporting Program (GHGRP)
        - Toxic Release Inventory (TRI)
        - Clean Air Markets

        Emissions are presented as facility-level aggregates and organized by pollutant
        and EPA program.

        The data was loaded and filtered to only states within the Chesapeake Bay
        Watershed.

        [The Chesapeake’s Airshed (Chesapeake Bay Foundation)](https://www.cbf.org/about-the-bay/maps/geography/the-chesapeakes-airshed.html)

        [Maps (Chesapeake Bay Program)](https://www.chesapeakebay.net/what/maps/chesapeake_bay_airshed#:~:text=The%20Airshed)

        The dataset was then filtered to only the NO2 pollutant emission.
        The original dataset was transformed and aggregated by HUC12 code and year
        and merged by year with the pollutant emissions of NO2 by year by Point Source.
        The data was correlated and filtered to only correlations >=.6 and <1.
        All correlated point sources for each HUC12 segment were aggregated by year.
        For those missing values, we used a mean of the HUC12 correlated point sources.

        """

        st.markdown(about_write,unsafe_allow_html=True)





if __name__ == '__main__':
    main()
