import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import statistics
from IFEEL import ifeel_transformation, ifeel_extraction
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.cluster import KMeans 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skfuzzy.cluster import cmeans as FuzzyCMeans
from sklearn.mixture import GaussianMixture
import seaborn as sns

from scipy.spatial.distance import cdist

######################################################################
############################## Data Cleaning ######################
######################################################################

def handle_missing_vals(dataframes, threshold):
    '''
    Input: Dictionary of dataframes of substation time series data with active power values
    Removes all substations that are missing > threshold % of their active power data
    Output: Dictionary
    '''
    substations_below_threshold = []
    for substation, df in dataframes.items():
        count_zero = (df['Active Power [kW]'] == 0).sum()
        if count_zero > (len(df) * threshold):
            substations_below_threshold.append(substation)
            
    for substation in substations_below_threshold:
        print(f"Substation {substation} has less than 50% available active power data. Dropping from dataframe.")
        del dataframes[substation]


    return dataframes   

def detect_bad_power_vals(df, active_upper_threshold = 700, active_lower_threshold = 0, reactive_upper_threshold = 250, reactive_lower_threshold = -100):
    '''
    Input: Dataframe with time series Active Power and Reactive Power data
    Removes dataframe dates with Active Power or Reactive Power data outside of the thresholds
    Output: Dataframe
    '''
    # Find Active Power values over 1500 kW
    bad_vals = df[(df['Active Power [kW]'] > active_upper_threshold) | (df['Active Power [kW]'] < active_lower_threshold) | (df['Reactive Power [kVAr]'] > reactive_upper_threshold) | (df['Reactive Power [kVAr]'] < reactive_lower_threshold)]
    
    drop_dates = bad_vals['Datetime'].dt.date.unique()

    df = df[~df['Datetime'].dt.date.isin(drop_dates)]
    
    return df 

def split_by_season(df):
    ''' 
    Input: Dataframe with time series data
    Output: Dataframe with time series data split into seasons
    '''
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['Month'] = df['Datetime'].dt.month
    
    spring = df[(df['Month'] >= 3) & (df['Month'] <= 5)]
    summer = df[(df['Month'] >= 6) & (df['Month'] <= 8)]
    fall = df[(df['Month'] >= 9) & (df['Month'] <= 11)]
    winter = df[(df['Month'] == 12) | ((df['Month'] >= 1) & (df['Month'] <= 2))]
    
    return spring, summer, fall, winter

def split_weekend_week(df):
    ''' 
    Input: dataframe with time series data
    Output: two dataframes, one with weekday data and one with weekend data
    '''
    df['weekday'] = df['Datetime'].dt.weekday
    week_df = df[df['weekday'].isin([0,1,2,3,4])]
    weekend_df = df[df['weekday'].isin([5,6])]
    return week_df, weekend_df

def drop_underful_substations(chopped_substation_dfs):
    '''
    Input: Dictionary that is indexed by season and time of week, containing substation time series dataframes
    Removes substations that have too few days for the time of week they represent (weekdays or weekends)
    Output: Dictionary
    '''
    # Drop substations with not enough data in sub-set
    drop_list = []
    for substation, seasons in chopped_substation_dfs.items():
        for season, days in seasons.items():
            for time_of_week, df in days.items():
                days_with_data = len(df['Datetime'].dt.date.unique())
                if time_of_week == 'week':
                    if days_with_data < 33: # 33 days is ~1/2 of the 65 days in a season during the week 
                        print(f"Substation {substation} has only {days_with_data} days of data in {season} {time_of_week} (<~1/2 of what should be there). Dropping from analysis.")
                        drop_list.append(str(substation) + "_" + str(season) + "_" + str(time_of_week))
                else:
                    if days_with_data < 13: #(has less than half of weekends)
                        print(f'Substation {substation} has only {days_with_data} days of data in {season} {time_of_week} (<~1/2 of what should be there). Dropping from analysis.')
                        drop_list.append(str(substation) + "_" + str(season) + "_" + str(time_of_week))
                        
    print("number of sub-datasets to drop,", len(drop_list))
    for to_drop in drop_list:
        substation = to_drop.split("_")[0]
        season = to_drop.split("_")[1]
        time_of_week = to_drop.split("_")[2]
        del chopped_substation_dfs[substation][season][time_of_week]
        
    return chopped_substation_dfs
                
                
######################################################################
############################## Data Exploration ######################
######################################################################


def plot_peak_hour_distributions(dataframes, active):
    ''' 
    Input: Dictionary of dataframes of substation time series data with active power values and reactive power values
    Plots histograms of the distribution of active power and reactive power values for each substation
    '''
    for substation, substation_data in dataframes.items():
        substation_data['Date'] = substation_data['Datetime'].dt.date
        substation_data['Hour'] = substation_data['Datetime'].dt.hour
        
        # Group the data by date
        daily_groups = substation_data.groupby(['Date'])

        if active:
        # Find the hour of peak Active Power for each date 
            peak_hour = daily_groups['Active Power [kW]'].idxmax().map(lambda x: substation_data.loc[x, 'Hour'])
                        # Count the number of times each hour appears as the peak hour for each day
            peak_counts = peak_hour.value_counts()
            
            plt.bar(peak_counts.index, peak_counts.values)
            plt.xlabel('Hour of the day')
            plt.ylabel('Frequency of being peak hour')
            plt.title(f'Peak Hour Distribution for {substation}')
            plt.savefig(f'peak_hr_histograms/peak_hour_distribution_{substation}.png')
            plt.show()
            
        else:
                # Find the hour of peak positive and negative Reactive power for each day
            peak_hour_max = daily_groups['Reactive Power [kVAr]'].idxmax().map(lambda x: substation_data.loc[x, 'Hour'])
            peak_hour_min = daily_groups['Reactive Power [kVAr]'].idxmin().map(lambda x: substation_data.loc[x, 'Hour'])
            
                # Count the number of times each hour appears as the peak hour for each day
            peak_counts_max = peak_hour_max.value_counts()
            peak_counts_min = peak_hour_min.value_counts()
            
            plt.bar(peak_counts_max.index, peak_counts_max.values)
            plt.xlabel('Hour of the day')
            plt.ylabel('Frequency of being peak hour')
            plt.title(f'Positive Reactive Power Peak Hour Distribution for {substation}')
            plt.show()
            plt.bar(peak_counts_min.index, peak_counts_min.values)
            plt.xlabel('Hour of the day')
            plt.ylabel('Frequency of being peak hour')
            plt.title(f'Negative Reactive Power Peak Hour Distribution for {substation}')
            plt.show()
            
            
######################################################################
############################## Feature Extraction ####################
######################################################################



def extract_global_features_v3(substation, df, active = True):
    '''
    Input: name of substation, and the dataframe of the substation's time series data
    Compute global features for the substation's time series data for active and reactive power
    Output: Dictionary of global features
    '''
    # convert datetime column to datetime type
    df['Datetime'] = pd.to_datetime(df['Datetime'])

    df['Date'] = df['Datetime'].dt.date
    df['Hour'] = df['Datetime'].dt.hour

    # set datetime column as index
    df = df.set_index('Datetime')

    first = True
    for feature_of_interest in ['Active Power [kW]', 'Reactive Power [kVAr]']:

        # group data by day to get daily load pattern (total energy consumed in a day)
        daily_groups = df.groupby(df['Date'])

        # Total daily power consumption
        total_power = daily_groups[feature_of_interest].sum()

        # Mean daily power consumption 
        mean_powers = daily_groups[feature_of_interest].mean()

        # Std deviation of daily power consumption
        sd_powers = daily_groups[feature_of_interest].std()

        # Max power consumption during a day
        max_powers = daily_groups[feature_of_interest].max()

        # Min power consumption during a day
        min_powers = daily_groups[feature_of_interest].min()

        # Range of power consumption during a day
        range_powers = max_powers - min_powers

        # Percent values above mean value in each day
        above_mean_counts = [(df[(df['Date'] == date) & (df[feature_of_interest] > mean)]).shape[0] for date, mean in mean_powers.iteritems()]
        percentage_above_mean = [above_mean_count / 24 * 100 for i, above_mean_count in enumerate(above_mean_counts)]


        if feature_of_interest=='Reactive Power [kVAr]':
            hot_vals = [(df[(df['Date'] == date) & (abs(df[feature_of_interest]) > 1.25 * abs(mean))]).shape[0] for date, mean in mean_powers.iteritems()]
            cold_vals = [(df[(df['Date'] == date) & (abs(df[feature_of_interest]) < 0.75 * abs(mean))]).shape[0] for date, mean in mean_powers.iteritems()]
        else:
            # Number of hours above 1.25 * mean value in each day
            hot_vals = [(df[(df['Date'] == date) & (df[feature_of_interest] > 1.25 * mean)]).shape[0] for date, mean in mean_powers.iteritems()]
            # Numer of hours below 0.75 * mean value in each day
            cold_vals = [(df[(df['Date'] == date) & (df[feature_of_interest] < 0.75 * mean)]).shape[0] for date, mean in mean_powers.iteritems()]
            
            

        # Filter the dataframe to include only the hours between 9 and 17
        filtered_business_hours = df[(df['Hour'] >= 9) & (df['Hour'] <= 18)]
        # Group by date
        grouped_business_hours = filtered_business_hours.groupby(by=['Date'])
        # Sum of net loads during business hours (9am-6pm)
        business_hour_loads = grouped_business_hours[feature_of_interest].sum()

        ####################### Now for non-business hours (the opposite) #############################
        filtered_non_business_hours = df[(df['Hour'] < 9) | (df['Hour'] > 18)]
        grouped_non_business_hours = filtered_non_business_hours.groupby(by=['Date'])
        non_business_hour_loads = grouped_non_business_hours[feature_of_interest].sum()

        # Skewness 
        skewness_daily_load_pattern = daily_groups[feature_of_interest].sum().skew()

        # Kurtosis 
        kurtosis_daily_load_pattern = daily_groups[feature_of_interest].sum().kurtosis()

        # Mode of 5-bin histogram for daily load pattern
        hist, bin_edges = np.histogram(daily_groups[feature_of_interest].sum(), bins=5)
        mode_5_bin_histogram = bin_edges[np.argmax(hist)]

        if first:
            features = {
                f'Mean total daily load consumption {feature_of_interest}' : total_power.mean(),
                f'Mean value of daily load pattern {feature_of_interest}' : mean_powers.mean(),
                f'SD of daily load pattern {feature_of_interest}': sd_powers.mean(),
                f'Max power consumption during a day {feature_of_interest}': max_powers.max(),
                f'Min power consumption during a day {feature_of_interest}': min_powers.min(),
                f'Range of power consumption during a day (max - min) {feature_of_interest}': range_powers.max(),
                f'Percent values above mean val (%) {feature_of_interest}': statistics.mean(percentage_above_mean),
                f'Number of hours above 1.25 * mean val {feature_of_interest}': statistics.mean(hot_vals),
                f'Number of hours below 0.75 * mean val {feature_of_interest}': statistics.mean(cold_vals),
                f'Sum of net loads during business hours (9am-6pm) {feature_of_interest}': business_hour_loads.mean(),
                f'Sum of net loads during non-business hours {feature_of_interest}': non_business_hour_loads.mean(),
                f'Skewness of the distribution of a daily load pattern {feature_of_interest}': skewness_daily_load_pattern.mean(),
                f'Kurtosis of distribution of a daily load pattern {feature_of_interest}': kurtosis_daily_load_pattern.mean(),
                f'Mode of 5-bin histogram for daily load pattern {feature_of_interest}': mode_5_bin_histogram
            }
        else:
            next = {
                f'Mean total daily load consumption {feature_of_interest}' : total_power.mean(),
                f'Mean value of daily load pattern {feature_of_interest}' : mean_powers.mean(),
                f'SD of daily load pattern {feature_of_interest}': sd_powers.mean(),
                f'Max power consumption during a day {feature_of_interest}': max_powers.max(),
                f'Min power consumption during a day {feature_of_interest}': min_powers.min(),
                f'Range of power consumption during a day (max - min) {feature_of_interest}': range_powers.max(),
                f'Percent values above mean val (%) {feature_of_interest}': statistics.mean(percentage_above_mean),
                f'Number of hours above 1.25 * mean val {feature_of_interest}': statistics.mean(hot_vals),
                f'Number of hours below 0.75 * mean val {feature_of_interest}': statistics.mean(cold_vals),
                f'Sum of net loads during business hours (9am-6pm) {feature_of_interest}': business_hour_loads.mean(),
                f'Sum of net loads during non-business hours {feature_of_interest}': non_business_hour_loads.mean(),
                f'Skewness of the distribution of a daily load pattern {feature_of_interest}': skewness_daily_load_pattern.mean(),
                f'Kurtosis of distribution of a daily load pattern {feature_of_interest}': kurtosis_daily_load_pattern.mean(),
                f'Mode of 5-bin histogram for daily load pattern {feature_of_interest}': mode_5_bin_histogram
            }
            features.update(next)
        first = False 

    
    return features

  
def get_peak_hour_distribution(substation_data, time_intervals, time_labels, feature_of_interest):
    '''
    Input: substation dataframe, intervals of time that designate parts of the day and concomitant labels, and the feature of interest (active or reactive power)
    Obtains the part of the day that peak and valley power consumption occurs in
    Output: Float of the most common peak and valley parts of the day for the feature of interest
    '''
    # Extract Date and Hour columns from Datetime column 
    substation_data['Date'] = substation_data['Datetime'].dt.date
    substation_data['Hour'] = substation_data['Datetime'].dt.hour

    # Group the data by Date
    daily_groups = substation_data.groupby(['Date'])

    # Find the hour of Peak positive and Valley negative Reactive power for each day
    peak_hour_max = daily_groups[feature_of_interest].idxmax().map(lambda x: substation_data.loc[x, 'Hour'])
    peak_hour_min = daily_groups[feature_of_interest].idxmin().map(lambda x: substation_data.loc[x, 'Hour'])

    # Count the number of times each hour appears as the peak hour for each day
    peak_counts_max = peak_hour_max.value_counts()
    peak_counts_min = peak_hour_min.value_counts()

    # Add the distributions to the dictionary
    most_common_peak_max = peak_counts_max.idxmax()
    most_common_peak_min = peak_counts_min.idxmax()

    for i, (start,end) in enumerate(time_intervals):
        if start <= most_common_peak_max < end:
            positive_peak_part_of_day = time_labels[i]
        if start <= most_common_peak_min < end:
            negative_peak_part_of_day = time_labels[i]
        

    return positive_peak_part_of_day, negative_peak_part_of_day


def get_peak_hour(substation_data, feature_of_interest):
    '''
    Input: substation dataframe and the feature of interest (active or reactive power)
    Obtains the hour of the day that peak and valley power consumption occurs in
    Output: Float of the most common peak and valley hours of the day for the feature of interest
    '''
      # Extract Date and Hour columns from Datetime column 
    substation_data['Date'] = substation_data['Datetime'].dt.date
    substation_data['Hour'] = substation_data['Datetime'].dt.hour

    # Group the data by Date
    daily_groups = substation_data.groupby(['Date'])

    # Find the hour of Peak positive and Valley negative Reactive power for each day
    peak_hour_max = daily_groups[feature_of_interest].idxmax().map(lambda x: substation_data.loc[x, 'Hour'])
    peak_hour_min = daily_groups[feature_of_interest].idxmin().map(lambda x: substation_data.loc[x, 'Hour'])

    # Count the number of times each hour appears as the peak hour for each day
    peak_counts_max = peak_hour_max.value_counts()
    peak_counts_min = peak_hour_min.value_counts()

    # Add the distributions to the dictionary
    most_common_peak_max = peak_counts_max.idxmax()
    most_common_peak_min = peak_counts_min.idxmax()
      
      

    return most_common_peak_max, most_common_peak_min


def plot_correlation_matrices(df_dict, global_active_features, global_reactive_features):
    '''
    Input: Dictionary of substation dataframes, lists of global active and reactive features
    Plots correlation matrices for features in each dataframe in the dictionary
    Output: None
    '''
        # Iterate through the dataframes in the dictionary
    for k,v in df_dict:
        df = df_dict[k,v]

        #df_dict[k,v]['Most common Active Power peak time of day'] = df_dict[k,v]['Most common Active Power peak time of day'].astype(float)
        df = df.astype({'Most common maximum Active Power [kW] peak time of day':'float',
                        'Most common minimum Active Power [kW] peak time of day':'float',
                        'Most common maximum Reactive Power [kVAr] peak time of day':'float',
                        'Most common minimum Reactive Power [kVAr] peak time of day': 'float'})
        
        # Build a correlatrion matrix between all the features in the dataframe (excluding cluster labels)
        corr_matrix = df.loc[:, ~df.columns.str.contains('cluster')].corr()
        
        # Build active and reactive features only correlation matrices
        ap_features_corr_matrix = df.loc[:, global_active_features + ['Most common maximum Active Power [kW] peak time of day', 'Most common minimum Active Power [kW] peak time of day']].corr()
        rp_features_corr_matrix = df.loc[:, global_reactive_features + ['Most common maximum Reactive Power [kVAr] peak time of day', 'Most common minimum Reactive Power [kVAr] peak time of day' ]].corr()

        # Create labels for the plots
        feature = 'Active'
        active_feature_labels = [f'GF-1 ({feature})', f'GF-2 ({feature})', f'GF-3 ({feature})', f'GF-4 ({feature})', f'GF-5 ({feature})',
                                f'GF-6 ({feature})', f'GF-7 ({feature})', f'GF-8 ({feature})', f'GF-9 ({feature})', f'GF-10 ({feature})', 
                                f'GF-11 ({feature})', f'GF-12 ({feature})',f'GF-13 ({feature})', f'GF-14 ({feature})',
                                f'PF-1 ({feature})', f'PF-2 ({feature})']
        feature = 'Reactive'
        reactive_feature_labels =[f'GF-1 ({feature})', f'GF-2 ({feature})', f'GF-3 ({feature})', f'GF-4 ({feature})', f'GF-5 ({feature})',
                                f'GF-6 ({feature})', f'GF-7 ({feature})', f'GF-8 ({feature})', f'GF-9 ({feature})', f'GF-10 ({feature})',
                                f'GF-11 ({feature})', f'GF-12 ({feature})', f'GF-13 ({feature})', f'GF-14 ({feature})',
                                f'PF-1 ({feature})', f'PF-2 ({feature})']
        
        # All feature labels (reorganized so in correct order)
        all_feature_labels = active_feature_labels[:-2] + reactive_feature_labels[:-2] + ['PF-1 (Active)', 'PF-2 (Active)' ,'PF-1 (Reactive)', 'PF-2 (Reactive)']
        
        
        # Create a heatmap of the correlation matrix with seaborn
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, vmin=-1, vmax=1, center=0, cmap='coolwarm',
                    annot=False, fmt='.2f', square=True, xticklabels=all_feature_labels, yticklabels=all_feature_labels)

        # Add a title
        plt.title(f'All Features Correlation Matrix for {v}, {k}')
        plt.tight_layout()
        plt.savefig(f"../figures/corr_matrix_{v}_{k}.png",facecolor = 'white', edgecolor = 'black')
        plt.show()


        # Create a heatmap of the correlation matrix with seaborn
        plt.figure(figsize=(10, 8))
        sns.heatmap(ap_features_corr_matrix, vmin=-1, vmax=1, center=0, cmap='coolwarm',
                    annot=False, fmt='.2f', square=True, xticklabels=active_feature_labels, yticklabels=active_feature_labels)

        plt.title(f'Active Power Features Correlation Matrix for {v}, {k}')
        plt.tight_layout()
        plt.savefig(f"../figures/AP_corr_matrix_{v}_{k}.png",facecolor = 'white', edgecolor = 'black')
        plt.show()



        plt.figure(figsize=(10, 8))
        sns.heatmap(rp_features_corr_matrix, vmin=-1, vmax=1, center=0, cmap='coolwarm',
                    annot=False, fmt='.2f', square=True, xticklabels=reactive_feature_labels, yticklabels=reactive_feature_labels)
        plt.title(f'Reactive Power Features Correlation Matrix for {v}, {k}')
        plt.tight_layout()
        plt.savefig(f"../figures/RP_corr_matrix_{v}_{k}.png", facecolor = 'white', edgecolor = 'black')
        plt.show()

def create_substation_features_dataframes(chopped_substation_dfs, time_labels, time_intervals):
    '''
    Inputs: Dictionary holding the chopped up dataframes for each substation, indexed by season and time of week, as well as intervals for times of the day and accompanying labels
    
    Outputs: Dictionary of dictionaries, indexed by substation, season, and time of week, that hold dictionaries of all the features values for each substation.
    '''
    features_dataframes = {}
    active = False 
    specific_time = True
    for substation, season in chopped_substation_dfs.items():
        features_dataframes[substation]  = {}
        for season_name, times_of_week in season.items():
            features_dataframes[substation][season_name] = {}
            for time_of_week, df in times_of_week.items():
                
                # Get global features in a dictionary
                features = extract_global_features_v3(substation, df, active)
                
                # Get peak hour distributions. What I really need is just for the peak hour distributions to be added to the features dictionary... hmmm... so I want a key that is 'peak hour distribution' and the value to be that distribution? but then that's different than all the other things
                
                for active in [True, False]:
                    if active:
                        feature_of_interest= 'Active Power [kW]'
                    else:
                        feature_of_interest = 'Reactive Power [kVAr]'
                        
                        
                    if specific_time:
                        features[f'Most common maximum {feature_of_interest} peak time of day'], features[f'Most common minimum {feature_of_interest} peak time of day'] = get_peak_hour(df,feature_of_interest)
                    else:
                        features[f'Most common maximum {feature_of_interest} peak time of day'], features[f'Most common minimum {feature_of_interest} peak time of day'] = get_peak_hour_distribution(df, time_intervals, time_labels, feature_of_interest)

        
                features_dataframes[substation][season_name][time_of_week] = features 
    return features_dataframes

def split_into_seasonal_dfs(features_dataframes, global_active_features, global_reactive_features, peak_features):
    '''
    Inputs: Dictionary of dictionaries, indexed by substation, season, and time of week, that holds a dictionary of features for each substation. Also, lists of the global active and reactive features, as well as the peak features.
    
    Outputs: A dictionary of dataframes, indexed by substation, season, and time of week, that holds the features for each substation. The dataframes are split into seasonal dataframes, and the seasonal dataframes are split into time of week dataframes.'''
    active_and_reactive_features = global_active_features + global_reactive_features + peak_features

    # Create an empty dictionary to hold the dataframes
    df_dict = {}
    PCA_clustering = True
    if PCA_clustering: # If we're doing PCA clusteirng, we are using large feature set, otherwise using smaller feature set for greater interpretability 
        feature_set = ['substation'] + active_and_reactive_features
    else:
        feature_set = ['substation'] + [global_active_features[i] for i in [0,7,8]] + [global_reactive_features[i] for i in [0,7,8]] + peak_features
        
        
    # Loop through the nested dictionary
    for substation, season_dict in features_dataframes.items():
        for season, time_dict in season_dict.items():
            for time_of_week, feature_dict in time_dict.items():
                # If the dataframe doesn't exist for this season/time_of_week combination, create it
                if (time_of_week, season) not in df_dict:
                    df_dict[(time_of_week, season)] = pd.DataFrame(columns= feature_set)
                # Create a new row of feature data for the substation
                feature_data = {"substation": substation}
                feature_data.update(feature_dict)

                # Convert the feature data into a dataframe and append it to the corresponding dataframe in df_dict
                df_dict[(time_of_week, season)] = df_dict[(time_of_week, season)].append(feature_data, ignore_index=True)
                
                
    return df_dict


######################################################################
############################## Dimensionality Reduction ##############
######################################################################

def pca_plot(df):
    '''
    Input: Seasonal/time-of-week dataframe with all substations and features extracted
    Plot scree plot
    Output: None
    '''
    scaler = StandardScaler()
    scaler.fit_transform(df)

    pca = PCA()
    pca.fit(df)

    np.set_printoptions(precision=4, suppress=True)
    print(pca.explained_variance_ratio_)

    plt.figure(figsize = (5,4))
    plt.plot(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_.cumsum(), marker = 'o', linestyle = '--')
    plt.title('Explained Variance by Components')
    plt.xlabel('Number of Components*')
    plt.ylabel('Cumulative Explained Variance')
    plt.ylim(0,1.0)
    plt.show()
    for key in df_dict:
        print(key)
        df = df_dict[key].set_index('substation')
        pca_plot(df)
        
    
    
def get_scores_pca(df, n_components):
    '''
    Input: Season/time-of-week dataframe with all substations and features, and a given number of components
    Fit PCA model and calculate scores
    Output: PCA scores
    '''
    pca = PCA(n_components = n_components)
    #Fit the model the our data with the selected number of components. In our case three.
    pca.fit(df)

    pca.transform(df)
    scores_pca = pca.transform(df)
    
    return scores_pca


######################################################################
############################## Clustering ############################
######################################################################


def cluster_on_pca_scores(df, scores_pca, n_clusters: int):
    '''
    Input: Seasonal dataframe with all substations and features for them, PCA scores and number of clusters (k) to use
    Use K-Means clustering on the 
    Output: Dataframe with cluster labels and K-Means model
    '''
        # We have chosen four clusters, so we run K-means with number of clusters equals four.
    # Same initializer and random state as before.
    kmeans_pca = KMeans(n_clusters = n_clusters, init = 'k-means++', random_state = 42)
    # We fit our data with the k-means pa model
    kmeans_pca.fit(scores_pca)
    # We create a new data frame with the original features and add the PC scores and assigned clusters.
    df_segm_pca_kmeans = pd.concat([df.reset_index(drop = True), pd.DataFrame(scores_pca)], axis = 1)
    df_segm_pca_kmeans.columns.values[-3: ] = ['Component 1','Component 2', 'Component 3']
    # The last column we add contains the pea k-means clustering labels.
    df_segm_pca_kmeans['Segment K-means PCA'] = kmeans_pca.labels_

    return df_segm_pca_kmeans,kmeans_pca

def implement_pca_clustering(df_dict):
    '''
    Inputs: Dictionary of dataframes, indexed by substation, season, and time of week, that holds the features for each substation. 
    
    Fits PCA model and applies K-Means clustering to PCA scores for each season/time of week combo
    
    Outputs: Clustering results for each season/time of week combo
    '''
    pca_cluster_results = {}
    n_components = 2
    range_of_k_vals = range(2,11)
    for key in df_dict.keys():
        
        # Put substation in index so that PCA can work (doesn't work with strings)
        df = df_dict[key].set_index('substation')
        # Plot PCA results to determine number of components to keep 
        pca_plot(df)
        # Get PCA scores
        scores_pca = get_scores_pca(df, n_components)
        
        # Initialize the dictionary entry for this season/time of week combo 
        pca_cluster_results[key] = df
        
        # Initialize arrays for elbow curve, silhouette scores, and Davies Bouldin index
        elbow_scores = []
        silhouette_scores = []
        davies_bouldin_scores = []
        
        # Run K-means on PCA scores for different numbers of clusters 
        for n_clusters in range_of_k_vals:
            # Get clustering results
            these_results,kmeans_pca = cluster_on_pca_scores(df, scores_pca, n_clusters)
            # Add results to dataframe
            pca_cluster_results[key][f'PCA_clustering_k={n_clusters}'] = np.array(these_results['Segment K-means PCA'])
            
            # Calculate and store elbow score
            elbow_scores.append(kmeans_pca.inertia_)
            
            # Calculate and store silhouette score
            sil_score = silhouette_score(scores_pca, kmeans_pca.labels_)
            silhouette_scores.append(sil_score)
            
            # Calculate and store Davies Bouldin index score
            db_score = davies_bouldin_score(scores_pca, kmeans_pca.labels_)
            davies_bouldin_scores.append(db_score)
            
        # Plot elbow curve
        plt.plot(range(2,11), elbow_scores)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title(f'Elbow Curve for {key}')
        plt.show()
        
        # Plot silhouette scores
        plt.plot(range(2,11), silhouette_scores)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title(f'Silhouette Scores for {key}')
        plt.show()
        
        # Plot Davies Bouldin index scores
        plt.plot(range(2,11), davies_bouldin_scores)
        plt.xlabel('Number of Clusters')
        plt.ylabel('Davies Bouldin Index Score')
        plt.title(f'Davies Bouldin Index Scores for {key}')
        plt.show()
    return pca_cluster_results


def k_means(df, k_values = range(2,11), active_only = False):
    '''
    Input: dataframe wtih all substation and feature values, desired range of k values to test, and whether to only use active power features or all featuers (bool)
    Conduct KMeans clustering for each value of k in k_values, plot silhouette scores, DBI, and elbow cost 
    Output: Dataframe with cluster label results, and a dict with the results
    '''
    if active_only:
        # Get the feature set - only active power features
        X = df.loc[:, df.columns.str.contains('Active Power')].values
    else:
        # Get the feature set - all features
        X = df.loc[:, df.columns != 'substation'].values

    # Create an empty list to hold the silhouette scores and DBI and elbow cost
    silhouette_scores = []
    dbi_scores = []
    wss = []
    
    # Create empty dict to hold results
    results = {}
    
    # Create an empty dictionary to hold the data points cloest to the center
    cluster_centers = {}
    
    # Loop through each value of k
    for k in k_values:
        # Fit the k-means model to the feature set
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X)
        # Calculate the silhouette score and DBI for the clustering
        score = silhouette_score(X, labels)
        db_index = davies_bouldin_score(X, labels)
        silhouette_scores.append(score)
        dbi_scores.append(db_index)
        
        # Assign cluster labels to substations
        df[f'cluster_{k}'] = kmeans.labels_
        
        # Calculate the elbow cost for the clustering
        kmeans.fit(X)
        wss.append(kmeans.inertia_)
        
        # Save clustering results
        results[k] = labels  

    # # Plot elbow curve
    # plt.plot(k_values, wss)
    # plt.xlabel('Number of clusters (k)')
    # plt.ylabel('WSS')
    # plt.title('Elbow Curve')
    # plt.show()

    # # Plot the silhouette scores and DBI for each k value
    # fig, ax = plt.subplots(2, 1, figsize=(10,10))
    # ax[0].plot(k_values, silhouette_scores)
    # ax[0].set_xlabel('Number of clusters (k)')
    # ax[0].set_ylabel('Silhouette score')
    # ax[0].set_title('Silhouette score for k-means clustering')
    ax[1].plot(k_values, dbi_scores)
    ax[1].set_xlabel('Number of clusters (k)')
    ax[1].set_ylabel('DBI')
    ax[1].set_title('Davies Bouldin Index for k-means clustering')
    plt.show()
    
    return df, results#, cluster_centers
