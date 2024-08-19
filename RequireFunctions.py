import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from statsmodels.multivariate.manova import MANOVA
from itertools import combinations

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

def vowel_space_expansion(in_file_path, target_vowels_IPA, register, feature_column_names, out_file_path, parentgender=None):
    """
    Calculates the vowel space expansion (convex hull area) for specific vowels
    based on given acoustic features, and saves the results to an Excel file.

    This function reads vowel data from a CSV file, filters it by the specified vowels, 
    and optionally by the parent's gender, computes the mean values for the given 
    acoustic features (such as F1, F2, or MFCC features) for each vowel, participant (spkid), 
    and age group, and then calculates the convex hull area to represent the vowel space expansion. 
    The results, including the convex hull area, are saved to an Excel file.

    Parameters:
    -----------
    in_file_path : str
        The path to the input CSV file containing vowel data with IPA symbols, acoustic features, 
        and metadata such as 'spkid', 'AgeMonth', and 'ParentGender'.
        
    target_vowels_IPA : list of str
        A list of target IPA symbols representing the vowels to be included in the analysis.
        
    register : str
        A string representing the register or context (e.g., "ADS" or "IDS") associated with the data.
        
    feature_column_names : list of str
        A list of column names in the DataFrame that represent the acoustic features to be used for 
        convex hull calculation. This can be a list containing F1 and F2 for formant analysis, or 
        a list of MFCC feature names (e.g., 'MFCC1', 'MFCC2', ..., 'MFCC13').

    out_file_path : str
        The path where the output Excel file with the convex hull areas will be saved.

    parentgender : str, optional
        A string representing the gender of the parent (e.g., "Male", "Female"). If specified, the 
        function filters the data to include only records where the 'ParentGender' column matches 
        this value. If not specified (default is None), no filtering by parent gender is performed.

    Returns:
    --------
    None
        The function saves the results directly to an Excel file specified by `out_file_path`.
    
    Example:
    --------
    in_file_path = "path/to/input_data.csv"
    target_vowels_IPA = ['a', 'e', 'i', 'o', 'u', 'æ']
    register = "ADS"
    feature_column_names = ['F1', 'F2']  # For formant analysis
    out_file_path = "path/to/output_file.xlsx"
    vowel_space_expansion(in_file_path, target_vowels_IPA, register, feature_column_names, out_file_path)
    
    feature_column_names = ['MFCC1', 'MFCC2', 'MFCC3', ..., 'MFCC13']  # For MFCC analysis
    vowel_space_expansion(in_file_path, target_vowels_IPA, register, feature_column_names, out_file_path, parentgender="Female")

    Notes:
    ------
    - The input CSV file should have columns for 'spkid', 'AgeMonth', 'IPA', and the specified feature columns.
    - The convex hull area or volume (Volume of the convex hull when input dimension > 2. When input points are 
      2-dimensional, this is the area of the convex hull) is calculated using the selected features for each vowel 
      group defined by 'spkid' and 'AgeMonth'.
    - The function assumes that there are at least three distinct vowel points for each participant and age group 
      to form a convex hull. If there are fewer than three points, an error will be raised by `ConvexHull`.
    - The results DataFrame saved to the Excel file includes columns for 'spkid', 'AgeMonth', 'Register', 
      'ParentGender' (if filtered), and 'ConvexHullArea'.
    """
    
    df = pd.read_csv(in_file_path)
    if parentgender != None:
        df = df[df['Parent'] == parentgender].copy()
        df = df[df['IPA'].isin(target_vowels_IPA)].copy()
        df.reset_index(inplace=True)
    else:
        df = df[df['IPA'].isin(target_vowels_IPA)].copy()
        df.reset_index(inplace=True)
    
    # Group by 'spkid', 'AgeMonth', and 'IPA' to compute the mean feature values for each vowel
    mean_features = df.groupby(['spkid', 'AgeMonth', 'IPA'])[feature_column_names].mean().reset_index()
    
    # Group by 'spkid' and 'AgeMonth' to compute the convex hull area
    results = []
    for (spkid, age), group in mean_features.groupby(['spkid', 'AgeMonth']):
        feature_values = group[feature_column_names].values
        if feature_values.shape[0] > feature_values.shape[1]:  # Check if there are at least 3 points
            area = ConvexHull(feature_values).volume
            results.append({'spkid': spkid, 'AgeMonth': age, 'Register': register, 'ConvexHullArea': round(area)})
        elif feature_values.shape[0] <= feature_values.shape[1] and feature_values.shape[1] > 2 and feature_values.shape[0] > 2:
            pca = PCA(n_components=feature_values.shape[0]-1)
            feature_values = pca.fit_transform(feature_values)
            area = ConvexHull(feature_values).volume
            results.append({'spkid': spkid, 'AgeMonth': age, 'Register': register, 'ConvexHullArea': round(area)})
        else:
            results.append({'spkid': spkid, 'AgeMonth': age, 'Register': register, 'ConvexHullArea': np.nan})
    
    # Convert results to DataFrame and save to Excel
    results_df = pd.DataFrame(results)
    results_df.dropna(inplace=True)
    if parentgender != None:
        results_df.to_excel(out_file_path + '_' + parentgender + '.xlsx', index=False)
    else:
        results_df.to_excel(out_file_path + '.xlsx', index=False)
    
    print("Convex hull areas have been saved")


def vowel_variability(in_file_path, target_vowels_IPA, register, feature_column_names, out_file_path, parentgender=None):
    """
    Calculates the vowel space variability (convex hull area/volume) for specific vowels
    based on given acoustic features, considering only points that are one standard deviation 
    away from the mean, and saves the results to an Excel file.

    This function reads vowel data from a CSV file, filters it by the specified vowels 
    and optionally by the parent's gender, computes the mean values and filters data points 
    that are two standard deviation away from the mean for the given acoustic features 
    (such as F1, F2, or MFCC features) of a vowel for each participant and age, and then calculates the convex hull area 
    or volume to represent vowel variability. The results are saved to an Excel file.

    Parameters:
    -----------
    in_file_path : str
        The path to the input CSV file containing vowel data with IPA symbols, acoustic features, 
        and metadata such as 'spkid', 'AgeMonth', and 'ParentGender'.
        
    target_vowels_IPA : list of str
        A list of target IPA symbols representing the vowels to be included in the analysis.
        
    register : str
        A string representing the register or context (e.g., "ADS" or "IDS") associated with the data.
        
    feature_column_names : list of str
        A list of column names in the DataFrame that represent the acoustic features to be used for 
        convex hull calculation. This can be a list containing F1 and F2 for formant analysis, or 
        a list of MFCC feature names (e.g., 'MFCC1', 'MFCC2', ..., 'MFCC13').

    out_file_path : str
        The path where the output Excel file with the convex hull areas/volumes will be saved.

    parentgender : str, optional
        A string representing the gender of the parent (e.g., "Male", "Female"). If specified, the 
        function filters the data to include only records where the 'ParentGender' column matches 
        this value. If not specified (default is None), no filtering by parent gender is performed.

    Returns:
    --------
    None
        The function saves the results directly to an Excel file specified by `out_file_path`.
    
    Example:
    --------
    in_file_path = "path/to/input_data.csv"
    target_vowels_IPA = ['a', 'e', 'i', 'o', 'u', 'æ']
    register = "ADS"
    feature_column_names = ['F1', 'F2']  # For formant analysis
    out_file_path = "path/to/output_file.xlsx"
    vowel_space_expansion(in_file_path, target_vowels_IPA, register, feature_column_names, out_file_path)
    
    feature_column_names = ['MFCC1', 'MFCC2', 'MFCC3', ..., 'MFCC13']  # For MFCC analysis
    vowel_space_expansion(in_file_path, target_vowels_IPA, register, feature_column_names, out_file_path, parentgender="Female")

    Notes:
    ------
    - The input CSV file should have columns for 'spkid', 'AgeMonth', 'IPA', and the specified feature columns.
    - The convex hull area or volume (Volume of the convex hull when input dimension > 2. When input points are 
      2-dimensional, this is the area of the convex hull) is calculated using the selected features for each vowel 
      defined by 'spkid' and 'AgeMonth'.
    - The function filters the data to include only points that are within one standard deviation from the mean.
    - The results DataFrame saved to the Excel file includes columns for 'spkid', 'AgeMonth', 'Register', 
      'ParentGender' (if filtered), and 'ConvexHullArea/Volume'.
    """
    
    df = pd.read_csv(in_file_path)
    if parentgender:
        df = df[df['Parent'] == parentgender].copy()
    
    df = df[df['IPA'].isin(target_vowels_IPA)].copy()
    df.reset_index(inplace=True)
    
    # Group by 'spkid', 'AgeMonth', and 'IPA' to compute the mean and std for each vowel
    mean_features = df.groupby(['spkid', 'AgeMonth', 'IPA'])[feature_column_names].mean().reset_index()
    std_features = df.groupby(['spkid', 'AgeMonth', 'IPA'])[feature_column_names].std().reset_index()
    
    # Filter data to include only points within one standard deviation from the mean
    filtered_df = pd.merge(df, mean_features, on=['spkid', 'AgeMonth', 'IPA'], suffixes=('', '_mean'))
    filtered_df = pd.merge(filtered_df, std_features, on=['spkid', 'AgeMonth', 'IPA'], suffixes=('', '_std'))

    condition = True
    for feature in feature_column_names:
        condition &= (filtered_df[feature] >= (filtered_df[feature + '_mean'] - 2 * filtered_df[feature + '_std'])) & \
                     (filtered_df[feature] <= (filtered_df[feature + '_mean'] + 2 * filtered_df[feature + '_std']))

    filtered_df = filtered_df[condition]

    # Group by 'spkid' and 'AgeMonth' to compute the convex hull area/volume
    results = []
    for (spkid, age, vowel), group in filtered_df.groupby(['spkid', 'AgeMonth', 'IPA']):
        feature_values = group[feature_column_names].values
        mean_feature_values = np.mean(feature_values, axis=0).reshape(1, -1) # mean of each feature
        eq_dist = cdist(feature_values, mean_feature_values, metric='euclidean').flatten()
        sum_eq_dist = np.sum(eq_dist) # Sum the distances
        average_sum_distance = sum_eq_dist / len(feature_values) # average sum of distances
        results.append({'spkid': spkid, 'AgeMonth': age,'IPA': vowel, 'Register': register, 'variability': round(average_sum_distance)})
    # Convert results to DataFrame and save to Excel
    results_df = pd.DataFrame(results)
    results_df.dropna(inplace=True)
    
    if parentgender:
        results_df.to_excel(out_file_path + '_' + parentgender + '.xlsx', index=False)
    else:
        results_df.to_excel(out_file_path + '.xlsx', index=False)
    
    print("Variabilities have been saved")



def separability(in_file_path, target_vowels_IPA, register, feature_column_names, out_file_path, parentgender=None):
    # Load data
    df = pd.read_csv(in_file_path)
    
    # Filter data based on the parentgender and target_vowels_IPA
    if parentgender is not None:
        df = df[df['Parent'] == parentgender].copy()
    df = df[df['IPA'].isin(target_vowels_IPA)].copy()
    df.reset_index(drop=True, inplace=True)

    # Generate all unique pairs of IPA categories
    ipa_pairs = list(combinations(target_vowels_IPA, 2))
    #total_pairs = len(ipa_pairs)
    
    results = []

    # Group by 'spkid' and 'AgeMonth' to compute the Pillai score using MANOVA test
    for (spkid, age), group in df.groupby(['spkid', 'AgeMonth']):
        #print(spkid, age)
        feature_values = group[feature_column_names].values
        #print(feature_values.shape)
        
        if len(group) < 2:
            continue
        
        # Standardizing the feature data
        #scaler = StandardScaler()
        #scaled_features = scaler.fit_transform(feature_values)
        
        # Prepare the data for MANOVA
        df_scaled = pd.DataFrame(feature_values, columns=feature_column_names)
        df_scaled['IPA'] = group['IPA'].values

        # Compute Pillai score for each pair
        all_pillai_scores = []
        for ipa1, ipa2 in ipa_pairs:
            try:
                subset = df_scaled[df_scaled['IPA'].isin([ipa1, ipa2])]
                
                if subset.shape[0] < 2:
                    continue  # Skip if there are not enough samples for the pair
                
                formula = f"{'+'.join(feature_column_names)} ~ C(IPA)"
                #formula = f"{' + '.join([f'`{col}`' for col in feature_column_names])} ~ C(IPA)"
                manova = MANOVA.from_formula(formula, data=subset)
                pillai_score = manova.mv_test().results['C(IPA)']['stat']['Value']['Pillai\'s trace']
                p_value = manova.mv_test().results['C(IPA)']['stat']['Pr > F']['Pillai\'s trace']
                if p_value < 0.05:
                    all_pillai_scores.append(pillai_score)
                else:
                    all_pillai_scores.append(np.nan)
            except Exception as e:
                print(f"MANOVA failed for spkid={spkid} at AgeMonth={age} for pair ({ipa1}, {ipa2}): {e}")
                all_pillai_scores.append(np.nan)

        # Compute the average Pillai score
        if all_pillai_scores:
            array = np.array(all_pillai_scores)
            no_pairs = np.sum(~np.isnan(array))
            average_pillai_score = np.nanmean(all_pillai_scores)
            results.append({'spkid': spkid, 'AgeMonth': age, 'Register': register, 'separability': average_pillai_score, 'Nopairs': no_pairs})

        else:
            average_pillai_score = np.nan  # Handle the case where no pairs were found
            results.append({'spkid': spkid, 'AgeMonth': age, 'Register': register, 'separability': average_pillai_score, 'Nopairs': np.nan})

    # Create results DataFrame
    # Convert results to DataFrame and save to Excel
    results_df = pd.DataFrame(results)
    results_df.dropna(inplace=True)
    
    if parentgender:
        results_df.to_excel(out_file_path + '_' + parentgender + '.xlsx', index=False)
    else:
        results_df.to_excel(out_file_path + '.xlsx', index=False)
    
    print("Pillai scores have been saved")

