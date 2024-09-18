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
warnings.filterwarnings("ignore", category=FutureWarning)

def vowel_space_expansion(in_dir_path, feature_name, feature_column_names, out_dir_path):
    """
    Computes the vowel space expansion (convex hull area) for specific vowels based on 
    acoustic features and saves the results to an Excel file. The function processes each CSV file 
    in the input directory and calculates the convex hull area for formant features (F1, F2) or 
    for MFCC features after dimensionality reduction using PCA if needed.

    The input CSV file names should follow the format:
    `featurename_language_register_numberofvowels_anythingnotimportant.csv`.
    For example: `formant_no_ADS_6_data.csv` for formant features of Norwegian ADS data with 6 vowels.

    For each speaker (`spkid`) and age group (`AgeMonth`), the function computes the mean feature values 
    for each vowel (e.g., using F1, F2 for formants or PCA1 and PCA2 for MFCCs). It then calculates the convex hull area 
    using these acoustic features to represent vowel space expansion.

    Parameters:
    -----------
    in_dir_path : str
        The directory containing input CSV files with vowel data. Each file must contain columns for 
        'spkid', 'AgeMonth', 'IPA', and the specified acoustic features in `feature_column_names`.

    feature_name : str
        The name of the feature type (e.g., 'formant' or 'mfcc') used to filter input files. 
        The input CSV filenames should start with this feature name.

    feature_column_names : list of str
        The list of column names corresponding to the acoustic features (e.g., ['F1', 'F2'] for formant analysis, 
        or ['MFCC1', 'MFCC2', ..., 'MFCC13'] for MFCC analysis).

    out_dir_path : str 
        The directory where the output Excel files with the convex hull areas will be saved. 
        The results will be saved with the same base name as the input CSV file but with a `.xlsx` extension.

    Returns:
    --------
    None
        The function saves an Excel file for each input CSV file, containing the computed convex hull areas 
        for each speaker and age group.

    Notes:
    ------
    - The input CSV files must include columns for 'spkid', 'AgeMonth', 'IPA', and the specified acoustic features.
    - The convex hull area (or volume for higher-dimensional data) is calculated for each speaker and age group 
      based on the mean feature values of the vowels.
    - If the feature data is higher than 2 dimensions (e.g., MFCCs), PCA is applied to reduce the data to two dimensions 
      before calculating the convex hull area.
    - The function skips any group with fewer than 3 vowel points (since at least 3 points are required to form a convex hull).
    - The output DataFrame saved to the Excel file includes the columns 'spkid', 'AgeMonth', 'Language', 'Register', 
      and 'ConvexHullArea'.
    
    Example:
    --------
    # For formant analysis:
    in_dir_path = "path/to/input"
    feature_name = 'formant'
    feature_column_names = ['F1', 'F2']
    out_dir_path = "path/to/output"
    vowel_space_expansion(in_dir_path, feature_name, feature_column_names, out_dir_path)

    # For MFCC analysis (13 features):
    feature_name = 'mfcc'
    feature_column_names = [f'MFCC{i}' for i in range(1, 14)]
    vowel_space_expansion(in_dir_path, feature_name, feature_column_names, out_dir_path)
    """
    for file in os.listdir(in_dir_path):
        if file.split('.csv')[0].split('_')[0] != feature_name:
            print('file name should start with featuer_name')
            continue

        df = pd.read_csv(os.path.join(in_dir_path, file))
        #df = df[df['IPA'].isin(target_vowels_IPA)].copy()
        #df.reset_index(inplace=True)
        
    
        # Group by 'spkid', 'AgeMonth', and 'IPA' to compute the mean feature values for each vowel
        #mean_features = df.groupby(['spkid', 'AgeMonth', 'IPA'])[feature_column_names].mean().reset_index()
    
        # Group by 'spkid' and 'AgeMonth' to compute the convex hull area
        results = []
        for (spkid, age), group in df.groupby(['spkid', 'AgeMonth']):
            
                if group['IPA'].nunique() != int(file.split('.csv')[0].split('_')[3]):
                    print(f"Skipping speaker {spkid}, age {age} due to missing classes.")
                    continue
                #mean_features = group.groupby(['spkid', 'AgeMonth', 'IPA'])[feature_column_names].mean().reset_index()
                #feature_values = mean_features[feature_column_names].values
            #if feature_values.shape[0] > feature_values.shape[1]:  # Check if there are at least 3 points
                try:
                    if len(feature_column_names) == 2:
                        mean_features = group.groupby(['spkid', 'AgeMonth', 'IPA'])[feature_column_names].mean().reset_index()
                        feature_values = mean_features[feature_column_names].values
                        area = ConvexHull(feature_values).volume
                        results.append({'spkid': spkid, 'AgeMonth': age, 'ConvexHullArea': round(area)})
            #elif feature_values.shape[0] <= feature_values.shape[1] and feature_values.shape[1] > 2 and feature_values.shape[0] > 2:
                    if len(feature_column_names) > 2:
                #pca = PCA(n_components=feature_values.shape[0]-1)
                        pca = PCA(n_components=2)
                        feature_values = pca.fit_transform(group[feature_column_names].values)
                        group[['PCA1', 'PCA2']] = feature_values
                        mean_features = group.groupby(['spkid', 'AgeMonth', 'IPA'])[['PCA1', 'PCA2']].mean().reset_index()
                        feature_values = mean_features[['PCA1', 'PCA2']].values
                        area = ConvexHull(feature_values).volume
                        results.append({'spkid': spkid, 'AgeMonth': age, 'ConvexHullArea': round(area)})
                except Exception as e:
                    print(f"Error during Creating ConvexHull for speaker {spkid}, age {age}: {e}")
                    continue
                #results.append({'spkid': spkid, 'AgeMonth': age, 'ConvexHullArea': np.nan})
    
        # Convert results to DataFrame and save to Excel
        results_df = pd.DataFrame(results)
        results_df.dropna(inplace=True)
        results_df['Language'] = file.split('.csv')[0].split('_')[1]
        results_df['Register'] = file.split('.csv')[0].split('_')[2]
        results_df.to_excel(os.path.join(out_dir_path, file.split('.csv')[0] + '.xlsx'), index=False)
    
        print("Convex hull areas have been saved")


def vowel_variability(in_dir_path, feature_name, feature_column_names, out_dir_path):
    """
    Computes vowel space variability (convex hull area or volume) for specific vowels using acoustic features 
    (formants or PCA(2) of MFCCs) with the data points that are within one standard deviation from the mean. 
    For MFCC features, PCA is applied to reduce dimensionality to two, and then for both features (formants and PCA of MFCCs) convex hulls are computed. 
    The results are saved to an Excel file for each input CSV file.

    The input CSV file names should follow the format:
    `featurename_language_register_numberofvowels_anythingnotimportant.csv`.
    For example: `formant_no_ADS_6_data.csv` for formant features of Norwegian ADS data with 6 vowels.

    The function filters data for each speaker, age, and vowel combination based on acoustic feature values, 
    retains only points within one standard deviation from the mean, and calculates the convex hull area (for 2D formants) 
    or (for MFCC features projected onto 2D space) to compute compactness as variability measures. 

    Parameters:
    -----------
    in_dir_path : str
        The directory containing input CSV files with vowel data. The files must contain columns for 'spkid', 'AgeMonth', 
        'IPA', and the acoustic features specified in `feature_column_names`.

    feature_name : str
        The name of the feature type (e.g., 'formant' or 'mfcc') used to filter input files. The input CSV filenames should 
        start with this feature name.

    feature_column_names : list of str
        The list of column names corresponding to the acoustic features (e.g., ['F1', 'F2'] for formant analysis, 
        or ['MFCC1', 'MFCC2', ..., 'MFCC13'] for MFCC analysis).

    out_dir_path : str
        The directory where the output Excel files will be saved. The results will be saved in Excel files with the same 
        base name as the input CSV file, but with a `.xlsx` extension.

    Returns:
    --------
    None
        The function saves an Excel file containing the computed vowel space variability for each speaker, age, and vowel 
        to the specified `out_dir_path`.

    Notes:
    ------
    - The function groups data by 'spkid', 'AgeMonth', and 'IPA' and filters points to include only those within one standard 
      deviation from the mean of the specified acoustic features (formants or PCA of MFCCs).
    - For formant analysis (2D), the convex hull area is computed, while for MFCC features, PCA is used to project data onto 
      two dimensions, and then convex hull area is computed based on the transformed data.
    - The output DataFrame saved in Excel includes columns for 'spkid', 'AgeMonth', 'IPA', 'Language', 'Register', '#Samples', 
      and 'variability' (the computed vowel space variability).
    - Files are processed one by one, and errors during computation for specific groups (e.g., due to insufficient data) 
      are logged and skipped.
    
    Example:
    --------
    # For formant analysis:
    in_dir_path = "path/to/input"
    feature_name = 'formant'
    feature_column_names = ['F1', 'F2']
    out_dir_path = "path/to/output"
    vowel_variability(in_dir_path, feature_name, feature_column_names, out_dir_path)

    # For MFCC analysis (13 features projected to 2D using PCA):
    feature_name = 'mfcc'
    feature_column_names = [f'MFCC{i}' for i in range(1, 14)]
    vowel_variability(in_dir_path, feature_name, feature_column_names, out_dir_path)
    """
    
    for file in os.listdir(in_dir_path):
        if file.split('.csv')[0].split('_')[0] != feature_name:
            print('file name should start with featuer_name')
            continue
        df = pd.read_csv(os.path.join(in_dir_path, file))
        results = []
        for (spkid, age, vowel), group in df.groupby(['spkid', 'AgeMonth', 'IPA']):
            try:
                if len(feature_column_names) == 2:
                    #compute the mean and std for each group
                    mean_features = group[feature_column_names].mean()
                    std_features = group[feature_column_names].std()
                    #Filter data to include only points within one standard deviation from the mean
                    condition = True
                    for feature in feature_column_names:
                        condition &= (group[feature] >= (mean_features[feature] - std_features[feature])) & \
                            (group[feature] <= (mean_features[feature] + std_features[feature]))

                    filtered_group = group[condition]
                    feature_values = filtered_group[feature_column_names].values
                    if len(feature_values) < 3:
                        print(f"Skipping speaker {spkid}, age {age} due to less than 3 samples.")
                        continue
                    area = ConvexHull(feature_values).volume
                    results.append({'spkid': spkid, 'AgeMonth': age,'IPA': vowel, '#Samples': len(feature_values), 'variability': round(area)})
            #elif feature_values.shape[0] <= feature_values.shape[1] and feature_values.shape[1] > 2 and feature_values.shape[0] > 2:
                if len(feature_column_names) > 2:
                        #pca = PCA(n_components=feature_values.shape[0]-1)
                    pca = PCA(n_components=2)
                    #feature_values = pca.fit_transform(feature_values)
                    feature_values = pca.fit_transform(group[feature_column_names].values)
                    group[['PCA1', 'PCA2']] = feature_values
                    #print(f'group df after PCA {group}')
                    #compute the mean and std for each group
                    mean_features = group[['PCA1', 'PCA2']].mean()
                    std_features = group[['PCA1', 'PCA2']].std()
                    #Filter data to include only points within one standard deviation from the mean
                    condition = True
                    for feature in ['PCA1', 'PCA2']:
                        condition &= (group[feature] >= (mean_features[feature] - std_features[feature])) & \
                                (group[feature] <= (mean_features[feature] + std_features[feature]))

                    filtered_group = group[condition]
                    #print(f'filtered group df {filtered_group}')
                    feature_values = filtered_group[['PCA1', 'PCA2']].values
                    if len(feature_values) < 3:
                        print(f"Skipping vowel {vowel} for speaker {spkid}, age {age} due to less than 3 samples.")
                        continue
                    area = ConvexHull(feature_values).volume
                    results.append({'spkid': spkid, 'AgeMonth': age,'IPA': vowel, '#Samples': len(feature_values),'variability': round(area)})
            except Exception as e:
                print(f"Error during Creating ConvexHull for speaker {spkid}, age {age}: {e}")
                continue
            #mean_feature_values = np.mean(feature_values, axis=0).reshape(1, -1) # mean of each feature
            #eq_dist = cdist(feature_values, mean_feature_values, metric='euclidean').flatten()
            #sum_eq_dist = np.sum(eq_dist) # Sum the distances
            #average_sum_distance = sum_eq_dist / len(feature_values) # average sum of distances
            #results.append({'spkid': spkid, 'AgeMonth': age,'IPA': vowel, '#Samples': len(feature_values),'variability': round(average_sum_distance)})
        # Convert results to DataFrame and save to Excel
        results_df = pd.DataFrame(results)
        results_df.dropna(inplace=True)
        results_df['Language'] = file.split('.csv')[0].split('_')[1]
        results_df['Register'] = file.split('.csv')[0].split('_')[2]
        results_df.to_excel(os.path.join(out_dir_path, file.split('.csv')[0] + '.xlsx'), index=False)

        print("Variabilities have been saved")



def vowel_separability(in_dir_path, feature_name, feature_column_names, out_dir_path):
    """
    Computes vowel separability based on acoustic features (formants or MFCCs) using Pillai's trace 
    from the MANOVA test for different pairs of vowels. The function processes each CSV file in the 
    input directory and computes the average Pillai score for vowel pairs, representing how separable 
    the vowels are within the acoustic space.

    The input CSV file names should follow the format:
    `featurename_language_register_numberofvowels_anythingnotimportant.csv`.
    For example: `formant_no_ADS_6_data.csv` for formant features of Norwegian ADS data with 6 vowels.

    For each speaker (`spkid`) and age group (`AgeMonth`), the function computes the MANOVA Pillai's trace 
    score for all unique vowel pairs and averages the scores to represent the overall vowel separability 
    for the speaker and age group.

    Parameters:
    -----------
    in_dir_path : str
        The directory containing input CSV files with vowel data. Each file must contain columns for 'spkid', 
        'AgeMonth', 'IPA', and the specified acoustic features in `feature_column_names`.

    feature_name : str
        The name of the feature type (e.g., 'formant' or 'mfcc') used to filter input files. 
        The input CSV filenames should start with this feature name.

    feature_column_names : list of str
        The list of column names corresponding to the acoustic features (e.g., ['F1', 'F2'] for formant analysis, 
        or ['MFCC1', 'MFCC2', ..., 'MFCC13'] for MFCC analysis).

    out_dir_path : str
        The directory where the output Excel files with the computed Pillai scores will be saved. 
        The results will be saved with the same base name as the input CSV file but with a `.xlsx` extension.

    Returns:
    --------
    None
        The function saves an Excel file containing the computed vowel separability (Pillai score) for each speaker, 
        age group, and vowel pair to the specified `out_dir_path`.

    Notes:
    ------
    - The function groups data by 'spkid', 'AgeMonth', and 'IPA' and generates all unique pairs of IPA categories 
      (vowels). For each vowel pair, it performs a MANOVA test and computes Pillai's trace score to represent 
      vowel separability.
    - The Pillai scores for each vowel pair are averaged to provide an overall vowel separability measure for each 
      speaker and age group.
    - The output DataFrame saved in Excel includes columns for 'spkid', 'AgeMonth', 'Language', 'Register', 
      '#Pairs', and 'separability' (the average Pillai score).
    - Files are processed one by one, and errors during computation for specific groups (e.g., due to insufficient data) 
      are logged and skipped.
    - If there are no valid vowel pairs for a speaker and age group, the separability score will be set to `NaN`.

    Example:
    --------
    # For formant analysis:
    in_dir_path = "path/to/input"
    feature_name = 'formant'
    feature_column_names = ['F1', 'F2']
    out_dir_path = "path/to/output"
    vowel_separability(in_dir_path, feature_name, feature_column_names, out_dir_path)

    # For MFCC analysis (13 features):
    feature_name = 'mfcc'
    feature_column_names = [f'MFCC{i}' for i in range(1, 14)]
    vowel_separability(in_dir_path, feature_name, feature_column_names, out_dir_path)
    """
    # Load data
    for file in os.listdir(in_dir_path):
        if file.split('.csv')[0].split('_')[0] != feature_name:
            print('file name should start with featuer_name')
            continue
        df = pd.read_csv(os.path.join(in_dir_path, file))
        # Filter data based on target_vowels_IPA
        #df = df[df['IPA'].isin(target_vowels_IPA)].copy()
        #df.reset_index(drop=True, inplace=True)

        # Generate all unique pairs of IPA categories
        ipa_pairs = list(combinations(df['IPA'].unique().tolist(), 2))
        #total_pairs = len(ipa_pairs)
    
        results = []

        # Group by 'spkid' and 'AgeMonth' to compute the Pillai score using MANOVA test
        for (spkid, age), group in df.groupby(['spkid', 'AgeMonth']):
            #print(spkid, age)
            if group['IPA'].nunique() != int(file.split('.csv')[0].split('_')[3]):
                print(f"Skipping speaker {spkid}, age {age} due to missing classes.")
                continue
            if len(group) < 2:
                continue

            feature_values = group[feature_column_names].values
            scaler = StandardScaler()

            if len(feature_column_names) > 2:
                
                pca = PCA(n_components=2)
                feature_values = pca.fit_transform(feature_values)
                scaled_features = scaler.fit_transform(feature_values)
                pca_column_names = ['PCA1', 'PCA2']
                df_scaled = pd.DataFrame(scaled_features, columns=pca_column_names)
                df_scaled['IPA'] = group['IPA'].values
            # Standardizing the feature data
            else:
                scaled_features = scaler.fit_transform(feature_values)
        
                # Prepare the data for MANOVA
                df_scaled = pd.DataFrame(scaled_features, columns=feature_column_names)
                df_scaled['IPA'] = group['IPA'].values

            # Compute Pillai score for each pair
            all_pillai_scores = []
            for ipa1, ipa2 in ipa_pairs:
                try:
                    subset = df_scaled[df_scaled['IPA'].isin([ipa1, ipa2])]
                
                    if subset.shape[0] < 2:
                        continue  # Skip if there are not enough samples for the pair
                    if feature_name == 'formant':
                        formula = f"{'+'.join(feature_column_names)} ~ C(IPA)"
                    else:
                        formula = f"{'+'.join(pca_column_names)} ~ C(IPA)"
                    #formula = f"{' + '.join([f'`{col}`' for col in feature_column_names])} ~ C(IPA)"
                    manova = MANOVA.from_formula(formula, data=subset)
                    pillai_score = manova.mv_test().results['C(IPA)']['stat']['Value']['Pillai\'s trace']
                    #p_value = manova.mv_test().results['C(IPA)']['stat']['Pr > F']['Pillai\'s trace']
                    all_pillai_scores.append(pillai_score)
                    #if p_value < 0.05:
                        #all_pillai_scores.append(pillai_score)
                    #else:
                        #all_pillai_scores.append(np.nan)
                except Exception as e:
                    #print(f"MANOVA failed for spkid={spkid} at AgeMonth={age} for pair ({ipa1}, {ipa2}): {e}")
                    all_pillai_scores.append(np.nan)

            # Compute the average Pillai score
            if all_pillai_scores:
                array = np.array(all_pillai_scores)
                no_pairs = np.sum(~np.isnan(array))
                average_pillai_score = np.nanmean(all_pillai_scores)
                results.append({'spkid': spkid, 'AgeMonth': age, 'separability': average_pillai_score, 'Nopairs': no_pairs})

            else:
                average_pillai_score = np.nan  # Handle the case where no pairs were found
                results.append({'spkid': spkid, 'AgeMonth': age, 'separability': average_pillai_score, 'Nopairs': np.nan})

        # Create results DataFrame
        # Convert results to DataFrame and save to Excel
        results_df = pd.DataFrame(results)
        results_df.dropna(inplace=True)
        results_df['Language'] = file.split('.csv')[0].split('_')[1]
        results_df['Register'] = file.split('.csv')[0].split('_')[2]
        results_df.to_excel(os.path.join(out_dir_path, file.split('.csv')[0] + '.xlsx'), index=False)
    
        print("Pillai scores have been saved")

