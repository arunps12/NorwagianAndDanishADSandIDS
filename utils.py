import os
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import ConvexHull
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import ConvexHull
import numpy as np
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from itertools import combinations
from statsmodels.multivariate.manova import MANOVA


def create_dir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)




def plot_convex_hull(df, spkid, age_month, feature_column_names, no_vowels, feature_type='formant', save_dir = None):
    """
    Plots the convex hull for a particular speaker and age group. Displays the convex hull based on formants (F1, F2)
    or PCA components (PCA1, PCA2) if using MFCC features.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing the data for the speaker with columns ['spkid', 'AgeMonth', 'IPA', feature_column_names].

    spkid : str or int
        The speaker ID for which the convex hull needs to be plotted.

    age_month : int
        The age in months for which the convex hull needs to be plotted.

    feature_column_names : list of str
        List of column names for the acoustic features (e.g., ['F1', 'F2'] for formants or ['PCA1', 'PCA2'] for PCA).

    feature_type : str, optional
        Either 'formant' or 'mfcc'. Determines the x and y axis labels.
    
    Returns:
    --------
    None
        The function plots and shows the convex hull with all data points for that speaker and age group.
    """
    
    # Filter data for the specific speaker and age month
    speaker_data = df[(df['spkid'] == spkid) & (df['AgeMonth'] == age_month)].copy()

    if speaker_data.empty:
        print(f"No data found for speaker {spkid} and age month {age_month}")
        return
    if speaker_data['IPA'].nunique() != no_vowels:
        print(f"speaker {spkid}, age {age_month} have missing classes. Try another spkid or age")
        return
    
    # Get the mean feature values for each vowel category (IPA)
    if len(feature_column_names) == 2: 
        mean_features = speaker_data.groupby('IPA')[feature_column_names].mean().reset_index()
    elif len(feature_column_names) > 2:
        pca = PCA(n_components=2)
        speaker_data.loc[:, ['PCA1', 'PCA2']] = pca.fit_transform(speaker_data[feature_column_names].values)
        mean_features = speaker_data.groupby('IPA')[['PCA1', 'PCA2']].mean().reset_index()

    # Plot all the data points (with transparency)
    plt.figure(figsize=(8, 6))
    # Define a color map 
    colors = {'a': 'blue', 'i': 'green', 'u': 'red'} 
    for vowel, group in speaker_data.groupby('IPA'):
        if len(feature_column_names) == 2:
            plt.scatter(group[feature_column_names[1]], group[feature_column_names[0]], 
                        color=colors[vowel], alpha=0.3)#, label=f'{vowel}: Data Points')
        else:
            plt.scatter(group['PCA2'], group['PCA1'], color=colors[vowel], alpha=0.3)#, label=f'{vowel}: Data Points')

    # Plot mean points for each vowel category
    for vowel, group in mean_features.groupby('IPA'):
        if len(feature_column_names) == 2:
            plt.scatter(group[feature_column_names[1]], group[feature_column_names[0]], 
                    color=colors[vowel], s=100, label=f'{vowel}:Mean', zorder=5)
        else:
            plt.scatter(group['PCA2'], group['PCA1'], color=colors[vowel], s=100, label=f'{vowel}:Mean', zorder=5)
    
        # Label mean points with their IPA category
    for _, row in mean_features.iterrows():
        if len(feature_column_names) == 2:
            plt.text(row[feature_column_names[1]]+20, row[feature_column_names[0]]+20, row['IPA'], 
                    fontsize=12, color=colors[row['IPA']], ha='right', va='bottom', zorder=6)
        else:
            plt.text(row['PCA2']+5, row['PCA1']+5, row['IPA'], fontsize=12, color=colors[row['IPA']], ha='right', va='bottom', zorder=6)

    # Create Convex Hull around the mean feature points
    try:
        if len(feature_column_names) == 2:
            hull = ConvexHull(mean_features[feature_column_names].values)
        else:
            hull = ConvexHull(mean_features[['PCA1', 'PCA2']].values)
            
        for simplex in hull.simplices:
            if len(feature_column_names) == 2:
                plt.plot(mean_features.iloc[simplex][feature_column_names[1]], 
                         mean_features.iloc[simplex][feature_column_names[0]], 'k-', label='_nolegend_')
            else:
                plt.plot(mean_features.iloc[simplex]['PCA2'], 
                         mean_features.iloc[simplex]['PCA1'], 'k-', label='_nolegend_')
                
        if len(feature_column_names) == 2:
            plt.fill(mean_features.iloc[hull.vertices][feature_column_names[1]], 
                     mean_features.iloc[hull.vertices][feature_column_names[0]], 
                     'gray', alpha=0.2, label='Convex Hull')
        else:
            plt.fill(mean_features.iloc[hull.vertices]['PCA2'], 
                     mean_features.iloc[hull.vertices]['PCA1'], 
                     'gray', alpha=0.2, label='Convex Hull')
    except Exception as e:
        print(f"Error creating Convex Hull: {e}")

    # Axis labels
    if feature_type == 'formant':
        plt.xlabel('F2 (Hz)')
        plt.ylabel('F1 (Hz)')
    elif feature_type == 'mfcc':
        plt.xlabel('PCA2')
        plt.ylabel('PCA1')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.tick_top()
    plt.gca().yaxis.set_label_position('right')
    plt.gca().yaxis.tick_right()
    
    plt.legend()
    plt.grid(True)

    if save_dir:
        path = 'VowelSpaceArea' + '_' + feature_type + '_' + str(no_vowels) + '_' + spkid + '_' + str(age_month) + '.pdf'
        save_path = os.path.join(save_dir, path)
        plt.savefig(save_path, format='pdf')
        path_jpg = 'VowelSpaceArea' + '_' + feature_type + '_' + str(no_vowels) + '_' + spkid + '_' + str(age_month) + '.jpg'
        save_path_jpg = os.path.join(save_dir, path_jpg)
        plt.savefig(save_path_jpg, format='jpg')
        print(f"Plot saved as {save_path}")
    plt.show()

def plot_variability(df, spkid, age_month, feature_column_names,no_vowels, feature_type='formant', save_dir=None):
    """
    Plots the vowel variability for a particular speaker and age group, showing points within one standard deviation
    with lower opacity and the convex hull for each vowel category (IPA) with higher opacity.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing the data for the speaker with columns ['spkid', 'AgeMonth', 'IPA', feature_column_names].

    spkid : str or int
        The speaker ID for which the variability needs to be plotted.

    age_month : int
        The age in months for which the variability needs to be plotted.

    feature_column_names : list of str
        List of column names for the acoustic features (e.g., ['F1', 'F2'] for formants or ['PCA1', 'PCA2'] for PCA).

    feature_type : str, optional
        Either 'formant' or 'mfcc'. Determines the x and y axis labels.

    save_path : str, optional
        File path to save the PDF. If not provided, the plot will just be shown but not saved.

    Returns:
    --------
    None
        The function plots and optionally saves the vowel variability for each category for that speaker and age group.
    """
    
    # Filter data for the specific speaker and age month
    speaker_data = df[(df['spkid'] == spkid) & (df['AgeMonth'] == age_month)].copy()

    if speaker_data.empty:
        print(f"No data found for speaker {spkid} and age month {age_month}")
        return
    if speaker_data['IPA'].nunique() != no_vowels:
        print(f"speaker {spkid}, age {age_month} have missing classes. Try another spkid or age")
        return

    # Apply PCA if necessary for MFCCs
    if len(feature_column_names) > 2:
        pca = PCA(n_components=2)
        speaker_data.loc[:, ['PCA1', 'PCA2']] = pca.fit_transform(speaker_data[feature_column_names].values)
        feature_column_names = ['PCA1', 'PCA2']

    plt.figure(figsize=(8, 6))

    # For each vowel category (IPA), plot convex hull and points within one standard deviation
    for vowel, group in speaker_data.groupby('IPA'):
        # Compute the mean and std for the group
        mean_features = group[feature_column_names].mean()
        std_features = group[feature_column_names].std()
        
        # Filter data to include only points within one standard deviation from the mean
        condition = True
        for feature in feature_column_names:
            condition &= (group[feature] >= (mean_features[feature] - std_features[feature])) & \
                         (group[feature] <= (mean_features[feature] + std_features[feature]))
        filtered_group = group[condition]
        
        # Define a color map 
        colors = {'a': 'blue', 'i': 'green', 'u': 'red'} 

        # Plot all data points with transparency (points outside 1 std dev)
        plt.scatter(group[feature_column_names[1]], group[feature_column_names[0]], 
            color=colors[vowel], alpha=0.3)#, label=f'{vowel}: Points outside 1 std dev')

        # Highlight the filtered points with higher opacity using the same color
        plt.scatter(filtered_group[feature_column_names[1]], filtered_group[feature_column_names[0]], 
            color=colors[vowel], alpha=0.8)#, label=f'{vowel}: Points within 1 std dev', zorder=5)
        # Compute and plot the convex hull around the filtered points
        if len(filtered_group) >= 3:
            feature_values = filtered_group[feature_column_names].values
            hull = ConvexHull(feature_values)
            for simplex in hull.simplices:
                plt.plot(feature_values[simplex, 1], feature_values[simplex, 0], 'k-', zorder=6)
            plt.fill(feature_values[hull.vertices, 1], feature_values[hull.vertices, 0], 
                     color=colors[vowel],alpha=0.5, label=f'{vowel}: Convex Hull with data points within 1 std dev', zorder=6)
            
            # Label the convex hull points with their IPA category
            mean_hull_point = filtered_group[feature_column_names].mean()
            plt.text(mean_hull_point[1], mean_hull_point[0], vowel, fontsize=12, color=colors[vowel], ha='right', va='bottom', zorder=7)

    # Axis labels
    if feature_type == 'formant':
        plt.xlabel('F2 (Hz)')
        plt.ylabel('F1 (Hz)')
    elif feature_type == 'mfcc':
        plt.xlabel('PCA2')
        plt.ylabel('PCA1')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.gca().xaxis.set_label_position('top')
    plt.gca().xaxis.tick_top()
    plt.gca().yaxis.set_label_position('right')
    plt.gca().yaxis.tick_right()
    plt.legend()
    plt.grid(True)

    # Save plot as PDF if save_path is provided
    if save_dir:
        path = 'Variability' + '_' + feature_type + '_' + str(no_vowels) + '_' + spkid + '_' + str(age_month) + '.pdf'
        save_path = os.path.join(save_dir, path)
        plt.savefig(save_path, format='pdf')
        path_jpg = 'Variability' + '_' + feature_type + '_' + str(no_vowels) + '_' + spkid + '_' + str(age_month) + '.jpg'
        save_path_jpg = os.path.join(save_dir, path_jpg)
        plt.savefig(save_path_jpg, format='jpg')
        print(f"Plot saved as {save_path}")

    # Show the plot
    plt.show()

def plot_vowel_separability(df, spkid, age_month, feature_column_names, no_vowels, feature_type='formant', save_dir=None):
    """
    Computes the vowel separability (Pillai scores) based on MANOVA for all vowel pairs for a particular speaker.
    Then plots the vowel pairs with their Pillai score and corresponding data points.

    Parameters:
    -----------
    df : pandas.DataFrame
        Dataframe containing the data for the speaker with columns ['spkid', 'AgeMonth', 'IPA', feature_column_names].

    spkid : str or int
        The speaker ID for which the vowel separability needs to be computed and plotted.

    age_month : int
        The age in months for which the vowel separability needs to be computed and plotted.

    feature_column_names : list of str
        List of column names for the acoustic features (e.g., ['F1', 'F2'] for formants or ['MFCC1', 'MFCC2'] for MFCC analysis).

    no_vowel_pairs : int, optional
        Maximum number of vowel pairs per row in the plot. Default is 3.

    Returns:
    --------
    None
        The function computes and plots the vowel separability for the given speaker and age month.
    """

    # Filter data for the specific speaker and age month
    speaker_data = df[(df['spkid'] == spkid) & (df['AgeMonth'] == age_month)]

    if speaker_data.empty:
        print(f"No data found for speaker {spkid} and age month {age_month}")
        return
    if speaker_data['IPA'].nunique() != no_vowels:
        print(f"speaker {spkid}, age {age_month} have missing classes. Try another spkid or age")
        return

    # Generate all unique vowel pairs (IPA)
    ipa_pairs = list(combinations(speaker_data['IPA'].unique().tolist(), 2))

    # Prepare the data for MANOVA, scaling and PCA if needed
    feature_values = speaker_data[feature_column_names].values
    scaler = StandardScaler()

    if len(feature_column_names) > 2:
        pca = PCA(n_components=2)
        feature_values = pca.fit_transform(feature_values)
        speaker_data.loc[:, ['PCA1', 'PCA2']] = pca.fit_transform(speaker_data[feature_column_names].values)
        scaled_features = scaler.fit_transform(feature_values)
        pca_column_names = ['PCA1', 'PCA2']
        df_scaled = pd.DataFrame(scaled_features, columns=pca_column_names)
        df_scaled['IPA'] = speaker_data['IPA'].values
        columns_for_manova = pca_column_names
    else:
        scaled_features = scaler.fit_transform(feature_values)
        df_scaled = pd.DataFrame(scaled_features, columns=feature_column_names)
        df_scaled['IPA'] = speaker_data['IPA'].values
        columns_for_manova = feature_column_names

    # Compute Pillai scores for each vowel pair
    pillai_scores = []
    for ipa1, ipa2 in ipa_pairs:
        try:
            subset = df_scaled[df_scaled['IPA'].isin([ipa1, ipa2])]
            if subset.shape[0] < 2:
                pillai_scores.append(np.nan)
                continue

            # Run MANOVA
            formula = f"{'+'.join(columns_for_manova)} ~ C(IPA)"
            manova = MANOVA.from_formula(formula, data=subset)
            pillai_score = manova.mv_test().results['C(IPA)']['stat']['Value']['Pillai\'s trace']
            pillai_scores.append(pillai_score)
        except Exception as e:
            print(f"MANOVA failed for {ipa1} vs {ipa2}: {e}")
            pillai_scores.append(np.nan)

    # Plot settings
    num_pairs = len(ipa_pairs)
    rows = (num_pairs // no_vowels) + int(num_pairs % no_vowels > 0)
    if len(feature_column_names) == 2:
        x_min = speaker_data[feature_column_names[1]].min()
        x_max = speaker_data[feature_column_names[1]].max()
        y_min = speaker_data[feature_column_names[0]].min()
        y_max = speaker_data[feature_column_names[0]].max()
    else:
        x_min = speaker_data['PCA2'].min()
        x_max = speaker_data['PCA2'].max()
        y_min = speaker_data['PCA1'].min()
        y_max = speaker_data['PCA1'].max()


    # Create the figure and axes objects
    fig, axes = plt.subplots(rows, no_vowels, figsize=(15, 5 * rows))

    for i, (pair, pillai) in enumerate(zip(ipa_pairs, pillai_scores)):
        ipa1, ipa2 = pair
        # Determine the subplot location (row, column)
        ax = axes.flat[i]  # Use flat indexing if axes are in multiple rows

        # Filter data for the two vowels in the pair
        subset = speaker_data[speaker_data['IPA'].isin([ipa1, ipa2])]
        
        
        # Define a color map for each vowel
        #colors = {ipa1: 'blue', ipa2: 'green'}
        colors = {'a': 'blue', 'i': 'green', 'u': 'red'} 
        
        # Plot data points for each vowel separately with their respective colors
        for ipa in [ipa1, ipa2]:
            vowel_subset = subset[subset['IPA'] == ipa]
            if len(feature_column_names) == 2:  # For formants (F1, F2)
                ax.scatter(vowel_subset[feature_column_names[1]], vowel_subset[feature_column_names[0]], 
                           color=colors[ipa], alpha=0.8, label=f'{ipa}')
                
                # Add text label for each vowel category near the mean position
                mean_x = vowel_subset[feature_column_names[1]].mean()
                mean_y = vowel_subset[feature_column_names[0]].mean()
                ax.text(mean_x+10, mean_y+10, ipa, fontsize=12, color=colors[ipa], ha='right', va='bottom')
                
            else:  # For MFCC (PCA1, PCA2)
                ax.scatter(vowel_subset['PCA2'], vowel_subset['PCA1'], 
                           color=colors[ipa], alpha=0.8, label=f'{ipa}')
                
                # Add text label for each vowel category near the mean position
                mean_x = vowel_subset['PCA2'].mean()
                mean_y = vowel_subset['PCA1'].mean()
                ax.text(mean_x+5, mean_y+5, ipa, fontsize=12, color=colors[ipa], ha='right', va='bottom')

        # Set the Pillai score as the title of each subplot
        ax.set_title(f'Pillai Score: {pillai:.2f}', fontsize=12)
        
        
        # # Set the same x and y limits for all subplots and Axis labels 
        if feature_type == 'formant':
            ax.set_xlim(x_min-100, x_max+100)
            ax.set_ylim(y_min-100, y_max+100)
            ax.set_xlabel('F2 (Hz)')
            ax.set_ylabel('F1 (Hz)')
        elif feature_type == 'mfcc':
            ax.set_xlim(x_min-5, x_max+10)
            ax.set_ylim(y_min-50, y_max+50)
            ax.set_xlabel('PCA2')
            ax.set_ylabel('PCA1')
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.xaxis.set_label_position('top')
        ax.xaxis.tick_top()
        ax.yaxis.set_label_position('right')
        ax.yaxis.tick_right()
    # Adjust layout
    plt.tight_layout()

    # Save the plot as a PDF if a save directory is provided
    if save_dir:
        path = f'Separability_{feature_type}_{no_vowels}_{spkid}_{age_month}.pdf'
        save_path = os.path.join(save_dir, path)
        plt.savefig(save_path, format='pdf')
        path_jpg = f'Separability_{feature_type}_{no_vowels}_{spkid}_{age_month}.jpg'
        save_path_jpg = os.path.join(save_dir, path_jpg)
        plt.savefig(save_path_jpg, format='jpg')
        print(f"Plot saved as {save_path}")

    # Show the plot
    plt.show()
    
        
      
