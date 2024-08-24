import os
from utils import create_dir
# Base directory of the project
basedir = os.path.abspath(os.path.dirname(__file__)) 

# WaveFiles directory
# Norwagian wave dir
#no_wav_dir = path/to/norwagian wavdir 'Uncomment it and give the path of your wav file dir'
# Danish wav dir
#da_wav_dir = path/to/danish wavdir 'Uncomment it and give the path of your if other language wav file dir'

#Metadata directory
meta_dir = os.path.join(basedir, 'Metadata')
create_dir(meta_dir)

# Danish matadata
path_meta_df_ADS_da = os.path.join(meta_dir, 'metadata_ADS_da.csv')
path_meta_df_IDS_da = os.path.join(meta_dir, 'metadata_IDS_da.csv')

# Norwagian meatadata
path_meta_df_ADS_no = os.path.join(meta_dir, 'metadata_ADS_no.csv')
path_meta_df_IDS_no = os.path.join(meta_dir, 'metadata_IDS_no.csv')

# Data count directory
data_count_dir = os.path.join(basedir, 'DataCount')
create_dir(data_count_dir)

#data count file full path
no_ADS_count = os.path.join(data_count_dir, 'no_ADS_count_per_spkid_age_IPA_df.xlsx')
no_IDS_count = os.path.join(data_count_dir, 'no_IDS_count_per_spkid_age_IPA_df.xlsx')
da_ADS_count = os.path.join(data_count_dir, 'da_ADS_count_per_spkid_age_IPA_df.xlsx')
da_IDS_count = os.path.join(data_count_dir, 'da_IDS_count_per_spkid_age_IPA_df.xlsx')

# Segemnt wise vowel features directory
clean_data_dir = os.path.join(basedir, 'SegmentWiseFeatures')
create_dir(clean_data_dir)

# path to save segment wise features
# Norwagian
# formant
out_path_formant_no_ADS_seg = os.path.join(clean_data_dir,'formant_no_ADS')
out_path_formant_no_IDS_seg = os.path.join(clean_data_dir, 'formant_no_IDS')

#MFCC
out_path_mfcc_no_ADS_seg = os.path.join(clean_data_dir, 'mfcc_no_ADS')
out_path_mfcc_no_IDS_seg = os.path.join(clean_data_dir, 'mfcc_no_IDS')

# Danish
#formant
out_path_formant_da_ADS_seg = os.path.join(clean_data_dir, 'formant_da_ADS')
out_path_formant_da_IDS_seg = os.path.join(clean_data_dir, 'formant_da_IDS')
#MFCC
out_path_mfcc_da_ADS_seg = os.path.join(clean_data_dir, 'mfcc_da_ADS')
out_path_mfcc_da_IDS_seg = os.path.join(clean_data_dir, 'mfcc_da_IDS')

# Framwise formant and mfcc features dir for ML models
out_dir_ML = os.path.join(basedir, 'MLDATA')
create_dir(out_dir_ML)

# path to save frame wise features
# Norwagian
# formant
out_path_formant_no_ADS_ML = os.path.join(out_dir_ML,'formant_no_ADS')
out_path_formant_no_IDS_ML = os.path.join(out_dir_ML, 'formant_no_IDS')

#MFCC
out_path_mfcc_no_ADS_ML = os.path.join(out_dir_ML, 'mfcc_no_ADS')
out_path_mfcc_no_IDS_ML = os.path.join(out_dir_ML, 'mfcc_no_IDS')

# Danish
#formant
out_path_formant_da_ADS_ML = os.path.join(out_dir_ML, 'formant_da_ADS')
out_path_formant_da_IDS_ML = os.path.join(out_dir_ML, 'formant_da_IDS')
#MFCC
out_path_mfcc_da_ADS_ML = os.path.join(out_dir_ML, 'mfcc_da_ADS')
out_path_mfcc_da_IDS_ML = os.path.join(out_dir_ML, 'mfcc_da_IDS')

# Computed vowel-based measures ouput dirs
out_base_dir = os.path.join(basedir, 'ComputedMeasures')
create_dir(out_base_dir)

#Measures out dir
area = os.path.join(out_base_dir, 'Area')
create_dir(area)

variability = os.path.join(out_base_dir, 'Variability')
create_dir(variability)

separability = os.path.join(out_base_dir, 'Separability')
create_dir(separability)




