import os
import numpy as np
import pandas as pd

from pydub import AudioSegment
import librosa

import statistics
import parselmouth
from parselmouth.praat import call

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Set the directory path to save spit audio segment
temp_seg_dir = os.path.join(os.getcwd(), 'audio_seg')

# Create the split audio segment directory if it doesn't exist
if not os.path.exists(temp_seg_dir):
    os.makedirs(temp_seg_dir)

# Function to split and save the audio file to the dir
def audio_split_(from_sec, to_sec, wav_file_path, new_wav_file_dir):
    """
    Extracts a segment from an audio file and saves it as a new WAV file.

    This function takes a start and end time (in seconds), reads the corresponding segment 
    from the input WAV file, and saves the extracted segment as a new WAV file in the specified directory.

    Parameters:
    -----------
    from_sec : float
        The starting time of the audio segment to be extracted, in seconds.
    to_sec : float
        The ending time of the audio segment to be extracted, in seconds.
    wav_file_path : str
        The full path to the original WAV audio file.
    new_wav_file_dir : str
        The directory where the extracted (split) audio segment will be saved.

    Returns:
    --------
    AudioSegment
        The extracted audio segment as a `pydub.AudioSegment` object.

    Example:
    --------
    >>> from_sec = 10.0
    >>> to_sec = 20.0
    >>> wav_file_path = r"~\sample.wav"
    >>> new_wav_file_dir = r"~\audio_seg"
    >>> audio_segment = audio_split_(from_sec, to_sec, wav_file_path, new_wav_file_dir)
    >>> print(audio_segment)

    Notes:
    ------
    - The extracted segment is saved as 'seg.wav' in the specified directory.
    - The input and output audio files must be in WAV format.
    - The function will overwrite any existing 'seg.wav' file in the output directory.
    """
    t1 = from_sec * 1000
    t2 = to_sec * 1000
    new_audio = AudioSegment.from_wav(wav_file_path)
    new_audio = new_audio[t1:t2]
    new_audio.export(os.path.join(new_wav_file_dir, 'seg.wav'), format = 'wav') #Exports to a wav file in the current path.
    return new_audio

# Function to delete the split audio segment from the dir
def delete_audio_from_dir(filename):
    """
    Deletes a specified audio file from the directory.

    This function checks if a file with the given filename exists in the predefined 
    audio directory. If the file exists, it deletes the file.

    Parameters:
    -----------
    filename : str
        The name of the file to be deleted (e.g., 'seg.wav').

    Example:
    --------
    >>> filename = 'seg.wav'
    >>> delete_audio_from_dir(filename)

    Notes:
    ------
    - The function expects the file to be located in the directory specified by `audio_dir`.
    - If the file does not exist, no action is taken.
    """
    filepath = os.path.join(temp_seg_dir, filename)
    if os.path.exists(filepath):
        os.remove(filepath)

def extract_formants_per_glottal_pulse(in_file_path, target_vowels_IPA, wav_dir, register, out_file_path, Frame = False, parentgender=None):
    """
    Extracts formant frequencies for each vowel segment within an audio file by analyzing glottal pulses.

    This function processes a CSV file containing metadata and annotations, extracts the corresponding
    audio segments, and computes formant frequencies either for each segments (F1, F2, F3, F4) or per frame of each segment (F1, F2) across glottal pulses.
    In case of Frame computes delta and delta-delta formants (df1, df2, ddf1, ddf2) for each frame.

    Parameters:
    -----------
    in_file_path : str
        Path to the CSV file containing metadata and annotations, including file names, time intervals, and IPA symbols.
    target_vowels_IPA : list of str
        List of target vowel IPA symbols to filter the dataset for analysis.
    wav_dir : str
        Directory containing the WAV audio files referenced in the CSV file.
    register : str
        Register type (e.g., "IDS", "ADS"), used in metadata and file output.
    Frame : bool, optional
        If True, extracts formants for each frame within a segment and computes delta and delta-delta formants. 
        If False, extracts average formant values across glottal pulses for each segment. Default is False.
    parentgender : str, optional
        Gender of the speaker's parent, used to set pitch and formant ceiling values.
        'F' for Father (lower pitch), 'M' for Mother (higher pitch). Default is None (treated as 'M').

    Returns:
    --------
    None
        The function saves the results directly to a csv file specified by `out_file_path`.
        A file containing formant values (F1, F2, F3, F4) for each analyzed segment. 
        If Frame is True, the file containing formant values (F1, F2) and also includes delta and delta-delta formants (dF1, dF2, ddF1, ddF2).
        Additional columns include speaker ID, IPA symbol, age in months, register, and segment duration.

    Notes:
    ------
    - The function assumes that the input CSV file has the following columns: 'file_name', 'time_start', 'time_end', 
      'spkid', 'IPA', 'AgeMonth', and 'Parent'.
    - Temporary audio segments are saved and processed in a specified directory before being deleted.
    - If the analysis fails for a segment, the function logs the error and skips the segment.
    """
    # Set default pitch and formant ceiling values based on gender
    if parentgender == 'F': # Father
        f0min = 75
        f0max = 300
        f_ceiling = 5000
    elif parentgender == 'M': # Mother
        f0min = 100
        f0max = 500
        f_ceiling = 5500
    else:  # Default for unspecified gender (Mother)
        f0min = 100
        f0max = 500
        f_ceiling = 5500

    # Fixed parameters
    frame_length = 400  # in samples
    hop_length = 160  # in samples
    sample_rate = 16000  # in Hz

    # Initialize an empty DataFrame to store all formant values
    temp_dfs = []

    # Load data
    df = pd.read_csv(in_file_path)

    # Filter data based on the parentgender and target_vowels_IPA
    if parentgender:
        df = df[df['Parent'] == parentgender].copy()
    
    df = df[df['IPA'].isin(target_vowels_IPA)].copy()
    df.reset_index(inplace=True)

    for ind, row in df.iterrows():
        file_name = row['file_name']
        if not file_name.endswith('.wav'):
            file_name += '.wav'
        # Construct the audio path
        audio_path = os.path.join(wav_dir, file_name)
        try:
            sound = audio_split_(row['time_start'], row['time_end'], audio_path, temp_seg_dir)
            seg_path = os.path.join(temp_seg_dir, 'seg.wav')

            # Load the sound
            sound = parselmouth.Sound(seg_path)

            # Downsample to 16000 Hz
            if sound.get_sampling_frequency() != sample_rate:
                sound = sound.resample(sample_rate)

            # Convert to mono if not already
            if sound.n_channels > 1:
                sound = sound.convert_to_mono()
            # Extract pitch and point process
            pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
            point_process = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)

            # Extract formants
            formants = call(sound, "To Formant (burg)", 0.0025, 5, f_ceiling, 0.005, 50)
            # Get the number of glottal pulses
            num_points = call(point_process, "Get number of points")
            if Frame:
                # Calculate the number of frames
                num_samples = sound.get_total_duration() * sample_rate
                num_frames = int((num_samples - frame_length) // hop_length) + 1

                # Initialize the matrix to store formant values
                formant_matrix = np.zeros((num_frames, 2))  # Only F1 and F2 are being stored

                # Extract formants for each frame
                for i in range(num_frames):
                    start_time = i * hop_length / sample_rate
                    end_time = start_time + frame_length / sample_rate

                    # Initialize lists to store formants for the current frame
                    formant_lists = [[] for _ in range(2)]  # For F1 and F2, Adujust according to number of formants

                    # Check each glottal pulse
                    for j in range(1, num_points + 1):
                        t = call(point_process, "Get time from index", j)
                        # Check if the pulse is within the current frame
                        if start_time <= t <= end_time:
                            for k in range(2): # adujust according to number of formants require
                                formant_value = call(formants, "Get value at time", k + 1, t, 'Hertz', 'Linear')
                                if not np.isnan(formant_value):
                                    formant_lists[k].append(formant_value)

                    # Calculate the mean formant values across glottal pulses in this frame
                    for k in range(2):
                        if formant_lists[k]:
                            formant_matrix[i, k] = statistics.mean(formant_lists[k])
                        else:
                            formant_matrix[i, k] = np.nan  # Or use NaN to indicate missing data

                # Calculate delta and delta-delta formants
                delta_formant_matrix = librosa.feature.delta(formant_matrix, width=3, order=1, axis=0)
                delta_delta_formant_matrix = librosa.feature.delta(formant_matrix, width=3, order=2, axis=0)

                # Create a temporary DataFrame for the current file
                df_temp = pd.DataFrame(formant_matrix, columns=['F1', 'F2'])
                df_temp['spkid'] = row['spkid']
                df_temp['IPA'] = row['IPA']
                df_temp['AgeMonth'] = row['AgeMonth']
                df_temp['Register'] = register
                df_temp['time_start'] = row['time_start'] # use this information to get number of frames for the current vowel segment
                df_temp['duration(ms)'] = 1000 * (row['time_end'] - row['time_start'])  # Duration in milliseconds

                # Add delta and delta-delta formants
                df_temp[['dF1', 'dF2']] = delta_formant_matrix
                df_temp[['ddF1', 'ddF2']] = delta_delta_formant_matrix

                # Rearrange columns: move F1, F2, dF1, dF2, ddF1, ddF2 to the end
                df_temp = df_temp[['spkid', 'IPA', 'AgeMonth', 'Register', 'time_start', 'duration(ms)', 'F1', 'F2', 'dF1', 'dF2', 'ddF1', 'ddF2']]

                temp_dfs.append(df_temp)

                # Delete the temporary audio segment
                delete_audio_from_dir('seg.wav')
            else:
                # Initialize formant lists
                formant_lists = [[] for _ in range(4)]
    
                # Measure formants at glottal pulses
                for i in range(1, num_points + 1):
                    t = call(point_process, "Get time from index", i)
                    for j in range(4):
                        formant_value = call(formants, "Get value at time", j + 1, t, 'Hertz', 'Linear')
                        if not np.isnan(formant_value):
                            formant_lists[j].append(formant_value)
    
                # Calculate mean formants across pulses
                mean_formants = []
                for formant_list in formant_lists:
                    if formant_list:
                        mean_formants.append(statistics.mean(formant_list))
                    else:
                        mean_formants.append(None)
                # Create a temporary DataFrame for the current file
                df_temp = pd.DataFrame([mean_formants], columns=['F1', 'F2', 'F3', 'F4'])
                df_temp['spkid'] = row['spkid']
                df_temp['IPA'] = row['IPA']
                df_temp['AgeMonth'] = row['AgeMonth']
                df_temp['Register'] = register
                df_temp['duration(ms)'] = 1000 * (row['time_end'] - row['time_start'])  # Duration in milliseconds

                # Rearrange columns: move F1, F2, dF1, dF2, ddF1, ddF2 to the end
                df_temp = df_temp[['spkid', 'IPA', 'AgeMonth', 'Register', 'duration(ms)', 'F1', 'F2', 'F3', 'F4']]

                temp_dfs.append(df_temp)
                # Delete the temporary audio segment
                delete_audio_from_dir('seg.wav')
        except Exception as e:
            # Ensure the temporary audio segment is deleted in case of error
            delete_audio_from_dir('seg.wav')
            #print(f'Error at index {ind}: {e}')
            continue

    df_all_formants = pd.concat(temp_dfs, ignore_index=True)
    df_all_formants.dropna(inplace=True)
    df_all_formants.reset_index(inplace=True)
    if parentgender != None:
        df_all_formants.to_csv(out_file_path + '_' + str(len(target_vowels_IPA)) + '_' + 'bordervowels' + '_' + parentgender + '.csv', index=False)
    else:
        df_all_formants.to_csv(out_file_path + '_' + str(len(target_vowels_IPA)) + '_' + 'bordervowels' + '.csv', index=False)
    print('Formant values have benn saved') 


def extract_mfcc(in_file_path, target_vowels_IPA, wav_dir, register,out_file_path, Frame=False, parentgender=None):
    """
    Extracts Mel-frequency cepstral coefficients (MFCCs) from audio segments and stores them in a DataFrame.

    This function processes audio segments based on their start and end times, computes MFCC features 
    and their deltas, and returns a DataFrame with these features. Depending on the `Frame` parameter, 
    MFCCs can be extracted for each frame of the audio segment or averaged over the entire segment.

    Parameters:
    -----------
    in_file_path : str
        Path to the CSV file containing metadata and annotations, which should include columns for 
        file names, start and end times, and additional metadata.
    target_vowels_IPA : list of str
        List of target vowel IPA symbols to filter the data.
    wav_dir : str
        Directory containing the WAV files for audio segments.
    register : str
        Register type (ADS or IDS), used for file output or metadata purposes.
    Frame : bool, optional
        If True, extracts MFCC features frame-by-frame. If False, computes the mean MFCCs for each segment.
    parentgender : str, optional
        Gender of the speaker's parent mother or father.

    Returns:
    --------
    None
        The function saves the results directly to a csv file specified by `out_file_path`.
        file containing MFCC values and delta MFCCs for each audio segment, If `Frame` is True, 
        each row corresponds to a frame with MFCCs and delta MFCCs. If `Frame` is False, each row represents 
        the mean MFCCs only.

    Notes:
    ------
    - The function expects the audio files to be in WAV format and located in the directory specified by `wav_dir`.
    - Delta MFCCs are computed using `librosa.feature.delta`.
    - Temporary audio segment files (`'seg.wav'`) are created and then deleted after processing.
    - Any errors encountered during processing will print an error message and skip to the next segment.

    Example:
    --------
    >>> df = pd.DataFrame({
            'file_name': ['audio1', 'audio2'],
            'time_start': [0, 5],
            'time_end': [5, 10],
            'spkid': ['001', '002'],
            'IPA': ['a', 'e'],
            'AgeMonth': [12, 15]
        })
    >>> wav_dir = r'C:\path\to\audio\files'
    >>> df_all_mfccs = extract_mfcc('metadata.csv', ['a', 'e'], wav_dir, 'ADS', Frame=True, parentgender='F')
    >>> print(df_all_mfccs)
    """
    # Fixed parameters
    n_fft = 400  # in samples
    hop_length = 160  # in samples
    sample_rate = 16000  # in Hz

    # Initialize an empty list to store DataFrames
    temp_dfs = []

    # Define the MFCC and delta-MFCC column names
    mfcc_columns = [f'mfcc{i+1}' for i in range(13)]
    dmfcc_columns = [f'dmfcc{i+1}' for i in range(13)]

    # Load data
    df = pd.read_csv(in_file_path)

    # Filter data based on the parentgender and target_vowels_IPA
    if parentgender:
        df = df[df['Parent'] == parentgender].copy()
    
    df = df[df['IPA'].isin(target_vowels_IPA)].copy()
    df.reset_index(drop=True, inplace=True)

    for ind, row in df.iterrows():
        file_name = row['file_name']
        if not file_name.endswith('.wav'):
            file_name += '.wav'
        
        # Construct the audio path
        audio_path = os.path.join(wav_dir, file_name)
        try:
            # Extract the audio segment
            sound = audio_split_(row['time_start'], row['time_end'], audio_path, temp_seg_dir)
            y, sr = librosa.load(os.path.join(temp_seg_dir, 'seg.wav'), sr=sample_rate)

            # Extract MFCC features
            mfccs_matrix = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
            dmfcc_matrix = librosa.feature.delta(mfccs_matrix, width=3, order=1, axis=0)

            if Frame:
                # Create a DataFrame with frame-level MFCCs
                df_temp = pd.DataFrame(mfccs_matrix.T, columns=mfcc_columns)
                df_temp['spkid'] = row['spkid']
                df_temp['IPA'] = row['IPA']
                df_temp['AgeMonth'] = row['AgeMonth']
                df_temp['Register'] = register
                df_temp['time_start'] = row['time_start']
                df_temp['duration(ms)'] = 1000 * (row['time_end'] - row['time_start'])

                # Add delta MFCCs
                for i, col in enumerate(dmfcc_columns):
                    df_temp[col] = dmfcc_matrix[i, :]
                # Shift mfccs column in last
                df_temp = df_temp[[col for col in ['spkid', 'IPA', 'AgeMonth', 'Register', 'time_start', 'duration(ms)'] + mfcc_columns+dmfcc_columns]]
                temp_dfs.append(df_temp)

            else:
                # Mean along the time axis (axis 1)
                mfccs_mean = np.mean(mfccs_matrix, axis=1)
                dmfcc_mean = np.mean(dmfcc_matrix, axis=1)

                # Create a DataFrame with a single row for the mean MFCCs
                df_temp = pd.DataFrame([mfccs_mean], columns=mfcc_columns)

                df_temp['spkid'] = row['spkid']
                df_temp['IPA'] = row['IPA']
                df_temp['AgeMonth'] = row['AgeMonth']
                df_temp['Register'] = register
                df_temp['time_start'] = row['time_start']
                df_temp['duration(ms)'] = 1000 * (row['time_end'] - row['time_start'])

                # Shift mfccs column in last
                df_temp = df_temp[[col for col in ['spkid', 'IPA', 'AgeMonth', 'Register', 'time_start', 'duration(ms)'] + mfcc_columns]]

                temp_dfs.append(df_temp)

            # Delete the temporary audio segment
            delete_audio_from_dir('seg.wav')

        except Exception as e:
            # Ensure the temporary audio segment is deleted in case of error
            delete_audio_from_dir('seg.wav')
            #print(f'Error at index {ind}: {e}')
            continue

    df_all_mfccs = pd.concat(temp_dfs, ignore_index=True)
    df_all_mfccs.dropna(inplace=True)
    df_all_mfccs.reset_index(drop=True, inplace=True)
    if parentgender != None:
        df_all_mfccs.to_csv(out_file_path + '_' + str(len(target_vowels_IPA)) + '_' + 'bordervowels' + '_' + parentgender + '.csv', index=False)
    else:
        df_all_mfccs.to_csv(out_file_path + '_' + str(len(target_vowels_IPA)) + '_' + 'bordervowels' + '.csv', index=False)
    print('MFCC values have been saved') 