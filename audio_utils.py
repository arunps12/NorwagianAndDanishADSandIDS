import os
import numpy as np
import pandas as pd

from pydub import AudioSegment
import librosa

import statistics
import parselmouth
from parselmouth.praat import call

# Set the directory path to save spit audio segment
audio_dir = r'C:\Users\arunps\OneDrive\Projects\Scripts\Python\Research Stay\audio_seg'

# Create the split audio segment directory if it doesn't exist
if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)

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
    filepath = os.path.join(audio_dir, filename)
    if os.path.exists(filepath):
        os.remove(filepath)

def extract_formants_mean_glottal_pulse(wave_file, f0min, f0max, no_formants, f_ceiling):
    # Load the sound and extract pitch and point process
    sound = parselmouth.Sound(wave_file)
    pitch = call(sound, "To Pitch (cc)", 0, f0min, 15, 'no', 0.03, 0.45, 0.01, 0.35, 0.14, f0max)
    point_process = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    
    # Extract formants
    formants = call(sound, "To Formant (burg)", 0.0025, no_formants, f_ceiling, 0.005, 50)
    num_points = call(point_process, "Get number of points")
    
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
    
    # Return the mean values of F1, F2, F3, and F4
    return mean_formants

def extract_mfcc(df, wav_dir):
    """
    Extracts MFCC features from audio segments and stores them in a DataFrame.

    This function iterates over each row of the input DataFrame, uses the start and end 
    times to extract the corresponding audio segment, computes the MFCC features from 
    the segment, and stores the mean of each MFCC coefficient in the DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame containing the timing and file information for each audio segment.
        Expected columns: ['file_name', 'time_start', 'time_end'].
    wav_dir : str
        The directory where the original audio files are stored.

    Returns:
    --------
    None
        The function modifies the input DataFrame in place by adding columns for each MFCC coefficient.

    Example:
    --------
    >>> df = pd.DataFrame({
            'file_name': ['audio1', 'audio2'],
            'time_start': [0, 5],
            'time_end': [5, 10]
        })
    >>> wav_dir = r'C:\path\to\audio\files'
    >>> mfcc(df, wav_dir)
    >>> print(df)

    Notes:
    ------
    - The function expects the audio files to be in WAV format and located in `wav_dir`.
    - MFCC features are extracted using `librosa`. The mean of each of the 13 MFCC coefficients 
      is calculated and stored in the DataFrame.
    - The temporary audio segment file ('seg.wav') is deleted after processing each segment.
    - If an error occurs while processing a segment, it is caught, the segment is deleted, 
      and the function proceeds to the next segment.
    """

    # Define the MFCC column names
    mfcc_columns = [f'mfcc{i+1}' for i in range(13)]

    # Ensure the DataFrame has the correct columns for MFCC features
    for col in mfcc_columns:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = np.nan

    for ind, row in df.iterrows():
        audio_path = os.path.join(wav_dir, row['file_name'] + '.wav')  # Audio file path
        try:
            sound = audio_split_(row['time_start'], row['time_end'], audio_path, wav_dir)
            y, sr = librosa.load(os.path.join(wav_dir, 'seg.wav'), sr=16000)
            
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=400, hop_length=160, win_length=400)
            # Mean along the time axis (axis 1)
            mfccs_mean = np.mean(mfccs, axis=1)
            
            # Assign values to the correct columns
            for i in range(len(mfccs_mean)):
                df.loc[ind, mfcc_columns[i]] = mfccs_mean[i]
            
            # Delete the temporary audio segment
            delete_audio_from_dir('seg.wav')
        except Exception as e:
            # Ensure the temporary audio segment is deleted in case of error
            delete_audio_from_dir('seg.wav')
            print(f'Error at index {ind}: {e}')
            continue

