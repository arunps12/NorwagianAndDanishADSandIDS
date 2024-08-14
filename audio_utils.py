import os
import numpy as np
import pandas as pd

from pydub import AudioSegment
import librosa

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

# Function to extact mfcc features for each extracted (split) audio segement
def mfcc(df, wav_dir):
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
    for ind, row in df.iterrows():
        audio_path = os.path.join(wav_dir, row['file_name'] + '.wav') #  audio file path
        try:
            sound = audio_split_(row['time_start'], row['time_end'], audio_path, audio_dir)
            y, sr = librosa.load(os.path.join(audio_dir, 'seg.wav'), sr=16000)
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft = 400, hop_length=160, win_length = 400)
            #mfccs_normalized = librosa.util.normalize(mfccs)
            #mean along the time axis (axis 1)
            mfccs_mean = np.mean(mfccs, axis=1)
            for i in range(len(mfccs_mean)):
                df.loc[ind, i+1] = mfccs_mean[i]
            delete_audio_from_dir('seg.wav')
        except Exception as e:
            delete_audio_from_dir('seg.wav')
            print(f'Error at index {ind}: {e}')
            continue

