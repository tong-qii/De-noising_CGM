import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from statsmodels.nonparametric.smoothers_lowess import lowess
from collections import defaultdict
import pywt
import scaleogram as scg
from ssqueezepy import cwt
from ssqueezepy.visuals import plot, imshow
from bokeh.plotting import figure, show, output_notebook
from bokeh.layouts import column
from bokeh.models import BoxZoomTool, ResetTool
from bokeh.models import Span
from bokeh.io import output_file

from datetime import datetime
from scipy.spatial import distance

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

###################### data exploration
#######################################

def check_time_format(df):
    import pandas as pd
    '''check if all the "Time" have the same required time format'''
    time_series = df['Time']
    time_format = "%Y-%m-%d %H:%M:%S"
    inconsistent_indices = []
    for index, value in time_series.items():
        try:
            # Attempt to parse each value in the Series as a time using the specified format
            pd.to_datetime(value, format=time_format)
        except ValueError:
            inconsistent_indices.append(index)
        
    return inconsistent_indices

def split_by_subject(df, subject_column):
    '''Group the DataFrame by the subject column
    return the dfs by subject order'''
    grouped = df.groupby(subject_column)
    
    dfs_by_subject = {}
    
    for subject, group in grouped:
        group = group.reset_index(drop=True)
        dfs_by_subject[subject] = group.copy()
    
    return dfs_by_subject

def check_NAs(df):
    '''check all possible NA for the whole dataset'''
    nan_count = df.isna().sum()
    if nan_count.sum() == 0:
        print("There are no missing values in the dataset.")
    else:
        print("There are missing values in the dataset:")
        for column, count in nan_count.items():
            if count > 0:
                print(f"- {column}: {count} missing values")


def remove_GlucoseNA(df):
    df_cleaned = df.dropna(subset=['Glucose'])
    return df_cleaned

def remove_duplicates(df):
    '''Function to normalize datetime to minute, ignoring seconds'''
    def normalize_to_minute(dt_str):
        try:
            dt = pd.to_datetime(dt_str, errors='coerce')
            return dt.floor('T')
        except ValueError:
            return dt_str
    df['normalized_time'] = df['Time'].apply(normalize_to_minute)
    df_cleaned = df.drop_duplicates(subset=['id', 'normalized_time', 'Glucose'])
    # df_cleaned = df_cleaned.groupby(['id', 'normalized_time'], as_index=False).agg({'Glucose': 'median'})
    df_cleaned = df_cleaned.drop(columns=['normalized_time'])
    return df_cleaned
        


def print_wrong_time_records(df,inconsistent_indices):
    '''print out the wrong time format records'''
    print(df.iloc[np.r_[inconsistent_indices],:])

def impute_average_time(df,inconsistent_indices):
    '''impute the average time for the wrong formate time records'''
    time_format = "%Y-%m-%d %H:%M:%S"
    for i in inconsistent_indices:
        max_time =  pd.Timestamp(df['Time'][i-1])
        min_time =  pd.Timestamp(df['Time'][i+1])
        mid_time = ((max_time - min_time) / 2) + min_time
        mid_time = mid_time.strftime(time_format)
        df.at[i, 'Time'] = mid_time

    return df

def rows_with_repeated_values(df, column_name):
    '''Identify the repeated values in the specified column'''
    repeated_values = df[column_name][df[column_name].duplicated(keep=False)]
    if repeated_values.empty:
        print("There are no repeated values in the specified column.")
        return pd.DataFrame(columns=df.columns)
    df_repeated = df[df[column_name].isin(repeated_values)]
    
    return df_repeated
        
def time_diff_minutes_series(df):
    '''
    # check the "Time" difference
    # less4 returns the number of records that the time diff is between 4 and 5, acceptable
    # more6 returns the number of records that the time diff is between 5 and 6, acceptable
    # total returns all the number of records that time diff is not 5 mins
    '''
    time_series = df['Time']
    input_format = "%Y-%m-%d %H:%M:%S"
    time_diff_minutes_list = []
    for i in range(1,len(time_series)):
        datetime1 = datetime.strptime(time_series[i-1], input_format)
        datetime2 = datetime.strptime(time_series[i], input_format)
        time_diff = (datetime2 - datetime1).total_seconds() / 60
        time_diff_minutes_list.append(time_diff)

    less4 = sum(4<i<5 for i in time_diff_minutes_list)
    more6 = sum(5<i<6 for i in time_diff_minutes_list)
    total = sum(i != 5 for i in time_diff_minutes_list )
        
    return time_diff_minutes_list,less4,more6,total


def check_missing_time_diff(time_diffs):
    '''returns statment of whether there is or not missing time differences'''
    sum1 = sum(np.isnan(time_diffs))
    if sum1==0:
        print('There is no missing time diffs.')
    else:
        print('There is ',sum1,' missing time diffs.')

def count_records_per_day(df):
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    #df['Date'] = df['Time'].dt.date
    records_per_day = df['Time'].dt.date.value_counts().sort_index()
    records_per_day_df = records_per_day.reset_index()
    records_per_day_df.columns = ['Date', 'Number of Records']
    total_days = len(records_per_day_df["Date"])
    
    return records_per_day_df,total_days

def find_missing_ranges(df):
    """Identify ranges of missing datetime values in the DataFrame.
    """
    missing_ranges = []
    missing_indices = df.index[df['Time'].isna()].tolist()
    if not missing_indices:
        return missing_ranges
    
    start_idx = missing_indices[0]
    end_idx = start_idx
    
    for idx in missing_indices[1:]:
        if idx == end_idx + 1:
            end_idx = idx
        else:
            missing_ranges.append((start_idx, end_idx))
            start_idx = idx
            end_idx = idx
    
    missing_ranges.append((start_idx, end_idx))
    print("Missing time range is ", missing_ranges)
    return missing_ranges

def impute_datetimes(df, freq='5min'):
    time_format = "%Y-%m-%d %H:%M"
    
    missing_ranges = find_missing_ranges(df)
    for start_idx, end_idx in missing_ranges:
        start_time = pd.to_datetime(df.loc[start_idx-1, 'Time'])
        end_time = pd.to_datetime(df.loc[end_idx+1, 'Time'])
        
        # Generate the imputed datetime range
        imputed = pd.date_range(start=start_time, end=end_time, freq=freq)
        
        # Ensure the imputed times have no seconds
        imputed = imputed.to_pydatetime()
        imputed = [ts.replace(second=0, microsecond=0) for ts in imputed]
        imputed = [ts.strftime(time_format) for ts in imputed]
        
        # Apply the imputed times to the missing values
        if len(imputed[1:-1]) == (end_idx - start_idx + 1):
            df.loc[start_idx:end_idx, 'Time'] = imputed[1:-1]
        else:
            print(f"Imputed range for indices {start_idx} to {end_idx} does not match the missing values range. Check the imputation logic.")
    
    return df


####------------------- check the repeated time chunks----------------------------
####------------------------------------------------------------------------------
def crossover_timechunk(df):
    '''Check the crossover time chunk
    '''
    break_indices = [0]  
    for i in range(1, len(df)):
        if df['Time'][i] < df['Time'][i - 1]:
            break_indices.append(i)
    break_indices.append(len(df))  
    dfs = []
    for idx in range(len(break_indices) - 1):
        part = df.iloc[break_indices[idx]:break_indices[idx + 1]].copy()
        if idx > 0: 
            part['id'] = part['id'].astype(str) + f"_{idx + 1}"
        dfs.append(part)
    return dfs
 
def split_crossover_timechunk(df_dicts):
    '''split the crossover time chunk
    '''
    split_dict = {}
    for date, date_dfs in df_dicts.items():
        middle_dict = {}
        for subject, df in date_dfs.items():
            final_df = crossover_timechunk(df)
            middle_dict[subject] = final_df
        split_dict[date] = middle_dict
    return split_dict




def remove_dup_dicts(df_dicts):
    '''remove the duplicated records
    and add the _2 for the subject having crossover time chunk'''
    removed_dict = {}
    for date, date_dfs in df_dicts.items():
        middle_dict = {}
        for subject, df_list in date_dfs.items():
            #final_df = remove_duplicates(df)
            #final_df_list = [remove_duplicates(df) for df in df_list]
            for i, df in enumerate(df_list):
                new_subject = subject if i == 0 else f"{subject}_2"
                middle_dict[new_subject] = remove_duplicates(df)
            #middle_dict[subject] = final_df_list
        removed_dict[date] = middle_dict
    return removed_dict



############################# functions for all subjects
##########################################################

def data_subject_info(df,freq='5min'):
    '''print out the basic info of each subject 
    return the 'good to use' for wavelet transform dataframe list with subject index in order'''

    print("There are in total ", len(df.id.unique()), "subjects.")
    dfs_good = {}
    dfs_by_subject = split_by_subject(df,"id")
    for subject, df_subject in dfs_by_subject.items():
        print('The information for subject: ', subject)
        subject_1 = df_subject
        check_NAs(subject_1)
        subject_1 = remove_GlucoseNA(subject_1)
        time_check = check_time_format(subject_1)
        df_gtime  = impute_average_time(subject_1,time_check)
        df_nomiss = impute_datetimes(df_gtime,freq=freq )
        records_per_day_df, total_days = count_records_per_day(df_nomiss)
        print("Records per day:")
        print(records_per_day_df)
        print("\nTotal number of days:", total_days)
        dfs_good[subject] = df_gtime.copy()

    return dfs_good



################################# wavelet transform
###################################################

def dwt_denoise(signal, wavelet='bior2.6', level=None, thresholding='soft', threshold_value=None):
    '''
    Do the DWT on one singal, wavelet family, level of decomposition,
    type of threshold and value can be customized
    '''
    coeffs = pywt.wavedec(signal, wavelet, level=level,axis = 0)
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    minimax_n = len(coeffs[-1])
    
    if threshold_value is None:
        threshold_value = sigma * np.sqrt(2 * np.log(len(signal)))

    if threshold_value == 'minimax':
        threshold_value = sigma * np.sqrt(2 * np.log(minimax_n))

    def hard_threshold(coeff, thresh):
        return coeff * (np.abs(coeff) >= thresh)
    
    def soft_threshold(coeff, thresh):
        return np.sign(coeff) * np.maximum(np.abs(coeff) - thresh, 0)
    
    def garrote_threshold(coeff, thresh):
        return coeff * (np.abs(coeff) >= thresh) - (thresh**2 / coeff) * (np.abs(coeff) >= thresh)
    
    if thresholding == 'hard':
        threshold_func = hard_threshold
    elif thresholding == 'soft':
        threshold_func = soft_threshold
    elif thresholding == 'garrote':
        threshold_func = garrote_threshold
    else:
        raise ValueError("Invalid thresholding method. Choose 'hard', 'soft', or 'garrote'.")
    
    coeffs[1:] = [threshold_func(c, threshold_value) for c in coeffs[1:]]
    
    denoised_signal = pywt.waverec(coeffs, wavelet)

    if len(denoised_signal) == len(signal):
        return denoised_signal
    else:
        denoised_signal_adj = denoised_signal[:-1]
        return denoised_signal_adj


def get_reconstruct_daily(df_good,best_parameter_dict):
    '''get reconstructions by daily data
    input is one dataframe on one subject
    it will be splited into daily data then do DWT on each day'''
    date_dfs = split_by_date_onesub(df_good)
    final_reconstruct_daily = {}
    for date, df_good in date_dfs.items():
        original_signal = df_good['Glucose'].values
        #t = df_good['Time'].values
        best_wavelet = best_parameter_dict[date][0]["wavelet"][0]
        best_level = best_parameter_dict[date][0]["level"][0]
        final_reconstructed = dwt_denoise(original_signal,wavelet=best_wavelet, level=best_level, thresholding='hard')
        final_reconstruct_daily[date]=final_reconstructed
    return final_reconstruct_daily



##################### metrics functions
#########################################

def mse(original_signal,reconstructed_signal):
    original = np.asarray(original_signal)
    reconstructed = np.asarray(reconstructed_signal)
    mse = np.mean((original - reconstructed) ** 2)
    return mse
    
def signal_noise_ratio(original_signal,reconstructed_signal):
    signal_power = np.mean(original_signal ** 2)
    noise = original_signal-  reconstructed_signal
    noise_power = np.mean(noise ** 2)
    snr = signal_power / noise_power
    # converts the SNR from a linear scale to decibels.
    snr_db = 10 * np.log10(snr)
    #return snr,snr_db
    return snr_db

def peak_snr(original, reconstructed):
    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel_value = np.max(original)
    psnr = 20 * np.log10(max_pixel_value / np.sqrt(mse))
    
    return psnr

def get_r2(original, reconstructed):
    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)
    mean_original = np.mean(original)
    ss_tot = np.sum((original - mean_original) ** 2)
    ss_res = np.sum((original - reconstructed) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    return r2

def MAGE(df, std=1):
    '''
    https://github.com/brinnaebent/cgmquantify/blob/master/cgmquantify/__init__.py#L503
    '''
    #extracting glucose values and incdices
    glucose = df['Glucose'].tolist()
    ix = [1*i for i in range(len(glucose))]
    stdev = std
    
    # local minima & maxima
    a = np.diff(np.sign(np.diff(glucose))).nonzero()[0] + 1      
    # local min
    valleys = (np.diff(np.sign(np.diff(glucose))) > 0).nonzero()[0] + 1 
    # local max
    peaks = (np.diff(np.sign(np.diff(glucose))) < 0).nonzero()[0] + 1         
    # +1 -- diff reduces original index number

    #store local minima and maxima -> identify + remove turning points
    excursion_points = pd.DataFrame(columns=['Index', 'Time', 'Glucose', 'Type'])
    k=0
    for i in range(len(peaks)):
        excursion_points.loc[k] = [peaks[i]] + [df['Time'][k]] + [df['Glucose'][k]] + ["P"]
        k+=1

    for i in range(len(valleys)):
        excursion_points.loc[k] = [valleys[i]] + [df['Time'][k]] + [df['Glucose'][k]] + ["V"]
        k+=1

    excursion_points = excursion_points.sort_values(by=['Index'])
    excursion_points = excursion_points.reset_index(drop=True)


    # selecting turning points
    turning_points = pd.DataFrame(columns=['Index', 'Time', 'Glucose', 'Type'])
    k=0
    for i in range(stdev,len(excursion_points.Index)-stdev):
        positions = [i-stdev,i,i+stdev]
        for j in range(0,len(positions)-1):
            if(excursion_points.Type[positions[j]] == excursion_points.Type[positions[j+1]]):
                if(excursion_points.Type[positions[j]]=='P'):
                    if excursion_points.Glucose[positions[j]]>=excursion_points.Glucose[positions[j+1]]:
                        turning_points.loc[k] = excursion_points.loc[positions[j+1]]
                        k+=1
                    else:
                        turning_points.loc[k] = excursion_points.loc[positions[j+1]]
                        k+=1
                else:
                    if excursion_points.Glucose[positions[j]]<=excursion_points.Glucose[positions[j+1]]:
                        turning_points.loc[k] = excursion_points.loc[positions[j]]
                        k+=1
                    else:
                        turning_points.loc[k] = excursion_points.loc[positions[j+1]]
                        k+=1

    if len(turning_points.index)<10:
        turning_points = excursion_points.copy()
        excursion_count = len(excursion_points.index)
    else:
        excursion_count = len(excursion_points.index)/2


    turning_points = turning_points.drop_duplicates(subset= "Index", keep= "first")
    turning_points=turning_points.reset_index(drop=True)
    excursion_points = excursion_points[excursion_points.Index.isin(turning_points.Index) == False]
    excursion_points = excursion_points.reset_index(drop=True)

    # calculating MAGE
    mage = turning_points.Glucose.sum()/excursion_count
    
    return round(mage,3)

def calculate_cgm_metrics(data):
    sd = data.std()
    mean_glucose = data.mean()
    cv = (sd / mean_glucose) * 100  

    eA1c = (46.7 + np.mean(data))/ 28.7 

    # def calculate_mage(glucose_levels, threshold=1):
    #     glucose_levels = np.array(glucose_levels)
    #     mean_glucose = np.mean(glucose_levels)
        
    #     excursions = []
    #     for i in range(1, len(glucose_levels)):
    #         if (glucose_levels[i-1] < mean_glucose and glucose_levels[i] > mean_glucose) or \
    #         (glucose_levels[i-1] > mean_glucose and glucose_levels[i] < mean_glucose):
    #             excursions.append(i)
      
    #     amplitudes = []
    #     for i in range(1, len(excursions)):
    #         amplitude = abs(glucose_levels[excursions[i]] - glucose_levels[excursions[i-1]])
    #         if amplitude > threshold:
    #             amplitudes.append(amplitude)

    #     mage = np.mean(amplitudes) if amplitudes else 0
    #     return mage
    # mage = calculate_mage(data)
    
    return sd, cv, eA1c

def hamming_distance(s1, s2):
    if len(s1) != len(s2):
        raise ValueError("Strings must be of the same length")
    distance = 0
    for i in range(len(s1)):
        if s1[i] != s2[i]:
            distance += 1
    return distance

def transition_rates(t1, t2):
    vec1 = np.array(t1)
    vec2 = np.array(t2)
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must be of the same length.")
    
    zeros_to_ones = np.sum((vec1 == 0) & (vec2 == 1))
    ones_to_zeros = np.sum((vec1 == 1) & (vec2 == 0))

    zeros_to_zeros = np.sum((vec1 == 0) & (vec2 == 0))
    ones_to_ones = np.sum((vec1 == 1) & (vec2 == 1))
    
    total_zeros =np.sum(vec1 == 0)    # len(vec1)
    total_ones = np.sum(vec1 == 1)   #  len(vec1)
    
    rate_zeros_to_ones = zeros_to_ones / total_zeros if total_zeros > 0 else 0
    rate_ones_to_zeros = ones_to_zeros / total_ones if total_ones > 0 else 0

    rate_zeros_to_zeros = zeros_to_zeros / total_zeros if total_zeros > 0 else 0
    rate_ones_to_ones = ones_to_ones / total_ones if total_ones > 0 else 0
    
    
    return rate_zeros_to_ones, rate_ones_to_zeros, rate_zeros_to_zeros, rate_ones_to_ones

def convert_to_binary(glucose_values, threshold=70):
    binary_array = [1 if value <= threshold else 0 for value in glucose_values]
    return binary_array

def get_metrics(original, reconstructed):
    '''
    get all metrics for one set of original and reconstructed signals
    '''
    mse1 = mse(original, reconstructed)
    snr_db = signal_noise_ratio(original, reconstructed)
    psnr = peak_snr(original, reconstructed)
    r2 = get_r2(original, reconstructed)
    metrics = {
        'MSE': [mse1],
        'PSNR': [psnr],
        'SNR': [snr_db],
        'R^2': [r2]
    }
    df_metrics = pd.DataFrame(metrics)
    
    return df_metrics

def get_all_metrics(original,denoised_hard,denoised_soft,denoised_garrote):
    df_hard = get_metrics(original,denoised_hard)
    df_soft = get_metrics(original,denoised_soft)
    df_garrote = get_metrics(original,denoised_garrote)
    df_all_metrics = pd.concat([df_hard, df_soft,df_garrote], axis=0)
    df_all_metrics['thresh'] = ['hard','soft','garrote']
    
    return df_all_metrics

def find_best_metrics(df):
    '''
    return a dataframe showing the best values on each metric
    '''
    metrics = []
    best_values = []
    names = []
    threshs = []
    
    for metric in ['MSE', 'PSNR', 'SNR', 'R^2']:
        if metric in ['PSNR', 'SNR','R^2']:
            value = df[metric].max()
            name = df.loc[df[metric].idxmax(), 'name']
            thresh = df.loc[df[metric].idxmax(), 'thresh']
        else:
            value = df[metric].min()
            name = df.loc[df[metric].idxmin(), 'name']
            thresh = df.loc[df[metric].idxmax(), 'thresh']
        metrics.append(metric)
        best_values.append(value)
        names.append(name)
        threshs.append(thresh)
    
    results_df = pd.DataFrame({
        'Metric': metrics,
        'Value': best_values,
        'Name': names,
        'Thresh': threshs
    })
    return results_df

def best_parameter(results_df):
    name = results_df['Name'][0]
    parts = name.split('_')
    best_wavelet = parts[0]
    best_level = int(parts[2])
    return best_wavelet, best_level
    




def get_best_parameters_allsubject(dfs_good,wavelet_candidates,level_candidates):
    ''' return the dataframe listing all subjects' best wavelet and best level
    return a dict of dataframe for each subject showing all detailed results'''
    
    allsubject_metrics_dict = {}
    best_parameter_dict = {}
    
    for subject, df_good in dfs_good.items():
        original_signal = df_good['Glucose'].values
        t =  df_good['Time'].values
        metrics_dict = {}
        for wavelet_choice in wavelet_candidates:
            for level_choice in level_candidates:
                denoised_hard = dwt_denoise(original_signal,wavelet=wavelet_choice, level=level_choice, thresholding='hard')
                denoised_soft = dwt_denoise(original_signal,wavelet=wavelet_choice, level=level_choice, thresholding='soft')
                denoised_garrote = dwt_denoise(original_signal,wavelet=wavelet_choice, level=level_choice, thresholding='garrote')
                name = wavelet_choice +'_level_' + str(level_choice)
                #metrics_dict[name] = get_all_metrics(original_signal,denoised_hard,denoised_soft,denoised_garrote)
                df = get_all_metrics(original_signal,denoised_hard,denoised_soft,denoised_garrote) #[:-1]
                df['name'] = name
                metrics_dict[name] = df
        
        metrics_allresults = pd.concat(metrics_dict.values(), ignore_index=True)
        compare_df = find_best_metrics(metrics_allresults)
        compare_df['subject'] = subject
        allsubject_metrics_dict[subject] = compare_df
        best_wavelet, best_level = best_parameter(compare_df)
        best_parameter_dict[subject] = pd.DataFrame({ 'wavelet': [best_wavelet],'level': [best_level],'subject': [subject]})
    
    metrics_allsubjects = pd.concat(allsubject_metrics_dict.values(), ignore_index=True)
    best_parameters_all = pd.concat(best_parameter_dict.values(), ignore_index=True)

    return  best_parameters_all, allsubject_metrics_dict

def get_best_parameters_onesub(df_good,wavelet_candidates,level_candidates):
    '''return one dataframe for one subject of the best wavelet and level
       return one big dataframe showing all the result for different combinations'''
    
    original_signal = df_good['Glucose'].values
    t =  df_good['Time'].values
    metrics_dict_one = {}
    for wavelet_choice in wavelet_candidates:
        for level_choice in level_candidates:
            denoised_hard = dwt_denoise(original_signal,wavelet=wavelet_choice, level=level_choice, thresholding='hard')
            denoised_soft = dwt_denoise(original_signal,wavelet=wavelet_choice, level=level_choice, thresholding='soft')
            denoised_garrote = dwt_denoise(original_signal,wavelet=wavelet_choice, level=level_choice, thresholding='garrote')
            name = wavelet_choice +'_level_' + str(level_choice)
            #metrics_dict[name] = get_all_metrics(original_signal,denoised_hard,denoised_soft,denoised_garrote)
            df = get_all_metrics(original_signal,denoised_hard,denoised_soft,denoised_garrote) #[:-1]
            df['name'] = name
            metrics_dict_one[name] = df
    
    metrics_allresults = pd.concat(metrics_dict_one.values(), ignore_index=True)
    compare_df = find_best_metrics(metrics_allresults)
    
    return compare_df, metrics_allresults
        


###################### plotting functions
###########################################

def plot_compare(original, reconstructed):
    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(original,linewidth=2, label='original signal',alpha = 0.8)
    ax.plot(reconstructed, label='DWT denoised', linewidth=2,linestyle='--')
    #ax.scatter(x=t,y=original,s=5, label='original signal',marker = 'o')
    #ax.scatter(x=t,y=reconstructed, label='DWT denoised', s=5,marker='v')
    ax.legend()
    ax.set_title('Denoising CGM with DWT', fontsize=18)
    ax.set_ylabel('Glucose', fontsize=16)
    ax.set_xlabel('Time', fontsize=16)
    plt.axhline(y=70, color='red', linestyle='-')
    plt.axhline(y=180, color='orange', linestyle='-')
    plt.legend(loc='upper right',fontsize=10)
    plt.show()
    

def plot_compare_zoom(orginal,final_reconstructed):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    import matplotlib.dates as mdates

    df = orginal.copy()
    original_signal = df['Glucose'].values
    df['Time_only'] = df['Time'].dt.strftime('%H:%M:%S')
    df['Date_only'] = df['Time'].dt.date
    df['Time_only'] = pd.to_datetime(df['Time_only'], format='%H:%M:%S')
    x = df['Time_only']
    # x =  df['Time'].values
    # x = np.arange(len(df['Time']))
    signal1 = original_signal
    signal2 = final_reconstructed
    
    difference = np.abs(signal1 - signal2)

    max_diff_index = np.argmax(difference)
    
    window_size = 30  # Number of points to include around the max difference
    start_index = max(0, max_diff_index - window_size // 2)
    end_index = min(len(x), max_diff_index + window_size // 2)
    
    # Plot the original signals
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(x, signal1, label='Original Values',linewidth = 2, marker='o',markersize = 2)
    ax.plot(x, signal2, label='Reconstructed Values',linewidth=2,linestyle='--', marker='v',markersize=2)
    plt.axhline(y=70, color='red', linestyle='-')
    plt.axhline(y=180, color='orange', linestyle='-')

    ax.set_xticklabels(x, rotation=45)
    myFmt = mdates.DateFormatter('%H:%M')
    ax.xaxis.set_major_formatter(myFmt)
    
    # Highlight the region with the maximum difference
    #ax.axvspan(x[start_index], x[end_index], color='yellow', alpha=0.3, label='Zoomed Area')
    
    # plot the zoomed area
    ax_inset = inset_axes(ax, width="20%", height="30%", loc="upper right")
    ax_inset.plot(x, signal1, label='Original Signal',marker='o',markersize = 2)
    ax_inset.plot(x, signal2, label='Reconstructed Signal',linestyle = "--", marker='v',markersize = 2)
    ax_inset.set_xlim(x[start_index], x[end_index])
    ax_inset.set_ylim(min(signal1[start_index:end_index].min(), signal2[start_index:end_index].min()) - 0.1,
                      max(signal1[start_index:end_index].max(), signal2[start_index:end_index].max()) + 0.1)
    ax_inset.set_title('Zoomed Area')
    plt.gca().set_xticks([])

    mark_inset(ax, ax_inset, loc1=2, loc2=4, fc="none", ec="0.5")

    ax.legend()
    ax.set_xlabel('Time', fontsize = 12)
    ax.set_ylabel('Glucose',fontsize = 12)
    ax.set_title(f'Denoising CGM with DWT for subject {df["id"][0]} on {df['Date_only'][0]}',fontsize = 14)
    ax.legend(loc='lower left',fontsize=10)
    
    plt.show()


def interactive_compare(df,final_reconstructed):
    from bokeh.plotting import figure, show, output_notebook
    from bokeh.layouts import column
    from bokeh.models import BoxZoomTool, ResetTool
    from bokeh.models import Span
    from bokeh.io import output_file
    from bokeh.models import DatetimeTickFormatter

    # output_file("two_line_compare.html")
    
    # Enable output in the notebook
    output_notebook()
    
    original_signal = df['Glucose'].values
    signal1 = original_signal
    signal2 = final_reconstructed
    df['Time'] = pd.to_datetime(df['Time'])
    df['Time_only'] = df['Time'].dt.strftime('%H:%M:%S')
    df['Date_only'] = df['Time'].dt.date
    df['Time_only'] = pd.to_datetime(df['Time_only'], format='%H:%M:%S')
    #x = np.linspace(0, len(signal1),len(signal1))
    x = df['Time_only']

    p = figure(title=f'Denoising CGM with DWT Interactive Plot for subject {df["id"][0]} on {df['Date_only'][0]}', 
               x_axis_label='Time',  y_axis_label='Glucose',
               tools="pan,wheel_zoom,box_zoom,reset", active_drag="box_zoom",width =1000,height=350,
               background_fill_color="#fafafa")
    # x_axis_type='datetime',
    
    # Add the signals to the plot
    p.line(x, signal1, legend_label="Original Values", line_width=2, color="#2ca02c")
    p.scatter(x, signal1,line_color ='#2ca02c', fill_color="#2ca02c", size=3)
    p.line(x, signal2, legend_label="Reconstructed Values", line_width=2, color="#ff7f0e")
    p.scatter(x, signal2,line_color ='#ff7f0e', fill_color="#ff7f0e", size=3)

    p.xaxis.major_label_orientation=1.5
    p.xaxis.formatter=DatetimeTickFormatter(hours="%H:%M", minutes="%H:%M")

    dst_low = Span(location=70, dimension='width',line_color='red', line_width=2)
    dst_high = Span(location=180, dimension='width',line_color='orange', line_width=2)
    p.add_layout(dst_low)
    p.add_layout(dst_high)

    p.add_tools(BoxZoomTool(), ResetTool())
    
    show(p)

def interactive_compare_three(df,noisy_df,final_reconstructed,):
    import numpy as np
    from bokeh.plotting import figure, show, output_notebook
    from bokeh.layouts import column
    from bokeh.models import BoxZoomTool, ResetTool
    from bokeh.models import Span
    from bokeh.models import DatetimeTickFormatter
    
    # Enable output in the notebook
    output_notebook()
    
    original_signal = df['Glucose'].values
    signal1 = original_signal
    signal2 = final_reconstructed
    noisysignal = noisy_df['Glucose'].values
    # x = np.linspace(0, len(signal1),len(signal1))
    df['Time'] = pd.to_datetime(df['Time'])
    df['Time_only'] = df['Time'].dt.strftime('%H:%M:%S')
    df['Date_only'] = df['Time'].dt.date
    df['Time_only'] = pd.to_datetime(df['Time_only'], format='%H:%M:%S')
    #x = np.linspace(0, len(signal1),len(signal1))
    x = df['Time_only']
    
    p = figure(title=f'Denoising CGM with DWT Interactive Plot for subject {df["id"][0]} on {df['Date_only'][0]}', 
               x_axis_label='Time', y_axis_label='Glucose',
               tools="pan,wheel_zoom,box_zoom,reset", active_drag="box_zoom",width =1000,height=350,
               background_fill_color="#fafafa")
    
    # Add the signals to the plot
    p.line(x, noisysignal, legend_label="Noisy Signal", line_width=2, color="khaki")
    p.scatter(x, noisysignal,line_color ='khaki', fill_color="khaki", size=3)
    p.line(x, signal1, legend_label="Original Signal", line_width=2, color="#2ca02c")
    p.scatter(x, signal1,line_color ='#2ca02c', fill_color="#2ca02c", size=3)
    p.line(x, signal2, legend_label="Reconstructed Signal", line_width=2, color="#ff7f0e")
    p.scatter(x, signal2,line_color ='#ff7f0e', fill_color="#ff7f0e", size=3)

    p.xaxis.major_label_orientation=1.5
    p.xaxis.formatter=DatetimeTickFormatter(hours="%H:%M", minutes="%H:%M")


    dst_low = Span(location=70, dimension='width',line_color='red', line_width=2)
    dst_high = Span(location=180, dimension='width',line_color='orange', line_width=2)
    p.add_layout(dst_low)
    p.add_layout(dst_high)

    p.add_tools(BoxZoomTool(), ResetTool())

    show(p)



def interactive_daily_compare(df, daily_reconstructions):
    ''' generate multiple plots for each day at the same time '''
    from bokeh.plotting import figure, show, output_notebook
    from bokeh.layouts import column
    from bokeh.models import BoxZoomTool, ResetTool
    from bokeh.models import Span
    # Enable output in the notebook
    output_notebook()

    date_dfs = split_by_date_onesub(df)
    ps = []

    max_length = max(len(data) for data in date_dfs.values())

    for date, df in date_dfs.items():
        original_signal = df['Glucose'].values
        signal1 = original_signal
        signal2 = daily_reconstructions[date]
        x = np.linspace(0, len(signal1),len(signal1))
        
        # Create a figure
        p = figure(title=f'Denoising CGM with DWT Interactive Plot for subject {df["id"][0]} on {date}', x_axis_label='Time', y_axis_label='Glucose',
                tools="pan,wheel_zoom,box_zoom,reset", active_drag="box_zoom",width =1000,height=350,background_fill_color="#fafafa",x_range=(0, max_length))
        
        # Add the signals to the plot
        p.line(x, signal1, legend_label="Original Signal", line_width=2, color="#2ca02c")
        p.scatter(x, signal1,line_color ='#2ca02c', fill_color="#2ca02c", size=3)
        p.line(x, signal2, legend_label="Reconstructed Signal", line_width=2, color="#ff7f0e")
        p.scatter(x, signal2,line_color ='#ff7f0e', fill_color="#ff7f0e", size=3)

        dst_low = Span(location=70, dimension='width',line_color='red', line_width=2)
        dst_high = Span(location=180, dimension='width',line_color='orange', line_width=2)
        p.add_layout(dst_low)
        p.add_layout(dst_high)

        # Add BoxZoom and Reset tools
        p.add_tools(BoxZoomTool(), ResetTool())

        ps.append(p)

        
    # Show the plot
    show(column(ps))


######################## split data with dates
#################################################

def split_by_date_allsub(dfs_good):
    '''Dictionary to store the DataFrames by date'''
    date_dfs = {}

    for subject, df in dfs_good.items():
        df['Time'] = pd.to_datetime(df['Time'])
        grouped = df.groupby(df['Time'].dt.date)
        for date, group in grouped:
            if date not in date_dfs:
                date_dfs[date] = {}
            date_dfs[date][subject] = group.reset_index(drop=True)

    return date_dfs

def split_by_date_onesub(df):
    '''Dictionary to store the DataFrames by date on one subject'''
    date_dfs = {}

    df['Time'] = pd.to_datetime(df['Time'])
    grouped = df.groupby(df['Time'].dt.date)

    for date, group in grouped:
        if date not in date_dfs:
            date_dfs[date] = {}
        date_dfs[date] = group.reset_index(drop=True)

    return date_dfs

def get_df_by_subject(date_dfs, subject):
    subject_df_list = []
    for date, subject_dict in date_dfs.items():
        if subject in subject_dict:
            subject_df_list.append(subject_dict[subject])
    if subject_df_list:
        return pd.concat(subject_df_list, ignore_index=True)
    else:
        print(f"No data found for subject: {subject}")
        return None
        
################ get the best parameters for all subjects on daily data
#######################################################################

def get_best_parameters_allsubject_alldates(dfs_good, wavelet_candidates, level_candidates):

    date_dfs = split_by_date_allsub(dfs_good)

    allsubject_metrics_dict = {}
    best_parameter_dict = {}

    for date, subject_dfs in date_dfs.items():
        for subject, df_good in subject_dfs.items():
            original_signal = df_good['Glucose'].values
            t = df_good['Time'].values
            metrics_dict = {}

            for wavelet_choice in wavelet_candidates:
                for level_choice in level_candidates:
                    denoised_hard = dwt_denoise(original_signal, wavelet=wavelet_choice, level=level_choice, thresholding='hard')
                    denoised_soft = dwt_denoise(original_signal, wavelet=wavelet_choice, level=level_choice, thresholding='soft')
                    denoised_garrote = dwt_denoise(original_signal, wavelet=wavelet_choice, level=level_choice, thresholding='garrote')
                    name = f'{wavelet_choice}_level_{level_choice}'
                    df = get_all_metrics(original_signal, denoised_hard, denoised_soft, denoised_garrote)
                    df['name'] = name
                    metrics_dict[name] = df

            metrics_allresults = pd.concat(metrics_dict.values(), ignore_index=True)
            compare_df = find_best_metrics(metrics_allresults)
            compare_df['subject'] = subject
            compare_df['date'] = date

            if subject not in allsubject_metrics_dict:
                allsubject_metrics_dict[subject] = []
            allsubject_metrics_dict[subject].append(compare_df)

            best_wavelet, best_level = best_parameter(compare_df)
            if subject not in best_parameter_dict:
                best_parameter_dict[subject] = []
            best_parameter_dict[subject].append(pd.DataFrame({'wavelet': [best_wavelet], 'level': [best_level], 'subject': [subject], 'date': [date]}))

    for subject in allsubject_metrics_dict:
        allsubject_metrics_dict[subject] = pd.concat(allsubject_metrics_dict[subject], ignore_index=True)
    best_parameters_all = pd.concat([pd.concat(value) for value in best_parameter_dict.values()], ignore_index=True)

    return best_parameters_all, allsubject_metrics_dict


########## 

def get_best_parameters_onesubject_alldates(df_good, wavelet_candidates, level_candidates):

    date_dfs = split_by_date_onesub(df_good)

    alldate_metrics_dict = {}
    best_parameter_dict = {}

    for date, df_good in date_dfs.items():
        original_signal = df_good['Glucose'].values
        t = df_good['Time'].values
        metrics_dict = {}

        for wavelet_choice in wavelet_candidates:
            for level_choice in level_candidates:
                denoised_hard = dwt_denoise(original_signal, wavelet=wavelet_choice, level=level_choice, thresholding='hard')
                denoised_soft = dwt_denoise(original_signal, wavelet=wavelet_choice, level=level_choice, thresholding='soft')
                denoised_garrote = dwt_denoise(original_signal, wavelet=wavelet_choice, level=level_choice, thresholding='garrote')
                name = f'{wavelet_choice}_level_{level_choice}'
                df = get_all_metrics(original_signal, denoised_hard, denoised_soft, denoised_garrote)
                df['name'] = name
                metrics_dict[name] = df

        metrics_allresults = pd.concat(metrics_dict.values(), ignore_index=True)
        compare_df = find_best_metrics(metrics_allresults)
        # compare_df['subject'] = subject
        compare_df['date'] = date

        if date not in alldate_metrics_dict:
            alldate_metrics_dict[date] = []
        alldate_metrics_dict[date].append(compare_df)

        best_wavelet, best_level = best_parameter(compare_df)
        if date not in best_parameter_dict:
            best_parameter_dict[date] = []
        best_parameter_dict[date].append(pd.DataFrame({'wavelet': [best_wavelet], 'level': [best_level], 'date': [date]}))

    for date in alldate_metrics_dict:
        alldate_metrics_dict[date] = pd.concat(alldate_metrics_dict[date], ignore_index=True)
    best_parameters_all = pd.concat([pd.concat(value) for value in best_parameter_dict.values()], ignore_index=True)

    return  alldate_metrics_dict,best_parameter_dict


####################### add noises
###################################
def add_noise_to_allglucose(df, sigma):
    noisy_df = df.copy()
    noise = np.random.normal(0, sigma, noisy_df['Glucose'].shape)
    noisy_df['Glucose'] += noise
    return noisy_df

def add_noise_to_spikeglucose(df, sigma):
    noisy_df = df.copy()
    glucose_values = noisy_df['Glucose']
    noise_indices = set()
    
    critical_indices = glucose_values[(glucose_values < 70) | (glucose_values > 180)].index
    #critical_indices = glucose_values[(glucose_values < 70)].index
    
    for idx in critical_indices:
        start_idx = max(0, idx - 5)
        end_idx = min(len(glucose_values), idx + 6)  # Use +6 because range end is exclusive
        noise_indices.update(range(start_idx, end_idx))
    
    noise_indices = list(noise_indices)
    noise = np.random.normal(0, sigma, len(noise_indices))
    noisy_df.loc[noise_indices, 'Glucose'] += noise
    
    return noisy_df

def add_allnoise_to_all_dfs(df_list, sigma):
    noisy_dfs_dict = {}
    for subject, df_good in df_list.items():
        noisy_df = add_noise_to_allglucose(df_good, sigma)
        noisy_dfs_dict[subject] = noisy_df
    return noisy_dfs_dict

def add_noise_to_all_dfs(df_list, sigma):
    noisy_dfs_dict = {}
    for subject, df_good in df_list.items():
        noisy_df = add_noise_to_spikeglucose(df_good, sigma)
        noisy_dfs_dict[subject] = noisy_df
    return noisy_dfs_dict




################################ CWT related
###################################################   

def filter_dfdict_by_record_count(nested_dict, min_records=250):
    filtered_dict = {}
    for date, subjects in nested_dict.items():
        filtered_subjects = {}
        for subject, df in subjects.items():
            if len(df) > min_records:
                filtered_subjects[subject] = df
        if filtered_subjects:
            filtered_dict[date] = filtered_subjects

    return filtered_dict

def merge_dates_for_subjects(df_dict):
    merged_dict = {}
    
    for date, subjects_dict in df_dict.items():
        for subject, df in subjects_dict.items():
            if subject not in merged_dict:
                merged_dict[subject] = []
            merged_dict[subject].append(df)
    
    for subject in merged_dict:
        merged_dict[subject] = pd.concat(merged_dict[subject], ignore_index=True)

    return merged_dict

def denoised_cgm_allsub_daily(allsub_df_dict, all_param_daily_df):
    denoised_signal_allsub = {}
    for date, date_dfs in allsub_df_dict.items():
        final_reconstruct = {}

        for subject, df_good in date_dfs.items():
            original_signal = df_good['Glucose'].values
            condition = (all_param_daily_df['subject'] == subject) # & (all_param_daily_df['date']==date)
            best_params = all_param_daily_df[condition]
            
            if best_params.empty:
                #print(f"No parameters found for subject {subject} on date {date}")
                continue
            
            best_wavelet = best_params.iloc[0]['wavelet']
            best_level = best_params.iloc[0]['level']
            final_reconstructed = dwt_denoise(original_signal,wavelet=best_wavelet, level=best_level, thresholding='hard')
            final_reconstruct[subject]=final_reconstructed
        denoised_signal_allsub[date] = final_reconstruct
    return denoised_signal_allsub
     
    
##### plot one scalogram for the CWT
def plot_scalogram_onesubdate(one_date, subject,coefficients_dict,scales):
    coefficients_one = coefficients_dict[one_date][subject]
    time = np.arange(0,coefficients_dict[one_date][subject].shape[1])
    plt.contourf(time, scales, np.abs(coefficients_one), cmap='Spectral_r')
    plt.colorbar(label='Magnitude')
    plt.title('Contour Diagram of CGM Data')
    plt.xlabel('Time points (every 5 minutes)')
    plt.ylabel('Scale')
    plt.ylim(min(scales), max(scales))  
    #plt.gca().invert_yaxis()  # Invert y-axis to match typical scale representation
    plt.show()


def plot_subject_glucose_histograms(data_dict, target_subject):
    glucose_values_by_segment = {}

    for date, subjects_dict in data_dict.items():
        if target_subject in subjects_dict:
            segments_dict = subjects_dict[target_subject]
            for segment, df in segments_dict.items():
                if segment not in glucose_values_by_segment:
                    glucose_values_by_segment[segment] = []
                glucose_values_by_segment[segment].extend(df['Glucose'].values)

    for segment, glucose_values in glucose_values_by_segment.items():
        plt.figure(figsize=(4, 3))
        plt.hist(glucose_values, bins=20, alpha=0.75, edgecolor='black')
        plt.title(f'Glucose Levels Histogram for Subject {target_subject} during {segment}')
        plt.xlabel('Glucose')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()


def label_glucose(data_dict):
    for date, subjects_dict in data_dict.items():
        for subject, df in subjects_dict.items():
            df['ConsecutiveLow'] = df['Glucose'].rolling(window=3).apply(lambda x: (x < 70).all()).fillna(0)
            if (df['ConsecutiveLow'] == 1).any():
                df['Label'] = 1
            else:
                df['Label'] = 0

            df.drop(columns='ConsecutiveLow', inplace=True)
            
            df['Label'] = df['Label'].max()
    return data_dict

def label_denoised_glucose(data_dict):
    denoised_label_dict = {}
    for date, subjects_dict in data_dict.items():
        mid_dict = {}
        for subject, df in subjects_dict.items():
            new_df = pd.DataFrame(df, columns=['Denoised_Glucose'])
            new_df['ConsecutiveLow'] = new_df['Denoised_Glucose'].rolling(window=3).apply(lambda x: (x < 70).all()).fillna(0)
            
            if (new_df['ConsecutiveLow'] == 1).any():
                new_df['Label'] = 1
            else:
                new_df['Label'] = 0
            new_df.drop(columns='ConsecutiveLow', inplace=True)
            
            mid_dict[subject] = new_df
        
        denoised_label_dict[date] = mid_dict
    
    return denoised_label_dict


def label_summary_df(data_dict):
    label_dfs = []
    for date, subjects_dict in data_dict.items():
        for subject, df in subjects_dict.items():
            if (df['Label'] == 1).any():
                label_dfs.append(pd.DataFrame({'subject': [subject], 'date': [date],'Label': [1]}))
            else:
                label_dfs.append(pd.DataFrame({'subject': [subject], 'date': [date],'Label': [0]}))
                
    label_df = pd.concat(label_dfs, ignore_index=True)
    return label_df


def compare_label_dfs(df1, df2):
    merged_df = pd.merge(df1, df2, on=['subject', 'date'], suffixes=('_df1', '_df2'))
    merged_df['Label_difference'] = merged_df['Label_df1'] != merged_df['Label_df2']
    differences = merged_df[merged_df['Label_difference']]
    return differences





def get_cwt_allsub_daily(denoised_signal_dict, scales, wavelet):
    coefficients_dict = {}
    frequencies_dict = {}
    for date,dfs in denoised_signal_dict.items():
        coefficients = {}
        frequencies = {}
        for subject,df in dfs.items():
            coefficients[subject], frequencies[subject] = pywt.cwt(df, scales, wavelet)
        
        coefficients_dict[date] = coefficients
        frequencies_dict[date]=frequencies

    return coefficients_dict,frequencies_dict

def compute_power_distribution(signal, scales, waveletname='cmor'):
    coefficients, frequencies = pywt.cwt(signal, scales, waveletname)
    power = np.sum(abs(coefficients) ** 2, axis=1)
    return power



####------------------------ Split data into 6-hour segments
####---------------------------------------------------------------

def split_into_6hour_segments(df):
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df = df.dropna(subset=['Time'])
    df.set_index('Time', inplace=True)
    segments = [("00:00", "05:59"), ("06:00", "11:59"), ("12:00", "17:59"), ("18:00", "23:59")]
    segmented_dfs = {}
    for start, end in segments:
        segment_start = start
        segment_end = end
        if segment_end == "00:00":
            segment_end = "23:59:59"
        else:
            segment_end = segment_end
            
        segment_df = df.between_time(segment_start, segment_end).reset_index()
        key = f"{segment_start}-{segment_end}"
        segmented_dfs[key] = segment_df
    
    return segmented_dfs

def split_into_6hour_dict(df_dicts):
    splited_dict = {}
    for date, date_dfs in df_dicts.items():
        middle_dict = {}
        for subject, df in date_dfs.items():
            final = split_into_6hour_segments(df)
            # for i, df in enumerate(df_list):
            #     new_subject = subject if i == 0 else f"{subject}_2"
            #     middle_dict[new_subject] = split_into_6hour_segments(df)
            middle_dict[subject] = final
        splited_dict[date] = middle_dict
    return splited_dict

def combine_3keys(nested_dict):
    combined_dict = {}
    for outer_key, middle_dict in nested_dict.items():
        for middle_key, inner_dict in middle_dict.items():
            for inner_key, value in inner_dict.items():
                combined_key = f"{outer_key}_{middle_key}_{inner_key}"
                combined_dict[combined_key] = value
    return combined_dict

def combine_2keys(nested_dict):
    combined_dict = {}
    for middle_key, inner_dict in nested_dict.items():
        for inner_key, value in inner_dict.items():
            combined_key = f"{middle_key}_{inner_key}"
            combined_dict[combined_key] = value
    return combined_dict

def split_combined_2keys(combined_dict):
    separated_dict = {}
    for combined_key, df in combined_dict.items():
        outer_key, middle_key = combined_key.split('_', 1)
        if outer_key not in separated_dict:
            separated_dict[outer_key] = {}
        others_key = f"{outer_key}_{middle_key}"
        separated_dict[outer_key][others_key] = df
    return separated_dict

def filter_6hourdict_by_record_count(nested_dict, fixed_records=72):
    filtered_dfs = {}
    for key, df in nested_dict.items():
        if len(df) == fixed_records:
            filtered_dfs[key] = df
    return filtered_dfs

def filter_6hourdict_by_more_count(nested_dict, fixed_records=72):
    filtered_dfs = {}
    for key, df in nested_dict.items():
        if len(df) > fixed_records:
            filtered_dfs[key] = df
    return filtered_dfs

def print_subjects_with_suffix_2(df_dicts):
    subjects_with_2_suffix = set() 
    for date, date_dfs in df_dicts.items():
        for subject, timesegment_dict in date_dfs.items():
            subject = str(subject)
            for timesegment in timesegment_dict.keys():
                if subject.endswith('_2'):
                    subjects_with_2_suffix.add(subject)
    return subjects_with_2_suffix

def split_combined_3keys(combined_dict):
    separated_dict = {}
    for combined_key, df in combined_dict.items():
        outer_key, middle_key, inner_key = combined_key.split('_', 2)
        if outer_key not in separated_dict:
            separated_dict[outer_key] = {}
        others_key = f"{outer_key}_{middle_key}_{inner_key}"
        separated_dict[outer_key][others_key] = df
    return separated_dict

def rearrange_subaskey_dict(data_dict):
    new_dict = {}
    for date, subjects_dict in data_dict.items():
        for subject, df in subjects_dict.items():
            if subject not in new_dict:
                new_dict[subject] = df.copy()
            else:
                new_dict[subject] = pd.concat([new_dict[subject], df], ignore_index=True)
    return new_dict

def print_all_2keys(df_dicts):
    multiple_chunk = []
    for date, date_dfs in df_dicts.items():
        print(f"Date: {date}")
        for subject in date_dfs.keys():
            print(f"  Subject: {subject}")
            parts = subject.split('_')  # Split subject by underscore
            if len(parts) >= 3 and (parts[-1] == '2' or parts[-1].startswith('2-')):
                multiple_chunk.append(subject)
            elif len(parts) >= 3 and (parts[-2] == '2' or parts[-2].startswith('2-')):
                multiple_chunk.append(subject)
    return multiple_chunk

def print_keys(df_dict):
    for date, date_dfs in df_dict.items():
        print(f"Date: {date}")
        for subject in date_dfs.keys():
            print(f"  Subject: {subject}")

def extract_glucose(allsub_6hour_2keydict):
    glucoses= {}
    for key, dict in allsub_6hour_2keydict.items():
        middle_dict = {}
        for subject, df in dict.items():
            middle_dict[subject] = df['Glucose'].values
        glucoses[key] = middle_dict
    return glucoses


def split_into_8hour_segments(df):
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df = df.dropna(subset=['Time'])
    df.set_index('Time', inplace=True)
    segments = [("00:00", "07:59"), ("08:00", "15:59"), ("16:00", "23:59")]
    segmented_dfs = {}
    for start, end in segments:
        segment_start = start
        segment_end = end
        if segment_end == "00:00":
            segment_end = "23:59:59"
        else:
            segment_end = segment_end
            
        segment_df = df.between_time(segment_start, segment_end).reset_index()
        key = f"{segment_start}-{segment_end}"
        segmented_dfs[key] = segment_df
    
    return segmented_dfs

def split_into_8hour_dict(df_dicts):
    splited_dict = {}
    for date, date_dfs in df_dicts.items():
        middle_dict = {}
        for subject, df in date_dfs.items():
            final = split_into_8hour_segments(df)
            # for i, df in enumerate(df_list):
            #     new_subject = subject if i == 0 else f"{subject}_2"
            #     middle_dict[new_subject] = split_into_6hour_segments(df)
            middle_dict[subject] = final
        splited_dict[date] = middle_dict
    return splited_dict

def metricdf_by_timesegment(allsub_6hour_2keydict,denoised_signal_dict,org_dict_labels,denoised_dict_labels):
    subjects = []
    mse_values = []
    psnr_values = []
    snr_values = []
    org_labels = []
    denoise_labels = []
    for date, dict in allsub_6hour_2keydict.items():
        for subject, df in dict.items():
            mse_1= get_metrics(df['Glucose'],denoised_signal_dict[date][subject])['MSE'].values 
            psnr_1= get_metrics(df['Glucose'],denoised_signal_dict[date][subject])['PSNR'].values
            snr_1= get_metrics(df['Glucose'],denoised_signal_dict[date][subject])['SNR'].values
            subjects.append(subject)
            mse_values.append(np.round(mse_1,6))
            psnr_values.append(np.round(psnr_1,6))
            snr_values.append(np.round(snr_1,6))
            org_labels.append(org_dict_labels[date][subject]['Label'][0])
            denoise_labels.append(denoised_dict_labels[date][subject]['Label'][0])

    result_df = pd.DataFrame({
        'subject': subjects,
        'MSE_values': mse_values,
        'PSNR_values': psnr_values,
        'SNR_values': snr_values,
        'org_label':org_labels,
        'denoise_label':denoise_labels
    })        
    return result_df

def summarydf_timesegment(df):
    df['time_segment'] = df['subject'].apply(lambda x: x.split('_')[-1])
    result = df.groupby(['time_segment','denoise_label']).agg( mean_MSE=('MSE_values', 'mean'), mean_PSNR=('PSNR_values', 'mean'), mean_SNR=('SNR_values', 'mean'),
                                                              count=('MSE_values', 'size')).reset_index() #,
                                            
    result2 = df.groupby(['time_segment']).agg(mean_MSE=('MSE_values', 'mean'), mean_PSNR=('PSNR_values', 'mean'), mean_SNR=('SNR_values', 'mean'),
                                               org_1s = ('org_label','sum'), denoise_1s = ('denoise_label','sum'), count=('MSE_values', 'size')).reset_index()
    return result, result2

def split_into_4hour_segments(df):
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df = df.dropna(subset=['Time'])
    df.set_index('Time', inplace=True)
    segments = [("00:00", "03:59"), ("04:00", "7:59"), ("8:00", "11:59"),
                ("12:00","15:59"),("16:00","19:59"),("20:00","23:59")]
    segmented_dfs = {}
    for start, end in segments:
        segment_start = start
        segment_end = end
        if segment_end == "00:00":
            segment_end = "23:59:59"
        else:
            segment_end = segment_end
            
        segment_df = df.between_time(segment_start, segment_end).reset_index()
        key = f"{segment_start}-{segment_end}"
        segmented_dfs[key] = segment_df
    
    return segmented_dfs

def split_into_4hour_dict(df_dicts):
    splited_dict = {}
    for date, date_dfs in df_dicts.items():
        middle_dict = {}
        for subject, df in date_dfs.items():
            final = split_into_4hour_segments(df)
            # for i, df in enumerate(df_list):
            #     new_subject = subject if i == 0 else f"{subject}_2"
            #     middle_dict[new_subject] = split_into_6hour_segments(df)
            middle_dict[subject] = final
        splited_dict[date] = middle_dict
    return splited_dict



def plot_TR_bar(df):
    conditions = [
        (df['Glucose'] < 54),
        (df['Glucose'] >= 54) & (df['Glucose'] <= 69),
        (df['Glucose'] >= 70) & (df['Glucose'] <= 180),
        (df['Glucose'] >= 181) & (df['Glucose'] <= 250),
        (df['Glucose'] > 250)
    ]
    choices = ['Very Low (<54 mg/dL)', 'Low (54-69  mg/dL)', 'Target Range (70-180  mg/dL)',
            'High (181-250  mg/dL)','Very High (>250  mg/dL)']
    df['Glucose Category'] = pd.cut(df['Glucose'], bins=[-float('inf'), 54, 70, 180, 250, float('inf')],
                                    labels=choices, right=False)
    category_counts = df['Glucose Category'].value_counts(normalize=True).sort_index() * 100

    stacked_data = category_counts.values.reshape(1, -1)
    categories = category_counts.index

    plt.figure(figsize=(2, 5))
    bar_width = 0.2  # Adjust the bar width here
    bottom = np.zeros(stacked_data.shape[0])
    colors_list = ['darkred','red', 'green', 'gold', 'darkorange']

    for i, category in enumerate(categories):
        plt.bar('Glucose Categories', stacked_data[:, i], bottom=bottom, width=bar_width,
                 label=category,color = colors_list[i],edgecolor = "black",linewidth=0.5)
        bottom += stacked_data[:, i]

        for j in range(stacked_data.shape[0]):
            height = stacked_data[j, i]
            plt.text('Glucose Categories', bottom[j] - height / 2, f'{height:.1f}%', ha='center', va='center', color='white', fontsize=10)

    #plt.xlabel('Glucose Categories')
    plt.ylabel('Frequency Percentage (%)')
    plt.title('Distribution of Glucose Values')

    handles, labels = plt.gca().get_legend_handles_labels() 
    order = [4, 3,2, 1, 0] 
    plt.legend([handles[i] for i in order], [labels[i] for i in order],title='Glucose Category',loc='center left', bbox_to_anchor=(1, 0.5),
               fontsize=10,labelspacing=2,frameon=False)
    
    for pos in ['right', 'top']: 
        plt.gca().spines[pos].set_visible(False) 
    plt.show()

def plot_TR_bar_reconstructed(reconstructed):
    final_reconstructed_df = pd.DataFrame()
    final_reconstructed_df['Glucose'] = reconstructed
    plot_TR_bar(final_reconstructed_df)

####-------------------------- CNN related ---------------------------
####------------------------------------------------------------------
'''
Prepare the data into good size as CNN input
'''
# def resize_coefficients(coefficients, size=(128, 128)):
#     resized_coeffs = cv2.resize(coefficients, size, interpolation=cv2.INTER_LINEAR)
#     return resized_coeffs
def pad_coefficients(coefficients, target_size=(128, 128)):
    current_size = coefficients.shape
    pad_width = ((0, max(0, target_size[0] - current_size[0])),
                 (0, max(0, target_size[1] - current_size[1])))
    padded_coeffs = np.pad(coefficients, pad_width, mode='constant', constant_values=0)
    return padded_coeffs

def coefficients_to_rgb(coefficients):
    norm_coeffs = (coefficients - np.min(coefficients)) / (np.max(coefficients) - np.min(coefficients))
    rgb_coeffs = np.stack((norm_coeffs,) * 3, axis=-1)
    return rgb_coeffs

def prepare_input_net(coefficients, target_size=(128, 128)):
    padded_coeffs = pad_coefficients(coefficients, target_size)
    rgb_coeffs = coefficients_to_rgb(padded_coeffs)
    return rgb_coeffs




####---------------------- plot simulation result------------------------
####---------------------------------------------------------------------
def plot_sim_result(sim_result,filename=None):
    df_expanded = sim_result.explode(['MSE_changes', 'PSNR_changes'])
    df_expanded['SNR_changes'] = df_expanded['SNR_changes'].apply(lambda x:x[1:9] )
    df_expanded['SNR_changes'] = pd.to_numeric(df_expanded['SNR_changes'])

    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(16, 8))

    axs[0, 0].plot(df_expanded['noise_levels'], df_expanded['SNR_changes'], label=r"$\Delta$ SNR",linewidth=2,color = '#ff7f0e')
    #axs[0, 0].plot(df_expanded['noise_levels'], df_expanded['MSE_changes'], label=r"$\Delta$ MSE",linewidth=2,color = '#2ca02c')
    #axs[0, 0].plot(df_expanded['noise_levels'], df_expanded['PSNR_changes'], label=r"$\Delta$ PSNR",linewidth=2,color = 'yellow')
    axs[0, 0].set_xlabel('Noise Levels')
    axs[0, 0].set_ylabel('SNR changes')
    axs[0, 0].set_title('SNR performance')
    axs[0,0].tick_params(axis="y", labelcolor='#ff7f0e')
    axs[0, 0].legend(fontsize = 9)

    axs2=axs[0, 0].twinx()
    axs2.plot(df_expanded['noise_levels'], df_expanded['SNR_out'], label='SNR after denoising',linewidth=2,color = "#2ca02c")
    axs2.fill_between(df_expanded['noise_levels'], df_expanded['snr_out_lower'], df_expanded['snr_out_upper'], color='#2ca02c',linewidth = 0, alpha=0.1)
    axs2.set_ylabel('SNR values')
    axs2.tick_params(axis="y", labelcolor='#2ca02c')
    axs2.legend(fontsize = 9)

    axs[0, 0].set_axisbelow(True)
    axs[0, 0].yaxis.grid(color='lightgray', linestyle='dotted')
    axs[0, 0].xaxis.grid(color='lightgray', linestyle='dotted')

    axs[0, 2].plot(df_expanded['noise_levels'], df_expanded['cv_org'], label='CV Original',linestyle = "dashed",linewidth=2)
    axs[0, 2].plot(df_expanded['noise_levels'], df_expanded['cv_noise'], label='CV Noisy',linewidth=2)
    axs[0, 2].plot(df_expanded['noise_levels'], df_expanded['cv_denoise'], label='CV Denoised',linewidth=2)
    axs[0, 2].fill_between(df_expanded['noise_levels'], df_expanded['cv1_lower'], df_expanded['cv1_upper'], color='#ff7f0e', linewidth = 0,alpha=0.1)
    axs[0, 2].fill_between(df_expanded['noise_levels'], df_expanded['cv2_lower'], df_expanded['cv2_upper'], color='#2ca02c',linewidth = 0, alpha=0.1)
    axs[0, 2].set_xlabel('Noise Levels')
    axs[0, 2].set_ylabel('Value in %')
    axs[0, 2].set_title('Coefficient of Variation (CV)')
    axs[0, 2].legend()
    axs[0, 2].set_axisbelow(True)
    axs[0, 2].yaxis.grid(color='lightgray', linestyle='dotted')
    axs[0, 2].xaxis.grid(color='lightgray', linestyle='dotted')
    

    axs[0, 1].plot(df_expanded['noise_levels'], df_expanded['sd_org'], label='SD Original',linestyle = "dashed",linewidth=2)
    axs[0, 1].plot(df_expanded['noise_levels'], df_expanded['sd_noise'], label='SD Noisy',linewidth=2)
    axs[0, 1].plot(df_expanded['noise_levels'], df_expanded['sd_denoise'], label='SD Denoised',linewidth=2)
    axs[0, 1].fill_between(df_expanded['noise_levels'], df_expanded['sd1_lower'], df_expanded['sd1_upper'], color='#ff7f0e', linewidth = 0,alpha=0.1)
    axs[0, 1].fill_between(df_expanded['noise_levels'], df_expanded['sd2_lower'], df_expanded['sd2_upper'], color='#2ca02c',linewidth = 0, alpha=0.1)
    axs[0, 1].set_xlabel('Noise Levels')
    axs[0, 1].set_ylabel('Value')
    axs[0, 1].set_title('Standard Deviation (SD)')
    axs[0, 1].legend()
    axs[0, 1].set_axisbelow(True)
    axs[0, 1].yaxis.grid(color='lightgray', linestyle='dotted')
    axs[0, 1].xaxis.grid(color='lightgray', linestyle='dotted')

    axs[0, 3].plot(df_expanded['noise_levels'], df_expanded['mage_org'], label='MAGE Original',linestyle = "dashed",linewidth=2)
    axs[0, 3].plot(df_expanded['noise_levels'], df_expanded['mage_noise'], label='MAGE Noisy',linewidth=2)
    axs[0, 3].plot(df_expanded['noise_levels'], df_expanded['mage_denoise'], label='MAGE Denoised',linewidth=2)
    axs[0, 3].fill_between(df_expanded['noise_levels'], df_expanded['mage1_lower'], df_expanded['mage1_upper'], color='#ff7f0e', linewidth = 0,alpha=0.1)
    axs[0, 3].fill_between(df_expanded['noise_levels'], df_expanded['mage2_lower'], df_expanded['mage2_upper'], color='#2ca02c',linewidth = 0, alpha=0.1)
    axs[0, 3].set_xlabel('Noise Levels')
    axs[0, 3].set_ylabel('Value')
    axs[0, 3].set_title('Mean Amplitude of Glycemic Excursions')
    axs[0, 3].legend()
    axs[0, 3].set_axisbelow(True)
    axs[0, 3].yaxis.grid(color='lightgray', linestyle='dotted')
    axs[0, 3].xaxis.grid(color='lightgray', linestyle='dotted')

    axs[1, 1].plot(df_expanded['noise_levels'], df_expanded['hamming_distance1'], label='Dist of Orginal and Noisy',linewidth=2,color = '#ff7f0e')
    axs[1, 1].plot(df_expanded['noise_levels'], df_expanded['hamming_distance2'], label='Dist of Orginal and Denoised',linewidth=2,color = '#1f77b4')
    axs[1, 1].fill_between(df_expanded['noise_levels'], df_expanded['hamming_distance1s_lower'], df_expanded['hamming_distance1s_upper'], color='#ff7f0e',linewidth = 0, alpha=0.1)
    axs[1, 1].fill_between(df_expanded['noise_levels'], df_expanded['hamming_distance2s_lower'], df_expanded['hamming_distance2s_upper'], color='#1f77b4',linewidth = 0, alpha=0.1)
    axs[1, 1].set_xlabel('Noise Levels')
    axs[1, 1].set_ylabel('Value')
    axs[1, 1].set_title('Hamming Distances')
    axs[1, 1].legend()
    axs[1, 1].set_axisbelow(True)
    axs[1, 1].yaxis.grid(color='lightgray', linestyle='dotted')
    axs[1, 1].xaxis.grid(color='lightgray', linestyle='dotted')

    axs[1, 2].plot(df_expanded['noise_levels'], df_expanded['jaccard_distance1'], label='Dist of Orginal and Noisy',linewidth=2,color = '#ff7f0e')
    axs[1, 2].plot(df_expanded['noise_levels'], df_expanded['jaccard_distance2'], label='Dist of Orginal and Denoised',linewidth=2,color = '#1f77b4')
    axs[1, 2].fill_between(df_expanded['noise_levels'], df_expanded['jaccard_distance1s_lower'], df_expanded['jaccard_distance1s_upper'], color='#ff7f0e',linewidth = 0, alpha=0.1)
    axs[1, 2].fill_between(df_expanded['noise_levels'], df_expanded['jaccard_distance2s_lower'], df_expanded['jaccard_distance2s_upper'], color='#1f77b4',linewidth = 0, alpha=0.1)
    axs[1, 2].set_xlabel('Noise Levels')
    axs[1, 2].set_ylabel('Value')
    axs[1, 2].set_title('Jaccard Distances')
    axs[1, 2].legend()
    axs[1, 2].set_axisbelow(True)
    axs[1, 2].yaxis.grid(color='lightgray', linestyle='dotted')
    axs[1, 2].xaxis.grid(color='lightgray', linestyle='dotted')

    axs[1, 3].plot(df_expanded['noise_levels'], df_expanded['rates_0_to_1'], label='0_to_1',linewidth=2,color='#e9c46a')
    axs[1, 3].plot(df_expanded['noise_levels'], df_expanded['rates_1_to_0'], label='1_to_0',linewidth=2,color='#f4a261')
    axs[1, 3].fill_between(df_expanded['noise_levels'], df_expanded['rates_0_to_1_lower'], df_expanded['rates_0_to_1_upper'], color='#e9c46a', linewidth = 0,alpha=0.1)
    axs[1, 3].fill_between(df_expanded['noise_levels'], df_expanded['rates_1_to_0_lower'], df_expanded['rates_1_to_0_upper'], color='#f4a261',linewidth = 0, alpha=0.1)
    
    axs2=axs[1, 3].twinx()
    axs2.plot(df_expanded['noise_levels'], df_expanded['rates_0_to_0'], label='0_to_0',linewidth=2,color='forestgreen') #264653
    axs2.plot(df_expanded['noise_levels'], df_expanded['rates_1_to_1'], label='1_to_1',linewidth=2,color='#2a9d8f')
    axs2.fill_between(df_expanded['noise_levels'], df_expanded['rates_0_to_0_lower'], df_expanded['rates_0_to_0_upper'], color='forestgreen', linewidth = 0,alpha=0.1)
    axs2.fill_between(df_expanded['noise_levels'], df_expanded['rates_1_to_1_lower'], df_expanded['rates_1_to_1_upper'], color='#2a9d8f',linewidth = 0, alpha=0.1)
    axs2.set_ylabel('Rate')
    axs2.tick_params(axis="y", labelcolor='forestgreen')
    

    axs[1, 3].set_xlabel('Noise Levels')
    axs[1, 3].set_ylabel('Rate')
    axs[1, 3].set_title('Transition Rates of Events')
    axs[1, 3].tick_params(axis="y", labelcolor='#f4a261')
    axs[1, 3].legend()
    axs2.legend(loc = 4)
    axs[1, 3].set_axisbelow(True)
    axs[1, 3].yaxis.grid(color='lightgray', linestyle='dotted')
    axs[1, 3].xaxis.grid(color='lightgray', linestyle='dotted')

    #axs[1, 0].plot(df_expanded['noise_levels'], df_expanded['MSE_out'], label='MSE after denoising',linewidth=2,color = '#2ca02c')
    #axs[1, 0].plot(df_expanded['noise_levels'], df_expanded['SNR_out'], label='SNR after denoising',linewidth=2,color = "#ff7f0e")
    axs[1, 0].plot(df_expanded['noise_levels'], df_expanded['MSE_in'], label='MSE before denoising',linewidth=2,color = '#ff7f0e')
    axs[1, 0].plot(df_expanded['noise_levels'], df_expanded['MSE_out'], label='MSE after denoising',linewidth=2,color = '#2ca02c')
    #axs[1, 0].plot(df_expanded['noise_levels'], df_expanded['SNR_in'], label='SNR before denoising',linewidth=2,color = "red")
    #axs[1, 0].plot(df_expanded['noise_levels'], df_expanded['MSE_in'], label='mse before denoising',linewidth=2,color = "yellow")
    #axs[1, 0].plot(df_expanded['noise_levels'], df_expanded['PSNR_in'], label='pSNR before denoising',linewidth=2,color = "pink")
    #axs[1, 0].plot(df_expanded['noise_levels'], df_expanded['PSNR_out'], label='pSNR after denoising',linewidth=2,color = "brown")
    #axs[1, 0].fill_between(df_expanded['noise_levels'], df_expanded['snr_out_lower'], df_expanded['snr_out_upper'], color='#ff7f0e',linewidth = 0, alpha=0.1)
    axs[1, 0].fill_between(df_expanded['noise_levels'], df_expanded['mse_out_lower'], df_expanded['mse_out_upper'], color='#2ca02c',linewidth = 0, alpha=0.1)
    axs[1, 0].fill_between(df_expanded['noise_levels'], df_expanded['mse_in_lower'], df_expanded['mse_in_upper'], color='#ff7f0e',linewidth = 0, alpha=0.1)
    axs[1, 0].set_xlabel('Noise Levels')
    axs[1, 0].set_ylabel('Value')
    axs[1, 0].set_title('MSE before and after Denoising')
    axs[1, 0].legend()
    axs[1, 0].set_axisbelow(True)
    axs[1, 0].yaxis.grid(color='lightgray', linestyle='dotted')
    axs[1, 0].xaxis.grid(color='lightgray', linestyle='dotted')

    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()
    plt.close(fig)


def plot_sim_variance(sim_result,filename=None):
    df_expanded = sim_result.explode(['MSE_changes', 'PSNR_changes'])

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

    axs[3].plot(df_expanded['noise_levels'], df_expanded['mage_org'], label='MAGE Original',linestyle = "dashed",linewidth=2)
    axs[3].plot(df_expanded['noise_levels'], df_expanded['mage_noise'], label='MAGE Noisy',linewidth=2)
    axs[3].plot(df_expanded['noise_levels'], df_expanded['mage_denoise'], label='MAGE Denoised',linewidth=2)
    axs[3].fill_between(df_expanded['noise_levels'], df_expanded['mage1_lower'], df_expanded['mage1_upper'], color='#ff7f0e', linewidth = 0,alpha=0.1)
    axs[3].fill_between(df_expanded['noise_levels'], df_expanded['mage2_lower'], df_expanded['mage2_upper'], color='#2ca02c',linewidth = 0, alpha=0.1)
    axs[3].set_xlabel('Noise Levels')
    axs[3].set_ylabel('Value')
    axs[3].set_title('Mean Amplitude of Glycemic Excursions')
    axs[3].legend()
    axs[3].set_axisbelow(True)
    axs[3].yaxis.grid(color='lightgray', linestyle='dotted')
    axs[3].xaxis.grid(color='lightgray', linestyle='dotted')

    axs[1].plot(df_expanded['noise_levels'], df_expanded['sd_org'], label='SD Original',linestyle = "dashed",linewidth=2)
    axs[1].plot(df_expanded['noise_levels'], df_expanded['sd_noise'], label='SD Noisy',linewidth=2)
    axs[1].plot(df_expanded['noise_levels'], df_expanded['sd_denoise'], label='SD Denoised',linewidth=2)
    axs[1].fill_between(df_expanded['noise_levels'], df_expanded['sd1_lower'], df_expanded['sd1_upper'], color='#ff7f0e', linewidth = 0,alpha=0.1)
    axs[1].fill_between(df_expanded['noise_levels'], df_expanded['sd2_lower'], df_expanded['sd2_upper'], color='#2ca02c',linewidth = 0, alpha=0.1)
    axs[1].set_xlabel('Noise Levels')
    axs[1].set_ylabel('Value')
    axs[1].set_title('Standard Deviation')
    axs[1].legend()
    axs[1].set_axisbelow(True)
    axs[1].yaxis.grid(color='lightgray', linestyle='dotted')
    axs[1].xaxis.grid(color='lightgray', linestyle='dotted')

    axs[2].plot(df_expanded['noise_levels'], df_expanded['cv_org'], label='CV Original',linestyle = "dashed",linewidth=2)
    axs[2].plot(df_expanded['noise_levels'], df_expanded['cv_noise'], label='CV Noisy',linewidth=2)
    axs[2].plot(df_expanded['noise_levels'], df_expanded['cv_denoise'], label='CV Denoised',linewidth=2)
    axs[2].fill_between(df_expanded['noise_levels'], df_expanded['cv1_lower'], df_expanded['cv1_upper'], color='#ff7f0e', linewidth = 0,alpha=0.1)
    axs[2].fill_between(df_expanded['noise_levels'], df_expanded['cv2_lower'], df_expanded['cv2_upper'], color='#2ca02c',linewidth = 0, alpha=0.1)
    axs[2].set_xlabel('Noise Levels')
    axs[2].set_ylabel('Value in %')
    axs[2].set_title('Coefficient of Variation')
    axs[2].legend()
    axs[2].set_axisbelow(True)
    axs[2].yaxis.grid(color='lightgray', linestyle='dotted')
    axs[2].xaxis.grid(color='lightgray', linestyle='dotted')

    axs[0].plot(df_expanded['noise_levels'], df_expanded['eHbA1c_org'], label='eHbA1c Original',linestyle = "dashed",linewidth=2)
    axs[0].plot(df_expanded['noise_levels'], df_expanded['eHbA1c_noise'], label='eHbA1c Noisy',linewidth=2)
    axs[0].plot(df_expanded['noise_levels'], df_expanded['eHbA1c_denoise'], label='eHbA1c Denoised',linewidth=2)
    axs[0].fill_between(df_expanded['noise_levels'], df_expanded['eHbA1c1_lower'], df_expanded['eHbA1c1_upper'], color='#ff7f0e', linewidth = 0,alpha=0.1)
    axs[0].fill_between(df_expanded['noise_levels'], df_expanded['eHbA1c2_lower'], df_expanded['eHbA1c2_upper'], color='#2ca02c',linewidth = 0, alpha=0.1)
    axs[0].set_xlabel('Noise Levels')
    axs[0].set_ylabel('Value')
    axs[0].set_title('eHbA1c')
    axs[0].legend()
    axs[0].set_axisbelow(True)
    axs[0].yaxis.grid(color='lightgray', linestyle='dotted')
    axs[0].xaxis.grid(color='lightgray', linestyle='dotted')

    plt.tight_layout()

    if filename:
        plt.savefig(filename)

    plt.show()
    plt.close(fig)


def plot_sim_distance(sim_result,filename=None):
    df_expanded = sim_result.explode(['MSE_changes', 'PSNR_changes'])

    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

    axs[0].plot(df_expanded['noise_levels'], df_expanded['hamming_distance1'], label='Dist of Orginal and Noisy',linewidth=2,color = '#ff7f0e')
    axs[0].plot(df_expanded['noise_levels'], df_expanded['hamming_distance2'], label='Dist of Orginal and Denoised',linewidth=2,color = '#1f77b4')
    axs[0].fill_between(df_expanded['noise_levels'], df_expanded['hamming_distance1s_lower'], df_expanded['hamming_distance1s_upper'], color='#ff7f0e',linewidth = 0, alpha=0.1)
    axs[0].fill_between(df_expanded['noise_levels'], df_expanded['hamming_distance2s_lower'], df_expanded['hamming_distance2s_upper'], color='#1f77b4',linewidth = 0, alpha=0.1)
    axs[0].set_xlabel('Noise Levels')
    axs[0].set_ylabel('Value')
    axs[0].set_title('Hamming Distances')
    axs[0].legend()
    axs[0].set_axisbelow(True)
    axs[0].yaxis.grid(color='lightgray', linestyle='dotted')
    axs[0].xaxis.grid(color='lightgray', linestyle='dotted')

    axs[1].plot(df_expanded['noise_levels'], df_expanded['jaccard_distance1'], label='Dist of Orginal and Noisy',linewidth=2,color = '#ff7f0e')
    axs[1].plot(df_expanded['noise_levels'], df_expanded['jaccard_distance2'], label='Dist of Orginal and Denoised',linewidth=2,color = '#1f77b4')
    axs[1].fill_between(df_expanded['noise_levels'], df_expanded['jaccard_distance1s_lower'], df_expanded['jaccard_distance1s_upper'], color='#ff7f0e',linewidth = 0, alpha=0.1)
    axs[1].fill_between(df_expanded['noise_levels'], df_expanded['jaccard_distance2s_lower'], df_expanded['jaccard_distance2s_upper'], color='#1f77b4',linewidth = 0, alpha=0.1)
    axs[1].set_xlabel('Noise Levels')
    axs[1].set_ylabel('Value')
    axs[1].set_title('Jaccard Distances')
    axs[1].legend()
    axs[1].set_axisbelow(True)
    axs[1].yaxis.grid(color='lightgray', linestyle='dotted')
    axs[1].xaxis.grid(color='lightgray', linestyle='dotted')

    axs[2].plot(df_expanded['noise_levels'], df_expanded['rates_0_to_1'], label='Rates 0_to_1',linewidth=2,color='#e9c46a')
    axs[2].plot(df_expanded['noise_levels'], df_expanded['rates_1_to_0'], label='Rates 1_to_0',linewidth=2, color='#f4a261')
    axs[2].fill_between(df_expanded['noise_levels'], df_expanded['rates_0_to_1_lower'], df_expanded['rates_0_to_1_upper'], color='#e9c46a', linewidth = 0,alpha=0.1)
    axs[2].fill_between(df_expanded['noise_levels'], df_expanded['rates_1_to_0_lower'], df_expanded['rates_1_to_0_upper'], color='#f4a261',linewidth = 0, alpha=0.1)
    axs[2].set_xlabel('Noise Levels')
    axs[2].set_ylabel('Value')
    axs[2].set_title('Transition Rates of Events')
    axs[2].legend()
    axs[2].set_axisbelow(True)
    axs[2].yaxis.grid(color='lightgray', linestyle='dotted')
    axs[2].xaxis.grid(color='lightgray', linestyle='dotted')

    axs[3].plot(df_expanded['noise_levels'], df_expanded['rates_0_to_0'], label='Rates 0_to_0',linewidth=2,color='forestgreen')
    axs[3].plot(df_expanded['noise_levels'], df_expanded['rates_1_to_1'], label='Rates 1_to_1',linewidth=2, color='#2a9d8f')
    axs[3].fill_between(df_expanded['noise_levels'], df_expanded['rates_0_to_0_lower'], df_expanded['rates_0_to_0_upper'], color='forestgreen', linewidth = 0,alpha=0.1)
    axs[3].fill_between(df_expanded['noise_levels'], df_expanded['rates_1_to_1_lower'], df_expanded['rates_1_to_1_upper'], color='#2a9d8f',linewidth = 0, alpha=0.1)
    axs[3].set_xlabel('Noise Levels')
    axs[3].set_ylabel('Value')
    axs[3].set_title('Remaining Rates of Events')
    axs[3].legend()
    axs[3].set_axisbelow(True)
    axs[3].yaxis.grid(color='lightgray', linestyle='dotted')
    axs[3].xaxis.grid(color='lightgray', linestyle='dotted')

    plt.tight_layout()

    if filename:
        plt.savefig(filename)

    plt.show()
    plt.close(fig)

def plot_sim_msesnr(sim_result,filename=None):

    df_expanded = sim_result.explode(['MSE_changes', 'PSNR_changes'])
    df_expanded['SNR_changes'] = df_expanded['SNR_changes'].apply(lambda x:x[1:9] )
    df_expanded['SNR_changes'] = pd.to_numeric(df_expanded['SNR_changes'])

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    axs[0].plot(df_expanded['noise_levels'], df_expanded['MSE_in'], label='MSE before denoising',linewidth=2,color = '#ff7f0e')
    axs[0].plot(df_expanded['noise_levels'], df_expanded['MSE_out'], label='MSE after denoising',linewidth=2,color = '#2ca02c')
    axs[0].fill_between(df_expanded['noise_levels'], df_expanded['mse_out_lower'], df_expanded['mse_out_upper'], color='#2ca02c',linewidth = 0, alpha=0.1)
    axs[0].fill_between(df_expanded['noise_levels'], df_expanded['mse_in_lower'], df_expanded['mse_in_upper'], color='#ff7f0e',linewidth = 0, alpha=0.1)
    #axs[0].fill_between(df_expanded['noise_levels'], df_expanded['rates_0_to_1_lower'], df_expanded['rates_0_to_1_upper'], color='#1f77b4', linewidth = 0,alpha=0.1)
    #axs[0].fill_between(df_expanded['noise_levels'], df_expanded['rates_1_to_0_lower'], df_expanded['rates_1_to_0_upper'], color='#ff7f0e',linewidth = 0, alpha=0.1)
    axs[0].set_xlabel('Noise Levels')
    axs[0].set_ylabel('Value')
    axs[0].set_title('MSE before and after Denoising')
    axs[0].legend()
    axs[0].set_axisbelow(True)
    axs[0].yaxis.grid(color='lightgray', linestyle='dotted')
    axs[0].xaxis.grid(color='lightgray', linestyle='dotted')

    
    axs[1].plot(df_expanded['noise_levels'], df_expanded['SNR_changes'], label=r"$\Delta$ SNR",linewidth=2,color = 'tomato')
    #axs[1].plot(df_expanded['noise_levels'], df_expanded['MSE_changes'], label=r"$\Delta$ MSE",linewidth=2,color = '#2ca02c')
    #axs[1]].plot(df_expanded['noise_levels'], df_expanded['PSNR_changes'], label=r"$\Delta$ PSNR",linewidth=2,color = 'yellow')
    axs2=axs[1].twinx()
    axs2.plot(df_expanded['noise_levels'], df_expanded['SNR_out'], label='SNR after denoising',linewidth=2,color = "#2ca02c")
    axs2.fill_between(df_expanded['noise_levels'], df_expanded['snr_out_lower'], df_expanded['snr_out_upper'], color='#2ca02c',linewidth = 0, alpha=0.1)
    axs2.set_ylabel('SNR values')
    axs2.tick_params(axis="y", labelcolor='#2ca02c')
    axs2.legend()
    #axs[1].plot(df_expanded['noise_levels'], df_expanded['SNR_out'], label='SNR after denoising',linewidth=2,color = "#2ca02c")
    #axs[1].fill_between(df_expanded['noise_levels'], df_expanded['snr_out_lower'], df_expanded['snr_out_upper'], color='#2ca02c',linewidth = 0, alpha=0.1)
    axs[1].set_xlabel('Noise Levels')
    axs[1].set_ylabel('SNR changes')
    axs[1].set_title('SNR performance')
    axs[1].tick_params(axis="y", labelcolor='tomato')
    axs[1].legend()
    axs[1].set_axisbelow(True)
    axs[1].yaxis.grid(color='lightgray', linestyle='dotted')
    axs[1].xaxis.grid(color='lightgray', linestyle='dotted')

    # #axs[2].plot(df_expanded['noise_levels'], df_expanded['MSE_out'], label='MSE after denoising',linewidth=2,color = '#2ca02c')
    # axs[2].plot(df_expanded['noise_levels'], df_expanded['SNR_in'], label='SNR before denoising',linewidth=2,color = "red")
    # axs[2].plot(df_expanded['noise_levels'], df_expanded['SNR_out'], label='SNR after denoising',linewidth=2,color = "#ff7f0e")
    # #axs[2].plot(df_expanded['noise_levels'], df_expanded['MSE_in'], label='mse before denoising',linewidth=2,color = "yellow")
    # axs[2].plot(df_expanded['noise_levels'], df_expanded['PSNR_in'], label='pSNR before denoising',linewidth=2,color = "pink")
    # axs[2].plot(df_expanded['noise_levels'], df_expanded['PSNR_out'], label='pSNR after denoising',linewidth=2,color = "brown")
    # axs[2].fill_between(df_expanded['noise_levels'], df_expanded['snr_out_lower'], df_expanded['snr_out_upper'], color='#ff7f0e',linewidth = 0, alpha=0.1)
    # #axs[2].fill_between(df_expanded['noise_levels'], df_expanded['mse_out_lower'], df_expanded['mse_out_upper'], color='#2ca02c',linewidth = 0, alpha=0.1)
    # axs[2].set_xlabel('Noise Levels')
    # axs[2].set_ylabel('Value')
    # axs[2].set_title('SNR and MSE after Denoising')
    # axs[2].legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename)

    plt.show()
    plt.close(fig)


def plot_sim_ehb_rate(sim_result,filename=None):
    df_expanded = sim_result.explode(['MSE_changes', 'PSNR_changes','SNR_changes'])

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))

    axs[0].plot(df_expanded['noise_levels'], df_expanded['eHbA1c_org'], label='eHbA1c Original',linestyle = "dashed",linewidth=2)
    axs[0].plot(df_expanded['noise_levels'], df_expanded['eHbA1c_noise'], label='eHbA1c Noisy',linewidth=2)
    axs[0].plot(df_expanded['noise_levels'], df_expanded['eHbA1c_denoise'], label='eHbA1c Denoised',linewidth=2)
    axs[0].fill_between(df_expanded['noise_levels'], df_expanded['eHbA1c1_lower'], df_expanded['eHbA1c1_upper'], color='#ff7f0e', linewidth = 0,alpha=0.1)
    axs[0].fill_between(df_expanded['noise_levels'], df_expanded['eHbA1c2_lower'], df_expanded['eHbA1c2_upper'], color='#2ca02c',linewidth = 0, alpha=0.1)
    axs[0].set_xlabel('Noise Levels')
    axs[0].set_ylabel('Value')
    axs[0].set_title('eHbA1c')
    axs[0].legend()
    axs[0].set_axisbelow(True)
    axs[0].yaxis.grid(color='lightgray', linestyle='dotted')
    axs[0].xaxis.grid(color='lightgray', linestyle='dotted')

    axs[1].plot(df_expanded['noise_levels'], df_expanded['rates_0_to_0'], label='Rates 0_to_0',linewidth=2)
    axs[1].plot(df_expanded['noise_levels'], df_expanded['rates_1_to_1'], label='Rates 1_to_1',linewidth=2)
    axs[1].fill_between(df_expanded['noise_levels'], df_expanded['rates_0_to_0_lower'], df_expanded['rates_0_to_0_upper'], color='#1f77b4', linewidth = 0,alpha=0.1)
    axs[1].fill_between(df_expanded['noise_levels'], df_expanded['rates_1_to_1_lower'], df_expanded['rates_1_to_1_upper'], color='#ff7f0e',linewidth = 0, alpha=0.1)
    axs[1].set_xlabel('Noise Levels')
    axs[1].set_ylabel('Value')
    axs[1].set_title('Remaining Rates of Events')
    axs[1].legend()
    axs[1].set_axisbelow(True)
    axs[1].yaxis.grid(color='lightgray', linestyle='dotted')
    axs[1].xaxis.grid(color='lightgray', linestyle='dotted')

    plt.tight_layout()

    if filename:
        plt.savefig(filename)

    plt.show()
    plt.close(fig)




def print_nn_result(y_test,y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn,fp,fn,tp = confusion_matrix(y_test, y_pred).ravel()
    print("Confusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy:.2f}")

    precision = precision_score(y_test, y_pred)
    print(f"Precision: {precision:.2f}")

    recall = recall_score(y_test, y_pred)
    print(f"Recall/Sensitivity: {recall:.2f}")

    specificity = tn/(tn+fp)
    print(f"Specificity: {specificity:.2f}")


    f1 = f1_score(y_test, y_pred)
    print(f"F1-score: {f1:.2f}")

    roc_auc = roc_auc_score(y_test, y_pred)
    print(f"ROC AUC: {roc_auc:.2f}")
