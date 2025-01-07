import pandas as pd
import pywt
import numpy as np
import math
import PanTompkins
import EntropyHub as EH
from pyts.image import GramianAngularField
import sklearn.metrics as skm
import scipy.io as sio
from sqi_frequency_distribution import ssqi, ksqi
from sqi_power_spectrum import psqi, bassqi
from sqi_rr_intervals import csqi, qsqi
from Morphology_Feature_Extract import *


def signal_reconstruction(signal: np.ndarray, r_nums: int, r_length: int, freq: int):
    # type :(np.ndarray, int, int, int) -> np.ndarray
    """
    reconstruct and slice signal
    :param signal: raw signal
    :param r_nums: reconstruction nums
    :param r_length: reconstruction lengths (seconds)
    :param freq: sample frequency
    :return: reconstructed signals
    """
    sample = r_length * freq
    result = np.zeros([r_nums, sample])
    interval = ((r_nums * r_length) - (len(signal) / freq)) / (r_nums - 1) * freq
    for i in range(r_nums):
        star = int(i * (sample - interval))
        end = star + sample
        result[i] = signal[star:end]
    return result


def denoise(data: np.ndarray):
    # type :(np.ndarray) -> np.ndarray
    """
    wavelet transform to denoise
    :param data:noised data
    :return:denoised data
    """
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=7)
    # cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    # 噪声置零
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')

    return rdata


def Data_Normalization(data: np.ndarray):
    # type :(np.ndarray) -> np.ndarray
    """
    Max-min data normalization
    :param data:
    :return:
    """
    max_value = np.max(data)
    min_value = np.min(data)
    Dvalue = max_value - min_value
    for idx, i in enumerate(data):
        data[idx] = (i - min_value) / Dvalue
    if np.max(data) > 1 or np.min(data) < -1:
        raise Exception("Error Normalization")
    return data


def Seq2Img(data: np.ndarray):
    # type : (np.ndarray) -> np.ndarray
    """
    transform signals to image
    :param data: signals
    :return: image
    """
    graf = GramianAngularField(image_size=512, method='s')
    # data = np.expand_dims(data, axis=0)
    result = np.ndarray((1, 512, 512))
    timestamp = np.zeros(4500)
    # denoised_data = denoise(data)
    for i in range(4500):
        timestamp[i] = (i + 1)

    signals = np.vstack((data, timestamp))
    img = graf.transform(signals)[0]
    result[0, :, :] = img

    return result


def cal_entropy(data: np.ndarray):
    # type :(np.ndarray) -> (float, float, float)
    """
    calculate entropy
    dimension = 1, down-sample = 1
    :param data: signals
    :return: sample entropy, fuzzy entropy, approximate entropy
    """
    sampEH = EH.SampEn(data)[0][1]
    fuzzEH = EH.FuzzEn(data)[0][-1]
    approxEH = EH.ApEn(data)[0][1]

    return sampEH, fuzzEH, approxEH


def cal_statistics_indis(data: np.ndarray, sample_rate: int):
    # type :(np.ndarray, int) -> (float, float, float, float, float, float)
    """
    calculate statistics indis
    include max, min, mean, var, pnn50, rmssd, rr_interval
    :param data:signals
    :param sample_rate:sample rate
    :return:max, min, mean, var, pnn50, rmssd, rr_interval
    """
    max_val = np.max(data)
    min_val = np.min(data)
    mean = np.mean(data)
    var = np.var(data)

    # PNN50
    hr = PanTompkins.heart_rate(data, sample_rate)
    result = hr.find_r_peaks()
    result = np.array(result)
    result = result[result > 0]
    loc = hr.peaks
    r_num = len(loc)
    count = 0
    for i in range(0, r_num - 2):
        if loc[i + 1] - loc[i] > 50:
            count += 1

    pnn50 = count / r_num

    # RMSSD
    time_gap = []
    for i in range(0, r_num - 2):
        time_gap.append((loc[i + 1] - loc[i]) / 360)
    for i in range(len(time_gap) - 1):
        time_gap[i] = time_gap[i] ** 2
    rmssd = math.sqrt(sum(time_gap) / r_num)
    # RR interval
    temp_interval = np.zeros(30)
    if len(loc) >= 31:
        for i in range(len(temp_interval)):
            temp_interval[i] = loc[i + 1] - loc[i]
    else:
        for i in range(len(loc)-1):
            temp_interval[i] = loc[i + 1] - loc[i]
    return max_val, min_val, mean, var


def cal_sqi(data: np.ndarray, sample_rate: int):
    # type :(np.ndarray, int) -> np.ndarray
    """
    calculate signal quality index
    :param data: data
    :param sample_rate: data sample rate
    :return: ndarray[ssqi, ksqi, psqi, bassqi, csqi, qsqi]
    """
    result = np.zeros(6)
    result[0] = ssqi(data)
    result[1] = ksqi(data)
    result[2] = psqi(data, sample_rate)
    result[3] = bassqi(data, sample_rate)
    result[4] = csqi(data, sample_rate)
    result[5] = qsqi(data, sample_rate)

    return result


def cal_morph_features(data: np.ndarray, sample_rate: int):
    # type :(np.ndarray, int) -> np.ndarray
    """
    calculate signal morphology features
    :param data: data
    :param sample_rate: data sample rate
    :return: features[]
    """
    r, s, q = EKG_QRS_detect(data, sample_rate, True, False)
    result = list()
    if r.shape[0] > 1:
        rr_interval = np.diff(r, n=1)
        ss_interval = np.diff(s, n=1)
        qq_interval = np.diff(q, n=1)
        # median
        result.append(np.median(rr_interval))
        result.append(np.median(ss_interval))
        result.append(np.median(qq_interval))
        # mean
        result.append(np.mean(rr_interval))
        result.append(np.mean(ss_interval))
        result.append(np.mean(qq_interval))

        # QRS duration time
        duration_time = list()
        if q.shape[0] >= s.shape[0]:
            for i in range(q.shape[0]):
                duration_time.append(s[i] - q[i])
        else:
            for i in range(s.shape[0]):
                duration_time.append(s[i] - q[i])
        result.append(np.median(np.asarray(duration_time)))
        result.append(np.mean(np.asarray(duration_time)))

        r_num = len(r)
        count = 0
        for i in range(0, r_num - 2):
            if r[i + 1] - r[i] > 50:
                count += 1
        pnn50 = count / r_num
        result.append(pnn50)

        # RMSSD
        time_gap = []
        for i in range(0, r_num - 2):
            time_gap.append((r[i + 1] - r[i]) / 360)
        for i in range(len(time_gap) - 1):
            time_gap[i] = time_gap[i] ** 2
        rmssd = math.sqrt(sum(time_gap) / r_num)
        result.append(rmssd)
    else:
        for i in range(10):
            result.append(0)
    return np.asarray(result)


def cal_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list):
    # type :(np.ndarray, np.ndarray, list) -> (float, np.ndarray, np.ndarray)
    """
    calculate statistics metrics of prediction result
    :param labels: labels of classification
    :param y_true: real labels value
    :param y_pred: prediction labels value
    :return: 1. accurate;
             2. precision, recall, F1-score;
             3. confusion matrix
    """
    # confusion matrix
    cm = skm.confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels)

    # cal metric of each type
    l = len(labels)
    metric = np.ndarray((l, 3))
    cr = skm.classification_report(y_true=y_true, y_pred=y_pred, labels=labels, output_dict=True, zero_division=0.0)
    for i in range(l):
        # precision
        metric[i, 0] = cr[labels[i]].get('precision')
        # sensitive
        metric[i, 1] = cr[labels[i]].get('recall')
        # F1-score
        metric[i, 2] = cr[labels[i]].get('f1-score')

    # cal accurate
    # acc = round((cr.get('accuracy') * 100), 2)
    acc = cr.get('accuracy')
    return acc, metric, cm


def down_sample(data: np.ndarray, sample_rate: float):
    # type :(np.ndarray, float) -> np.ndarray
    """
    data down sample
    :param data: object data
    :param sample_rate: float down sample rate
    :return: target data(down sampled)
    """
    stride = int(1 / sample_rate)
    result = np.ndarray((1, math.ceil(len(data) * sample_rate)))
    j = 0
    for i in range(0, len(data), stride):
        result[0, j] = sum(data[i:i + stride]) / stride
        j += 1
    return result[0, :]



    df3.to_csv("PTB-XL/downsample/" + data_type + "_labels.csv", index=False)


def cinc_padding(data: np.ndarray, res_lens: int):
    # type :(np.ndarray, int) -> np.ndarray
    """
    padding zero to shorten data
    :param data: raw data
    :param res_lens: result data length
    :return: padded data
    """
    result = np.ndarray(res_lens)
    diff = res_lens % len(data)
    counts = int(res_lens / len(data))
    for i in range(counts):
        result[i*len(data):(i+1)*len(data)] = data
    result[counts*len(data):] = data[0:diff]
    return result


def data_separate(data: np.ndarray, lens: int):
    # type :(np.ndarray, int) -> (np.ndarray, int)
    """
    split data into lens parts
    :param data: one-dimension original ndarray data
    :param lens: splited data length
    :return: separate data(two-dimension) and nums of data
    """
    nums = math.floor(data.shape[0] / lens)
    left = data.shape[0] % lens

    result = np.ndarray((nums, lens))
    for i in range(nums):
        result[i, :] = data[i*lens:(i+1)*lens]

    if left >= 0.75 * lens:
        result = np.vstack((result, cinc_padding(data[(nums*lens):], 3000)))
        nums += 1
    return result, nums


def cinc_data_split(data: str):
    # type :str -> None
    """
    reconstruct CinC data into csv files
    :param data: data position
    :return: None
    """
    ref = pd.read_csv('CinC2017/Raw_Data/REFERENCE.csv').values
    labels = []
    base_data = np.zeros((1, 3000))
    for i in range(ref.shape[0]):
        f = sio.loadmat(data + ref[i, 0] + '.mat')['val'][0, :]
        if 3000 < f.shape[0] <= 18000:
            # use signal reconstruction in article by Zhang et al.
            # set hyperparameters r_nums = 6, r_length = 10
            f = signal_reconstruction(f, 6, 10, 300)
            f = denoise(f)
            base_data = np.vstack((base_data, f))
            for n in range(6):
                labels.append(ref[i, -1])

    base_data = np.delete(base_data, 0, 0)
    if base_data.shape[0] != len(labels):
        raise ValueError("base_data.shape[0] != len(labels)")

    df1 = pd.DataFrame(base_data)
    df2 = pd.DataFrame(labels)
    df1.to_csv('CinC2017/split_data_0901.csv', index=False)
    df2.to_csv('CinC2017/split_labels_0901.csv', index=False)
