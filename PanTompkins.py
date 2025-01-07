import wfdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sg

global freq
freq = 300


class Pan_Tompkins_QRS:

    def band_pass_filter(self, signals):
        # Initialize result
        result = None

        # Create a copy of the input signal
        sig = signals.copy()

        # Apply the low pass filter using the equation given
        for index in range(len(signals)):
            sig[index] = signals[index]

            if index >= 1:
                sig[index] += 2 * sig[index - 1]

            if index >= 2:
                sig[index] -= sig[index - 2]

            if index >= 6:
                sig[index] -= 2 * signals[index - 6]

            if index >= 12:
                sig[index] += signals[index - 12]

                # Copy the result of the low pass filter
        result = sig.copy()

        # Apply the high pass filter using the equation given
        for index in range(len(signals)):
            result[index] = -1 * sig[index]

            if index >= 1:
                result[index] -= result[index - 1]

            if index >= 16:
                result[index] += 32 * sig[index - 16]

            if index >= 32:
                result[index] += sig[index - 32]

        # Normalize the result from the high pass filter
        max_val = max(max(result), -min(result))
        result = result / max_val

        return result

    def derivative(self, signals):
        # Initialize result
        result = signals.copy()

        # Apply the derivative filter using the equation given
        for index in range(len(signals)):
            result[index] = 0

            if index >= 1:
                result[index] -= 2 * signals[index - 1]

            if index >= 2:
                result[index] -= signals[index - 2]

            if 2 <= index <= len(signals) - 2:
                result[index] += 2 * signals[index + 1]

            if 2 <= index <= len(signals) - 3:
                result[index] += signals[index + 2]

            result[index] = (result[index] * freq) / 8

        return result

    def squaring(self, signals):
        # Initialize result
        result = signals.copy()

        # Apply the squaring using the equation given
        for index in range(len(signals)):
            result[index] = signals[index] ** 2

        return result

    def moving_window_integration(self, signals):
        # Initialize result and window size for integration
        result = signals.copy()
        win_size = round(0.150 * freq)
        sum = 0

        # Calculate the sum for the first N terms
        for j in range(win_size):
            sum += signals[j] / win_size
            result[j] = sum

        # Apply the moving window integration using the equation given
        for index in range(win_size, len(signals)):
            sum += signals[index] / win_size
            sum -= signals[index - win_size] / win_size
            result[index] = sum

        return result

    def solve(self, signal):
        # Bandpass Filter
        global bpass
        bpass = self.band_pass_filter(signal.copy())

        # Derivative Function
        global der
        der = self.derivative(bpass.copy())

        # Squaring Function
        global sqr
        sqr = self.squaring(der.copy())

        # Moving Window Integration Function
        global mwin
        mwin = self.moving_window_integration(sqr.copy())

        return mwin


class heart_rate():

    def __init__(self, data, samp_freq):
        # Initialize variables
        QRS_detector = Pan_Tompkins_QRS()
        self.RR1, self.RR2, self.probable_peaks, self.r_locs, self.peaks, self.result = ([] for i in range(6))
        self.SPKI, self.NPKI, self.Threshold_I1, self.Threshold_I2, self.SPKF, self.NPKF, self.Threshold_F1, self.Threshold_F2 = (
            0 for i in range(8))

        self.T_wave = False
        self.m_win = QRS_detector.solve(data)
        self.b_pass = bpass
        self.samp_freq = samp_freq
        self.signal = data
        self.win_150ms = round(0.15 * self.samp_freq)

        self.RR_Low_Limit = 0
        self.RR_High_Limit = 0
        self.RR_Missed_Limit = 0
        self.RR_Average1 = 0
        self.outofindex = False

    def approx_peak(self):
        # FFT convolution
        slopes = sg.fftconvolve(self.m_win, np.full((25,), 1) / 25, mode='same')

        # Finding approximate peak locations
        for i in range(round(0.2 * self.samp_freq) + 1, len(slopes) - 1):
            if (slopes[i] > slopes[i - 1]) and (slopes[i + 1] < slopes[i]):
                self.peaks.append(i)

    def adjust_rr_interval(self, ind):
        # Finding the eight most recent RR intervals
        self.RR1 = np.diff(self.peaks[max(0, ind - 8): ind + 1]) / self.samp_freq

        # Calculating RR Averages
        self.RR_Average1 = np.mean(self.RR1)
        RR_Average2 = self.RR_Average1

        # Finding the eight most recent RR intervals lying between RR Low Limit and RR High Limit
        if ind >= 8:
            for i in range(0, 8):
                if self.RR_Low_Limit < self.RR1[i] < self.RR_High_Limit:
                    self.RR2.append(self.RR1[i])

                    if len(self.RR2) > 8:
                        self.RR2.remove(self.RR2[0])
                        RR_Average2 = np.mean(self.RR2)

                        # Adjusting the RR Low Limit and RR High Limit
        if len(self.RR2) > 7 or ind < 8:
            self.RR_Low_Limit = 0.92 * RR_Average2
            self.RR_High_Limit = 1.16 * RR_Average2
            self.RR_Missed_Limit = 1.66 * RR_Average2

    def searchback(self, peak_val, RRn, sb_win):
        # Check if the most recent RR interval is greater than the RR Missed Limit
        global x_max
        if RRn > self.RR_Missed_Limit:
            # Initialize a window to searchback
            win_rr = self.m_win[peak_val - sb_win + 1: peak_val + 1]

            # Find the x locations inside the window having y values greater than Threshold I1
            coord = np.asarray(win_rr > self.Threshold_I1).nonzero()[0]

            # Find the x location of the max peak value in the search window
            if len(coord) > 0:
                for pos in coord:
                    if win_rr[pos] == max(win_rr[coord]):
                        x_max = pos
                        break
            else:
                x_max = None

            # If the max peak value is found
            if x_max is not None:
                # Update the thresholds corresponding to moving window integration
                self.SPKI = 0.25 * self.m_win[x_max] + 0.75 * self.SPKI
                self.Threshold_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
                self.Threshold_I2 = 0.5 * self.Threshold_I1

                # Initialize a window to searchback
                win_rr = self.b_pass[x_max - self.win_150ms: min(len(self.b_pass) - 1, x_max)]

                # Find the x locations inside the window having y values greater than Threshold F1
                coord = np.asarray(win_rr > self.Threshold_F1).nonzero()[0]

                # Find the x location of the max peak value in the search window
                if len(coord) > 0:
                    for pos in coord:
                        if win_rr[pos] == max(win_rr[coord]):
                            r_max = pos
                            break
                else:
                    r_max = None

                # If the max peak value is found
                if r_max is not None:
                    # Update the thresholds corresponding to bandpass filter
                    if self.b_pass[r_max] > self.Threshold_F2:
                        self.SPKF = 0.25 * self.b_pass[r_max] + 0.75 * self.SPKF
                        self.Threshold_F1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
                        self.Threshold_F2 = 0.5 * self.Threshold_F1

                        # Append the probable R peak location
                        self.r_locs.append(r_max)

    def find_t_wave(self, peak_val, RRn, ind, prev_ind):
        if self.m_win[peak_val] >= self.Threshold_I1:
            if ind > 0 and 0.20 < RRn < 0.36:
                # Find the slope of current and last waveform detected
                curr_slope = max(np.diff(self.m_win[peak_val - round(self.win_150ms / 2): peak_val + 1]))
                last_slope = max(
                    np.diff(self.m_win[self.peaks[prev_ind] - round(self.win_150ms / 2): self.peaks[prev_ind] + 1]))

                # If current waveform slope is less than half of last waveform slope
                if curr_slope < 0.5 * last_slope:
                    # T Wave is found and update noise threshold
                    self.T_wave = True
                    self.NPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.NPKI

            if not self.T_wave:
                # T Wave is not found and update signal thresholds
                if self.probable_peaks[ind] > self.Threshold_F1:
                    self.SPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.SPKI
                    self.SPKF = 0.125 * self.b_pass[ind] + 0.875 * self.SPKF

                    # Append the probable R peak location
                    self.r_locs.append(self.probable_peaks[ind])

                else:
                    self.SPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.SPKI
                    self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

                    # Update noise thresholds
        elif (self.m_win[peak_val] < self.Threshold_I1) or (
                self.Threshold_I1 < self.m_win[peak_val] < self.Threshold_I2):
            self.NPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.NPKI
            self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

    def adjust_thresholds(self, peak_val, ind):
        if self.m_win[peak_val] >= self.Threshold_I1:
            # Update signal threshold
            self.SPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.SPKI

            if self.probable_peaks[ind] > self.Threshold_F1:
                self.SPKF = 0.125 * self.b_pass[ind] + 0.875 * self.SPKF

                # Append the probable R peak location
                self.r_locs.append(self.probable_peaks[ind])

            else:
                # Update noise threshold
                self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

                # Update noise thresholds
        elif (self.m_win[peak_val] < self.Threshold_I2) or (
                self.Threshold_I2 < self.m_win[peak_val] < self.Threshold_I1):
            self.NPKI = 0.125 * self.m_win[peak_val] + 0.875 * self.NPKI
            self.NPKF = 0.125 * self.b_pass[ind] + 0.875 * self.NPKF

    def update_thresholds(self):
        self.Threshold_I1 = self.NPKI + 0.25 * (self.SPKI - self.NPKI)
        self.Threshold_F1 = self.NPKF + 0.25 * (self.SPKF - self.NPKF)
        self.Threshold_I2 = 0.5 * self.Threshold_I1
        self.Threshold_F2 = 0.5 * self.Threshold_F1
        self.T_wave = False

    def ecg_searchback(self):
        # Filter the unique R peak locations
        self.r_locs = np.unique(np.array(self.r_locs).astype(int))

        # Initialize a window to searchback
        win_200ms = round(0.2 * self.samp_freq)

        for r_val in self.r_locs:
            coord = np.arange(r_val - win_200ms, min(len(self.signal), r_val + win_200ms + 1), 1)

            # Find the x location of the max peak value
            if len(coord) > 0:
                for pos in coord:
                    if self.signal[pos] == max(self.signal[coord]):
                        x_max = pos
                        break
            else:
                x_max = None

            # Append the peak location
            if x_max is not None:
                self.result.append(x_max)

    def find_r_peaks(self):
        # Find approximate peak locations
        self.approx_peak()

        # Iterate over possible peak locations
        for ind in range(len(self.peaks)):

            # Initialize the search window for peak detection
            peak_val = self.peaks[ind]
            win_300ms = np.arange(max(0, self.peaks[ind] - self.win_150ms),
                                  min(self.peaks[ind] + self.win_150ms, len(self.b_pass) - 1), 1)
            max_val = max(self.b_pass[win_300ms], default=0)

            # Find the x location of the max peak value
            if max_val != 0:
                x_coord = np.asarray(self.b_pass == max_val).nonzero()
                self.probable_peaks.append(x_coord[0][0])

            if ind < len(self.probable_peaks) and ind != 0:
                # Adjust RR interval and limits
                self.adjust_rr_interval(ind)

                # Adjust thresholds in case of irregular beats
                if self.RR_Average1 < self.RR_Low_Limit or self.RR_Average1 > self.RR_Missed_Limit:
                    self.Threshold_I1 /= 2
                    self.Threshold_F1 /= 2

                RRn = self.RR1[-1]

                # Searchback
                self.searchback(peak_val, RRn, round(RRn * self.samp_freq))

                # T Wave Identification
                self.find_t_wave(peak_val, RRn, ind, ind - 1)

            else:
                # Adjust threholds
                # debugger
                # if ind >= len(self.probable_peaks):
                #     self.outofindex = True
                # else:
                self.adjust_thresholds(peak_val, ind)

            # Update threholds for next iteration
            self.update_thresholds()

        # Searchback in ECG signal
        self.ecg_searchback()

        # return self.outofindex
        return self.result


'''
filename = f'F:\DataSet\MIT-BIH\\102'
record = wfdb.rdrecord(filename, sampfrom=0, sampto=1080, )
freq = 360

QRS_detector = Pan_Tompkins_QRS()
ecg = pd.DataFrame(np.array([list(range(len(record.adc()))), record.adc()[:, 0]]).T, columns=['TimeStamp', 'ecg'])
output_singal = QRS_detector.solve(ecg)

signal = ecg.iloc[:, 1].to_numpy()

# Find the R peak locations
hr = heart_rate(signal, freq)
result = hr.find_r_peaks()
result = np.array(result)

# Clip the x locations less than 0 (Learning Phase)
result = result[result > 0]

# Plotting the R peak locations in ECG signal
plt.figure(figsize=(20, 4), dpi=100)
plt.xticks(np.arange(0, len(signal) + 1, 150))
plt.plot(signal, color='blue')
plt.scatter(result, signal[result], color='red', s=50, marker='*')
plt.xlabel('Samples')
plt.ylabel('MLIImV')
plt.title("R Peak Locations")
plt.show()
'''
