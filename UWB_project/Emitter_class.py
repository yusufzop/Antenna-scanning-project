import random
from types import NoneType
import numpy as np
import matplotlib.pyplot as plt
from Wave_Generator_Functions import pri_values, generate_waveform
from Antenna_Scanning_Functions import load_radiation_pattern, circular_scanning, unidirectional_sector_scanning, bidirectional_sector_scanning, raster_scanning, raster_sector_scanning_centering_a_target
global c
c= 299792458
class Emitter:
    def __init__(self,ID=None, wave_type_range=None, emitter_complexity=None, PW_range=None,
                 number_of_pulse_range=None, PRI_type_range=None, PRI_agile_ampl_range=None, 
                 amplitude_range=None, fc_range=None, PRI_mean_range=None,wave_param_B_range=None,start_position=None,
                 velocity=None,locked_target=None,
                 antenna_gain=None,SNR_aim=None,current_noise_power=None, Antenna_Scanning_Type=None,
                 Antenna_Scanning_step=None, boresight_location_azimuth_deg_start=None,boresight_location_elevation_deg_start=None,
                 sector_angle_azimuth_deg=None,sector_angle_elevation_deg=None,scanning_area_deg=None,
                 target_azimuth_deg=None,target_elevation_deg=None,
                 antenna_radiation_pattern=None, fs=None, frame_duration=None):
        # Identification & basic configuration
        self.ID = ID
        self.emitter_complexity = emitter_complexity
        self.frame_duration = frame_duration
        self.fs = fs

        # Pulse Params
        self.wave_type_range = wave_type_range
        self.PW_range = PW_range
        self.number_of_pulse_range = number_of_pulse_range
        self.wave_param_L = None
        self.wave_param_B = None
        self.fc_range = fc_range
        self.wave_param_B_range = wave_param_B_range
        self.amplitude_range = amplitude_range

        # PRI Params
        self.PRI_type_range = PRI_type_range
        self.PRI_agile_ampl_range = PRI_agile_ampl_range
        self.PRI_mean_range = PRI_mean_range

        # Motion & tracking
        self.start_position = np.array(start_position, dtype=float) if start_position is not None else None
        self.velocity = np.array(velocity, dtype=float) if velocity is not None else None
        self.locked_target = np.array(locked_target, dtype=float) if isinstance(locked_target, (list, tuple, np.ndarray)) else locked_target
        self.locked_target_position = None

        # Antenna configuration
        self.antenna_gain = antenna_gain
        self.SNR_aim = SNR_aim
        self.Antenna_Scanning_Type = Antenna_Scanning_Type
        self.Antenna_Scanning_step = Antenna_Scanning_step
        self.boresight_location_azimuth_deg_start = boresight_location_azimuth_deg_start
        self.boresight_location_elevation_deg_start = boresight_location_elevation_deg_start
        self.sector_angle_azimuth_deg = sector_angle_azimuth_deg
        self.sector_angle_elevation_deg = sector_angle_elevation_deg
        self.antenna_radiation_pattern = antenna_radiation_pattern
        self.scanning_area_deg = scanning_area_deg
        self.is_scanning=True
        self.scan_index=0
        self.first_transmit=True
        self.target_azimuth_deg = target_azimuth_deg
        self.target_elevation_deg = target_elevation_deg
        


    def set_waveform_type(self):
        self.wave_type = random.choice(self.wave_type_range)
        return self
    
    def set_carrier_frequency(self):
        
        max_fc = self.fc_range[1] - (self.right_bandwidth_freq-self.fs/4)   # Maximum f_c to avoid aliasing
        min_fc = self.fc_range[0] - (self.left_bandwidth_freq-self.fs/4) # Minimum f_c to keep signal in positive frequencies
        fc = random.uniform(min_fc, max_fc)
        self.fc = fc
        return self

    def set_number_of_pulse(self):
        self.number_of_pulse = random.randint(self.number_of_pulse_range[0], self.number_of_pulse_range[1])
        return self

    def set_PRI_params_calculate_PRI_values(self):
        self.PRI_type = random.choice(self.PRI_type_range)
        self.PRI_agile_ampl = random.uniform(self.PRI_agile_ampl_range[0], self.PRI_agile_ampl_range[1])
        max_PRI = min(self.PRI_mean_range[1],self.frame_duration/(self.number_of_pulse+2))
        if max_PRI<self.PRI_mean_range[0]:
            a=1
        self.PRI_mean = random.uniform(self.PRI_mean_range[0], max_PRI)
        self.PRI_values = pri_values(self)
        return self

    def set_pulse_width(self):
        min_PRI=min(self.PRI_values)
        max_PW= min(self.PW_range[1],min_PRI/1.5)
        self.pulse_width = random.uniform(self.PW_range[0], max_PW)
        if max_PW<self.PW_range[0]:
            self.pulse_width = random.uniform(max_PW/2, max_PW)
        if min_PRI<self.pulse_width:
            a=1
        
        
        return self

    def set_wave_param_L(self):
        if self.wave_type == "barker":
            self.wave_param_L = random.choice([2, 7, 11, 13])
        elif self.wave_type == "costas":
            self.wave_param_L = random.choice([7, 11, 13, 17])
        elif self.wave_type in ["frank", "p1", "p2"]:
            self.wave_param_L = random.choice([ 2, 3, 4])
        elif self.wave_type in ["p3", "p4"]:
            self.wave_param_L = random.randint(4, 16)
        else:
            self.wave_param_L = 1
        return self

    def set_wave_param_B(self):
        if self.wave_type == "costas" or self.wave_type == "lfm":
            self.wave_param_B = random.uniform(self.wave_param_B_range[0], self.wave_param_B_range[1])
        elif self.wave_type in ["frank", "p1", "p2"]:
            self.wave_param_B = self.wave_param_L*self.wave_param_L/self.pulse_width
        elif self.wave_type in ["rect", "barker", "p3", "p4"]:
            self.wave_param_B = self.wave_param_L/self.pulse_width
        return self

    def generate_waveform(self):
        self.set_waveform_type()
        self.set_number_of_pulse()
        self.set_PRI_params_calculate_PRI_values()
        self.set_pulse_width()
        self.set_wave_param_L()
        self.set_wave_param_B()
        self.fc=self.fs/4

        
        if min(self.PRI_values)<self.pulse_width:
            a=1

        self.waveform=generate_waveform(self)
        if self.wave_type in ["rect"]:
            self.calculate_global_bandwidth(bandwidth_threshold_db=15)
        elif self.wave_type in ["barker","frank", "p1", "p2", "p3", "p4"]:
            self.calculate_global_bandwidth(bandwidth_threshold_db=15)
        elif self.wave_type in ["costas","lfm"]:
            self.calculate_global_bandwidth(bandwidth_threshold_db=15)
        self.set_carrier_frequency()

        self.waveform=generate_waveform(self)
        if self.wave_type in ["rect"]:
            self.calculate_global_bandwidth(bandwidth_threshold_db=15)
        elif self.wave_type in ["barker","frank", "p1", "p2", "p3", "p4"]:
            self.calculate_global_bandwidth(bandwidth_threshold_db=15)
        elif self.wave_type in ["costas","lfm"]:
            self.calculate_global_bandwidth(bandwidth_threshold_db=15)
        return self

    def calculate_bandwidth(self,bandwidth_threshold_db):
        pulse_length = min(256, len(self.waveform))
        #pulse_length = len(self.waveform)
        pulse = self.waveform[0:pulse_length]
        # If pulse is shorter than 256, pad it to 256 samples
        if len(pulse) < 256:
            pulse = np.pad(pulse, (0, 256 - len(pulse)), mode='constant')
        elif len(pulse) > 256:
            pulse = pulse[0:256]
        # Compute FFT
        fft_pulse = np.fft.fft(pulse)
        fft_freq = np.fft.fftfreq(len(pulse), 1/self.fs)
        
        # Shift to center frequencies around 0
        fft_pulse_shifted = np.fft.fftshift(fft_pulse)
        fft_freq_shifted = np.fft.fftshift(fft_freq)
        
        # Get magnitude spectrum
        magnitude = np.abs(fft_pulse_shifted)
        magnitude_db = 20 * np.log10(magnitude + 1e-12)
        
        # Restrict to non-negative frequencies when searching for the peak
        positive_mask = fft_freq_shifted >= 0
        magnitude_pos = magnitude[positive_mask]
        magnitude_db_pos = magnitude_db[positive_mask]
        fft_freq_shifted_pos = fft_freq_shifted[positive_mask]
        peak_pos_idx = np.argmax(magnitude_pos)
        peak_magnitude = magnitude_pos[peak_pos_idx]
        peak_magnitude_db = magnitude_db_pos[peak_pos_idx]
        peak_freq = fft_freq_shifted_pos[peak_pos_idx]
        # map back to original index space for bandwidth calculations
        peak_idx = np.where(positive_mask)[0][peak_pos_idx]
        
        bandwidth_threshold = peak_magnitude / 10**(bandwidth_threshold_db/20)
        threshold_db_line = peak_magnitude_db - bandwidth_threshold_db
        
        bandwidth_threshold = peak_magnitude * 10**(-bandwidth_threshold_db / 20.0)

        # --- Left side search ---
        left_bandwidth_idx = peak_idx
        for i in range(peak_idx - 1, -1, -1):
            if magnitude[i] >= bandwidth_threshold:
                left_bandwidth_idx = i
            else:
                # Crossing between i and i+1
                if i + 1 < len(magnitude):
                    m1 = magnitude[i]      # below threshold
                    m2 = magnitude[i + 1]  # above threshold
                    if m2 != m1:
                        alpha = (bandwidth_threshold - m1) / (m2 - m1)
                    else:
                        alpha = 0.0
                    left_bandwidth_idx = i + alpha
                break

        # --- Right side search ---
        right_bandwidth_idx = peak_idx
        for j in range(peak_idx + 1, len(magnitude)):
            if magnitude[j] >= bandwidth_threshold:
                right_bandwidth_idx = j
            else:
                # Crossing between j-1 and j
                if j - 1 >= 0:
                    m1 = magnitude[j - 1]  # above threshold
                    m2 = magnitude[j]      # below threshold
                    if m2 != m1:
                        alpha = (bandwidth_threshold - m1) / (m2 - m1)
                    else:
                        alpha = 0.0
                    right_bandwidth_idx = (j - 1) + alpha
                break

        # --- Linear index → frequency ---
        freq_step = fft_freq_shifted[1] - fft_freq_shifted[0]
        f0 = fft_freq_shifted[0]

        left_bandwidth_freq = f0 + left_bandwidth_idx * freq_step
        right_bandwidth_freq = f0 + right_bandwidth_idx * freq_step
        if left_bandwidth_freq <0:
            left_bandwidth_freq=0
        self.left_bandwidth_freq=left_bandwidth_freq
        self.right_bandwidth_freq=right_bandwidth_freq
        self.bandwidth = right_bandwidth_freq - left_bandwidth_freq
        self._plot_bandwidth_spectrum(fft_freq_shifted, magnitude_db, peak_freq, left_bandwidth_freq, right_bandwidth_freq, self.bandwidth, threshold_db_line)
        return self

    def calculate_global_bandwidth(self, bandwidth_threshold_db, N=256, hop=None):
        N=256
        if hop is None:
            hop = N // 4

        x = self.waveform
        fs = self.fs
        L = len(x)

        if L < N:
            # Pad once and just reuse your existing method
            x = np.pad(x, (0, N - L), mode='constant')
            L = len(x)

        window = np.hamming(N)

        # Number of frames
        num_frames = 1 + (L - N) // hop

        # We will store max magnitude over time for each frequency bin
        # Use first frame to initialize freq axis
        first_frame = x[0:N] * window
        fft_first = np.fft.fft(first_frame)
        fft_freq = np.fft.fftfreq(N, 1/fs)
        fft_freq_shifted = np.fft.fftshift(fft_freq)

        max_magnitude = np.zeros_like(fft_first, dtype=float)

        # Slide over all frames
        for m in range(num_frames):
            start = m * hop
            frame = x[start:start+N]
            if len(frame) < N:
                frame = np.pad(frame, (0, N - len(frame)), mode='constant')  # zero-pad to N
            frame = frame * window
            fft_frame = np.fft.fft(frame)
            fft_frame_shifted = np.fft.fftshift(fft_frame)
            mag = np.abs(fft_frame_shifted)
            max_magnitude = np.maximum(max_magnitude, mag)

        # Now max_magnitude is M[k] = max_t |X[k,t]|
        magnitude = max_magnitude
        magnitude_db = 20 * np.log10(magnitude + 1e-12)

        # Restrict to non-negative frequencies for peak search
        positive_mask = fft_freq_shifted >= 0
        magnitude_pos = magnitude[positive_mask]
        magnitude_db_pos = magnitude_db[positive_mask]
        fft_freq_shifted_pos = fft_freq_shifted[positive_mask]

        peak_pos_idx = np.argmax(magnitude_pos)
        peak_magnitude = magnitude_pos[peak_pos_idx]
        peak_magnitude_db = magnitude_db_pos[peak_pos_idx]
        peak_freq = fft_freq_shifted_pos[peak_pos_idx]
        peak_idx = np.where(positive_mask)[0][peak_pos_idx]  # back to full index

        # Threshold in linear
        bandwidth_threshold = peak_magnitude * 10**(-bandwidth_threshold_db / 20.0)
        threshold_db_line = peak_magnitude_db - bandwidth_threshold_db

        # --- Left search ---
        left_bandwidth_idx = peak_idx
        for i in range(peak_idx - 1, -1, -1):
            if magnitude[i] >= bandwidth_threshold:
                left_bandwidth_idx = i
            else:
                if i + 1 < len(magnitude):
                    m1 = magnitude[i]
                    m2 = magnitude[i + 1]
                    if m2 != m1:
                        alpha = (bandwidth_threshold - m1) / (m2 - m1)
                    else:
                        alpha = 0.0
                    left_bandwidth_idx = i + alpha
                break

        # --- Right search ---
        right_bandwidth_idx = peak_idx
        for j in range(peak_idx + 1, len(magnitude)):
            if magnitude[j] >= bandwidth_threshold:
                right_bandwidth_idx = j
            else:
                if j - 1 >= 0:
                    m1 = magnitude[j - 1]
                    m2 = magnitude[j]
                    if m2 != m1:
                        alpha = (bandwidth_threshold - m1) / (m2 - m1)
                    else:
                        alpha = 0.0
                    right_bandwidth_idx = (j - 1) + alpha
                break

        # Index → frequency
        freq_step = fft_freq_shifted[1] - fft_freq_shifted[0]
        f0 = fft_freq_shifted[0]

        left_bandwidth_freq = f0 + left_bandwidth_idx * freq_step
        right_bandwidth_freq = f0 + right_bandwidth_idx * freq_step

        if left_bandwidth_freq < 0:
            left_bandwidth_freq = 0.0

        self.left_bandwidth_freq = left_bandwidth_freq
        self.right_bandwidth_freq = right_bandwidth_freq
        self.bandwidth = right_bandwidth_freq - left_bandwidth_freq

        # self._plot_bandwidth_spectrum(
        #     fft_freq_shifted,
        #     magnitude_db,
        #     peak_freq,
        #     left_bandwidth_freq,
        #     right_bandwidth_freq,
        #     self.bandwidth,
        #     threshold_db_line,
        # )

        return self


    def _plot_bandwidth_spectrum(self, frequencies, magnitude_db, peak_freq, left_freq, right_freq, bandwidth, threshold_db_line):
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        freq_mhz = frequencies / 1e6

        ax.plot(freq_mhz, magnitude_db, 'b-', linewidth=1.5, label='Magnitude Spectrum')
        ax.axvline(peak_freq/1e6, color='r', linestyle='--', linewidth=1.5, label=f'Peak at {peak_freq/1e6:.4f} MHz')
        ax.axvline(left_freq/1e6, color='g', linestyle='--', linewidth=1.5, label='Bandwidth edges')
        ax.axvline(right_freq/1e6, color='g', linestyle='--', linewidth=1.5)
        ax.axhline(threshold_db_line, color='orange', linestyle=':', linewidth=1.2, alpha=0.8, label=f'Threshold ({threshold_db_line:.2f} dB)')

        mask = (frequencies >= left_freq) & (frequencies <= right_freq)
        ax.fill_between(freq_mhz, magnitude_db.min() - 10, magnitude_db,
                        where=mask, alpha=0.25, color='yellow',
                        label=f'Bandwidth: {bandwidth/1e6:.4f} MHz')

        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Magnitude (dB)')
        ax.set_title('Pulse Spectrum with Bandwidth Highlight')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(freq_mhz.min(), freq_mhz.max())
        ax.legend(loc='best', fontsize=9)
        plt.tight_layout()
        plt.show()
    
    def calculate_start_index(self,frame_length):
        max_start = int(frame_length - len(self.waveform)) 
        min_start = 0
        
        # Check if waveform fits in the frame
        if max_start < min_start:
            # Waveform is too long - print parameters and use safe start position
            print(f"Warning: Waveform too long. Length: {len(self.waveform)}, Frame length: {frame_length}")
            print(f"  number_of_pulse: {self.number_of_pulse}")
            print(f"  pulse_width (PW): {self.pulse_width}")
            print(f"  PRI_values: {self.PRI_values}")
            self.waveform=[0]
            # Ensure it fits by starting at the latest possible position
            safe_max_start = frame_length - len(self.waveform)
            if safe_max_start < min_start:
                # Waveform is longer than the available space - use min_start but it will overflow
                self.start_index = min_start
            else:
                # Use the safe maximum start position
                self.start_index = random.randint(min_start, safe_max_start)
        else:
            self.start_index = random.randint(min_start, max_start)
        return self

    def update_position(self,time):
        self.current_position = self.start_position + self.velocity*time
        return self

    def locking_to_a_target(self, locked_emitter):
        if isinstance(self.locked_target, Emitter):
            self.locked_target_position = locked_emitter.current_position
        elif isinstance(self.locked_target, np.ndarray):
            self.locked_target_position = self.locked_target
        return self

    def power_control(self):
        wavelength = c / self.fc  # Speed of light / carrier frequency
        distance_to_target = np.linalg.norm(self.current_position - self.locked_target_position)
        path_loss_linear = (4 * np.pi * distance_to_target / wavelength) ** 4
        linear_SNR_aim = 10**(self.SNR_aim / 10)
        required_signal_power = linear_SNR_aim * self.current_noise_power * path_loss_linear / self.antenna_gain
        self.amplitude = max (np.sqrt(required_signal_power),self.amplitude_range[1])
        return self
    
    def set_radiation_pattern(self):
        self.radiation_pattern = load_radiation_pattern(self.antenna_radiation_pattern)
        return self

    def set_scanning_type(self):
        if self.Antenna_Scanning_Type == "No_Scanning":
            self.scan_az_deg_list = [0]
            self.scan_el_deg_list = [0]
        elif self.Antenna_Scanning_Type == "circular_scanning":
            self.scan_az_deg_list, self.scan_el_deg_list = circular_scanning(
                scanning_step_size_deg=self.Antenna_Scanning_step, 
                boresight_location_azimuth_deg_start=self.boresight_location_azimuth_deg_start, 
                boresight_location_elevation_deg_start=self.boresight_location_elevation_deg_start
                )
        elif self.Antenna_Scanning_Type == "unidirectional_sector_scanning":
            self.scan_az_deg_list, self.scan_el_deg_list = unidirectional_sector_scanning(
                scanning_step_size_deg=self.Antenna_Scanning_step, 
                sector_angle_deg=self.sector_angle_azimuth_deg, 
                Target_azimuth_deg=self.boresight_location_azimuth_deg_start, 
                Target_elevation_deg=self.boresight_location_elevation_deg_start
                )
        elif self.Antenna_Scanning_Type == "bidirectional_sector_scanning":
            self.scan_az_deg_list, self.scan_el_deg_list = bidirectional_sector_scanning(
                scanning_step_size_deg=self.Antenna_Scanning_step, 
                sector_angle_deg=self.sector_angle_azimuth_deg, 
                boresight_location_azimuth_deg_start=self.boresight_location_azimuth_deg_start, 
                boresight_location_elevation_deg_start=self.boresight_location_elevation_deg_start
                )
        elif self.Antenna_Scanning_Type == "raster_scanning":
            self.scan_az_deg_list, self.scan_el_deg_list = raster_scanning(
                scanning_step_size_deg=self.Antenna_Scanning_step, 
                boresight_location_azimuth_deg_start=self.boresight_location_azimuth_deg_start, 
                boresight_location_elevation_deg_start=self.boresight_location_elevation_deg_start
                )
        elif self.Antenna_Scanning_Type == "raster_sector_scanning_centering_a_target":
            self.scan_az_deg_list, self.scan_el_deg_list = raster_sector_scanning_centering_a_target(
                scanning_step_size_deg=self.Antenna_Scanning_step,
                square_side_deg=self.scanning_area_deg,
                target_azimuth_deg=self.target_azimuth_deg,
                target_elevation_deg=self.target_elevation_deg
            )
        return self
        
    def check_parameter_limits_valid(self):
        max_waveform_length=self.number_of_pulse_range[1]*self.PRI_mean_range[1]+self.PW_range[1]
        if max_waveform_length>self.frame_duration:
            print("Waveform length is greater than frame duration")
        min_waveform_length=self.PW_range[0]*self.number_of_pulse_range[0]*self.PRI_mean_range[0]
        if min_waveform_length<0:
            print("Waveform length is less than frame duration")
        min_chip_duration=self.PW_range[0]/25
        print(f"Minimum chip duration is {min_chip_duration*self.fs} sample")

        max_frequency=self.fc_range[1]+self.wave_param_B_range[1]
        if max_frequency>self.fs/2:
            print("Maximum frequency is greater than Nyquist frequency")
        
        max_frequency=self.fc_range[1]+20/self.PW_range[0]
        if max_frequency>self.fs/2:
            print("Maximum frequency is greater than Nyquist frequency")
        return 0

    def antenna_gain_calculator(self, scan_az_deg, scan_el_deg, target_az_deg, target_el_deg):
        az_vals = np.asarray( self.radiation_pattern["Az_deg"], dtype=float)
        el_vals = np.asarray( self.radiation_pattern["El_deg"], dtype=float)
        gain_vals = np.asarray( self.radiation_pattern["Gain_linear"], dtype=float)
        max_idx = np.argmax(gain_vals)
        max_gain = gain_vals[max_idx]
        max_gain_az = az_vals[max_idx]
        max_gain_el = el_vals[max_idx]
        rel_el = float(target_el_deg) - float(scan_el_deg)
        rel_az = (float(target_az_deg) - float(scan_az_deg)) % 360.0
        rel_az = (rel_az + 180.0) % 360.0 - 180.0
        rel_el = max(-90.0, min(90.0, rel_el))
        
        az_diff = np.abs((az_vals - rel_az + 180.0) % 360.0 - 180.0)
        el_diff = np.abs(el_vals - rel_el)
        total_diff = np.hypot(az_diff, el_diff)

        idx_sorted = np.argsort(total_diff)
        i1, i2 = idx_sorted[:2]
        d1, d2 = total_diff[i1], total_diff[i2]
        g1, g2 = gain_vals[i1], gain_vals[i2]

        # handle identical distance case
        if d1 == 0:
            gain = g1
        else:
            gain = (g1 * d2 + g2 * d1) / (d1 + d2)
        return gain    

    def scanning(self):
        if self.is_scanning==True:
            target_vector = -1 * self.current_position
            target_distance_xy = np.hypot(target_vector[0], target_vector[1])
            target_el = np.rad2deg(np.arctan2(target_vector[2], target_distance_xy))
            target_az = (np.rad2deg(np.arctan2(target_vector[1], target_vector[0])) + 180.0) % 360.0 - 180.0
            if self.Antenna_Scanning_Type == "circular_scanning":
                scan_az_deg = self.scan_az_deg_list[self.scan_index]
                scan_el_deg = self.scan_el_deg_list[self.scan_index]+target_el
            elif self.Antenna_Scanning_Type == "unidirectional_sector_scanning":
                scan_az_deg = self.scan_az_deg_list[self.scan_index]+target_az
                scan_el_deg = self.scan_el_deg_list[self.scan_index]+target_el
            elif self.Antenna_Scanning_Type == "bidirectional_sector_scanning":
                scan_az_deg = self.scan_az_deg_list[self.scan_index]+target_az
                scan_el_deg = self.scan_el_deg_list[self.scan_index]+target_el
            elif self.Antenna_Scanning_Type == "raster_scanning":
                scan_az_deg = self.scan_az_deg_list[self.scan_index]
                scan_el_deg = self.scan_el_deg_list[self.scan_index]
            elif self.Antenna_Scanning_Type == "raster_sector_scanning_centering_a_target":
                scan_az_deg = self.scan_az_deg_list[self.scan_index]+target_az
                scan_el_deg = self.scan_el_deg_list[self.scan_index]+target_el
            elif self.Antenna_Scanning_Type == "No_Scanning":
                scan_az_deg = 0.0
                scan_el_deg = 0.0
            self.scan_index+=1
            if self.scan_index==len(self.scan_az_deg_list):
                self.is_scanning=False
                self.scan_index=0
        else:
            if random.random()<0.95:
                self.is_scanning=True
                self.scan_index=0
                target_vector = -1 * self.current_position
                target_distance_xy = np.hypot(target_vector[0], target_vector[1])
                target_el = np.rad2deg(np.arctan2(target_vector[2], target_distance_xy))
                target_az = (np.rad2deg(np.arctan2(target_vector[1], target_vector[0])) + 180.0) % 360.0 - 180.0
                if self.Antenna_Scanning_Type == "circular_scanning":
                    scan_az_deg = self.scan_az_deg_list[self.scan_index]
                    scan_el_deg = self.scan_el_deg_list[self.scan_index]+target_el
                elif self.Antenna_Scanning_Type == "unidirectional_sector_scanning":
                    scan_az_deg = self.scan_az_deg_list[self.scan_index]+target_az
                    scan_el_deg = self.scan_el_deg_list[self.scan_index]+target_el
                elif self.Antenna_Scanning_Type == "bidirectional_sector_scanning":
                    scan_az_deg = self.scan_az_deg_list[self.scan_index]+target_az
                    scan_el_deg = self.scan_el_deg_list[self.scan_index]+target_el
                elif self.Antenna_Scanning_Type == "raster_scanning":
                    scan_az_deg = self.scan_az_deg_list[self.scan_index]
                    scan_el_deg = self.scan_el_deg_list[self.scan_index]
                elif self.Antenna_Scanning_Type == "raster_sector_scanning_centering_a_target":
                    scan_az_deg = self.scan_az_deg_list[self.scan_index]+target_az
                    scan_el_deg = self.scan_el_deg_list[self.scan_index]+target_el
                elif self.Antenna_Scanning_Type == "No_Scanning":
                    scan_az_deg = 0.0
                    scan_el_deg = 0.0
                self.scan_index+=1
            else :
                
                scan_az_deg = None
                scan_el_deg = None
                target_az = None
                target_el = None
                if self.Antenna_Scanning_Type == "No_Scanning":
                    scan_az_deg = 0.0
                    scan_el_deg = 0.0
                    target_az = 0.0
                    target_el = 0.0

                

        return scan_az_deg, scan_el_deg, target_az,target_el

    