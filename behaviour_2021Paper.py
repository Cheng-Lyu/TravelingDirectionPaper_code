import functions_2021Paper as fc
from neo.io import AxonIO
from neo.io import WinEdrIO

import plotting_help_2021Paper as ph
from scipy import ndimage

import numpy as np
import scipy.cluster as sc

import sys, os, glob, copy, math, time, pickle, scipy
from simplification.cutil import simplify_coords

reload(fc)
reload(ph)


filename_to_valid_time_dict = {
    # Walking data
    '2017_05_22_0001': [0,150],     '2017_05_22_0005': [0,540],     '2017_05_23_0006': [0,100],     '2017_05_23_0007': [0,100],
    '2017_05_27_0013': [0,790],     '2017_06_16_0003': [0,103],     '2017_06_16_0004': [0,103],     '2017_06_16_0006': [0,148],
    '2018_03_22_0001': [0,160],     '2018_03_23_0007': [0,775],     '2018_03_23_0008': [0,100],     '2018_03_23_0009': [0,86],
    '2018_03_23_0017': [0,260],     '2018_03_26_0007': [0,65],      '2018_03_26_0008': [0,485],     '2018_03_26_0003': [0,315],
    '2018_03_27_0001': [0,250],     '2018_05_13_0004': [0,1674],    '2018_05_13_0005': [0,1982],    '2018_05_14_0002': [0,642],
    '2018_05_14_0003': [0,1600],    '2018_05_14_0004': [0,1340],    '2018_06_03_0001': [0,1880],    '2018_06_11_0005': [1945,-1],
    '2018_06_11_0012': [0,1797],    '2018_06_11_0014': [0,1400],    '2018_08_24_0002': [0, 250],    '2018_08_27_0005': [0, 50],
    '2018_08_27_0015': [50, 200],   '2018_08_27_0006': [0, 900],    '2018_08_28_0001': [20,195],    '2018_08_28_0002': [0,192],
    '2018_08_28_0011': [0,200],     '2018_08_28_0012': [200,600],   '2018_08_28_0005': [180,-1],    '2018_08_23_0002': [0,1980],
    '2018_08_23_0003': [0,280],
    '2018_08_29_0001': [0,368],     '2018_08_30_0001': [0,101],     '2018_08_31_0001': [0,125],     '2018_09_13_0004': [0,2240],
    '2018_09_30_0001': [0,26],      '2018_10_05_0003': [0,210],     '2018_09_30_0007': [0,14.8],    '2018_10_05_0001': [27,128],
    '2018_10_05_0002': [0,45],      '2018_10_16_0003': [0,24.7],    '2018_10_16_0004': [0,15.1],    '2018_11_13_0002': [0,66],
    '2018_11_13_0005': [19,-1],     '2018_11_13_0007': [0,76.1],    '2018_11_14_0001': [0,495],     '2018_11_18_0001': [0,141],
    '2018_11_18_0008': [0,15],      '2018_11_21_0001': [0,41],      '2019_04_19_0002': [0, 1220],   '2019_04_23_0011': [0, 1220],
    '2019_04_24_0002': [0,1220],
    '2019_06_06_0001': [0, 705],    '2019_06_07_0000': [0,550],     '2019_06_08_0000': [0, 2480],   '2019_06_08_0001': [0,3500],
    '2019_06_08_0003': [0, 440],
    '2019_06_09_0004': [0,160],     '2019_06_09_0014': [170,215],   '2019_06_13_0000': [0,1043],    '2019_06_26_0002': [0, 465],
    '2019_06_27_0002': [0,1720],    '2019_06_29_0000': [0,349],     '2019_06_29_0001': [0,1167],    '2019_06_29_0002': [0,2275],
    '2019_06_27_0004': [0,476],     '2019_06_30_0000': [0,2708],    '2019_07_17_0000': [0,372.1],
    '2019_07_17_0001': [150,-1],    '2019_07_17_0011': [0,150],     '2019_07_23_0000': [0,228],     '2019_07_24_0005': [0,160],
    '2019_07_24_0008': [0,3000],    '2019_08_02_0014': [0,3630],

    # '2019_06_20_0001': [0,300],
}


def read_abf(filename, io=None):
    """
    Returns a dictionary. Keys are the channel names in lowercase, values are the channels.

    # Usage:
    abf = read_abf('2015_08_03_0000.abf'). abf = dict, abf_filename = string pointing to the abf file.

    # To show all channel keys:
    abf.keys()

    # Channels have useful attributes:
    temp = abf['temp']
    temp_array = np.array(temp)
    t = np.array(temp.times)

    # To plot:
    plt.plot(t, temp_array)

    # Can also get sampling rate and period:
    temp.sampling_rate
    temp.sampling_period
    """

    if io is None:
        extension = filename.split('.')[-1]
        if extension == 'EDR':
            io = WinEdrIO
        elif extension == 'abf':
            io = AxonIO
        else:
            raise ValueError('Extension not recognized.')

    analog_signals_dict = {}
    if io == AxonIO:

        fh = io(filename=filename)
        segments = fh.read_block().segments

        if len(segments) > 1:
            print 'More than one segment in file.'
            return 0

        analog_signals_ls = segments[0].analogsignals
        for analog_signal in analog_signals_ls:
            analog_signals_dict[analog_signal.name.lower()] = analog_signal

    if io == WinEdrIO:
        fh = WinEdrIO(filename=filename)
        seg = fh.read_segment(lazy=False)

        for analog_signal in seg.analogsignals:
            analog_signals_dict[analog_signal.name.lower()] = analog_signal

    return analog_signals_dict


def volts2degs(volts, maxvolts=10):
    degs = volts*360/maxvolts
    degs[degs>360] = 360.
    degs[degs<0] = 0.
    return degs


# Collects recordings into a list of flies, each of which is a tuple of GCaMP_ABF recordings.

def get_recs(recnamess, parent_folder='./', bhtype='Flight', **kwargs):

    def get_trial_name(recname):
        year = recname[:4]
        month = recname[4:6]
        day = recname[6:8]
        recnum = recname[8:10]
        date = year + '_' + month + '_' + day
        return date + '/' + date + '_00' + recnum + '.abf'

    if bhtype == 'Flight':
        recs = [
            [Flight(parent_folder + get_trial_name(recname), **kwargs)
             for recname in recnames]
            for recnames in recnamess
        ]
    if bhtype == 'Walk':
        recs = [
            [Walk(parent_folder + get_trial_name(recname), **kwargs)
             for recname in recnames]
            for recnames in recnamess
        ]
    return recs

# adapted from Peter MP
# bhtype only Walk for now
# for now, only use it for reference
def get_abfs(recnamess, parent_folder='./',  light=False, reprocess=False, **kwargs):

    # creates a list of Walk objects using a list of filenames of abf or EDR files

    # def get_filename(recname):
    #     year = recname[:4]
    #     month = recname[4:6]
    #     day = recname[6:8]
    #     recnum = recname[8:10]
    #     date = year + '_' + month + '_' + day
    #     return date + '/' + date + '_00' + recnum + '.abf'

    def get_prefix(recname):
        year = recname[:4]
        month = recname[4:6]
        day = recname[6:8]
        recnum = recname[8:10]
        date = year + '_' + month + '_' + day
        return date + '/' + date + '_00' + recnum

    recss = []
    for i, recnames in enumerate(recnamess):
        recs = []
        for j, recname in enumerate(recnames):

            # Get filename and filetype/ext (.abf, .EDR, or .pickle)
            prefix = parent_folder + get_prefix(recname)
            if glob.glob(prefix + '.abf'):
                ext = 'abf'
            else:
                ext = 'EDR'

            # pickle_available = True if glob.glob(prefix + '.pickle') else False
            # if pickle_available and light and not reprocess:
            #     rec = pickle.load(open(prefix + '.pickle', 'rb'))
            # else:
            #     rec = bhtype(prefix + '.' + ext, light=light,
            #              **kwargs)
            #     if light:
            #         pickle.dump(rec, open(prefix + '.pickle', 'wb') )
            rec = Walk(prefix + '.' + ext, **kwargs)

            recs.append(rec)
        recss.append(recs)

    return recss

def slope_to_headning_angle_menotaxis(dx, dy):
    deg = np.degrees(np.arctan(dy / dx))
    if dx < 0:
        deg += 90
    else:
        deg -= 90
    return deg

def circdist_incomplete_arena_menotaxis(a, b, low=-135, high=135):
    # push bar positions that are in the dark to the edge of the seeable boundaries, described by low & high
    if a < low:
        a = low
    elif a > high:
        a = high
    if b < low:
        b = low
    elif b > high:
        b = high

    signal = np.array([a, b])
    period = high - low
    unwrapped = fc.unwrap(signal, period=period)
    return np.abs(unwrapped[-1] - unwrapped[0])

def get_stims_window_menotaxis(rec_stimid, t,):

    def get_trial_ts(rec_stimid, t, mode='normal', erosion_step=5, stitch_time=0.1):
        trials = np.gradient(rec_stimid) == 0
        trials = ndimage.binary_erosion(trials, iterations=erosion_step)
        trials = ndimage.binary_dilation(trials, iterations=erosion_step).astype(int)

        trial_starts = t[np.where(np.diff(trials) > 0)[0] + 1]
        trial_ends = t[np.where(np.diff(trials) < 0)[0] + 1]

        if len(trial_starts):
            trial_starts = np.insert(trial_starts, 0, t[0])
            trial_ends = np.append(trial_ends, t[-1])

            if mode == 'stitch':  # added by Cheng, Feb.9 2019. To deal with the blink noise in stimid recording ch.
                trial_starts_new = []
                trial_ends_new = []
                i_start = 0
                i_end = 0
                for i_trial in range(len(trial_starts) - 1):
                    stimid0 = get_stimid_in_window(t, [trial_starts[i_trial], trial_ends[i_trial]], rec_stimid)
                    stimid1 = get_stimid_in_window(t, [trial_starts[i_trial + 1], trial_ends[i_trial + 1]], rec_stimid)
                    if (stimid0 == stimid1) & (
                            trial_starts[i_trial + 1] - trial_ends[i_trial] <= stitch_time):  # stitch
                        i_end = i_trial + 1
                    else:  # output into trial_starts_new & trial_ends_new
                        trial_starts_new.append(trial_starts[i_start])
                        trial_ends_new.append(trial_ends[i_end])
                        i_start = i_trial + 1
                        i_end = i_end + 1
                trial_starts_new.append(trial_starts[i_start])
                trial_ends_new.append(trial_ends[i_end])
                return zip(np.array(trial_starts_new), np.array(trial_ends_new))
            else:
                return zip(trial_starts, trial_ends)
        else:
            return zip([t[0]], [t[-1]])

    def get_stimid_in_window(t, ts, rec_stimid):
        # window: ts: (t0, t1)
        t_ind = (t > ts[0]) & (t < ts[1])
        if len(np.unique(np.round(rec_stimid[t_ind]))) != 1:
            # print 'Error: stimid in period: %3.1f-%3.1f not unique!!' % (ts[0], ts[1])
            return np.round(np.mean(rec_stimid[t_ind]))
        return np.unique(np.round(rec_stimid[t_ind]))[0]

    stim_window_info = []
    tss = get_trial_ts(rec_stimid, t, mode='stitch')
    for i_ts, ts in enumerate(tss):
        if ts[-1] - ts[0] <= 2:
            continue
        stimid = get_stimid_in_window(t, ts, rec_stimid)
        ind0 = np.where(t >= ts[0])[0][0]
        ind1 = np.where(t <= ts[-1])[0][-1]
        stim_window_info.append([stimid, ind0, ind1])

    return stim_window_info

def get_stims_window_menotaxis_10secLED_20190824(stimid, led_trig, subsampling_rate):
    """
    :param stimid: stimid == 0 means the stimulus is in CL bar, -9 means dark
    :param led_trig: off 5min -- (on 10sec -- off 10min) x 3

    stimid baseline: -1 to -7, 5 min each, led - sham - led - sham - led - sham: 1 - 2 - 3 - 4 - 5 - 6, 10 sec each, i.e.

    -1 (off,5min), 1 (on 10sec), -2 (off,5min), 2 (sham,10sec) -3 (off,5min), 3 (on 10sec), -4 (off,5min), 4 (sham 10sec), -5 (off,5min), 5 (on 10sec), -6 (off,5min), 6 (sham 10sec), -7 (off,5min)
    """

    stim_window_info = []
    baseline_inds = (led_trig < 2.5) & (stimid == 0)
    baseline_idxs = fc.get_contiguous_inds(baseline_inds, min_contig_len=0, trim_left=0, trim_right=0, keep_ends=True)

    if len(baseline_idxs) != 4:
        raise ValueError('led_trig channel does not complete 3 trials, not 4 baseline periods')

    # detect sham and assign stimid
    stim_window_info.append([-1, int(baseline_idxs[0][0]), int(baseline_idxs[0][-1])])

    for i_trial in [1,2,3]:
        _idx_sham_on = int(baseline_idxs[i_trial][0] + 300 * subsampling_rate)
        _idx_sham_off = int(baseline_idxs[i_trial][0] + 310 * subsampling_rate)

        stim_window_info.append([i_trial*2-1, int(baseline_idxs[i_trial-1][-1]), int(baseline_idxs[i_trial][0])])       # LED on
        stim_window_info.append([-1 * (i_trial * 2), int(baseline_idxs[i_trial][0]), _idx_sham_on])                     # interval
        stim_window_info.append([i_trial*2, _idx_sham_on, _idx_sham_off])                                               # Sham
        stim_window_info.append([-1 * (i_trial * 2 + 1), _idx_sham_off, int(baseline_idxs[i_trial][-1])])               # interval

    return stim_window_info

def initialize_fb_variables_for_multiple_stimids(Nvariable, Nstimid):
    variables = []

    for i in range(Nvariable):
        variables.append([])
        for j in range(Nstimid):
            variables[-1].append([])

    return tuple(variables)


class ABF():

    def __init__(self, abffile, maxvolts=None, spike_detection=False, spike_detection_algo='butterworth', positive_spike=True,
                 convolve_sp=False, sp_min_contig_len=3, sp_butterworth_cutoff=100, sp_butterworth_N=8,
                 ext='abf', angle_offset=86, spike_halfwidth=.01, **kwargs):
        self.basename = ''.join(abffile.split('.')[:-1])
        self.abffile = abffile
        self.recid = abffile[-19:-4]
        self.maxvolts = maxvolts
        self.angle_offset = angle_offset  # used to be -274 deg
        self.arena_edge = 135
        self.ext = ext

        # spike analysis related
        self.spike_halfwidth = spike_halfwidth
        self.spike_threshold = 0.3 # hpf 100, N=8
        self.spike_detection = spike_detection
        self.convolve_sp = convolve_sp
        self.spike_detection_algo = spike_detection_algo
        self.positive_spike = positive_spike
        # 'butterworth' algorithm
        self.sp_min_contig_len = sp_min_contig_len
        self.sp_butterworth_cutoff = sp_butterworth_cutoff
        self.sp_butterworth_N = sp_butterworth_N

    def open(self, subsampling_rate=50, delay_ms=15, nstims=None, **kwargs):
        self.nstims = nstims
        self.t0 = 0
        abf = read_abf(self.abffile)
        self.createdtime = time.ctime(os.path.getctime(self.abffile))

        # only analyze the data in the time window specified in filename_to_thresh_dict{}
        if self.abffile[-19:-4] in filename_to_valid_time_dict:
            valid_time = filename_to_valid_time_dict[self.abffile[-19:-4]]
            if valid_time[0] > 0:
                valid_time_ind0 = np.where(np.array(abf[abf.keys()[0]].times) >= valid_time[0])[0][0]
            else:
                valid_time_ind0 = 0
            if valid_time[-1] > 0:
                valid_time_ind1 = np.where(np.array(abf[abf.keys()[0]].times) >= valid_time[-1])[0][0]
            else:
                valid_time_ind1 = len(abf[abf.keys()[0]].times)
            for key in abf.keys():
                abf[key] = abf[key][valid_time_ind0:valid_time_ind1]
            self.t0 = np.array(abf[abf.keys()[0]].times)[0]

        # Collect subsampling indices based on camera triggers.
        camtriglabel = fc.contains(['cameratri', 'camtrig', 'walkcamtri', 'flycamtri'], abf.keys())
        if 0:
        # if camtriglabel:
            # Use cameratrigger to get framerate for wingbeat tracking.
            # Better to use if cameratrigger is available because it aligns the
            # wing beats to the camera acquisition.
            camtrig = abf[camtriglabel]
            self.camtrig = np.array(camtrig)
            
            self.sampling_period = float(camtrig.sampling_period)
            self.sampling_rate = float(camtrig.sampling_rate)
            risingtrig = np.where(np.diff((self.camtrig>1).astype(np.int8))>0)[0]
            if len(risingtrig):
                subsampling_period_idx = np.diff(risingtrig).mean()
                self.subsampling_period = subsampling_period_idx*self.sampling_period
                self.subsampling_rate = 1./self.subsampling_period

                subsampling_inds = np.round(risingtrig - 1.5*subsampling_period_idx).astype(int)
                trim_idx = np.sum(subsampling_inds<0)
                self.subsampling_inds = subsampling_inds[trim_idx:]
                self.behaviour_inds = (risingtrig-10)[trim_idx:]

                self.t_orig = np.array(camtrig.times) - self.t0
                self.t = self.t_orig[self.subsampling_inds]
            else:
                print 'Camera Triggers are not available.'
                camtriglabel = None
        else:
            # print 'Camera Triggers are not available.'
            example = abf[abf.keys()[0]]
            self.sampling_period = float(example.sampling_period)
            self.sampling_rate = float(example.sampling_rate)
            self.subsampling_step = int(example.sampling_rate/subsampling_rate)
            self.subsampling_rate = float(example.sampling_rate/self.subsampling_step)
            self.subsampling_period = 1.0 / self.subsampling_rate

            inds = np.arange(len(example))
            self.subsampling_inds = inds[::self.subsampling_step]
            self.behaviour_inds = self.subsampling_inds + int(round((delay_ms/1000.)*self.sampling_rate))
            trim_idx = np.sum(self.behaviour_inds>len(example))
            if trim_idx:
                self.subsampling_inds = self.subsampling_inds[:-trim_idx]
                self.behaviour_inds = self.behaviour_inds[:-trim_idx]
            
            self.t_orig = np.array(example.times) - self.t0
            self.t = self.t_orig[self.subsampling_inds]

        # channels do not need to be subsampled
        ch1label = fc.contains(['ch1_patch', 'ipatch'], abf.keys())
        if ch1label:
            self.ch1 = np.array(abf[ch1label])
            self.ch1 -= 13
            self.ch1_lpf = fc.butterworth(self.ch1, cutoff=10, passside='low', N=2, sampling_freq=10000)
            self.ch1_subsampled = self.subsample_ephy_to_behavior(self.ch1_lpf)

        ch2label = fc.contains(['ch2_patch', 'i_mtest1'], abf.keys())
        if ch2label:
            self.ch2 = np.array(abf[ch2label])

        pwmlabel = fc.contains(['pwmdutycyc'], abf.keys())
        if pwmlabel:
            self.pwm = np.array(abf[pwmlabel])
            self.pwm_dc, self.pwm_dc_stimdcs, self.pwm_dc_idx_starts, self.pwm_dc_idx_ends, self.pwm_dc_subsampled = self.pwm_analysis(self.pwm)

        pwmlabel = fc.contains(['pwm2'], abf.keys())
        if pwmlabel:
            self.pwm2 = np.array(abf[pwmlabel])
            self.pwm2_dc, self.pwm2_dc_stimdcs, self.pwm2_dc_idx_starts, self.pwm2_dc_idx_ends, self.pwm2_dc_subsampled = self.pwm_analysis(self.pwm2)

        # Other channels to be subsampled.
        templabel = fc.contains(['temp'], abf.keys())
        if templabel:
            if self.ext == 'abf':
                self.temp = np.array(abf['temp'])[self.subsampling_inds]
            else:
                self.temp = np.ndarray.flatten(np.array(abf['temp']))[self.subsampling_inds]
        
        xstimlabel = fc.contains(['vstim_x18', 'vstim_x', 'stim_x', 'xstim', 'x_ch'], abf.keys())
        if xstimlabel:
            if self.ext == 'abf':
                self.xstim_orig = np.array(abf[xstimlabel])
            else:
                self.xstim_orig = np.ndarray.flatten(np.array(abf[xstimlabel]))
            self.xstim = self.xstim_orig[self.subsampling_inds]
            self.xstim_old = self.xstim + 96
            self.xbar = volts2degs(self.xstim, 10) + self.angle_offset
            self.xbar = fc.wrap(self.xbar)
            self.xbar_unwrap = fc.unwrap(self.xbar)
            # self.dxbar = fc.circgrad(self.xbar)*self.subsampling_rate
                
        ystimlabel = fc.contains(['vstim_y', 'stim_y', 'ystim', 'y_ch'], abf.keys())
        if ystimlabel:
            if self.ext == 'abf':
                self.ystim = np.array(abf[ystimlabel])[self.subsampling_inds]
            else:
                self.ystim = np.ndarray.flatten(np.array(abf[ystimlabel]))[self.subsampling_inds]
            self.ystim_uw = fc.unwrap(self.ystim, period=10)

        stimidlabel = fc.contains(['stimid', 'patid', 'patname'], abf.keys())
        if stimidlabel:
            if self.ext == 'abf':
                self.stimid_raw_orig = np.array(abf[stimidlabel])
            else:
                self.stimid_raw_orig = np.ndarray.flatten(np.array(abf[stimidlabel]))
            self.stimid_raw = self.stimid_raw_orig[self.subsampling_inds]
            if self.nstims:
                self.stimid = self.get_stimid(nstims)
                # self.stimid = self.stimid_orig[self.subsampling_inds]
            else:
                self.stimid = np.round(self.stimid_raw*10)/10.0

        pressurelabel = fc.contains(['pressure', 'pippuff'], abf.keys())
        if pressurelabel:
            self.pressure_orig = np.array(abf[pressurelabel])
            self.pressure = self.pressure_orig[self.subsampling_inds]
        
        vmlabel = fc.contains(['vm_b', 'vm'], abf.keys())
        if vmlabel:
            self.vm_orig = np.array(abf[vmlabel])
            self.vm = self.vm_orig[self.subsampling_inds]

        imlabel = fc.contains(['im_b', 'im'], abf.keys())
        if imlabel:
            self.im_orig = np.array(abf[imlabel])
            self.im = self.im_orig[self.subsampling_inds]

        pufflabel = fc.contains(['puff'], abf.keys())
        if pufflabel:
            self.puff = np.array(abf[pufflabel])[self.subsampling_inds]

        picopumplabel = fc.contains(['picopump', 'picotrig'], abf.keys())
        if picopumplabel:
            self.picopump = np.array(abf[picopumplabel])[self.subsampling_inds]
        # Channels that are not subsampled.
        frametriglabel = fc.contains(['frame tri', '2pframest', 'frametri'], abf.keys())
        if frametriglabel:
            self.frametrig = np.array(abf[frametriglabel])
        
        if 'pockels' in abf.keys():
            self.pockels = np.array(abf['pockels'])
        
        aolabel = fc.contains(['ao', 'arena_sti'], abf.keys())    
        if aolabel:
            self.arena_stim = np.array(abf[aolabel])

        magtriglabel = fc.contains(['magtrig'], abf.keys())
        if magtriglabel:
            self.magtrig = np.array(abf[magtriglabel])[self.subsampling_inds]

        pwmlabel = fc.contains(['pwm laser'], abf.keys())
        if pwmlabel:
            self.pwm = np.array(abf[pwmlabel])[self.subsampling_inds]

        vcamtriglabel = fc.contains(['vicamtrig', 'behcamtrig'], abf.keys())
        if vcamtriglabel:
            vcamtrig = abf[vcamtriglabel]
            vcamtrig = np.array(vcamtrig)
            # fallingtrig = np.where(np.diff((vcamtrig < 1).astype(np.int8)) > 0)[0]
            fallingtrig = np.where(np.diff((vcamtrig > 1).astype(np.int8)) > 0)[0]
            if len(fallingtrig):
                subsampling_inds = np.round(fallingtrig + .0 * np.diff(fallingtrig).mean()).astype(int)
                self.t_vcam = np.array(abf[abf.keys()[0]].times)[subsampling_inds]

        led_triglabel = fc.contains(['led_trig'], abf.keys())
        if led_triglabel:
            # self.led_trig = np.array(abf[led_triglabel])[self.subsampling_inds]
            if self.ext == 'abf':
                self.led_trig = np.array(abf[led_triglabel])[self.subsampling_inds]
            else:
                self.led_trig = np.ndarray.flatten(np.array(abf[led_triglabel])[self.subsampling_inds])

        # tags = AxonIO(filename=self.abffile).read_header()['listTag']
        # self.tags = np.array([tag['lTagTime'] for tag in tags])
        # if len(self.tags):
        #     self.t_tags = self.t_orig[self.tags]
        
        return abf

    def pwm_analysis_before20210731(self, pwm):
        print 'in def pwm_analysis'
        pwm_raw = (pwm >= 4.0)
        moving_average_nframe = 204  # first quick filter
        pwm_smooth = np.convolve(pwm_raw, np.ones((moving_average_nframe,)) / moving_average_nframe, mode='same')
        pwm_dc_rough = np.round(pwm_smooth * 50) / 50.  # duty cycle accurate to 0.02
        # the same duty cycle has to last at least 0.1 second
        idxss = fc.get_contiguous_inds(np.gradient(pwm_dc_rough) == 0, min_contig_len=1000, keep_ends=True)
        pwm_dc = copy.copy(pwm_dc_rough)
        for i_idxs, idxs in enumerate(idxss):
            if i_idxs < len(idxss) - 1:
                dc0 = pwm_dc_rough[idxs[-1]]  # duty cycle of previous period
                dc1 = pwm_dc_rough[idxss[i_idxs + 1][0]]  # duty cycle of next period
                idx_m = int(np.round((idxs[-1] + idxss[i_idxs + 1][0]) / 2.))  # middle point between two periods
                # if starting from idx_m, moves toward the period with larger dc, first time hit pwm > 4.8 decides the onset time
                if dc0 < dc1:
                    temp = np.where(pwm_raw[idx_m:idx_m+204])[0]
                    if len(temp):
                        idx = temp[0] + idx_m
                    else:
                        idx = idx_m
                else:
                    temp = np.where(pwm_raw[idx_m - 204:idx_m])[0]
                    if len(temp):
                        idx = temp[-1] + idx_m - 204
                    else:
                        idx = idx_m
                pwm_dc[idxs[-1]:idx] = pwm_dc_rough[idxs[-1]]
                pwm_dc[idx:idxss[i_idxs + 1][0]] = pwm_dc_rough[idxss[i_idxs + 1][0]]

        pwm_dc_stimdcs = []
        pwm_dc_idx_starts = []
        pwm_dc_idx_ends = []
        for pwm_dc_unique in np.unique(pwm_dc):
            idxss = fc.get_contiguous_inds(pwm_dc == pwm_dc_unique, min_contig_len=50, keep_ends=True)
            for idxs in idxss:
                pwm_dc_stimdcs.append(pwm_dc[idxs[0]])
                pwm_dc_idx_starts.append(idxs[0])
                pwm_dc_idx_ends.append(idxs[-1])

        return np.array(pwm_dc), np.array(pwm_dc_stimdcs), np.array(pwm_dc_idx_starts), np.array(pwm_dc_idx_ends), self.subsample_ephy_to_behavior(pwm_dc)

    def pwm_analysis(self, pwm):
        pwm_raw = (pwm >= 4.0)
        moving_average_nframe = 204  # first quick filter
        pwm_smooth = np.convolve(pwm_raw, np.ones((moving_average_nframe,)) / moving_average_nframe, mode='same')
        pwm_dc_rough = np.round(pwm_smooth * 100) / 100.  # duty cycle accurate to 0.02

        # non-zero pwm
        idxss = fc.get_contiguous_inds(pwm_dc_rough > 0.01, min_contig_len=1000, keep_ends=True)
        pwm_dc = pwm_dc_rough
        pwm_dc_stimdcs = []
        pwm_dc_idx_starts = []
        pwm_dc_idx_ends = []
        for i_idxs, idxs in enumerate(idxss):
            pwm_dc_stimdcs.append(np.round(np.mean(pwm_dc_rough[idxs]) * 100) / 100.)
            pwm_dc_idx_starts.append(idxs[0])
            pwm_dc_idx_ends.append(idxs[-1])

        # zero pwm
        idxss = fc.get_contiguous_inds(pwm_dc_rough < 0.01, min_contig_len=1000, keep_ends=True)
        pwm_dc = pwm_dc_rough
        pwm_dc_stimdcs = []
        pwm_dc_idx_starts = []
        pwm_dc_idx_ends = []
        for i_idxs, idxs in enumerate(idxss):
            pwm_dc_stimdcs.append(np.round(np.mean(pwm_dc_rough[idxs]) * 100) / 100.)
            pwm_dc_idx_starts.append(idxs[0])
            pwm_dc_idx_ends.append(idxs[-1])

        return np.array(pwm_dc), np.array(pwm_dc_stimdcs), np.array(pwm_dc_idx_starts), np.array(pwm_dc_idx_ends), self.subsample_ephy_to_behavior(pwm_dc)

    def subsample_ephy_to_behavior(self, signal):
        unit_num = len(self.t)
        ind_unit_length = int((self.sampling_rate / self.subsampling_rate))
        inds = np.arange(ind_unit_length * unit_num).reshape(unit_num, ind_unit_length)
        if len(self.t_orig) <= inds.size:
            inds[inds >= len(self.t_orig)] = len(self.t_orig)-1
        signal_subsampled = np.mean(signal[inds], axis=1)
        return signal_subsampled

    def get_temp_transition(self, transition_temp):
        if not hasattr(self, 'temp'):
            return 0
        
        t_heating_transitions = self.t[np.where(np.diff(self.temp > transition_temp) > 0)]
        #cooling_transitions = self.t[np.where(np.diff(self.temp > transition_temp) < 0)]
        if len(t_heating_transitions) > 0:
            return t_heating_transitions[0]
        else:
            return 0

    def get_stimid(self, nstims):
        centroids, _ = sc.vq.kmeans(self.stimid_raw, nstims+1)
        centroids = np.sort(centroids)
        stimid, _ = sc.vq.vq(self.stimid_raw, centroids)
        return np.array(stimid)

    def _is_hot(self, trial_inds, hot_min=30, hot_max=40):
        local_temp = self.temp[trial_inds]
        local_max_temp = max(local_temp)
        local_min_temp = min(local_temp)
        is_hot = (local_min_temp > hot_min) and (local_max_temp < hot_max)
        return is_hot.astype(int)

    def exponential_kernal(self, kernal_length_ind=3001, tc_ind=500):
        k = np.zeros(kernal_length_ind)
        for x in range(int(kernal_length_ind)):
            k[x] = np.exp(-x / tc_ind)
        return k

    def gaussian_kernal(self, kernal_length_ind=3001, tc_ind=500):
        if not kernal_length_ind % 2:
            kernal_length_ind += 1
        t0_ind = kernal_length_ind / 2 + 1
        k = np.zeros(kernal_length_ind)
        for x in range(int(kernal_length_ind)):
            k[x] = np.exp(-(x - t0_ind) ** 2 / 2.0 / tc_ind ** 2)
        return k, t0_ind

class Walk(ABF):
    angle_offset = 86  #used to be -274 deg
    arena_edge = 135

    def __init__(self, abffile, nstims=None, ball_diam_mm=6.34, maxvolts=2, bar_jump=True, **kwargs):
        ABF.__init__(self, abffile, **kwargs)
        self.bhtype = 'walk'
        self.ball_diam_mm = ball_diam_mm
        self.maxvolts = maxvolts
        self.open(nstims=nstims, bar_jump=bar_jump, **kwargs)
    
    def open(self, nstims=None, bar_jump=True, **kwargs):
        abf = ABF.open(self, subsampling_rate=50, delay_ms=13.6, nstims=nstims, **kwargs)

        headlabel = fc.contains(['ball_head', 'ball_headi'], abf.keys())
        # in case the last behaviour_inds exceeds the abf array
        if self.behaviour_inds[-1] == len(abf[headlabel]):
            self.behaviour_inds[-1] -= 1

        if self.ext == 'abf':
            self.head = volts2degs(np.array(abf[headlabel])[self.behaviour_inds], maxvolts=self.maxvolts)
        else:
            self.head = volts2degs(np.ndarray.flatten(np.array(abf[headlabel]))[self.behaviour_inds], maxvolts=self.maxvolts)
        self.head = fc.wrap(self.head)
        self.dhead = fc.circgrad(self.head)*self.subsampling_rate
        
        sidelabel = fc.contains(['ball_side', 'int_posy'], abf.keys())
        if self.ext == 'abf':
            self.side = volts2degs(np.array(abf[sidelabel])[self.behaviour_inds], maxvolts=self.maxvolts)
        else:
            self.side = volts2degs(np.ndarray.flatten(np.array(abf[sidelabel]))[self.behaviour_inds], maxvolts=self.maxvolts)
        self.side = fc.wrap(self.side)
        self.dside = self.deg2mm(fc.circgrad(self.side))*self.subsampling_rate
        self.sideuw = self.deg2mm(fc.unwrap(self.side))
        
        forwlabel = fc.contains(['ball_forw', 'int_posx'], abf.keys())
        if self.ext == 'abf':
            self.forw = volts2degs(np.array(abf[forwlabel])[self.behaviour_inds], maxvolts=self.maxvolts)
        else:
            self.forw = volts2degs(np.ndarray.flatten(np.array(abf[forwlabel]))[self.behaviour_inds], maxvolts=self.maxvolts)
        self.forw = fc.wrap(self.forw)
        self.dforw = self.deg2mm(fc.circgrad(self.forw))*self.subsampling_rate
        self.forwuw = self.deg2mm(fc.unwrap(self.forw))

        self.speed = np.hypot(self.dside, self.dforw)
        
        self.x, self.y, self.d = self.get_trajectory(bar_jump=bar_jump)

        thresh_stand_speed = 1  # speed criterion of standing, below, mm/s
        thresh_walk_speed = 2  # speed criterion of walking, above, mm/s
        thresh_stand_length = 0.2 * self.subsampling_rate  # duration criterion of standing, below, s*fps
        thresh_walk_length = 0.1 * self.subsampling_rate  # duration criterion of walking, below, s
        # detect when the fly is walking or standing
        # self.detect_walking_bouts(thresh_stand_speed, thresh_walk_speed, thresh_stand_length, thresh_walk_length)
    
    def parse(self):
        pass
    
    def _is_active(self, start_t, end_t, time_thresh=0.95):
        # Check that fly is walking forward for at least 95% of the trial
        #t = self.t
        #if end_t > t.max(): return False
        #flight_trial = self.flight[(t>=start_t) & (t<end_t)]
        #flying = np.sum(flight_trial)/len(flight_trial) > time_thresh
        return True
    
    def deg2mm(self, deg):
        circumference_mm = self.ball_diam_mm*np.pi
        mm = deg*(circumference_mm/360)
        return mm
    
    def get_dxdy(self, head_rad, side_mm, forw_mm):
        theta0 = math.atan2(side_mm, forw_mm)
        thetaf = theta0 + head_rad
        r = np.hypot(side_mm, forw_mm)
        dx = r*math.cos(thetaf)
        dy = r*math.sin(thetaf)
        return dx, dy

    def get_xiyi(self, x0, y0, head_rad, side_mm, forw_mm):
        dx, dy = self.get_dxdy(head_rad, forw_mm, side_mm)
        xi = x0 + dx
        yi = y0 + dy
        return xi, yi
    
    def total_forw_distance(self):
        dist = self.deg2mm(fc.circgrad(self.forw)).sum()
        return dist
    
    def get_trajectory(self, bar_jump=True, proj_forw=False):
        x = np.zeros(len(self.head))
        y = np.zeros(len(self.head))
        d = np.zeros(len(self.head))
        if bar_jump:
            # head_rads = np.radians(self.xbar) + np.pi/2
            head_rads = np.radians(self.xbar)
        else:
            head_rads = np.radians(self.head)
        dforw_mms = self.deg2mm(fc.circgrad(self.forw, np.diff))
        if proj_forw:
            dside_mms = np.zeros(len(self.dside))
        else:
            dside_mms = self.deg2mm(fc.circgrad(self.side, np.diff))
        
        for i in xrange(1, len(x)):
            headi = head_rads[i-1:i+1].mean()   # avg heading between i-1 and i
            dside = dside_mms[i-1]              # diff between i-1 and i
            dforw = dforw_mms[i-1]              # diff between i-1 and i
            
            xi, yi = self.get_xiyi(x[i-1], y[i-1], headi, dside, dforw)
            dx, dy = self.get_dxdy(headi, dside, dforw)
            x[i] = xi
            y[i] = yi
            d[i] = d[i-1] + np.sqrt(dx*dx+dy*dy)
        return x, y, d

    def detect_walking_bouts(self, thresh_stand_speed, thresh_walk_speed, thresh_stand_length, thresh_walk_length):
        speed = self.speed
        ls = np.zeros(self.N)       # locomotion_status, 0 means standing, 1 means walking

        if speed[0] < thresh_walk_speed:
            ls[0] = 0
        else:
            ls[0] = 1

        # Schmitt trigger
        for i in range(1, self.N):
            if ls[i-1]:
                if speed[i] <= thresh_stand_speed:
                    ls[i] = 0
                    continue
                else:
                    ls[i] = 1
                    continue
            else:
                if speed[i] > thresh_walk_speed:
                    ls[i] = 1
                    continue
                else:
                    ls[i] = 0
                    continue

        # Minimum duration for standing and walking bouts
        ind_trial = ls == 0
        inds_stand = fc.get_contiguous_inds(ind_trial, keep_ends=True, min_contig_len=0, trim_left=0, trim_right=0)
        ind_trial = ls == 1
        inds_walk = fc.get_contiguous_inds(ind_trial, keep_ends=True, min_contig_len=0, trim_left=0, trim_right=0)

        i_stand = 0                                                     # denoting next standing bout to examine
        i_walk = 0                                                      # denoting next walking bout to examine
        flag_next = np.mod(ls[0]+1, 2)
        while i_walk < len(inds_walk) or i_stand < len(inds_stand):
            if flag_next:                                               # now is standing, next time point is walking
                if i_walk >= len(inds_walk):
                    break
                if len(inds_walk[i_walk]) <= thresh_walk_length:        # change walking to standing in ls
                    inds = inds_walk[i_walk]
                    ls[inds[0]:inds[-1]+1] = 0
                    i_walk += 1
                    i_stand += 1
                    continue
                else:                                                   # jump to next standing bout
                    flag_next = np.mod(flag_next + 1, 2)
                    i_walk += 1
            else:                                                       # now is walking, next time point is standing
                if i_stand >= len(inds_stand):
                    break
                if len(inds_stand[i_stand]) <= thresh_stand_length:        # change walking to standing in ls
                    inds = inds_stand[i_stand]
                    ls[inds[0]:inds[-1]+1] = 1
                    i_walk += 1
                    i_stand += 1
                    continue
                else:                                                   # jump to next standing bout
                    flag_next = np.mod(flag_next + 1, 2)
                    i_stand += 1

        self.ls = ls


# created Apr21, 2019
class WalkBar_backup(Walk):

    def __init__(self, abffile, multi_stimid=False, **kwargs):
        Walk.__init__(self, abffile, **kwargs)
        self.multi_stimid = multi_stimid
        self.open_walkbar()

    def open_walkbar(self):
        self.step_size = 1                   # transform from time coordinate to distance coordinate
        self.smooth_window_size = 11         # savitzky golay filter, size of the smoothing window
        self.smooth_order = 5                # savitzky golay filter, order of the polynomials of fitting curves
        self.mhv_half_length = 100           # in unit of mm
        self.cri_mhv_height = 0.75
        self.cri_mhv_duration_mm = 200
        self.cri_mhv_avespeed = 2
        self.cri_similar_orientation = 20
        self.cri_DP_tol = 100

        # smooth trajectory
        self.smooth_trajectory()

        # convert some important variables from time coordinates to distance coordinates
        self.convert_to_distance_coordinate()

        # calculate the mean heading vector in distance coordinate
        self.mhv_x, self.mhv_y, self.mhv_xbar, self.mhv_amp = self.mean_heading_vector(self.mhv_half_length)

        if not self.multi_stimid:
            # separate individual fixation bout
            self.fb_xbar_raw, self.fb_dist_raw, self.fb_dshift_raw, self.fb_ave_speed_raw, self.fb_idxs_raw, self.fb_idxs_d_raw, \
            self.fb_xbar, self.fb_dist, self.fb_dshift, self.fb_ave_speed, self.fb_idxs, self.fb_idxs_d \
                = self.get_individual_fixation_bout(self.cri_mhv_height, self.cri_mhv_duration_mm, self.cri_mhv_avespeed, self.mhv_amp, self.mhv_xbar)

            # merge fixation bouts
            self.fb_xbar_merge, self.fb_dist_merge, self.fb_dshift_merge, self.fb_idxs_merge, self.fb_idxs_d_merge \
                = self.merge_fixation_bouts(self.cri_similar_orientation, self.fb_xbar, self.fb_dist, self.fb_dshift, self.fb_idxs, self.fb_idxs_d)

            # separate individual fixation bout using Douglas-Peuker algorithm
            self.DPfb_xbar_raw, self.DPfb_dist_raw, self.DPfb_dshift_raw, self.DPfb_ave_speed_raw, self.DPfb_idxs_raw, \
            self.DPfb_xbar, self.DPfb_dist, self.DPfb_dshift, self.DPfb_ave_speed, self.DPfb_idxs \
                = self.get_individual_fixation_bout_DP(self.cri_DP_tol, self.cri_mhv_duration_mm, self.cri_mhv_avespeed)

            # merge Douglas-Peuker fixation bouts
            self.DPfb_xbar_merge, self.DPfb_dist_merge, self.DPfb_dshift_merge, self.DPfb_idxs_merge \
                = self.merge_fixation_bouts_DP(self.cri_similar_orientation, self.DPfb_xbar, self.DPfb_dist, self.DPfb_dshift, self.DPfb_idxs)

        else:
            # get the info about different trial stages, defined by stims: self.stim_window_info = [[stimid, ind0, ind1], ...]
            self.get_stims_window()

            self.fb_xbar_raw, self.fb_dist_raw, self.fb_dshift_raw, self.fb_ave_speed_raw, self.fb_idxs_raw, \
            self.fb_idxs_d_raw, self.fb_xbar, self.fb_dist, self.fb_dshift, self.fb_ave_speed, self.fb_idxs, \
            self.fb_idxs_d, self.fb_xbar_merge, self.fb_dist_merge, self.fb_dshift_merge, self.fb_idxs_merge, \
            self.fb_idxs_d_merge, self.DPfb_xbar_raw, self.DPfb_dist_raw, self.DPfb_dshift_raw, self.DPfb_ave_speed_raw, \
            self.DPfb_idxs_raw, self.DPfb_xbar, self.DPfb_dist, self.DPfb_dshift, self.DPfb_ave_speed, self.DPfb_idxs, \
            self.DPfb_xbar_merge, self.DPfb_dist_merge, self.DPfb_dshift_merge, self.DPfb_idxs_merge \
                = self.initialize_fb_variables_for_multiple_stimids(len(self.stim_window_info))

            for i_stim, (stimid, ind0, ind1) in enumerate(self.stim_window_info):

                # separate fixation bouts
                d0, d1 = self.d[ind0], self.d[ind1]
                inds_ = (self.d_d >= d0) & (self.d_d <= d1)
                self.fb_xbar_raw[i_stim], self.fb_dist_raw[i_stim], self.fb_dshift_raw[i_stim], \
                self.fb_ave_speed_raw[i_stim], self.fb_idxs_raw[i_stim], self.fb_idxs_d_raw[i_stim], self.fb_xbar[i_stim], \
                self.fb_dist[i_stim], self.fb_dshift[i_stim], self.fb_ave_speed[i_stim], self.fb_idxs[i_stim], self.fb_idxs_d[i_stim] \
                    = self.get_individual_fixation_bout(self.cri_mhv_height, self.cri_mhv_duration_mm,
                                                        self.cri_mhv_avespeed, self.mhv_amp, self.mhv_xbar, inds_)

                # merge fixation bouts
                self.fb_xbar_merge[i_stim], self.fb_dist_merge[i_stim], self.fb_dshift_merge[i_stim], self.fb_idxs_merge[i_stim], self.fb_idxs_d_merge[i_stim] \
                    = self.merge_fixation_bouts(self.cri_similar_orientation, self.fb_xbar[i_stim], self.fb_dist[i_stim],
                                                self.fb_dshift[i_stim], self.fb_idxs[i_stim], self.fb_idxs_d[i_stim])

                # separate individual fixation bout using Douglas-Peuker algorithm
                self.DPfb_xbar_raw[i_stim], self.DPfb_dist_raw[i_stim], self.DPfb_dshift_raw[i_stim],\
                self.DPfb_ave_speed_raw[i_stim], self.DPfb_idxs_raw[i_stim], self.DPfb_xbar[i_stim], \
                self.DPfb_dist[i_stim], self.DPfb_dshift[i_stim], self.DPfb_ave_speed[i_stim], self.DPfb_idxs[i_stim] \
                    = self.get_individual_fixation_bout_DP(self.cri_DP_tol, self.cri_mhv_duration_mm, self.cri_mhv_avespeed, ind0, ind1)

                # merge Douglas-Peuker fixation bouts
                self.DPfb_xbar_merge[i_stim], self.DPfb_dist_merge[i_stim], self.DPfb_dshift_merge[i_stim], \
                self.DPfb_idxs_merge[i_stim] \
                    = self.merge_fixation_bouts_DP(self.cri_similar_orientation, self.DPfb_xbar[i_stim], self.DPfb_dist[i_stim],
                                                   self.DPfb_dshift[i_stim], self.DPfb_idxs[i_stim])

    def smooth_trajectory(self):
        self.x_smooth = fc.savitzky_golay(self.x, self.smooth_window_size, self.smooth_order)
        self.y_smooth = fc.savitzky_golay(self.y, self.smooth_window_size, self.smooth_order)
        self.d_smooth = np.zeros(len(self.x))
        self.d_smooth[1:] = np.sqrt(np.square(self.x_smooth[1:] - self.x_smooth[:-1]) + np.square(self.y_smooth[1:] - self.y_smooth[:-1]))
        self.d_smooth = np.cumsum(self.d_smooth)

    def convert_to_distance_coordinate(self):
        xbar_unwrap = fc.unwrap(self.xbar)
        d = self.d
        idx0 = 0
        xbar_d = []
        hv_x = []
        hv_y = []
        for i in np.arange(self.step_size, d[-1] - .01, self.step_size):
            idx = np.where(d <= i)[0][-1]
            if idx == idx0:
                continue
            xbar_d.append(np.radians(np.mean(xbar_unwrap[idx0:idx])))
            hv_x.append(np.cos(xbar_d[-1] + np.pi / 2))
            hv_y.append(np.sin(xbar_d[-1] + np.pi / 2))
            idx0 = idx

        self.xbar_d = np.array(xbar_d)
        self.hv_x = np.array(hv_x)
        self.hv_y = np.array(hv_y)
        self.D = len(xbar_d)
        self.d_d = np.arange(self.D) + self.step_size / 2.
        self.r = np.hypot(self.x, self.y)

    def mean_heading_vector(self, mhv_half_length):
        mhv_x = np.zeros(self.D)
        mhv_y = np.zeros(self.D)
        mhv_xbar = np.zeros(self.D)
        mhv_amp = np.zeros(self.D)
        for i in np.arange(mhv_half_length, self.D - mhv_half_length):
            mhv_x[i] = np.mean(self.hv_x[i - mhv_half_length:i + mhv_half_length])
            mhv_y[i] = np.mean(self.hv_y[i - mhv_half_length:i + mhv_half_length])
            mhv_xbar[i] = self.slope_to_headning_angle(mhv_x[i], mhv_y[i])
            mhv_amp[i] = np.hypot(mhv_x[i], mhv_y[i])

        return np.array(mhv_x), np.array(mhv_y), np.array(mhv_xbar), np.array(mhv_amp)

    def get_individual_fixation_bout(self, cri_mhv_height, cri_mhv_duration_mm, cri_mhv_avespeed, mhv_amp, mhv_xbar, _inds=[]):
        # separate fixation bouts
        # fb_dshift and DPfd_dist are calculating the same thing, not fb_dist
        inds = mhv_amp > cri_mhv_height
        if len(_inds):
            inds = inds & _inds

        idxss = fc.get_contiguous_inds(inds, min_contig_len=0, trim_left=0, trim_right=0, keep_ends=True)

        # calculate fixation angle, in unit of deg
        _fb_xbar_raw, _fb_dist_raw, _fb_dshift_raw, _fb_ave_speed_raw, _fb_idxs_raw, _fb_idxs_d_raw = [], [], [], [], [], []
        for i, idxs in enumerate(idxss):
            _fb_xbar_raw.append(fc.circmean(mhv_xbar[idxs]))
            _fb_dist_raw.append(len(idxs)*self.step_size)

            ind0 = np.where(self.d <= self.d_d[idxs[0]])[0][-1]
            ind1 = np.where(self.d >= self.d_d[idxs[-1]])[0][0]
            t0 = self.t[ind0]
            t1 = self.t[ind1]
            _fb_dshift_raw.append(np.hypot((self.x[ind0] - self.x[ind1]), (self.y[ind0] - self.y[ind1])))
            _fb_ave_speed_raw.append(_fb_dist_raw[-1]/(t1-t0))
            _fb_idxs_raw.append([np.where(self.d <= self.d_d[idxs[0]])[0][-1], np.where(self.d >= self.d_d[idxs[-1]])[0][0]])
            _fb_idxs_d_raw.append([idxs[0], idxs[-1]])

        fb_xbar_raw, fb_dist_raw, fb_dshift_raw, fb_ave_speed_raw, fb_idxs_raw, fb_idxs_d_raw = \
            np.array(_fb_xbar_raw), np.array(_fb_dist_raw), np.array(_fb_dshift_raw), np.array(_fb_ave_speed_raw), _fb_idxs_raw, _fb_idxs_d_raw

        # applying duration and average speed criteria
        _fb_xbar, _fb_dist, _fb_dshift, _fb_ave_speed, _fb_idxs, _fb_idxs_d = [], [], [], [], [], []
        for i_fb in range(len(fb_xbar_raw)):
            if (fb_dist_raw[i_fb] >= cri_mhv_duration_mm) \
                    and (fb_ave_speed_raw[i_fb] >= cri_mhv_avespeed):
                _fb_xbar.append(fb_xbar_raw[i_fb])
                _fb_dist.append(fb_dist_raw[i_fb])
                _fb_dshift.append(fb_dshift_raw[i_fb])
                _fb_ave_speed.append(fb_ave_speed_raw[i_fb])
                _fb_idxs.append(fb_idxs_raw[i_fb])
                _fb_idxs_d.append(fb_idxs_d_raw[i_fb])
        fb_xbar, fb_dist, fb_dshift, fb_ave_speed, fb_idxs, fb_idxs_d = \
            np.array(_fb_xbar), np.array(_fb_dist), np.array(_fb_dshift), np.array(_fb_ave_speed), _fb_idxs, _fb_idxs_d

        return fb_xbar_raw, fb_dist_raw, fb_dshift_raw, fb_ave_speed_raw, fb_idxs_raw, fb_idxs_d_raw, \
               fb_xbar, fb_dist, fb_dshift, fb_ave_speed, fb_idxs, fb_idxs_d

    def get_individual_fixation_bout_DP(self, cri_DP_tol, cri_mhv_duration_mm, cri_mhv_avespeed, _idx0=0, _idx1=0):
        if (not _idx0) and (not _idx1):
            traj = list(np.array([self.x, self.y]).T)
        else:
            traj = list(np.array([self.x[_idx0:_idx1], self.y[_idx0:_idx1]]).T)
        simplified = np.array(simplify_coords(traj, cri_DP_tol))
        N_DPfb = len(simplified) - 1

        _DPfb_xbar_raw, _DPfb_dist_raw, _DPfb_dshift_raw, _DPfb_ave_speed_raw, _DPfb_idxs_raw, _DPfb_turn_idx = \
            np.zeros(N_DPfb), np.zeros(N_DPfb), np.zeros(N_DPfb), np.zeros(N_DPfb), [], []

        for i in range(N_DPfb+1):
            idx = np.where(self.x == simplified[i][0])[0][0]
            _DPfb_turn_idx.append(idx)

        for i in range(N_DPfb):
            dx = simplified[i + 1][0] - simplified[i][0]
            dy = simplified[i + 1][1] - simplified[i][1]
            _DPfb_xbar_raw[i] = self.slope_to_headning_angle(dx, dy)
            _DPfb_dist_raw[i] = self.d[_DPfb_turn_idx[i + 1]] - self.d[_DPfb_turn_idx[i]]
            _DPfb_dshift_raw[i] = math.hypot(dx, dy)
            _DPfb_ave_speed_raw[i] = _DPfb_dist_raw[i] / (self.t[_DPfb_turn_idx[i+1]] - self.t[_DPfb_turn_idx[i]])
            _DPfb_idxs_raw.append([_DPfb_turn_idx[i], _DPfb_turn_idx[i+1]])

        DPfb_xbar_raw, DPfb_dist_raw, DPfb_dshift_raw, DPfb_ave_speed_raw, DPfb_idxs_raw = \
            np.array(_DPfb_xbar_raw), np.array(_DPfb_dist_raw), np.array(_DPfb_dshift_raw), np.array(_DPfb_ave_speed_raw), _DPfb_idxs_raw

        _DPfb_xbar, _DPfb_dist, _DPfb_dshift, _DPfb_ave_speed, _DPfb_idxs = [], [], [], [], []
        for i_fb in range(len(DPfb_xbar_raw)):
            if (DPfb_dist_raw[i_fb] >= cri_mhv_duration_mm) \
                    and (DPfb_ave_speed_raw[i_fb] >= cri_mhv_avespeed):
                _DPfb_xbar.append(DPfb_xbar_raw[i_fb])
                _DPfb_dist.append(DPfb_dist_raw[i_fb])
                _DPfb_dshift.append(DPfb_dshift_raw[i_fb])
                _DPfb_ave_speed.append(DPfb_ave_speed_raw[i_fb])
                _DPfb_idxs.append(DPfb_idxs_raw[i_fb])

        DPfb_xbar, DPfb_dist, DPfb_dshift, DPfb_ave_speed, DPfb_idxs = \
            np.array(_DPfb_xbar), np.array(_DPfb_dist), np.array(_DPfb_dshift), np.array(_DPfb_ave_speed), _DPfb_idxs

        return DPfb_xbar_raw, DPfb_dist_raw, DPfb_dshift_raw, DPfb_ave_speed_raw, DPfb_idxs_raw, \
               DPfb_xbar, DPfb_dist, DPfb_dshift, DPfb_ave_speed, DPfb_idxs

    def merge_fixation_bouts(self, cri_similar_orientation, fb_xbar, fb_dist, fb_dshift, fb_idxs, fb_idxs_d):

        def new_queue():
            xbar = fb_xbar[i_fb]
            dist = fb_dist[i_fb]
            dshift = fb_dshift[i_fb]
            idxs = fb_idxs[i_fb]
            idxs_d = fb_idxs_d[i_fb]
            return xbar, dist, dshift, idxs, idxs_d

        def fill_queue(xbar, dist, dshift, idxs, idxs_d):
            angles = fc.unwrap(np.array([xbar, fb_xbar[i_fb]]))
            angle = (angles[0] * dist + angles[-1] * fb_dist[i_fb]) / (dist + fb_dist[i_fb])
            xbar = fc.wrap(np.array([angle]))[0]
            dist += fb_dist[i_fb]
            dshift += fb_dshift[i_fb]
            idxs = [idxs[0], fb_idxs[i_fb][-1]]
            idxs_d = [idxs_d[0], fb_idxs_d[i_fb][-1]]
            return xbar, dist, dshift, idxs, idxs_d

        def clear_queue(xbar, dist, dshift, idxs, idxs_d):
            fb_xbar_merge.append(xbar)
            fb_dist_merge.append(dist)
            fb_dshift_merge.append(dshift)
            fb_idxs_merge.append(idxs)
            fb_idxs_d_merge.append(idxs_d)

        fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge, fb_idxs_d_merge = [], [], [], [], []

        # deal with recording that doesn't have or only has one fixation bout
        if len(fb_xbar) == 0:
            fb_xbar_merge = np.array([])
            fb_dist_merge = np.array([])
            fb_idxs_merge = []
            fb_idxs_d_merge = []
            return fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge, fb_idxs_d_merge

        elif len(fb_xbar) == 1:
            fb_xbar_merge = [fb_xbar[0]]
            fb_dist_merge = [fb_dist[0]]
            fb_dshift_merge = [fb_dshift[0]]
            fb_idxs_merge = [fb_idxs[0]]
            fb_idxs_d_merge = [fb_idxs_d[0]]
            return fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge, fb_idxs_d_merge

        else:
            xbar = fb_xbar[0]
            dist = fb_dist[0]
            dshift = fb_dshift[0]
            idxs = fb_idxs[0]
            idxs_d = fb_idxs_d[0]

        for i_fb in np.arange(1, len(fb_xbar)):
            if self.circdist_incomplete_arena(xbar, fb_xbar[i_fb]) <= cri_similar_orientation:
                xbar, dist, dshift, idxs, idxs_d = fill_queue(xbar, dist, dshift, idxs, idxs_d)     # merge
            else:
                clear_queue(xbar, dist, dshift, idxs, idxs_d)                               # clear queue
                xbar, dist, dshift, idxs, idxs_d = new_queue()                              # new queue
            if i_fb == (len(fb_xbar) - 1):  # last one, clear queue
                clear_queue(xbar, dist, dshift, idxs, idxs_d)

        fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge, fb_idxs_d_merge = \
            np.array(fb_xbar_merge), np.array(fb_dist_merge), np.array(fb_dshift_merge), fb_idxs_merge, fb_idxs_d_merge

        return fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge, fb_idxs_d_merge

    def merge_fixation_bouts_DP(self, cri_similar_orientation, fb_xbar, fb_dist, fb_dshift, fb_idxs):

        def new_queue():
            xbar = fb_xbar[i_fb]
            dist = fb_dist[i_fb]
            dshift = fb_dshift[i_fb]
            idxs = fb_idxs[i_fb]
            return xbar, dist, dshift, idxs

        def fill_queue(xbar, dist, dshift, idxs):
            angles = fc.unwrap(np.array([xbar, fb_xbar[i_fb]]))
            angle = (angles[0] * dist + angles[-1] * fb_dist[i_fb]) / (dist + fb_dist[i_fb])
            xbar = fc.wrap(np.array([angle]))[0]
            dist += fb_dist[i_fb]
            dshift += fb_dshift[i_fb]
            idxs = [idxs[0], fb_idxs[i_fb][-1]]
            return xbar, dist, dshift, idxs

        def clear_queue(xbar, dist, dshift, idxs):
            fb_xbar_merge.append(xbar)
            fb_dist_merge.append(dist)
            fb_dshift_merge.append(dshift)
            fb_idxs_merge.append(idxs)

        fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge = [], [], [], []

        # deal with recording that doesn't have or only has one fixation bout
        if len(fb_xbar) == 0:
            fb_xbar_merge = np.array([])
            fb_dist_merge = np.array([])
            fb_dshift_merge = np.array([])
            fb_idxs_merge = []
            return fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge

        elif len(self.fb_xbar) == 1:
            fb_xbar_merge = [fb_xbar[0]]
            fb_dist_merge = [fb_dist[0]]
            fb_dshift_merge = [fb_dshift[0]]
            fb_idxs_merge = [fb_idxs[0]]
            return fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge

        else:
            xbar = fb_xbar[0]
            dist = fb_dist[0]
            dshift = fb_dshift[0]
            idxs = fb_idxs[0]

        for i_fb in np.arange(1, len(fb_xbar)):
            if self.circdist_incomplete_arena(xbar, fb_xbar[i_fb]) <= cri_similar_orientation:
                xbar, dist, dshift, idxs = fill_queue(xbar, dist, dshift, idxs)     # merge
            else:
                clear_queue(xbar, dist, dshift, idxs)                               # clear queue
                xbar, dist, dshift, idxs = new_queue()                              # new queue
            if i_fb == (len(fb_xbar) - 1):  # last one, clear queue
                clear_queue(xbar, dist, dshift, idxs)

        fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge = \
            np.array(fb_xbar_merge), np.array(fb_dist_merge), np.array(fb_dshift_merge), fb_idxs_merge

        return fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge

    def slope_to_headning_angle(self, dx, dy):
        deg = np.degrees(np.arctan(dy / dx))
        if dx < 0:
            deg += 90
        else:
            deg -= 90
        return deg

    def circdist_incomplete_arena(self, a, b, low=-135, high=135):
        # push bar positions that are in the dark to the edge of the seeable boundaries, described by low & high
        if a < low:
            a = low
        elif a > high:
            a = high
        if b < low:
            b = low
        elif b > high:
            b = high

        signal = np.array([a, b])
        period = high - low
        unwrapped = fc.unwrap(signal, period=period)
        return np.abs(unwrapped[-1] - unwrapped[0])

    def get_stims_window(self):
        self.stim_window_info = []
        tss = self.get_trial_ts(mode='stitch')
        for i_ts, ts in enumerate(tss):
            if ts[-1] - ts[0] <= 2:
                continue
            stimid = self.get_stimid_in_window_sf(ts)
            ind0 = np.where(self.t >= ts[0])[0][0]
            ind1 = np.where(self.t <= ts[-1])[0][-1]
            self.stim_window_info.append([stimid, ind0, ind1])

    def get_trial_ts(self, mode='normal', erosion_step=5, stitch_time=0.1):
        trials = np.gradient(self.stimid) == 0
        trials = ndimage.binary_erosion(trials, iterations=erosion_step)
        trials = ndimage.binary_dilation(trials, iterations=erosion_step).astype(int)

        trial_starts = self.t[np.where(np.diff(trials) > 0)[0] + 1]
        trial_ends = self.t[np.where(np.diff(trials) < 0)[0] + 1]

        if len(trial_starts):
            if trial_starts[0] > trial_ends[0]:
                # trial_starts = trial_starts
                trial_ends = np.append(trial_ends[1:], self.t[-1])

            if mode == 'stitch':    # added by Cheng, Feb.9 2019. To deal with the blink noise in stimid recording ch.
                t = self.t
                trial_starts_new = []
                trial_ends_new = []
                i_start = 0
                i_end = 0
                for i_trial in range(len(trial_starts)-1):
                    stimid0 = self.get_stimid_in_window_sf([trial_starts[i_trial], trial_ends[i_trial]])
                    stimid1 = self.get_stimid_in_window_sf([trial_starts[i_trial+1], trial_ends[i_trial+1]])
                    if (stimid0 == stimid1) & (trial_starts[i_trial+1] - trial_ends[i_trial] <= stitch_time):    # stitch
                        i_end = i_trial + 1
                    else:   # output into trial_starts_new & trial_ends_new
                        trial_starts_new.append(trial_starts[i_start])
                        trial_ends_new.append(trial_ends[i_end])
                        i_start = i_trial + 1
                        i_end = i_end + 1
                trial_starts_new.append(trial_starts[i_start])
                trial_ends_new.append(trial_ends[i_end])
                return zip(np.array(trial_starts_new), np.array(trial_ends_new))
            else:
                return zip(trial_starts, trial_ends)
        else:
            return zip([self.t[0]], [self.t[-1]])

    def get_stimid_in_window_sf(self, ts):
        # window: ts: (t0, t1)
        t_ind = (self.t > ts[0]) & (self.t < ts[1])
        if len(np.unique(np.round(self.stimid[t_ind]))) != 1:
            # print 'Error: stimid in period: %3.1f-%3.1f not unique!!' % (ts[0], ts[1])
            return np.round(np.mean(self.stimid[t_ind]))
        return np.unique(np.round(self.stimid[t_ind]))[0]

    def initialize_fb_variables_for_multiple_stimids(self, Nstimid):
        fb_xbar_raw, fb_dist_raw, fb_dshift_raw, fb_ave_speed_raw, fb_idxs_raw, fb_idxs_d_raw, fb_xbar, fb_dist, \
        fb_dshift, fb_ave_speed, fb_idxs, fb_idxs_d, fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge, \
        fb_idxs_d_merge, DPfb_xbar_raw, DPfb_dist_raw, DPfb_dshift_raw, DPfb_ave_speed_raw, DPfb_idxs_raw, DPfb_xbar, \
        DPfb_dist, DPfb_dshift, DPfb_ave_speed, DPfb_idxs, DPfb_xbar_merge, DPfb_dist_merge, DPfb_dshift_merge, DPfb_idxs_merge \
            = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

        for i in range(Nstimid):
            fb_xbar_raw.append([]), fb_dist_raw.append([]), fb_dshift_raw.append([]), fb_ave_speed_raw.append([]), \
            fb_idxs_raw.append([]), fb_idxs_d_raw.append([]), fb_xbar.append([]), fb_dist.append([]), \
            fb_dshift.append([]), fb_ave_speed.append([]), fb_idxs.append([]), fb_idxs_d.append([]), \
            fb_xbar_merge.append([]), fb_dist_merge.append([]), fb_dshift_merge.append([]), fb_idxs_merge.append([]), \
            fb_idxs_d_merge.append([]), DPfb_xbar_raw.append([]), DPfb_dist_raw.append([]), DPfb_dshift_raw.append([]), \
            DPfb_ave_speed_raw.append([]), DPfb_idxs_raw.append([]), DPfb_xbar.append([]), DPfb_dist.append([]), \
            DPfb_dshift.append([]), DPfb_ave_speed.append([]), DPfb_idxs.append([]), DPfb_xbar_merge.append([]),\
            DPfb_dist_merge.append([]), DPfb_dshift_merge.append([]), DPfb_idxs_merge.append([])

        return fb_xbar_raw, fb_dist_raw, fb_dshift_raw, fb_ave_speed_raw, fb_idxs_raw, fb_idxs_d_raw, fb_xbar, fb_dist,\
               fb_dshift, fb_ave_speed, fb_idxs, fb_idxs_d, fb_xbar_merge, fb_dist_merge, fb_dshift_merge, \
               fb_idxs_merge, fb_idxs_d_merge, DPfb_xbar_raw, DPfb_dist_raw, DPfb_dshift_raw, DPfb_ave_speed_raw, \
               DPfb_idxs_raw, DPfb_xbar, DPfb_dist, DPfb_dshift, DPfb_ave_speed, DPfb_idxs, DPfb_xbar_merge, \
               DPfb_dist_merge, DPfb_dshift_merge, DPfb_idxs_merge

    def recalculate_fb(self, mhv_half_length, cri_mhv_height, cri_mhv_duration_mm, cri_mhv_avespeed, cri_similar_orientation):
        self.mhv_half_length, self.cri_mhv_height, self.cri_mhv_duration_mm, self.cri_mhv_avespeed, self.cri_similar_orientation \
            = mhv_half_length, cri_mhv_height, cri_mhv_duration_mm, cri_mhv_avespeed, cri_similar_orientation

        # calculate the mean heading vector in distance coordinate
        self.mhv_x, self.mhv_y, self.mhv_xbar, self.mhv_amp = self.mean_heading_vector(self.mhv_half_length)

        # separate individual fixation bout
        self.fb_xbar_raw, self.fb_dist_raw, self.fb_dshift_raw, self.fb_ave_speed_raw, self.fb_idxs_raw, self.fb_idxs_d_raw, \
        self.fb_xbar, self.fb_dist, self.fb_dshift, self.fb_ave_speed, self.fb_idxs, self.fb_idxs_d \
            = self.get_individual_fixation_bout(self.cri_mhv_height, self.cri_mhv_duration_mm, self.cri_mhv_avespeed,
                                                self.mhv_amp, self.mhv_xbar)

        # merge fixation bouts
        self.fb_xbar_merge, self.fb_dist_merge, self.fb_dshift_merge, self.fb_idxs_merge, self.fb_idxs_d_merge \
            = self.merge_fixation_bouts(self.cri_similar_orientation, self.fb_xbar, self.fb_dist, self.fb_dshift,
                                        self.fb_idxs, self.fb_idxs_d)

    def recalculate_fb_DP(self, cri_DP_tol, cri_mhv_duration_mm, cri_mhv_avespeed, cri_similar_orientation):
        self.cri_DP_tol, self.cri_mhv_duration_mm, self.cri_mhv_avespeed, self.cri_similar_orientation \
            = cri_DP_tol, cri_mhv_duration_mm, cri_mhv_avespeed, cri_similar_orientation

        # separate individual fixation bout using Douglas-Peuker algorithm
        self.DPfb_xbar_raw, self.DPfb_dist_raw, self.DPfb_dshift_raw, self.DPfb_ave_speed_raw, self.DPfb_idxs_raw, \
        self.DPfb_xbar, self.DPfb_dist, self.DPfb_dshift, self.DPfb_ave_speed, self.DPfb_idxs \
            = self.get_individual_fixation_bout_DP(self.cri_DP_tol, self.cri_mhv_duration_mm, self.cri_mhv_avespeed)

        # merge Douglas-Peuker fixation bouts
        self.DPfb_xbar_merge, self.DPfb_dist_merge, self.DPfb_dshift_merge, self.DPfb_idxs_merge \
            = self.merge_fixation_bouts_DP(self.cri_similar_orientation, self.DPfb_xbar, self.DPfb_dist,
                                           self.DPfb_dshift, self.DPfb_idxs)

# created Apr21, 2019, put calculate fixation bout outside the class
class WalkBar(Walk):

    def __init__(self, abffile, multi_stimid=False, stimid_algorithm='multi_stimid', **kwargs):
        Walk.__init__(self, abffile, **kwargs)
        self.multi_stimid = multi_stimid
        self.stimid_algorithm = stimid_algorithm
        self.open_walkbar()

    def open_walkbar(self):
        self.step_size = 1                   # transform from time coordinate to distance coordinate
        self.smooth_window_size = 11         # savitzky golay filter, size of the smoothing window
        self.smooth_order = 5                # savitzky golay filter, order of the polynomials of fitting curves
        self.mhv_half_length = 100           # in unit of mm
        self.cri_mhv_height = 0.75
        self.cri_mhv_duration_mm = 200
        self.cri_mhv_avespeed = 2
        self.cri_similar_orientation = 20
        self.cri_DP_tol = 100

        # smooth trajectory
        self.smooth_trajectory()

        # convert some important variables from time coordinates to distance coordinates
        self.convert_to_distance_coordinate()

        if self.stimid_algorithm == 'multi_stimid':
            # abf channel 'stimid'
            self.stim_window_info = get_stims_window_menotaxis(self.stimid, self.t)
        elif self.stimid_algorithm == 'led_trig_20190824':
            # abf channel 'led_trig': off 5min -- (on 10sec -- off 10min) x 3
            self.stim_window_info = get_stims_window_menotaxis_10secLED_20190824(self.stimid, self.led_trig, self.subsampling_rate)
        else:
            # single stimid
            self.stim_window_info = []

        _, _, _, _, _, self.mhv_x, self.mhv_y, self.mhv_xbar, self.mhv_amp, \
        self.fb_xbar_raw, self.fb_dist_raw, self.fb_dshift_raw, self.fb_ave_speed_raw, self.fb_idxs_raw, self.fb_idxs_d_raw, \
        self.fb_xbar, self.fb_dist, self.fb_dshift, self.fb_ave_speed, self.fb_idxs, self.fb_idxs_d, \
        self.fb_xbar_merge, self.fb_dist_merge, self.fb_dshift_merge, self.fb_idxs_merge, self.fb_idxs_d_merge = \
            calculate_fixationbout_dynamic(self.x, self.y, self.d, self.t, self.d_d, self.hv_x, self.hv_y, self.step_size,
                                       self.multi_stimid, self.mhv_half_length, self.cri_mhv_height, self.cri_mhv_duration_mm,
                                       self.cri_mhv_avespeed, self.cri_similar_orientation, self.stim_window_info)

        _, _, _, _, self.DPfb_xbar_raw, self.DPfb_dist_raw, self.DPfb_dshift_raw, self.DPfb_ave_speed_raw, self.DPfb_idxs_raw, \
        self.DPfb_xbar, self.DPfb_dist, self.DPfb_dshift, self.DPfb_ave_speed, self.DPfb_idxs, \
        self.DPfb_xbar_merge, self.DPfb_dist_merge, self.DPfb_dshift_merge, self.DPfb_idxs_merge = \
            calculate_fixationbout_geometric(self.x, self.y, self.d, self.t, self.d_d, self.multi_stimid, self.cri_DP_tol,
                                             self.cri_mhv_duration_mm, self.cri_mhv_avespeed,
                                             self.cri_similar_orientation, self.stim_window_info)

    def smooth_trajectory(self):
        self.x_smooth = fc.savitzky_golay(self.x, self.smooth_window_size, self.smooth_order)
        self.y_smooth = fc.savitzky_golay(self.y, self.smooth_window_size, self.smooth_order)
        self.d_smooth = np.zeros(len(self.x))
        self.d_smooth[1:] = np.sqrt(np.square(self.x_smooth[1:] - self.x_smooth[:-1]) + np.square(self.y_smooth[1:] - self.y_smooth[:-1]))
        self.d_smooth = np.cumsum(self.d_smooth)

    def convert_to_distance_coordinate(self):
        xbar_unwrap = fc.unwrap(self.xbar)
        d = self.d
        idx0 = 0
        xbar_d = []
        hv_x = []
        hv_y = []
        for i in np.arange(self.step_size, d[-1] - .01, self.step_size):
            idx = np.where(d <= i)[0][-1]
            if idx == idx0:
                continue
            xbar_d.append(np.radians(np.mean(xbar_unwrap[idx0:idx])))
            hv_x.append(np.cos(xbar_d[-1] + np.pi / 2))
            hv_y.append(np.sin(xbar_d[-1] + np.pi / 2))
            idx0 = idx

        self.xbar_d = np.array(xbar_d)
        self.hv_x = np.array(hv_x)
        self.hv_y = np.array(hv_y)
        self.D = len(xbar_d)
        self.d_d = np.arange(self.D) + self.step_size / 2.
        self.r = np.hypot(self.x, self.y)


class Flight(ABF):
    angle_offset = -103.5
    arena_edge = 108
    def __init__(self, abffile, tracktype='strokelitude', tach_detection=False, nstims=None,
                 detect_stop_with_tach=True, fix_wb_tracking_error=True, fix_wb_tracking_error_ignore_rec_end=False,
                 tach_use_pickles=False,
                 pickle_parental_folder='./', **kwargs):
        ABF.__init__(self, abffile, **kwargs)
        self.bhtype = 'flight'
        self.tracktype = tracktype
        self.detect_stop_with_tach = detect_stop_with_tach      # use wbf extracted from tach to detect stopping event
        self.fix_wb_tracking_error = fix_wb_tracking_error      # fix wb tracking errors (mostly dropping to zero, due to bad wing lighting), by connecting the flying-but-bad-tracking wb
        self.fix_wb_tracking_error_ignore_rec_end = fix_wb_tracking_error_ignore_rec_end
        self.tach_use_pickles = tach_use_pickles                # import tach and wbf using pickles, if doesn't exist, export pickles
        self.pickle_parental_folder = pickle_parental_folder
        self.open(nstims=nstims, tach_detection=tach_detection, **kwargs)

    def open(self, nstims=None, tach_detection=False, **kwargs):
        abf = ABF.open(self, delay_ms=12, **kwargs)
        wa_factor = 0.038095
        wa_offset = -15
        lwalabel = fc.contains(['lwa', 'wba_l'], abf.keys())
        if lwalabel:
            self.lwa = np.array(abf[lwalabel])[self.subsampling_inds]
            self.lwa = self.lwa/wa_factor + wa_offset

        rwalabel = fc.contains(['rwa', 'wba_r'], abf.keys())
        if rwalabel:
            self.rwa = np.array(abf[rwalabel])[self.subsampling_inds]
            self.rwa = self.rwa/wa_factor + wa_offset

        if lwalabel and rwalabel:
            self.lmr = self.lwa - self.rwa
            self.lpr = self.lwa + self.rwa
            self.dhead = self.lwa - self.rwa

        if self.fix_wb_tracking_error and fc.contains(['puff'], abf.keys()) and lwalabel:
            self.fix_wingbeat_tracking_error()

        tachlabel = fc.contains(['tacmeter'], abf.keys())
        if tachlabel and tach_detection:
            if self.tach_use_pickles:
                pickle_folder = self.pickle_parental_folder + 'pickles/'
                pickle_name = self.recid + '.pickle'
                tachpicklefile = (glob.glob(pickle_folder + pickle_name))
                if tachpicklefile:
                    self.tach, self.wbf = pickle.load(open(pickle_folder + pickle_name, 'rb'))
                else:
                    self.tach, self.wbf = self.calculate_wbf_using_tach(abf, tachlabel)
                    pickle.dump((self.tach, self.wbf), open(pickle_folder + pickle_name, 'wb'))
            else:
                self.tach, self.wbf = self.calculate_wbf_using_tach(abf, tachlabel)

    def calculate_wbf_using_tach(self, abf, tachlabel,
                                 n=40, ave_subsampling_num = 5, freq_cut=400, freq_noise=5, freq_thresh=200):
        # 2019 Jan, by Cheng
        sub_inds = self.subsampling_inds

        wbf = np.zeros(len(self.t) - ave_subsampling_num * 2)
        axlen = (sub_inds[ave_subsampling_num * 2] - sub_inds[0]) * n
        wbf_low_thresh = freq_noise * self.sampling_period
        wbf_high_thresh = freq_cut * self.sampling_period
        freq = np.fft.fftfreq(axlen, 1. / n) / n
        inds = (freq >= wbf_low_thresh) & (freq < wbf_high_thresh)
        freq = freq[inds]
        tach = np.array(abf[tachlabel])

        for i in range(len(wbf)):
            data = tach[sub_inds[i]:sub_inds[i + ave_subsampling_num * 2]]
            fft1 = np.fft.fft(data, axlen, 0)
            power = np.abs(fft1) ** 2
            power = power[inds]
            power = power / power.max()

            wbf_ = freq[np.argmax(power)] / self.sampling_period
            idx_half = np.argmax(power) / 2
            wbf[i] = wbf_ if (wbf_ < freq_thresh) or (np.max(power[idx_half - 100:idx_half + 300]) < 0.1)\
                else freq[idx_half] / self.sampling_period

        wbf[wbf > freq_cut] = wbf[wbf > freq_cut] / 2.
        selfwbf = np.zeros(len(self.t))
        selfwbf[ave_subsampling_num:-ave_subsampling_num] = wbf
        for i in range(ave_subsampling_num):
            selfwbf[i] = wbf[0]
            j = i + 1
            selfwbf[-j] = wbf[-1]

        return tach, selfwbf

    def get_flight_stopping_inds(self, min_len_s=0.5, wing_amp_thresh=-10, tach_thresh=50):
        # output the inds of abf when fly is not flying
        # tricks used to avoid tracking error:
        # 1. stopping too short (less than 0.5 here)
        # 2. no puffing
        # 3. at the end of a recording
        # 4. using tach info, if tach data exist

        lwa = self.lwa
        rwa = self.rwa
        file_length = len(self.lwa)
        left_thresh = (lwa < wing_amp_thresh)
        right_thresh = (rwa < wing_amp_thresh)
        min_len_frames = min_len_s * self.subsampling_rate
        left_inds_initial = fc.get_contiguous_inds(left_thresh, trim_left=0, keep_ends=True,
                                                   min_contig_len=min_len_frames)
        right_inds_initial = fc.get_contiguous_inds(right_thresh, trim_left=0, keep_ends=True,
                                                    min_contig_len=min_len_frames)
        puff_inds = fc.get_contiguous_inds(self.puff > 1, trim_left=0, keep_ends=True, min_contig_len=0)
        for i in range(len(puff_inds)):
            puff_inds[i] = puff_inds[i][0]
        left_inds = []
        right_inds = []
        puff_flag = 0

        for inds in left_inds_initial:
            for puff_ind in puff_inds:
                if (puff_ind >= inds[0]) & (puff_ind <= inds[-1]):
                    puff_flag = 1
                    break
            if puff_flag:
                puff_flag = 0
                left_inds.append(inds)
            elif ((inds[-1] + 1) == file_length) and not self.fix_wb_tracking_error_ignore_rec_end:
                left_inds.append(inds)
        for inds in right_inds_initial:
            for puff_ind in puff_inds:
                if (puff_ind >= inds[0]) & (puff_ind <= inds[-1]):
                    puff_flag = 1
                    break
            if puff_flag:
                puff_flag = 0
                right_inds.append(inds)
            elif (inds[-1] + 1) == file_length and not self.fix_wb_tracking_error_ignore_rec_end:
                right_inds.append(inds)

        temp = copy.copy(left_inds)
        temp.extend(right_inds)
        temp = np.array(temp)
        if len(temp):
            stop_inds = np.unique(np.hstack(np.hstack(np.array(temp))))
        else:
            stop_inds = []

        # 4. using tach info, if tach data exist
        if self.detect_stop_with_tach and hasattr(self, 'wbf'):
            wbf = getattr(self, 'wbf')
            inds_wbf = np.where(wbf <= tach_thresh)[0]
            stop_inds = np.unique(np.hstack((stop_inds, inds_wbf)))

        return stop_inds

    def fix_wingbeat_tracking_error(self, wing_amp_thresh=-10):
        stop_inds = self.get_flight_stopping_inds()
        lwa = self.lwa
        rwa = self.rwa
        self.lwa_orig = copy.copy(self.lwa)
        self.rwa_orig = copy.copy(self.rwa)
        left_thresh_inds = np.where(lwa < wing_amp_thresh)[0]
        right_thresh_inds = np.where(rwa < wing_amp_thresh)[0]
        for ind in left_thresh_inds:
            if ind not in stop_inds:
                self.lwa[ind] = self.iteration_to_first_normal_wingbeat_detection(ind-1, l_r='left', ite_count=0)
        for ind in right_thresh_inds:
            if ind not in stop_inds:
                self.rwa[ind] = self.iteration_to_first_normal_wingbeat_detection(ind-1, l_r='right', ite_count=0)

    def iteration_to_first_normal_wingbeat_detection(self, ind, wing_amp_iteration_thresh=30, l_r='left', ite_count=0):
        if l_r == 'left':
            if ite_count > 10:
                return self.lwa[ind-1]
            if self.lwa[ind-1] < wing_amp_iteration_thresh:
                self.lwa[ind - 1] = self.iteration_to_first_normal_wingbeat_detection(ind-1, l_r='left', ite_count=ite_count+1)
                return self.lwa[ind-1]
            else:
                return self.lwa[ind-1]
        if l_r == 'right':
            if ite_count > 10:
                return self.rwa[ind-1]
            if self.rwa[ind-1] < wing_amp_iteration_thresh:
                self.rwa[ind - 1] = self.iteration_to_first_normal_wingbeat_detection(ind-1, l_r='right', ite_count=ite_count+1)
                return self.rwa[ind - 1]
            else:
                return self.rwa[ind-1]

    def get_trajectory(self, proj_forw=False):
        x = np.zeros(len(self.xbar))
        y = np.zeros(len(self.xbar))
        d = np.zeros(len(self.xbar))

        head_rads = np.radians(self.xbar) + np.pi / 2

        dforw_mms = self.deg2mm(fc.circgrad(self.forw, np.diff))

        if proj_forw:
            dside_mms = np.zeros(len(self.dside))
        else:
            dside_mms = self.deg2mm(fc.circgrad(self.side, np.diff))

        for i in xrange(1, len(x)):
            headi = head_rads[i - 1:i + 1].mean()  # avg heading between i-1 and i
            dside = dside_mms[i - 1]  # diff between i-1 and i
            dforw = dforw_mms[i - 1]  # diff between i-1 and i

            xi, yi = self.get_xiyi(x[i - 1], y[i - 1], headi, dside, dforw)
            dx, dy = self.get_dxdy(headi, dforw, dside)
            x[i] = xi
            y[i] = yi
            d[i] = d[i - 1] + np.sqrt(dx * dx + dy * dy)
        return x, y, d

# backup Apr22, 2019
class FlightSun_backup(Flight):

    def __init__(self, abffile, stimid_split_mode='stitch', mhv_half_length_sec=5, sunjumps=False, **kwargs):
        Flight.__init__(self, abffile, **kwargs)
        self.stimid_split_mode = stimid_split_mode  # different ways to separate stimid trials
        self.mhv_half_length_sec = mhv_half_length_sec
        self.sunjumps = sunjumps
        self.open_flightsun()

    def open_flightsun(self):
        # params for trajectory
        # transform from time coordinate to distance coordinate
        self.step_size = 1. / self.subsampling_rate * 5.

        # savitzky golay filter, size of the smoothing window
        self.smooth_window_size = int(0.2*self.subsampling_rate)/2*2+1

        # savitzky golay filter, order of the polynomials of fitting curves
        self.smooth_order = 5

        # params for fixation bouts
        self.mhv_half_length = int(1. * self.mhv_half_length_sec / self.step_size)  # unit: transfer seconds to number in the distance coordinate
        self.cri_mhv_height = 0.8
        self.cri_mhv_duration_mm = 5
        self.cri_similar_orientation = 10

        # params for sun jumps
        self.sunjump_threshold_deg = 15
        self.sunjump_tooclose_threshold_sec = 5

        # calculate virtual 2D trajectory
        self.x, self.y, self.d = self.get_trajectory()

        # smooth trajectory
        self.smooth_trajectory()

        # get the info about different trial stages, defined by stims: self.stim_window_info = [[stimid, ind0, ind1], ...]
        self.get_stims_window()

        # convert some important variables from time coordinates to distance coordinates
        self.convert_to_distance_coordinate()

        # calculate the mean heading vector in distance coordinate
        self.mhv_x, self.mhv_y, self.mhv_xbar, self.mhv_amp \
            = self.mean_heading_vector(self.mhv_half_length)

        # separate individual fixation bout
        self.fb_xbar_raw, self.fb_dist_travelled_raw, self.fb_dist_shifted_raw, self.fb_idxs_raw, self.fb_idxs_d_raw, \
        self.fb_xbar, self.fb_dist_travelled, self.fb_dist_shifted, self.fb_idxs, self.fb_idxs_d \
            = self.get_individual_fixation_bout(self.cri_mhv_height, self.cri_mhv_duration_mm, self.mhv_amp, self.mhv_xbar)

        # merge fixation bouts
        self.fb_xbar_merge, self.fb_dist_travelled_merge, self.fb_dist_shifted_merge, self.fb_idxs_merge, \
        self.fb_idxs_d_merge \
            = self.merge_fixation_bouts(self.cri_similar_orientation, self.fb_xbar, self.fb_dist_travelled, self.fb_dist_shifted, self.fb_idxs, self.fb_idxs_d)

        # detect sun jumping indexes
        if self.sunjumps:
            self.detect_sunjumps()

    def get_dxdy(self, head_rad):
        r = self.subsampling_period
        dx = r*math.cos(head_rad)
        dy = r*math.sin(head_rad)
        return dx, dy

    def get_xiyi(self, x0, y0, head_rad):
        dx, dy = self.get_dxdy(head_rad)
        xi = x0 + dx
        yi = y0 + dy
        return xi, yi

    def get_trajectory(self):
        x = np.zeros(len(self.xbar))
        y = np.zeros(len(self.xbar))
        d = np.zeros(len(self.xbar))
        head_rads = np.radians(self.xbar) + np.pi / 2
        stop_inds = self.get_flight_stopping_inds()

        for i in xrange(1, len(x)):
            if i in stop_inds:
                x[i] = x[i - 1]
                y[i] = y[i - 1]
                d[i] = d[i - 1]
            else:
                headi = head_rads[i - 1:i + 1].mean()  # avg heading between i-1 and i
                xi, yi = self.get_xiyi(x[i - 1], y[i - 1], headi)
                dx, dy = self.get_dxdy(headi)
                x[i] = xi
                y[i] = yi
                d[i] = d[i - 1] + np.sqrt(dx * dx + dy * dy)
        return x, y, d

    def smooth_trajectory(self):
        self.x_smooth = fc.savitzky_golay(self.x, self.smooth_window_size, self.smooth_order)
        self.y_smooth = fc.savitzky_golay(self.y, self.smooth_window_size, self.smooth_order)
        self.d_smooth = np.zeros(len(self.x))
        self.d_smooth[1:] = np.sqrt(np.square(self.x_smooth[1:] - self.x_smooth[:-1]) + np.square(self.y_smooth[1:] - self.y_smooth[:-1]))
        self.d_smooth = np.cumsum(self.d_smooth)

    def get_stims_window(self):
        self.stim_window_info = []
        tss = self.get_trial_ts(mode=self.stimid_split_mode)
        for i_ts, ts in enumerate(tss):
            if ts[-1] - ts[0] <= 2:
                continue
            stimid = self.get_stimid_in_window_sf(ts)
            ind0 = np.where(self.t >= ts[0])[0][0]
            ind1 = np.where(self.t <= ts[-1])[0][-1]
            self.stim_window_info.append([stimid, ind0, ind1])

    def convert_to_distance_coordinate(self):
        xbar_unwrap = fc.unwrap(self.xbar)
        d = self.d
        idx0 = 0
        xbar_d = []
        hv_x = []
        hv_y = []
        for i in np.arange(self.step_size/2., d[-1] - 1e-10, self.step_size):
            idx = np.where(d <= i)[0][-1]
            if idx == idx0:
                continue
            xbar_d.append(np.radians(np.mean(xbar_unwrap[idx0:idx])))
            hv_x.append(np.cos(xbar_d[-1] + np.pi / 2))
            hv_y.append(np.sin(xbar_d[-1] + np.pi / 2))
            idx0 = idx

        self.xbar_d = np.array(xbar_d)
        self.hv_x = np.array(hv_x)
        self.hv_y = np.array(hv_y)
        self.N = len(xbar_d)
        self.d_d = np.arange(self.step_size/2., d[-1] - 1e-10, self.step_size)

    def mean_heading_vector(self, mhv_half_length):
        mhv_x = np.zeros(self.N)
        mhv_y = np.zeros(self.N)
        mhv_xbar = np.zeros(self.N)
        mhv_amp = np.zeros(self.N)
        for i in np.arange(mhv_half_length, self.N - mhv_half_length):
            mhv_x[i] = np.mean(self.hv_x[i - mhv_half_length:i + mhv_half_length])
            mhv_y[i] = np.mean(self.hv_y[i - mhv_half_length:i + mhv_half_length])
            mhv_xbar[i] = self.slope_to_headning_angle(mhv_x[i], mhv_y[i])
            mhv_amp[i] = np.hypot(mhv_x[i], mhv_y[i])

        return np.array(mhv_x), np.array(mhv_y), np.array(mhv_xbar), np.array(mhv_amp)

    def get_individual_fixation_bout(self, cri_mhv_height, cri_mhv_duration_mm, mhv_amp, mhv_xbar):
        # initializing variables
        fb_xbar_raw, fb_dist_travelled_raw, fb_dist_shifted_raw, fb_idxs_raw, fb_idxs_d_raw, fb_xbar, \
        fb_dist_travelled, fb_dist_shifted, fb_idxs, fb_idxs_d = [], [], [], [], [], [], [], [], [], []
        for i in range(len(self.stim_window_info)):
            fb_xbar_raw.append([]), fb_dist_travelled_raw.append([]), fb_dist_shifted_raw.append([]), \
            fb_idxs_raw.append([]), fb_idxs_d_raw.append([]), fb_xbar.append([]), fb_dist_travelled.append([]), \
            fb_dist_shifted.append([]), fb_idxs.append([]), fb_idxs_d.append([])

        # loop of different stimid
        for i_stim, (stimid, ind0, ind1) in enumerate(self.stim_window_info):
            # separate fixation bouts
            d0, d1 = self.d[ind0], self.d[ind1]
            inds = (mhv_amp > cri_mhv_height) & (self.d_d >= d0) & (self.d_d <= d1)
            idxss = fc.get_contiguous_inds(inds, min_contig_len=0, trim_left=0, trim_right=0, keep_ends=True)

            # calculate fixation angle, in unit of deg
            _fb_xbar_raw, _fb_dist_travelled_raw, _fb_dist_shifted_raw, _fb_idxs_raw, _fb_idxs_d_raw = [], [], [], [], []
            for i, idxs in enumerate(idxss):
                _fb_xbar_raw.append(fc.circmean(mhv_xbar[idxs]))
                _fb_dist_travelled_raw.append(len(idxs) * self.step_size)

                idx0 = np.where(self.d >= self.d_d[idxs[0]])[0][0]
                idx1 = np.where(self.d <= self.d_d[idxs[-1]])[0][-1]
                x0, x1, y0, y1 = self.x[idx0], self.x[idx1], self.y[idx0], self.y[idx1]
                _fb_dist_shifted_raw.append(np.hypot(x1-x0, y1-y0))
                _fb_idxs_raw.append(
                    [np.where(self.d <= self.d_d[idxs[0]])[0][-1], np.where(self.d >= self.d_d[idxs[-1]])[0][0]])
                _fb_idxs_d_raw.append([idxs[0], idxs[-1]])

            fb_xbar_raw[i_stim] = np.array(_fb_xbar_raw)
            fb_dist_travelled_raw[i_stim] = np.array(_fb_dist_travelled_raw)
            fb_dist_shifted_raw[i_stim] = np.array(_fb_dist_shifted_raw)
            fb_idxs_raw[i_stim] = _fb_idxs_raw
            fb_idxs_d_raw[i_stim] = _fb_idxs_d_raw

            # applying duration criterion
            _fb_xbar, _fb_dist_travelled, _fb_dist_shifted, _fb_idxs, _fb_idxs_d = [], [], [], [], []
            for i_fb in range(len(fb_xbar_raw[i_stim])):
                if fb_dist_shifted_raw[i_stim][i_fb] >= cri_mhv_duration_mm:
                    _fb_xbar.append(_fb_xbar_raw[i_fb])
                    _fb_dist_travelled.append(_fb_dist_travelled_raw[i_fb])
                    _fb_dist_shifted.append(_fb_dist_shifted_raw[i_fb])
                    _fb_idxs.append(_fb_idxs_raw[i_fb])
                    _fb_idxs_d.append(_fb_idxs_d_raw[i_fb])
            fb_xbar[i_stim] = np.array(_fb_xbar)
            fb_dist_travelled[i_stim] = np.array(_fb_dist_travelled)
            fb_dist_shifted[i_stim] = np.array(_fb_dist_shifted)
            fb_idxs[i_stim] = _fb_idxs
            fb_idxs_d[i_stim] = _fb_idxs_d

        return fb_xbar_raw, fb_dist_travelled_raw, fb_dist_shifted_raw, fb_idxs_raw, fb_idxs_d_raw, fb_xbar, \
               fb_dist_travelled, fb_dist_shifted, fb_idxs, fb_idxs_d

    def merge_fixation_bouts(self, cri_similar_orientation, fb_xbar, fb_dist_travelled, fb_dist_shifted, fb_idxs, fb_idxs_d):

        def new_queue():
            _xbar = copy.copy(fb_xbar[i_stim][i_fb])
            _dist_travelled = copy.copy(fb_dist_travelled[i_stim][i_fb])
            _dist_shifted = copy.copy(fb_dist_shifted[i_stim][i_fb])
            _idxs = copy.copy(fb_idxs[i_stim][i_fb])
            _idxs_d = copy.copy(fb_idxs_d[i_stim][i_fb])
            return _xbar, _dist_travelled, _dist_shifted, _idxs, _idxs_d

        def fill_queue(_xbar, _dist_travelled, _dist_shifted, _idxs, _idxs_d):
            _angles = fc.unwrap(np.array([_xbar, fb_xbar[i_stim][i_fb]]))
            _angle = (_angles[0] * _dist_shifted + _angles[-1] * fb_dist_shifted[i_stim][i_fb]) \
                    / (_dist_shifted + fb_dist_shifted[i_stim][i_fb])
            _xbar = fc.wrap(np.array([_angle]))[0]
            _dist_travelled += fb_dist_travelled[i_stim][i_fb]
            _dist_shifted += fb_dist_shifted[i_stim][i_fb]
            _idxs = [_idxs[0], fb_idxs[i_stim][i_fb][-1]]
            _idxs_d = [_idxs_d[0], fb_idxs_d[i_stim][i_fb][-1]]
            return _xbar, _dist_travelled, _dist_shifted, _idxs, _idxs_d

        def clear_queue(_xbar, _dist_travelled, _dist_shifted, _idxs, _idxs_d):
            _fb_xbar_merge.append(_xbar)
            _fb_dist_travelled_merge.append(_dist_travelled)
            _fb_dist_shifted_merge.append(_dist_shifted)
            _fb_idxs_merge.append(_idxs)
            _fb_idxs_d_merge.append(_idxs_d)

        fb_xbar_merge, fb_dist_travelled_merge, fb_dist_shifted_merge, fb_idxs_merge, fb_idxs_d_merge = [], [], [], [], []
        for i in range(len(self.stim_window_info)):
            fb_xbar_merge.append([]), fb_dist_travelled_merge.append([]), fb_dist_shifted_merge.append([]), \
            fb_idxs_merge.append([]), fb_idxs_d_merge.append([])

        # loop of different stimid
        for i_stim, _ in enumerate(self.stim_window_info):
            _fb_xbar_merge, _fb_dist_travelled_merge, _fb_dist_shifted_merge, _fb_idxs_merge, _fb_idxs_d_merge = [], [], [], [], []

            # deal with recording that doesn't have or only has one fixation bout
            if len(fb_xbar[i_stim]) == 0:
                fb_xbar_merge[i_stim] = np.array([])
                fb_dist_travelled_merge[i_stim] = np.array([])
                fb_dist_shifted_merge[i_stim] = np.array([])
                fb_idxs_merge[i_stim] = []
                fb_idxs_d_merge[i_stim] = []
            elif len(fb_xbar[i_stim]) == 1:
                fb_xbar_merge[i_stim] = [fb_xbar[i_stim][0]]
                fb_dist_travelled_merge[i_stim] = [fb_dist_travelled[i_stim][0]]
                fb_dist_shifted_merge[i_stim] = [fb_dist_shifted[i_stim][0]]
                fb_idxs_merge[i_stim] = [fb_idxs[i_stim][0]]
                fb_idxs_d_merge[i_stim] = [fb_idxs_d[i_stim][0]]
            else:
                _xbar = fb_xbar[i_stim][0]
                _dist_travelled = fb_dist_travelled[i_stim][0]
                _dist_shifted = fb_dist_shifted[i_stim][0]
                _idxs = fb_idxs[i_stim][0]
                _idxs_d = fb_idxs_d[i_stim][0]

                for i_fb in np.arange(1, len(fb_xbar[i_stim])):
                    if self.circdist_incomplete_arena(_xbar, fb_xbar[i_stim][i_fb]) <= cri_similar_orientation:
                        _xbar, _dist_travelled, _dist_shifted, _idxs, _idxs_d = \
                            fill_queue(_xbar, _dist_travelled, _dist_shifted, _idxs, _idxs_d)  # merge
                    else:
                        clear_queue(_xbar, _dist_travelled, _dist_shifted, _idxs, _idxs_d)  # clear queue
                        _xbar, _dist_travelled, _dist_shifted, _idxs, _idxs_d = new_queue()  # new queue
                    if i_fb == (len(fb_xbar[i_stim]) - 1):  # last one, clear queue
                        clear_queue(_xbar, _dist_travelled, _dist_shifted, _idxs, _idxs_d)

                fb_xbar_merge[i_stim] = np.array(_fb_xbar_merge)
                fb_dist_travelled_merge[i_stim] = np.array(_fb_dist_travelled_merge)
                fb_dist_shifted_merge[i_stim] = np.array(_fb_dist_shifted_merge)
                fb_idxs_merge[i_stim] = _fb_idxs_merge
                fb_idxs_d_merge[i_stim] = _fb_idxs_d_merge

        return fb_xbar_merge, fb_dist_travelled_merge, fb_dist_shifted_merge, fb_idxs_merge, fb_idxs_d_merge

    def slope_to_headning_angle(self, dx, dy):
        deg = np.degrees(np.arctan(dy / dx))
        if dx < 0:
            deg += 90
        else:
            deg -= 90
        return deg

    def circdist_incomplete_arena(self, a, b, low=-165, high=165):
        # push bar positions that are in the dark to the edge of the seeable boundaries, described by low & high
        if a < low:
            a = low
        elif a > high:
            a = high
        if b < low:
            b = low
        elif b > high:
            b = high

        signal = np.array([a, b])
        period = high - low
        unwrapped = fc.unwrap(signal, period=period)
        return np.abs(unwrapped[-1] - unwrapped[0])

    def get_trial_ts(self, mode='normal', erosion_step=5, stitch_time=0.1):
        trials = np.gradient(self.stimid) == 0
        trials = ndimage.binary_erosion(trials, iterations=erosion_step)
        trials = ndimage.binary_dilation(trials, iterations=erosion_step).astype(int)

        trial_starts = self.t[np.where(np.diff(trials) > 0)[0] + 1]
        trial_ends = self.t[np.where(np.diff(trials) < 0)[0] + 1]

        if len(trial_starts):
            if trial_starts[0] > trial_ends[0]:
                # trial_starts = trial_starts
                trial_ends = np.append(trial_ends[1:], self.t[-1])

            if mode == 'stitch':    # added by Cheng, Feb.9 2019. To deal with the blink noise in stimid recording ch.
                t = self.t
                trial_starts_new = []
                trial_ends_new = []
                i_start = 0
                i_end = 0
                for i_trial in range(len(trial_starts)-1):
                    stimid0 = self.get_stimid_in_window_sf([trial_starts[i_trial], trial_ends[i_trial]])
                    stimid1 = self.get_stimid_in_window_sf([trial_starts[i_trial+1], trial_ends[i_trial+1]])
                    if (stimid0 == stimid1) & (trial_starts[i_trial+1] - trial_ends[i_trial] <= stitch_time):    # stitch
                        i_end = i_trial + 1
                    else:   # output into trial_starts_new & trial_ends_new
                        trial_starts_new.append(trial_starts[i_start])
                        trial_ends_new.append(trial_ends[i_end])
                        i_start = i_trial + 1
                        i_end = i_end + 1
                trial_starts_new.append(trial_starts[i_start])
                trial_ends_new.append(trial_ends[i_end])
                return zip(np.array(trial_starts_new), np.array(trial_ends_new))
            else:
                return zip(trial_starts, trial_ends)
        else:
            return zip([self.t[0]], [self.t[-1]])

    def get_stimid_in_window_sf(self, ts):
        # window: ts: (t0, t1)
        t_ind = (self.t > ts[0]) & (self.t < ts[1])
        if len(np.unique(np.round(self.stimid[t_ind]))) != 1:
            # print 'Error: stimid in period: %3.1f-%3.1f not unique!!' % (ts[0], ts[1])
            return np.round(np.mean(self.stimid[t_ind]))
        return np.unique(np.round(self.stimid[t_ind]))[0]

    def recalculate_fb(self, mhv_half_length_sec, cri_mhv_height, cri_mhv_duration_mm, cri_similar_orientation):
        self.mhv_half_length_sec = mhv_half_length_sec
        self.mhv_half_length = int(1. * self.mhv_half_length_sec / self.step_size)
        self.cri_mhv_height = cri_mhv_height
        self.cri_mhv_duration_mm = cri_mhv_duration_mm
        self.cri_similar_orientation = cri_similar_orientation

        # calculate the mean heading vector in distance coordinate
        self.mhv_x, self.mhv_y, self.mhv_xbar, self.mhv_amp \
            = self.mean_heading_vector(self.mhv_half_length)

        # separate individual fixation bout
        self.fb_xbar_raw, self.fb_dist_travelled_raw, self.fb_dist_shifted_raw, self.fb_idxs_raw, self.fb_idxs_d_raw, \
        self.fb_xbar, self.fb_dist_travelled, self.fb_dist_shifted, self.fb_idxs, self.fb_idxs_d \
            = self.get_individual_fixation_bout(self.cri_mhv_height, self.cri_mhv_duration_mm, self.mhv_amp,
                                                self.mhv_xbar)

        # mergge fixation bouts
        self.fb_xbar_merge, self.fb_dist_travelled_merge, self.fb_dist_shifted_merge, self.fb_idxs_merge, \
        self.fb_idxs_d_merge \
            = self.merge_fixation_bouts(self.cri_similar_orientation, self.fb_xbar, self.fb_dist_travelled,
                                        self.fb_dist_shifted, self.fb_idxs, self.fb_idxs_d)

    def detect_sunjumps(self):
        xbar_orig = volts2degs(self.xstim_orig, 10) + self.angle_offset
        xbar_grad_abs = np.abs(np.gradient(fc.unwrap(xbar_orig)))

        _inds = xbar_grad_abs > self.sunjump_threshold_deg
        _idxss = fc.get_contiguous_inds(_inds, keep_ends=True, )
        idxs_continous = []
        for _idxs in _idxss:
            idxs_continous.append(np.mean(_idxs))
        idxs_continous = np.array(idxs_continous)
        _idxs_todelete = \
            np.where(np.abs(idxs_continous[:-1] - idxs_continous[1:]) <=
                     (self.sampling_rate * self.sunjump_tooclose_threshold_sec))[0]
        idxs_todelete = np.concatenate((_idxs_todelete, (_idxs_todelete + 1)))
        idxs_continous[idxs_todelete] = np.nan
        _idxs_sunjumps = idxs_continous[~np.isnan(idxs_continous)].astype(int)
        self.sunjumps_idxs = (1. * _idxs_sunjumps / self.subsampling_step).astype(int)

    def calculate_fb_adjacent_to_sunjumps(self, stimid_special, cri_nearby_pre_sec=3, cri_nearby_post_sec=15,
                                          mhv_half_length_sec=3, cri_mhv_height=.8, cri_mhv_duration_mm=10, cri_similar_orientation=10):

        # recalculate fixation bouts, with different set of parameters, compensating for the sudden sun jumping
        mhv_half_length = int(1. * mhv_half_length_sec / self.step_size)
        _, _, mhv_xbar, mhv_amp = self.mean_heading_vector(mhv_half_length)

        for sj_idx in self.sunjumps_idxs:
            _idx_d = np.where(self.d_d <= self.d[sj_idx])[0][-1]
            mhv_amp[_idx_d] = 0

        _, _, _, _, _, _fb_xbar, _fb_dist_travelled, _fb_dist_shifted, _fb_idxs, _fb_idxs_d \
            = self.get_individual_fixation_bout(cri_mhv_height, cri_mhv_duration_mm, mhv_amp, mhv_xbar)

        # _fb_xbar_merge, _fb_dist_travelled_merge, _fb_dist_shifted_merge, _fb_idxs_merge, _fb_idxs_d_merge \
        #     = self.merge_fixation_bouts(cri_similar_orientation, _fb_xbar, _fb_dist_travelled, _fb_dist_shifted, _fb_idxs, _fb_idxs_d)

        # find the indexes for stimid special
        flag_find_stimid = 0
        for i_stim, stimid_info in enumerate(self.stim_window_info):
            stimid, idx0_id, idx1_id = stimid_info
            if stimid == stimid_special:
                # fb_idxs = fb_idxs_merge[i_stim]
                # fb_xbar = fb_xbar_merge[i_stim]
                fb_idxs = _fb_idxs[i_stim]
                fb_xbar = _fb_xbar[i_stim]
                flag_find_stimid = 1
                break
        if flag_find_stimid == 0:
            print 'Did not find the special stimid %1.1f for function "calculate_fb_adjacent_to_sunjumps in fly: %s"' \
                  % (stimid_special, self.recid)
            return False

        # apply criteria on the fixation bouts, to find the right adjacent fixation bouts
        fb_adj_sj = []                      # structure: [sj_idx, [xbar_pre, idx_pre_start, idx_pre_end], [xbar_post, idx_post_start, idx_post_end]]
        fb_idxs = np.asarray(fb_idxs)       # structure: fb_idxs[starting & ending idx, # of fb]

        stop_idxs = self.get_flight_stopping_inds()
        for sj_idx in self.sunjumps_idxs:
            if sj_idx >= idx0_id and sj_idx <= idx1_id:
                # find the adj fb bout before
                if len(np.where(fb_idxs[:,1] < sj_idx)[0]):
                    fborder_pre = np.where(fb_idxs[:,1] < sj_idx)[0][-1]
                    xbar_pre = fb_xbar[fborder_pre]
                    idx_pre_start = fb_idxs[fborder_pre, 0]
                    idx_pre_end = fb_idxs[fborder_pre, 1]
                    if (sj_idx - idx_pre_end) >= (cri_nearby_pre_sec * self.subsampling_rate):   # the fixation bout before sj is too far away
                            continue
                else:
                    continue

                # find the adj fb bout after
                if len(np.where(fb_idxs[:,0] > sj_idx)[0]):
                    fborder_post = np.where(fb_idxs[:,0] > sj_idx)[0][0]
                    xbar_post = fb_xbar[fborder_post]
                    idx_post_start = fb_idxs[fborder_post, 0]
                    idx_post_end = fb_idxs[fborder_post, 1]
                    if (idx_post_start - sj_idx) >= (cri_nearby_post_sec * self.subsampling_rate):   # the fixation bout before sj is too far away
                        continue
                else:
                    continue

                # apply stopping criteria on the fixation bouts
                if not np.sum((stop_idxs > idx_pre_end) & (stop_idxs < idx_post_start)):
                    fb_adj_sj.append([sj_idx, [xbar_pre, idx_pre_start, idx_pre_end], [xbar_post, idx_post_start, idx_post_end]])

        return fb_adj_sj

    def print_fb_params(self):
        print 'mhv_half_length_sec = %2.2f\ncri_mhv_height = %2.2f\ncri_mhv_duration_mm = %2.2f\n' \
              'cri_similar_orientation = %2.2f' % \
              (self.mhv_half_length_sec, self.cri_mhv_height, self.cri_mhv_duration_mm, self.cri_similar_orientation)

# created Apr22, 2019, put calculate fixation bout outside the class
class FlightSun(Flight):

    def __init__(self, abffile, stimid_split_mode='stitch', mhv_half_length_sec=5, sunjumps=False, **kwargs):
        Flight.__init__(self, abffile, **kwargs)
        self.stimid_split_mode = stimid_split_mode  # different ways to separate stimid trials
        self.mhv_half_length_sec = mhv_half_length_sec
        self.sunjumps = sunjumps
        self.open_flightsun()

    def open_flightsun(self):
        # params for trajectory
        self.step_size = 1. / self.subsampling_rate * 5.
        self.smooth_window_size = int(0.2*self.subsampling_rate)/2*2+1  # savitzky golay filter, size of the smoothing window
        self.smooth_order = 5                       # savitzky golay filter, order of the polynomials of fitting curves

        # params for fixation bouts
        self.mhv_half_length = int(1. * self.mhv_half_length_sec / self.step_size)  # unit: transfer seconds to number in the distance coordinate
        self.cri_mhv_height = 0.8
        self.cri_mhv_duration_mm = 5
        self.cri_similar_orientation = 10
        self.cri_DP_tol = 5

        # params for sun jumps
        self.sunjump_threshold_deg = 15
        self.sunjump_tooclose_threshold_sec = 5

        # calculate virtual 2D trajectory
        self.x, self.y, self.d = self.get_trajectory()
        self.smooth_trajectory()

        # get the info about different trial stages, defined by stims: self.stim_window_info = [[stimid, ind0, ind1], ...]
        self.stim_window_info = get_stims_window_menotaxis(self.stimid, self.t)

        # convert some important variables from time coordinates to distance coordinates
        self.convert_to_distance_coordinate()

        _, _, _, _, _, self.mhv_x, self.mhv_y, self.mhv_xbar, self.mhv_amp, \
        self.fb_xbar_raw, self.fb_dist_raw, self.fb_dshift_raw, _, self.fb_idxs_raw, self.fb_idxs_d_raw, \
        self.fb_xbar, self.fb_dist, self.fb_dshift, _, self.fb_idxs, self.fb_idxs_d, \
        self.fb_xbar_merge, self.fb_dist_merge, self.fb_dshift_merge, self.fb_idxs_merge, self.fb_idxs_d_merge = \
            calculate_fixationbout_dynamic(self.x, self.y, self.d, self.t, self.d_d, self.hv_x, self.hv_y,
                                           self.step_size, True, self.mhv_half_length, self.cri_mhv_height,
                                           self.cri_mhv_duration_mm, 0, self.cri_similar_orientation, self.stim_window_info)

        _, _, _, _, self.DPfb_xbar_raw, self.DPfb_dist_raw, self.DPfb_dshift_raw, _, self.DPfb_idxs_raw, \
        self.DPfb_xbar, self.DPfb_dist, self.DPfb_dshift, _, self.DPfb_idxs, \
        self.DPfb_xbar_merge, self.DPfb_dist_merge, self.DPfb_dshift_merge, self.DPfb_idxs_merge = \
            calculate_fixationbout_geometric(self.x, self.y, self.d, self.t, self.d_d, True, self.cri_DP_tol,
                                             self.cri_mhv_duration_mm, 0,
                                             self.cri_similar_orientation, self.stim_window_info)

        # detect sun jumping indexes
        if self.sunjumps:
            self.detect_sunjumps()

    def get_dxdy(self, head_rad):
        r = self.subsampling_period
        dx = r*math.cos(head_rad)
        dy = r*math.sin(head_rad)
        return dx, dy

    def get_xiyi(self, x0, y0, head_rad):
        dx, dy = self.get_dxdy(head_rad)
        xi = x0 + dx
        yi = y0 + dy
        return xi, yi

    def get_trajectory(self):
        x = np.zeros(len(self.xbar))
        y = np.zeros(len(self.xbar))
        d = np.zeros(len(self.xbar))
        head_rads = np.radians(self.xbar) + np.pi / 2
        stop_inds = self.get_flight_stopping_inds()

        for i in xrange(1, len(x)):
            if i in stop_inds:
                x[i] = x[i - 1]
                y[i] = y[i - 1]
                d[i] = d[i - 1]
            else:
                headi = head_rads[i - 1:i + 1].mean()  # avg heading between i-1 and i
                xi, yi = self.get_xiyi(x[i - 1], y[i - 1], headi)
                dx, dy = self.get_dxdy(headi)
                x[i] = xi
                y[i] = yi
                d[i] = d[i - 1] + np.sqrt(dx * dx + dy * dy)
        return x, y, d

    def smooth_trajectory(self):
        self.x_smooth = fc.savitzky_golay(self.x, self.smooth_window_size, self.smooth_order)
        self.y_smooth = fc.savitzky_golay(self.y, self.smooth_window_size, self.smooth_order)
        self.d_smooth = np.zeros(len(self.x))
        self.d_smooth[1:] = np.sqrt(np.square(self.x_smooth[1:] - self.x_smooth[:-1]) + np.square(self.y_smooth[1:] - self.y_smooth[:-1]))
        self.d_smooth = np.cumsum(self.d_smooth)

    def convert_to_distance_coordinate(self):
        xbar_unwrap = fc.unwrap(self.xbar)
        d = self.d
        idx0 = 0
        xbar_d = []
        hv_x = []
        hv_y = []
        for i in np.arange(self.step_size/2., d[-1] - 1e-10, self.step_size):
            idx = np.where(d <= i)[0][-1]
            if idx == idx0:
                continue
            xbar_d.append(np.radians(np.mean(xbar_unwrap[idx0:idx])))
            hv_x.append(np.cos(xbar_d[-1] + np.pi / 2))
            hv_y.append(np.sin(xbar_d[-1] + np.pi / 2))
            idx0 = idx

        self.xbar_d = np.array(xbar_d)
        self.hv_x = np.array(hv_x)
        self.hv_y = np.array(hv_y)
        self.N = len(xbar_d)
        self.d_d = np.arange(self.step_size/2., d[-1] - 1e-10, self.step_size)

    def detect_sunjumps(self):
        xbar_orig = volts2degs(self.xstim_orig, 10) + self.angle_offset
        xbar_grad_abs = np.abs(np.gradient(fc.unwrap(xbar_orig)))

        _inds = xbar_grad_abs > self.sunjump_threshold_deg
        _idxss = fc.get_contiguous_inds(_inds, keep_ends=True, )
        idxs_continous = []
        for _idxs in _idxss:
            idxs_continous.append(np.mean(_idxs))
        idxs_continous = np.array(idxs_continous)
        _idxs_todelete = \
            np.where(np.abs(idxs_continous[:-1] - idxs_continous[1:]) <=
                     (self.sampling_rate * self.sunjump_tooclose_threshold_sec))[0]
        idxs_todelete = np.concatenate((_idxs_todelete, (_idxs_todelete + 1)))
        idxs_continous[idxs_todelete] = np.nan
        _idxs_sunjumps = idxs_continous[~np.isnan(idxs_continous)].astype(int)
        self.sunjumps_idxs = (1. * _idxs_sunjumps / self.subsampling_step).astype(int)

    def calculate_fb_adjacent_to_sunjumps(self, stimid_special, cri_nearby_pre_sec=3, cri_nearby_post_sec=15,
                                          mhv_half_length_sec=3, cri_mhv_height=.8, cri_mhv_duration_mm=10, cri_similar_orientation=10):

        # recalculate fixation bouts, with different set of parameters, compensating for the sudden sun jumping
        mhv_half_length = int(1. * mhv_half_length_sec / self.step_size)
        _, _, mhv_xbar, mhv_amp = self.mean_heading_vector(mhv_half_length)

        for sj_idx in self.sunjumps_idxs:
            _idx_d = np.where(self.d_d <= self.d[sj_idx])[0][-1]
            mhv_amp[_idx_d] = 0

        _, _, _, _, _, _fb_xbar, _fb_dist_travelled, _fb_dist_shifted, _fb_idxs, _fb_idxs_d \
            = self.get_individual_fixation_bout(cri_mhv_height, cri_mhv_duration_mm, mhv_amp, mhv_xbar)

        # _fb_xbar_merge, _fb_dist_travelled_merge, _fb_dist_shifted_merge, _fb_idxs_merge, _fb_idxs_d_merge \
        #     = self.merge_fixation_bouts(cri_similar_orientation, _fb_xbar, _fb_dist_travelled, _fb_dist_shifted, _fb_idxs, _fb_idxs_d)

        # find the indexes for stimid special
        flag_find_stimid = 0
        for i_stim, stimid_info in enumerate(self.stim_window_info):
            stimid, idx0_id, idx1_id = stimid_info
            if stimid == stimid_special:
                # fb_idxs = fb_idxs_merge[i_stim]
                # fb_xbar = fb_xbar_merge[i_stim]
                fb_idxs = _fb_idxs[i_stim]
                fb_xbar = _fb_xbar[i_stim]
                flag_find_stimid = 1
                break
        if flag_find_stimid == 0:
            print 'Did not find the special stimid %1.1f for function "calculate_fb_adjacent_to_sunjumps in fly: %s"' \
                  % (stimid_special, self.recid)
            return False

        # apply criteria on the fixation bouts, to find the right adjacent fixation bouts
        fb_adj_sj = []                      # structure: [sj_idx, [xbar_pre, idx_pre_start, idx_pre_end], [xbar_post, idx_post_start, idx_post_end]]
        fb_idxs = np.asarray(fb_idxs)       # structure: fb_idxs[starting & ending idx, # of fb]

        stop_idxs = self.get_flight_stopping_inds()
        for sj_idx in self.sunjumps_idxs:
            if sj_idx >= idx0_id and sj_idx <= idx1_id:
                # find the adj fb bout before
                if len(np.where(fb_idxs[:,1] < sj_idx)[0]):
                    fborder_pre = np.where(fb_idxs[:,1] < sj_idx)[0][-1]
                    xbar_pre = fb_xbar[fborder_pre]
                    idx_pre_start = fb_idxs[fborder_pre, 0]
                    idx_pre_end = fb_idxs[fborder_pre, 1]
                    if (sj_idx - idx_pre_end) >= (cri_nearby_pre_sec * self.subsampling_rate):   # the fixation bout before sj is too far away
                            continue
                else:
                    continue

                # find the adj fb bout after
                if len(np.where(fb_idxs[:,0] > sj_idx)[0]):
                    fborder_post = np.where(fb_idxs[:,0] > sj_idx)[0][0]
                    xbar_post = fb_xbar[fborder_post]
                    idx_post_start = fb_idxs[fborder_post, 0]
                    idx_post_end = fb_idxs[fborder_post, 1]
                    if (idx_post_start - sj_idx) >= (cri_nearby_post_sec * self.subsampling_rate):   # the fixation bout before sj is too far away
                        continue
                else:
                    continue

                # apply stopping criteria on the fixation bouts
                if not np.sum((stop_idxs > idx_pre_end) & (stop_idxs < idx_post_start)):
                    fb_adj_sj.append([sj_idx, [xbar_pre, idx_pre_start, idx_pre_end], [xbar_post, idx_post_start, idx_post_end]])

        return fb_adj_sj

    def print_fb_params(self):
        print 'mhv_half_length_sec = %2.2f\ncri_mhv_height = %2.2f\ncri_mhv_duration_mm = %2.2f\n' \
              'cri_similar_orientation = %2.2f' % \
              (self.mhv_half_length_sec, self.cri_mhv_height, self.cri_mhv_duration_mm, self.cri_similar_orientation)


def calculate_fixationbout_dynamic(x, y, d, t, d_d, hv_x, hv_y, step_size,
                                   multi_stimid, mhv_half_length, cri_mhv_height, cri_mhv_duration_mm,
                                   cri_mhv_avespeed, cri_similar_orientation, stim_window_info):

    def mean_heading_vector(mhv_half_length):
        mhv_x = np.zeros(len(hv_x))
        mhv_y = np.zeros(len(hv_x))
        mhv_xbar = np.zeros(len(hv_x))
        mhv_amp = np.zeros(len(hv_x))
        for i in np.arange(mhv_half_length, len(hv_x) - mhv_half_length):
            mhv_x[i] = np.mean(hv_x[i - mhv_half_length:i + mhv_half_length])
            mhv_y[i] = np.mean(hv_y[i - mhv_half_length:i + mhv_half_length])
            mhv_xbar[i] = slope_to_headning_angle_menotaxis(mhv_x[i], mhv_y[i])
            mhv_amp[i] = np.hypot(mhv_x[i], mhv_y[i])

        return np.array(mhv_x), np.array(mhv_y), np.array(mhv_xbar), np.array(mhv_amp)

    def get_individual_fixation_bout(cri_mhv_height, cri_mhv_duration_mm, cri_mhv_avespeed, mhv_amp, mhv_xbar, _inds=[]):
        # separate fixation bouts
        # fb_dshift and DPfd_dist are calculating the same thing, not fb_dist
        inds = mhv_amp > cri_mhv_height
        if len(_inds):
            inds = inds & _inds

        idxss = fc.get_contiguous_inds(inds, min_contig_len=0, trim_left=0, trim_right=0, keep_ends=True)

        # calculate fixation angle, in unit of deg
        _fb_xbar_raw, _fb_dist_raw, _fb_dshift_raw, _fb_ave_speed_raw, _fb_idxs_raw, _fb_idxs_d_raw = [], [], [], [], [], []
        for i, idxs in enumerate(idxss):
            _fb_xbar_raw.append(fc.circmean(mhv_xbar[idxs]))
            _fb_dist_raw.append(len(idxs)*step_size)

            ind0 = np.where(d <= d_d[idxs[0]])[0][-1]
            ind1 = np.where(d >= d_d[idxs[-1]])[0][0]
            t0 = t[ind0]
            t1 = t[ind1]
            _fb_dshift_raw.append(np.hypot((x[ind0] - x[ind1]), (y[ind0] - y[ind1])))
            _fb_ave_speed_raw.append(_fb_dist_raw[-1]/(t1-t0))
            _fb_idxs_raw.append([np.where(d <= d_d[idxs[0]])[0][-1], np.where(d >= d_d[idxs[-1]])[0][0]])
            _fb_idxs_d_raw.append([idxs[0], idxs[-1]])

        fb_xbar_raw, fb_dist_raw, fb_dshift_raw, fb_ave_speed_raw, fb_idxs_raw, fb_idxs_d_raw = \
            np.array(_fb_xbar_raw), np.array(_fb_dist_raw), np.array(_fb_dshift_raw), np.array(_fb_ave_speed_raw), _fb_idxs_raw, _fb_idxs_d_raw

        # applying duration and average speed criteria
        _fb_xbar, _fb_dist, _fb_dshift, _fb_ave_speed, _fb_idxs, _fb_idxs_d = [], [], [], [], [], []
        for i_fb in range(len(fb_xbar_raw)):
            if (fb_dist_raw[i_fb] >= cri_mhv_duration_mm) \
                    and (fb_ave_speed_raw[i_fb] >= cri_mhv_avespeed):
                _fb_xbar.append(fb_xbar_raw[i_fb])
                _fb_dist.append(fb_dist_raw[i_fb])
                _fb_dshift.append(fb_dshift_raw[i_fb])
                _fb_ave_speed.append(fb_ave_speed_raw[i_fb])
                _fb_idxs.append(fb_idxs_raw[i_fb])
                _fb_idxs_d.append(fb_idxs_d_raw[i_fb])
        fb_xbar, fb_dist, fb_dshift, fb_ave_speed, fb_idxs, fb_idxs_d = \
            np.array(_fb_xbar), np.array(_fb_dist), np.array(_fb_dshift), np.array(_fb_ave_speed), _fb_idxs, _fb_idxs_d

        return fb_xbar_raw, fb_dist_raw, fb_dshift_raw, fb_ave_speed_raw, fb_idxs_raw, fb_idxs_d_raw, \
               fb_xbar, fb_dist, fb_dshift, fb_ave_speed, fb_idxs, fb_idxs_d

    def merge_fixation_bouts(cri_similar_orientation, fb_xbar, fb_dist, fb_dshift, fb_idxs, fb_idxs_d):

        def new_queue():
            xbar = fb_xbar[i_fb]
            dist = fb_dist[i_fb]
            dshift = fb_dshift[i_fb]
            idxs = fb_idxs[i_fb]
            idxs_d = fb_idxs_d[i_fb]
            return xbar, dist, dshift, idxs, idxs_d

        def fill_queue(xbar, dist, dshift, idxs, idxs_d):
            angles = fc.unwrap(np.array([xbar, fb_xbar[i_fb]]))
            angle = (angles[0] * dist + angles[-1] * fb_dist[i_fb]) / (dist + fb_dist[i_fb])
            xbar = fc.wrap(np.array([angle]))[0]
            dist += fb_dist[i_fb]
            dshift += fb_dshift[i_fb]
            idxs = [idxs[0], fb_idxs[i_fb][-1]]
            idxs_d = [idxs_d[0], fb_idxs_d[i_fb][-1]]
            return xbar, dist, dshift, idxs, idxs_d

        def clear_queue(xbar, dist, dshift, idxs, idxs_d):
            fb_xbar_merge.append(xbar)
            fb_dist_merge.append(dist)
            fb_dshift_merge.append(dshift)
            fb_idxs_merge.append(idxs)
            fb_idxs_d_merge.append(idxs_d)

        fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge, fb_idxs_d_merge = [], [], [], [], []

        # deal with recording that doesn't have or only has one fixation bout
        if len(fb_xbar) == 0:
            fb_xbar_merge = np.array([])
            fb_dist_merge = np.array([])
            fb_idxs_merge = []
            fb_idxs_d_merge = []
            return fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge, fb_idxs_d_merge

        elif len(fb_xbar) == 1:
            fb_xbar_merge = [fb_xbar[0]]
            fb_dist_merge = [fb_dist[0]]
            fb_dshift_merge = [fb_dshift[0]]
            fb_idxs_merge = [fb_idxs[0]]
            fb_idxs_d_merge = [fb_idxs_d[0]]
            return fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge, fb_idxs_d_merge

        else:
            xbar = fb_xbar[0]
            dist = fb_dist[0]
            dshift = fb_dshift[0]
            idxs = fb_idxs[0]
            idxs_d = fb_idxs_d[0]

        for i_fb in np.arange(1, len(fb_xbar)):
            if circdist_incomplete_arena_menotaxis(xbar, fb_xbar[i_fb]) <= cri_similar_orientation:
                xbar, dist, dshift, idxs, idxs_d = fill_queue(xbar, dist, dshift, idxs, idxs_d)     # merge
            else:
                clear_queue(xbar, dist, dshift, idxs, idxs_d)                               # clear queue
                xbar, dist, dshift, idxs, idxs_d = new_queue()                              # new queue
            if i_fb == (len(fb_xbar) - 1):  # last one, clear queue
                clear_queue(xbar, dist, dshift, idxs, idxs_d)

        fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge, fb_idxs_d_merge = \
            np.array(fb_xbar_merge), np.array(fb_dist_merge), np.array(fb_dshift_merge), fb_idxs_merge, fb_idxs_d_merge

        return fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge, fb_idxs_d_merge

    # calculate the mean heading vector in distance coordinate
    mhv_x, mhv_y, mhv_xbar, mhv_amp = mean_heading_vector(mhv_half_length)

    if not multi_stimid:
        # separate individual fixation bout
        fb_xbar_raw, fb_dist_raw, fb_dshift_raw, fb_ave_speed_raw, fb_idxs_raw, fb_idxs_d_raw, \
        fb_xbar, fb_dist, fb_dshift, fb_ave_speed, fb_idxs, fb_idxs_d \
            = get_individual_fixation_bout(cri_mhv_height, cri_mhv_duration_mm, cri_mhv_avespeed, mhv_amp, mhv_xbar)

        # merge fixation bouts
        fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge, fb_idxs_d_merge \
            = merge_fixation_bouts(cri_similar_orientation, fb_xbar, fb_dist, fb_dshift, fb_idxs, fb_idxs_d)

    else:
        fb_xbar_raw, fb_dist_raw, fb_dshift_raw, fb_ave_speed_raw, fb_idxs_raw, \
        fb_idxs_d_raw, fb_xbar, fb_dist, fb_dshift, fb_ave_speed, fb_idxs, \
        fb_idxs_d, fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge, \
        fb_idxs_d_merge, DPfb_xbar_raw, DPfb_dist_raw \
            = initialize_fb_variables_for_multiple_stimids(19, len(stim_window_info))

        for i_stim, (stimid, ind0, ind1) in enumerate(stim_window_info):
            # separate fixation bouts
            d0, d1 = d[ind0], d[ind1]
            inds_ = (d_d >= d0) & (d_d <= d1)
            fb_xbar_raw[i_stim], fb_dist_raw[i_stim], fb_dshift_raw[i_stim], fb_ave_speed_raw[i_stim], \
            fb_idxs_raw[i_stim], fb_idxs_d_raw[i_stim], fb_xbar[i_stim], fb_dist[i_stim], fb_dshift[i_stim], \
            fb_ave_speed[i_stim], fb_idxs[i_stim], fb_idxs_d[i_stim] \
                = get_individual_fixation_bout(cri_mhv_height, cri_mhv_duration_mm, cri_mhv_avespeed, mhv_amp, mhv_xbar, inds_)

            # merge fixation bouts
            fb_xbar_merge[i_stim], fb_dist_merge[i_stim], fb_dshift_merge[i_stim], fb_idxs_merge[i_stim], fb_idxs_d_merge[i_stim] \
                = merge_fixation_bouts(cri_similar_orientation, fb_xbar[i_stim], fb_dist[i_stim], fb_dshift[i_stim],
                                       fb_idxs[i_stim], fb_idxs_d[i_stim])

    return mhv_half_length, cri_mhv_height, cri_mhv_duration_mm, cri_mhv_avespeed, cri_similar_orientation, \
           mhv_x, mhv_y, mhv_xbar, mhv_amp, \
           fb_xbar_raw, fb_dist_raw, fb_dshift_raw, fb_ave_speed_raw, fb_idxs_raw, fb_idxs_d_raw,\
           fb_xbar, fb_dist, fb_dshift, fb_ave_speed, fb_idxs, fb_idxs_d, \
           fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge, fb_idxs_d_merge

def calculate_fixationbout_geometric(x, y, d, t, d_d, multi_stimid, cri_DP_tol, cri_mhv_duration_mm, cri_mhv_avespeed,
                                     cri_similar_orientation, stim_window_info):

    def get_individual_fixation_bout_DP(cri_DP_tol, cri_mhv_duration_mm, cri_mhv_avespeed, _idx0=0, _idx1=0):
        if (not _idx0) and (not _idx1):
            traj = list(np.array([x, y]).T)
        else:
            traj = list(np.array([x[_idx0:_idx1], y[_idx0:_idx1]]).T)
        simplified = np.array(simplify_coords(traj, cri_DP_tol))
        N_DPfb = len(simplified) - 1

        _DPfb_xbar_raw, _DPfb_dist_raw, _DPfb_dshift_raw, _DPfb_ave_speed_raw, _DPfb_idxs_raw, _DPfb_turn_idx = \
            np.zeros(N_DPfb), np.zeros(N_DPfb), np.zeros(N_DPfb), np.zeros(N_DPfb), [], []

        for i in range(N_DPfb+1):
            idx = np.where(x == simplified[i][0])[0][0]
            _DPfb_turn_idx.append(idx)

        for i in range(N_DPfb):
            dx = simplified[i + 1][0] - simplified[i][0]
            dy = simplified[i + 1][1] - simplified[i][1]
            _DPfb_xbar_raw[i] = slope_to_headning_angle_menotaxis(dx, dy)
            _DPfb_dist_raw[i] = d[_DPfb_turn_idx[i + 1]] - d[_DPfb_turn_idx[i]]
            _DPfb_dshift_raw[i] = math.hypot(dx, dy)
            _DPfb_ave_speed_raw[i] = _DPfb_dist_raw[i] / (t[_DPfb_turn_idx[i+1]] - t[_DPfb_turn_idx[i]])
            _DPfb_idxs_raw.append([_DPfb_turn_idx[i], _DPfb_turn_idx[i+1]])

        DPfb_xbar_raw, DPfb_dist_raw, DPfb_dshift_raw, DPfb_ave_speed_raw, DPfb_idxs_raw = \
            np.array(_DPfb_xbar_raw), np.array(_DPfb_dist_raw), np.array(_DPfb_dshift_raw), np.array(_DPfb_ave_speed_raw), _DPfb_idxs_raw

        _DPfb_xbar, _DPfb_dist, _DPfb_dshift, _DPfb_ave_speed, _DPfb_idxs = [], [], [], [], []
        for i_fb in range(len(DPfb_xbar_raw)):
            if (DPfb_dist_raw[i_fb] >= cri_mhv_duration_mm) \
                    and (DPfb_ave_speed_raw[i_fb] >= cri_mhv_avespeed):
                _DPfb_xbar.append(DPfb_xbar_raw[i_fb])
                _DPfb_dist.append(DPfb_dist_raw[i_fb])
                _DPfb_dshift.append(DPfb_dshift_raw[i_fb])
                _DPfb_ave_speed.append(DPfb_ave_speed_raw[i_fb])
                _DPfb_idxs.append(DPfb_idxs_raw[i_fb])

        DPfb_xbar, DPfb_dist, DPfb_dshift, DPfb_ave_speed, DPfb_idxs = \
            np.array(_DPfb_xbar), np.array(_DPfb_dist), np.array(_DPfb_dshift), np.array(_DPfb_ave_speed), _DPfb_idxs

        return DPfb_xbar_raw, DPfb_dist_raw, DPfb_dshift_raw, DPfb_ave_speed_raw, DPfb_idxs_raw, \
               DPfb_xbar, DPfb_dist, DPfb_dshift, DPfb_ave_speed, DPfb_idxs

    def merge_fixation_bouts_DP(cri_similar_orientation, fb_xbar, fb_dist, fb_dshift, fb_idxs):

        def new_queue():
            xbar = fb_xbar[i_fb]
            dist = fb_dist[i_fb]
            dshift = fb_dshift[i_fb]
            idxs = fb_idxs[i_fb]
            return xbar, dist, dshift, idxs

        def fill_queue(xbar, dist, dshift, idxs):
            angles = fc.unwrap(np.array([xbar, fb_xbar[i_fb]]))
            angle = (angles[0] * dist + angles[-1] * fb_dist[i_fb]) / (dist + fb_dist[i_fb])
            xbar = fc.wrap(np.array([angle]))[0]
            dist += fb_dist[i_fb]
            dshift += fb_dshift[i_fb]
            idxs = [idxs[0], fb_idxs[i_fb][-1]]
            return xbar, dist, dshift, idxs

        def clear_queue(xbar, dist, dshift, idxs):
            fb_xbar_merge.append(xbar)
            fb_dist_merge.append(dist)
            fb_dshift_merge.append(dshift)
            fb_idxs_merge.append(idxs)

        fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge = [], [], [], []

        # deal with recording that doesn't have or only has one fixation bout
        if len(fb_xbar) == 0:
            fb_xbar_merge = np.array([])
            fb_dist_merge = np.array([])
            fb_dshift_merge = np.array([])
            fb_idxs_merge = []
            return fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge

        elif len(fb_xbar) == 1:
            fb_xbar_merge = [fb_xbar[0]]
            fb_dist_merge = [fb_dist[0]]
            fb_dshift_merge = [fb_dshift[0]]
            fb_idxs_merge = [fb_idxs[0]]
            return fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge

        else:
            xbar = fb_xbar[0]
            dist = fb_dist[0]
            dshift = fb_dshift[0]
            idxs = fb_idxs[0]

        for i_fb in np.arange(1, len(fb_xbar)):
            if circdist_incomplete_arena_menotaxis(xbar, fb_xbar[i_fb]) <= cri_similar_orientation:
                xbar, dist, dshift, idxs = fill_queue(xbar, dist, dshift, idxs)     # merge
            else:
                clear_queue(xbar, dist, dshift, idxs)                               # clear queue
                xbar, dist, dshift, idxs = new_queue()                              # new queue
            if i_fb == (len(fb_xbar) - 1):  # last one, clear queue
                clear_queue(xbar, dist, dshift, idxs)

        fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge = \
            np.array(fb_xbar_merge), np.array(fb_dist_merge), np.array(fb_dshift_merge), fb_idxs_merge

        return fb_xbar_merge, fb_dist_merge, fb_dshift_merge, fb_idxs_merge

    if not multi_stimid:

        # separate individual fixation bout using Douglas-Peuker algorithm
        DPfb_xbar_raw, DPfb_dist_raw, DPfb_dshift_raw, DPfb_ave_speed_raw, DPfb_idxs_raw, \
        DPfb_xbar, DPfb_dist, DPfb_dshift, DPfb_ave_speed, DPfb_idxs \
            = get_individual_fixation_bout_DP(cri_DP_tol, cri_mhv_duration_mm, cri_mhv_avespeed)

        # merge Douglas-Peuker fixation bouts
        DPfb_xbar_merge, DPfb_dist_merge, DPfb_dshift_merge, DPfb_idxs_merge \
            = merge_fixation_bouts_DP(cri_similar_orientation, DPfb_xbar, DPfb_dist, DPfb_dshift, DPfb_idxs)

    else:
        DPfb_xbar_raw, DPfb_dist_raw, DPfb_dshift_raw, DPfb_ave_speed_raw, DPfb_idxs_raw, \
        DPfb_xbar, DPfb_dist, DPfb_dshift, DPfb_ave_speed, DPfb_idxs, \
        DPfb_xbar_merge, DPfb_dist_merge, DPfb_dshift_merge, DPfb_idxs_merge \
            = initialize_fb_variables_for_multiple_stimids(14, len(stim_window_info))

        for i_stim, (stimid, ind0, ind1) in enumerate(stim_window_info):
            d0, d1 = d[ind0], d[ind1]
            inds_ = (d_d >= d0) & (d_d <= d1)

            # separate individual fixation bout using Douglas-Peuker algorithm
            DPfb_xbar_raw[i_stim], DPfb_dist_raw[i_stim], DPfb_dshift_raw[i_stim], \
            DPfb_ave_speed_raw[i_stim], DPfb_idxs_raw[i_stim], DPfb_xbar[i_stim], \
            DPfb_dist[i_stim], DPfb_dshift[i_stim], DPfb_ave_speed[i_stim], DPfb_idxs[i_stim] \
                = get_individual_fixation_bout_DP(cri_DP_tol, cri_mhv_duration_mm, cri_mhv_avespeed, ind0, ind1)

            # merge Douglas-Peuker fixation bouts
            DPfb_xbar_merge[i_stim], DPfb_dist_merge[i_stim], DPfb_dshift_merge[i_stim], DPfb_idxs_merge[i_stim] \
                = merge_fixation_bouts_DP(cri_similar_orientation, DPfb_xbar[i_stim], DPfb_dist[i_stim],
                                          DPfb_dshift[i_stim], DPfb_idxs[i_stim])

    return cri_DP_tol, cri_mhv_duration_mm, cri_mhv_avespeed, cri_similar_orientation,\
           DPfb_xbar_raw, DPfb_dist_raw, DPfb_dshift_raw, DPfb_ave_speed_raw, DPfb_idxs_raw, \
           DPfb_xbar, DPfb_dist, DPfb_dshift, DPfb_ave_speed, DPfb_idxs, \
           DPfb_xbar_merge, DPfb_dist_merge, DPfb_dshift_merge, DPfb_idxs_merge




