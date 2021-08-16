import imaging_2021Paper as im
import functions_2021Paper as fc
import plotting_help_2021Paper as ph
import traveldirection_2021Paper as td

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import numpy as np
import matplotlib
from scipy import stats, ndimage, optimize
import random
from matplotlib import cm
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid.inset_locator import inset_axes
import copy, os, glob, csv

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib import colors as mcolors

reload(ph)
reload(fc)


class LocomotionIndex():
    def __init__(self, filename):
        self.filename = filename
        self.Gal4, self.indicator, loc = filename.split('/')[-1].split('-')
        self.loc = loc[:2]
        self.open()

    def open(self):
        with open(self.filename, 'rb') as csvfile:
            reader = csv.reader(csvfile)
            mylist = list(reader)[0]
            if len(mylist) != 4:
                raise ValueError('data in file %s do not have correct format!' % self.filename)

            if mylist[0] == 'nan':
                raise ValueError('file: %s does not have a valid xcorr_index!' % self.filename)
            else:
                self.xcorr_mean = float(mylist[0])

            if mylist[1] == 'nan':
                self.xcorr_std = 0
            else:
                self.xcorr_std = float(mylist[1])

            if mylist[2] == 'nan':
                raise ValueError('file: %s does not have a valid forw_select_index!' % self.filename)
            else:
                self.forw_selective_mean = float(mylist[2])

            if mylist[3] == 'nan':
                self.forw_selective_std = 0
            else:
                self.forw_selective_std = float(mylist[3])


def seq_light_colours():

    return [(1,.4,.4), (.4,.7,1), (1,1,.4), (1,.4,1), (.4,1,.7),
            (.4,1,1), (1,.7,.4), (.7,.4,1), (.7,1,.4), (1,.4,.7)]


def seq_dark_colours():
    return [(.8,0,0), (0,.4,.8), (.8,0,.4), (.8,.4,0), (.6,.6,0), (.8,.8,0),
            (.4,.8,0), (0,.8,.8), (.4,0,.8), (.8,0,.8), (0,.8,.4), (.6,0,.6), (0,.6,.6)]


def get_recs_no(recnamess, celltype, parent_folder='./', **kwargs):

    def get_trial_name(recname):
        folder = recname[11:]
        year = recname[:4]
        month = recname[4:6]
        day = recname[6:8]
        recnum = recname[8:10]
        date = year + '_' + month + '_' + day
        return folder + '/' + date + '/' + date + '-0' + recnum + '/'

    recs = [
        [im.GCaMP_ABF(parent_folder + get_trial_name(recname), celltype, **kwargs)
         for recname in recnames]
        for recnames in recnamess
    ]
    return recs


def get_trial_ts(rec):
    trials = np.diff(rec.abf.stimid) == 0
    trials = ndimage.binary_erosion(trials, iterations=15)
    trials = ndimage.binary_dilation(trials, iterations=15).astype(int)
    trial_starts = rec.abf.t[np.where(np.diff(trials)>0)[0]+1]
    trial_ends = rec.abf.t[np.where(np.diff(trials)<0)[0]+1]

    if len(trial_starts):
        if trial_starts[0] > trial_ends[0]:
            trial_ends = trial_ends[1:]
        return zip(trial_starts, trial_ends)
    else:
        # print rec.name
        return None


def get_walking_bouts_inds(rec):
    # returns a list of the onset and offset inds for each detected walking bouts

    thresh_speed = 4
    thresh_turn = 25
    standing_dur = 3
    thresh_bouts_dur = 10
    thresh_forward_ball_rotation = 1 * 360

    walking_bouts_inds = []
    minlen_idx = int(np.ceil(standing_dur * rec.abf.subsampling_rate))

    idx = (np.abs(rec.abf.dhead) < thresh_turn) & (rec.abf.speed < thresh_speed)
    inds = fc.get_contiguous_inds(idx, trim_left=0, keep_ends=True, min_contig_len=minlen_idx)
    t = rec.abf.t
    standing_num = len(inds)
    for i, ind in enumerate(inds):
        if i != (standing_num-1):
            if (t[inds[i+1][0]] - t[ind[-1]]) >= thresh_bouts_dur:
                forw = fc.unwrap(rec.abf.forw[ind[-1]:inds[i+1][0]])
                if (np.max(forw) - np.min(forw)) >= thresh_forward_ball_rotation:
                    walking_bouts_inds.append([ind[-1], inds[i+1][0]])
        else:
            if (t[-1] - t[ind[-1]]) >= thresh_bouts_dur:
                forw = fc.unwrap(rec.abf.forw[ind[-1]:])
                if (np.max(forw) - np.min(forw)) >= thresh_forward_ball_rotation:
                    walking_bouts_inds.append([ind[-1], len(t)-1])
    return walking_bouts_inds


def correlate(A, V, dt=1, fig=1):
    length = len(A)
    x = (np.arange(0, length) - length/2) * dt
    a = (A - np.mean(A)) / (np.std(A) * length)
    v = (V - np.mean(V)) / np.std(V)

    if fig == 1:
        rcParams['font.size'] = 20
        rcParams['xtick.labelsize'] = 'large'
        rcParams['ytick.labelsize'] = 'large'
        rcParams['axes.labelsize'] = 'large'
        plt.figure(1, (10, 5))
        gs = gridspec.GridSpec(1, 1)

        ax = plt.subplot(gs[0, 0])
        ax.plot(x, np.correlate(a, v, 'same'), c='green', lw=1)
        plt.xlim(-5, 5)
        plt.xlabel('delay (s)')
        plt.ylabel('xcorr')
    return x, np.correlate(a, v, 'same')


def plot_scatter_intensity_velocity_onefly(rec, location='fb', fig=1):
    # fig=1: scatter + bin ave
    if location=='pb':
        intensity = rec.pb.c1.al.mean(axis=-1)
    elif location == 'fb':
        intensity = rec.fb.c1.al.mean(axis=-1)
    elif location == 'no':
        intensity = rec.no.c1.al.mean(axis=-1)
    velocity = rec.subsample('dforw')
    rotation_speed = rec.subsample('dhead')

    if fig == 1:
        rcParams['font.size'] = 20
        rcParams['xtick.labelsize'] = 'large'
        rcParams['ytick.labelsize'] = 'large'
        rcParams['axes.labelsize'] = 'large'
        plt.figure(1, (15, 5))
        gs = gridspec.GridSpec(1, 2)

        ax = plt.subplot(gs[0, 0])
        ax.scatter(velocity, intensity, facecolor='none', alpha=.2 )
        ax.set_xlabel('Forward velocity (mm/s)')
        ax.set_ylabel('Intensity')

        ax = plt.subplot(gs[0, 1])
        ax.scatter(np.abs(rotation_speed), intensity, facecolor='none', alpha=.2)
        ax.set_xlabel('Ro. speed (deg/s)')
        ax.set_ylabel('Intensity')


def plot_scatter_intensity_velocity_group(group, location='fb', bins=40, fig=1):
    # fig=1: scatter + bin ave
    markersize = 2
    alpha = 0.2

    intensity = []
    velocity = []
    rotation_speed = []

    for recs in group:
        for rec in recs:
            if location == 'pb':
                # intensity.extend(np.nanmean(rec.pb.c1.al, axis=-1))
                intensity.extend(rec.pb.c1.mean_pl)
            elif location == 'fb':
                intensity.extend(rec.fb.c1.al.mean(axis=-1))
            elif location == 'no':
                intensity.extend(rec.no.c1.al.mean(axis=-1))
            elif location == 'eb':
                intensity.extend(rec.eb.c1.mean_pl)
            velocity.extend(rec.subsample('dforw'))
            rotation_speed.extend(rec.subsample('dhead'))

    intensity = np.array(intensity)
    velocity = np.array(velocity)
    rotation_speed = np.array(rotation_speed)

    if fig == 1:
        rcParams['font.size'] = 20
        rcParams['xtick.labelsize'] = 'large'
        rcParams['ytick.labelsize'] = 'large'
        rcParams['axes.labelsize'] = 'large'
        plt.figure(1, (15, 5))
        gs = gridspec.GridSpec(1, 2)

        ax = plt.subplot(gs[0, 0])
        ax.scatter(velocity, intensity, facecolor='none', alpha=alpha, s=markersize)
        x, y_mean, y_std = fc.scatter_to_errorbar(x_input=velocity, y_input=intensity, binx=bins, xmin=-5, xmax=10)
        ax.errorbar(x, y_mean, y_std, fmt='o')
        ax.set_xlabel('Forward velocity (mm/s)')
        ax.set_ylabel('Ave. Intensity')
        ax.set_xlim([-4.2, 10.5])
        ax.set_xticks(np.arange(-4, 8.1, 4))
        ax.set_ylim([0, 1])
        ax.set_yticks(np.arange(0, 1.1, .25))

        ax = plt.subplot(gs[0, 1])
        ax.scatter(np.abs(rotation_speed), intensity, facecolor='none', alpha=alpha, s=markersize)
        x, y_mean, y_std = fc.scatter_to_errorbar(x_input=np.abs(rotation_speed), y_input=intensity, binx=bins, xmin=0, xmax=400)
        ax.errorbar(x, y_mean, y_std, fmt='o')
        ax.set_xlabel('Ro. speed (deg/s)')
        ax.set_xlim([-10, 410])
        ax.set_xticks(np.arange(0, 400.1, 100))
        ax.set_ylim([0, 1])
        ax.set_yticks(np.arange(0, 1.1, .25))


def plot_xcorr_no_onefly(rec, r_A=[], r_B=[], fig=1):
    # fig=1: cross correlation
    # default -- A: dhead, B: noasym
    ts = get_trial_ts(rec)
    A_dark = []
    A_bar = []
    B_dark = []
    B_bar = []
    if len(r_A) == 0:
        r_A = rec.subsample('dhead', lag_ms=0)
    if len(r_B) == 0:
        r_B = rec.no.c1.al[:,1] - rec.no.c1.al[:,0]
    r_stimid = rec.subsample('stimid', lag_ms=0)
    for trial_repeat_index, (t0, t1) in enumerate(ts):
        idx = (rec.no.t >= t0) & (rec.no.t < t1)
        istimid = int(r_stimid[idx].mean().round())
        if istimid == 0:
            #idx = idx & (np.abs(rec.no.xstim) < 135)
            A_bar.append(r_A[idx])
            B_bar.append(r_B[idx])
        else:
            A_dark.append(r_A[idx])
            B_dark.append(r_B[idx])
    A_dark = np.array(A_dark)
    A_bar = np.array(A_bar)
    B_dark = np.array(B_dark)
    B_bar = np.array(B_bar)

    if fig == 1:
        rcParams['font.size'] = 20
        rcParams['xtick.labelsize'] = 'large'
        rcParams['ytick.labelsize'] = 'large'
        rcParams['axes.labelsize'] = 'large'
        plt.figure(1, (15, 5))
        gs = gridspec.GridSpec(1, 2)

        dt = rec.no.sampling_period
        colorlist = {0:'red', 1:'blue', 2:'green', 3:'orange', 4:'pink', 5:'purple'}

        ax = plt.subplot(gs[0, 0])
        for index in range(len(A_bar)):
            length = len(A_bar[index])
            x = (np.arange(0, length) - length/2) * dt
            a = (A_bar[index] - np.mean(A_bar[index])) / (np.std(A_bar[index]) * length)
            v = (B_bar[index] - np.mean(B_bar[index])) / np.std(B_bar[index])
            ax.plot(x, np.correlate(a, v, 'same'), c=colorlist[index], lw=1)
            ax.set_xlim([-5, 5])
        ax.set_ylabel('auto-correlation')
        ax.set_xlabel('delay (s)')
        ax.set_title('Bar')

        ax = plt.subplot(gs[0, 1])
        for index in range(len(A_dark)):
            length = len(A_dark[index])
            x = (np.arange(0, length) - length/2) * dt
            a = (A_dark[index] - np.mean(A_dark[index])) / (np.std(A_dark[index]) * length)
            v = (B_dark[index] - np.mean(B_dark[index])) / np.std(B_dark[index])
            ax.plot(x, np.correlate(a, v, 'same'), c=colorlist[index], lw=1)
            ax.set_xlim([-5, 5])
        ax.set_ylabel('auto-correlation')
        ax.set_xlabel('delay (s)')
        ax.set_title('Dark')


def plot_xcorr_pb_onefly(rec, r_A=[], r_B=[], fig=1):
    # fig=1: cross correlation
    # default -- A: dhead, B: noasym
    ts = get_trial_ts(rec)
    A_dark = []
    A_bar = []
    B_dark = []
    B_bar = []
    if len(r_A) == 0:
        r_A = rec.subsample('dhead', lag_ms=0)
    if len(r_B) == 0:
        r_B = rec.pb.c1.al[:,1] - rec.pb.c1.al[:,0]
    r_stimid = rec.subsample('stimid', lag_ms=0)
    for trial_repeat_index, (t0, t1) in enumerate(ts):
        idx = (rec.pb.t >= t0) & (rec.pb.t < t1)
        istimid = int(r_stimid[idx].mean().round())
        if istimid == 0:
            idx = idx & (np.abs(rec.pb.xstim) < 135)
            A_bar.append(r_A[idx])
            B_bar.append(r_B[idx])
        else:
            A_dark.append(r_A[idx])
            B_dark.append(r_B[idx])
    A_dark = np.array(A_dark)
    A_bar = np.array(A_bar)
    B_dark = np.array(B_dark)
    B_bar = np.array(B_bar)

    if fig == 1:
        rcParams['font.size'] = 20
        rcParams['xtick.labelsize'] = 'large'
        rcParams['ytick.labelsize'] = 'large'
        rcParams['axes.labelsize'] = 'large'
        plt.figure(1, (15, 5))
        gs = gridspec.GridSpec(1, 2)

        dt = rec.pb.sampling_period
        colorlist = {0:'red', 1:'blue', 2:'green', 3:'orange', 4:'pink', 5:'purple'}

        ax = plt.subplot(gs[0, 0])
        for index in range(len(A_bar)):
            length = len(A_bar[index])
            x = (np.arange(0, length) - length/2) * dt
            a = (A_bar[index] - np.mean(A_bar[index])) / (np.std(A_bar[index]) * length)
            v = (B_bar[index] - np.mean(B_bar[index])) / np.std(B_bar[index])
            ax.plot(x, np.correlate(a, v, 'same'), c=colorlist[index], lw=1)
            ax.set_xlim([-5, 5])
        ax.set_ylabel('auto-correlation')
        ax.set_xlabel('delay (s)')
        ax.set_title('Bar')

        ax = plt.subplot(gs[0, 1])
        for index in range(len(A_dark)):
            length = len(A_dark[index])
            x = (np.arange(0, length) - length/2) * dt
            a = (A_dark[index] - np.mean(A_dark[index])) / (np.std(A_dark[index]) * length)
            v = (B_dark[index] - np.mean(B_dark[index])) / np.std(B_dark[index])
            ax.plot(x, np.correlate(a, v, 'same'), c=colorlist[index], lw=1)
            ax.set_xlim([-5, 5])
        ax.set_ylabel('auto-correlation')
        ax.set_xlabel('delay (s)')
        ax.set_title('Dark')


def plot_xcorr_fb_onefly(rec, r_A=[], r_B=[], fig=1):
    # fig=1: cross correlation
    # default -- A: dhead, B: noasym
    ts = get_trial_ts(rec)
    A_dark = []
    A_bar = []
    B_dark = []
    B_bar = []
    if len(r_A) == 0:
        r_A = rec.subsample('dhead', lag_ms=0)
    if len(r_B) == 0:
        r_B = rec.fb.c1.al[:,1] - rec.fb.c1.al[:,0]
    r_stimid = rec.subsample('stimid', lag_ms=0)
    for trial_repeat_index, (t0, t1) in enumerate(ts):
        idx = (rec.fb.t >= t0) & (rec.fb.t < t1)
        istimid = int(r_stimid[idx].mean().round())
        if istimid == 0:
            # idx = idx & (np.abs(rec.fb.xstim) < 135)
            A_bar.append(r_A[idx])
            B_bar.append(r_B[idx])
        else:
            A_dark.append(r_A[idx])
            B_dark.append(r_B[idx])
    A_dark = np.array(A_dark)
    A_bar = np.array(A_bar)
    B_dark = np.array(B_dark)
    B_bar = np.array(B_bar)

    if fig == 1:
        rcParams['font.size'] = 20
        rcParams['xtick.labelsize'] = 'large'
        rcParams['ytick.labelsize'] = 'large'
        rcParams['axes.labelsize'] = 'large'
        plt.figure(1, (15, 5))
        gs = gridspec.GridSpec(1, 2)

        dt = rec.fb.sampling_period
        colorlist = {0:'red', 1:'blue', 2:'green', 3:'orange', 4:'pink', 5:'purple'}

        ax = plt.subplot(gs[0, 0])
        for index in range(len(A_bar)):
            length = len(A_bar[index])
            x = (np.arange(0, length) - length/2) * dt
            a = (A_bar[index] - np.mean(A_bar[index])) / (np.std(A_bar[index]) * length)
            v = (B_bar[index] - np.mean(B_bar[index])) / np.std(B_bar[index])
            ax.plot(x, np.correlate(a, v, 'same'), c=colorlist[index], lw=1)
            ax.set_xlim([-5, 5])
        ax.set_ylabel('auto-correlation')
        ax.set_xlabel('delay (s)')
        ax.set_title('Bar')

        ax = plt.subplot(gs[0, 1])
        for index in range(len(A_dark)):
            length = len(A_dark[index])
            x = (np.arange(0, length) - length/2) * dt
            a = (A_dark[index] - np.mean(A_dark[index])) / (np.std(A_dark[index]) * length)
            v = (B_dark[index] - np.mean(B_dark[index])) / np.std(B_dark[index])
            ax.plot(x, np.correlate(a, v, 'same'), c=colorlist[index], lw=1)
            ax.set_xlim([-5, 5])
        ax.set_ylabel('auto-correlation')
        ax.set_xlabel('delay (s)')
        ax.set_title('Dark')


def plot_intensity_asymmetry_vs_rotation_onefly(rec, r_intensity_diff=[], fig=1):
    # Nodule asymmetry by default
    # fig=1: errorbar
    # fig=2: plot
    # fig=3: scatter
    # fig=9: scatter & errorbar
    if len(r_intensity_diff) == 0:
        r_intensity_diff = rec.no.c1.al[:,1] - rec.no.c1.al[:,0]
    r_rotation_speed = rec.subsample('dhead')
    rotation_max = np.max(np.abs(r_rotation_speed))
    xlim = np.ceil(rotation_max/100)*100

    bins = np.arange(-xlim, xlim+0.1, 25)
    inds = np.digitize(r_rotation_speed, bins)

    intensity_mean = np.zeros(len(bins) - 1)
    intensity_std = np.zeros(len(bins) - 1)
    intensity_mean[:] = np.nan
    intensity_std[:] = np.nan
    for i in range(1, len(bins)):
        idx = inds == i
        if idx.sum() > 0:
            intensity_mean[i-1] = r_intensity_diff[idx].mean()
            intensity_std[i-1] = r_intensity_diff[idx].std()

    if fig != 0:
        rcParams['font.size'] = 20
        rcParams['xtick.labelsize'] = 'large'
        rcParams['ytick.labelsize'] = 'large'
        rcParams['axes.labelsize'] = 'large'

        if fig != 9:
            plt.figure(1, (8, 5))
            gs = gridspec.GridSpec(1, 1)
            ax = plt.subplot(gs[0, 0])
            x = (bins[:-1] + bins[1:])/2.0
            if fig == 1:
                ax.errorbar(x, intensity_mean, intensity_std)
            elif fig == 2:
                ax.plot(x, intensity_mean)
            elif fig == 3:
                ax.scatter(r_rotation_speed, r_intensity_diff)
            ax.set_xlabel('turning velocity (deg/s)')
            ax.set_ylabel('Right - Left\n Bridge')
        else:
            plt.figure(1, (15, 5))
            gs = gridspec.GridSpec(1, 2)

            intensity_max = np.max(np.abs(r_intensity_diff))
            rotation_max = np.max(np.abs(r_rotation_speed))
            ylim = np.ceil(intensity_max*20)/20.0
            xlim = np.ceil(rotation_max/50)*50

            ax = plt.subplot(gs[0, 0])
            ax.scatter(r_rotation_speed, r_intensity_diff, alpha=0.2)
            ax.set_xlabel('turning velocity (deg/s)')
            ax.set_ylabel('Right - Left\n Bridge')
            ax.set_xlim([-xlim, xlim])
            ax.set_ylim([-ylim, ylim])

            ax = plt.subplot(gs[0, 1])
            x = (bins[:-1] + bins[1:])/2.0
            ax.errorbar(x, intensity_mean, intensity_std)
            ax.set_xlabel('turning velocity (deg/s)')
            ax.set_xlim([-xlim, xlim])
            ax.set_ylim([-ylim, ylim])


def plot_intensity_asymmetry_vs_rotation_no_population(group, r_intensity_diff=[], fig=1):
    # Nodule asymmetry by default
    # fig=1: plot individual traces and averaged traces over fly
    # fig=2: plot individual traces only

    bins = np.arange(-500, 500.1, 25)
    count = 0
    for fly in group:
        for rec in fly:
            count += 1
    intensity_mean = np.zeros((count, len(bins) - 1))
    intensity_mean[:, :] = np.nan

    count = 0
    for fly in group:
        for rec in fly:
            if len(r_intensity_diff) == 0:
                r_intensity_diff = rec.no.c1.al[:,1] - rec.no.c1.al[:,0]
            r_rotation_speed = rec.subsample('dhead')
            inds = np.digitize(r_rotation_speed, bins)
            for i in range(1, len(bins)):
                idx = inds == i
                if idx.sum() > 0:
                    intensity_mean[count][i-1] = r_intensity_diff[idx].mean()
            count += 1

    if fig == 1:
        rcParams['font.size'] = 20
        rcParams['xtick.labelsize'] = 'large'
        rcParams['ytick.labelsize'] = 'large'
        rcParams['axes.labelsize'] = 'large'
        plt.figure(1, (5, 5))
        gs = gridspec.GridSpec(1, 1)

        ax = plt.subplot(gs[0, 0])
        x = (bins[:-1] + bins[1:])/2.0
        ax.set_xlabel('turning velocity (deg/s)')
        ax.set_ylabel('Right - Left\n Bridge')
        for index in range(count):
            ax.plot(x, intensity_mean[index], c=[1, .8, .6], lw=1)
        ax.plot(x, np.mean(intensity_mean, axis=0), c=[1, 0.4, 0], lw=2)


    if fig == 2:
        rcParams['font.size'] = 20
        rcParams['xtick.labelsize'] = 'large'
        rcParams['ytick.labelsize'] = 'large'
        rcParams['axes.labelsize'] = 'large'
        plt.figure(1, (8, 5))
        gs = gridspec.GridSpec(1, 1)

        ax = plt.subplot(gs[0, 0])
        x = (bins[:-1] + bins[1:])/2.0
        for index in range(count):
            ax.plot(x, intensity_mean[index])
        ax.set_xlabel('turning velocity (deg/s)')
        ax.set_ylabel('Right - Left\n Bridge')


def plot_isolated_turns(group, trigger='dhead', channels=['dhead', 'no.c1.rmll'], ylabel='RML', fig=1):

    peak_dict_left = im.get_isolated_peak_trials(group, direction='right', trigger=trigger, channels=channels)
    peak_dict_right = im.get_isolated_peak_trials(group, direction='left', trigger=trigger, channels=channels)
    # for index in range(len(peak_dict_left['dhead'])):
    #     temp = fc.unwrap(peak_dict_left['dhead'][index][1])
    #     level = np.mean(temp)
    #     peak_dict_left['dhead'][index][1] = fc.wrap(temp - level)
    # for index in range(len(peak_dict_right['dhead'])):
    #     temp = fc.unwrap(peak_dict_right['dhead'][index][1])
    #     level = np.mean(temp)
    #     peak_dict_right['dhead'][index][1] = fc.wrap(temp - level)

    if fig == 1:
        rcParams['font.size'] = 20
        rcParams['xtick.labelsize'] = 'large'
        rcParams['ytick.labelsize'] = 'large'
        rcParams['axes.labelsize'] = 'large'
        plt.figure(1, (30, 15))
        gs = gridspec.GridSpec(2, 2)

        ax = plt.subplot(gs[0, 0])
        for item in peak_dict_left[channels[0]]:
            ax.plot(item[0], item[1], c=[.7, .7, .7], lw=1)
        ax.plot(peak_dict_left[channels[0]][0][0], np.mean(peak_dict_left[channels[0]], axis=0)[1], c=[0, 0, 0], lw=3)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-400, 400)
        ax.set_ylabel('Turning Velocity\n (deg/s)')
        ax.set_xticks([])
        ax.set_yticks([-400, -200, 0, 200, 400])
        ax.set_xticks(np.arange(-1.5, 1.6, .5), minor=True)
        ax.set_yticks(np.arange(-400, 400.1, 100), minor=True)
        ax.grid(which='minor', alpha=.9)

        ax = plt.subplot(gs[0, 1])
        for item in peak_dict_right[channels[0]]:
            ax.plot(item[0], item[1], c=[.7, .7, .7], lw=1)
        ax.plot(peak_dict_right[channels[0]][0][0], np.mean(peak_dict_right[channels[0]], axis=0)[1], c=[0, 0, 0], lw=3)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-400, 400)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticks(np.arange(-1.5, 1.6, .5), minor=True)
        ax.set_yticks(np.arange(-400, 400.1, 100), minor=True)
        ax.grid(which='minor', alpha=0.9)

        ax = plt.subplot(gs[1, 0])
        for item in peak_dict_left[channels[1]]:
            ax.plot(item[0], item[1], c=[1, .8, .6], lw=1)
        ax.plot(peak_dict_left[channels[1]][0][0], np.mean(peak_dict_left[channels[1]], axis=0)[1], c=[1, 0.4, 0], lw=3)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-.6, .6)
        ax.set_yticks(np.arange(-.6, .61, .3))
        ax.set_xticks(np.arange(-1.5, 1.6, .5), minor=True)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Time (s)')

        ax = plt.subplot(gs[1, 1])
        for item in peak_dict_right[channels[1]]:
            ax.plot(item[0], item[1], c=[1, .8, .6], lw=1)
        ax.plot(peak_dict_right[channels[1]][0][0], np.mean(peak_dict_right[channels[1]], axis=0)[1], c=[1, 0.4, 0], lw=3)
        ax.set_yticks([])
        ax.set_xticks(np.arange(-1.5, 1.6, .5), minor=True)
        ax.set_yticks(np.arange(-400, 400.1, 100), minor=True)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-.6, .6)
        ax.set_xlabel('Time (s)')


def plot_phase_xstim(rec):
    rcParams['font.size'] = 20
    rcParams['xtick.labelsize'] = 'large'
    rcParams['ytick.labelsize'] = 'large'
    rcParams['axes.labelsize'] = 'large'

    plt.figure(1, (20, 5))
    plt.plot(rec.fb.t, im.fc.unwrap(rec.fb.c1.phase), '.')
    plt.plot(rec.fb.t, rec.subsample('xstim_unwrap'))
    plt.xlabel('Time (s)')
    plt.ylabel('deg')
    plt.xlim([0, 300])
    plt.legend(['phase', 'xstim'])


def plot_behavior_vs_asymm_no_onefly(rec, fig_num=1):
    # provide global, fast glance of a trace, focusing on speed and intensity asymmetry in Nodule
    rcParams['font.size'] = 40
    rcParams['xtick.labelsize'] = 'large'
    rcParams['ytick.labelsize'] = 'large'
    rcParams['axes.labelsize'] = 'large'
    plt.figure(fig_num, (50, 30))
    gs = gridspec.GridSpec(6, 1, height_ratios=[1,1,1,1,1,.3])

    ax = plt.subplot(gs[0, 0])
    ax.plot(rec.no.t, rec.no.c1.al[:,1] + rec.no.c1.al[:,0], c='k')
    ax.plot(np.array([0, rec.no.t[-1]]), np.array([0, 0]), '--', c='grey')
    plt.ylabel('R+L\n Nodule')
    ax.set_xticks([])
    ax.set_xticks(np.arange(0, rec.no.t[-1], rec.no.t[-1]/10), minor=True)
    ax.grid(which='minor', alpha=1)
    ax.set_xlim([0, rec.no.t[-1]])

    ax = plt.subplot(gs[1, 0])
    ax.plot(rec.no.t, rec.no.c1.al[:,1] - rec.no.c1.al[:,0], c='k')
    ax.plot(np.array([0, rec.no.t[-1]]), np.array([0, 0]), '--', c='grey')
    plt.ylabel('R-L\n Nodule')
    ax.set_xticks([])
    ax.set_xticks(np.arange(0, rec.no.t[-1], rec.no.t[-1]/10), minor=True)
    ax.grid(which='minor', alpha=1)
    ax.set_xlim([0, rec.no.t[-1]])

    ax = plt.subplot(gs[2, 0])
    dhead = rec.subsample('dhead')
    ax.plot(rec.no.t, dhead, c='k')
    plt.ylabel('turning velocity\n (deg/s)')
    ax.set_xticks([])
    ax.set_xticks(np.arange(0, rec.no.t[-1], rec.no.t[-1]/10), minor=True)
    ax.grid(which='minor', alpha=1)
    ax.set_xlim([0, rec.no.t[-1]])

    ax = plt.subplot(gs[3, 0])
    dforw = rec.subsample('dforw')
    ax.plot(rec.no.t, dforw, c='k')
    plt.ylabel('forward speed\n (mm/s)')
    ax.set_xticks([])
    ax.set_xticks(np.arange(0, rec.no.t[-1], rec.no.t[-1]/10), minor=True)
    ax.grid(which='minor', alpha=1)
    ax.set_xlim([0, rec.no.t[-1]])

    ax = plt.subplot(gs[4, 0])
    dside = rec.subsample('dside')
    ax.plot(rec.no.t, dside, c='k')
    plt.ylabel('side speed\n (mm/s)')
    plt.xlabel('Time (s)')
    ax.set_xticks([])
    ax.set_xticks(np.arange(0, rec.no.t[-1], rec.no.t[-1]/10), minor=True)
    ax.grid(which='minor', alpha=1)
    ax.set_xlim([0, rec.no.t[-1]])

    ax = plt.subplot(gs[5, 0])
    stimid = rec.subsample('stimid_raw')
    ax.plot(rec.no.t, stimid, c='k')
    plt.ylabel('Stimulus ID\n (mm/s)')
    ax.set_xticks([])
    ax.set_xticks(np.arange(0, rec.no.t[-1], rec.no.t[-1]/10), minor=True)
    ax.grid(which='minor', alpha=1)
    ax.set_xlim([0, rec.no.t[-1]])
    ax.set_ylim([-1, 4])

    ax.set_xticks(np.arange(0, rec.no.t[-1], rec.no.t[-1]/5))
    plt.xlabel('Time (s)')



def filter_locomotion(rec, filtertype=0):
    """
    :param filtertype:
        0: no filter
        1: not standing
        2: walking forward, small turning speed
    :return:
    """

    cri_dforw_standing = 2
    cri_dhead_standing = 10
    cri_dhead_straight = 50

    dforw = rec.subsample('dforw')
    dhead = np.abs(rec.subsample('dhead'))
    N = len(dforw)

    inds = np.ones(N).astype(bool)
    if filtertype == 1:
        inds = (dforw > cri_dforw_standing) | (dhead > cri_dhead_standing)
    elif filtertype == 2:
        inds = ((dforw > cri_dforw_standing) | (dhead > cri_dhead_standing)) & (dhead < cri_dhead_straight)

    return inds


def plot_FFT_of_glomeruli_intensity(recss, ax, filtertype_locomotion=0, ch=1, sigtype='an', period_bound=32,
                                    color=ph.grey8, lw_indi=1, lw=4, alpha=0.5, fs=15, legend='WT', proj2pb=False,
                                    ylim=[0,0.5], yticks=[0,0.2,0.4], plotcurvesonly=False, irecss=0, showflynum=True):
    """
    :param recss:
    :param ax:
    :param filtertype_locomotion:
        0: no filter
        1: not standing
        2: walking forward, small turning speed
    :return:
    """

    if not ax:
        ax = ph.large_ax([5, 3])
    ph.set_fontsize(fs)

    powers = []
    for recs in recss:

        data_fly = []
        for rec in recs:

            # inds = filter_locomotion(rec, filtertype_locomotion)
            inds = rec.ims[0].t > -100
            if ch == 1:
                if sigtype == 'an':
                    data_fly.extend(rec.ims[0].c1.an[inds])
                elif sigtype == 'al':
                    data_fly.extend(rec.ims[0].c1.al[inds])
            elif ch == 2:
                if sigtype == 'an':
                    data_fly.extend(rec.ims[0].c2.an[inds])
                elif sigtype == 'al':
                    data_fly.extend(rec.ims[0].c2.al[inds])

        if len(data_fly):
            data_fly = np.array(data_fly)
            if np.isnan(data_fly[0][0]):
                data_fly = data_fly[:,1:-1]
            if np.isnan(data_fly[0][8]):
                data_fly = data_fly[:,[0,1,2,3,4,5,6,7,10,11,12,13,14,15,16,17]]
            if proj2pb:
                data_fly = data_fly[:, [0,2,4,6,8,10,12,14,15,1,3,5,7,9,11,13]]
            power, period, _ = fc.powerspec(data_fly)
            power_fly_m = np.mean(power, axis=0)
            powers.append(power_fly_m)
    m = np.mean(powers, axis=0)
    s = fc.nansem(powers, 0)

    indx = np.where(period <= period_bound)[0][0]
    period = period[indx:]
    m = m[indx:]
    flynum = len(powers)
    for i in range(flynum):
        ax.plot(period, powers[i][indx:], c=color, lw=lw_indi, alpha=alpha)
    ax.plot(period, m, c=color, lw=lw, label=legend)
    ax.legend(prop={'size': fs}, loc='upper right')

    if showflynum:
        ytext = ylim[-1] - (ylim[-1] - ylim[0]) / 11. * irecss
        ax.text(2, ytext, '%i  flies' % flynum, ha='left', va='top', fontsize=fs, color=color)

    if not plotcurvesonly:
        ax.set_xlabel('Periodicity / Glomerulus', fontsize=fs)
        ax.set_ylabel('Amplitude (a.u.)', fontsize=fs)
        ph.adjust_spines(ax, ['left', 'bottom'], lw=1, xticks=[2,4,8,16], yticks=yticks, xlim=[2, period_bound], ylim=ylim)
        ax.patch.set_facecolor('white')
        ax.grid(which='major', alpha=0)

def get_FFT_of_glomeruli_intensity_Arrayinput(data, period_bound=32,):

    power, period, _ = fc.powerspec(data)
    power = np.mean(power, axis=0)
    indx = np.where(period <= period_bound)[0][0]

    return period[indx:], power[indx:]



def plot_dphase_dhead_xcorr(recss, ax, ch=1, t_window=[-2.2,2.2], color=ph.grey8, lw_indi=1, lw=4, alpha=0.5, fs=15,
                            ylim=[-0.1,0.8], yticks=[0,0.4,0.8], plotcurvesonly=False, irecss=0, showflynum=False):

    if not ax:
        ax = ph.large_ax([5, 3])
    ph.set_fontsize(fs)

    t_lag = np.arange(t_window[0] - 0.1, t_window[1] + 0.1, 0.005)
    xcorrs = []
    for recs in recss:

        xcorr_fly = []
        for rec in recs:

            if ch == 1:
                dphase = rec.ims[0].c1.dphase
            elif ch == 2:
                dphase = rec.ims[0].c2.dphase
            dhead = -rec.subsample('dhead')
            t_lag_trial, xcorr_trial = fc.xcorr(dphase, dhead, sampling_period=rec.ims[0].sampling_period)
            f = interp1d(t_lag_trial, xcorr_trial, kind='linear')
            xcorr_interp = f(t_lag)
            xcorr_fly.append(xcorr_interp)

        if len(xcorr_fly):
            xcorr_fly = np.array(xcorr_fly)
            xcorr_fly_m = np.mean(xcorr_fly, axis=0)
            xcorrs.append(xcorr_fly_m)

    m = np.mean(xcorrs, axis=0)
    # s = fc.nansem(xcorrs, 0)

    flynum = len(xcorrs)
    for i in range(flynum):
        ax.plot(t_lag, xcorrs[i], c=color, lw=lw_indi, alpha=alpha)
    ax.plot(t_lag, m, c=color, lw=lw)
    ax.axvline(0, c=ph.grey5, ls='--', alpha=0.4)
    if showflynum:
        ytext = ylim[-1] - (ylim[-1] - ylim[0]) / 11. * irecss
        ax.text(t_window[0], ytext, '%i  flies' % flynum, ha='left', va='top', fontsize=fs)

    if not plotcurvesonly:
        y0 = ylim[-1]
        y1 = ylim[-1] - (ylim[-1] - ylim[0]) * 0.2
        ax.text(t_window[0], y0, '$\Delta$phase v.s.\n$\Delta$bar', ha='left', va='top')
        ax.set_xlabel('t lag (s)')
        ax.set_ylabel('xcorr')
        ph.adjust_spines(ax, ['left', 'bottom'], lw=1, xticks=[-2, 0, 2], xlim=t_window,
                         yticks=yticks, ylim=ylim)
        ax.patch.set_facecolor('white')
        ax.grid(which='major', alpha=0)


def plot_intensity_locomotion_xcorr(recss, ax, sig_type='mean_an', loc_num=0, ch=1, t_window=[-2.2,2.2], color=ph.grey8,
                                    behtype='forw', lw_indi=1, lw=4, alpha=0.5, fs=15, ylim=[-0.1,0.8], yticks=[0,0.4,0.8],
                                    plotcurvesonly=False, irecss=0, showflynum=False, noaxis=False,):

    def get_sig(rec, loc_num, sig_type):
        _sig = 0
        if sig_type == 'rmln':
            _sig = rec.ims[loc_num].c1.rmln
        elif sig_type == 'rmll':
            _sig = rec.ims[loc_num].c1.rmll
        elif sig_type == 'rmlz':
            _sig = rec.ims[loc_num].c1.rmlz
        elif sig_type == 'mean_al':
            _sig = rec.ims[loc_num].c1.mean_al
        elif sig_type == 'mean_an':
            _sig = rec.ims[loc_num].c1.mean_an
        elif sig_type == 'mean_az':
            _sig = rec.ims[loc_num].c1.mean_az
        elif sig_type == 'mean_pn':
            _sig = rec.ims[loc_num].c1.mean_pn
        elif sig_type == 'd_mean_an':
            _sig = np.gradient(rec.ims[loc_num].c1.mean_an)
        elif sig_type == 'peak_an':
            _sig = np.nanmax(rec.ims[loc_num].c1.an, axis=-1)
        elif sig_type == 'd_peak_an':
            _sig = np.gradient(np.nanmax(rec.ims[loc_num].c1.an, axis=-1))
        return _sig

    if not ax:
        ax = ph.large_ax([5, 3])
    ph.set_fontsize(fs)

    t_lag = np.arange(t_window[0] - 0.1, t_window[1] + 0.1, 0.005)
    xcorrs = []
    for recs in recss:

        xcorr_fly = []
        for rec in recs:

            # if ch == 1:
            #     Fluo_mean = rec.ims[0].c1.mean_an
            # elif ch == 2:
            #     Fluo_mean = rec.ims[0].c2.mean_an
            Fluo_mean = get_sig(rec, loc_num, sig_type)

            if behtype == 'forw':
                beh = rec.subsample('dforw')
            elif behtype == 'head_abs':
                beh = np.abs(rec.subsample('dhead'))
            elif behtype == 'abs_forw':
                beh = np.abs(rec.subsample('dforw'))
            elif behtype == 'head':
                beh = rec.subsample('dhead')
            elif behtype == 'side':
                beh = rec.subsample('dside')
            elif behtype == 'speed':
                beh = rec.subsample('speed')

            if len(Fluo_mean):
                t_lag_trial, xcorr_trial = fc.xcorr(Fluo_mean, beh, sampling_period=rec.ims[0].sampling_period)
                f = interp1d(t_lag_trial, xcorr_trial, kind='linear')
                xcorr_interp = f(t_lag)
                xcorr_fly.append(xcorr_interp)

        if len(xcorr_fly):
            xcorr_fly = np.array(xcorr_fly)
            xcorr_fly_m = np.mean(xcorr_fly, axis=0)
            xcorrs.append(xcorr_fly_m)

    xcorrs = np.array(xcorrs)
    m = np.mean(xcorrs, axis=0)

    flynum = len(xcorrs)
    for i in range(flynum):
        ax.plot(t_lag, xcorrs[i], c=color, lw=lw_indi, alpha=alpha)
    ax.plot(t_lag, m, c=color, lw=lw)
    ax.axvline(0, c=ph.grey5, ls='--', alpha=0.4, lw=.5)
    if showflynum:
        ytext = ylim[-1] - (ylim[-1] - ylim[0]) / 11. * irecss
        ax.text(t_window[0], ytext, '%i  flies' % flynum, ha='left', va='top', fontsize=fs)

    if not plotcurvesonly:
        ax.text(t_window[0], ylim[-1], '$\Delta$F v.s.\n$\Delta$d%s' % behtype, ha='left', va='top', fontsize=fs)

        if noaxis:
            ph.adjust_spines(ax, ['left', 'bottom'], lw=1, xlim=t_window, xticks=[], ylim=ylim, yticks=[])
        else:
            ax.set_xlabel('t lag (s)')
            ax.set_ylabel('xcorr')
            ph.adjust_spines(ax, ['left', 'bottom'], lw=1, xticks=[-2, 0, 2], xlim=t_window, yticks=yticks, ylim=ylim)

        ax.patch.set_facecolor('white')
        ax.grid(which='major', alpha=0)


def plot_dphase_dhead_lingre(recss, ax, ch=1, filtertype_locomotion=0, bins=20, color=ph.grey8, lw_indi=1, lw=4,
                             alpha=0.5, fs=15, xlim=[-300, 300], ylim=[-150, 150], yticks=[-150,0,150],
                             ylabel_xcoords=-0.1, plotcurvesonly=False, irecss=0, showflynum=False):

    if not ax:
        ax = ph.large_ax([5, 5])
    ph.set_fontsize(fs)

    dhead_group = []
    dphase_group = []
    for recs in recss:

        dhead_fly = []
        dphase_fly = []
        for rec in recs:

            inds = filter_locomotion(rec, filtertype_locomotion)
            if ch == 1:
                dphase = rec.ims[0].c1.dphase[inds]
            elif ch == 2:
                dphase = rec.ims[0].c2.dphase[inds]
            dhead = -rec.subsample('dhead')[inds]

            dhead_fly.extend(dhead)
            dphase_fly.extend(dphase)

        x, y, _ = fc.scatter_to_errorbar(x_input=dhead_fly, y_input=dphase_fly, binx=bins, xmin=xlim[0], xmax=xlim[-1])
        dhead_group.append(x)
        dphase_group.append(y)

    dhead_m = np.nanmean(dhead_group, axis=0)
    dphase_m = np.nanmean(dphase_group, axis=0)

    flynum = len(dhead_group)
    for i in range(flynum):
        ax.plot(dhead_group[i], dphase_group[i], c=color, alpha=alpha, lw=lw_indi)
    ax.plot(dhead_m, dphase_m, c=color, lw=lw)
    ax.axvline(0, c=ph.grey5, ls='--', alpha=0.4)
    ax.axhline(0, c=ph.grey5, ls='--', alpha=0.4)
    if showflynum:
        ytext = ylim[-1] - (ylim[-1] - ylim[0]) / 11. * irecss
        ax.text(xlim[0], ytext, '%i  flies' % flynum, ha='left', va='top', fontsize=fs)

    if not plotcurvesonly:
        ax.set_xlabel('Heading vel (deg/s)')
        ax.set_ylabel('Phase vel (deg/s)')
        ax.get_yaxis().set_label_coords(ylabel_xcoords, 0.5)
        ph.adjust_spines(ax, ['left', 'bottom'], lw=1, xticks=[xlim[0], 0, xlim[-1]], xlim=xlim,
                         yticks=yticks, ylim=ylim)
        ax.patch.set_facecolor('white')
        ax.grid(which='major', alpha=0)


colours = {'EIP': ph.blue, 'PEN': ph.orange, 'PFR': ph.blue, 'PFN': ph.red, 'FBLN': ph.green,
           'PFLC': ph.green, 'FBCre': ph.green, 'FBNo': ph.purple, 'Speed': ph.green}
dcolours = {'EIP': ph.nblue, 'PEN': ph.dorange, 'PFR': ph.nblue, 'PFN': ph.dorange, 'FBLN': ph.green,
            'PFLC': ph.green, 'FBCre': ph.green, 'FBNo': ph.purple, 'Speed': ph.green}
cmaps = {'EIP': plt.cm.Blues, 'PEN': plt.cm.Oranges, 'PFR': plt.cm.Oranges, 'PFN': plt.cm.Oranges, 'FBLN': plt.cm.jet,
         'PFLC': plt.cm.Oranges, 'FBCre': plt.cm.Greens, 'FBNo': plt.cm.Purples, 'Speed': plt.cm.Greens}


def plot(self, tlim=None, powerspec=False, imaging=['fb.c1.an'], phaseoverlay=True,
         overlay=['c1.phase', 'xstim'], channels=[], channels_narrow=[], mask_powerspec=False, vmin=None,
         vmax=None, toffset=None, phase_offsets=[0], cmap=None, lines=[],
         dark_gain=None, gs=None, row=0, trig_t=0, unwrap=False, cancel_phase=True,
         show_axes=True, sf=1, reset_t=False, suffix='', save=False, ar=2, size=None,
         filetype='png', highlight='arena', colorbar=0, fig_num=1, title=[], barjump_trig=False, two_location=False,
         xedges=None):
    """
    Most versatile plotting functions. Used to get explore datasets.
    :param self: GCaMP_ABF recording.
    :param tlim: Time window.
    :param powerspec: True/False. True to plot power spectrum.
    :param imaging: List of labels, starting with a. eg. 'an' = dF/F normalized.
    :param phaseoverlay: True/False. True to overlay phase on the imaging dataset.
    :param overlay: List of labels to plot together. Typically phase and bar or ball.
    :param channels: List of labels to plot under separate columns. Must be abf channels. eg. 'forw' or 'stimid'
    :param mask_powerspec: Do not plot power spectrum, but leave space for it.
    :param vmin: Minimum value in imaging colormap lookup table.
    :param vmax: Maximum value in imaging colormap lookup table.
    :param toffset: Time at which to cancel the offset between phase and bar. Can be an interval, in which case
    the circular mean offset will be computed.
    :param offset: Amount by which to offset the phase. If not defined (ie. None), will be computed using toffset.
    :param cmap: Colormap to plot imaging.
    :param lines: Times to plot horizontal lines across all columns.
    :param dark_gain: Gain with which to multiply the fly's heading. If None, it is computed as the slope between
    the phase and ball velocities in the tlim time window.
    :param gs: Used under plot_trials.
    :param row: Used under plot_trials.
    :param trig_t: Used under plot_trials.
    :param unwrap: True/False. True to unwrap circular data.
    :param show_axes: True to show_axes. Used under plot_trials.
    :param sf: Scale factor for figure size.
    :param reset_t: Sets the x-axis to start at 0.
    :param suffix: String appended to figure name.
    :param save: True to save figure.
    :param ar: Aspect ratio.
    :param filetype: eg.'png', 'svg', 'pdf', etc.
    :return:
    """

    rcParams['font.size'] = 25
    rcParams['ytick.labelsize'] = 'large'
    rcParams['xtick.labelsize'] = 'large'
    rcParams['axes.labelsize'] = 'large'

    if highlight == 'arena' and unwrap:
        highlight = 'all'

    color_dict = {
        plt.cm.Blues: ph.blue,
        plt.cm.Greens: ph.dgreen,
        plt.cm.Purples: ph.dpurple,
        plt.cm.Oranges: ph.orange,
        plt.cm.CMRmap: ph.dpurple,
        plt.cm.jet: ph.blue
    }

    if cmap is None: cmap = cmaps[self.ims[0].c1.celltype]
    if not type(cmap) is tuple: cmap = [cmap]
    if len(cmap) == 1: cmap = list(cmap) + [plt.cm.Reds]
    if not tlim: tlim = [self.ims[0].t[0], self.ims[0].t[-1]]
    if type(tlim) is int:
        trial_times = self.abf.t[self.abf.stimid == tlim]
        tlim = (trial_times[0], trial_times[-1])
    if toffset is None: toffset = tlim[0]

    if reset_t:
        t0 = tlim[0]
        tlim = [0, tlim[1]-tlim[0]]
    else:
        t0 = trig_t

    tabf = self.abf.t - t0
    tabf_orig = self.abf.t_orig - t0

    # Setup Figure if not triggered
    if not gs:
        ncols = np.array([len(imaging), powerspec, len(channels), len(channels_narrow), len(overlay) > 0]).sum()
        def wr_fcn(imaging_label):
            if 'eb' in imaging_label:
                return 1
            elif 'pb' in imaging_label:
                return 2
            elif 'fb' in imaging_label:
                return 1
            elif 'sg' in imaging_label:
                return 1
        wr = map(wr_fcn, imaging) + [1]*powerspec + [1]*(len(overlay)>0) + [1]*len(channels) +[0.5]*len(channels_narrow) # width ratios
        hr = [12, 1]
        if size is None:
            size = (10*sf*np.sqrt(ar), 10*sf/np.sqrt(ar))
        fig = plt.figure(fig_num, size)
        # plt.suptitle(self.folder, size=10)
        rcParams['xtick.direction'] = 'out'
        rcParams['ytick.direction'] = 'out'
        gs = gridspec.GridSpec(2, ncols, width_ratios=wr, height_ratios=hr)
        gs.update(wspace=0.02, hspace=.03)
    axs = []

    if show_axes:
        1+1
        # stim = 'BAR' if self.abf.ystim.mean() < 5 else 'DARK'
        # temp = int(np.round(self.abf.temp.mean()))
        # print '%s, %iC' %(stim, temp)
        # if title:
        #     fig.suptitle(title + ' -- ' + str(temp) + 'C', fontsize=40)
        # else:
        #     print '%iC' % temp

    # Plot Imaging
    if trig_t > 0:
        red_pulse = self.ims[0].c2.an[(self.ims[0].t>0) & (self.ims[0].t<.3)].mean(axis=0)
        pipette = np.argmax(red_pulse) + .5

    irow = row * 2
    for icol, label in enumerate(imaging):
        # Get imtype (pb or eb), channel and gc
        imlabel, clabel, alabel = self.get_labels(label)
        im, c, _ = self.get_objects(label)
        if xedges is None:
            (t, xedges), gc = self.get_signal(label)
        else:
            (t, _), gc = self.get_signal(label)

        # Setup axis
        axgc = plt.subplot(gs[irow, icol])
        if trig_t > 0:
            axgc.arrow(pipette, -.4, 0, .2, head_width=0.5, head_length=0.1, fc='r', ec='r', lw=2)

        # if 0:
        if show_axes:
            # axgc.set_xlabel(imlabel + ' ' + c.celltype, labelpad=20)
            if imlabel == 'pb':
                axgc.set_xticks(np.array([0, 9, 18]))
            elif imlabel == 'eb':
                xticks = np.array([0, 8, 16])
                axgc.set_xticks(xticks)
                axgc.set_xticklabels(xticks)
            spines = []
            if icol == 0:
                spines.append('left')
                # axgc.set_ylabel('Time (s)', rotation='horizontal', labelpad=40)
        else:
            spines = []
        ph.adjust_spines(axgc, spines, ylim=tlim, xlim=[xedges[0], xedges[-1]])

        # Plot gc
        gc[np.isnan(gc)] = 0
        img = plt.pcolormesh(xedges, t-t0, gc, cmap=cmaps[c.celltype])
        # img = plt.pcolormesh(xedges, t - t0, gc, cmap=plt.cm.Blues)
        if not vmin is None: img.set_clim(vmin=vmin)
        if not vmax is None: img.set_clim(vmax=vmax)

        # Overlay phase
        if phaseoverlay:
            phase = c.phase * c.period / 360  +.5 # scale phase to periodicity of the pb or eb, +.5 is to place it in the middle of the glomerulus (since the glomerulus number maps to the left edges of the pcolormesh grid)
            phase[phase < 0] = phase[phase < 0] + c.period
            ph.circplot(phase, im.tc-t0, c=(.1, .1, .1), alpha=.8, period=c.period, zorder=11, lw=2)
        axgc.invert_yaxis()
        axs.append(axgc)

        # Draw colorbar
        if colorbar == 1:
            cbaxes = fig.add_axes([0.17, 0.17, 0.1, 0.01])
            cb = plt.colorbar(ax=axgc, cax=cbaxes,
                              use_gridspec=False, orientation='horizontal')
            cb.ax.yaxis.set_ticks_position('left')
            cb.set_label(r'$\Delta$F/F')
            if vmax:
                cmax = vmax
            else:
                idx = (t>=tlim[0]) & (t<tlim[1])
                cmax = np.floor(gc[idx].max())
                if cmax==0:
                    cmax = np.floor(gc[idx].max()*10)/10.
            cb.set_ticks([0, cmax])

    if powerspec:
        icol += 1
        topax = plt.subplot(gs[irow, icol])
        im = self.ims[0]
        power, period, phase = fc.powerspec(im.c1.an, mask=False, cmap=plt.cm.Greys, show=False, n=10)


        # Plot power spectrum over time
        ph.adjust_spines(topax, ['top'])
        # topax.set_ylabel('Time (s)', rotation='horizontal', ha='right', labelpad=20)
        if not mask_powerspec:
            yedges = im.t-t0
            xedges = period
            img = topax.pcolormesh(xedges, yedges, power, cmap=plt.cm.Greys)
        topax.set_xlabel('Period (glomeruli)', labelpad=20)
        # ax1.scatter(self.tif.t, periods, color='black', alpha=.8, lw=0, s=5, clip_on=True)
        topax.set_xlim(2, 32)
        topax.set_ylim(tlim)
        topax.invert_yaxis()
        topax.set_xscale('log', basex=2)
        topax.xaxis.set_major_formatter(ScalarFormatter())
        topax.set_xticks([2, 8, 32])

        # # Draw colorbar
        # cb = plt.colorbar(img, ax=topax, aspect=4, shrink=.15, pad=0.05, anchor=(0, .9),
        #                   use_gridspec=False)
        # # cb.set_label(r'$\Delta$F/F')
        # cb.set_clim([0, 1])
        # cb.set_ticks([0, 1])

        # Plot average power spectrum
        botax = plt.subplot(gs[irow+1, icol])
        ph.adjust_spines(botax, ['left'])
        idx1 = (im.t-t0 >= tlim[0]) & (im.t-t0 < tlim[1])
        idx2 = (period<=32)
        h = power[:, idx2]
        h = h[idx1].mean(axis=0)
        ymax = fc.round(h.max()*10, 1, 'up')/10.
        botax.set_ylim(0, ymax)
        botax.set_yticks([0, ymax])
        botax.plot(period[period<=32], h, c='black', lw=1)
        botax.set_xlim(2, 32)
        botax.set_xscale('log', basex=2)
        ph.adjust_spines(botax, ['left'])

    # Plot overlayed circ plot
    if overlay:
        icol += 1

        # Setup axis
        ax = plt.subplot(gs[irow, icol])
        if unwrap:
            xlim = None
            xlabel = r'Accumulated Rotation ($\pi$ rad)'
        else:
            xlim = (-1, 1)
            xlabel = r'Rotation ($\pi$ rad)'

        # if show_axes:
        #     ax.set_xlabel(xlabel, labelpad=20)
        #     ph.adjust_spines(ax, ['top'], xlim=xlim, xticks=(-1, 0, 1), ylim=tlim)
        # else:
        #     ph.adjust_spines(ax, [], xlim=xlim, xticks=[], ylim=tlim)
        ph.adjust_spines(ax, [], xlim=xlim, xticks=[], ylim=tlim)
        # ax.axvline(0, c='black', ls='--', alpha=0.4)

        intmin = 0.
        intmax = 0.
        for i, label in enumerate(overlay):
            t, a = self.get_signal(label)
            # a = copy.copy(a)
            color = 'grey'
            if 'head' in label:
                if dark_gain is None:
                    dark_gain = self.get_dark_gain((t0+tlim[0], t0+tlim[1]), 'pb.c1.dphase')
                print 'Dark gain: %.2f' %dark_gain
                head = -self.get_headint(dark_gain)
                head = fc.unwrap(head)
                head = self.subsample(head)
                t = self.ims[0].t
                if unwrap:

                    idx0 = np.where(t-t0>=tlim[0])[0][0]
                    head = head - head[idx0]
                a = fc.wrap(head + phase_offsets[i%len(phase_offsets)])
                ph.circplot(head/180., t-t0, c=color, alpha=1, lw=2, period=2, zorder=2)
            elif 'phase' in label:
                if unwrap and fc.contains(['abf.head'], overlay):
                    phase = fc.unwrap(a)
                    if cancel_phase:
                        idx_offset = np.where(t>=toffset)[0][0]
                        offset = phase[idx_offset] - head[idx_offset]
                        phase = phase - offset
                else:
                    phase = copy.copy(a)
                    if cancel_phase:
                        offset = self.get_offset(toffset, label)
                        # offset = 230
                        phase = fc.wrap(phase - offset)

                _, c, _ = self.get_objects(label)
                color = colours[c.celltype] #if 'eb' in label else 'grey'
                if two_location:
                    if 'pb' in label:
                        # color = ph.blue
                        color = (0.2,1,1)
                    elif 'fb' in label:
                        color = ph.orange
                    elif 'eb' in label:
                        # color = ph.green
                        color = (0,0.3,0.6)
                a = fc.wrap(phase + phase_offsets[i%len(phase_offsets)])
                ph.circplot(phase/180., t-t0, c=color, alpha=1, lw=1, period=2, zorder=2)
            # if 'xstim' == label:
            #     t = self.abf.t
            #     a = self.abf.xstim
            #     ph.circplot(a/180., t-t0, c=color, alpha=1, lw=2, period=2, zorder=2)
            if 'xstim_old' == label:
                t = self.abf.t
                a = self.abf.xstim_old
                ph.circplot(a/180., t-t0, c=color, alpha=1, lw=2, period=2, zorder=2)
            else:
                # b = fc.circ_moving_average(a, n=3)
                ph.circplot(a/180., t-t0, c=color, alpha=1, lw=2, period=2, zorder=2)
            a = a / 180.
            a_tlim = a[(t-t0 >= tlim[0]) & (t-t0 < tlim[1])]
            intmin = min(intmin, a_tlim.min())
            intmax = max(intmax, a_tlim.max())

        if unwrap:
            intmin = np.floor(intmin)
            intmax = np.ceil(intmax)
            # intmax = 3
            ax.set_xlim(intmin, intmax)
            xticks = np.arange(intmin, intmax+1, 1)
            ax.set_xticks(xticks)
        if highlight == 'arena':
            ax.fill_betweenx(tabf, x1=-1, x2=-self.abf.arena_edge/180., color='black', alpha=0.1, zorder=1)
            ax.fill_betweenx(tabf, x1=self.abf.arena_edge/180., x2=1, color='black', alpha=0.1, zorder=1)
        elif highlight == 'all':
            if unwrap:
                ax.fill_betweenx(tabf, x1=intmin, x2=intmax, color='black', alpha=0.1, zorder=1)
            else:
                ax.fill_betweenx(tabf, x1=-1, x2=1, color='black', alpha=0.1, zorder=1)
        # dotline1 = np.zeros(len(t)) + 0.5
        # dotline2 = np.zeros(len(t)) - 0.5
        # dotline3 = np.zeros(len(t)) + 0.25
        # dotline4 = np.zeros(len(t)) - 0.25
        # ph.circplot(dotline1, t-t0, c=color, alpha=1, lw=0.2, period=2, zorder=2, ls='--')
        # ph.circplot(dotline2, t-t0, c=color, alpha=1, lw=0.2, period=2, zorder=2, ls='--')
        # ph.circplot(dotline3, t-t0, c=color, alpha=1, lw=0.2, period=2, zorder=2, ls='--')
        # ph.circplot(dotline4, t-t0, c=color, alpha=1, lw=0.2, period=2, zorder=2, ls='--')
        ax.axvline(ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
        axs.append(ax)
        ax.invert_yaxis()

    # Plot channels individually
    for i, label in enumerate(channels):
        icol += 1
        if fc.contains(['head_opposite'], [label]):
            t, signal = self.get_signal(label[:-9])
        else:
            t, signal = self.get_signal(label)
        ax = plt.subplot(gs[irow, icol])
        if fc.contains(['head_opposite'], [label]):
            ax.set_xlim(-180, 180)
            ax.set_xticks([-180, 0, 180])
            ph.circplot(-signal, t, circ='x', c='black', lw=1)
        elif fc.contains(['head', 'side', 'forw', 'xstim'], [label]):
            ax.set_xlim(-180, 180)
            ax.set_xticks([])
            ax.set_xticks([-180,-90,0,90,180], minor=True)
            ax.grid(which='minor', alpha=.3)
            ax.axvline(-180, c='black', ls='-', alpha=.3)
            ax.axvline(180, c='black', ls='-', alpha=.3)
            ph.circplot(signal, t, circ='x', c='black', lw=1)
        elif fc.contains(['xbar'], [label]):
            ax.set_xlim(-180, 180)
            ax.set_xticks([])
            ax.set_xticks([-180,-90,0,90,180], minor=True)
            ax.grid(which='minor', alpha=.3)
            ax.axvline(-180, c='black', ls='-', alpha=.3)
            ax.axvline(-90, c=ph.grey5, ls='-', alpha=.3)
            ax.axvline(-0, c=ph.grey5, ls='-', alpha=.3)
            ax.axvline(90, c=ph.grey5, ls='-', alpha=.3)
            ax.axvline(180, c='black', ls='-', alpha=.3)
            ph.circplot(signal, t, circ='x', c='black', lw=1)
        elif fc.contains(['dhead'], [label]):
            ax.set_xlim(-30, 30)
            ax.plot(signal, t-t0, c='black', lw=1)
        elif fc.contains(['speed'], [label]):
            ax.set_xlim(0, 5)
            ax.plot(signal, t-t0, c='black', lw=1)
        elif fc.contains(['glo_peak_fluo'], [label]):
            ax.set_xlim(0, 1)
            ax.plot(signal, t-t0, c='black', lw=1)
        elif fc.contains(['mean_al', 'mean_an', 'mean_p', 'apm_pl', 'apm_pn'], [label]):
            if self.abf.bhtype == 'walk':
                _, speed = self.get_signal('speed')
                speed = speed * .12
                ax.plot(speed, t-t0, c='black', lw=0.3)
            _, puff = self.get_signal('puff')
            puff[puff > 0.9] = 1
            puff[puff <= 0.9] = -1
            ax.set_xlim(0, 1.2)
            ax.set_xticks([0, 1])
            ax.plot(signal, t-t0, c=colours[self.ims[0].c1.celltype], lw=1)
            ax.plot(puff, t-t0, c='red', lw=.5)
            ax.set_xticks([0,0.5,1], minor=True)
            ax.grid(which='minor', alpha=.3)
        else:
            # idx = (t-t0 >= tlim[0]) & (t-t0 < tlim[1])
            xmin = np.floor(signal.min())
            xmax = np.ceil(signal.max())
            ax.set_xlim(xmin, xmax)
            ax.set_xticks([xmin, xmax])
            ax.plot(signal, t-t0, c='black', lw=1)

        ph.adjust_spines(ax, ['bottom'], ylim=tlim)
        # if show_axes:
        #     ax.set_xlabel(label.title(), labelpad=20)
        ax.invert_yaxis()
        axs.append(ax)

    for i, label in enumerate(channels_narrow):
        icol += 1
        if fc.contains(['head_opposite'], [label]):
            t, signal = self.get_signal(label[:-9])
        else:
            t, signal = self.get_signal(label)
        ax = plt.subplot(gs[irow, icol])
        if fc.contains(['head_opposite'], [label]):
            ax.set_xlim(-180, 180)
            ax.set_xticks([-180, 0, 180])
            ph.circplot(-signal, t, circ='x', c='black', lw=1)
        elif fc.contains(['head', 'side', 'forw'], [label]):
            ax.set_xlim(-180, 180)
            ax.set_xticks([])
            ax.set_xticks([-180,0,180], minor=True)
            ax.grid(which='minor', alpha=.3)
            # ax.axvline(c='black', ls='--', alpha=.3)
            ph.circplot(signal, t, circ='x', c='black', lw=1)
        elif fc.contains(['dhead'], [label]):
            ax.set_xlim(-30, 30)
            ax.plot(signal, t-t0, c='black', lw=1)
        elif fc.contains(['speed', 'dforw'], [label]):
            ax.set_xlim(0, 5)
            ax.plot(signal, t-t0, c='black', lw=1)
        elif fc.contains(['glo_peak_fluo'], [label]):
            ax.set_xlim(0, 1)
            ax.plot(signal, t-t0, c='black', lw=1)
        elif fc.contains(['stimid'], [label]):
            ax.set_xlim(-10, 10)
            ax.plot(signal, t-t0, c='black', lw=1)
            ax.set_xticks([-5, 5])
            ax.set_xticks([-5,0,5], minor=True)
            ax.grid(which='minor', alpha=.3)
        elif fc.contains(['abf.lwa', 'abf.rwa'], [label]):
            ax.set_xlim(-5, 90)
            ax.plot(signal, t-t0, c='black', lw=1)
            ax.set_xticks([40])
            ax.set_xticks([0,40,80], minor=True)
            ax.grid(which='minor', alpha=.3)
        elif fc.contains(['xstim'], [label]):
                # ax.set_xlim(-180, 180)
                # ax.set_xticks([-180, 0, 180])
                # ph.circplot(signal, t, circ='x', c='black', lw=1)
            ax.set_xlim(0, 10)
            ax.set_xticks([0, 10])
            ph.circplot(signal, t, circ='x', c='black', lw=1)
        else:
            # idx = (t-t0 >= tlim[0]) & (t-t0 < tlim[1])
            xmin = np.floor(signal.min())
            xmax = np.ceil(signal.max())
            ax.set_xlim(xmin, xmax)
            ax.set_xticks([xmin, xmax])
            ax.plot(signal, t-t0, c='black', lw=1)

        ax.axvline(ax.get_xlim()[0], c=ph.grey5, lw=1, alpha=0.5)
        ax.axvline(ax.get_xlim()[-1], c=ph.grey5, lw=1, alpha=0.5)
        ph.adjust_spines(ax, ['bottom'], ylim=tlim)
        # if show_axes:
        #     ax.set_xlabel(label.title(), labelpad=20)
        ax.invert_yaxis()
        axs.append(ax)

    if trig_t:
        lines.append(0)

    if barjump_trig:
        bj_trig_ts = self.abf.t[:-1][np.abs(np.diff(self.abf.xstim_unwrap)) > 50]
        for ax in axs:
            for bj_trig_t in bj_trig_ts:
                ax.axhline(bj_trig_t-t0, c='red', ls='--', lw=2)

    for ax in axs:
        ax.patch.set_facecolor('white')
        ax.grid(which='major', alpha=0)
        ax.grid(which='minor', alpha=0)
        for line in lines:
            ax.axhline(line-t0, c='black', ls='--', lw=1)

    if save:
        figname = '%s_%i-%is' % (self.basename, tlim[0], tlim[1])
        if self.basename[0] == '/':
            figname = '.' + figname
        ph.save(figname.split('/')[-1] + suffix, exts=['png'])
        # plt.savefig(figname + '.' + filetype, bbox_inches='tight')


def plot_truncated(self, tlim=None, powerspec=False, imaging=['fb.c1.an'], phaseoverlay=False,
         overlay=['c1.phase', 'xstim'], channels=[], channels_narrow=[], mask_powerspec=False, vmin=None,
         vmax=None, toffset=None, offset=None, cmap=None, lines=[], _cmaps=[plt.cm.Blues],
         dark_gain=1, gs=None, row=0, trig_t=0, unwrap=False, cancel_phase=True,
         show_axes=True, sf=1, reset_t=False, suffix='', save=False, ar=2, size=None,
         filetype='png', highlight='arena', colorbar=0, fig_num=1, title=[], barjump_trig=False, two_location=False,
                   xedges=None, colors_overlay=['black'], phase_offset=0):
    """
    Most versatile plotting functions. Used to get explore datasets.
    :param self: GCaMP_ABF recording.
    :param tlim: Time window.
    :param powerspec: True/False. True to plot power spectrum.
    :param imaging: List of labels, starting with a. eg. 'an' = dF/F normalized.
    :param phaseoverlay: True/False. True to overlay phase on the imaging dataset.
    :param overlay: List of labels to plot together. Typically phase and bar or ball.
    :param channels: List of labels to plot under separate columns. Must be abf channels. eg. 'forw' or 'stimid'
    :param mask_powerspec: Do not plot power spectrum, but leave space for it.
    :param vmin: Minimum value in imaging colormap lookup table.
    :param vmax: Maximum value in imaging colormap lookup table.
    :param toffset: Time at which to cancel the offset between phase and bar. Can be an interval, in which case
    the circular mean offset will be computed.
    :param offset: Amount by which to offset the phase. If not defined (ie. None), will be computed using toffset.
    :param cmap: Colormap to plot imaging.
    :param lines: Times to plot horizontal lines across all columns.
    :param dark_gain: Gain with which to multiply the fly's heading. If None, it is computed as the slope between
    the phase and ball velocities in the tlim time window.
    :param gs: Used under plot_trials.
    :param row: Used under plot_trials.
    :param trig_t: Used under plot_trials.
    :param unwrap: True/False. True to unwrap circular data.
    :param show_axes: True to show_axes. Used under plot_trials.
    :param sf: Scale factor for figure size.
    :param reset_t: Sets the x-axis to start at 0.
    :param suffix: String appended to figure name.
    :param save: True to save figure.
    :param ar: Aspect ratio.
    :param filetype: eg.'png', 'svg', 'pdf', etc.
    :return:
    """

    ph.set_fontsize(20)

    if highlight == 'arena' and unwrap:
        highlight = 'all'

    color_dict = {
        plt.cm.Blues: ph.blue,
        plt.cm.Greens: ph.dgreen,
        plt.cm.Purples: ph.dpurple,
        plt.cm.Oranges: ph.orange,
        plt.cm.CMRmap: ph.dpurple,
        plt.cm.jet: ph.blue
    }

    if cmap is None: cmap = cmaps[self.ims[0].c1.celltype]
    if not type(cmap) is tuple: cmap = [cmap]
    if len(cmap) == 1: cmap = list(cmap) + [plt.cm.Reds]
    if not tlim: tlim = [self.ims[0].t[0], self.ims[0].t[-1]]
    if type(tlim) is int:
        trial_times = self.abf.t[self.abf.stimid == tlim]
        tlim = (trial_times[0], trial_times[-1])
    if toffset is None: toffset = tlim[0]

    if reset_t:
        t0 = tlim[0]
        tlim = [0, tlim[1]-tlim[0]]
    else:
        t0 = trig_t

    tabf = self.abf.t - t0
    tabf_orig = self.abf.t_orig - t0

    # get the starting and ending inds for imaging and behavior
    (t, _), _ = self.get_signal(imaging[0])
    inds_image = np.arange(np.where(t <= tlim[0])[0][-1], np.where(t >= tlim[-1])[0][0])
    inds_abf = np.arange(np.where(tabf <= tlim[0])[0][-1], np.where(tabf >= tlim[-1])[0][0])
    tabf = tabf[inds_abf]

    # Setup Figure if not triggered
    if not gs:
        ncols = np.array([len(imaging), powerspec, len(channels), len(channels_narrow), len(overlay) > 0]).sum()
        def wr_fcn(imaging_label):
            if 'eb' in imaging_label:
                return 1
            elif 'pb' in imaging_label:
                return 1.5
            elif 'fb' in imaging_label:
                return 1
            elif 'sg' in imaging_label:
                return 2
        wr = map(wr_fcn, imaging) + [1]*powerspec + [1]*(len(overlay)>0) + [1]*len(channels) +[0.5]*len(channels_narrow) # width ratios
        hr = [12, 1]
        if size is None:
            size = (10*sf*np.sqrt(ar), 10*sf/np.sqrt(ar))
        fig = plt.figure(fig_num, size)
        # plt.suptitle(self.folder, size=10)
        rcParams['xtick.direction'] = 'out'
        rcParams['ytick.direction'] = 'out'
        gs = gridspec.GridSpec(2, ncols, width_ratios=wr, height_ratios=hr)
        gs.update(wspace=0.02, hspace=.03)
    axs = []

    if show_axes:
        # stim = 'BAR' if self.abf.ystim.mean() < 5 else 'DARK'
        # temp = int(np.round(self.abf.temp.mean()))
        # print '%s, %iC' %(stim, temp)
        if title:
            fig.suptitle(title, y=.91, fontsize=10)
        # else:
        #     print '%iC' % temp

    # Plot Imaging
    if trig_t > 0:
        red_pulse = self.ims[0].c2.an[(self.ims[0].t>0) & (self.ims[0].t<.3)].mean(axis=0)
        pipette = np.argmax(red_pulse) + .5

    irow = row * 2
    for icol, label in enumerate(imaging):
        # Get imtype (pb or eb), channel and gc
        imlabel, clabel, alabel = self.get_labels(label)
        im, c, _ = self.get_objects(label)
        # if xedges is None:
        (t, xedges), gc = self.get_signal(label)
        # else:
        #     (t, _), gc = self.get_signal(label)
        # Setup axis
        axgc = plt.subplot(gs[irow, icol])
        if trig_t > 0:
            axgc.arrow(pipette, -.4, 0, .2, head_width=0.5, head_length=0.1, fc='r', ec='r', lw=2)

        # if 0:
        if show_axes:
            # axgc.set_xlabel(imlabel + ' ' + c.celltype, labelpad=20)
            if imlabel == 'pb':
                axgc.set_xticks(np.array([0, 9, 18]))
            elif imlabel == 'eb':
                xticks = np.array([0, 8, 16])
                axgc.set_xticks(xticks)
                axgc.set_xticklabels(xticks)
            spines = []
            if icol == 0:
                spines.append('left')
            else:
                spines.append('bottom')
                # axgc.set_ylabel('Time (s)', rotation='horizontal', labelpad=40)
        else:
            spines = []
        ph.adjust_spines(axgc, spines, ylim=tlim, xlim=[xedges[0], xedges[-1]])

        # Plot gc
        gc[np.isnan(gc)] = 0
        cm = _cmaps[icol%len(_cmaps)]
        #
        # img = plt.pcolormesh(xedges, t[inds_image]-t0, gc[inds_image], cmap=cmaps[c.celltype])
        if 'fb' in label:
            img = plt.pcolormesh(xedges, t[inds_image] - t0, np.roll(gc[inds_image], 1, axis=1), cmap=cm)
        else:
            img = plt.pcolormesh(xedges, t[inds_image] - t0, gc[inds_image], cmap=cm)
        #
        if not vmin is None: img.set_clim(vmin=vmin[icol])
        if not vmax is None: img.set_clim(vmax=vmax[icol])
        # Overlay phase
        if phaseoverlay:
            phase = c.phase * c.period / 360  +.5 # scale phase to periodicity of the pb or eb, +.5 is to place it in
            # the middle of the glomerulus (since the glomerulus number maps to the left edges of the pcolormesh grid)
            phase[phase < 0] = phase[phase < 0] + c.period
            ph.circplot(phase, im.tc-t0, c=(.1, .1, .1), alpha=.8, period=c.period, zorder=11, lw=2)
        axgc.invert_yaxis()
        axs.append(axgc)

        # Draw colorbar
        if colorbar == 1:
            cbaxes = fig.add_axes([0.17, 0.17, 0.1, 0.01])
            cb = plt.colorbar(ax=axgc, cax=cbaxes,
                              use_gridspec=False, orientation='horizontal')
            cb.ax.yaxis.set_ticks_position('left')
            # cb.set_label(r'$\Delta$F/F')
            if vmax:
                cmax = vmax[icol]
            else:
                idx = (t>=tlim[0]) & (t<tlim[1])
                cmax = np.floor(gc[idx].max())
                if cmax==0:
                    cmax = np.floor(gc[idx].max()*10)/10.
            cb.set_ticks([0, cmax])

    if powerspec:
        icol += 1
        topax = plt.subplot(gs[irow, icol])
        im = self.ims[0]
        power, period, phase = fc.powerspec(im.c1.an, mask=False, cmap=plt.cm.Greys, show=False, n=10)


        # Plot power spectrum over time
        ph.adjust_spines(topax, ['top'])
        # topax.set_ylabel('Time (s)', rotation='horizontal', ha='right', labelpad=20)
        if not mask_powerspec:
            yedges = im.t-t0
            xedges = period
            img = topax.pcolormesh(xedges, yedges, power, cmap=plt.cm.Greys)
        topax.set_xlabel('Period (glomeruli)', labelpad=20)
        # ax1.scatter(self.tif.t, periods, color='black', alpha=.8, lw=0, s=5, clip_on=True)
        topax.set_xlim(2, 32)
        topax.set_ylim(tlim)
        topax.invert_yaxis()
        topax.set_xscale('log', basex=2)
        topax.xaxis.set_major_formatter(ScalarFormatter())
        topax.set_xticks([2, 8, 32])

        # # Draw colorbar
        # cb = plt.colorbar(img, ax=topax, aspect=4, shrink=.15, pad=0.05, anchor=(0, .9),
        #                   use_gridspec=False)
        # # cb.set_label(r'$\Delta$F/F')
        # cb.set_clim([0, 1])
        # cb.set_ticks([0, 1])

        # Plot average power spectrum
        botax = plt.subplot(gs[irow+1, icol])
        ph.adjust_spines(botax, ['left'])
        idx1 = (im.t-t0 >= tlim[0]) & (im.t-t0 < tlim[1])
        idx2 = (period<=32)
        h = power[:, idx2]
        h = h[idx1].mean(axis=0)
        ymax = fc.round(h.max()*10, 1, 'up')/10.
        botax.set_ylim(0, ymax)
        botax.set_yticks([0, ymax])
        botax.plot(period[period<=32], h, c='black', lw=1)
        botax.set_xlim(2, 32)
        botax.set_xscale('log', basex=2)
        ph.adjust_spines(botax, ['left'])

    # Plot overlayed circ plot
    if overlay:
        icol += 1

        # Setup axis
        ax = plt.subplot(gs[irow, icol])
        if unwrap:
            xlim = None
            xlabel = r'Accumulated Rotation ($\pi$ rad)'
        else:
            xlim = (-1, 1)
            xlabel = r'Rotation ($\pi$ rad)'

        # if show_axes:
        #     ax.set_xlabel(xlabel, labelpad=20)
        #     ph.adjust_spines(ax, ['top'], xlim=xlim, xticks=(-1, 0, 1), ylim=tlim)
        # else:
        #     ph.adjust_spines(ax, [], xlim=xlim, xticks=[], ylim=tlim)
        ph.adjust_spines(ax, [], xlim=xlim, xticks=[], ylim=tlim)
        # ax.axvline(0, c='black', ls='--', alpha=0.4)

        intmin = 0.
        intmax = 0.
        for i, label in enumerate(overlay):
            t, a = self.get_signal(label)
            # a = copy.copy(a)
            color = colors_overlay[i%len(colors_overlay)]
            if 'head' in label:
                if dark_gain is None:
                    dark_gain = self.get_dark_gain((t0+tlim[0], t0+tlim[1]), 'pb.c1.dphase')
                print 'Dark gain: %.2f' %dark_gain
                head = -self.get_headint(dark_gain)
                # head = fc.unwrap(head)
                # head = self.subsample(head)
                # t = self.ims[0].t
                t = self.abf.t
                if unwrap:
                    idx0 = np.where(t-t0>=tlim[0])[0][0]
                    head = head - head[idx0]
                a = head
                ph.circplot(head/180., t-t0, c=color, alpha=0.7, lw=0.5, period=2, zorder=2)
            elif 'phase' in label:
                if unwrap and fc.contains(['abf.head'], overlay):
                    phase = fc.unwrap(a)
                    if cancel_phase:
                        idx_offset = np.where(t>=toffset)[0][0]
                        offset = phase[idx_offset] - head[idx_offset]
                        phase = phase - offset
                else:
                    phase = copy.copy(a)
                    if cancel_phase:
                        # offset = self.get_offset(toffset, label)
                        # offset = 270  # PFN-EPG
                        # offset = 240  # PFN3--83H07
                        # offset = -45
                        # print phase_offset
                        # offset = phase_offset
                        # if 'pb' in label:  # PFN
                        #     offset = 0  # PFN
                        #     # offset = 240  # EPG
                        # if 'fb' in label:  # PFN
                        #     offset = 180  # PFN
                        #     # offset = 240  # EPG
                        # if 'eb' in label:  # PFN-EPG
                        #     offset = 180  # PFN-EPG
                        #     # offset = 240  # EPG
                        xbar = self.subsample('xbar')
                        offset = fc.circmean(phase - xbar)
                        phase = fc.wrap(phase - offset)

                _, c, _ = self.get_objects(label)
                # color = colours[c.celltype] #if 'eb' in label else 'grey'
                if two_location:
                    if 'pb' in label:
                        # color = ph.blue
                        # color = (0,0.3,0.6) # EPG
                        color = (1,.5,0)  # PFN
                    elif 'fb' in label:
                        color = ph.orange
                        # color = (1,.7,.3)    # PFN
                    elif 'eb' in label:
                        color = ph.green
                        # color = (0.2,1,1) # EPG
                a = phase
                ph.circplot(phase/180., t-t0, c=color, alpha=.5, lw=2, period=2, zorder=2)
            elif 'xstim_old' == label:
                t = self.abf.t[inds_abf]
                a = self.abf.xstim_old[inds_abf]
                ph.circplot(a/180., t-t0, c=color, alpha=1, lw=1, period=2, zorder=2)
            else:
                # b = fc.circ_moving_average(a, n=3)
                ph.circplot(a/180., t-t0, c=color, alpha=.5, lw=2, period=2, zorder=1)
            a = a / 180.
            a_tlim = a[(t-t0 >= tlim[0]) & (t-t0 < tlim[1])]
            intmin = min(intmin, a_tlim.min())
            intmax = max(intmax, a_tlim.max())

        if unwrap:
            intmin = np.floor(intmin)
            intmax = np.ceil(intmax)
            # intmax = 3
            ax.set_xlim(intmin, intmax)
            xticks = np.arange(intmin, intmax+1, 1)
            ax.set_xticks(xticks)
        # highlight = 'arena'
        if highlight == 'arena':
            ax.fill_betweenx(tabf, x1=-1, x2=-self.abf.arena_edge/180., color='black', alpha=0.1, zorder=1)
            ax.fill_betweenx(tabf, x1=self.abf.arena_edge/180., x2=1, color='black', alpha=0.1, zorder=1)
        elif highlight == 'all':
            if unwrap:
                ax.fill_betweenx(tabf, x1=intmin, x2=intmax, color='black', alpha=0.1, zorder=1)
            else:
                ax.fill_betweenx(tabf, x1=-1, x2=1, color='black', alpha=0.1, zorder=1)
        # dotline1 = np.zeros(len(t)) + 0.5
        # dotline2 = np.zeros(len(t)) - 0.5
        # dotline3 = np.zeros(len(t)) + 0.25
        # dotline4 = np.zeros(len(t)) - 0.25
        # ph.circplot(dotline1, t-t0, c=color, alpha=1, lw=0.2, period=2, zorder=2, ls='--')
        # ph.circplot(dotline2, t-t0, c=color, alpha=1, lw=0.2, period=2, zorder=2, ls='--')
        # ph.circplot(dotline3, t-t0, c=color, alpha=1, lw=0.2, period=2, zorder=2, ls='--')
        # ph.circplot(dotline4, t-t0, c=color, alpha=1, lw=0.2, period=2, zorder=2, ls='--')
        ax.axvline(ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
        axs.append(ax)
        ax.invert_yaxis()

    # Plot channels individually
    for i, label in enumerate(channels):
        icol += 1
        if fc.contains(['head_opposite'], [label]):
            t, signal = self.get_signal(label[:-9])
        else:
            t, signal = self.get_signal(label)
        if (1./np.mean(np.diff(t))) > 40:
            t = t[inds_abf]
            signal = signal[[inds_abf]]
        else:
            t = t[inds_image]
            signal = signal[[inds_image]]
        ax = plt.subplot(gs[irow, icol])
        if fc.contains(['head_opposite'], [label]):
            ax.set_xlim(-180, 180)
            ax.set_xticks([-180, 0, 180])
            ph.circplot(-signal, t, circ='x', c='black', lw=1)
        elif fc.contains(['head', 'side', 'forw', 'xstim'], [label]):
            if self.abf.bhtype == 'flight':
                ax.set_xlim(-0.1, 10.1)
                ax.set_xticks([0, 5, 10], minor=True)
            else:
                ax.set_xlim(-180, 180)
                ax.set_xticks([-180, -90, 0, 90, 180], minor=True)
            ax.set_xticks([])
            ax.grid(which='minor', alpha=.3)
            if self.abf.bhtype == 'walk':
                ax.axvline(-180, c='black', ls='-', alpha=.3)
                ax.axvline(180, c='black', ls='-', alpha=.3)
            ph.circplot(signal, t, circ='x', c='black', lw=1)
        elif fc.contains(['dhead'], [label]):
            ax.set_xlim(-30, 30)
            ax.plot(signal, t-t0, c='black', lw=1)
        elif fc.contains(['speed'], [label]):
            ax.set_xlim(0, 5)
            ax.plot(signal, t-t0, c='black', lw=1)
        elif fc.contains(['glo_peak_fluo'], [label]):
            ax.set_xlim(0, 1)
            ax.plot(signal, t-t0, c='black', lw=1)
        elif fc.contains(['mean_al', 'mean_an', 'mean_p', 'apm_pl', 'apm_pn'], [label]):
            if self.abf.bhtype == 'walk':
                # _, speed = self.get_signal('speed')
                speed = self.subsample('dforw')
                speed = speed * .1
                speed = speed[[inds_image]]
                ax.plot(speed, t-t0, c='blue', lw=0.2)
            _, puff = self.get_signal('puff')
            puff[puff > 0.9] = 1
            puff[puff <= 0.9] = -1
            puff = puff[inds_image]
            ax.set_xlim(0, 0.8)
            ax.plot(signal, t-t0, c='black', lw=1)
            ax.plot(puff, t-t0, c='red', lw=.5)

            # if self.abf.bhtype == 'walk':
            #     _, speed = self.get_signal('dforw')
            #     speed = speed * .1
            #     speed = speed[[inds_image]]
            #     ax.plot(speed, t-t0, c=blue, lw=0.3)
            # _, puff = self.get_signal('puff')
            # puff[puff > 1] = 1
            # puff[puff <= 1] = -1
            # puff = puff[inds_image]
            # ax.set_xlim(0, .7)
            # ax.set_xticks([0, .5])
            # ax.plot(signal, t-t0, c=colours[self.ims[0].c1.celltype], lw=1)
            # ax.plot(puff, t-t0, c='red', lw=.5)
            # ax.set_xticks([0,0.5], minor=True)
            # ax.grid(which='minor', alpha=.3)
        elif fc.contains(['stimid', 'stimid_raw'], [label]):
            if self.abf.bhtype == 'walk':
                ax.set_xlim(-0.1, 1.1)
                ax.set_xticks([0, 1])
            if self.abf.bhtype == 'flight':
                ax.set_xlim(-10, 0)
            ax.plot(signal, t-t0, c='black', lw=1)
        else:
            # idx = (t-t0 >= tlim[0]) & (t-t0 < tlim[1])
            xmin = np.floor(signal.min())
            xmax = np.ceil(signal.max())
            ax.set_xlim(xmin, xmax)
            ax.set_xticks([xmin, xmax])
            ax.plot(signal, t-t0, c='black', lw=1)

        ph.adjust_spines(ax, ['bottom'], ylim=tlim)
        # if show_axes:
        #     ax.set_xlabel(label.title(), labelpad=20)
        ax.invert_yaxis()
        axs.append(ax)

    for i, label in enumerate(channels_narrow):
        icol += 1
        if fc.contains(['head_opposite'], [label]):
            t, signal = self.get_signal(label[:-9])
        else:
            t, signal = self.get_signal(label)
        ax = plt.subplot(gs[irow, icol])
        if fc.contains(['head_opposite'], [label]):
            ax.set_xlim(-180, 180)
            ax.set_xticks([-180, 0, 180])
            ph.circplot(-signal, t, circ='x', c='black', lw=1)
        elif fc.contains(['head', 'side', 'forw'], [label]):
            ax.set_xlim(-180, 180)
            ax.set_xticks([])
            ax.set_xticks([-180,0,180], minor=True)
            ax.grid(which='minor', alpha=.3)
            ax.axvline(-180, c='black', lw=1)
            ax.axvline(180, c='black', lw=1)
            ph.circplot(signal, t, circ='x', c='black', lw=1)
        elif fc.contains(['dhead'], [label]):
            ax.set_xlim(-30, 30)
            ax.plot(signal, t-t0, c='black', lw=1)
        elif fc.contains(['speed'], [label]):
            ax.set_xlim(0, 5)
            ax.plot(signal, t-t0, c='black', lw=1)
        elif fc.contains(['glo_peak_fluo'], [label]):
            ax.set_xlim(0, 1)
            ax.plot(signal, t-t0, c='black', lw=1)
        elif fc.contains(['stimid', 'stimid_raw'], [label]):
            if self.abf.bhtype == 'walk':
                ax.set_xlim(-0.1, 1.1)
                ax.set_xticks([0, 1])
            if self.abf.bhtype == 'flight':
                ax.set_xlim(-10, 10)
                ax.set_xticks([10, 10])
            ax.plot(signal, t-t0, c='black', lw=1)
        elif fc.contains(['xstim'], [label]):
            ax.set_xlim(0, 10)
            ax.set_xticks([0, 10])
            ph.circplot(signal, t, circ='x', c='black', lw=1)
        elif fc.contains(['abf.lwa', 'abf.rwa'], [label]):
            ax.set_xlim(-5, 90)
            ax.plot(signal, t-t0, c='black', lw=1)
            ax.set_xticks([40])
            ax.set_xticks([0,40,80], minor=True)
            ax.grid(which='minor', alpha=.3)
        else:
            # idx = (t-t0 >= tlim[0]) & (t-t0 < tlim[1])
            xmin = np.floor(signal.min())
            xmax = np.ceil(signal.max())
            ax.set_xlim(xmin, xmax)
            ax.set_xticks([xmin, xmax])
            ax.plot(signal, t-t0, c='black', lw=1)

        ph.adjust_spines(ax, ['bottom'], ylim=tlim)
        # if show_axes:
        #     ax.set_xlabel(label.title(), labelpad=20)
        ax.invert_yaxis()
        axs.append(ax)

    if trig_t:
        lines.append(0)

    if barjump_trig:
        bj_trig_ts = self.abf.t[:-1][np.abs(np.diff(self.abf.xstim_unwrap)) > 50]
        for ax in axs:
            for bj_trig_t in bj_trig_ts:
                ax.axhline(bj_trig_t-t0, c='red', ls='--', lw=2)

    for ax in axs:
        ax.patch.set_facecolor('white')
        ax.grid(which='major', alpha=0)
        ax.grid(which='minor', alpha=0)
        for line in lines:
            ax.axhline(line-t0, c='black', ls='--', lw=1)

    if save:
        figname = '%s_%i-%is' % (self.basename, tlim[0], tlim[1])
        if self.basename[0] == '/':
            figname = '.' + figname
        ph.save(figname.split('/')[-1] + suffix, exts=['png'])
        # plt.savefig(figname + '.' + filetype, bbox_inches='tight')

def plot_2c(rec, axs=[], fig_size=(15, 3), tlim=[1,200], gs_width_ratios=[50,1], plot_bar=True,
                cmaps=[plt.cm.Blues, plt.cm.Oranges], vm=[0,1.6], colors=[ph.blue, ph.dorange], bar_length=10, fs=7, auto_offset=True,
                ms=1.5, vlines=[]):
    ph.set_fontsize(fs)
    if not len(axs):
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(3, 2, width_ratios=gs_width_ratios)
        axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[2, 0])]

    idx0 = np.where(rec.ims[0].t >= tlim[0])[0][0]
    idx1 = np.where(rec.ims[1].t <= tlim[1])[0][-1]
    idx0_beh = np.where(rec.abf.t >= tlim[0])[0][0]
    idx1_beh = np.where(rec.abf.t <= tlim[1])[0][-1]
    xlim = tlim

    for i_ in range(2):
        ax = axs[i_*2]
        ph.adjust_spines(ax, [], ylim=[rec.ims[1-i_].c1.xedges[0], rec.ims[1-i_].c1.xedges[-1]], xlim=xlim)
        ax.pcolormesh(rec.ims[1-i_].t[idx0:idx1], rec.ims[1-i_].c1.xedges, rec.ims[1-i_].c1.al[idx0:idx1].T, cmap=cmaps[i_],
                      vmin=vm[0], vmax=vm[1])

        ax = axs[i_*2+1]
        ph.plot_colormap(ax, colormap=cmaps[i_], reverse_cm=False, ylim=[0, 256], yticks=[0, 256],
                         yticklabels=['%.1f' % vm[0], '>%.1f' % vm[1]],
                         ylabel='', label_rotation=90, label_pos="right", label_axeratio=1)

    ax = axs[4]
    ph.adjust_spines(ax, [], lw=1, ylim=[-180, 180], yticks=[], xlim=xlim, xticks=[], pad=0)
    xbar_ss = fc.wrap(rec.subsample('xstim')[idx0:idx1])
    phase_eb = rec.ims[1].c1.phase[idx0:idx1]
    if auto_offset:
        offset = td.calculate_offsets(rec, phase_eb, xbar_ss, flight=False)
    else:
        offset = 0
    if plot_bar:
        ax.plot(rec.abf.t[idx0_beh:idx1_beh], fc.wrap(rec.abf.xstim[idx0_beh:idx1_beh]), alpha=1, zorder=1, ls='none', marker='o', ms=ms, mec='none', mfc='black')
    for i_ in range(2):
        ax.plot(rec.ims[1-i_].t[idx0:idx1], fc.wrap(rec.ims[1-i_].c1.phase[idx0:idx1]-offset), alpha=1, zorder=2, ls='none', marker='o', ms=ms, mec='none', mfc=colors[i_])

    ph.plot_x_scale_text(ax, bar_length=bar_length, text='%is' % bar_length, y_text_lower=0.85, x_text_lefter=.05, fontsize=fs)
    ax.fill_between(xlim, y1=-180, y2=-135., facecolor='black', edgecolor='none', alpha=0.1, zorder=1)
    ax.fill_between(xlim, y1=135, y2=180, facecolor='black', edgecolor='none', alpha=0.1, zorder=1)

    for vline in vlines:
        ax.axvline(vline, ls='--', lw=2, c=ph.dred, alpha=.6, zorder=1)

def plot_2c_walking0(rec, axs=[], fig_size=(15, 3), tlim=[1,200], gs_width_ratios=[200,1], plot_bar=True, plot_phase_diff=False,
                cmaps=[plt.cm.Blues, plt.cm.Oranges], vm=[0,1.6], colors=[ph.blue, ph.dorange], behs=['speed'],
                bar_length=10, fs=7, auto_offset=True, ms=1.5, lw=1, vlines=[], n_movwin=7):
    ph.set_fontsize(fs)
    if not len(axs):
        _ = plt.figure(1, fig_size)
        if not plot_phase_diff:
            gs = gridspec.GridSpec(4, 2, width_ratios=gs_width_ratios, height_ratios=[1,1,1,.6])
            axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]),
                   plt.subplot(gs[2, 0]), plt.subplot(gs[3, 0]),]
        else:
            gs = gridspec.GridSpec(5, 2, width_ratios=gs_width_ratios, height_ratios=[1, 1, 1, .6, .6])
            axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]),
                   plt.subplot(gs[2, 0]), plt.subplot(gs[3, 0]), plt.subplot(gs[4, 0]),]

    idx0 = np.where(rec.ims[0].t >= tlim[0])[0][0]
    idx1 = np.where(rec.ims[1].t <= tlim[1])[0][-1]
    idx0_beh = np.where(rec.abf.t >= tlim[0])[0][0]
    idx1_beh = np.where(rec.abf.t <= tlim[1])[0][-1]
    xlim = tlim

    # Heatmap
    for i_ in range(2):
        ax = axs[i_*2]
        ph.adjust_spines(ax, [], ylim=[rec.ims[1-i_].c1.xedges[0], rec.ims[1-i_].c1.xedges[-1]], xlim=xlim)
        ax.pcolormesh(rec.ims[1-i_].t[idx0:idx1], rec.ims[1-i_].c1.xedges, rec.ims[1-i_].c1.al[idx0:idx1].T, cmap=cmaps[i_],
                      vmin=vm[0], vmax=vm[1])

        ax = axs[i_*2+1]
        ph.plot_colormap(ax, colormap=cmaps[i_], reverse_cm=False, ylim=[0, 256], yticks=[0, 256],
                         yticklabels=['%.1f' % vm[0], '>%.1f' % vm[1]],
                         ylabel='', label_rotation=90, label_pos="right", label_axeratio=1)

    # Plot phases
    ax = axs[4]
    ph.adjust_spines(ax, [], lw=1, ylim=[-180, 180], yticks=[], xlim=xlim, xticks=[], pad=0)
    xbar_ss = fc.wrap(rec.subsample('xstim')[idx0:idx1])
    phase_eb = rec.ims[1].c1.phase[idx0:idx1]
    if auto_offset:
        offset = td.calculate_offsets(rec, phase_eb, xbar_ss, flight=False)
    else:
        offset = 0
    if plot_bar:
        ax.plot(rec.abf.t[idx0_beh:idx1_beh], fc.wrap(rec.abf.xstim[idx0_beh:idx1_beh]), alpha=1, zorder=1, ls='none', marker='o', ms=ms, mec='none', mfc='black')
    for i_ in range(2):
        ax.plot(rec.ims[1-i_].t[idx0:idx1], fc.wrap(rec.ims[1-i_].c1.phase[idx0:idx1]-offset), alpha=1, zorder=2, ls='none', marker='o', ms=ms, mec='none', mfc=colors[i_])

    ph.plot_x_scale_text(ax, bar_length=bar_length, text='%is' % bar_length, y_text_lower=0.85, x_text_lefter=.05, fontsize=fs)
    ax.fill_between(xlim, y1=-180, y2=-135., facecolor='black', edgecolor='none', alpha=0.1, zorder=1)
    ax.fill_between(xlim, y1=135, y2=180, facecolor='black', edgecolor='none', alpha=0.1, zorder=1)

    for vline in vlines:
        ax.axvline(vline, ls='--', lw=2, c=ph.dred, alpha=.6, zorder=1)

    # plot phase difference
    if plot_phase_diff:
        ax = axs[5]
        ax_speed = axs[6]
        ph.adjust_spines(ax, [], lw=1, ylim=[0, 180], yticks=[0,180], xlim=xlim, xticks=[], pad=0)

        # need to interpolate
        t_interp = np.arange(tlim[0]-0.1, tlim[1]+0.2, .1)
        idx_2interp = np.arange(idx0-5, idx1+5).astype(int)
        t_im1, t_im0 = rec.ims[1].t[idx_2interp], rec.ims[0].t[idx_2interp]
        phase_im1, phase_im0 = rec.ims[1].c1.phase[idx_2interp], rec.ims[0].c1.phase[idx_2interp]
        f_im1 = interp1d(t_im1, fc.unwrap(phase_im1), kind='linear')
        f_im0 = interp1d(t_im0, fc.unwrap(phase_im0), kind='linear')
        phase_interp_im1 = fc.wrap(f_im1(t_interp))
        phase_interp_im0 = fc.wrap(f_im0(t_interp))
        dphase = fc.circsubtract(phase_interp_im1, phase_interp_im0)
        ax.plot(t_interp, np.abs(dphase), zorder=2, ls='none', marker='o', ms=ms, mec='none', mfc='black')
    else:
        ax_speed = axs[5]

    # Plot speed
    ph.adjust_spines(ax_speed, [], lw=1, ylim=[0, 7], yticks=[0, 5], xlim=xlim, xticks=[], pad=0)
    # speed = np.convolve(rec.abf.speed, np.ones((n_movwin,)) / n_movwin, mode='same')
    speed = np.convolve(np.abs(rec.abf.dforw), np.ones((n_movwin,)) / n_movwin, mode='same')
    # ax.plot(rec.abf.t[idx0_beh:idx1_beh], speed[idx0_beh:idx1_beh], alpha=1, zorder=1, ls='none', marker='o', ms=ms, mec='none', mfc='black')
    ax_speed.plot(rec.abf.t[idx0_beh:idx1_beh], speed[idx0_beh:idx1_beh], alpha=1, zorder=1, lw=lw, c=ph.grey7)

def plot_2c_walking(rec, axs=[], fig_size=(15, 3), tlim=[1,200], gs_width_ratios=[200,1], plot_bar=True,
        cmaps=[plt.cm.Blues, plt.cm.Oranges], vm=[0,1.6], colors=[ph.blue, ph.dorange], behs=['dforw'], abs_mode=True,
        ylim_beh=[0,7], bar_length=10, fs=7, auto_offset=True, ms=1.5, lw=1, vlines=[], n_movwin=7):
    ph.set_fontsize(fs)
    if not len(axs):
        _ = plt.figure(1, fig_size)
        nrow = 3 + len(behs)
        height_ratios = [1,1,1]
        for i in range(len(behs)):
            height_ratios.append(0.6)
        gs = gridspec.GridSpec(nrow, 2, width_ratios=gs_width_ratios, height_ratios=height_ratios)

    idx0 = np.where(rec.ims[0].t >= tlim[0])[0][0]
    idx1 = np.where(rec.ims[1].t <= tlim[1])[0][-1]
    idx0_beh = np.where(rec.abf.t >= tlim[0])[0][0]
    idx1_beh = np.where(rec.abf.t <= tlim[1])[0][-1]
    xlim = tlim

    # Heatmap
    for i_ in range(2):
        ax = plt.subplot(gs[i_, 0])
        ph.adjust_spines(ax, [], ylim=[rec.ims[1-i_].c1.xedges[0], rec.ims[1-i_].c1.xedges[-1]], xlim=xlim)
        ax.pcolormesh(rec.ims[1-i_].t[idx0:idx1], rec.ims[1-i_].c1.xedges, rec.ims[1-i_].c1.al[idx0:idx1].T, cmap=cmaps[i_],
                      vmin=vm[0], vmax=vm[1])
        ax = plt.subplot(gs[i_, 1])
        ph.plot_colormap(ax, colormap=cmaps[i_], reverse_cm=False, ylim=[0, 256], yticks=[0, 256],
                         yticklabels=['%.1f' % vm[0], '>%.1f' % vm[1]],
                         ylabel='', label_rotation=90, label_pos="right", label_axeratio=1)

    # Plot phases
    ax = plt.subplot(gs[2, 0])
    ph.adjust_spines(ax, [], lw=1, ylim=[-180, 180], yticks=[], xlim=xlim, xticks=[], pad=0)
    xbar_ss = fc.wrap(rec.subsample('xstim')[idx0:idx1])
    phase_eb = rec.ims[1].c1.phase[idx0:idx1]
    if auto_offset:
        offset = td.calculate_offsets(rec, phase_eb, xbar_ss, flight=False)
    else:
        offset = 0
    if plot_bar:
        ax.plot(rec.abf.t[idx0_beh:idx1_beh], fc.wrap(rec.abf.xstim[idx0_beh:idx1_beh]), alpha=1, zorder=1, ls='none', marker='o', ms=ms, mec='none', mfc='black')
    for i_ in range(2):
        ax.plot(rec.ims[1-i_].t[idx0:idx1], fc.wrap(rec.ims[1-i_].c1.phase[idx0:idx1]-offset), alpha=1, zorder=2, ls='none', marker='o', ms=ms, mec='none', mfc=colors[i_])
    ph.plot_x_scale_text(ax, bar_length=bar_length, text='%is' % bar_length, y_text_lower=0.85, x_text_lefter=.05, fontsize=fs)
    if plot_bar:
        ax.fill_between(xlim, y1=-180, y2=-135., facecolor='black', edgecolor='none', alpha=0.1, zorder=1)
        ax.fill_between(xlim, y1=135, y2=180, facecolor='black', edgecolor='none', alpha=0.1, zorder=1)
    for vline in vlines:
        ax.axvline(vline, ls='--', lw=2, c=ph.dred, alpha=.6, zorder=1)

    # Plot behavior
    for i_ in range(len(behs)):
        ax = plt.subplot(gs[3+i_, 0])
        ph.adjust_spines(ax, [], lw=1, ylim=ylim_beh, yticks=[], xlim=xlim, xticks=[], pad=0)
        speed_orig = getattr(rec.abf, behs[i_])
        if abs_mode:
            speed_orig = np.abs(speed_orig)
        speed = np.convolve(speed_orig, np.ones((n_movwin,)) / n_movwin, mode='same')
        ax.plot(rec.abf.t[idx0_beh:idx1_beh], speed[idx0_beh:idx1_beh], alpha=1, zorder=2, lw=lw, c=ph.grey7)
        ax.axhline(0, ls='--', lw=lw, c=ph.grey4, alpha=.6, zorder=1)


def plot_colorful_trajectory(rec, location='pb', color_val=['RML'], spike=False, spike_thresh=0.2):

    x, y = rec.abf.get_trajectory()
    t_traj = rec.abf.t
    t_tif = rec.subsample('t')
    if location == 'pb':
        t_period = rec.pb.sampling_period
    elif location == 'fb':
        t_period = rec.fb.sampling_period
    elif location == 'no':
        t_period = rec.no.sampling_period
    else:
        print '%s data is not available now' % location
        return
    dforw = rec.abf.dforw
    bj_x = x[:-1][np.abs(np.diff(rec.abf.xstim_unwrap)) > 50]
    bj_y = y[:-1][np.abs(np.diff(rec.abf.xstim_unwrap)) > 50]

    def t_to_val(i):
        list_t = np.where(t_tif < t_traj[i])[0]
        if len(list_t) == 0:
            return v_min - 1
        if len(list_t) == len(t_tif):
            return v_max + 1
        ind_t = list_t[-1]
        return val[ind_t] + (val[ind_t+1]-val[ind_t]) * (t_traj[i] - t_tif[ind_t]) / t_period
    def val_to_color_index(val):
        if val <= v_min:
            return 0
        if val >= v_max:
            return 255
        return (val - v_min)/(v_max - v_min)*255
    def speed_to_lw(speed):
        lw_base = 3
        if speed <= 0:
            return lw_base
        return lw_base - speed/speed_max*2.9
    def speed_to_opaqueness(speed):
        op_base = 0.05
        if speed <= 0:
            return op_base
        if speed >= speed_max*0.8:
            return 1
        return op_base + speed/(speed_max*0.8)*(1-op_base)

    def axis_ticks(x, y, num_ish=4):
        x_space = np.round((np.max(x)-np.min(x)) / (num_ish+1)/50) * 50
        y_space = np.round((np.max(y)-np.min(y)) / (num_ish+1)/50) * 50
        xticks = [0]
        yticks = [0]
        count = -x_space
        while count > np.min(x):
            xticks.insert(0, count)
            count = count - x_space
        count = x_space
        while count < np.max(x):
            xticks.append(count)
            count = count + x_space

        count = -y_space
        while count > np.min(y):
            yticks.insert(0, count)
            count = count - y_space
        count = y_space
        while count < np.max(y):
            yticks.append(count)
            count = count + y_space
        return xticks, yticks
    def axis_ticks_minor(x, y, space=50):
        xticks_minor = [0]
        yticks_minor = [0]

        count = -space
        while count > np.min(x):
            xticks_minor.insert(0, count)
            count = count - space
        count = space
        while count < np.max(x):
            xticks_minor.append(count)
            count = count + space
        count = -space
        while count > np.min(y):
            yticks_minor.insert(0, count)
            count = count - space
        count = space
        while count < np.max(y):
            yticks_minor.append(count)
            count = count + space
        return xticks_minor, yticks_minor
    def axis_lim(x, y, space=50):
        xlim = [np.floor(np.min(x)/space)*space, np.ceil(np.max(x)/space)*space]
        ylim = [np.floor(np.min(y)/space)*space, np.ceil(np.max(y)/space)*space]
        return xlim, ylim

    rcParams['font.size'] = 15
    rcParams['xtick.labelsize'] = 'large'
    rcParams['ytick.labelsize'] = 'large'
    rcParams['axes.labelsize'] = 'large'
    fig_num = len(color_val)
    if not fig_num:
        return
    elif fig_num == 1:
        figure = plt.figure(1, (12, 10))
        gs = gridspec.GridSpec(1, 1)
    elif fig_num == 2:
        figure = plt.figure(1, (17, 8))
        gs = gridspec.GridSpec(1, 2)

    for fig_i, label_val in enumerate(color_val):
        ax = plt.subplot(gs[0, fig_i])

        # sign val for different cases
        if label_val == 'RML':
            colormap = cm.coolwarm
            cm_name = 'coolwarm'
            if location == 'pb':
                val = rec.pb.c1.rmll
            elif location == 'no':
                val = rec.no.c1.rmll
            else:
                print '%s has no attribute as %s' % (location, label_val)
                return
        elif label_val == 'AVE':
            colormap = cm.jet
            cm_name = 'jet'
            if location == 'pb':
                val = rec.pb.c1.mean_al
            elif location == 'no':
                val = rec.no.c1.mean_al
            elif location == 'fb':
                val = rec.fb.c1.mean_al
            else:
                print '%s has no attribute as %s' % (location, label_val)
                return
        else:
            print 'Have no value like %s' % label_val
            return

        v_min = np.min(val)
        v_max = np.max(val)
        speed_max = np.max(dforw)

        # Main plot sentence !!
        if label_val == 'AVE':
            for i in range(len(t_traj) - 1):
                color = colormap(int(np.round(val_to_color_index(t_to_val(i)))))
                opaqueness = speed_to_opaqueness(dforw[i])
                ax.plot([x[i], x[i+1]], [y[i], y[i+1]], c=color, alpha=opaqueness, lw=2)
        elif label_val == 'RML':
            for i in range(len(t_traj) - 1):
                color = colormap(int(np.round(val_to_color_index(t_to_val(i)))))
                linewidth = speed_to_lw(dforw[i])
                ax.plot([x[i], x[i+1]], [y[i], y[i+1]], c=color, lw=linewidth)

        ax.plot(x[0], y[0], marker='o', markersize=8, markeredgecolor='grey', markeredgewidth=2, markerfacecolor='none')
        ax.plot(x[-1], y[-1], marker='o', markersize=8, markeredgecolor='grey', markeredgewidth=2, markerfacecolor='grey')
        ax.plot(bj_x, bj_y, ls='none', marker='o', markersize=4, markeredgecolor='none', markeredgewidth=2, markerfacecolor='red')
        if spike:
            inds = np.diff(val >= spike_thresh)
            for item in t_tif[inds][::5]:
                spike_ind = np.where(t_traj >= item)[0][0]
                ax.plot(x[spike_ind], y[spike_ind], ls='none', marker='o', markersize=8, markeredgecolor='none', markerfacecolor='orange')


        axis_unit = 50
        ax.set_title(label_val + ' Fluorescence (a.u.)', fontsize=15)
        ax.set_xlabel('x (mm)')
        xticks, yticks = axis_ticks(x, y)
        ax.set_xticks(xticks)
        if fig_i == 0:
            ax.set_ylabel('y (mm)')
            ax.set_yticks(yticks)
        if fig_i == 1:
            ax.set_yticks([])
        xticks_minor, yticks_minor = axis_ticks_minor(x, y, axis_unit)
        ax.set_xticks(xticks_minor, minor=True)
        ax.set_yticks(yticks_minor, minor=True)
        ax.grid(which='minor', alpha=.5)
        ax.grid(which='major', alpha=1)
        xlim, ylim = axis_lim(x, y, axis_unit)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        gradient = np.linspace(255, -1, 256)
        gradient = np.vstack((gradient, gradient))
        if fig_num == 2:
            cbaxes = figure.add_axes([.465 + 0.42*fig_i, 0.135, 0.01, 0.18])
        else:
            cbaxes = figure.add_axes([.885, 0.135, 0.01, 0.18])
        cbaxes.imshow(gradient.T, aspect='auto', cmap=plt.get_cmap(cm_name))
        plt.yticks([0, 255], ['%1.2f' % v_max, '%1.2f' % v_min], fontsize=10)
        plt.xticks([])


def plot_noduli_properties(group):
    im.plot_isolated_turns2(group, channels=('dhead', 'no.c1.rmll'), fig_num=1)
    im.plot_isolated_turns2(group, channels=('dhead', 'no.c1.mean_al'), fig_num=2)
    plot_behavior_vs_asymm_no_onefly(group[0][0], fig_num=3)


def plot_PFN_asym_properties(group, imtype):
    if imtype == im.Bridge:
        im.plot_isolated_turns2(group, channels=('dhead', 'pb.c1.rmll'), fig_num=1, ylims=([-0.4, 0.4], [-400, 400]))
        im.plot_isolated_turns2(group, channels=('dhead', 'pb.c1.mean_al'), fig_num=2, ylims=([0, 1], [-400, 400]))
    elif imtype == im.Noduli:
        im.plot_isolated_turns2(group, channels=('dhead', 'no.c1.rmll'), fig_num=1, ylims=([-0.4, 0.4], [-400, 400]))
        im.plot_isolated_turns2(group, channels=('dhead', 'no.c1.mean_al'), fig_num=2, ylims=([0, 1], [-400, 400]))


arr18 = np.arange(1., 19.)
arr16 = np.arange(1., 17.)
arr8 = np.arange(1.,9.)
x_dict = {'pb': {'PEN': arr18, 'EIP': arr18, 'PFN': arr18, 'PFR': arr8, 'PFLC': arr18},
          'eb': {'PEN': arr16[::2] + .5, 'EIP': arr16, 'PFN': arr16[::2] + .5},
          'fb': {'PEN': arr16[::2] + .5, 'EIP': arr16, 'PFN': arr16, 'PFR': arr16, 'FBCre': arr16}}


def plot_phase_subtraction_average(group, ax=False,location='fb', offset=180, max_ro=15, max_speed=3, FWHM=False,
                                   diff_condition=False, ylim=[0, 2.5], yticks=np.arange(0, 2.6, 1), color=ph.orange,
                                   channel=1):
    r = group[0][0]                                     # get the x axis for ploting
    _, _, x = r.get_objects('%s.c%i.xedges_interp' % (location, channel))
    if diff_condition:
        abs = np.zeros((len(group), 2, 2, len(x)))          # abs contains all the info for ploting, 2 for locomotion, 2 for stimuli
    else:
        abs = np.zeros((len(group), len(x)))
    abs[:] = np.nan
    _, _, celltype = r.get_objects('%s.c%i.celltype' % (location, channel))

    def plot_FWHM():
        _, _, temp = r.get_objects('%s.c%i.xedges_interp' % (location, channel))
        bar_length = 0.05
        half_len = int(len(temp) / 2)
        curve = abs_mean[row][col]
        max_height = np.max(curve)
        min_height = np.min(curve)
        half_height = (max_height + min_height) / 2.0
        left_width = np.where(curve[:half_len] >= half_height)[0][0] / 10.0 + .5
        right_width = (np.where(curve[half_len:] <= half_height)[0][0] + half_len) / 10.0 + .5
        fwhm = right_width - left_width
        ax.plot([-1, 20], [half_height, half_height], '--', c=[.8, .8, .8], lw=1)
        ax.plot([left_width, left_width], [half_height - bar_length, half_height + bar_length], '--', c=[.6, .6, .6],
                lw=1)
        ax.plot([right_width, right_width], [half_height - bar_length, half_height + bar_length], '--', c=[.6, .6, .6],
                lw=1)
        ax.text(1, .38, 'FWHM=%i$^\circ$' % int(fwhm * 360 / 16), fontsize=13)

    for ifly, fly in enumerate(group):
        if diff_condition:
            flyab = np.zeros((len(fly),  2, 2, len(x)))
        else:
            flyab = np.zeros((len(fly), len(x)))
        flyab[:] = np.nan

        for irec, rec in enumerate(fly):

            # extract the info of this single recording
            _, _, phase = rec.get_objects('%s.c%i.phase' % (location, channel))
            im, ch, an_cubspl = rec.get_objects('%s.c%i.an_cubspl' % (location, channel))
            an_nophase = im.cancel_phase(an_cubspl, phase, 10, ch.celltype, offset=offset)
            stimid = rec.subsample('stimid')
            idx_total = np.zeros((2, len(phase)))
            dhead = rec.subsample('dhead')
            speed = rec.subsample('speed')

            if diff_condition:
                # 1.two stimuli, 2.two locomotion states
                # 1. two stimuli
                idx_total[0] = (stimid == 1)                     # idx_total[0]: CL bar
                idx_total[1] = (stimid == 2)                     # idx_total[1]: Dark
                for i_stim in (0,1):
                    inds = fc.get_contiguous_inds(idx_total[i_stim], trim_left=0, keep_ends=True, min_contig_len=50)
                    idx_total[i_stim][:] = False
                    idx_total[i_stim][inds] = True
                    idx_total = idx_total.astype(bool)

                # 2.two locomotion states
                for i_stim in (0,1):                       # iterate two stimuli: i_stim -- 0: CL bar,  1: Dark
                    for i_motion in (0,1):             # iterate two locomotion states
                        if i_motion == 0:              # i_motion -- 0: standing, 1: moving
                            idx = idx_total[i_stim] & (np.abs(dhead) < max_ro) & (speed < max_speed)
                        elif i_motion == 1:
                            idx = idx_total[i_stim] & ((np.abs(dhead) >= max_ro) | (speed >= max_speed))
                        flyab[irec][i_motion][i_stim] = an_nophase[idx].mean(axis=0)
            else:
                flyab[irec] = an_nophase.mean(axis=0)

        abs[ifly] = np.nanmean(flyab, axis=0)
        abs_mean = np.nanmean(abs, axis=0)

    # # Subsample back into glomerular / wedge resolution from cubic spline interpolation
    idx = np.arange(len(x)).reshape((len(x)/10, 10))
    if diff_condition:
        ab_sub = np.nanmean(abs[:, :, :, idx], axis=-1)
    else:
        ab_sub = np.nanmean(abs[:, idx], axis=-1)

    # Compute mean and standard error
    ab_mean = np.nanmean(ab_sub, axis=0)
    n = (np.isnan(ab_sub)==False).sum(axis=0)
    ab_sem = np.nanstd(ab_sub, axis=0) / np.sqrt(n)

    # PLOT
    # PLOT
    ph.set_fontsize(20)
    if diff_condition:
        plt.figure(1, (12, 8))
        gs = gridspec.GridSpec(2, 2)
        fig_text = [['standing, bar', 'standing, dark'], ['walking, bar', 'walking, dark']]
    else:
        if not ax:
            plt.figure(1, (6, 4))
            gs = gridspec.GridSpec(1, 1)
    # x = np.arange(1, 17.)
    x = x_dict[location[:2]][celltype]

    if diff_condition:
        for count in range(4):
            row = count / 2
            col = count % 2
            ax = plt.subplot(gs[row, col])
            line = ab_mean[row, col, :]
            ax.plot(x, line, c=ph.orange, lw=3)
            sem = ab_sem[row, col, :]
            ax.fill_between(x, line - sem, line + sem, facecolor=ph.orange, edgecolor='none', alpha=.3)

            ax.patch.set_facecolor('white')
            ax.grid(which='major', alpha=0)
            ph.adjust_spines(ax, ['left', 'bottom'], lw=.25, xlim=[1,16], ylim=[0,1.5], yticks=np.arange(0,1.51,0.5))

            ax.text(11, 1, fig_text[row][col], fontsize=13)

            # plot the full width at half maximum
            if FWHM:
                plot_FWHM()
    else:
        if not ax:
            ax = plt.subplot(gs[0, 0])
        line = ab_mean
        ax.plot(x, line, c=color, lw=3)
        for i_fly in range(len(ab_sub)):
            ax.plot(x, ab_sub[i_fly], c=color, lw=1, alpha=0.5)
        # sem = ab_sem
        # ax.fill_between(x, line - sem, line + sem, facecolor=ph.orange, edgecolor='none', alpha=.3)

        if location[:2] == 'eb':
            ax.set_xticks(np.arange(1, 17), minor=True)
            ax.set_xticks([1, 16], minor=False)
            ax.set_xlim(1, 16)
        elif location[:2] == 'pb':
            ax.set_xticks(np.arange(1, 19), minor=True)
            plt.xticks([1, 9, 10, 18], ['1', '9', '1', '9'])
            ax.set_xlim(1, 18)
        elif location[:2] == 'fb':
            ax.set_xticks(np.arange(1, 19), minor=True)
            plt.xticks([1, 16], ['1', '16'])
            ax.set_xlim(1, 16)

        ph.adjust_spines(ax, ['left', 'bottom'], lw=.25, ylim=ylim, yticks=yticks)
        ax.patch.set_facecolor('white')
        ax.grid(which='major', alpha=0)


def plot_phase_subtraction_average_3d(groups, ax=False, neuropils='fb', offsets=180, normalization='an',
                                      colors=[ph.grey8, ph.blue], channel=1,
                                      zlim=[0,4], zticks=np.linspace(0,4,3), lw=1.5, add_tuning_proj=False,
                                      bin_type='forw_speed', vel_max=11, bin_num=10, alpha_fill=0.1, plot_type='3d', fs=15,
                                      show_colorbar=True, colormap='rainbow', include_standing=False, color_shape=ph.purple, rainbow_color=False,
                                      lw_indi=1, alpha=0.5, plotcurvesonly=False, kicknan=False, plotcosfit=True,):
    def cc(arg, alpha):
        '''
        Shorthand to convert 'named' colors to rgba format at 60% opacity.
        '''
        return mcolors.to_rgba(arg, alpha=alpha)

    def polygon_errorbar(xlist, ylist, xlist2, ylist2):
        '''
        Construct the vertex list which defines the polygon filling the space under
        the (xlist, ylist) line graph.  Assumes the xs are in ascending order.
        '''
        return list(zip(xlist, ylist)) + list(zip(xlist2, ylist2))


    # -1- defining parameters
    location_dict = {
        'pb': 'pb', 'fb': 'fb', 'ebproj': 'pb', 'eb': 'eb'
    }

    low_turn_speed = 100
    low_forw_speed = 2

    speed_max = vel_max
    speed_standing = 0.2
    turn_max = vel_max
    if bin_type == 'forw_speed':
        if include_standing:
            bins_orig = np.linspace(0, speed_max, bin_num+1)
            bins = np.zeros(len(bins_orig)+1)
            bins[2:] = bins_orig[1:]
            bins[1] = speed_standing
            bin_num += 1
        else:
            bins = np.linspace(0, speed_max, bin_num + 1)

    elif bin_type == 'turn_speed':
        bins = np.linspace(0, turn_max, bin_num+1)

    Ngroups = len(groups)
    for igroup, group in enumerate(groups):

        if type(neuropils) == list:
            neuropil = neuropils[igroup]
        else:
            neuropil = neuropils
        location = location_dict[neuropil]

        if type(offsets) == list:
            offset = offsets[igroup]
        else:
            offset = offsets

        color = colors[igroup]

        # -2- extracting data
        r = group[0][0]  # get the x axis for ploting
        _, _, x = r.get_objects('%s.c%i.xedges_interp' % (location, channel))
        _, _, celltype = r.get_objects('%s.c%i.celltype' % (location, channel))
        abs = np.zeros((bin_num, len(group), len(x)))
        abs[:] = np.nan

        for ifly, fly in enumerate(group):
            flyab = np.zeros((bin_num, len(fly), len(x)))
            flyab[:] = np.nan

            for irec, rec in enumerate(fly):
                # extract the info of this single recording
                _, _, phase = rec.get_objects('%s.c%i.phase' % (location, channel))
                if normalization == 'an':
                    im, ch, an_cubspl = rec.get_objects('%s.c%i.an_cubspl' % (location, channel))
                    a_nophase = im.cancel_phase(an_cubspl, phase, 10, ch.celltype, offset=offset)
                elif normalization == 'al':
                    im, ch, al_cubspl = rec.get_objects('%s.c%i.al_cubspl' % (location, channel))
                    a_nophase = im.cancel_phase(al_cubspl, phase, 10, ch.celltype, offset=offset)

                dhead = rec.subsample('dhead')
                # speed = rec.subsample('speed')
                speed = rec.subsample('dforw')

                for ibin in range(bin_num):
                    if bin_type == 'forw_speed':
                        idx = (np.abs(dhead) < low_turn_speed) & (speed >= bins[ibin]) & (speed < bins[ibin+1])
                    elif bin_type == 'turn_speed':
                        idx = (speed < low_forw_speed) & (np.abs(dhead) >= bins[ibin]) & (np.abs(dhead) < bins[ibin+1])

                    flyab[ibin][irec] = a_nophase[idx].mean(axis=0)

            for ibin in range(bin_num):
                abs[ibin][ifly] = np.nanmean(flyab[ibin], axis=0)

        # # Subsample back into glomerular / wedge resolution from cubic spline interpolation
        idx = np.arange(len(x)).reshape((len(x) / 10, 10))
        ab_sub = np.nanmean(abs[:, :, idx], axis=-1)

        # Compute mean and standard error
        ab_mean = np.nanmean(ab_sub, axis=1)
        n = (np.isnan(ab_sub) == False).sum(axis=1)
        ab_sem = np.nanstd(ab_sub, axis=1) / np.sqrt(n)

        if neuropil == 'ebproj':
            eb_mean = []
            eb_sem = []
            for ibin in range(bin_num):
                eb_mean.append(rec.pb.get_ebproj(ab_mean[ibin], celltype, 'merge'))
                eb_sem.append(rec.pb.get_ebproj(ab_sem[ibin], celltype, 'merge'))
            ab_mean = np.array(eb_mean)
            ab_sem = np.array(eb_sem)

        ############################################################
        if '1d_shape' in plot_type: # no speed bins
            r = group[0][0]  # get the x axis for ploting
            _, _, x = r.get_objects('%s.c%i.xedges_interp' % (location, channel))
            _, _, celltype = r.get_objects('%s.c%i.celltype' % (location, channel))
            abs = np.zeros((len(group), len(x)))
            abs[:] = np.nan

            for ifly, fly in enumerate(group):
                flyab = np.zeros((len(fly), len(x)))
                flyab[:] = np.nan

                for irec, rec in enumerate(fly):
                    # extract the info of this single recording
                    _, _, phase = rec.get_objects('%s.c%i.phase' % (location, channel))
                    if normalization == 'an':
                        im, ch, an_cubspl = rec.get_objects('%s.c%i.an_cubspl' % (location, channel))
                        a_nophase = im.cancel_phase(an_cubspl, phase, 10, ch.celltype, offset=offset)
                    elif normalization == 'al':
                        im, ch, al_cubspl = rec.get_objects('%s.c%i.al_cubspl' % (location, channel))
                        a_nophase = im.cancel_phase(al_cubspl, phase, 10, ch.celltype, offset=offset)

                    dhead = rec.subsample('dhead')
                    speed = rec.subsample('dforw')

                    idx = (np.abs(dhead) < low_turn_speed) & (speed >= low_forw_speed)
                    flyab[irec] = a_nophase[idx].mean(axis=0)

                abs[ifly] = np.nanmean(flyab, axis=0)

            # # Subsample back into glomerular / wedge resolution from cubic spline interpolation
            idx = np.arange(len(x)).reshape((len(x) / 10, 10))
            ab_sub = np.nanmean(abs[:, idx], axis=-1)

            # Compute mean and standard error
            ab_mean = np.nanmean(ab_sub, axis=0)
            n = (np.isnan(ab_sub) == False).sum(axis=0)
            ab_sem = np.nanstd(ab_sub, axis=0) / np.sqrt(n)
            if neuropil == 'ebproj':
                eb_sub = np.zeros((ab_sub.shape[0], ab_sub.shape[1]-2))
                for i_fly in range(len(ab_sub)):
                    eb_sub[i_fly] = rec.pb.get_ebproj(ab_sub[i_fly], celltype, 'merge')
                eb_mean = rec.pb.get_ebproj(ab_mean, celltype, 'merge')
                eb_sem = rec.pb.get_ebproj(ab_sem, celltype, 'merge')
                ab_mean = np.array(eb_mean)
                ab_sem = np.array(eb_sem)


        # -3- PLOT
        ph.set_fontsize(fs)

        if plot_type == '3d':
            if not ax:
                fig = plt.figure(1, (7, 7))
                ax = fig.gca(projection='3d')

            if kicknan:         # kick out the nan in EPG PEN PFN data
                if np.isnan(ab_mean[0][0]):
                    ab_mean = ab_mean[:,1:-1]
                    ab_sem = ab_sem[:, 1:-1]
                if np.isnan(ab_mean[0][8]):
                    ab_mean = ab_mean[:,[0,1,2,3,4,5,6,7,10,11,12,13,14,15,16,17]]
                    ab_sem = ab_sem[:, [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 16, 17]]
                nglom = 16
                x = np.arange(1, nglom + 1)
            else:
                r = group[0][0]  # get the x axis for ploting
                _, _, x = r.get_objects('%s.c%i.xedges' % (location, channel))
                nglom = len(x) - 1
                x = np.arange(1, nglom + 1)

            for ibin in range(bin_num):
                y = np.zeros(len(x)) + (bins[ibin]+bins[ibin+1])/2.
                line = ab_mean[ibin]
                if np.sum(np.isnan(line)):
                    continue
                ax.plot(x, y, line, c=color, lw=lw)

                sem = ab_sem[ibin]
                line_errpos = line + sem
                line_errneg = line - sem
                verts = [polygon_errorbar(x, line_errpos, x[::-1], line_errneg[::-1])]
                poly = PolyCollection(verts, facecolors=[cc(color, alpha_fill)])
                # ax.add_collection3d(poly, zs=y, zdir='y')     # commented by cheng, July 14, 2019

            # plot the projection of the tuning curve
            if add_tuning_proj:
                ys = np.mean(np.array([bins[:-1], bins[1:]]), axis=0)
                zm = np.zeros(bin_num)
                ze = np.zeros(bin_num)
                x3d = np.zeros(bin_num)
                line_errpos = np.zeros(bin_num)
                line_errneg = np.zeros(bin_num)
                for ibin in range(bin_num):
                    imax = np.argmax(ab_mean[ibin])
                    zm[ibin] = ab_mean[ibin][imax]
                    ze[ibin] = ab_sem[ibin][imax]
                    x3d[ibin] = imax+1
                    line_errpos[ibin] = zm[ibin] + ze[ibin]
                    line_errneg[ibin] = zm[ibin] - ze[ibin]

                # plot on the projection plane
                x_tuning = np.zeros(bin_num) + x[0]

                ax.plot(x_tuning, ys, zm, c=color, alpha=.8, lw=lw)
                ax.plot(x_tuning, ys, zm, ls='none', marker='o', mfc=color, mec='none', ms=5, alpha=.8)

                # plot shades on the projection plane
                verts = [polygon_errorbar(ys, line_errpos, ys[::-1], line_errneg[::-1])]
                poly = PolyCollection(verts, facecolors=[cc(color, alpha_fill)])
                ax.add_collection3d(poly, zs=1, zdir='x')

                # plot on the 3d curves
                ax.plot(x3d, ys, zm, ls='none', marker='o', mfc=color, mec='none', ms=5, alpha=.8)

                # plot projection dash lines
                for ibin in range(bin_num):
                    imax = np.argmax(ab_mean[ibin])
                    zm[ibin] = ab_mean[ibin][imax]
                    ax.plot([1,imax+1], [ys[ibin],ys[ibin]], [zm[ibin],zm[ibin]], ls='--', c=color, alpha=.8, lw=1)

            if not plotcurvesonly and (igroup == Ngroups - 1):
                if neuropil[:2] == 'eb':
                    ax.set_xticks(np.arange(1, 17), minor=True)
                    ax.set_xticks([1, 16], minor=False)
                    ax.set_xlim(1, 16)
                elif neuropil[:2] == 'pb':
                    ax.set_xticks(np.arange(1, 19), minor=True)
                    plt.xticks([1, 9, 10, 18], ['1', '9', '1', '9'])
                    ax.set_xlim(1, 18)
                elif neuropil[:2] == 'fb':
                    ax.set_xticks(np.arange(1, nglom+1), minor=True)
                    ax.set_xticks([1, nglom], minor=False)
                    ax.set_xlim(1, nglom)
                ax.set_xlabel('Glom #')

                ax.view_init(35, -30)
                ax.dist = 11

                ax.set_zlim(zlim)
                ax.set_zticks(zticks)
                ax.set_zlabel(r'$\Delta$F/F0', rotation=0)
                ax.set_ylim([0, speed_max])
                ax.set_yticks(np.linspace(0,10,3))
                ax.set_ylabel('forw speed (mm/s)')
                ax.yaxis.set_label_coords(-.2, .5, -.2)

        elif plot_type == '2d_shape':

            if not ax:
                fig = plt.figure(1, (5.5, 5))
                gs = gridspec.GridSpec(10, 2, width_ratios=[20,1])
                ax = plt.subplot(gs[:, 0])

            cm = matplotlib.cm.get_cmap(colormap)
            x = x_dict[neuropil[:2]][celltype]

            for ibin in range(bin_num):
                if rainbow_color:
                    color = cm(ibin * 1. / (bin_num-1))
                else:
                    color = color_shape
                line = ab_mean[ibin]
                ax.plot(x, line, c=color, lw=lw)

                sem = ab_sem[ibin]
                line_errpos = line + sem
                line_errneg = line - sem
                # ax.fill_between(x, line_errneg, line_errpos, color=color, alpha=alpha_fill, zorder=1)

            ax.set_ylabel(r'$\Delta$F/F0', rotation=0)
            ax.yaxis.set_label_coords(-.2, .5,)

            if not plotcurvesonly and (igroup == Ngroups - 1):
                if neuropil[:2] == 'eb':
                    ax.set_xticks(np.arange(1, 17), minor=True)
                    xticks = [1, 16]
                    xlim = [1, 16]
                elif neuropil[:2] == 'pb':
                    ax.set_xticks(np.arange(1, 19), minor=True)
                    xticks = [1, 9, 10, 18]
                    xlim = [1, 18]
                elif neuropil[:2] == 'fb':
                    r = group[0][0]  # get the x axis for ploting
                    _, _, x = r.get_objects('%s.c%i.xedges' % (location, channel))
                    nglom = len(x) - 1
                    ax.set_xticks(np.arange(1, nglom+1), minor=True)
                    xticks = [1, nglom]
                    xlim = [1, nglom]
                ax.set_xlabel('Glomeruli #')
                ph.adjust_spines(ax, ['left', 'bottom'], lw=.25, xticks=xticks, xlim=xlim, yticks=zticks, ylim=zlim)
                ax.patch.set_facecolor('white')
                ax.grid(which='major', alpha=0)

                if show_colorbar:
                    ax = plt.subplot(gs[:, 1])
                    ph.plot_colormap(ax, colormap=cm, reverse_cm=False, ylim=[0, 256], yticks=[0, 256],
                                     yticklabels=['0', '>%.1f' % vel_max],
                                     ylabel='', label_rotation=90, label_pos="right", label_axeratio=1)
                    # ax = plt.subplot(gs[1:-1, 1])
                    # gradient = np.linspace(0, 1, 256)
                    # gradient = np.vstack((gradient, gradient)).T[::-1,:]
                    # ax.imshow(gradient, aspect='auto', cmap=plt.get_cmap(colormap))
                    # ax.set_yticks([0, 255])
                    # ax.set_yticklabels([str(speed_max), '0'])
                    # ax.yaxis.tick_right()
                    # ax.set_ylabel('Forward\nspeed\n(mm/s)', rotation=0, va='bottom', ha='left')
                    # ax.yaxis.set_label_position("right")
                    # ax.set_xticks([])
                    # ax.set_ylim(zlim)

        elif plot_type == '2d_tuning':
            fig = plt.figure(1, (6, 5))
            gs = gridspec.GridSpec(1, 1)

            # plot curves
            ax = plt.subplot(gs[0, 0])
            xs = np.mean(np.array([bins[:-1], bins[1:]]), axis=0)
            ym = np.zeros(bin_num)
            ye = np.zeros(bin_num)
            for ibin in range(bin_num):
                imax = np.argmax(ab_mean[ibin])
                ym[ibin] = ab_mean[ibin][imax]
                ye[ibin] = ab_sem[ibin][imax]
            ax.plot(xs, ym, c=color, alpha=1, lw=lw)
            ax.plot(xs, ym, ls='none', marker='o', mfc=color, mec='none', ms=5)
            ax.fill_between(xs, ym-ye, ym+ye, edgecolor='none', facecolor=color, alpha=alpha_fill)

            if not plotcurvesonly and (igroup == Ngroups - 1):
                yticks = zticks
                ax.set_ylabel(r'$\Delta$F/F0', rotation=0)
                ax.yaxis.set_label_coords(-.2, .5)

                xlim = [0, speed_max]
                xticks = np.linspace(0, 10, 3)
                ax.set_xlabel('Forward speed (mm/s)')
                ph.adjust_spines(ax, ['left', 'bottom'], lw=.25, xticks=xticks, xlim=xlim, yticks=yticks, ylim=zlim)
                ax.patch.set_facecolor('white')
                ax.grid(which='major', alpha=0)

        elif plot_type == '1d_shape':

            if not ax:
                fig = plt.figure(1, (5.5, 5))
                gs = gridspec.GridSpec(10, 2, width_ratios=[20,1])
                ax = plt.subplot(gs[:, 0])

            cm = matplotlib.cm.get_cmap(colormap)
            # x = x_dict[neuropil[:2]][celltype]
            r = group[0][0]  # get the x axis for ploting
            _, _, x = r.get_objects('%s.c%i.xedges' % (location, channel))
            nglom = len(x) - 1
            x = np.arange(1, nglom+1)

            if neuropil == 'ebproj':
                for i in range(len(eb_sub)):
                    ax.plot(x[:-2], eb_sub[i], c=color, lw=lw_indi, alpha=alpha)
                ax.plot(x[:-2], eb_mean, c=color, lw=lw)
            else:
                for i in range(len(ab_sub)):
                    ax.plot(x, ab_sub[i], c=color, lw=lw_indi, alpha=alpha)
                ax.plot(x, ab_mean, c=color, lw=lw)

            yticks = zticks
            ax.set_ylabel(r'$\Delta$F/F0', rotation=90, fontsize=fs)
            ax.yaxis.set_label_coords(-.2, .5,)

            if not plotcurvesonly and (igroup == Ngroups - 1):
                if neuropil[:2] == 'eb':
                    ax.set_xticks(np.arange(1, 17), minor=True)
                    xticks = [1, 16]
                    xlim = [1, 16]
                elif neuropil[:2] == 'pb':
                    ax.set_xticks(np.arange(1, 19), minor=True)
                    xticks = [1, 9, 10, 18]
                    xlim = [1, 18]
                elif neuropil[:2] == 'fb':
                    ax.set_xticks(np.arange(1, nglom + 1), minor=True)
                    xticks = [1, nglom]
                    xlim = [1, nglom]
                ax.set_xlabel('Glomeruli #', fontsize=fs)
                ph.adjust_spines(ax, ['left', 'bottom'], lw=.25, xticks=xticks, xlim=xlim, yticks=yticks, ylim=zlim)
                ax.patch.set_facecolor('white')
                ax.grid(which='major', alpha=0)

        elif plot_type == '1d_shape_norm':

            if not ax:
                fig = plt.figure(1, (5.5, 5))
                gs = gridspec.GridSpec(10, 2, width_ratios=[20, 1])
                ax = plt.subplot(gs[:, 0])

            cm = matplotlib.cm.get_cmap(colormap)
            # x = x_dict[neuropil[:2]][celltype]
            r = group[0][0]  # get the x axis for ploting
            _, _, x = r.get_objects('%s.c%i.xedges' % (location, channel))
            nglom = len(x) - 1
            x = np.arange(1, nglom + 1)

            zlength = zlim[-1] - zlim[0]
            z0 = zlim[0] + zlength * 0.1
            z1 = zlim[0] + zlength * 0.9

            if neuropil == 'ebproj':
                y0 = np.min(eb_mean)
                y1 = np.max(eb_mean)
                eb_mean_norm = (eb_mean - y0) / (y1 - y0) * (z1 - z0) + z0
                ax.plot(x[:-2], eb_mean_norm, ls='--', c=color, lw=lw, alpha=0.7)
            else:
                y0 = np.min(ab_mean)
                y1 = np.max(ab_mean)
                ab_mean_norm = (ab_mean - y0) / (y1 - y0) * (z1 - z0) + z0
                ax.plot(x, ab_mean_norm, ls='--', c=color, lw=lw, alpha=0.7)

            yticks = zticks
            ax.set_ylabel(r'Norm F', rotation=90, fontsize=fs)
            ax.yaxis.set_label_coords(-.1, .5)

            if not plotcurvesonly and (igroup == Ngroups - 1):
                if plotcosfit:
                    theta = np.linspace(np.pi, np.pi * 3, 256)
                    x_cos = (theta - theta[0]) / (2 * np.pi) * (x[-1] - x[0]) + x[0]
                    y_cos = (np.cos(theta) - (-1)) / 2. * (z1 - z0) + z0
                    ax.plot(x_cos, y_cos, lw=6, c=ph.grey9, alpha=0.3, zorder=0)
                if neuropil[:2] == 'eb':
                    ax.set_xticks(np.arange(1, 17), minor=True)
                    xticks = [1, 16]
                    xlim = [1, 16]
                elif neuropil[:2] == 'pb':
                    ax.set_xticks(np.arange(1, 19), minor=True)
                    xticks = [1, 9, 10, 18]
                    xlim = [1, 18]
                elif neuropil[:2] == 'fb':
                    # ax.set_xticks(np.arange(1, nglom + 1), minor=True)
                    xticks = [1, nglom]
                    xlim = [1, nglom]
                ax.set_xlabel('Glomeruli #', fontsize=fs)
                ph.adjust_spines(ax, ['left', 'bottom'], lw=.25, xticks=xticks, xlim=xlim, yticks=[], ylim=zlim)
                ax.patch.set_facecolor('white')
                ax.grid(which='major', alpha=0)


#############################
# Walking initiation analysis


def get_walk_initiation_data(group, sig='fb.c1.mean_al', move_style='turn', direction='right', standing_dur=2,
                             flip_head_in_forw=False, para_type='strict'):
    # modified Feb25 & Mar2 2017
    # added unwrapped behavior and changed behavior threshold of walking onset time from 0 to
    # thresh_dhead_standing and thresh_dhead_standing

    filter_window_s = 0.2                                              # numbers for thresholds
    if para_type == 'strict':
        thresh_dhead_standing = 5
        thresh_dforw_standing = 0.5
        thresh_dhead_yes = 20
        thresh_dhead_no = 50
        thresh_head_contrast_yes = 25
        thresh_head_contrast_no = 45
        thresh_dforw_yes = 2
        thresh_dforw_no = 1
        thresh_dforw_nonfiltered_no = 2.8
        thresh_forw_contrast_no = 3
        plot_stim_bouts_no = 3
        thresh_dside_yes = 2
        thresh_dhead_uni = 5
        thresh_dforw_uni = 0.2
        thresh_dside_uni = 0.2
        thresh_standing_dur = standing_dur
        exhibition_window_length = 3.0

    elif para_type == 'loose':
        thresh_dhead_standing = 10
        thresh_dforw_standing = 2
        thresh_dhead_yes = 15
        thresh_dhead_no = 200
        thresh_head_contrast_yes = 20
        thresh_head_contrast_no = 100
        thresh_dforw_yes = 1.5
        thresh_dforw_no = 3
        thresh_dforw_nonfiltered_no = 5
        thresh_forw_contrast_no = 5
        plot_stim_bouts_no = 3
        thresh_dside_yes = 2
        thresh_dhead_uni = 5
        thresh_dforw_uni = 0.2
        thresh_dside_uni = 0.2
        thresh_standing_dur = standing_dur
        exhibition_window_length = 3.0

    elif para_type == 'smallturn_in_forw':
        thresh_dhead_standing = 10
        thresh_dforw_standing = 2
        thresh_dhead_yes = 30
        thresh_dhead_no = 50
        thresh_head_contrast_yes = 20
        thresh_head_contrast_no = 100
        thresh_dforw_yes = 1.5
        thresh_dforw_no = 1
        thresh_dforw_nonfiltered_no = 5
        thresh_forw_contrast_no = 5
        plot_stim_bouts_no = 3
        thresh_dside_yes = 2
        thresh_dhead_uni = 5
        thresh_dforw_uni = 0.2
        thresh_dside_uni = 0.2
        thresh_standing_dur = standing_dur
        exhibition_window_length = 3.0

    data = {'t_abf':[], 'dhead_abf':[], 'dhead_abf_filted':[], 'dforw_abf':[], 'dside_abf':[], 'initiation_t':[],
            't_tif':[], 'imaging':[], 'head_abf':[], 'forw_abf':[], 'side_abf':[], 'fly_num':[], 'fileinfo':[]}

    def fill_data(data, flip_head=False):
        data['t_abf'].append(tc[ind_s:ind_e] - tc[ind_ini_abs])
        data['dhead_abf'].append(dhead[ind_s:ind_e])
        data['dhead_abf_filted'].append(dhead_filted[ind_s:ind_e])
        data['dforw_abf'].append(dforw[ind_s:ind_e])
        data['dside_abf'].append(dside[ind_s:ind_e])
        data['initiation_t'].append(tc[ind_ini_abs])
        data['t_tif'].append(t_imaging[ind_s_tif:ind_e_tif] - tc[ind_ini_abs])
        data['imaging'].append(imaging[ind_s_tif:ind_e_tif])
        data['fly_num'].append(ifly)
        data['fileinfo'].append([irec, tc[ind_ini_abs]])

        baseline_inds_s = int((ind_ini_abs - ind_s) - minlen_idx)
        baseline_inds_e = int(ind_ini_abs - ind_s)
        walking_inds_s = int(ind_ini_abs - ind_s)
        walking_inds_e = int(ind_e - ind_s)

        head_wrapped = fc.unwrap(rec.abf.head[ind_s:ind_e])
        head_baseline = np.nanmean(head_wrapped[baseline_inds_s:baseline_inds_e])
        head_wrapped -= head_baseline
        if flip_head:
            headmean_post = np.nanmean(head_wrapped[walking_inds_s:walking_inds_e])
            if headmean_post < 0:
                data['head_abf'].append(-head_wrapped)
            else:
                data['head_abf'].append(head_wrapped)

        else:
            data['head_abf'].append(head_wrapped)

        forw_wrapped = fc.unwrap(rec.abf.forw[ind_s:ind_e])
        forw_baseline = np.nanmean(forw_wrapped[baseline_inds_s:baseline_inds_e])
        forw_wrapped -= forw_baseline
        data['forw_abf'].append(forw_wrapped)
        side_wrapped = fc.unwrap(rec.abf.side[ind_s:ind_e])
        side_baseline = np.nanmean(side_wrapped[baseline_inds_s:baseline_inds_e])
        side_wrapped -= side_baseline
        data['side_abf'].append(side_wrapped)
        return data

    def cri_contrast_forw_no():
        starting_ind = 10
        forw_wrapped = fc.unwrap(rec.abf.forw[ind_ini_abs:ind_e])
        max_value = np.max(forw_wrapped[starting_ind:])
        max_index = np.argmax(forw_wrapped[starting_ind:])
        if max_index <= (1+starting_ind):
            return True
        else:
            min_value = np.min(forw_wrapped[:max_index])
            if (max_value - min_value) <= thresh_forw_contrast_no:
                return True
            else:
                return False

    def cri_contrast_head_yes():
        starting_ind = 0
        head_unwrapped = fc.unwrap(rec.abf.head[ind_ini_abs:ind_e])
        max_value = np.max(head_unwrapped[starting_ind:])
        min_value = np.min(head_unwrapped[starting_ind:])
        if (max_value - min_value) >= thresh_head_contrast_yes:
            return True
        else:
            return False

    def cri_contrast_head_no():
        starting_ind = 0
        head_unwrapped = fc.unwrap(rec.abf.head[ind_ini_abs:ind_e])
        max_value = np.max(head_unwrapped[starting_ind:])
        min_value = np.min(head_unwrapped[starting_ind:])
        if (max_value - min_value) <= thresh_head_contrast_no:
            return True
        else:
            return False

    for ifly, fly in enumerate(group):
        for irec, rec in enumerate(fly):
            dt = np.diff(rec.abf.tc).mean()
            sigma = filter_window_s / dt
            minlen_idx = np.ceil(thresh_standing_dur/dt)        # minimum length of standing
            mindis_s = 0.5                      # minimum distance between two standing events
            bout_detection_len_idx = int(np.ceil(2.0/dt))
            window_half_len_idx = np.ceil(exhibition_window_length/dt)     # window length for criteria and exhibition, for half length

            dhead = rec.abf.dhead
            dforw = rec.abf.dforw
            dside = rec.abf.dside
            tc = rec.abf.tc
            if 'peak' not in sig:
                t_imaging, imaging=rec.get_signal(sig)
            else:
                t_imaging = rec.ims[0].t
                if 'al' in sig:
                    imaging = np.nanmax(rec.ims[0].c1.al, axis=-1)
                if 'an' in sig:
                    imaging = np.nanmax(rec.ims[0].c1.an, axis=-1)

            dhead_filted = ndimage.filters.gaussian_filter1d(dhead, sigma)     # gaussian filt the raw speed
            dforw_filted = ndimage.filters.gaussian_filter1d(dforw, sigma)
            dside_filted = ndimage.filters.gaussian_filter1d(dside, sigma)

            idx_standing = (np.abs(dhead_filted) <= thresh_dhead_standing) \
                & (np.abs(dforw_filted) <= thresh_dforw_standing)              # get the raw standing events
            inds_standing = fc.get_contiguous_inds(idx_standing, keep_ends=True, min_contig_len=minlen_idx)

            # delete the standing events that are too close to the next one
            for i in range(len(inds_standing)-1):
                if tc[inds_standing[i+1][0]] - tc[inds_standing[i][-1]] <= mindis_s:
                    inds_standing[i] = []
            inds_standing = [item for item in inds_standing if len(item) > 0]
            # delete the last standing events if it's too close to the end
            if len(inds_standing):
                if tc[-1] - tc[inds_standing[-1][-1]] <= (exhibition_window_length+0.5):
                    del inds_standing[-1]

            # check whether each standing events is followed by a rotating bout
            if move_style == 'turn':
                for ind in inds_standing:
                    temp_dhead = dhead_filted[ind[0]:ind[-1]+bout_detection_len_idx]
                    if direction == 'left':
                        temp_dhead = -temp_dhead
                    temp_dforw = dforw_filted[ind[0]:ind[-1]+bout_detection_len_idx]
                    temp_dforw_nonfiltered = dforw[ind[0]:ind[-1] + bout_detection_len_idx]
                    # higher than threshold
                    if (np.max(temp_dhead) > thresh_dhead_yes) and (np.max(np.abs(temp_dforw)) <= thresh_dforw_no) \
                            and ((np.max(temp_dforw_nonfiltered) <= thresh_dforw_nonfiltered_no)):
                        # uni-direction
                        if np.min(temp_dhead) > -thresh_dhead_uni:
                            ind_max = np.where(temp_dhead >= thresh_dhead_yes)[0][0]      # find the bout initiation ind
                            temp = np.where(temp_dhead >= thresh_dhead_standing)[0]
                            if len(temp) == 0:
                                continue
                            ind_ini = temp[0]
                            ind_ini_abs = ind_ini + ind[0]
                            ind_s = int(ind_ini_abs - window_half_len_idx)
                            ind_e = int(ind_ini_abs + window_half_len_idx)
                            if (tc[ind_s] <= t_imaging[0]) or (tc[ind_e] >= t_imaging[-1]):
                                continue
                            ind_s_tif = int(np.where(t_imaging <= tc[ind_s])[0][-1])
                            ind_e_tif = int(np.where(t_imaging >= tc[ind_e])[0][0])
                            if cri_contrast_forw_no() and cri_contrast_head_yes(): # last criterion: forward contrast
                                data = fill_data(data)
            elif move_style == 'walk':
                for ind in inds_standing:
                    temp_dforw = dforw_filted[ind[0]:ind[-1]+bout_detection_len_idx]
                    if direction == 'backward':
                        temp_dforw = -temp_dforw
                    temp_dhead = dhead_filted[ind[0]:ind[-1]+bout_detection_len_idx]
                    # higher than threshold
                    if (np.max(temp_dforw) > thresh_dforw_yes) and (np.max(np.abs(temp_dhead)) <= thresh_dhead_no):
                        # uni-direction
                        if np.min(temp_dforw) > -thresh_dforw_uni:
                            ind_max = np.where(temp_dforw >= thresh_dforw_yes)[0][0]      # find the bout initiation ind
                            temp = np.where(temp_dforw >= thresh_dforw_standing)[0]
                            if len(temp) == 0:
                                continue
                            ind_ini = temp[0]
                            ind_ini_abs = ind_ini + ind[0]
                            ind_s = int(ind_ini_abs - window_half_len_idx)
                            ind_e = int(ind_ini_abs + window_half_len_idx)
                            if (tc[ind_s] <= t_imaging[0]) or (tc[ind_e] >= t_imaging[-1]):
                                continue
                            ind_s_tif = int(np.where(t_imaging <= tc[ind_s])[0][-1])
                            ind_e_tif = int(np.where(t_imaging >= tc[ind_e])[0][0])
                            if cri_contrast_head_no():
                                data = fill_data(data, flip_head=flip_head_in_forw)
            if move_style == 'sidewalk':
                for ind in inds_standing:
                    temp_dhead = dhead_filted[ind[0]:ind[-1]+bout_detection_len_idx]
                    temp_dside = dside_filted[ind[0]:ind[-1]+bout_detection_len_idx]
                    if direction == 'leftside':
                        temp_dside = -temp_dside
                    temp_dforw = dforw_filted[ind[0]:ind[-1]+bout_detection_len_idx]
                    # higher than threshold
                    if (np.max(temp_dside) > thresh_dside_yes) and (np.max(np.abs(temp_dforw)) <= thresh_dforw_no):
                        # uni-direction
                        if np.min(temp_dside) > -thresh_dside_uni:
                            ind_max = np.where(temp_dside >= thresh_dside_yes)[0][0]      # find the bout initiation ind
                            temp = np.where(temp_dside[:ind_max] <= thresh_dforw_standing)[0]
                            if len(temp) == 0:
                                continue
                            ind_ini = temp[-1]
                            ind_ini_abs = ind_ini + ind[0]
                            ind_s = ind_ini_abs - window_half_len_idx
                            ind_e = ind_ini_abs + window_half_len_idx
                            if (tc[ind_s] <= t_imaging[0]) or (tc[ind_e] >= t_imaging[-1]):
                                continue
                            ind_s_tif = np.where(t_imaging <= tc[ind_s])[0][-1]
                            ind_e_tif = np.where(t_imaging >= tc[ind_e])[0][0]
                            data = fill_data(data)


    return data


def plot_walk_initiation(group, tlim=(-2, 2), signal_type='fb.c1.mean_an', channel_type='abs', c=[],
                         directions=['left', 'right', 'forward'], standing_dur=2, lw_individ=1, lw=3,
                         show_indi_fly=False, show_all=False, errorbar=True, ylims=(), yticks=[],xticks=[],
                         imchannel=0):
    rcParams['ytick.labelsize'] = 20
    rcParams['xtick.labelsize'] = 20
    rcParams['font.size'] = 20
    rcParams['axes.labelsize'] = 20

    # Setup figure
    size = (2 * len(directions), 9)
    fig = plt.figure(1, size)
    gs = gridspec.GridSpec(3, len(directions))
    axs = []
    if channel_type == 'speed':
        channels = ['dhead_abf', 'dforw_abf', 'dside_abf', 'imaging']
        channels_ylabel = ['turning speed (deg/s)', 'forward speed (mm/s)', 'side speed (mm/s)',
                           'signal intensity (a.u.)']
    else:
        channels = ['head_abf', 'forw_abf', 'imaging']
        channels_ylabel = ['heading (deg)', 'forward (mm)', 'signal intensity (a.u.)']
    move_style_dict = {'left': 'turn', 'right': 'turn', 'turn': 'turn',
                       'forward': 'walk', 'backward': 'walk',
                       'leftside': 'sidewalk', 'rightside': 'sidewalk'}

    def plot_temporal_curve():
        for i, y in enumerate(sig):
            if label == 'imaging':
                t = data['t_tif'][i]
            else:
                t = data['t_abf'][i]
            if len(t) == 0:
                continue
            f = interp1d(t, y, kind='linear')
            y_interp[i] = f(t_interp)

        if show_indi_fly:
            # plot the average of individual fly and population
            fly_num = len(group)
            fly_data = []
            population_data = []
            for i_fly in range(fly_num):
                for i in range(len(sig)):
                    if data['fly_num'][i] == i_fly:
                        fly_data.append(y_interp[i])
                if len(fly_data):
                    fly_mean = np.nanmean(fly_data, axis=0)
                    population_data.append(fly_mean)
                    fly_data = []
                    ax.plot(t_interp, fly_mean, lw=lw_individ, c=color, alpha=0.3, zorder=2)
            if len(population_data):
                population_mean = np.nanmean(population_data, axis=0)
                ax.plot(t_interp, population_mean, lw=lw, c=color, alpha=1, zorder=1)
        else:
            mean = np.nanmean(y_interp, axis=0)
            ax.plot(t_interp, mean, lw=lw, c=color, zorder=2)
            # ax.scatter(t_interp, mean, lw=lw, c='black', zorder=3, s=.2)
            if show_all:
                for i, y in enumerate(sig):
                    if label == 'imaging':
                        t = data['t_tif'][i]
                    else:
                        t = data['t_abf'][i]
                    ax.plot(t, y, c=color, lw=lw_individ, alpha=.2, zorder=1)
            if errorbar:
                sem = fc.nansem(y_interp, axis=0)
                ax.fill_between(t_interp, mean - sem, mean + sem, edgecolor='none', facecolor=color, alpha=.4)

    for col, direction in enumerate(directions):
        move_style = move_style_dict[direction]
        if direction == 'turn':
            data = get_walk_initiation_data(group, sig=signal_type, move_style=move_style, direction='right',
                                                standing_dur=standing_dur)
            data2 = get_walk_initiation_data(group, sig=signal_type, move_style=move_style, direction='left',
                                                standing_dur=standing_dur)
            for key in data.keys():
                if 'head' in key:
                    for item in data2[key]:
                        data[key].append(-item)
                else:
                    data[key].extend(data2[key])
        else:
            data = get_walk_initiation_data(group, sig=signal_type, move_style=move_style, direction=direction,
                                        standing_dur=standing_dur)
        print '%i %s %ss' % (len(data.items()[0][1]), direction.capitalize(), move_style.capitalize())

        for ichannel, label in enumerate(channels):
            row = ichannel
            sig = data[label]
            # convert deg to mm
            if ('forw' in label) or ('side' in label):
                for i_trial in range(len(sig)):
                    sig[i_trial] = fc.deg2mm(sig[i_trial])
            ax = plt.subplot(gs[row, col])
            axs.append(ax)
            show_axes = []
            if row == len(channels) - 1:
                show_axes.append('bottom')
            # if row == 0:
            #     show_axes.append('top')
            #     ax.set_xticklabels([])
            #     ax.set_xlabel(direction, labelpad=2)
            if col == 0:
                show_axes.append('left')
                ax.set_ylabel(channels_ylabel[ichannel])
                ax.get_yaxis().set_label_coords(-0.5, 0.5)
            ph.adjust_spines(ax, show_axes, lw=.25)
            ax.axvline(ls='--', c=ph.grey5, zorder=1, lw=1, dashes=[3, 3])

            # Plot mean bout curves
            if 'abf' in label:
                dt = .02
                color = 'black'
                ax.axhline(ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
            else:
                dt = .1
                celltype = group[0][0].ims[imchannel].c1.celltype
                if not len(c):
                    color = colours[celltype]
                else:
                    color = c

            t_interp = np.arange(tlim[0] - .3, tlim[1] + .3, dt)
            y_interp = np.zeros((len(sig), len(t_interp)))
            plot_temporal_curve()

            show_axes = []
            ax.set_xlim(tlim)
            ph.adjust_spines(ax, show_axes, lw=1, xlim=tlim, xticks=xticks[col], ylim=ylims[row], yticks=yticks[row])
            ax.patch.set_facecolor('white')
            ax.grid(which='major', alpha=0)

    fig.text(0.5, 0.02, 'Time (s)', horizontalalignment='center', verticalalignment='bottom', fontsize='20')


def plot_walk_initiation_individual(group, tlim=(-2, 2), signal_type='fb.c1.mean_al', direction='left',
                                     lw=1, standing_dur=1, channel_type='speed', ylims=()):

    if channel_type == 'speed':
        channels = ['dhead_abf', 'dforw_abf', 'imaging']
    else:
        channels = ['head_abf', 'forw_abf', 'imaging']
    move_style_dict = {'left': 'turn', 'right': 'turn', 'turn': 'turn',
                       'forward': 'walk', 'backward': 'walk',
                       'leftside': 'sidewalk', 'rightside': 'sidewalk'}

    # Find walking initiation
    move_style = move_style_dict[direction]
    data = get_walk_initiation_data(group, sig=signal_type, move_style=move_style, direction=direction,
                                        standing_dur=standing_dur)
    trial_num = len(data['t_tif'])
    # Setup figure
    col_num = 5
    row_num = (trial_num-1)/col_num + 1
    size = (13, 4*row_num)
    fig = plt.figure(1, size)
    gs = gridspec.GridSpec(row_num, col_num)
    axs = []

    for i in range(trial_num):
        row = i / col_num
        col = i % col_num
        ax = plt.subplot(gs[row, col])

        for ichannel, label in enumerate(channels):
            loc_dict = {0:9, 1:10, 2:8}
            loc = loc_dict[ichannel]
            sig = data[label][i]
            # convert deg to mm
            if ('forw' in label) or ('side' in label):
                for i_trial in range(len(sig)):
                    sig[i_trial] = fc.deg2mm(sig[i_trial])
            sub_ax = inset_axes(ax, height="32%", width="100%", loc=loc)
            if (ichannel == 0) or (ichannel == 1):
                t = data['t_abf'][i]
                c = 'black'
                sub_ax.axhline(ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
            else:
                t = data['t_tif'][i]
                c = ph.yellow
                sub_ax.axhline(0.1, ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
            sub_ax.plot(t, sig, c=c, lw=1)
            sub_ax.axvline(ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
            sub_ax.patch.set_facecolor('white')
            sub_ax.set_xlim(tlim)
            sub_ax.set_xticks([])
            sub_ax.set_yticks([])
            if ylims:
                sub_ax.set_ylim(ylims[ichannel][:2])
                # if len(ylims[ichannel]) == 3:
                #     ytickstep = ylims[ichannel][2]
                # else:
                #     ytickstep = ylims[ichannel][1]
                # ax.set_yticks(np.arange(ylims[ichannel][0], ylims[ichannel][1]+ytickstep/2., ytickstep))

        ax.set_xlim(tlim)
        ax.set_xticks([])
        ax.set_yticks([])


def plot_walk_initiation_imaging_for_gaby_grant(group, axss=[], tlim=(-2, 2), signal_type='fb.c1.mean_an', channel_type='abs',
                                                directions=['left', 'forward'], standing_dur=2, lw_individ=1, lw=3, lw_dash=1,
                                                 show_indi_fly=False, show_all=False, errorbar=True, ylims=(), yticks=[],
                                                float_xaxis_unit=True, show_left_axis=True, color=ph.orange, alpha=0.5, fs=15,
                                                ylabel_xcoords=-0.8, plotcurvesonly=0,
                                                **kwargs):
    ph.set_fontsize(fs)

    # Setup figure
    if not len(axss):
        size = (1 * len(directions), 10)
        fig = plt.figure(1, size)
        gs = gridspec.GridSpec(3, len(directions))
        axs = []

    if channel_type == 'speed':
        channels = ['dhead_abf', 'dforw_abf', 'dside_abf', 'imaging']
        channels_ylabel = ['turning speed (deg/s)', 'forward speed (mm/s)', 'side speed (mm/s)',
                           '\Delta F/F (a.u.)']
    else:
        channels = ['head_abf', 'forw_abf', 'imaging']
        channels_ylabel = ['heading\n(deg)', 'forward\n(mm)', '$\Delta$F/F\n(a.u.)']

    move_style_dict = {'left': 'turn', 'right': 'turn', 'turn': 'turn',
                        'forward': 'walk', 'backward': 'walk',
                       'leftside': 'sidewalk', 'rightside': 'sidewalk'}

    def plot_temporal_curve():
        for i, y in enumerate(sig):
            if label == 'imaging':
                t = data['t_tif'][i]
            else:
                t = data['t_abf'][i]
            if len(t) == 0:
                continue
            f = interp1d(t, y, kind='linear')
            y_interp[i] = f(t_interp)

        if show_indi_fly:
            # plot the average of individual fly and population
            fly_num = len(group)
            fly_data = []
            population_data = []
            for i_fly in range(fly_num):
                for i in range(len(sig)):
                    if data['fly_num'][i] == i_fly:
                        fly_data.append(y_interp[i])
                if len(fly_data):
                    fly_mean = np.nanmean(fly_data, axis=0)
                    population_data.append(fly_mean)
                    fly_data = []
                    ax.plot(t_interp, fly_mean, lw=lw_individ, c=c, alpha=alpha, zorder=2)
            if len(population_data):
                population_mean = np.nanmean(population_data, axis=0)
                ax.plot(t_interp, population_mean, lw=lw, c=c, alpha=1, zorder=1)
        else:
            mean = np.nanmean(y_interp, axis=0)
            ax.plot(t_interp, mean, lw=lw, c=c, zorder=2)
            # ax.scatter(t_interp, mean, lw=lw, c='black', zorder=3, s=.2)
            if show_all:
                for i, y in enumerate(sig):
                    if label == 'imaging':
                        t = data['t_tif'][i]
                    else:
                        t = data['t_abf'][i]
                    ax.plot(t, y, c=c, lw=lw_individ, alpha=alpha, zorder=1)
            if errorbar:
                sem = fc.nansem(y_interp, axis=0)
                ax.fill_between(t_interp, mean - sem, mean + sem, edgecolor='none', facecolor=c, alpha=alpha)

    for col, direction in enumerate(directions):
        # Find walking initiation
        if direction == 'turn':
            move_style = move_style_dict['turn']
            data = get_walk_initiation_data(group, sig=signal_type, move_style=move_style, direction='right',
                                                standing_dur=standing_dur, **kwargs)
            data2 = get_walk_initiation_data(group, sig=signal_type, move_style=move_style, direction='left',
                                                standing_dur=standing_dur, **kwargs)
            for key in data.keys():
                if 'head' in key:
                    for item in data2[key]:
                        data[key].append(-item)
                else:
                    data[key].extend(data2[key])
        else:
            move_style = move_style_dict[direction]
            data = get_walk_initiation_data(group, sig=signal_type, move_style=move_style, direction=direction,
                                                standing_dur=standing_dur, **kwargs)

        print '%i %s %ss' % (len(data.items()[0][1]), direction.capitalize(), move_style.capitalize())

        for ichannel, label in enumerate(channels):
            row = ichannel
            sig = data[label]
            # convert deg to mm
            if ('forw' in label) or ('side' in label):
                for i_trial in range(len(sig)):
                    sig[i_trial] = fc.deg2mm(sig[i_trial])

            if not len(axss):
                ax = plt.subplot(gs[row, col])
                axs.append(ax)
            else:
                ax = axss[row][col]

            show_axes = []
            if (row == len(channels)-1) and not float_xaxis_unit:
                show_axes.append('bottom')
            if col == 0:
                if show_left_axis:
                    show_axes.append('left')
                    ax.set_ylabel(channels_ylabel[ichannel], labelpad=2)
                    ax.get_yaxis().set_label_coords(ylabel_xcoords, 0.5)
            # ph.adjust_spines(ax, show_axes, lw=.25)
            ax.axvline(ls='--', c=ph.grey5, zorder=1, lw=lw_dash, dashes=[5,5])

            # Plot mean bout curves
            if 'abf' in label:
                dt = .02
                c = 'black'
                ax.axhline(ls='--', c=ph.grey5, zorder=1, lw=lw_dash, dashes=[5,5])
            else:
                dt = .1
                c = color

            t_interp = np.arange(tlim[0] - .3, tlim[1] + .3, dt)
            y_interp = np.zeros((len(sig), len(t_interp)))
            plot_temporal_curve()

            if not plotcurvesonly:
                if ylims[row]:
                    ax.set_ylim(ylims[row][:2])
                if len(yticks[row]):
                    ax.set_yticks(np.arange(yticks[row][0], yticks[row][1] + yticks[row][-1] / 2., yticks[row][-1]))
                ax.set_xlim(tlim)
                if not float_xaxis_unit:
                    if col == 0:
                        ax.set_xticks(np.arange(tlim[0], tlim[-1] + 1))
                        ax.set_xticklabels([-2,'',0,'',2])
                    else:
                        ax.set_xticks(np.arange(tlim[0], tlim[-1] + 1))
                        ax.set_xticklabels([])

                if row == 0:
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    text1 = str(len(data.items()[0][1]))
                    if direction == 'turn':
                        text2 = direction + 's'
                    else:
                        text2 = direction
                    ax.text(xlim[0], ylim[0]*.9+ylim[-1]*.1, text1, ha='left', va='bottom', fontsize=fs)
                    ax.text(xlim[0], ylim[0], text2, ha='left', va='bottom', fontsize=fs)
                if (col == 0) and (row == len(channels)-1) and float_xaxis_unit:
                    ph.plot_x_scale_text(ax, 1, '1s', x_text_lefter=None, y_text_lower=None)
                ph.adjust_spines(ax, show_axes, lw=1, xticks='none', yticks=yticks[row])
                ax.patch.set_facecolor('white')
                ax.grid(which='major', alpha=0)

    if not float_xaxis_unit:
        fig.text(0.5, 0.02, 'Time (s)', horizontalalignment='center', verticalalignment='bottom', fontsize=20)

# added May 2019
def plot_walk_initiation_imaging_paper(group, axss=[], tlim=(-2, 2), signal_type='fb.c1.mean_an', channel_type='abs',
                                       directions=['turn', 'forward'], color_directions=[ph.dblue_pure, ph.dred_pure],
                                       standing_dur=2, lw_individ=1, lw=3, lw_dash=1,
                                       ybar_length_list = [1,5,50], ybar_text_list = ['1', '5\nmm', '50\ndeg'],
                                       show_indi_fly=False, show_all=True, errorbar=False, ylims=(), yticks=[], yticks_minor=[[],[],[]],
                                       float_xaxis_unit=True, show_left_axis=True, flip_head_in_forw=False,
                                       alpha=0.5, fs=10, ylabel_xcoords=-0.2, plotcurvesonly=0, **kwargs):

    ph.set_fontsize(fs)

    # Setup figure
    if not len(axss):
        size = (.6 * len(directions), 2)
        fig = plt.figure(1, size)
        gs = gridspec.GridSpec(3, len(directions), height_ratios=[2,1,1])
        axs = []

    if channel_type == 'speed':
        channels = ['dhead_abf', 'dforw_abf', 'dside_abf', 'imaging']
        channels_ylabel = ['turning speed (deg/s)', 'forward speed (mm/s)', 'side speed (mm/s)',
                           '\Delta F/F (a.u.)']
    else:
        channels = ['imaging', 'forw_abf', 'head_abf']
        channels_ylabel = ['$\Delta F/F_0$', 'forw.', 'turn']

    move_style_dict = {'left': 'turn', 'right': 'turn', 'turn': 'turn',
                       'forward': 'walk', 'backward': 'walk',
                       'leftside': 'sidewalk', 'rightside': 'sidewalk'}

    def plot_temporal_curve():
        for i, y in enumerate(sig):
            if label == 'imaging':
                t = data['t_tif'][i]
            else:
                t = data['t_abf'][i]
            if len(t) == 0:
                continue
            f = interp1d(t, y, kind='linear')
            y_interp[i] = f(t_interp)

        if show_indi_fly:
            # plot the average of individual fly and population
            fly_num = len(group)
            fly_data = []
            population_data = []
            for i_fly in range(fly_num):
                for i in range(len(sig)):
                    if data['fly_num'][i] == i_fly:
                        fly_data.append(y_interp[i])
                if len(fly_data):
                    fly_mean = np.nanmean(fly_data, axis=0)
                    population_data.append(fly_mean)
                    fly_data = []
                    ax.plot(t_interp, fly_mean, lw=lw_individ, c=c, alpha=alpha, zorder=2)
            if len(population_data):
                population_mean = np.nanmean(population_data, axis=0)
                ax.plot(t_interp, population_mean, lw=lw, c=c, alpha=1, zorder=1)
        else:
            mean = np.nanmean(y_interp, axis=0)
            ax.plot(t_interp, mean, lw=lw, c=c, zorder=2)
            # ax.scatter(t_interp, mean, lw=lw, c='black', zorder=3, s=.2)
            if show_all:
                for i, y in enumerate(sig):
                    if label == 'imaging':
                        t = data['t_tif'][i]
                    else:
                        t = data['t_abf'][i]
                    ax.plot(t, y, c=c, lw=lw_individ, alpha=alpha, zorder=1)
            if errorbar:
                sem = fc.nansem(y_interp, axis=0)
                ax.fill_between(t_interp, mean - sem, mean + sem, edgecolor='none', facecolor=c, alpha=alpha)

    for col, direction in enumerate(directions):
        c = color_directions[col]
        # Find walking initiation
        if direction == 'turn':
            move_style = move_style_dict['turn']
            data = get_walk_initiation_data(group, sig=signal_type, move_style=move_style, direction='right',
                                            standing_dur=standing_dur, flip_head_in_forw=flip_head_in_forw, **kwargs)
            data2 = get_walk_initiation_data(group, sig=signal_type, move_style=move_style, direction='left',
                                             standing_dur=standing_dur, flip_head_in_forw=flip_head_in_forw, **kwargs)
            for key in data.keys():
                if 'head' in key:
                    for item in data2[key]:
                        data[key].append(-item)
                else:
                    data[key].extend(data2[key])
        else:
            move_style = move_style_dict[direction]
            data = get_walk_initiation_data(group, sig=signal_type, move_style=move_style, direction=direction,
                                            standing_dur=standing_dur, flip_head_in_forw=flip_head_in_forw, **kwargs)

        print '%i %s %ss' % (len(data.items()[0][1]), direction.capitalize(), move_style.capitalize())

        for ichannel, label in enumerate(channels):
            row = ichannel
            sig = data[label]
            # convert deg to mm
            if ('forw' in label) or ('side' in label):
                for i_trial in range(len(sig)):
                    sig[i_trial] = fc.deg2mm(sig[i_trial])

            if not len(axss):
                ax = plt.subplot(gs[row, col])
                axs.append(ax)
            else:
                ax = axss[row][col]

            if col == 0:
                if show_left_axis:
                    ax.set_ylabel(channels_ylabel[ichannel], labelpad=2)
                    ax.get_yaxis().set_label_coords(ylabel_xcoords, 0.5)
            ax.axvline(ls='--', c=c, zorder=1, lw=lw_dash, dashes=[5, 5])

            # Plot mean bout curves
            if 'abf' in label:
                dt = .02
                ax.axhline(ls='--', c=ph.grey5, zorder=1, lw=lw_dash, dashes=[5, 5])
            else:
                dt = .1

            t_interp = np.arange(tlim[0] - .3, tlim[1] + .3, dt)
            y_interp = np.zeros((len(sig), len(t_interp)))
            plot_temporal_curve()

            if not plotcurvesonly:
                if ylims[row]:
                    ax.set_ylim(ylims[row][:2])
                if len(yticks[row]):
                    ax.set_yticks(np.arange(yticks[row][0], yticks[row][1] + yticks[row][-1] / 2., yticks[row][-1]))
                ax.set_xlim(tlim)
                if not float_xaxis_unit:
                    if col == 0:
                        ax.set_xticks(np.arange(tlim[0], tlim[-1] + 1))
                        ax.set_xticklabels([-2, '', 0, '', 2])
                    else:
                        ax.set_xticks(np.arange(tlim[0], tlim[-1] + 1))
                        ax.set_xticklabels([])

                if row == (len(channels)-1):
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    ax.text((xlim[0] + xlim[1]) / 2., ylim[0] - (ylim[1] - ylim[0]) * .1, '%i' % len(data.items()[0][1]), ha='center', va='top', fontsize=fs)
                    if col == 0:
                        ax.text(xlim[0] - (xlim[1] - xlim[0]) * .1, ylim[0] - (ylim[1] - ylim[0]) * .1, 'n', ha='left', va='top', fontsize=fs)
                if (col == 0) and (row == 0) and float_xaxis_unit:
                    ph.plot_x_scale_text(ax, 1, '1s', x_text_lefter=.6, y_text_lower=.2)

                ph.adjust_spines(ax, [], lw=1, xticks='none', yticks=yticks[row], yticks_minor=yticks_minor[row])
                ax.patch.set_facecolor('white')
                ax.grid(which='major', alpha=0)

            if col == 1:
                ph.plot_y_scale_text(ax, bar_length=ybar_length_list[row], text=ybar_text_list[row], bar_pos='bot',
                                     x_text_righter=0.2, text2bar_ratio=.05, color='black', rotation='horizontal',
                                     ha='left', va='bottom', fs=fs, lw=1)

    if not float_xaxis_unit:
        fig.text(0.5, 0.02, 'Time (s)', horizontalalignment='center', verticalalignment='bottom', fontsize=20)


# Stimuli Screen Flight


def get_flight_stopping_inds(rec, min_len_s=0.5, wing_amp_thresh=-10):
    # output the inds of abf when fly is not flying
    # tricks used to avoid tracking error:
    # 1. stopping too short (less than 0.5 here)
    # 2. no puffing
    # 3. at the end of a recording

    lwa = rec.abf.lwa
    rwa = rec.abf.rwa
    file_length = len(rec.abf.lwa)
    left_thresh = (lwa < wing_amp_thresh)
    right_thresh = (rwa < wing_amp_thresh)
    min_len_frames = min_len_s * rec.abf.subsampling_rate
    left_inds_initial = fc.get_contiguous_inds(left_thresh, trim_left=0, keep_ends=True, min_contig_len=min_len_frames)
    right_inds_initial = fc.get_contiguous_inds(right_thresh, trim_left=0, keep_ends=True, min_contig_len=min_len_frames)
    puff_inds = fc.get_contiguous_inds(rec.abf.puff > 1, trim_left=0, keep_ends=True, min_contig_len=0)
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
        elif (inds[-1]+1) == file_length:
            left_inds.append(inds)

    for inds in right_inds_initial:
        for puff_ind in puff_inds:
            if (puff_ind >= inds[0]) & (puff_ind <= inds[-1]):
                puff_flag = 1
                break
        if puff_flag:
            puff_flag = 0
            right_inds.append(inds)
        elif (inds[-1]+1) == file_length:
            right_inds.append(inds)

    return left_inds, right_inds


def get_six_stimuli_data(group, signal_type='fb.c1.mean_al', t_precede=4):
    # corresponding to the 2p data collected BEFORE Jan. 1st, 2017
    datas = [{}, {}, {}, {}, {}, {}]
    for i in range(len(datas)):
        datas[i] = {'t_abf':[], 'lwa':[], 'rwa':[], 't_tif':[], 'imaging':[]}
    stimid_dict = {-3:0, -4:1, -5:2, -6:3, -7:4, -8:5}

    def fill_data(datas):
        datas[i_data]['t_abf'].append(tc[ind_s:ind_e] - ts[0])
        datas[i_data]['lwa'].append(lwa[ind_s:ind_e])
        datas[i_data]['rwa'].append(rwa[ind_s:ind_e])
        datas[i_data]['t_tif'].append(t_tif[ind_s_tif:ind_e_tif] - ts[0])
        datas[i_data]['imaging'].append(imaging[ind_s_tif:ind_e_tif])
        return datas

    def stop_detection():
        for ind in stop_inds:
            if (ind >= ind_s) & (ind <= ind_e):
                return True
        return False

    for fly in group:
        for rec in fly:
            delta_inds_abf = np.ceil((t_precede + 1) * rec.abf.subsampling_rate)
            delta_inds_tif = np.ceil((t_precede + 1) * rec.fb.sampling_rate)
            tss = get_trial_ts(rec)
            left_inds, right_inds = get_flight_stopping_inds(rec)
            temp = copy.copy(left_inds)
            temp.extend(right_inds)
            stop_inds = np.unique(np.hstack(np.hstack(np.array(temp))))
            for i, ts in enumerate(tss):
                if ts[-1]-ts[0] <= 2:
                    continue
                stimid = fc.get_stimid_in_window(rec, ts)
                xgain = fc.get_xgain_in_window(rec, ts)
                if (stimid in stimid_dict.keys()) and np.abs(xgain) < 60:
                    t_tif, imaging = rec.get_signal(signal_type)
                    tc = rec.abf.tc
                    lwa = rec.abf.lwa
                    rwa = rec.abf.rwa
                    ind_s = np.where(tc <= ts[0])[0][-1] - delta_inds_abf
                    ind_e = np.where(tc >= ts[-1])[0][0]
                    ind_s_tif = np.where(t_tif <= ts[0])[0][-1] - delta_inds_tif
                    ind_e_tif = np.where(t_tif >= ts[-1])[0][0]
                    i_data = stimid_dict[stimid]
                    if not stop_detection():
                        datas = fill_data(datas)
    return datas


def get_six_xgains_data(group, signal_type='fb.c1.mean_al', t_precede=4):
    # corresponding to the 2p data collected before Jan. 1st, 2017
    datas = [{}, {}, {}, {}, {}, {}]
    for i in range(len(datas)):
        datas[i] = {'t_abf':[], 'lwa':[], 'rwa':[], 't_tif':[], 'imaging':[]}
    # xgain_dict = {5:0, 9:1, 13:2, 14:2, 17:3, 18:3, 21:4, 22:4, 26:5}
    xgain_dict = {4:0, 5:0, 6:0,
                  10:1, 11:1, 12:1,
                  15:2, 16:2, 17:2,
                  21:3, 22:3, 23:3,
                  26:4, 27:4, 28:4,
                  32:5, 33:5, 34:5}

    def fill_data(datas):
        datas[i_data]['t_abf'].append(tc[ind_s:ind_e] - ts[0])
        datas[i_data]['lwa'].append(lwa[ind_s:ind_e])
        datas[i_data]['rwa'].append(rwa[ind_s:ind_e])
        datas[i_data]['t_tif'].append(t_tif[ind_s_tif:ind_e_tif] - ts[0])
        datas[i_data]['imaging'].append(imaging[ind_s_tif:ind_e_tif])
        return datas

    def stop_detection():
        for ind in stop_inds:
            if (ind >= ind_s) & (ind <= ind_e):
                return True
        return False

    for fly in group:
        for rec in fly:
            delta_inds_abf = np.ceil((t_precede + 1) * rec.abf.subsampling_rate)
            delta_inds_tif = np.ceil((t_precede + 1) * rec.fb.sampling_rate)
            tss = get_trial_ts(rec)
            left_inds, right_inds = get_flight_stopping_inds(rec)
            temp = copy.copy(left_inds)
            temp.extend(right_inds)
            stop_inds = np.unique(np.hstack(np.hstack(np.array(temp))))
            for i, ts in enumerate(tss):
                stimid = fc.get_stimid_in_window(rec, ts)
                xgain = np.round(fc.get_xgain_in_window(rec, ts))
                if (stimid == -3) and np.abs(xgain) < 60:
                    t_tif, imaging = rec.get_signal(signal_type)
                    tc = rec.abf.tc
                    lwa = rec.abf.lwa
                    rwa = rec.abf.rwa
                    ind_s = np.where(tc <= ts[0])[0][-1] - delta_inds_abf
                    ind_e = np.where(tc >= ts[-1])[0][0]
                    ind_s_tif = np.where(t_tif <= ts[0])[0][-1] - delta_inds_tif
                    ind_e_tif = np.where(t_tif >= ts[-1])[0][0]
                    # if xgain in xgain_dict.keys():
                    i_data = xgain_dict[xgain]
                    if not stop_detection():
                        datas = fill_data(datas)
    return datas


def plot_stimuli_transition_flight(group, stim_type='six', t_precede=4, t_follow=8, signal_type='fb.c1.mean_al',
                                   title=[], show_all=True, errorbar=True):
    # corresponding to the 2p data collected before Jan. 1st, 2017
    rcParams['ytick.labelsize'] = 20
    rcParams['xtick.labelsize'] = 20
    # rcParams['font.size'] = 25
    rcParams['axes.labelsize'] = 20
    size = (25, 15)
    fig = plt.figure(1, size)
    gs = gridspec.GridSpec(3, 6)
    if stim_type == 'six':
        data = get_six_stimuli_data(group, signal_type, t_precede)
        subtitle_list = ['progressive', 'regressive', 'pitch up', 'pitch down', 'yaw leftward', 'yaw rightward']
    elif stim_type == 'pro':
        data = get_six_xgains_data(group, signal_type, t_precede)
        subtitle_list = ['gain: 5', 'gain: 11', 'gain: 16', 'gain: 22', 'gain: 27', 'gain: 33']
    else:
        print 'Error: does not have this type of stim_type'
        return
    ylabel_list = ['Ave Signal', 'lpr WA', 'lmr WA']
    ylim_list = [[0, 1.2], [20, 80], [-45, 45]]
    yticks_list = [np.arange(0,1.21,0.4), np.arange(20, 81, 20), np.arange(-30,31,30)]

    for row, ylabel in enumerate(ylabel_list):
        for col, xlabel in enumerate(subtitle_list):
            ax = plt.subplot(gs[row, col])

            # Plot mean bout curves
            if row == 0:
                dt = 0.05
            else:
                dt = 0.02
            t_interp = np.arange(-t_precede-0.3, t_follow-0.11, dt)
            y_interp = np.zeros((len(data[col]['t_tif']), len(t_interp)))
            for i in range(len(data[col]['t_tif'])):
                if row == 0:
                    t = data[col]['t_tif'][i]
                    y = data[col]['imaging'][i]
                    if show_all:
                        ax.plot(t, y, c='grey', alpha=.2, zorder=1)
                elif row == 1:
                    t = data[col]['t_abf'][i]
                    y = (data[col]['lwa'][i] + data[col]['rwa'][i]) / 2.0
                    if show_all:
                        ax.plot(t, y, c='grey', alpha=.2, zorder=1)
                elif row == 2:
                    t = data[col]['t_abf'][i]
                    y = data[col]['lwa'][i] - data[col]['rwa'][i]
                    if show_all:
                        ax.plot(t, y, c='grey', alpha=.2, zorder=1)
                if len(t) == 0:
                    continue
                f = interp1d(t, y, kind='linear')
                # print t[0], t[-1], t_interp[0], t_interp[-1]
                y_interp[i] = f(t_interp)
            ax.axvline(x=0, ls='--', c='black', zorder=1, lw=1, dashes=[5, 5])
            mean = np.nanmean(y_interp, axis=0)
            c = seq_dark_colours()[col]
            ax.plot(t_interp, mean, lw=2, c=c, zorder=2)
            if errorbar:
                sem = fc.nansem(y_interp, axis=0)
                ax.fill_between(t_interp, mean-sem, mean+sem, edgecolor='none', facecolor=c, alpha=.4)
            ax.set_xlim([-t_precede, t_follow])
            ax.set_xticks(np.arange(-3, t_follow-0.12, 3))
            ax.set_ylim(ylim_list[row])
            ax.set_yticks(yticks_list[row])

            show_axes = []
            if row == 2:
                show_axes.append('bottom')
                ax.axhline(y=0, ls='--', c='black', zorder=1, lw=1, dashes=[5, 5])
            if row == 0:
                show_axes.append('top')
                ax.set_xticklabels([])
                ax.set_xlabel(subtitle_list[col])
            if col == 0:
                show_axes.append('left')
                ax.set_ylabel(ylabel_list[row])
            ph.adjust_spines(ax, show_axes, lw=.25)
            ax.axhline(ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])

    fig.text(0.5, 0.07, 'Time (s)', horizontalalignment='center', verticalalignment='bottom', fontsize='20')
    if title:
        fig.suptitle(title, fontsize='20')


def get_stim_bouts_data(group, stim_type='speed_6', signal_type='fb.c1.mean_al', t_precede=2, t_follow=6,
                        imaging_type='phase'):
    # corresponding to the 2p data collected AFTER Jan. 1st, 2017

    def stim_type_info(stim_type):
        if stim_type == 'screen':
            return 10, [[-3,30,2],[-3,-30,2],[-4,30,4],[-4,-30,4],[-5,30,6],
                    [-5,-30,6],[-6,30,8],[-6,-30,8],[-7,30,10],[-7,-30,10]]
        elif stim_type == 'speed':
            return 7, [[-3, -30, 3], [-3, 30, 1], [-3, 30, 2], [-3, 30, 3],
                       [-3, 30, 4], [-3, 30, 5], [-3, 30, 6]]
        elif stim_type == 'speed_6':
            return 7, [[-3, -30, 3], [-3, 30, 1], [-3, 30, 2], [-3, 30, 3],
                       [-3, 30, 4], [-3, 30, 5], [-3, 30, 6]]
        elif stim_type == 'offcenter':
            return 9, [[-3, 30, 1], [-3, 30, 2], [-3, 30, 3], [-3, 30, 4],
                       [-3, 30, 5], [-3, 30, 6], [-3, 30, 7], [-3, 30, 8], [-3, 30, 9]]
        elif stim_type == 'pro_reg':
            return 4, [[-3, -30, 6], [-3, -30, 3], [-3, 30, 3], [-3, 30, 6]]
        elif stim_type == 'pro_reg_v2':
            return 3, [[-3, -30, 4], [-3, 0, 4], [-3, 30, 4]]
        elif stim_type == 'heading_vs_head':
            return 7, [[-3, 30, 1], [-3, 30, 5], [-4, 30, 1], [-4, 30, 5], [-4, 30, 10], [-5, 30, 10], [-5, -30, 10]]
        elif stim_type == 'heading_vs_head_v2':
            return 8, [[-3, 30, 1], [-3, 30, 5], [-4, 30, 1], [-4, 30, 5], [-4, 30, 10], [-5, 30, 10], [-5, -30, 10],
                       [-4, -30, 1]]
        elif stim_type == 'heading_vs_head_v4':
            return 5, [[-1, 46, 1], [-1, 46, 2], [-1, 46, 3], [-1, 46, 4], [-1, 46, 5]]
        return 'stim_type is not in the list'

    gain_tolerance = 2

    def fill_data(datas):
        # imaging_fill = imaging[:, glomerulus_num]  # for Ari's individual glomerilus

        datas[i_data].t_abf.append(tc[ind_s:ind_e] - ts[0])
        datas[i_data].lwa.append(lwa[ind_s:ind_e])
        datas[i_data].rwa.append(rwa[ind_s:ind_e])
        datas[i_data].t_tif.append(t_tif[ind_s_tif:ind_e_tif] - ts[0])

        ### for unwrap and subtract baseline if imaging signal is phase
        if imaging_type == 'phase':
            phase = imaging[ind_s_tif:ind_e_tif]
            phase_unwrap = fc.unwrap(phase, 360)
            phase_base_subtract = phase_unwrap - phase_unwrap[ind_0_tif-ind_s_tif]
            datas[i_data].imaging.append(phase_base_subtract)
        elif imaging_type == 'tif':
            datas[i_data].imaging.append(imaging[ind_s_tif:ind_e_tif])

        # datas[i_data].imaging.append(imaging_fill[ind_s_tif:ind_e_tif])  # for Ari's individual glomerilus
        datas[i_data].xstim.append(xstim[ind_s:ind_e])
        datas[i_data].fly_num.append(ifly)
        datas[i_data].stop_detection.append(stop_detection())
        datas[i_data].fileinfo.append([irec, ts[0]])
        return datas

    def stop_detection():
        for ind in stop_inds:
            if (ind >= ind_s) & (ind <= ind_e):
                return True
        return False

    def inrange_detection():
        if (ind_s <= 0) or (ind_s_tif <= 0):
            return True
        if ind_e >= len(tc):
            return True
        if (tc[ind_e] >= t_tif[-1]) or (ind_e_tif >= len(t_tif)):
            return True
        return False

    class stim_bouts_flight():
        def __init__(self):
            self.t_abf = []
            self.lwa = []
            self.rwa = []
            self.t_tif = []
            self.imaging = []
            self.fly_num = []
            self.xstim = []
            self.stop_detection = []
            self.fileinfo = []  # [irec, abs time]

    stim_num, cri_list = stim_type_info(stim_type)
    datas = []
    for i in range(stim_num):
        datas.append(stim_bouts_flight())

    for ifly, fly in enumerate(group):
        for irec, rec in enumerate(fly):
            delta_precede_inds_abf = np.ceil((t_precede + 1) * rec.abf.subsampling_rate)
            delta_precede_inds_tif = np.ceil((t_precede + 1) * rec.ims[0].sampling_rate)
            if t_follow:
                delta_follow_inds_abf = np.ceil((t_follow + 1) * rec.abf.subsampling_rate)
                delta_follow_inds_tif = np.ceil((t_follow + 1) * rec.ims[0].sampling_rate)
            tss = get_trial_ts(rec)
            left_inds, right_inds = get_flight_stopping_inds(rec)
            temp = copy.copy(left_inds)
            temp.extend(right_inds)
            temp = np.array(temp)
            if temp.ndim == 2:
                stop_inds = np.unique(np.hstack(np.hstack(np.array(temp))))
            else:
                if temp.size:
                    stop_inds = np.unique(np.hstack(np.hstack(np.array(temp))))
                else:
                    stop_inds = []
            t_tif, imaging = rec.get_signal(signal_type)
            if len(t_tif) == 2:
                t_tif = t_tif[0]
            # t_tif = t_tif[0]  # for Ari's individual glomerilus
            tc = rec.abf.tc
            lwa = rec.abf.lwa
            rwa = rec.abf.rwa
            xstim = rec.abf.xstim
            for ts in tss:
                if ts[-1]-ts[0] <= 2:
                    continue
                stimid = fc.get_stimid_in_window(rec, ts)
                xgain, xgain2 = fc.get_xgain_in_window(rec, ts)
                ystim = fc.get_ystim_in_window(rec, ts)
                i_data = -1
                for i in range(stim_num):
                    cri = cri_list[i]
                    if (stimid == cri[0]) & (xgain >= cri[1] - gain_tolerance) & (xgain <= cri[1] + gain_tolerance) \
                            & (xgain2 >= cri[1] - gain_tolerance) & (xgain2 <= cri[1] + gain_tolerance) &\
                            (ystim == cri[2]):
                        i_data = i
                        break
                if i_data >= 0:
                    if t_follow == 0:
                        ind_s = np.where(tc <= ts[0])[0][-1] - delta_precede_inds_abf
                        ind_e = np.where(tc >= ts[-1])[0][0]
                        ind_0_tif = np.where(t_tif <= ts[0])[0][-1]
                        ind_s_tif = ind_0_tif - delta_precede_inds_tif
                        ind_e_tif = np.where(t_tif >= ts[-1])[0][0]
                    else:
                        ind_s = np.where(tc <= ts[0])[0][-1] - delta_precede_inds_abf
                        ind_e = np.where(tc <= ts[0])[0][-1] + delta_follow_inds_abf
                        ind_0_tif = np.where(t_tif <= ts[0])[0][-1]
                        ind_s_tif = ind_0_tif - delta_precede_inds_tif
                        ind_e_tif = np.where(t_tif <= ts[0])[0][-1] + delta_follow_inds_tif
                    ind_s = int(ind_s)
                    ind_e = int(ind_e)
                    ind_0_tif = int(ind_0_tif)
                    ind_s_tif = int(ind_s_tif)
                    ind_e_tif = int(ind_e_tif)
                    if not inrange_detection():
                        datas = fill_data(datas)


    return datas


def get_stim_bouts_data_walking(group, stim_type='OLOF6', signal_type='mean_pl', t_precede=2, t_follow=2):
    # corresponding to the 2p data collected AFTER March, 2017

    def stim_type_info(stim_type):
        if stim_type == 'OLOF6':
            return 6, [[-3,-30,3],[-3,30,3],[-3,-30,6],[-3,30,6],[-3,-30,10],[-3,30,10]]
        elif stim_type == 'pro_reg_v2':
            return 3, [[-3, -30, 4], [-3, 0, 4], [-3, 30, 4]]
        return 'stim_type is not in the list'
    gain_tolerance = 4

    def fill_data(datas):
        datas[i_data].t_abf.append(tc[ind_s:ind_e] - ts[0])
        datas[i_data].t_tif.append(t_tif[ind_s_tif:ind_e_tif] - ts[0])
        datas[i_data].imaging.append(imaging[ind_s_tif:ind_e_tif])
        datas[i_data].xstim.append(xstim[ind_s:ind_e])
        datas[i_data].fly_num.append(ifly)
        datas[i_data].dhead.append(rec.subsample('dhead')[ind_s_tif:ind_e_tif])
        datas[i_data].dforw.append(rec.subsample('dforw')[ind_s_tif:ind_e_tif])

        unwrapped = fc.unwrap(rec.abf.head[ind_s:ind_e])
        baseline = unwrapped[delta_precede_inds_abf]
        unwrapped -= baseline
        datas[i_data].head.append(unwrapped)
        unwrapped = fc.unwrap(rec.abf.forw[ind_s:ind_e])
        baseline = unwrapped[delta_precede_inds_abf]
        unwrapped -= baseline
        datas[i_data].forw.append(unwrapped)
        unwrapped = fc.unwrap(rec.abf.side[ind_s:ind_e])
        baseline = unwrapped[delta_precede_inds_abf]
        unwrapped -= baseline
        datas[i_data].side.append(unwrapped)

        return datas

    class stim_bouts_walking():
        def __init__(self):
            self.t_abf = []
            self.t_tif = []
            self.forw = []
            self.head = []
            self.side = []
            self.dhead = []
            self.dforw = []
            self.imaging = []
            self.fly_num = []
            self.xstim = []

    def inrange_detection():
        if (ind_s <= 0) or (ind_s_tif <= 0):
            return True
        if ind_e >= len(tc):
            return True
        if (tc[ind_e] >= t_tif[-1]) or (ind_e_tif >= len(t_tif)):
            return True
        return False

    stim_num, cri_list = stim_type_info(stim_type)
    datas = []
    for i in range(stim_num):
        datas.append(stim_bouts_walking())

    for ifly, fly in enumerate(group):
        for rec in fly:
            delta_precede_inds_abf = int(np.ceil((t_precede + 1) * rec.abf.subsampling_rate))
            delta_precede_inds_tif = int(np.ceil((t_precede + 1) * rec.ims[0].sampling_rate))
            if t_follow:
                delta_follow_inds_abf = int(np.ceil((t_follow + 1) * rec.abf.subsampling_rate))
                delta_follow_inds_tif = int(np.ceil((t_follow + 1) * rec.ims[0].sampling_rate))
            tss = get_trial_ts(rec)
            t_tif, imaging = rec.get_signal(signal_type)
            tc = rec.abf.tc
            xstim = rec.abf.xstim
            for ts in tss:
                stimid = fc.get_stimid_in_window(rec, ts)
                xgain, xgain2 = fc.get_xgain_in_window(rec, ts)
                ystim = fc.get_ystim_in_window(rec, ts)
                i_data = -1
                for i in range(stim_num):
                    cri = cri_list[i]
                    if (stimid == cri[0]) & (xgain >= cri[1]-gain_tolerance) & (xgain <= cri[1]+gain_tolerance) \
                            & (xgain2 >= cri[1] - gain_tolerance) & (xgain2 <= cri[1] + gain_tolerance) & (ystim == cri[2]):
                        i_data = i
                        break
                if i_data >= 0:
                    if t_follow == 0:
                        ind_s = np.where(tc <= ts[0])[0][-1] - delta_precede_inds_abf
                        ind_e = np.where(tc >= ts[-1])[0][0]
                        ind_s_tif = np.where(t_tif <= ts[0])[0][-1] - delta_precede_inds_tif
                        ind_e_tif = np.where(t_tif >= ts[-1])[0][0]
                    else:
                        ind_s = np.where(tc <= ts[0])[0][-1] - delta_precede_inds_abf
                        ind_e = np.where(tc <= ts[0])[0][-1] + delta_follow_inds_abf
                        ind_s_tif = np.where(t_tif <= ts[0])[0][-1] - delta_precede_inds_tif
                        ind_e_tif = np.where(t_tif <= ts[0])[0][-1] + delta_follow_inds_tif
                    ind_s = int(ind_s)
                    ind_e = int(ind_e)
                    ind_s_tif = int(ind_s_tif)
                    ind_e_tif = int(ind_e_tif)
                    if not inrange_detection():
                        datas = fill_data(datas)

    return datas


def plot_stim_bouts(group, stim_type='heading_vs_head', t_precede=2, t_follow=6, signal_type='fb.c1.mean_al',
                    title='', show_all=False, errorbar=False, axvline=[], flying_cri=True,
                    indi_alpha=0.5, indi_lw=1, lw=3,
                    ylim=[[0, 3], [20, 80], [-45, 45], [0, 10]],
                    yticks=[np.arange(0,1.21,0.4), np.arange(20, 81, 20), np.arange(-30,31,30), np.arange(0,5,10)],
                    glomerulus_num=0):
    # corresponding to the 2p data collected AFTER Jan. 1st, 2017
    fs = 12
    ph.set_fontsize(fs)
    size = (7, 8)
    fig = plt.figure(1, size)
    if stim_type == 'screen':
        stim_num = 10
        subtitle_list = ['progressive', 'regressive', 'yaw leftward', 'yaw rightward', 'roll rightward',
                         'roll leftward', 'pitch downward', 'pitch upward', 'lift downward', 'lift upward']
    elif stim_type == 'speed':
        stim_num = 7
        subtitle_list = ['speed: - 0.33 m/s', 'speed: 0.11 m/s', 'speed: 0.22 m/s', 'speed: 0.33 m/s', 'speed: 0.44 m/s', 'speed: 0.55 m/s', 'speed: 0.66 m/s']
    elif stim_type == 'offcenter':
        stim_num = 9
        subtitle_list = ['-90', '-67.5', '-45', '-22.5', '0', '22.5', '45', '67.5', '90']
    elif stim_type == 'pro_reg':
        stim_num = 4
        subtitle_list = ['speed: -0.30 m/s', 'speed: -0.15 m/s', 'speed: 0.15 m/s', 'speed: 0.30 m/s']
    elif stim_type == 'pro_reg_v2':
        stim_num = 3
        subtitle_list = ['velocity: -0.20 m/s', 'velocity: 0 m/s', 'velocity: 0.20 m/s']
    elif stim_type == 'heading_vs_head':
        stim_num = 7
        subtitle_list = ['-90 deg', '-40 deg', '0 deg', '40 deg', '90 deg', 'LW yaw', 'RW yaw']
    elif stim_type == 'heading_vs_head_v4':
        stim_num = 5
        subtitle_list = ['-90 deg', '-45 deg', '0 deg', '45 deg', '90 deg']
    else:
        print 'Error: does not have this type of stim_type'
        return

    gs = gridspec.GridSpec(4, stim_num)
    data = get_stim_bouts_data(group, stim_type, signal_type, t_precede, glomerulus_num=glomerulus_num)
    fly_num = len(group)
    color = ph.green

    if 'phase' in signal_type:
        ylabel_list = ['phase\n(deg)', 'L+R WA\n(deg)', 'L-R WA\n(deg)', 'stim x\n(volt)'] # for phase
    else:
        ylabel_list = ['$\Delta F/F$', 'L+R WA\n(deg)', 'L-R WA\n(deg)', 'stim x\n(volt)']

    for row, ylabel in enumerate(ylabel_list):
        for col, xlabel in enumerate(subtitle_list):
            ax = plt.subplot(gs[row, col])
            # Plot mean bout curves
            if row == 0:
                dt = 0.05
            else:
                dt = 0.02

            fly_mean = []
            t_interp = np.arange(-t_precede - 0.3, t_follow - 0.11, dt)
            for i_fly in range(fly_num):
                y_interp = np.zeros((len(data[col].t_tif), len(t_interp)))
                for i1 in range(len(y_interp)):
                    for i2 in range(len(y_interp[i1])):
                        y_interp[i1][i2] = np.nan
                for i in range(len(data[col].t_tif)):
                    if flying_cri:
                        if data[col].stop_detection[i]:
                            continue
                    if data[col].fly_num[i] == i_fly:
                        if row == 0:
                            t = data[col].t_tif[i]
                            y = data[col].imaging[i]
                        elif row == 3:
                            t = data[col].t_abf[i]
                            y = data[col].xstim[i]
                        elif row == 1:
                            t = data[col].t_abf[i]
                            y = (data[col].lwa[i] + data[col].rwa[i]) / 2.0
                        elif row == 2:
                            t = data[col].t_abf[i]
                            y = data[col].lwa[i] - data[col].rwa[i]
                        f = interp1d(t, y, kind='linear')
                        # print t[0], t[-1], t_interp[0], t_interp[-1]
                        y_interp[i] = f(t_interp)
                mean = np.nanmean(y_interp, axis=0)
                ax.plot(t_interp, mean, lw=indi_lw, c=color, zorder=1, alpha=indi_alpha)
                fly_mean.append(mean)
            ax.plot(t_interp, np.nanmean(fly_mean, axis=0), lw=lw, c=color, zorder=1)

            if show_all:
                t_interp = np.arange(-t_precede-0.3, t_follow-0.11, dt)
                y_interp = np.zeros((len(data[col].t_tif), len(t_interp)))
                for i1 in range(len(y_interp)):
                    for i2 in range(len(y_interp[i1])):
                        y_interp[i1][i2] = np.nan

                for i in range(len(data[col].t_tif)):
                    if row == 0:
                        t = data[col].t_tif[i]
                        y = data[col].imaging[i]
                    elif row == 3:
                        t = data[col].t_abf[i]
                        y = data[col].xstim[i]
                    elif row == 1:
                        t = data[col].t_abf[i]
                        y = (data[col].lwa[i] + data[col].rwa[i]) / 2.0
                    elif row == 2:
                        t = data[col].t_abf[i]
                        y = data[col].lwa[i] - data[col].rwa[i]

                    if len(t) == 0:
                        continue
                    f = interp1d(t, y, kind='linear')
                    # print t[0], t[-1], t_interp[0], t_interp[-1]
                    y_interp[i] = f(t_interp)
                    ax.plot(t, y, c='grey', alpha=.2, zorder=1)
                mean = np.nanmean(y_interp, axis=0)
                c = seq_dark_colours()[col]
                ax.plot(t_interp, mean, lw=2, c=c, zorder=2)
                if errorbar:
                    sem = fc.nansem(y_interp, axis=0)
                    ax.fill_between(t_interp, mean-sem, mean+sem, edgecolor='none', facecolor=c, alpha=.4)

            ax.patch.set_facecolor('white')
            ax.grid(which='major', alpha=0)
            show_axes = []
            if row == 3:
                show_axes.append('bottom')
                ax.axhline(y=0, ls='--', c='black', zorder=1, lw=1, dashes=[5, 5])
            if row == 0:
                ax.set_xticklabels([])
                ax.set_xlabel(subtitle_list[col])
                ax.xaxis.set_label_coords(0.5, 1.4)
            if col == 0:
                show_axes.append('left')
                ax.set_ylabel(ylabel_list[row])
                ax.get_yaxis().set_label_coords(-0.5, 0.6)
            ph.adjust_spines(ax, show_axes, lw=.25, xlim=[-t_precede, t_follow], ylim=(ylim[row]),
                             yticks=(yticks[row]), xticks=(np.arange(-t_precede, t_follow-0.12, 2)))
            ax.patch.set_facecolor('white')
            ax.grid(which='major', alpha=0)

            if (row == 0) and (col == 0):
                xlim_text = ax.get_xlim()
                ylim_text = ax.get_ylim()
                text = str(len(group)) + 'flies'
                ax.text(xlim_text[0]*.95+xlim_text[-1]*.05, ylim_text[0]*.15+ylim_text[-1]*.85, text, ha='left', va='bottom', fontsize=fs)
            # for x in axvline:
            #     ax.axvline(x=x, ls='--', c='black', zorder=1, lw=1, dashes=[5, 5])
            ax.fill_between(axvline, ylim[row][0], ylim[row][-1], edgecolor='none', facecolor=ph.grey1, alpha=1)

    fig.text(0.5, 0.01, 'Time (s)', horizontalalignment='center', verticalalignment='bottom', fontsize=fs)
    # fig.text(0.5, 1, 'lobe number = %i' % (glomerulus_num+1), horizontalalignment='center', verticalalignment='bottom', fontsize='30')
    if title:
        fig.suptitle(title, fontsize=fs)

def plot_stim_bouts_statedependent_paper(group, stim_type='pro_reg_v2', loco_type='flight', t_precede=2, t_follow=4,
                                         signal_type='fb.c1.mean_an', twindow_shade=[.17,2],
                                         show_timebar=True, time_bar_length=1, time_text='1 s',
                                         show_ybar=True,
                                         show_legend=True, legend_list=['reg.', 'static', 'prog.'],
                                         leg_loc='upper left', bbox_to_anchor=(0,1), fs_legend=10,
                                         flying_cri=True, standing_cri=False, show_indi=False, show_flynum=True,
                                         colors = [ph.dred_pure, 'black', ph.dblue_pure],indi_alpha=0.5, indi_lw=1,
                                         lw=3, alpha_sem=.2, ylim=[-1, 3], fs=12,):
    def standing_or_not():
        thresh_dhead_standing = 15
        thresh_dforw_standing = 2
        if (np.max(np.abs(data[i_stim].dhead[i])) <= thresh_dhead_standing) &\
                (np.max(np.abs(data[i_stim].dforw[i])) <= thresh_dforw_standing):
            return True
        return False

    ph.set_fontsize(fs)
    fig = plt.figure(1, (2, 2.5))
    gs = gridspec.GridSpec(1, 1)
    if loco_type == 'flight':
        data = get_stim_bouts_data(group, stim_type, signal_type, t_precede)
    else:
        data = get_stim_bouts_data_walking(group, stim_type, signal_type, t_precede, t_follow)
    stim_num = 3
    dt = 0.05
    fly_num = len(group)
    ax = plt.subplot(gs[0, 0])

    t_interp = np.arange(-t_precede - 0.3, t_follow - 0.11, dt)
    for i_stim in range(stim_num):
        data_onestim = []
        color = colors[i_stim]

        for i_fly in range(fly_num):

            y_interp = np.zeros((len(data[i_stim].t_tif), len(t_interp)))
            for i1 in range(len(y_interp)):
                for i2 in range(len(y_interp[i1])):
                    y_interp[i1][i2] = np.nan
            for i in range(len(data[i_stim].t_tif)):
                if flying_cri:
                    if data[i_stim].stop_detection[i]:
                        continue
                if standing_cri:
                    if not standing_or_not():
                        continue

                if data[i_stim].fly_num[i] == i_fly:
                    t = data[i_stim].t_tif[i]
                    y = data[i_stim].imaging[i]
                    f = interp1d(t, y, kind='linear')
                    y_interp[i] = f(t_interp)

            mean_onefly = np.nanmean(y_interp, axis=0)
            if show_indi:
                ax.plot(t, y, c=color, alpha=.2, zorder=1)
                ax.plot(t_interp, mean_onefly, lw=indi_lw, c=color, zorder=1, alpha=indi_alpha)
            data_onestim.append(mean_onefly)

        mean_stim = np.nanmean(data_onestim, axis=0)
        sem_stim = np.nanstd(data_onestim, axis=0) / np.sqrt(len(data_onestim))
        ax.plot(t_interp, mean_stim, lw=lw, c=color, zorder=2, label=legend_list[i_stim])
        ax.fill_between(t_interp, mean_stim-sem_stim, mean_stim+sem_stim, edgecolor='none', facecolor=color,
                        alpha=alpha_sem, zorder=2)

    ax.patch.set_facecolor('white')
    ax.grid(which='major', alpha=0)
    ph.adjust_spines(ax, [], lw=1, xlim=[-t_precede, t_follow], xticks=[], ylim=ylim, yticks=[])

    if show_timebar:
        ph.plot_x_scale_text(ax, bar_length=time_bar_length, text=time_text, y_text_lower=.2)

    if show_ybar:
        bar_length = 1
        ybar_text = '$\Delta F/F_0$' + ' 1'
        xlim = [-t_precede, t_follow]
        y0 = ylim[0]
        y1 = ylim[0] + bar_length
        x_bar = xlim[0] - (xlim[-1] - xlim[0]) * .1
        x_text = xlim[0] - (xlim[-1] - xlim[0]) * .13
        y_text = ylim[0]
        ax.plot([x_bar, x_bar], [y0, y1], c='black', lw=2.5, solid_capstyle='butt', clip_on=False)
        ax.text(x_text, y_text, ybar_text, rotation='horizontal', ha='right', va='bottom')

    if show_flynum:
        xlim_text = ax.get_xlim()
        ylim_text = ax.get_ylim()
        text = str(len(group)) + 'flies'
        ax.text(xlim_text[0]*.95+xlim_text[-1]*.05, ylim_text[0]*.95+ylim_text[-1]*.05, text, ha='left', va='bottom', fontsize=fs)

    if show_legend:
        if bbox_to_anchor:
            ax.legend(prop={'size': fs_legend}, loc=leg_loc, bbox_to_anchor=bbox_to_anchor, ncol=1, frameon=False)
        else:
            ax.legend(prop={'size': fs_legend}, loc=leg_loc, ncol=1, frameon=False)

    if len(twindow_shade):
        ax.fill_between(twindow_shade, ylim[0], ylim[-1], edgecolor='none', facecolor=ph.grey05, alpha=1, zorder=0)


def plot_stim_bouts_subtract_two_phases(group, stim_type='heading_vs_head', t_precede=2, t_follow=6,
                                        signal_types=['fb.c1.phase' 'eb.c1.phase'], imaging_type='phase',
                                        title='', show_all=False, errorbar=False, axvline=[], flying_cri=True,
                                        indi_alpha=0.5, indi_lw=1, lw=3,
                                        ylim=[[-180, 180], [20, 80], [-45, 45], [0, 10]],
                                        yticks=[[-180,0,180], np.arange(20, 81, 20), np.arange(-30,31,30), np.arange(0,5,10)],
                                        glomerulus_num=0):
    fs = 12
    ph.set_fontsize(fs)
    size = (7, 8)
    fig = plt.figure(1, size)
    if stim_type == 'screen':
        stim_num = 10
        subtitle_list = ['progressive', 'regressive', 'yaw leftward', 'yaw rightward', 'roll rightward',
                         'roll leftward', 'pitch downward', 'pitch upward', 'lift downward', 'lift upward']
    elif stim_type == 'speed':
        stim_num = 7
        subtitle_list = ['speed: - 0.33 m/s', 'speed: 0.11 m/s', 'speed: 0.22 m/s', 'speed: 0.33 m/s', 'speed: 0.44 m/s', 'speed: 0.55 m/s', 'speed: 0.66 m/s']
    elif stim_type == 'offcenter':
        stim_num = 9
        subtitle_list = ['-90', '-67.5', '-45', '-22.5', '0', '22.5', '45', '67.5', '90']
    elif stim_type == 'pro_reg':
        stim_num = 4
        subtitle_list = ['speed: -0.30 m/s', 'speed: -0.15 m/s', 'speed: 0.15 m/s', 'speed: 0.30 m/s']
    elif stim_type == 'pro_reg_v2':
        stim_num = 3
        subtitle_list = ['velocity: -0.20 m/s', 'velocity: 0 m/s', 'velocity: 0.20 m/s']
    elif stim_type == 'heading_vs_head':
        stim_num = 7
        subtitle_list = ['-90 deg', '-40 deg', '0 deg', '+40 deg', '+90 deg', 'LW yaw', 'RW yaw']
    elif stim_type == 'heading_vs_head_v2':
        stim_num = 8
        subtitle_list = ['-90 deg', '-40 deg', '0 deg', '+40 deg', '+90 deg', 'LW yaw', 'RW yaw', '-180 deg']
    elif stim_type == 'heading_vs_head_v4':
        stim_num = 5
        subtitle_list = ['-90 deg', '-45 deg', '0 deg', '+45 deg', '+90 deg']
    else:
        print 'Error: does not have this type of stim_type'
        return

    gs = gridspec.GridSpec(4, stim_num)
    fly_num = len(group)
    color = ph.green

    if 'phase' in signal_types[0]:
        ylabel_list = ['$\Delta$ phase\nFB-EB\n(deg)', 'L+R WA\n(deg)', 'L-R WA\n(deg)', 'stim x\n(volt)'] # for phase
    else:
        ylabel_list = ['$\Delta F/F$', 'L+R WA\n(deg)', 'L-R WA\n(deg)', 'stim x\n(volt)']

    data = get_stim_bouts_data(group, stim_type, signal_types[0], t_precede, t_follow, imaging_type='phase')
    data_add = get_stim_bouts_data(group, stim_type, signal_types[1], t_precede, t_follow, imaging_type='phase')

    for row, ylabel in enumerate(ylabel_list):
        for col, xlabel in enumerate(subtitle_list):
            ax = plt.subplot(gs[row, col])
            # Plot mean bout curves
            if row == 0:
                dt = 0.05
            else:
                dt = 0.02

            fly_mean = []
            t_interp = np.arange(-t_precede - 0.3, t_follow - 0.11, dt)
            for i_fly in range(fly_num):
                y_interp = np.zeros((len(data[col].t_tif), len(t_interp)))
                for i1 in range(len(y_interp)):
                    for i2 in range(len(y_interp[i1])):
                        y_interp[i1][i2] = np.nan
                for i in range(len(data[col].t_tif)):
                    if flying_cri:
                        if data[col].stop_detection[i]:
                            continue
                    if data[col].fly_num[i] == i_fly:
                        if row == 0:
                            t0 = data[col].t_tif[i]
                            t1 = data_add[col].t_tif[i]
                            y0 = data[col].imaging[i]
                            y1 = data_add[col].imaging[i]
                            f0 = interp1d(t0, y0, kind='linear', axis=0)
                            f1 = interp1d(t1, y1, kind='linear', axis=0)
                            y_interp0 = f0(t_interp)
                            y_interp1 = f1(t_interp)
                            y_interp[i] = y_interp0 - y_interp1
                        else:
                            if row == 3:
                                t = data[col].t_abf[i]
                                y = data[col].xstim[i]
                            elif row == 1:
                                t = data[col].t_abf[i]
                                y = (data[col].lwa[i] + data[col].rwa[i]) / 2.0
                            elif row == 2:
                                t = data[col].t_abf[i]
                                y = data[col].lwa[i] - data[col].rwa[i]
                            f = interp1d(t, y, kind='linear')
                            y_interp[i] = f(t_interp)
                mean = np.nanmean(y_interp, axis=0)
                ax.plot(t_interp, mean, lw=indi_lw, c=color, zorder=1, alpha=indi_alpha)
                fly_mean.append(mean)
            ax.plot(t_interp, np.nanmean(fly_mean, axis=0), lw=lw, c=color, zorder=1)

            if show_all:
                t_interp = np.arange(-t_precede-0.3, t_follow-0.11, dt)
                y_interp = np.zeros((len(data[col].t_tif), len(t_interp)))
                for i1 in range(len(y_interp)):
                    for i2 in range(len(y_interp[i1])):
                        y_interp[i1][i2] = np.nan

                for i in range(len(data[col].t_tif)):
                    if row == 0:
                        t0 = data[col].t_tif[i]
                        t1 = data_add[col].t_tif[i]
                        y0 = data[col].imaging[i]
                        y1 = data_add[col].imaging[i]
                        f0 = interp1d(t0, y0, kind='linear', axis=0)
                        f1 = interp1d(t1, y1, kind='linear', axis=0)
                        y_interp0 = f0(t_interp)
                        y_interp1 = f1(t_interp)
                        y_interp[i] = y_interp0 - y_interp1
                        ax.plot(t, y0, c='grey', alpha=.2, zorder=1)
                        ax.plot(t, y1, c='blue', alpha=.2, zorder=1)
                    else:
                        if row == 3:
                            t = data[col].t_abf[i]
                            y = data[col].xstim[i]
                        elif row == 1:
                            t = data[col].t_abf[i]
                            y = (data[col].lwa[i] + data[col].rwa[i]) / 2.0
                        elif row == 2:
                            t = data[col].t_abf[i]
                            y = data[col].lwa[i] - data[col].rwa[i]

                        if len(t) == 0:
                            continue
                        f = interp1d(t, y, kind='linear')
                        # print t[0], t[-1], t_interp[0], t_interp[-1]
                        y_interp[i] = f(t_interp)
                        ax.plot(t, y, c='grey', alpha=.2, zorder=1)
                mean = np.nanmean(y_interp, axis=0)
                c = seq_dark_colours()[col]
                ax.plot(t_interp, mean, lw=2, c=c, zorder=2)
                if errorbar:
                    sem = fc.nansem(y_interp, axis=0)
                    ax.fill_between(t_interp, mean-sem, mean+sem, edgecolor='none', facecolor=c, alpha=.4)

            ax.patch.set_facecolor('white')
            ax.grid(which='major', alpha=0)
            show_axes = []
            if row == 3:
                show_axes.append('bottom')
                ax.axhline(y=0, ls='--', c='black', zorder=1, lw=1, dashes=[5, 5])
            if row == 0:
                ax.set_xticklabels([])
                ax.set_xlabel(subtitle_list[col])
                ax.xaxis.set_label_coords(0.5, 1.4)
            if col == 0:
                show_axes.append('left')
                ax.set_ylabel(ylabel_list[row])
                ax.get_yaxis().set_label_coords(-0.4, 0.6)
            ph.adjust_spines(ax, show_axes, lw=.25, xlim=[-t_precede, t_follow], ylim=(ylim[row]),
                             yticks=(yticks[row]), xticks=(np.arange(-t_precede, t_follow-0.12, 2)))

            if (row == 0) and (col == 0):
                xlim_text = ax.get_xlim()
                ylim_text = ax.get_ylim()
                text = str(len(group)) + 'flies'
                ax.text(xlim_text[0]*.95+xlim_text[-1]*.05, ylim_text[0]*.15+ylim_text[-1]*.85, text, ha='left', va='bottom', fontsize=11)
            ax.fill_between(axvline, ylim[row][0], ylim[row][-1], edgecolor='none', facecolor=ph.grey1, alpha=1)

    fig.text(0.5, 0.01, 'Time (s)', horizontalalignment='center', verticalalignment='bottom', fontsize=fs)
    # fig.text(0.5, 1, 'lobe number = %i' % (glomerulus_num+1), horizontalalignment='center', verticalalignment='bottom', fontsize='30')
    if title:
        fig.suptitle(title, fontsize=fs)


def plot_stim_bouts_slope(group, stim_type='speed_6', t_precede=3, t_window_slope=1, signal_type='fb.c1.mean_al',
                          ylim=[], ms=5):

    if stim_type == 'screen':
        stim_num = 10
        subtitle_list = ['progressive', 'regressive', 'yaw leftward', 'yaw rightward', 'roll rightward',
                         'roll leftward', 'pitch downward', 'pitch upward', 'lift downward', 'lift upward']
    elif stim_type == 'speed':
        stim_num = 9
        subtitle_list = ['speed: - 0.15 m/s', 'speed: 0.05 m/s', 'speed: 0.10 m/s', 'speed: 0.15 m/s',
                         'speed: 0.20 m/s', 'speed: 0.25 m/s', 'speed: 0.30 m/s', 'speed: 0.35 m/s', 'speed: 0.40 m/s']
    elif stim_type == 'speed_6':
        stim_num = 7
        subtitle_list = ['-30', '10', '20', '30', '40', '50', '60']
    elif stim_type == 'offcenter':
        stim_num = 9
        subtitle_list = ['-90', '-67.5', '-45', '-22.5', '0', '22.5', '45', '67.5', '90']
    else:
        print 'Error: does not have this type of stim_type'
        return

    data = get_stim_bouts_data(group, stim_type, signal_type, t_precede)

    rcParams['ytick.labelsize'] = 20
    rcParams['xtick.labelsize'] = 20
    # rcParams['font.size'] = 25
    rcParams['axes.labelsize'] = 20
    fig = plt.figure(1, (13, 6))
    ax = plt.subplot(111)

    fly_num = len(group)
    xaxis_ticks = np.arange(stim_num) + 0.5
    data_population = []
    for i in range(stim_num):
        data_population.append([])
    t_interp = np.arange(0, t_window_slope, 0.05)

    # extract data
    for i_fly in range(fly_num):
        data_fly = []
        for i in range(stim_num):
            data_fly.append([])
        # go through the whole data for each fly
        for col in range(stim_num):
            for i in range(len(data[col].stop_detection)):
                if not data[col].stop_detection[i]:
                    if data[col].fly_num[i] == i_fly:
                        t = data[col].t_tif[i]
                        y = data[col].imaging[i]
                        f = interp1d(t, y, kind='linear')
                        y_interp = f(t_interp)
                        data_fly[col].append(y_interp)
        # average the samples inside each fly
        for col in range(stim_num):
            if len(data_fly[col]):
                y_mean = np.nanmean(data_fly[col], axis=0)
                slope, intercept, r_value, p_value, std_err = stats.linregress(t_interp, y_mean)
                data_population[col].append(slope)

    # plot errorbar with individuals
    for i_fly in range(fly_num):
        color = seq_dark_colours()[i_fly]
        for col in range(len(data)):
            x_random = (random.random()-0.5) * 0.2 + xaxis_ticks[col]
            if i_fly < len(data_population[col]):
                ax.plot(x_random, data_population[col][i_fly], 'o', ms=ms, c=color)
    mean = np.zeros(stim_num)
    sem = np.zeros(stim_num)
    for col in range(stim_num):
        mean[col] = np.nanmean(data_population[col])
        sem[col] = np.nanstd(data_population[col]) / np.sqrt(np.sum(~np.isnan(data_population[col])))
    ax.errorbar(xaxis_ticks, mean, sem, fmt='o', c='grey', ms=ms + 2)

    # set axis and texts
    ax.patch.set_facecolor('white')
    ax.axhline(ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
    ax.set_xticks(xaxis_ticks)
    ax.set_xticklabels(subtitle_list)
    # for label in ax.xaxis.get_ticklabels():
    #     label.set_rotation(45)
    ax.set_xlim(0, xaxis_ticks[-1]+.5)
    if ylim:
        ax.set_ylim(ylim[:-1])
        ax.set_yticks(np.arange(ylim[0], ylim[1] + ylim[2]/2.0, ylim[2]))
    ax.set_xlabel('speed of progressive optic flow (cm/s)')
    ax.set_ylabel('slope of signal intensity')
    ax.grid(which='major', alpha=0)
    # fig.suptitle('firing rate of the speed neuron increases with the speed of progressive optic flow', fontsize=15)


def plot_stim_bout_individual_trace(group, stim_type='heading_vs_head', t_precede=2, t_follow=6,
                                    signal_types=['fb.c1.al', 'eb.c1.al'], phase_types=['fb.c1.phase', 'eb.c1.phase'], title='', flying_cri=True, i_data=0,
                                    axvline=[0], lw=1.5, imaging_type='tif',
                                    ylim=[[0, 16], [0, 16], [-360, 360], [30, 90], [30, 90]],
                                    yticks=[[0, 16], 'none', [-360, 360], 'none', [30, 90]]):
    ph.set_fontsize(14)
    data = []           # tif data
    data_phase = []     # phase data
    for i_signal_type in range(len(signal_types)):
        data.append([])
        data_phase.append([])
        data[i_signal_type] = get_stim_bouts_data(group, stim_type, signal_types[i_signal_type], t_precede, t_follow,
                                                  imaging_type='tif')
        data_phase[i_signal_type] = get_stim_bouts_data(group, stim_type, phase_types[i_signal_type], t_precede, t_follow,
                                                  imaging_type='phase')
    if not len(data[0][i_data].imaging):
        print 'No valid data'
        return

    # plotting related params
    xedges = np.arange(data[0][i_data].imaging[0].shape[-1]+1)
    total_trial_num = -1
    if flying_cri:
        for i in range(len(data[0][i_data].lwa)):
            if not data[0][i_data].stop_detection[i]:
                total_trial_num += 1
    else:
        total_trial_num = len(data[0][i_data].lwa)
    col_num = 4
    row_num = total_trial_num/col_num + 1
    num_per_row = 5
    size = (13, 9*row_num)
    fig = plt.figure(1, size)
    hr_unit = [4, 4, 2.5, 1.5, 1.5]
    height_ratios = []
    for i in range(row_num):
        height_ratios.extend(hr_unit)
    gs = gridspec.GridSpec(row_num * num_per_row, col_num, height_ratios=height_ratios)
    color = 'black'
    cmap = plt.cm.Blues
    ylabel_list = ['PFR-FB', 'EPG-EB', '$\Delta\phi$\nFB-EB', 'LWA\n(deg)', 'RWA\n(deg)']

    # plot
    trial_count = 0
    for i_trial in range(len(data[0][i_data].lwa)):         # each trial
        if flying_cri:
            if data[0][i_data].stop_detection[i_trial]:
                continue
        row = trial_count/col_num
        col = trial_count%col_num
        trial_count += 1

        for sub_row in range(num_per_row):                  # each sub row, within each trial
            ax = plt.subplot(gs[row*num_per_row+sub_row, col])
            if sub_row <= 2:
                dt = 0.2
            else:
                dt = 0.02
            t_interp = np.arange(-t_precede - 0.3, t_follow - 0.11, dt)

            if sub_row <= 1:                                # plot tif
                t = data[sub_row][i_data].t_tif[i_trial]
                y = data[sub_row][i_data].imaging[i_trial]
                f = interp1d(t, y, kind='linear', axis=0)
                y_interp = f(t_interp)
                plt.pcolormesh(t_interp, xedges, y_interp.T, cmap=cmap)
            elif sub_row == 2:                              # plot two phases subtraction
                t0 = data_phase[0][i_data].t_tif[i_trial]
                t1 = data_phase[1][i_data].t_tif[i_trial]
                y0 = data_phase[0][i_data].imaging[i_trial]
                y1 = data_phase[1][i_data].imaging[i_trial]
                f0 = interp1d(t0, y0, kind='linear', axis=0)
                f1 = interp1d(t1, y1, kind='linear', axis=0)
                y_interp0 = f0(t_interp)
                y_interp1 = f1(t_interp)
                ax.plot(t_interp, y_interp0-y_interp1, c=color, lw=lw)
                ax.axhline(0, ls='--', c=ph.grey3, zorder=1, lw=1, dashes=[5, 5])
                ax.patch.set_facecolor('white')
                ax.grid(which='major', alpha=0)
            else:                                           # plot wing beat amplitude
                t = data[0][i_data].t_abf[i_trial]
                if sub_row == 3:
                    y = data[0][i_data].lwa[i_trial]
                else:
                    y = data[0][i_data].rwa[i_trial]
                f = interp1d(t, y, kind='linear', axis=0)
                y_interp = f(t_interp)
                ax.plot(t_interp, y_interp, c=color, lw=lw)
                ax.patch.set_facecolor('white')
                ax.grid(which='major', alpha=0)

            show_axes = []
            ytick = 'none'
            if (row == row_num-1) and (sub_row == num_per_row-1):
                show_axes.append('bottom')
            if col == 0:
                show_axes.append('left')
                if row == (row_num-1):
                    ax.set_ylabel(ylabel_list[sub_row])
                    ax.get_yaxis().set_label_coords(-0.3, 0.6)
                    ytick = yticks[sub_row]
            ph.adjust_spines(ax, show_axes, lw=.25, xlim=[-t_precede, t_follow], ylim=(ylim[sub_row]),
                             yticks=ytick, xticks=(np.arange(-t_precede, t_follow - 0.12, 2)))
            for x in axvline:
                ax.axvline(x=x, ls='--', c=ph.grey6, zorder=1, lw=1, dashes=[5, 5])

        ax.text(0, 85, '%i,%i,%3.1f' % (data[0][i_data].fly_num[i_trial], data[0][i_data].fileinfo[i_trial][0],
                                         data[0][i_data].fileinfo[i_trial][1]), fontsize=13)

    if title:
        fig.suptitle(title, fontsize='20', y=0.91)


# Stimuli Screen Walking

def plot_stim_bouts_walking_nobehavior(group, stim_type='OLOF6', t_precede=2, t_follow=2, signal_type='mean_pl',
                            title='', stim_dur=0.8,
                            lw_individ = 1, lw=3,
                            xlim=[-2, 2], ylim=[0, 1], xticks=[-2,0,2], yticks=[0, 1]):
    # corresponding to the 2p data collected AFTER March, 2017
    thresh_dhead_standing = 15
    thresh_dforw_standing = 2
    thresh_dforw_forwarding = 6
    dt = 0.05
    t_interp = np.arange(xlim[0]-0.2, xlim[-1]+0.2, dt)
    c = ph.green

    def plot_temporal_curve():
        # plot the average of individual fly and population
        fly_num = len(group)
        fly_data = []
        population_data = []
        for i_fly in range(fly_num):
            for i_trial in range(len(sig)):
                if data[col].fly_num[i_trial] == i_fly:
                    if class_criteria(i_trial):
                        fly_data.append(y_interp[i_trial])
            if len(fly_data):
                fly_mean = np.nanmean(fly_data, axis=0)
                population_data.append(fly_mean)
                fly_data = []
                ax.plot(t_interp, fly_mean, lw=lw_individ, c=c, alpha=0.3, zorder=2)
        if len(population_data):
            population_mean = np.nanmean(population_data, axis=0)
            ax.plot(t_interp, population_mean, lw=lw, c=c, alpha=1, zorder=1)

    def class_criteria(i_trial):
        t_tif = data[col].t_tif[i_trial]
        dhead = data[col].dhead[i_trial]
        dforw = data[col].dforw[i_trial]
        ind_0 = np.where(t_tif >= 0)[0][0]
        ind_s = np.where(t_tif >= xlim[0])[0][0]
        ind_e = np.where(t_tif >= xlim[-1])[0][0]
        ind_stim = np.where(t_tif >= stim_dur)[0][0]
        standing = (np.max(np.abs(dhead[ind_s:ind_e])) <= thresh_dhead_standing) \
                   and (np.max(np.abs(dforw[ind_s:ind_e])) <= thresh_dforw_standing)
        forwarding = np.max(dforw[ind_0:ind_stim]) >= thresh_dforw_forwarding
        if row == 0:
            if standing:
                return True
            else:
                return False
        elif row == 1:
            if forwarding:
                return True
            else:
                return False
        elif row == 2:
            if (not standing) and (not forwarding):
                return True
            else:
                return False
        return False

    ph.set_fontsize(20)
    size = (30, 16)
    fig = plt.figure(1, size)
    if stim_type == 'OLOF6':
        stim_num = 6
        subtitle_list = ['Regressive\nspeed: -15cm/s', 'Progressive\nspeed: 15cm/s', 'Regressive\nspeed: -30cm/s',
                         'Progressive\nspeed: 30cm/s', 'Yaw Leftward', 'Yaw Rightward']
        ylabel_list = ['Standing', 'Forwarding', 'Moving but\nnot forwarding']
    else:
        print 'Error: does not have this type of stim_type'
        return
    gs = gridspec.GridSpec(3, stim_num)
    data = get_stim_bouts_data_walking(group, stim_type, signal_type, t_precede, t_follow)

    for row, ylabel in enumerate(ylabel_list):
        for col, xlabel in enumerate(subtitle_list):
            ax = plt.subplot(gs[row, col])

            sig = data[col].imaging
            y_interp = np.zeros((len(data[col].t_tif), len(t_interp)))
            for i, y in enumerate(sig):
                t = data[col].t_tif[i]
                if len(t) == 0:
                    continue
                f = interp1d(t, y, kind='linear')
                y_interp[i] = f(t_interp)

            plot_temporal_curve()

            show_axes = []
            if row == 2:
                show_axes.append('bottom')
            if row == 0:
                ax.set_xlabel(subtitle_list[col])
                ax.xaxis.set_label_coords(0.5, 1.2)
            if col == 0:
                show_axes.append('left')
                ax.set_ylabel(ylabel_list[row])
            ph.adjust_spines(ax, show_axes, lw=.25, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
            # ax.axvline(ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
            # ax.axvline(stim_dur, ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
            ax.patch.set_facecolor('white')
            ax.grid(which='major', alpha=0)
            ax.fill_between([0,stim_dur], ylim[0], ylim[1], facecolor=ph.grey2, alpha=0.2)

    fig.text(0.5, 0.07, 'Time (s)', horizontalalignment='center', verticalalignment='bottom', fontsize='20')
    if title:
        fig.suptitle(title, fontsize='20')


def plot_stim_bouts_walking_per_behavior_type(group, stim_type='OLOF6', t_precede=2, t_follow=2, signal_type='mean_pl',
                                              title='', stim_dur=0.8, behavior_type=0, lw_individ = 1, lw=3,
                                              xlim=[-2, 2], ylim=[[0,1],[-50,50],[-20,20]],
                                              xticks=[-2,0,2], yticks=[[0,1],[-50,50],[-20,20]]):
    # corresponding to the 2p data collected AFTER March, 2017
    thresh_dhead_standing = 15
    thresh_dforw_standing = 2
    thresh_dforw_forwarding = 6

    c = ph.green

    def plot_temporal_curve():
        # plot the average of individual fly and population
        fly_num = len(group)
        fly_data = []
        population_data = []
        for i_fly in range(fly_num):
            for i_trial in range(len(data[col].t_tif)):
                if data[col].fly_num[i_trial] == i_fly:
                    if class_criteria(i_trial):
                        fly_data.append(y_interp[i_trial])
            if len(fly_data):
                fly_mean = np.nanmean(fly_data, axis=0)
                population_data.append(fly_mean)
                fly_data = []
                ax.plot(t_interp, fly_mean, lw=lw_individ, c=c, alpha=0.3, zorder=2)
        if len(population_data):
            population_mean = np.nanmean(population_data, axis=0)
            ax.plot(t_interp, population_mean, lw=lw, c=c, alpha=1, zorder=1)

    def class_criteria(i_trial):
        t_tif = data[col].t_tif[i_trial]
        dhead = data[col].dhead[i_trial]
        dforw = data[col].dforw[i_trial]
        ind_0 = np.where(t_tif >= 0)[0][0]
        ind_s = np.where(t_tif >= xlim[0])[0][0]
        ind_e = np.where(t_tif >= xlim[-1])[0][0]
        ind_stim = np.where(t_tif >= stim_dur)[0][0]
        standing = (np.max(np.abs(dhead[ind_s:ind_e])) <= thresh_dhead_standing) \
                   and (np.max(np.abs(dforw[ind_s:ind_e])) <= thresh_dforw_standing)
        forwarding = np.max(dforw[ind_0:ind_stim]) >= thresh_dforw_forwarding
        if behavior_type == 0:
            if standing:
                return True
            else:
                return False
        elif behavior_type == 1:
            if forwarding:
                return True
            else:
                return False
        elif behavior_type == 2:
            if (not standing) and (not forwarding):
                return True
            else:
                return False
        return False

    ph.set_fontsize(20)
    size = (30, 16)
    fig = plt.figure(1, size)
    if stim_type == 'OLOF6':
        stim_num = 6
        subtitle_list = ['Regressive\nspeed: -15cm/s', 'Progressive\nspeed: 15cm/s', 'Regressive\nspeed: -30cm/s',
                         'Progressive\nspeed: 30cm/s', 'Yaw Leftward', 'Yaw Rightward']
        ylabel_list = ['$\Delta F/F$', 'Heading\n(deg)', 'Forward\n(mm)']
    else:
        print 'Error: does not have this type of stim_type'
        return
    gs = gridspec.GridSpec(3, stim_num)
    data = get_stim_bouts_data_walking(group, stim_type, signal_type, t_precede, t_follow)

    for row, ylabel in enumerate(ylabel_list):
        for col, xlabel in enumerate(subtitle_list):
            ax = plt.subplot(gs[row, col])

            if row == 0:
                dt = 0.05
            else:
                dt = 0.02
            t_interp = np.arange(xlim[0] - 0.2, xlim[-1] + 0.2, dt)
            y_interp = np.zeros((len(data[col].t_tif), len(t_interp)))
            for i in range(len(data[col].t_tif)):
                if row == 0:
                    t = data[col].t_tif[i]
                    y = data[col].imaging[i]
                    if len(t) == 0:
                        continue
                elif row == 1:
                    t = data[col].t_abf[i]
                    y = data[col].head[i]
                elif row == 2:
                    t = data[col].t_abf[i]
                    y = fc.deg2mm(data[col].forw[i])
                f = interp1d(t, y, kind='linear')
                y_interp[i] = f(t_interp)

            plot_temporal_curve()

            show_axes = []
            if row == 2:
                show_axes.append('bottom')
            if row == 0:
                ax.set_xlabel(subtitle_list[col])
                ax.xaxis.set_label_coords(0.5, 1.2)
            if col == 0:
                show_axes.append('left')
                ax.set_ylabel(ylabel_list[row])
            ph.adjust_spines(ax, show_axes, lw=.25, xlim=xlim, ylim=ylim[row], xticks=xticks, yticks=yticks[row])
            # ax.axvline(ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
            # ax.axvline(stim_dur, ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
            ax.patch.set_facecolor('white')
            ax.grid(which='major', alpha=0)
            ax.fill_between([0,stim_dur], ylim[row][0], ylim[row][1], facecolor=ph.grey2, alpha=0.2)

    fig.text(0.5, 0.07, 'Time (s)', horizontalalignment='center', verticalalignment='bottom', fontsize='20')
    if title:
        fig.suptitle(title, fontsize='20')


def plot_stim_bouts_walking_overlap(group, stim_type='OLOF6', t_precede=2, t_follow=2, signal_type='mean_pl',
                                              title='', stim_dur=0.8, lw_individ = 1, lw=3,
                                              xlim=[-2, 2], ylim=[[0,1],[-50,50],[-20,20]],
                                              xticks=[-2,0,2], yticks=[[0,1],[-50,50],[-20,20]]):
    # corresponding to the 2p data collected AFTER March, 2017
    thresh_dhead_standing = 15
    thresh_dforw_standing = 2
    thresh_dforw_forwarding = 6

    def plot_temporal_curve():
        # plot the average of individual fly and population
        fly_num = len(group)
        fly_data = []
        population_data = []
        for i_fly in range(fly_num):
            for i_trial in range(len(data[col].t_tif)):
                if data[col].fly_num[i_trial] == i_fly:
                    if class_criteria(i_trial):
                        fly_data.append(y_interp[i_trial])
            if len(fly_data):
                fly_mean = np.nanmean(fly_data, axis=0)
                population_data.append(fly_mean)
                fly_data = []
                ax.plot(t_interp, fly_mean, lw=lw_individ, c=c, alpha=0.3, zorder=2)
        if len(population_data):
            population_mean = np.nanmean(population_data, axis=0)
            ax.plot(t_interp, population_mean, lw=lw, c=c, alpha=1, zorder=1)

    def class_criteria(i_trial):
        t_tif = data[col].t_tif[i_trial]
        dhead = data[col].dhead[i_trial]
        dforw = data[col].dforw[i_trial]
        ind_0 = np.where(t_tif >= 0)[0][0]
        ind_s = np.where(t_tif >= xlim[0])[0][0]
        ind_e = np.where(t_tif >= xlim[-1])[0][0]
        ind_stim = np.where(t_tif >= stim_dur)[0][0]
        standing = (np.max(np.abs(dhead[ind_s:ind_e])) <= thresh_dhead_standing) \
                   and (np.max(np.abs(dforw[ind_s:ind_e])) <= thresh_dforw_standing)
        forwarding = np.max(dforw[ind_0:ind_stim]) >= thresh_dforw_forwarding
        if behavior_type == 0:
            if standing:
                return True
            else:
                return False
        elif behavior_type == 1:
            if forwarding:
                return True
            else:
                return False
        elif behavior_type == 2:
            if (not standing) and (not forwarding):
                return True
            else:
                return False
        return False

    ph.set_fontsize(40)
    size = (30, 16)
    fig = plt.figure(1, size)
    if stim_type == 'OLOF6':
        stim_num = 6
        subtitle_list = ['Regressive\nspeed: -15cm/s', 'Progressive\nspeed: 15cm/s', 'Regressive\nspeed: -30cm/s',
                         'Progressive\nspeed: 30cm/s', 'Yaw Leftward', 'Yaw Rightward']
        ylabel_list = ['$\Delta F/F$', 'Heading\n(deg)', 'Forward\n(mm)']
    else:
        print 'Error: does not have this type of stim_type'
        return
    gs = gridspec.GridSpec(3, stim_num)
    data = get_stim_bouts_data_walking(group, stim_type, signal_type, t_precede, t_follow)

    for row, ylabel in enumerate(ylabel_list):
        for col, xlabel in enumerate(subtitle_list):
            ax = plt.subplot(gs[row, col])

            c = ph.green
            behavior_type = 0
            if row == 0:
                dt = 0.05
            else:
                dt = 0.02
            t_interp = np.arange(xlim[0] - 0.2, xlim[-1] + 0.2, dt)
            y_interp = np.zeros((len(data[col].t_tif), len(t_interp)))
            for i in range(len(data[col].t_tif)):
                if row == 0:
                    t = data[col].t_tif[i]
                    y = data[col].imaging[i]
                    if len(t) == 0:
                        continue
                elif row == 1:
                    t = data[col].t_abf[i]
                    y = data[col].head[i]
                elif row == 2:
                    t = data[col].t_abf[i]
                    y = fc.deg2mm(data[col].forw[i])
                f = interp1d(t, y, kind='linear')
                y_interp[i] = f(t_interp)
            plot_temporal_curve()

            c = ph.red
            behavior_type = 1
            if row == 0:
                dt = 0.05
            else:
                dt = 0.02
            t_interp = np.arange(xlim[0] - 0.2, xlim[-1] + 0.2, dt)
            y_interp = np.zeros((len(data[col].t_tif), len(t_interp)))
            for i in range(len(data[col].t_tif)):
                if row == 0:
                    t = data[col].t_tif[i]
                    y = data[col].imaging[i]
                    if len(t) == 0:
                        continue
                elif row == 1:
                    t = data[col].t_abf[i]
                    y = data[col].head[i]
                elif row == 2:
                    t = data[col].t_abf[i]
                    y = fc.deg2mm(data[col].forw[i])
                f = interp1d(t, y, kind='linear')
                y_interp[i] = f(t_interp)
            plot_temporal_curve()

            c = ph.yellow
            behavior_type = 2
            if row == 0:
                dt = 0.05
            else:
                dt = 0.02
            t_interp = np.arange(xlim[0] - 0.2, xlim[-1] + 0.2, dt)
            y_interp = np.zeros((len(data[col].t_tif), len(t_interp)))
            for i in range(len(data[col].t_tif)):
                if row == 0:
                    t = data[col].t_tif[i]
                    y = data[col].imaging[i]
                    if len(t) == 0:
                        continue
                elif row == 1:
                    t = data[col].t_abf[i]
                    y = data[col].head[i]
                elif row == 2:
                    t = data[col].t_abf[i]
                    y = fc.deg2mm(data[col].forw[i])
                f = interp1d(t, y, kind='linear')
                y_interp[i] = f(t_interp)
            plot_temporal_curve()

            show_axes = []
            if row == 2:
                show_axes.append('bottom')
            if row == 0:
                ax.set_xlabel(subtitle_list[col])
                ax.xaxis.set_label_coords(0.5, 1.4)
            if col == 0:
                show_axes.append('left')
                ax.set_ylabel(ylabel_list[row])
                ax.get_yaxis().set_label_coords(-0.5, 0.5)
            ph.adjust_spines(ax, show_axes, lw=.25, xlim=xlim, ylim=ylim[row], xticks=xticks, yticks=yticks[row])
            # ax.axvline(ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
            # ax.axvline(stim_dur, ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
            ax.patch.set_facecolor('white')
            ax.grid(which='major', alpha=0)
            ax.fill_between([0,stim_dur], ylim[row][0], ylim[row][1], facecolor=ph.grey2, alpha=0.2)

    fig.text(0.5, 0.07, 'Time (s)', horizontalalignment='center', verticalalignment='bottom', fontsize='20')
    if title:
        fig.suptitle(title, fontsize='20')


def plot_stim_bouts_walking_overlap_bar(group, stim_type='OLOF6', t_precede=2, t_follow=2, signal_type='mean_pl',
                                              title='', stim_dur=0.8, lw_individ = 1, lw=3,
                                              xlim=[-2, 2], ylim=[[0,1],[-50,50],[-20,20]],
                                              xticks=[-2,0,2], yticks=[[0,1],[-50,50],[-20,20]]):
    # corresponding to the 2p data collected AFTER March, 2017
    thresh_dhead_standing = 15
    thresh_dforw_standing = 2
    thresh_dforw_forwarding = 6

    def plot_temporal_curve():
        # plot the average of individual fly and population
        fly_num = len(group)
        fly_data = []
        population_data = []
        for i_fly in range(fly_num):
            for i_trial in range(len(data[col].t_tif)):
                if data[col].fly_num[i_trial] == i_fly:
                    if class_criteria(i_trial):
                        fly_data.append(y_interp[i_trial])
            if len(fly_data):
                fly_mean = np.nanmean(fly_data, axis=0)
                population_data.append(fly_mean)
                fly_data = []
                ax.plot(t_interp, fly_mean, lw=lw_individ, c=c, alpha=0.3, zorder=2)
        if len(population_data):
            population_mean = np.nanmean(population_data, axis=0)
            ax.plot(t_interp, population_mean, lw=lw, c=c, alpha=1, zorder=1)

    def class_criteria(i_trial):
        t_tif = data[col].t_tif[i_trial]
        dhead = data[col].dhead[i_trial]
        dforw = data[col].dforw[i_trial]
        ind_0 = np.where(t_tif >= 0)[0][0]
        ind_s = np.where(t_tif >= xlim[0])[0][0]
        ind_e = np.where(t_tif >= xlim[-1])[0][0]
        ind_stim = np.where(t_tif >= stim_dur)[0][0]
        standing = (np.max(np.abs(dhead[ind_s:ind_e])) <= thresh_dhead_standing) \
                   and (np.max(np.abs(dforw[ind_s:ind_e])) <= thresh_dforw_standing)
        forwarding = np.max(dforw[ind_0:ind_stim]) >= thresh_dforw_forwarding
        if behavior_type == 0:
            if standing:
                return True
            else:
                return False
        elif behavior_type == 1:
            if forwarding:
                return True
            else:
                return False
        elif behavior_type == 2:
            if (not standing) and (not forwarding):
                return True
            else:
                return False
        return False

    ph.set_fontsize(20)
    size = (10, 4)
    fig = plt.figure(1, size)
    if stim_type == 'OLOF6':
        stim_num = 6
        subtitle_list = ['Regressive\nspeed: -15cm/s', 'Progressive\nspeed: 15cm/s', 'Regressive\nspeed: -30cm/s',
                         'Progressive\nspeed: 30cm/s', 'Yaw Leftward', 'Yaw Rightward']
    else:
        print 'Error: does not have this type of stim_type'
        return
    gs = gridspec.GridSpec(3, stim_num)
    data = get_stim_bouts_data_walking(group, stim_type, signal_type, t_precede, t_follow)

    for row, ylabel in enumerate(ylabel_list):
        for col, xlabel in enumerate(subtitle_list):
            ax = plt.subplot(gs[row, col])

            c = ph.green
            behavior_type = 0
            if row == 0:
                dt = 0.05
            else:
                dt = 0.02
            t_interp = np.arange(xlim[0] - 0.2, xlim[-1] + 0.2, dt)
            y_interp = np.zeros((len(data[col].t_tif), len(t_interp)))
            for i in range(len(data[col].t_tif)):
                if row == 0:
                    t = data[col].t_tif[i]
                    y = data[col].imaging[i]
                    if len(t) == 0:
                        continue
                elif row == 1:
                    t = data[col].t_abf[i]
                    y = data[col].head[i]
                elif row == 2:
                    t = data[col].t_abf[i]
                    y = fc.deg2mm(data[col].forw[i])
                f = interp1d(t, y, kind='linear')
                y_interp[i] = f(t_interp)
            plot_temporal_curve()

            c = ph.red
            behavior_type = 1
            if row == 0:
                dt = 0.05
            else:
                dt = 0.02
            t_interp = np.arange(xlim[0] - 0.2, xlim[-1] + 0.2, dt)
            y_interp = np.zeros((len(data[col].t_tif), len(t_interp)))
            for i in range(len(data[col].t_tif)):
                if row == 0:
                    t = data[col].t_tif[i]
                    y = data[col].imaging[i]
                    if len(t) == 0:
                        continue
                elif row == 1:
                    t = data[col].t_abf[i]
                    y = data[col].head[i]
                elif row == 2:
                    t = data[col].t_abf[i]
                    y = fc.deg2mm(data[col].forw[i])
                f = interp1d(t, y, kind='linear')
                y_interp[i] = f(t_interp)
            plot_temporal_curve()

            show_axes = []
            if row == 2:
                show_axes.append('bottom')
            if row == 0:
                ax.set_xlabel(subtitle_list[col])
                ax.xaxis.set_label_coords(0.5, 1.2)
            if col == 0:
                show_axes.append('left')
                ax.set_ylabel(ylabel_list[row])
            ph.adjust_spines(ax, show_axes, lw=.25, xlim=xlim, ylim=ylim[row], xticks=xticks, yticks=yticks[row])
            # ax.axvline(ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
            # ax.axvline(stim_dur, ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
            ax.patch.set_facecolor('white')
            ax.grid(which='major', alpha=0)
            ax.fill_between([0,stim_dur], ylim[row][0], ylim[row][1], facecolor=ph.grey2, alpha=0.2)

    fig.text(0.5, 0.07, 'Time (s)', horizontalalignment='center', verticalalignment='bottom', fontsize='20')
    if title:
        fig.suptitle(title, fontsize='20')


def plot_stim_bouts_walking_standing(group, stim_type='pro_reg_v2', t_precede=2, t_follow=4, signal_type='fb.c1.mean_an',
                                     title='', axvline=[], indi_alpha=0.5, indi_lw=1, lw=3,
                                     ylim=[[0, 3], [-20, 20], [-5, 5], [0, 10]],
                                     yticks=[np.arange(0,3.1,1), np.arange(-20, 21, 20), np.arange(-5,6,5), np.arange(0,11,10)]):
    # corresponding to the 2p data collected AFTER Jan. 1st, 2017
    ph.set_fontsize(12)
    size = (7, 8)
    fig = plt.figure(1, size)
    if stim_type == 'pro_reg':
        stim_num = 4
        subtitle_list = ['speed: -0.30 m/s', 'speed: -0.15 m/s', 'speed: 0.15 m/s', 'speed: 0.30 m/s']
    elif stim_type == 'pro_reg_v2':
        stim_num = 3
        subtitle_list = ['velocity: -0.20 m/s', 'velocity: 0 m/s', 'velocity: 0.20 m/s']
    else:
        print 'Error: does not have this type of stim_type'
        return

    gs = gridspec.GridSpec(4, stim_num)
    data = get_stim_bouts_data_walking(group, stim_type, signal_type, t_precede, t_follow)
    fly_num = len(group)
    color = ph.green

    ylabel_list = ['$\Delta F/F$','Heading\n(deg)', 'Forward\n(mm)', 'stim x\n(volt)']

    def standing_cri():
        thresh_dhead_standing = 15
        thresh_dforw_standing = 2
        if (np.max(np.abs(data[col].dhead[i])) <= thresh_dhead_standing) &\
                (np.max(np.abs(data[col].dforw[i])) <= thresh_dforw_standing):
            return True
        return False


    for row, ylabel in enumerate(ylabel_list):
        for col, xlabel in enumerate(subtitle_list):
            ax = plt.subplot(gs[row, col])
            # Plot mean bout curves
            if row == 0:
                dt = 0.05
            else:
                dt = 0.02

            fly_mean = []
            t_interp = np.arange(-t_precede - 0.3, t_follow + 0.1, dt)
            for i_fly in range(fly_num):
                y_interp = np.zeros((len(data[col].t_tif), len(t_interp)))
                for i1 in range(len(y_interp)):
                    for i2 in range(len(y_interp[i1])):
                        y_interp[i1][i2] = np.nan

                for i in range(len(data[col].t_tif)):
                    if not standing_cri():
                        continue
                    if data[col].fly_num[i] == i_fly:
                        if row == 0:
                            t = data[col].t_tif[i]
                            y = data[col].imaging[i]
                        elif row == 3:
                            t = data[col].t_abf[i]
                            y = data[col].xstim[i]
                        elif row == 1:
                            t = data[col].t_abf[i]
                            y = data[col].head[i]
                        elif row == 2:
                            t = data[col].t_abf[i]
                            y = fc.deg2mm(data[col].forw[i])
                        f = interp1d(t, y, kind='linear')
                        # print t[0], t[-1], t_interp[0], t_interp[-1]
                        y_interp[i] = f(t_interp)
                mean = np.nanmean(y_interp, axis=0)
                ax.plot(t_interp, mean, lw=indi_lw, c=color, zorder=1, alpha=indi_alpha)
                fly_mean.append(mean)
            ax.plot(t_interp, np.nanmean(fly_mean, axis=0), lw=lw, c=color, zorder=1)

            ax.patch.set_facecolor('white')
            ax.grid(which='major', alpha=0)
            show_axes = []
            if row == 3:
                show_axes.append('bottom')
                ax.axhline(y=0, ls='--', c='black', zorder=1, lw=1, dashes=[5, 5])
            if row == 0:
                ax.set_xticklabels([])
                ax.set_xlabel(subtitle_list[col])
                ax.xaxis.set_label_coords(0.5, 1.4)
            if col == 0:
                show_axes.append('left')
                ax.set_ylabel(ylabel_list[row])
                ax.get_yaxis().set_label_coords(-0.3, 0.6)
            ph.adjust_spines(ax, show_axes, lw=.25, xlim=[-t_precede, t_follow], ylim=(ylim[row]),
                             yticks=(yticks[row]), xticks=(np.arange(-t_precede, t_follow+0.1, 2)))

            if (row == 0) and (col == 0):
                xlim_text = ax.get_xlim()
                ylim_text = ax.get_ylim()
                text = str(len(group)) + 'flies'
                ax.text(xlim_text[0]*.95+xlim_text[-1]*.05, ylim_text[0]*.15+ylim_text[-1]*.85, text, ha='left', va='bottom', fontsize=11)
            if (col == 0) and (row == 0):
                ph.plot_x_scale_text(ax, 1, '1s', x_text_lefter=None, y_text_lower=None)
            # for x in axvline:
            #     ax.axvline(x=x, ls='--', c='black', zorder=1, lw=1, dashes=[5, 5])
            ax.fill_between(axvline, ylim[row][0], ylim[row][-1], edgecolor='none', facecolor=ph.grey1, alpha=1)

    fig.text(0.5, 0.01, 'Time (s)', horizontalalignment='center', verticalalignment='bottom', fontsize='30')
    if title:
        fig.suptitle(title, fontsize='20')


def plot_stim_bouts_walking_forward_vel_tuning_curve(group, stim_type='pro_reg_v2', t_precede=2, t_follow=4, stim_dur=2,
                                                     signal_type='fb.c1.mean_pn', title='', indi_alpha=0.5, bins=40,
                                                     indi_lw=1, lw=3, ylim=[0,1], yticks=np.arange(0,1.1,0.5),
                                                     xlim=[-4,10], xticks=np.arange(-4,10,4)):
    # corresponding to the 2p data collected AFTER Jan. 1st, 2017
    ph.set_fontsize(12)
    size = (7, 1.5)
    fig = plt.figure(1, size)
    if stim_type == 'pro_reg':
        stim_num = 4
        subtitle_list = ['speed: -0.30 m/s', 'speed: -0.15 m/s', 'speed: 0.15 m/s', 'speed: 0.30 m/s']
    elif stim_type == 'pro_reg_v2':
        stim_num = 3
        subtitle_list = ['velocity: -0.20 m/s', 'velocity: 0 m/s', 'velocity: 0.20 m/s']
    else:
        print 'Error: does not have this type of stim_type'
        return

    gs = gridspec.GridSpec(1, stim_num)
    data = get_stim_bouts_data_walking(group, stim_type, signal_type, t_precede, t_follow)
    fly_num = len(group)
    color = ph.green
    ylabel = '$\Delta F/F$'

    rec = group[0][0]
    ind0 = np.ceil((t_precede + 1) * rec.ims[0].sampling_rate)
    ind_dur = np.floor(stim_dur * rec.ims[0].sampling_rate)
    inds = np.arange(ind0, ind0+ind_dur).astype(int)
    def standing_cri():
        thresh_dhead_standing = 15
        thresh_dforw_standing = 2
        if (np.max(np.abs(data[col].dhead[i])) <= thresh_dhead_standing) &\
                (np.max(np.abs(data[col].dforw[i])) <= thresh_dforw_standing):
            return True
        return False

    for col, xlabel in enumerate(subtitle_list):
        ax = plt.subplot(gs[0, col])
        # Plot mean bout curves

        population_x = []
        population_y = []
        for i_fly in range(fly_num):
            fly_signal = []
            fly_behavior = []
            for i in range(len(data[col].t_tif)):
                if standing_cri():
                    continue
                if data[col].fly_num[i] == i_fly:
                    fly_signal.extend(data[col].imaging[i][inds])
                    fly_behavior.extend(data[col].dforw[i][inds])
            x, y, _ = fc.scatter_to_errorbar(x_input=fly_behavior, y_input=fly_signal, binx=bins, xmin=xlim[0],
                                             xmax=xlim[-1])
            population_x.append(x)
            population_y.append(y)

        x_mean = np.nanmean(population_x, axis=0)
        y_mean = np.nanmean(population_y, axis=0)
        for i in range(len(population_x)):
            ax.plot(population_x[i], population_y[i], c=color, alpha=indi_alpha, lw=indi_lw)
        ax.plot(x_mean, y_mean, c=color, alpha=1, lw=lw)
        ax.patch.set_facecolor('white')
        ax.grid(which='major', alpha=0)
        ax.axvline(ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])

        show_axes = []
        if col == 0:
            ax.set_ylabel('$\Delta$F/F')
            ax.get_yaxis().set_label_coords(-0.3, 0.6)
            ax.set_xlabel('Forward speed\n(mm/s)')
            show_axes.append('left')
        show_axes.append('bottom')
        ph.adjust_spines(ax, show_axes, lw=.25, xticks=xticks, yticks=yticks, ylim=ylim, xlim=xlim)

        if col == 0:
            xlim_text = ax.get_xlim()
            ylim_text = ax.get_ylim()
            text = str(len(group)) + 'flies'
            ax.text(xlim_text[0] * .95 + xlim_text[-1] * .05, ylim_text[0] * .15 + ylim_text[-1] * .85, text, ha='left',
                    va='bottom', fontsize=11)
        # ax.text(np.mean(xlim), ylim[-1]*1.1, subtitle_list[col], horizontalalignment='center', verticalalignment='bottom', fontsize='25')

    if title:
        fig.suptitle(title, fontsize='20')


def plot_stim_bouts_walking_forward_vel_tuning_curve_overlap_meancurve(group, stim_type='pro_reg_v2', t_precede=2, t_follow=4, stim_dur=2,
                                                     signal_type='fb.c1.mean_an', title='', indi_alpha=0.5, bins=40,
                                                     indi_lw=1, lw=2, ylim=[0,1], yticks=np.arange(0,1.1,0.5),
                                                     xlim=[-4,10], xticks=np.arange(-4,10,4)):
    # corresponding to the 2p data collected AFTER Jan. 1st, 2017
    ph.set_fontsize(20)
    size = (6, 4)
    fig = plt.figure(1, size)
    if stim_type == 'pro_reg':
        stim_num = 4
        subtitle_list = ['speed: -0.30 m/s', 'speed: -0.15 m/s', 'speed: 0.15 m/s', 'speed: 0.30 m/s']
    elif stim_type == 'pro_reg_v2':
        stim_num = 3
        subtitle_list = ['velocity: -0.20 m/s', 'velocity: 0 m/s', 'velocity: 0.20 m/s']
    else:
        print 'Error: does not have this type of stim_type'
        return

    gs = gridspec.GridSpec(1, 1)
    data = get_stim_bouts_data_walking(group, stim_type, signal_type, t_precede, t_follow)
    fly_num = len(group)
    color = ph.green
    ylabel = '$\Delta F/F$'
    ax = plt.subplot(gs[0, 0])

    rec = group[0][0]
    ind0 = np.ceil((t_precede + 1) * rec.ims[0].sampling_rate)
    ind_dur = np.floor(stim_dur * rec.ims[0].sampling_rate)
    inds = np.arange(ind0, ind0+ind_dur).astype(int)
    def standing_cri():
        thresh_dhead_standing = 15
        thresh_dforw_standing = 2
        if (np.max(np.abs(data[col].dhead[i])) <= thresh_dhead_standing) &\
                (np.max(np.abs(data[col].dforw[i])) <= thresh_dforw_standing):
            return True
        return False

    for col, xlabel in enumerate(subtitle_list):
        # Plot mean bout curves

        population_x = []
        population_y = []
        for i_fly in range(fly_num):
            fly_signal = []
            fly_behavior = []
            for i in range(len(data[col].t_tif)):
                if standing_cri():
                    continue
                if data[col].fly_num[i] == i_fly:
                    fly_signal.extend(data[col].imaging[i][inds])
                    fly_behavior.extend(data[col].dforw[i][inds])
            x, y, _ = fc.scatter_to_errorbar(x_input=fly_behavior, y_input=fly_signal, binx=bins, xmin=xlim[0],
                                             xmax=xlim[-1])
            population_x.append(x)
            population_y.append(y)

        x_mean = np.nanmean(population_x, axis=0)
        y_mean = np.nanmean(population_y, axis=0)
        color = seq_dark_colours()[col]
        ax.plot(x_mean, y_mean, c=color, alpha=1, lw=lw, label=subtitle_list[col])
    ax.patch.set_facecolor('white')
    ax.grid(which='major', alpha=0)

    show_axes = []
    ax.axvline(ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
    ax.set_ylabel('$\Delta$F/($F_{max}-F_0$)')
    ax.set_xlabel('Forward speed\n(mm/s)')
    show_axes.append('left')
    show_axes.append('bottom')
    ph.adjust_spines(ax, show_axes, lw=.25, xticks=xticks, yticks=yticks, ylim=ylim, xlim=xlim)
    ax.legend()
    # ax.text(np.mean(xlim), ylim[-1]*1.1, subtitle_list[col], horizontalalignment='center', verticalalignment='bottom', fontsize='25')

    if title:
        fig.suptitle(title, fontsize='20')

def plot_stim_bouts_walking_fvtcom_paper(group, stim_type='pro_reg_v2', t_precede=2, t_follow=4, stim_dur=2,
                                         signal_type='fb.c1.mean_an', bins=20, xlim=[-4,10], xticks=np.arange(-4,10,4),
                                         show_ybar=True, ylim=[-1,3], yticks=[-1,3],
                                         show_legend=True, legend_list=['reg.', 'static', 'prog.'],
                                         leg_loc='upper left', bbox_to_anchor=(0, 1), fs_legend=10, ylabel_axeratio=-.2,
                                         show_flynum=True, colors=[ph.dred_pure, 'black', ph.dblue_pure],
                                         lw=3, alpha_sem=.2, fs=12,):
    def standing_or_not():
        thresh_dhead_standing = 15
        thresh_dforw_standing = 2
        if (np.max(np.abs(data[i_stim].dhead[i])) <= thresh_dhead_standing) &\
                (np.max(np.abs(data[i_stim].dforw[i])) <= thresh_dforw_standing):
            return True
        return False

    ph.set_fontsize(fs)
    fig = plt.figure(1, (2, 2.5))
    gs = gridspec.GridSpec(1, 1)
    data = get_stim_bouts_data_walking(group, stim_type, signal_type, t_precede, t_follow)
    stim_num = 3
    fly_num = len(group)
    ax = plt.subplot(gs[0, 0])

    rec = group[0][0]
    ind0 = np.ceil((t_precede + 1) * rec.ims[0].sampling_rate)
    ind_dur = np.floor(stim_dur * rec.ims[0].sampling_rate)
    inds = np.arange(ind0, ind0+ind_dur).astype(int)

    for i_stim in range(stim_num):
        color = colors[i_stim]
        population_x = []
        population_y = []
        for i_fly in range(fly_num):
            fly_signal = []
            fly_behavior = []
            for i in range(len(data[i_stim].t_tif)):
                if standing_or_not():
                    continue
                if data[i_stim].fly_num[i] == i_fly:
                    fly_signal.extend(data[i_stim].imaging[i][inds])
                    fly_behavior.extend(data[i_stim].dforw[i][inds])
            x, y, _ = fc.scatter_to_errorbar(x_input=fly_behavior, y_input=fly_signal, binx=bins, xmin=xlim[0],
                                             xmax=xlim[-1])
            xs = x
            population_y.append(y)

        y_mean = np.nanmean(population_y, axis=0)
        y_sem = np.nanstd(population_y, axis=0) / np.sqrt(len(population_y))
        ax.plot(xs, y_mean, c=color, alpha=1, lw=lw, zorder=2, label=legend_list[i_stim])
        ax.fill_between(xs, y_mean - y_sem, y_mean + y_sem, edgecolor='none', facecolor=color, alpha=alpha_sem, zorder=2)

    # ax.set_ylabel('$\Delta F/F_0$', rotation=90, va='center')
    # ax.get_yaxis().set_label_coords(ylabel_axeratio, 0.5)
    ax.set_xlabel('Forward velocity (mm/s)')
    ph.adjust_spines(ax, ['bottom'], lw=1, xticks=xticks, yticks=[], ylim=ylim, xlim=xlim)

    if show_ybar:
        bar_length = 1
        ybar_text = '$\Delta F/F_0$' + ' 1'
        y0 = ylim[0]
        y1 = ylim[0] + bar_length
        x_bar = xlim[0] - (xlim[-1] - xlim[0]) * .1
        x_text = xlim[0] - (xlim[-1] - xlim[0]) * .13
        y_text = ylim[0]
        ax.plot([x_bar, x_bar], [y0, y1], c='black', lw=2.5, solid_capstyle='butt', clip_on=False)
        ax.text(x_text, y_text, ybar_text, rotation='horizontal', ha='right', va='bottom')

    if show_legend:
        if bbox_to_anchor:
            ax.legend(prop={'size': fs_legend}, loc=leg_loc, bbox_to_anchor=bbox_to_anchor, ncol=1, frameon=False)
        else:
            ax.legend(prop={'size': fs_legend}, loc=leg_loc, ncol=1, frameon=False)

    if show_flynum:
        xlim_text = ax.get_xlim()
        ylim_text = ax.get_ylim()
        text = str(len(group)) + 'flies'
        ax.text(xlim_text[0]*.95+xlim_text[-1]*.05, ylim_text[0]*.95+ylim_text[-1]*.05, text, ha='left', va='bottom', fontsize=fs)

    ax.patch.set_facecolor('white')
    ax.grid(which='major', alpha=0)


def plot_stim_bouts_walking_forward_vel_tuning_curve_slope(group, stim_type='pro_reg_v2', t_precede=2, t_follow=4, stim_dur=2,
                                                           signal_type='fb.c1.mean_pn', bins=40, ms=5, vel_range=[-4,8]):
    # corresponding to the 2p data collected AFTER Jan. 1st, 2017
    ph.set_fontsize(12)
    size = (2, 1.5)
    fig = plt.figure(1, size)
    if stim_type == 'pro_reg':
        stim_num = 4
        subtitle_list = ['speed: -0.30 m/s', 'speed: -0.15 m/s', 'speed: 0.15 m/s', 'speed: 0.30 m/s']
    elif stim_type == 'pro_reg_v2':
        stim_num = 3
        subtitle_list = ['Regressive', 'Stationary', 'Progressive']
    else:
        print 'Error: does not have this type of stim_type'
        return

    gs = gridspec.GridSpec(1, 1)
    data = get_stim_bouts_data_walking(group, stim_type, signal_type, t_precede, t_follow)
    fly_num = len(group)
    color = ph.green
    ylabel = '$\Delta F/F$'
    ax = plt.subplot(gs[0, 0])

    rec = group[0][0]
    ind0 = np.ceil((t_precede + 1) * rec.ims[0].sampling_rate)
    ind_dur = np.floor(stim_dur * rec.ims[0].sampling_rate)
    inds = np.arange(ind0, ind0+ind_dur).astype(int)
    def standing_cri():
        thresh_dhead_standing = 15
        thresh_dforw_standing = 2
        if (np.max(np.abs(data[col].dhead[i])) <= thresh_dhead_standing) &\
                (np.max(np.abs(data[col].dforw[i])) <= thresh_dforw_standing):
            return True
        return False

    # plot
    population_slope = [[], [], []]
    for i_fly in range(fly_num):
        for col, xlabel in enumerate(subtitle_list):
            fly_signal = []
            fly_behavior = []
            for i in range(len(data[col].t_tif)):
                if standing_cri():
                    continue
                if data[col].fly_num[i] == i_fly:
                    fly_signal.extend(data[col].imaging[i][inds])
                    fly_behavior.extend(data[col].dforw[i][inds])
            x, y, _ = fc.scatter_to_errorbar(x_input=fly_behavior, y_input=fly_signal, binx=bins, xmin=vel_range[0],
                                             xmax=vel_range[-1])
            inds = np.where(~np.isnan(y))[0]
            x = x[inds]
            y = y[inds]
            slope, _, _, _, _ = stats.linregress(x, y)
            population_slope[col].append(slope)

    xaxis_ticks = np.arange(stim_num) + 0.5
    for col in range(stim_num):
        for i_fly in range(len(population_slope[col])):
            color = seq_dark_colours()[i_fly]
            x_random = (random.random() - 0.5) * 0.2 + xaxis_ticks[col]
            ax.plot(x_random, population_slope[col][i_fly], 'o', ms=ms, c=color)
    mean = np.zeros(stim_num)
    sem = np.zeros(stim_num)
    for col in range(stim_num):
        mean[col] = np.nanmean(population_slope[col])
        sem[col] = np.nanstd(population_slope[col]) / np.sqrt(np.sum(~np.isnan(population_slope[col])))
    ax.errorbar(xaxis_ticks, mean, sem, fmt='o', c='black', ms=ms + 3)

    ax.patch.set_facecolor('white')
    ax.grid(which='major', alpha=0)
    ax.set_ylabel('slope')
    show_axes = ['bottom', 'left']
    ph.adjust_spines(ax, show_axes, lw=.25, xticks=np.arange(0.5,stim_num,1), xlim=[0,stim_num],
                     ylim=[0, 0.1], yticks=[0, 0.1])
    ax.set_xticklabels(subtitle_list, rotation=45, ha='right', )
    # ax.text(stim_num/2.0, -0.033, 'velocity (m/s)', horizontalalignment='center', verticalalignment='bottom',
    #         fontsize='12')


def plot_intensity_velocity_walking_bouts(group, axs=False, walking_bouts=True, bins=40, indi_alpha=0.2, lw_indi=1.5, lw=5, c=[],
                                          vel_range=[[-4, 10],[0,400],[-2,5]], xticks=[[],[],[]], yticks=[], ylim=[0,1],
                                          loc_num=0, sig_type='mean_al', ch_num=1, subplot_num=3, plotcurvesonly=False, fs=9):
    # fig=1: scatter + bin ave

    ph.set_fontsize(fs)
    if not axs:
        plt.figure(1, (2+subplot_num*4, 4))
        gs = gridspec.GridSpec(1, subplot_num)
    if not len(c):
        color = colours[group[0][0].ims[0].c1.celltype]
    else:
        color = c
    xlabel = ['Forward speed\n(mm/s)', 'Turning vel\n(deg/s)', 'Forward acce\n(mm/s$^2$)']

    population_x = [[],[],[]]
    population_y = [[],[],[]]
    behavior_num = len(population_y)

    for fly in group:
        fly_intensity = []
        fly_behavior = [[],[],[]]
        for rec in fly:
            if walking_bouts:
                inds = get_walking_bouts_inds(rec)
                for ind in inds:
                    inds_start = np.where(rec.ims[loc_num].t >= rec.abf.t[ind[0]])[0]
                    inds_end = np.where(rec.ims[loc_num].t <= rec.abf.t[ind[-1]])[0]
                    if len(inds_start) and len(inds_end):
                        t_ind = np.arange(inds_start[0], inds_end[-1])
                    else:
                        continue
                    if sig_type == 'mean_al':
                        if ch_num == 1:
                            fly_intensity.extend(rec.ims[loc_num].c1.mean_al[t_ind])
                        else:
                            fly_intensity.extend(rec.ims[loc_num].c2.mean_al[t_ind])
                    if sig_type == 'mean_an':
                        if ch_num == 1:
                            fly_intensity.extend(rec.ims[loc_num].c1.mean_an[t_ind])
                        else:
                            fly_intensity.extend(rec.ims[loc_num].c2.mean_an[t_ind])
                    if sig_type == 'd_mean_an':
                        if ch_num == 1:
                            fly_intensity.extend(np.gradient(rec.ims[loc_num].c1.mean_an[t_ind]))
                        else:
                            fly_intensity.extend(np.gradient(rec.ims[loc_num].c2.mean_an[t_ind]))
                    elif sig_type == 'peak_al':
                        if ch_num == 1:
                            fly_intensity.extend(np.nanmax(rec.ims[loc_num].c1.al[t_ind], axis=-1))
                        else:
                            fly_intensity.extend(np.nanmax(rec.ims[loc_num].c2.al[t_ind], axis=-1))
                    elif sig_type == 'peak_an':
                        if ch_num == 1:
                            fly_intensity.extend(np.nanmax(rec.ims[loc_num].c1.an[t_ind], axis=-1))
                        else:
                            fly_intensity.extend(np.nanmax(rec.ims[loc_num].c2.an[t_ind], axis=-1))
                    if sig_type == 'd_peak_an':
                        if ch_num == 1:
                            fly_intensity.extend(np.gradient(np.nanmax(rec.ims[loc_num].c1.an[t_ind], axis=-1)))
                        else:
                            fly_intensity.extend(np.gradient(np.nanmax(rec.ims[loc_num].c2.an[t_ind], axis=-1)))
                    fly_behavior[0].extend(rec.subsample('dforw')[t_ind])
                    fly_behavior[1].extend(np.abs(rec.subsample('dhead')[t_ind]))
                    fly_behavior[2].extend(np.gradient(rec.subsample('dforw')[t_ind]))
            else:
                if sig_type == 'mean_al':
                    if ch_num == 1:
                        fly_intensity.extend(rec.ims[loc_num].c1.mean_al)
                    else:
                        fly_intensity.extend(rec.ims[loc_num].c2.mean_al)
                if sig_type == 'mean_an':
                    if ch_num == 1:
                        fly_intensity.extend(rec.ims[loc_num].c1.mean_an)
                    else:
                        fly_intensity.extend(rec.ims[loc_num].c2.mean_an)
                if sig_type == 'd_mean_an':
                    if ch_num == 1:
                        fly_intensity.extend(np.gradient(rec.ims[loc_num].c1.mean_an))
                    else:
                        fly_intensity.extend(np.gradient(rec.ims[loc_num].c2.mean_an))
                elif sig_type == 'peak_al':
                    if ch_num == 1:
                        fly_intensity.extend(np.nanmax(rec.ims[loc_num].c1.al, axis=-1))
                    else:
                        fly_intensity.extend(np.nanmax(rec.ims[loc_num].c2.al, axis=-1))
                elif sig_type == 'peak_an':
                    if ch_num == 1:
                        fly_intensity.extend(np.nanmax(rec.ims[loc_num].c1.an, axis=-1))
                    else:
                        fly_intensity.extend(np.nanmax(rec.ims[loc_num].c2.an, axis=-1))
                elif sig_type == 'd_peak_an':
                    if ch_num == 1:
                        fly_intensity.extend(np.gradient(np.nanmax(rec.ims[loc_num].c1.an, axis=-1)))
                    else:
                        fly_intensity.extend(np.gradient(np.nanmax(rec.ims[loc_num].c2.an, axis=-1)))

                fly_behavior[0].extend(rec.subsample('dforw'))
                fly_behavior[1].extend(np.abs(rec.subsample('dhead')))
                fly_behavior[2].extend(np.gradient(rec.subsample('dforw')))

        fly_intensity = np.array(fly_intensity)
        if len(fly_intensity):
            for i in range(behavior_num):
                fly_behavior[i] = np.array(fly_behavior[i])
            for i in range(behavior_num):
                x, y, y_std = fc.scatter_to_errorbar(x_input=fly_behavior[i], y_input=fly_intensity, binx=bins,
                                                     xmin=vel_range[i][0], xmax=vel_range[i][-1])
                # if i == 0:
                #     print len(y), np.sum(~np.isnan(y))
                population_x[i].append(x)
                population_y[i].append(y)

    for col in range(subplot_num):
        if axs:
            ax = axs[col]
        else:
            ax = plt.subplot(gs[0, col])
        x_mean = np.nanmean(population_x[col], axis=0)
        y_mean = np.nanmean(population_y[col], axis=0)
        for i in range(len(population_x[col])):
            ax.plot(population_x[col][i], population_y[col][i], c=color, alpha=indi_alpha, lw=lw_indi)
        ax.plot(x_mean, y_mean, c=color, alpha=1, lw=lw)
        ax.axvline(0, ls='--', c=ph.grey3, lw=1, zorder=1)

        if not plotcurvesonly:
            ax.set_xlabel(xlabel[col])
            show_axes = []
            if col == 0:
                # ax.axvline(ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
                ax.set_ylabel('$\Delta$F/$F_0$', rotation=90, va='center')
                ax.get_yaxis().set_label_coords(-0.25, 0.5)
                show_axes.append('left')
            show_axes.append('bottom')
            ph.adjust_spines(ax, show_axes, lw=1, xticks=xticks[col], xlim=vel_range[col], yticks=yticks, ylim=ylim)
            ax.patch.set_facecolor('white')
            ax.grid(which='major', alpha=0)


def plot_intensity_velocity_walking_bouts_2d(group, walking_bouts=True, bins=40, colormap=plt.cm.plasma,
                                             vel_range=[[-4, 10],[0,400]], xticks=[[],[]], yticks=[],
                                             ylim=[0,1], loc_num=0, sig_type='mean_al'):
    # fig=1: scatter + bin ave

    ph.set_fontsize(20)
    plt.figure(1, (8, 8))
    gs = gridspec.GridSpec(1, 1)

    xlabel = ['Forward speed\n(mm/s)', 'Turning velocity\n(deg/s)']

    population = []
    behavior_num = 2

    for fly in group:
        fly_intensity = []
        fly_behavior = [[],[]]
        for rec in fly:
            if walking_bouts:
                inds = get_walking_bouts_inds(rec)
                for ind in inds:
                    inds_start = np.where(rec.ims[loc_num].t >= rec.abf.t[ind[0]])[0]
                    inds_end = np.where(rec.ims[loc_num].t <= rec.abf.t[ind[-1]])[0]
                    if len(inds_start) and len(inds_end):
                        t_ind = np.arange(inds_start[0], inds_end[-1])
                    else:
                        continue
                    if sig_type == 'mean_al':
                        fly_intensity.extend(rec.ims[loc_num].c1.mean_al[t_ind])
                    elif sig_type == 'peak_al':
                        fly_intensity.extend(np.nanmax(rec.ims[loc_num].c1.al[t_ind], axis=-1))
                    elif sig_type == 'peak_an':
                        fly_intensity.extend(np.nanmax(rec.ims[loc_num].c1.an[t_ind], axis=-1))
                    fly_behavior[0].extend(rec.subsample('dforw')[t_ind])
                    fly_behavior[1].extend(np.abs(rec.subsample('dhead')[t_ind]))
            else:
                if sig_type == 'mean_al':
                    fly_intensity.extend(rec.ims[loc_num].c1.mean_al)
                elif sig_type == 'peak_al':
                    fly_intensity.extend(np.nanmax(rec.ims[loc_num].c1.al, axis=-1))
                elif sig_type == 'peak_an':
                    fly_intensity.extend(np.nanmax(rec.ims[loc_num].c1.an, axis=-1))
                fly_behavior[0].extend(rec.subsample('dforw'))
                fly_behavior[1].extend(np.abs(rec.subsample('dhead')))

        fly_intensity = np.array(fly_intensity)
        if len(fly_intensity):
            for i in range(behavior_num):
                fly_behavior[i] = np.array(fly_behavior[i])

            x, y, z_mean = fc.scatter_to_heatmap(x_input=fly_behavior[0], y_input=fly_behavior[1],
                                                 z_input=fly_intensity, binx=bins, biny=bins,
                                                 xmin=vel_range[0][0], xmax=vel_range[0][-1],
                                                 ymin=vel_range[1][0], ymax=vel_range[1][-1],)

            population.append(z_mean)

    ax = plt.subplot(gs[0, 0])
    z = np.nanmean(population, axis=0).reshape(bins, bins)
    z[np.isnan(z)] = 0
    img = plt.pcolormesh(y, x, z, cmap=colormap)
    #
    # show_axes = []
    # if col == 0:
    #     ax.axvline(ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
    #     ax.set_ylabel('$\Delta$F/($F_{max}-F_0$)')
    #     show_axes.append('left')
    # show_axes.append('bottom')
    # ph.adjust_spines(ax, show_axes, lw=.25, xticks=xticks[col], yticks=yticks, ylim=ylim)
    #
    # ax.patch.set_facecolor('white')
    # ax.grid(which='major', alpha=0)


def plot_xcorr_walking_bouts(group, signal_type=['forward\nspeed', 'turning\nvelocity', 'forward\nacceleration'],
                             indi_alpha=0.2, lw_indi=1, lw=3, c=[],
                             xlim=[-2, 2], ylim=[0,1], xticks=[], yticks=[], yticklabels=[], xticklabels=[]):

    ph.set_fontsize(20)
    col_num = len(signal_type)
    plt.figure(1, (5*col_num, 4))
    gs = gridspec.GridSpec(1, col_num)
    if not len(c):
        color = colours[group[0][0].ims[0].c1.celltype]
    else:
        color = c

    xcorr = [[], [], []]
    t_lag = np.arange(xlim[0]-0.1, xlim[1]+0.1, 0.02)

    for fly in group:
        xcorr_fly = [[], [], []]
        for rec in fly:
            behavior = [[], [], []]
            inds = get_walking_bouts_inds(rec)
            for ind in inds:
                inds_start = np.where(rec.ims[0].t <= rec.abf.t[ind[0]])[0]
                inds_end = np.where(rec.ims[0].t >= rec.abf.t[ind[-1]])[0]
                if len(inds_start) and len(inds_end):
                    t_ind = np.arange(inds_start[-1], inds_end[0])
                else:
                    continue
                signal = rec.ims[0].c1.mean_pl[t_ind]
                behavior[0] = rec.subsample('dforw')[t_ind]
                behavior[1] = np.abs(rec.subsample('dhead')[t_ind])
                behavior[2] = np.gradient(rec.subsample('dforw')[t_ind])
                if len(signal):
                    for i in range(3):
                        t_lag_trial, xcorr_trial = fc.xcorr(signal, behavior[i], sampling_period=1.0/rec.ims[0].sampling_rate)
                        f = interp1d(t_lag_trial, xcorr_trial, kind='linear')
                        xcorr_interp = f(t_lag)
                        xcorr_fly[i].append(xcorr_interp)

        if len(xcorr_fly[0]):
            for i in range(3):
                xcorr[i].append(np.nanmean(xcorr_fly[i], axis=0))

    for col in range(col_num):
        ax = plt.subplot(gs[0, col])
        xcorr_mean = np.nanmean(xcorr[col], axis=0)
        for i in range(len(xcorr[col])):
            ax.plot(t_lag, xcorr[col][i], c=color, alpha=indi_alpha, lw=lw_indi)
        ax.plot(t_lag, xcorr_mean, c=color, alpha=1, lw=lw)
        ax.axvline(ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
        ax.patch.set_facecolor('white')
        ax.grid(which='major', alpha=0)

        show_axes = []
        if col == 0:
            ax.set_ylabel('correlation\nbetween\n2p signal\nand behavior', y=0.3, labelpad= 60, rotation=0)
            ax.set_xlabel('Lag (s)')
            show_axes.append('left')
        show_axes.append('bottom')
        ph.adjust_spines(ax, show_axes, lw=.25, xticks=xticks, yticks=yticks, xlim=xlim, ylim=ylim)
        if col == 0:
            ax.set_xticklabels(xticklabels)
            ax.set_yticklabels(yticklabels)
        else:
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        ax.text(xlim[0], ylim[-1]*0.9, signal_type[col], fontsize=15)


def cal_xcorr_and_selectivity(recss, dforw_thresh_high=11, dforw_thresh_low=2, dhead_thresh_high=130,
                              dhead_thresh_low=80, driverline='46F02', effector='GC6f', loc='fb'):
    xcorr_indexs = []
    forw_select_indexs = []
    for recs in recss:
        xcorr_index = []
        forw_select_index = []
        for rec in recs:
            if rec.ims[0].c1.an.ndim == 2:
                tif = np.nanmean(rec.ims[0].c1.an, axis=-1)
            else:
                tif = rec.ims[0].c1.an

            t_lag_trial, xcorr_trial = fc.xcorr(tif, rec.subsample('dforw'), sampling_period=1.0 / rec.ims[0].sampling_rate)
            inds_xcorr = (t_lag_trial >= -1) & (t_lag_trial <= 1)
            xcorr_index.append(np.max(xcorr_trial[inds_xcorr]))

            dforw = rec.subsample('dforw')
            dhead = np.abs(rec.subsample('dhead'))
            inds_dforw = (dforw >= dforw_thresh_high) & (dhead <= dhead_thresh_low)
            inds_dhead = (dforw <= dforw_thresh_low) & (dhead >= dhead_thresh_high)
            tif_dforw = tif[inds_dforw].mean()
            tif_dhead = tif[inds_dhead].mean()
            forw_select_index.append((tif_dforw - tif_dhead) / (tif_dforw + tif_dhead))
        xcorr_indexs.append(np.nanmean(xcorr_index))
        forw_select_indexs.append(np.nanmean(forw_select_index))

    if len(xcorr_indexs) and len(forw_select_indexs):
        xcorr_indexs = np.array(xcorr_indexs)
        forw_select_indexs = np.array(forw_select_indexs)
        print xcorr_indexs.mean(), xcorr_indexs.var(ddof=1)
        print forw_select_indexs.mean(), forw_select_indexs.var(ddof=1)

    filename =  driverline + '-' + effector + '-' + loc + '.csv'
    with open(filename, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='|', quoting=csv.QUOTE_MINIMAL)
        spamwriter.writerow([xcorr_indexs.mean(), xcorr_indexs.var(ddof=1),
                             forw_select_indexs.mean(), forw_select_indexs.var(ddof=1)])


def plot_xcorr_and_selectivity(folder, color='black', fs=20, xlim=[0,1], ylim=[-.3,1],
                               xticks=np.array([0, 1]), xticks_minor=np.arange(0,1.1,.5),
                               yticks=np.array([0, 1]), yticks_minor=np.arange(0,1.1,.5),
                               emphasize_Gal4s=[], marker='o', alpha=1, mew=1.2, ms=6, capsize=4,
                               label_axeratio=True, bbox_to_anchor=False,
                               legend=False, legend_list=[], leg_loc='upper right'):
    recs = []
    filenames = (glob.glob(folder + os.path.sep + '*.csv'))
    for filename in filenames:
        recs.append(LocomotionIndex(filename))

    ph.set_fontsize(fs)
    fig = plt.figure(1, (1.5, 1.5))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])
    for rec in recs:

        if rec.Gal4 in emphasize_Gal4s:
            _color = 'black'
        elif rec.Gal4 in []:
            _color = ph.yellow
        # elif rec.Gal4 in ['46F02'] and rec.loc=='fb':
        #     _color = 'black'
        # else:
        #     continue
        else:
            _color = color
        ax.errorbar(rec.xcorr_mean, rec.forw_selective_mean, rec.xcorr_std, rec.forw_selective_std,
                    c=_color, marker=marker, mew=mew, mec=color, mfc='none', ms=ms, capsize=capsize, alpha=alpha)

    show_axes = ['left', 'bottom']
    # show_axes = []
    # ax.axvline(0, ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
    ax.axhline(0, ls='--', c=ph.grey5, zorder=1, lw=.5, dashes=[2, 2])
    ph.adjust_spines(ax, show_axes, lw=.8, xticks=xticks, xticks_minor=xticks_minor, yticks=yticks,
                     yticks_minor=yticks_minor, xlim=xlim, ylim=ylim, ticklength=2)
    ax.patch.set_facecolor('white')
    ax.grid(which='major', alpha=0)
    ax.set_xlabel('cross correlation\nwith forw.')
    ax.set_ylabel('forw. selectivity\nover turning')
    ax.get_yaxis().set_label_coords(-.3, 0.5)


# tuning curve, NEW, 2019 Oct

def heatmap_asym_forwside(group, axs=[], walking_bouts=False, bins=20, loc_num=0, sig_type='al0',
                          colormap=plt.cm.hot, show_cbar=True, vmin=None, vmax=None, show_axes=['left', 'bottom'],
                          vel_range=[[-10, 10],[-250,250]], xticks=[-10,0,10], yticks=[-250,0,250], fs=10):
    # fig=1: scatter + bin ave
    small_turn_vel = [-300,300]

    ph.set_fontsize(fs)
    population = []
    for fly in group:
        fly_intensity = []
        fly_behavior = [[],[]]
        fly_turnvel = []

        for rec in fly:
            if sig_type == 'al0':
                _sig = rec.ims[loc_num].c1.al[:,0]
            elif sig_type == 'al1':
                _sig = rec.ims[loc_num].c1.al[:,1]
            elif sig_type == 'an0':
                _sig = rec.ims[loc_num].c1.an[:, 0]
            elif sig_type == 'an1':
                _sig = rec.ims[loc_num].c1.an[:, 1]

            if walking_bouts:
                inds = get_walking_bouts_inds(rec)
                for ind in inds:
                    inds_start = np.where(rec.ims[loc_num].t >= rec.abf.t[ind[0]])[0]
                    inds_end = np.where(rec.ims[loc_num].t <= rec.abf.t[ind[-1]])[0]
                    if len(inds_start) and len(inds_end):
                        t_ind = np.arange(inds_start[0], inds_end[-1])
                    else:
                        continue
                    fly_intensity.extend(_sig[t_ind])
                    fly_behavior[0].extend(rec.subsample('dforw')[t_ind])
                    fly_behavior[1].extend(rec.subsample('dside')[t_ind])
                    fly_turnvel.extend(rec.subsample('dhead')[t_ind])
            else:
                fly_intensity.extend(_sig)
                fly_behavior[0].extend(rec.subsample('dforw'))
                fly_behavior[1].extend(rec.subsample('dside'))
                fly_turnvel.extend(rec.subsample('dhead'))

        fly_intensity = np.array(fly_intensity)
        if len(fly_intensity):
            for i in range(2):
                fly_behavior[i] = np.array(fly_behavior[i])
            fly_turnvel = np.array(fly_turnvel)

            if len(small_turn_vel):  # apply small forward velocity condition
                fly_intensity = fly_intensity[
                    (fly_turnvel > small_turn_vel[0]) & (fly_turnvel < small_turn_vel[-1])]
                fly_behavior[0] = fly_behavior[0][
                    (fly_turnvel > small_turn_vel[0]) & (fly_turnvel < small_turn_vel[-1])]
                fly_behavior[1] = fly_behavior[1][
                    (fly_turnvel > small_turn_vel[0]) & (fly_turnvel < small_turn_vel[-1])]

            x, y, z_mean = fc.scatter_to_heatmap(x_input=fly_behavior[0], y_input=fly_behavior[1],
                                                 z_input=fly_intensity, binx=bins, biny=bins,
                                                 xmin=vel_range[0][0], xmax=vel_range[0][-1],
                                                 ymin=vel_range[1][0], ymax=vel_range[1][-1], nan_to_num=-100)
            population.append(z_mean)
    z = np.nanmean(population, axis=0).reshape(bins, bins)
    z[np.isnan(z)] = 0

    if not len(axs):
        fig = plt.figure(1, (3, 2.5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[30, 1])
        ax = plt.subplot(gs[0, 0])
    else:
        ax = axs[0]

    mat = z.T
    img = ax.pcolormesh(x, y, mat, cmap=colormap)
    ph.adjust_spines(ax, show_axes, xlim=vel_range[0], xticks=xticks, ylim=vel_range[1], yticks=yticks, pad=0, ticklength=0)
    if 'bottom' in show_axes:
        ax.set_xlabel('forward velocity (mm/s)')
    if 'left' in show_axes:
        ax.set_ylabel('sideslip velocity (mm/s)')

    if show_cbar:
        ax = axs[1] if len(axs) else plt.subplot(gs[0, 1])
        if (vmin is None) or (vmax is None):
            ph.plot_colormap(ax, colormap=colormap, reverse_cm=False, ylim=[0, 256], yticks=[0, 256],
                             yticklabels=['%.1f' % np.nanmin(z),
                                          '%.1f' % np.nanmax(z)],
                             ylabel='Vm (mV)', label_rotation=90, label_pos="right", label_axeratio=2)
        else:
            ph.plot_colormap(ax, colormap=colormap, reverse_cm=False, ylim=[0, 256], yticks=[0, 256],
                             yticklabels=['%.1f' % vmin, '>%.1f' % vmax],
                             ylabel='Vm (mV)', label_rotation=90, label_pos="right", label_axeratio=2)

    if not vmin is None: img.set_clim(vmin=vmin)
    if not vmax is None: img.set_clim(vmax=vmax)

def heatmap_asym_sideturn(group, axs=[], walking_bouts=False, bins=20, loc_num=0, sig_type='rmln',
                          colormap=plt.cm.hot, show_cbar=True, vmin=None, vmax=None, show_axes=['left', 'bottom'],
                          vel_range=[[-10, 10],[-250,250]], xticks=[-10,0,10], yticks=[-250,0,250], fs=10):
    # fig=1: scatter + bin ave
    small_forw_vel = [-2, 2]

    ph.set_fontsize(fs)
    population = []
    for fly in group:
        fly_intensity = []
        fly_behavior = [[],[]]
        fly_forwvel = []

        for rec in fly:
            if sig_type == 'rmln':
                _sig = rec.ims[loc_num].c1.rmln
            elif sig_type == 'rmll':
                _sig = rec.ims[loc_num].c1.rmll
            elif sig_type == 'rmlz':
                _sig = rec.ims[loc_num].c1.rmlz

            if walking_bouts:
                inds = get_walking_bouts_inds(rec)
                for ind in inds:
                    inds_start = np.where(rec.ims[loc_num].t >= rec.abf.t[ind[0]])[0]
                    inds_end = np.where(rec.ims[loc_num].t <= rec.abf.t[ind[-1]])[0]
                    if len(inds_start) and len(inds_end):
                        t_ind = np.arange(inds_start[0], inds_end[-1])
                    else:
                        continue
                    fly_intensity.extend(_sig[t_ind])
                    fly_behavior[0].extend(rec.subsample('dside')[t_ind])
                    fly_behavior[1].extend(rec.subsample('dhead')[t_ind])
                    fly_forwvel.extend(rec.subsample('dforw')[t_ind])
            else:
                fly_intensity.extend(_sig)
                fly_behavior[0].extend(rec.subsample('dside'))
                fly_behavior[1].extend(rec.subsample('dhead'))
                fly_forwvel.extend(rec.subsample('dforw'))

        fly_intensity = np.array(fly_intensity)
        if len(fly_intensity):
            for i in range(2):
                fly_behavior[i] = np.array(fly_behavior[i])
            fly_forwvel = np.array(fly_forwvel)

            if len(small_forw_vel):  # apply small forward velocity condition
                fly_intensity = fly_intensity[
                    (fly_forwvel > small_forw_vel[0]) & (fly_forwvel < small_forw_vel[-1])]
                fly_behavior[0] = fly_behavior[0][
                    (fly_forwvel > small_forw_vel[0]) & (fly_forwvel < small_forw_vel[-1])]
                fly_behavior[1] = fly_behavior[1][
                    (fly_forwvel > small_forw_vel[0]) & (fly_forwvel < small_forw_vel[-1])]

            x, y, z_mean = fc.scatter_to_heatmap(x_input=fly_behavior[0], y_input=fly_behavior[1],
                                                 z_input=fly_intensity, binx=bins, biny=bins,
                                                 xmin=vel_range[0][0], xmax=vel_range[0][-1],
                                                 ymin=vel_range[1][0], ymax=vel_range[1][-1], nan_to_num=-100)
            population.append(z_mean)
    z = np.nanmean(population, axis=0).reshape(bins, bins)
    z[np.isnan(z)] = 0

    if not len(axs):
        fig = plt.figure(1, (3, 2.5))
        gs = gridspec.GridSpec(1, 2, width_ratios=[30, 1])
        ax = plt.subplot(gs[0, 0])
    else:
        ax = axs[0]

    mat = z.T
    img = ax.pcolormesh(x, y, mat, cmap=colormap)
    ph.adjust_spines(ax, show_axes, xlim=vel_range[0], xticks=xticks, ylim=vel_range[1], yticks=yticks, pad=0, ticklength=0)
    if 'bottom' in show_axes:
        ax.set_xlabel('sideslip velocity (mm/s)')
    if 'left' in show_axes:
        ax.set_ylabel('turn velocity (deg/s)')

    if show_cbar:
        ax = axs[1] if len(axs) else plt.subplot(gs[0, 1])
        if (vmin is None) or (vmax is None):
            ph.plot_colormap(ax, colormap=colormap, reverse_cm=False, ylim=[0, 256], yticks=[0, 256],
                             yticklabels=['%.1f' % np.nanmin(z),
                                          '%.1f' % np.nanmax(z)],
                             ylabel='Vm (mV)', label_rotation=90, label_pos="right", label_axeratio=2)
        else:
            ph.plot_colormap(ax, colormap=colormap, reverse_cm=False, ylim=[0, 256], yticks=[0, 256],
                             yticklabels=['%.1f' % vmin, '>%.1f' % vmax],
                             ylabel='Vm (mV)', label_rotation=90, label_pos="right", label_axeratio=2)

    if not vmin is None: img.set_clim(vmin=vmin)
    if not vmax is None: img.set_clim(vmax=vmax)

def plot_tc_asym_sideturn(groups, ax=False, walking_bouts=False, speed_type='side', loc_num=0, sig_type='rmln',
                bins=20, color_lines=[ph.dorange, ph.dgreen], vel_range=[-10, 10], xticks=[-10,0,10],
                ylim=[0,8], yticks=[0,4,8], show_yaxis=True, lw_indi=1, alpha_indi=.3, lw_group=3, alpha_group=.8,
                alpha_sem=.3, fs=10):
    small_side_vel = [-1,1]
    small_turn_vel = [-50,50]
    small_forw_vel = [-2, 8]
    ph.set_fontsize(fs)
    if not ax:
        fig = plt.figure(1, (2.2, 2))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    for i_group, group in enumerate(groups):
        color = color_lines[i_group]
        ys_onegroup = []

        for fly in group:
            fly_intensity = []
            fly_behavior = [[],[]]
            fly_forwvel = []

            for rec in fly:
                if sig_type == 'rmln':
                    _sig = rec.ims[loc_num].c1.rmln
                elif sig_type == 'rmll':
                    _sig = rec.ims[loc_num].c1.rmll
                elif sig_type == 'rmlz':
                    _sig = rec.ims[loc_num].c1.rmlz

                if walking_bouts:
                    inds = get_walking_bouts_inds(rec)
                    for ind in inds:
                        t_ind = np.arange(ind[0], ind[-1] + 1)
                        fly_intensity.extend(_sig[t_ind])
                        fly_behavior[0].extend(rec.subsample('dside')[t_ind])
                        fly_behavior[1].extend(rec.subsample('dhead')[t_ind])
                        fly_forwvel.extend(rec.subsample('dforw')[t_ind])
                else:
                    fly_intensity.extend(_sig)
                    fly_behavior[0].extend(rec.subsample('dside'))
                    fly_behavior[1].extend(rec.subsample('dhead'))
                    fly_forwvel.extend(rec.subsample('dforw'))

            fly_intensity = np.array(fly_intensity)
            if len(fly_intensity):
                for i in range(2):
                    fly_behavior[i] = np.array(fly_behavior[i])
                fly_forwvel = np.array(fly_forwvel)

                if len(small_forw_vel):     # apply small forward velocity condition
                    fly_intensity = fly_intensity[
                        (fly_forwvel > small_forw_vel[0]) & (fly_forwvel < small_forw_vel[-1])]
                    fly_behavior[0] = fly_behavior[0][
                        (fly_forwvel > small_forw_vel[0]) & (fly_forwvel < small_forw_vel[-1])]
                    fly_behavior[1] = fly_behavior[1][
                        (fly_forwvel > small_forw_vel[0]) & (fly_forwvel < small_forw_vel[-1])]

                xbinnum = bins if speed_type == 'side' else 1
                ybinnum = bins if speed_type == 'turn' else 1
                xvelrange = vel_range if speed_type == 'side' else small_side_vel
                yvelrange = vel_range if speed_type == 'turn' else small_turn_vel
                x, y, z_mean = fc.scatter_to_heatmap(x_input=fly_behavior[0], y_input=fly_behavior[1],
                                                     z_input=fly_intensity, binx=xbinnum, biny=ybinnum,
                                                     xmin=xvelrange[0], xmax=xvelrange[-1],
                                                     ymin=yvelrange[0], ymax=yvelrange[-1], nan_to_num=np.nan)

                xs = x if speed_type == 'side' else y
                ys = z_mean
                ys_onegroup.append(ys)

                ax.plot(xs, ys, lw=lw_indi, color=color, alpha=alpha_indi, zorder=1)

        ys_onegroup_mean = np.nanmean(ys_onegroup, axis=0)
        ys_onegroup_sem = np.nanstd(ys_onegroup, axis=0) / np.sqrt(len(ys_onegroup))
        ax.plot(xs, ys_onegroup_mean, c=color, lw=lw_group, alpha=alpha_group, zorder=2)
        ax.fill_between(xs, ys_onegroup_mean - ys_onegroup_sem, ys_onegroup_mean + ys_onegroup_sem,
                        facecolor=color, edgecolor='none', alpha=alpha_sem)

        # trim the plot
        xlabel = 'Sideslip velocity (mm/s)' if speed_type == 'side' else 'Turning velocity (deg/s)'
        ax.set_xlabel(xlabel)
        if show_yaxis:
            ax.set_ylabel('rml')
        show_axes = ['left', 'bottom'] if show_yaxis else ['bottom']
        ph.adjust_spines(ax, show_axes, xlim=vel_range, xticks=xticks, ylim=ylim, yticks=yticks)

# keep fixing: _sig extraction (add rpl mean),  delay_beh_sec
def get_tc_1d(group, sig_type='rmlz', speed_type='side', vel_range=[-6, 6], binnum=20, loc_num=0,
              filters=[['forw', [-2,2]], ['head', [-50, 50]]], delay_beh_ms=100):
    """
    :param filters: [[_filter], [_filter]], each _filter = [f_name, f_range] e.g. ['forw', [-2, 2]]
    'forw': 0, 'head': 1, 'side': 2
    :param walking_bouts: only include data when a fly is in a 'walking bout' (see def get_walking_bouts_inds_patching)
    :return:
    """

    def f_name_to_index(f_name):
        index = -1
        if f_name == 'forw':
            index = 0
        elif f_name == 'head':
            index = 1
        elif f_name == 'side':
            index = 2
        # elif f_name == 'stimid':
        #     index = -1
        return index

    def get_sig(rec, loc_num, sig_type):
        if sig_type == 'rmln':
            _sig = rec.ims[loc_num].c1.rmln
        elif sig_type == 'rmll':
            _sig = rec.ims[loc_num].c1.rmll
        elif sig_type == 'rmlz':
            _sig = rec.ims[loc_num].c1.rmlz
        elif sig_type == 'mean_al':
            _sig = rec.ims[loc_num].c1.mean_al
        elif sig_type == 'mean_an':
            _sig = rec.ims[loc_num].c1.mean_an
        elif sig_type == 'mean_az':
            _sig = rec.ims[loc_num].c1.mean_az
        return _sig

    # subsampling_rate = group[0][0].subsampling_rate
    beh_ind_plot = f_name_to_index(speed_type)

    ys = []
    for fly in group:
        fly_sig = []
        fly_beh = []
        for _i in range(3):
            fly_beh.append([])

        for rec in fly:
            behaviors = [rec.subsample('dforw', lag_ms=delay_beh_ms),
                         rec.subsample('dhead', lag_ms=delay_beh_ms), rec.subsample('dside', lag_ms=delay_beh_ms)]
            _sig = get_sig(rec, loc_num, sig_type)

            for _i in range(3):
                fly_beh[_i].extend(behaviors[_i])
            fly_sig.extend(_sig)

        for _i in range(3):
            fly_beh[_i] = np.array(fly_beh[_i])
        fly_sig = np.array(fly_sig)

        # apply filters
        inds_filtered = (fly_beh[0] < 1e5)
        for _filter in filters:
            if len(_filter):
                f_name, f_range = _filter
                beh_index_filter = f_name_to_index(f_name)
                inds_filtered = inds_filtered & (fly_beh[beh_index_filter] > f_range[0]) & (fly_beh[beh_index_filter] < f_range[-1])
        fly_beh = fly_beh[beh_ind_plot][inds_filtered]
        fly_sig = fly_sig[inds_filtered]

        _, fly_y, _ = fc.scatter_to_errorbar(x_input=fly_beh, y_input=fly_sig, binx=binnum,
                                             xmin=vel_range[0], xmax=vel_range[-1])

        ys.append(fly_y)

    return ys

def get_tc_amp_1d(group, sig_type='mean', location='fb', loc_num=None,
                  speed_type='side', vel_range=[-6, 6], binnum=20,
                  filters=[['forw', [-2,2]], ['head', [-50, 50]]], delay_beh_ms=100):
    """
    :param filters: [[_filter], [_filter]], each _filter = [f_name, f_range] e.g. ['forw', [-2, 2]]
    'forw': 0, 'head': 1, 'side': 2
    :param walking_bouts: only include data when a fly is in a 'walking bout' (see def get_walking_bouts_inds_patching)
    :return:
    """
    def _cosine_func(x, amp, baseline, phase):
        return amp * np.sin(x + phase) + baseline

    def f_name_to_index(f_name):
        index = -1
        if f_name == 'forw':
            index = 0
        elif f_name == 'head':
            index = 1
        elif f_name == 'side':
            index = 2
        # elif f_name == 'stimid':
        #     index = -1
        return index

    def get_curves(rec, location, loc_num):
        if loc_num is None:
            _, _, phase = rec.get_objects('%s.c1.phase' % location)
            im, ch, an_cubspl = rec.get_objects('%s.c1.an_cubspl' % location)
        else:
            phase = rec.ims[loc_num].c1.phase
            im = rec.ims[loc_num]
            an_cubspl = im.c1.an_cubspl
        an_nophase = im.cancel_phase(an_cubspl, phase, 10, im.c1.celltype, offset=0)
        return an_nophase

    def get_sigs(curves_bins_fly, location, binnum, sig_type):
        amps_fly = []
        for ibin in range(binnum):
            amps_fly.append(np.nan)

        for ibin in range(binnum):
            curves_bin = curves_bins_fly[ibin]
            if not isinstance(curves_bin, np.ndarray):
                continue
            if location == 'pb':
                len_shape = len(curves_bin)
                curve1 = curves_bin[:(len_shape / 2)]
                curve1 = curve1[~np.isnan(curve1)]
                curve2 = curves_bin[(len_shape / 2):]
                curve2 = curve2[~np.isnan(curve2)]
            else:
                curve1 = curves_bin
                curve2 = curves_bin
            x_data = np.radians(np.linspace(0, 360, len(curve1)))
            params1, _ = optimize.curve_fit(_cosine_func, x_data, curve1, p0=[1, 1, 1])
            params2, _ = optimize.curve_fit(_cosine_func, x_data, curve2, p0=[1, 1, 1])
            amp1, amp2 = np.abs(params1[0]), np.abs(params2[0])
            if sig_type == 'mean':
                amps_fly[ibin] = (amp1+amp2)/2.
            elif sig_type == 'lmr':
                amps_fly[ibin] = amp1 - amp2
        return amps_fly

    # subsampling_rate = group[0][0].subsampling_rate
    beh_ind_plot = f_name_to_index(speed_type)
    ys = []

    for fly in group:
        fly_curve = []
        fly_beh = []
        for _i in range(3):
            fly_beh.append([])

        for rec in fly:
            behaviors = [rec.subsample('dforw', lag_ms=delay_beh_ms),
                         rec.subsample('dhead', lag_ms=delay_beh_ms), rec.subsample('dside', lag_ms=delay_beh_ms)]
            for _i in range(3):
                fly_beh[_i].extend(behaviors[_i])

            _curve = get_curves(rec, location, loc_num)
            fly_curve.extend(_curve)

        for _i in range(3):
            fly_beh[_i] = np.array(fly_beh[_i])
        fly_curve = np.array(fly_curve)

        # apply filters
        inds_filtered = (fly_beh[0] < 1e5)
        for _filter in filters:
            if len(_filter):
                f_name, f_range = _filter
                beh_index_filter = f_name_to_index(f_name)
                inds_filtered = inds_filtered & (fly_beh[beh_index_filter] > f_range[0]) & (fly_beh[beh_index_filter] < f_range[-1])
        fly_beh = fly_beh[beh_ind_plot][inds_filtered]
        fly_curve = fly_curve[inds_filtered]

        # get nophase interpolated curves for each bin
        _, curves_bins_fly, _ = fc.scatter_to_errorbar2(x_input=fly_beh, y_input=fly_curve, binx=binnum,
                                             xmin=vel_range[0], xmax=vel_range[-1])

        # calculate amp of each bin, save them to ys
        fly_y = get_sigs(curves_bins_fly, location, binnum, sig_type)
        ys.append(fly_y)

    return ys

def plot_tc_1d_2asymm_1mean_20191017(groups, binnum=20, loc_num=0, colors=['black'], delay_beh_ms=100,
    dside_range=[-6,6], dhead_range=[-80,80], dforw_range=[-4,10], filter_dside=[-1,1], filter_dhead=[-50,50],
    filter_dforw=[-2,8], xticks_dside=[-6,0,6], xticks_dhead=[-80,0,80], xticks_dforw=[-4,0,4,8],
    ylim_asymm=[[-.5,.5],], yticks_asymm=[[-.5,0,.5],], ylim_mean=[[0,.5],], yticks_mean=[[0,.5],],
    lw=2, lw_indi=1, alpha_indi=.5, alpha_sem=.2, show_flynum=True, fs=8):

    def plot_curves(ax, ys, color='black', xlabel='', ylabel='', show_axes=[],
                    xlim=[0,1], xticks=[], ylim=[0,1], yticks=[], ylabel_xcoords=-.25):
        for y in ys:
            ax.plot(xs, y, lw=lw_indi, color=color, alpha=alpha_indi)
        ymean = np.nanmean(ys, axis=0)
        ystd = np.nanstd(ys, axis=0)
        ysem = ystd / np.sqrt(len(ys))
        ax.plot(xs, ymean, lw=lw, color=color)
        ax.fill_between(xs, ymean - ysem, ymean + ysem, facecolor=color, edgecolor='none', alpha=alpha_sem)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel, rotation='vertical', va='center')
        ax.get_yaxis().set_label_coords(ylabel_xcoords, 0.5)
        ax.patch.set_facecolor('white')
        ax.grid(which='major', alpha=0)
        ph.adjust_spines(ax, show_axes, xlim=xlim, xticks=xticks, ylim=ylim, yticks=yticks)

    ph.set_fontsize(fs)
    nrow = len(groups)
    _ = plt.figure(1, (2.5*2.5+.5, 2*nrow))
    gs = gridspec.GridSpec(nrow, 4, width_ratios=[2.5,2.5,.5,2.5])

    for row, group in enumerate(groups):
        color = colors[row % nrow]

        ax = plt.subplot(gs[row, 0])
        xs = np.linspace(dside_range[0], dside_range[-1], binnum)
        ys = get_tc_1d(group, sig_type='rmlz', speed_type='side', vel_range=dside_range, binnum=binnum, loc_num=loc_num,
                       filters=[['forw',filter_dforw], ['head',filter_dhead]], delay_beh_ms=delay_beh_ms)
        _xlabel = 'Sideslip velocity (mm/s)' if row == nrow - 1 else ''
        plot_curves(ax, ys, color=color, xlabel=_xlabel, ylabel='R-L\ndF/F', show_axes=['left', 'bottom'],
                    xlim=dside_range, xticks=xticks_dside, ylim=ylim_asymm[row], yticks=yticks_asymm[row])
        if show_flynum:
            ax.text(dside_range[-1], ylim_asymm[row][-1], '%i flies' % len(ys), fontsize=fs, va='top', ha='right')

        ax = plt.subplot(gs[row, 1])
        xs = np.linspace(dhead_range[0], dhead_range[-1], binnum)
        ys = get_tc_1d(group, sig_type='rmlz', speed_type='head', vel_range=dhead_range, binnum=binnum, loc_num=loc_num,
                       filters=[['forw', filter_dforw], ['side', filter_dside]], delay_beh_ms = delay_beh_ms)
        plot_curves(ax, ys, color=color, xlabel='Heading velocity (mm/s)', ylabel='', show_axes=['bottom'],
                    xlim=dhead_range, xticks=xticks_dhead, ylim=ylim_asymm[row], yticks=yticks_asymm[row])

        ax = plt.subplot(gs[row, 3])
        xs = np.linspace(dforw_range[0], dforw_range[-1], binnum)
        ys = get_tc_1d(group, sig_type='mean_an', speed_type='forw', vel_range=dforw_range, binnum=binnum,
                           loc_num=loc_num, delay_beh_ms = delay_beh_ms, filters=[],)
        plot_curves(ax, ys, color=color, xlabel='Forward velocity (mm/s)', ylabel='R+L\n$\Delta F/F$', show_axes=['left', 'bottom'],
                    xlim=dforw_range, xticks=xticks_dforw, ylim=ylim_mean[row], yticks=yticks_mean[row], ylabel_xcoords=-.1)

def plot_tc_1d_2asymm_1mean_20191105(groups, binnum=20, loc_num=0, colors=['black'],
                                     dside_range=[-6, 6], dhead_range=[-80, 80], dforw_range=[-4, 10],
                                     filter_dside=[-1, 1], filter_dhead=[-50, 50],
                                     xticks_dside=[-6, 0, 6], xticks_dhead=[-80, 0, 80], xticks_dforw=[-4, 0, 4, 8],
                                     ylim_asymm=[-.5, .5], yticks_asymm=[-.5, 0, .5], ylim_mean=[0, .5],
                                     yticks_mean=[0, .5], binnum_forw=4, cm=plt.cm.viridis,
                                     lw=2, lw_indi=1, alpha_indi=.5, alpha_sem=.2, show_flynum=True, fs=8):
    # try to plot asym strength as a func of dforw, results not as expected, probably GCaMP not fast enough
    def plot_curves(ax, ys, color='black', xlabel='', ylabel='', show_axes=[],
                    xlim=[0, 1], xticks=[], ylim=[0, 1], yticks=[], label=''):
        # for y in ys:
        #     ax.plot(xs, y, lw=lw_indi, color=color, alpha=alpha_indi)
        ymean = np.nanmean(ys, axis=0)
        ystd = np.nanstd(ys, axis=0)
        ysem = ystd / np.sqrt(len(ys))
        ax.plot(xs, ymean, lw=lw, color=color, label=label)
        ax.fill_between(xs, ymean - ysem, ymean + ysem, facecolor=color, edgecolor='none', alpha=alpha_sem)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel, rotation='vertical', va='center')
        ax.get_yaxis().set_label_coords(-.25, 0.5)
        ax.patch.set_facecolor('white')
        ax.grid(which='major', alpha=0)
        ph.adjust_spines(ax, show_axes, xlim=xlim, xticks=xticks, ylim=ylim, yticks=yticks)

    ph.set_fontsize(fs)
    nrow = len(groups)
    _ = plt.figure(1, (2.5 * 3 + .5, 2 * nrow))
    gs = gridspec.GridSpec(nrow, 4, width_ratios=[2.5, 2.5, .5, 2.5])

    for row, group in enumerate(groups):

        ax = plt.subplot(gs[row, 0])
        # color = colors[row % nrow]
        dforw_list = np.linspace(dforw_range[0], dforw_range[-1], binnum_forw + 1)
        color_ind_seq = np.linspace(0, 1, binnum_forw)

        for i_dforw in range(binnum_forw):
            filter_dforw = ['forw', [dforw_list[i_dforw], dforw_list[i_dforw + 1]]]
            forw_vel = (dforw_list[i_dforw] + dforw_list[i_dforw + 1]) / 2.

            xs = np.linspace(dside_range[0], dside_range[-1], binnum)
            ys = get_tc_1d(group, sig_type='rmlz', speed_type='side', vel_range=dside_range, binnum=binnum,
                           loc_num=loc_num, filters=[filter_dforw, ['head', filter_dhead]],
                           delay_beh_sec=0.04)
            _xlabel = 'Sideslip velocity (mm/s)' if row == nrow - 1 else ''
            plot_curves(ax, ys, color=cm(color_ind_seq[i_dforw]), xlabel=_xlabel, ylabel='RmL\nZscore',
                        show_axes=['left', 'bottom'], xlim=dside_range, xticks=xticks_dside, ylim=ylim_asymm,
                        yticks=yticks_asymm, label='%.1f' % forw_vel)
        if show_flynum:
            ax.text(dside_range[-1], ylim_asymm[-1], '%i flies' % len(ys), fontsize=fs, va='top', ha='right')
        ax.legend(prop={'size': fs}, loc='upper right')

        # ax = plt.subplot(gs[row, 1])
        # xs = np.linspace(dhead_range[0], dhead_range[-1], binnum)
        # ys = get_tc_1d(group, sig_type='rmlz', speed_type='head', vel_range=dhead_range, binnum=binnum,
        #                loc_num=loc_num, filters=[['forw', filter_dforw], ['side', filter_dside]],
        #                delay_beh_sec=0.04)
        # plot_curves(ax, ys, color=color, xlabel='Heading velocity (mm/s)', ylabel='', show_axes=['bottom'],
        #             xlim=dhead_range, xticks=xticks_dhead, ylim=ylim_asymm, yticks=yticks_asymm)
        #
        # ax = plt.subplot(gs[row, 3])
        # xs = np.linspace(dforw_range[0], dforw_range[-1], binnum)
        # ys = get_tc_1d(group, sig_type='mean_al', speed_type='forw', vel_range=dforw_range, binnum=binnum,
        #                loc_num=loc_num, delay_beh_sec=0.04, filters=[], )
        # plot_curves(ax, ys, color=color, xlabel='Forward velocity (mm/s)', ylabel='Ave.$\Delta F/F$',
        #             show_axes=['left', 'bottom'],
        #             xlim=dforw_range, xticks=xticks_dforw, ylim=ylim_mean, yticks=yticks_mean)


# Stim bouts, NEW, 2019 Oct

def get_stimbout_td_vs_h_flight(recss, sig_type='rmlz', stimid_special=1, t_win_plot=[-4,4,4], t_win_stop=[-8,8],
                                stim_method='dypos', loc_num=0):

    def get_sig(rec, loc_num, sig_type):
        if sig_type == 'rmln':
            _sig = rec.ims[loc_num].c1.rmln
        elif sig_type == 'rmll':
            _sig = rec.ims[loc_num].c1.rmll
        elif sig_type == 'rmlz':
            _sig = rec.ims[loc_num].c1.rmlz
        elif sig_type == 'mean_al':
            _sig = rec.ims[loc_num].c1.mean_al
        elif sig_type == 'mean_an':
            _sig = rec.ims[loc_num].c1.mean_an
        elif sig_type == 'mean_az':
            _sig = rec.ims[loc_num].c1.mean_az
        elif sig_type == 'c2.rmlz':
            _sig = rec.ims[loc_num].c2.rmlz
        elif sig_type == 'c2.mean_az':
            _sig = rec.ims[loc_num].c2.mean_az
        elif sig_type == 'c2.azl':
            _sig = rec.ims[loc_num].c2.az[:,0]
        elif sig_type == 'c2.azr':
            _sig = rec.ims[loc_num].c2.az[:,1]
        return _sig

    t_interp = np.arange(t_win_plot[0] - .5, (t_win_plot[1]+t_win_plot[2])+.5, 0.05)
    sig_group = []
    for fly in recss:
        sig_fly = []

        for rec in fly:
            if stim_method == 'dypos':
                of_stimids, of_start_idxs, _ = td.get_stim_info_20191007(rec)
            elif stim_method == 'dxpos':
                of_stimids, of_start_idxs, _ = td.get_stim_info_20200224_xposonly(rec)
            stop_inds = td.get_flight_stopping_inds(rec)
            ssr_tif = rec.ims[0].sampling_rate
            ssr_abf = rec.abf.subsampling_rate

            for start_idx in of_start_idxs[of_stimids==stimid_special]:
                start_idx_abf = np.where(rec.abf.t < rec.ims[0].t[start_idx])[0][-1]
                idx_stopwindow = [start_idx_abf - np.ceil(np.abs(t_win_stop[0])*ssr_abf),
                                  start_idx_abf + np.ceil(np.abs(t_win_stop[-1])*ssr_abf)]

                if not np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
                    t0 = rec.abf.t[start_idx_abf]
                    idx1 = int(start_idx)
                    idx0 = int(idx1 - np.ceil((np.abs(t_win_plot[0])+1) * ssr_tif))
                    idx2 = int(idx1 + np.ceil(t_win_plot[1] * ssr_tif))
                    idx3 = int(idx2 + np.ceil((t_win_plot[2]+1) * ssr_tif))
                    t_tif = rec.ims[0].t[idx0:idx3] - t0
                    sig = get_sig(rec, loc_num, sig_type)[idx0:idx3]
                    # print t_tif[0], t_tif[1]
                    # print t_interp[0], t_interp[1]
                    _f = interp1d(t_tif, sig, kind='linear')
                    sig_interp = fc.wrap(_f(t_interp))
                    sig_fly.append(sig_interp)

        if not np.sum(np.isnan(np.nanmean(sig_fly, axis=0))):
            sig_group.append(np.nanmean(sig_fly, axis=0))

    return t_interp, sig_group

def plot_stimbout_td_vs_h_flight_unit(recss, ax=False, sig_type='rmlz', stimid_special=1, t_win_plot=[-4,4,4],
                                      t_win_stop=[-8,8], stim_method='dypos', loc_num=0, lw=2, lw_indi=1, alpha_indi=.4, alpha_sem=.2,
                                      color='black', xlabel='time (s)', ylabel='rmlz', show_axes=['left'],
                                      xticks=[-4,0,4,8], ylim=[-1,1], yticks=[-1,0,1], show_xscale=True,
                                      label_axeratio=-.25, fs=10, show_stimid_special=True, show_indi=True,
                                      show_flynum=True):

    if not ax:
        _ = plt.figure(1, (3, 2))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    xlim = [t_win_plot[0], t_win_plot[1]+t_win_plot[2]]
    xs, ys = get_stimbout_td_vs_h_flight(recss, sig_type=sig_type, stimid_special=stimid_special, stim_method=stim_method,
                                         t_win_plot=t_win_plot, t_win_stop=t_win_stop, loc_num=loc_num)
    ymean = np.nanmean(ys, axis=0)
    ystd = np.nanstd(ys, axis=0)
    ysem = ystd / np.sqrt(len(ys))
    if show_indi:
        for y in ys:
            ax.plot(xs, y, lw=lw_indi, color=color, alpha=alpha_indi)
    ax.plot(xs, ymean, lw=lw, color=color)
    ax.fill_between(xs, ymean - ysem, ymean + ysem, facecolor=color, edgecolor='none', alpha=alpha_sem)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation='vertical', va='center')
    ax.get_yaxis().set_label_coords(label_axeratio, 0.5)
    ax.patch.set_facecolor('white')
    ax.grid(which='major', alpha=0)
    ph.adjust_spines(ax, show_axes, xlim=xlim, xticks=xticks, ylim=ylim, yticks=yticks)

    ax.axvline(0, ls='--', lw=1, c=ph.red, alpha=1, zorder=3)
    ax.axvline(t_win_plot[1], ls='--', lw=1, c=ph.red, alpha=1, zorder=3)
    ax.axvline(8, ls='--', lw=1, c=ph.red, alpha=1, zorder=3)
    ax.axhline(0, ls='--', lw=1, c=ph.grey5, alpha=.3, zorder=1)
    if show_xscale:
        ph.plot_x_scale_text(ax, bar_length=2, text='2s', y_text_lower=0.15, x_text_lefter=.05, fontsize=fs)

    if show_stimid_special:
        ax.text(xlim[0], ylim[-1], 'id = %i' % stimid_special, fontsize=fs, va='top', ha='left')

    if show_flynum:
        ax.text(xlim[0], ylim[-1], '%i flies' % len(recss), fontsize=fs, va='top', ha='left')

def plot_stimbout_td_vs_h_flight_8dir_20191020(recss, sig_type='rmlz', t_win_plot=[-4,4,4], t_win_stop=[-8,8],
                                               loc_num=0, lw=2, lw_indi=1, alpha_indi=.4, alpha_sem=.2,
                                               color='black', xlabel='', ylabel='rmlz', show_axes=['left'],
                                               xticks=[-4,0,4,8], ylim=[-1,1], yticks=[-1,0,1], show_xscale=True,
                                               label_axeratio=-.25, fs=10, show_stimid_special=True, show_indi=True):
    ph.set_fontsize(fs)
    nrow, ncol = 2, 4
    _ = plt.figure(1, (2.5 * 4, 2 * 2))
    gs = gridspec.GridSpec(nrow, ncol)

    for row in range(2):
        for col in range(4):
            stimid_special = row * 4 + col + 1
            ax = plt.subplot(gs[row, col])
            _ylabel = ylabel if col == 0 else ''
            _show_axes = show_axes if col == 0 else []
            plot_stimbout_td_vs_h_flight_unit(recss, ax, sig_type, stimid_special, t_win_plot, t_win_stop, loc_num, lw,
                                              lw_indi, alpha_indi, alpha_sem, color, xlabel, _ylabel, _show_axes,
                                              xticks, ylim, yticks, show_xscale, label_axeratio, fs,
                                              show_stimid_special, show_indi)

def plot_stimbout_td_vs_h_flight_8dir_20191103(groups, sig_type='rmlz', stimids=[3,2,1,6,5,4,7,8],
                                               t_win_plot=[-4,4,4], t_win_stop=[-8,8], stim_method='dypos',
                                               loc_num=0, lw=2, lw_indi=.7, alpha_indi=.3, alpha_sem=.4,
                                               colors=['black'], ylabel='rmlz', show_axes=['left'],
                                               xlabels=['-120','-60','0','60','120','180','Lyaw','Ryaw'],
                                               xticks=[-4,0,4,8], ylim=[-1,1], yticks=[-1,0,1], show_xscale=True,
                                               label_axeratio=-.25, fs=10, show_stimid_special=True, show_indi=True,
                                               show_flynum=True):
    ph.set_fontsize(fs)
    nrow = len(groups)
    nrol = len(stimids)
    _ = plt.figure(1, (2.5 * nrol, 2 * nrow))
    gs = gridspec.GridSpec(nrow, nrol)

    for row in range(nrow):
        recss = groups[row]
        for col in range(nrol):
            stimid_special = stimids[col]
            ax = plt.subplot(gs[row, col])
            _ylabel = ylabel if col == 0 else ''
            _show_axes = show_axes if col == 0 else []
            _show_xscale = show_xscale if col == 0 and row == 0 else False
            _show_flynum = show_flynum if col == 0 else False
            color = colors[row%nrow]
            xlabel = xlabels[col] if (row == nrow - 1) else ''
            plot_stimbout_td_vs_h_flight_unit(recss, ax, sig_type, stimid_special, t_win_plot, t_win_stop, stim_method,
                                              loc_num, lw, lw_indi, alpha_indi, alpha_sem, color, xlabel, _ylabel,
                                              _show_axes, xticks, ylim, yticks, _show_xscale, label_axeratio, fs,
                                              show_stimid_special, show_indi, _show_flynum)


def get_ave_stimbout_td_vs_h_flight(recss, sig_type='rmlz', stimid_special=1, t_before_end=-2, t_end=0, t_win_stop=[-12,4],
                                    t_win_interp=[-3,1], loc_num=0):

    def get_sig(rec, loc_num, sig_type):
        if sig_type == 'rmln':
            _sig = rec.ims[loc_num].c1.rmln
        elif sig_type == 'rmll':
            _sig = rec.ims[loc_num].c1.rmll
        elif sig_type == 'rmlz':
            _sig = rec.ims[loc_num].c1.rmlz
        elif sig_type == 'mean_al':
            _sig = rec.ims[loc_num].c1.mean_al
        elif sig_type == 'mean_an':
            _sig = rec.ims[loc_num].c1.mean_an
        elif sig_type == 'mean_az':
            _sig = rec.ims[loc_num].c1.mean_az
        return _sig

    t_interp = np.arange(t_before_end, t_end, 0.1)
    sig_group = []
    for fly in recss:
        sig_fly = []

        for rec in fly:
            of_stimids, of_start_idxs, of_end_idxs = td.get_stim_info_20191007(rec)
            stop_inds = td.get_flight_stopping_inds(rec)
            ssr_tif = rec.ims[0].sampling_rate
            ssr_abf = rec.abf.subsampling_rate

            for end_idx in of_end_idxs[of_stimids == stimid_special]:
                end_idx_abf = np.where(rec.abf.t < rec.ims[0].t[end_idx])[0][-1]
                idx_stopwindow = [end_idx_abf - np.ceil(np.abs(t_win_stop[0]) * ssr_abf),
                                  end_idx_abf + np.ceil(np.abs(t_win_stop[-1]) * ssr_abf)]

                if not np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
                    t0 = rec.abf.t[end_idx_abf]
                    idx_end = int(end_idx)
                    idx0 = int(idx_end - np.ceil(np.abs(t_win_interp[0]) * ssr_tif))
                    idx1 = int(idx_end + np.ceil(t_win_interp[1] * ssr_tif))

                    t_tif = rec.ims[0].t[idx0:idx1] - t0
                    sig = get_sig(rec, loc_num, sig_type)[idx0:idx1]
                    _f = interp1d(t_tif, sig, kind='linear')
                    sig_interp = fc.wrap(_f(t_interp))
                    sig_fly.append(np.nanmean(sig_interp, axis=0))

        sig_group.append(np.nanmean(sig_fly, axis=0))

    return sig_group

def plot_ave_stimbout_td_vs_h_flight_20191105(recss, sig_type='rmlz', ylim=[-2.5,2.5], yticks=[-2,-1,0,1,2],
    stimids=[3, 2, 1, 6, 5, 4, 7, 8], t_before_end=-2, t_end=0, t_win_stop=[-12, 4], t_win_interp=[-3, 1], loc_num=0,
    group_labels=['-120','-60','0','60','120','180','Lyaw','Ryaw'], ylabel='R-L zscore', label_axeratio=-.25,
    c_marker=ph.grey5, fs=10):

    _ = plt.figure(1, (2.5, 2.5))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])

    xss = []
    for stimid_special in stimids:
        xs = get_ave_stimbout_td_vs_h_flight(recss, sig_type=sig_type, stimid_special=stimid_special,
                                             t_before_end=t_before_end, t_end=t_end, t_win_stop=t_win_stop,
                                             t_win_interp=t_win_interp, loc_num=loc_num)
        xss.append(xs)
    group_info = [[i] for i in range(len(stimids))]
    ph.errorbar_lc_xss(xss=xss, ax=ax, group_info=group_info, group_labels=group_labels,
                       colors=[c_marker],
                       ylabel=ylabel, yticks=yticks, ylim=ylim, yticks_minor=[], fs=fs,
                       ro_x=45, label_axeratio=label_axeratio, rotate_ylabel=False, plot_bar=False,
                       subgroup_samefly=False, circstat=True, legend_list=group_labels,
                       ms_indi=4, alpha_indi=.6,)


def plot_sample_trace_walking(rec, tlim=[0, 10], imaging='pb.c1.an', curvetype='c1.mean_an', plot_tif=True,
                              ylims=[[0, 2], [-2, 15], [0, 500]], yticks=[[0, 1, 2], [0, 5, 10, 15], [0, 200, 400]],
                              ybar_length_list = [1,10,180], ybar_text_list = ['1', '20\nmm', '180\ndeg'],
                              velocity=False, cmap=plt.cm.Blues, subsample_beh=False,
                              time_bar_length=5, time_text='5s', colors=['black', 'black', 'black'], fs=15, lw=2,
                              colorbar=True, vmin=0, vmax=None, show_flyid=True, turning_moments=[], forw_moments=[],
                              turn_color=ph.dblue_pure, forw_color=ph.dred_pure, alpha_shade=.1,
                              hide_ylabels=False, label_axeratio=-.07,):
    def hr_fcn(imaging_label):
        if 'eb' in imaging_label:
            return 1.5
        elif 'pb' in imaging_label:
            return 2
        elif 'fb' in imaging_label:
            return 1.5
        elif 'no' in imaging_label:
            return 0.5
        elif 'sg' in imaging_label:
            return 2
        return 1.5

    ph.set_fontsize(fs)
    if plot_tif:
        fig = plt.figure(1, (3, 3))
        gs = gridspec.GridSpec(50, 50)
    else:
        fig = plt.figure(1, (10, 4))
        gs = gridspec.GridSpec(3, 1)

    tif_t = rec.ims[0].t
    abf_t = rec.abf.t
    tif_inds = np.arange(np.where(tif_t <= tlim[0])[0][-1], np.where(tif_t >= tlim[-1])[0][0])
    abf_inds = np.arange(np.where(abf_t <= tlim[0])[0][-1], np.where(abf_t >= tlim[-1])[0][0])

    #############
    # plot tif
    if plot_tif:
        imlabel, clabel, alabel = rec.get_labels(imaging)
        im, c, _ = rec.get_objects(imaging)
        (t, xedges), gc = rec.get_signal(imaging)

        # Setup axis
        ax = plt.subplot(gs[:10, :48])
        ph.adjust_spines(ax, [], ylim=[xedges[0], xedges[-1]], xlim=tlim)

        # Plot gc
        gc[np.isnan(gc)] = 0
        gc = gc.T
        img = ax.pcolormesh(tif_t[tif_inds], xedges[1:], gc[:,tif_inds], cmap=cmap)
        if not vmin is None: img.set_clim(vmin=vmin)
        if not vmax is None: img.set_clim(vmax=vmax)
        ax.set_ylabel('Glom.', rotation=90)
        ax.yaxis.set_label_coords(label_axeratio, .5)

        # Draw colorbar
        if colorbar:
            ax = plt.subplot(gs[1:9, 49:])
            gradient = np.linspace(0, 1, 256)
            gradient = np.vstack((gradient, gradient)).T
            ax.imshow(gradient, aspect='auto', cmap=cmap)
            if vmax:
                cmax = vmax
            else:
                idx = (t >= tlim[0]) & (t < tlim[1])
                cmax = np.floor(gc[:,idx].max())
                if cmax == 0:
                    cmax = np.floor(gc[:,idx].max() * 10) / 10.
            ax.set_yticklabels(['%1.0f' % vmin, '%1.0f' % cmax])
            ax.yaxis.tick_right()
            ax.tick_params('both', length=2, width=1, which='major')
            # ax.set_ylabel('$\Delta$F/F', rotation=0, va='bottom', ha='left')
            ax.yaxis.set_label_position("right")
            ph.adjust_spines(ax, ['right'], pad=0, lw=0, ylim=[0, 255], yticks=[0, 255], xticks=[])
            ax.spines['right'].set_visible(False)

    #############
    # plot curves
    if subsample_beh:
        ts = [tif_t[tif_inds], tif_t[tif_inds], tif_t[tif_inds]]
    else:
        ts = [tif_t[tif_inds], abf_t[abf_inds], abf_t[abf_inds]]

    _, sig_tif = rec.get_signal(curvetype)

    if velocity:
        if subsample_beh:
            sigs = [sig_tif[tif_inds], rec.subsample(rec.abf.dforw)[tif_inds],
                    rec.subsample(rec.abf.dhead)[tif_inds]]
        else:
            sigs = [sig_tif[tif_inds], rec.abf.dforw[abf_inds], np.abs(rec.abf.dhead[abf_inds])]
        ylabels = ['$\Delta$F/F0\nAverage', 'Forward\nvelocity\n(mm/s)', 'Turning\nspeed\n(deg/s)']
    else:
        if subsample_beh:
            sigs = [sig_tif[tif_inds], fc.deg2mm(rec.subsample(rec.abf.forw)[tif_inds]),
                    np.abs(rec.subsample(rec.abf.head)[tif_inds])]
        else:
            sigs = [sig_tif[tif_inds], fc.deg2mm(rec.abf.forw[abf_inds]), np.abs(rec.abf.head[abf_inds])]
        ylabels = ['$\Delta F/F_0$', 'Forw.', 'Turn']

    noline_thresholds = [0, 8, 135]
    bar_length_list = ybar_length_list
    bar_text_list = ybar_text_list
    for row in range(3):
        if row == 0:
            ax = plt.subplot(gs[10:30, :48])
        else:
            ax = plt.subplot(gs[(20 + row * 10):(30 + row * 10), :48])
        ph.adjust_spines(ax, [], lw=2, ylim=ylims[row], yticks=yticks[row],
                         xlim=[tlim[0], tlim[1] + (tlim[1] - tlim[0]) * 0.01], xticks=[])
        ax.patch.set_facecolor('white')
        ax.grid(which='major', alpha=0)

        if row == 0:
            ax.plot(ts[row], sigs[row], c=colors[row], lw=lw)
            ph.plot_x_scale_text(ax, bar_length=time_bar_length, text=time_text, y_text_lower=.2, x_text_lefter=.05)
        else:
            ph.plot_noline_between_lims(ax, ts[row], sigs[row], threshold=noline_thresholds[row], color=colors[row], lw=lw)

        ph.plot_y_scale_text(ax, bar_length=bar_length_list[row], text=bar_text_list[row], bar_pos='bot',
                             x_text_righter=0.05, text2bar_ratio=.02, color='black', rotation='horizontal',
                             ha='left', va='bottom', fs=fs,lw=1)

        if not hide_ylabels:
            ax.set_ylabel(ylabels[row], rotation='vertical', va='center')
            ax.get_yaxis().set_label_coords(label_axeratio, 0.5)

        if row == 2 and show_flyid:
            x_text = tlim[0] + (tlim[-1] - tlim[0]) / 2.
            y_text = ylims[row][0] - (ylims[row][-1] - ylims[row][0]) / 20.
            ax.text(x_text, y_text, '%s' % rec.abf.abffile[-19:-4], ha='center', va='top', fontsize=fs / 2)

        for turning_moment in turning_moments:
            [t0, t1] = turning_moment
            ax.fill_between([t0, t1], ylims[row][0], ylims[row][-1], edgecolor='none', facecolor=turn_color, alpha=alpha_shade)

        for forw_moment in forw_moments:
            [t0, t1] = forw_moment
            ax.fill_between([t0, t1], ylims[row][0], ylims[row][-1], edgecolor='none', facecolor=forw_color, alpha=alpha_shade)


# added May 03, 2019
def plot_sample_trace_walking_paper(rec, tlim=[1, 10], curvetype='c1.mean_an', beh_type='abs',
                                    ylims=[[0, 2],[-10, 10], [-180, 180]], ybar_fluo=1,
                                    yticks=[[], [], []], yticks_minors=[[], [],[]],
                                    colors=['black', 'black', 'black'],
                                    subsample_beh=False, time_bar_length=5, time_text='5s', fs=15, lw=2,
                                    hide_ylabels=False, label_axeratio=-.07, show_flyid=False,
                                    turning_moments = [], forw_moments = [],
                                    turn_color=ph.dblue_pure, forw_color=ph.dred_pure, alpha_shade=.1):
    ph.set_fontsize(fs)
    # fig = plt.figure(1, (2.5, 2))
    fig = plt.figure(1, (10, 2.5))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2,1,1])

    tif_t = rec.ims[0].t
    abf_t = rec.abf.t
    tif_inds = np.arange(np.where(tif_t <= tlim[0])[0][-1], np.where(tif_t >= tlim[-1])[0][0])
    abf_inds = np.arange(np.where(abf_t <= tlim[0])[0][-1], np.where(abf_t >= tlim[-1])[0][0])

    # plot curves

    if subsample_beh:
        ts = [tif_t[tif_inds], tif_t[tif_inds], tif_t[tif_inds]]
    else:
        ts = [tif_t[tif_inds], abf_t[abf_inds], abf_t[abf_inds]]

    _, sig_tif = rec.get_signal(curvetype)

    if beh_type == 'abs':
        if subsample_beh:
            sigs = [sig_tif[tif_inds], fc.deg2mm(rec.subsample(rec.abf.forw)[tif_inds]),
                    np.abs(rec.subsample(rec.abf.head)[tif_inds])]
        else:
            sigs = [sig_tif[tif_inds], fc.deg2mm(rec.abf.forw[abf_inds]), np.abs(rec.abf.head[abf_inds])]
        ylabels = ['$\Delta F/F_0$', 'Forw.', 'Turning']
    else:
        if subsample_beh:
            sigs = [sig_tif[tif_inds], rec.subsample(rec.abf.dforw)[tif_inds],
                    rec.subsample(rec.abf.dhead)[tif_inds]]
        else:
            # sigs = [sig_tif[tif_inds], rec.abf.dforw[abf_inds], np.abs(rec.abf.dhead[abf_inds])]
            n_movwin = 7
            sigs = [sig_tif[tif_inds],
                   np.convolve(rec.abf.dforw[abf_inds], np.ones((n_movwin,)) / n_movwin, mode='same'),
                   # np.convolve(np.abs(rec.abf.dhead[inds]), np.ones((n_movwin,)) / n_movwin, mode='same')]
                   np.convolve(rec.abf.dhead[abf_inds], np.ones((n_movwin,)) / n_movwin, mode='same')]
        ylabels = ['$\Delta F/F_0$', 'Forward\nvelocity', 'Turning\nspeed']

    noline_thresholds = [0, 8, 135]
    bar_length_list = [ybar_fluo, 20, 360] if beh_type == 'abs' else [ybar_fluo, 10, 100]
    bar_text_list = [str(ybar_fluo), '20\nmm', '360\ndeg'] if beh_type == 'abs' else [str(ybar_fluo), '10\nmm/s', '100\ndeg/s']
    for row in range(3):
        ax = plt.subplot(gs[row, 0])
        ph.adjust_spines(ax, [], lw=2, ylim=ylims[row], yticks=yticks[row], yticks_minor=yticks_minors[row],
                         xlim=[tlim[0], tlim[1] + (tlim[1] - tlim[0]) * 0.01], xticks=[])
        ax.patch.set_facecolor('white')
        ax.grid(which='major', alpha=0)

        if row == 0:
            ax.plot(ts[row], sigs[row], c=colors[row], lw=lw)
            ph.plot_x_scale_text(ax, bar_length=time_bar_length, text=time_text, y_text_lower=.1, x_text_lefter=.05)
        else:
            ph.plot_noline_between_lims(ax, ts[row], sigs[row], threshold=noline_thresholds[row], color=colors[row], lw=lw)

        ph.plot_y_scale_text(ax, bar_length=bar_length_list[row], text=bar_text_list[row], bar_pos='bot',
                             x_text_righter=0.05, text2bar_ratio=.02, color='black', rotation='horizontal',
                             ha='left', va='bottom', fs=fs, lw=1)

        if not hide_ylabels:
            ax.set_ylabel(ylabels[row], rotation='vertical', va='center')
            ax.get_yaxis().set_label_coords(label_axeratio, 0.5)

        if row == 2 and show_flyid:
            x_text = tlim[0] + (tlim[-1] - tlim[0]) / 2.
            y_text = ylims[row][0] - (ylims[row][-1] - ylims[row][0]) / 20.
            ax.text(x_text, y_text, '%s' % rec.abf.abffile[-19:-4], ha='center', va='top', fontsize=fs / 2)

        for turning_moment in turning_moments:
            [t0, t1] = turning_moment
            ax.fill_between([t0, t1], ylims[row][0], ylims[row][-1], edgecolor='none', facecolor=turn_color, alpha=alpha_shade)

        for forw_moment in forw_moments:
            [t0, t1] = forw_moment
            ax.fill_between([t0, t1], ylims[row][0], ylims[row][-1], edgecolor='none', facecolor=forw_color, alpha=alpha_shade)


# plot summary for 2p data of each genotype


def plot_2p_summary(groups, colors=[ph.grey8, ph.blue], ch_num=2, filtertype_locomotion=0, kicknan=False, nophase=False,
                    signal_type='pb.c2.mean_an', neuropil='ebproj', offset=45, legend_list=['KirM', 'Kir'], alpha=0.35,
                    proj2pb=False, ylim_FFT=[0, 0.5], yticks_FFT=[0, 0.2, 0.4],
                    ylim_xcorr_phase=[-0.1, 0.6], yticks_xcorr_phase=[0, 0.3, 0.6],
                    ylim_xcorr_vel=[-0.2, 0.8],  yticks_xcorr_vel=[0,0.4,0.8],
                    ylim_dphase_vel=[-150, 150], yticks_dphase_vel=[-150, 0, 150],
                    ylim_an_mean=[0, 4.1], yticks_an_mean=np.arange(0, 4.1, 2),
                    ylim_an_peak=[0, 5.5], yticks_an_peak=np.arange(0, 5.5, 2.5), fs=10):

    N_group = len(groups)
    ph.set_fontsize(fs)

    if not nophase:
        sizex = 15
        sizey = 17

        plt.figure(1, (sizex, sizey))
        gs = gridspec.GridSpec(16, 24)

        for i_group in range(N_group):

            if type(signal_type) == list:
                i_signal_type = signal_type[i_group]
            else:
                i_signal_type = signal_type

            if type(neuropil) == list:
                i_neuropil = neuropil[i_group]
            else:
                i_neuropil = neuropil

            if type(offset) == list:
                i_offset = offset[i_group]
            else:
                i_offset = offset

            if type(proj2pb) == list:
                i_proj2pb = proj2pb[i_group]
            else:
                i_proj2pb = proj2pb

            if type(ch_num) == list:
                i_ch_num = ch_num[i_group]
            else:
                i_ch_num = ch_num

            recss = groups[i_group]

            ax = plt.subplot(gs[:3, 0:7])
            plot_FFT_of_glomeruli_intensity(recss, ax, filtertype_locomotion=filtertype_locomotion, ch=i_ch_num,
                                            color=colors[i_group], plotcurvesonly=i_group, irecss=i_group, proj2pb=i_proj2pb,
                                            ylim=ylim_FFT, yticks=yticks_FFT, legend=legend_list[i_group], alpha=alpha)

            ax = plt.subplot(gs[:3, 9:13])
            plot_dphase_dhead_lingre(recss, ax, ch=i_ch_num, color=colors[i_group], ylabel_xcoords=-0.2,
                                     plotcurvesonly=i_group, filtertype_locomotion=filtertype_locomotion,
                                     ylim=ylim_dphase_vel, yticks=yticks_dphase_vel, alpha=alpha)

            ax = plt.subplot(gs[5:8, :4])
            plot_phase_subtraction_average_3d([recss, ], ax=ax, neuropils=i_neuropil, offsets=i_offset, normalization='an',
                                              lw_indi=1, lw=4, alpha=alpha, colors=[colors[i_group]], channel=i_ch_num,
                                              zlim=ylim_an_peak, zticks=yticks_an_peak, plot_type='1d_shape',
                                              plotcurvesonly=i_group)

            ax = plt.subplot(gs[5:8, 5:9])
            plot_phase_subtraction_average_3d([recss, ], ax=ax, neuropils=i_neuropil, offsets=i_offset, normalization='an',
                                              lw_indi=1, lw=4, alpha=alpha, colors=[colors[i_group]], channel=i_ch_num,
                                              zlim=ylim_an_peak, zticks=yticks_an_peak, plot_type='1d_shape_norm',
                                              plotcurvesonly=i_group)

            ax = plt.subplot(gs[5:8, 10:13])
            plot_dphase_dhead_xcorr(recss, ax, ch=i_ch_num, color=colors[i_group], plotcurvesonly=i_group,
                                    ylim=ylim_xcorr_phase, yticks=yticks_xcorr_phase, alpha=alpha)

            x00, x01, x10, x11  = 18, 21, 21, 24
            y00, y01, y10, y11, y20, y21 = 0, 3, 3, 6, 6, 9
            axss = [[plt.subplot(gs[y00:y01, x00:x01]), plt.subplot(gs[y00:y01, x10:x11])],
                [plt.subplot(gs[y10:y11, x00:x01]), plt.subplot(gs[y10:y11, x10:x11])],
                [plt.subplot(gs[y20:y21, x00:x01]), plt.subplot(gs[y20:y21, x10:x11])]]

            plot_walk_initiation_imaging_for_gaby_grant(recss, axss=axss, signal_type=i_signal_type,
                                                        directions=['turn', 'forward'], channel_type='abs', show_all=True,
                                                        errorbar=False, ylims=[[-30, 30], [-2.5, 5], ylim_an_mean],
                                                        yticks=[np.arange(-40, 41, 40), np.arange(0, 5.1, 5),
                                                                yticks_an_mean], ylabel_xcoords=-0.3,
                                                        color=colors[i_group], tlim=(-1.8, 1.8), para_type='loose',
                                                        plotcurvesonly=i_group, alpha=alpha)

            ax = plt.subplot(gs[10:13, 11:14])
            plot_intensity_locomotion_xcorr(recss, ax, ch=i_ch_num, behtype='forw', color=colors[i_group],
                                            plotcurvesonly=i_group, ylim=ylim_xcorr_vel, yticks=yticks_xcorr_vel,)

            ax = plt.subplot(gs[10:13, 14:17])
            plot_intensity_locomotion_xcorr(recss, ax, ch=i_ch_num, behtype='head', color=colors[i_group],
                                            plotcurvesonly=i_group, ylim=ylim_xcorr_vel, noaxis=True)

            axs = [plt.subplot(gs[10:13, 18:21]), plt.subplot(gs[10:13, 21:24])]
            plot_intensity_velocity_walking_bouts(recss, axs=axs, walking_bouts=False, bins=30, indi_alpha=0.5,
                                                  lw_indi=1, lw=4, c=colors[i_group], vel_range=[[-4,10], [0,300], [0,3]],
                                                  xticks=[np.arange(-4,8.1,4), np.arange(0,301,150), [0,1]],
                                                  yticks=yticks_an_mean, ylim=ylim_an_mean, loc_num=0, sig_type='mean_an',
                                                  ch_num=i_ch_num, subplot_num=2, plotcurvesonly=i_group)

        # ax = plt.subplot(gs[9:14, :8], projection='3d')
        # plot_phase_subtraction_average_3d(groups, ax=ax, neuropils=neuropil, offsets=offset, normalization='an',
        #                                   colors=colors, channel=i_ch_num,
        #                                   zlim=ylim_an_peak, zticks=yticks_an_peak, add_tuning_proj=True, plot_type='3d',
        #                                   plotcurvesonly=False, kicknan=kicknan)

    else:
        sizex = 8
        sizey = 17

        plt.figure(1, (sizex, sizey))
        gs = gridspec.GridSpec(16, 13)

        if type(signal_type) == list:
            i_signal_type = signal_type[i_group]
        else:
            i_signal_type = signal_type

        if type(neuropil) == list:
            i_neuropil = neuropil[i_group]
        else:
            i_neuropil = neuropil

        if type(offset) == list:
            i_offset = offset[i_group]
        else:
            i_offset = offset

        if type(proj2pb) == list:
            i_proj2pb = proj2pb[i_group]
        else:
            i_proj2pb = proj2pb

        if type(ch_num) == list:
            i_ch_num = ch_num[i_group]
        else:
            i_ch_num = ch_num

        for i_group in range(N_group):
            recss = groups[i_group]

            x00, x01, x10, x11 = 4, 7, 7, 10
            y00, y01, y10, y11, y20, y21 = 0, 3, 3, 6, 6, 9
            axss = [[plt.subplot(gs[y00:y01, x00:x01]), plt.subplot(gs[y00:y01, x10:x11])],
                    [plt.subplot(gs[y10:y11, x00:x01]), plt.subplot(gs[y10:y11, x10:x11])],
                    [plt.subplot(gs[y20:y21, x00:x01]), plt.subplot(gs[y20:y21, x10:x11])]]

            plot_walk_initiation_imaging_for_gaby_grant(recss, axss=axss, signal_type=signal_type,
                                                        directions=['turn', 'forward'], channel_type='abs',
                                                        show_all=True,
                                                        errorbar=False, ylims=[[-30, 30], [-2.5, 5], ylim_an_mean],
                                                        yticks=[np.arange(-40, 41, 40), np.arange(0, 5.1, 5),
                                                                yticks_an_mean], ylabel_xcoords=-0.3,
                                                        color=colors[i_group], tlim=(-2.2, 2.2), para_type='loose',
                                                        plotcurvesonly=i_group, alpha=alpha)

            ax = plt.subplot(gs[10:13, 0:3])
            plot_intensity_locomotion_xcorr(recss, ax, ch=i_ch_num, behtype='forw', color=colors[i_group], alpha=alpha,
                                            plotcurvesonly=i_group, ylim=ylim_xcorr_vel, yticks=yticks_xcorr_vel,)

            ax = plt.subplot(gs[10:13, 3:6])
            plot_intensity_locomotion_xcorr(recss, ax, ch=i_ch_num, behtype='head', color=colors[i_group], alpha=alpha,
                                            plotcurvesonly=i_group, ylim=ylim_xcorr_vel, noaxis=True)

            axs = [plt.subplot(gs[10:13, 7:10]), plt.subplot(gs[10:13, 10:13])]
            plot_intensity_velocity_walking_bouts(recss, axs=axs, walking_bouts=False, bins=30, indi_alpha=alpha,
                                                  lw_indi=1, lw=4, c=colors[i_group],
                                                  vel_range=[[-4, 10], [0, 300], [0, 3]],
                                                  xticks=[np.arange(-4, 8.1, 4), np.arange(0, 301, 150), [0, 1]],
                                                  yticks=yticks_an_mean, ylim=ylim_an_mean, loc_num=0,
                                                  sig_type='mean_an',
                                                  ch_num=i_ch_num, subplot_num=2, plotcurvesonly=i_group)




























