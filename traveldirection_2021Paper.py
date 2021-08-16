import behaviour_2021Paper as bh
import imaging_2021Paper as im
import fbscreening_2021Paper as fbs
import plotting_help_2021Paper as ph
import functions_2021Paper as fc

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
from scipy import stats, ndimage, optimize
from scipy.stats import chisquare

import os, pickle, copy, matplotlib


# help functions
def dump_batch_recss_using_pickel(recss, prefix='37G12_GC7f_TD', overwrite=False):
    recss_short = [[GCaMP_ABF_TD(rec) for rec in recs] for recs in recss]
    directory = 'pickles/'
    filename = prefix + '.pickle'

    if not os.path.exists(directory):
        os.makedirs(directory)

    if not os.path.isfile(directory + filename):
        pickle.dump(recss_short, open(directory + filename, 'wb'))
    else:
        if overwrite:
            pickle.dump(recss_short, open(directory + filename, 'wb'))

def get_batch_recss_using_pickel(prefix, parent_folder='pickles/'):
    return pickle.load(open(parent_folder + prefix + '.pickle', 'rb'))

class GCaMP_TD():

    def __init__(self, rec):
        self.sampling_rate = rec.ims[0].sampling_rate
        self.t, self.an, self.al = rec.ims[0].t, rec.ims[0].c1.an, rec.ims[0].c1.al
        self.phase, self.xedges = rec.ims[0].c1.phase, rec.ims[0].c1.xedges
        self.forw, self.head, self.side = rec.subsample('forw'), rec.subsample('head'), rec.subsample('side')
        self.dforw, self.dhead, self.dside = rec.subsample('dforw'), rec.subsample('dhead'), rec.subsample('dside')
        self.xbar = rec.subsample('xbar')

class ABF_TD():

    def __init__(self, rec):
        self.t = rec.abf.t
        self.forw, self.head, self.side = rec.abf.forw, rec.abf.head, rec.abf.side
        self.dforw, self.dhead, self.dside = rec.abf.dforw, rec.abf.dhead, rec.abf.dside
        self.xbar = rec.abf.xbar

class GCaMP_ABF_TD():

    def __init__(self, rec, puff_thresh=1.5):
        self.recid = rec.abf.recid
        self.tif = GCaMP_TD(rec)
        self.abf = ABF_TD(rec)
        self.tc_puff = rec.abf.t[rec.abf.puff>puff_thresh]

def detect_standing(rec, subsample=True, th_stand_dforw_ub=2, th_stand_dhead_ub=30, th_stand_dside_ub=2,):
    if subsample:
        inds_stand = (np.abs(rec.subsample('dforw')) < th_stand_dforw_ub) & (np.abs(rec.subsample('dhead')) < th_stand_dhead_ub) \
                     & (np.abs(rec.subsample('dside')) < th_stand_dside_ub)
    else:
        inds_stand = (np.abs(rec.abf.dforw) < th_stand_dforw_ub) & (np.abs(rec.abf.dhead) < th_stand_dhead_ub) \
                     & (np.abs(rec.abf.dside) < th_stand_dside_ub)
    return inds_stand

def slope_to_headning_angle(dx, dy):
    # inherited from bar_fixation.slope_to_headning_angle, take np.array inputs now
    # dx is dside, positive means ball rotates to the left, fly walks to her right
    # dy is dforw, positive means fly walks forward
    # deg is also a np.array, when deg is 0, fly walks straight forward
    # when deg is positive, it means the fly is travelling to her left.
    deg = np.degrees(np.arctan(dy / dx))
    deg[dx<0] += 90
    deg[dx>0] -= 90
    return deg

def calculate_td(rec, exclude_standing=True, subsample=True, th_stand_dforw_ub=2, th_stand_dhead_ub=30, th_stand_dside_ub=2,):
    if subsample:
        phase_td = rec.subsample('xbar') + slope_to_headning_angle(rec.subsample('dside'), rec.subsample('dforw'))
    else:
        phase_td = rec.abf.xbar + slope_to_headning_angle(rec.abf.dside, rec.abf.dforw)

    if exclude_standing:
        inds_stand = detect_standing(rec, subsample, th_stand_dforw_ub, th_stand_dhead_ub, th_stand_dside_ub)
        if subsample:
            phase_td[inds_stand] = rec.subsample('xbar')[inds_stand]
        else:
            phase_td[inds_stand] = rec.abf.xbar[inds_stand]
    return fc.wrap(phase_td)

def get_walking_bouts_inds(rec, th_stand_dforw_ub=2, th_stand_dhead_ub=30, th_stand_dside_ub=2, standing_dur=3, thresh_bouts_dur = 10):
    # returns a list of the onset and offset inds for each detected walking bouts
    thresh_forward_ball_rotation = 1 * 360

    walking_bouts_inds = []
    minlen_idx = int(np.ceil(standing_dur * rec.ims[0].sampling_rate))

    inds_stand = detect_standing(rec, th_stand_dforw_ub, th_stand_dhead_ub, th_stand_dside_ub)
    inds = fc.get_contiguous_inds(inds_stand, trim_left=0, keep_ends=True, min_contig_len=minlen_idx)
    t = rec.ims[0].t
    standing_num = len(inds)
    if standing_num:
        for i, ind in enumerate(inds):
            if i != (standing_num-1):
                if (t[inds[i+1][0]] - t[ind[-1]]) >= thresh_bouts_dur:
                    forw = fc.unwrap(rec.subsample('forw')[ind[-1]:inds[i+1][0]])
                    if (np.max(forw) - np.min(forw)) >= thresh_forward_ball_rotation:
                        walking_bouts_inds.append([ind[-1], inds[i+1][0]])
            else:
                if (t[-1] - t[ind[-1]]) >= thresh_bouts_dur:
                    forw = fc.unwrap(rec.subsample('forw')[ind[-1]:])
                    if (np.max(forw) - np.min(forw)) >= thresh_forward_ball_rotation:
                        walking_bouts_inds.append([ind[-1], len(t)-1])
    else:
        walking_bouts_inds.append([0,len(t)-1])
    return walking_bouts_inds

def calculate_phaseoffset(recs, exclude_standing=True, exclude_puff=True, th_stand_dforw_ub=2,
                          th_stand_dhead_ub=30, th_stand_dside_ub=2, win_puff_s=1):
    _phase = []
    _xbar = []
    for rec in recs:
        inds_plot = rec.tif.phase < 1e4
        if exclude_standing:
            inds_stand = detect_standing(rec, th_stand_dforw_ub, th_stand_dhead_ub, th_stand_dside_ub)
            inds_plot = inds_plot & ~inds_stand
        if exclude_puff:
            inds_nopuff = rec.tif.t > -1
            for t_puff in rec.tc_puff:
                inds_nopuff = inds_nopuff & ((rec.tif.t < t_puff) | (rec.tif.t > (t_puff + win_puff_s)))
            inds_plot = inds_plot & inds_nopuff
        _phase.extend(rec.tif.phase[inds_plot])
        _xbar.extend(rec.tif.xbar[inds_plot])
    offset = fc.circmean(np.array(_phase) - np.array(_xbar))
    return offset

def get_flight_stopping_inds(rec, min_len_s=0.5, wing_amp_thresh=-10):
    # inheritated from the function in fbs, modified so that returns stop_inds directly, 20191007
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

    left_inds.extend(right_inds)
    stop_inds = np.unique(np.hstack(np.hstack(np.array(left_inds)))) if len(left_inds) else np.array([])

    return stop_inds

def get_2p_ind_flight(rec, add_right=0, **kwargs):
    # get imaging index of flying fly: inds_flight
    t_beh = rec.abf.t
    len_beh = len(t_beh)
    t_img = rec.ims[0].t

    stop_inds = get_flight_stopping_inds(rec, **kwargs)
    inds_flight = t_img > 0
    if len(stop_inds):
        idxss = fc.get_contiguous_inds(np.gradient(stop_inds) < 1.1, trim_left=0, trim_right=0, keep_ends=True, min_contig_len=1)
        stop_t_start = [t_beh[stop_inds[idxs[0]]] for idxs in idxss]
        stop_t_end = []
        for idxs in idxss:
            stop_idx_beh_ = int(min(stop_inds[idxs[-1]] + add_right, len_beh-1))
            stop_t_end.append(t_beh[stop_idx_beh_])
        for i in range(len(stop_t_start)):
            inds_flight = inds_flight & ((t_img < stop_t_start[i]) | (t_img > stop_t_end[i]))

    return inds_flight

def filter_name_to_signal(rec, filter_name,):
    if filter_name == 'dforw':
        filter_sig = np.abs(rec.subsample('dforw'))
    elif filter_name == 'dhead':
        filter_sig = np.abs(rec.subsample('dhead'))
    elif filter_name == 'xbar':
        filter_sig = np.abs(rec.subsample('xbar'))
    else:
        raise NameError('in traveldirection.py, func get_filter_signal, effective filter_name: %s not found' % filter_name)
    return filter_sig

def get_inds_walking(rec, filters=[['dforw',[1,20]],['dhead',[50,600]]]):
    inds_stand = rec.subsample('dforw') < -1e3
    for filter in filters:
        filter_sig = filter_name_to_signal(rec, filter[0])
        sig_range = filter[-1]
        inds_stand = inds_stand | (filter_sig < sig_range[0])
    return ~inds_stand

def get_stim_visible(rec, filter=['xbar',[0,135]]):
    inds = rec.subsample('xbar') < 1e3
    filter_sig = filter_name_to_signal(rec, filter[0])
    sig_range = filter[-1]
    inds = inds & (filter_sig > sig_range[0]) & (filter_sig < sig_range[-1])
    return inds

def calculate_offsets(rec, sig1, sig2, flight=True):
    if flight:
        inds_flight = td.get_2p_ind_flight(rec)
        sig1 = sig1[inds_flight]
        sig2 = sig2[inds_flight]
    offset = fc.circmean(fc.circsubtract(sig1, sig2))
    return offset

# walking individual trace plots

def plot_indi_trace_opto_unit(rec, axs=[], t0=10, sig_type='phase_subtraction', beh_type=['forw','side','head'],
                              thresholds_behline=[7,7,135],
                              twin=[-4,8],dt=0.1, colors_sig=[ph.dorange, ph.nblue], vlines=[0,4],
                              ylims=[[-180,180],[-10,10],[-10,10],[-180,180]],
                              yticks=[[-180,180],[-10,10],[-10,10],[-180,180]],
                              time_bar_length=2, time_text='2s',
                              ylabels=['PFR-EPG','forw','side','head'], show_axes=['left'], show_flyid=True,
                              label_axeratio=-0.1, fs=10,):
    ph.set_fontsize(fs)
    if not len(axs):
        nrow = len(beh_type) + 1
        _ = plt.figure(1, (3, 1.5*nrow))
        gs = gridspec.GridSpec(nrow, 1)
        axs = []
        for i in range(nrow):
            axs.append(plt.subplot(gs[i, 0]))

    # plot brain signal
    ax = axs[0]
    t_interp = np.arange(twin[0]-dt*2, twin[1]+dt*2, dt)
    if sig_type == 'phase_subtraction':
        inds_sig = (rec.ims[0].t > (twin[0]+t0-dt*10)) & (rec.ims[0].t < (twin[1]+t0+dt*10))
        t_im1, t_im0 = rec.ims[1].t[inds_sig] - t0, rec.ims[0].t[inds_sig] - t0
        phase_im1, phase_im0 = rec.ims[1].c1.phase[inds_sig], rec.ims[0].c1.phase[inds_sig]
        f_im1 = interp1d(t_im1, fc.unwrap(phase_im1), kind='linear')
        f_im0 = interp1d(t_im0, fc.unwrap(phase_im0), kind='linear')
        phase_interp_im1 = fc.wrap(f_im1(t_interp))
        phase_interp_im0 = fc.wrap(f_im0(t_interp))
        dphase = fc.circsubtract(phase_interp_im0, phase_interp_im1)

        ph.plot_noline_between_lims(ax, t_interp, dphase, threshold=135, color='black', lw=3, zorder=3, alpha=1)
        ph.plot_noline_between_lims(ax, t_interp, phase_interp_im1, threshold=135, color=colors_sig[1], lw=2, zorder=2, alpha=0.5)
        ph.plot_noline_between_lims(ax, t_interp, phase_interp_im0, threshold=135, color=colors_sig[0], lw=2, zorder=2, alpha=0.5)
        for vline in vlines:
            ax.axvline(vline, ls='--', lw=2, c=ph.dred, alpha=.6, zorder=1)
        ax.axhline(0, ls='--', lw=2, c=ph.dred, alpha=.6, zorder=1)

    # plot behavior
    inds_beh = (rec.abf.t > (twin[0] + t0 - dt * 10)) & (rec.abf.t < (twin[1] + t0 + dt * 10))
    t_beh = rec.abf.t[inds_beh] - t0
    for ibeh in range(len(beh_type)):
        ax = axs[ibeh+1]
        behsig = getattr(rec.abf, beh_type[ibeh])[inds_beh]
        if beh_type[ibeh] == 'forw' or beh_type[ibeh] == 'side':
            behsig = rec.abf.deg2mm(behsig)
        ph.plot_noline_between_lims(ax, t_beh, behsig, threshold=thresholds_behline[ibeh], color='black', lw=3, zorder=3, alpha=1)
        for vline in vlines:
            ax.axvline(vline, ls='--', lw=2, c=ph.dred, alpha=.6, zorder=1)

    # trim the plot
    for iax, ax in enumerate(axs):
        ph.adjust_spines(ax, show_axes, lw=1, xlim=twin, ylim=ylims[iax], xticks=[], yticks=yticks[iax])

        if iax == 0:
            ph.plot_x_scale_text(ax, bar_length=time_bar_length, text=time_text, y_text_lower=.05, x_text_lefter=.05)

        if 'left' in show_axes:
            ax.set_ylabel(ylabels[iax], rotation='vertical', va='center')
            ax.get_yaxis().set_label_coords(label_axeratio, 0.5)

        if show_flyid and not iax:
            ax.set_title(rec.abf.recid, ha='center', va='top', fontsize=fs)

def plot_indi_trace_opto_20200127(recss, max_ncol=7, **kwargs):
    nrow = np.sum([len(recs) for recs in recss])
    hr = []
    for irow in range(nrow):
        hr.extend([2,1,1,1,0.5])
    plt.figure(1, (3 * max_ncol, 5 * nrow))
    gs = gridspec.GridSpec(nrow*5, max_ncol, height_ratios=hr)

    icount = -1
    for ifly, recs in enumerate(recss):
        for irec, rec in enumerate(recs):
            icount += 1
            idxs = rec.abf.pwm_dc_idx_starts[rec.abf.pwm_dc_stimdcs > 0.05]
            for itrial, idx in enumerate(idxs):
                axs = []
                for i in range(4):
                    axs.append(plt.subplot(gs[i+icount*5, itrial]))
                _show_flyid = True if itrial == 0 else False
                _show_axes = ['left'] if itrial == 0 else []
                plot_indi_trace_opto_unit(rec, axs=axs, t0=rec.abf.t_orig[idx], show_flyid=_show_flyid,
                                          show_axes=_show_axes, **kwargs)

# walking sample trace plots

def plot_sample_trace_walking(rec, tlim=[0, 10], ylims=[[-11, 11], [-180, 180], [-20, 20]], phase_offset_pcolor=False,
                              phase_offset_curve=False,
                              yticks=[[0, 1, 2], [0, 5, 10, 15], [0, 200, 400]], cmap=plt.cm.Blues, color_phase=ph.blue,
                              ybar_length_list = [1,10,180], ybar_text_list = ['1', '20\nmm', '180\ndeg'],
                              time_bar_length=5, time_text='5s', color='black', fs=9, lw=2,
                              colorbar=True, vmin=0, vmax=None, show_flyid=True,
                              hide_ylabels=False, label_axeratio=-.07,):
    ph.set_fontsize(fs)
    _ = plt.figure(1, (10, 10))
    gs = gridspec.GridSpec(50, 50)
    gs_rows = [[20,30],[30,40],[40,50]]

    tif_t = rec.tif.t
    abf_t = rec.abf.t
    tif_inds = np.arange(np.where(tif_t <= tlim[0])[0][-1], np.where(tif_t >= tlim[-1])[0][0])
    abf_inds = np.arange(np.where(abf_t <= tlim[0])[0][-1], np.where(abf_t >= tlim[-1])[0][0])

    # plot tif
    xedges = rec.tif.xedges
    gc = rec.tif.al

    # Setup axis
    ax = plt.subplot(gs[10:20, :48])
    ph.adjust_spines(ax, [], ylim=[xedges[0], xedges[-1]], xlim=tlim)

    # Plot gc
    gc[np.isnan(gc)] = 0
    gc = gc.T
    img = ax.pcolormesh(tif_t[tif_inds], xedges, gc[:,tif_inds], cmap=cmap)
    if not vmin is None: img.set_clim(vmin=vmin)
    if not vmax is None: img.set_clim(vmax=vmax)
    ax.set_ylabel('Glom.', rotation=90)
    ax.yaxis.set_label_coords(label_axeratio, .5)

    period = xedges[-1]
    if not phase_offset_pcolor:
        phase_offset_pcolor = calculate_phaseoffset([rec,])
        print 'pcolor phase offset = %.1f' % phase_offset_pcolor
    phase = (rec.tif.phase - phase_offset_pcolor + 180) * period / 360 + .5
    phase[phase < 0] = phase[phase < 0] + period
    ax.plot(tif_t[tif_inds], phase[tif_inds], alpha=1, zorder=3, ls='none', marker='.', ms=4, mec='black', mfc='black')

    # Draw colorbar
    if colorbar:
        ax = plt.subplot(gs[11:19, 49:])
        gradient = np.linspace(0, 1, 256)
        gradient = np.vstack((gradient, gradient)).T
        ax.imshow(gradient, aspect='auto', cmap=cmap)
        if vmax:
            cmax = vmax
        else:
            idx = (tif_t >= tlim[0]) & (tif_t < tlim[1])
            cmax = np.floor(gc[:,idx].max())
            if cmax == 0:
                cmax = np.floor(gc[:,idx].max() * 10) / 10.
        ax.set_yticklabels(['%1.0f' % vmin, '%1.0f' % cmax])
        ax.yaxis.tick_right()
        ax.tick_params('both', length=2, width=1, which='major')
        ax.set_ylabel('$\Delta$F/F', rotation=0, va='bottom', ha='left')
        ax.yaxis.set_label_position("right")
        ph.adjust_spines(ax, ['right'], pad=0, lw=0, ylim=[0, 255], yticks=[0, 255], xticks=[])
        ax.spines['right'].set_visible(False)


    # plot phase and xbar
    ax = plt.subplot(gs[:10, :48])
    ph.adjust_spines(ax, [], lw=1, ylim=[-180,180], yticks=[], xlim=[tlim[0], tlim[1] + (tlim[1] - tlim[0]) * 0.01],
                     xticks=[], pad=0)
    if not phase_offset_curve:
        phase_offset_curve = calculate_phaseoffset([rec,])
        print 'plot phase offset = %.1f' % phase_offset_curve
    ax.plot(tif_t[tif_inds], fc.wrap(rec.tif.phase[tif_inds]-phase_offset_curve), alpha=1, zorder=3, ls='none',
            marker='.', ms=4, mec=color_phase, mfc=color_phase)
    # ax.plot(abf_t[abf_inds], rec.abf.xbar[abf_inds], c=ph.grey9, lw=lw)
    ph.plot_noline_between_lims(ax, abf_t[abf_inds], rec.abf.xbar[abf_inds], threshold=135, color=ph.grey9, lw=lw)
    ph.plot_x_scale_text(ax, bar_length=time_bar_length, text=time_text, y_text_lower=0, x_text_lefter=.05)
    ax.fill_between(abf_t[abf_inds], y1=-180, y2=-135., color='black', alpha=0.1, zorder=1)
    ax.fill_between(abf_t[abf_inds], y1=135, y2=180, color='black', alpha=0.1, zorder=1)
    ax.set_ylabel('xbar & phase', rotation='vertical', va='center')
    ax.get_yaxis().set_label_coords(label_axeratio, 0.5)


    # plot curves
    ts = [abf_t[abf_inds], abf_t[abf_inds], abf_t[abf_inds]]
    sigs = [fc.deg2mm(rec.abf.forw[abf_inds]), rec.abf.head[abf_inds], fc.deg2mm(rec.abf.side[abf_inds])]
    ylabels = ['Forw', 'Head', 'Side']

    noline_thresholds = [8, 135, 8]
    bar_length_list = ybar_length_list
    bar_text_list = ybar_text_list
    for row in [0,1,2]:
        ax = plt.subplot(gs[gs_rows[row][0]:gs_rows[row][-1], :48])
        ph.adjust_spines(ax, ['bottom', 'left', 'up', 'right'], lw=1, ylim=ylims[row], yticks=yticks[row],
                         xlim=[tlim[0], tlim[1] + (tlim[1] - tlim[0]) * 0.01], xticks=[], pad=0)
        ph.plot_noline_between_lims(ax, ts[row], sigs[row], threshold=noline_thresholds[row], color=color, lw=lw)

        ph.plot_y_scale_text(ax, bar_length=bar_length_list[row], text=bar_text_list[row], bar_pos='bot',
                             x_text_righter=0.05, text2bar_ratio=.02, color='black', rotation='horizontal',
                             ha='left', va='bottom', fs=fs,lw=1)

        if not hide_ylabels:
            ax.set_ylabel(ylabels[row], rotation='vertical', va='center')
            ax.get_yaxis().set_label_coords(label_axeratio, 0.5)

        if row == 2 and show_flyid:
            x_text = tlim[0] + (tlim[-1] - tlim[0]) / 2.
            y_text = ylims[row][0] - (ylims[row][-1] - ylims[row][0]) / 20.
            ax.text(x_text, y_text, '%s' % rec.recid, ha='center', va='top', fontsize=fs)

def plot_indi_trial_walking_unit(rec, axs=[], t0=10, t_win=[0,12,0], ylabels=['PFFR', 'EPG'], vlines=[4,8],
                         colors=[ph.dorange, ph.blue], cmaps=[plt.cm.Oranges, plt.cm.Blues],
                         label_axeratio=-.1, show_label=True, show_recid=True, vm=[0,1.2], fs=11):
    # idx0,1,2,3 corresponds to beginning of plot, beginning of optic flow, end of optic flow, and end of plot, respctl.
    ph.set_fontsize(fs)
    if not len(axs):
        _ = plt.figure(1, (2, 5))
        gs = gridspec.GridSpec(5, 1, height_ratios=[1.5,1.5,1.5,1,1])
        axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[1, 0]), plt.subplot(gs[2, 0]),
               plt.subplot(gs[3, 0]), plt.subplot(gs[4, 0])]

    ssr = rec.ims[0].sampling_rate
    idx_start = np.where(rec.ims[0].t <= t0)[0][-1]
    idx1 = int(idx_start)
    idx0 = int(idx1 - np.ceil(np.abs(t_win[0] * ssr)))
    idx2 = int(idx1 + np.ceil(t_win[1] * ssr))
    idx3 = int(idx2 + np.ceil(t_win[2] * ssr))
    idx0_abf = np.where(rec.abf.t < rec.ims[0].t[idx0])[0][-1]
    idx3_abf = np.where(rec.abf.t > rec.ims[1].t[idx3])[0][0]
    tif_ims = [rec.ims[1].c1.al, rec.ims[0].c1.al]
    xlim = [rec.ims[0].t[idx0], rec.ims[1].t[idx3]]

    for locnum in range(2):
        ax = axs[locnum]
        ph.adjust_spines(ax, [], ylim=[rec.ims[locnum].c1.xedges[0], rec.ims[locnum].c1.xedges[-1]], xlim=xlim)
        ax.pcolormesh(rec.ims[locnum].t[idx0:idx3], rec.ims[locnum].c1.xedges, tif_ims[locnum][idx0:idx3].T, cmap=cmaps[locnum],
                      vmin=vm[0], vmax=vm[1])
        if show_label:
            ax.set_ylabel(ylabels[locnum], rotation='horizontal', va='center', fontsize=fs)
            ax.get_yaxis().set_label_coords(label_axeratio, 0.5)
        if show_recid and locnum == 0:
            ax.set_title(rec.abf.recid, fontsize=fs)

    ax = axs[2]
    ph.adjust_spines(ax, ['left'], lw=1, ylim=[-180,180], yticks=[-180,180], xlim=xlim, xticks=[], pad=0)
    for locnum in range(2):
        ax.plot(rec.ims[locnum].t[idx0:idx3], rec.ims[locnum].c1.phase[idx0:idx3], alpha=1, zorder=3, ls='none',
                marker='.', ms=3, mec=colors[locnum], mfc=colors[locnum])
    ph.plot_x_scale_text(ax, bar_length=2, text='2s', y_text_lower=0.15, x_text_lefter=.05)
    ax.fill_between(xlim, y1=-180, y2=-135., color='black', alpha=0.1, zorder=1)
    ax.fill_between(xlim, y1=135, y2=180, color='black', alpha=0.1, zorder=1)
    if show_label:
        ax.set_ylabel('phases', rotation='horizontal', va='center', fontsize=fs)
        ax.get_yaxis().set_label_coords(label_axeratio, 0.5)

    ylims = [[-15, 15], [-15, 15]]
    yticks = [[-15, 15], [-15, 15]]
    dforw = rec.abf.dforw
    dside = rec.abf.dside
    sigma = 5
    dforw_filted = ndimage.filters.gaussian_filter1d(dforw, sigma)
    dside_filted = ndimage.filters.gaussian_filter1d(dside, sigma)
    behsigs = [dforw_filted, dside_filted]
    ylabels = ['dforw\nmm/s','dside\nmm/s']
    for ibeh in range(2):
        ax = axs[ibeh+3]
        ph.adjust_spines(ax, ['left'], lw=1, ylim=ylims[ibeh], yticks=yticks[ibeh], xlim=xlim, xticks=[], pad=0)
        ax.plot(rec.abf.t[idx0_abf:idx3_abf], behsigs[ibeh][idx0_abf:idx3_abf], color='black', lw=1, zorder=2, alpha=1)
        ax.axhline(0, ls='--', lw=1, c=ph.grey9, alpha=.5, zorder=1)
        if show_label:
            ax.set_ylabel(ylabels[ibeh], rotation='horizontal', va='center', fontsize=fs)
            ax.get_yaxis().set_label_coords(label_axeratio, 0.5)

    for ax in axs:
        for vline in vlines:
            ax.axvline(rec.ims[0].t[idx1]+vline, ls='--', lw=1, c=ph.grey9, alpha=.5, zorder=1)

def plot_indi_trial_walking_paper_20200320(rec, axs=[], t0=10, t_win=[0,12,0], vlines=[0],
            colors=[ph.blue, ph.dorange], cmaps=[plt.cm.Blues, plt.cm.Oranges], label_axeratio=-.2, show_label=True,
            ylims=[[-20,15],[-20,20]], yticks=[[-15,0,15],[-20,0,20]],
            vm=[0,1], fs=11, show_axes=['left'], ms=1.5, lw_dash=.5, lw=.7):
    # idx0,1,2,3 corresponds to beginning of plot, beginning of optic flow, end of optic flow, and end of plot, respctl.
    ph.set_fontsize(fs)
    if not len(axs):
        _ = plt.figure(1, (3, 3))
        gs = gridspec.GridSpec(3, 3, width_ratios=[5,1,5])
        axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[1, 0]), plt.subplot(gs[2, 0]),
               plt.subplot(gs[0, 2]), plt.subplot(gs[1, 2]), plt.subplot(gs[2, 2])]

    ssr = rec.ims[0].sampling_rate
    idx_start = np.where(rec.ims[0].t <= t0)[0][-1]
    idx1 = int(idx_start)
    idx0 = int(idx1 - np.ceil(np.abs(t_win[0] * ssr)))
    idx2 = int(idx1 + np.ceil(t_win[1] * ssr))
    idx3 = int(idx2 + np.ceil(t_win[2] * ssr))
    idx0_abf = np.where(rec.abf.t < rec.ims[0].t[idx0])[0][-1]
    idx3_abf = np.where(rec.abf.t > rec.ims[1].t[idx3])[0][0]
    tif_ims = [rec.ims[1].c1.al, rec.ims[0].c1.al]
    xlim = [rec.ims[0].t[idx0], rec.ims[1].t[idx3]]

    # pcolormesh EPG and PFR signal
    for i in range(2):
        ax = axs[i]
        locnum = 1 - i
        ph.adjust_spines(ax, [], ylim=[rec.ims[locnum].c1.xedges[0], rec.ims[locnum].c1.xedges[-1]], xlim=xlim)
        ax.pcolormesh(rec.ims[locnum].t[idx0:idx3], rec.ims[locnum].c1.xedges, rec.ims[locnum].c1.al[idx0:idx3].T, cmap=cmaps[i],
                      vmin=vm[0], vmax=vm[1])

    # phase plot
    ax = axs[2]
    ph.adjust_spines(ax, [], lw=.5, ylim=[-180,180], yticks=[-180,180], xlim=xlim, xticks=[], pad=0)
    for i in range(2):
        locnum = 1 - i
        ax.plot(rec.ims[locnum].t[idx0:idx3], rec.ims[locnum].c1.phase[idx0:idx3], alpha=1, zorder=3, ls='none',
                marker='.', ms=ms, mec=colors[i], mfc=colors[i])
    ax.fill_between(xlim, y1=-180, y2=-135., facecolor='black', edgecolor='none', alpha=0.1, zorder=1)
    ax.fill_between(xlim, y1=135, y2=180, facecolor='black', edgecolor='none', alpha=0.1, zorder=1)

    # Delta phase plot
    ax = axs[5]
    dt_sig = 0.1
    t_interp_sig = np.arange(t_win[0] - dt_sig * 2, t_win[1] + t_win[2] + dt_sig * 2, dt_sig)
    inds_sig = (rec.ims[0].t > (t_win[0] + t0 - dt_sig * 10)) & (
                rec.ims[0].t < (t_win[1] + t_win[2] + t0 + dt_sig * 10))
    t_im1, t_im0 = rec.ims[1].t[inds_sig] - t0, rec.ims[0].t[inds_sig] - t0
    phase_im1, phase_im0 = rec.ims[1].c1.phase[inds_sig], rec.ims[0].c1.phase[inds_sig]
    f_im1 = interp1d(t_im1, fc.unwrap(phase_im1), kind='linear')
    f_im0 = interp1d(t_im0, fc.unwrap(phase_im0), kind='linear')
    phase_interp_im1 = fc.wrap(f_im1(t_interp_sig))
    phase_interp_im0 = fc.wrap(f_im0(t_interp_sig))
    dphase = fc.circsubtract(phase_interp_im1, phase_interp_im0)
    ph.adjust_spines(ax, [], lw=.5, xlim=xlim, ylim=[-180,180], xticks=[], yticks=[])
    ax.plot(t_interp_sig+t0, dphase, ls='none', marker='o', ms=ms, mec='none', mfc='black', zorder=2, alpha=1)
    ax.axhline(0, ls='--', lw=lw_dash, c=ph.grey9, alpha=.5, zorder=1)

    # Behavior plot
    dforw = rec.abf.dforw
    dside = rec.abf.dside
    sigma = 5
    dforw_filted = ndimage.filters.gaussian_filter1d(dforw, sigma)
    dside_filted = ndimage.filters.gaussian_filter1d(dside, sigma)
    behsigs = [dforw_filted, dside_filted]
    ylabels = ['dforw\n(mm/s)','dside\n(mm/s)']
    for ibeh in range(2):
        ax = axs[ibeh+3]
        ph.adjust_spines(ax, show_axes, lw=.5, ylim=ylims[ibeh], yticks=yticks[ibeh], xlim=xlim, xticks=[], pad=0)
        ax.plot(rec.abf.t[idx0_abf:idx3_abf], behsigs[ibeh][idx0_abf:idx3_abf], color='black', lw=lw, zorder=2, alpha=1)
        ax.axhline(0, ls='--', lw=lw_dash, c=ph.grey9, alpha=.5, zorder=1)
        if show_label:
            ax.set_ylabel(ylabels[ibeh], rotation='horizontal', va='center', fontsize=fs)
            ax.get_yaxis().set_label_coords(label_axeratio, 0.5)
        if ibeh == 0:
            ph.plot_x_scale_text(ax, bar_length=2, text='2s', y_text_lower=0.15, x_text_lefter=.05)

    for ax in axs:
        for vline in vlines:
            ax.axvline(rec.ims[0].t[idx1]+vline, ls='--', lw=lw_dash, c=ph.grey9, alpha=.5, zorder=1)

def plot_bww_sample_traj_unit(rec, fig_size=(.8,1.5), idx_bww_s=1000, idx_bww_e=1200, t_pre=2, t_post=2, filter_window_s=0.05,
            head_colors=[ph.blue, ph.red, ph.blue], arrow_itvs=[5,4,10], ms=2, arrow_l=.7, tail_w=1, head_w=.1, head_alpha=0.7, scalebar=True, bar_length=1, fs=8):
    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])
    ph.set_fontsize(fs)

    dt = np.diff(rec.abf.tc).mean()
    sigma = filter_window_s / dt
    x = ndimage.filters.gaussian_filter1d(rec.abf.x, sigma)
    y = ndimage.filters.gaussian_filter1d(rec.abf.y, sigma)
    xbar = np.radians(rec.abf.xbar + 90)

    n_idx_pre = int(t_pre * rec.abf.subsampling_rate)
    n_idx_post = int(t_post * rec.abf.subsampling_rate)
    idx0 = idx_bww_s - n_idx_pre
    idx3 = idx_bww_e + n_idx_post
    idx1, idx2 = idx_bww_s, idx_bww_e

    # plot traj and arrows
    idxss = [[idx0, idx1],[idx1, idx2],[idx2, idx3]]
    for i in range(3):
        # ax.plot(x[idxss[i][0]:idxss[i][1]], y[idxss[i][0]:idxss[i][1]], ls='none', marker='.', ms=ms, mec='none', mfc=ph.grey8, alpha=0.7, zorder=2)
        ax.plot(x[idxss[i][0]:idxss[i][1]], y[idxss[i][0]:idxss[i][1]], lw=.5, color=ph.grey8, alpha=0.7, zorder=2)


        idxs_arrow = np.arange(idxss[i][0], idxss[i][1], arrow_itvs[i]).astype(int)
        n_step = len(idxs_arrow)
        if i == 0 or i == 2:
            zorder_ = 2
            for idx_ in idxs_arrow:
                ax.arrow(x[idx_], y[idx_], np.cos(xbar[idx_])*arrow_l, np.sin(xbar[idx_])*arrow_l, color=head_colors[i],
                         width=tail_w, length_includes_head=True, head_width=head_w, alpha=head_alpha, zorder=zorder_)
        # if i == 1:
        #     cm = matplotlib.cm.get_cmap('Reds')
        #     zorder_ = 2
        #     for i_, idx_ in enumerate(idxs_arrow[::-1]):
        #         color = cm(.3 + .7 * i_ / n_step)
        #         ax.arrow(x[idx_], y[idx_], np.cos(xbar[idx_])*arrow_l, np.sin(xbar[idx_])*arrow_l, color=color,
        #                  length_includes_head=True, head_width=head_w, alpha=head_alpha, zorder=zorder_)
        if i == 1:
            zorder_ = 3
            for idx_ in idxs_arrow:
                ax.arrow(x[idx_], y[idx_], np.cos(xbar[idx_])*arrow_l, np.sin(xbar[idx_])*arrow_l, color=head_colors[i],
                         width=tail_w, length_includes_head=True, head_width=head_w, alpha=head_alpha, zorder=zorder_)

    # trim plot
    ax.set_aspect('equal', 'datalim')
    x0, x1, y0, y1 = np.min(x[idx0:idx3]), np.max(x[idx0:idx3]), np.min(y[idx0:idx3]), np.max(y[idx0:idx3])
    xl, yl = x1 - x0, y1 - y0
    xlim = [x0 - xl * .15, x1 + xl * .15]
    ylim = [y0 - yl * .1, y1 + yl * .1]
    ph.adjust_spines(ax, [], lw=.25, xticks='none', yticks='none', xlim=xlim, ylim=ylim)
    if scalebar:
        # ph.plot_xy_scalebar(ax, bar_length=bar_length, fs=fs, unit='mm', scale=1, mode='top_center_to_right')
        ph.plot_x_scale_text(ax, bar_length=bar_length, text='%imm' % bar_length, y_text_lower=0.35, x_text_lefter=.05)

def plot_2c_sample_trace_td_walk_paper_20200409(rec, axs=[], fig_size=(15, 3), tlim=[1,200], gs_width_ratios=[50,1], plot_bar=True,
                cmaps=[plt.cm.Blues, plt.cm.Oranges], vm=[0,1.6], colors=[ph.blue, ph.dorange], bar_length=10, fs=7, auto_offset=True,
                ms=1.5, ms_beh=1, lw=.3, vlines=[], n_movwin=7):
    ph.set_fontsize(fs)
    if not len(axs):
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(6, 2, width_ratios=gs_width_ratios, height_ratios=[1,1,1,1,.7,.7])
        axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]),
               plt.subplot(gs[2, 0]), plt.subplot(gs[3, 0]), plt.subplot(gs[4, 0]), plt.subplot(gs[5, 0])]

    idx0 = np.where(rec.fb.t >= tlim[0])[0][0]
    idx1 = np.where(rec.eb.t <= tlim[1])[0][-1]
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

    # plot phases
    sigs_ss = [rec.subsample('xbar'), calculate_td(rec, exclude_standing=False, subsample=True)]
    sigs_beh = [rec.abf.xbar, calculate_td(rec, exclude_standing=False, subsample=False)]
    phase_eb = rec.eb.c1.phase[idx0:idx1]
    for i_dir in range(2):
        ax = axs[i_dir+4]
        ph.adjust_spines(ax, [], lw=1, ylim=[-180, 180], yticks=[], xlim=xlim, xticks=[], pad=0)
        offset = calculate_offsets(rec, phase_eb, sigs_ss[0][idx0:idx1], flight=False) if auto_offset else 0
        if plot_bar:
            ax.plot(rec.abf.t[idx0_beh:idx1_beh], sigs_beh[i_dir][idx0_beh:idx1_beh], alpha=.5, zorder=1, ls='none', marker='o', ms=ms_beh, mec='none', mfc='black')
        for i_ in range(2):
            ax.plot(rec.ims[1-i_].t[idx0:idx1], fc.wrap(rec.ims[1-i_].c1.phase[idx0:idx1]-offset), alpha=1, zorder=2, ls='none', marker='o', ms=ms, mec='none', mfc=colors[i_])
        if i_dir == 0:
            ph.plot_x_scale_text(ax, bar_length=bar_length, text='%is' % bar_length, y_text_lower=0.85, x_text_lefter=.05, fontsize=fs)
        ax.fill_between(xlim, y1=-180, y2=-135., facecolor='black', edgecolor='none', alpha=0.1, zorder=1)
        ax.fill_between(xlim, y1=135, y2=180, facecolor='black', edgecolor='none', alpha=0.1, zorder=1)
        for vline in vlines:
            ax.axvline(vline, ls='--', lw=1, c=ph.dred, alpha=.6, zorder=1)

    # plot beh
    vels = [np.convolve(rec.abf.dforw, np.ones((n_movwin,)) / n_movwin, mode='same'),
            np.convolve(rec.abf.dside, np.ones((n_movwin,)) / n_movwin, mode='same')]
    for i_dir in range(2):
        ax = axs[i_dir+6]
        ph.adjust_spines(ax, [], lw=1, ylim=[-15, 15], yticks=[], xlim=xlim, xticks=[], pad=0)
        ax.plot(rec.abf.t[idx0_beh:idx1_beh], vels[i_dir][idx0_beh:idx1_beh], alpha=1, zorder=1, c='black', lw=lw)
        ax.axhline(0, ls='--', lw=1, c=ph.grey9, alpha=.6, zorder=1)
        ph.plot_y_scale_text(ax, bar_length=10, text=10, x_text_right=-.02,
                             color='black', rotation='horizontal', ha='right', va='center', fontsize=fs)


# flight sample trace plots, 2019 10 01

def plot_indi_trial_unit(rec, axs=[], idx_start=0, t_win=[0,12,0], ylabels=['PFFR', 'EPG'], vlines=[4,8],
                         colors=[ph.dorange, ph.blue], cmaps=[plt.cm.Oranges, plt.cm.Blues], vm=[], ms=2,
                         label_axeratio=-.1, show_label=True, show_recid=True, show_xscale=True, lw_dash=1, fs=11):
    # idx0,1,2,3 corresponds to beginning of plot, beginning of optic flow, end of optic flow, and end of plot, respctl.
    if not len(axs):
        _ = plt.figure(1, (3, 5))
        gs = gridspec.GridSpec(3, 1)
        axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[1, 0]), plt.subplot(gs[2, 0])]

    ssr = rec.ims[0].sampling_rate
    idx1 = int(idx_start)
    idx0 = int(idx1 + np.ceil(t_win[0] * ssr))
    idx2 = int(idx1 + np.ceil(t_win[1] * ssr))
    idx3 = int(idx2 + np.ceil(t_win[2] * ssr))
    idx0_abf = np.where(rec.abf.t < rec.ims[0].t[idx0])[0][-1]
    idx3_abf = np.where(rec.abf.t > rec.ims[1].t[idx3])[0][0]
    # tif_ims = [rec.ims[1].c1.al, rec.ims[0].c1.al]
    tif_ims = [rec.ims[0].c1.al, rec.ims[1].c1.al]
    xlim = [rec.ims[0].t[idx0], rec.ims[1].t[idx3]]

    for locnum in range(2):
        ax = axs[locnum]
        ph.adjust_spines(ax, [], ylim=[rec.ims[locnum].c1.xedges[0], rec.ims[locnum].c1.xedges[-1]], xlim=xlim)
        if len(vm)==2:
            ax.pcolormesh(rec.ims[locnum].t[idx0:idx3], rec.ims[locnum].c1.xedges, tif_ims[locnum][idx0:idx3].T, cmap=cmaps[locnum], vmin=vm[0], vmax=vm[1])
        else:
            ax.pcolormesh(rec.ims[locnum].t[idx0:idx3], rec.ims[locnum].c1.xedges, tif_ims[locnum][idx0:idx3].T,
                          cmap=cmaps[locnum])
        if show_label:
            ax.set_ylabel(ylabels[locnum], rotation='horizontal', va='center', fontsize=fs)
            ax.get_yaxis().set_label_coords(label_axeratio, 0.5)
        if show_recid:
            ax.set_title(rec.abf.recid, fontsize=fs)

    ax = axs[2]
    ph.adjust_spines(ax, [], lw=1, ylim=[-180, 180], yticks=[], xlim=xlim, xticks=[], pad=0)
    for locnum in range(2):
        ax.plot(rec.ims[locnum].t[idx0:idx3], rec.ims[locnum].c1.phase[idx0:idx3], alpha=1, zorder=3, ls='none',
                marker='.', ms=ms, mec='none', mfc=colors[locnum])

    # ph.plot_noline_between_lims(ax, rec.abf.t[idx0_abf:idx3_abf], rec.abf.xbar[idx0_abf:idx3_abf], threshold=135,
    #                             color=ph.grey9, lw=1)
    if show_xscale:
        ph.plot_x_scale_text(ax, bar_length=2, text='2s', y_text_lower=0.15, x_text_lefter=.05)
    ax.fill_between(xlim, y1=-180, y2=-135., facecolor='black', alpha=0.1, zorder=1, edgecolor='none')
    ax.fill_between(xlim, y1=135, y2=180, facecolor='black', alpha=0.1, zorder=1, edgecolor='none')
    if show_label:
        ax.set_ylabel('sun & phases', rotation='vertical', va='center', fontsize=fs)
        ax.get_yaxis().set_label_coords(label_axeratio, 0.5)

    for ax in axs:
        for vline in vlines:
            ax.axvline(rec.ims[0].t[idx1]+vline, ls='--', lw=lw_dash, c=ph.grey9, alpha=.5)

def plot_indi_trial_2ch_unit(rec, axs=[], idx_start=0, t_win=[0,12,0], ylabels=['PFFR', 'EPG'], vlines=[4,8],
                         locnums=[0,1], chnums=[1,1], colors=[ph.dorange, ph.blue], cmaps=[plt.cm.Oranges, plt.cm.Blues], vm=[], ms=2,
                         label_axeratio=-.1, show_label=True, show_recid=True, show_xscale=True, lw_dash=1, fs=11):
    # idx0,1,2,3 corresponds to beginning of plot, beginning of optic flow, end of optic flow, and end of plot, respctl.
    if not len(axs):
        _ = plt.figure(1, (3, 5))
        gs = gridspec.GridSpec(3, 1)
        axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[1, 0]), plt.subplot(gs[2, 0])]

    ssr = rec.ims[0].sampling_rate
    idx1 = int(idx_start)
    idx0 = int(idx1 + np.ceil(t_win[0] * ssr))
    idx2 = int(idx1 + np.ceil(t_win[1] * ssr))
    idx3 = int(idx2 + np.ceil(t_win[2] * ssr))
    idx0_abf = np.where(rec.abf.t < rec.ims[locnums[0]].t[idx0])[0][-1]
    idx3_abf = np.where(rec.abf.t > rec.ims[locnums[1]].t[idx3])[0][0]
    # tif_ims = [rec.ims[1].c1.al, rec.ims[0].c1.al]
    chs = [getattr(rec.ims[locnums[0]], 'c%i' % chnums[0]), getattr(rec.ims[locnums[1]], 'c%i' % chnums[1])]
    tif_ims = [chs[0].al, chs[1].al]
    xlim = [rec.ims[locnums[0]].t[idx0], rec.ims[locnums[1]].t[idx3]]

    for iax in range(2):
        ax = axs[iax]
        ph.adjust_spines(ax, [], ylim=[chs[0].xedges[0], chs[1].xedges[-1]], xlim=xlim)
        if len(vm)==2:
            ax.pcolormesh(rec.ims[locnums[iax]].t[idx0:idx3], chs[iax].xedges, tif_ims[iax][idx0:idx3].T, cmap=cmaps[iax], vmin=vm[0], vmax=vm[1])
        else:
            ax.pcolormesh(rec.ims[locnums[iax]].t[idx0:idx3], chs[iax].xedges, tif_ims[iax][idx0:idx3].T, cmap=cmaps[iax])
        if show_label:
            ax.set_ylabel(ylabels[iax], rotation='horizontal', va='center', fontsize=fs)
            ax.get_yaxis().set_label_coords(label_axeratio, 0.5)
        if show_recid:
            ax.set_title(rec.abf.recid, fontsize=fs)

    ax = axs[2]
    ph.adjust_spines(ax, [], lw=1, ylim=[-180, 180], yticks=[], xlim=xlim, xticks=[], pad=0)
    for i in range(2):
        ax.plot(rec.ims[locnums[i]].t[idx0:idx3], chs[i].phase[idx0:idx3], alpha=1, zorder=3, ls='none',
                marker='.', ms=ms, mec='none', mfc=colors[i])

    # ph.plot_noline_between_lims(ax, rec.abf.t[idx0_abf:idx3_abf], rec.abf.xbar[idx0_abf:idx3_abf], threshold=135,
    #                             color=ph.grey9, lw=1)
    if show_xscale:
        ph.plot_x_scale_text(ax, bar_length=2, text='2s', y_text_lower=0.15, x_text_lefter=.05)
    ax.fill_between(xlim, y1=-180, y2=-135., facecolor='black', alpha=0.1, zorder=1, edgecolor='none')
    ax.fill_between(xlim, y1=135, y2=180, facecolor='black', alpha=0.1, zorder=1, edgecolor='none')
    if show_label:
        ax.set_ylabel('sun & phases', rotation='vertical', va='center', fontsize=fs)
        ax.get_yaxis().set_label_coords(label_axeratio, 0.5)

    for ax in axs:
        for vline in vlines:
            ax.axvline(rec.ims[locnums[i]].t[idx1]+vline, ls='--', lw=lw_dash, c=ph.grey9, alpha=.5)


def plot_indi_trial_unit_1c(rec, axs=[], idx_start=0, t_win_plot=[-1,4,1], ylabel='hDeltaK', vlines=[0,4],
                         color=ph.magenta, cmap=ph.Magentas, vm=[], ms=4, ylim=[-.2,2], lw_dash=1, fs=11, show_axes=['left'],
                         label_axeratio=-.1, show_label=True, show_recid=True, show_xscale=True, ):
    # idx0,1,2,3 corresponds to beginning of plot, beginning of optic flow, end of optic flow, and end of plot, respctl.
    if not len(axs):
        _ = plt.figure(1, (3, 3))
        gs = gridspec.GridSpec(2, 1)
        axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[1, 0])]

    ssr = rec.ims[0].sampling_rate
    idx1 = int(idx_start)
    idx0 = int(idx1 + np.ceil(t_win_plot[0] * ssr))
    idx2 = int(idx1 + np.ceil(t_win_plot[1] * ssr))
    idx3 = int(idx2 + np.ceil(t_win_plot[2] * ssr))
    xlim = [rec.ims[0].t[idx0], rec.ims[0].t[idx3]]

    ax = axs[0]
    ph.adjust_spines(ax, [], ylim=[rec.ims[0].c1.xedges[0], rec.ims[0].c1.xedges[-1]], xlim=xlim)
    if len(vm)==2:
        ax.pcolormesh(rec.ims[0].t[idx0:idx3], rec.ims[0].c1.xedges, rec.ims[0].c1.al[idx0:idx3].T, cmap=cmap, vmin=vm[0], vmax=vm[1])
    else:
        ax.pcolormesh(rec.ims[0].t[idx0:idx3], rec.ims[0].c1.xedges, rec.ims[0].c1.al[idx0:idx3].T, cmap=cmap)
    if show_label:
        ax.set_ylabel(ylabel, rotation='vertical', va='center', fontsize=fs)
        ax.get_yaxis().set_label_coords(label_axeratio, 0.5)
    if show_recid:
        ax.set_title(rec.abf.recid, fontsize=fs)

    ax = axs[1]
    ph.adjust_spines(ax, show_axes, lw=1, ylim=ylim, yticks=ylim, xlim=xlim, xticks=[], pad=0)
    ax.plot(rec.ims[0].t[idx0:idx3], rec.ims[0].c1.mean_an[idx0:idx3], alpha=1, zorder=3, ls='none',
            marker='.', ms=ms, mec='none', mfc=color)
    if show_xscale:
        ph.plot_x_scale_text(ax, bar_length=2, text='2s', y_text_lower=0.15, x_text_lefter=.05)

    if show_label:
        ax.set_ylabel('dF/F0', rotation='vertical', va='center', fontsize=fs)
        ax.get_yaxis().set_label_coords(label_axeratio, 0.5)

    for ax in axs:
        for vline in vlines:
            ax.axvline(rec.ims[0].t[idx1]+vline, ls='--', lw=lw_dash, c=ph.grey9, alpha=.5)

def get_stim_info_20191007(rec, stimid_N=10):
    thresh_dystim_stimon_lb = 0.02
    min_contig_len_stimon_s = 1

    min_contig_len_stimon_nframe = np.ceil(min_contig_len_stimon_s * rec.ims[0].sampling_rate)
    of_stimids, of_start_idxs, of_end_idxs = [], [], []

    ystim = rec.subsample('ystim')
    dystim = np.gradient(ystim)
    stimid = rec.subsample('stimid')
    idxss_stimon = fc.get_contiguous_inds(np.abs(dystim) > thresh_dystim_stimon_lb, trim_left=0, keep_ends=True,
                                          min_contig_len=min_contig_len_stimon_nframe)
    for _idxs in idxss_stimon:
        of_start_idxs.append(_idxs[0])
        of_end_idxs.append(_idxs[-1])
        # of_stimids.append(np.round(np.mean(stimid[_idxs])))
        of_stimids.append(np.round(np.mean(stimid[_idxs])*stimid_N)/1./stimid_N)
    return np.array(of_stimids), np.array(of_start_idxs), np.array(of_end_idxs)

def get_stim_info_20200117_pwmdc(rec):
    min_contig_len_stimon_s = 1

    min_contig_len_stimon_nframe = np.ceil(min_contig_len_stimon_s * rec.ims[0].sampling_rate)
    of_stimids, of_start_idxs, of_end_idxs = [], [], []

    pwmdc = rec.subsample('pwm_dc_subsampled')
    idxss_stimon = fc.get_contiguous_inds(pwmdc > 0.01, trim_left=0, keep_ends=True,
                                          min_contig_len=min_contig_len_stimon_nframe)
    for _idxs in idxss_stimon:
        of_start_idxs.append(_idxs[0])
        of_end_idxs.append(_idxs[-1])
        of_stimids.append(np.round(np.mean(pwmdc[_idxs])*20)/2.)

    return np.array(of_stimids), np.array(of_start_idxs), np.array(of_end_idxs)

def get_stim_info_20200224_xposonly(rec):
    thresh_dystim_stimon_lb = 0.02
    min_contig_len_stimon_s = 1

    min_contig_len_stimon_nframe = np.ceil(min_contig_len_stimon_s * rec.ims[0].sampling_rate)
    of_stimids, of_start_idxs, of_end_idxs = [], [], []

    xstim = rec.subsample('xstim')
    dxstim = np.gradient(xstim)
    stimid = rec.subsample('stimid')
    idxss_stimon = fc.get_contiguous_inds(np.abs(dxstim) > thresh_dystim_stimon_lb, trim_left=0, keep_ends=True,
                                          min_contig_len=min_contig_len_stimon_nframe)
    for _idxs in idxss_stimon:
        of_start_idxs.append(_idxs[0])
        of_end_idxs.append(_idxs[-1])
        of_stimids.append(np.round(np.mean(stimid[_idxs])))

    return np.array(of_stimids), np.array(of_start_idxs), np.array(of_end_idxs)

def get_stim_info_20210408_stimidonly(rec):
    min_contig_len_stimon_s = 1
    min_contig_len_stimon_nframe = np.ceil(min_contig_len_stimon_s * rec.ims[0].sampling_rate)
    of_stimids, of_start_idxs, of_end_idxs = [], [], []

    stimid = rec.subsample('stimid')
    dstimid = np.gradient(stimid)
    idxss_stimon = fc.get_contiguous_inds(np.abs(dstimid) < 0.1, trim_left=0, keep_ends=True,
                                          min_contig_len=min_contig_len_stimon_nframe)
    for _idxs in idxss_stimon:
        of_start_idxs.append(_idxs[0])
        of_end_idxs.append(_idxs[-1])
        of_stimids.append(np.round(np.mean(stimid[_idxs])))

    return np.array(of_stimids), np.array(of_start_idxs), np.array(of_end_idxs)

def plot_indi_trial_flight_20191007(recss, stimid_special=1, t_win_plot=[0,12,0], t_win_stop=[-15,10],
                                    stimtype='get_stim_info_20191007',
                                    ylabels=['PFFR', 'EPG'], vlines=[4,8], colors=[ph.dorange, ph.blue],
                                    cmaps=[plt.cm.Blues, plt.cm.Oranges], label_axeratio=-.1, fs=11):
    nrec = np.sum([len(recs) for recs in recss])
    ncol = 7
    _ = plt.figure(1, (3 * ncol, 5 * nrec))
    gs = gridspec.GridSpec(3 * nrec, 1 * ncol)

    irec = -1
    for recs in recss:
        for rec in recs:
            irec += 1
            if stimtype == 'get_stim_info_20191007':
                of_stimids, of_start_idxs, _ = get_stim_info_20191007(rec)
            elif stimtype == 'get_stim_info_20200117_pwmdc':
                of_stimids, of_start_idxs, _ = get_stim_info_20200117_pwmdc(rec)
            stop_inds = get_flight_stopping_inds(rec)
            ssr_abf = rec.abf.subsampling_rate

            col = -1
            for start_idx in of_start_idxs[of_stimids==stimid_special]:
                start_idx_abf = np.where(rec.abf.t < rec.fb.t[start_idx])[0][-1]
                idx_stopwindow = [start_idx_abf - np.ceil(np.abs(t_win_stop[0])*ssr_abf),
                                  start_idx_abf + np.ceil(np.abs(t_win_stop[-1])*ssr_abf)]
                if not np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
                    col += 1
                    axs = [plt.subplot(gs[irec*3, col]), plt.subplot(gs[irec*3+1, col]), plt.subplot(gs[irec*3+2, col])]
                    _show_recid = True if col == 0 else False
                    _show_label = True if col == 0 else False
                    plot_indi_trial_unit(rec, axs=axs, idx_start=start_idx, t_win=t_win_plot, ylabels=ylabels,
                         colors=colors, cmaps=cmaps, label_axeratio=label_axeratio, show_label=_show_label,
                                         show_recid=_show_recid, fs=fs, vlines=vlines)

def plot_2c_sample_trace_paper_20200320(rec, axs=[], fig_size=(15, 3), tlim=[1,200], gs_width_ratios=[50,1], plot_bar=True,
                cmaps=[plt.cm.Blues, plt.cm.Oranges], vm=[0,1.6], colors=[ph.blue, ph.dorange], bar_length=10, fs=7, auto_offset=True,
                ms=1.5, ms_dot=.5, c_dot=ph.grey8, vlines=[], ):
    ph.set_fontsize(fs)
    if not len(axs):
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(3, 2, width_ratios=gs_width_ratios)
        axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[2, 0])]

    idx0 = np.where(rec.fb.t >= tlim[0])[0][0]
    idx1 = np.where(rec.eb.t <= tlim[1])[0][-1]
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
    xbar_ss = rec.subsample('xbar')[idx0:idx1]
    phase_eb = rec.eb.c1.phase[idx0:idx1]
    if auto_offset:
        offset = calculate_offsets(rec, phase_eb, xbar_ss, flight=False)
    else:
        offset = 0
    if plot_bar:
        ax.plot(rec.abf.t[idx0_beh:idx1_beh], rec.abf.xbar[idx0_beh:idx1_beh], alpha=1, zorder=1, ls='none', marker='o',
                ms=ms_dot, mec='none', mfc=c_dot)
    for i_ in range(2):
        ax.plot(rec.ims[1-i_].t[idx0:idx1], fc.wrap(rec.ims[1-i_].c1.phase[idx0:idx1]-offset), alpha=1, zorder=2, ls='none', marker='o', ms=ms, mec='none', mfc=colors[i_])

    ph.plot_x_scale_text(ax, bar_length=bar_length, text='%is' % bar_length, y_text_lower=0.85, x_text_lefter=.05, fontsize=fs)
    ax.fill_between(xlim, y1=-180, y2=-135., facecolor='black', edgecolor='none', alpha=0.1, zorder=1)
    ax.fill_between(xlim, y1=135, y2=180, facecolor='black', edgecolor='none', alpha=0.1, zorder=1)

    for vline in vlines:
        ax.axvline(vline, ls='--', lw=2, c=ph.dred, alpha=.6, zorder=1)

def plot_2c_sample_trace_paper_raw_20210704(rec, axs=[], fig_size=(15, 3), tlim=[1,200], gs_width_ratios=[50,1], plot_bar=True,
                cmaps=[plt.cm.Blues, plt.cm.Oranges], vms=[[0,300],[0,1000]], colors=[ph.blue, ph.dorange], bar_length=10, fs=7, auto_offset=True,
                ms=1.5, ms_dot=.5, c_dot=ph.grey8, vlines=[], ):
    ph.set_fontsize(fs)
    if not len(axs):
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(3, 2, width_ratios=gs_width_ratios)
        axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[2, 0])]

    idx0 = np.where(rec.fb.t >= tlim[0])[0][0]
    idx1 = np.where(rec.eb.t <= tlim[1])[0][-1]
    idx0_beh = np.where(rec.abf.t >= tlim[0])[0][0]
    idx1_beh = np.where(rec.abf.t <= tlim[1])[0][-1]
    xlim = tlim

    for i_ in range(2):
        ax = axs[i_*2]
        ph.adjust_spines(ax, [], ylim=[rec.ims[1-i_].c1.xedges[0], rec.ims[1-i_].c1.xedges[-1]], xlim=xlim)
        ax.pcolormesh(rec.ims[1-i_].t[idx0:idx1], rec.ims[1-i_].c1.xedges, rec.ims[1-i_].c1.a[idx0:idx1].T, cmap=cmaps[i_],
                      vmin=vms[i_][0], vmax=vms[i_][1])

        ax = axs[i_*2+1]
        ph.plot_colormap(ax, colormap=cmaps[i_], reverse_cm=False, ylim=[0, 256], yticks=[0, 256],
                         yticklabels=['%.1f' % vms[i_][0], '>%.1f' % vms[i_][1]],
                         ylabel='', label_rotation=90, label_pos="right", label_axeratio=1)

    ax = axs[4]
    ph.adjust_spines(ax, [], lw=1, ylim=[-180, 180], yticks=[], xlim=xlim, xticks=[], pad=0)
    xbar_ss = rec.subsample('xbar')[idx0:idx1]
    phase_eb = rec.eb.c1.phase_raw[idx0:idx1]
    if auto_offset:
        offset = calculate_offsets(rec, phase_eb, xbar_ss, flight=False)
    else:
        offset = 0
    if plot_bar:
        ax.plot(rec.abf.t[idx0_beh:idx1_beh], rec.abf.xbar[idx0_beh:idx1_beh], alpha=1, zorder=1, ls='none', marker='o',
                ms=ms_dot, mec='none', mfc=c_dot)
    for i_ in range(2):
        ax.plot(rec.ims[1-i_].t[idx0:idx1], fc.wrap(rec.ims[1-i_].c1.phase_raw[idx0:idx1]-offset), alpha=1, zorder=2, ls='none', marker='o', ms=ms, mec='none', mfc=colors[i_])

    ph.plot_x_scale_text(ax, bar_length=bar_length, text='%is' % bar_length, y_text_lower=0.85, x_text_lefter=.05, fontsize=fs)
    ax.fill_between(xlim, y1=-180, y2=-135., facecolor='black', edgecolor='none', alpha=0.1, zorder=1)
    ax.fill_between(xlim, y1=135, y2=180, facecolor='black', edgecolor='none', alpha=0.1, zorder=1)

    for vline in vlines:
        ax.axvline(vline, ls='--', lw=2, c=ph.dred, alpha=.6, zorder=1)

def plot_2c_sample_trace_2pb_paper_20200410(rec, axs=[], fig_size=(15, 3), tlim=[1,200], gs_width_ratios=[50,1],
                gs_height_ratios=[1,1,1], plot_bar=True, cmaps=[plt.cm.Blues, plt.cm.Reds], vm=[0,1.6],
                colors=[ph.blue, ph.dorange], color_dot='black', bar_length=10, fs=7, auto_offset=True,
                ms_dot=1, ms=1.5, vlines=[], alpha_marker=1):
    ph.set_fontsize(fs)
    if not len(axs):
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(3, 2, width_ratios=gs_width_ratios, height_ratios=gs_height_ratios)
        axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[2, 0])]

    idx0 = np.where(rec.pb.t >= tlim[0])[0][0]
    idx1 = np.where(rec.pb.t <= tlim[1])[0][-1]
    idx0_beh = np.where(rec.abf.t >= tlim[0])[0][0]
    idx1_beh = np.where(rec.abf.t <= tlim[1])[0][-1]
    xlim = tlim

    ax = axs[0]
    ph.adjust_spines(ax, [], ylim=[rec.pb.c2.xedges[0], rec.pb.c2.xedges[-1]], xlim=xlim)
    ax.pcolormesh(rec.pb.t[idx0:idx1], rec.pb.c2.xedges, rec.pb.c2.al[idx0:idx1].T, cmap=cmaps[0], vmin=vm[0], vmax=vm[1])
    ax = axs[2]
    ph.adjust_spines(ax, [], ylim=[rec.pb.c1.xedges[0], rec.pb.c1.xedges[-1]], xlim=xlim)
    ax.pcolormesh(rec.pb.t[idx0:idx1], rec.pb.c1.xedges, rec.pb.c1.al[idx0:idx1].T, cmap=cmaps[1], vmin=vm[0], vmax=vm[1])

    for i_ in range(2):
        ax = axs[i_*2+1]
        ph.plot_colormap(ax, colormap=cmaps[i_], reverse_cm=False, ylim=[0, 256], yticks=[0, 256],
                         yticklabels=['%.1f' % vm[0], '>%.1f' % vm[1]],
                         ylabel='', label_rotation=90, label_pos="right", label_axeratio=1)

    ax = axs[4]
    ph.adjust_spines(ax, [], lw=1, ylim=[-180, 180], yticks=[], xlim=xlim, xticks=[], pad=0)
    xbar_ss = rec.subsample('xbar')[idx0:idx1]
    phase_epg = rec.pb.c2.phase[idx0:idx1]
    if auto_offset:
        offset = calculate_offsets(rec, phase_epg, xbar_ss, flight=False)
    else:
        offset = 0
    if plot_bar:
        ax.plot(rec.abf.t[idx0_beh:idx1_beh], rec.abf.xbar[idx0_beh:idx1_beh], alpha=alpha_marker, zorder=1, ls='none', marker='o', ms=ms_dot, mec='none', mfc=color_dot)
    ax.plot(rec.pb.t[idx0:idx1], fc.wrap(rec.pb.c2.phase[idx0:idx1] - offset), alpha=alpha_marker, zorder=3, ls='none', marker='o',
            ms=ms, mec='none', mfc=colors[0])
    ax.plot(rec.pb.t[idx0:idx1], fc.wrap(rec.pb.c1.phase[idx0:idx1] - offset), alpha=alpha_marker, zorder=2, ls='none', marker='o',
            ms=ms, mec='none', mfc=colors[1])

    ph.plot_x_scale_text(ax, bar_length=bar_length, text='%is' % bar_length, y_text_lower=0.85, x_text_lefter=.05, fontsize=fs)
    ax.fill_between(xlim, y1=-180, y2=-135., facecolor='black', edgecolor='none', alpha=0.1, zorder=1)
    ax.fill_between(xlim, y1=135, y2=180, facecolor='black', edgecolor='none', alpha=0.1, zorder=1)

    for vline in vlines:
        ax.axvline(vline, ls='--', lw=.5, c=ph.dred, alpha=.6, zorder=1)

def plot_2c_sample_trace_2pb_vertical_paper_20200512(rec, axs=[], fig_size=(3, 15), tlim=[1,200], gs_height_ratios=[50,1], plot_bar=False,
                cmaps=[plt.cm.Blues, plt.cm.Reds], vm=[0,1.6], colors=[ph.blue, ph.dorange], bar_length=10, fs=7, auto_offset=True,
                ms=1.5, hlines=[]):
    ph.set_fontsize(fs)
    if not len(axs):
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(2, 3, height_ratios=gs_height_ratios)
        axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[1, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 1]), plt.subplot(gs[0, 2])]

    idx0 = np.where(rec.pb.t >= tlim[0])[0][0]
    idx1 = np.where(rec.pb.t <= tlim[1])[0][-1]
    idx0_beh = np.where(rec.abf.t >= tlim[0])[0][0]
    idx1_beh = np.where(rec.abf.t <= tlim[1])[0][-1]
    ylim = tlim

    ax = axs[0]
    ph.adjust_spines(ax, [], xlim=[rec.pb.c2.xedges[0], rec.pb.c2.xedges[-1]], ylim=ylim)
    ax.pcolormesh(rec.pb.c2.xedges, rec.pb.t[idx0:idx1], rec.pb.c2.al[idx0:idx1], cmap=cmaps[0], vmin=vm[0], vmax=vm[1])
    ax.invert_yaxis()
    ax = axs[2]
    ph.adjust_spines(ax, [], xlim=[rec.pb.c1.xedges[0], rec.pb.c1.xedges[-1]], ylim=ylim)
    ax.pcolormesh(rec.pb.c1.xedges, rec.pb.t[idx0:idx1], rec.pb.c1.al[idx0:idx1], cmap=cmaps[1], vmin=vm[0], vmax=vm[1])
    ax.invert_yaxis()

    for i_ in range(2):
        ax = axs[i_*2+1]
        ph.plot_colormap(ax, colormap=cmaps[i_], reverse_cm=False, ylim=[0, 256], yticks=[0, 256],
                         yticklabels=['%.1f' % vm[0], '>%.1f' % vm[1]], flipxy=True,
                         ylabel='', label_rotation=90, label_pos="right", label_axeratio=1)

    ax = axs[4]
    ph.adjust_spines(ax, [], lw=1, xlim=[-180, 180], yticks=[], ylim=ylim, xticks=[], pad=0)
    ax.invert_yaxis()
    xbar_ss = rec.subsample('xbar')[idx0:idx1]
    phase_epg = rec.pb.c2.phase[idx0:idx1]
    if auto_offset:
        offset = calculate_offsets(rec, phase_epg, xbar_ss, flight=False)
    else:
        offset = 0
    if plot_bar:
        ax.plot(rec.abf.xbar[idx0_beh:idx1_beh], rec.abf.t[idx0_beh:idx1_beh], alpha=1, zorder=1, ls='none', marker='o', ms=ms, mec='none', mfc='black')
    ax.plot(fc.wrap(rec.pb.c2.phase[idx0:idx1] - offset), rec.pb.t[idx0:idx1], alpha=1, zorder=2, ls='none', marker='o',
            ms=ms, mec='none', mfc=colors[0])
    ax.plot(fc.wrap(rec.pb.c1.phase[idx0:idx1] - offset), rec.pb.t[idx0:idx1], alpha=1, zorder=2, ls='none', marker='o',
            ms=ms, mec='none', mfc=colors[1])

    ph.plot_y_scale_text(ax, bar_length=bar_length, text='%is' % bar_length, y_text_upper=0.85, x_text_right=.05, fontsize=fs)
    # ax.fill_betweenx(ylim, x1=-180, x2=-135., facecolor='black', edgecolor='none', alpha=0.1, zorder=1)
    # ax.fill_betweenx(ylim, x1=135, x2=180, facecolor='black', edgecolor='none', alpha=0.1, zorder=1)

    for hline in hlines:
        ax.axhline(hline, ls='--', lw=2, c=ph.dred, alpha=.6, zorder=1)

def plot_2c_sample_trace_wba_paper_20200526(rec, axs=[], fig_size=(15, 3), tlim=[1,200], gs_width_ratios=[50,1], plot_bar=True,
                cmaps=[plt.cm.Blues, plt.cm.Oranges], vm=[0,1.6], colors=[ph.blue, ph.dorange], bar_length=10, fs=7, auto_offset=True,
                ms=1.5, vlines=[]):
    ph.set_fontsize(fs)
    if not len(axs):
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(7, 2, width_ratios=gs_width_ratios)
        axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]),
               plt.subplot(gs[2, 0]), plt.subplot(gs[3, 0]), plt.subplot(gs[4, 0]), plt.subplot(gs[5, 0]), plt.subplot(gs[6, 0])]

    idx0 = np.where(rec.fb.t >= tlim[0])[0][0]
    idx1 = np.where(rec.eb.t <= tlim[1])[0][-1]
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
    xbar_ss = rec.subsample('xbar')[idx0:idx1]
    phase_eb = rec.eb.c1.phase[idx0:idx1]
    if auto_offset:
        offset = calculate_offsets(rec, phase_eb, xbar_ss, flight=False)
    else:
        offset = 0
    if plot_bar:
        ax.plot(rec.abf.t[idx0_beh:idx1_beh], rec.abf.xbar[idx0_beh:idx1_beh], alpha=1, zorder=1, ls='none', marker='o', ms=ms, mec='none', mfc='black')
    for i_ in range(2):
        ax.plot(rec.ims[1-i_].t[idx0:idx1], fc.wrap(rec.ims[1-i_].c1.phase[idx0:idx1]-offset), alpha=1, zorder=2, ls='none', marker='o', ms=ms, mec='none', mfc=colors[i_])

    ph.plot_x_scale_text(ax, bar_length=bar_length, text='%is' % bar_length, y_text_lower=0.85, x_text_lefter=.05, fontsize=fs)
    ax.fill_between(xlim, y1=-180, y2=-135., facecolor='black', edgecolor='none', alpha=0.1, zorder=1)
    ax.fill_between(xlim, y1=135, y2=180, facecolor='black', edgecolor='none', alpha=0.1, zorder=1)

    for vline in vlines:
        ax.axvline(vline, ls='--', lw=2, c=ph.dred, alpha=.6, zorder=1)

    ax = axs[5]
    ph.adjust_spines(ax, [], lw=1, ylim=[30, 90], yticks=[], xlim=xlim, xticks=[], pad=0)
    ax.plot(rec.abf.t[idx0_beh:idx1_beh], rec.abf.lwa[idx0_beh:idx1_beh], alpha=1, zorder=1, lw=.3, c='black')
    ax = axs[6]
    ph.adjust_spines(ax, [], lw=1, ylim=[30, 90], yticks=[], xlim=xlim, xticks=[], pad=0)
    ax.plot(rec.abf.t[idx0_beh:idx1_beh], rec.abf.rwa[idx0_beh:idx1_beh], alpha=1, zorder=1, lw=.3, c='black')
    ax = axs[7]
    ph.adjust_spines(ax, [], lw=1, ylim=[-50, 50], yticks=[], xlim=xlim, xticks=[], pad=0)
    ax.plot(rec.abf.t[idx0_beh:idx1_beh], rec.abf.lmr[idx0_beh:idx1_beh], alpha=1, zorder=1, lw=.3, c='black')
    ax.axhline(0, ls='--', lw=1, c=ph.grey9, alpha=.3, zorder=1)
    ax = axs[8]
    ph.adjust_spines(ax, [], lw=1, ylim=[-180, 180], yticks=[], xlim=xlim, xticks=[], pad=0)
    ax.plot(rec.ims[0].t[idx0:idx1], fc.circsubtract(rec.ims[1].c1.phase[idx0:idx1], rec.ims[0].c1.phase[idx0:idx1]),
            alpha=1, zorder=2, ls='none', marker='o', ms=ms, mec='none', mfc='black')
    ax.axhline(0, ls='--', lw=1, c=ph.grey9, alpha=.3, zorder=1)

def plot_2c_sample_trace_2eb_paper_20210323(rec, axs=[], fig_size=(15, 3), tlim=[1,200], gs_width_ratios=[50,1],
                gs_height_ratios=[1,1,1], plot_bar=True, cmaps=[plt.cm.Blues, plt.cm.Reds], vm=[0,1.6],
                colors=[ph.blue, ph.dorange], color_dot='black', bar_length=10, fs=7, auto_offset=True,
                ms_dot=1, ms=1.5, vlines=[], alpha_marker=1):
    ph.set_fontsize(fs)
    if not len(axs):
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(3, 2, width_ratios=gs_width_ratios, height_ratios=gs_height_ratios)
        axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[2, 0])]

    idx0 = np.where(rec.ims[0].t >= tlim[0])[0][0]
    idx1 = np.where(rec.ims[0].t <= tlim[1])[0][-1]
    idx0_beh = np.where(rec.abf.t >= tlim[0])[0][0]
    idx1_beh = np.where(rec.abf.t <= tlim[1])[0][-1]
    xlim = tlim

    ax = axs[0]
    ph.adjust_spines(ax, [], ylim=[rec.ims[0].c2.xedges[0], rec.ims[0].c2.xedges[-1]], xlim=xlim)
    ax.pcolormesh(rec.ims[0].t[idx0:idx1], rec.ims[0].c2.xedges, rec.ims[0].c2.al[idx0:idx1].T, cmap=cmaps[0], vmin=vm[0], vmax=vm[1])
    ax = axs[2]
    ph.adjust_spines(ax, [], ylim=[rec.ims[0].c1.xedges[0], rec.ims[0].c1.xedges[-1]], xlim=xlim)
    ax.pcolormesh(rec.ims[0].t[idx0:idx1], rec.ims[0].c1.xedges, rec.ims[0].c1.al[idx0:idx1].T, cmap=cmaps[1], vmin=vm[0], vmax=vm[1])

    for i_ in range(2):
        ax = axs[i_*2+1]
        ph.plot_colormap(ax, colormap=cmaps[i_], reverse_cm=False, ylim=[0, 256], yticks=[0, 256],
                         yticklabels=['%.1f' % vm[0], '>%.1f' % vm[1]],
                         ylabel='', label_rotation=90, label_pos="right", label_axeratio=1)

    ax = axs[4]
    ph.adjust_spines(ax, [], lw=1, ylim=[-180, 180], yticks=[], xlim=xlim, xticks=[], pad=0)
    xbar_ss = rec.subsample('xbar')[idx0:idx1]
    phase_epg = rec.ims[0].c2.phase[idx0:idx1]
    if auto_offset:
        offset = calculate_offsets(rec, phase_epg, xbar_ss, flight=False)
    else:
        offset = 0
    if plot_bar:
        ax.plot(rec.abf.t[idx0_beh:idx1_beh], rec.abf.xbar[idx0_beh:idx1_beh], alpha=alpha_marker, zorder=1, ls='none', marker='o', ms=ms_dot, mec='none', mfc=color_dot)
    ax.plot(rec.ims[0].t[idx0:idx1], fc.wrap(rec.ims[0].c2.phase[idx0:idx1] - offset), alpha=alpha_marker, zorder=3, ls='none', marker='o',
            ms=ms, mec='none', mfc=colors[0])
    ax.plot(rec.ims[0].t[idx0:idx1], fc.wrap(rec.ims[0].c1.phase[idx0:idx1] - offset), alpha=alpha_marker, zorder=2, ls='none', marker='o',
            ms=ms, mec='none', mfc=colors[1])

    ph.plot_x_scale_text(ax, bar_length=bar_length, text='%is' % bar_length, y_text_lower=0.85, x_text_lefter=.05, fontsize=fs)
    ax.fill_between(xlim, y1=-180, y2=-135., facecolor='black', edgecolor='none', alpha=0.1, zorder=1)
    ax.fill_between(xlim, y1=135, y2=180, facecolor='black', edgecolor='none', alpha=0.1, zorder=1)

    for vline in vlines:
        ax.axvline(vline, ls='--', lw=.2, c='black', alpha=.6, zorder=1)

def plot_2c_sample_trace_2eb_paper_4row_20210408(rec, axs=[], fig_size=(15, 3), tlim=[1,200], gs_width_ratios=[50,1],
                gs_height_ratios=[1,1,1,1], plot_bar=True, cmaps=[plt.cm.Blues, plt.cm.Reds], vm=[0,1.6],
                colors=[ph.blue, ph.dorange], color_dot='black', bar_length=10, fs=7, auto_offset=True,
                ms_dot=1, ms=1.5, vlines=[], alpha_marker=1):
    ph.set_fontsize(fs)
    if not len(axs):
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(4, 2, width_ratios=gs_width_ratios, height_ratios=gs_height_ratios)
        axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]),
               plt.subplot(gs[2, 0]), plt.subplot(gs[3, 0])]

    idx0 = np.where(rec.ims[0].t >= tlim[0])[0][0]
    idx1 = np.where(rec.ims[0].t <= tlim[1])[0][-1]
    idx0_beh = np.where(rec.abf.t >= tlim[0])[0][0]
    idx1_beh = np.where(rec.abf.t <= tlim[1])[0][-1]
    xlim = tlim

    ax = axs[0]
    ph.adjust_spines(ax, [], ylim=[rec.ims[0].c2.xedges[0], rec.ims[0].c2.xedges[-1]], xlim=xlim)
    ax.pcolormesh(rec.ims[0].t[idx0:idx1], rec.ims[0].c2.xedges, rec.ims[0].c2.al[idx0:idx1].T, cmap=cmaps[0], vmin=vm[0], vmax=vm[1])
    ax = axs[2]
    ph.adjust_spines(ax, [], ylim=[rec.ims[0].c1.xedges[0], rec.ims[0].c1.xedges[-1]], xlim=xlim)
    ax.pcolormesh(rec.ims[0].t[idx0:idx1], rec.ims[0].c1.xedges, rec.ims[0].c1.al[idx0:idx1].T, cmap=cmaps[1], vmin=vm[0], vmax=vm[1])

    for i_ in range(2):
        ax = axs[i_*2+1]
        ph.plot_colormap(ax, colormap=cmaps[i_], reverse_cm=False, ylim=[0, 256], yticks=[0, 256],
                         yticklabels=['%.1f' % vm[0], '>%.1f' % vm[1]],
                         ylabel='', label_rotation=90, label_pos="right", label_axeratio=1)

    ax = axs[4]
    ph.adjust_spines(ax, [], lw=1, ylim=[-180, 180], yticks=[], xlim=xlim, xticks=[], pad=0)
    xbar_ss = rec.subsample('xbar')[idx0:idx1]
    phase_epg = rec.ims[0].c2.phase[idx0:idx1]
    if auto_offset:
        offset = calculate_offsets(rec, phase_epg, xbar_ss, flight=False)
    else:
        offset = 0
    if plot_bar:
        ax.plot(rec.abf.t[idx0_beh:idx1_beh], rec.abf.xbar[idx0_beh:idx1_beh], alpha=alpha_marker, zorder=1, ls='none', marker='o', ms=ms_dot, mec='none', mfc=color_dot)
    ax.plot(rec.ims[0].t[idx0:idx1], fc.wrap(rec.ims[0].c2.phase[idx0:idx1] - offset), alpha=alpha_marker, zorder=3, ls='none', marker='o',
            ms=ms, mec='none', mfc=colors[0])
    ax.plot(rec.ims[0].t[idx0:idx1], fc.wrap(rec.ims[0].c1.phase[idx0:idx1] - offset), alpha=alpha_marker, zorder=2, ls='none', marker='o',
            ms=ms, mec='none', mfc=colors[1])

    ph.plot_x_scale_text(ax, bar_length=bar_length, text='%is' % bar_length, y_text_lower=0.85, x_text_lefter=.05, fontsize=fs)
    ax.fill_between(xlim, y1=-180, y2=-135., facecolor='black', edgecolor='none', alpha=0.1, zorder=1)
    ax.fill_between(xlim, y1=135, y2=180, facecolor='black', edgecolor='none', alpha=0.1, zorder=1)

    for vline in vlines:
        ax.axvline(vline, ls='--', lw=.2, c='black', alpha=.6, zorder=1)

    ax = axs[5]
    ph.adjust_spines(ax, [], lw=1, ylim=[-180, 180], yticks=[], xlim=xlim, xticks=[], pad=0)
    ys = fc.circsubtract(fc.wrap(rec.ims[0].c2.phase[idx0:idx1] - offset), fc.wrap(rec.ims[0].c1.phase[idx0:idx1] - offset))
    ax.plot(rec.ims[0].t[idx0:idx1], ys, alpha=alpha_marker, zorder=3, ls='none', marker='o', ms=ms, mec='none', mfc='black')
    ax.axhline(0, ls='--', lw=.2, c='black', alpha=.6, zorder=1)
    for vline in vlines:
        ax.axvline(vline, ls='--', lw=.2, c='black', alpha=.6, zorder=1)

# raw, df/f, normalized df/f
def plot_2c_sample_trace_paper_raw_20210711(rec, axs=[], fig_size=(3, 15), tlim=[1,200], gs_height_ratios=[50,1],
                sig_type='dff0', locnums=[0,1],chnums=[1,1],
                cmaps=[plt.cm.Blues, plt.cm.Reds], colors=[ph.blue, ph.dorange], vms=[[0,1.6],[0,1.6]],
                bar_length=2, fs=6, ms=1.5, hlines=[]):
    ph.set_fontsize(fs)
    if not len(axs):
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(2, 3, height_ratios=gs_height_ratios)
        axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[1, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 1]), plt.subplot(gs[0, 2])]

    idx0 = np.where(rec.ims[locnums[0]].t >= tlim[0])[0][0]
    idx1 = np.where(rec.ims[locnums[0]].t <= tlim[1])[0][-1]
    ylim = tlim

    chs = [getattr(rec.ims[locnums[0]], 'c%i' % chnums[0]), getattr(rec.ims[locnums[1]], 'c%i' % chnums[1])]
    if sig_type == 'dff0':
        heatmap_sigs = [chs[0].an, chs[1].an]
        phases = [chs[0].phase_dff0, chs[1].phase_dff0]
    elif sig_type == 'raw':
        heatmap_sigs = [chs[0].a, chs[1].a]
        phases = [chs[0].phase_raw, chs[1].phase_raw]
    elif sig_type == 'norm':
        heatmap_sigs = [chs[0].al, chs[1].al]
        phases = [chs[0].phase, chs[1].phase]

    for ic in range(2):
        ax = axs[ic*2]
        ph.adjust_spines(ax, [], xlim=[chs[ic].xedges[0], chs[ic].xedges[-1]], ylim=ylim)
        ax.pcolormesh(chs[ic].xedges, rec.ims[locnums[ic]].t[idx0:idx1+1], heatmap_sigs[ic][idx0:idx1], cmap=cmaps[ic], vmin=vms[ic][0], vmax=vms[ic][1])
        ax.invert_yaxis()

        ax = axs[ic*2+1]
        ph.plot_colormap(ax, colormap=cmaps[ic], reverse_cm=False, ylim=[0, 256], yticks=[0, 256],
                         yticklabels=['%.1f' % vms[ic][0], '>%.1f' % vms[ic][1]], flipxy=True,
                         ylabel='', label_rotation=90, label_pos="right", label_axeratio=1)

    ax = axs[4]
    ph.adjust_spines(ax, [], lw=1, xlim=[-180, 180], yticks=[], ylim=ylim, xticks=[], pad=0)
    ax.invert_yaxis()
    for ic in range(2):
        ax.plot(fc.wrap(phases[ic][idx0:idx1]), rec.ims[locnums[ic]].t[idx0:idx1], alpha=1, zorder=2, ls='none', marker='o',
                ms=ms, mec='none', mfc=colors[ic])

    ph.plot_y_scale_text(ax, bar_length=bar_length, text='%is' % bar_length, y_text_upper=0.85, x_text_right=.05, fontsize=fs)

    for hline in hlines:
        ax.axhline(hline, ls='--', lw=2, c=ph.dred, alpha=.6, zorder=1)



# single compass imaging walking analysis plots

def plot_titrate_standing_params(rec, th_stand_dforw_ub=2, th_stand_dhead_ub=30, th_stand_dside_ub=2, min_contig_len=3):

    plt.figure(1, [20, 15])
    gs = gridspec.GridSpec(3, 1)
    sigs_tif_posi = [rec.tif.forw, rec.tif.head, rec.tif.side]
    sigs_tif_deri = [rec.tif.dforw, rec.tif.dhead, rec.tif.dside]
    sigs_abf_posi = [rec.abf.forw, rec.abf.head, rec.abf.side]
    sigs_abf_deri = [rec.abf.dforw, rec.abf.dhead, rec.abf.dside]
    ylims = [[-15, 90], [-300, 2000], [-15, 90], ]
    yticks = [np.arange(-10, 21, 10), np.arange(-300, 301, 300), np.arange(-10, 21, 10)]
    ths = [th_stand_dforw_ub, th_stand_dhead_ub, th_stand_dside_ub]
    ylabels = ['forw', 'head', 'side']

    # np.convolve(x, np.ones((N,))/N, mode='valid')
    inds_stand = (np.abs(rec.tif.dforw) < th_stand_dforw_ub) & (np.abs(rec.tif.dhead) < th_stand_dhead_ub) \
                 & (np.abs(rec.tif.dside) < th_stand_dside_ub)
    idxss_stand = fc.get_contiguous_inds(inds_stand, keep_ends=True, min_contig_len=min_contig_len)

    for irow in [0, 1, 2]:
        ax = plt.subplot(gs[irow, 0])
        ax2 = ax.twinx()
        # ax.plot(rec.tif.t, sigs_tif_posi[irow], c=ph.grey9, lw=2, alpha=.9, zorder=1)
        ax.plot(rec.abf.t, sigs_abf_posi[irow], c=ph.blue, lw=1, alpha=.5, zorder=2)
        ax2.plot(rec.tif.t, sigs_tif_deri[irow], c=ph.grey9, lw=1, alpha=.9, zorder=1)
        ax2.plot(rec.abf.t, sigs_abf_deri[irow], c=ph.blue, lw=1, alpha=.5, zorder=2)
        ax2.axhline(ths[irow], c=ph.grey3, ls='--', alpha=.3)
        ax2.axhline(-ths[irow], c=ph.grey3, ls='--', alpha=.3)

        ax.set_xlim([rec.tif.t[0], rec.tif.t[-1]])
        ax.set_ylim([-360, 181])
        ax.set_yticks(np.arange(-180, 180, 90))
        ax.set_ylabel(ylabels[irow])
        ax2.set_ylim(ylims[irow])
        ax2.set_yticks(yticks[irow])

        for _idxs in idxss_stand:
            ax.plot(rec.tif.t[_idxs[0]:_idxs[-1] + 1], sigs_tif_posi[irow][_idxs[0]:_idxs[-1] + 1], c=ph.red, lw=1,
                    zorder=3)
            ax2.plot(rec.tif.t[_idxs[0]:_idxs[-1] + 1], sigs_tif_deri[irow][_idxs[0]:_idxs[-1] + 1], c=ph.red, lw=1,
                     zorder=3)

def plot_compare_heading_with_td_unit(rec, th_stand_dforw_ub=2, th_stand_dhead_ub=30, th_stand_dside_ub=2, min_contig_len=3):

    plt.figure(1, [20, 5])
    gs = gridspec.GridSpec(1, 1)
    sigs_tif_posi = [rec.tif.forw, rec.tif.head, rec.tif.side]
    sigs_tif_deri = [rec.tif.dforw, rec.tif.dhead, rec.tif.dside]
    sigs_abf_posi = [rec.abf.forw, rec.abf.head, rec.abf.side]
    sigs_abf_deri = [rec.abf.dforw, rec.abf.dhead, rec.abf.dside]
    ylims = [[-15, 90], [-300, 2000], [-15, 90], ]
    yticks = [np.arange(-10, 21, 10), np.arange(-300, 301, 300), np.arange(-10, 21, 10)]
    ths = [th_stand_dforw_ub, th_stand_dhead_ub, th_stand_dside_ub]
    ylabels = ['forw', 'head', 'side']

    # np.convolve(x, np.ones((N,))/N, mode='valid')
    inds_stand = detect_standing(rec, th_stand_dforw_ub, th_stand_dhead_ub, th_stand_dside_ub)
    idxss_stand = fc.get_contiguous_inds(inds_stand, keep_ends=True, min_contig_len=min_contig_len)

    for irow in [0, 1, 2]:
        ax = plt.subplot(gs[irow, 0])
        ax2 = ax.twinx()
        # ax.plot(rec.tif.t, sigs_tif_posi[irow], c=ph.grey9, lw=2, alpha=.9, zorder=1)
        ax.plot(rec.abf.t, sigs_abf_posi[irow], c=ph.blue, lw=1, alpha=.5, zorder=2)
        ax2.plot(rec.tif.t, sigs_tif_deri[irow], c=ph.grey9, lw=1, alpha=.9, zorder=1)
        ax2.plot(rec.abf.t, sigs_abf_deri[irow], c=ph.blue, lw=1, alpha=.5, zorder=2)
        ax2.axhline(ths[irow], c=ph.grey3, ls='--', alpha=.3)
        ax2.axhline(-ths[irow], c=ph.grey3, ls='--', alpha=.3)

        ax.set_xlim([rec.tif.t[0], rec.tif.t[-1]])
        ax.set_ylim([-360, 181])
        ax.set_yticks(np.arange(-180, 180, 90))
        ax.set_ylabel(ylabels[irow])
        ax2.set_ylim(ylims[irow])
        ax2.set_yticks(yticks[irow])

        for _idxs in idxss_stand:
            ax.plot(rec.tif.t[_idxs[0]:_idxs[-1] + 1], sigs_tif_posi[irow][_idxs[0]:_idxs[-1] + 1], c=ph.red, lw=1,
                    zorder=3)
            ax2.plot(rec.tif.t[_idxs[0]:_idxs[-1] + 1], sigs_tif_deri[irow][_idxs[0]:_idxs[-1] + 1], c=ph.red, lw=1,
                     zorder=3)


def plot_phase_scatter_unit(recs, ax=False, xtype='Heading', exclude_standing=True, show_axes=['bottom', 'left'],
                            exclude_puff=True, th_stand_dforw_ub=2, th_stand_dhead_ub=30, th_stand_dside_ub=2, win_puff_s=1):
    if not ax:
        ax = ph.large_ax([3, 6])

    for rec in recs:
        inds_plot = rec.tif.phase < 1e4
        if exclude_standing:
            inds_stand = detect_standing(rec, th_stand_dforw_ub, th_stand_dhead_ub, th_stand_dside_ub)
            inds_plot = inds_plot & ~inds_stand
        if exclude_puff:
            inds_nopuff = rec.tif.t > -1
            for t_puff in rec.tc_puff:
                inds_nopuff = inds_nopuff & ((rec.tif.t < t_puff) | (rec.tif.t > (t_puff + win_puff_s)))
            inds_plot = inds_plot & inds_nopuff

        sig_heading = rec.tif.xbar if xtype == 'Heading' else calculate_td(rec, exclude_standing=True)
        ax.plot(sig_heading[inds_plot], rec.tif.phase[inds_plot], ls='none', marker='.',
                mfc=ph.blue, mec=ph.blue, ms=4, alpha=.4, zorder=2)
        ax.plot(sig_heading[inds_plot], rec.tif.phase[inds_plot] + 360, ls='none', marker='.',
                mfc=ph.blue, mec=ph.blue, ms=4, alpha=.4, zorder=2)

    ax.fill_betweenx([-180, 540], x1=-180, x2=-145, color='black', alpha=0.1, zorder=1)
    ax.fill_betweenx([-180, 540], x1=145., x2=180, color='black', alpha=0.1, zorder=1)
    alpha_diag = .2
    ax.plot([-180, 180], [-360, 0], lw=2, c='black', alpha=alpha_diag, zorder=1)
    ax.plot([-180, 180], [-180, 180], lw=2, c='black', alpha=alpha_diag, zorder=1)
    ax.plot([-180, 180], [0, 360], lw=2, c='black', alpha=alpha_diag, zorder=1)
    ax.plot([-180, 180], [180, 540], lw=2, c='black', alpha=alpha_diag, zorder=1)
    ax.plot([-180, 180], [360, 720], lw=2, c='black', alpha=alpha_diag, zorder=1)

    ax.set_aspect('equal', 'datalim')
    ph.adjust_spines(ax, show_axes, lw=1, xlim=[-181, 180], ylim=[-180, 541],
                     xticks=np.arange(-180, 181, 90), yticks=np.arange(-180, 541, 90))
    ax.set_xlabel(xtype)
    ax.set_ylabel('phase')
    ax.set_title('%s' % rec.recid)

def plot_xcorr_phase_heading_and_td(group, ax=False, im_num=0, ch_num=1):
    gs = gridspec.GridSpec(1, 2)
    if not ax:
        ax = ph.large_ax([4, 2])
    xlim = [-2, 2]
    t_lag = np.arange(xlim[0] - 0.1, xlim[1] + 0.1, 0.005)
    colors = ['black', ph.orange]
    titles = ['xbar', 'travel direction']

    def chanelnum_to_beh(rec, i):
        if i == 0:
            return rec.subsample('xbar')
        else:
            return calculate_td(rec, exclude_standing=True)

    for icol in [0, 1]:
        ax = plt.subplot(gs[0, icol])

        xcorr_group = []
        for recs in group:
            xcorr_fly = []
            for rec in recs:
                beh = chanelnum_to_beh(rec, icol)
                walking_bouts_inds = get_walking_bouts_inds(rec, )
                for ind in walking_bouts_inds:
                    idxs = np.arange(ind[0], ind[-1] + 1)
                    ch = getattr(rec.ims[im_num], 'c%i' % ch_num)
                    _sig = fc.unwrap(ch.phase[idxs])
                    _beh = fc.unwrap(beh[idxs])
                    t_lag_trial, xcorr_trial = fc.xcorr(_sig, _beh, sampling_period=1.0 / rec.ims[0].sampling_rate)
                    f = interp1d(t_lag_trial, xcorr_trial, kind='linear')
                    xcorr_interp = f(t_lag)
                    xcorr_fly.append(xcorr_interp)

            if len(xcorr_fly):
                xcorr_group.append(np.nanmean(xcorr_fly, axis=0))

        ax.plot(t_lag, np.nanmean(xcorr_group, axis=0), c=colors[icol], alpha=1, lw=2)
        ax.set_title(titles[icol])
        for xcorr_trial in xcorr_group:
            ax.plot(t_lag, xcorr_trial, c=colors[icol], alpha=.5, lw=1)
        ph.adjust_spines(ax, ['bottom', 'left'], lw=1, xlim=[-2, 2], ylim=[-.3, 1],
                         xticks=np.arange(-2, 2.1, 2), yticks=np.arange(-.25, 1.1, .25))

def plot_xcorr_phase_heading_and_td_deviation(group):
    gs = gridspec.GridSpec(1, 1)
    ax = ph.large_ax([3,3])
    xlim = [-2, 2]
    t_lag = np.arange(xlim[0] - 0.1, xlim[1] + 0.1, 0.005)

    xcorr_group = []
    for recs in group:
        offset_phase = calculate_phaseoffset(recs)
        xcorr_fly = []

        for rec in recs:
            walking_bouts_inds = get_walking_bouts_inds(rec, )
            for ind in walking_bouts_inds:
                idxs = np.arange(ind[0], ind[-1] + 1)
                _sig = fc.wrap(rec.tif.phase[idxs] - rec.tif.xbar[idxs] - offset_phase)
                _beh = slope_to_headning_angle(rec.tif.dside, rec.tif.dforw)[idxs]

                t_lag_trial, xcorr_trial = fc.xcorr(_sig, _beh, sampling_period=1.0 / rec.tif.sampling_rate)
                f = interp1d(t_lag_trial, xcorr_trial, kind='linear')
                xcorr_interp = f(t_lag)
                xcorr_fly.append(xcorr_interp)

        if len(xcorr_fly):
            xcorr_group.append(np.nanmean(xcorr_fly, axis=0))

    ax.plot(t_lag, np.nanmean(xcorr_group, axis=0), c='black', alpha=1, lw=2)
    # ax.set_title(titles[icol])
    for xcorr_trial in xcorr_group:
        ax.plot(t_lag, xcorr_trial, c='black', alpha=.5, lw=1)
    ph.adjust_spines(ax, ['bottom', 'left'], lw=1, xlim=[-2, 2], ylim=[-.3, 1],
                     xticks=np.arange(-2, 2.1, 2), yticks=np.arange(-.25, 1.1, .25))

def plot_1c_walk_sample_trace_paper_20200324(rec, axs=[], fig_size=(1.5, .7), tlim=[1,100], gs_width_ratios=[50,1], plot_bar=True,
                cmap=plt.cm.Oranges, vm=[0,1.6], color=ph.dorange, bar_length=10, fs=7, auto_offset=True, ms=1.5):
    ph.set_fontsize(fs)
    if not len(axs):
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(2, 2, width_ratios=gs_width_ratios)
        axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[1, 0])]

    idx0 = np.where(rec.ims[0].t >= tlim[0])[0][0]
    idx1 = np.where(rec.ims[0].t <= tlim[1])[0][-1]
    idx0_beh = np.where(rec.abf.t >= tlim[0])[0][0]
    idx1_beh = np.where(rec.abf.t <= tlim[1])[0][-1]
    xlim = tlim

    ax = axs[0]
    ph.adjust_spines(ax, [], ylim=[rec.ims[0].c1.xedges[0], rec.ims[0].c1.xedges[-1]], xlim=xlim)
    ax.pcolormesh(rec.ims[0].t[idx0:idx1], rec.ims[0].c1.xedges, rec.ims[0].c1.al[idx0:idx1].T, cmap=cmap, vmin=vm[0], vmax=vm[1])

    ax = axs[1]
    ph.plot_colormap(ax, colormap=cmap, reverse_cm=False, ylim=[0, 256], yticks=[0, 256],
                     yticklabels=['%.1f' % vm[0], '>%.1f' % vm[1]],
                     ylabel='', label_rotation=90, label_pos="right", label_axeratio=1)

    ax = axs[2]
    ph.adjust_spines(ax, [], lw=1, ylim=[-180, 180], yticks=[], xlim=xlim, xticks=[], pad=0)
    xbar_ss = rec.subsample('xbar')[idx0:idx1]
    phase = rec.ims[0].c1.phase[idx0:idx1]
    if auto_offset:
        offset = calculate_offsets(rec, phase, xbar_ss, flight=False)
    else:
        offset = 0
    if plot_bar:
        ax.plot(rec.abf.t[idx0_beh:idx1_beh], rec.abf.xbar[idx0_beh:idx1_beh], alpha=1, zorder=1, ls='none', marker='o', ms=ms, mec='none', mfc='black')
    ax.plot(rec.ims[0].t[idx0:idx1], fc.wrap(rec.ims[0].c1.phase[idx0:idx1]-offset), alpha=1, zorder=2, ls='none', marker='o', ms=ms, mec='none', mfc=color)

    ph.plot_x_scale_text(ax, bar_length=bar_length, text='%is' % bar_length, y_text_lower=0.85, x_text_lefter=.05, fontsize=fs)
    ax.fill_between(xlim, y1=-180, y2=-135., edgecolor='none', facecolor='black', alpha=0.1, zorder=1)
    ax.fill_between(xlim, y1=135, y2=180, edgecolor='none', facecolor='black', alpha=0.1, zorder=1)



# dual compasses imaging walking analysis plots

def plot_tc_td_phase_deviation_unit(group, ax=False, shift_beh_n=1, exclude_standing=True, exclude_puff=True,
                                    th_stand_dforw_ub=2, th_stand_dhead_ub=30, th_stand_dside_ub=2, win_puff_s=1,
                                    thresh_smallturn_dhead_ub=30, binnum=20, xlim=[-180,180], ylim=[-180, 180],
                                    smooth_order=1,):
    # calculate offset: phase_shifted = phase - offset
    if not ax:
        ax = ph.large_ax([3, 3])
    ys = []
    for recs in group:
        sig_fly = []
        beh_fly = []
        offset_phase = calculate_phaseoffset(recs)
        for rec in recs:
            inds_plot = rec.tif.phase < 1e4
            if exclude_standing:
                inds_stand = detect_standing(rec, th_stand_dforw_ub, th_stand_dhead_ub, th_stand_dside_ub)
                inds_plot = inds_plot & ~inds_stand
            if exclude_puff:
                inds_nopuff = rec.tif.t > -1
                for t_puff in rec.tc_puff:
                    inds_nopuff = inds_nopuff & ((rec.tif.t < t_puff) | (rec.tif.t > (t_puff + win_puff_s)))
                inds_plot = inds_plot & inds_nopuff
            inds_plot = inds_plot & (np.abs(rec.tif.dhead) < thresh_smallturn_dhead_ub)

            deviation_angle = slope_to_headning_angle(rec.tif.dside, rec.tif.dforw)
            deviation_angle_smoothed = np.convolve(deviation_angle, np.ones((smooth_order,)) / smooth_order, mode='same')
            # deviation_angle_smoothed = deviation_angle

            if shift_beh_n > 0:
                sig_fly.extend(fc.wrap(rec.tif.phase[inds_plot] - rec.tif.xbar[inds_plot] - offset_phase)[:-shift_beh_n])
                beh_fly.extend(deviation_angle_smoothed[inds_plot][shift_beh_n:])
            elif shift_beh_n < 0:
                sig_fly.extend(fc.wrap(rec.tif.phase[inds_plot] - rec.tif.xbar[inds_plot] - offset_phase)[-shift_beh_n:])
                beh_fly.extend(deviation_angle_smoothed[inds_plot][:shift_beh_n])
            else:
                sig_fly.extend(fc.wrap(rec.tif.phase[inds_plot] - rec.tif.xbar[inds_plot] - offset_phase))
                beh_fly.extend(deviation_angle_smoothed[inds_plot])

        x, y, _ = fc.scatter_to_errorbar(x_input=beh_fly, y_input=sig_fly, binx=binnum, xmin=xlim[0], xmax=xlim[-1])
        ys.append(y)

    for y in ys:
        ax.plot(x, y, lw=1, c='black', alpha=.6)
    ax.plot(x, np.nanmean(ys, axis=0), lw=2, c='black', alpha=1)
    ax.axhline(0, c=ph.blue, ls='--', alpha=.9, lw=2, zorder=3)
    # ax.set_aspect('equal', 'datalim')
    ph.adjust_spines(ax, ['bottom', 'left'], lw=1, xlim=xlim, ylim=ylim, xticks=xlim, yticks=ylim)
    ax.set_title('n=%i' % shift_beh_n)

def get_trial_criterion_info_bww(rec, t0, t1, filter_window_s=0.1, thresh_dforw_yes=-6, thresh_dforw_dur_yes=1,
                                 thresh_dforw_no=-0.5, thresh_dside_ratio_uni=0.8):
    """check whether the trial between t0 and t1 meet the critera for backward walking"""
    dt = np.diff(rec.abf.tc).mean()
    sigma = filter_window_s / dt
    t = rec.abf.t

    dforw = rec.abf.dforw
    dside = rec.abf.dside

    dforw_filted = ndimage.filters.gaussian_filter1d(dforw, sigma)
    dside_filted = ndimage.filters.gaussian_filter1d(dside, sigma)

    flag_trialcri_fail = 0      # whether this trial meet the criteria for bww
    dside_pos = True            # direction of dside

    _inds_firstpeak = np.where((t >= t0) & (t <= t1) & (dforw_filted < thresh_dforw_yes))[0]
    if not len(_inds_firstpeak):        # check whether the bww is fast enough, cri 1
        flag_trialcri_fail = 1
    else:
        # go to the beginning and ending of bww, check whether it is long enough, cri 2
        ind_dforw_pass_thresh = _inds_firstpeak[0]
        _inds_pre = np.where((t >= t0) & (t < t[ind_dforw_pass_thresh]) & (dforw_filted >= thresh_dforw_no))[0]
        if len(_inds_pre):
            ind_negdforw_start = int(_inds_pre[-1])
        else:
            ind_negdforw_start = int(np.where(t >= t0)[0][0])

        _inds_post = np.where((t <= t1) & (t > t[ind_dforw_pass_thresh]) & (dforw_filted >= thresh_dforw_no))[0]
        if len(_inds_post):
            ind_negdforw_end = int(_inds_post[0])
        else:
            ind_negdforw_end = int(np.where(t <= t1)[0][-1])

        if ((ind_negdforw_end - ind_negdforw_start) * dt) < thresh_dforw_dur_yes:   # check whether it is long enough, cri 2
            flag_trialcri_fail = 2
        else:
            ratio_dside_pos = 1. * np.sum(dside_filted[ind_negdforw_start:ind_negdforw_end] > 0) / len(
                dside_filted[ind_negdforw_start:ind_negdforw_end])
            if ratio_dside_pos > thresh_dside_ratio_uni:
                dside_pos = True
            elif ratio_dside_pos < (1 - thresh_dside_ratio_uni):
                dside_pos = False
            else:
                flag_trialcri_fail = 3

    t_bwwstart = t[ind_negdforw_start] if 'ind_negdforw_start' in locals() else 0
    return flag_trialcri_fail, dside_pos, t_bwwstart

def plot_bww_trace_alignbybeh(recss, twin=[-4,8], dt_sig=0.1, dt_beh=0.02, filter_window_s=0.1, thresh_dforw_yes=-6,
                              thresh_dforw_dur_yes=1, thresh_dforw_no=-0.5, thresh_dside_ratio_uni=0.8, vlines=[0,4],
                   ylims=[[-180,180],[-10,10],[-10,10]], yticks=[[-180,180],[-10,10],[-10,10]],
                   time_bar_length=2, time_text='2s', ylabels=['PFR-EPG','forw','side'],
                   titles=['rightward', 'leftward'], label_axeratio=-0.1, fs=10):
    # for OL0046B, natual avoiding response, implement criteria on beh
    ph.set_fontsize(fs)
    nrow, ncol, hr = 3, 2, [2,1,1]
    _ = plt.figure(1, (4, 5))
    gs = gridspec.GridSpec(nrow, ncol, height_ratios=hr)
    t_interp_sig = np.arange(twin[0] - dt_sig * 2, twin[1] + dt_sig * 2, dt_sig)
    t_interp_beh = np.arange(twin[0] - dt_beh * 2, twin[1] + dt_beh * 2, dt_beh)
    dphases, dforws, dsides = [[],[]], [[],[]], [[],[]]

    # extract data
    for ifly, recs in enumerate(recss):
        for rec in recs:
            dt = np.diff(rec.abf.tc).mean()
            sigma = filter_window_s / dt
            dforw = rec.abf.dforw
            dside = rec.abf.dside
            dforw_filted = ndimage.filters.gaussian_filter1d(dforw, sigma)
            dside_filted = ndimage.filters.gaussian_filter1d(dside, sigma)
            ts_start = rec.abf.t_orig[rec.abf.pwm_dc_idx_starts[rec.abf.pwm_dc_stimdcs > 0.05]]
            ts_end = rec.abf.t_orig[rec.abf.pwm_dc_idx_ends[rec.abf.pwm_dc_stimdcs > 0.05]]

            for i_trial in range(len(ts_start)):
                flag_trialcri_fail, dside_pos, t0_bww = \
                    get_trial_criterion_info_bww(rec, ts_start[i_trial], ts_end[i_trial], filter_window_s,
                                                 thresh_dforw_yes, thresh_dforw_dur_yes, thresh_dforw_no,
                                                 thresh_dside_ratio_uni)

                if not flag_trialcri_fail:  # one qualified trial
                    # t0 = ts_start[i_trial]    # align trial based on when laser turned on
                    t0 = t0_bww                 # align trial based on when bww start
                    direction = int(dside_pos)

                    # extract and save imaging sig
                    inds_sig = (rec.ims[0].t > (twin[0] + t0 - dt_sig * 10)) & (rec.ims[0].t < (twin[1] + t0 + dt_sig * 10))
                    t_im1, t_im0 = rec.ims[1].t[inds_sig] - t0, rec.ims[0].t[inds_sig] - t0
                    phase_im1, phase_im0 = rec.ims[1].c1.phase[inds_sig], rec.ims[0].c1.phase[inds_sig]
                    f_im1 = interp1d(t_im1, fc.unwrap(phase_im1), kind='linear')
                    f_im0 = interp1d(t_im0, fc.unwrap(phase_im0), kind='linear')
                    phase_interp_im1 = fc.wrap(f_im1(t_interp_sig))
                    phase_interp_im0 = fc.wrap(f_im0(t_interp_sig))
                    dphase = fc.circsubtract(phase_interp_im0, phase_interp_im1)
                    dphases[direction].append(dphase)

                    # extract and save behavior data
                    inds_beh = (rec.abf.t > (twin[0] + t0 - dt_beh * 10)) & (rec.abf.t < (twin[1] + t0 + dt_beh * 10))
                    t_beh = rec.abf.t[inds_beh] - t0
                    f_dforw = interp1d(t_beh, dforw_filted[inds_beh], kind='linear')
                    f_dside = interp1d(t_beh, dside_filted[inds_beh], kind='linear')
                    dforws[direction].append(f_dforw(t_interp_beh))
                    dsides[direction].append(f_dside(t_interp_beh))

    # plot
    xs = [t_interp_sig, t_interp_beh, t_interp_beh]
    for col in range(ncol):
        ys = [dphases[col], dforws[col], dsides[col]]
        for row in range(3):
            ax = plt.subplot(gs[row, col])
            for y in ys[row]:
                if row == 0:
                    # ph.plot_noline_between_lims(ax, xs[row], y, threshold=135, color='black', lw=1, zorder=2, alpha=0.3)
                    ax.plot(xs[row], y, ls = 'none', marker = 'o', ms = 2, mec = 'none', mfc = 'black', zorder=2, alpha=0.3)
                else:
                    ax.plot(xs[row], y, color='black', lw=1, zorder=2, alpha=0.3)
            if row == 0:
                # ph.plot_noline_between_lims(ax, xs[row], fc.circmean(ys[row], axis=0), threshold=135,
                #                             color='black', lw=2, zorder=3, alpha=1)
                ax.plot(xs[row], fc.circmean(ys[row], axis=0), ls='none', marker='o', ms=4, mec='none', mfc='black', zorder=2, alpha=0.9)
            else:
                ax.plot(xs[row], np.nanmean(ys[row], axis=0), color='black', lw=2, zorder=3, alpha=1)
            for vline in vlines:
                ax.axvline(vline, ls='--', lw=2, c=ph.dred, alpha=.6, zorder=1)
            ax.axhline(0, ls='--', lw=2, c=ph.dred, alpha=.6, zorder=1)

            # trim the plot
            show_axes = ['left'] if col == 0 else []
            ph.adjust_spines(ax, show_axes, lw=1, xlim=twin, ylim=ylims[row], xticks=[], yticks=yticks[row])
            if row == 0:
                ph.plot_x_scale_text(ax, bar_length=time_bar_length, text=time_text, y_text_lower=.1,
                                     x_text_lefter=.05)
                ax.set_title(titles[col], ha='center', va='top', fontsize=fs)
            if 'left' in show_axes:
                ax.set_ylabel(ylabels[row], rotation='vertical', va='center')
                ax.get_yaxis().set_label_coords(label_axeratio, 0.5)

def get_trial_criterion_info_bww_3dir(rec, t0, t1, filter_window_s=0.1, thresh_dforw_yes=-6, thresh_dforw_dur_yes=1,
                                      thresh_dforw_no=-0.5, thresh_dside_ratio_uni=0.8, thresh_bww_dev_angle=30):
    """check whether the trial between t0 and t1 meet the critera for backward walking"""
    dt = np.diff(rec.abf.tc).mean()
    sigma = filter_window_s / dt
    t = rec.abf.t

    dforw = rec.abf.dforw
    dside = rec.abf.dside

    dforw_filted = ndimage.filters.gaussian_filter1d(dforw, sigma)
    dside_filted = ndimage.filters.gaussian_filter1d(dside, sigma)

    flag_trialcri_fail = 0      # whether this trial meet the criteria for bww
    dside_pos = True            # direction of dside

    _inds_firstpeak = np.where((t >= t0) & (t <= t1) & (dforw_filted < thresh_dforw_yes))[0]
    if not len(_inds_firstpeak):        # check whether the bww is fast enough, cri 1
        flag_trialcri_fail = 1
    else:
        # go to the beginning and ending of bww, check whether it is long enough, cri 2
        ind_dforw_pass_thresh = _inds_firstpeak[0]
        _inds_pre = np.where((t >= t0) & (t < t[ind_dforw_pass_thresh]) & (dforw_filted >= thresh_dforw_no))[0]
        if len(_inds_pre):
            ind_negdforw_start = int(_inds_pre[-1])
        else:
            ind_negdforw_start = int(np.where(t >= t0)[0][0])

        _inds_post = np.where((t <= t1) & (t > t[ind_dforw_pass_thresh]) & (dforw_filted >= thresh_dforw_no))[0]
        if len(_inds_post):
            ind_negdforw_end = int(_inds_post[0])
        else:
            ind_negdforw_end = int(np.where(t <= t1)[0][-1])

        if ((ind_negdforw_end - ind_negdforw_start) * dt) < thresh_dforw_dur_yes:   # check whether it is long enough, cri 2
            flag_trialcri_fail = 2
        else:
            ratio_dside_pos = 1. * np.sum(dside_filted[ind_negdforw_start:ind_negdforw_end] > 0) / len(
                dside_filted[ind_negdforw_start:ind_negdforw_end])
            if ratio_dside_pos > thresh_dside_ratio_uni:
                dside_pos = True
            elif ratio_dside_pos < (1 - thresh_dside_ratio_uni):
                dside_pos = False
            else:
                flag_trialcri_fail = 3

    if flag_trialcri_fail == 0:     # calculated backward shift distance v.s. sideward shift distance
        d_back = np.abs(np.sum(dforw[ind_negdforw_start:ind_negdforw_end]))
        d_side = np.abs(np.sum(dside[ind_negdforw_start:ind_negdforw_end]))
        bww_dev_angle = np.arctan(d_side/d_back) * 180 / np.pi
        if bww_dev_angle < thresh_bww_dev_angle:
            dside_pos = 1
        else:
            dside_pos = int(dside_pos) * 2

    t_bwwstart = t[ind_negdforw_start] if 'ind_negdforw_start' in locals() else 0
    return flag_trialcri_fail, dside_pos, t_bwwstart

def plot_bww_trace_alignbybeh_3dir(recss, twin=[-4,8], dt_sig=0.1, dt_beh=0.02, filter_window_s=0.1, thresh_dforw_yes=-6,
                thresh_dforw_dur_yes=1, thresh_dforw_no=-0.5, thresh_dside_ratio_uni=0.8, thresh_bww_dev_angle=30,
                vlines=[0,4], ylims=[[-180,180],[-10,10],[-10,10]], yticks=[[-180,180],[-10,10],[-10,10]],
                   time_bar_length=2, time_text='2s', ylabels=['PFR-EPG','forw','side'],
                   titles=['leftward','backward','rightward'], label_axeratio=-0.1, fs=10):
    # for OL0046B, natual avoiding response, implement criteria on beh
    ph.set_fontsize(fs)
    nrow, ncol, hr = 3, 3, [2,1,1]
    _ = plt.figure(1, (6, 5))
    gs = gridspec.GridSpec(nrow, ncol, height_ratios=hr)
    t_interp_sig = np.arange(twin[0] - dt_sig * 2, twin[1] + dt_sig * 2, dt_sig)
    t_interp_beh = np.arange(twin[0] - dt_beh * 2, twin[1] + dt_beh * 2, dt_beh)
    dphases, dforws, dsides = [[],[],[]], [[],[],[]], [[],[],[]]

    # extract data
    for ifly, recs in enumerate(recss):
        for rec in recs:
            dt = np.diff(rec.abf.tc).mean()
            sigma = filter_window_s / dt
            dforw = rec.abf.dforw
            dside = rec.abf.dside
            dforw_filted = ndimage.filters.gaussian_filter1d(dforw, sigma)
            dside_filted = ndimage.filters.gaussian_filter1d(dside, sigma)
            ts_start = rec.abf.t_orig[rec.abf.pwm_dc_idx_starts[rec.abf.pwm_dc_stimdcs > 0.05]]
            ts_end = rec.abf.t_orig[rec.abf.pwm_dc_idx_ends[rec.abf.pwm_dc_stimdcs > 0.05]]

            for i_trial in range(len(ts_start)):
                flag_trialcri_fail, dside_pos, t0_bww = \
                    get_trial_criterion_info_bww_3dir(rec, ts_start[i_trial], ts_end[i_trial], filter_window_s,
                                                 thresh_dforw_yes, thresh_dforw_dur_yes, thresh_dforw_no,
                                                 thresh_dside_ratio_uni, thresh_bww_dev_angle)

                if not flag_trialcri_fail:  # one qualified trial
                    # t0 = ts_start[i_trial]    # align trial based on when laser turned on
                    t0 = t0_bww                 # align trial based on when bww start
                    direction = dside_pos       # 0: left, 1: bww, 2: right

                    # extract and save imaging sig
                    inds_sig = (rec.ims[0].t > (twin[0] + t0 - dt_sig * 10)) & (rec.ims[0].t < (twin[1] + t0 + dt_sig * 10))
                    t_im1, t_im0 = rec.ims[1].t[inds_sig] - t0, rec.ims[0].t[inds_sig] - t0
                    phase_im1, phase_im0 = rec.ims[1].c1.phase[inds_sig], rec.ims[0].c1.phase[inds_sig]
                    f_im1 = interp1d(t_im1, fc.unwrap(phase_im1), kind='linear')
                    f_im0 = interp1d(t_im0, fc.unwrap(phase_im0), kind='linear')
                    phase_interp_im1 = fc.wrap(f_im1(t_interp_sig))
                    phase_interp_im0 = fc.wrap(f_im0(t_interp_sig))
                    dphase = fc.circsubtract(phase_interp_im0, phase_interp_im1)
                    dphases[direction].append(dphase)

                    # extract and save behavior data
                    inds_beh = (rec.abf.t > (twin[0] + t0 - dt_beh * 10)) & (rec.abf.t < (twin[1] + t0 + dt_beh * 10))
                    t_beh = rec.abf.t[inds_beh] - t0
                    f_dforw = interp1d(t_beh, dforw_filted[inds_beh], kind='linear')
                    f_dside = interp1d(t_beh, dside_filted[inds_beh], kind='linear')
                    dforws[direction].append(f_dforw(t_interp_beh))
                    dsides[direction].append(f_dside(t_interp_beh))

    # plot
    xs = [t_interp_sig, t_interp_beh, t_interp_beh]
    for col in range(ncol):
        ys = [dphases[col], dforws[col], dsides[col]]
        for row in range(3):
            ax = plt.subplot(gs[row, col])
            for y in ys[row]:
                if row == 0:
                    # ph.plot_noline_between_lims(ax, xs[row], y, threshold=135, color='black', lw=1, zorder=2, alpha=0.3)
                    ax.plot(xs[row], y, ls = 'none', marker = 'o', ms = 2, mec = 'none', mfc = 'black', zorder=2, alpha=0.3)
                else:
                    ax.plot(xs[row], y, color='black', lw=1, zorder=2, alpha=0.3)
            if row == 0:
                # ph.plot_noline_between_lims(ax, xs[row], fc.circmean(ys[row], axis=0), threshold=135,
                #                             color='black', lw=2, zorder=3, alpha=1)
                ax.plot(xs[row], fc.circmean(ys[row], axis=0), ls='none', marker='o', ms=4, mec='none', mfc='black', zorder=2, alpha=0.9)
            else:
                ax.plot(xs[row], np.nanmean(ys[row], axis=0), color='black', lw=2, zorder=3, alpha=1)
            for vline in vlines:
                ax.axvline(vline, ls='--', lw=2, c=ph.dred, alpha=.6, zorder=1)
            ax.axhline(0, ls='--', lw=2, c=ph.dred, alpha=.6, zorder=1)

            # trim the plot
            show_axes = ['left'] if col == 0 else []
            ph.adjust_spines(ax, show_axes, lw=1, xlim=twin, ylim=ylims[row], xticks=[], yticks=yticks[row])
            if row == 0:
                ph.plot_x_scale_text(ax, bar_length=time_bar_length, text=time_text, y_text_lower=.1,
                                     x_text_lefter=.05)
                ax.set_title(titles[col], ha='center', va='top', fontsize=fs)
            if 'left' in show_axes:
                ax.set_ylabel(ylabels[row], rotation='vertical', va='center')
                ax.get_yaxis().set_label_coords(label_axeratio, 0.5)

def get_trial_criterion_info_bww_5dir(rec, t0, t1, filter_window_s=0.1, thresh_dforw_yes=-6, thresh_dforw_dur_yes=1,
                                      thresh_dforw_no=-0.5, thresh_dside_ratio_uni=0.8, thresh_bww_dev_angles=[30,60]):
    """check whether the trial between t0 and t1 meet the critera for backward walking"""
    dt = np.diff(rec.abf.tc).mean()
    sigma = filter_window_s / dt
    t = rec.abf.t

    dforw = rec.abf.dforw
    dside = rec.abf.dside

    dforw_filted = ndimage.filters.gaussian_filter1d(dforw, sigma)
    dside_filted = ndimage.filters.gaussian_filter1d(dside, sigma)

    flag_trialcri_fail = 0      # whether this trial meet the criteria for bww
    dside_pos = True            # direction of dside

    _inds_firstpeak = np.where((t >= t0) & (t <= t1) & (dforw_filted < thresh_dforw_yes))[0]
    if not len(_inds_firstpeak):        # check whether the bww is fast enough, cri 1
        flag_trialcri_fail = 1
    else:
        # go to the beginning and ending of bww, check whether it is long enough, cri 2
        ind_dforw_pass_thresh = _inds_firstpeak[0]
        _inds_pre = np.where((t >= t0) & (t < t[ind_dforw_pass_thresh]) & (dforw_filted >= thresh_dforw_no))[0]
        if len(_inds_pre):
            ind_negdforw_start = int(_inds_pre[-1])
        else:
            ind_negdforw_start = int(np.where(t >= t0)[0][0])

        _inds_post = np.where((t <= t1) & (t > t[ind_dforw_pass_thresh]) & (dforw_filted >= thresh_dforw_no))[0]
        if len(_inds_post):
            ind_negdforw_end = int(_inds_post[0])
        else:
            ind_negdforw_end = int(np.where(t <= t1)[0][-1])

        if ((ind_negdforw_end - ind_negdforw_start) * dt) < thresh_dforw_dur_yes:   # check whether it is long enough, cri 2
            flag_trialcri_fail = 2
        else:
            ratio_dside_pos = 1. * np.sum(dside_filted[ind_negdforw_start:ind_negdforw_end] > 0) / len(
                dside_filted[ind_negdforw_start:ind_negdforw_end])
            if ratio_dside_pos > thresh_dside_ratio_uni:
                dside_pos = True
            elif ratio_dside_pos < (1 - thresh_dside_ratio_uni):
                dside_pos = False
            else:
                flag_trialcri_fail = 3

    if flag_trialcri_fail == 0:     # calculated backward shift distance v.s. sideward shift distance
        d_back = np.abs(np.sum(dforw[ind_negdforw_start:ind_negdforw_end]))
        d_side = np.abs(np.sum(dside[ind_negdforw_start:ind_negdforw_end]))
        bww_dev_angle = np.arctan(d_side/d_back) * 180 / np.pi
        if bww_dev_angle < thresh_bww_dev_angles[0]:
            dside_pos = 2
        elif bww_dev_angle < thresh_bww_dev_angles[1]:
            dside_pos = int(dside_pos) * 2 + 1
        else:
            dside_pos = int(dside_pos) * 4
        if dside_pos == 2:
            print rec.abf.recid, t0
    t_bwwstart = t[ind_negdforw_start] if 'ind_negdforw_start' in locals() else 0
    return flag_trialcri_fail, dside_pos, t_bwwstart

def plot_bww_trace_alignbybeh_5dir(recss, twin=[-4,8], dt_sig=0.1, dt_beh=0.02, filter_window_s=0.1, thresh_dforw_yes=-6,
                thresh_dforw_dur_yes=1, thresh_dforw_no=-0.5, thresh_dside_ratio_uni=0.8, thresh_bww_dev_angles=[30,60],
                vlines=[0,4], ylims=[[-180,180],[-10,10],[-10,10]], yticks=[[-180,180],[-10,10],[-10,10]],
                   time_bar_length=2, time_text='2s', ylabels=['PFR-EPG','forw','side'],
                   titles=['-2','-1', '0', '1','2'], label_axeratio=-0.1, fs=10):
    # for OL0046B, natual avoiding response, implement criteria on beh
    ph.set_fontsize(fs)
    nrow, ncol, hr = 3, 5, [2,1,1]
    _ = plt.figure(1, (10, 5))
    gs = gridspec.GridSpec(nrow, ncol, height_ratios=hr)
    t_interp_sig = np.arange(twin[0] - dt_sig * 2, twin[1] + dt_sig * 2, dt_sig)
    t_interp_beh = np.arange(twin[0] - dt_beh * 2, twin[1] + dt_beh * 2, dt_beh)
    dphases, dforws, dsides = [[],[],[],[],[]], [[],[],[],[],[]], [[],[],[],[],[]]

    # extract data
    for ifly, recs in enumerate(recss):
        for rec in recs:
            dt = np.diff(rec.abf.tc).mean()
            sigma = filter_window_s / dt
            dforw = rec.abf.dforw
            dside = rec.abf.dside
            dforw_filted = ndimage.filters.gaussian_filter1d(dforw, sigma)
            dside_filted = ndimage.filters.gaussian_filter1d(dside, sigma)
            ts_start = rec.abf.t_orig[rec.abf.pwm_dc_idx_starts[rec.abf.pwm_dc_stimdcs > 0.05]]
            ts_end = rec.abf.t_orig[rec.abf.pwm_dc_idx_ends[rec.abf.pwm_dc_stimdcs > 0.05]]

            for i_trial in range(len(ts_start)):
                flag_trialcri_fail, dside_pos, t0_bww = \
                    get_trial_criterion_info_bww_5dir(rec, ts_start[i_trial], ts_end[i_trial], filter_window_s,
                                                 thresh_dforw_yes, thresh_dforw_dur_yes, thresh_dforw_no,
                                                 thresh_dside_ratio_uni, thresh_bww_dev_angles)

                if not flag_trialcri_fail:  # one qualified trial
                    # t0 = ts_start[i_trial]    # align trial based on when laser turned on
                    t0 = t0_bww                 # align trial based on when bww start
                    direction = dside_pos       # 0-1: left, 2: bww, 3-4: right

                    # extract and save imaging sig
                    inds_sig = (rec.ims[0].t > (twin[0] + t0 - dt_sig * 10)) & (rec.ims[0].t < (twin[1] + t0 + dt_sig * 10))
                    t_im1, t_im0 = rec.ims[1].t[inds_sig] - t0, rec.ims[0].t[inds_sig] - t0
                    phase_im1, phase_im0 = rec.ims[1].c1.phase[inds_sig], rec.ims[0].c1.phase[inds_sig]
                    f_im1 = interp1d(t_im1, fc.unwrap(phase_im1), kind='linear')
                    f_im0 = interp1d(t_im0, fc.unwrap(phase_im0), kind='linear')
                    phase_interp_im1 = fc.wrap(f_im1(t_interp_sig))
                    phase_interp_im0 = fc.wrap(f_im0(t_interp_sig))
                    dphase = fc.circsubtract(phase_interp_im0, phase_interp_im1)
                    dphases[direction].append(dphase)

                    # extract and save behavior data
                    inds_beh = (rec.abf.t > (twin[0] + t0 - dt_beh * 10)) & (rec.abf.t < (twin[1] + t0 + dt_beh * 10))
                    t_beh = rec.abf.t[inds_beh] - t0
                    f_dforw = interp1d(t_beh, dforw_filted[inds_beh], kind='linear')
                    f_dside = interp1d(t_beh, dside_filted[inds_beh], kind='linear')
                    dforws[direction].append(f_dforw(t_interp_beh))
                    dsides[direction].append(f_dside(t_interp_beh))

    # plot
    xs = [t_interp_sig, t_interp_beh, t_interp_beh]
    for col in range(ncol):
        ys = [dphases[col], dforws[col], dsides[col]]
        for row in range(3):
            ax = plt.subplot(gs[row, col])
            for y in ys[row]:
                if row == 0:
                    # ph.plot_noline_between_lims(ax, xs[row], y, threshold=135, color='black', lw=1, zorder=2, alpha=0.3)
                    ax.plot(xs[row], y, ls = 'none', marker = 'o', ms = 2, mec = 'none', mfc = 'black', zorder=2, alpha=0.3)
                else:
                    ax.plot(xs[row], y, color='black', lw=1, zorder=2, alpha=0.3)
            if row == 0:
                # ph.plot_noline_between_lims(ax, xs[row], fc.circmean(ys[row], axis=0), threshold=135,
                #                             color='black', lw=2, zorder=3, alpha=1)
                ax.plot(xs[row], fc.circmean(ys[row], axis=0), ls='none', marker='o', ms=4, mec='none', mfc='black', zorder=2, alpha=0.9)
            else:
                ax.plot(xs[row], np.nanmean(ys[row], axis=0), color='black', lw=2, zorder=3, alpha=1)
            for vline in vlines:
                ax.axvline(vline, ls='--', lw=2, c=ph.dred, alpha=.6, zorder=1)
            ax.axhline(0, ls='--', lw=2, c=ph.dred, alpha=.6, zorder=1)

            # trim the plot
            show_axes = ['left'] if col == 0 else []
            ph.adjust_spines(ax, show_axes, lw=1, xlim=twin, ylim=ylims[row], xticks=[], yticks=yticks[row])
            if row == 0:
                ph.plot_x_scale_text(ax, bar_length=time_bar_length, text=time_text, y_text_lower=.1,
                                     x_text_lefter=.05)
                ax.set_title(titles[col], ha='center', va='top', fontsize=fs)
            if 'left' in show_axes:
                ax.set_ylabel(ylabels[row], rotation='vertical', va='center')
                ax.get_yaxis().set_label_coords(label_axeratio, 0.5)


def plot_bww_trace_alignbybeh_2dir_paper_20200331(recss, axs=[], twin=[-4,8], dt_sig=0.1, dt_beh=0.02, filter_window_s=0.1, thresh_dforw_yes=-6,
                thresh_dforw_dur_yes=1, thresh_dforw_no=-0.5, thresh_dside_ratio_uni=0.8,
                vlines=[0,4], ylims=[[-20,15],[-20,20],[-182,182]], time_bar_length=2, fs=10,
                ms_indi=1.5, lw_indi=.5, lw_dash=.5, lw=.7, return_data=False):
    # for OL0046B, natual avoiding response, implement criteria on beh
    ncol, nrow = 2, 3
    ph.set_fontsize(fs)
    if not len(axs):
        _ = plt.figure(1, (2.67, 3))
        gs = gridspec.GridSpec(nrow, ncol)
        axs = [[plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])],
               [plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1])],
               [plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1])]]

    t_interp_sig = np.arange(twin[0] - dt_sig * 2, twin[1] + dt_sig * 2, dt_sig)
    t_interp_beh = np.arange(twin[0] - dt_beh * 2, twin[1] + dt_beh * 2, dt_beh)
    dphases, dforws, dsides = [[],[]], [[],[]], [[],[]]

    # extract data
    for ifly, recs in enumerate(recss):
        for rec in recs:
            dt = np.diff(rec.abf.tc).mean()
            sigma = filter_window_s / dt
            dforw = rec.abf.dforw
            dside = rec.abf.dside
            dforw_filted = ndimage.filters.gaussian_filter1d(dforw, sigma)
            dside_filted = ndimage.filters.gaussian_filter1d(dside, sigma)
            ts_start = rec.abf.t_orig[rec.abf.pwm_dc_idx_starts[rec.abf.pwm_dc_stimdcs > 0.05]]
            ts_end = rec.abf.t_orig[rec.abf.pwm_dc_idx_ends[rec.abf.pwm_dc_stimdcs > 0.05]]

            for i_trial in range(len(ts_start)):
                flag_trialcri_fail, dside_pos, t0_bww = \
                    get_trial_criterion_info_bww(rec, ts_start[i_trial], ts_end[i_trial], filter_window_s,
                                                 thresh_dforw_yes, thresh_dforw_dur_yes, thresh_dforw_no,
                                                 thresh_dside_ratio_uni)

                if not flag_trialcri_fail:  # one qualified trial
                    # t0 = ts_start[i_trial]    # align trial based on when laser turned on
                    t0 = t0_bww                 # align trial based on when bww start
                    direction = dside_pos       # 0: left, 1: right

                    # extract and save imaging sig
                    inds_sig = (rec.ims[0].t > (twin[0] + t0 - dt_sig * 10)) & (rec.ims[0].t < (twin[1] + t0 + dt_sig * 10))
                    t_im1, t_im0 = rec.ims[1].t[inds_sig] - t0, rec.ims[0].t[inds_sig] - t0
                    phase_im1, phase_im0 = rec.ims[1].c1.phase[inds_sig], rec.ims[0].c1.phase[inds_sig]
                    f_im1 = interp1d(t_im1, fc.unwrap(phase_im1), kind='linear')
                    f_im0 = interp1d(t_im0, fc.unwrap(phase_im0), kind='linear')
                    phase_interp_im1 = fc.wrap(f_im1(t_interp_sig))
                    phase_interp_im0 = fc.wrap(f_im0(t_interp_sig))
                    dphase = fc.circsubtract(phase_interp_im1, phase_interp_im0)
                    dphases[direction].append(dphase)

                    # extract and save behavior data
                    inds_beh = (rec.abf.t > (twin[0] + t0 - dt_beh * 10)) & (rec.abf.t < (twin[1] + t0 + dt_beh * 10))
                    t_beh = rec.abf.t[inds_beh] - t0
                    f_dforw = interp1d(t_beh, dforw_filted[inds_beh], kind='linear')
                    f_dside = interp1d(t_beh, dside_filted[inds_beh], kind='linear')
                    dforws[direction].append(f_dforw(t_interp_beh))
                    dsides[direction].append(f_dside(t_interp_beh))

    # plot
    xs = [t_interp_beh, t_interp_beh, t_interp_sig]
    for col in range(ncol):
        ys = [dforws[col], dsides[col], dphases[col]]
        for row in range(3):
            ax = axs[row][col]

            for y in ys[row]:       # individual traces
                if row == 2:
                    ax.plot(xs[row], y, ls='none', marker='o', ms=ms_indi, mec='none', mfc=ph.grey7, zorder=2, alpha=0.3)
                    # ax.plot(xs[row], y+360, ls='none', marker='o', ms=2, mec='none', mfc='black', zorder=2, alpha=0.3)
                else:
                    ax.plot(xs[row], y, color=ph.grey7, lw=lw_indi, zorder=2, alpha=0.3)

            if row == 2:            # population average traces
                # ax.plot(xs[row], fc.circmean(ys[row], axis=0), ls='none', marker='o', ms=1.5, mec='none', mfc='black', zorder=2, alpha=1)
                ax.plot(xs[row], fc.circmean(ys[row], axis=0), color='black', lw=lw, zorder=3, alpha=1)
            else:
                ax.plot(xs[row], np.nanmean(ys[row], axis=0), color='black', lw=lw, zorder=3, alpha=1)

            for vline in vlines:
                ax.axvline(vline, ls='--', lw=lw_dash, c=ph.grey9, alpha=.5, zorder=1)
            ax.axhline(0, ls='--', lw=lw_dash, c=ph.grey9, alpha=.5, zorder=1)

            # trim the plot
            ph.adjust_spines(ax, [], lw=.5, xlim=twin, ylim=ylims[row], xticks=[], yticks=[])
            if row == 0 and col == 0:
                ph.plot_x_scale_text(ax, bar_length=time_bar_length, text='%is' % time_bar_length, y_text_lower=.1,
                                     x_text_lefter=.05)

    if return_data:
        return t_interp_sig, dphases

def plot_sample_traj_n_bww_trace_alignbybeh_2dir_paper_20200331(recss, rec, fig_size=(6, 2.475), t0=1, t_win_sample=[-1.4,1.4,4],
            colors=[ph.blue, ph.dorange], cmaps=[plt.cm.Blues, plt.cm.Oranges], vm=[0,1.2], twin=[-4,8],
            filter_window_s=0.05, ylims=[[-20,15],[-20,20],[-182,182]], time_bar_length=2, fs=5,
            ms=1.5, ms_indi=.5, lw_indi=.5, lw_dash=.5, lw=.7, vlines=[0]):

    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(3, 5, width_ratios=[6,2,6,5,5])
    axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[1, 0]), plt.subplot(gs[2, 0]),
               plt.subplot(gs[0, 2]), plt.subplot(gs[1, 2]), plt.subplot(gs[2, 2])]
    plot_indi_trial_walking_paper_20200320(rec, axs, t0=t0, t_win=t_win_sample, vlines=[0], colors=colors, cmaps=cmaps,
                                           vm=vm, show_label=False, ylims=ylims, yticks=[[],[],[]], show_axes=[], fs=fs,
                                           ms=ms, lw_dash=lw_dash, lw=lw)

    axs = [[plt.subplot(gs[0, 3]), plt.subplot(gs[0, 4])],
           [plt.subplot(gs[1, 3]), plt.subplot(gs[1, 4])],
           [plt.subplot(gs[2, 3]), plt.subplot(gs[2, 4])]]
    plot_bww_trace_alignbybeh_2dir_paper_20200331(recss, axs, twin=twin, filter_window_s=filter_window_s,
                vlines=vlines, ylims=ylims, time_bar_length=time_bar_length, fs=8, ms_indi=ms_indi, lw_dash=lw_dash, lw=lw, lw_indi=lw_indi)

def plot_bww_trace_alignbybeh_3dir_paper_20200320(recss, axs=[], twin=[-4,8], dt_sig=0.1, dt_beh=0.02, filter_window_s=0.1, thresh_dforw_yes=-6,
                thresh_dforw_dur_yes=1, thresh_dforw_no=-0.5, thresh_dside_ratio_uni=0.8, thresh_bww_dev_angle=30,
                vlines=[0,4], ylims=[[-20,15],[-20,20],[-182,182]], time_bar_length=2, fs=10,
                ms_indi=1.5, lw_indi=.5, lw_dash=.5, lw=.7):
    # for OL0046B, natual avoiding response, implement criteria on beh
    ncol, nrow = 3, 3
    ph.set_fontsize(fs)
    if not len(axs):
        _ = plt.figure(1, (4, 3))
        gs = gridspec.GridSpec(3, 3)
        axs = [[plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1]), plt.subplot(gs[0, 2])],
               [plt.subplot(gs[1, 0]), plt.subplot(gs[1, 1]), plt.subplot(gs[1, 2])],
               [plt.subplot(gs[2, 0]), plt.subplot(gs[2, 1]), plt.subplot(gs[2, 2])]]

    t_interp_sig = np.arange(twin[0] - dt_sig * 2, twin[1] + dt_sig * 2, dt_sig)
    t_interp_beh = np.arange(twin[0] - dt_beh * 2, twin[1] + dt_beh * 2, dt_beh)
    dphases, dforws, dsides = [[],[],[]], [[],[],[]], [[],[],[]]

    # extract data
    for ifly, recs in enumerate(recss):
        for rec in recs:
            dt = np.diff(rec.abf.tc).mean()
            sigma = filter_window_s / dt
            dforw = rec.abf.dforw
            dside = rec.abf.dside
            dforw_filted = ndimage.filters.gaussian_filter1d(dforw, sigma)
            dside_filted = ndimage.filters.gaussian_filter1d(dside, sigma)
            ts_start = rec.abf.t_orig[rec.abf.pwm_dc_idx_starts[rec.abf.pwm_dc_stimdcs > 0.05]]
            ts_end = rec.abf.t_orig[rec.abf.pwm_dc_idx_ends[rec.abf.pwm_dc_stimdcs > 0.05]]

            for i_trial in range(len(ts_start)):
                flag_trialcri_fail, dside_pos, t0_bww = \
                    get_trial_criterion_info_bww_3dir(rec, ts_start[i_trial], ts_end[i_trial], filter_window_s,
                                                 thresh_dforw_yes, thresh_dforw_dur_yes, thresh_dforw_no,
                                                 thresh_dside_ratio_uni, thresh_bww_dev_angle)

                if not flag_trialcri_fail:  # one qualified trial
                    # t0 = ts_start[i_trial]    # align trial based on when laser turned on
                    t0 = t0_bww                 # align trial based on when bww start
                    direction = dside_pos       # 0: left, 1: bww, 2: right

                    # extract and save imaging sig
                    inds_sig = (rec.ims[0].t > (twin[0] + t0 - dt_sig * 10)) & (rec.ims[0].t < (twin[1] + t0 + dt_sig * 10))
                    t_im1, t_im0 = rec.ims[1].t[inds_sig] - t0, rec.ims[0].t[inds_sig] - t0
                    phase_im1, phase_im0 = rec.ims[1].c1.phase[inds_sig], rec.ims[0].c1.phase[inds_sig]
                    f_im1 = interp1d(t_im1, fc.unwrap(phase_im1), kind='linear')
                    f_im0 = interp1d(t_im0, fc.unwrap(phase_im0), kind='linear')
                    phase_interp_im1 = fc.wrap(f_im1(t_interp_sig))
                    phase_interp_im0 = fc.wrap(f_im0(t_interp_sig))
                    dphase = fc.circsubtract(phase_interp_im0, phase_interp_im1)
                    dphases[direction].append(dphase)

                    # extract and save behavior data
                    inds_beh = (rec.abf.t > (twin[0] + t0 - dt_beh * 10)) & (rec.abf.t < (twin[1] + t0 + dt_beh * 10))
                    t_beh = rec.abf.t[inds_beh] - t0
                    f_dforw = interp1d(t_beh, dforw_filted[inds_beh], kind='linear')
                    f_dside = interp1d(t_beh, dside_filted[inds_beh], kind='linear')
                    dforws[direction].append(f_dforw(t_interp_beh))
                    dsides[direction].append(f_dside(t_interp_beh))

    # plot
    xs = [t_interp_beh, t_interp_beh, t_interp_sig]
    for col in range(ncol):
        ys = [dforws[col], dsides[col], dphases[col]]
        for row in range(3):
            ax = axs[row][col]

            for y in ys[row]:       # individual traces
                if row == 2:
                    ax.plot(xs[row], y, ls='none', marker='o', ms=ms_indi, mec='none', mfc=ph.grey7, zorder=2, alpha=0.3)
                    # ax.plot(xs[row], y+360, ls='none', marker='o', ms=2, mec='none', mfc='black', zorder=2, alpha=0.3)
                else:
                    ax.plot(xs[row], y, color=ph.grey7, lw=lw_indi, zorder=2, alpha=0.3)

            if row == 2:            # population average traces
                # ax.plot(xs[row], fc.circmean(ys[row], axis=0), ls='none', marker='o', ms=1.5, mec='none', mfc='black', zorder=2, alpha=1)
                ax.plot(xs[row], fc.circmean(ys[row], axis=0), color='black', lw=lw, zorder=3, alpha=1)
            else:
                ax.plot(xs[row], np.nanmean(ys[row], axis=0), color='black', lw=lw, zorder=3, alpha=1)

            for vline in vlines:
                ax.axvline(vline, ls='--', lw=lw_dash, c=ph.grey9, alpha=.5, zorder=1)
            ax.axhline(0, ls='--', lw=lw_dash, c=ph.grey9, alpha=.5, zorder=1)

            # trim the plot
            ph.adjust_spines(ax, [], lw=.5, xlim=twin, ylim=ylims[row], xticks=[], yticks=[])
            if row == 0 and col == 0:
                ph.plot_x_scale_text(ax, bar_length=time_bar_length, text='%is' % time_bar_length, y_text_lower=.1,
                                     x_text_lefter=.05)

def plot_sample_traj_n_bww_trace_alignbybeh_3dir_paper_20200321(recss, rec, fig_size=(6, 2.475), t0=1, t_win_sample=[-1.4,1.4,4],
            colors=[ph.blue, ph.dorange], cmaps=[plt.cm.Blues, plt.cm.Oranges], vm=[0,1.2], twin=[-4,8],
            filter_window_s=0.05, ylims=[[-20,15],[-20,20],[-182,182]], time_bar_length=2, fs=5,
            ms=1.5, ms_indi=.5, lw_indi=.5, lw_dash=.5, lw=.7):

    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(3, 6, width_ratios=[6,2,6,5,5,5])
    axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[1, 0]), plt.subplot(gs[2, 0]),
               plt.subplot(gs[0, 2]), plt.subplot(gs[1, 2]), plt.subplot(gs[2, 2])]
    plot_indi_trial_walking_paper_20200320(rec, axs, t0=t0, t_win=t_win_sample, vlines=[0], colors=colors, cmaps=cmaps,
                                           vm=vm, show_label=False, ylims=ylims, yticks=[[],[],[]], show_axes=[], fs=fs,
                                           ms=ms, lw_dash=lw_dash, lw=lw)

    axs = [[plt.subplot(gs[0, 3]), plt.subplot(gs[0, 4]), plt.subplot(gs[0, 5])],
           [plt.subplot(gs[1, 3]), plt.subplot(gs[1, 4]), plt.subplot(gs[1, 5])],
           [plt.subplot(gs[2, 3]), plt.subplot(gs[2, 4]), plt.subplot(gs[2, 5])]]
    # plot_bww_trace_alignbybeh_3dir_paper_20200320(recss, axs, twin=twin, filter_window_s=filter_window_s,
    #             vlines=[0], ylims=ylims, time_bar_length=time_bar_length, fs=8, ms_indi=ms_indi, lw_dash=lw_dash, lw=lw, lw_indi=lw_indi)



def plot_bww_trace_alignbylight(recss, twin=[-4,8], dt_sig=0.1, dt_beh=0.02, vlines=[0,4], thresh_dforw_yes=-2,
                                thresh_dforw_no=-0.5, filter_window_s=0.1, ylims=[[-180,180],[-10,10],[-10,10]],
                                yticks=[[-180,180],[-10,10],[-10,10]], time_bar_length=2, time_text='2s',
                                ylabels=['PFR-EPG','forw','side'], label_axeratio=-0.1, fs=10):
    # for VT050660, motor bww, just align by light, just one filter: dforw
    ph.set_fontsize(fs)
    nrow, ncol, hr = 3, 1, [2,1,1]
    _ = plt.figure(1, (2, 5))
    gs = gridspec.GridSpec(nrow, ncol, height_ratios=hr)
    t_interp_sig = np.arange(twin[0] - dt_sig * 2, twin[1] + dt_sig * 2, dt_sig)
    t_interp_beh = np.arange(twin[0] - dt_beh * 2, twin[1] + dt_beh * 2, dt_beh)
    dphases, dforws, dsides = [], [], []

    # extract data
    for ifly, recs in enumerate(recss):
        for rec in recs:
            dt = np.diff(rec.abf.tc).mean()
            sigma = filter_window_s / dt
            t = rec.abf.t
            dforw = rec.abf.dforw
            dside = rec.abf.dside
            dforw_filted = ndimage.filters.gaussian_filter1d(dforw, sigma)
            dside_filted = ndimage.filters.gaussian_filter1d(dside, sigma)
            ts_start = rec.abf.t_orig[rec.abf.pwm_dc_idx_starts[rec.abf.pwm_dc_stimdcs > 0.05]]
            ts_end = rec.abf.t_orig[rec.abf.pwm_dc_idx_ends[rec.abf.pwm_dc_stimdcs > 0.05]]

            for i_trial in range(len(ts_start)):
                _inds_firstpeak = np.where((t >= ts_start[i_trial]) & (t <= ts_end[i_trial]) & (dforw_filted < thresh_dforw_yes))[0]
                if len(_inds_firstpeak):  # check whether the bww is fast enough
                    ind_dforw_pass_thresh = _inds_firstpeak[0]
                    _inds_pre = \
                    np.where((t >= ts_start[i_trial]) & (t < t[ind_dforw_pass_thresh]) & (dforw_filted >= thresh_dforw_no))[0]
                    if len(_inds_pre):
                        ind_negdforw_start = int(_inds_pre[-1])
                    else:
                        ind_negdforw_start = int(np.where(t >= t0)[0][0])
                    t0_bww = t[ind_negdforw_start]

                    # t0 = ts_start[i_trial]    # align trial based on when laser turned on
                    t0 = t0_bww                 # align trial based on when bww start

                    # extract and save imaging sig
                    inds_sig = (rec.ims[0].t > (twin[0] + t0 - dt_sig * 10)) & (rec.ims[0].t < (twin[1] + t0 + dt_sig * 10))
                    t_im1, t_im0 = rec.ims[1].t[inds_sig] - t0, rec.ims[0].t[inds_sig] - t0
                    phase_im1, phase_im0 = rec.ims[1].c1.phase[inds_sig], rec.ims[0].c1.phase[inds_sig]
                    f_im1 = interp1d(t_im1, fc.unwrap(phase_im1), kind='linear')
                    f_im0 = interp1d(t_im0, fc.unwrap(phase_im0), kind='linear')
                    phase_interp_im1 = fc.wrap(f_im1(t_interp_sig))
                    phase_interp_im0 = fc.wrap(f_im0(t_interp_sig))
                    dphase = fc.circsubtract(phase_interp_im0, phase_interp_im1)
                    dphases.append(dphase)

                    # extract and save behavior data
                    inds_beh = (rec.abf.t > (twin[0] + t0 - dt_beh * 10)) & (rec.abf.t < (twin[1] + t0 + dt_beh * 10))
                    t_beh = rec.abf.t[inds_beh] - t0
                    f_dforw = interp1d(t_beh, dforw_filted[inds_beh], kind='linear')
                    f_dside = interp1d(t_beh, dside_filted[inds_beh], kind='linear')
                    dforws.append(f_dforw(t_interp_beh))
                    dsides.append(f_dside(t_interp_beh))

    # plot
    xs = [t_interp_sig, t_interp_beh, t_interp_beh]
    ys = [dphases, dforws, dsides]
    for row in range(3):
        ax = plt.subplot(gs[row, 0])
        for y in ys[row]:
            if row == 0:
                ph.plot_noline_between_lims(ax, xs[row], y, threshold=135, color='black', lw=1, zorder=2, alpha=0.1)
            else:
                ax.plot(xs[row], y, color='black', lw=1, zorder=2, alpha=0.1)
        if row == 0:
            ph.plot_noline_between_lims(ax, xs[row], fc.circmean(ys[row], axis=0), threshold=135,
                                        color='black', lw=2, zorder=3, alpha=1)
        else:
            ax.plot(xs[row], np.nanmean(ys[row], axis=0), color='black', lw=2, zorder=3, alpha=1)
        for vline in vlines:
            ax.axvline(vline, ls='--', lw=2, c=ph.dred, alpha=.6, zorder=1)
        ax.axhline(0, ls='--', lw=2, c=ph.dred, alpha=.6, zorder=1)

        # trim the plot
        ph.adjust_spines(ax, ['left'], lw=1, xlim=twin, ylim=ylims[row], xticks=[], yticks=yticks[row])
        if row == 0:
            ph.plot_x_scale_text(ax, bar_length=time_bar_length, text=time_text, y_text_lower=.1,
                                 x_text_lefter=.05)
        ax.set_ylabel(ylabels[row], rotation='vertical', va='center')
        ax.get_yaxis().set_label_coords(label_axeratio, 0.5)


def plot_td_dltphase_xcorr_two_phase(recss, ax=False, fig_size=(2,2), twin=[-2,2], dt=0.1, t_behdelay_s=0.04,
                         xticks=[-2,0,2], ylim=[-.1,.4], yticks=[0,.2,.4], lw=2, lw_indi=1, alpha_indi=.3, fs=8):
    ph.set_fontsize(fs)
    if not ax:
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    t_interp = np.arange(twin[0] - dt * 2, twin[1] + dt * 2, dt)
    ys = []
    for recs in recss:
        ys_fly = []
        for rec in recs:
            # dside = rec.subsample('dside')
            tvd = slope_to_headning_angle(-rec.subsample('dside'), rec.subsample('dforw'))
            dltphase = fc.circsubtract(rec.eb.c1.phase, rec.fb.c1.phase)
            t, y = fc.xcorr(tvd, dltphase, rec.fb.sampling_period)
            f = interp1d(t, y, kind='linear')
            ys_fly.append(f(t_interp+t_behdelay_s))

        if len(ys_fly):
            ys.append(np.mean(ys_fly, axis=0))

    for y in ys:
        ax.plot(t_interp, y, lw=lw_indi, c='black', alpha=alpha_indi, zorder=2)
    ax.plot(t_interp, np.nanmean(ys, axis=0), lw=lw, c='black', zorder=3)
    ax.axvline(0, ls='--', lw=.5, c=ph.grey5, zorder=1)
    ph.adjust_spines(ax, ['left','bottom'], lw=.3, xlim=twin, ylim=ylim, xticks=xticks, yticks=yticks)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('xcorr', rotation=0)

def plot_td_dltphase_xcorr_two_phase1(recss, ax=False, fig_size=(2,2), twin=[-2,2], dt=0.1,
                         xticks=[-2,0,2], ylim=[-.1,.4], yticks=[0,.2,.4], lw=2, lw_indi=1, alpha_indi=.3, fs=8):
    # data points from interpolation
    ph.set_fontsize(fs)
    if not ax:
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    t_interp_xc = np.arange(twin[0] - dt * 2, twin[1] + dt * 2, dt)
    ys = []
    for recs in recss:
        ys_fly = []
        for rec in recs:
            td_bodycentric = slope_to_headning_angle(-rec.abf.dside, rec.abf.dforw)
            t_abf = rec.abf.t
            t_tif = rec.fb.t
            dltphase = fc.circsubtract(rec.eb.c1.phase, rec.fb.c1.phase)
            f_beh = interp1d(t_abf, td_bodycentric)
            t, y = fc.xcorr(f_beh(t_tif), dltphase, rec.fb.sampling_period)
            f = interp1d(t, y, kind='linear')
            ys_fly.append(f(t_interp_xc))

        if len(ys_fly):
            ys.append(np.mean(ys_fly, axis=0))

    for y in ys:
        ax.plot(t_interp_xc, y, lw=lw_indi, c='black', alpha=alpha_indi, zorder=2)
    ax.plot(t_interp_xc, np.nanmean(ys, axis=0), lw=lw, c='black', zorder=3)
    ax.axvline(0, ls='--', c=ph.grey5, zorder=1)
    ph.adjust_spines(ax, ['left','bottom'], lw=.3, xlim=twin, ylim=ylim, xticks=xticks, yticks=yticks)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('xcorr', rotation=0)

def plot_td_dltphase_xcorr_two_phase2(recss, ax=False, fig_size=(2,2), twin=[-2,2], dt=0.1, t_behdelay_s=0.04,
                         xticks=[-2,0,2], ylim=[-.1,.4], yticks=[0,.2,.4], lw=2, lw_indi=1, alpha_indi=.3, fs=8):
    # walking bouts only
    ph.set_fontsize(fs)
    if not ax:
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    t_interp = np.arange(twin[0] - dt * 2, twin[1] + dt * 2, dt)
    ys = []
    for recs in recss:
        ys_fly = []
        for rec in recs:
            # dside = rec.subsample('dside')
            tvd = slope_to_headning_angle(-rec.subsample('dside'), rec.subsample('dforw'))
            dltphase = fc.circsubtract(rec.eb.c1.phase, rec.fb.c1.phase)

            walking_bouts_idxs = get_walking_bouts_inds(rec, th_stand_dforw_ub=1, th_stand_dhead_ub=20,
                                                        th_stand_dside_ub=1, standing_dur=3, thresh_bouts_dur=10)
            for idx_ in walking_bouts_idxs:
                idxs = np.arange(idx_[0], idx_[-1] + 1).astype(int)
                t, y = fc.xcorr(tvd[idxs], dltphase[idxs], rec.fb.sampling_period)
                f = interp1d(t, y, kind='linear')
                ys_fly.append(f(t_interp+t_behdelay_s))

        if len(ys_fly):
            ys.append(np.mean(ys_fly, axis=0))

    for y in ys:
        ax.plot(t_interp, y, lw=lw_indi, c='black', alpha=alpha_indi, zorder=2)
    ax.plot(t_interp, np.nanmean(ys, axis=0), lw=lw, c='black', zorder=3)
    ax.axvline(0, ls='--', lw=.5, c=ph.grey5, zorder=1)
    ph.adjust_spines(ax, ['left','bottom'], lw=.3, xlim=twin, ylim=ylim, xticks=xticks, yticks=yticks)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('xcorr', rotation=0)

def plot_td_dltphase_ave_two_phase(recss, ax=False, fig_size=(2,2), binnum=20,
                         xlim=[-90,90], xticks=[-90,0,90], ylim=[-90,90], yticks=[-90,0,90], lw=2, lw_indi=1, alpha_indi=.3, fs=8):
    ph.set_fontsize(fs)
    if not ax:
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    ys = []
    for recs in recss:
        beh_fly = []
        sig_fly = []
        for rec in recs:
            tvd = slope_to_headning_angle(-rec.subsample('dside'), rec.subsample('dforw'))
            dltphase = fc.circsubtract(rec.eb.c1.phase, rec.fb.c1.phase)
            beh_fly.extend(tvd), sig_fly.extend(dltphase)
        x, y_mean, _ = fc.scatter_to_errorbar(x_input=beh_fly, y_input=sig_fly, binx=binnum, xmin=xlim[0], xmax=xlim[1])
        if len(y_mean):
            ys.append(y_mean)

    for y in ys:
        ax.plot(x, y, lw=lw_indi, c='black', alpha=alpha_indi, zorder=2)
    ax.plot(x, np.nanmean(ys, axis=0), lw=lw, c='black', zorder=3)
    ax.axvline(0, ls='--', c=ph.grey5, zorder=1)
    ph.adjust_spines(ax, ['left', 'bottom'], lw=.3, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)

    ax.set_xlabel('traveling direction (deg)')
    ax.set_ylabel('$\Delta$phase', rotation=0)





# flight single compass analysis plots
def get_stimbout_td_vs_h_flight(recss, sig_type='rmlz', stimid_special=1, stimid_N=4, t_win_plot=[-4,4,4], t_win_stop=[-8,8],
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
        elif sig_type == 'azl':
            _sig = rec.ims[loc_num].c1.az[:,0]
            # _sig = rec.ims[loc_num].c1.leftz
        elif sig_type == 'azr':
            _sig = rec.ims[loc_num].c1.az[:, 1]
            # _sig = rec.ims[loc_num].c1.rightz
        elif sig_type == 'peak_an':
            _sig = np.nanmax(rec.ims[loc_num].c1.an, axis=-1)
        elif sig_type == 'peak_az':
            _sig = np.nanmax(rec.ims[loc_num].c1.az, axis=-1)
        return _sig

    t_interp = np.arange(t_win_plot[0] - .5, (t_win_plot[1]+t_win_plot[2])+.5, 0.05)
    sig_group = []
    for fly in recss:
        sig_fly = []

        for rec in fly:
            if stim_method == 'dypos':
                of_stimids, of_start_idxs, _ = get_stim_info_20191007(rec, stimid_N=stimid_N)
            elif stim_method == 'dxpos':
                of_stimids, of_start_idxs, _ = get_stim_info_20200224_xposonly(rec)
            stop_inds = get_flight_stopping_inds(rec)
            ssr_tif = rec.ims[0].sampling_rate
            ssr_abf = rec.abf.subsampling_rate

            for start_idx in of_start_idxs[of_stimids==stimid_special]:
                start_idx_abf = np.where(rec.abf.t < rec.ims[0].t[start_idx])[0][-1]
                idx_stopwindow = [start_idx_abf - np.ceil(np.abs(t_win_stop[0])*ssr_abf),
                                  start_idx_abf + np.ceil(np.abs(t_win_stop[-1])*ssr_abf)]

                if not np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
                # if 1:
                    t0 = rec.abf.t[start_idx_abf]
                    idx1 = int(start_idx)
                    idx0 = int(idx1 - np.ceil((np.abs(t_win_plot[0])+1) * ssr_tif))
                    idx2 = int(idx1 + np.ceil(t_win_plot[1] * ssr_tif))
                    idx3 = int(idx2 + np.ceil((t_win_plot[2]+1) * ssr_tif))
                    t_tif = rec.ims[0].t[idx0:idx3] - t0
                    sig = get_sig(rec, loc_num, sig_type)[idx0:idx3]
                    # print stimid_special, idx0, idx3, t_tif
                    if len(t_tif):
                        _f = interp1d(t_tif, sig, kind='linear')
                        sig_interp = fc.wrap(_f(t_interp))
                        sig_fly.append(sig_interp)

        if not np.sum(np.isnan(np.nanmean(sig_fly, axis=0))):
            sig_group.append(np.nanmean(sig_fly, axis=0))

    return t_interp, sig_group

def plot_stimbout_td_vs_h_flight_unit(recss, ax=False, sig_type='rmlz', stimid_special=1, stimid_N=4, t_win_plot=[-4,4,4],
                                      t_win_stop=[-8,8], stim_method='dypos', loc_num=0, lw=2, lw_indi=1, alpha_indi=.4, alpha_sem=.2,
                                      color='black', xlabel='time (s)', ylabel='rmlz', show_axes=['left'],
                                      xticks=[-4,0,4,8], ylim=[-1,1], yticks=[-1,0,1], show_xscale=True, show_yscale=False,
                                      label_axeratio=-.25, fs=10, show_stimid_special=True, show_indi=True,
                                      show_flynum=True):

    if not ax:
        _ = plt.figure(1, (3, 2))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    xlim = [t_win_plot[0], t_win_plot[1]+t_win_plot[2]]
    xs, ys = get_stimbout_td_vs_h_flight(recss, sig_type=sig_type, stimid_special=stimid_special, stim_method=stim_method,
                                         stimid_N=stimid_N, t_win_plot=t_win_plot, t_win_stop=t_win_stop, loc_num=loc_num)
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

    color_dashline = ph.grey9
    ax.axvline(0, ls='--', lw=.5, c=color_dashline, alpha=.6, zorder=1)
    ax.axvline(t_win_plot[1], ls='--', lw=.4, c=color_dashline, alpha=.6, zorder=1)
    # ax.axvline(4, ls='--', lw=.5, c=color_dashline, alpha=.6, zorder=1)
    # ax.axvline(8, ls='--', lw=.5, c=color_dashline, alpha=.6, zorder=1)
    ax.axhline(0, ls='--', lw=.4, c=color_dashline, alpha=.6, zorder=1)
    if show_xscale:
        ph.plot_x_scale_text(ax, bar_length=2, text='2s', y_text_lower=0.15, x_text_lefter=.05, fontsize=fs)

    if show_yscale:
        ph.plot_y_scale_text(ax, bar_length=1, text='1', x_text_right=.1, y_text_upper=.05, color=color, fontsize=fs, horizontalalignment='right', verticalalignment='center')

    if show_stimid_special:
        ax.text(xlim[0], ylim[-1], 'id = %i' % stimid_special, fontsize=fs, va='top', ha='left')

    if show_flynum:
        ax.text(xlim[0], ylim[-1], '%i flies' % len(recss), fontsize=fs, va='top', ha='left')

def plot_stimbout_td_vs_h_flight_8dir_20191103(groups, sig_type='rmlz', stimids=[3,2,1,6,5,4,7,8],
            t_win_plot=[-4,4,4], t_win_stop=[-8,8], stim_method='dypos', loc_num=0, lw=2, lw_indi=.7, alpha_indi=.3, alpha_sem=.4,
            colors=['black'], ylabel='rmlz', show_axes=['left'], xlabels=['-120','-60','0','60','120','180','Lyaw','Ryaw'],
            xticks=[-4,0,4,8], ylim=[-1,1], yticks=[-1,0,1], show_xscale=True, show_yscale=True,
            label_axeratio=-.25, fs=10, show_stimid_special=True, show_indi=True, show_flynum=True):
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
            _show_yscale = show_yscale if col == 0 else False
            _show_flynum = show_flynum if col == 0 else False
            color = colors[row%nrow]
            xlabel = xlabels[col] if (row == nrow - 1) else ''
            plot_stimbout_td_vs_h_flight_unit(recss, ax, sig_type, stimid_special, t_win_plot, t_win_stop, stim_method,
                                              loc_num, lw, lw_indi, alpha_indi, alpha_sem, color, xlabel, _ylabel,
                                              _show_axes, xticks, ylim, yticks, _show_xscale, _show_yscale, label_axeratio, fs,
                                              show_stimid_special, show_indi, _show_flynum)

def get_ave_stimbout_td_vs_h_flight(recss, sig_type='rmlz', stimid_special=1, stimid_N=10, t_before_end=-2, t_win_stop=[-12,4],
                                    t_win_interp=[-3,1], loc_num=0, stim_method='dypos', location='fb'):

    def _cosine_func(x, amp, baseline, phase):
        return amp * np.sin(x + phase) + baseline

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

        elif sig_type == 'azl':
            _sig = rec.ims[loc_num].c1.az[:, 0]
            # _sig = rec.ims[loc_num].c1.leftz
        elif sig_type == 'azr':
            _sig = rec.ims[loc_num].c1.az[:, 1]
            # _sig = rec.ims[loc_num].c1.rightz
        elif sig_type == 'anl':
            _sig = rec.ims[loc_num].c1.an[:,0]
        elif sig_type == 'anr':
            _sig = rec.ims[loc_num].c1.an[:,1]
        elif sig_type == 'all':
            _sig = rec.ims[loc_num].c1.al[:,0]
        elif sig_type == 'alr':
            _sig = rec.ims[loc_num].c1.al[:,1]
        elif sig_type == 'c2.azl':
            _sig = rec.ims[loc_num].c2.az[:,0]
        elif sig_type == 'c2.azr':
            _sig = rec.ims[loc_num].c2.az[:,1]
        elif sig_type == 'amp_mmm':
            shapes_group_ = get_bolusshape_unit([[rec,],], stimid_special=stimid_special, stimid_N=stimid_N,
                    location=location, loc_num=loc_num, stim_method=stim_method, t_before_end=t_before_end, t_end=0,
                    t_win_stop=t_win_stop, offset=90)
            shape_ = shapes_group_[0]
            _sig = (np.nanmax(shape_) - np.nanmin(shape_))/2.
        elif sig_type == 'amp_fit':
            shapes_group_ = get_bolusshape_unit([[rec, ], ], stimid_special=stimid_special, stimid_N=stimid_N,
                                                location=location, loc_num=loc_num, stim_method=stim_method,
                                                t_before_end=t_before_end, t_end=0,
                                                t_win_stop=t_win_stop, offset=90)
            if location == 'fb':
                if len(shapes_group_):
                    shape_ = shapes_group_[0]
                    x_data = np.radians(np.linspace(0, 360, len(shape_)))
                    params, _ = optimize.curve_fit(_cosine_func, x_data, shape_, p0=[1, 1, 1])
                    _sig = np.abs(params[0])
                else:
                    _sig = np.nan
            elif location == 'pb':
                if len(shapes_group_):
                    len_shape = len(shapes_group_[0])
                    shape_ = shapes_group_[0][:(len_shape / 2)]
                    shape_ = shape_[~np.isnan(shape_)]
                    x_data = np.radians(np.linspace(0, 360, len(shape_)))
                    params, _ = optimize.curve_fit(_cosine_func, x_data, shape_, p0=[1, 1, 1])

                    _sig1 = np.abs(params[0])
                    shape_ = shapes_group_[0][(len_shape / 2):]
                    shape_ = shape_[~np.isnan(shape_)]
                    x_data = np.radians(np.linspace(0, 360, len(shape_)))
                    params, _ = optimize.curve_fit(_cosine_func, x_data, shape_, p0=[1, 1, 1])
                    _sig2 = np.abs(params[0])

                    _sig = (_sig1 + _sig2)/2.
                else:
                    _sig = np.nan

        return _sig

    t_interp = np.arange(t_before_end, 0, 0.1)
    sig_group = []
    for fly in recss:
        sig_fly = []

        for rec in fly:
            if stim_method == 'dypos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20191007(rec, stimid_N=stimid_N)
            elif stim_method == 'dxpos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20200224_xposonly(rec)

            stop_inds = get_flight_stopping_inds(rec)
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
                    if (sig_type == 'amp_mmm') or (sig_type == 'amp_fit'):
                        sig_fly.append(get_sig(rec, loc_num, sig_type))
                    else:
                        sig = get_sig(rec, loc_num, sig_type)[idx0:idx1]
                        _f = interp1d(t_tif, sig, kind='linear')
                        sig_interp = fc.wrap(_f(t_interp))
                        sig_fly.append(np.nanmean(sig_interp, axis=0))

        sig_group.append(np.nanmean(sig_fly, axis=0))

    return sig_group

def plot_ave_stimbout_td_vs_h_flight_20191105(recss, sig_type='rmlz', ylim=[-2.5,2.5], yticks=[-2,-1,0,1,2],
    stimids=[3, 2, 1, 6, 5, 4, 7, 8], t_before_end=-2, t_win_stop=[-12, 4], t_win_interp=[-3, 1], loc_num=0,
    group_labels=['-120','-60','0','60','120','180','Lyaw','Ryaw'], ylabel='R-L zscore', label_axeratio=-.25,
    c_marker=ph.grey5, fs=10):

    _ = plt.figure(1, (2.5, 2.5))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])

    xss = []
    for stimid_special in stimids:
        xs = get_ave_stimbout_td_vs_h_flight(recss, sig_type=sig_type, stimid_special=stimid_special,
                                             t_before_end=t_before_end, t_win_stop=t_win_stop,
                                             t_win_interp=t_win_interp, loc_num=loc_num)
        xss.append(xs)
    group_info = [[i] for i in range(len(stimids))]
    ph.errorbar_lc_xss(xss=xss, ax=ax, group_info=group_info, group_labels=group_labels,
                       colors=[c_marker],
                       ylabel=ylabel, yticks=yticks, ylim=ylim, yticks_minor=[], fs=fs,
                       ro_x=45, label_axeratio=label_axeratio, rotate_ylabel=False, plot_bar=False,
                       subgroup_samefly=False, circstat=True, legend_list=group_labels)

def plot_stim_trace_neural_corr_td_flight_paper_20200322(groups, nrow=3, ncol=9, fig_size=(4.5, 1.8), loc_num=0,
                t_win_plot=[-3,4,3], t_win_stop=[-5,8.5], stimids=[4,3,2,1,6,5,4,7,8], stimid_N=4, show_xscale=True, show_flynum=True,
                sig_type_='rmlz', lw_indi=0.15, alpha_indi=0.2, alpha_sem=0.2, lw=.6, colors=[ph.ngreen,ph.dred,ph.nblue],
                stim_method='dypos', ylim=[-3,3], fs=8):
    ph.set_fontsize(fs)
    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(nrow, ncol)
    if type(ylim[0]) is list:
        ylims = ylim
    else:
        ylims = []
        for i in range(nrow):
            ylims.append(ylim)

    for row in range(nrow):
        recss = groups[row]
        # trace plot
        for col in range(ncol):
            ax = plt.subplot(gs[row, col])
            stimid_special = stimids[col]
            _show_xscale = show_xscale if col == 0 and row == 0 else False
            _show_yscale = True if col == 0 else False
            _show_flynum = show_flynum if col == 0 else False
            plot_stimbout_td_vs_h_flight_unit(recss, ax, sig_type=sig_type_, stimid_special=stimid_special,
                                              t_win_plot=t_win_plot, stimid_N=stimid_N, loc_num=loc_num,
                                              t_win_stop=t_win_stop, stim_method=stim_method, lw=lw, lw_indi=lw_indi,
                                              alpha_indi=alpha_indi, alpha_sem=alpha_sem,
                                              color=colors[row], xlabel='', ylabel='', show_axes=[], xticks=[],
                                              ylim=ylims[row], yticks=[], show_xscale=_show_xscale, show_yscale=_show_yscale,
                                              fs=fs, show_stimid_special=False, show_indi=True,
                                              show_flynum=_show_flynum)

def scatter_stim_neural_corr_td_flight_paper_20200322(groups, axs=[], nrow=3, ncol=1, fig_size=(1.6,3), plottype='bar', data_xs=[],
                t_win_stop=[-5,8.5], stimids=[4,3,2,1,6,5,4,0,7,8], stimid_N=10, loc_num=0, stim_method='dypos', location='fb',
                sig_type_='rmlz', t_before_end=-2, t_win_interp=[-3, 1], colors=[ph.ngreen,ph.dred,ph.nblue],ylim=[-3,3],
                lw_indi=0.5, alpha_indi=0.5, lw=2, xlim=[-1,1], xticks=[], yticks=[],
                ms_indi=2, ms=5, fs=8, xlabel='', ylabel='', plot_cosfit=False, cfit=ph.grey3, alphafit=1,
                print_activity=False, print_phase=False):
    def _cosine_func(x, amp, baseline, phase):
        return amp * np.cos(x + phase) + baseline

    ph.set_fontsize(fs)
    if not len(axs):
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(nrow, ncol)

    if type(ylim[0]) is list:
        ylims = ylim
    else:
        ylims = []
        for i in range(nrow):
            ylims.append(ylim)

    for row in range(nrow):
        ax = plt.subplot(gs[row,0]) if not len(axs) else axs[row]
        recss = groups[row]

        xss = []
        for stimid_special in stimids:
            xs = get_ave_stimbout_td_vs_h_flight(recss, sig_type=sig_type_, stimid_special=stimid_special, stimid_N=stimid_N,
                                                 t_before_end=t_before_end, t_win_stop=t_win_stop, stim_method=stim_method,
                                                 t_win_interp=t_win_interp, loc_num=loc_num, location=location)
            xss.append(xs)
        # print np.nanmean(xss, axis=1)
        if plottype == 'bar':
            group_info = [[i] for i in range(len(stimids))]
            ph.errorbar_lc_xss(xss=xss, ax=ax, group_info=group_info, show_xlabel=False, colors=[colors[row]],
                               col_width=0.2, subgroup_gap=0.2, group_gap=1, plot_scale=1, margin_to_yaxis=.5,
                               figsize_height=5, ms_indi=2, alpha_indi=1,
                               noyaxis=True, ylabel='', yticks=[], ylim=ylims[row], yticks_minor=[], fs=fs,
                               ro_x=0, label_axeratio=-.1, rotate_ylabel=False, plot_bar=False,
                               subgroup_samefly=False, circstat=True, show_flynum=False, show_nancol=False)

            ph.plot_y_scale_text(ax, bar_length=1, text='1', x_text_right=.1, y_text_upper=.05, color=colors[row], fontsize=fs,
                                 horizontalalignment='right', verticalalignment='center')

            ax.axhline(0, ls='--', lw=.5, c=ph.grey9, alpha=.5, zorder=1)
        elif plottype == 'line':
            xss = np.array(xss)
            nstim = len(xss)
            nfly = len(xss[0])
            if len(data_xs) != nstim:
                data_xs = np.arange(nstim)
            # plot individual flies
            for ifly in range(nfly):
                ys = []
                for istim in range(nstim):
                    ys.append(xss[istim][ifly])
                ax.plot(data_xs, ys, c=colors[row], lw=lw_indi, alpha=alpha_indi)
            # plot mean
            ys = []
            for istim in range(nstim):
                ys.append(np.nanmean(xss[istim]))
            ax.plot(data_xs, ys, c=colors[row], lw=lw)

            ax.grid(which='major', alpha=0)
            ph.adjust_spines(ax, ['left', 'bottom'], xlim=xlim, xticks=xticks, ylim=ylims[row], yticks=yticks)
        elif plottype == 'scatter':
            xss = np.array(xss)
            nstim = len(xss)
            nfly = len(xss[0])
            if len(data_xs) != nstim:
                data_xs = np.arange(nstim)

            for istim in range(nstim):
                for ifly in range(nfly):
                    ax.plot(data_xs[istim], xss[istim][ifly], ls='none', marker='o', ms=ms_indi, mec=colors[row],
                            mew=.4, mfc='none', alpha=alpha_indi)
                ax.plot(data_xs[istim], np.nanmean(xss[istim]), ls='none', marker='o', ms=ms, mec=colors[row], mew=1,
                        mfc=colors[row], )

            ax.grid(which='major', alpha=0)
            ph.adjust_spines(ax, ['left', 'bottom'], xlim=xlim, xticks=xticks, ylim=ylims[row], yticks=yticks)

        if print_activity:
            ystr = ''
            for y in np.nanmean(xss, axis=1):
                ystr = ystr + str(y) + ', '
            print ystr

        if xlabel:
            ax.set_xlabel(xlabel, fontsize=fs)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=fs)

        if plot_cosfit:
            phases = []
            x_f = np.radians(np.linspace(0, 360, 7))
            xss_swapped = np.swapaxes(xss, 0, 1)
            for xs in xss_swapped:
                if np.sum(np.isnan(xs)):
                    continue
                params, _ = optimize.curve_fit(_cosine_func, x_f, xs, p0=[1, 1, 1])
                amp, baseline, phase = params
                phases.append(np.degrees(phase))
            params, _ = optimize.curve_fit(_cosine_func, x_f, np.nanmean(xss, axis=1), p0=[1, 1, 1])
            amp, baseline, phase = params
            ax.plot(np.linspace(0.6, 7.8, 80), _cosine_func(np.radians(np.linspace(0, 360, 80)), amp, baseline, phase),
                    c=cfit, alpha=alphafit, lw=1, zorder=1)

            if print_phase:
                phases = np.array(phases)
                phases[phases < 0] += 180
                print -phases
                print -np.nanmean(phases), stats.sem(phases)

def plot_boluswidth_unit(recss, stimid_special=1, location='eb', t_before_end=-6.51, t_end=-4.1, t_win_stop=[-2,13],
                         offset=0, stimid_N=10):
    # return xs
    width_group = []
    height_group = []

    for fly in recss:
        shapes_fly = []

        for rec in fly:
            of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20191007(rec, stimid_N=stimid_N)
            stop_inds = get_flight_stopping_inds(rec)
            ssr_tif = rec.ims[0].sampling_rate
            ssr_abf = rec.abf.subsampling_rate

            _, _, phase = rec.get_objects('%s.c1.phase' % location)
            im, ch, an_cubspl = rec.get_objects('%s.c1.an_cubspl' % location)
            an_nophase = im.cancel_phase(an_cubspl, phase, 10, ch.celltype, offset=offset)

            for end_idx in of_end_idxs[of_stimids==stimid_special]:
                end_idx_abf = np.where(rec.abf.t < rec.fb.t[end_idx])[0][-1]
                idx_stopwindow = [end_idx_abf - np.ceil(np.abs(t_win_stop[0])*ssr_abf),
                                  end_idx_abf + np.ceil(np.abs(t_win_stop[-1])*ssr_abf)]

                if not np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
                    t0 = rec.abf.t[end_idx_abf]
                    idx_end = int(end_idx)
                    idx1 = int(idx_end + np.ceil(t_end * ssr_tif))
                    idx0 = int(idx_end + np.ceil(t_before_end * ssr_tif))
                    shapes_fly.extend(an_nophase[idx0:idx1])

        if len(shapes_fly):
            shape_fly = np.nanmean(shapes_fly, axis=0)
            half_len = int(len(shape_fly) / 2)
            max_height = np.max(shape_fly)
            min_height = np.min(shape_fly)
            half_height = (max_height + min_height) / 2.0
            left_width = np.where(shape_fly[:half_len] >= half_height)[0][0] / 10.0 + .5
            right_width = (np.where(shape_fly[half_len:] <= half_height)[0][0] + half_len) / 10.0 + .5
            fwhm_wedge = right_width - left_width
            fwhm_deg = fwhm_wedge * 360 / 16
            width_group.append(fwhm_deg)
            height_group.append(max_height + min_height)

    return width_group, height_group

def plot_trial_compass_and_mean_1c(recss, stimid_special=1, t_win_plot=[-1,4,1],t_win_stop=[-5,8.5], ylabel='hDeltaK',
            vlines=[0,4], color=ph.magenta, cmap=ph.Magentas, vm=[], ms=4, ylim=[-.2, 2], lw_dash=1, fs=11, stimid_N=4,
            label_axeratio=-.1,):
    ph.set_fontsize(fs)
    nrow = len(recss)
    _ = plt.figure(1, (2.5*4,2.5*nrow))
    gs = gridspec.GridSpec(nrow*2, 4)

    for ifly, fly in enumerate(recss):
        irec = -1
        for rec in fly:
            of_stimids, of_start_idxs, _ = get_stim_info_20191007(rec, stimid_N=stimid_N)
            stop_inds = get_flight_stopping_inds(rec)
            ssr_abf = rec.abf.subsampling_rate
            for start_idx in of_start_idxs[of_stimids==stimid_special]:
                start_idx_abf = np.where(rec.abf.t < rec.ims[0].t[start_idx])[0][-1]
                idx_stopwindow = [start_idx_abf - np.ceil(np.abs(t_win_stop[0])*ssr_abf),
                                  start_idx_abf + np.ceil(np.abs(t_win_stop[-1])*ssr_abf)]
                if not np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
                    irec += 1
                    if irec == 4:
                        break
                    show_label_ = False if irec else True
                    # show_yscale_ = False if irec else True
                    axs = [plt.subplot(gs[ifly*2, irec]), plt.subplot(gs[ifly*2+1, irec])]
                    plot_indi_trial_unit_1c(rec, axs=axs, idx_start=start_idx, t_win_plot=t_win_plot, ylabel=ylabel, vlines=vlines,
                        color=color, cmap=cmap, vm=vm, ms=ms, ylim=ylim, lw_dash=lw_dash, label_axeratio=label_axeratio,
                        show_label=show_label_, show_recid=True, show_xscale=True, fs=fs, )



def plot_sig_ave_1ch_unit(recss, ax=False, locnum=0, chnum=1, stimid_special=1, t_win_plot=[-4,4,4],
                    vlines=[0,4], t_win_stop=[-8,8], stim_method='dypos', show_axes=['left'], xlabel='',
                    stimid_N=10, show_timescale=True, ylim=[-.2,1],
                    fs=11, alpha_indi=0.3, color=ph.grey9, lw_indi=1, lw=2, lw_dash=1, stopdetection=True,):
    if not ax:
        _ = plt.figure(1, (3, 2))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    t_interp = np.arange(t_win_plot[0] - .3, (t_win_plot[1]+t_win_plot[2])+.3, 0.1)
    xlim = [t_win_plot[0], t_win_plot[1]+t_win_plot[2]]
    ph.adjust_spines(ax, show_axes, lw=1, ylim=ylim, yticks=ylim, xlim=xlim,
                     xticks=[], pad=0)

    sig_group = []
    for fly in recss:
        sig_fly = []
        for rec in fly:
            if stim_method == 'dypos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20191007(rec, stimid_N=stimid_N)
            elif stim_method == 'pwm':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20200117_pwmdc(rec)
            elif stim_method == 'dxpos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20200224_xposonly(rec)
            elif stim_method == 'stimid':
                of_stimids, of_start_idxs, _ = get_stim_info_20210408_stimidonly(rec)
            stop_inds = get_flight_stopping_inds(rec)
            ssr_tif = rec.ims[0].sampling_rate
            ssr_abf = rec.abf.subsampling_rate
            ch = getattr(rec.ims[locnum], 'c%i' % chnum)

            for i_stim, start_idx in enumerate(of_start_idxs[of_stimids == stimid_special]):
                start_idx_abf = np.where(rec.abf.t < rec.ims[0].t[start_idx])[0][-1]

                # stop detection
                if stopdetection:
                    idx_stopwindow = [start_idx_abf - np.ceil(np.abs(t_win_stop[0])*ssr_abf),
                                      start_idx_abf + np.ceil(np.abs(t_win_stop[-1])*ssr_abf)]
                    if np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
                        continue

                t0 = rec.abf.t[start_idx_abf]
                idx1 = int(start_idx)
                # idx0 = int(idx1 - np.ceil((np.abs(t_win_plot[0])+1) * ssr_tif))
                idx0 = int(idx1 + np.ceil((t_win_plot[0]-1) * ssr_tif))
                idx2 = int(idx1 + np.ceil(t_win_plot[1] * ssr_tif))
                idx3 = int(idx2 + np.ceil((t_win_plot[2]+1) * ssr_tif))

                t_sig = rec.ims[locnum].t[idx0:idx3] - t0
                if (t_sig[0] > t_interp[0]) | (t_sig[-1] < t_interp[-1]):
                    continue
                sig_orig = ch.mean_an[idx0:idx3]
                f_sig = interp1d(t_sig, sig_orig, kind='linear')
                sig_interp = fc.wrap(f_sig(t_interp))
                sig_fly.append(sig_interp)

        if not np.sum(np.isnan(np.nanmean(sig_fly, axis=0))):
            sig_group.append(np.nanmean(sig_fly, axis=0))

        if len(sig_fly):
            ax.plot(t_interp, np.nanmean(sig_fly, axis=0), lw=lw_indi, zorder=2, alpha=alpha_indi, c=color)

    ax.plot(t_interp, np.nanmean(sig_group, axis=0), lw=lw, zorder=3, c=color)

    for vline in vlines:
        ax.axvline(vline, ls='--', lw=lw_dash, c=ph.grey9, alpha=.5)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fs)
    if show_timescale:
        ph.plot_x_scale_text(ax, bar_length=2, text='2s', y_text_lower=0.15, x_text_lefter=.05, fontsize=fs)

def plot_1c_20210422(group, rec, fig_size=(12, 3.3), ncol=6, stimids=[4,3.75,3.5,3.25,4.5,4.25,4], stim_method='dypos',
            stimid_N=4, trial_id=[1,1,1,1,1,1], t_win_plot=[-1,4,1], t_win_stop=[-5,8.5],vlines=[0,4], locnum=0, chnum=1,
            color=ph.magenta, cmap=ph.Magentas, vm=[], ms=4, ylim=[-.2, 2],
            stopdetection=True, lw_indi=0.5, alpha_indi=0.4, lw_dash=1, lw=2, fs=11,):

    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(3, ncol)

    if stim_method == 'dypos':
        of_stimids, of_start_idxs, _ = get_stim_info_20191007(rec, stimid_N=stimid_N)
    elif stim_method == 'pwm':
        of_stimids, of_start_idxs, _ = get_stim_info_20200117_pwmdc(rec)
    elif stim_method == 'dxpos':
        of_stimids, of_start_idxs, _ = get_stim_info_20200224_xposonly(rec)
    elif stim_method == 'stimid':
        of_stimids, of_start_idxs, _ = get_stim_info_20210408_stimidonly(rec)

    for col in range(ncol):
        # plot sample trace
        axs = [plt.subplot(gs[0, col]), plt.subplot(gs[1, col])]
        stimid_special = stimids[col]
        start_idx = of_start_idxs[of_stimids==stimid_special][trial_id[col]]

        _show_xscale = True if col == 0 else False
        show_axes = ['left'] if col == 0 else []
        plot_indi_trial_unit_1c(rec, axs=axs, idx_start=start_idx, t_win_plot=t_win_plot, ylabel='celltype', vlines=vlines,
                                color=color, cmap=cmap, vm=vm, ms=ms, ylim=ylim, show_axes=show_axes,
                                lw_dash=lw_dash, fs=fs, show_label=False, show_recid=False,
                                show_xscale=_show_xscale, )

        # plot population average
        ax = plt.subplot(gs[2, col])
        plot_sig_ave_1ch_unit(group, ax=ax, locnum=locnum, chnum=chnum, stimid_special=stimid_special, t_win_plot=t_win_plot, vlines=vlines,
                             t_win_stop=t_win_stop, stim_method=stim_method, show_axes=show_axes, xlabel='',
                             show_timescale=_show_xscale, lw_indi=lw_indi, alpha_indi=alpha_indi, lw=lw, stimid_N=stimid_N,
                             ylim=ylim, color=color, fs=fs, lw_dash=lw_dash, stopdetection=stopdetection)



# bolus shape related
def barplot_boluswidth(recss, fig_size=(2, 3.3), stimids=[4,3,2,1,6,5,4,7,8], plot_width=True, location='eb', t_before_end=-2, t_end=0,
                       t_win_stop=[-12,4], offset=0, label_axeratio=-.2, fs=11, stimid_N=10, yticks=[0,45,90,135,180,225], ylim=[0,225],
                       c=ph.blue, group_labels=['-180','-120','-60','0','60','120','180','Lyaw','Ryaw'],):
    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])

    xss_w, xss_h = [], []
    for stimid_special in stimids:
        xs_w, xs_h = plot_boluswidth_unit(recss, stimid_special=stimid_special, location=location, t_before_end=t_before_end,
                                  t_end=t_end, t_win_stop=t_win_stop, offset=offset, stimid_N=stimid_N)
        xss_w.append(xs_w)
        xss_h.append(xs_h)
    group_info = [[i] for i in range(len(stimids))]
    xss = xss_w if plot_width else xss_h
    ph.errorbar_lc_xss(xss=xss, ax=ax, group_info=group_info, group_labels=group_labels, colors=[c],
                       col_width=0.2, subgroup_gap=0.2, group_gap=1, plot_scale=1, margin_to_yaxis=.5,
                       figsize_height=5, ms_indi=2, alpha_indi=0.4,
                       ylabel='', yticks=yticks, ylim=ylim, yticks_minor=[], fs=fs,
                       ro_x=45, label_axeratio=label_axeratio, rotate_ylabel=False, plot_bar=False, circ_low=0, circ_high=360,
                       subgroup_samefly=False, circstat=True, legend_list=group_labels, show_flynum=False, show_nancol=False)


def get_bolusshape_unit(recss, stim_method='dypos', stimid_special=1, location='eb', loc_num=None, stimid_N=10,
                         t_before_end=-6.51, t_end=-4.1, t_win_stop=[-5,1], offset=0, thresh_dphase=100, flight_detection=True):
    # return xs
    shapes_group = []

    for fly in recss:
        shapes_fly = []

        for rec in fly:
            if stim_method == 'dypos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20191007(rec, stimid_N=stimid_N)
            elif stim_method == 'dxpos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20200224_xposonly(rec)

            stop_inds = get_flight_stopping_inds(rec)
            ssr_tif = rec.ims[0].sampling_rate
            ssr_abf = rec.abf.subsampling_rate

            if loc_num is None:
                _, _, phase = rec.get_objects('%s.c1.phase' % location)
                im, ch, an_cubspl = rec.get_objects('%s.c1.an_cubspl' % location)
            else:
                phase = rec.ims[loc_num].c1.phase
                im = rec.ims[loc_num]
                an_cubspl = im.c1.an_cubspl
            an_nophase = im.cancel_phase(an_cubspl, phase, 10, im.c1.celltype, offset=offset)
            # for end_idx in of_end_idxs[of_stimids==stimid_special]:
            #     end_idx_abf = np.where(rec.abf.t < rec.fb.t[end_idx])[0][-1]
            #     idx_stopwindow = [end_idx_abf - np.ceil(np.abs(t_win_stop[0])*ssr_abf),
            #                       end_idx_abf + np.ceil(np.abs(t_win_stop[-1])*ssr_abf)]

            for i_idx, start_idx in enumerate(of_start_idxs[of_stimids == stimid_special]):
                start_idx_abf = np.where(rec.abf.t < rec.ims[0].t[start_idx])[0][-1]
                idx_stopwindow = [start_idx_abf - np.ceil(np.abs(t_win_stop[0]) * ssr_abf),
                                  start_idx_abf + np.ceil(np.abs(t_win_stop[-1]) * ssr_abf)]

                flight_flag = (not np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1]))) if flight_detection else 1
                # if not np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
                if flight_flag:
                    end_idx = of_end_idxs[of_stimids==stimid_special][i_idx]
                    # t0 = rec.abf.t[end_idx_abf]
                    idx_end = int(end_idx)
                    idx1 = int(idx_end + np.ceil(t_end * ssr_tif))
                    idx0 = int(idx_end + np.ceil(t_before_end * ssr_tif))
                    # shapes_fly.extend(an_nophase[idx0:idx1])
                    dphase = im.c1.dphase[idx0:idx1]
                    shapes_fly.extend(an_nophase[idx0:idx1][np.abs(dphase)<thresh_dphase])

        if len(shapes_fly):
            shape_fly = np.nanmean(shapes_fly, axis=0)
            shapes_group.append(shape_fly)

    return shapes_group

def barplot_bolusshape_interpolation(recss, fig_size=(20,3), stimids=[4,3,2,1,6,5,4,7,8], stimid_N=10, location='eb', loc_num=None,
                       stim_method='dypos', t_before_end=-2, t_end=0, thresh_dphase=1000,
                       t_win_stop=[-12,4], offset=0, c='black', ylim=[0,3], fs=8, lw=1,):
    def _plot_shape(show_yscale=True):
        xs = np.linspace(-180, 180, len(shapes_group[0]))
        ymean = np.nanmean(shapes_group, axis=0)
        ysem = fc.nansem(shapes_group, axis=0)
        ax.plot(xs, ymean, lw=lw, c=c)
        ax.fill_between(xs, ymean - ysem, ymean + ysem, facecolor=c, edgecolor='none', alpha=.2)
        ph.adjust_spines(ax, ['bottom'], lw=1, ylim=ylim, yticks=[], xlim=[-180,180], xticks=[], pad=0)
        if show_yscale:
            ph.plot_y_scale_text(ax, bar_length=1, text='1', x_text_right=.1, y_text_upper=.05, color=c,
                                 fontsize=fs, horizontalalignment='right', verticalalignment='center')
        # ax.set_title(group_labels[col])

    ncol = len(stimids)
    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(1, ncol)

    for col, stimid_special in enumerate(stimids):
        ax = plt.subplot(gs[0, col])
        shapes_group = get_bolusshape_unit(recss, stimid_special=stimid_special, stimid_N=stimid_N,
                                            location=location, loc_num=loc_num, thresh_dphase=thresh_dphase,
                                            stim_method=stim_method, t_before_end=t_before_end, t_end=t_end,
                                            t_win_stop=t_win_stop, offset=offset)
        show_yscale = True if col == 0 else False
        _plot_shape(show_yscale)

def barplot_bolusshape_subsampled(recss, fig_size=(20,3), stimids=[4,3,2,1,6,5,4,7,8], stimid_N=10, location='eb', loc_num=None,
                       stim_method='dypos', t_before_end=-2, t_end=0, thresh_dphase=1000, cfit=ph.grey7, plot_fit=True,
                       t_win_stop=[-12,4], offset=0, c='black', ylim=[0,3], fs=8, lw=1, lw_indi=.5, alpha_indi=.5, show_chisquare=False):
    flies = []
    chisquares = []

    def _cosine_func(x, baseline, amp, phase):
        return amp * np.sin(x - phase) + baseline

    def _plot_shape(show_yscale=True):
        # subsample back to column unit
        _, _, x = recss[0][0].get_objects('%s.c%i.xedges_interp' % (location, 1))
        idx = np.arange(len(x)).reshape((len(x) / 10, 10))
        shapes_sub = []
        for shape_fly in shapes_group:
            shapes_sub.append(np.nanmean(shape_fly[idx], axis=-1))

        nglom = len(shapes_sub[0])
        xs_ = np.linspace(-180, 180, nglom+1)
        xs = (xs_[1:]+xs_[:-1])/2.

        ymean = np.nanmean(shapes_sub, axis=0)
        flies.append(ymean)
        ysem = fc.nansem(shapes_sub, axis=0)
        # for y in shapes_sub:
        #     ax.plot(xs, y, lw=lw_indi, c=c, alpha=alpha_indi)
        ax.plot(xs, ymean, lw=lw, c=c)
        ax.fill_between(xs, ymean - ysem, ymean + ysem, facecolor=c, edgecolor='none', alpha=.2)
        ph.adjust_spines(ax, ['bottom'], lw=1, ylim=ylim, yticks=[], xlim=[-180,180], xticks=[], pad=0)
        if show_yscale:
            ph.plot_y_scale_text(ax, bar_length=1, text='1', x_text_right=.1, y_text_upper=.05, color=c,
                                 fontsize=fs, horizontalalignment='right', verticalalignment='center')
        # ax.set_title(group_labels[col])

        if plot_fit:  # fit and plot
            if location == 'pb':
                len_fullshape = len(ymean)
                xsfit_ = np.linspace(-180, 180, 8 + 1)
                xsfit = (xsfit_[1:] + xsfit_[:-1]) / 2.

                shape_ = ymean[:(len_fullshape / 2)]
                shape_ = shape_[~np.isnan(shape_)]
                x_f = np.radians(xsfit)
                params, _ = optimize.curve_fit(_cosine_func, x_f, shape_, p0=[1, 1, 1])
                baseline, amp, phase = params
                xs_fitplot = xs[:(len_fullshape/2-1)]
                ax.plot(xs_fitplot, _cosine_func(np.radians(xsfit), baseline, amp, phase), c=cfit, lw=lw, zorder=1)
                chisquares.append(chisquare(shape_, _cosine_func(np.radians(xsfit), baseline, amp, phase))[1])

                shape_ = ymean[(len_fullshape/2):]
                shape_ = shape_[~np.isnan(shape_)]
                x_f = np.radians(xsfit)
                params, _ = optimize.curve_fit(_cosine_func, x_f, shape_, p0=[1, 1, 1])
                baseline, amp, phase = params
                xs_fitplot = xs[(len_fullshape / 2+1):]
                ax.plot(xs_fitplot, _cosine_func(np.radians(xsfit), baseline, amp, phase), c=cfit, lw=lw, zorder=1)
                chisquares.append(chisquare(shape_, _cosine_func(np.radians(xsfit), baseline, amp, phase))[1])

    ncol = len(stimids)
    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(1, ncol)

    for col, stimid_special in enumerate(stimids):
        ax = plt.subplot(gs[0, col])
        shapes_group = get_bolusshape_unit(recss, stimid_special=stimid_special, stimid_N=stimid_N,
                                            location=location, loc_num=loc_num, thresh_dphase=thresh_dphase,
                                            stim_method=stim_method, t_before_end=t_before_end, t_end=t_end,
                                            t_win_stop=t_win_stop, offset=offset)
        show_yscale = True if col == 0 else False
        _plot_shape(show_yscale)

    if show_chisquare:
        print 'chi square p values:'
        ystr = ''
        for y in chisquares:
            ystr = ystr + '%.8f' % y + ', '
        print ystr

def barplot_bolusshape_subsampled_chisquare(recss, fig_size=(20,3), stimids=[4,3,2,1,6,5,4,7,8], stimid_N=10, location='eb', loc_num=None,
                       stim_method='dypos', t_before_end=-2, t_end=0, thresh_dphase=1000, cfit=ph.grey7, plot_fit=True,
                       t_win_stop=[-12,4], offset=0, c='black', ylim=[0,3], fs=8, lw=1, lw_indi=.5, alpha_indi=.5, show_chisquare=False):
    flies = []
    chisquares = []
    pvalues = []

    def _cosine_func(x, baseline, amp, phase):
        return amp * np.sin(x - phase) + baseline
    def my_chisquare(p, data, var, angles):
        return np.sum((data - (p[0] * np.sin(np.radians(angles + p[1])) + p[2])) ** 2 / var)
    def my_curve(p, angles):
        return p[0] * np.sin(np.radians(angles + p[1])) + p[2]

    def _plot_shape(show_yscale=True):
        # subsample back to column unit
        _, _, x = recss[0][0].get_objects('%s.c%i.xedges_interp' % (location, 1))
        idx = np.arange(len(x)).reshape((len(x) / 10, 10))
        shapes_sub = []
        for shape_fly in shapes_group:
            shapes_sub.append(np.nanmean(shape_fly[idx], axis=-1))

        nglom = len(shapes_sub[0])
        xs_ = np.linspace(-180, 180, nglom+1)
        xs = (xs_[1:]+xs_[:-1])/2.

        ymean = np.nanmean(shapes_sub, axis=0)
        var = np.var(shapes_sub, axis=0)
        flies.append(ymean)
        ysem = fc.nansem(shapes_sub, axis=0)
        # for y in shapes_sub:
        #     ax.plot(xs, y, lw=lw_indi, c=c, alpha=alpha_indi)
        ax.plot(xs, ymean, lw=lw, c=c)
        ax.fill_between(xs, ymean - ysem, ymean + ysem, facecolor=c, edgecolor='none', alpha=.2)
        ph.adjust_spines(ax, ['bottom'], lw=1, ylim=ylim, yticks=[], xlim=[-180,180], xticks=[], pad=0)
        if show_yscale:
            ph.plot_y_scale_text(ax, bar_length=1, text='1', x_text_right=.1, y_text_upper=.05, color=c,
                                 fontsize=fs, horizontalalignment='right', verticalalignment='center')
        # ax.set_title(group_labels[col])

        dof = 8 - 3
        if plot_fit:  # fit and plot
            if location == 'pb':
                len_fullshape = len(ymean)
                xsfit_ = np.linspace(-180, 180, 8 + 1)
                xsfit = (xsfit_[1:] + xsfit_[:-1]) / 2.

                shape_ = ymean[:(len_fullshape / 2)]
                shape_ = shape_[~np.isnan(shape_)]
                var_ = copy.copy(var[:(len_fullshape / 2)])
                var_ = var_[~np.isnan(var_)]
                p = optimize.fmin(my_chisquare, [1, 1, 1], args=(shape_, var_, xsfit), disp=False)
                xs_fitplot = xs[:(len_fullshape/2-1)]
                ax.plot(xs_fitplot, my_curve(p, xsfit), c=cfit, lw=lw, zorder=1)
                chisquares.append(my_chisquare(p, shape_, var_, xsfit) / dof)
                pvalues.append(stats.distributions.chi2.sf(my_chisquare(p, shape_, var_, xsfit), dof))

                shape_ = ymean[(len_fullshape/2):]
                shape_ = shape_[~np.isnan(shape_)]
                var_ = copy.copy(var[(len_fullshape/2):])
                var_ = var_[~np.isnan(var_)]
                p = optimize.fmin(my_chisquare, [1, 1, 1], args=(shape_, var_, xsfit), disp=False)
                xs_fitplot = xs[(len_fullshape / 2+1):]
                ax.plot(xs_fitplot, my_curve(p, xsfit), c=cfit, lw=lw, zorder=1)
                chisquares.append(my_chisquare(p, shape_, var_, xsfit) / dof)
                pvalues.append(stats.distributions.chi2.sf(my_chisquare(p, shape_, var_, xsfit), dof))

    ncol = len(stimids)
    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(1, ncol)

    for col, stimid_special in enumerate(stimids):
        ax = plt.subplot(gs[0, col])
        shapes_group = get_bolusshape_unit(recss, stimid_special=stimid_special, stimid_N=stimid_N,
                                            location=location, loc_num=loc_num, thresh_dphase=thresh_dphase,
                                            stim_method=stim_method, t_before_end=t_before_end, t_end=t_end,
                                            t_win_stop=t_win_stop, offset=offset)
        show_yscale = True if col == 0 else False
        _plot_shape(show_yscale)

    if show_chisquare:
        print 'chi square:'
        ystr = ''
        for y in chisquares:
            ystr = ystr + '%.8f' % y + ', '
        print ystr
        print 'p values:'
        ystr = ''
        for y in pvalues:
            ystr = ystr + '%.8f' % y + ', '
        print ystr


    # return flies

def barplot_bolusshape_stimids_sameplot(recss, ax=False, fig_size=(3.15,3), stimids=[4,3,2,1,6,5,4,7,8], stimid_N=4, location='eb', loc_num=None,
                       stim_method='dypos', t_before_end=-2, t_end=0, cm=plt.cm.viridis,
                       t_win_stop=[-12,4], offset=0, ylim=[0,3], fs=8, lw=1,):
    def _plot_shape(show_yscale=True):
        xs = np.linspace(-180, 180, len(shapes_group[0]))
        ymean = np.nanmean(shapes_group, axis=0)
        ysem = fc.nansem(shapes_group, axis=0)
        ax.plot(xs, ymean, lw=lw, c=c)
        ax.fill_between(xs, ymean - ysem, ymean + ysem, facecolor=c, edgecolor='none', alpha=.2)
        ph.adjust_spines(ax, ['bottom'], lw=1, ylim=ylim, yticks=[], xlim=[-180,180], xticks=[], pad=0)
        if show_yscale:
            ph.plot_y_scale_text(ax, bar_length=.5, text='.5', x_text_right=.1, y_text_upper=.05, color=c,
                                 fontsize=fs, horizontalalignment='right', verticalalignment='center')
        # ax.set_title(group_labels[col])

    if not ax:
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(1, 2, width_ratios=[20,1])
        ax = plt.subplot(gs[0, 0])

    nstim = len(stimids)
    color_ind_seq = np.linspace(0, 1, nstim)
    for istimid, stimid_special in enumerate(stimids):
        shapes_group = get_bolusshape_unit(recss, stimid_special=stimid_special, stimid_N=stimid_N, location=location, loc_num=loc_num,
                                            stim_method=stim_method, t_before_end=t_before_end, t_end=t_end,
                                            t_win_stop=t_win_stop, offset=offset)
        show_yscale = True if istimid == 0 else False
        c = cm(color_ind_seq[istimid])
        _plot_shape(show_yscale)

    # ax = plt.subplot(gs[0, 1])
    # ph.plot_colormap(ax, colormap=cm, reverse_cm=False, ylim=[0, 256], yticks=[0, 256],
    #                  yticklabels=cm_ticklabels, ylabel=cm_label, label_rotation=90, label_pos="right", label_axeratio=1)

def barplot_bolusshape_threespeeds(recss, fig_size=(20,3), stimids=[4,3,2,1,6,5,4,7,8], stimid_N=10, location='eb', loc_num=None,
                                   stim_method='dypos', t_before_end=-2, t_end=0, t_win_stop=[-12,4], offset=0,
                                   c='black', alphas=[.3,.7,1], ylim=[0,3], fs=8, lw=1,):
    def _plot_shape(show_yscale=True):
        xs = np.linspace(-180, 180, len(shapes_group[0]))
        ymean = np.nanmean(shapes_group, axis=0)
        ysem = fc.nansem(shapes_group, axis=0)
        ax.plot(xs, ymean, lw=lw, c=c, alpha=alpha)
        # ax.fill_between(xs, ymean - ysem, ymean + ysem, facecolor=c, edgecolor='none', alpha=.2)
        ph.adjust_spines(ax, ['bottom'], lw=1, ylim=ylim, yticks=[], xlim=[-180,180], xticks=[], pad=0)
        if show_yscale:
            ph.plot_y_scale_text(ax, bar_length=1, text='1', x_text_right=.1, y_text_upper=.05, color=c,
                                 fontsize=fs, horizontalalignment='right', verticalalignment='center')
        # ax.set_title(group_labels[col])

    ncol = len(stimids)
    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(1, ncol)

    for col, stimid_special in enumerate(stimids):
        ax = plt.subplot(gs[0, col])
        for i_speed, stimid_special_ in enumerate([stimid_special, stimid_special+3, stimid_special+6]):
            shapes_group = get_bolusshape_unit(recss, stimid_special=stimid_special_, stimid_N=stimid_N,
                                                location=location, loc_num=loc_num,
                                                stim_method=stim_method, t_before_end=t_before_end, t_end=t_end,
                                                t_win_stop=t_win_stop, offset=offset)
            show_yscale = True if col == 0 else False
            alpha = alphas[i_speed]
            _plot_shape(show_yscale)


def barplot_bolusshape_overlap(recss, fig_size=(3,3), stimids=[4,3,2,1,6,5,4,7,8], location='eb', loc_num=None,
                       stim_method='dypos', t_before_end=-2, t_end=0,
                       t_win_stop=[-12,4], offset=0, colors=[ph.purple, ph.green, ph.yellow], ylim=[0,3], fs=8, lw=1,):
    def _plot_shape(show_yscale=True):
        xs = np.linspace(-180, 180, len(shapes_group[0]))
        ymean = np.nanmean(shapes_group, axis=0)
        ysem = fc.nansem(shapes_group, axis=0)
        ax.plot(xs, ymean, lw=lw, c=c)
        # ax.fill_between(xs, ymean - ysem, ymean + ysem, facecolor=c, edgecolor='none', alpha=.2)
        ph.adjust_spines(ax, ['bottom'], lw=1, ylim=ylim, yticks=[], xlim=[-180,180], xticks=[], pad=0)
        if show_yscale:
            ph.plot_y_scale_text(ax, bar_length=1, text='1', x_text_right=.1, y_text_upper=.05, color=c,
                                 fontsize=fs, horizontalalignment='right', verticalalignment='center')
        # ax.set_title(group_labels[col])

    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])

    for i_stim, stimid_special in enumerate(stimids):
        shapes_group = get_bolusshape_unit(recss, stimid_special=stimid_special, location=location, loc_num=loc_num,
                                            stim_method=stim_method, t_before_end=t_before_end, t_end=t_end,
                                            t_win_stop=t_win_stop, offset=offset)
        show_yscale = True if i_stim == 0 else False
        c = colors[i_stim]
        _plot_shape(show_yscale)

def plot_bolusshape_amplitude_interpolation(recss, ax=False, fig_size=(3,3), stimids=[4,3,2,1,6,5,4,7,8], stimid_N=4, location='fb',
                              PB_left=True, loc_num=None, stim_method='dypos', amp_type='mean', t_before_end=-2, t_end=0,
                              t_win_stop=[-12,4], offset=0, c='black', ylim=[0,3], fs=8,
                              plot_linefit=False, plot_cosfit=False, lw=1, cfit=ph.grey7, alphafit=1):
    def _cosine_func(x, amp, baseline, phase):
        return amp * np.sin(x + phase) + baseline

    def _cosine_func_135(x, amp, baseline):
        return amp * np.sin(x + np.pi*0.75) + baseline

    def _cosine_func_neg135(x, amp, baseline):
        return amp * np.sin(x + np.pi*0.25) + baseline

    def _baseline_func(x, baseline):
        return x*0 + baseline

    if not ax:
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    xss = []
    for istim, stimid_special in enumerate(stimids):
        shapes_group_ = get_bolusshape_unit(recss, stimid_special=stimid_special, stimid_N=stimid_N,
                                            location=location, loc_num=loc_num,
                                            stim_method=stim_method, t_before_end=t_before_end, t_end=t_end,
                                            t_win_stop=t_win_stop, offset=offset)
        xs = []

        if location == 'pb':
            shapes_group = []
            len_shape = len(shapes_group_[0])
            if PB_left:
                for shape_ in shapes_group_:
                    shape__ = shape_[:(len_shape / 2)]
                    shapes_group.append(shape__[~np.isnan(shape__)])
            else:
                for shape_ in shapes_group_:
                    shape__ = shape_[(len_shape / 2):]
                    shapes_group.append(shape__[~np.isnan(shape__)])
        else:
            shapes_group = shapes_group_

        if amp_type == 'mean':
            for shape in shapes_group:
                xs.append(np.nanmean(shape))
        if amp_type == 'mean_fit':
            for shape in shapes_group:
                x_data = np.radians(np.linspace(0, 360, len(shape)))
                params, _ = optimize.curve_fit(_cosine_func, x_data, shape, p0=[1, 1, 1])
                xs.append(params[1])
        if amp_type == 'max':
            for shape in shapes_group:
                xs.append(np.nanmax(shape))
        if amp_type == 'max_fit':
            for shape in shapes_group:
                x_data = np.radians(np.linspace(0, 360, len(shape)))
                params, _ = optimize.curve_fit(_cosine_func, x_data, shape, p0=[1, 1, 1])
                xs.append(params[0]+params[1])
        if amp_type == 'amp_mmm':
            for shape in shapes_group:
                xs.append((np.nanmax(shape) - np.nanmin(shape))/2.)
        if amp_type == 'amp_fit':
            for shape in shapes_group:
                x_data = np.radians(np.linspace(0, 360, len(shape)))
                params, _ = optimize.curve_fit(_cosine_func, x_data, shape, p0=[1, 1, 1])
                xs.append(np.abs(params[0]))
        xss.append(xs)

    group_info = [[i] for i in range(len(stimids))]
    # print np.nanmean(xss, axis=1)
    ph.errorbar_lc_xss(xss=xss, ax=ax, group_info=group_info, show_xlabel=False, colors=[c],
                       col_width=0.2, subgroup_gap=0.2, group_gap=1, plot_scale=1, margin_to_yaxis=.5,
                       figsize_height=5, ms_indi=2, alpha_indi=1,
                       noyaxis=True, ylabel='', yticks=[], ylim=ylim, yticks_minor=[], fs=fs,
                       ro_x=0, label_axeratio=-.1, rotate_ylabel=False, plot_bar=False,
                       subgroup_samefly=False, circstat=True, show_flynum=False, show_nancol=False)

    ph.plot_y_scale_text(ax, bar_length=0.5, text='0.5', x_text_right=.1, y_text_upper=.05, color=c, fontsize=fs,
                         horizontalalignment='right', verticalalignment='center')
    ax.axhline(0, ls='--', lw=.5, c=ph.grey9, alpha=.5, zorder=1)

    if plot_linefit:
        x_f = np.radians(np.linspace(0, 360, 7))
        params, _ = optimize.curve_fit(_baseline_func, x_f, np.nanmean(xss, axis=1), p0=[1])
        baseline = params
        ax.plot(np.linspace(0.6, 7.8, 80), np.linspace(0.6, 7.8, 80)*0 + baseline, c=cfit,
                alpha=alphafit, lw=lw)
    elif plot_cosfit:
        x_f = np.radians(np.linspace(0, 360, 7))
        params, _ = optimize.curve_fit(_cosine_func, x_f, np.nanmean(xss, axis=1), p0=[1, 1, 1])
        amp, baseline, phase = params
        ax.plot(np.arange(0.6, 7.9, 1.2), _cosine_func(x_f, amp, baseline, phase), c=cfit, alpha=alphafit, lw=lw, marker='o')
        # if PB_left:
        #     params, _ = optimize.curve_fit(_cosine_func_135, x_f, np.nanmean(xss, axis=1), p0=[1, 1])
        #     amp, baseline = params
        #     ax.plot(np.linspace(0.6,7.8,80), _cosine_func_135(np.radians(np.linspace(0, 360, 80)), amp, baseline), c=cfit, alpha=alphafit, lw=lw)
        # else:
        #     params, _ = optimize.curve_fit(_cosine_func_neg135, x_f, np.nanmean(xss, axis=1), p0=[1, 1])
        #     amp, baseline = params
        #     ax.plot(np.linspace(0.6,7.8,80), _cosine_func_neg135(np.radians(np.linspace(0, 360, 80)), amp, baseline), c=cfit, alpha=alphafit, lw=lw)

def plot_bolusshape_amplitude_subsampled(recss, ax=False, fig_size=(3,3), stimids=[4,3,2,1,6,5,4,7,8], stimid_N=4, location='fb',
                              PB_left=True, loc_num=None, stim_method='dypos', amp_type='mean', t_before_end=-2, t_end=0,
                              t_win_stop=[-12,4], offset=0, c='black', ylim=[0,3], fs=8, flight_detection=True,
                              plot_linefit=False, fit_offset=True, plot_cosfit=False, lw=1, cfit=ph.grey7, alphafit=1,
                            print_indi=False, print_mean=False):
    def _cosine_func(x, amp, baseline, phase):
        return amp * np.cos(x + phase) + baseline

    def _cosine_func_135(x, amp, baseline):
        return amp * np.sin(x + np.pi*0.75) + baseline

    def _cosine_func_neg135(x, amp, baseline):
        return amp * np.sin(x + np.pi*0.25) + baseline

    def _baseline_func(x, baseline):
        return x*0 + baseline

    if not ax:
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    xss = []
    for istim, stimid_special in enumerate(stimids):
        shapes_group_ = get_bolusshape_unit(recss, stimid_special=stimid_special, stimid_N=stimid_N,
                                            location=location, loc_num=loc_num, flight_detection=flight_detection,
                                            stim_method=stim_method, t_before_end=t_before_end, t_end=t_end,
                                            t_win_stop=t_win_stop, offset=offset)
        # subsample
        _, _, x = recss[0][0].get_objects('%s.c%i.xedges_interp' % (location, 1))
        idx = np.arange(len(x)).reshape((len(x) / 10, 10))
        shapes_sub = []
        for shape_fly in shapes_group_:
            shapes_sub.append(np.nanmean(shape_fly[idx], axis=-1))

        if location == 'pb':
            shapes_group = []
            len_shape = len(shapes_sub[0])
            if PB_left:
                for shape_ in shapes_sub:
                    shape__ = shape_[:(len_shape / 2)]
                    shapes_group.append(shape__[~np.isnan(shape__)])
            else:
                for shape_ in shapes_sub:
                    shape__ = shape_[(len_shape / 2):]
                    shapes_group.append(shape__[~np.isnan(shape__)])
        else:
            shapes_group = shapes_sub

        nglom = len(shapes_group[0])
        x_data_ = np.linspace(0, 360, nglom + 1)
        x_data = np.radians((x_data_[1:] + x_data_[:-1]) / 2.)

        xs = []
        if amp_type == 'mean':
            for shape in shapes_group:
                xs.append(np.nanmean(shape))
        if amp_type == 'mean_fit':
            for shape in shapes_group:
                # x_data = np.radians(np.linspace(0, 360, len(shape)))
                params, _ = optimize.curve_fit(_cosine_func, x_data, shape, p0=[1, 1, 1])
                xs.append(params[1])
        if amp_type == 'max':
            for shape in shapes_group:
                xs.append(np.nanmax(shape))
        if amp_type == 'max_fit':
            for shape in shapes_group:
                # x_data = np.radians(np.linspace(0, 360, len(shape)))
                params, _ = optimize.curve_fit(_cosine_func, x_data, shape, p0=[1, 1, 1])
                xs.append(params[0]+params[1])
        if amp_type == 'amp_mmm':
            for shape in shapes_group:
                xs.append((np.nanmax(shape) - np.nanmin(shape))/2.)
        if amp_type == 'amp_fit':
            for shape in shapes_group:
                x_data = np.radians(np.linspace(0, 360, len(shape)))
                # print x_data
                # print shape
                params, _ = optimize.curve_fit(_cosine_func, x_data, shape, p0=[1, 1, 1])
                xs.append(np.abs(params[0]))
        xss.append(xs)

    group_info = [[i] for i in range(len(stimids))]

    if print_indi:
        print 'indi:'
        for xs in xss:
            ystr = ''
            for y in xs:
                ystr = ystr + '%.4f' % y + ', '
            print ystr
    if print_mean:
        ystr = ''
        for y in np.nanmean(xss, axis=1):
            ystr = ystr + '%.4f' % y + ', '
        print 'mean: %s' % ystr
        ystr = ''
        for y in stats.sem(xss, axis=1):
            ystr = ystr + '%.4f' % y + ', '
        print 'sem: %s' % ystr
        ystr = ''
        for y in np.var(xss, axis=1):
            ystr = ystr + '%.4f' % y + ', '
        print 'variance: %s' % ystr

    ph.errorbar_lc_xss(xss=xss, ax=ax, group_info=group_info, show_xlabel=False, colors=[c],
                       col_width=0.2, subgroup_gap=0.2, group_gap=1, plot_scale=1, margin_to_yaxis=.5,
                       figsize_height=5, ms_indi=2, alpha_indi=1,
                       noyaxis=True, ylabel='', yticks=[], ylim=ylim, yticks_minor=[], fs=fs,
                       ro_x=0, label_axeratio=-.1, rotate_ylabel=False, plot_bar=False,
                       subgroup_samefly=False, circstat=True, show_flynum=False, show_nancol=False)

    ph.plot_y_scale_text(ax, bar_length=0.5, text='0.5', x_text_right=.1, y_text_upper=.05, color=c, fontsize=fs,
                         horizontalalignment='right', verticalalignment='center')
    ax.axhline(0, ls='--', lw=.5, c=ph.grey9, alpha=.5, zorder=1)

    if plot_linefit:
        x_f = np.radians(np.linspace(0, 360, 7))
        params, _ = optimize.curve_fit(_baseline_func, x_f, np.nanmean(xss, axis=1), p0=[1])
        baseline = params
        ax.plot(np.linspace(0.6, 7.8, 80), np.linspace(0.6, 7.8, 80)*0 + baseline, c=cfit,
                alpha=alphafit, lw=lw)
        print baseline

    elif plot_cosfit:
        x_f = np.radians(np.linspace(0, 360, 7))
        if fit_offset:
            # phases = []
            # xss_swapped = np.swapaxes(xss, 0, 1)
            # for xs in xss_swapped:
            #     params, _ = optimize.curve_fit(_cosine_func, x_f, xs, p0=[1, 1, 1])
            #     amp, baseline, phase = params
            #     phases.append(np.degrees(phase))
            # print phases
            params, _ = optimize.curve_fit(_cosine_func, x_f, np.nanmean(xss, axis=1), p0=[1, 1, 1])
            amp, baseline, phase = params
            ax.plot(np.linspace(0.6,7.8,80), _cosine_func(np.radians(np.linspace(0,360,80)), amp, baseline, phase), c=cfit, alpha=alphafit, lw=lw)
            ystr = ''
            for y in _cosine_func(np.radians(np.linspace(0,360,7)), amp, baseline, phase):
                ystr = ystr + '%.4f' % y + ', '
            print 'amp: %.4f  baseline: %.4f' % (amp, baseline)
            print 'fit: ' + ystr

        else:
            if PB_left:
                params, _ = optimize.curve_fit(_cosine_func_135, x_f, np.nanmean(xss, axis=1), p0=[1, 1])
                amp, baseline = params
                ax.plot(np.linspace(0.6,7.8,80), _cosine_func_135(np.radians(np.linspace(0, 360, 80)), amp, baseline), c=cfit, alpha=alphafit, lw=lw)
            else:
                params, _ = optimize.curve_fit(_cosine_func_neg135, x_f, np.nanmean(xss, axis=1), p0=[1, 1])
                amp, baseline = params
                ax.plot(np.linspace(0.6,7.8,80), _cosine_func_neg135(np.radians(np.linspace(0, 360, 80)), amp, baseline), c=cfit, alpha=alphafit, lw=lw)

def plot_bolusshape_amplitude_subsampled_chisquare_20210709(recss, ax=False, fig_size=(3,3), stimids=[4,3,2,1,6,5,4,7,8], stimid_N=4, location='fb',
                              PB_left=True, loc_num=None, stim_method='dypos', amp_type='mean', t_before_end=-2, t_end=0,
                              t_win_stop=[-12,4], offset=0, c='black', ylim=[0,3], fs=8, flight_detection=True,
                            lw=1, cfit=ph.grey7, alphafit=1, least_square=True):
    def _cosine_func(x, amp, baseline, phase):
        return amp * np.cos(x + phase) + baseline
    def my_square(p, data, var, angles):
        return np.sum((data - (p[0] * np.cos(np.radians(angles + p[1])) + p[2])) ** 2)
    def my_chisquare(p, data, var, angles):
        return np.sum((data - (p[0] * np.cos(np.radians(angles + p[1])) + p[2])) ** 2 / var)
    def my_curve(p, angles):
        return p[0] * np.cos(np.radians(angles + p[1])) + p[2]

    if not ax:
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    xss = []
    for istim, stimid_special in enumerate(stimids):
        shapes_group_ = get_bolusshape_unit(recss, stimid_special=stimid_special, stimid_N=stimid_N,
                                            location=location, loc_num=loc_num, flight_detection=flight_detection,
                                            stim_method=stim_method, t_before_end=t_before_end, t_end=t_end,
                                            t_win_stop=t_win_stop, offset=offset)
        # subsample
        _, _, x = recss[0][0].get_objects('%s.c%i.xedges_interp' % (location, 1))
        idx = np.arange(len(x)).reshape((len(x) / 10, 10))
        shapes_sub = []
        for shape_fly in shapes_group_:
            shapes_sub.append(np.nanmean(shape_fly[idx], axis=-1))

        if location == 'pb':
            shapes_group = []
            len_shape = len(shapes_sub[0])
            if PB_left:
                for shape_ in shapes_sub:
                    shape__ = shape_[:(len_shape / 2)]
                    shapes_group.append(shape__[~np.isnan(shape__)])
            else:
                for shape_ in shapes_sub:
                    shape__ = shape_[(len_shape / 2):]
                    shapes_group.append(shape__[~np.isnan(shape__)])
        else:
            shapes_group = shapes_sub

        nglom = len(shapes_group[0])
        x_data_ = np.linspace(0, 360, nglom + 1)
        x_data = np.radians((x_data_[1:] + x_data_[:-1]) / 2.)

        xs = []
        if amp_type == 'mean':
            for shape in shapes_group:
                xs.append(np.nanmean(shape))
        if amp_type == 'mean_fit':
            for shape in shapes_group:
                # x_data = np.radians(np.linspace(0, 360, len(shape)))
                params, _ = optimize.curve_fit(_cosine_func, x_data, shape, p0=[1, 1, 1])
                xs.append(params[1])
        if amp_type == 'max':
            for shape in shapes_group:
                xs.append(np.nanmax(shape))
        if amp_type == 'max_fit':
            for shape in shapes_group:
                # x_data = np.radians(np.linspace(0, 360, len(shape)))
                params, _ = optimize.curve_fit(_cosine_func, x_data, shape, p0=[1, 1, 1])
                xs.append(params[0]+params[1])
        if amp_type == 'amp_mmm':
            for shape in shapes_group:
                xs.append((np.nanmax(shape) - np.nanmin(shape))/2.)
        if amp_type == 'amp_fit':
            for shape in shapes_group:
                x_data = np.radians(np.linspace(0, 360, len(shape)))
                # print x_data
                # print shape
                params, _ = optimize.curve_fit(_cosine_func, x_data, shape, p0=[1, 1, 1])
                xs.append(np.abs(params[0]))
        xss.append(xs)

    group_info = [[i] for i in range(len(stimids))]

    ph.errorbar_lc_xss(xss=xss, ax=ax, group_info=group_info, show_xlabel=False, colors=[c],
                       col_width=0.2, subgroup_gap=0.2, group_gap=1, plot_scale=1, margin_to_yaxis=.5,
                       figsize_height=5, ms_indi=2, alpha_indi=1,
                       noyaxis=True, ylabel='', yticks=[], ylim=ylim, yticks_minor=[], fs=fs,
                       ro_x=0, label_axeratio=-.1, rotate_ylabel=False, plot_bar=False,
                       subgroup_samefly=False, circstat=True, show_flynum=False, show_nancol=False)

    ph.plot_y_scale_text(ax, bar_length=0.5, text='0.5', x_text_right=.1, y_text_upper=.05, color=c, fontsize=fs,
                         horizontalalignment='right', verticalalignment='center')
    ax.axhline(0, ls='--', lw=.5, c=ph.grey9, alpha=.5, zorder=1)

    angles = [-180, -120, -60, 0, 60, 120]
    angles_plot = np.linspace(-180, 180, 80)
    data = np.nanmean(xss, axis=1)[:6]
    var = np.var(xss, axis=1)[:6]
    dof = 3
    if least_square:
        p = optimize.fmin(my_square, [1, 1, 1], args=(data, var, angles), disp=False)
    else:
        p = optimize.fmin(my_chisquare, [1, 1, 1], args=(data, var, angles), disp=False)

    print 'chi-square: %.4f' % (my_chisquare(p, data, var, angles) / dof)
    print 'p-value: %.6f' % stats.distributions.chi2.sf(my_chisquare(p, data, var, angles), dof)
    print 'amp: %.4f, baseline: %.4f, offset: %.3f' % (p[0], p[2], p[1])

    ax.plot(np.linspace(0.6,7.8,80), my_curve(p, angles_plot), c=cfit, alpha=alphafit, lw=lw)


def plot_speedtuned_amp(groups, fig_size=(7,2), t_win_stop=[-5,1], location='fb', PB_left=True, loc_num=None,
            stimids_base=np.array([1,.75,.5,.25,1.5,1.25,1,]), stim_step=1.5, stimcycle=4, stimid_N=4,
            stim_method='dypos', amp_type='amp_fit', t_before_end=-2, t_end=0, offset=0,
            cm_cycle=plt.cm.viridis, colors_celltype=['black'], flight_detection=True,
            ylim_curve=[0,1.5], yticks_curve=[0,.5,1,1.5], speeds=[.0875,.175,.35,.7],
            xlim_amp=[0, .75], xticks_amp=[0,.25,.5,.75], ylim_amp=[0,1.5], yticks_amp=[0,.5,1,1.5],
            ms=6, lw=2, alpha_sem=0.1, fs=10, ylabels=['celltype']):
    def _cosine_func(x, amp, baseline, phase):
        return amp * np.sin(x + phase) + baseline

    ph.set_fontsize(fs)
    nrow = len(groups)
    ncol = 3
    nstim = len(stimids_base)
    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(nrow, ncol, width_ratios=[2,1,1])
    x_fit = np.radians(np.linspace(-180,180, 7))
    colorcycle_idxs = np.linspace(0, 1, stimcycle +1)

    for row in range(nrow):
        recss = groups[row]
        print ylabels[row]
        ax = plt.subplot(gs[row, 0])
        bsls, amps = np.zeros(stimcycle), np.zeros(stimcycle),
        for ispeed in range(stimcycle):
            stimids = stimids_base + ispeed * stim_step
            xss = []
            for istim, stimid_special in enumerate(stimids):
                shapes_group_ = get_bolusshape_unit(recss, stimid_special=stimid_special, stimid_N=stimid_N,
                                                    location=location, loc_num=loc_num, flight_detection=flight_detection,
                                                    stim_method=stim_method, t_before_end=t_before_end, t_end=t_end,
                                                    t_win_stop=t_win_stop, offset=offset)
                # subsample
                _, _, x = recss[0][0].get_objects('%s.c%i.xedges_interp' % (location, 1))
                idx = np.arange(len(x)).reshape((len(x) / 10, 10))
                shapes_sub = []
                for shape_fly in shapes_group_:
                    shapes_sub.append(np.nanmean(shape_fly[idx], axis=-1))

                if location == 'pb':
                    shapes_group = []
                    len_shape = len(shapes_sub[0])
                    if PB_left:
                        for shape_ in shapes_sub:
                            shape__ = shape_[:(len_shape / 2)]
                            shapes_group.append(shape__[~np.isnan(shape__)])
                    else:
                        for shape_ in shapes_sub:
                            shape__ = shape_[(len_shape / 2):]
                            shapes_group.append(shape__[~np.isnan(shape__)])
                else:
                    shapes_group = shapes_sub

                nglom = len(shapes_group[0])
                x_data_ = np.linspace(0, 360, nglom + 1)
                x_data = np.radians((x_data_[1:] + x_data_[:-1]) / 2.)

                xs = []
                if amp_type == 'mean':
                    for shape in shapes_group:
                        xs.append(np.nanmean(shape))
                if amp_type == 'mean_fit':
                    for shape in shapes_group:
                        # x_data = np.radians(np.linspace(0, 360, len(shape)))
                        params, _ = optimize.curve_fit(_cosine_func, x_data, shape, p0=[1, 1, 1])
                        xs.append(params[1])
                if amp_type == 'max':
                    for shape in shapes_group:
                        xs.append(np.nanmax(shape))
                if amp_type == 'max_fit':
                    for shape in shapes_group:
                        # x_data = np.radians(np.linspace(0, 360, len(shape)))
                        params, _ = optimize.curve_fit(_cosine_func, x_data, shape, p0=[1, 1, 1])
                        xs.append(params[0] + params[1])
                if amp_type == 'amp_mmm':
                    for shape in shapes_group:
                        xs.append((np.nanmax(shape) - np.nanmin(shape)) / 2.)
                if amp_type == 'amp_fit':
                    for shape in shapes_group:
                        x_data = np.radians(np.linspace(0, 360, len(shape)))
                        # print x_data
                        # print shape
                        params, _ = optimize.curve_fit(_cosine_func, x_data, shape, p0=[1, 1, 1])
                        xs.append(np.abs(params[0]))
                xss.append(xs)

            # plot four speed curves
            x_plot = np.linspace(-180, 180, nstim)
            ymean, ysem = np.zeros(nstim), np.zeros(nstim)
            for istim in range(nstim):
                ymean[istim] = np.nanmean(xss[istim])
                ysem[istim] = np.nanstd(xss[istim]) / np.sqrt(len(xss[istim]))

            ax.plot(x_plot, ymean, c=cm_cycle(colorcycle_idxs[ispeed]), lw=lw)
            ax.fill_between(x_plot, ymean - ysem, ymean + ysem, facecolor=cm_cycle(colorcycle_idxs[ispeed]),
                            edgecolor='none', alpha=alpha_sem)

            # print mean or sem
            # ystr = ''
            # for y in ysem:
            #     ystr = ystr + str(y) + ', '
            # print ystr

            # compute bsl and amp
            params, _ = optimize.curve_fit(_cosine_func, x_fit, ymean, p0=[1, 1, 1])
            amp, bsl, _ = params
            bsls[ispeed] = bsl
            amps[ispeed] = np.abs(amp)

            # print sem of fit amp
            amps_, bsls_ = [], []
            # print xss
            for ifly in range(len(recss)):
                y = np.zeros(nstim)
                for istim in range(nstim):
                    # print istim, ifly
                    y[istim] = xss[istim][ifly]
                params, _ = optimize.curve_fit(_cosine_func, x_fit, y, p0=[1, 1, 1])
                amp, bsl, _ = params
                amps_.append(np.abs(amp))
                bsls_.append(bsl)
            print np.nanstd(amps_) / np.sqrt(len(amps_)), np.nanstd(bsls_) / np.sqrt(len(bsls_))

        ax.grid(which='major', alpha=0)
        ph.adjust_spines(ax, ['left', 'bottom'], xlim=[-185, 185], xticks=[-180, -90, 0, 90, 180], ylim=ylim_curve,
                         yticks=yticks_curve)
        ax.set_ylabel(ylabels[row], fontsize=fs)

        ax = plt.subplot(gs[row, 1])
        ax.plot(speeds, bsls, c=colors_celltype[row], lw=lw, marker='s', mec='none', ms=ms, mfc=colors_celltype[row])
        ax.set_title('Baseline', fontsize=fs)
        ph.adjust_spines(ax, ['left', 'bottom'], xlim=xlim_amp, xticks=xticks_amp, ylim=ylim_amp, yticks=yticks_amp)

        ax = plt.subplot(gs[row, 2])
        ax.plot(speeds, amps, c=colors_celltype[row], lw=lw, marker='s', mec='none', ms=ms, mfc=colors_celltype[row])
        ax.set_title('Amplitude', fontsize=fs)
        ph.adjust_spines(ax, ['left', 'bottom'], xlim=xlim_amp, xticks=xticks_amp, ylim=ylim_amp, yticks=yticks_amp)


def print_bolusshape_amplitude_subsampled_phase(recss, stimids=[4,3,2,1,6,5,4,7,8], stimid_N=4, location='fb',
                              PB_left=True, loc_num=None, stim_method='dypos', amp_type='mean', t_before_end=-2, t_end=0,
                              t_win_stop=[-12,4], offset=0, c='black', ylim=[0,3], fs=8,
                              plot_linefit=False, plot_cosfit=False, lw=1, cfit=ph.grey7, alphafit=1):
    def _cosine_func(x, amp, baseline, phase):
        return amp * np.cos(x + phase) + baseline

    xss = []
    for istim, stimid_special in enumerate(stimids):
        shapes_group_ = get_bolusshape_unit(recss, stimid_special=stimid_special, stimid_N=stimid_N,
                                            location=location, loc_num=loc_num,
                                            stim_method=stim_method, t_before_end=t_before_end, t_end=t_end,
                                            t_win_stop=t_win_stop, offset=offset)
        # subsample
        _, _, x = recss[0][0].get_objects('%s.c%i.xedges_interp' % (location, 1))
        idx = np.arange(len(x)).reshape((len(x) / 10, 10))
        shapes_sub = []
        for shape_fly in shapes_group_:
            shapes_sub.append(np.nanmean(shape_fly[idx], axis=-1))

        if location == 'pb':
            shapes_group = []
            len_shape = len(shapes_sub[0])
            if PB_left:
                for shape_ in shapes_sub:
                    shape__ = shape_[:(len_shape / 2)]
                    shapes_group.append(shape__[~np.isnan(shape__)])
            else:
                for shape_ in shapes_sub:
                    shape__ = shape_[(len_shape / 2):]
                    shapes_group.append(shape__[~np.isnan(shape__)])
        else:
            shapes_group = shapes_sub

        nglom = len(shapes_group[0])
        x_data_ = np.linspace(0, 360, nglom + 1)
        x_data = np.radians((x_data_[1:] + x_data_[:-1]) / 2.)

        xs = []
        if amp_type == 'mean':
            for shape in shapes_group:
                xs.append(np.nanmean(shape))
        if amp_type == 'mean_fit':
            for shape in shapes_group:
                # x_data = np.radians(np.linspace(0, 360, len(shape)))
                params, _ = optimize.curve_fit(_cosine_func, x_data, shape, p0=[1, 1, 1])
                xs.append(params[1])
        if amp_type == 'max':
            for shape in shapes_group:
                xs.append(np.nanmax(shape))
        if amp_type == 'max_fit':
            for shape in shapes_group:
                # x_data = np.radians(np.linspace(0, 360, len(shape)))
                params, _ = optimize.curve_fit(_cosine_func, x_data, shape, p0=[1, 1, 1])
                xs.append(params[0]+params[1])
        if amp_type == 'amp_mmm':
            for shape in shapes_group:
                xs.append((np.nanmax(shape) - np.nanmin(shape))/2.)
        if amp_type == 'amp_fit':
            for shape in shapes_group:
                # x_data = np.radians(np.linspace(0, 360, len(shape)))
                # print x_data
                # print shape
                params, _ = optimize.curve_fit(_cosine_func, x_data, shape, p0=[1, 1, 1])
                xs.append(np.abs(params[0]))
        xss.append(xs)

    x_f = np.radians(np.linspace(-180, 180, 7))
    params, _ = optimize.curve_fit(_cosine_func, x_f, np.nanmean(xss, axis=1), p0=[1, 1, 1])
    amp, baseline, phase = params
    print -(np.degrees(phase) - (1 - np.sign(amp)) * 90)

def plot_shapes_fit_afo_driftangle_interpolation(recss, fig_size=(4.5,2), stimids=[4,3,2,1,6,5,4,7,8], stimid_N=4, location='pb',
                                   loc_num=0, stim_method='dypos', t_before_end=-2, t_end=0, t_win_stop=[-12,4],
                                   offset=0, cdata='black', mew=0.5, ms=2, ylim=[0,3], lw=0.6, cfit=ph.grey7, alphafit=1):
    def _cosine_func(x, baseline, amp, phase):
        return amp * np.sin(x - phase) + baseline

    ncol = len(stimids)
    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(2, ncol)

    for col, stimid_special in enumerate(stimids):
        shapes_group = get_bolusshape_unit(recss, stimid_special=stimid_special, stimid_N=stimid_N,
                                            location=location, loc_num=loc_num,
                                            stim_method=stim_method, t_before_end=t_before_end, t_end=t_end,
                                            t_win_stop=t_win_stop, offset=offset)
        show_yscale = True if col == 0 else False

        for row in range(2):
            # plot data
            ax = plt.subplot(gs[row, col])
            PB_left = True if row == 0 else False

            shapes_group_half = []
            len_fullshape = len(shapes_group[0])
            if PB_left:
                for shape_ in shapes_group:
                    shape__ = shape_[:(len_fullshape / 2)]
                    shapes_group_half.append(shape__[~np.isnan(shape__)])
            else:
                for shape_ in shapes_group:
                    shape__ = shape_[(len_fullshape / 2):]
                    shapes_group_half.append(shape__[~np.isnan(shape__)])

            xs = np.linspace(-180, 180, len(shapes_group_half[0]))
            ymean = np.nanmean(shapes_group_half, axis=0)
            ax.plot(xs, ymean, ls='none', marker='o', mec=cdata, mfc='none', mew=mew, ms=ms)
            ph.adjust_spines(ax, ['bottom'], lw=1, ylim=ylim, yticks=[], xlim=[-180,180], xticks=[], pad=0)
            if show_yscale:
                ph.plot_y_scale_text(ax, bar_length=1, text='1', x_text_right=.1, y_text_upper=.05, color=cdata,
                                     fontsize=5, horizontalalignment='right', verticalalignment='center')

            # fit and plot
            x_f = np.radians(np.linspace(-180, 180, len(shapes_group_half[0])))
            params, _ = optimize.curve_fit(_cosine_func, x_f, np.nanmean(shapes_group_half, axis=0), p0=[1, 1, 1])
            baseline, amp, phase = params
            ax.plot(np.degrees(x_f), _cosine_func(x_f, baseline, amp, phase), c=cfit, alpha=alphafit, lw=lw)

def plot_shapes_fit_afo_driftangle_subsample(recss, fig_size=(4.5,2), stimids=[4,3,2,1,6,5,4,7,8], stimid_N=4, location='pb',
                                   loc_num=0, stim_method='dypos', t_before_end=-2, t_end=0, t_win_stop=[-12,4],
                                   offset=0, cdata='black', mew=0.5, ms=2, ylim=[0,3], lw=0.6, cfit=ph.grey7, alphafit=1):
    def _cosine_func(x, baseline, amp, phase):
        return amp * np.sin(x - phase) + baseline

    ncol = len(stimids)
    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(2, ncol)

    for col, stimid_special in enumerate(stimids):
        shapes_group_ = get_bolusshape_unit(recss, stimid_special=stimid_special, stimid_N=stimid_N,
                                            location=location, loc_num=loc_num,
                                            stim_method=stim_method, t_before_end=t_before_end, t_end=t_end,
                                            t_win_stop=t_win_stop, offset=offset)
        # subsample
        _, _, x = recss[0][0].get_objects('%s.c%i.xedges_interp' % (location, 1))
        idx = np.arange(len(x)).reshape((len(x) / 10, 10))
        shapes_sub = []
        for shape_fly in shapes_group_:
            shapes_sub.append(np.nanmean(shape_fly[idx], axis=-1))
        shapes_group = shapes_sub
        show_yscale = True if col == 0 else False

        for row in range(2):
            # plot data
            ax = plt.subplot(gs[row, col])
            PB_left = True if row == 0 else False

            shapes_group_half = []
            len_fullshape = len(shapes_group[0])
            if PB_left:
                for shape_ in shapes_group:
                    shape__ = shape_[:(len_fullshape / 2)]
                    shapes_group_half.append(shape__[~np.isnan(shape__)])
            else:
                for shape_ in shapes_group:
                    shape__ = shape_[(len_fullshape / 2):]
                    shapes_group_half.append(shape__[~np.isnan(shape__)])

            nglom = len(shapes_group_half[0])
            xs_ = np.linspace(-180, 180, nglom + 1)
            xs = (xs_[1:] + xs_[:-1]) / 2.
            ymean = np.nanmean(shapes_group_half, axis=0)
            # ax.plot(xs, ymean, ls='none', marker='o', mec=cdata, mfc='none', mew=mew, ms=ms)
            ax.plot(xs, ymean, lw=lw, c=cdata)
            ph.adjust_spines(ax, ['bottom'], lw=1, ylim=ylim, yticks=[], xlim=[-180,180], xticks=[], pad=0)
            if show_yscale:
                ph.plot_y_scale_text(ax, bar_length=1, text='1', x_text_right=.1, y_text_upper=.05, color=cdata,
                                     fontsize=5, horizontalalignment='right', verticalalignment='center')

            # fit and plot
            x_f = np.radians(xs)
            params, _ = optimize.curve_fit(_cosine_func, x_f, ymean, p0=[1, 1, 1])
            baseline, amp, phase = params
            x_fit = np.linspace(-180, 180, 80)
            ax.plot(x_fit, _cosine_func(np.radians(x_fit), baseline, amp, phase), c=cfit, alpha=alphafit, lw=lw)




def plot_onephase_stim_unit(recss, ax=False, locnum=1, stimid_special=1, t_win_plot=[1,11,0], vlines=[4,8], t_win_stop=[-2,14],
                         stim_method='dypos', show_axes=['left'], xlabel='', show_ylable=True,
                         show_timescale=True, ms=2, fs=11, alpha_indi=0.3, lw_indi=1, lw=2, c=ph.blue, lw_dash=1):
    if not ax:
        _ = plt.figure(1, (3, 2))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    t_interp = np.arange(t_win_plot[0] - .3, (t_win_plot[1]+t_win_plot[2])+.3, 0.1)
    xlim = [t_win_plot[0], t_win_plot[1]+t_win_plot[2]]
    ph.adjust_spines(ax, show_axes, lw=1, ylim=[-180, 180], yticks=[-180, -90, 0, 90, 180], xlim=xlim,
                     xticks=[], pad=0)
    dphase_group = []

    for fly in recss:
        dphase_fly = []

        for rec in fly:
            if stim_method == 'dypos':
                of_stimids, of_start_idxs, _ = get_stim_info_20191007(rec)
            elif stim_method == 'pwm':
                of_stimids, of_start_idxs, _ = get_stim_info_20200117_pwmdc(rec)
            elif stim_method == 'dxpos':
                of_stimids, of_start_idxs, _ = get_stim_info_20200224_xposonly(rec)
            stop_inds = get_flight_stopping_inds(rec)
            ssr_tif = rec.ims[0].sampling_rate
            ssr_abf = rec.abf.subsampling_rate

            for start_idx in of_start_idxs[of_stimids==stimid_special]:
                start_idx_abf = np.where(rec.abf.t < rec.ims[0].t[start_idx])[0][-1]
                idx_stopwindow = [start_idx_abf - np.ceil(np.abs(t_win_stop[0])*ssr_abf),
                                  start_idx_abf + np.ceil(np.abs(t_win_stop[-1])*ssr_abf)]
                if not np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
                    t0 = rec.abf.t[start_idx_abf]
                    idx1 = int(start_idx)
                    # idx0 = int(idx1 - np.ceil((np.abs(t_win_plot[0])+1) * ssr_tif))
                    idx0 = int(idx1 + np.ceil((t_win_plot[0]-1) * ssr_tif))
                    idx2 = int(idx1 + np.ceil(t_win_plot[1] * ssr_tif))
                    idx3 = int(idx2 + np.ceil((t_win_plot[2]+1) * ssr_tif))
                    idx_stimmove = int(idx1 + np.floor((vlines[0]) * ssr_tif))

                    t = rec.ims[locnum].t[idx0:idx3] - t0
                    if (t[0] > t_interp[0]) | (t[-1] < t_interp[-1]):
                        continue
                    phase_baseline = np.nanmean(fc.unwrap(rec.ims[locnum].c1.phase[idx0:idx_stimmove]))
                    phase_im = fc.unwrap(rec.ims[locnum].c1.phase[idx0:idx3]) - phase_baseline
                    f = interp1d(t, phase_im, kind='linear')
                    phase_interp = f(t_interp)
                    dphase_fly.append(phase_interp)

        if not np.sum(np.isnan(fc.circmean(dphase_fly, axis=0))):
            dphase_group.append(np.nanmean(dphase_fly, axis=0))
        ax.plot(t_interp, np.nanmean(dphase_fly, axis=0), lw=lw_indi, zorder=2, alpha=alpha_indi, c=c)
        # ph.plot_noline_between_lims(ax, t_interp, np.nanmean(dphase_fly, axis=0), threshold=135, color=c_indi, alpha=alpha_indi, lw=lw_indi, zorder=2)
        # ax.plot(t_interp, np.nanmean(dphase_fly, axis=0), ls='none', marker='.', mfc='none', mec=c, zorder=2, ms=ms, alpha=alpha_indi)
    ax.plot(t_interp, np.nanmean(dphase_group, axis=0), lw=lw, zorder=3, c=c)
    # ph.plot_noline_between_lims(ax, t_interp, np.nanmean(dphase_group, axis=0), threshold=135, color=c, lw=lw_mean, zorder=3)
    # ax.plot(t_interp, np.nanmean(dphase_group, axis=0), ls='none', marker='.', mfc='none', mec=c, zorder=3, ms=ms)
    for vline in vlines:
        ax.axvline(vline, ls='--', lw=lw_dash, c=ph.grey9, alpha=.5)
        # ax.axvline(vline, ls='--', lw=lw_dash, c=ph.dred, alpha=.8)
    ax.axhline(0, ls='--', lw=lw_dash, c=ph.grey9, alpha=.6, zorder=1)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fs)
    if show_ylable:
        ax.set_ylabel('phase (deg.)', rotation='horizontal', va='center', fontsize=fs)
    if show_timescale:
        ph.plot_x_scale_text(ax, bar_length=2, text='2s', y_text_lower=0.15, x_text_lefter=.05, fontsize=fs)

def plot_onephase_stim(group, fig_size=(12, 1.3), ncol=9, locnum=1, stimids=[4,3,2,1,6,5,4,7,8], stim_method='dypos',
                       t_win_plot=[1,11,0], vlines=[4,8], t_win_stop=[-2,14],
                       ms_phase_ave=.8, lw=2, lw_indi=0.5, c_indi=ph.grey7, alpha_indi=0.4, c=ph.blue, lw_dash=1, fs=7,):
    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(1, ncol)

    rec = group[0][0]
    if stim_method == 'dypos':
        of_stimids, of_start_idxs, _ = get_stim_info_20191007(rec)
    elif stim_method == 'pwm':
        of_stimids, of_start_idxs, _ = get_stim_info_20200117_pwmdc(rec)
    elif stim_method == 'dxpos':
        of_stimids, of_start_idxs, _ = get_stim_info_20200224_xposonly(rec)
    stop_inds = get_flight_stopping_inds(rec)
    ssr_abf = rec.abf.subsampling_rate

    for col in range(ncol):
        # plot sample trace
        stimid_special = stimids[col]

        # plot population average
        ax = plt.subplot(gs[0, col])
        plot_onephase_stim_unit(group, ax=ax, stimid_special=stimid_special, locnum=locnum, t_win_plot=t_win_plot, vlines=vlines,
                             t_win_stop=t_win_stop, stim_method=stim_method, show_axes=[], xlabel='', show_ylable=False,
                             show_timescale=False, ms=ms_phase_ave, lw_indi=lw_indi, alpha_indi=alpha_indi,
                             c=c, fs=fs, lw_dash=lw_dash)

## FFT

def get_fft_unit(recss, stim_method='dypos', stimid_special=1, loc_num=0, ch_num=1, stimid_N=10, period_bound=32,
                         t_before_end=-6.51, t_end=-4.1, t_win_stop=[-5,1], flight_detection=True):

    powers_group = []
    for fly in recss:
        data_fly = []

        for rec in fly:
            if stim_method == 'dypos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20191007(rec, stimid_N=stimid_N)
            elif stim_method == 'dxpos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20200224_xposonly(rec)

            stop_inds = get_flight_stopping_inds(rec)
            ssr_tif = rec.ims[0].sampling_rate
            ssr_abf = rec.abf.subsampling_rate

            ch = getattr(rec.ims[loc_num], 'c%i' % ch_num)
            data_an = ch.an

            for i_idx, start_idx in enumerate(of_start_idxs[of_stimids == stimid_special]):
                start_idx_abf = np.where(rec.abf.t < rec.ims[0].t[start_idx])[0][-1]
                idx_stopwindow = [start_idx_abf - np.ceil(np.abs(t_win_stop[0]) * ssr_abf),
                                  start_idx_abf + np.ceil(np.abs(t_win_stop[-1]) * ssr_abf)]
                flight_flag = (not np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1]))) if flight_detection else 1
                if flight_flag:
                    end_idx = of_end_idxs[of_stimids==stimid_special][i_idx]
                    idx_end = int(end_idx)
                    idx1 = int(idx_end + np.ceil(t_end * ssr_tif))
                    idx0 = int(idx_end + np.ceil(t_before_end * ssr_tif))
                    data_fly.extend(data_an[idx0:idx1])

        if len(data_an):
            period, powers = fbs.get_FFT_of_glomeruli_intensity_Arrayinput(np.array(data_fly), period_bound=period_bound,)
            powers_group.append(powers)
    return period, powers_group








# flight dual compass analysis plots

def plot_2c_td_paper_20200320(group, rec, fig_size=(12, 3.3), ncol=9, stimids=[4,3,2,1,6,5,4,7,8], stim_method='dypos',
                              trial_id=[1,1,1,1,1,1,1,1,1], t_win_plot=[1,10,0], vlines=[4,8], t_win_stop=[-2,14],
                              cmaps=[plt.cm.Blues, plt.cm.Oranges], vm=[0,1.2], colors=[ph.dorange, ph.blue],
                              ms=2, ms_phase_ave=.8, additional_filters=[], flip_neuropil_order=False,
                              lw_indi=0.5, c_indi=ph.grey7, alpha_indi=0.4, c='black', lw_dash=1, fs=7,):
    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(4, ncol)

    if stim_method == 'dypos':
        of_stimids, of_start_idxs, _ = get_stim_info_20191007(rec, stimid_N=1)
    elif stim_method == 'pwm':
        of_stimids, of_start_idxs, _ = get_stim_info_20200117_pwmdc(rec)
    elif stim_method == 'dxpos':
        of_stimids, of_start_idxs, _ = get_stim_info_20200224_xposonly(rec)
    stop_inds = get_flight_stopping_inds(rec)
    ssr_abf = rec.abf.subsampling_rate

    for col in range(ncol):
        # plot sample trace
        # axs = [plt.subplot(gs[0, col]), plt.subplot(gs[1, col]), plt.subplot(gs[2, col])] # edited at 20201025
        if flip_neuropil_order:
            axs = [plt.subplot(gs[1, col]), plt.subplot(gs[0, col]), plt.subplot(gs[2, col])]
        else:
            axs = [plt.subplot(gs[0, col]), plt.subplot(gs[1, col]), plt.subplot(gs[2, col])]
        stimid_special = stimids[col]
        start_idx = of_start_idxs[of_stimids==stimid_special][trial_id[col]]
        start_idx_abf = np.where(rec.abf.t < rec.fb.t[start_idx])[0][-1]        # check if flying stopped
        idx_stopwindow = [start_idx_abf - np.ceil(np.abs(t_win_stop[0])*ssr_abf),
                          start_idx_abf + np.ceil(np.abs(t_win_stop[-1])*ssr_abf)]
        if not np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
        # if 1:
            _show_xscale = True if col == 0 else False
            plot_indi_trial_unit(rec, axs=axs, idx_start=start_idx, t_win=t_win_plot, colors=colors, cmaps=cmaps,
                                 vlines=vlines, show_label=False, show_recid=False, show_xscale=_show_xscale, ms=ms,
                                 vm=vm, lw_dash=lw_dash, fs=fs)
        # plot population average
        ax = plt.subplot(gs[3, col])
        plot_dphase_ave_unit(group, ax=ax, stimid_special=stimid_special, t_win_plot=t_win_plot, vlines=vlines,
                             t_win_stop=t_win_stop, stim_method=stim_method, show_axes=[], xlabel='', show_ylable=False,
                             show_timescale=False, ms=ms_phase_ave, lw_indi=lw_indi, c_indi=c_indi, alpha_indi=alpha_indi,
                             c=c, fs=fs, lw_dash=lw_dash, additional_filters=additional_filters)

def plot_2c_2ch_20210324(group, rec, fig_size=(12, 3.3), ncol=9, stimids=[4,3,2,1,6,5,4,7,8], stim_method='dypos',
                         trial_id=[1,1,1,1,1,1,1,1,1], t_win_plot=[1,10,0], vlines=[4,8], t_win_stop=[-2,14],
                         locnums=[0,1], chnums=[1,1], cmaps=[plt.cm.Blues, plt.cm.Oranges], vm=[0,1.2],
                         colors=[ph.dorange, ph.blue], ms=2, ms_phase_ave=.8, additional_filters=[], flip_neuropil_order=False,
                        lw_indi=0.5, c_indi=ph.grey7, alpha_indi=0.4, c='black', lw_dash=1, fs=7,stopdetection=True):
    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(4, ncol)

    if stim_method == 'dypos':
        of_stimids, of_start_idxs, _ = get_stim_info_20191007(rec, stimid_N=1)
    elif stim_method == 'pwm':
        of_stimids, of_start_idxs, _ = get_stim_info_20200117_pwmdc(rec)
    elif stim_method == 'dxpos':
        of_stimids, of_start_idxs, _ = get_stim_info_20200224_xposonly(rec)
    elif stim_method == 'stimid':
        of_stimids, of_start_idxs, _ = get_stim_info_20210408_stimidonly(rec)
    stop_inds = get_flight_stopping_inds(rec)
    ssr_abf = rec.abf.subsampling_rate

    for col in range(ncol):
        # plot sample trace
        # axs = [plt.subplot(gs[0, col]), plt.subplot(gs[1, col]), plt.subplot(gs[2, col])] # edited at 20201025
        if flip_neuropil_order:
            axs = [plt.subplot(gs[1, col]), plt.subplot(gs[0, col]), plt.subplot(gs[2, col])]
        else:
            axs = [plt.subplot(gs[0, col]), plt.subplot(gs[1, col]), plt.subplot(gs[2, col])]
        stimid_special = stimids[col]
        start_idx = of_start_idxs[of_stimids==stimid_special][trial_id[col]]
        start_idx_abf = np.where(rec.abf.t < rec.ims[0].t[start_idx])[0][-1]        # check if flying stopped
        idx_stopwindow = [start_idx_abf - np.ceil(np.abs(t_win_stop[0])*ssr_abf),
                          start_idx_abf + np.ceil(np.abs(t_win_stop[-1])*ssr_abf)]
        if not np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
        # if 1:
            _show_xscale = True if col == 0 else False
            plot_indi_trial_2ch_unit(rec, axs=axs, idx_start=start_idx, t_win=t_win_plot, colors=colors, cmaps=cmaps,
                                 vlines=vlines, show_label=False, show_recid=False, show_xscale=_show_xscale, ms=ms,
                                 vm=vm, lw_dash=lw_dash, fs=fs, locnums=locnums, chnums=chnums)
        # plot population average
        ax = plt.subplot(gs[3, col])
        plot_dphase_ave_2ch_unit(group, ax=ax, locnums=locnums, chnums=chnums, stimid_special=stimid_special, t_win_plot=t_win_plot, vlines=vlines,
                             t_win_stop=t_win_stop, stim_method=stim_method, show_axes=[], xlabel='', show_ylable=False,
                             show_timescale=False, ms=ms_phase_ave, lw_indi=lw_indi, c_indi=c_indi, alpha_indi=alpha_indi,
                             c=c, fs=fs, lw_dash=lw_dash, additional_filters=additional_filters, stopdetection=stopdetection)

def plot_dphase_ave_unit(recss, ax=False, stimid_special=1, t_win_plot=[-4,4,4], vlines=[0,4], t_win_stop=[-8,8],
                         stim_method='dypos', show_axes=['left'], xlabel='', show_ylable=True, add180_phase=False,
                         additional_filters=[], stimid_N=10, show_timescale=True, ms=2,
                         fs=11, alpha_indi=0.3, c_indi=ph.grey9, lw_indi=1, c=ph.blue, lw_dash=1):
    if not ax:
        _ = plt.figure(1, (3, 2))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    t_interp = np.arange(t_win_plot[0] - .3, (t_win_plot[1]+t_win_plot[2])+.3, 0.1)
    xlim = [t_win_plot[0], t_win_plot[1]+t_win_plot[2]]
    ph.adjust_spines(ax, show_axes, lw=1, ylim=[-180, 180], yticks=[-180, -90, 0, 90, 180], xlim=xlim,
                     xticks=[], pad=0)
    dphase_group = []

    for fly in recss:
        dphase_fly = []

        for rec in fly:
            if stim_method == 'dypos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20191007(rec, stimid_N=stimid_N)
            elif stim_method == 'pwm':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20200117_pwmdc(rec)
            elif stim_method == 'dxpos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20200224_xposonly(rec)
            stop_inds = get_flight_stopping_inds(rec)
            ssr_tif = rec.ims[0].sampling_rate
            ssr_abf = rec.abf.subsampling_rate

            for i_stim, start_idx in enumerate(of_start_idxs[of_stimids == stimid_special]):
                start_idx_abf = np.where(rec.abf.t < rec.ims[0].t[start_idx])[0][-1]

                # stop detection
                idx_stopwindow = [start_idx_abf - np.ceil(np.abs(t_win_stop[0])*ssr_abf),
                                  start_idx_abf + np.ceil(np.abs(t_win_stop[-1])*ssr_abf)]
                if np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
                    continue

                # additional filters
                flag_filter_fail = False
                for additional_filter in additional_filters:
                    fail_flag = get_sig_ave_unit_check_filter(rec, additional_filter, start_idx,
                                                              of_end_idxs[of_stimids == stimid_special][i_stim])
                    if fail_flag:
                        flag_filter_fail = True
                        break
                if flag_filter_fail:
                    continue

                t0 = rec.abf.t[start_idx_abf]
                idx1 = int(start_idx)
                # idx0 = int(idx1 - np.ceil((np.abs(t_win_plot[0])+1) * ssr_tif))
                idx0 = int(idx1 + np.ceil((t_win_plot[0]-1) * ssr_tif))
                idx2 = int(idx1 + np.ceil(t_win_plot[1] * ssr_tif))
                idx3 = int(idx2 + np.ceil((t_win_plot[2]+1) * ssr_tif))

                t_im1, t_im0 = rec.ims[1].t[idx0:idx3] - t0, rec.ims[0].t[idx0:idx3] - t0
                if (t_im1[0] > t_interp[0]) | (t_im1[-1] < t_interp[-1]) | (t_im0[0] > t_interp[0]) | (t_im0[-1] < t_interp[-1]):
                    continue
                phase_im1, phase_im0 = rec.ims[1].c1.phase[idx0:idx3], rec.ims[0].c1.phase[idx0:idx3]
                f_im1 = interp1d(t_im1, fc.unwrap(phase_im1), kind='linear')
                f_im0 = interp1d(t_im0, fc.unwrap(phase_im0), kind='linear')
                phase_interp_im1 = fc.wrap(f_im1(t_interp)) if not add180_phase else fc.wrap(f_im1(t_interp)) + 180
                phase_interp_im0 = fc.wrap(f_im0(t_interp))
                dphase = fc.circsubtract(phase_interp_im1, phase_interp_im0)
                dphase_fly.append(dphase)

        if not np.sum(np.isnan(fc.circmean(dphase_fly, axis=0))):
            dphase_group.append(fc.circmean(dphase_fly, axis=0))

        if len(dphase_fly):
            # ax.plot(t_interp, fc.circmean(dphase_fly, axis=0), lw=lw_indi, zorder=2, alpha=alpha_indi, c=c_indi)
            ph.plot_noline_between_lims(ax, t_interp, fc.circmean(dphase_fly, axis=0), threshold=135, color=c_indi, alpha=alpha_indi, lw=lw_indi, zorder=2)
            # ax.plot(t_interp, fc.circmean(dphase_fly, axis=0), ls='none', marker='.', mfc='none', mec=c, zorder=2, ms=ms, alpha=alpha_indi)
    # ax.plot(t_interp, fc.circmean(dphase_group, axis=0), lw=2, zorder=3, c=ph.blue)
    # ph.plot_noline_between_lims(ax, t_interp, fc.circmean(dphase_group, axis=0), threshold=135, color=c, lw=lw, zorder=3)
    ax.plot(t_interp, fc.circmean(dphase_group, axis=0), ls='none', marker='o', mec='none', mfc=c, zorder=3, ms=ms)
    for vline in vlines:
        ax.axvline(vline, ls='--', lw=lw_dash, c=ph.grey9, alpha=.5)
        # ax.axvline(vline, ls='--', lw=lw_dash, c=ph.dred, alpha=.8)
    ax.axhline(0, ls='--', lw=lw_dash, c=ph.grey9, alpha=.6, zorder=1)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fs)
    if show_ylable:
        ax.set_ylabel('$\Delta \Theta$', rotation='horizontal', va='center', fontsize=fs)
    if show_timescale:
        ph.plot_x_scale_text(ax, bar_length=2, text='2s', y_text_lower=0.15, x_text_lefter=.05, fontsize=fs)

def plot_dphase_ave_2ch_unit(recss, ax=False, locnums=[0,1], chnums=[1,1], stimid_special=1, t_win_plot=[-4,4,4],
                    vlines=[0,4], t_win_stop=[-8,8], stim_method='dypos', show_axes=['left'], xlabel='',
                    show_ylable=True, add180_phase=False, additional_filters=[], stimid_N=10, show_timescale=True, ms=2,
                         fs=11, alpha_indi=0.3, c_indi=ph.grey9, lw_indi=1, c=ph.blue, lw_dash=1, stopdetection=True,):
    if not ax:
        _ = plt.figure(1, (3, 2))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    t_interp = np.arange(t_win_plot[0] - .3, (t_win_plot[1]+t_win_plot[2])+.3, 0.1)
    xlim = [t_win_plot[0], t_win_plot[1]+t_win_plot[2]]
    ph.adjust_spines(ax, show_axes, lw=1, ylim=[-180, 180], yticks=[-180, -90, 0, 90, 180], xlim=xlim,
                     xticks=[], pad=0)
    dphase_group = []

    for fly in recss:
        dphase_fly = []

        for rec in fly:
            if stim_method == 'dypos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20191007(rec, stimid_N=stimid_N)
            elif stim_method == 'pwm':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20200117_pwmdc(rec)
            elif stim_method == 'dxpos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20200224_xposonly(rec)
            elif stim_method == 'stimid':
                of_stimids, of_start_idxs, _ = get_stim_info_20210408_stimidonly(rec)
            stop_inds = get_flight_stopping_inds(rec)
            ssr_tif = rec.ims[0].sampling_rate
            ssr_abf = rec.abf.subsampling_rate
            chs = [getattr(rec.ims[locnums[0]], 'c%i' % chnums[0]), getattr(rec.ims[locnums[1]], 'c%i' % chnums[1])]

            for i_stim, start_idx in enumerate(of_start_idxs[of_stimids == stimid_special]):
                start_idx_abf = np.where(rec.abf.t < rec.ims[0].t[start_idx])[0][-1]

                # stop detection
                if stopdetection:
                    idx_stopwindow = [start_idx_abf - np.ceil(np.abs(t_win_stop[0])*ssr_abf),
                                      start_idx_abf + np.ceil(np.abs(t_win_stop[-1])*ssr_abf)]
                    if np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
                        continue

                # additional filters
                flag_filter_fail = False
                for additional_filter in additional_filters:
                    fail_flag = get_sig_ave_unit_check_filter(rec, additional_filter, start_idx,
                                                              of_end_idxs[of_stimids == stimid_special][i_stim])
                    if fail_flag:
                        flag_filter_fail = True
                        break
                if flag_filter_fail:
                    continue

                t0 = rec.abf.t[start_idx_abf]
                idx1 = int(start_idx)
                # idx0 = int(idx1 - np.ceil((np.abs(t_win_plot[0])+1) * ssr_tif))
                idx0 = int(idx1 + np.ceil((t_win_plot[0]-1) * ssr_tif))
                idx2 = int(idx1 + np.ceil(t_win_plot[1] * ssr_tif))
                idx3 = int(idx2 + np.ceil((t_win_plot[2]+1) * ssr_tif))

                t_im1, t_im0 = rec.ims[locnums[1]].t[idx0:idx3] - t0, rec.ims[locnums[0]].t[idx0:idx3] - t0
                if (t_im1[0] > t_interp[0]) | (t_im1[-1] < t_interp[-1]) | (t_im0[0] > t_interp[0]) | (t_im0[-1] < t_interp[-1]):
                    continue
                phase_im1, phase_im0 = chs[1].phase[idx0:idx3], chs[0].phase[idx0:idx3]
                f_im1 = interp1d(t_im1, fc.unwrap(phase_im1), kind='linear')
                f_im0 = interp1d(t_im0, fc.unwrap(phase_im0), kind='linear')
                phase_interp_im1 = fc.wrap(f_im1(t_interp)) if not add180_phase else fc.wrap(f_im1(t_interp)) + 180
                phase_interp_im0 = fc.wrap(f_im0(t_interp))
                dphase = fc.circsubtract(phase_interp_im1, phase_interp_im0)
                dphase_fly.append(dphase)

        if not np.sum(np.isnan(fc.circmean(dphase_fly, axis=0))):
            dphase_group.append(fc.circmean(dphase_fly, axis=0))

        if len(dphase_fly):
            # ax.plot(t_interp, fc.circmean(dphase_fly, axis=0), lw=lw_indi, zorder=2, alpha=alpha_indi, c=c_indi)
            ph.plot_noline_between_lims(ax, t_interp, fc.circmean(dphase_fly, axis=0), threshold=135, color=c_indi, alpha=alpha_indi, lw=lw_indi, zorder=2)
            # ax.plot(t_interp, fc.circmean(dphase_fly, axis=0), ls='none', marker='.', mfc='none', mec=c, zorder=2, ms=ms, alpha=alpha_indi)
    # ax.plot(t_interp, fc.circmean(dphase_group, axis=0), lw=2, zorder=3, c=ph.blue)
    # ph.plot_noline_between_lims(ax, t_interp, fc.circmean(dphase_group, axis=0), threshold=135, color=c, lw=lw, zorder=3)
    ax.plot(t_interp, fc.circmean(dphase_group, axis=0), ls='none', marker='o', mec='none', mfc=c, zorder=3, ms=ms)
    for vline in vlines:
        ax.axvline(vline, ls='--', lw=lw_dash, c=ph.grey9, alpha=.5)
        # ax.axvline(vline, ls='--', lw=lw_dash, c=ph.dred, alpha=.8)
    ax.axhline(0, ls='--', lw=lw_dash, c=ph.grey9, alpha=.6, zorder=1)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fs)
    if show_ylable:
        ax.set_ylabel('$\Delta \Theta$', rotation='horizontal', va='center', fontsize=fs)
    if show_timescale:
        ph.plot_x_scale_text(ax, bar_length=2, text='2s', y_text_lower=0.15, x_text_lefter=.05, fontsize=fs)

def plot_wba_ave_unit(recss, ax=False, stimid_special=1, t_win_plot=[1,11,0], vlines=[4,8], t_win_stop=[-2,14],
                      behtype='lmr', stim_method='dypos', show_axes=['left'], ylim=[40,90], yticks=[40,60,80],
                      xlabel='', show_ylable=True, show_timescale=True, fs=11, alpha_indi=0.3, lw=2, lw_indi=1,lw_dash=1):
    if not ax:
        _ = plt.figure(1, (3, 2))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    t_interp = np.arange(t_win_plot[0] - .3, (t_win_plot[1]+t_win_plot[2])+.3, 0.01)
    xlim = [t_win_plot[0], t_win_plot[1]+t_win_plot[2]]
    ph.adjust_spines(ax, show_axes, lw=1, ylim=ylim, yticks=yticks, xlim=xlim, xticks=[], pad=0)
    wba_group = []

    for fly in recss:
        wba_fly = []

        for rec in fly:
            if stim_method == 'dypos':
                of_stimids, of_start_idxs, _ = get_stim_info_20191007(rec)
            elif stim_method == 'pwm':
                of_stimids, of_start_idxs, _ = get_stim_info_20200117_pwmdc(rec)
            elif stim_method == 'dxpos':
                of_stimids, of_start_idxs, _ = get_stim_info_20200224_xposonly(rec)
            stop_inds = get_flight_stopping_inds(rec)
            ssr = rec.abf.subsampling_rate
            t = rec.abf.t

            for start_idx_tif in of_start_idxs[of_stimids==stimid_special]:
                start_idx = np.where(t < rec.ims[0].t[start_idx_tif])[0][-1]
                idx_stopwindow = [start_idx - np.ceil(np.abs(t_win_stop[0])*ssr),
                                  start_idx + np.ceil(np.abs(t_win_stop[-1])*ssr)]
                if not np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
                    t0 = t[start_idx]
                    idx1 = int(start_idx)
                    idx0 = int(idx1 + np.ceil((t_win_plot[0]-1) * ssr))
                    idx2 = int(idx1 + np.ceil(t_win_plot[1] * ssr))
                    idx3 = int(idx2 + np.ceil((t_win_plot[2]+1) * ssr))

                    ts = t[idx0:idx3] - t0
                    if (ts[0] > t_interp[0]) | (ts[-1] < t_interp[-1]):
                        continue
                    # wba = rec.abf.lmr if behtype == 'lmr' else (rec.abf.lwa + rec.abf.rwa)/2.
                    wba = rec.abf.lmr if behtype == 'lmr' else (rec.abf.lwa + rec.abf.rwa) / 1.
                    wba = wba[idx0:idx3]
                    f = interp1d(ts, wba, kind='linear')
                    wba_interp = f(t_interp)
                    wba_fly.append(wba_interp)

        if not np.sum(np.isnan(np.nanmean(wba_fly, axis=0))):
            wba_group.append(np.nanmean(wba_fly, axis=0))

    for wba in wba_group:
        ax.plot(t_interp, wba, color='black', alpha=alpha_indi, lw=lw_indi, zorder=2)
    ax.plot(t_interp, np.nanmean(wba_group, axis=0), lw=lw, zorder=3, c='black')

    c_lines = ph.grey5
    for vline in vlines:
        ax.axvline(vline, ls='--', lw=lw_dash, c=c_lines, alpha=.8)
    if behtype == 'lmr':
        ax.axhline(0, ls='--', lw=lw_dash, c=c_lines, alpha=.6, zorder=1)

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fs)
    if show_ylable:
        ax.set_ylabel(behtype, rotation='horizontal', va='center', fontsize=fs)
    if show_timescale:
        ph.plot_x_scale_text(ax, bar_length=2, text='2s', y_text_lower=0.15, x_text_lefter=.05, fontsize=fs)

def plot_phase_and_wba_ave_unit_2subgroup(recss, axs=[], stimid_special=1, t_win_plot=[1,11,0], vlines=[4,8], t_win_stop=[-2,14],
                      beh_range=1, beh_sigh=1, stim_method='dypos',
                      xlabel='', show_timescale=True, fs=11, alpha_indi=0.3, lw=2, lw_indi=1, lw_dash=1):
    if not len(axs):
        _ = plt.figure(1, (3, 4))
        gs = gridspec.GridSpec(2, 1)
        axs = [plt.subplot(gs[0, 0]), plt.subplot(gs[1, 0])]

    t_interp_beh = np.arange(t_win_plot[0] - .3, (t_win_plot[1]+t_win_plot[2])+.3, 0.01)
    t_interp_sig = np.arange(t_win_plot[0] - .3, (t_win_plot[1] + t_win_plot[2]) + .3, 0.1)
    xlim = [t_win_plot[0], t_win_plot[1]+t_win_plot[2]]
    dphase_group = []
    wba_group = []

    for fly in recss:
        dphase_fly = []
        wba_fly = []

        # first, calculate the behavior range index for each trial
        wba_aves = []
        for rec in fly:
            if stim_method == 'dypos':
                of_stimids, of_start_idxs, _ = get_stim_info_20191007(rec)
            elif stim_method == 'pwm':
                of_stimids, of_start_idxs, _ = get_stim_info_20200117_pwmdc(rec)
            elif stim_method == 'dxpos':
                of_stimids, of_start_idxs, _ = get_stim_info_20200224_xposonly(rec)
            stop_inds = get_flight_stopping_inds(rec)
            ssr_abf = rec.abf.subsampling_rate
            t = rec.abf.t
            for start_idx_tif in of_start_idxs[of_stimids==stimid_special]:
                start_idx_abf = np.where(t < rec.ims[0].t[start_idx_tif])[0][-1]
                idx_stopwindow = [start_idx_abf - np.ceil(np.abs(t_win_stop[0])*ssr_abf),
                                  start_idx_abf + np.ceil(np.abs(t_win_stop[-1])*ssr_abf)]
                if not np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
                    t0 = t[start_idx_abf]
                    idx00 = int(start_idx_abf)
                    idx0 = int(idx00 + np.ceil((t_win_plot[0] - 1) * ssr_abf))
                    idx1 = int(idx00 + np.ceil((vlines[0] - 1) * ssr_abf))
                    idx2 = int(idx00 + np.ceil(vlines[1] * ssr_abf))
                    idx3 = int(idx2 + np.ceil((t_win_plot[1] + 1) * ssr_abf))

                    ts = t[idx0:idx3] - t0
                    if (ts[0] > t_interp_beh[0]) | (ts[-1] < t_interp_beh[-1]):
                        continue
                    wba_aves.append(np.nanmean(rec.abf.lmr[idx1:idx2])-np.nanmean(rec.abf.lmr[idx0:idx1]))

        wba_aves = np.array(wba_aves)
        flags = wba_aves > np.mean(wba_aves) if beh_sigh else wba_aves <= np.mean(wba_aves)

        # second, extract data and plot
        flag_count = -1
        for rec in fly:
            if stim_method == 'dypos':
                of_stimids, of_start_idxs, _ = get_stim_info_20191007(rec)
            elif stim_method == 'pwm':
                of_stimids, of_start_idxs, _ = get_stim_info_20200117_pwmdc(rec)
            elif stim_method == 'dxpos':
                of_stimids, of_start_idxs, _ = get_stim_info_20200224_xposonly(rec)
            stop_inds = get_flight_stopping_inds(rec)
            ssr_tif = rec.ims[0].sampling_rate
            ssr_abf = rec.abf.subsampling_rate
            t = rec.abf.t
            for start_idx_tif in of_start_idxs[of_stimids == stimid_special]:
                start_idx_abf = np.where(t < rec.ims[0].t[start_idx_tif])[0][-1]
                idx_stopwindow = [start_idx_abf - np.ceil(np.abs(t_win_stop[0]) * ssr_abf),
                                  start_idx_abf + np.ceil(np.abs(t_win_stop[-1]) * ssr_abf)]
                if not np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
                    flag_count += 1

                    if flags[flag_count] == beh_range:
                        # calculate wba
                        t0 = t[start_idx_abf]
                        idx1 = int(start_idx_abf)
                        idx0 = int(idx1 + np.ceil((t_win_plot[0] - 1) * ssr_abf))
                        idx2 = int(idx1 + np.ceil(t_win_plot[1] * ssr_abf))
                        idx3 = int(idx2 + np.ceil((t_win_plot[2] + 1) * ssr_abf))
                        ts = t[idx0:idx3] - t0
                        if (ts[0] > t_interp_beh[0]) | (ts[-1] < t_interp_beh[-1]):
                            continue
                        wba = rec.abf.lmr[idx0:idx3]
                        f = interp1d(ts, wba, kind='linear')
                        wba_interp = f(t_interp_beh)
                        wba_fly.append(wba_interp)

                        # calculate phase
                        t0 = t[start_idx_abf]
                        idx1 = int(start_idx_tif)
                        idx0 = int(idx1 + np.ceil((t_win_plot[0] - 1) * ssr_tif))
                        idx2 = int(idx1 + np.ceil(t_win_plot[1] * ssr_tif))
                        idx3 = int(idx2 + np.ceil((t_win_plot[2] + 1) * ssr_tif))
                        t_im1, t_im0 = rec.ims[1].t[idx0:idx3] - t0, rec.ims[0].t[idx0:idx3] - t0
                        # print idx0, idx3
                        # print t_im1, t_im0
                        if (t_im1[0] > t_interp_sig[0]) | (t_im1[-1] < t_interp_sig[-1]) | (t_im0[0] > t_interp_sig[0]) | (
                                t_im0[-1] < t_interp_sig[-1]):
                            continue
                        phase_im1, phase_im0 = rec.ims[1].c1.phase[idx0:idx3], rec.ims[0].c1.phase[idx0:idx3]
                        f_im1 = interp1d(t_im1, fc.unwrap(phase_im1), kind='linear')
                        f_im0 = interp1d(t_im0, fc.unwrap(phase_im0), kind='linear')
                        phase_interp_im1 = fc.wrap(f_im1(t_interp_sig))
                        phase_interp_im0 = fc.wrap(f_im0(t_interp_sig))
                        dphase = fc.circsubtract(phase_interp_im1, phase_interp_im0)
                        dphase_fly.append(dphase)

        if not np.sum(np.isnan(np.nanmean(wba_fly, axis=0))):
            wba_group.append(np.nanmean(wba_fly, axis=0))
        if not np.sum(np.isnan(fc.circmean(dphase_fly, axis=0))):
            dphase_group.append(fc.circmean(dphase_fly, axis=0))

    for wba in wba_group:
        axs[1].plot(t_interp_beh, wba, color='black', alpha=alpha_indi, lw=lw_indi, zorder=2)
    axs[1].plot(t_interp_beh, np.nanmean(wba_group, axis=0), lw=lw, zorder=3, c='black')
    for dphase_fly in dphase_group:
        ph.plot_noline_between_lims(axs[0], t_interp_sig, dphase_fly, threshold=135, color=ph.blue,
                                alpha=alpha_indi, lw=lw_indi, zorder=2)
    axs[0].plot(t_interp_sig, fc.circmean(dphase_group, axis=0), lw=2, zorder=3, c=ph.blue)
    ph.adjust_spines(axs[0], [], lw=1, ylim=[-180,180], yticks=[], xlim=xlim, xticks=[], pad=0)
    ph.adjust_spines(axs[1], [], lw=1, ylim=[-30, 30], yticks=[], xlim=xlim, xticks=[], pad=0)

    for ax in axs:
        for vline in vlines:
            ax.axvline(vline, ls='--', lw=lw_dash, c=ph.dred, alpha=.8)
        ax.axhline(0, ls='--', lw=lw_dash, c=ph.dred, alpha=.6, zorder=1)

        if xlabel:
            ax.set_xlabel(xlabel, fontsize=fs)

        if show_timescale:
            ph.plot_x_scale_text(ax, bar_length=2, text='2s', y_text_lower=0.15, x_text_lefter=.05, fontsize=fs)

def plot_dphase_ave(recss, stimids=[4,3,2,1,6,5,4,7,8], t_win_plot=[-4,4,4], t_win_stop=[-8,8], vlines=[0,4],
                    ms_phase_ave=5, fs=11, xlabels=['-180','-120','-60','0','60','120','180','Lyaw','Ryaw'],
                    stimid_N=10, stim_method='dypos', add180_phase=False):
    ncol = len(stimids)
    _ = plt.figure(1, (2.5*ncol, 2))
    gs = gridspec.GridSpec(1, ncol)

    for col, stimid_special in enumerate(stimids):
        ax = plt.subplot(gs[0, col])
        show_axes = ['left'] if not col else []
        show_ylable = True if not col else False
        show_timescale = True if not col else False
        plot_dphase_ave_unit(recss, ax=ax, stimid_special=stimid_special, t_win_plot=t_win_plot, vlines=vlines,
                             t_win_stop=t_win_stop, stim_method=stim_method, add180_phase=add180_phase,
                             show_axes=show_axes, xlabel=xlabels[col], show_ylable=show_ylable,
                             ms=ms_phase_ave, show_timescale=show_timescale, stimid_N=stimid_N, fs=fs)

def plot_dphase_plus_wba_ave(group, fig_size=(12, 3.3), ncol=9, stimids=[4,3,2,1,6,5,4,7,8], stim_method='dypos',
                              t_win_plot=[1,10,0], vlines=[4,8], t_win_stop=[-2,14], ms_phase_ave=.8,
                             lw=2, lw_indi=0.5, c_indi=ph.grey7, alpha_indi=0.4, c='black', lw_dash=1, fs=7,):
    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(3, ncol, height_ratios=[1,1,1])

    rec = group[0][0]
    if stim_method == 'dypos':
        of_stimids, of_start_idxs, _ = get_stim_info_20191007(rec)
    elif stim_method == 'pwm':
        of_stimids, of_start_idxs, _ = get_stim_info_20200117_pwmdc(rec)
    elif stim_method == 'dxpos':
        of_stimids, of_start_idxs, _ = get_stim_info_20200224_xposonly(rec)

    for col in range(ncol):
        stimid_special = stimids[col]
        ax = plt.subplot(gs[0, col])
        plot_dphase_ave_unit(group, ax=ax, stimid_special=stimid_special, t_win_plot=t_win_plot, vlines=vlines,
                             t_win_stop=t_win_stop, stim_method=stim_method, show_axes=[], xlabel='', show_ylable=False,
                             show_timescale=False, ms=ms_phase_ave, lw_indi=lw_indi, c_indi=c_indi,
                             alpha_indi=alpha_indi,
                             c=c, fs=fs, lw_dash=lw_dash)

        ax = plt.subplot(gs[1, col])
        plot_wba_ave_unit(group, ax=ax, stimid_special=stimid_special, t_win_plot=t_win_plot, vlines=vlines,
                          t_win_stop=t_win_stop, behtype='lmr', stim_method=stim_method,
                          ylim=[-40,40], yticks=[-40,0,40], show_axes=[], xlabel='',
                          show_ylable=False, show_timescale=False, fs=fs, alpha_indi=alpha_indi, lw=lw, lw_indi=lw_indi,
                          lw_dash=lw_dash)

        ax = plt.subplot(gs[2, col])
        plot_wba_ave_unit(group, ax=ax, stimid_special=stimid_special, t_win_plot=t_win_plot, vlines=vlines,
                          t_win_stop=t_win_stop, behtype='lpr', stim_method=stim_method,
                          ylim=[110,150], yticks=[110,130,150], show_axes=[], xlabel='',
                          show_ylable=False, show_timescale=False, fs=fs, alpha_indi=alpha_indi, lw=lw, lw_indi=lw_indi,
                          lw_dash=lw_dash)

def plot_dphase_plus_wba_ave_2subgroup(group, fig_size=(8, 2.2), ncol=6, stimids=[3,2,6,5,7,8], beh_sighs=[1,1,0,0,1,0], stim_method='dypos',
                              t_win_plot=[1,10,0], vlines=[4,8], t_win_stop=[-2,14], beh_range=1,
                              lw=2, lw_indi=0.5, alpha_indi=0.4, lw_dash=1, fs=7,):
    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(2, ncol)

    rec = group[0][0]
    if stim_method == 'dypos':
        of_stimids, of_start_idxs, _ = get_stim_info_20191007(rec)
    elif stim_method == 'pwm':
        of_stimids, of_start_idxs, _ = get_stim_info_20200117_pwmdc(rec)
    elif stim_method == 'dxpos':
        of_stimids, of_start_idxs, _ = get_stim_info_20200224_xposonly(rec)

    for col in range(ncol):
        stimid_special = stimids[col]
        axs = [plt.subplot(gs[0, col]), plt.subplot(gs[1, col])]
        plot_phase_and_wba_ave_unit_2subgroup(group, axs=axs, stimid_special=stimid_special, t_win_plot=t_win_plot, vlines=vlines,
                            t_win_stop=t_win_stop, stim_method=stim_method, show_timescale=False, fs=fs,
                            alpha_indi=alpha_indi, lw=lw, lw_indi=lw_indi, lw_dash=lw_dash,
                            beh_range=beh_range, beh_sigh=beh_sighs[col], )

def plot_dphase_ave_multirecss(groups, stimid=8, t_win_plot=[-5,17,0], vlines=[0,12], t_win_stop=[-1,18], lw_mean=2,
                               fs=11, alpha_indi=0.2, xlabels=['left', 'right', 'both'], show_ylable=True,
                               show_axes=['left'], stim_method='dypos'):
    ncol = len(groups)
    _ = plt.figure(1, (2.5*ncol, 2))
    gs = gridspec.GridSpec(1, ncol)

    for col, recss in enumerate(groups):
        ax = plt.subplot(gs[0, col])
        _show_axes = show_axes if not col else []
        _show_ylable = show_ylable if not col else False
        show_timescale = True if not col else False
        plot_dphase_ave_unit(recss, ax=ax, stimid_special=stimid, t_win_plot=t_win_plot, vlines=vlines,
                             t_win_stop=t_win_stop, stim_method=stim_method,
                             show_axes=_show_axes, xlabel=xlabels[col], show_ylable=_show_ylable,
                             show_timescale=show_timescale, fs=fs, alpha_indi=alpha_indi)


def get_dphase_ave_unit(recss, stimid_special=1, t_before_end=-2, t_end=0, t_win_stop=[-12,4], t_win_interp=[-3,1],
                        stim_method='dypos', flipsign_dphase=False, add180_phase1=False, add180_phase2=False, stimid_N=10):
    # return xs
    t_interp = np.arange(t_before_end, t_end, 0.1)
    dphase_group = []

    for fly in recss:
        dphase_fly = []

        for rec in fly:
            if stim_method == 'dypos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20191007(rec, stimid_N=stimid_N)
            elif stim_method == 'dxpos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20200224_xposonly(rec)
            stop_inds = get_flight_stopping_inds(rec)
            ssr_tif = rec.ims[0].sampling_rate
            ssr_abf = rec.abf.subsampling_rate

            for end_idx in of_end_idxs[of_stimids==stimid_special]:
                end_idx_abf = np.where(rec.abf.t < rec.fb.t[end_idx])[0][-1]
                idx_stopwindow = [end_idx_abf - np.ceil(np.abs(t_win_stop[0])*ssr_abf),
                                  end_idx_abf + np.ceil(np.abs(t_win_stop[-1])*ssr_abf)]

                if not np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
                # if 1:
                    t0 = rec.abf.t[end_idx_abf]
                    idx_end = int(end_idx)
                    idx1 = int(idx_end + np.ceil(t_win_interp[1] * ssr_tif))
                    idx0 = int(idx_end - np.ceil(np.abs(t_win_interp[0]) * ssr_tif))
                    t_eb, t_fb = rec.ims[1].t[idx0:idx1] - t0, rec.ims[0].t[idx0:idx1] - t0
                    phase_eb, phase_fb = rec.ims[1].c1.phase[idx0:idx1]*0, rec.ims[0].c1.phase[idx0:idx1]
                    f_fb = interp1d(t_fb, fc.unwrap(phase_fb), kind='linear')
                    f_eb = interp1d(t_eb, fc.unwrap(phase_eb), kind='linear')
                    phase_interp_fb = fc.wrap(f_fb(t_interp)) if not add180_phase1 else fc.wrap(f_fb(t_interp)) + 180
                    phase_interp_eb = fc.wrap(f_eb(t_interp)) if not add180_phase2 else fc.wrap(f_eb(t_interp)) + 180
                    if flipsign_dphase:
                        dphase = fc.circsubtract(phase_interp_eb, phase_interp_fb)
                    else:
                        dphase = fc.circsubtract(phase_interp_fb, phase_interp_eb)
                    dphase_fly.append(fc.circmean(dphase, axis=0))

        dphase_group.append(fc.circmean(dphase_fly, axis=0))
    return dphase_group

def barplot_dphase_ave_20191105(recss, fig_size=(2, 3.3), stimids=[4,3,2,1,6,5,4,7,8], t_before_end=-2, t_end=0, t_win_stop=[-12,4],
                                t_win_interp=[-3,1], flipsign_dphase=True, label_axeratio=-.2, fs=11,
                                group_labels=['-180','-120','-60','0','60','120','180','Lyaw','Ryaw'], stimid_N=10,
                                ms_indi=2, add180_phase1=False, add180_phase2=False, print_data=False):
    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])

    xss, x2ss, x3ss = [], [], []
    for stimid_special in stimids:
        xs = get_dphase_ave_unit(recss, stimid_special=stimid_special, t_before_end=t_before_end, t_end=t_end,
                                   t_win_stop=t_win_stop, t_win_interp=t_win_interp, flipsign_dphase=flipsign_dphase,
                                   add180_phase1=add180_phase1, add180_phase2=add180_phase2, stimid_N=stimid_N)
        xss.append(xs)
        x2ss.append(np.array(xs)+360)
        x3ss.append(np.array(xs) - 360)
    group_info = [[i] for i in range(len(stimids))]
    if print_data:
        ystr = ''
        for y in fc.circmean(xss, axis=1):
            ystr = ystr + '%.3f' % y + ', '
        print 'mean: %s' % ystr

        ystr = ''
        for xs in xss:
            ystr = ystr + '%.3f' % (fc.circstd(xs)/np.sqrt(len(xs))) + ', '
        print 'sem: %s' % ystr

    ph.errorbar_lc_xss(xss=xss, ax=ax, group_info=group_info, group_labels=group_labels, colors=[ph.grey2],
                       ylabel='', yticks=[], ylim=[-210, 540], yticks_minor=[], fs=fs,
                       ro_x=45, label_axeratio=label_axeratio, rotate_ylabel=False, plot_bar=False, ms_indi=ms_indi,
                       subgroup_samefly=False, circstat=True, legend_list=group_labels, show_flynum=False, show_nancol=False)

    ph.errorbar_lc_xss(xss=x2ss, ax=ax, group_info=group_info, group_labels=group_labels, colors=[ph.grey2],
                       ylabel='$\Delta$phase', yticks=np.arange(-180, 541, 180), ylim=[-210, 540], yticks_minor=[],
                       fs=fs, ro_x=45, label_axeratio=label_axeratio, rotate_ylabel=False, plot_bar=False, ms_indi=ms_indi,
                       subgroup_samefly=False, circstat=True, circ_low=180, circ_high=540,
                       legend_list=group_labels, show_flynum=False, show_nancol=False)

    ph.errorbar_lc_xss(xss=x3ss, ax=ax, group_info=group_info, group_labels=group_labels, colors=[ph.grey2],
                       ylabel='$\Delta$phase', yticks=np.arange(-180, 541, 180), ylim=[-210, 540], yticks_minor=[],
                       fs=fs, ro_x=45, label_axeratio=label_axeratio, rotate_ylabel=False, plot_bar=False,
                       ms_indi=ms_indi,
                       subgroup_samefly=False, circstat=True, circ_low=-540, circ_high=-180,
                       legend_list=group_labels, show_flynum=False, show_nancol=False)

def barplot_dphase_ave_20200414(recss, fig_size=(2, 2), stimids=[4,3,2,1,6,5,4,7,8], t_before_end=-2, t_end=0, t_win_stop=[-12,4],
                                t_win_interp=[-3,1], flipsign_dphase=True, label_axeratio=-.2, fs=11, stim_method='dypos',
                                group_labels=['-180','-120','-60','0','60','120','180','Lyaw','Ryaw'],
                                ms_indi=2, add180_phase1=False, add180_phase2=False):
    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])

    xss, x2ss, x3ss = [], [], []
    for stimid_special in stimids:
        xs = get_dphase_ave_unit(recss, stimid_special=stimid_special, t_before_end=t_before_end, t_end=t_end,
                                 t_win_stop=t_win_stop, t_win_interp=t_win_interp, stim_method=stim_method,
                                 flipsign_dphase=flipsign_dphase, add180_phase1=add180_phase1, add180_phase2=add180_phase2)
        xss.append(xs)
        x2ss.append(np.array(xs) + 360)
        x3ss.append(np.array(xs) - 360)
    group_info = [[i] for i in range(len(stimids))]

    ph.errorbar_lc_xss(xss=xss, ax=ax, group_info=group_info, group_labels=group_labels, colors=[ph.grey2],
                       ylabel='', yticks=[-210,540], ylim=[-210,540], yticks_minor=[], fs=fs,
                       ro_x=45, label_axeratio=label_axeratio, rotate_ylabel=False, plot_bar=False, ms_indi=ms_indi,
                       subgroup_samefly=False, circstat=True, legend_list=group_labels, show_flynum=False,
                       show_nancol=False)

    ph.errorbar_lc_xss(xss=x2ss, ax=ax, group_info=group_info, group_labels=group_labels, colors=[ph.grey2],
                       ylabel='', yticks=[], ylim=[-210, 540], yticks_minor=[], fs=fs,
                       ro_x=45, label_axeratio=label_axeratio, rotate_ylabel=False, plot_bar=False, ms_indi=ms_indi,
                       subgroup_samefly=False, circstat=True, circ_low=180, circ_high=540, legend_list=group_labels, show_flynum=False,
                       show_nancol=False)

    ph.errorbar_lc_xss(xss=x3ss, ax=ax, group_info=group_info, group_labels=group_labels, colors=[ph.grey2],
                       ylabel='', yticks=np.arange(-180, 181, 180), ylim=[-210, 540], yticks_minor=[], fs=fs,
                       ro_x=45, label_axeratio=label_axeratio, rotate_ylabel=False, plot_bar=False, ms_indi=ms_indi,
                       subgroup_samefly=False, circstat=True, circ_low=-540, circ_high=-180,legend_list=group_labels, show_flynum=False,
                       show_nancol=False)

def get_dphase_ave_2ch_unit(recss, locnums=[0,1], chnums=[1,1], stimid_special=1, t_before_end=-2, t_end=0,
                            t_win_stop=[-12,4], t_win_interp=[-3,1], stim_method='dypos', flipsign_dphase=False,
                            add180_phase1=False, add180_phase2=False, stopdetection=True, stimid_N=10,):
    # return xs
    t_interp = np.arange(t_before_end, t_end, 0.1)
    dphase_group = []

    for fly in recss:
        dphase_fly = []

        for rec in fly:
            if stim_method == 'dypos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20191007(rec, stimid_N=stimid_N)
            elif stim_method == 'dxpos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20200224_xposonly(rec)
            stop_inds = get_flight_stopping_inds(rec)
            ssr_tif = rec.ims[0].sampling_rate
            ssr_abf = rec.abf.subsampling_rate
            chs = [getattr(rec.ims[locnums[0]], 'c%i' % chnums[0]), getattr(rec.ims[locnums[1]], 'c%i' % chnums[1])]

            for end_idx in of_end_idxs[of_stimids==stimid_special]:
                end_idx_abf = np.where(rec.abf.t < rec.ims[0].t[end_idx])[0][-1]
                if stopdetection:
                    idx_stopwindow = [end_idx_abf - np.ceil(np.abs(t_win_stop[0])*ssr_abf),
                                      end_idx_abf + np.ceil(np.abs(t_win_stop[-1])*ssr_abf)]
                    stop_flag = np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1]))
                else:
                    stop_flag = False
                if not stop_flag:
                    t0 = rec.abf.t[end_idx_abf]
                    idx_end = int(end_idx)
                    idx1 = int(idx_end + np.ceil(t_win_interp[1] * ssr_tif))
                    idx0 = int(idx_end - np.ceil(np.abs(t_win_interp[0]) * ssr_tif))
                    t_eb, t_fb = rec.ims[locnums[1]].t[idx0:idx1] - t0, rec.ims[locnums[0]].t[idx0:idx1] - t0
                    phase_eb, phase_fb = chs[1].phase[idx0:idx1], chs[0].phase[idx0:idx1]
                    f_fb = interp1d(t_fb, fc.unwrap(phase_fb), kind='linear')
                    f_eb = interp1d(t_eb, fc.unwrap(phase_eb), kind='linear')
                    phase_interp_fb = fc.wrap(f_fb(t_interp)) if not add180_phase1 else fc.wrap(f_fb(t_interp)) + 180
                    phase_interp_eb = fc.wrap(f_eb(t_interp)) if not add180_phase2 else fc.wrap(f_eb(t_interp)) + 180
                    if flipsign_dphase:
                        dphase = fc.circsubtract(phase_interp_eb, phase_interp_fb)
                    else:
                        dphase = fc.circsubtract(phase_interp_fb, phase_interp_eb)
                    dphase_fly.append(fc.circmean(dphase, axis=0))

        dphase_group.append(fc.circmean(dphase_fly, axis=0))
    return dphase_group


def barplot_dphase_ave_2ch_20210324(recss, fig_size=(2, 2), locnums=[0,0], chnums=[1,2], stimids=[4,3,2,1,6,5,4,7,8],
                                t_before_end=-2, t_end=0, t_win_stop=[-12,4], stopdetection=True,
                                t_win_interp=[-3,1], flipsign_dphase=True, label_axeratio=-.2, fs=11, stim_method='dypos',
                                group_labels=['-180','-120','-60','0','60','120','180','Lyaw','Ryaw'],
                                print_indi=False, print_mean=False, stimid_N=10,
                                ms_indi=2, add180_phase1=False, add180_phase2=False,yticks=[-180,540], ylim=[-210,570],):
    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])

    xss, x2ss, x3ss = [], [], []
    for stimid_special in stimids:
        xs = get_dphase_ave_2ch_unit(recss, locnums=locnums, chnums=chnums, stimid_special=stimid_special, t_before_end=t_before_end, t_end=t_end,
                                 t_win_stop=t_win_stop, t_win_interp=t_win_interp, stim_method=stim_method, stopdetection=stopdetection,
                                 flipsign_dphase=flipsign_dphase, add180_phase1=add180_phase1, add180_phase2=add180_phase2, stimid_N=stimid_N)
        xss.append(xs)
        x2ss.append(np.array(xs) + 360)
        x3ss.append(np.array(xs) - 360)
    group_info = [[i] for i in range(len(stimids))]


    ph.errorbar_lc_xss(xss=xss, ax=ax, group_info=group_info, group_labels=group_labels, colors=[ph.grey2],
                       ylabel='', yticks=yticks, ylim=ylim, yticks_minor=[], fs=fs,
                       ro_x=45, label_axeratio=label_axeratio, rotate_ylabel=False, plot_bar=False, ms_indi=ms_indi,
                       subgroup_samefly=False, circstat=True, legend_list=group_labels, show_flynum=False,
                       show_nancol=False)
    if print_indi:
        print 'indi:'
        for xs in xss:
            ystr = ''
            for y in xs:
                ystr = ystr + '%.3f' % y + ', '
            print ystr
    if print_mean:
        print 'mean:'
        ystr = ''
        for y in fc.circmean(xss, axis=1):
            ystr = ystr + '%.3f' % y + ', '
        print ystr
    ph.errorbar_lc_xss(xss=x2ss, ax=ax, group_info=group_info, group_labels=group_labels, colors=[ph.grey2],
                       ylabel='', yticks=[], ylim=[-210, 540], yticks_minor=[], fs=fs,
                       ro_x=45, label_axeratio=label_axeratio, rotate_ylabel=False, plot_bar=False, ms_indi=ms_indi,
                       subgroup_samefly=False, circstat=True, circ_low=180, circ_high=540, legend_list=group_labels, show_flynum=False,
                       show_nancol=False)

    ph.errorbar_lc_xss(xss=x3ss, ax=ax, group_info=group_info, group_labels=group_labels, colors=[ph.grey2],
                       ylabel='', yticks=np.arange(-180, 181, 180), ylim=[-240, 540], yticks_minor=[], fs=fs,
                       ro_x=45, label_axeratio=label_axeratio, rotate_ylabel=False, plot_bar=False, ms_indi=ms_indi,
                       subgroup_samefly=False, circstat=True, circ_low=-540, circ_high=-180,legend_list=group_labels, show_flynum=False,
                       show_nancol=False)

    return xss


def get_phase_and_td_unit(recss, stimids=[1,2,3,4,5,6], of_offsets=[0,-60,-120,180,120,60], binnum=36, allo=True,
                          t_before_end=-6.51, t_end=-4.1, t_win_stop=[-2,13], flipsign_dphase=False, add180_phase=False):
    # return xs
    y_group = []

    for fly in recss:
        pfr_fly = []
        td_fly = []

        for i_stim, stimid_special in enumerate(stimids):
            for rec in fly:
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20191007(rec)
                stop_inds = get_flight_stopping_inds(rec)
                ssr_tif = rec.ims[0].sampling_rate
                ssr_abf = rec.abf.subsampling_rate

                for end_idx in of_end_idxs[of_stimids == stimid_special]:
                    end_idx_abf = np.where(rec.abf.t < rec.fb.t[end_idx])[0][-1]
                    idx_stopwindow = [end_idx_abf - np.ceil(np.abs(t_win_stop[0])*ssr_abf),
                                      end_idx_abf + np.ceil(np.abs(t_win_stop[-1])*ssr_abf)]

                    # if not np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
                    if 1:
                        idx_end = int(end_idx)
                        idx0 = int(idx_end + np.ceil(t_before_end * ssr_tif))
                        idx1 = int(idx_end + np.ceil(t_end * ssr_tif))
                        phase_eb, phase_fb = rec.ims[1].c1.phase[idx0:idx1], rec.ims[0].c1.phase[idx0:idx1]
                        addphase = 0 if not add180_phase else 180
                        if not flipsign_dphase:
                            pfr_fly.extend(fc.wrap(phase_fb + addphase))
                            if allo:
                                td_fly.extend(fc.wrap(phase_eb - of_offsets[i_stim]))
                            else:
                                td_fly.extend(phase_eb*0 - of_offsets[i_stim])
                        else:
                            pfr_fly.extend(fc.wrap(phase_eb + addphase))
                            if allo:
                                td_fly.extend(fc.wrap(phase_fb - of_offsets[i_stim]))
                            else:
                                td_fly.extend(phase_fb*0 - of_offsets[i_stim])

        x, ymean, _ = fc.scatter_to_circ_errorbar(x_input=td_fly, y_input=pfr_fly, binx=binnum, xmin=-180, xmax=180)
        y_group.append(ymean)

    return x, y_group

def plot_phase_and_td_unit(recss, ax=False, stimids=[1,2,3,4,5,6], of_offsets=[0,-60,-120,180,120,60], binnum=36,
                           t_before_end=-6.51, t_end=-4.1, t_win_stop=[-2,13], fs=8, show_ylabel=True, show_xlabel=True,
                           show_axes=['left', 'bottom'], lim=[-225,225], lw_indi=1, alpha_indi=.3, lw=2, ms_indi=3,
                           ms=5, c_indi=ph.grey5, c='black', flipsign_dphase=False, add180_phase=False, allo=True,):
    ph.set_fontsize(fs)
    if not ax:
        _ = plt.figure(1, (4, 4))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    x, y_group = get_phase_and_td_unit(recss, stimids=stimids, of_offsets=of_offsets, binnum=binnum,
                        t_before_end=t_before_end, t_end=t_end, t_win_stop=t_win_stop,
                        flipsign_dphase=flipsign_dphase, add180_phase=add180_phase, allo=allo)
    for y in y_group:
        # ax.plot(x, y-360, ls='none', marker='o', mfc='none', mec=c_indi, zorder=2, ms=ms_indi, alpha=alpha_indi, mew=.4)
        ax.plot(x, y, ls='none', marker='o', mfc='none', mec=c_indi, zorder=2, ms=ms_indi, alpha=alpha_indi, mew=.4)
        # ax.plot(x, y+360, ls='none', marker='o', mfc='none', mec=c_indi, zorder=2, ms=ms_indi, alpha=alpha_indi, mew=.4)
    # ax.plot(x, fc.circnanmean(y_group, axis=0, nan_policy='omit')-360, ls='none', marker='o', mec='none', mfc=c, zorder=2, ms=ms)
    ax.plot(x, fc.circnanmean(y_group, axis=0, nan_policy='omit'), ls='none', marker='o', mec='none', mfc=c, zorder=2, ms=ms)
    # ax.plot(x, fc.circnanmean(y_group, axis=0, nan_policy='omit')+360, ls='none', marker='o', mec='none', mfc=c, zorder=2, ms=ms)

    ax.set_aspect('equal', 'datalim')
    ph.adjust_spines(ax, show_axes, lw=1, xlim=lim, ylim=lim,
                     xticks=np.arange(-180, 181, 90), yticks=np.arange(-180, 181, 90))
    # ax.plot([-180,180], [-180,180], lw=1, ls='--', alpha=.5, c=ph.red)

    if show_xlabel:
        ax.set_xlabel('world-centric traveling direction (deg.)')
    if show_ylabel:
        ax.set_ylabel('PFR phase (deg.)', rotation=0)




def get_phase_during_flight(rec, im_num=1, ch_num=1,):
    inds_flight = get_2p_ind_flight(rec)
    ch = getattr(rec.ims[im_num], 'c%i' % ch_num)
    phase = fc.wrap(ch.phase)
    return phase[inds_flight]

def plot_scatter_two_phases(recss, ax=False, im_num=[0,0], ch_num=[2,1], binnum=32,
                            xlabel='\Theta neurons 1', ylabel='\Theta neurons2', ylabel_axeratio=-.1,
                            show_axes=['bottom', 'left'], c_indi=ph.nblue, alpha_indi=.15, fs=10,):
    ph.set_fontsize(fs)
    if not ax:
        _ = plt.figure(1, (2, 4))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    ys = []
    for fly in recss:
        phase1, phase2 = [], []

        for rec in fly:
            phase1.extend(get_phase_during_flight(rec, im_num=im_num[0], ch_num=ch_num[0]))
            phase2.extend(get_phase_during_flight(rec, im_num=im_num[1], ch_num=ch_num[1]))

            ax.plot(phase1, phase2, c=c_indi, ls='none', marker='o', ms=3, mec='none', mfc=c_indi, alpha=alpha_indi)
            ax.plot(phase1, np.array(phase2) + 360., c=c_indi, ls='none', marker='o', ms=3, mec='none', mfc=c_indi, alpha=alpha_indi)

        if len(phase1):
            x, _ys, _ = fc.scatter_to_circ_errorbar(phase1, phase2, binnum, -180, 180)
            ys.append(_ys)

    # for y in ys:
        # ax.plot(x, y, c=c_indi, lw=lw_indi, alpha=alpha_indi)
        # ax.plot(x, y, c=c_indi, ls='none', marker='o', ms=3, mec='none', mfc='black', alpha=alpha_indi)
        # ax.plot(x, y + 360., c=c_indi, ls='none', marker='o', ms=3, mec='none', mfc='black', alpha=alpha_indi)
    ax.plot(x, fc.circmean(ys, axis=0), c=c_indi, ls='none', marker='o', ms=4, mec='none', mfc='black')
    ax.plot(x, fc.circmean(ys, axis=0) + 360., c=c_indi, ls='none', marker='o', ms=4, mec='none', mfc='black')

    ax.set_aspect('equal', 'datalim')
    ph.adjust_spines(ax, show_axes, lw=1, xlim=[-181, 180], ylim=[-180, 540],
                     xticks=np.arange(-180, 181, 90), yticks=np.arange(-180, 541, 90))

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation=0)
    ax.get_yaxis().set_label_coords(ylabel_axeratio, 0.5)

def plot_wba_dltphase_xcorr_two_phase0(recss, ax=False, fig_size=(2,2), twin=[-2,2], dt=0.1, min_contig_s=1,
                         xticks=[-2,0,2], ylim=[-.1,.4], yticks=[0,.2,.4], lw=2, lw_indi=1, alpha_indi=.3, fs=8):
    ph.set_fontsize(fs)
    if not ax:
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    t_interp = np.arange(twin[0] - dt * 2, twin[1] + dt * 2, dt)
    ys = []
    for recs in recss:
        ys_fly = []
        for rec in recs:
            min_contig_nframe = np.ceil(rec.ims[0].sampling_rate * min_contig_s)
            lmr = rec.subsample('lmr')
            dltphase = fc.circsubtract(rec.eb.c1.phase, rec.fb.c1.phase)
            flight_inds = get_2p_ind_flight(rec)
            flight_idxss = fc.get_contiguous_inds(flight_inds, trim_left=0, keep_ends=True,
                                                  min_contig_len=min_contig_nframe)
            for flight_idxs in flight_idxss:
                t, y = fc.xcorr(lmr[flight_idxs], dltphase[flight_idxs], rec.fb.sampling_period)
                f = interp1d(t, y, kind='linear')
                ys_fly.append(f(t_interp))

        if len(ys_fly):
            ys.append(np.mean(ys_fly, axis=0))

    for y in ys:
        ax.plot(t_interp, y, lw=lw_indi, c='black', alpha=alpha_indi, zorder=2)
    ax.plot(t_interp, np.nanmean(ys, axis=0), lw=lw, c='black', zorder=3)
    ax.axvline(0, ls='--', c=ph.grey5, zorder=1)
    ph.adjust_spines(ax, ['left','bottom'], lw=.3, xlim=twin, ylim=ylim, xticks=xticks, yticks=yticks)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('xcorr', rotation=0)

def plot_wba_dltphase_xcorr_two_phase(recss, ax=False, fig_size=(2,2), twin=[-2,2], dt=0.1, min_contig_s=1, filters=[],
                         xticks=[-2,0,2], ylim=[-.1,.4], yticks=[0,.2,.4], lw=2, lw_indi=1, alpha_indi=.3, fs=8):
    ph.set_fontsize(fs)
    if not ax:
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    t_interp_xc = np.arange(twin[0] - dt * 2, twin[1] + dt * 2, dt)
    ys = []
    for recs in recss:
        ys_fly = []
        for rec in recs:
            min_contig_nframe = np.ceil(rec.ims[0].sampling_rate * min_contig_s)
            lmr = rec.abf.lmr
            t_abf = rec.abf.t
            f_beh = interp1d(t_abf, lmr, kind='linear')
            t_tif = rec.eb.t
            dltphase = fc.circsubtract(rec.eb.c1.phase, rec.fb.c1.phase)
            flight_inds = get_2p_ind_flight(rec)
            for filter_ in filters:
                filter_name, filter_range = filter_
                sig_filter = rec.subsample(filter_name)
                flight_inds = flight_inds & (sig_filter > filter_range[0]) & (sig_filter < filter_range[1])
            flight_idxss = fc.get_contiguous_inds(flight_inds, trim_left=0, keep_ends=True,
                                                  min_contig_len=min_contig_nframe)
            for flight_idxs in flight_idxss:
                t, y = fc.xcorr(f_beh(t_tif[flight_idxs]), dltphase[flight_idxs], rec.fb.sampling_period)
                f = interp1d(t, y, kind='linear')
                ys_fly.append(f(t_interp_xc))

        if len(ys_fly):
            ys.append(np.mean(ys_fly, axis=0))

    for y in ys:
        ax.plot(t_interp_xc, y, lw=lw_indi, c='black', alpha=alpha_indi, zorder=2)
    ax.plot(t_interp_xc, np.nanmean(ys, axis=0), lw=lw, c='black', zorder=3)
    ax.axvline(0, ls='--', c=ph.grey5, zorder=1, lw=lw_indi)
    ph.adjust_spines(ax, ['left', 'bottom'], lw=.3, xlim=twin, ylim=ylim, xticks=xticks, yticks=yticks)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('xcorr', rotation=0)

def plot_wba_dltphase_ave_two_phase(recss, ax=False, fig_size=(2,2), xlim=[-20,20], binnum=20, filters=[],
                         xticks=[-20,0,20], ylim=[-50,50], yticks=[-50,0,50], lw=2, lw_indi=1, alpha_indi=.3, fs=8):
    ph.set_fontsize(fs)
    if not ax:
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    ys = []
    for recs in recss:
        beh_fly = []
        sig_fly = []
        for rec in recs:
            lmr = rec.subsample('lmr')
            dltphase = fc.circsubtract(rec.eb.c1.phase, rec.fb.c1.phase)
            flight_inds = get_2p_ind_flight(rec)
            for filter_ in filters:
                filter_name, filter_range = filter_
                sig_filter = rec.subsample(filter_name)
                flight_inds = flight_inds & (sig_filter > filter_range[0]) & (sig_filter < filter_range[1])
            beh_fly.extend(lmr[flight_inds]), sig_fly.extend(dltphase[flight_inds])
        x, y_mean, _ = fc.scatter_to_errorbar(x_input=beh_fly, y_input=sig_fly, binx=binnum, xmin=xlim[0], xmax=xlim[1])
        if len(y_mean):
            ys.append(y_mean)

    for y in ys:
        ax.plot(x, y, lw=lw_indi, c='black', alpha=alpha_indi, zorder=2)
    ax.plot(x, np.nanmean(ys, axis=0), lw=lw, c='black', zorder=3)
    ax.axvline(0, ls='--', c=ph.grey5, zorder=1, lw=lw_indi)
    ph.adjust_spines(ax, ['left', 'bottom'], lw=.3, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)

    ax.set_xlabel('lmr (deg)')
    ax.set_ylabel('$\Delta$phase', rotation=0)



# single neural activity level
def get_sig_(rec, loc_num, sig_type):
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

def get_sig_ave_unit_check_filter(rec, filter, idx0, idx1):
    filter_type, filter_value = filter
    if filter_type == 'dypos':
        threshold = .05
        ystim_uw = rec.subsample('ystim_uw')
        dystim = np.gradient(ystim_uw)
        dymean = np.mean(np.abs(dystim[idx0:idx1]))
        if dymean > threshold:
            flag_fail = 0 if filter_value == 1 else 1
        else:
            flag_fail = 0 if filter_value == 0 else 1
    return flag_fail

def get_sig_ave_unit(recss, stimid_special=1, loc_num=0, sig_type='mean_az', t_win_plot=[-4,4,4], t_win_stop=[-8,8],
                     stim_method='pwm', stop_detection=False, additional_filters=[]):

    t_interp = np.arange(t_win_plot[0] - .3, (t_win_plot[1]+t_win_plot[2])+.3, 0.05)
    xlim = [t_win_plot[0], t_win_plot[1]+t_win_plot[2]]
    sig_group = []

    for fly in recss:
        sig_fly = []

        for rec in fly:
            if stim_method == 'dypos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20191007(rec)
            elif stim_method == 'pwm':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20200117_pwmdc(rec)
            elif stim_method == 'dxpos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20200224_xposonly(rec)
            stop_inds = get_flight_stopping_inds(rec)
            ssr_tif = rec.ims[0].sampling_rate
            ssr_abf = rec.abf.subsampling_rate

            for i_stim, start_idx in enumerate(of_start_idxs[of_stimids==stimid_special]):
                start_idx_abf = np.where(rec.abf.t < rec.ims[0].t[start_idx])[0][-1]

                if stop_detection:
                    idx_stopwindow = [start_idx_abf - np.ceil(np.abs(t_win_stop[0])*ssr_abf),
                                      start_idx_abf + np.ceil(np.abs(t_win_stop[-1])*ssr_abf)]
                    if np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
                        continue

                flag_filter_fail = False
                for additional_filter in additional_filters:
                    fail_flag = get_sig_ave_unit_check_filter(rec, additional_filter, start_idx, of_end_idxs[of_stimids==stimid_special][i_stim])
                    if fail_flag:
                        flag_filter_fail = True
                        break
                if flag_filter_fail:
                    continue

                t0 = rec.abf.t[start_idx_abf]
                idx1 = int(start_idx)
                idx0 = int(idx1 + np.ceil((t_win_plot[0]-1) * ssr_tif))
                idx2 = int(idx1 + np.ceil(t_win_plot[1] * ssr_tif))
                idx3 = int(idx2 + np.ceil((t_win_plot[2]+1) * ssr_tif))

                t_im = rec.ims[0].t[idx0:idx3] - t0
                if (t_im[0] > t_interp[0]) | (t_im[-1] < t_interp[-1]):
                    continue
                sig = get_sig_(rec, loc_num, sig_type)[idx0:idx3]
                _f = interp1d(t_im, sig, kind='linear')
                sig_interp = fc.wrap(_f(t_interp))
                sig_fly.append(sig_interp)

        if not np.sum(np.isnan(np.nanmean(sig_fly, axis=0))):
            sig_group.append(np.nanmean(sig_fly, axis=0))

    return t_interp, sig_group

def plot_sig_ave_unit(recss, ax=False, sig_type='rmlz', stimid_special=1, t_win_plot=[-4,4,4], stop_detection=False,
                      t_win_stop=[-8,8], stim_method='pwm', loc_num=0, lw=2, lw_indi=1, alpha_indi=.4, alpha_sem=.2,
                      color='black', xlabel='time (s)', ylabel='rmlz', show_axes=['left'],
                      xticks=[-4,0,4,8], ylim=[-1,1], yticks=[-1,0,1], show_xscale=True, show_yscale=False,
                      label_axeratio=-.25, fs=10, show_stimid_special=True, show_indi=True, show_flynum=True,
                      additional_filters=[]):

    if not ax:
        _ = plt.figure(1, (3, 2))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    xlim = [t_win_plot[0], t_win_plot[1]+t_win_plot[2]]
    xs, ys = get_sig_ave_unit(recss, sig_type=sig_type, stimid_special=stimid_special, stim_method=stim_method,
                              t_win_plot=t_win_plot, t_win_stop=t_win_stop, loc_num=loc_num,
                              stop_detection=stop_detection, additional_filters=additional_filters)
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

    color_dashline = ph.grey9
    ax.axvline(0, ls='--', lw=.2, c=color_dashline, alpha=.5, zorder=1)
    ax.axvline(5, ls='--', lw=.2, c=color_dashline, alpha=.5, zorder=1)
    ax.axvline(t_win_plot[1], ls='--', lw=.2, c=color_dashline, alpha=.5, zorder=1)
    ax.axhline(0, ls='--', lw=.2, c=color_dashline, alpha=.5, zorder=1)
    if show_xscale:
        ph.plot_x_scale_text(ax, bar_length=2, text='2s', y_text_lower=0.15, x_text_lefter=.05, fontsize=fs)

    if show_yscale:
        ph.plot_y_scale_text(ax, bar_length=1, text='1', x_text_right=.1, y_text_upper=.05, color=color, fontsize=fs, horizontalalignment='right', verticalalignment='center')

    if show_stimid_special:
        ax.text(xlim[0], ylim[-1], 'id = %i' % stimid_special, fontsize=fs, va='top', ha='left')

    if show_flynum:
        ax.text(xlim[0], ylim[-1], '%i flies' % len(recss), fontsize=fs, va='top', ha='left')

def plot_sig_ave(groups, nrow=3, ncol=9, fig_size=(4.5, 1.8), t_win_plot=[-3,4,3], t_win_stop=[-5,8.5], stim_method='pwm',
                 stimids=[4,3,2,1,6,5,4,7,8], stop_detection=False, show_xscale=True, show_flynum=True, additional_filters=[],
                 sig_type_='mean_az', lw_indi=0.15, alpha_indi=0.2, alpha_sem=0.2, lw=.6,
                 colors=[ph.ngreen,ph.dred,ph.nblue], ylim=[-3,3], fs=8):
    ph.set_fontsize(fs)
    _ = plt.figure(1, fig_size)
    gs = gridspec.GridSpec(nrow, ncol)
    if type(ylim[0]) is list:
        ylims = ylim
    else:
        ylims = []
        for i in range(nrow):
            ylims.append(ylim)

    for row in range(nrow):
        recss = groups[row]
        # trace plot
        for col in range(ncol):
            ax = plt.subplot(gs[row, col])
            stimid_special = stimids[col]
            _show_xscale = show_xscale if col == 0 and row == 0 else False
            _show_yscale = True if col == 0 else False
            _show_flynum = show_flynum if col == 0 else False
            plot_sig_ave_unit(recss, ax, sig_type=sig_type_, stimid_special=stimid_special, t_win_plot=t_win_plot,
                              t_win_stop=t_win_stop,  stop_detection=stop_detection, stim_method=stim_method,
                              additional_filters=additional_filters, lw=lw, lw_indi=lw_indi, alpha_indi=alpha_indi,
                              alpha_sem=alpha_sem, color=colors[row], xlabel='', ylabel='', show_axes=[], xticks=[],
                              ylim=ylims[row], yticks=[], show_xscale=_show_xscale, show_yscale=_show_yscale,
                              fs=fs, show_stimid_special=False, show_indi=True, show_flynum=_show_flynum)


# walking and flight general

def plot_hist_angular_variable(group, ax=False, fig_size=(1.5, 1.5), bin_num=36, flightstop_detection=True,
                    color=ph.blue, ylim=[0,0.175], range=[0,400], alpha_indi=0.4, lw_indi=0.5, lw=1.5, ):
    if not ax:
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])


    ys = []
    for recs in group:
        sigs = []
        for rec in recs:
            sig = rec.ims[0].c1.a
            inds = sig[:,0] < 1e10
            if flightstop_detection:
                inds_flight = get_2p_ind_flight(rec)
                inds = inds & inds_flight
            sigs.extend(sig[inds].flatten())

        hist_orig, bin_edges = np.histogram(sigs, bins=bin_num, range=range)
        hist_cumsum = np.cumsum(hist_orig)
        hist_norm = hist_orig / float(hist_cumsum[-1]) / ((range[-1]-range[0])/bin_num)
        xticks = np.mean([bin_edges[:-1], bin_edges[1:]], axis=0)
        ys.append(hist_norm)

    for y in ys:
        ax.plot(xticks, y, lw=lw_indi, color=color, alpha=alpha_indi, zorder=2, )
    ax.plot(xticks, np.nanmean(ys, axis=0), lw=lw, color=color, alpha=1, zorder=3, )

    ph.adjust_spines(ax, ['left', 'bottom'], lw=.2, xlim=range, ylim=ylim, xticks=range, yticks=ylim)
    ax.set_ylabel('Prob. density\n(deg$^{-1}$)')

def plot_scatter_bar_and_phase_unit(rec, ax=False, im_num=0, ch_num=1, xtype='Heading', im_num2=0, ch_num2=1,
                            exclude_standing=True, flightstop_detection=False,
                                    filters=[['dforw',[1,20]], ['xbar',[-135,135]]], two_cycle=True,
                            title='condition', xlabel='Bar $\Theta$', ylabel='PFFR $\Theta$', ylabel_axeratio=-.3,
                            show_axes=['bottom', 'left'], ms=2, color='black', alpha=.7, fs=12,):
    ph.set_fontsize(fs)
    if not ax:
        _ = plt.figure(1, (1, 1.2))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    speed_standing = 1
    dhead_standing = 25
    inds = rec.ims[0].c1.phase < 1e3
    if flightstop_detection:
        inds_flight = get_2p_ind_flight(rec)
        inds = inds & inds_flight
    if exclude_standing:
        dhead = rec.subsample('dhead')
        speed = rec.subsample('speed')
        inds = inds & ((speed > speed_standing) | (np.abs(dhead) > dhead_standing))

    for filter_ in filters:
        filter_name, filter_range = filter_
        sig_filter = rec.subsample(filter_name)
        inds = inds & (sig_filter > filter_range[0]) & (sig_filter < filter_range[1])

    ch = getattr(rec.ims[im_num], 'c%i' % ch_num)
    phase = ch.phase[inds]

    if xtype == 'secondphase':
        x_signal = rec.ims[im_num2].c1.phase if ch_num2 == 1 else rec.ims[im_num2].c2.phase
    else:
        x_signal = rec.subsample('xbar') if xtype == 'Heading' else calculate_td(rec, exclude_standing=False)

    x_signal = x_signal[inds]
    ax.plot(x_signal, phase, ls='none', marker='.', ms=ms, alpha=alpha, mfc=color, mec='none', zorder=2)
    if two_cycle:
        ax.plot(x_signal, phase+360, ls='none', marker='.', ms=ms, alpha=alpha, mfc=color, mec='none', zorder=2)

    # highlight arena
    ax.fill_betweenx([-181,541], x1=-180, x2=-135, facecolors='black', edgecolors='none', alpha=0.1, zorder=1)
    ax.fill_betweenx([-181,541], x1=135, x2=180, facecolors='black', edgecolors='none', alpha=0.1, zorder=1)

    ax.set_title(title)
    ax.set_aspect('equal', 'box')
    ylim_ = [-181, 541] if two_cycle else [-181, 181]
    ph.adjust_spines(ax, show_axes, lw=1, xlim=[-181, 181], ylim=ylim_, xticks=[], yticks=[])
                     # xticks=np.arange(-180, 181, 180), yticks=np.arange(-180, 541, 180))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel, rotation=90)
    ax.get_yaxis().set_label_coords(ylabel_axeratio, 0.5)

def plot_scatter_bar_and_phase(groups, group_labels=['cond #1','cond #2'], xlabel='Bar $\Theta$',
                               ylabel='PFFR $\Theta$', **kwargs):
    nrow = len(groups)
    _ns = [len(group) for group in groups]
    ncol = np.max(_ns)
    # ncol = 10

    _ = plt.figure(1, (2.2 * ncol, 4.2 * nrow))
    gs = gridspec.GridSpec(nrow, ncol)

    for row in range(nrow):
        recss = groups[row]
        col = 0

        for recs in recss:
            for rec in recs:
                ax = plt.subplot(gs[row, col])
                _xlabel = xlabel if (row == nrow - 1) else ''
                _ylabel = ylabel if (col == 0) else ''
                _title = group_labels[row] if col == 0 else ''
                _show_axes = []
                if col == 0:
                    _show_axes.append('left')
                if row == nrow - 1:
                    _show_axes.append('bottom')
                plot_scatter_bar_and_phase_unit(rec, ax=ax, title=_title, xlabel=_xlabel, ylabel=_ylabel,
                                                show_axes=_show_axes, **kwargs)
                col += 1

def plot_polarhist_2angular_variables(group, ax=False, fig_size=(1.5, 1.5), label_sig1='eb.c1.phase', label_sig2='xbar', bin_num=36,
            flightstop_detection=True, walkstand_detection=False, stim_visible=False, cancel_offset=True, filters=[],
            ylim=[0,0.175], color=ph.blue, alpha_indi=0.4, lw_indi=0.5, lw=1.5, ms=4,):
    if not ax:
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0], polar=True)

    ys = []
    for recs in group:
        for rec in recs:
            _, sig1 = rec.get_signal(label_sig1)
            if label_sig2 == 'td':
                sig2 = calculate_td(rec, exclude_standing=False)
            else:
                _, sig2 = rec.get_signal(label_sig2)

            # sig1 = rec.ims[0].c1.phase
            # sig2 = rec.ims[1].c1.phase
            inds = sig1 < 1e3
            if flightstop_detection:
                inds_flight = get_2p_ind_flight(rec)
                inds = inds & inds_flight
            if walkstand_detection:
                inds_walking = get_inds_walking(rec)
                inds = inds & inds_walking
            if stim_visible:
                inds_stimvisible = get_stim_visible(rec)
                inds = inds & inds_stimvisible
            for filter_ in filters:
                filter_name, filter_range = filter_
                sig_filter = rec.subsample(filter_name)
                inds = inds & (sig_filter > filter_range[0]) & (sig_filter < filter_range[1])
            sig1 = sig1[inds]
            sig2 = sig2[inds]
            if cancel_offset:
                offset = fc.circmean(fc.circsubtract(sig1, sig2))
                sig2 += offset

            delta_angles = np.radians(fc.circsubtract(sig1, sig2))
            hist_orig, bin_edges = np.histogram(delta_angles, bins=bin_num, range=[-np.pi, np.pi])
            hist_cumsum = np.cumsum(hist_orig)
            hist_norm = hist_orig / float(hist_cumsum[-1])
            xticks = np.mean([bin_edges[:-1], bin_edges[1:]], axis=0)
            ax.plot(xticks, hist_norm, lw=lw_indi, color=color, alpha=alpha_indi, zorder=2)
            ys.append(hist_norm)
    ax.plot(xticks, np.nanmean(ys, axis=0), lw=lw, color=color, alpha=1, zorder=3)

    ax.plot(0, 0, ls='none', marker='o', mfc=ph.red, mec='none', ms=ms, zorder=3)
    ax.set_ylim(ylim)
    ax.set_yticks([])
    ax.spines['polar'].set_visible(False)
    ax.set_theta_zero_location('N', offset=0)
    ax.set_theta_direction(1)
    ax.set_thetagrids([], visible=False)
    # ax.fill_between(np.linspace(0, 2*np.pi, 50), 0, ylim[1], facecolor=ph.grey1, edgecolor='none', alpha=0.5,
    #                 zorder=1)
    ax.plot(np.linspace(0, 2*np.pi, 120), np.zeros(120)+ylim[1], color='black', lw=.3, zorder=1)
    ax.grid(which='major', alpha=0)

def plot_hist_2angular_variables(groups, ax=False, fig_size=(1.5, 1.5), polar=True, bin_num=36,
            label_sigs1=['eb.c1.phase'], label_sigs2=['xbar'], colors=[ph.blue], filter_lag_ms=0,
            flightstop_detection=True, walkstand_detection=False, stim_visible=False, cancel_offset=True, filters=[],
            specific_stimid=0, shortest_stimid_dur_s=2, cut_start_stimid_s=30, cut_end_stimid_s=2, pwm_specificstimid=0,
            ylim=[0,0.175], alpha_indi=0.4, lw_indi=0.5, lw=1.5, ms=4, print_indi=False, return_data=False, vlines=[]):
    if not ax:
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0], polar=polar)

    ys_groups = []
    for igroup, group in enumerate(groups):
        ys_group = []
        delta_angles_group = []
        for recs in group:
            delta_angles_fly = []
            for rec in recs:
                ssr = rec.ims[0].sampling_rate
                _, sig1 = rec.get_signal(label_sigs1[igroup])
                if label_sigs2[igroup] == 'td':
                    sig2 = calculate_td(rec, exclude_standing=False)
                elif label_sigs2[igroup] == 'constant':
                    sig2 = sig1 * 0
                else:
                    _, sig2 = rec.get_signal(label_sigs2[igroup])

                # sig1 = rec.ims[0].c1.phase
                # sig2 = rec.ims[1].c1.phase
                inds = sig1 < 1e3
                if flightstop_detection:
                    inds_flight = get_2p_ind_flight(rec)
                    inds = inds & inds_flight
                if walkstand_detection:
                    inds_walking = get_inds_walking(rec)
                    inds = (inds & ~inds_walking) if walkstand_detection == 'stand' else (inds & inds_walking)
                if stim_visible:
                    inds_stimvisible = get_stim_visible(rec)
                    inds = inds & inds_stimvisible
                for filter_ in filters:
                    filter_name, filter_range = filter_
                    if filter_name == 'dypos':
                        ystim_uw = rec.subsample('ystim_uw', lag_ms=filter_lag_ms)
                        sig_filter = np.gradient(ystim_uw)
                    else:
                        sig_filter = rec.subsample(filter_name, lag_ms=filter_lag_ms)
                    inds = inds & (sig_filter > filter_range[0]) & (sig_filter < filter_range[1])
                if specific_stimid:
                    inds_stimid = rec.subsample('stimid') == specific_stimid
                    idxs = fc.get_contiguous_inds(inds_stimid, min_contig_len=np.ceil(shortest_stimid_dur_s*ssr),
                                        trim_left=np.ceil(cut_start_stimid_s*ssr), trim_right=np.ceil(cut_end_stimid_s*ssr), keep_ends=True)
                    inds[:int(idxs[0][0])] = False
                    inds[int(idxs[0][-1]):] = False
                if pwm_specificstimid:
                    inds_pwm = sig1 > 1e3       # all False
                    of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20200117_pwmdc(rec)
                    for i_stim, stimid in enumerate(of_stimids):
                        if stimid == pwm_specificstimid:
                            idx0 = int(of_start_idxs[i_stim]+np.ceil(cut_start_stimid_s*ssr))
                            idx1 = int(of_end_idxs[i_stim]-np.ceil(cut_end_stimid_s*ssr))
                            inds_pwm[idx0:idx1] = True
                    inds = inds & inds_pwm
                sig1 = sig1[inds]
                sig2 = sig2[inds]
                if cancel_offset:
                    offset = fc.circmean(fc.circsubtract(sig1, sig2))
                    sig2 += offset

                delta_angles = np.radians(fc.circsubtract(sig1, sig2))
                delta_angles_fly.extend(delta_angles)
            delta_angles_group.append(delta_angles_fly)
            hist_orig, bin_edges = np.histogram(delta_angles_fly, bins=bin_num, range=[-np.pi, np.pi])
            hist_cumsum = np.cumsum(hist_orig)
            hist_norm = hist_orig / float(hist_cumsum[-1]) / (360./bin_num)
            xticks = np.mean([bin_edges[:-1], bin_edges[1:]], axis=0)
            ys_group.append(hist_norm)
            ax.plot(xticks, hist_norm, lw=lw_indi, color=colors[igroup], alpha=alpha_indi, zorder=2, )

        ys_groups.append(ys_group)
        ax.plot(xticks, np.nanmean(ys_group, axis=0), lw=lw, color=colors[igroup], alpha=1, zorder=3, )
        if print_indi:
            # print fc.circmean(fc.circsubtract(sig1, sig2))
            print 'indi (radians):'
            ystr = ''
            angle_group = []
            for ys in ys_group:
                z0 = 0*np.exp(1j*0)
                for i in range(len(ys)):
                    z0 = z0 + ys[i] * np.exp(1j * xticks[i])
                angle_fly = np.degrees(np.angle(z0))
                angle_group.append(angle_fly)
                ystr = ystr + '%.3f' % angle_fly + ', '
            angle_group = np.array(angle_group)
            angle_group = angle_group[~np.isnan(angle_group)]
            print 'mean: %.3f' % fc.circmean(angle_group)
            print ystr
    if polar:
        ax.plot(0, 0, ls='none', marker='o', mfc=ph.red, mec='none', ms=ms, zorder=3)
        ax.set_ylim(ylim)
        ax.set_yticks([])
        ax.spines['polar'].set_visible(False)
        ax.set_theta_zero_location('N', offset=0)
        ax.set_theta_direction(1)
        ax.set_thetagrids([], visible=False)
        # ax.fill_between(np.linspace(0, 2*np.pi, 50), 0, ylim[1], facecolor=ph.grey1, edgecolor='none', alpha=0.5,
        #                 zorder=1)
        ax.plot(np.linspace(0, 2*np.pi, 120), np.zeros(120)+ylim[1], color='black', lw=.3, zorder=1)
        ax.grid(which='major', alpha=0)
    else:
        ax.axvline(0, ls='--', lw=lw_indi, color=ph.grey7, zorder=1)
        ph.adjust_spines(ax, ['left', 'bottom'], lw=.2, xlim=[-np.pi,np.pi], ylim=ylim, xticks=[-np.pi,0,np.pi], yticks=ylim)
        ax.set_ylabel('Prob. density\n(deg$^{-1}$)')
        if len(vlines):
            for vline in vlines:
                ax.axvline(vline, ls='--', lw=2, c=ph.dred, alpha=.6, zorder=1)
    if return_data:
        # return xticks, np.nanmean(ys_group, axis=0), delta_angles_group
        return xticks, ys_group, delta_angles_group
        # return xticks, ys_groups

def barplot_2angular_variables(groups, ax=False, fig_size=(1.5, 1.5), label_sig1='eb.c1.phase', label_sig2='fb.c1.phase',
            y_type='std', specific_stimids=0, shortest_stimid_dur_s=2, cut_start_stimid_s=30, cut_end_stimid_s=2, pwm_specificstimid=0,
            group_info=[], group_labels=[], label_axeratio=-.1, ms_indi=2, fs=8,
            ylim=[0,120], yticks=[0,50,100], circstat=True, two_cycles=True, print_indi=False,
            flightstop_detection=True, walkstand_detection=False, stim_visible=False, cancel_offset=True, filters=[], **kwargs):
    if not ax:
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])
    if type(specific_stimids) is not list:
        specific_stimids = np.zeros(len(groups)) + specific_stimids
    xss, x2ss = [], []
    for igroup, group in enumerate(groups):
        xs = []
        for recs in group:
            delta_angles = []
            for rec in recs:
                ssr = rec.ims[0].sampling_rate
                _, sig1 = rec.get_signal(label_sig1)
                if label_sig2 == 'td':
                    sig2 = calculate_td(rec, exclude_standing=False)
                else:
                    _, sig2 = rec.get_signal(label_sig2)
                inds = sig1 < 1e3
                if flightstop_detection:
                    inds_flight = get_2p_ind_flight(rec)
                    inds = inds & inds_flight
                if walkstand_detection:
                    inds_walking = get_inds_walking(rec)
                    inds = inds & inds_walking
                if stim_visible:
                    inds_stimvisible = get_stim_visible(rec)
                    inds = inds & inds_stimvisible
                for filter_ in filters:
                    filter_name, filter_range = filter_
                    sig_filter = rec.subsample(filter_name)
                    inds = inds & (sig_filter > filter_range[0]) & (sig_filter < filter_range[1])
                specific_stimid = specific_stimids[igroup]
                if specific_stimid:
                    inds_stimid = rec.subsample('stimid') == specific_stimid
                    idxs = fc.get_contiguous_inds(inds_stimid, min_contig_len=np.ceil(shortest_stimid_dur_s*ssr),
                                        trim_left=np.ceil(cut_start_stimid_s*ssr), trim_right=np.ceil(cut_end_stimid_s*ssr), keep_ends=True)
                    inds[:int(idxs[0][0])] = False
                    inds[int(idxs[0][-1]):] = False
                if pwm_specificstimid:
                    inds_pwm = sig1 > 1e3       # all False
                    of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20200117_pwmdc(rec)
                    for i_stim, stimid in enumerate(of_stimids):
                        if stimid == pwm_specificstimid:
                            idx0 = int(of_start_idxs[i_stim]+np.ceil(cut_start_stimid_s*ssr))
                            idx1 = int(of_end_idxs[i_stim]-np.ceil(cut_end_stimid_s*ssr))
                            inds_pwm[idx0:idx1] = True
                    inds = inds & inds_pwm
                sig1 = sig1[inds]
                sig2 = sig2[inds]
                if cancel_offset:
                    offset = fc.circmean(fc.circsubtract(sig1, sig2))
                    sig2 += offset

                delta_angles.extend(fc.circsubtract(sig1, sig2))
            if y_type == 'std':
                xs.append(fc.circstd(delta_angles))
            elif y_type == 'mean':
                xs.append(fc.circmean(delta_angles))
            elif y_type == 'absmean':
                xs.append(np.nanmean(np.abs(delta_angles)))
        xss.append(xs)
        x2ss.append(np.array(xs) + 360)

    if not len(group_info):
        group_info = [[i] for i in range(len(groups))]
    if not len(group_labels):
        group_labels = [i for i in range(len(groups))]
    ph.errorbar_lc_xss(xss=xss, ax=ax, group_info=group_info, group_labels=group_labels, colors=[ph.grey2],
                       ylabel='', yticks=yticks, ylim=ylim, yticks_minor=[], fs=fs,
                       ro_x=45, label_axeratio=label_axeratio, rotate_ylabel=False, plot_bar=False, ms_indi=ms_indi,
                       subgroup_samefly=False, circstat=circstat, legend_list=group_labels, show_flynum=False,
                       show_nancol=False, **kwargs)
    if two_cycles:
        ph.errorbar_lc_xss(xss=x2ss, ax=ax, group_info=group_info, group_labels=group_labels, colors=[ph.grey2],
                           ylabel='', yticks=yticks, ylim=ylim, yticks_minor=[], fs=fs,
                           ro_x=45, label_axeratio=label_axeratio, rotate_ylabel=False, plot_bar=False, ms_indi=ms_indi,
                           subgroup_samefly=False, circstat=circstat, legend_list=group_labels, show_flynum=False,
                           circ_low=180, circ_high=540, show_nancol=False, **kwargs)
    if print_indi:
        print 'indi:'
        for xs in xss:
            ystr = ''
            for y in xs:
                ystr = ystr + '%.3f' % y + ', '
            print ystr


def get_dphase_ave_unit_spcase1(recss, t_before_end=-2, t_end=0, t_win_stop=[-12,4], t_win_interp=[-3,1],
                        stim_method='dypos', flipsign_dphase=False, ):
    # return xs
    t_interp = np.arange(t_before_end, t_end, 0.1)
    dphase_group = []

    for fly in recss:
        dphase_fly = []

        for rec in fly:
            if stim_method == 'dypos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20191007(rec, stimid_N=10)
            elif stim_method == 'dxpos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20200224_xposonly(rec)
            stop_inds = get_flight_stopping_inds(rec)
            ssr_tif = rec.ims[0].sampling_rate
            ssr_abf = rec.abf.subsampling_rate

            for end_idx in of_end_idxs[of_stimids==1]:
                end_idx_abf = np.where(rec.abf.t < rec.fb.t[end_idx])[0][-1]
                idx_stopwindow = [end_idx_abf - np.ceil(np.abs(t_win_stop[0])*ssr_abf),
                                  end_idx_abf + np.ceil(np.abs(t_win_stop[-1])*ssr_abf)]

                if not np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
                    t0 = rec.abf.t[end_idx_abf]
                    idx_end = int(end_idx)
                    idx1 = int(idx_end + np.ceil(t_win_interp[1] * ssr_tif))
                    idx0 = int(idx_end - np.ceil(np.abs(t_win_interp[0]) * ssr_tif))
                    t_eb, t_fb = rec.ims[1].t[idx0:idx1] - t0, rec.ims[0].t[idx0:idx1] - t0
                    phase_eb, phase_fb = rec.ims[1].c1.phase[idx0:idx1], rec.ims[0].c1.phase[idx0:idx1]
                    f_fb = interp1d(t_fb, fc.unwrap(phase_fb), kind='linear')
                    f_eb = interp1d(t_eb, fc.unwrap(phase_eb), kind='linear')
                    phase_interp_fb = fc.wrap(f_fb(t_interp))
                    phase_interp_eb = fc.wrap(f_eb(t_interp))
                    if flipsign_dphase:
                        dphase = fc.circsubtract(phase_interp_eb, phase_interp_fb)
                    else:
                        dphase = fc.circsubtract(phase_interp_fb, phase_interp_eb)
                    dphase_fly.append(fc.circmean(dphase, axis=0))

            for end_idx in of_end_idxs[of_stimids==2]:
                end_idx_abf = np.where(rec.abf.t < rec.fb.t[end_idx])[0][-1]
                idx_stopwindow = [end_idx_abf - np.ceil(np.abs(t_win_stop[0])*ssr_abf),
                                  end_idx_abf + np.ceil(np.abs(t_win_stop[-1])*ssr_abf)]

                if not np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
                    t0 = rec.abf.t[end_idx_abf]
                    idx_end = int(end_idx)
                    idx1 = int(idx_end + np.ceil(t_win_interp[1] * ssr_tif))
                    idx0 = int(idx_end - np.ceil(np.abs(t_win_interp[0]) * ssr_tif))
                    t_eb, t_fb = rec.ims[1].t[idx0:idx1] - t0, rec.ims[0].t[idx0:idx1] - t0
                    phase_eb, phase_fb = rec.ims[1].c1.phase[idx0:idx1], rec.ims[0].c1.phase[idx0:idx1]
                    f_fb = interp1d(t_fb, fc.unwrap(phase_fb), kind='linear')
                    f_eb = interp1d(t_eb, fc.unwrap(phase_eb), kind='linear')
                    phase_interp_fb = fc.wrap(f_fb(t_interp))
                    phase_interp_eb = fc.wrap(f_eb(t_interp))
                    if not flipsign_dphase:
                        dphase = fc.circsubtract(phase_interp_eb, phase_interp_fb)
                    else:
                        dphase = fc.circsubtract(phase_interp_fb, phase_interp_eb)
                    dphase_fly.append(fc.circmean(dphase, axis=0))

        dphase_group.append(fc.circmean(dphase_fly, axis=0))
    return dphase_group

def barplot_2angular_variables_spcase1(groups, ax=False, fig_size=(1.5, 1.5),
            t_win_stop=[-2,20], stim_method='dypos',
            label_axeratio=-.1, ms_indi=2, fs=8, flipsign_dphase=False,
            ylim=[0,120], yticks=[0,50,100], circstat=True,  **kwargs):
    if not ax:
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    xss = []
    group_info = []
    count = 0

    for recss in groups:
        group_info_unit = []
        for iperiod in range(5):
            cut_start_s = iperiod * 4 + 2 - 20
            cut_end_s = iperiod * 4 + 4 - 20
            xs = get_dphase_ave_unit_spcase1(recss, t_before_end=cut_start_s,
                                     t_end=cut_end_s, t_win_stop=t_win_stop, t_win_interp=[cut_start_s-1,1],
                                     stim_method=stim_method, flipsign_dphase=flipsign_dphase,)
            xss.append(xs)
            group_info_unit.append(count)
            count += 1
        group_info.append(group_info_unit)

    group_labels = np.arange(5)
    # print group_info
    # print len(xss)
    # for xs in xss:
    #     print xs

    ph.errorbar_lc_xss(xss=xss, ax=ax, group_info=group_info, group_labels=group_labels, colors=[ph.grey2],
                       ylabel='', yticks=yticks, ylim=ylim, yticks_minor=[], fs=fs,
                       ro_x=45, label_axeratio=label_axeratio, rotate_ylabel=False, plot_bar=False, ms_indi=ms_indi,
                       subgroup_samefly=False, circstat=circstat, legend_list=group_labels, show_flynum=False,
                       show_nancol=False, **kwargs)


def get_dphase_ave_unit_spcase1_reverseorder(recss, t_before_end=-2, t_end=0, t_before_end2=-2, t_end2=0,
                        t_win_stop=[-12,4], t_win_interp=[-3,1],
                        stim_method='dypos', flipsign_dphase=False, ):
    # return xs
    t_interp = np.arange(t_before_end, t_end, 0.1)
    t_interp2 = np.arange(t_before_end2, t_end2, 0.1)
    dphase_group = []

    for fly in recss:
        dphase_fly = []

        for rec in fly:
            if stim_method == 'dypos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20191007(rec, stimid_N=10)
            elif stim_method == 'dxpos':
                of_stimids, of_start_idxs, of_end_idxs = get_stim_info_20200224_xposonly(rec)
            stop_inds = get_flight_stopping_inds(rec)
            ssr_tif = rec.ims[0].sampling_rate
            ssr_abf = rec.abf.subsampling_rate

            for end_idx in of_end_idxs[of_stimids==1]:
                end_idx_abf = np.where(rec.abf.t < rec.fb.t[end_idx])[0][-1]
                idx_stopwindow = [end_idx_abf - np.ceil(np.abs(t_win_stop[0])*ssr_abf),
                                  end_idx_abf + np.ceil(np.abs(t_win_stop[-1])*ssr_abf)]

                if not np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
                    t0 = rec.abf.t[end_idx_abf]
                    idx_end = int(end_idx)
                    idx1 = int(idx_end + np.ceil(t_win_interp[1] * ssr_tif))
                    idx0 = int(idx_end - np.ceil(np.abs(t_win_interp[0]) * ssr_tif))
                    t_eb, t_fb = rec.ims[1].t[idx0:idx1] - t0, rec.ims[0].t[idx0:idx1] - t0
                    phase_eb, phase_fb = rec.ims[1].c1.phase[idx0:idx1], rec.ims[0].c1.phase[idx0:idx1]
                    f_fb = interp1d(t_fb, fc.unwrap(phase_fb), kind='linear')
                    f_eb = interp1d(t_eb, fc.unwrap(phase_eb), kind='linear')
                    phase_interp_fb = fc.wrap(f_fb(t_interp))
                    phase_interp_eb = fc.wrap(f_eb(t_interp))
                    if flipsign_dphase:
                        dphase = fc.circsubtract(phase_interp_eb, phase_interp_fb)
                    else:
                        dphase = fc.circsubtract(phase_interp_fb, phase_interp_eb)
                    dphase_fly.append(fc.circmean(dphase, axis=0))

            for end_idx in of_end_idxs[of_stimids==2]:
                end_idx_abf = np.where(rec.abf.t < rec.fb.t[end_idx])[0][-1]
                idx_stopwindow = [end_idx_abf - np.ceil(np.abs(t_win_stop[0])*ssr_abf),
                                  end_idx_abf + np.ceil(np.abs(t_win_stop[-1])*ssr_abf)]

                if not np.sum((stop_inds > idx_stopwindow[0]) & (stop_inds < idx_stopwindow[-1])):
                    t0 = rec.abf.t[end_idx_abf]
                    idx_end = int(end_idx)
                    idx1 = int(idx_end + np.ceil(t_win_interp[1] * ssr_tif))
                    idx0 = int(idx_end - np.ceil(np.abs(t_win_interp[0]) * ssr_tif))
                    t_eb, t_fb = rec.ims[1].t[idx0:idx1] - t0, rec.ims[0].t[idx0:idx1] - t0
                    phase_eb, phase_fb = rec.ims[1].c1.phase[idx0:idx1], rec.ims[0].c1.phase[idx0:idx1]
                    f_fb = interp1d(t_fb, fc.unwrap(phase_fb), kind='linear')
                    f_eb = interp1d(t_eb, fc.unwrap(phase_eb), kind='linear')
                    phase_interp_fb = fc.wrap(f_fb(t_interp2))
                    phase_interp_eb = fc.wrap(f_eb(t_interp2))
                    if flipsign_dphase:
                        dphase = fc.circsubtract(phase_interp_eb, phase_interp_fb)
                    else:
                        dphase = fc.circsubtract(phase_interp_fb, phase_interp_eb)
                    dphase_fly.append(fc.circmean(dphase, axis=0))

        dphase_group.append(fc.circmean(dphase_fly, axis=0))
    return dphase_group

def barplot_2angular_variables_spcase1reverseorder(groups, ax=False, fig_size=(1.5, 1.5),
            t_win_stop=[-2,20], stim_method='dypos',
            label_axeratio=-.1, ms_indi=2, fs=8, flipsign_dphase=False,
            ylim=[0,120], yticks=[0,50,100], circstat=True,  **kwargs):
    if not ax:
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    xss = []
    group_info = []
    count = 0

    for recss in groups:
        group_info_unit = []
        for iperiod in range(5):
            cut_start_s = iperiod * 4 + 2 - 20
            cut_end_s = iperiod * 4 + 4 - 20
            cut_start_s2 = (4-iperiod) * 4 + 2 - 20
            cut_end_s2 = (4-iperiod) * 4 + 4 - 20
            cut_start_s_min = min(cut_start_s, cut_start_s2)
            xs = get_dphase_ave_unit_spcase1(recss, t_before_end=cut_start_s,t_end=cut_end_s,
                    t_before_end2=cut_start_s2, t_end2=cut_end_s2, t_win_stop=t_win_stop, t_win_interp=[cut_start_s_min-1,1],
                    stim_method=stim_method, flipsign_dphase=flipsign_dphase,)
            xss.append(xs)
            group_info_unit.append(count)
            count += 1
        group_info.append(group_info_unit)

    group_labels = np.arange(5)
    # print group_info
    # print len(xss)
    # for xs in xss:
    #     print xs

    ph.errorbar_lc_xss(xss=xss, ax=ax, group_info=group_info, group_labels=group_labels, colors=[ph.grey2],
                       ylabel='', yticks=yticks, ylim=ylim, yticks_minor=[], fs=fs,
                       ro_x=45, label_axeratio=label_axeratio, rotate_ylabel=False, plot_bar=False, ms_indi=ms_indi,
                       subgroup_samefly=False, circstat=circstat, legend_list=group_labels, show_flynum=False,
                       show_nancol=False, **kwargs)






































































