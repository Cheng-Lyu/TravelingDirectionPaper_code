import matplotlib.pyplot as plt
import numpy as np
import copy, math
import numpy.fft as fft
import scipy.signal as signal
from scipy import stats
import matplotlib.gridspec as gridspec
import plotting_help_2021Paper as ph
from scipy.interpolate import interp1d
import functions as fc
from matplotlib.ticker import ScalarFormatter
from matplotlib import rcParams
import operator

pi = np.pi

# Vector manipulation
def cart2polar(x, y):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta
            
def polar2cart(r, theta):
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return x, y

def perpendicular(a) :   #not used
    b = np.empty_like(a)
    b[:, 0] = -a[:, 1]
    b[:, 1] = a[:, 0]
    return b

def get_vector(slope, origin, length=20, orthogonal=True):
    # Generate 2D vector for each slope
    v = np.array([np.ones(len(slope)), slope]).T
    v[v.T[1] == np.inf] = np.array([0, 1])
    v[v.T[1] == -np.inf] = np.array([0, -1])
    
    # Set all vectors to the same length
    v = v/np.linalg.norm(v, axis=1).reshape(len(v), 1)
    v *= length/2.
    
    if orthogonal:
        v = perpendicular(v)
    
    # Define a line extending in both directions from the origin
    vf = np.empty([len(v), 2, 2])
    vf[:, 0] = origin - v
    vf[:, 1] = origin + v
    
    return vf

def get_slope(points, edge_order=2):
    x, y = points.T
    dx = np.gradient(x)
    slope = np.gradient(y, dx, edge_order=edge_order)
    return slope

def get_strings(mask):  
    """
    Extracts string of neighbouring masked values - used for finding skel in bridge
    """
    def get_sudoku(p):
            x0 = p[0]
            y0 = p[1]
            sudoku = [(x0-1, y0-1), (x0, y0-1), (x0+1, y0-1), 
                      (x0-1, y0),               (x0+1, y0), 
                      (x0-1, y0+1), (x0, y0+1), (x0+1, y0+1)]
            return sudoku
        
    def get_next_point(points_sorted, points):
        p0 = points_sorted[-1]
        for s in get_sudoku(p0):
            if s in points and s not in points_sorted:
                points_sorted.append(s)
                get_next_point(points_sorted, points)
                break
        return points_sorted
    
    def get_string(start, points):
        string1 = get_next_point([start], points)
        string = get_next_point([string1[-1]], points)
        start2 = min([string[0], string[-1]])
        string = get_next_point([start2], points)
        
        return string
    
    mask = copy.copy(mask)
    x, y = np.where(mask.T)
    points = zip(x, y)
    start = points[0]
    strings = []
    while 1:
        string = get_string(start, points)
        strings.append(string)
        sx, sy = zip(*string)
        mask[sy, sx] = 0
        x, y = np.where(mask.T)
        points = zip(x, y)
        if len(points):
            start = points[0]
        else:
            break
    
    # Order strings
    starts = [min([string[0], string[-1]]) for string in strings]
    starts, strings = zip(*sorted(zip(starts, strings)))  # Sort strings based on starts
    strings = map(np.array, strings)
    
    return strings


# FFT  
def powerspec(data, axis=-1, n=100, show=False, axs=None, fig_filename='', vmin=None,
              vmax=None, gate=[6, 11], cmap=plt.cm.jet, logscale=True, t=None,
              mask=False, norm=True):
    data = copy.copy(data)
    data = data - data.mean()
    if mask:
        data[:, 80:101] = 0
    axlen = data.shape[axis]*n
    fft1 = fft.fft(data, axlen, axis)
    power = np.abs(fft1)**2
    if norm:
        power = power / power.max()
    freq = fft.fftfreq(axlen, 1./n)/n
    phase = np.angle(fft1)
    
    midpoint = freq.size/2
    freq = freq[1:midpoint]
    period = (1./freq)
    power = power[:, 1:midpoint]
    phase = phase[:, 1:midpoint]
    
    if show or axs:
        if axs is None:
            plt.figure(1, (3, 7))
            gs = gridspec.GridSpec(2, 1, height_ratios=[1, 6])
            gs.update(hspace=0.2)
            ax1 = plt.subplot(gs[1, 0])
            ax2 = plt.subplot(gs[0, 0])#, sharex=ax1)
        else:
            ax2, ax1 = axs

        # Plot power spectrum over time
        ph.adjust_spines(ax1, ['left', 'top'])
        unit = 'frame #' if (t is None) else 's'
        ax1.set_ylabel('Time (%s)' %unit, rotation='horizontal', ha='right', labelpad=20)
        yedges = np.arange(power.shape[0]) if (t is None) else t
        xedges = period
        img = ax1.pcolormesh(xedges, yedges, power, cmap=cmap)
        if vmax: img.set_clim(vmax=vmax)
        if vmin: img.set_clim(vmin=vmin)
        ax1.set_xlabel('Period (glomeruli)', labelpad=20)
        # ax1.scatter(self.tif.t, periods, color='black', alpha=.8, lw=0, s=5, clip_on=True)
        ax1.set_xlim(2, 32)

        ax1.invert_yaxis()


        # Plot average power spectrum

        ax2.set_xticks([])
        h = power[:, period<=32].mean(axis=0)
        ymax = fc.round(h.max()*10, 1, 'up')/10.
        ax2.set_ylim(0, ymax)
        ax2.set_yticks([0, ymax])
#         ax2.fill_between(period[period<=pmax], h, facecolor=blue, lw=0)
        ax2.plot(period[period<=32], h, c='black', lw=1)
        ax2.set_xlim(2, 30)
        
        if logscale:
            ax1.set_xscale('log', basex=2)
            ax1.xaxis.set_major_formatter(ScalarFormatter())
            ax2.set_xscale('log', basex=2)
            ph.adjust_spines(ax2, ['left'])
        
        if fig_filename:
            plt.savefig(fig_filename, bbox_inches='tight')

        return img
    else:    
        return power, period, phase

def powerspec_peak(data, mode='constant', gate=[6, 11], axis=-1, t=None, cval=None, n=100, show=False, fig_filename=''): #mode used to be 'peak'
    """
    'peak' mode: find largest peak within gate. If no peak is detected, return nan.
    'max' mode: find largest value within gate. Same as peak, except returns largest value if no peak is detected.
    'constant' mode: use same value for each time frame.
    'median' mode: First runs powerspec_peak in 'peak' mode to find peaks within gate.
        Then runs again in peak mode, except return median peak if no peak is detected.
    """
    power, period, phase = powerspec(data, axis, n, False)
    if mode=='max':
        gateidx = np.where((period>gate[0]) & (period < gate[1]))[0]
    elif mode == 'median':
        _, pperiod, _ = fc.powerspec_peak(data, axis, 'peak', gate, show=False)
        medval = np.nanmedian(pperiod)
        medidx = np.where(period<=medval)[0][0]
    elif mode=='constant':
        if cval is None:
            power_mean = power.mean(axis=0)
            constidx = np.argmax(power_mean[(period >= gate[0]) & (period < gate[1])])
        else:
            constidx = np.where(period<=cval)[0][0]
    peak_phase = np.zeros(len(data))
    peak_period = np.zeros(len(data))
    peak_power = np.zeros(len(data))
    for i, row in enumerate(power):
        if mode in ['peak', 'median']:
            peakinds = detect_peaks(row, mph=.1, show=False)
            peakvals = row[peakinds]
            idx = np.argsort(peakvals)[::-1]
            peakpers = period[peakinds][idx]
            if gate:
                peakpers = peakpers[(peakpers>gate[0]) & (peakpers<gate[1])]
            if peakpers.size:
                iperiod = peakpers[0]
                iphase = phase[i, period==iperiod]
                ipower = power[i, period==iperiod]
            elif mode == 'median':
                iperiod = period[medidx]
                iphase = phase[i, medidx]
                ipower = power[i, medidx]
            else:
                iperiod = np.nan
                iphase = np.nan
                ipower = np.nan
        elif mode == 'max':
            rowgated = row[gateidx]
            idx = np.argmax(rowgated)
            idx += gateidx[0]
            iperiod = period[idx]
            iphase = phase[i, idx]
            ipower = power[i, idx]
        elif mode == 'constant':
            iperiod = period[constidx]
            iphase = phase[i, constidx]
            ipower = power[i, constidx]
            
        peak_period[i] = iperiod
        peak_phase[i] = iphase
        peak_power[i] = ipower
    peak_phase = peak_phase*(180/np.pi)
    
    if show:
        # Setup Figure
        rcParams['xtick.direction'] = 'out'
        rcParams['ytick.direction'] = 'out'
        plt.figure(1, (7, 5))
        gs = gridspec.GridSpec(2, 1, height_ratios=[1, 6])
        gs.update(wspace=0.05)

        # Plot period over time
        ax1 = plt.subplot(gs[1, 0])
        y = np.arange(len(peak_period)) if (t is None) else t
        ax1.scatter(peak_period, y, color='black', alpha=.8, lw=0, s=5, clip_on=True)
        ph.adjust_spines(ax1, ['left', 'bottom'])
        ax1.set_xlabel('Period (glomeruli)', labelpad=20)
        pmax = 18
        ax1.set_xlim(0, pmax)
        ax1.set_ylim(0, len(peak_period))
        ax1.invert_yaxis()
        plt.grid()

        # Plot histogram of period
        ax2 = plt.subplot(gs[0, 0])
        ph.adjust_spines(ax2, ['left'])
        binwidth = .1
        bins = np.arange(0, 18, binwidth)
        period_vals = peak_period[np.isnan(peak_period)==False]
        h, xedges = np.histogram(period_vals, bins=bins, density=True)
        ax2.bar(xedges[:-1], h, color='grey', lw=0, width=binwidth)
        ax2.set_xlim(0, pmax)
        ymax = np.round(h.max(), decimals=1)
        ax2.set_yticks([0, ymax])
        ax2.set_ylim(0, ymax)
        
        for ax in [ax1, ax2]:
            for line in gate:
                ax.axvline(line, alpha=.3, ls='--')
        
        if fig_filename:
            plt.savefig(fig_filename, bbox_inches='tight')
    else:
        return peak_phase, peak_period, peak_power


# Peaks
def detect_peaks(x, mph=None, mpd=1, threshold=0, edge='rising',
                 kpsh=False, valley=False, show=False, ax=None):

    """Detect peaks in data based on their amplitude and other features.

    Parameters
    ----------
    x : 1D array_like
        data.
    mph : {None, number}, optional (default = None)
        detect peaks that are greater than minimum peak height.
    mpd : positive integer, optional (default = 1)
        detect peaks that are at least separated by minimum peak distance (in
        number of data).
    threshold : positive number, optional (default = 0)
        detect peaks (valleys) that are greater (smaller) than `threshold`
        in relation to their immediate neighbors.
    edge : {None, 'rising', 'falling', 'both'}, optional (default = 'rising')
        for a flat peak, keep only the rising edge ('rising'), only the
        falling edge ('falling'), both edges ('both'), or don't detect a
        flat peak (None).
    kpsh : bool, optional (default = False)
        keep peaks with same height even if they are closer than `mpd`.
    valley : bool, optional (default = False)
        if True (1), detect valleys (local minima) instead of peaks.
    show : bool, optional (default = False)
        if True (1), plot data in matplotlib figure.
    ax : a matplotlib.axes.Axes instance, optional (default = None).

    Returns
    -------
    ind : 1D array_like
        indeces of the peaks in `x`.

    Notes
    -----
    The detection of valleys instead of peaks is performed internally by simply
    negating the data: `ind_valleys = detect_peaks(-x)`
    
    The function can handle NaN's 

    See this IPython Notebook [1]_.

    References
    ----------
    .. [1] http://nbviewer.ipython.org/github/demotu/BMC/blob/master/notebooks/DetectPeaks.ipynb

    Examples
    --------
    >>> from detect_peaks import detect_peaks
    >>> x = np.random.randn(100)
    >>> x[60:81] = np.nan
    >>> # detect all peaks and plot data
    >>> ind = detect_peaks(x, show=True)
    >>> print(ind)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # set minimum peak height = 0 and minimum peak distance = 20
    >>> detect_peaks(x, mph=0, mpd=20, show=True)

    >>> x = [0, 1, 0, 2, 0, 3, 0, 2, 0, 1, 0]
    >>> # set minimum peak distance = 2
    >>> detect_peaks(x, mpd=2, show=True)

    >>> x = np.sin(2*np.pi*5*np.linspace(0, 1, 200)) + np.random.randn(200)/5
    >>> # detection of valleys instead of peaks
    >>> detect_peaks(x, mph=0, mpd=20, valley=True, show=True)

    >>> x = [0, 1, 1, 0, 1, 1, 0]
    >>> # detect both edges
    >>> detect_peaks(x, edge='both', show=True)

    >>> x = [-2, 1, -2, 2, 1, 1, 3, 0]
    >>> # set threshold = 2
    >>> detect_peaks(x, threshold = 2, show=True)
    """

    x = np.atleast_1d(x).astype('float64')
    if x.size < 3:
        return np.array([], dtype=int)
    if valley:
        x = -x
    # find indices of all peaks
    dx = x[1:] - x[:-1]
    # handle NaN's
    indnan = np.where(np.isnan(x))[0]
    if indnan.size:
        x[indnan] = np.inf
        dx[np.where(np.isnan(dx))[0]] = np.inf
    ine, ire, ife = np.array([[], [], []], dtype=int)
    if not edge:
        ine = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) > 0))[0]
    else:
        if edge.lower() in ['rising', 'both']:
            ire = np.where((np.hstack((dx, 0)) <= 0) & (np.hstack((0, dx)) > 0))[0]
        if edge.lower() in ['falling', 'both']:
            ife = np.where((np.hstack((dx, 0)) < 0) & (np.hstack((0, dx)) >= 0))[0]
    ind = np.unique(np.hstack((ine, ire, ife)))
    # handle NaN's
    if ind.size and indnan.size:
        # NaN's and values close to NaN's cannot be peaks
        ind = ind[np.in1d(ind, np.unique(np.hstack((indnan, indnan-1, indnan+1))), invert=True)]
    # first and last values of x cannot be peaks
    if ind.size and ind[0] == 0:
        ind = ind[1:]
    if ind.size and ind[-1] == x.size-1:
        ind = ind[:-1]
    # remove peaks < minimum peak height
    if ind.size and mph is not None:
        ind = ind[x[ind] >= mph]
    # remove peaks - neighbors < threshold
    if ind.size and threshold > 0:
        dx = np.min(np.vstack([x[ind]-x[ind-1], x[ind]-x[ind+1]]), axis=0)
        ind = np.delete(ind, np.where(dx < threshold)[0])
    # detect small peaks closer than minimum peak distance
    if ind.size and mpd > 1:
        ind = ind[np.argsort(x[ind])][::-1]  # sort ind by peak height
        idel = np.zeros(ind.size, dtype=bool)
        for i in range(ind.size):
            if not idel[i]:
                # keep peaks with the same height if kpsh is True
                idel = idel | (ind >= ind[i] - mpd) & (ind <= ind[i] + mpd) \
                    & (x[ind[i]] > x[ind] if kpsh else True)
                idel[i] = 0  # Keep current peak
        # remove the small peaks and sort back the indices by their occurrence
        ind = np.sort(ind[~idel])

    def _plot(x, mph, mpd, threshold, edge, valley, ax, ind):
        """Plot results of the detect_peaks function, see its help."""
        
        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 4))

        ax.plot(x, 'b', lw=1)
        if ind.size:
            label = 'valley' if valley else 'peak'
            label = label + 's' if ind.size > 1 else label
            ax.plot(ind, x[ind], '+', mfc=None, mec='r', mew=2, ms=8,
                    label='%d %s' % (ind.size, label))
            ax.legend(loc='best', framealpha=.5, numpoints=1)
        ax.set_xlim(-.02*x.size, x.size*1.02-1)
        ymin, ymax = x[np.isfinite(x)].min(), x[np.isfinite(x)].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.1*yrange, ymax + 0.1*yrange)
        ax.set_xlabel('Data #', fontsize=14)
        ax.set_ylabel('Amplitude', fontsize=14)
        mode = 'Valley detection' if valley else 'Peak detection'
        ax.set_title("%s (mph=%s, mpd=%d, threshold=%s, edge='%s')"
                     % (mode, str(mph), mpd, str(threshold), edge))
        # plt.grid()
        plt.show()

    if show:
        if indnan.size:
            x[indnan] = np.nan
        if valley:
            x = -x
        _plot(x, mph, mpd, threshold, edge, valley, ax, ind)

    return ind

def get_halfwidth_peak(xs, ys):
    x_wrap = wrap(xs)
    y_wrap = np.tile(ys, 3)
    interpf = interp1d(x_wrap, y_wrap, 'linear')
    x_interp = np.arange(-180, 180.1, .1)
    y_interp = interpf(x_interp)
    halfwidth_idxs = np.where(np.diff(y_interp>=ys.max()/2.)>0)[0]
    if halfwidth_idxs.size == 2:
        peak = circmean(x_interp[halfwidth_idxs])
        return peak
    else:
        return None


# Sampling data
def get_inds_start_stop(inds, keep_ends=True):
    diff = np.diff(inds.astype(np.int8))
    start_inds = np.where(diff>0)[0] + 1
    stop_inds = np.where(diff<0)[0] + 1


    if len(stop_inds) == 0:
        stop_inds = np.array([len(inds)])

    if len(start_inds) == 0:
        start_inds = np.array([0])

    if stop_inds[0] <= start_inds[0]:
        if keep_ends:
            start_inds = np.insert(start_inds, 0, 0)
        else:
            stop_inds = stop_inds[1:]

    if start_inds[-1] >= stop_inds[-1]:
        if keep_ends:
            stop_inds = np.append(stop_inds, len(inds))
        else:
            start_inds = start_inds[:-1]
    return start_inds, stop_inds

def get_contiguous_inds(inds, min_contig_len=0, trim_left=0, trim_right=0, **kwargs):
    """
    Splits a boolean array into a list of lists, each list containing the indices for contiguous True elements.
    """
    if inds.sum():
        start_inds, stop_inds = get_inds_start_stop(inds, **kwargs)
        cont_inds = [np.arange(start+trim_left, stop-trim_right).astype(int) for start, stop in zip(start_inds, stop_inds) if stop-start > min_contig_len]
        return cont_inds
    else:
        return []

def get_merged_inds(inds):
    merged_inds = []
    if len(inds) == 0:
        return []
    for item in inds:
        merged_inds = merged_inds + list(item)
    return merged_inds

def rising_trig(signal, trigger=3, mindist=0):
    trigs = np.where(np.diff((signal>trigger).astype(int))>0)[0]
    trigs = trigs[np.insert(np.diff(trigs)>mindist, 0, 1)]
    return trigs

def dense_sample(data, edges, metric=np.mean, **kwargs):    # not used
    datab = np.zeros(len(edges)-1)
    for i in xrange(len(edges)-1):
        idx = np.arange(edges[i], edges[i+1])
        datab[i] = metric(data[idx], **kwargs)
    return datab

def sparse_sample(data, leftinds, indint, metric=np.mean, **kwargs):
    ind0 = np.arange(0, indint)
    inds = np.tile(ind0, (len(leftinds), 1)) + leftinds[:, None]
    datab = metric(data[inds], axis=1, **kwargs)
    return datab

def sparse_sample_deprecated(data, leftinds, indint, metric=np.mean, **kwargs):
    datab = np.zeros_like(leftinds)
    for i, leftind in enumerate(leftinds):
        idx = np.arange(leftind, leftind+indint)
        datab[i] = metric(data[idx], **kwargs)
    return datab

def binyfromx(x, y, xbins, metric=np.median, nmin=0, **kwargs):
    binidx = np.digitize(x, xbins)
    yshape = list(y.shape)
    yshape[0] = len(xbins)-1
    ystat = np.zeros(yshape)
    for ibin in range(1, len(xbins)):
        yi = y[ibin==binidx]
        ystat[ibin-1] = metric(yi, **kwargs) if (len(yi) > nmin) else np.nan
    return ystat

def butterworth(input_signal, cutoff, passside='low', N=8, sampling_freq=100):
    Nyquist_freq = sampling_freq / 2.
    Wn = cutoff/Nyquist_freq
    b, a = signal.butter(N, Wn, passside)
    output_signal = signal.filtfilt(b, a, input_signal)
    return output_signal


# Manipulating Circular data
def unwrap(signal, period=360):
    unwrapped = np.unwrap(signal*2*np.pi/period)*period/np.pi/2
    return unwrapped

def wrap(arr, cmin=-180, cmax=180):
    period = cmax - cmin
    arr = arr%period
    arr[arr>=cmax] = arr[arr>=cmax] - period
    arr[arr<cmin] = arr[arr<cmin] + period
    return arr

def circgrad(signal, method=np.gradient, **kwargs):
    signaluw = unwrap(signal, **kwargs)
    dsignal = method(signaluw)
    return dsignal

def circmean(arr, low=-180, high=180, axis=None):
    return stats.circmean(arr, high, low, axis)

def _contains_nan(a, nan_policy='propagate'):
    policies = ['propagate', 'raise', 'omit']
    if nan_policy not in policies:
        raise ValueError("nan_policy must be one of {%s}" %
                         ', '.join("'%s'" % s for s in policies))
    try:
        # Calling np.sum to avoid creating a huge array into memory
        # e.g. np.isnan(a).any()
        with np.errstate(invalid='ignore'):
            contains_nan = np.isnan(np.sum(a))
    except TypeError:
        # This can happen when attempting to sum things which are not
        # numbers (e.g. as in the function `mode`). Try an alternative method:
        try:
            contains_nan = np.nan in set(a.ravel())
        except TypeError:
            # Don't know what to do. Fall back to omitting nan values and
            # issue a warning.
            contains_nan = False
            nan_policy = 'omit'
            warnings.warn("The input array could not be properly checked for nan "
                          "values. nan values will be ignored.", RuntimeWarning)

    if contains_nan and nan_policy == 'raise':
        raise ValueError("The input contains nan values")

    return (contains_nan, nan_policy)

def _circfuncs_common(samples, high, low, nan_policy='propagate'):
    # Ensure samples are array-like and size is not zero
    samples = np.asarray(samples)
    if samples.size == 0:
        return np.nan, np.asarray(np.nan), np.asarray(np.nan), None

    # Recast samples as radians that range between 0 and 2 pi and calculate
    # the sine and cosine
    sin_samp = np.sin((samples - low)*2.*np.pi / (high - low))
    cos_samp = np.cos((samples - low)*2.*np.pi / (high - low))

    # Apply the NaN policy
    contains_nan, nan_policy = _contains_nan(samples, nan_policy)
    if contains_nan and nan_policy == 'omit':
        mask = np.isnan(samples)
        # Set the sines and cosines that are NaN to zero
        sin_samp[mask] = 0.0
        cos_samp[mask] = 0.0
    else:
        mask = None

    return samples, sin_samp, cos_samp, mask

def circnanmean(samples, low=-180, high=180, axis=None, nan_policy='omit'):
    """
    Compute the circular mean for samples in a range.
    Parameters
    ----------
    samples : array_like
        Input array.
    high : float or int, optional
        High boundary for circular mean range.  Default is ``2*pi``.
    low : float or int, optional
        Low boundary for circular mean range.  Default is 0.
    axis : int, optional
        Axis along which means are computed.  The default is to compute
        the mean of the flattened array.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'.
    Returns
    -------
    circmean : float
        Circular mean.
    Examples
    --------
    >>> from scipy.stats import circmean
    >>> circmean([0.1, 2*np.pi+0.2, 6*np.pi+0.3])
    0.2
    >>> from scipy.stats import circmean
    >>> circmean([0.2, 1.4, 2.6], high = 1, low = 0)
    0.4
    """
    samples, sin_samp, cos_samp, nmask = _circfuncs_common(samples, high, low,
                                                           nan_policy=nan_policy)
    sin_sum = sin_samp.sum(axis=axis)
    cos_sum = cos_samp.sum(axis=axis)
    res = np.arctan2(sin_sum, cos_sum)

    mask_nan = ~np.isnan(res)
    if mask_nan.ndim > 0:
        mask = res[mask_nan] < 0
    else:
        mask = res < 0

    if mask.ndim > 0:
        mask_nan[mask_nan] = mask
        res[mask_nan] += 2*pi
    elif mask:
        res += 2*pi

    # Set output to NaN if no samples went into the mean
    if nmask is not None:
        if nmask.all():
            res = np.full(shape=res.shape, fill_value=np.nan)
        else:
            # Find out if any of the axis that are being averaged consist
            # entirely of NaN.  If one exists, set the result (res) to NaN
            nshape = 0 if axis is None else axis
            smask = nmask.shape[nshape] == nmask.sum(axis=axis)
            if smask.any():
                res[smask] = np.nan

    return res*(high - low)/2.0/pi + low

def circnanstd(samples, high=2*pi, low=0, axis=None, nan_policy='omit'):
    """
    Compute the circular standard deviation for samples assumed to be in the
    range [low to high].
    Parameters
    ----------
    samples : array_like
        Input array.
    high : float or int, optional
        High boundary for circular standard deviation range.
        Default is ``2*pi``.
    low : float or int, optional
        Low boundary for circular standard deviation range.  Default is 0.
    axis : int, optional
        Axis along which standard deviations are computed.  The default is
        to compute the standard deviation of the flattened array.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan. 'propagate' returns nan,
        'raise' throws an error, 'omit' performs the calculations ignoring nan
        values. Default is 'propagate'.
    Returns
    -------
    circstd : float
        Circular standard deviation.
    Notes
    -----
    This uses a definition of circular standard deviation that in the limit of
    small angles returns a number close to the 'linear' standard deviation.
    Examples
    --------
    >>> from scipy.stats import circstd
    >>> circstd([0, 0.1*np.pi/2, 0.001*np.pi, 0.03*np.pi/2])
    0.063564063306
    """
    samples, sin_samp, cos_samp, mask = _circfuncs_common(samples, high, low,
                                                          nan_policy=nan_policy)
    if mask is None:
        sin_mean = sin_samp.mean(axis=axis)
        cos_mean = cos_samp.mean(axis=axis)
    else:
        nsum = np.asarray(np.sum(~mask, axis=axis).astype(float))
        nsum[nsum == 0] = np.nan
        sin_mean = sin_samp.sum(axis=axis) / nsum
        cos_mean = cos_samp.sum(axis=axis) / nsum
    R = np.hypot(sin_mean, cos_mean)

    return ((high - low)/2.0/pi) * np.sqrt(-2*np.log(R))

def circstd(arr, low=-180, high=180, axis=None):
    return stats.circstd(arr, high, low, axis)

def circ2lin(xs, ys, period=360):
    def shift2line(xs, ys, b, period=360):
        """
        Shift points greater than period/2 y distance away from diagonal + b closer to the line.
        """
        yline = xs + b
        ydist = ys - yline
        ys[ydist > period/2.] -= period
        ys[ydist < -period/2.] += period
        return ys
    
    # Take 1st guess at line with slope 1, shift wrapped values closer to the line
    y0 = ys[(xs>0) & (xs<10)].mean()
    x0 = 5
    b0 = y0 - x0
    ys = shift2line(xs, ys, b0, period)
    
    # Fit data to line with slope 1 (find b=y-intercept)
    def fitfunc(x, b):
        return x+b
    params, _ = curve_fit(fitfunc, xs, ys)
    b = params[0]
    # Re-shift to best fit line
    ys = shift2line(xs, ys, b, period)
    
    return ys, b

def tile(arr, period=360):
        arr2 = np.tile(arr, 3)
        arr2[:arr.size] -= period
        arr2[-arr.size:] += period
        return arr2

def circ_moving_average(a, n=3, low=-180, high=180):
    assert len(a) > n # ensure that the array is long enough
    assert n%2 != 0 # make sure moving average is odd, or this screws up time points
    shoulder = (n-1) / 2
    ma = np.zeros(len(a))
    ind0 = np.arange(-shoulder, shoulder+1)
    inds = np.tile(ind0, (len(a)-2, 1)) + np.arange(1, len(a)-1)[:, None]
    ma[1:-1] = circmean(a[inds], low, high, axis=1)
    ma[[0, -1]] = a[[0, -1]]
    return ma


# Miscellaneous
def contains(list1, list2):
    for source in list1:
        for target in list2:
            if source in target:
                return target
    return False

def xcorr(a, v, sampling_period=1, norm=True, mode='same'):
    """
    Computes normalized ross-correlation between a and v.
    :param a: 1-d vector.
    :param v: 1-d vector of same length as a.
    :param sampling_period: time steps associated with indices in a and v.
    :return: t, delay a_t - v_t (if t>0, a comes after v)
            xc, full cross-correlation between a and v.
    """
    if not len(a)==len(v):
        print 'len(a) must equal len(v).'
        return
    if norm:
        a = (a - np.mean(a)) / np.std(a)
        v = (v - np.mean(v)) / np.std(v)
    l = len(a)
    a = a / (len(a) - 1)
    xc = np.correlate(a, v, mode)
    if mode == 'full':
        t = np.arange(-(l-1), (l-1)+1)*sampling_period
    elif mode == 'same':
        if l%2 == 1:
            t = np.arange(-(l/2), l/2+1) * sampling_period
        else:
            t = (np.arange(-l/2, l/2) + 0.5) * sampling_period
    return t, xc

def neighbouring_xcorr(img, save=True, vmin=.8, vmax=1):    # not used
    def xcorr_square(y, x, img):
        a = np.zeros(8)
        m = 0
        c = img[:, y, x]
        c = (c - c.mean())/c.std()/c.size
        for i in range(x-1, x+2):
            for j in range(y-1, y+2):
                if i==x and j==y: continue
                p = img[:, j, i]
                p = (p - p.mean())/p.std()
                a[m] = (c*p).sum()
                m += 1
        return a.mean()

    newimg = np.zeros((img.shape[-2], img.shape[-1]))
    for y in range(1, img.shape[-2]-1):
        for x in range(1, img.shape[-1]-1):
            newimg[y, x] = xcorr_square(y, x, img)
    
    if save:
        plt.imshow(newimg, vmin=vmin, vmax=vmax)
        plt.savefig(newimg)
        
    return newimg

def nan2zero(arr, make_copy=False):
    if make_copy:
        arr2 = copy.copy(arr)
        arr2[np.isnan(arr2)] = 0
        return arr2
    else:
        arr[np.isnan(arr)] = 0

def zero2nan(arr, make_copy=False):
    if make_copy:
        arr2 = copy.copy(arr)
        arr2[arr2 == 0] = np.nan
        return arr2
    else:
        arr[arr == 0] = np.nan

def nansem(arr, axis):
    std = np.nanstd(arr, axis=axis)
    n = (np.isnan(arr)==False).sum(axis=axis)
    sem = std / np.sqrt(n)
    return sem

def round(x, base=180, dir='up'):
    if dir == 'up':
        return int(base * np.ceil(float(x)/base))
    else:
        return int(base * np.floor(float(x)/base))

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        # linear interpolation of NaNs
        nans, x= nan_helper(y)
        y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def interpolate(x, y, mask, kind='linear'):
    """
    Replaces masked values with linearly interpolated values in x, y.
    """
    f = interp1d(x[~mask], y[~mask], kind, axis=-1)
    return f(x)

def centroid_weightedring(data):
    if len(data.shape) != 2:
        print 'Wrong Dimension of signals for calculating phase.'
        return nan
    phase = np.zeros(data.shape[0])
    pi = np.pi
    num = data.shape[1]
    # angle = np.arange(num)*2.0*pi/num - pi
    angle = wrap(np.arange(num) * 2.0 * pi / num - pi + pi/16., cmin=-pi, cmax=pi)
    for irow, row in enumerate(data):
        x = row * np.cos(angle)
        y = row * np.sin(angle)
        phase[irow] = np.arctan2(y.mean(), x.mean())
    return phase

def centroid_weightedring_phaseandlength(data):
    if len(data.shape) != 2:
        print 'Wrong Dimension of signals for calculating phase.'
        return nan
    phase = np.zeros(data.shape[0])
    length = np.zeros(data.shape[0])
    pi = np.pi
    num = data.shape[1]
    angle = wrap(np.arange(num) * 2.0 * pi / num - pi + pi/16., cmin=-pi, cmax=pi)
    for irow, row in enumerate(data):
        x = row * np.cos(angle)
        y = row * np.sin(angle)
        ym = y.mean()
        xm = x.mean()
        phase[irow] = np.arctan2(ym, xm)
        length[irow] = np.sqrt(ym**2+xm**2)
    return phase, length

def circdis(x, y, low=-180, high=180):
    if len(x) != len(y):
        print 'two input lists are of different length ! !'
        return 0
    z = np.zeros(len(x))
    z = np.array(x) - np.array(y)
    for index in range(len(z)):
        if z[index] < low:
            z[index] += 360
        elif z[index] >= high:
            z[index] -= 360
    return np.abs(z)

def circsubtract(x, y, low=-180, high=180):
    if len(x) != len(y):
        print 'two input lists are of different length ! !'
        return 0
    z = np.zeros(len(x))
    z = np.array(x) - np.array(y)
    periodicity = high-low
    for index in range(len(z)):
        if z[index] < low:
            z[index] += periodicity
        elif z[index] >= high:
            z[index] -= periodicity
    return np.array(z)

def circ_dist_item(a, b, low=-180, high=180):
    z = a - b
    if z < low:
        z += 360
    elif z >= high:
        z -= 360
    return np.abs(z)

def circplus(x, y, low=-180, high=180):
    z = x + y
    for item in z:
        if item > high:
            item -= 360
    return z

def scatter_to_pcolor(x_input, y_input, binx, biny, xmin=0, ymin=0, xmax=0, ymax=0):
    if len(x_input) != len(y_input):
        return np.nan, np.nan, np.nan
    if xmax == 0:
        xmax = np.max(x_input)
    if ymax == 0:
        ymax = np.max(y_input)
    x = np.arange(xmin, xmax+1e-10, (xmax-xmin)*1.0/(binx))
    y = np.arange(ymin, ymax+1e-10, (ymax-ymin)*1.0/(biny))
    D = np.zeros((biny, binx))
    for index in range(len(x_input)):
        x_index = np.floor((x_input[index] - xmin) * 1.0 / (xmax - xmin) * binx)
        y_index = np.floor((y_input[index] - ymin) * 1.0 / (ymax - ymin) * biny)
        if (x_index < 0) | (x_index >= binx) | (y_index < 0) | (y_index >= biny):
            continue
        D[y_index][x_index] += 1

    return x, y, D

def scatter_to_errorbar(x_input, y_input, binx, xmin=0, xmax=0):
    if len(x_input) != len(y_input):
        return np.nan, np.nan, np.nan
    if xmax == 0:
        xmax = np.max(x_input)
    x = np.arange(xmin + (xmax-xmin)/2.0/(binx), xmax, (xmax-xmin)*1.0/(binx))
    y = []
    y_mean = []
    y_sem = []
    for index in range(binx):
        y.append([])
    for index in range(len(x_input)):
        if math.isnan(x_input[index]):
            continue
        x_index = int(np.floor((x_input[index] - xmin) * 1.0 / (xmax - xmin) * binx))
        if (x_index < 0) | (x_index >= binx):
            continue
        y[x_index].append(y_input[index])
    for index in range(binx):
        y_mean.append(np.mean(y[index]))
        y_sem.append(stats.sem(y[index]))

    y_mean = np.array(y_mean)
    y_sem = np.array(y_sem)
    return x, y_mean, y_sem

def scatter_to_errorbar2(x_input, y_input, binx, xmin=0, xmax=0):
    if len(x_input) != len(y_input):
        return np.nan, np.nan, np.nan
    if xmax == 0:
        xmax = np.max(x_input)
    x = np.arange(xmin + (xmax-xmin)/2.0/(binx), xmax, (xmax-xmin)*1.0/(binx))
    y = []
    y_mean = []
    y_sem = []
    for index in range(binx):
        y.append([])
    for index in range(len(x_input)):
        if math.isnan(x_input[index]):
            continue
        x_index = int(np.floor((x_input[index] - xmin) * 1.0 / (xmax - xmin) * binx))
        if (x_index < 0) | (x_index >= binx):
            continue
        y[x_index].append(y_input[index])
    for index in range(binx):
        y_mean.append(np.nanmean(y[index], axis=0))
        y_sem.append(stats.sem(y[index], axis=0))

    y_mean = np.array(y_mean)
    y_sem = np.array(y_sem)
    return x, y_mean, y_sem

def scatter_to_circ_errorbar(x_input, y_input, binx, xmin=0, xmax=0):
    if len(x_input) != len(y_input):
        return np.nan, np.nan, np.nan
    if xmax == 0:
        xmax = np.max(x_input)
    x = np.arange(xmin + (xmax-xmin)/2.0/(binx), xmax, (xmax-xmin)*1.0/(binx))
    y = []
    y_mean = []
    y_sem = []
    for index in range(binx):
        y.append([])
    for index in range(len(x_input)):
        if math.isnan(x_input[index]):
            continue
        x_index = int(np.floor((x_input[index] - xmin) * 1.0 / (xmax - xmin) * binx))
        if (x_index < 0) | (x_index >= binx):
            continue
        y[x_index].append(y_input[index])
    for index in range(binx):
        y_mean.append(circnanmean(y[index]))
        y_sem.append(circstd(y[index])/np.sqrt(len(y[index])))

    y_mean = np.array(y_mean)
    y_sem = np.array(y_sem)
    return x, y_mean, y_sem

def scatter_to_heatmap(x_input, y_input, z_input, binx, biny, xmin=0, xmax=0, ymin=0, ymax=0, nan_to_num=0):

    if len(x_input) != len(z_input) or len(x_input) != len(y_input):
        return np.nan, np.nan, np.nan
    if xmax == 0:
        xmax = np.max(x_input)
    if ymax == 0:
        ymax = np.max(y_input)
    xs = np.linspace(xmin, xmax, binx+1)
    ys = np.linspace(ymin, ymax, biny+1)
    z = []
    z_mean = np.zeros((binx, biny))

    for j in range(binx):
        temp = []
        for i in range(biny):
            temp.append([])
        z.append(temp)

    for i_frame in range(len(x_input)):
        ix = int(np.floor((x_input[i_frame] - xmin) * 1.0 / (xmax - xmin) * binx))
        iy = int(np.floor((y_input[i_frame] - ymin) * 1.0 / (ymax - ymin) * biny))
        if (ix < 0) | (ix >= binx) | (iy < 0) | (iy >= biny):
            continue
        z[ix][iy].append(z_input[i_frame])

    for ix in range(binx):
        for iy in range(biny):
            z_mean[ix][iy] = np.nanmean(z[ix][iy]) if len(z[ix][iy]) else nan_to_num

    z_mean = np.ndarray.flatten(z_mean)

    return xs, ys, z_mean

def get_stimid_in_window(rec, ts):
    # window: ts: (t0, t1)
    t_ind = (rec.abf.t > ts[0]) & (rec.abf.t < ts[1])
    if len(np.unique(np.round(rec.abf.stimid[t_ind]))) != 1:
        print 'Error: stimid in period: %3.1f-%3.1f not unique!!' % (ts[0], ts[1])
        return np.round(np.mean(rec.abf.stimid))
    return np.unique(np.round(rec.abf.stimid[t_ind]))[0]

def get_stimid_in_window_patching(rec, ts):
    # window: ts: (t0, t1)
    t_ind = (rec.t > ts[0]) & (rec.t < ts[1])
    if len(np.unique(np.round(rec.stimid[t_ind]))) != 1:
        print 'Error: stimid in period: %3.1f-%3.1f not unique!!' % (ts[0], ts[1])
        return np.round(np.mean(rec.stimid))
    return np.unique(np.round(rec.stimid[t_ind]))[0]

def get_xgain_in_window(rec, ts, total_frame=300):
    # window: ts: (t0, t1)
    t_ind = (rec.abf.t > ts[0]) & (rec.abf.t < ts[1])
    # xstim = fc.unwrap(rec.abf.xstim[t_ind][50:-100])
    ind_num = np.sum(t_ind)
    ind0 = int(ind_num*0.25)
    ind1 = int(ind_num*0.7)
    ind2 = int(ind_num * 0.9)
    xstim = fc.unwrap(rec.abf.xstim[t_ind][ind0:ind1])
    xstim2 = fc.unwrap(rec.abf.xstim[t_ind][ind0:ind2])
    return np.round((xstim[-1] - xstim[0])/len(xstim) * rec.abf.subsampling_rate*total_frame/10), \
           np.round((xstim2[-1] - xstim2[0])/len(xstim2) * rec.abf.subsampling_rate*total_frame/10)

def get_xgain_in_window_patching(rec, ts, total_frame=300, t_delay=0.1547):
    # window: ts: (t0, t1)
    t_ind = (rec.t > ts[0]) & (rec.t < ts[1])
    # xstim = fc.unwrap(rec.abf.xstim[t_ind][50:-100])
    xstim = fc.unwrap(rec.xstim[t_ind][30:-50])
    xstim2 = fc.unwrap(rec.xstim[t_ind][30:-100])
    return np.round((xstim[-1] - xstim[0])/len(xstim) * rec.subsampling_rate*total_frame/10), \
           np.round((xstim2[-1] - xstim2[0])/len(xstim2) * rec.subsampling_rate*total_frame/10)

def get_xgain_in_window_fewxpos_patching(rec, ts):
    xstim_stationary_thresh = 0.007
    xstim_stationary_smooth_length = 10
    xstim_gain_smooth_length = 0.3

    ind = [0, 0]
    ind[0] = np.where(rec.t_orig >= ts[0])[0][0]
    ind[-1] = np.where(rec.t_orig <= ts[-1])[0][-1]
    inds = np.arange(ind[0], ind[-1])
    xstim = rec.xstim_orig[inds]
    xstim_stationary = np.abs(np.gradient(xstim)) < xstim_stationary_thresh
    inds = get_contiguous_inds(xstim_stationary, trim_left=0, keep_ends=True, min_contig_len=5)
    xstim_interval_length = []
    for ind in inds:
        xstim_interval_length.append(len(ind) / xstim_stationary_smooth_length * xstim_stationary_smooth_length)
    xstim_interval = max(set(xstim_interval_length), key=xstim_interval_length.count)
    gain = 1e4 / xstim_interval
    gain = np.round(np.round(gain/xstim_gain_smooth_length) * xstim_gain_smooth_length)
    return gain

def get_ystim_in_window(rec, ts):
    # window: ts: (t0, t1)
    t_ind = (rec.abf.t > ts[0]) & (rec.abf.t < ts[1])
    if len(np.unique(np.round(rec.abf.ystim[t_ind]))) != 1:
        # print 'Error: ystim in period: %3.1f-%3.1f not unique!!' % (ts[0], ts[1])
        return np.round(np.mean(rec.abf.ystim))
    return np.unique(np.round(rec.abf.ystim[t_ind]))[0]

def get_ystim_in_window_patching(rec, ts, num=10):
    # window: ts: (t0, t1)
    alpha = 10.0 / num
    t_ind = (rec.t > ts[0]) & (rec.t < ts[1] - 1)
    ystim = rec.ystim[t_ind] / alpha
    if len(np.unique(np.round(ystim))) != 1:
        # print 'Error: ystim in period: %3.1f-%3.1f not unique!!' % (ts[0], ts[1])
        return np.round(np.mean(rec.stimid))
    return np.unique(np.round(ystim))[0]

def deg2mm(deg, ball_diam_mm=6.34):
    circumference_mm = ball_diam_mm*np.pi
    mm = deg*(circumference_mm/360)
    return mm

def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    """Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')

def confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    se = stats.sem(a)
    h = se * stats.t._ppf((1+confidence)/2., n-1)
    return h

def t_test_twosample_ownwrite(a, b, sign_positive=True):
    N1 = len(a)
    N2 = len(b)
    if N1 <= 2 or N2 <= 2:
        raise NameError("Samples for t-test are too small in size!")
    # For unbiased max likelihood estimate we have to divide the var by N-1, and therefore the parameter ddof = 1
    var_a = a.var(ddof=1)
    var_b = b.var(ddof=1)

    # std deviation
    s = np.sqrt(var_a / N1 + var_b / N2)

    ## Calculate the t-statistics
    if sign_positive:
        t = (a.mean() - b.mean()) / s
    else:
        t = (b.mean() - a.mean()) / s
    ## Compare with the critical t-value
    # Degrees of freedom
    df = N1 + N2 - 2

    # p-value after comparison with the t
    p = 1 - stats.t.cdf(t, df=df)

    return p


def vertical_distance_least_squares(xs, ys):
    N = len(xs)
    xm = np.mean(xs)
    ym = np.mean(ys)
    lxy = 0
    lxx = 0
    lyy = 0
    for i in range(N):
        lxx += (xs[i] - xm) * (xs[i] - xm)
        lxy += (xs[i] - xm) * (ys[i] - ym)
        lyy += (ys[i] - ym) * (ys[i] - ym)

    a = (lyy - lxx) + np.sqrt(np.square(lyy-lxx) + 4*lxy*lxy) / 2. / lxy
    b = ym - a * xm
    vd = 0
    alpha = np.sqrt(1+a*a)
    for i in range(N):
        vd += np.abs(ys[i] - a*xs[i] - b) / alpha
    vdm = vd / N
    return a, b, vdm



































