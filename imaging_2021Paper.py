import behaviour_2021Paper as bh
import tiffile_2021Paper as tiff
import functions_2021Paper as fc
import fbscreening_2021Paper as fbs
import traveldirection_2021Paper as td

import xml.etree.ElementTree as ET
import copy, os, glob, csv, time
from bisect import bisect

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from matplotlib.collections import LineCollection
import plotting_help as ph
from matplotlib.ticker import ScalarFormatter
from matplotlib import animation

import numpy as np
from scipy import ndimage, stats, optimize
from scipy.interpolate import interp1d
from skimage import transform
import circstats as cs
from scipy.ndimage.filters import gaussian_filter
import math, matplotlib
import seaborn as sns
import skimage.measure as sm
from PIL import Image

reload(bh)
reload(fc)
reload(ph)

# The classes

class Channel():
    """
    Class for handling one channel in a TIF recording.
    """

    def __init__(self, file, celltype):
        self.file = file
        self.celltype = celltype


class GCaMP():
    """
    Parent Class for handling TIF files.
    """
    def __init__(self, folder, celltype, celltype2=None):
        self.basename = folder + folder.split(os.path.sep)[-2]

        # TIF file
        greenfile = \
        (glob.glob(folder + os.path.sep + '*reg_gauss.tif') +
         glob.glob(folder + os.path.sep + 'C1*reg.tif') +
         glob.glob(folder + os.path.sep + '*_reg.tif') +
         glob.glob(folder + os.path.sep + '*_reg_interleaved.tif') +
         glob.glob(folder + os.path.sep + 'C1*.tif') +
         glob.glob(self.basename + '.tif')
         )[0]

        redfile = \
        (glob.glob(folder + os.path.sep + 'C2*reg.tif') +
        glob.glob(folder + os.path.sep + 'C2*.tif') +
        glob.glob(folder + os.path.sep + '*_red.tif')
        )

        if redfile and celltype2:
            redfile = redfile[0]
        else:
            redfile = None
        self.c1 = Channel(greenfile, celltype)
        self.c2 = Channel(redfile, celltype2)

        self.folder = folder
        self.basename = folder + folder.split(os.path.sep)[-2]
        self.xmlfile = self.basename + '.xml'

        self.info = {}
        self.info['filename'] = folder
        self.info['basename'] = self.basename
        self.info['pockels'] = self._get_pockels(self.xmlfile)
        self.info['laserpower_mW'] = self._get_laser_power(self.xmlfile)


        transformfile = glob.glob(self.basename + '_transform.csv')
        if transformfile:
            self.transformfile = transformfile[0]
            self.reg_header, self.reg_xys, self.reg_dist = self._get_reg_xys(self.transformfile)


        ctlmaskfile = glob.glob(folder + os.path.sep + '*_ctlmask.tif')
        if ctlmaskfile:
            self.c1.ctlmaskfile = ctlmaskfile[0]
            self.c1.ctlmask = tiff.imread(self.c1.ctlmaskfile) < 50

    def open(self, tiffile):
        # Open tif file.
        tif = tiff.imread(tiffile)

        # Reshape tif so that multi-z and single-z tifs have the same shape dimension
        if len(tif.shape) == 3:
            newshape = list(tif.shape)
            newshape.insert(1, 1)
            newshape = tuple(newshape)
            tif = tif.reshape(newshape)

        # Extract acquisition info.
        self.len = tif.shape[0]
        self.zlen = tif.shape[1]
        self.info['dimensions_pixels'] = tif.shape[-2:]
        self.info['zlen'] = self.zlen

        return tif

    def roi(self, tif, mask, metric=np.mean):
        mask = np.tile(mask, (self.len, 1, 1, 1))
        masked = tif[mask]
        masked = masked.reshape((self.len, masked.size / self.len))
        return metric(masked, axis=-1)

    def norm_over_time(self, signal, mode='dF/F0'):
        if mode == 'zscore':
            newsignal = signal - signal.mean(axis=0)
            div = newsignal.std(axis=0)
            div[div == 0] = 1
            newsignal /= div
        elif mode == 'dF/F0':
            signal_sorted = np.sort(signal, axis=0)
            f0 = signal_sorted[:int(len(signal)*.05)].mean(axis=0)
            # std = signal.std(axis=0)
            # std[std==0] = np.nan
            if f0.ndim > 1:
                f0[f0 == 0] = np.nan
            newsignal = (signal - f0) / f0
        elif mode == 'linear':
            signal_sorted = np.sort(signal, axis=0)
            f0 = signal_sorted[:int(len(signal)*.05)].mean(axis=0)
            fm = signal_sorted[int(len(signal)*.97):].mean(axis=0)
            if f0.ndim > 1:
                f0[f0 == 0] = np.nan
                fm[fm == 0] = np.nan
            newsignal = (signal - f0) / (fm - f0)
        return newsignal

    def norm_per_row(self, signal):
        signal = signal - signal.mean(axis=-1).reshape((len(signal), 1))
        signal = signal / signal.std(axis=-1).reshape((len(signal), 1))
        return signal

    def plot_masks(self, show_skel=True, show_mask=True, fig_filename=''):
        ax = plt.subplot(111)
        if show_mask:
            img = self.tifm
        else:
            img = self.tif
        plt.pcolormesh(img.var(axis=0).max(axis=0))
        ny, nx = self.tif.shape[-2:]
        plt.xlim(0, nx)
        plt.ylim(0, ny)
        ax.invert_yaxis()

        if show_skel:
            xs = [x for x, _ in self.points]
            ys = [y for _, y in self.points]
            plt.scatter(xs, ys, c=np.arange(len(xs)), marker=',', lw=0, cmap=plt.cm.jet)

        if fig_filename:
            plt.savefig(fig_filename, bbox_inches='tight', dpi=150)

    def plot(self, data='skel', cmap=plt.cm.CMRmap, vmin=None, vmax=None):
        if hasattr(self, data):
            data = getattr(self, data)
        else:
            print 'Undefined data type.'
            return

        plt.figure(1, (6, 20))
        ax1 = plt.subplot(111)
        xedges = np.arange(0, data.shape[1] + 1)
        img = plt.pcolormesh(xedges, range(self.len), data, cmap=cmap)
        if vmin:
            img.set_clim(vmin=vmin)
        if vmax:
            img.set_clim(vmax=vmax)
        ax1.invert_yaxis()
        plt.xlim([0, dxedges[-1]])

    def _get_pv_times(self, xmlfile):
        tree = ET.parse(xmlfile)
        root = tree.getroot()
        times = []
        for sequence in root.getchildren():
            for frame in sequence:
                if 'relativeTime' in frame.attrib:
                    time = float(frame.attrib['relativeTime'])
                    times.append(time)
        times = np.array(times)
        return times

    def _get_pockels(self, xmlfile):
        with open(xmlfile) as xml:
            for line in xml:
                if 'laserPower_0' in line:
                    value = int(line.split()[-2].split('"')[-2])
                    return value

    def _get_laser_power(self, xmlfile):
        with open(xmlfile) as xml:
            for line in xml:
                if 'twophotonLaserPower_0' in line:
                    value = int(line.split()[-2].split('"')[-2])
                    return value

    def _get_reg_xys(self, transformfile):
        with open(transformfile) as fh:
            v = csv.reader(fh)
            table = []
            for line in v:
                try:
                    table.append(map(float, line))
                except ValueError:
                    header = line
        table = np.array(table).T
        table[0] -= 1
        dimlens = np.max(table[:3], axis=-1).astype(int) + 1
        xys = np.zeros(list(dimlens) + [2]).astype(float)
        for (c, z, t, x, y) in table.T:
            c, z, t = map(int, [c, z, t])
            xys[c, z, t] = [x, y]
        header = [h[0].lower() for h in header]

        def f(ar):
            return np.hypot(*ar)
        dist = np.apply_along_axis(f, 0, xys.T).T
        dist = dist[0].max(axis=0)
        return header, xys, dist

    def get_subregdist_mask(self, thresh=5, regdist=None):
        if regdist is None:
            regdist = self.reg_dist
        mask = (regdist - regdist.mean()) < thresh
        mask = ndimage.binary_erosion(mask, iterations=1)
        return mask


class Bridge(GCaMP):
    def __init__(self, folder, celltype, celltype2=None, width=16, mode_phase_algo='pb', seq_mask=0,
                 tomato_background=False, prairie2019=False, **kwargs):
        """
        self.gctypes = []

        maskfile = glob.glob(self.folder + os.path.sep + '*_mask.tif')
        if maskfile:
            self.maskfile = maskfile[0]
            self.mask = tiff.imread(self.maskfile) > 0

        skelfile = glob.glob(self.folder + os.path.sep + '*_skel.tif')
        if skelfile:
            self.skelfile = skelfile[0]
            self.skelmask = tiff.imread(self.skelfile) > 0
        """
        GCaMP.__init__(self, folder, celltype, celltype2)

        segfile = glob.glob(self.folder + os.path.sep + '*pbmask.tif')
        segfile_ch1 = glob.glob(self.folder + os.path.sep + '*ch1_pbmask.tif')
        segfile_ch2 = glob.glob(self.folder + os.path.sep + '*ch2_pbmask.tif')
        segfile_ch1 = segfile_ch1 if segfile_ch1 else segfile
        segfile_ch2 = segfile_ch2 if segfile_ch2 else segfile

        if segfile_ch1:
            self.c1.maskfile = segfile_ch1[0]
            self.c2.maskfile = segfile_ch2[0]

            ### To deal with the new mask.tif
            if len(tiff.imread(self.c1.maskfile).shape) == 4:
                self.c1.mask = tiff.imread(self.c1.maskfile)[:,0,:]
                self.c2.mask = tiff.imread(self.c2.maskfile)[:,0,:]
            else:
                self.c1.mask = tiff.imread(self.c1.maskfile)
                self.c2.mask = tiff.imread(self.c2.maskfile)
            if np.max(self.c1.mask) > 255:
                self.c1.mask = self.c1.mask / 65280. * 255
            if np.max(self.c2.mask) > 255:
                self.c2.mask = self.c2.mask / 65280. * 255

            # if prairie2019:
            #     if len(tiff.imread(self.c1.maskfile).shape) == 4:
            #         self.c1.mask = self.c2.mask = np.array(Image.open(self.c1.maskfile))[:,0,:]
            #     else:
            #         self.c1.mask = self.c2.mask = np.array(Image.open(self.c1.maskfile))
            # else:
            #     if len(tiff.imread(self.c1.maskfile).shape) == 4:
            #         self.c1.mask = self.c2.mask = tiff.imread(self.c1.maskfile)[:,0,:]
            #     else:
            #         self.c1.mask = self.c2.mask = tiff.imread(self.c1.maskfile)

        self.width = width
        self.seq_mask = seq_mask
        self.tomato_background = tomato_background
        self.prairie2019 = prairie2019
        self.open()

        if hasattr(self.c1, 'ctlmask'):
            self.c1.ctl = self.roi(self.c1.tif, self.c1.ctlmask)

    def open(self, mode_phase_algo='pb'):
        tif = GCaMP.open(self, self.c1.file)
        if len(tif.shape) == 4:
            self.c1.tif = tif
            self.c2.tif = GCaMP.open(self, self.c2.file) if self.c2.file else None
        elif len(tif.shape) == 5:
            self.c1.tif = tif[:, :, 0]
            self.c2.tif = tif[:, :, 1]

        # mask.shape: nz, nx, ny. This is to deal with masks that only covers part of z planes
        if not self.c1.tif.shape[1] == self.c1.mask.shape[0]:
            mask_ch1 = np.zeros(list(self.c1.tif.shape)[-3:])
            if self.seq_mask == 0:
                mask_ch1[:self.c1.mask.shape[0]] = self.c1.mask
            else:
                mask_ch1[-self.c1.mask.shape[0]:] = self.c1.mask
            self.c1.mask = mask_ch1

        if not self.c2.tif is None:
            if not self.c2.tif.shape[1] == self.c2.mask.shape[0]:
                mask_ch2 = np.zeros(list(self.c2.tif.shape)[-3:])
                if self.seq_mask == 0:
                    mask_ch2[:self.c2.mask.shape[0]] = self.c2.mask
                else:
                    mask_ch2[-self.c2.mask.shape[0]:] = self.c2.mask
                self.c2.mask = mask_ch2

        self.process_channel(self.c1, mode_phase_algo)
        self.process_channel(self.c2, mode_phase_algo, tomato_background=self.tomato_background)

    def process_channel(self, channel, mode_phase_algo='pb', tomato_background=False):
        if channel.celltype is None: return
        channel.ppg = 1.
        # Parse images into glomerular representation1 and normalize

        channel.mean_p = self.get_pixel_mean(channel.tif, channel.mask)
        channel.mean_pn = self.norm_over_time(channel.mean_p, mode='dF/F0')
        channel.mean_pl = self.norm_over_time(channel.mean_p, mode='linear')

        channel.a = self._get_gloms(channel.tif, channel.mask, channel.celltype)
        if tomato_background:
            channel.a = channel.a / self.c1.a
        channel.an = self.norm_over_time(channel.a, mode='dF/F0')
        channel.az = self.norm_over_time(channel.a, mode='zscore')
        channel.al = self.norm_over_time(channel.a, mode='linear')
        channel.mean_an = np.nanmean(channel.an, axis=-1)
        channel.mean_al = np.nanmean(channel.al, axis=-1)
        channel.mean_az = np.nanmean(channel.az, axis=-1)

        channel.acn = self.get_continuous_bridge(channel.an, channel.celltype)
        channel.acl = self.get_continuous_bridge(channel.al, channel.celltype)
        channel.ac = self.get_continuous_bridge(channel.a, channel.celltype)
        if mode_phase_algo == 'pb':
            cval = 8
            if channel.celltype == 'PFLC':
                cval = 8
            channel.phase, channel.period = self.get_fftphase(channel.acl, div=1, mode='constant', cval=cval)
            channel.phase_dff0, _ = self.get_fftphase(channel.acn, div=1, mode='constant', cval=cval)
            channel.phase_raw, _ = self.get_fftphase(channel.ac, div=1, mode='constant', cval=cval)
        else:   # 'eb' mode
            channel.phase, channel.period = self.get_centroidphase(channel.acl)
        # edited on Feb 7th, 2021
        # if channel.celltype == 'EIP': channel.phase += 45
        if (channel.celltype == 'PEN') or (channel.celltype == 'PFN'):
            channel.phase = channel.phase - 45
            channel.phase_raw = channel.phase_raw - 45
            channel.phase_dff0 = channel.phase_dff0 - 45
        channel.phase = fc.wrap(channel.phase + 22.5-180)
        channel.phase_raw = fc.wrap(channel.phase_raw + 22.5 - 180)
        channel.phase_dff0 = fc.wrap(channel.phase_dff0 + 22.5 - 180)

        sorted_al = np.sort(channel.al, axis=-1)
        top_two = sorted_al[:, -2:]
        channel.glo_peak_fluo = top_two.mean(axis=-1)
        channel.phasef = fc.circ_moving_average(channel.phase)
        channel.power = self.get_fftpower_at_peak(channel.acn, channel.period)
        channel.peak = np.nanmax(channel.an, axis=1)
        channel.xedges = np.arange(19)

        # Interpolate glomeruli and cancel phase
        channel.xedges_interp, channel.an_cubspl = self.interpolate_glomeruli(channel.an, channel.celltype)
        channel.an_nophase = self.cancel_phase(channel.an_cubspl, channel.phase, 10, channel.celltype)

        ind0 = np.arange(0, 10)
        idx = np.tile(ind0, (18, 1)) + 10*np.arange(18)[:, None]
        channel.an_nophase_sub = np.nanmean(channel.an_nophase[:, idx], axis=-1)

        _, channel.az_cubspl = self.interpolate_glomeruli(channel.az, channel.celltype)
        channel.az_nophase = self.cancel_phase(channel.az_cubspl, channel.phase, 10, channel.celltype)
        _, channel.al_cubspl = self.interpolate_glomeruli(channel.al, channel.celltype)
        channel.al_nophase = self.cancel_phase(channel.al_cubspl, channel.phase, 10, channel.celltype)

        # Calculate Right - Left Bridge using zscore
        x = channel.xedges[:-1]
        channel.rightz = np.nanmean(channel.az[:, x >= 9], axis=1)
        channel.leftz = np.nanmean(channel.az[:, x < 9], axis=1)
        channel.rmlz = channel.rightz - channel.leftz
        channel.rplz = channel.rightz + channel.leftz

        channel.rightn = np.nanmean(channel.an[:, x >= 9], axis=1)
        channel.leftn = np.nanmean(channel.an[:, x < 9], axis=1)
        channel.rmln = channel.rightn - channel.leftn
        channel.rpln = channel.rightn + channel.leftn

        channel.rightl = np.nanmean(channel.al[:, x >= 9], axis=1)
        channel.leftl = np.nanmean(channel.al[:, x < 9], axis=1)
        channel.rmll = channel.rightl - channel.leftl
        channel.rpll = channel.rightl + channel.leftl

        # Calculate the expected projection onto the ellipsoid body given known anatomy.
        channel.an_ebproj = self.get_ebproj(channel.an.T, channel.celltype).T

    def get_continuous_bridge(self, bridge, celltype):
        """
        Since glomeruli L9 and R1 are adjacent and map to the same ellipsoid body wedge,
        these are averaged into a single glomerulus in the bridge for the purpose of extracting
        a phase and period from the signal.
        """
        if celltype == 'EIP' or celltype == 'EPG' or celltype == 'PFR':
            bridgec = np.zeros((len(bridge), self.width))
            if self.width == 16:
                bridgec[:, :8] = bridge[:, 1:9]
                bridgec[:, -8:] = bridge[:, -9:-1]
            else:
                bridgec[:, :7] = bridge[:, 1:8]
                bridgec[:, -7:] = bridge[:, -8:-1]
                bridgec[:, 7] = bridge[:, 8:10].mean(axis=-1)
        elif celltype == 'PEN' or celltype == 'PFN':
            bridgec = np.zeros_like(bridge)
            bridgec[:] = bridge[:]
            bridgec[:, 8:10] = np.mean([bridge[:, :2], bridge[:, -2:]], axis=0)
        elif celltype == 'PFLC':
            bridgec = np.zeros((len(bridge), 14))
            bridgec[:, :7] = bridge[:, 2:9]
            bridgec[:, -7:] = bridge[:, -9:-2]
        else:
            bridgec = bridge
        nans = np.isnan(bridgec[0]).sum()
        if nans > 0:
            if nans == 1:
                print '%i glomerulus is missing.' %nans
                print bridgec[0]
            else:
                print '%i glomeruli are missing.' %nans
                print bridgec[0]
            fc.nan2zero(bridgec)
        return bridgec

    def get_ebproj(self, pb, celltype, proj='merge'):
        """
        Projects a matrix with protecerebral bridge as its first or last (0 or -1) axis to
        an ellipsoid body axis. Assumes 18 glomeruli with no interpolation.
        :param pb: PB matrix. Axis 0 or -1 must be the protocerebral bridge axis (ie. len=18)
        :param celltype: 'EIP' or 'PEN', designating the celltype.
        :return: eb matrix.
        """
        def map_eb2pb(ebcol, celltype):
            if celltype == 'PEN':
                col1 = int(((ebcol - 1) % (8))) # Left bridge
                col2 = int((ebcol % (8) + 10))  # Right bridge
                pbcols = [col1, col2]
            if celltype == 'PFN':
                # e2pmap = [0, 10, 1, 11, 2, 12, 3, 13, 4, 14, 5, 15, 6, 16, 7, 17]
                e2pmap = [0, 17, 1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7, 16]
                pbcols = e2pmap[ebcol]
            elif celltype == 'EIP' or celltype == 'EPG' or celltype == 'PFLC':
                if ebcol % 2 == 0:  # If eb wedge is even, then the pb glomerulus is on the left side
                    if ebcol == 0:
                        pbcols = [8]
                    else:
                        pbcols = [ebcol / 2]
                else:  # If eb wedge is odd, then the pb glomerulus is on the right side
                    pbcols = [ebcol / 2 + 9]
            elif celltype == 'PFR':
                e2pmap = [8,1,10,2,11,3,12,4,13,5,14,6,15,7,16,9]
                pbcols = e2pmap[ebcol]
            return np.array(pbcols)

        pb2eb_PFR = {1:(0,1), 2:(0,3), 3:(0,5), 4:(0,7), 5:(0,9), 6:(0,11), 7:(0,13), 8:(0,0),
                     9:(1,15),10:(1,2),11:(1,4),12:(1,6),13:(1,8),14:(1,10),15:(1,12),16:(1,14)}

        nwedges_dict = {'EIP': 16, 'PEN': 8, 'EPG': 16, 'PFN': 16, 'PFR': 16, 'PFLC': 14}
        ncols = nwedges_dict[celltype]

        if pb.shape[-1] == 18:
            pbdata = pb.T
        else:
            pbdata = pb

        if proj == 'merge':
            shape = list(pbdata.shape)
            shape = [ncols] + shape[1:]
            eb = np.zeros(shape)
            eb[:] = np.nan
            for ebcol in range(ncols):
                pbcols = map_eb2pb(ebcol, celltype)
                ebi = np.nansum(pbdata[pbcols], axis=0)
                eb[ebcol] = ebi

            if pb.shape[-1] == 18:
                return eb.T
            else:
                return eb

        elif proj == 'preserve':
            shape = list(pbdata.shape)
            shape = [2, ncols] + shape[1:]
            eb = np.zeros(shape)
            eb[:] = np.nan
            for ebcol in range(ncols):
                pbcols = map_eb2pb(ebcol, celltype)
                if celltype == 'EIP' or celltype == 'EPG' or celltype == 'PFLC':
                    eb[ebcol%2, ebcol] = pbdata[pbcols]
                elif celltype == 'PEN' or celltype == 'PFN':
                    eb[:, ebcol] = pbdata[pbcols]
                elif celltype == 'PFR':
                    col0, col1 = pb2eb_PFR[int(pbcols)]
                    eb[col0,col1] = pbdata[pbcols]

            if pb.shape[-1] == 18:
                return np.array([ebi.T for ebi in eb])
            else:
                return eb

    def get_xcphase(self, data, period):
        nx = data.shape[1]
        xs = np.arange(nx * 2)
        f = 1. / period
        siny = np.sin(2 * np.pi * f * xs) * 2
        phases = np.zeros(len(data))
        for irow, row in enumerate(data):
            #     row = bh.fc.butterworth(row, 5)  # Uncomment to lowpass filter each row before cross-correlation. Much slower, no real difference so far.
            xc = np.correlate(row, siny, mode='full')
            xc = xc[xc.size / 2:]
            phase = xs[np.argmax(xc)]
            phase *= 360 / period
            phases[irow] = phase
        xcphases = fc.wrap(phases)
        return xcphases

    def get_fftphase(self, data, div=1, **kwargs):
        phase, period, _ = fc.powerspec_peak(data, **kwargs)
        return -phase, period[0] / div

    def get_fftpower_at_peak(self, gcn, peak_period):
        power, period, phase = fc.powerspec(gcn, norm=False)
        return power[:, np.where(period >= peak_period)[0][0]]

    def _get_gloms(self, signal, segmask, celltype):
        if not self.prairie2019:
            centroids = [13, 25, 38, 51, 64, 76, 89, 102, 115, 128, 140, 154, 166, 179, 192, 205, 217, 230]
        else:
            # new prairie version, 2019
            centroids = [17, 37, 54, 70, 85, 99, 112, 125, 137, 149, 161, 172, 183, 194, 204, 215, 225, 235]
            # print np.unique(segmask)
        gloms = np.zeros((self.len, len(centroids)))
        gloms.fill(np.nan)
        for glomid, val in enumerate(centroids):  # glomid, mask in enumerate(glomask):
            mask = (segmask == val)
            if mask.sum():
                glom = signal * mask
                gloms[:, glomid] = np.nansum(glom.reshape(self.len, signal[0].size), axis=1) / mask.sum()

        if celltype == 'PEN' or celltype == 'PFN':
            gloms[:, 8:10] = np.nan
        elif celltype == 'EIP' or celltype == 'EPG' or celltype == 'PFR' or celltype == 'PFLC':
            gloms[:, [0, 17]] = np.nan
        else:
            idx = (gloms != 0)
            idn = np.zeros(idx.sum())
            idn = (idn == 0)
            idn = idn.reshape(self.len, idn.size/self.len)
            res = gloms[idn]
            res = res.reshape(self.len, idn.size/self.len)
            return res

        return gloms

    def get_pixel_mean(self, tif, mask):
        # calculate the mean value across all the valid pixels in the tif
        tifm = copy.copy(tif).astype(float)
        tifm[:, mask == False] = np.nan
        return np.nanmean(tifm, axis=tuple(range(1, tifm.ndim)))

    def interpolate_glomeruli(self, glom, celltype, kind='cubic', dinterp=.1):
        tlen, glomlen = glom.shape[:2]
        row_interp = np.zeros((tlen, int(glomlen/dinterp)))
        if celltype == 'EIP' or celltype == 'EPG' or celltype == 'PFR' or celltype == 'PFLC':
            nans, x = fc.nan_helper(np.nanmean(glom, axis=0))
            wrap = np.zeros_like(glom)
            wrap[:, 1:17] = glom[:, 1:17]
            wrap[:, [0, 17]] = glom[:, [8, 9]]
            if celltype == 'PFLC':
                wrap[:, [0, 17]] = glom[:, [8, 9]]
                wrap[:, [1, 16]] = glom[:, [9, 8]]
            x = np.arange(0, 18, 1)
            f = interp1d(x, wrap, kind, axis=-1)
            # f = np.interp(x, wrap, kind, axis=-1)
            # x_interp = np.arange(1, 17, dinterp)
            x_interp = np.arange(.5, 16.5, dinterp)     # modified by Cheng, 2020 Apr
            row_interp[:, int(1./dinterp) : int(17./dinterp)] = f(x_interp)
        elif celltype == 'PEN' or celltype == 'PFN':
            wrap_left = np.zeros((tlen, 10))
            wrap_left[:, 1:9] = glom[:, :8]
            wrap_left[:, [0, 9]] = glom[:, [7, 0]]
            x = np.arange(0, 10, 1)
            fleft = interp1d(x, wrap_left, kind, axis=-1)

            wrap_right = np.zeros((tlen, 10))
            wrap_right[:, 1:9] = glom[:, -8:]
            wrap_right[:, [0, 9]] = glom[:, [17, 10]]
            fright = interp1d(x, wrap_right, kind, axis=-1)

            # x_interp = np.arange(1, 9, dinterp)
            x_interp = np.arange(.5, 8.5, dinterp)      # modified by Cheng, 2020 Apr
            row_interp[:, :int(8/dinterp)] = fleft(x_interp)
            row_interp[:, -int(8/dinterp):] = fright(x_interp)

        fc.zero2nan(row_interp)
        x_interp = np.arange(0, 18, dinterp)
        # plt.plot(row_interp)
        return x_interp, row_interp

    def cancel_phase(self, gc, phase, ppg, celltype, offset=0):
        period = 8 #if celltype=='PEN' else 9
        # period_inds = int(self.period * ppg)
        period_inds = int(period*ppg)
        offset = int(offset * period_inds / 360)
        gc_nophase = np.zeros_like(gc)
        x = np.arange(0, 18, 1./ppg)
        for i in xrange(len(gc)):
            shift = int(np.round((-phase[i] + 180) * period_inds / 360)) + offset
            row = np.zeros(len(x))
            row[:] = np.nan

            if celltype == 'EIP' or celltype == 'EPG' or celltype == 'PFR':
                left_ind = (x < 9) & (x >= 1)
                right_ind = (x >= 9) & (x < 17)
            elif celltype == 'PEN' or celltype == 'PFN':
                left_ind = (x < 8)
                right_ind = (x >= 10)
            elif celltype == 'PFLC':
                left_ind = (x < 9) & (x >= 2)
                right_ind = (x >= 9) & (x < 16)
            row[left_ind] = np.roll(gc[i, left_ind], shift)
            row[right_ind] = np.roll(gc[i, right_ind], shift)

            # row = np.array([gc[i][(j - shift) % period_inds] for j in range(gc.shape[1])])
            gc_nophase[i] = row
        return gc_nophase

    def _get_skel(self, signal, points):
        skel = []
        for x, y in points:
            datapoint = signal[:, y, x]
            if datapoint.sum() > 0:
                skel.append(datapoint)
        skel = np.array(skel).transpose()
        return skel

    def get_transform(self, skelmask, segmask=None, length=25, xmax=180, spacing=8, offset=1, plot=False, save=False):
        # Get control points orthogonal to skeleton
        strings = fc.get_strings(skelmask)
        assert len(strings) == 1
        points = strings[0]

        if segmask is None:
            # Subsample points by spacing, offset optional
            spoints = points[offset::spacing]
        else:
            # Get points where segmask crosses skelmask
            segskel = skelmask & segmask
            segx, segy = np.where(segskel.T)
            segpoints = zip(segx, segy)

            # Sort segmentation points along the skeleton
            segpoints_mask = np.array(map(lambda pt: (pt[0], pt[1]) in segpoints, points))
            spoints = points[segpoints_mask]

        slope = fc.get_slope(spoints)
        v = fc.get_vector(slope, spoints, length, orthogonal=True)
        dst = np.concatenate([v[:, 0], spoints, v[:, 1]])

        # Compute corresponding control points as a rectangular lattice
        ny, nx = skelmask.shape
        xs = np.linspace(0, xmax, len(spoints))
        ys = np.ones(len(spoints))
        src = np.concatenate([np.array([xs, ys * i * length / 2.]).T for i in range(3)])

        # get transform of image
        tform = transform.PiecewiseAffineTransform()
        tform.estimate(src, dst)
        output_shape = (length, xmax)

        if plot:
            plt.figure(1, (nx / 20., ny / 20.))
            rcParams['xtick.direction'] = 'in'
            rcParams['ytick.direction'] = 'in'
            lc = LineCollection(v, array=np.zeros(len(spoints)), cmap=plt.cm.Greys)
            ax = plt.subplot(111, axisbg=plt.cm.RdBu_r(0))
            ax.add_collection(lc)
            ax.set_xlim(0, nx)
            ax.set_ylim(0, ny)
            ax.invert_yaxis()

            mx, my = np.where(self.mask.T)
            plt.scatter(mx, my, c=ph.bgreen, s=35, lw=0, marker=',')

            x, y = points.T
            plt.plot(x, y, c='red', lw=3)

            sx, sy = spoints.T
            plt.scatter(sx, sy, c='black', s=40, zorder=10)

            if save:
                plt.savefig('%s_bridge_transform.png' % (self.basename), bbox_inches='tight')

        return tform, output_shape

    def get_bridge(self, signal, skelmask, mask, splice=False, fill=False, **kwargs):
        # Get transform with output shape
        if hasattr(self, 'tform'):
            tform = self.tform
            output_shape = self.tform_output_shape
        else:
            tform, output_shape = self.get_transform(skelmask, **kwargs)
            self.tform, self.tform_output_shape = tform, output_shape

        # Signal cannot exceed 1
        scale = np.abs(signal).max()
        s = signal / scale  # scale down - signal cannot exceed 1.
        maskw = transform.warp(mask, tform,
                               output_shape=output_shape) < .99  # Define warped mask in order to take mean.
        z = 0  # Only supports z-projection so far.
        data = np.zeros([len(s), output_shape[1]])  # Initialize data array.
        for it in xrange(len(s)):
            d = transform.warp(s[it, z], tform, output_shape=output_shape)
            dm = np.ma.masked_array(d, maskw)
            data[it] = dm.mean(axis=0)

        data = data * scale  # scale back up to original range

        xmax = output_shape[1]
        ppg = xmax / 18
        if 'segmask' in kwargs.keys():
            if splice:  # For PENs: see what the signal looks like it you splice out the two unrepresented center glomeruli
                spliced_data = np.zeros((len(data), xmax - ppg * 2))
                spliced_data[:, :8 * ppg] = data[:, :8 * ppg]
                spliced_data[:, -8 * ppg:] = data[:, -8 * ppg:]
                data = spliced_data
            else:
                splidx = np.arange(8 * ppg, 10 * ppg + 1)
                if fill:  # for PENs: fill in the two unrepresented center glomeruli
                    print 'Filling in center glomeruli.'
                    period = 8 * ppg
                    data[:, splidx] = (data[:, (splidx + period) % (18 * ppg)] + data[:, splidx - period]) / 2
        fc.nan2zero(data)
        return data

    def plot_warped_bridge(self, save=False):
        orig = self.tifm.mean(axis=0).mean(axis=0)
        orig = orig / np.abs(orig).max()
        warped = transform.warp(orig, self.tform, output_shape=self.tform_output_shape)

        ny, nx = orig.shape
        plt.figure(1, (nx / 20., ny / 20.))
        ax = plt.subplot(111)
        plt.pcolormesh(orig)
        plt.xlim(0, self.skelmask.shape[1])
        plt.ylim(0, self.skelmask.shape[0])
        ax.invert_yaxis()

        if save:
            plt.savefig('%s_bridge.png' % (self.basename), bbox_inches='tight')

        ny, nx = warped.shape
        plt.figure(2, (nx / 20., ny / 20.))
        ax = plt.subplot(111)
        plt.pcolormesh(warped)
        ax.invert_yaxis()
        plt.xlim(0, self.tform_output_shape[1])
        plt.ylim(0, self.tform_output_shape[0])

        if save:
            plt.savefig('%s_warped_bridge.png' % (self.basename), bbox_inches='tight')

    def get_centroidphase(self, data):
        """
        project eip intensity to eb, then average to get centroid position
        """
        if self.c1.celltype == 'PFLC':
            data_projection = copy.copy(data)
            data_projection[:, 0] = data[:,6]
            data_projection[:,1::2] = data[:,7:]
            data_projection[:,2::2] = data[:,0:6]
            return fc.centroid_weightedring(data_projection), 7
        else:
            if data.shape[-1] != 16:
                print 'the number of glomerulo is not 16.'
                return nan, nan
            data_eb = copy.copy(data)
            data_eb[:,2::2] = data[:,0:7]
            data_eb[:,0] = data[:,7]
            data_eb[:,1::2] = data[:,8:]
            phase = fc.centroid_weightedring(data_eb)
            period = np.zeros(data.shape[0]).reshape(data.shape[0], 1) + 8
        return phase, period


class FanshapeBody(GCaMP):
    def __init__(self, folder, celltype, celltype2=None, seq_mask=0, seg_algo_fb='auto_seg', wedge_num=16,
                 maskfile_seq=0, tomato_background=False, prairie2019=False, **kwargs):
        """
        seq_mask: indicates the position of mask file in tif z-dim, 0 at the top, z at the bottom
        maskfile_seq: indicates the filename of mask file
        """
        GCaMP.__init__(self, folder, celltype, celltype2)

        if maskfile_seq == 0:
            maskfile = 'fbmask.tif'
        else:
            maskfile = 'seq%i_fbmask.tif' % maskfile_seq

        self.seg_algo = seg_algo_fb
        self.seq_mask = seq_mask
        self.wedge_num = wedge_num
        self.tomato_background = tomato_background

        segfile = glob.glob(self.folder + os.path.sep + '*' + maskfile)
        segfile_ch1 = glob.glob(self.folder + os.path.sep + '*ch1_' + maskfile)
        segfile_ch2 = glob.glob(self.folder + os.path.sep + '*ch2_' + maskfile)
        segfile_ch1 = segfile_ch1 if segfile_ch1 else segfile
        segfile_ch2 = segfile_ch2 if segfile_ch2 else segfile

        if segfile_ch1:
            self.c1.maskfile = segfile_ch1[0]
            self.c2.maskfile = segfile_ch2[0]
            if self.seg_algo == 'hand_seg':
                self.prairie2019 = prairie2019
                self.c1.mask = tiff.imread(self.c1.maskfile)
                self.c2.mask = tiff.imread(self.c2.maskfile)
                if len(self.c1.mask.shape) == 4:
                    self.c1.mask = self.c1.mask[:, 0, :]
                    self.c2.mask = self.c2.mask[:, 0, :]
                if np.max(self.c1.mask) > 255:          # new prairie20019
                    self.c1.mask = self.c1.mask / 65280. * 255
                    self.c2.mask = self.c2.mask / 65280. * 255

            elif self.seg_algo == 'auto_seg':
                mask_ch1 = tiff.imread(self.c1.maskfile)
                mask_ch2 = tiff.imread(self.c2.maskfile)

                # changes made on Jan26, 2020. To allow 95% grey work instead of pure black which contains drawing lines.
                if len(mask_ch1.shape) == 4:
                    mask_ch1 = mask_ch1[:, 0, :]
                    mask_ch2 = mask_ch2[:, 0, :]

                # go with 95% grey color
                if len(np.unique(mask_ch1)) > 2:
                    if np.max(mask_ch1) > 255:          # new prairie20019
                        self.c1.mask = ((mask_ch1 / 65280. * 255) == 17)
                        self.c2.mask = ((mask_ch2 / 65280. * 255) == 17)
                    else:                               # old prairie
                        self.c1.mask = (mask_ch1 == 13)
                        self.c2.mask = (mask_ch2 == 13)
                # go with 100% black
                else:
                    self.c1.mask = (mask_ch1 == 0)
                    self.c2.mask = (mask_ch2 == 0)

                boundaryfile = glob.glob(self.folder + os.path.sep + '*boundary.tif')
                self.boundaryfile = boundaryfile[0]

                # tiff not working well (65280 instead of 255, probably changed after update), using PIL.Image.open instead. changed made on Jul 13, 2019
                self.prairie2019 = prairie2019
                if not prairie2019:
                    self.boundarymask = tiff.imread(self.boundaryfile) == 255
                else:
                    self.boundarymask = np.array(Image.open(self.boundaryfile)) == 255

        self.open()

    def open(self):
        tif = GCaMP.open(self, self.c1.file)
        if len(tif.shape) == 4:
            self.c1.tif = tif
            self.c2.tif = GCaMP.open(self, self.c2.file) if self.c2.file else None
        elif len(tif.shape) == 5:
            self.c1.tif = tif[:, :, 0]
            self.c2.tif = tif[:, :, 1]

        if not self.c1.tif.shape[1] == self.c1.mask.shape[0]:
            mask_ch1 = np.zeros(list(self.c1.tif.shape)[-3:])
            if self.seq_mask == 0:
                # print self.c1.mask.shape
                # print mask_ch1.shape
                mask_ch1[:self.c1.mask.shape[0]] = self.c1.mask
            else:
                mask_ch1[-self.c1.mask.shape[0]:] = self.c1.mask
            self.c1.mask = mask_ch1

        if not self.c2.tif is None:
            if not self.c2.tif.shape[1] == self.c2.mask.shape[0]:
                mask_ch2 = np.zeros(list(self.c2.tif.shape)[-3:])
                if self.seq_mask == 0:
                    mask_ch2[:self.c2.mask.shape[0]] = self.c2.mask
                else:
                    mask_ch2[-self.c2.mask.shape[0]:] = self.c2.mask
                self.c2.mask = mask_ch2

        self.process_channel(self.c1)
        self.process_channel(self.c2, tomato_background=self.tomato_background)

    def process_channel(self, channel, tomato_background=False):
        if channel.celltype is None: return
        channel.ppg = 1.
        # Parse images into glomeruli representation and normalize
        if self.seg_algo == 'hand_seg':
            channel.a = self._get_subgloms(channel.tif, channel.mask)
        else:
            channel.rois, channel.theta_list, channel.a = self.get_wedges(channel.tif, channel.mask, self.wedge_num)

        if tomato_background:
            channel.a = channel.a / self.c1.a

        channel.mean_p = self.get_pixel_mean(channel.tif, channel.mask)
        channel.mean_pn = self.norm_over_time(channel.mean_p, mode='dF/F0')
        channel.mean_pl = self.norm_over_time(channel.mean_p, mode='linear')

        # print channel.a.shape, channel.a[0]
        channel.an = self.norm_over_time(channel.a, mode='dF/F0')
        # print channel.an.shape, channel.an[0]
        channel.an = channel.an[~np.isnan(channel.an)]
        # print channel.an.shape, channel.an[0]
        channel.an = channel.an.reshape(self.len, channel.an.size/self.len)
        channel.al = self.norm_over_time(channel.a, mode='linear')
        channel.al = channel.al[~np.isnan(channel.al)]
        channel.al = channel.al.reshape(self.len, channel.al.size/self.len)

        channel.az = self.norm_over_time(channel.a, mode='zscore')
        channel.az = channel.az[~np.isnan(channel.az)]
        channel.az = channel.az.reshape(self.len, channel.az.size / self.len)

        channel.phase = self.get_centroidphase(channel.al) * 180.0 / np.pi
        channel.phase_dff0 = self.get_centroidphase(channel.an) * 180.0 / np.pi
        channel.phase_raw = self.get_centroidphase(channel.a) * 180.0 / np.pi
        channel.phasef = fc.circ_moving_average(channel.phase)
        channel.mean_an = channel.an.mean(axis=-1)
        channel.mean_al = channel.al.mean(axis=-1)
        # channel.mean_az = channel.az.mean(axis=-1)
        channel.mean_az = np.nanmean(channel.az, axis=-1)

        channel.xedges = np.arange(channel.al.shape[-1]+1)
        channel.xedges_interp, channel.an_cubspl = self.interpolate_wedges(channel.an, channel.xedges)
        channel.an_nophase = self.cancel_phase(channel.an_cubspl, channel.phase)
        _, channel.al_cubspl = self.interpolate_wedges(channel.al, channel.xedges)
        channel.al_nophase = self.cancel_phase(channel.al_cubspl, channel.phase)

    def _get_subgloms(self, signal, segmask):
        if not self.prairie2019:
            centroids = [13, 25, 38, 51, 64, 76, 89, 102, 115, 128, 140, 154, 166, 179, 192, 205, 217, 230]
        else:
            # new prairie version, 2019
            centroids = [17, 37, 54, 70, 85, 99, 112, 125, 137, 149, 161, 172, 183, 194, 204, 215, 225, 235]
        gloms = np.zeros((self.len, len(centroids)))
        gloms.fill(np.nan)
        for glomid, val in enumerate(centroids):  # glomid, mask in enumerate(glomask):
            mask = (segmask == val)
            if mask.sum():
                glom = signal * mask
                gloms[:, glomid] = np.nansum(glom.reshape(self.len, signal[0].size), axis=1) / mask.sum()
        for i in range(len(gloms)):
            for j in range(len(gloms[0])):
                if gloms[i][j] == 0:
                    gloms[i][j] = 1e-6

        idx = (gloms != 0)
        idn = np.zeros(idx.sum())
        idn = (idn == 0)
        idn = idn.reshape(self.len, idn.size/self.len)
        res = gloms[idn]
        res = res.reshape(self.len, idn.size/self.len)

        return res

    def get_pixel_mean(self, tif, mask):
        # calculate the mean value across all the valid pixels in the tif
        tifm = copy.copy(tif).astype(float)
        tifm[:, mask == False] = np.nan
        return np.nanmean(tifm, axis=tuple(range(1, tifm.ndim)))

    def get_centroidphase(self, data):
        """
        project eip intensity to eb, then average to get centroid position
        """
        phase = fc.centroid_weightedring(data)
        return phase

    def get_wedges(self, tif, mask, wedge_num=16):
        boundarymask = self.boundarymask.T      # invert x and y, so that it's [x, y]
        boundarymask = boundarymask[:, ::-1]     # invert y axis, so the data in the top of the original mask file is larger
        def lines_intersection((a1, b1), (a2, b2)):
            return (b2-b1)/(a1-a2), (a1*b2-a2*b1)/(a1-a2)

        def wrap_rad_simple(angle):
            if angle < 0:
                return angle + np.pi
            else:
                return angle

        def larger_than(points, a, b):
            return points[1] >= (a*points[0] + b)

        def smaller_than(points, a, b):
            return points[1] < (a*points[0] + b)

        # find the indexes for the connected parts: ite_list
        mat_con, num_con = sm.label(boundarymask, neighbors=8, return_num=True)
        assert (num_con == 2), 'Boundary file does not have two boundary lines!'
        total_points_list = [(mat_con==i).sum() for i in range(num_con)]
        background_num = total_points_list.index(max(total_points_list))
        ite_list = [0, 1, 2]
        del ite_list[background_num]

        # find xs, ys for the left and right boundaries
        if np.where(mat_con == ite_list[0])[0].mean() < np.where(mat_con == ite_list[1])[0].mean():
            left_x, left_y = np.where(mat_con == ite_list[0])
            right_x, right_y = np.where(mat_con == ite_list[1])
        else:
            left_x, left_y = np.where(mat_con == ite_list[1])
            right_x, right_y = np.where(mat_con == ite_list[0])
        a1, b1, r, p, e = stats.linregress(left_x, left_y)
        a2, b2, r, p, e = stats.linregress(right_x, right_y)

        # calculate the paras of two boundaries and their intersection
        x, y = lines_intersection((a1, b1), (a2, b2))
        self.center = np.array([x, y]) + 0.5
        self.a1 = a1
        self.a2 = a2

        # average along z axis the signal values in the masks
        tifm = copy.copy(tif).astype(float)
        tifm[:, mask==False] = np.nan
        tifmz = np.nanmean(tifm, axis=1)

        # calculate the iteration slopes for each ROI
        dims = tif.shape[-2:]
        points = np.array([(i, j) for j in np.arange(dims[0]) for i in np.arange(dims[1])]).T + .5
        rois16 = np.zeros((wedge_num, dims[0], dims[1])).astype(bool)
        theta_left = wrap_rad_simple(np.arctan(self.a1))
        theta_right = wrap_rad_simple(np.arctan(self.a2))
        theta_list = np.arange(theta_left, theta_right-0.001, (theta_right-theta_left)/1.0/wedge_num)
        a_list = np.tan(theta_list)
        b_list = self.center[1] - a_list*self.center[0]

        # calculate each ROI
        for i in range(wedge_num):
            if a_list[i] < 0:
                if a_list[i+1] <= 0:
                    roi = larger_than(points, a_list[i], b_list[i]) & smaller_than(points, a_list[i+1], b_list[i+1])
                else:
                    roi = larger_than(points, a_list[i], b_list[i]) & larger_than(points, a_list[i+1], b_list[i+1])
            else:
                roi = smaller_than(points, a_list[i], b_list[i]) & larger_than(points, a_list[i+1], b_list[i+1])
            rois16[i] = roi.reshape(dims[0], dims[1])[::-1, :]

        # extract imaging values in each ROI
        wedges = np.zeros((self.len, wedge_num))
        for i, roi in enumerate(rois16):
            wedge = tifmz * roi[None, :, :]
            total_pixel_num = np.sum(~np.isnan(wedge[0]))
            count_zero = 0
            for row in wedge[0]:
                count_zero += list(row).count(0)
            # wedges[:, i] = np.nansum(wedge.reshape((self.len, roi.size)), axis=1) / roi.sum()
            wedges[:, i] = np.nansum(wedge.reshape((self.len, roi.size)), axis=1) / (total_pixel_num - count_zero)

        return rois16, theta_list, wedges

    def interpolate_wedges(self, fb, xedges, kind='cubic', dinterp=.1):
        period_inds = fb.shape[1]
        tlen, glomlen = fb.shape[:2]
        wrap = np.zeros((fb.shape[0], fb.shape[1]+2))
        wrap[:, 1:period_inds+1] = fb[:, :period_inds]
        wrap[:, [0, period_inds+1]] = fb[:, [period_inds-1, 0]]
        x = np.arange(0, period_inds+2, 1)
        f = interp1d(x, wrap, kind, axis=-1)
        x_interp = np.arange(.5, period_inds+.5, dinterp)
        row_interp = f(x_interp)

        x_interp -= .5
        return x_interp, row_interp

    def cancel_phase(self, gc, phase, ppg=None, celltype=None, offset=180):
        period_inds = gc.shape[1]
        offset = int(offset * period_inds / 360)
        gc_nophase = np.zeros_like(gc)
        for i in xrange(len(gc)):
            shift = int(np.round((-phase[i] + 180) * period_inds / 360)) + offset
            row = np.roll(gc[i], shift)
            # row = np.zeros(gc.shape[1])
            # row = np.array([gc[i][(j - shift) % period_inds] for j in range(period_inds)])
            gc_nophase[i] = row
        return gc_nophase


class EllipsoidBody(GCaMP):
    def __init__(self, folder, celltype, celltype2=None, seq_mask=0, prairie2019=False, firstwedge_cheng=True, **kwargs):
        self.firstwedge_cheng = firstwedge_cheng
        GCaMP.__init__(self, folder, celltype, celltype2)

        segfile_c1 = glob.glob(self.folder + os.path.sep + 'C1*_seg.tif')
        segfile_c2 = glob.glob(self.folder + os.path.sep + 'C2*_seg.tif')
        if segfile_c1 and segfile_c2:
            self.c1.maskfile = segfile_c1[0]
            self.c2.maskfile = segfile_c2[0]
            self.c1.mask = tiff.imread(self.c1.maskfile) == 0
            self.c2.mask = tiff.imread(self.c2.maskfile) == 0
        else:
            segfile = (glob.glob(self.folder + os.path.sep + '*_seg.tif') +
                    glob.glob(self.folder + os.path.sep + '*ebmask.tif'))
            # old, before Jan 26, 2020
            # if segfile:
            #     self.c1.maskfile = self.c2.maskfile = segfile[0]
            #     mask = tiff.imread(self.c1.maskfile)
            #     if len(mask.shape) == 5:
            #         mask = mask[:, :, 1]
            #     if len(mask.shape) == 4:
            #         self.c1.mask = mask[:, 0,] == 0
            #         self.c2.mask = mask[:, 1] == 0
            #     elif len(mask.shape) == 3:
            #         self.c1.mask = mask == 0

            # changes made on Jan26, 2020. To allow 95% grey work instead of pure black which contains drawing lines.
            if segfile:
                self.c1.maskfile = self.c2.maskfile = segfile[0]
                mask = tiff.imread(self.c1.maskfile)
                if len(mask.shape) == 4:
                    mask = mask[:, 0, :]

                # go with 95% grey color
                if len(np.unique(mask)) > 2:
                    if np.max(mask) > 255:          # new prairie20019
                        self.c1.mask = ((mask / 65280. * 255) == 17)
                    else:                               # old prairie
                        self.c1.mask = (mask == 13)
                # go with 100% black
                else:
                    self.c1.mask = (mask == 0)
                self.c2.mask = self.c1.mask

        if len(self.c1.mask.shape) == 2:
            self.c1.mask = self.c1.mask[None, :, :]

        if hasattr(self.c2, 'mask'):
            if len(self.c2.mask.shape) == 2:
                self.c2.mask = self.c1.mask[None, :, :]
        else:
            self.c2.mask = None
        self.seq_mask = seq_mask

        centerfile = glob.glob(self.folder + os.path.sep + '*center.tif')
        if centerfile:
            self.centerfile = centerfile[0]
            if not prairie2019:
                self.centermask = tiff.imread(self.centerfile) == 255 # 0?
            else:
                self.centermask = np.array(Image.open(self.centerfile)) == 255

        self.nwedges_dict = {'EIP': 16, 'PEN': 16}

        self.open()

    def open(self):
        tif = GCaMP.open(self, self.c1.file)
        if len(tif.shape) == 4:
            self.c1.tif = tif
            self.c2.tif = GCaMP.open(self, self.c2.file) if self.c2.file else None
        elif len(tif.shape) == 5:
            self.c1.tif = tif[:, :, 0]
            self.c2.tif = tif[:, :, 1]
        #
        # print self.c1.tif.shape
        # print self.c1.mask.shape
        if not self.c1.tif.shape[1] == self.c1.mask.shape[0]:
            mask = np.zeros(list(self.c1.tif.shape)[-3:])
            if self.seq_mask == 0:
                mask[:self.c1.mask.shape[0]] = self.c1.mask
            else:
                mask[-self.c1.mask.shape[0]:] = self.c1.mask
            self.c1.mask = self.c2.mask = mask

        self.process_channel(self.c1)
        self.process_channel(self.c2)

    def process_channel(self, channel):
        if channel.celltype is None: return
        channel.xedges, channel.wedge_rois, channel.theta, channel.a = \
            self.get_wedges(channel.tif, channel.mask, channel.celltype)

        channel.mean_p = self.get_pixel_mean(channel.tif, channel.mask)
        channel.mean_pn = self.norm_over_time(channel.mean_p, mode='dF/F0')
        channel.mean_pl = self.norm_over_time(channel.mean_p, mode='linear')

        channel.an = self.norm_over_time(channel.a, mode='dF/F0')
        channel.al = self.norm_over_time(channel.a, mode='linear')
        channel.mean_an = np.nanmean(channel.an, axis=-1)
        channel.mean_al = np.nanmean(channel.al, axis=-1)
        # channel.phase = self.get_phase(channel.al, channel.theta)
        channel.phase = fc.wrap(self.get_phase(channel.al, channel.theta) - 180)        # edited 2019 Oct. 07
        channel.phase_dff0 = fc.wrap(self.get_phase(channel.an, channel.theta) - 180)  # edited 2019 Oct. 07
        channel.phase_raw = fc.wrap(self.get_phase(channel.a, channel.theta) - 180)  # edited 2019 Oct. 07
        channel.phasef = fc.circ_moving_average(channel.phase, 3)
        channel.period = self.nwedges_dict[channel.celltype]
        channel.xedges_interp, channel.an_cubspl = self.interpolate_wedges(channel.an, channel.xedges)
        channel.an_nophase = self.cancel_phase(channel.an_cubspl, channel.phase)

        channel.az = self.norm_over_time(channel.a, mode='zscore')
        _, channel.az_cubspl = self.interpolate_wedges(channel.az, channel.xedges)
        channel.az_nophase = self.cancel_phase(channel.az_cubspl, channel.phase)

    def get_phase(self, r, theta):
        phase = np.zeros(len(r))
        for i in xrange(len(r)):
            xi, yi = fc.polar2cart(r[i], theta)
            ri, thetai = fc.cart2polar(xi.mean(), yi.mean())
            phase[i] = thetai
        phase *= 180 / np.pi
        return phase

    def get_wedge_rois(self, center, nwedges, offset=0):
        thetas = np.arange(0, np.pi, np.pi/8)
        ms = np.tan(thetas)
        bs = center[1] - ms*center[0]
        dims = self.info['dimensions_pixels']

        # use the center of each pixel as a threshold for segmenting into each wedge
        points = np.array([(i, j) for j in np.arange(dims[0]) for i in np.arange(dims[1])]).T + .5

        def greater_than(points, m, b):
            return points[1] >= (m*points[0] + b)

        def less_than(points, m, b):
            return points[1] < (m*points[0] + b)

        rois16 = np.zeros((16, points.shape[1])).astype(bool)
        for i in range(16):
            j = i%8
            if i < 4:
                roi = greater_than(points, ms[j], bs[j]) & less_than(points, ms[j+1], bs[j+1])
            elif i == 4:
                roi = ((points[0]) < center[0]) & greater_than(points, ms[j+1], bs[j+1])
            elif i > 4 and i < 7:
                roi = less_than(points, ms[j], bs[j]) & greater_than(points, ms[j+1], bs[j+1])
            elif i == 7:
                roi = less_than(points, ms[j], bs[j]) & ((points[1]) >= center[1])
            elif i > 7 and i < 12:
                roi = less_than(points, ms[j], bs[j]) & greater_than(points, ms[j+1], bs[j+1])
            elif i == 12:
                roi = ( points[0] >= center[0] ) & less_than(points, ms[j+1], bs[j+1])
            elif i > 12:
                roi = greater_than(points, ms[j], bs[j]) & less_than(points, ms[(i+1)%8], bs[(i+1)%8])
            rois16[i] = roi

        rois16 = rois16.reshape((16, dims[0], dims[1]))
        rois16 = np.flipud(rois16) # for some reason need this to match the tiffile orientation
        if nwedges == 8:
            rois8 = np.zeros((8, dims[0], dims[1]))
            for i in range(8):
                rois8[i] = rois16[2*i] | rois16[2*i-1]
            rois = np.roll(rois8[::-1], -1, axis=0)
            xedges = np.arange(0, 17, 2)
        elif nwedges == 16:
            rois = np.roll(rois16[::-1], -3, axis=0)
            xedges = np.arange(0, 17)

        return xedges, rois

    def get_wedges(self, tif, mask, celltype):
        # Compute center of ellipsoid body from centermask
        x, y = np.where(self.centermask.T)
        self.center = (x[0], y[0])
        if self.center == (0, 0):
            self.centermask = self.centermask == False
            x, y = np.where(self.centermask.T)
            self.center = (x[0], y[0])
        self.center = np.array(self.center) + .5

        # Apply mask to each z-plane of tif, and take mean across z
        tifm = copy.copy(tif).astype(float)
        tifm[:, mask==False] = np.nan
        tifmz = np.nanmean(tifm, axis=1)

        # Compute ROIs using center coordinate
        xedges, rois = self.get_wedge_rois(self.center, self.nwedges_dict[celltype])

        # Compute mean value in each wedge ROI for each time frame
        wedges = np.zeros((self.len, self.nwedges_dict[celltype]))
        for i, roi in enumerate(rois):
            wedge = tifmz * roi[None, :, :]
            wedges[:, i] = np.nansum(wedge.reshape((self.len, roi.size)), axis=1) / roi.sum()

        # Thetas are the central angle of each wedge, and is used to compute the phase,
        # with zero pointing down (-pi/2).
        theta = np.arange(0, 2*np.pi, 2*np.pi/self.nwedges_dict[celltype])
        if self.nwedges_dict[celltype] == 16:
            theta -= 2*np.pi / 32

        if self.firstwedge_cheng:        # first wedge is the bottom left wedge, for FB, EB alignment
            return xedges, np.roll(rois, -1, axis=0), np.roll(theta, -1), np.roll(wedges, -1, axis=1)
        else:                       # first wedge is the bottom right wedge, Jonathan used, for pb eb alignment
            return xedges, rois, theta, wedges

    def get_pixel_mean(self, tif, mask):
        # calculate the mean value across all the valid pixels in the tif
        tifm = copy.copy(tif).astype(float)
        tifm[:, mask == False] = np.nan
        return np.nanmean(tifm, axis=tuple(range(1, tifm.ndim)))

    def get_polar_warping_rois(self, channel, mask, center, celltype):
        def reproject_image_into_polar(self, data, origin, nbins):
            """Reprojects a 3D numpy array ("data") into a polar coordinate system.
            "origin" is a tuple of (x0, y0) and defaults to the center of the image."""
            def index_coords(data, origin=None):
                """Creates x & y coords for the indicies in a numpy array "data".
                "origin" defaults to the center of the image. Specify origin=(0,0)
                to set the origin to the lower left corner of the image."""
                ny, nx = data.shape[:2]
                if origin is None:
                    origin_x, origin_y = nx // 2, ny // 2
                else:
                    origin_x, origin_y = origin
                x, y = np.meshgrid(np.arange(nx), np.arange(ny))
                x -= origin_x
                y -= origin_y
                return x, y

            ny, nx = data.shape[:2]
            if origin is None:
                origin = (nx // 2, ny // 2)

            # Determine that the min and max r and theta coords will be...
            x, y = index_coords(data, origin=origin)
            r, theta = fc.cart2polar(x, y)

            # Make a regular (in polar space) grid based on the min and max r & theta
            r_i = np.linspace(r.min(), r.max(), nx)
            theta_i = np.linspace(-np.pi, np.pi, nbins)
            theta_grid, r_grid = np.meshgrid(theta_i, r_i)

            # Project the r and theta grid back into pixel coordinates
            xi, yi = fc.polar2cart(r_grid, theta_grid)
            xi += origin[0]  # We need to shift the origin back to
            yi += origin[1]  # back to the lower-left corner...
            xi, yi = xi.flatten(), yi.flatten()
            coords = np.vstack((xi, yi))  # (map_coordinates requires a 2xn array)

            # Reproject data into polar coordinates
            output = ndimage.map_coordinates(data.T, coords, order=1).reshape((nx, nbins))

            return output, r_i, theta_i

        channelm = copy.copy(channel).astype(float)
        channelm[:, ~mask] = np.nan
        channelmz = np.nanmean(channelm, axis=1)
        # channelmzn = self.norm_over_time(channelmz, mode='dF/F0')
        nbins = 16 if celltype == 'EIP' else 16
        polar_mask, r, theta = self.reproject_image_into_polar(mask.max(axis=0), center, nbins)
        channelp = np.zeros((len(channelmz), theta.size))
        for i, cart in enumerate(channelmz):
            polar_grid, r, theta = reproject_image_into_polar(cart, center, nbins)
            polar_grid[polar_mask < .99] = np.nan
            channelp[i] = np.nanmean(polar_grid, axis=0)
        return theta, channelp

    def interpolate_wedges(self, eb, xedges, kind='cubic', dinterp=.1):
        period_inds = eb.shape[1]
        tlen, glomlen = eb.shape[:2]
        row_interp = np.zeros((tlen, int(glomlen/dinterp)))
        wrap = np.zeros((eb.shape[0], eb.shape[1]+2))
        wrap[:, 1:period_inds+1] = eb[:, :period_inds]
        wrap[:, [0, period_inds+1]] = eb[:, [period_inds-1, 0]]
        x = np.arange(0, period_inds+2, 1)
        f = interp1d(x, wrap, kind, axis=-1)
        x_interp = np.arange(.5, period_inds+.5, dinterp)
        row_interp = f(x_interp)

        x_interp -= .5
        return x_interp, row_interp

    def cancel_phase(self, gc, phase, ppg=None, celltype=None, offset=180):
        if np.isnan(phase[0]):
            return np.nan
        period_inds = gc.shape[1]
        offset = int(offset * period_inds / 360)
        gc_nophase = np.zeros_like(gc)
        for i in xrange(len(gc)):
            shift = int(np.round((-phase[i] + 180) * period_inds / 360)) + offset
            row = np.roll(gc[i], shift)
            # row = np.zeros(gc.shape[1])
            # row = np.array([gc[i][(j - shift) % period_inds] for j in range(period_inds)])
            gc_nophase[i] = row
        return gc_nophase


class GCaMP_ABF():
    """
    Class to handle an imaging experiment, both TIF and ABF together.
    """

    def __init__(self, folder, celltype, celltype2=None, bhtype=bh.Walk, imtype=Bridge,
                 nstims=None, width=16, mode_phase_algo='pb', maskfile_seqlist=[], **kwargs):
        self.folder = folder

        # ABF file
        self.abffile = glob.glob(folder + os.path.sep + '*.abf')[0]
        print self.abffile
        self.abf = bhtype(self.abffile, nstims, **kwargs)
        self.basename = self.abf.basename
        self.name = self.basename.split(os.path.sep)[-1]
        # print self.abf.recid

        # Imaging Files
        self._get_frame_ts()
        zlen = self.abf.zlen
        imlabel_dict = {Bridge: 'pb', EllipsoidBody: 'eb', FanshapeBody: 'fb',}

        def get_zslice(imtype, mask_zlen):
            if imtype == Bridge:
                return slice(0, mask_zlen)
            elif imtype == EllipsoidBody:
                return slice(zlen - mask_zlen, zlen)
            elif imtype == SubGlomeruli:
                return slice(0, mask_zlen)
        def get_zslice_cheng(imtype, mask_zlen, num_imtype):
            if num_imtype == 0:
                return slice(0, mask_zlen)
            else:
                return slice(zlen - mask_zlen, zlen)

        if not type(imtype) is tuple: imtype = [imtype]
        self.ims = []
        self.imlabels = []

        for num_imtype, i_imtype in enumerate(imtype):
            label = imlabel_dict[i_imtype]
            if len(maskfile_seqlist):
                maskfile_nameseq = maskfile_seqlist[num_imtype]

            if i_imtype == Bridge:
                setattr(self, label, i_imtype(folder, celltype, celltype2, width, mode_phase_algo, seq_mask=num_imtype, **kwargs))
            elif i_imtype == FanshapeBody:
                if len(maskfile_seqlist):
                    setattr(self, label, i_imtype(folder, celltype, celltype2, seq_mask=num_imtype, maskfile_seq=maskfile_nameseq, **kwargs))
                else:
                    setattr(self, label, i_imtype(folder, celltype, celltype2, seq_mask=num_imtype, **kwargs))
            else:
                setattr(self, label, i_imtype(folder, celltype, celltype2=celltype2, seq_mask=num_imtype, **kwargs))

            tif = getattr(self, label)
            mask_len = (tif.c1.mask.sum(axis=-1).sum(axis=-1) > 0).sum()
            zslice = get_zslice_cheng(i_imtype, mask_len, num_imtype)
            self._get_tif_ts(tif, zslice)

            if label == 'pb' or label == 'eb' or label == 'fb':
                tif.c1.dphasef = fc.circgrad(tif.c1.phasef)*tif.sampling_rate
                tif.c1.dphase = fc.circgrad(tif.c1.phase)*tif.sampling_rate
                if tif.c2.celltype:
                    tif.c2.dphase = fc.circgrad(tif.c2.phase)*tif.sampling_rate
                    tif.c2.dphasef = fc.circgrad(tif.c2.phasef)*tif.sampling_rate
            if hasattr(tif, 'xstim'):
                tif.xstim = self.subsample('xstim', lag_ms=0, tiflabel=tif, metric=fc.circmean)
                tif.dxstim = fc.circgrad(tif.xstim) * tif.sampling_rate
            self.ims.append(tif)
            self.imlabels.append(label)

    def get_pockels_on(self, trigs, pockel_thresh=.05, plot=False, tlim=None):
        pockels = self.abf.pockels
        t = self.abf.t_orig

        trig_period = np.diff(trigs[:100]).mean()
        pockel_ind = np.arange(trig_period * .1, trig_period * .9, 1).round().astype(int)
        pockel_inds = np.tile(pockel_ind, (len(trigs), 1))
        pockel_inds += trigs.reshape((len(trigs), 1))
        pishape = pockel_inds.shape
        pockels_on = pockels[pockel_inds.flatten()].reshape(pishape).mean(axis=1) > pockel_thresh

        if plot:
            plt.figure(1, (15, 5))
            plt.plot(t, self.abf.frametrig, c='black', alpha=.8)
            plt.plot(t, self.abf.pockels, c=ph.blue, alpha=.4)
            plt.scatter(t[trigs], np.ones(len(trigs)), c=pockels_on, lw=0, s=40, zorder=10, cmap=plt.cm.bwr)
            if tlim:
                plt.xlim(tlim)
            else:
                plt.xlim(-.1, 1.2)

        return pockels_on

    def _get_zlen(self, pockels_on):
        if pockels_on[:-10].all():
            return 1
        else:
            start, stop = fc.get_inds_start_stop(pockels_on)
            zlens = np.diff(np.vstack([start, stop]), axis=0)[0]
            # assert (zlens[0] == zlens).all()
            return zlens[0]

    def _get_frame_ts(self):
        # Extract frame starts from abffile.
        # print np.where(np.diff(self.abf.frametrig.T>3).astype(int)>0)[0]
        # print np.sum(np.diff(self.abf.frametrig > 3)>0)
        trigs = fc.rising_trig(self.abf.frametrig, 3)

        # Set t=0 to when 2p acquisition begins.
        t_acq_start = self.abf.t_orig[trigs[0]]
        self.abf.t -= t_acq_start
        self.abf.tc = self.abf.t
        self.abf.t_orig -= t_acq_start

        # Get frame triggers times
        self.abf.ft = self.abf.t_orig[trigs]  # Frame triggers
        self.abf.dft = np.diff(self.abf.ft[:100]).mean()  # Frame trigger period

        self.abf.pockels_on = self.get_pockels_on(trigs)  # Need to take into account lost frame triggers due to flip-back of piezo z-motor.
        self.abf.tz = self.abf.ft[self.abf.pockels_on]  # t is not shifted by half a frame because they act as edges when plotted using pcolormesh.
        self.abf.zlen = self._get_zlen(self.abf.pockels_on)

    def _get_tif_ts(self, tif, zslice=None):
        if zslice is None:
            zslice = slice(0, self.abf.zlen)

        tif.zlen = zslice.stop - zslice.start

        tif.int = self.abf.dft * tif.zlen  # Integration period for one z-stack (includes z=1)
        total_zlen = self.abf.zlen
        tif.t = self.abf.tz[zslice.start::total_zlen][:tif.len]
        # Print warning if TIF images corresponds to number of frame triggers.
        if len(tif.t) != tif.len:
            print 'Number of frames and triggers do not match: \nNumber of frames: %i \nNumber of triggers: %i.' % (tif.len, len(tif.t))
        tif.tc = tif.t + tif.int / 2.

        tif.ind = np.array([np.where(self.abf.t >= t)[0][0] for t in tif.t])
        tif.indint = int(np.round(tif.int / self.abf.subsampling_period))

        # Set sampling rate and period
        tif.sampling_period = np.diff(tif.t).mean()
        tif.subsampling_period = tif.sampling_period
        tif.sampling_rate = 1. / tif.sampling_period
        tif.info['sampling_period'] = tif.sampling_period
        tif.info['sampling_rate'] = tif.sampling_rate
        tif.info['recording_len'] = tif.t[-1] + tif.sampling_period

    def _get_stimon_tifidx(self):
        xstim = copy.copy(self.abf.xstim)
        degperpix = 360 / (24 * 8)
        # 6-pixel wide bar, angle of bar is defined at its center, therefore 3 pixels past the 'edge'
        # the bar is no longer visible, use 2.5 pixels as threshold to compensate for noise
        xstim[xstim < (135 + 2.5 * degperpix)] += 360
        stimon_abfidx = xstim > (225 - 2.5 * degperpix)

        stimon_tifidx = np.zeros(len(self.tif.t)).astype(bool)
        for i, idx in enumerate(self.tif.ind):
            inds = np.zeros(len(xstim)).astype(bool)
            inds[range(idx, idx + self.tif.indint)] = True
            stimon = (inds * stimon_abfidx).sum() > 0
            stimon_tifidx[i] = stimon

        return stimon_tifidx

    def rebin_tif(self, tbin=0.05):
        gcbinneds = []
        nrun = 0
        window_len = self.abf.window_lens[0]
        shoulder_len = self.abf.shoulder_lens[0]
        t0 = -shoulder_len
        tf = window_len + shoulder_len
        ti = np.arange(t0, tf, tbin)
        for stimid in range(self.abf.nstims):
            t = self.tif.tparsed[stimid][nrun]
            gc = self.tif.parsed['normed'][stimid][nrun]
            rows = [[] for _ in ti]
            for itrial, gctrial in enumerate(gc):
                for iline, tline in enumerate(t[itrial]):
                    bi = bisect(ti, tline) - 1
                    rows[bi].append(gctrial[:, iline])

            rows = [np.array(row) for row in rows]
            binned = np.array([row.mean(axis=0) for row in rows])
            gcbinneds.append(binned)

        gcbinneds = np.array(gcbinneds)

        return ti, gcbinneds

    def check_timing(self, tlim=[-1, 2]):
        plt.figure(1, (15, 5))
        plt.plot(self.abf.t_orig, self.abf.frametrig)
        plt.plot(self.abf.t_orig, self.abf.pockels)
        plt.scatter(self.tif.t, np.zeros_like(self.tif.t), zorder=3)
        plt.xlim(tlim)

    def subsample(self, signal, lag_ms=0, tiflabel=None, **kwargs):
        """
        Subsample abf channel so that it matches the 2P scanning sampling frequency and integration time.
        Signal is averaged over the integration time of scanning, which is less than the full scanning period.
        """

        if type(signal) is str:
            if 'abf' in signal:
                signal = signal.split('abf.')[1]
            signal = getattr(self.abf, signal)

        if tiflabel is None:
            tif = self.ims[0]
        elif type(tiflabel) is str:
            tif = getattr(self, tiflabel)
        else:
            tif = tiflabel

        indlag = int(np.round(lag_ms / 1000. / self.abf.subsampling_period))
        inds = tif.ind - indlag
        signal_subsampled = fc.sparse_sample(signal, inds, tif.indint, **kwargs)
        # t_subsampled = self.abf.t[inds] + self.tif.int / 2.
        return signal_subsampled

    def bin_gcamp(self, imlabel, binlabel, bins, lag_ms=0, metric=np.nanmean):
        """
        Bins gc time frames as a function of the binsignal during each time frame.
        Takes mean of time frames within same binsignal bin.
        Lag is positive if the bridge signal comes after the binsignal.
        Returns a 2D matrix: len(bins)-1 x bridge.shape[1]
        """
        # Collect signals
        if type(imlabel) is str:
            _, a = self.get_signal(imlabel)
        else:
            a = imlabel

        if type(binlabel) is str:
            _, x = self.get_signal(binlabel)
        else:
            x = binlabel

        x = self.subsample(binlabel, lag_ms=lag_ms, metric=metric)  # subsample binsignal to match tif sampling
        # elif len(x) == len(a):
        #     lag_int = int(np.round(lag_ms/1000. / im.int))
        #     if lag_int > 0:
        #         a = a[:-lag_int]
        #         x = x[lag_int:]
        #     elif lag_int < 0:
        #         lag_int *= -1
        #         a = a[lag_int:]
        #         x = x[:-lag_int]

        ab = np.zeros((len(bins) - 1, a.shape[1]))
        ab[:] = np.nan
        digitized = np.digitize(x, bins)
        for i in range(1, len(bins)):
            idx = digitized == i
            if idx.sum() > 0:
                ab[i-1] = np.nanmean(a[idx], axis=0)
        return ab

    def get_offset(self, toffset=None, phase_label='phase'):
        im, c, phase = self.get_objects(phase_label)
        if type(toffset) is tuple or type(toffset) is list:
            idx = (im.tc>=toffset[0]) & (im.tc<toffset[1])
            offset = fc.circmean(phase[idx] - im.xstim[idx])
            return offset
        else:
            if toffset is None:
                toffset = im.tc[np.where((im.tc >= 2) & (im.xstim > -135) & (im.xstim < 135))[0][0]]
            idx = np.where(im.tc >= toffset)[0]
            if len(idx) > 0:
                if len(idx) > 71:
                    # offset = fc.circmean(phase[idx[0]+1000:idx[0]+1400]) - fc.circmean(im.xstim[idx[0]+1000:idx[0]+1400])
                    offset = fc.circmean(phase[idx[0]+300:idx[0]+800]) - fc.circmean(im.xstim[idx[0]+300:idx[0]+800])
                else:
                    offset = phase[idx[0]+1] - im.xstim[idx[0]]
                return offset
            else:
                return None

    def get_delay_matrix(self, signal, max_delay=5, metric=fc.circmean, **kwargs):
        if type(signal) is str: signal = getattr(self.abf, signal)
        t = self.abf.t
        sr = self.abf.subsampling_rate
        sp = self.abf.subsampling_period
        subind = self.tif.ind
        subint = self.tif.indint
        max_delay_ind = int(np.round(max_delay * sr))
        delay_matrix = np.zeros((2 * max_delay_ind + 1, len(subind)))
        for i, dind in enumerate(xrange(-max_delay_ind, max_delay_ind + 1)):
            row = bh.fc.sparse_sample(signal, subind + dind, subint, metric, **kwargs)
            delay_matrix[i] = row
        delays = np.arange(-delay, delay + sp, sp)

        self.delays, self.delay_matrix = delays, delay_matrix
        return delays, delay_matrix

    def interpolate_phase(self, phase, power, thresh):
        phase_interp = fc.interpolate(self.tif.t, phase, power>thresh)
        return phase_interp

    def get_labels(self, label):
        """
        :param label: String pointing to an object. If parent objects are not specified,
         then they will be filled in with defaults.
         Defaults: tif > abf. c1 > c2. c = None if signal label is not found in c.

         Examples: 'pb.c1.phase', or 'abf.head' are complete.
         'phase' will be completed to tif.c1.phase (tif = self.imlabels[0]).
         'head' will be completed to tif.head (to be subsampled on the fly, if necessary),
         unless label = 'abf.head', in which case it is left as is.
        :return:
        """
        labels = label.split('.')
        if labels[0] == 'abf':
            return labels
        else:
            c = self.ims[0].c1
            if len(labels) < 2:

                if labels[-1] in dir(c):
                    clabel = 'c1'
                else:
                    clabel = self.imlabels[0]
                labels = [clabel] + labels
            if len(labels) < 3 and labels[-1] in dir(c):
                labels = [self.imlabels[0]] + labels
        return labels

    def get_objects(self, label, lag_ms=0):
        """
        :param label: String pointing to a signal. If parent objects are not specified,
        they are are filled in with defaults (see get_labels).
        :param lag_ms: Time lag in ms to subsample signal if not already subsampled.
        :return:
        """
        labels = self.get_labels(label)
        if labels[0] == 'abf':
            abf = getattr(self, labels[0])
            signal = getattr(abf, labels[-1])
            return abf, None, signal
        else:
            tif = getattr(self, labels[0])
            if len(labels) < 3:
                # c = None
                c = getattr(tif, 'c1')
                signal = self.subsample(labels[-1], lag_ms, tif)
            else:
                c = getattr(tif, labels[1])
                signal = getattr(c, labels[-1])

            return tif, c, signal

    def get_signal(self, label, lag_ms=0):
        labels = self.get_labels(label)
        par, c, signal = self.get_objects(label, lag_ms)

        if len(labels) < 3:
            if 'orig' in labels[-1]:
                return par.t_orig, signal
            else:
                return par.tc, signal
        elif labels[-1][0] == 'a':
            if len(signal.shape) > 1:
                if signal.shape[1] > 18:
                    return (par.t, c.xedges_interp), signal
                else:
                    return (par.t, c.xedges), signal
            else:
                return (par.t, c.xedges), signal
        else:
            return par.tc, signal

    def get_dark_gain(self, tlim, phaselabel='pb.c1.dphase'):
        head = self.subsample('head', lag_ms=250, metric=fc.circmean)
        head = fc.circ_moving_average(head, n=3)
        im, _, _ = self.get_objects(phaselabel)
        dhead = fc.circgrad(head) * im.sampling_rate
        # head = fc.unwrap(head)
        t, dphase = self.get_signal(phaselabel)
        idx = (t>=tlim[0]) & (t<tlim[1])
        a, b, r, p, e = stats.linregress(dhead[idx], dphase[idx])
        return -a

    def get_headint(self, gain, tiflabel=None, lag_ms=0):
        dhead = fc.circgrad(self.abf.head)
        headint = np.cumsum(dhead * gain)
        return fc.wrap(headint)


# Collects recordings into a list of flies, each of which is a tuple of GCaMP_ABF recordings.

def get_recs(recnamess, celltype, parent_folder='./', **kwargs):

    def get_trial_name(recname):
        year = recname[:4]
        month = recname[4:6]
        day = recname[6:8]
        recnum = recname[8:10]
        date = year + '_' + month + '_' + day
        return date + '/' + date + '-0' + recnum + '/'

    recs = [
        [GCaMP_ABF(parent_folder + get_trial_name(recname), celltype, **kwargs)
         for recname in recnames]
        for recnames in recnamess
    ]
    return recs


colours = {'EIP': ph.blue, 'PEN': ph.orange, 'PFR': ph.purple, 'PFN': ph.orange, 'FBLN': ph.green,
           'PFLC': ph.green, 'FBCre': ph.green, 'FBNo': ph.purple, 'Speed': ph.green}
dcolours = {'EIP': ph.nblue, 'PEN': ph.dorange, 'PFR': ph.purple, 'PFN': ph.dorange, 'FBLN': ph.green,
            'PFLC': ph.green, 'FBCre': ph.green, 'FBNo': ph.purple, 'Speed': ph.green}
cmaps = {'EIP': plt.cm.Blues, 'PEN': plt.cm.Oranges, 'PFR': plt.cm.Blues, 'PFN': plt.cm.Oranges, 'FBLN': plt.cm.jet,
         'PFLC': plt.cm.jet, 'FBCre': plt.cm.Greens, 'FBNo': plt.cm.Purples, 'Speed': plt.cm.Greens}



def get_trial_ts(rec):
    trials = np.diff(rec.abf.stimid) == 0
    trials = ndimage.binary_erosion(trials, iterations=25)
    trials = ndimage.binary_dilation(trials, iterations=25).astype(int)
    trial_starts = rec.abf.t[ np.where(np.diff( trials ) > 0)[0]+1 ]
    trial_ends = rec.abf.t[ np.where(np.diff( trials ) < 0)[0]+1 ]

    if len(trial_starts):
        if trial_starts[0] > trial_ends[0]:
            trial_ends = trial_ends[1:]
        return zip(trial_starts, trial_ends)
    else:
        # print rec.name
        return None


def get_trials_ts2(rec, tlim=(-2, 5), trigger='pressure', trigthresh=20, pulsethresh=.04):
    trig_inds = bh.fc.rising_trig(getattr(rec.abf, trigger), trigger=trigthresh,
                                mindist=.01 * rec.abf.subsampling_rate)
    if len(trig_inds) == 0:
        print 'No triggers detected.'
        return
    trig_ts = rec.abf.t[trig_inds]

    ttif = rec.tif.t + rec.tif.int / 2.

    ttrials = []
    greentrials = []
    redtrials = []
    phasetrials = []
    for trig_t in trig_ts:
        twindow = list(tlim + trig_t)
        inds = (ttif >= twindow[0]) & (ttif < twindow[1])
        baseline_inds = (ttif >= twindow[0] - tlim[0] - .5) & (ttif < twindow[0] - tlim[0] - .1)
        ttrial = ttif[inds] - trig_t
        redtrial = rec.tif.rcn[inds]
        red_pulse = redtrial[(ttrial > 0) & (ttrial < .5)]
        if len(red_pulse) and red_pulse.max() < pulsethresh:
            continue
        ttrials.append(ttrial)
        redtrials.append(redtrial)
        greentrials.append(rec.tif.gcn[inds])
        phasetrials.append(rec.tif.phase[inds])

    return zip(ttrials, greentrials, redtrials, phasetrials)


def get_trial_data(groups, metric, summary_fcn, stim_dict, axis=-1, barmintime=.5, minspeed=None,
                   lag_ms=300, temp_thresh=25, zero_offset=True, ntrials=2, maxregdist=4,
                   mintrialtime=10, maxdhead=None, minintensity=None):

    temps = ['cool', 'hot']
    labels = stim_dict.values()
    max_nflies = max(map(len, groups))
    data = [{temp: {label: [] for label in labels} for temp in temps} for _ in groups]

    for igroup, group in enumerate(groups):
        for ifly, fly in enumerate(group):
            flydata = {temp: {label: [] for label in labels} for temp in temps}
            for rec in fly:
                trial_ends = get_trial_ts(rec)
                if trial_ends is None: continue
                im = rec.ims[0]
                trial_tif_inds = [(im.t > start) & (im.t < end) for start, end in trial_ends]
                trial_abf_inds = [(rec.abf.t > start) & (rec.abf.t < end) for start, end in trial_ends]

                xstim = rec.subsample('xstim', lag_ms=lag_ms, metric=fc.circmean)
                barmask = (xstim > -135) & (xstim < 135)
                if barmintime != 0 and not barmintime is None:
                    iters = int(np.ceil(barmintime * im.sampling_rate))
                    barmask = ndimage.binary_erosion(barmask, iterations=iters)

                if not minspeed is None:
                    speedmask = rec.subsample('speed', lag_ms=lag_ms, metric=np.mean) > minspeed

                if not maxregdist is None:
                    regdistmask = im.get_subregdist_mask(thresh=maxregdist)

                if not maxdhead is None:
                    dheadmask = np.abs(rec.abf.dhead) < maxdhead
                    dheadmask = rec.subsample(dheadmask, lag_ms=lag_ms, metric=np.mean)
                    dheadmask = np.floor(dheadmask).astype(bool)
                    dheadmask = ndimage.binary_erosion(barmask, iterations=1)

                if not minintensity is None:
                    intensitymask = np.nanmax(im.c1.an, axis=1) > minintensity

                for itrial, (trial_abf_ind, trial_tif_ind) in enumerate(zip(trial_abf_inds, trial_tif_inds)):
                    temp = 'hot' if rec.abf.temp[trial_abf_ind].mean() > temp_thresh else 'cool'

                    # Get head and phase correctly sampled, with correct units
                    stimid = rec.abf.stimid[trial_abf_ind].mean()
                    if (stimid - np.floor(stimid)) > 0:
                        print rec.name
                        print rec.abf.t[trial_abf_ind].min(), rec.abf.t[trial_abf_ind].max()
                        print 'Trial range is incorrect.'
                    try:
                        label = stim_dict[stimid]
                    except KeyError:
                        print rec.name
                        print stimid
                    trial_ind = trial_tif_ind

                    if ('bar' in label.lower()) and (not barmintime is None):
                        trial_ind = trial_ind & barmask

                    if not minspeed is None:
                        trial_ind = trial_ind & speedmask

                    if not maxregdist is None:
                        trial_ind = trial_ind & regdistmask

                    if not 'bar' in label.lower() and not maxdhead is None:
                        trial_ind = trial_ind & dheadmask

                    if not minintensity is None:
                        trial_ind = trial_ind & intensitymask

                    if (trial_ind.sum() * im.sampling_period) < mintrialtime:
                        continue

                    flydata[temp][label].append( metric(rec, trial_ind, label) )

            for temp in temps:
                for label in labels:
                    if len(flydata[temp][label]):
                        w = summary_fcn(flydata[temp][label], axis=axis)
                        data[igroup][temp][label].append( w )

    return data


def get_trial_data2(group, stim_dict, channel, lag_ms=300, temp_thresh=25):
    """
    :param group: List of fly objects, each of which is a tuple of recs.
    :param stim_dict: Dictionary of stimulus ID to stimulus label.
    :param channel: String label of the channel to be parsed.
    :param lag_ms: Lag of channel with respect to tif triggers.
    :param temp_thresh: Threshold for parsing into hot and cool trials.
    :return: Numpy array with dimensions:
    len(group), len(stim_dict), len(temps), ntrials, max len of a trial
    """
    temps = [0, 1]
    rec = group[0][0]
    max_trial_len_tifinds = 0
    max_ntrials = 0
    for fly in group:
        for rec in fly:
            trial_ends = get_trial_ts(rec)
            if not trial_ends: continue
            imax_trial_len_sec = max(map(lambda x: x[1]-x[0], trial_ends))
            sr = rec.tif.sampling_rate
            imax_trial_len_tifinds = imax_trial_len_sec * sr
            max_trial_len_tifinds = max(max_trial_len_tifinds, imax_trial_len_tifinds)

            ntrials = len(trial_ends) / len(stim_dict)
            max_ntrials = max(max_ntrials, ntrials)

    data = np.empty((len(group), len(stim_dict), len(temps), max_ntrials, max_trial_len_tifinds))
    data.fill(np.nan)

    if hasattr(rec.tif, channel):
        channeltype = 'tif'
    elif hasattr(rec.abf, channel):
        channeltype = 'abf'

    for ifly, fly in enumerate(group):
        for rec in fly:
            trial_ends = get_trial_ts(rec)
            if trial_ends is None: continue
            sp = rec.tif.sampling_period
            tif_inds_set = [(rec.tif.t > (start + sp)) & (rec.tif.t < (end - sp)) for start, end in trial_ends]
            abf_inds_set = [(rec.abf.t > (start + sp)) & (rec.abf.t < (end - sp)) for start, end in trial_ends]
            stimids = [rec.abf.stimid[ind].mean() for ind in abf_inds_set]
            temps = [rec.abf.temp[ind].mean() > temp_thresh for ind in abf_inds_set]

            if channeltype == 'tif':
                rec_channel = getattr(rec.tif, channel)
            elif channeltype == 'abf':
                rec_channel = rec.subsample(channel, lag_ms)

            stim_counter = np.zeros(len(stim_dict))
            for itrial, tif_ind in enumerate(tif_inds_set):
                istimid = stimids[itrial] - 1 # stimids start at 1, need to go to zero-based indexing
                itemp = temps[itrial]
                istim = stim_counter[istimid] # the trial number of this specific stimid
                ichannel = rec_channel[tif_ind]

                try:
                    data[ifly, istimid, itemp, istim, :len(ichannel)] = ichannel
                except IndexError:
                    print rec.name

                stim_counter[istimid] += 1

    return data


def get_trial_inds(rec, abf_or_tif, trial_ts, tlag_s=2):
    subrec = getattr(rec, abf_or_tif[:3])
    t = subrec.t_orig if 'orig' in abf_or_tif else subrec.t
    peak_inds = np.array([np.where(t >= trial_t)[0][0] for trial_t in trial_ts])
    sr = subrec.sampling_rate if fc.contains(['orig', 'tif'], abf_or_tif) else subrec.subsampling_rate
    indlag = np.ceil(tlag_s * sr)
    abf_ind = np.arange(-indlag, indlag+1)
    abf_inds = np.tile(abf_ind, (len(peak_inds), 1)) + peak_inds[None, :].T
    return abf_inds.astype(int)


def get_abf_trials(rec, channel, trial_ts, tlag_s=2):
    t = rec.abf.t_orig if 'orig' in channel else rec.abf.t
    peak_inds = np.array([np.where(t >= peak_t)[0][0] for peak_t in trial_ts])
    sr = rec.abf.sampling_rate if 'orig' in channel else rec.abf.subsampling_rate
    indlag = tlag_s * sr
    ch = getattr(rec.abf, channel)

    ch_trials = [ch[ind - indlag : ind + indlag + 1]
                    for ind in peak_inds
                    if (ind-indlag>0) and (ind+indlag+1<len(ch))]

    t_trials = [t[ind - indlag : ind + indlag + 1] - t[ind]
                for ind in peak_inds
                if (ind-indlag>0) and (ind+indlag+1<len(t))]

    return t_trials, ch_trials


def get_tif_trials(rec, metric, trial_ts, tlag_s=2):
#     peak_ts = rec.abf.t[abf_peak_inds]
    gcz = rec.tif.gcz
    glomeruli = rec.tif.xedges[:-1]
    if type(metric) is str:
        metric_values = getattr(rec.tif, metric)
    else:
        metric_values = metric(rec)
    t = rec.tif.t
    metric_trials = []
    t_trials = []
    lag_ind = int(np.ceil(tlag_s*rec.tif.sampling_rate))
    indmax = len(t)-1
    for peak_t in trial_ts:
        if peak_t > t.max(): continue
        peak_ind = np.where(t >= peak_t)[0][0]
        idx = np.arange(peak_ind - lag_ind, peak_ind + lag_ind + 1)
        if (idx[0] > 0) and (idx[-1] < indmax):
            metric_trials.append(metric_values[idx])
            t_trials.append(t[idx] - peak_t)

#     t_trials = np.concatenate(t_trials)
#     rml_trials = np.concatenate(rml_trials)
    return t_trials, metric_trials


def get_group_data(group, label, filler_len=100, subsample=False):
    # Collect data
    if subsample:
        nanfill = np.zeros((2, filler_len))
        nanfill[:] = np.nan
        t_trigger = np.hstack(
            [np.hstack([
                        np.hstack([np.vstack([r.pb.t, r.subsample(label)]), nanfill])
                        for r in fly])
             for fly in group])
        t, trigger = t_trigger
        dt = np.zeros_like(t)
        dt[:-1] = np.diff(t)
        dt[-1] = np.nan
        return dt, trigger

    elif label != 'pb.c1.acn':
        nanfill = np.zeros((2, filler_len))
        nanfill[:] = np.nan
        t_trigger = np.hstack(
            [np.hstack([
                        np.hstack([r.get_signal(label), nanfill])
                        for r in fly])
             for fly in group])
        t, trigger = t_trigger
        dt = np.zeros_like(t)
        dt[:-1] = np.diff(t)
        dt[-1] = np.nan
        return dt, trigger

    elif label == 'pb.c1.acn':
        nanfill = np.zeros((16, filler_len))
        nanfill[:] = np.nan
        trigs = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
        for fly in group:
            for r in fly:
                _, trig = r.get_signal(label)
                trigs = np.hstack([trigs, trig.T])
                trigs = np.hstack([trigs, nanfill])
        return trigs


# Plotting trial data

def plot_trials(self, tlim=(-2, 5), imaging=('rcn', 'gcn'), gcoverlay=('phase'), overlay=('phase', 'xstimb'),
                channels=(), trigger='pressure', trigthresh=20, sf=1, save=True, **kwargs):
    """
    Plots all trials in a row. Essentially serially hijacks the plot function.
    :param self: GCaMP_ABF recording.
    :param tlim: Time window, 0 = stimulation time for each panel.
    :param imaging: See plot.
    :param gcoverlay: See plot.
    :param overlay: See plot.
    :param channels: See plot.
    :param trigger: The channel on which to trigger trials.
    :param trigthresh: Threshold with which to trigger each trial.
    :param sf: Scale factor for figure size.
    :param save: True to save figure.
    :param kwargs: More plot options.
    :return:
    """
    trig_inds = bh.fc.rising_trig(getattr(self.abf, trigger), trigger=trigthresh,
                                  mindist=.01 * self.abf.subsampling_rate)
    if len(trig_inds) == 0:
        print 'No triggers detected.'
        return

    trig_ts = self.abf.t[trig_inds]
    trig_ts = trig_ts[np.insert(np.diff(trig_ts) > 1, 0, 1)]
    nrows = len(trig_ts)
    ncols = np.array([len(imaging), powerspec, len(channels), len(overlay) > 0]).sum()
    def wr_fcn(imaging_label):
        if 'eb' in imaging_label:
            return 1
        elif 'pb' in imaging_label:
            return 2
    plt.figure(1, (3 * ncols * sf, 3 * nrows * sf))
    # plt.suptitle(self.folder, size=10)
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    wr = map(wr_fcn, imaging) + [1]*powerspec + [2]*(len(overlay)>0) + [1]*(len(channels))  # width ratios
    gs = gridspec.GridSpec(nrows*2, ncols, width_ratios=wr)
    ###
    plt.title(self.folder, size=10)
    gs.update(wspace=0.2)

    for row, trig_t in enumerate(trig_ts):
        show_axes = (row == 0)
        offset = self.get_offset(toffset=trig_t-1)
        plot(self, tlim, imaging, gcoverlay, overlay, channels, offset=offset, gs=gs, row=row,
            trig_t=trig_t, show_axes=show_axes, save=False, **kwargs)

    if save:
        plt.savefig('%s_%i-%is_plot.png' % (self.abf.basename, tlim[0], tlim[1]), bbox_inches='tight')

    plt.show()


def plot_trial_mean(self, tlim=(-2, 5), tbinsize=.2, t_collapse=(.2, .6), sf=1, save=True, **kwargs):
    """
    Plots mean across trials. Useful for stimulation experiments.
    :param self: GCaMP_ABF recording.
    :param tlim: Time window to plot.
    :param tbinsize: Time bins.
    :param t_collapse: Time window to average in 1D plot.
    :param sf: Scale factor for figure size.
    :param save: True to save figure.
    :param kwargs:
    :return:
    """

    ttrials, greentrials, redtrials, phasetrials = get_trials(self, tlim,  **kwargs)

    ntrials = len(ttrials)
    ttrials = np.concatenate(ttrials)
    greentrials = np.concatenate(greentrials)
    redtrials = np.concatenate(redtrials)

    tbins = np.arange(tlim[0], tlim[1] + tbinsize, tbinsize)
    green_mean = bh.fc.binyfromx(ttrials, greentrials, tbins, metric=np.median, axis=0)
    red_mean = bh.fc.binyfromx(ttrials, redtrials, tbins, metric=np.median, axis=0)

    plt.figure(1, (8 * sf, 5 * sf))
    rcParams['font.size'] = 9 * sf
    ph.set_tickdir('out')
    gs = gridspec.GridSpec(2, 3, height_ratios=[1, 10], width_ratios=[10, 10, 1])
    plt.ylabel('Time after stimulation (s)')
    plt.xlabel('Z-Score (stdev)')

    # Red Channel heatmap
    axred = plt.subplot(gs[1, 0])
    dx = np.diff(self.tif.xedges)[0] / 2.
    xedges = self.tif.xedges + dx
    xlim = [xedges[0], xedges[-1]]
    ph.adjust_spines(axred, ['left', 'bottom'], xticks=xedges[:-1:2] + dx)
    axred.set_ylabel('Time after stimulation (s)')
    axred.set_xlabel('Alexa568 (%i trials)' % ntrials)
    yedges = tbins
    axred.pcolormesh(xedges, yedges, red_mean, cmap=plt.cm.Reds)
    axred.axhline(c='black', ls='--', alpha=.8)
    axred.invert_yaxis()

    # Green channel heatmap (GCaMP)
    axgc = plt.subplot(gs[1, 1])
    ph.adjust_spines(axgc, ['bottom'], xticks=xedges[:-1:2] + dx)
    axgc.set_xlabel('GCaMP (%i trials)' % ntrials)
    yedges = tbins
    axgc.pcolormesh(xedges, yedges, green_mean, cmap=plt.cm.Blues, alpha=1)
    axgc.axhline(c='black', ls='--', alpha=.8)
    axgc.invert_yaxis()

    # 1D snapshot of green and red channels along bridge
    # Green channel
    ax = plt.subplot(gs[0, 1])
    ph.adjust_spines(ax, ['top'], xticks=xedges[:-1:2] + .5)
    g = xedges[1:-2] + dx
    tb = tbins[:-1] + tbinsize / 2.
    post_inds = (tb >= t_collapse[0]) & (tb < t_collapse[1])
    pre_inds = (tb >= -.4) & (tb < 0)
    green_tcollapse = green_mean[post_inds, 1:-1].mean(axis=0) - green_mean[pre_inds, 1:-1].mean(axis=0)
    ax.plot(g, green_tcollapse, c=ph.blue)
    # Red channel
    ax2 = ax.twinx()
    ph.adjust_spines(ax2, ['top'])
    redinds = (tb >= 0) & (tb < .4)
    red_tcollapse = red_mean[redinds, 1:-1].mean(axis=0)  # - red_mean[pre_inds, 1:-1].mean(axis=0)
    ax2.plot(g, red_tcollapse, c=ph.red)
    ax2.set_xlim(xlim)

    # show phases of green and red channels
    # green_peak_inds = fc.detect_peaks(green_tcollapse, mph=0)
    # red_peak_glom = np.argmax(red_tcollapse)+2 # +1 fora zero-based index, +1 for missing first glomerulus
    # green_peak_glom = green_peak_inds[np.argmin(np.abs(green_peak_inds - red_peak_glom))]+2   # +1 fora zero-based index, +1 for missing first glomerulus
    # print red_peak_glom
    # print green_peak_glom
    # ax.axvline(red_peak_glom, c='red') # +1 adjusts for missing first glomerulus
    # ax.axvline(green_peak_glom, c='blue') # +1 adjusts for missing first glomerulus
    # ax.set_xlim(xlim)

    # Draw an arrow where the pipette was positioned (maximum signal in red channel)
    red_maxglom = np.argmax(red_tcollapse) + 1
    axgc.arrow(red_maxglom + 1, -.4, 0, .2, head_width=0.5, head_length=0.1, fc='r', ec='r', lw=2)
    axred.arrow(red_maxglom + 1, -.4, 0, .2, head_width=0.5, head_length=0.1, fc='r', ec='r', lw=2)

    # 1D snapshot of peak glomerulus in green and red channels over time     
    # ax = plt.subplot(gs[1, 2])
    # ph.adjust_spines(ax, ['right'])
    # ax.axhline(c='black', ls='--', alpha=.8)
    # ax.plot(red_mean.mean(axis=1), tb, c=ph.red)
    # ax.invert_yaxis()
    # ax2 = ax.twiny()
    # ph.adjust_spines(ax2, ['right'])
    # ax2.plot(green_mean[:, np.argmax(green_tcollapse)], tb, c=ph.blue)

    if save:
        plt.savefig('%s_%i-%is_2d_median_trials.png' % (self.abf.basename, tlim[0], tlim[1]), bbox_inches='tight')



# Miscellaneous plots

def plot_offsets(group, color='black', nstims=2, mode='mean', figname=''):
    """
    Plots offsets under two modes. Mode 'mean' plots the circular mean and circular standard
    deviation for each fly. Mode 'all' plots the circular mean for each trial in each fly.
    :param group: List of fly objects (each a list of recs).
    :param color: Color to plot.
    :param nstims: Number of stimuli in each recording.
    :param mode: 'mean' or 'all'. See above.
    :param figname: Figure name to save.
    :return:
    """
    stim_dict = {1: 'Bar', 2: 'Dark'}
    ntrials_per_stim = 3
    offsets = np.zeros((len(group), ntrials_per_stim*2))
    offsets[:] = np.nan
    for ifly, fly in enumerate(group):
        fly_offsets = []
        for rec in fly:
            ts = get_trial_ts(rec)
            stimid = rec.subsample('stimid')
            xstim = fc.circ_moving_average(rec.pb.xstim, n=3)
            phase = rec.pb.c1.phase

            rec_offsets = np.zeros(len(ts)/nstims)
            bar_cnt = 0
            for i, (t0, t1) in enumerate(ts):
                idx = (rec.pb.t >= t0) & (rec.pb.t < t1)
                istimid = int(stimid[idx].mean().round())
                if istimid==0:
                    print stimid[idx].mean()
                    print rec.name
                if stim_dict[istimid] == 'Bar':
                    idx = idx & (rec.pb.xstim > -135) & (rec.pb.xstim < 135)
                    if idx.sum()*rec.pb.sampling_period < 10: continue
                    ioffset = fc.circmean(phase[idx] - xstim[idx])
                    rec_offsets[bar_cnt] = ioffset
                    bar_cnt += 1

            fly_offsets.append(rec_offsets)
        fly_offsets = np.concatenate(fly_offsets)
        fly_ntrials = (np.isnan(fly_offsets)==False).sum()
        offsets[ifly, :fly_ntrials] = fly_offsets
    offsets = offsets / 180.
    if mode == 'mean':
        # Plot median offset for each fly
        plt.figure(1, (4, 4))
        ax = plt.subplot(111)
        stimlabels = ['Bar', 'Dark']
        plt.xticks(np.arange(len(group))+1)
        plt.xlim(0, len(group)+1.5)
        plt.ylim(-1, 1)
        plt.yticks([-1,0,1])
        ph.adjust_spines(ax, ['left', 'bottom'])

        mean = fc.circmean(offsets, axis=1)
        err = fc.circstd(offsets, axis=1)
        mean, err = zip(*sorted(zip(mean, err)))
        plt.errorbar(np.arange(len(group))+1, mean, err, fmt='_', c=color, lw=1)
    elif mode == 'all':
        # Plot offset for each trial in each fly
        ncols = 3
        nrows = int(np.ceil(len(group)/float(ncols)))
        plt.figure(1, (nrows*3, ncols*3))
        gs = gridspec.GridSpec(nrows, ncols)
        for ifly in range(len(group)):
            irow = ifly/3
            icol = ifly%3
            ax = plt.subplot(gs[irow, icol])
            ax.grid()
            spines = ['left']
            if irow==nrows-1 and icol==0:
                # ax.set_ylabel('Offset (pi rad)')
                # ax.set_xlabel('Trial Number')
                spines.append('bottom')
            ph.adjust_spines(ax, spines, ylim=(-1, 1), yticks=[-1, 0, 1], xlim=(.5, offsets.shape[1]+.5))
            x = np.arange(offsets.shape[1])+1
            ax.scatter(x, offsets[ifly], edgecolor=color, facecolor=color, s=40)

    if figname:
        ph.save(figname, rec)


def plot_periods(group, color='black', figname='', show_labels=True):
    periods = np.zeros((2, len(group)))
    for ifly, fly in enumerate(group):
        for rec in fly:
            if rec.pb.c1.celltype == 'PEN':
                a = rec.pb.c1.acn
            elif rec.pb.c1.celltype == 'EIP':
                a = rec.pb.c1.an[:, 1:-1]
            power, period, phase = fc.powerspec(a)
            window = (period > 2) & (period < 18)
            period = period[window]
            stimid = rec.subsample('stimid')
            clbar = (stimid == 1) & (rec.pb.xstim > -135) & (rec.pb.xstim < 135)
            dark = (stimid == 2)
            pbar = period[np.argmax(power[clbar].mean(axis=0)[window])]
            pdark = period[np.argmax(power[dark].mean(axis=0)[window])]
            if pbar >10 or pdark > 10:
                print pbar, pdark
                print rec.name
            periods[0, ifly] = period[np.argmax(power[clbar].mean(axis=0)[window])]
            periods[1, ifly] = period[np.argmax(power[dark].mean(axis=0)[window])]

    # Plot periods per stimulus
    plt.figure(1, (6, 6))
    ax = plt.subplot(111)
    stimlabels = ['Bar', 'Dark']
    plt.xticks(range(len(group)), stimlabels)
    plt.xlim(-.5, 2-.5)
    plt.ylim(6, 10)
    plt.yticks([6, 8, 10])
    ph.adjust_spines(ax, ['left', 'bottom'])

    for istim, stimlabel in enumerate(stimlabels):
        x = np.random.normal(istim, .05, len(group))
        med = np.nanmedian(periods[istim])
        std = np.nanstd(periods[istim])
        plt.scatter(x, periods[istim], facecolor='none', edgecolor=color, s=100, lw=2)
        plt.errorbar(istim+.2, med, std, fmt='_', c=color)

    if figname:
        ph.save(figname, rec)


def plot_frames(rec, ts, cmap=plt.cm.Blues, save=True, hspace=.1):
    """
    Plots frames at different times, with a gaussian filter applied.
    :param rec: GCaMP_ABF object.
    :param ts: Time points to plot frames.
    :param cmap: Colormap for the frames.
    :param save: True will save the figure.
    :param hspace: How far the frames are spaced. Ranges (0, 1).
    :return:
    """
    plt.figure(1, (6, 6))
    gs = gridspec.GridSpec(len(ts), 1)
    gs.update(hspace=hspace)
    t = rec.fb.t
    tifmax = rec.fb.c1.tif.max(axis=1)
    for i, ti in enumerate(ts):
        ax1 = plt.subplot(gs[i, 0])
        ph.adjust_spines(ax1, [])
        tidx = np.where(t>=ti)[0][0]
        ax1.set_ylabel('%.1f s' %(t[tidx] - ts[0]), rotation='horizontal', labelpad=30)
        frame = gaussian_filter(tifmax[tidx], 1.5)
        ax1.imshow(frame, cmap=cmap)
    if save:
        plt.savefig('%s_frames.png' %rec.basename, bbox_inches='tight')


def plot_segmentations(rec, cmap=plt.cm.Blues, save=True, hspace=.1):
    """
    Plots the outline of the mask for each z-plane.
    :param rec: GCaMP_ABF object.
    :param cmap: Colormap for the frames.
    :param save: True to save figure.
    :param hspace: Spacing between frames.
    :return:
    """
    plt.figure(1, (6, 6))
    zs = [0, 9, 18]
    gs = gridspec.GridSpec(len(zs), 1)
    gs.update(hspace=hspace)
    tifmean = rec.pb.c1.tif.mean(axis=0)
    maskoutline = (rec.pb.c1.mask==0)

    for iz, z in enumerate(zs):
        ax1 = plt.subplot(gs[iz, 0])
        ph.adjust_spines(ax1, [])
        itifmean = copy.copy(tifmean[iz])
        itifmean[maskoutline[iz]] = 2**15
#         ax1.set_ylabel(r'%i $\u$m' %z, rotation='horizontal', labelpad=30)
        ax1.imshow(itifmean, cmap=cmap)
    if save:
        plt.savefig('%s_segments.png' %rec.basename, bbox_inches='tight')


def plot_stimulus_onoff(self, save=True):
    stimon_tifidx = self._get_stimon_tifidx()
    tifshape = self.tif.tif.shape
    newshape = (tifshape[0], tifshape[1], tifshape[2] * tifshape[3])
    tifctl = (self.tif.tif * self.tif.ctlmask).reshape(newshape).mean(axis=-1)
    tifbridge = (self.tif.tifm).reshape(newshape).mean(axis=-1)

    ctl_baron = tifctl[stimon_tifidx]
    ctl_baroff = tifctl[stimon_tifidx == False]
    print 'Control, Bar Hidden mean, std:', ctl_baroff.mean(), ctl_baroff.std()
    print 'Control, Bar Visible mean, std:', ctl_baron.mean(), ctl_baron.std()

    bridge_baron = tifbridge[stimon_tifidx]
    bridge_baroff = tifbridge[stimon_tifidx == False]
    print 'Bridge, Bar Hidden mean, std:', bridge_baroff.mean(), bridge_baroff.std()
    print 'Bridge, Bar Visible mean, std:', bridge_baron.mean(), bridge_baron.std()

    plt.figure(1, (8, 6))
    plt.suptitle(self.folder)
    ax = plt.subplot(111)
    plt.scatter(np.random.normal(0, .1, len(ctl_baroff)), ctl_baroff, c=ph.dgrey, lw=0, s=5)
    plt.scatter(np.random.normal(1, .1, len(ctl_baron)), ctl_baron, c=ph.blue, lw=0, s=5)
    plt.scatter(np.random.normal(2, .1, len(bridge_baroff)), bridge_baroff, c=ph.dgrey, lw=0, s=5)
    plt.scatter(np.random.normal(3, .1, len(bridge_baron)), bridge_baron, c=ph.blue, lw=0, s=5)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Control\nBar hidden', 'Control\nBar visible', 'Bridge\nBar hidden', 'Bridge\nBar visible'])
    plt.grid()

    if save:
        plt.savefig('%s_stimulus_onoff_intensities.png' % (self.abf.basename), bbox_inches='tight')


def plot_distribution(group, label, xbins=None, figname=''):
    dt, dforw = get_group_data(group, label)
    mask = np.isnan(dforw)
    if xbins is None: xbins = 100
    h, x = np.histogram(dforw[~mask], bins=xbins, density=True)
    celltype = group[0][0].pb.c1.celltype
    color = colours[celltype]

    plt.figure(1, (6, 4))
    ph.set_tickdir('out')
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'])
    plt.bar(x[:-1], h, width=np.diff(x)[0], color=color, edgecolor='black')

    if figname:
        ph.save(figname)


def plot_intensity_vs_age(groups, stim_dict, ages):
    stim_map = {v: k for k, v in stim_dict.items()}

    plt.figure(1, (4, 10))
    gs = gridspec.GridSpec(len(groups), 1)
    gs.update(hspace=.5)

    for igroup, (age, group) in enumerate(zip(ages, groups)):
        intensities = np.zeros(len(group))
        intensities[:] = np.nan
        for ifly, fly in enumerate(group):
            fly_intensities = np.zeros((2, len(fly)/2))
            cnt = np.zeros(2)
            for rec in fly:
                itemp = rec.abf.temp.mean() > 26
                dark = rec.subsample('stimid') == stim_map['Dark']
                fly_intensities[itemp, cnt[itemp]] = np.nanmean(rec.pb.c1.an[dark])
                cnt[itemp] += 1
            fly_intensity = fly_intensities[1].mean() / fly_intensities[0].mean()
            intensities[ifly] = fly_intensity

        ax = plt.subplot(gs[igroup, 0])
        ax.axhline(1, c='grey', ls='--')
        ax.axhline(0, c='grey', ls='--')
        ax.set_ylim(0, 1.5)
        ax.set_yticks(np.arange(0, 1.6, .5))
        if igroup == len(groups)-1:
            ph.adjust_spines(ax, ['left', 'bottom'])
            ax.set_xlabel('Age (days)')
        else:
            ph.adjust_spines(ax, ['left'])
        ax.scatter(age, intensities)


def plot_regtransform_distribution(rec, cutoff=5):
    filename = '%s_transform.csv' %rec.pb.basename
    with open(filename) as fh:
        v = csv.reader(fh)
        ar = []
        for line in v:
            try:
                ar.append(map(float, line))
            except ValueError:
                pass
    def f(ar):
        return np.hypot(*ar)
    ar = np.array(ar).T
    dist = np.apply_along_axis(f, 0, ar[-2:])
    bins = np.arange(0, 10, .25)

    ax = plt.subplot(111)
    a = plt.hist(dist, bins=bins)


    print 'Mean = %.2f' %dist.mean()
    print 'No > %i = %i'%(cutoff, (np.abs(dist - dist.mean()) > cutoff).sum())
    ax.axvline(dist.mean()+cutoff)


def plot_steadystate_phase_histogram(group, maxvel=.2, wait_s=1, minlen_s=2, save=True):
    dt, phase = get_group_data(group, 'pb.c1.phase')
    phase = fc.wrap(phase)
    # dt, dphase = get_group_data(group, 'pb.c1.dphasef')
    dt, dhead = get_group_data(group, 'pb.dhead')
    dt, stimid = get_group_data(group, 'abf.stimid', subsample=True)
    idx = (dhead < maxvel) & (dhead > -maxvel) #& (stimid==1)
    wait_idx = int(np.ceil(wait_s / .15))
    minlen_idx = int(np.ceil(minlen_s / .15))
    inds = fc.get_contiguous_inds(idx, trim_left=wait_idx, keep_ends=True)
    phases = np.array([phase[ind].mean() for ind in inds])

    celltype = group[0][0].pb.c1.celltype
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'])
    binwidth = 5
    bins = np.arange(-180, 180, binwidth)
    # h, x = np.histogram(phases, bins=bins)
    # ax.bar(x[:-1], h, facecolor=ph.blue, width=binwidth)
    plt.hist(phases, bins=bins, facecolor=colours[celltype])
    ax.set_xlim(-180, 180)
    ax.set_xticks(np.arange(-180, 181, 45))
    ax.set_ylabel('Counts')
    ax.set_xlabel('Phase')
    ylim = ax.get_ylim()
    ax.set_yticks(np.arange(ylim[0], ylim[1]+1))

    if save:

        ph.save('%s_Steady_state_phase_histogram' %celltype)



# Plotting Right - Left (or Right, Left alone) Bridge vs phase or heading velocity

def plot_rml_v_dhead(groups, alabel='az', vel='abf.dhead', binsize=30, binlim=360, lag_ms=300, ylim=None, figname=''):
    if not type(groups) is list: groups = [groups]
    if not type(groups[0]) is list: groups = [groups]
    turn_bins = np.arange(-binlim-binsize/2., binlim+1.5*binsize, binsize)

    # Setup Figure
    plt.figure(1, (4, 4))
    ph.set_tickdir('out')
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'])
    ax.set_xticks(np.arange(-binlim, binlim+1, 180)/180.)
    # ax.grid()
    ax.set_xlabel('Turning velocity (pi rad/s)', labelpad=20)
    ax.set_ylabel('Right - Left\nBridge (stdev)', labelpad=20)
    ax.axhline(ls='--', c='grey')
    ax.axvline(ls='--', c='grey')

    # Plot groups of flies
    colours = {'EIP': ph.blue, 'PEN': ph.orange}
    tbc = turn_bins[:-1] + np.diff(turn_bins)[0] / 2.  # Turn bin centers
    glomeruli = groups[0][0][0].pb.c1.xedges[:-1]
    for group in groups:
        rmls = np.zeros((len(group), len(tbc)))
        for ifly, fly in enumerate(group):
            ab = np.nanmean([rec.bin_gcamp('pb.c1.' + alabel, vel, turn_bins, lag_ms) for rec in fly], axis=0)
            rml = np.nanmean(ab[:, glomeruli>=9], axis=1) - np.nanmean(ab[:, glomeruli<9], axis=1)
            rmls[ifly] = rml

        med = np.nanmedian(rmls, axis=0)
        n = (np.isnan(rmls)==False).sum(axis=0)
        sem = np.nanstd(rmls, axis=0) / np.sqrt(n)
        colour = colours[rec.pb.c1.celltype]
        ax.plot(tbc/180., med, c=colour, lw=2)
        plt.fill_between(tbc/180., med-sem, med+sem, facecolor=colour, edgecolor='none', alpha=.2)
        # for rml in rmls:
                #     ax.plot(tbc, rml, c=colours[i], lw=1, alpha=.5)
    if ylim:
        ax.set_ylim(ylim)
        ax.set_yticks(np.arange(ylim[0], ylim[1]*2, ylim[1]))
    if figname:
        ph.save(figname + '_%s_%ims_lag' % (alabel, lag_ms))


def plot_rml_dhead_slope(groups, alabel='az', vel='abf.dhead', lags_ms=np.arange(-2000, 2001, 100), ylim=None, figname='',
                                  filetype='png'):
    if not type(groups) is list: groups = [groups]
    if not type(groups[0]) is list: groups = [groups]
    turn_bins = np.arange(-200, 200.1, 25)

    # Setup Figure
    plt.figure(1, (10, 4))
    ph.set_tickdir('out')
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'])
    # ax.grid()
    ax.axhline(c='grey', ls='--')
    ax.set_xlabel('Lag (ms)')
    ax.set_ylabel('Slope (std / (pi rad/s))', labelpad=20)

    # Extract slope for each time lag for each group of flies
    colours = {'EIP': ph.blue, 'PEN': ph.orange}
    tbc = turn_bins[:-1] + np.diff(turn_bins)[0] / 2.  # Turn bin centers
    glomeruli = groups[0][0][0].pb.c1.xedges[:-1]
    for igroup, group in enumerate(groups):
        slopes = np.zeros((len(group), len(lags_ms)))
        for ifly, fly in enumerate(group):
            for ilag, lag_ms in enumerate(lags_ms):
                ab = np.nanmean([bin_gcamp(rec, 'pb.c1.' + alabel, 'abf.dhead', turn_bins, lag_ms) for rec in fly], axis=0)
                rml = np.nanmean(ab[:, glomeruli>=9], axis=1) - np.nanmean(ab[:, glomeruli<9], axis=1)

                a, b, r, p, e = stats.linregress(tbc, rml)
                slopes[ifly, ilag] = a

        slopes *= 180.
        med = np.nanmedian(slopes, axis=0)
        n = (np.isnan(slopes)==False).sum(axis=0)
        sem = np.nanstd(slopes, axis=0) / np.sqrt(n)
        colour = colours[rec.pb.c1.celltype]
        plt.plot(lags_ms, med, c=colour, lw=2)
        plt.fill_between(lags_ms, med-sem, med+sem, facecolor=colour, edgecolor='none', alpha=.2)

    if ylim:
        ax.set_ylim(ylim)

    if figname:
        binsize = int(np.diff(lags_ms)[0])
        ph.save(figname + ('_%s_%ims_bins' % (alabel, binsize)))

# Jonathan's old functions, updated before Dec.5

def plot_isolated_turns0(group, tlim=(-2, 2), save=True, overlay=True,
                        trigger='pb.c1.dphasef', lines=[],  **kwargs):
    """
    Plots phase and R-L triggered on a change in phase (using detect peaks on dphase).
    :param group: List of flies.
    :param tlim: Time window.
    :param save: True to save figure.
    :param overlay: Overlay mean and sem of phase and R-L on same plot.
    Otherwise, plot them separately with individual traces.
    :param kwargs: kwargs for get_isolated_peak_inds.s
    :return:
    """
    ph.set_tickdir('out')
    ph.set_fontsize(15)
    # Setup figure
    if overlay:
        plt.figure(1, (5, 5))
        gs = gridspec.GridSpec(1, 2)
    else:
        plt.figure(1, (7, 7))
        gs = gridspec.GridSpec(2, 2)
    celltype = group[0][0].ims[0].c1.celltype
    colour = colours[celltype]

    for col, direction in enumerate(['right', 'left']):
        # Find peaks
        # Collect data from all groups
        filler_len = max(100, int(mpd_s*10))
        dt, trigger = get_group_data(group, trigger, filler_len)

        # find peaks
        dt, peak_inds = get_isolated_peak_inds(trigger, dt,  direction=direction, **kwargs)
        window_ind0 = np.round(np.arange((tlim[0]-1)/dt[0], (tlim[1]+1)/dt[0]+1)).astype(int)
        window_inds = np.tile(window_ind0, (len(peak_inds), 1)) + peak_inds[:, None]
        t0 = np.where((window_ind0==0))[0][0]-1

        # Plot Phase
        phase_dt, phase = get_group_data(group, 'pb.c1.dphase')
        phase_turns = phase[window_inds]
        ax = plt.subplot(gs[0, col])
        if overlay:
            axes = ['bottom', 'left']
            if col == 0:
                ph.adjust_spines(ax, ['bottom', 'left'])
                ax.set_ylabel('Phase (pi rad)', color=colour)
                ax.set_xlabel('Time (s)')
            else:
                ph.adjust_spines(ax, ['bottom'])
            ax.set_ylim(-.5, .5)
            ax.set_yticks([-.5, 0, .5])
        else:
            if col == 0:
                ph.adjust_spines(ax, ['left'])
                ax.set_ylabel('Phase (pi rad)')
            else:
                ph.adjust_spines(ax, [])
            ax.set_ylim(-1, 1)
            ax.set_yticks([-1, 0, 1])

        ax.set_xlim(tlim)
        ax.axhline(ls='--', c='black', alpha=.5)
        ax.axvline(ls='--', c='black', alpha=.5)


        t_interp = np.arange(tlim[0], tlim[1], .1)
        phase_interp = np.zeros((len(phase_turns), len(t_interp)))
        for i, (dti, phasei) in enumerate(zip(phase_dt, phase_turns)):
            t = window_ind0.astype(float)*dti
            phasei = fc.unwrap(phasei) / 180.
            phasei = phasei - phasei[t0]

            if not overlay:
                plt.plot(t, phasei, c='grey', alpha=.07)

            f = interp1d(t, phasei)
            phase_interp[i] = f(t_interp)

        # Plot mean
        mean = np.nanmean(phase_interp, axis=0)
        if overlay:
            std = np.nanstd(phase_interp, axis=0)
            n = (~np.isnan(phase_interp)).sum(axis=0)
            sem = std / np.sqrt(n)
            ax.fill_between(t_interp, mean-sem, mean+sem, edgecolor='none', facecolor=colour, alpha=.3)
            ax.plot(t_interp, mean, lw=1, c=colour)
        else:
            ax.plot(t_interp, mean, lw=2, c=colour)


        # Plot R-L
        rml_dt, rml = get_group_data(group, 'pb.c1.rmlz')
        rml_turns = rml[window_inds]

        if overlay:
            ax = ax.twinx()
            if col==1:
                ph.adjust_spines(ax, ['bottom', 'right'])
                ax.set_ylabel('Right - Left (std)')
            else:
                ph.adjust_spines(ax, ['bottom'])
            ax.set_ylim(-.5, .5)
            ax.set_yticks(np.arange(-.5, .6, .5))
        else:
            ax = plt.subplot(gs[1, col])
            if col ==0:
                ph.adjust_spines(ax, ['left', 'bottom'])
                ax.set_ylabel('Right - Left (std)')
                ax.set_xlabel('Time (s)')
            elif col == 1:
                ph.adjust_spines(ax, ['bottom'])
            ax.set_ylim(-1, 1)
            ax.set_yticks(np.arange(-1, 1.1, .5))
        ax.set_xticks(np.arange(tlim[0], tlim[1]+.5, 1))
        ax.set_xlim(tlim)
        ax.axhline(ls='--', c='black', alpha=.5)
        ax.axvline(ls='--', c='black', alpha=.5)
        if lines:
            for line in lines:
                ax.axvline(line, ls='--', c='black', alpha=.5)

        rml_interp = np.zeros((len(rml_turns), len(t_interp)))
        for i, (dti, rmli) in enumerate(zip(rml_dt, rml_turns)):
            t = window_ind0.astype(float)*dti
            # rmli = rmli - rmli[t0]
            if not overlay:
                plt.plot(t, rmli, c='grey', alpha=.07)

            f = interp1d(t, rmli)
            rml_interp[i] = f(t_interp)

        # Plot mean
        mean = np.nanmean(rml_interp, axis=0)
        if overlay:
            std = np.nanstd(rml_interp, axis=0)
            n = (~np.isnan(rml_interp)).sum(axis=0)
            sem = std / np.sqrt(n)
            ax.fill_between(t_interp, mean-sem, mean+sem, edgecolor='none', facecolor='grey', alpha=.3)
            ax.plot(t_interp, mean, lw=1, c=colour)
        else:
            ax.plot(t_interp, mean, lw=2, c=colour)


        # Plot

    if save:
        figname = '%s_R-L_triggered_mean' %celltype
        if overlay:
            figname += '_overlay'
        ph.save(figname)


def get_isolated_peak_inds(trigger, dt, direction='right', edge='peak',
                           mph=200, mph_neighbor=100, mpd_s=.5, mpw_s=None, mpw_thresh=10):
    """
    Find peak inds that are separated from other peaks, and of a minimum width.
    :param group: List of flies.
    :param trigger: typically 'c1.dphase' or 'pb.dhead'
    str Label pointing to the signal.
    :param direction: 'left' or 'right' for direction of phase or fly turn.
    :param edge:    'peak' return output of detect_peaks
                    'left' return start of peak
                    'right' return end of peak
    :param mph: Minimum peak height (see detect peaks).
    :param mph_neighbor: Minimum neighboring peak height.
    :param mpd_s: Minimum neighboring peak distance in seconds.
    :param mpw_s: Minimum peak width in seconds.
    :param mpw_thresh: Threshold with which to calculate the peak width.
    ie. Peak edges are defined as the first value to cross mpw_thresh from the peak
    on either side. Peak width is defined as right edge - left edge.
    :return:
    """
    if type(dt) is np.float64:
        a = np.zeros(len(trigger))
        a.fill(dt)
        dt = a

    # Detect peaks
    valley = False if direction == 'right' else True
    peak_inds = fc.detect_peaks(trigger, mph=mph, valley=valley, edge='rising')
    # print 'Raw peaks > mph:\t%s' %len(peak_inds)

    # Select peaks that are far away from neighboring peaks with height greater than mph_neighbor
    if not mph_neighbor is None:
        neighbor_peak_inds_right = fc.detect_peaks(trigger, mph=mph_neighbor, show=False)
        neighbor_peak_inds_left = fc.detect_peaks(trigger, mph=mph_neighbor, valley=True, show=False)
        neighbor_peak_inds = np.unique(np.concatenate([neighbor_peak_inds_right, neighbor_peak_inds_left]))

        neighbor_peak_tile = np.tile(neighbor_peak_inds, (len(peak_inds), 1))
        peak_dist = (neighbor_peak_tile - peak_inds[:, None]) * dt[peak_inds[:, None]]
        peak_dist[peak_dist==0] = np.nan
        peak_dist = np.nanmin(np.abs(peak_dist), axis=1)
        peak_inds = peak_inds[peak_dist > mpd_s]
    # print 'Separated peaks:\t%s' %len(peak_inds)

    # Select peaks that have width greater than mpw_s.
    if (not mpw_s is None) or (not edge == 'peak'):
        if direction == 'right':
            peak_edges = np.where(np.diff(trigger > mpw_thresh))[0]
        else:
            peak_edges = np.where(np.diff(trigger < -mpw_thresh))[0]

        peak_edges_tile = np.tile(peak_edges, (len(peak_inds), 1))
        peak_edges_dist = (peak_edges_tile - peak_inds[:, None]).astype(float)

        # Get left edge distance from peak (negative)
        peak_edges_left = copy.copy(peak_edges_dist)
        peak_edges_left[peak_edges_dist>=0] = np.nan
        peak_edge_left = np.nanmax(peak_edges_left, axis=1)

        left_edge_inds = peak_inds + peak_edge_left

        # Get right edge distance from peak (positive)
        peak_edges_right = copy.copy(peak_edges_dist)
        peak_edges_right[peak_edges_dist<=0] = np.nan
        peak_edge_right = np.nanmin(peak_edges_right, axis=1)

        right_edge_inds = peak_inds + peak_edge_right

        # get width from right-left
        peak_width = peak_edge_right - peak_edge_left * dt[peak_inds]
        peak_inds = peak_inds[peak_width>mpw_s]

    # print 'Wide peaks:\t\t%s' %len(peak_inds)

    if edge == 'left':
        idx = left_edge_inds[~np.isnan(left_edge_inds)].astype(int)
    elif edge == 'right':
        idx = right_edge_inds[~np.isnan(right_edge_inds)].astype(int)
    elif edge == 'peak':
        idx = peak_inds[~np.isnan(peak_inds)].astype(int)

    return dt[idx], idx


def get_isolated_peak_trials(group, trigger, channels=('c1.dphase',), window=(-4, 4), **kwargs):
    # buffer the window by 1s:
    window = list(window)
    window[0], window[1] = window[0]-1, window[1]+1
    window = tuple(window)

    mintifdt = .1
    tifinds = np.arange(window[0]/mintifdt, window[1]/mintifdt+1, 1).astype(int)

    minabfdt = .02
    abfinds = np.arange(window[0]/minabfdt, window[1]/minabfdt+1, 1).astype(int)


    trigger_label = copy.copy(trigger)
    data = [[] for _ in channels]

    for fly in group:
        for rec in fly:
            ttrig, trigger = rec.get_signal(trigger_label)
            dt = np.diff(ttrig).mean()
            dt, idx = get_isolated_peak_inds(trigger, dt, **kwargs)
            tpeaks = ttrig[idx]
            for ichannel, channel in enumerate(channels):

                ts, signal = rec.get_signal(channel)
                for tpeak in tpeaks:
                    if 'abf' in channel:
                        idxi = abfinds + np.where(ts>=tpeak)[0][0]
                    else:
                        idxi = tifinds + np.where(ts>=tpeak)[0][0]

                    if idxi[0] >= 0 and idxi[-1] < len(ts):
                        data[ichannel].append([ts[idxi]-tpeak, signal[idxi]])

    for i in range(len(data)):
        data[i] = np.array(data[i])

    peaks = {channel: idata for channel, idata in zip(channels, data)}

    return peaks


def plot_isolated_turns2(group, tlim=(-2, 2), save=True, overlay=True, trigger='dhead', lines=(),
                         show_all=True, show_err=True, channels=('pb.c1.dphase', 'pb.c1.rmlz'), ylims=(),
                         yticks=(), toffset=True, fig_num=1, **kwargs):
    """
    Plots phase and R-L triggered on a change in phase (using detect peaks on dphase).
    :param group: List of flies.
    :param tlim: Time window.
    :param save: True to save figure.
    :param overlay: Overlay mean and sem of phase and R-L on same plot.
    Otherwise, plot them separately with individual traces.
    :param kwargs: kwargs for get_isolated_peak_inds.s
    :return:
    """
    ph.set_fontsize(15)
    # Setup figure
    if overlay and len(channels)==2:
        plt.figure(fig_num, (9, 9))
        gs = gridspec.GridSpec(2, 2)
    else:
        plt.figure(fig_num, (16, 16))
        gs = gridspec.GridSpec(len(channels), 2)
    celltype = group[0][0].ims[0].c1.celltype
    colour = colours[celltype]
    axs = []

    for col, direction in enumerate(['right', 'left']):
        # Find peaks
        peak_dict = get_isolated_peak_trials(group, trigger, channels, window=tlim,
                direction=direction, **kwargs)
        print '%i %s Turns' %(len(peak_dict.items()[0][1]), direction.capitalize())

        for ichannel, label in enumerate(channels):
            row = len(channels) - 1 - ichannel
            peaks = peak_dict[label]
            ax = plt.subplot(gs[row, col])
            axs.append(ax)
            show_axes = []
            if row == len(peak_dict)-1:
                show_axes.append('bottom')
            if col == 0:
                show_axes.append('left')
                ax.set_ylabel(label, labelpad=20)
            if ichannel == 0:
                ax.set_xlabel('Time (s)')
            ph.adjust_spines(ax, show_axes)
            ax.axhline(ls='--', c='black', alpha=.5)
            ax.axvline(ls='--', c='black', alpha=.5)

            if lines:
                for line in lines:
                    ax.axvline(line, ls='--', c='black', alpha=.5)

            # Plot mean peak
            if 'abf' in label:
                dt = .02
                c = 'black'
            else:
                dt = .05
                c = colour
            t_interp = np.arange(tlim[0]-.5, tlim[1]+.5, dt)
            y_interp = np.zeros((len(peaks), len(t_interp)))
            for i, (t, y) in enumerate(peaks):
                f = interp1d(t, y)
                y_interp[i] = f(t_interp)

            mean = np.nanmean(y_interp, axis=0)
            if ichannel==0:
                zero_thresh = 2
                if direction == 'left':
                    zeros = np.where(np.diff(mean<-zero_thresh))[0].astype(float)
                elif direction == 'right':
                    zeros = np.where(np.diff(mean>zero_thresh))[0].astype(float)
                t0_idx = np.argmin(np.abs(t_interp)).astype(float)
                zero_dist = zeros-t0_idx
                zero_dist[zero_dist>=0] = np.nan
                t0_idx_new = t0_idx + np.nanmax(zero_dist)
                t_offset = t_interp[t0_idx_new]
            if toffset:
                t_interp -= t_offset
            ax.plot(t_interp, mean, lw=4, c=c, zorder=2)
            if show_all:
                for t, y in peaks:
                    if toffset:
                        t -= t_offset
                    ax.plot(t, y, c='grey', alpha=.1, zorder=1)
            if show_err:
                sem = fc.nansem(y_interp, axis=0)
                # sem = np.nanstd(y_interp, axis=0)
                ax.fill_between(t_interp, mean-sem, mean+sem, edgecolor='none', facecolor=c, alpha=.5)
            ax.set_xlim(tlim)
            if ylims:
                ax.set_ylim(ylims[row])
            if yticks:
                ax.set_yticks(yticks[row])
        ax.set_title('%i %s Turns' % (len(peak_dict.items()[0][1]), direction.capitalize()))

    if save:
        figname = '%s_R-L_triggered_mean' %celltype
        if overlay:
            figname += '_overlay'
        ph.save(figname)


# Jonathan's new functions, updated Dec.5

def get_peak_inds(trigger, dt, direction='right', edge='left',
                           mph=200, mph_neighbor=100, mpd_s=.5, minpw_s=None,
                           maxpw_s=None, minpw_thresh=10, gaussfilt_s=None):
    """
    Find peak inds that are separated from other peaks, and of a minimum width.
    :param group: List of flies.
    :param trigger: typically 'c1.dphase' or 'pb.dhead'
    str Label pointing to the signal.
    :param direction: 'left' or 'right' for direction of phase or fly turn.
    :param edge:    'peak' return output of detect_peaks
                    'left' return start of peak
                    'right' return end of peak
    :param mph: Minimum peak height (see detect peaks).
    :param mph_neighbor: Minimum neighboring peak height.
    :param mpd_s: Minimum neighboring peak distance in seconds.
    :param minpw_s: Minimum peak width in seconds.
    :param mpw_thresh: Threshold with which to calculate the peak width.
    ie. Peak edges are defined as the first value to cross mpw_thresh from the peak
    on either side. Peak width is defined as right edge - left edge.
    :return:
    """

    if not gaussfilt_s is None:
        sigma = gaussfilt_s/dt
        trigger = ndimage.filters.gaussian_filter1d(trigger, sigma)

    if type(dt) is np.float64:
        a = np.zeros(len(trigger))
        a.fill(dt)
        dt = a

    # Detect peaks
    valley = False if direction == 'right' else True
    peak_inds = fc.detect_peaks(trigger, mph=mph, valley=valley, edge='rising')
    # print 'Raw peaks > mph:\t%s' %len(peak_inds)

    # Select peaks that are far away from neighboring peaks with height greater than mph_neighbor
    if not mph_neighbor is None:
        neighbor_peak_inds_right = fc.detect_peaks(trigger, mph=mph_neighbor, show=False)
        neighbor_peak_inds_left = fc.detect_peaks(trigger, mph=mph_neighbor, valley=True, show=False)
        neighbor_peak_inds = np.unique(np.concatenate([neighbor_peak_inds_right, neighbor_peak_inds_left]))

        neighbor_peak_tile = np.tile(neighbor_peak_inds, (len(peak_inds), 1))
        peak_dist = (neighbor_peak_tile - peak_inds[:, None]) * dt[peak_inds[:, None]]
        peak_dist[peak_dist==0] = np.nan
        peak_dist = np.nanmin(np.abs(peak_dist), axis=1)
        peak_inds = peak_inds[peak_dist > mpd_s]
    # print 'Separated peaks:\t%s' %len(peak_inds)

    # Select peaks that have width greater than minpw_s.
    filter = np.ones(len(peak_inds)).astype(bool)
    if (not minpw_s is None) or (not maxpw_s is None) or (not edge == 'peak'):
        if direction == 'right':
            peak_edges = np.where(np.diff(trigger > minpw_thresh))[0]
        else:
            peak_edges = np.where(np.diff(trigger < -minpw_thresh))[0]

        peak_edges_tile = np.tile(peak_edges, (len(peak_inds), 1))
        peak_edges_dist = (peak_edges_tile - peak_inds[:, None]).astype(float)

        # Get left edge distance from peak (negative)
        peak_edges_left = copy.copy(peak_edges_dist)
        peak_edges_left[peak_edges_dist>=0] = np.nan
        peak_edge_left = np.nanmax(peak_edges_left, axis=1)

        left_edge_inds = peak_inds + peak_edge_left

        # Get right edge distance from peak (positive)
        peak_edges_right = copy.copy(peak_edges_dist)
        peak_edges_right[peak_edges_dist<=0] = np.nan
        peak_edge_right = np.nanmin(peak_edges_right, axis=1)

        right_edge_inds = peak_inds + peak_edge_right

        # get width from right-left
        peak_width = peak_edge_right - peak_edge_left * dt[peak_inds]

        # filter peaks based on peak_width
        if minpw_s:
            filter = filter & (peak_width > minpw_s)
        if maxpw_s:
            filter = filter & (peak_width < maxpw_s)


    # print 'Wide peaks:\t\t%s' %len(peak_inds)
    # print

    if edge == 'left':
        notnan = ~np.isnan(left_edge_inds)
        idx = left_edge_inds[notnan & filter].astype(int)
    elif edge == 'right':
        notnan = ~np.isnan(right_edge_inds)
        idx = right_edge_inds[notnan & filter].astype(int)
    elif edge == 'peak':
        notnan = ~np.isnan(peak_inds)
        idx = peak_inds[notnan & filter].astype(int)

    return dt[idx], idx


def get_trigger_fcn(trigger_label, direction='right', dir_label=None, dir_delay_s=0,
                    range=None):

    def fcn(rec, **kwargs):
        ttrig, trigger = rec.get_signal(trigger_label)
        dt = np.diff(ttrig).mean()

        if not range is None:
            idx = (trigger >= range[0]) & (trigger < range[1])
        else:
            if dir_label is None:
                dt, idx = get_peak_inds(trigger, dt, direction=direction, **kwargs)
            else:
                dt, idx = get_peak_inds(trigger, dt, direction='right', **kwargs)
                tdir, dir_channel = rec.get_signal(dir_label)
                dir_inds = fc.get_nearest_t_inds(ttrig[idx] + dir_delay_s, tdir)
                dirs = dir_channel[dir_inds] > 0
                dir_dict = {'left': 0, 'right': 1}
                idx = idx[dirs == dir_dict[direction]]

        return ttrig[idx]
    return fcn


def get_triggered_data(group, trigger_fcn, channels,
                             window=(-4, 4), stim=None, **kwargs):

    # buffer the window by 1s:
    window = list(window)
    window[0], window[1] = window[0]-1, window[1]+1
    window = tuple(window)

    # generate baseline indices to be offset by each triggered index
    mintifdt = min(map(lambda fly: fly[0].ims[0].sampling_period, group))
    tifinds = np.arange(window[0]/mintifdt, window[1]/mintifdt+1, 1).astype(int)

    minabfdt = min(map(lambda fly: fly[0].abf.subsampling_period, group))
    abfinds = np.arange(window[0]/minabfdt, window[1]/minabfdt+1, 1).astype(int)


    data = {channel: [] for channel in channels}
    for fly in group:
        for rec in fly:
            ttrigs = trigger_fcn(rec, **kwargs)
            if not stim is None:
                tabf, stimid = rec.get_signal('abf.stimid')
                # trig_idx = [np.where(tabf>ttrig)[0][0] for ttrig in ttrigs]
                abf_idx = fc.get_nearest_t_inds(ttrigs, tabf)
                trig_stimids = stimid[abf_idx]
                inds = (trig_stimids == stim)

                if stim == 1:
                    # Make sure that bar is visible for 'bar_on_after_trigger_s' after trigger
                    bar_on_after_trigger_s = 1
                    tabf, xstim = rec.get_signal('abf.xstim')
                    sr = rec.abf.subsampling_rate
                    clipped = (abf_idx + 1*sr) >= len(xstim)-1
                    abf_idx[clipped] = 0 # this is just so it doesn't crash when it indexes the xstim array, will remove this index after
                    abf_ind0 = np.arange(0, bar_on_after_trigger_s*sr).astype(int)
                    abf_idx = np.tile(abf_ind0, (len(abf_idx), 1)) + abf_idx[:, None]
                    trig_xstims = xstim[abf_idx]
                    bar_visible = ((trig_xstims > -135) & (trig_xstims < 135)).all(axis=1)
                    bar_visible[clipped] = False
                    inds = inds & bar_visible
                ttrigs = ttrigs[inds]

            # Remove peaks that are outside of the 2P scanning time range
            tmax = rec.ims[0].t.max()
            ttrigs = ttrigs[ttrigs+window[0]-1 >= 0]
            ttrigs = ttrigs[ttrigs+window[1]+1 < tmax]

            # Collect data from each channel
            for ichannel, channel in enumerate(channels):
                ts, signal = rec.get_signal(channel)

                for ttrig in ttrigs:
                    if 'abf' in channel:
                        idxi = abfinds + np.where(ts>=ttrig)[0][0]
                    else:
                        idxi = tifinds + np.where(ts>=ttrig)[0][0]

                    within_bounds = (idxi[0] >= 0) and (idxi[-1] < len(ts))
                    if within_bounds:
                        data[channel].append([ts[idxi]-ttrig, signal[idxi]])
                    # else:
                    #     print 'Trigger out of bounds:', ttrig
    try:
        for key in data.keys():
            data[key] = np.array(data[key])
    except:
        pass
    return data


def plot_isolated_turns(groups, tlim=(-2, 2), save=False, overlay=True,
                        trigger='abf.head', dir_label=None,
                         lines=(), show_all=False, dir_delay_s=0,
                         channels=('abf.head', 'no.c1.rmlz'), ylims=(),
                         size=None, lw=1, suffix='', stim=None,
                         colors=None, directions=['left', 'right'], plotvlines=True, **kwargs):
    """
    Plots phase and R-L triggered on a change in phase (using detect peaks on dphase).
    :param group: List of flies.
    :param tlim: Time window.
    :param save: True to save figure.
    :param overlay: Overlay mean and sem of phase and R-L on same plot.
    Otherwise, plot them separately with individual traces.
    :param kwargs: kwargs for get_isolated_peak_inds.s
    :return:
    """

    # Setup figure
    ph.set_fontsize(15)
    if overlay and len(channels)==2:
        if size is None: size = (8, 10)
        plt.figure(1, size)
        gs = gridspec.GridSpec(2, 2)
    else:
        if size is None: size = (10, 10)
        plt.figure(1, size)
        gs = gridspec.GridSpec(len(channels), 2)
    axs = []

    for igroup, group in enumerate(groups):
        for col, direction in enumerate(directions):
            # Find peaks
            trigger_fcn = get_trigger_fcn(trigger, direction=direction, dir_label=dir_label, dir_delay_s=dir_delay_s)
            peak_dict = get_triggered_data(group, trigger_fcn, channels, window=tlim,
                                           stim=stim, **kwargs)

            print '%i %s Turns' %(len(peak_dict.items()[0][1]), direction.capitalize())

            for ichannel, label in enumerate(channels):
                row = ichannel
                peaks = peak_dict[label]
                ax = plt.subplot(gs[row, col])
                axs.append(ax)
                show_axes = []
                if row == len(channels)-1:
                    show_axes.append('bottom')
                if col == 0:
                    show_axes.append('left')
                    ax.set_ylabel(label, labelpad=2)
                ph.adjust_spines(ax, show_axes, lw=.25)
                ax.axhline(ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])
                # ax.axvline(ls='--', c='black', alpha=.5)

                if lines:
                    for line in lines:
                        ax.axvline(line, ls='--', c=ph.grey5, zorder=1, lw=.35, dashes=[2, 2])

                # Plot mean peak
                if 'abf' in label:
                    dt = .02
                    c = 'black'
                else:
                    dt = .1
                    _, imchannel, _ = group[0][0].get_objects(label)
                    celltype = imchannel.celltype
                    c = colours[celltype]

                if colors:
                    if len(groups) == 1:
                        c = colors[ichannel]
                    else:
                        c = colors[igroup]

                t_interp = np.arange(tlim[0]-.5, tlim[1]+.5, dt)
                y_interp = np.zeros((len(peaks), len(t_interp)))
                for i, (t, y) in enumerate(peaks):
                    if label.split('.')[-1] == 'phase':
                        y = fc.unwrap(y) - y[np.argmin(np.abs(t))]
                    f = interp1d(t, y, kind='linear')
                    y_interp[i] = f(t_interp)

                mean = np.nanmean(y_interp, axis=0)
                ax.plot(t_interp, mean, lw=lw, c=c, zorder=2)
                # ax.scatter(t_interp, mean, lw=lw, c='black', zorder=3, s=.2)
                if show_all:
                    for t, y in peaks:
                        ax.plot(t, y, c='grey', alpha=.2, zorder=1)
                else:
                    sem = fc.nansem(y_interp, axis=0)
                    # sem = np.nanstd(y_interp, axis=0)
                    ax.fill_between(t_interp,
                                    mean-sem,
                                    mean+sem,
                                    edgecolor='none',
                                    facecolor=c,
                                    alpha=.4)
                ax.set_xlim(tlim)
                ax.set_xticks(np.arange(tlim[0], tlim[-1]+1))
                if ylims:
                    ax.set_ylim(ylims[row][:2])
                    if len(ylims)==3:
                        ytickstep = ylims[row][2]
                    else:
                        ytickstep = ylims[row][1]

                    ax.set_yticks(np.arange(ylims[row][0], ylims[row][1]+ytickstep/2., ytickstep))
                if plotvlines:
                    ax.axvline(c=ph.grey7, ls='--', alpha=.5)

    if save:
        channel_labels = '_'.join([channel.split('.')[-1] for channel in channels])
        figname = '%s_%s_triggered_mean_%s' %(celltype, trigger.split('.')[-1], channel_labels)
        if overlay:
            figname += '_overlay'
        figname += suffix
        ph.save(figname)



# Plotting time difference


def plot_tif_xcorr(group, metric, tlim=(-2, 2), ylim=None, stim=None, figname=''):
    """
    Plots cross-correlation between two arrays with the same tif time base.
    :param group: List of flies.
    :param tlim: Cross-correlation time-window.
    :param ylim: Y-axis limits.
    :param save: True to save figure.s
    :return:
    """
    plt.figure(1, (5, 5))
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'])
    # ax.set_xticks(np.arange(tlim[0], tlim[1]+.5, .5))
    ax.axvline(c='black', ls='--', alpha=.3)
    ax.set_ylabel('Correlation Coeff')
    ax.set_xlabel('Lag (s)')

    xi = np.arange(tlim[0], tlim[1], .1)
    xcs = np.zeros((len(group), len(xi)))
    xcs[:] = np.nan
    for ifly, fly in enumerate(group):
        flyxc = []
        for rec in fly:
            tif = rec.ims[0]
            if not stim is None:
                stimid = rec.subsample('stimid')
                idx = stimid == stim
            else:
                idx = np.ones(tif.len).astype(bool)
            signal1, signal2 = metric(rec)
            t, xc = fc.xcorr(signal1[idx], signal2[idx], tif.sampling_period)

            c = 'grey'
            idx = (t>tlim[0]-1) & (t<tlim[1]+1)
            ax.plot(t[idx], xc[idx], c=c, alpha=.3)

            f = interp1d(t[idx], xc[idx], 'cubic')
            flyxc.append(f(xi))

        xcs[ifly] = np.nanmean(flyxc, axis=0)
    ax.plot(xi, np.nanmean(xcs, axis=0), c=c, lw=3)
    ax.set_xlim(tlim)
    ax.set_xticks(np.arange(tlim[0], tlim[1]+1, 1))
    if ylim:
        ax.set_ylim(ylim)
    else:
        ylim = ax.get_ylim()

    if figname:
        ph.save(figname)


def plot_rml_dphase_xcorr(groups, alabel='az', tlim=(-2, 2), ylim=None, acceleration=False, figname=''):
    """
    Plot cross correlation between R-L bridge and phase velocity.
    :param groups: List of flies.
    :param alabel: Label of bridge array.
    :param tlim: Time window.
    :param figname: Figure name.
    :return:
    """
    plt.figure(1, (5, 5))
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'])
    # ax.set_xticks(np.arange(tlim[0], tlim[1]+.5, .5))
    ax.axvline(c='black', ls='--', alpha=.3)
    ax.set_ylabel('Correlation Coeff')
    ax.set_xlabel('Lag (s)')
    xi = np.arange(tlim[0], tlim[1], .1)
    for group in groups:
        xcs = np.zeros((len(group), len(xi)))
        xcs[:] = np.nan
        for ifly, fly in enumerate(group):
            flyxc = []
            for rec in fly:
                az = getattr(rec.pb.c1, alabel)

                x = rec.pb.c1.xedges[:-1]
                rml = np.nanmean(az[:, x>=9], axis=1) - np.nanmean(az[:, x<9], axis=1)
                # rml = np.nanmean(az, axis=1)
                if acceleration:
                    dphase = np.gradient(rec.pb.c1.dphase)
                else:
                    dphase = rec.pb.c1.dphase
                t, xc = fc.xcorr(rml, dphase, rec.pb.sampling_period)

                c = colours[rec.pb.c1.celltype]
                idx = (t>tlim[0]-1) & (t<tlim[1]+1)
                ax.plot(t[idx], xc[idx], c=c, alpha=.3)

                f = interp1d(t[idx], xc[idx], 'cubic')
                flyxc.append(f(xi))

            xcs[ifly] = np.nanmean(flyxc, axis=0)
        mean = np.nanmean(xcs, axis=0)
        print '%s Peak delay (s): %.3f' %(rec.pb.c1.celltype, xi[np.argmin(mean)])
        ax.plot(xi, np.nanmean(xcs, axis=0), c=c, lw=3)
    ax.set_xlim(tlim)
    if ylim:
        ax.set_ylim(ylim)
    if figname:
        ph.save(figname)


def plot_rpl_speed_xcorr(groups, alabel='az', tlim=(-2, 2), figname=''):
    """
    Plot cross correlation between R+L bridge and speed.
    :param groups: List of flies.
    :param alabel: Label of bridge array.
    :param tlim: Time window.
    :param figname: Figure name.
    :return:
    """
    plt.figure(1, (5, 5))
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'])
    # ax.set_xticks(np.arange(tlim[0], tlim[1]+.5, .5))
    ax.axvline(c='black', ls='--', alpha=.3)
    ax.set_ylabel('Correlation Coeff')
    ax.set_xlabel('Lag (s)')
    xi = np.arange(tlim[0], tlim[1], .1)
    for group in groups:
        xcs = np.zeros((len(group), len(xi)))
        xcs[:] = np.nan
        for ifly, fly in enumerate(group):
            flyxc = []
            for rec in fly:
                az = getattr(rec.pb.c1, alabel)

                speed = subsample(rec, 'speed', lag_ms=300, metric=np.nanmean)
                rpl = np.nanmean(az, axis=1)
                t, xc = fc.xcorr(rpl, speed, rec.pb.sampling_period)

                c = colours[rec.pb.c1.celltype]
                idx = (t>tlim[0]-1) & (t<tlim[1]+1)
                ax.plot(t[idx], xc[idx], c=c, alpha=.3)

                f = interp1d(t[idx], xc[idx], 'cubic')
                flyxc.append(f(xi))

            xcs[ifly] = np.nanmean(flyxc, axis=0)
        ax.plot(xi, np.nanmean(xcs, axis=0), c=c, lw=3)
    ax.set_xlim(tlim)

    if figname:
        ph.save(figname)


def plot_phase_head_xcorr(groups, tlim=(-2, 2), vel=True, ylim=None, stim=None, figname=''):
    """
    Plot cross correlation between phase and heading position or velocity.
    :param groups: List of flies.
    :param tlim: Time window tuple.
    :param vel: True/False. True to use velocity, otherwise position.
    :param ylim: Scale y-axis.
    :param stim: If vel, only use points under stim.
    :param figname: Figure name.
    :return:
    """
    plt.figure(1, (5, 5))
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'])
    ax.axvline(c='black', ls='--', alpha=.3)
    ax.set_ylabel('Correlation Coeff')
    ax.set_xlabel('Lag (s)')
    colours = {'EIP': ph.blue, 'PEN': ph.orange}
    xi = np.arange(tlim[0], tlim[1], .1)
    if ylim: ax.set_ylim(ylim)

    for group in groups:
        xcs = np.zeros((len(group), len(xi)))
        xcs[:] = np.nan
        for ifly, fly in enumerate(group):
            flyxc = []
            for rec in fly:

                c = colours[rec.pb.c1.celltype]
                dhead = rec.subsample('dhead', lag_ms=0)
                head = rec.subsample('head', lag_ms=0, metric=fc.circmean)
                xstim = rec.subsample('xstim', lag_ms=0, metric=fc.circmean)
                stimid = rec.subsample('stimid', lag_ms=0)
                baridx = (xstim > -135) & (xstim < 135) & (stimid == 1)

                trial_ts = get_trial_ts(rec)
                t = rec.pb.t
                for t0, t1 in trial_ts:
                    tidx = (t >= t0) & (t < t1)
                    if vel:
                        if stim:
                            idx = tidx & (stimid==stim)
                        else:
                            idx = tidx

                        p = rec.pb.c1.dphase[idx]
                        h = dhead[idx]
                    else:
                        idx =  baridx & tidx
                        p = fc.unwrap(rec.pb.c1.phase[idx])
                        h = fc.unwrap(head[idx])


                    if idx.sum()*rec.pb.sampling_period > 30:
                        txc, xc = fc.xcorr(p, h, rec.pb.sampling_period)
                        xcidx = (txc>tlim[0]-1) & (txc<tlim[1]+1)
                        ax.plot(txc[xcidx], xc[xcidx], c=c, alpha=.1)

                        f = interp1d(txc[xcidx], xc[xcidx], 'cubic')
                        flyxc.append(f(xi))

            xcs[ifly] = np.nanmean(flyxc, axis=0)
        mean = np.nanmean(xcs, axis=0)
        print '%s Peak Lag (s): %.3fs' %(rec.pb.c1.celltype, xi[np.argmin(mean)])
        ax.plot(xi, np.nanmean(xcs, axis=0), c=c, lw=3)
    ax.set_xlim(tlim)

    if figname:
        ph.save(figname)


def plot_c1c2_phasediff_dphase1_xcorr(group, save=False, **kwargs):
    """
    Plots cross-correlation between phase difference between c1 and c2 and the phase
    velocity, measured by c1 (usually EIPs).
    :param group: List of flies.
    :param tlim: Cross-correlation time-window.
    :param ylim: Y-axis limits.
    :param save: True to save figure.s
    :return:
    """

    def metric(rec):
        tif = rec.ims[0]
        phase_diff = fc.wrap(tif.c2.phase - tif.c1.phase)
        signal1 = fc.wrap(phase_diff - fc.circmean(phase_diff))
        signal2 = tif.c1.dphase
        return signal1, signal2

    if save:
        tif = rec.ims[0]
        figname = 'Phasediff_dphase_xcorr_%s_%s' %(tif.c1.celltype, tif.c2.celltype)
    else:
        figname = ''

    plot_tif_xcorr(group, metric, figname=figname, **kwargs)


def plot_dphase1_dphase2_xcorr(group, save=True, **kwargs):
    """
    Plots cross-correlation between phase difference between c1 and c2 and the phase
    velocity, measured by c1 (usually EIPs).
    :param group: List of flies.
    :param tlim: Cross-correlation time-window.
    :param ylim: Y-axis limits.
    :param save: True to save figure.s
    :return:
    """
    def metric(rec):
        tif = rec.ims[0]
        return tif.c1.dphase, tif.c2.dphase

    if save:
        tif = group[0][0].ims[0]
        figname = 'dphase1_dphase2_xcorr_%s_%s' %(tif.c1.celltype, tif.c2.celltype)
    else:
        figname = ''

    plot_tif_xcorr(group, metric, figname=figname, **kwargs)


def plot_phase1_phase2_xcorr(group, save=True, **kwargs):
    """
    Plots cross-correlation between phase difference between c1 and c2 and the phase
    velocity, measured by c1 (usually EIPs).
    :param group: List of flies.
    :param tlim: Cross-correlation time-window.
    :param ylim: Y-axis limits.
    :param save: True to save figure.s
    :return:
    """
    def metric(rec):
        tif = rec.ims[0]
        return fc.unwrap(tif.c1.dphase), fc.unwrap(tif.c2.dphase)

    if save:
        tif = group[0][0].ims[0]
        figname = 'phase1_phase2_xcorr_%s_%s' %(tif.c1.celltype, tif.c2.celltype)
    else:
        figname = ''

    plot_tif_xcorr(group, metric, figname=figname, **kwargs)



# Plotting Correlations between phase and bar/ball position/velocity

def plot_scatter_phase_v_xstim(groups, stim_dict, genotype='', save=True, **kwargs):
    labels = stim_dict.values()
    def metric(rec, trial_ind, label):
        im = rec.ims[0]
        xstim = rec.subsample(rec.abf.xstim, lag_ms=300, metric=fc.circmean)
        if 'bar' in label.lower():
            iphase = im.c1.phase[trial_ind]
            ixstim = xstim[trial_ind]
            offset = fc.circmean(fc.wrap(iphase-ixstim))
            data = np.vstack([ixstim, iphase-offset])
        else:
            dxstim = fc.circgrad(xstim)[trial_ind] * im.sampling_rate
            dphase = im.c1.dphasef[trial_ind]
            data = np.vstack([dxstim, dphase])
        return data

    phase_v_xstim = get_trial_data(groups, metric, np.concatenate, stim_dict, **kwargs)

    ################

    plt.figure(1, (28, 9))
    gs = gridspec.GridSpec(len(groups), len(stim_dict)*3, width_ratios=[5, 5, 2]*len(stim_dict), height_ratios=[5]*len(groups))
    gs.update(wspace=.05, hspace=.4)
    color = [ph.blue, ph.orange]
    for igroup in range(len(groups)):
        for ilabel, label in enumerate(labels):
            for itemp, temp in enumerate(['cool', 'hot']):
                ax = plt.subplot(gs[igroup, ilabel*3+itemp])

                if igroup == len(groups)-1 and itemp==0 and (ilabel==0 or ilabel==2):
                    ph.adjust_spines(ax, ['left', 'bottom'])
                    ax.set_ylabel('Phase (deg)')
                    if 'bar' in label.lower():
                        ax.set_xlabel('Bar Position (deg)')
                        ax.set_ylabel('Phase (deg/s)')
                    else:
                        ax.set_xlabel('Turning Velocity (deg/s)')
                        ax.set_ylabel('Phase Velocity (deg/s)')
                else:
                    ph.adjust_spines(ax, [])
                    # ax.set_xticklabels([])


                if 'bar' in label.lower():
                    ax.set_xticks(np.arange(-180, 181, 90))
                    ax.set_yticks(np.arange(-180, 181, 90))
                    ax.set_xlim(-180, 180)
                    ax.set_ylim(-180, 180)

                else:
                    ax.set_xticks(np.arange(-360, 361, 180))
                    ax.set_yticks(np.arange(-360, 361, 180))
                    ax.set_xlim(-360, 360)
                    ax.set_ylim(-360, 360)


                ax.grid()

                data = np.hstack(phase_v_xstim[igroup][temp][label])

                ax.scatter(data[0], data[1], edgecolor=color[itemp], facecolor='none',
                           alpha=.15)

    if save:
        if genotype:
            genotype = '_' + genotype
        ph.save('Scatter_phase_v_xstim' + genotype)


def plot_R_phase_xstim(groups, stim_dict, ylim=(-.5, 1), genotype='', save=True,
                       show_pcnt=True, **kwargs):
    labels = stim_dict.values()
    def metric(rec, trial_ind, label):
        xstim = rec.subsample('xstim', lag_ms=300, metric=fc.circmean)
        im = rec.ims[0]
        if 'bar' in label.lower():
            phase = im.c1.phase[trial_ind]
            ixstim = xstim[trial_ind]
            ri = cs.corrcc(np.deg2rad(ixstim+180), np.deg2rad(phase+180))
        else:
            dxstim = fc.circgrad(xstim)[trial_ind] * im.sampling_rate
            dphase = im.c1.dphasef[trial_ind]
            a, b, r, p, e = stats.linregress(dxstim, dphase)
            ri = r
        return ri

    r = get_trial_data(groups, metric, np.mean, stim_dict, **kwargs)

    ################

    # Setup Figure
    plt.figure(1, (10, 5))
    ph.set_tickdir('out')
    ax = plt.subplot(111)
    ax.axhline(c='black', alpha=.3)
    ph.adjust_spines(ax, ['left', 'bottom'], xlim=[-.5, 3.5])
    plt.ylabel('R', rotation=0, labelpad=20, ha='right')
    xlabels = []
    color = [ph.blue, ph.orange] * len(labels)
    ngroups = len(groups)
    inter_group_dist = .2
    if show_pcnt:
        barwidth = (1-inter_group_dist) / (ngroups)
    else:
        barwidth = (1-inter_group_dist) / (ngroups*2)
    for igroup in range(ngroups):
        for ilabel, label in enumerate(labels):
            rs = []
            for itemp, temp in enumerate(['cool', 'hot']):
                pts = np.array(r[igroup][temp][label])
                rs.append(pts)
                if not show_pcnt:
                    left_edge = ilabel + barwidth*(-1*(ngroups-1) - 1  + igroup*2 + itemp)
                    center_gauss = np.random.normal(left_edge + barwidth / 2, .01, len(pts))
                    plt.scatter(center_gauss, pts, c='none', zorder=2)

                    std = np.nanstd(pts)
                    n = (np.isnan(pts)==False).sum()
                    sem = std / np.sqrt(n)

                    plt.bar(left_edge, np.nanmean(pts), width=barwidth, facecolor=color[itemp], yerr=sem, ecolor='black')
                    xlabels.append('%s\n%s' % (label, temp))
            if show_pcnt:
                rs = np.array(rs)
                ratios = rs[1] / rs[0]
                means = np.nanmean(ratios)
                left_edge = ilabel + barwidth*(-1*(ngroups-1)+1  + igroup)
                center_gauss = np.random.normal(left_edge + barwidth / 2, .01, len(ratios))
                plt.scatter(center_gauss, ratios, c='none', zorder=2)
                plt.bar(left_edge, means, width=barwidth, facecolor=ph.blue, ecolor='black')

    label_pos = np.arange(len(labels))
    plt.xticks(label_pos, labels, ha='center')
    plt.xlim([-.5, len(labels) - .5])
    plt.yticks(np.arange(-1, 1.1, .5))
    if ylim:
        plt.ylim(ylim)
    else:
        plt.ylim([-1, 1])

    if save:
        if genotype:
            genotype = '_' + genotype
        ph.save('R_phase_xstim' + genotype)


def plot_R_v_powerthresh(groups, stim_dict, range=(0, 10, .2), thresh_channel='peak', temp_thresh=25):
    temps = [0, 1]
    stim_keys = stim_dict.keys()
    thresh_range = np.arange(range[0], range[1], range[2])

    plt.figure(1, (10, 10))
    gs = gridspec.GridSpec(len(groups)*3, len(stim_dict), height_ratios=[3, 1.5, 1]*len(groups))

    for igroup, group in enumerate(groups):
        phase = get_trial_data2(group, stim_dict, 'phase')
        xstim = get_trial_data2(group, stim_dict, 'xstim', lag_ms=300)
        power = get_trial_data2(group, stim_dict, thresh_channel)

        # sr = group[0][0].tif.sampling_rate

        for istim in xrange(len(stim_dict)):
            r = np.zeros((len(group), len(temps), len(thresh_range))).astype(float)
            r.fill(np.nan)
            frac_pts = np.zeros_like(r)
            for ifly in xrange(len(group)):
                for itemp in temps:
                    for ithresh, thresh in enumerate(thresh_range):
                        ipower = power[ifly, istim, itemp]
                        mask = ipower > thresh
                        frac_pts[ifly, itemp, ithresh] = float(mask.sum()) / (np.isnan(ipower)==False).sum()

                        ddt = True
                        if 'bar' in stim_dict[istim+1].lower():
                            ris = np.zeros(phase.shape[3])
                            for itrial in xrange(phase.shape[3]):
                                iphase = phase[ifly, istim, itemp, itrial]
                                ixstim = xstim[ifly, istim, itemp, itrial]
                                notnans = np.isnan(iphase)==False
                                bar_visible = (ixstim > -135) & (ixstim < 135)
                                idx = notnans & bar_visible & mask[itrial]
                                iphase = np.deg2rad(iphase+180)[idx]
                                ixstim = np.deg2rad(ixstim+180)[idx]

                                ris[itrial] = cs.corrcc(ixstim, iphase)
                            r[ifly, itemp, ithresh] = np.mean(ris)
                        else:
                            mask = mask.flatten()
                            iphase = phase[ifly, istim, itemp].flatten()
                            ixstim = xstim[ifly, istim, itemp].flatten()
                            inds_set = fc.get_contiguous_inds(mask, min_contig_len=3, keep_ends=True)

                            if inds_set:
                                dphase = np.concatenate([fc.circgrad(iphase[inds]) for inds in inds_set])
                                dxstim = np.concatenate([fc.circgrad(ixstim[inds]) for inds in inds_set])
                                notnans = np.isnan(dphase)==False
                                a, b, ri, p, e = stats.linregress(dxstim[notnans], dphase[notnans])
                                r[ifly, itemp, ithresh] = ri

            rmean = np.nanmean(r, axis=0)
            rstd = np.nanstd(r, axis=0)
            n = (np.isnan(r)==False).astype(int).sum(axis=0)
            rsem = rstd / np.sqrt(n)
            error = rsem

            frac_mean = np.nanmean(frac_pts, axis=0)
            frac_std = np.nanstd(frac_pts, axis=0)

            ax = plt.subplot(gs[igroup*3, istim])
            if igroup==len(groups)-1 and istim==0:
                ph.adjust_spines(ax, ['left', 'bottom'])
                ax.set_ylabel('R')
            else:
                ph.adjust_spines(ax, [])
            colors = [ph.blue, ph.orange]

            ax.plot(thresh_range, rmean[0], c=ph.blue)
            ax.fill_between(thresh_range, rmean[0]-error[0], rmean[0]+error[0], facecolor=ph.blue, edgecolor='none', alpha=.4)
            ax.plot(thresh_range, rmean[1], c=ph.orange)
            ax.fill_between(thresh_range, rmean[1]-error[1], rmean[1]+error[1], facecolor=ph.orange, edgecolor='none', alpha=.4)
            ax.set_ylim(-.505, 1)
            ax.set_xlim(range[0], range[1])

            ax.grid()
            ax.set_yticks(np.arange(-.5, 1.1, .5))
            ax.axhline(0, ls='--', lw=1, alpha=.6, c='black')


            ax2 = plt.subplot(gs[igroup*3+1, istim])
            if igroup==len(groups)-1 and istim==0:
                ph.adjust_spines(ax2, ['left', 'bottom'])
                ax2.set_xlabel('Power threshold')
                ax2.set_ylabel('Fraction of data\nabove threshold', labelpad=15)
            else:
                ph.adjust_spines(ax2, [])
            colors = [ph.blue, ph.orange]
            ax2.plot(thresh_range, frac_mean[0], c=ph.blue)
            ax2.plot(thresh_range, frac_mean[1], c=ph.orange)
            ax2.set_ylim(-.01, 1)
            ax2.set_xlim(range[0], range[1])

            ax2.grid()
            ax2.set_yticks(np.arange(0, 1.1, .5))
            ax2.axhline(0, ls='--', lw=1, alpha=.6, c='black')


def plot_correlations(group, color='black', show_labels=True, nstims=2, figname=''):
    stim_dict = {1: 'Bar', 2: 'Dark'}
    ntrials_per_stim = 3
    rs = np.zeros((nstims, len(group)))
    for ifly, fly in enumerate(group):
        fly_rs = []
        for rec in fly:
            ts = get_trial_ts(rec)
            stimid = rec.subsample('stimid')
            xstim = fc.circ_moving_average(rec.pb.xstim, n=3)
#             xstim = rec.pb.xstim

            rec_rs = np.zeros((nstims, len(ts)/nstims))
            bar_cnt = 0
            dark_cnt = 0
            for i, (t0, t1) in enumerate(ts):
                idx = (rec.pb.t >= t0) & (rec.pb.t < t1)
                istimid = int(stimid[idx].mean().round())
                if istimid==0:
                    print stimid[idx].mean()
                    print rec.name
                if stim_dict[istimid] == 'Bar':
                    idx = idx & (rec.pb.xstim > -135) & (rec.pb.xstim < 135)
                    if idx.sum()*rec.pb.sampling_period < 10: continue
                    ixstim = np.deg2rad(xstim[idx]+180)
                    iphase = np.deg2rad(rec.pb.c1.phase[idx]+180)
                    ri = cs.corrcc(ixstim, iphase)
                    rec_rs[istimid-1, bar_cnt] = ri
                    bar_cnt += 1
                elif stim_dict[istimid] == 'Dark':
                    if idx.sum()*rec.pb.sampling_period < 10: continue
                    idxstim = fc.circgrad(xstim[idx])
                    idphase = rec.pb.c1.dphase[idx]
                    a, b, ri, p, e = stats.linregress(idxstim, idphase)
                    rec_rs[istimid-1, dark_cnt] = ri
                    dark_cnt += 1

            fly_rs.append(rec_rs)
        fly_rs = np.hstack(fly_rs)
        rs[:, ifly] = np.mean(fly_rs, axis=1)

    # Plot periods per stimulus
    plt.figure(1, (6, 6))
    ax = plt.subplot(111)
    stimlabels = ['Bar', 'Dark']
    plt.xticks(range(len(group)), stimlabels)
    plt.xlim(-.5, 2-.5)
    plt.ylim(0, 1)
    plt.yticks([0, 1])
    ph.adjust_spines(ax, ['left', 'bottom'])
    for istim, stimlabel in enumerate(stimlabels):
        x = np.random.normal(istim, .05, len(group))
        med = np.nanmedian(rs[istim])
        std = np.nanstd(rs[istim])
        plt.scatter(x, rs[istim], facecolor='none', edgecolor=color, s=100, lw=2)
        plt.errorbar(istim+.2, med, std, fmt='_', c=color)

    if figname:
        ph.save(figname, rec)


# Plotting the phase-subtracted signal
arr18 = np.arange(1., 19.)
arr16 = np.arange(1., 17.)
x_dict = {'pb': {'PEN': arr18, 'EIP': arr18, 'PFN': arr18, 'PFR': arr18, 'PFLC': arr18},
          'eb': {'PEN': arr16[::2] + .5, 'EIP': arr16, 'PFN': arr16[::2] + .5, 'PFR': arr16,},
          'fb': {'PEN': arr16[::2] + .5, 'EIP': arr16, 'PFN': arr16[::2] + .5, 'PFR': arr16,}}


def plot_phase_subtracted_mean(group, neuropil='pb', proj='preserve', a='an_cubspl',
                               turn_vels=(0,), lag_ms=300, binsize=90, channels=(1,),
                               right_dashed=True, save=False, separate_eip=True,
                               phase_label='c1.phase', c1_ylim=None, c2_ylim=None,
                               c1_yticks=None, c2_yticks=None, offset=0):
    """
    Plot the phase subtracted mean signal across the bridge or ellipsoid body.
    :param group: List of flies.
    :param neuropil: 'pb', 'eb', or 'ebproj'. Type of signal to plot.
    :param proj: 'merge' or 'preserve'. Type of projection to apply to bridge signal
    to arrive at ellipsoid body signal.
    :param turn_vels: Center of turning velocity bin to plot.
    :param lag_ms: Lag to use for binning heading velocity.
    :param binsize: Binsize for turning bins.
    :param channels: tuple, channels to plot.
    :param right_dashed: Draw right bridge with dashed line.
    :param save: Save figure.
    :param separate_eip: If proj = 'preserve', plot left and right EIPs separately,
    with right dashed and left solid linestyle. Otherwise, connect them, alternating
    left, right points with solid and open circles.
    :param phase_channel: Will use the phase from this channel to 'cancel' the phase
    from both channels, if two exist.
    :param c1_ylim: ylim for channel 1.
    :param c2_ylim: ylim for channel 2.
    :param c1_yticks: yticks for channel 1.
    :param c2_yticks: yticks for channel 2.
    :return:
    """

    rcParams['font.size'] = 20
    rcParams['xtick.labelsize'] = 'large'
    rcParams['ytick.labelsize'] = 'large'
    rcParams['axes.labelsize'] = 'large'

    # Draw Figure
    ph.set_tickdir('out')
    if neuropil[:2] == 'eb':
        plt.figure(1, (3.5, 3))
    elif neuropil[:2] == 'pb':
        plt.figure(1, (8, 3))
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'])

    neuropils = copy.copy(neuropil)
    if not type(neuropils) is tuple:
        neuropils = (neuropils,)
    ylims = [c1_ylim, c2_ylim]
    yticks = [c1_yticks, c2_yticks]
    plot_count = 0
    for ineuropil, neuropil in enumerate(neuropils):
        if ineuropil == 1:
            ax = ax.twinx()
            ph.adjust_spines(ax, ['bottom', 'right'])

        imlabel_dict = {'pb':'pb', 'eb':'eb', 'ebproj':'pb', 'fb':'fb'}
        imlabel = imlabel_dict[neuropil]

        # Collect data from group of flies
        turn_bins = np.arange(-360-binsize/2., 360.1+binsize/2., binsize)
        turn_bin_idx = [np.where(turn_bins>=turn_vel)[0][0]-1 for turn_vel in turn_vels]

        r = group[0][0]
        _, _, x = r.get_objects('%s.c1.xedges_interp' %imlabel)
        abs = np.zeros((len(group), len(channels), len(turn_vels), len(x)))
        abs[:] = np.nan
        for ifly, fly in enumerate(group):
            flyab = np.zeros((len(fly), len(channels), len(turn_vels), len(x)))
            flyab[:] = np.nan
            for irec, rec in enumerate(fly):
                for ichannel, channel in enumerate(channels):
                    _, _, phase = rec.get_objects(phase_label)
                    im, ch, an_cubspl = rec.get_objects('%s.c%i.%s' %(imlabel, channel, a))
                    az_nophase = im.cancel_phase(an_cubspl, phase, 10, ch.celltype, offset=offset)
                    ab = rec.bin_gcamp(az_nophase, 'abf.dhead', turn_bins, lag_ms=lag_ms)
                    flyab[irec, ichannel] = ab[turn_bin_idx]
            abs[ifly] = np.nanmean(flyab, axis=0)

        # # Subsample back into glomerular / wedge resolution from cubic spline interpolation
        idx = np.arange(len(x)).reshape((len(x)/10, 10))
        ab_sub = np.nanmean(abs[:, :, :, idx], axis=-1)
        # Compute mean and standard error
        ab_mean = np.nanmean(ab_sub, axis=0)
        n = (np.isnan(ab_sub)==False).sum(axis=0)
        ab_sem = np.nanstd(ab_sub, axis=0) / np.sqrt(n)
        print ab_mean
        if neuropil == 'ebproj':
            eb_mean = []
            eb_sem = []
            for ichannel, channel in enumerate(channels):
                ch = getattr(rec.pb, 'c%i' %channel)
                eb_mean.append(rec.pb.get_ebproj(ab_mean[ichannel], ch.celltype, proj))
                eb_sem.append(rec.pb.get_ebproj(ab_sem[ichannel], ch.celltype, proj))
            ab_mean = eb_mean
            ab_sem = eb_sem
            print eb_mean

        # Now we plot
        rec = group[0][0]
        celltypes = []
        for ic, channel in enumerate(channels):
            if ic == 1:
                ax = ax.twinx()
                ph.adjust_spines(ax, ['right'])
            _, _, celltype = rec.get_objects('%s.c%i.celltype' %(imlabel, channel))
            ax.set_ylabel(celltype + ' ' + neuropil, labelpad=40)
            ax.set_xlabel('glomeruli #', labelpad=40)
            celltypes.append(celltype)
            colour = colours[celltype] if ineuropil == 0 else 'grey'
            for it in range(len(turn_vels)):
                if len(channels)==1 and len(turn_vels)>1 and turn_vels[it] == 0:
                    c = 'grey'
                elif len(channels)==2 and (rec.ims[0].c1.celltype==rec.ims[0].c2.celltype):
                    c = ph.green if ic == 0 else ph.orange
                else:
                    c = colour
                c = ph.purple
                if neuropil=='ebproj' and proj=='preserve':
                    x = x_dict[neuropil[:2]][celltype]
                    if celltype == 'PEN' or celltype == 'PFN':
                        line = ab_mean[ic][:, it]
                        ax.plot(x, line[0], c=c, lw=2)
                        ax.plot(x, line[1], c=c, lw=2, ls='--')
                        sem = ab_sem[ic][:, it]
                        ax.fill_between(x,
                                        line[0] - sem[0],
                                        line[0] + sem[0],
                                        facecolor=c, edgecolor='none', alpha=.3)
                        ax.fill_between(x,
                                        line[1] - sem[1],
                                        line[1] + sem[1],
                                        facecolor=c, edgecolor='none', alpha=.3)
                    elif celltype == 'EIP' or celltype == 'EPG' or celltype == 'PFLC':
                        x = x
                        print x
                        print x[::2]
                        print ab_mean
                        print ab_mean[ic][0, it, ::2]
                        if separate_eip:
                            ax.plot(x[::2], ab_mean[ic][0, it, ::2], c=c,  lw=2)
                            ax.plot(x[::2]+1, ab_mean[ic][1, it, 1::2], c=c, lw=2, ls='--')
                        else:
                            ax.scatter(x[::2], ab_mean[ic][0, it, ::2], c=c,  edgecolor=c, lw=2)
                            ax.scatter(x[::2]+1, ab_mean[ic][1, it, 1::2], c='white', edgecolor=c, lw=2, zorder=3)
                            line = np.nansum(ab_mean[ic][:, it], axis=0)
                            ax.plot(x, line, c=c, lw=1)
                            sem = np.nansum(ab_sem[ic][:, it], axis=0)
                            ax.fill_between(x,
                                            line - sem,
                                            line + sem,
                                            facecolor=c, edgecolor='none', alpha=.3)
                    elif celltype == 'PFR':
                        indexs_left = [0,1,3,5,7,9,11,13]
                        indexs_right = [2,4,6,8,10,12,14,15]
                        ax.plot(x[indexs_left], ab_mean[ic][0, it, indexs_left], c=c, lw=2)
                        ax.plot(x[indexs_right], ab_mean[ic][1, it, indexs_right], c=c, lw=2, ls='--')

                else:
                    x = x_dict[neuropil[:2]]['EIP']
                    if neuropil=='pb' and right_dashed:
                        ax.plot(x[x<9], ab_mean[ic][it, x<9], c=c, lw=2)
                        ax.plot(x[x>=9], ab_mean[ic][it, x>=9], c=c, lw=2, ls='--')
                    else:
                        ax.plot(x, ab_mean[ic][it], c=c, lw=2)

                    ax.fill_between(x, ab_mean[ic][it]-ab_sem[ic][it],
                                    ab_mean[ic][it]+ab_sem[ic][it],
                                    facecolor=c, edgecolor='none', alpha=.3)

                # Y limits
                if ylims[plot_count]:   #if ylims[channel-1]:
                    ylim = ylims[plot_count]
                    ax.set_ylim(ylim)
                else:
                    ylim = list(ax.get_ylim())
                    ylim[0] = 0
                    ax.set_ylim(ylim)
                    # if ylim[1] < 1: ylim[1] = 1
                    # ax.set_ylim(ylim)

                # Yticks
                if yticks[plot_count]:
                    ax.set_yticks(yticks[plot_count])
                else:
                    ax.set_yticks([ylim[0], ylim[1]])
                plot_count += 1

        if neuropil[:2] == 'eb':
            ax.set_xticks(np.arange(1, 17), minor=True)
            ax.set_xticks([1, 16], minor=False)
            ax.set_xlim(1, 16)
        elif neuropil[:2] == 'pb':
            ax.set_xticks(np.arange(1, 19), minor=True)
            plt.xticks([1, 9, 10, 18], ['1', '9', '1', '9'])
            ax.set_xlim(1, 18)
        ax.patch.set_facecolor('white')
        ax.grid(which='major', alpha=0)


    if save:
        celltype_str = '_'.join(['%s' %celltype for celltype in celltypes])
        neuropil_str = '_'.join(list(neuropils))
        if 'proj' in neuropil:
            neuropil_str = neuropil + '_' + proj
            # if proj=='preserve' and separate_eip:
            #     neuropil_str += '_eipsep'
        figname = celltype_str +  '_' + neuropil_str + ''.join(['_%i' %vel for vel in turn_vels])
        ph.save(figname)


def plot_phase_subtracted_mean_multistim(groups, stim_dict, normalize=False, genotype='', save=True, **kwargs):
    """
    Plot the average phase signal for multiple stimuli, cool and hot.
    Used for shibire experiments, where neurons are 'wildtype' (cool) and then inhibited
    (hot).
    :param groups: List of groups of flies (ie. experimental conditions).
    :param stim_dict: Dictionary mapping the stimid to a stimulus label. ie. 1: 'CL Bar'
    :param normalize: Normalize the hot condition to the maximum average response in the
    cool condition. Only plots this normalized hot condition.
    :param genotype: str, genotype label.
    :param save: True to save figure.
    :param kwargs: For get_trial_data.
    :return:
    """
    labels = stim_dict.values()
    def metric(rec, trial_ind, label):
        data = np.nanmean(rec.tif.gcn_nophase[trial_ind], axis=0)
        return data

    phase_cancelled = get_trial_data(groups, metric, np.mean, stim_dict, **kwargs)

    ################

    plt.figure(1, (15, 3))
    gs = gridspec.GridSpec(1, len(stim_dict))
    rec = groups[0][0][0]
    x = rec.tif.xedges_interp
    color = [ph.blue, ph.orange]
    linestyle = ['-', '--', ':']
    for ilabel, label in enumerate(labels):
        ax = plt.subplot(gs[0, ilabel])

        if ilabel == 0:
            ph.adjust_spines(ax, ['left', 'bottom'])
            ax.set_ylabel('$\Delta$F/F')

        else:
            ph.adjust_spines(ax, ['bottom'])
            ax.set_yticks([])
        ax.set_xlabel(label)

        xticks = np.array([0, 8, 10, 17])
        plt.xticks(xticks, (xticks % 9 + .5).astype(int))
        ax.set_xlim(0, 18)
        for igroup in range(len(groups)):
            if normalize:
                ax.axhline(1, ls='--', color='black', alpha=.4)
                ax.set_ylim([0, 1.25])
                if ilabel == 0: ax.set_yticks([0, 1])
                ipcs_cool = np.array(phase_cancelled[igroup]['cool'][label])
                ipc_cool_mean = np.nanmean(ipcs_cool, axis=0)
                ipcs_hot = np.array(phase_cancelled[igroup]['hot'][label])
                ipc_hot_mean = np.nanmean(ipcs_hot, axis=0)
                ipc_cool_mean_peak = np.max(ipc_cool_mean)
                ipc_hot_mean_normd = ipc_hot_mean / ipc_cool_mean_peak
                ax.plot(x, ipc_hot_mean_normd, c=color[1], ls=linestyle[igroup], lw=2)
            else:
                ax.set_ylim([0, 6])
                if ilabel == 0: ax.set_yticks(np.arange(0, 7, 2))
                for itemp, temp in enumerate(['cool', 'hot']):
                    ipcs = np.array(phase_cancelled[igroup][temp][label])
                    ipc_mean = np.nanmean(ipcs, axis=0)
                    ipc_sem = stats.sem(ipcs, axis=0)
                    ax.plot(x, ipc_mean, c=color[itemp], ls=linestyle[igroup], lw=2)
            #         ax.fill_between(x, ipc_mean - ipc_sem, ipc_mean + ipc_sem, facecolor=color, edgecolor='none', alpha=.2)

    if save:
        if genotype:
            genotype = '_' + genotype
        ph.save('Phase_subtracted_mean_multistim' + genotype)


def plot_period_v_dhead(group, binsize=60, lag_ms=400, ylim=None, save=True):
    dhead_bins = np.arange(-360-.5*binsize, 360+1.5*binsize, binsize)
    period_binned = np.zeros((len(group), len(dhead_bins)-1))
    for ifly, fly in enumerate(group):
        flyperiod = []
        flydhead = []
        for rec in fly:
            power, period, phase = fc.powerspec(rec.pb.c1.acn)
            gate = (period > 6) & (period < 8.5)
            peak_period = period[gate][np.argmax(power[:, gate], axis=1)]
            dhead = subsample(rec, 'dhead', lag_ms=lag_ms)
            flyperiod.append(peak_period)
            flydhead.append(dhead)
        flyperiod = np.concatenate(flyperiod)
        flydhead = np.concatenate(flydhead)
        period_binned[ifly] = fc.binyfromx(flydhead, flyperiod, dhead_bins)

    bin_centers = dhead_bins[:-1] + np.diff(dhead_bins)[0]/2.

    med = np.nanmedian(period_binned, axis=0)
    n = (np.isnan(period_binned)==False).sum(axis=0)
    sem = np.nanstd(period_binned, axis=0) / np.sqrt(n)
    c = colours[rec.pb.c1.celltype]

    plt.figure(1, (5, 5))
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'])
    ax.plot(bin_centers, med, c=c, lw=2)
    ax.fill_between(bin_centers, med-sem, med+sem, facecolor=c, edgecolor='none', alpha=.3)
    if ylim:
        ax.set_ylim(ylim)
    if save:
        celltype = rec.pb.c1.celltype
        figname = '%s_period_v_dhead_%ims_lag' %(celltype, lag_ms)
        ph.save(figname)



# Plotting phase difference between two channels

def plot_c1c2_phasediff(group, save=True):
    phase_diff_means = []
    phase_diff_stds = []
    for fly in group:
        iphase_diffs = []
        for rec in fly:
            tif = rec.ims[0]
            phase_diff = fc.wrap(tif.c1.phase - tif.c2.phase)
            iphase_diffs.append( phase_diff )

        iphase_diffs = np.hstack(iphase_diffs)
        phase_diff_means.append( fc.circmean(iphase_diffs) )
        phase_diff_stds.append( fc.circstd(iphase_diffs) )

    phase_diff_means = np.array(phase_diff_means)
    phase_diff_stds = np.array(phase_diff_stds)

    plt.figure(1, (5, 5))
    ax = plt.subplot(111)
    ph.adjust_spines(ax, ['left', 'bottom'])
    nflies = len(group)
    x = np.arange(nflies) + 1
    phase_diff_means = fc.wrap(phase_diff_means, cmin=-360, cmax=0)
    ax.scatter(x, phase_diff_means, marker='o', c='black')
    ax.errorbar(x, phase_diff_means, phase_diff_stds, c='black', linestyle='none')
    ax.set_xlim(.5, nflies + .5)
    ax.set_xticks(np.arange(len(group))+1, minor=True)
    ax.set_xticks([1, len(group)], minor=False)
    ax.set_ylim(-360, 0)
    ax.set_yticks(np.arange(-360, 1, 90))
    ax.set_xlabel('Fly')
    ax.set_ylabel('Phase offset, PEN - EIP (deg)')

    print phase_diff_means.mean()

    if save:
        ph.save('PEN-EIP_phase_diff')



# Analysis for phase/bar slope vs position - now believe due to bleaching

def plot_slope_bardensity(self, channel='green', phase='phase', tbinsize=30, xbinsize=15,
                          tlim=None, slope_lim=None, forw_lim=None, idx_offset=-1,
                          minspeed=.2, x_range=[-45, 45], period=360, alpha=1, lines=[],
                          save=True, sf=1):
    ch = getattr(self, channel)
    t = self.tif.t
    if not tlim: tlim = [t[0], t[-1]]
    colours = [ph.red, ph.purple, 'black']
    t = self.abf.t
    tbins = np.arange(t[0], t[-1], tbinsize)

    # Setup Figure
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    plt.figure(1, (sf * 9, sf * 7))
    plt.suptitle(self.folder, size=10)
    gs = gridspec.GridSpec(4, 1)
    axs = []

    # Plot slope
    t = self.tif.t
    xstimb = self.abf.xstimb
    phase = getattr(ch, phase)

    if not minspeed is None:
        speedb = self.subsample(np.abs(self.abf.dforw))
        walking = speedb > minspeed
    else:
        walking = np.ones(len(t)).astype(bool)

    in_x_range = (xstimb >= x_range[0]) & (xstimb < x_range[1])
    walking_and_in_x_range = walking & in_x_range

    inds = [(t >= tbins[i]) & (t < tbins[i + 1]) & walking_and_in_x_range for i in range(len(tbins) - 1)]

    slope = np.zeros(len(inds))
    std_err = np.zeros(len(inds))
    for i, ind in enumerate(inds):
        idx = np.where(ind)[0]
        ixstimb = xstimb[idx + idx_offset]
        if ixstimb.size:
            iphase, _ = fc.circ2lin(ixstimb, phase[idx]) if ixstimb.size else np.nan
            islope, intercept, r_value, p_value, istd_err = stats.linregress(ixstimb, iphase)
            slope[i] = islope
            std_err[i] = istd_err
        else:
            slope[i] = np.nan
            std_err[i] = np.nan
    axslope = plt.subplot(gs[0, 0])
    axs.append(axslope)
    ph.adjust_spines(axslope, ['left'], xlim=tlim, yticks=np.arange(-1, 4, .5))
    axslope.set_ylabel('Slope (phi/theta)', labelpad=10, rotation='horizontal', ha='right')
    if slope_lim:
        axslope.set_ylim(slope_lim)
    tb = tbins[:-1] + tbinsize / 2.
    notnan = np.isnan(slope) == False
    axslope.plot(tb[notnan], slope[notnan], c=ph.bgreen, lw=2, zorder=2)
    # axslope.scatter(tb[notnan], slope[notnan], c='none', zorder=4)
    axslope.errorbar(tb[notnan], slope[notnan], yerr=std_err[notnan], c='black', fmt='o', lw=1, zorder=3)
    # axslope.set_ylim(tlim)
    # axslope.invert_yaxis()

    # Plot bar density
    t = self.abf.t
    xstim = self.abf.xstim
    if not minspeed is None:
        speed = np.abs(self.abf.dforw)
        walking = speed > minspeed
    else:
        walking = np.ones(len(t)).astype(bool)
    inds = [(t >= tbins[i]) & (t < tbins[i + 1]) & walking for i in range(len(tbins) - 1)]
    xbins = np.arange(-180, 181, xbinsize)
    bardensity = np.zeros((len(inds), len(xbins) - 1))
    for i, ind in enumerate(inds):
        bardensity[i, :], _ = np.histogram(xstim[ind], bins=xbins, density=True)
    axdensity = plt.subplot(gs[1, 0])
    axs.append(axdensity)
    ph.adjust_spines(axdensity, ['left'], ylim=[-180, 180], yticks=[-180, 0, 180], xlim=tlim)
    axdensity.pcolormesh(tbins, xbins, bardensity.T, cmap=plt.cm.Blues)
    axdensity.set_ylabel('Bar density', labelpad=10, rotation='horizontal', ha='right')
    # axdensity.invert_yaxis()

    # Plot bar position
    axstim = plt.subplot(gs[2, 0])
    axs.append(axstim)
    axstim.invert_yaxis()
    ph.adjust_spines(axstim, ['left'], ylim=[-180, 180], yticks=[-180, 0, 180], xlim=tlim)
    axstim.set_ylabel('Bar Position (deg)', labelpad=10, rotation='horizontal', ha='right')
    axstim.fill_between(self.abf.t, y1=-180, y2=-self.abf.arena_edge, color='black', alpha=0.1, zorder=10)
    axstim.fill_between(self.abf.t, y1=self.abf.arena_edge, y2=180, color='black', alpha=0.1, zorder=10)
    axstim.axvline(0, c='black', ls='--', alpha=0.4)
    ph.circplot(self.abf.t, self.abf.xstim, circ='y', c='black', alpha=.8)
    # ph.circplot(xstimb, t, c='black', alpha=.8)
    # Plot phase
    # ph.circplot(phase, t, c=ph.nblue, alpha=1, lw=1)
    # axstim.invert_yaxis()

    # Plot forward rotation
    axforw = plt.subplot(gs[3, 0])
    axs.append(axforw)
    ph.adjust_spines(axforw, ['bottom', 'left'], xlim=tlim)
    axforw.set_ylabel('Forward rotation (deg)', labelpad=10, rotation='horizontal', ha='right')
    axforw.axvline(0, c='black', ls='--', alpha=0.4)
    forwint = self.abf.forwuw
    axforw.plot(self.abf.t, forwint, c='black', alpha=.8)
    if forw_lim:
        axforw.set_ylim(forw_lim)
        axforw.set_yticks(forw_lim)
    axdforw = axforw.twinx()
    axdforw.plot(self.abf.t, self.abf.dforw, c='black', alpha=.6)
    ph.adjust_spines(axdforw, ['bottom', 'right'], xlim=tlim, ylim=[-10, 50])

    if lines:
        for line in lines:
            for ax in axs:
                ax.axvline(line, alpha=.6, c=ph.nblue, ls='--')

    if save:
        suffix = ''
        if tlim: suffix += '_%is-%is' % (tlim[0], tlim[1])
        plt.savefig('%s_phase_v_xpos_%s.png' % (self.abf.basename, suffix), bbox_inches='tight')


def plot_slope_v_xposdensity(recs, save=True, sf=1, method='polyreg', suffix=''):
    # Setup Figure
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    plt.figure(1, (sf * 7, sf * 7))
    # plt.suptitle(self.folder, size=10)
    ax = plt.subplot(111)

    # Collect data
    def get_phase_xpos_slope_xposdensity(self, channel='green', phase='phase', tlim=None, idx_offset=-1, binsize=10,
                                         minspeed=None, method=method, period=360):
        ch = getattr(self, channel)
        if not tlim:
            tlim = [ch.t[0], ch.t[-1]]
        abft = self.abf.t
        abfidx = (abft >= tlim[0]) & (abft < tlim[-1])
        tift = self.tif.t
        tifidx = (tift >= tlim[0]) & (tift < tlim[-1])
        if not minspeed is None:
            speed = np.abs(self.abf.dforw)
            abfidx2 = abfidx & (speed > minspeed)
            abfidx = abfidx2

            speedb = self.subsample(speed)
            tifidx2 = tifidx & (speedb > minspeed)
            tifidx = tifidx2

        # Get histogram of xposition
        xbins = np.arange(-180, 181, binsize)
        xb = xbins[:-1] + binsize / 2  # Bin centers
        t = self.abf.t
        xstim = copy.copy(self.abf.xstim)
        xstim = xstim[abfidx]
        h, _ = np.histogram(xstim, bins=xbins, density=True)
        hinterpf = interp1d(fc.tile(xb), np.tile(h, 3), 'linear')

        # Get slope of phase vs binned xposition
        # Linearize phase, find best fit diagonal (ie. find b in y=x+b)
        xstimb = self.abf.xstimb
        phases = getattr(ch, phase)
        phases = fc.wrap(phases)
        t = ch.t
        idx = np.where(tifidx)[0]
        xstimb = xstimb[idx + idx_offset]
        phases = phases[idx]
        phases, b = fc.circ2lin(xstimb, phases)

        if method == 'smooth':
            x = [-180, 180]
            y = x + b

            # Convert xstim, phase coordinates into distance from diagonal
            def xy2diag(xs, ys, b):
                ts = np.zeros_like(xs)
                ds = np.zeros_like(xs)
                for i, (x0, y0) in enumerate(zip(xs, ys)):
                    # (x1, y1) = closest point on diagonal
                    x1 = (x0 + y0 - b) / 2.
                    y1 = x1 + b
                    d = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
                    if y0 < y1: d *= -1
                    ds[i] = d
                    t = np.sqrt(x1 ** 2 + (y1 - b) ** 2)
                    if x1 < 0: t *= -1
                    ts[i] = t
                return ts, ds

            ts, ds = xy2diag(xstimb, phases, b)

            # Bin values along diagonal
            diaglen = np.sqrt(2 * (360 ** 2))
            tbins = np.arange(-diaglen / 2., diaglen / 2. + 1, binsize)
            tb = tbins[:-1] + binsize / 2.
            db = bh.fc.binyfromx(ts, ds, tbins, np.median)
            idx = np.isnan(db) == False
            db = db[idx]
            tb = tb[idx]

            def diag2xy(t, d, b):
                a = np.sqrt((t ** 2) / 2.)
                a[t < 0] *= -1
                e = np.sqrt((d ** 2) / 2.)
                ex = copy.copy(e)
                ex[d > 0] *= -1
                ey = copy.copy(e)
                ey[d < 0] *= -1
                x = a + ex
                y = a + b + ey
                return x, y

            step = 5
            # Convert binned distances from diagonal to x, y coordinates, take slope
            x, y = diag2xy(tb, db, b)
            ytile = fc.tile(y[::step])
            xtile = fc.tile(x[::step])
            dy = np.gradient(ytile, np.gradient(xtile))
            # Find density of xposition at points defined by binning along the diagonal
            xposdensity = hinterpf(x[::step])
            slope = dy[len(y[::step]):-len(y[::step])]
        elif method == 'polyreg':
            p = np.poly1d(np.polyfit(fc.tile(xstimb, period), fc.tile(phases, period), 25))
            x = np.arange(-180, 181)
            dy = np.gradient(p(x), np.gradient(x))
            slope = np.array([dy[x == xbin] for xbin in xb]).ravel()
            xposdensity = h
        return xposdensity, slope

    # Collect slopes and densities for all recordings
    slopes = []
    xposdensities = []
    for rec in recs:
        tlim = None
        period = 360
        if rec.folder == '2015_03_27-004/': tlim = [100, 240]
        if rec.folder == '2015_03_27-008/': period = 270
        if rec.folder == '2015_04_14-003/': tlim = [0, 180]
        xposdensity, slope = get_phase_xpos_slope_xposdensity(rec, tlim=tlim, minspeed=.1, period=period)
        slopes.append(slope)
        xposdensities.append(xposdensity)
    xposdensities = np.concatenate(xposdensities)
    slopes = np.concatenate(slopes)

    # Plot data
    print 'npoints:', len(slopes)
    ax.set_ylabel('Slope\n(phase deg/bar deg)', rotation='horizontal', labelpad=80, size=14)
    ax.set_xlabel('Bar Position Density', labelpad=20, size=14)
    ax.scatter(xposdensities, slopes, c='none', lw=1, edgecolor='black')
    xlim = [xposdensities.min() - .001, xposdensities.max() + .001]
    ylim = [slopes.min() - .5, slopes.max() + .5]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.grid()

    # Fit data to line
    slope, intercept, r_value, p_value, std_err = stats.linregress(xposdensities, slopes)
    x = np.arange(-1, 2)
    y = slope * x + intercept
    plt.plot(x, y, c='red')
    text = 'R = %.3f\nR2 = %.3f\np = %.3e' % (r_value, r_value ** 2, p_value)
    plt.text(.006, 0, text, fontsize=12)

    if save:
        if len(recs) == 1: prefix += recs[0].abf.basename
        plt.savefig('slope_v_xposdensity%s.png' % suffix, bbox_inches='tight')

def plot_intensity_v_xstim(self, tlim=None, tlims=None, metric=np.sum, save=True):
    colors = [ph.nblue, red]

    bridge_intensity = metric(self.tif.gcn, axis=1)
    theta = self.abf.xstimb
    print theta.min(), theta.max()
    dforwb = self.subsample(self.abf.dforw)

    t = self.tif.t
    if tlims is None: tlims = [tlim]
    if tlims[0] is None: tlims[0] = [t[0], t[-1]]

    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    plt.figure(1, (8, 6))
    plt.suptitle(self.abf.basename.split(os.path.sep)[-1])
    ax = plt.subplot(111)
    plt.xlim(-180, 180)
    plt.xticks(np.arange(-180, 181, 45))
    plt.xlabel('Bar position (deg)')
    plt.ylabel('Intensity (au)')
    xbinsize = 20
    xbins = np.arange(-180, 181, xbinsize)
    xb = xbins[:-1] + xbinsize / 2.
    for i, tlim in enumerate(tlims):
        idx = (t >= tlim[0]) & (t < tlim[1])
        # idx = idx & (dforwb>.000001)
        c = colors[i]
        plt.scatter(theta[idx], bridge_intensity[idx], c='none', edgecolor=colors[i], s=20, alpha=.4)
        sum_mean = bh.fc.binyfromx(theta[idx], bridge_intensity[idx], xbins, np.mean, nmin=4)
        sum_std = bh.fc.binyfromx(theta[idx], bridge_intensity[idx], xbins, stats.sem, nmin=4)
        plt.errorbar(xb, sum_mean, yerr=sum_std, linestyle="None", c=c, zorder=2)
        plt.scatter(xb, sum_mean, s=30, c=c, zorder=3, lw=0, label='%i-%is' % (tlim[0], tlim[1]))
        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles, labels, loc='upper right')

    plt.grid()

    if save:
        suffix = 'max' if metric is np.max else 'summed'
        plt.savefig('%s_%i-%is_%s_intensity_v_theta.png' % (self.abf.basename, tlim[0], tlim[1], suffix),
                    bbox_inches='tight')

def plot_test_sca(data1, data2):
    """
    test, Cheng
    """
    ax = plt.subplot(111)
    ax.scatter(data1, data2)

def plot_test(self, x=0):
    ax = plt.subplot(111)
    if x is None:
        ax.plot(x, self)
    else:
        ax.plot(self)

def plot_offset_sca(self, color='black', nstims=2, mode='mean', figname=''):
    """
    Plots offsets under two modes. Mode 'mean' plots the circular mean and circular standard
    deviation for each fly. Mode 'all' plots the circular mean for each trial in each fly.
    :param group: List of fly objects (each a list of recs).
    :param color: Color to plot.
    :param nstims: Number of stimuli in each recording.
    :param mode: 'mean' or 'all'. See above.
    :param figname: Figure name to save.
    :return:
    """
    stim_dict = {1: 'Bar', 2: 'Dark'}

    fly_offsets = []
    rec = self
    ts = get_trial_ts(rec)
    stimid = rec.subsample('stimid')
    xstim = fc.circ_moving_average(rec.pb.xstim, n=3)
    phase = rec.pb.c1.phase
    for i, (t0, t1) in enumerate(ts):
        idx = (rec.pb.t >= t0) & (rec.pb.t < t1)
        istimid = int(stimid[idx].mean().round())
        if istimid==0:
            print stimid[idx].mean()
            print rec.name
        if stim_dict[istimid] == 'Bar':
            idx = idx & (rec.pb.xstim > -100) & (rec.pb.xstim < 100)
            if idx.sum()*rec.pb.sampling_period < 10: continue
            fly_offsets.append(phase[idx] - xstim[idx])
            temp_max =fly_offsets[-1].max()
            temp_min =fly_offsets[-1].min()
            for item2 in range(0, len(fly_offsets[-1])):
                if fly_offsets[-1][item2] < (temp_max+temp_min)/2:
                    fly_offsets[-1][item2] += 360

    ax = plt.subplot(111)
    for i in range (0,len(fly_offsets)):
        ax.plot(fly_offsets[i])
    #plt.ylim([67.08, 67.1])

    if figname:
        ph.save(figname, rec)

    return fly_offsets

def plot_phase_peak(group):
    stim_dict = {1: 'Bar', 2: 'Dark'}
    mean_bar = np.zeros(len(group))
    std_bar = np.zeros(len(group))
    mean_dark = np.zeros(len(group))
    std_dark = np.zeros(len(group))
    mean_all = np.zeros(len(group))
    std_all = np.zeros(len(group))
    for ifly, fly in enumerate(group):
        peak_cleaned_bar = []
        peak_cleaned_dark = []
        peak_cleaned_all = []
        for rec in fly:
            ts = get_trial_ts(rec)
            stimid = rec.subsample('stimid')
            xstim = fc.circ_moving_average(rec.pb.xstim, n=3)
            _, peak_period, _ = fc.powerspec_peak(rec.pb.c1.acn, mode='peak')
            for i, (t0, t1) in enumerate(ts):
                idx = (rec.pb.t >= t0) & (rec.pb.t < t1)
                istimid = int(stimid[idx].mean().round())
                if stim_dict[istimid] == 'Bar':
                    peak_cleaned_bar = np.hstack((peak_cleaned_bar, [x for x in peak_period[idx] if x > 0]))
                if stim_dict[istimid] == 'Dark':
                    peak_cleaned_dark = np.hstack((peak_cleaned_dark, [x for x in peak_period[idx] if x > 0]))
        peak_cleaned_all = np.hstack((peak_cleaned_bar, peak_cleaned_dark))
        mean_bar[ifly] = np.mean(peak_cleaned_bar)
        std_bar[ifly] = np.std(peak_cleaned_bar)
        mean_dark[ifly] = np.mean(peak_cleaned_dark)
        std_dark[ifly] = np.std(peak_cleaned_dark)
        mean_all[ifly] = np.mean(peak_cleaned_all)
        std_all[ifly] = np.std(peak_cleaned_all)

    ax = plt.subplot(111)
    plt.errorbar(np.arange(len(group))+1, mean_bar, std_bar)
    plt.errorbar(np.arange(len(group))+1, mean_dark, std_dark)
    plt.errorbar(np.arange(len(group))+1, mean_all, std_all)
    plt.legend(['bar', 'dark', 'all'])
    ax.set_xlabel('fly number')
    ax.set_ylabel('period (glomeruli number')
    ax.set_title('peak of period from DFT: %1.3f' %np.mean(mean_all))

def plot_bleach_cali(group, area=[50, 100]):
    total_fluo = [[], [], [], [], [], []]
    stim_dict = {1: 'Bar', 2: 'Dark'}
    for ifly, fly in enumerate(group):
        for irec, rec in enumerate(fly):
            ts = get_trial_ts(rec)
            stimid = rec.subsample('stimid')
            for i, (t0, t1) in enumerate(ts):
                idx = (rec.pb.t >= t0) & (rec.pb.t < t1)
                istimid = int(stimid[idx].mean().round())
                if stim_dict[istimid] == 'Bar':
                    temp = rec.pb.c1.acn[idx]
                    fluo = []
                    for index_area in range(area[0], area[1]):
                        fluo.append(np.mean(temp[index_area]))
                    total_fluo[irec*3 + i/2].append(np.mean(fluo))
    gs = gridspec.GridSpec(1, 2)
    ax = plt.subplot(gs[0, 0])
    sns.stripplot(data = total_fluo, jitter=True)
    sns.pointplot(data = total_fluo)
    ax = plt.subplot(gs[0, 1])
    plt.errorbar(np.arange(6)+1, np.mean(total_fluo, axis=-1), np.std(total_fluo, axis=-1))
    ax.set_xlim([0.5, 6.5])
    ax.set_xlabel('trial squence number')
    ax.set_ylabel('ave fluo over bridge')
    ax.set_title('bleach effect')

def plot_offset_dist(group, maxvel=30, erosion_ite=2):
    stim_dict = {1: 'Bar', 2: 'Dark'}
    offsets = []
    for ifly, fly in enumerate(group):
        for rec in fly:
            ts = get_trial_ts(rec)
            stimid = rec.subsample('stimid')
            xstim = fc.circ_moving_average(rec.pb.xstim, n=3)
            phase = rec.pb.c1.phase
            dhead = rec.subsample('dhead')
            for i, (t0, t1) in enumerate(ts):
                idx = (rec.pb.t >= t0) & (rec.pb.t < t1)
                istimid = int(stimid[idx].mean().round())
                if stim_dict[istimid] == 'Bar':
                    iidx = idx & (abs(xstim) < 135) & (abs(dhead) < maxvel)
                    if (iidx.sum()*rec.pb.sampling_period) < 10:
                        continue
                    if erosion_ite != 0:
                        eidx = ndimage.binary_erosion(iidx, iterations=erosion_ite)
                    else:
                           eidx = iidx
                    if not math.isnan(fc.circmean(phase[eidx] - xstim[eidx])):
                        offsets.append(fc.circmean(phase[eidx] - xstim[eidx]))
    binwidth = 5
    bins = np.arange(-180, 180, binwidth)
    ax = plt.subplot(111)
    plt.hist(offsets, bins=bins)

def plot_offset_cali(group, channel_num=3, erosion_ite=0, maxvel=30):
    stim_dict = {1: 'Bar', 2: 'Dark'}
    plot_range = []
    offsets = [[]]
    for i in range(0, channel_num):
        plot_range.append([270./channel_num*i-135, 270./channel_num*(i+1)-135])
        if i != 0:
            offsets.append([])

    for ifly, fly in enumerate(group):
        for rec in fly:
            ts = get_trial_ts(rec)
            stimid = rec.subsample('stimid')
            xstim = fc.circ_moving_average(rec.pb.xstim, n=3)
            phase = rec.pb.c1.phase
            dhead = rec.subsample('dhead')
            for i, (t0, t1) in enumerate(ts):
                idx = (rec.pb.t >= t0) & (rec.pb.t < t1)
                istimid = int(stimid[idx].mean().round())
                if stim_dict[istimid] == 'Bar':
                    for m in range(0, channel_num):
                        iidx = idx & (rec.pb.xstim > plot_range[m][0]) & (rec.pb.xstim < plot_range[m][1]) & (abs(dhead) < maxvel)
                        if (iidx.sum()*rec.pb.sampling_period) < 10:
                            continue
                        if erosion_ite != 0:
                            eidx = ndimage.binary_erosion(iidx, iterations=erosion_ite)
                        else:
                            eidx = iidx
                        if not math.isnan(fc.circstd(phase[eidx] - xstim[eidx])):
                            offsets[m].append(fc.circstd(phase[eidx] - xstim[eidx]))
    mean_plot = np.zeros(channel_num)
    std_plot = np.zeros(channel_num)
    for i in range(0, channel_num):
        mean_plot[i] = np.mean(offsets[i])
        std_plot[i] = np.std(offsets[i])


    ax = plt.subplot(111)
    plt.errorbar(np.arange(channel_num)+1, mean_plot, std_plot)
    plt.xlim([0, channel_num+1])
    ax.set_xlabel('xstim area')
    ax.set_ylabel('std in each area')
    ax.set_title('distribution of offsets in each xstim area')

def plot_offset_effectfromdark(group, erosion_ite=0, maxvel=30, repeat_of_each_fly=2):
    stim_dict = {1: 'Bar', 2: 'Dark'}
    offsets = [[]]
    for i in range(0, len(group)*2):
        if i != 0:
            offsets.append([])

    count = 0
    for ifly, fly in enumerate(group):
        for rec in fly:
            ts = get_trial_ts(rec)
            stimid = rec.subsample('stimid')
            xstim = fc.circ_moving_average(rec.pb.xstim, n=3)
            phase = rec.pb.c1.phase
            dhead = rec.subsample('dhead')
            for i, (t0, t1) in enumerate(ts):
                idx = (rec.pb.t >= t0) & (rec.pb.t < t1) & (abs(dhead) < maxvel)
                istimid = int(stimid[idx].mean().round())
                if stim_dict[istimid] == 'Bar':
                    if idx.sum()*rec.pb.sampling_period < 10: continue
                    if erosion_ite != 0:
                        eidx = ndimage.binary_erosion(idx, iterations=erosion_ite)
                    else:
                        eidx = idx
                    if not math.isnan(fc.circstd(phase[eidx] - xstim[eidx])):
                        offsets[count].append(fc.circmean(phase[eidx] - xstim[eidx]))
                        offsets[count].append(fc.circstd(phase[eidx] - xstim[eidx]))
            count += 1

    ax = plt.subplot(111)
    for i in range(0, len(offsets)):
        if len(offsets[i]) == 6:
            offsets[i][0::2] -= offsets[i][0]
            plt.errorbar(np.arange(3)+1, offsets[i][0::2], offsets[i][1::2])
    plt.xlim([0.5, 3.5])
    ax.set_xlabel('CL sequence')
    ax.set_ylabel('offsets in each CL trial')

def plot_fluo_vs_xstimarea(group, erosion_ite_o=2, erosion_ite_f=2, maxvel_o=30, maxvel_f=20 ,phasearea=[0, 22.5], offset_std_lim=20, fig=1):
    stim_dict = {1: 'Bar', 2: 'Dark'}
    fluo = [[]]
    for i in range(0, 15):
        fluo.append([])

    for ifly, fly in enumerate(group):
        for rec in fly:
            ts = get_trial_ts(rec)
            stimid = rec.subsample('stimid')
            xstim = fc.circ_moving_average(rec.pb.xstim, n=3)
            phase = rec.pb.c1.phase
            dhead = rec.subsample('dhead')

            # first, get offset; each CL trial has its own offset
            # then, fluo
            # iidx, eidx is used for estimating offset; idx is used for fluo
            for i, (t0, t1) in enumerate(ts):
                idx = (rec.pb.t >= t0) & (rec.pb.t < t1) & (abs(dhead) < maxvel_f) & (abs(xstim) < 135)
                iidx = (rec.pb.t >= t0) & (rec.pb.t < t1) & (abs(dhead) < maxvel_o)
                istimid = int(stimid[idx].mean().round())
                offset = 10000
                if stim_dict[istimid] == 'Bar':

                    # estimating offset
                    if idx.sum()*rec.pb.sampling_period < 10: continue
                    if erosion_ite_o != 0:
                        eidx = ndimage.binary_erosion(iidx, iterations=erosion_ite_o)
                    else:
                        eidx = iidx
                    if not math.isnan(fc.circstd(phase[eidx] - xstim[eidx])):
                        if fc.circstd(phase[eidx] - xstim[eidx]) < offset_std_lim:
                            offset = fc.circmean(phase[eidx] - xstim[eidx])

                    # take fluo data
                    if offset != 10000:
                        xstim_rescale = fc.circplus(xstim, offset)
                        idx = idx & (xstim_rescale >= phasearea[0]) & (xstim_rescale < phasearea[1])
                        if erosion_ite_f != 0:
                            idx = np.where(ndimage.binary_erosion(idx, iterations=erosion_ite_f))[0]
                        if idx.sum() > 0:
                            for time_slice in idx:
                                for glo_index in range(0, 16):
                                    fluo[glo_index].append(rec.pb.c1.acn[time_slice][glo_index])

    mean_plot = np.zeros(16)
    std_plot = np.zeros(16)
    for glo_index in range(0, 16):
        mean_plot[glo_index] = np.mean(fluo[glo_index])
        std_plot[glo_index] = np.std(fluo[glo_index])
    if fig == 1:
        ax = plt.subplot(111)
        plt.errorbar(np.arange(16)+1, mean_plot, std_plot)
        plt.xlim([0.5, 16.5])
        ax.set_xlabel('glomerulus number')
        ax.set_ylabel('fluorescence intensity (F/F0)')
        print 'number of data for ploting is: [%2.1f, %2.1f]--%i' %(phasearea[0], phasearea[1], len(fluo[0]))
    return mean_plot, len(fluo[0])

def plot_xstim_fluo_phase(group, x_interval=5):
    if type(x_interval) is not int or 360 % x_interval != 0:
        print 'wrong input phase interval'
        return np.nan
    phase_point = np.zeros(360/x_interval)
    phase_area = []
    plot_x = []
    plot_y = []
    for i in range(0, len(phase_point)):
        phase_area.append([])
        phase_area[i].append(x_interval*i)
        phase_area[i].append(x_interval*(i+1))
    phase_area = np.array(phase_area)
    phase_area = phase_area - 180

    temp =[[], []]
    for i in range(0, len(phase_area)):
        glo_fluo, data_size = plot_fluo_vs_xstimarea(group, fig=0, phasearea=phase_area[i])
        temp[0] = glo_fluo
        temp[1] = glo_fluo
        temp = np.array(temp)
        if data_size > 0:
            phase_value, _, _ = fc.powerspec_peak(temp, mode='constant', cval=8.435)
            plot_x.append(np.mean(phase_area[i]))
            plot_y.append(-phase_value[0])

    ax = plt.subplot(111)
    ax.plot(plot_x, plot_y, marker='o')
    ax.set_xlabel('phase by def')
    ax.set_ylabel('phase from glo mean intensity')

def plot_phase_running(group, minspeed=4, minvel=0, stim_type='Dark', binwidth=5, cl=1, fig=1):
    _, phase = get_group_data(group, 'pb.c1.phase')
    phase = fc.wrap(phase)
    _, dhead = get_group_data(group, 'pb.dhead')
    _, speed = get_group_data(group, 'pb.speed')
    dt, stimid = get_group_data(group, 'abf.stimid', subsample=True)
    if cl == 0:
        stim_dict = {'Bar': 1, 'Dark': 2}
    else:
        stim_dict = {'Bar': 0, 'Dark': 1}
    stimID = stim_dict[stim_type]
    idx = ((abs(dhead) > minvel) | (speed > minspeed)) & (stimid == stimID)
    point_in_running = phase[idx]
    if fig == 1:
        bins = np.arange(-180, 180, binwidth)
        plt.figure(1, (20, 5))
        gs = gridspec.GridSpec(1, 1)

        ax = plt.subplot(gs[0, 0])
        ax.hist(point_in_running, bins=bins)
        ax.set_xlabel('phase in running')
        ax.set_ylabel('count')
        ax.set_xlim(-180, 180)
        ax.set_xticks(np.arange(-180, 181, 90))

def plot_ss_cali(group, maxvel=15, maxspeed=4, minvel=0, wait_s=0, minlen_s=1, stim_type='Dark', thr_std=10, binwidth=5, cl=0, fig=1):
    _, phase = get_group_data(group, 'pb.c1.phase')
    phase = fc.wrap(phase)
    _, dhead = get_group_data(group, 'pb.dhead')
    _, speed = get_group_data(group, 'pb.speed')
    dt, stimid = get_group_data(group, 'abf.stimid', subsample=True)
    if cl == 0:
        stim_dict = {'Bar': 1, 'Dark': 2}
    else:
        stim_dict = {'Bar': 0, 'Dark': 1}
    stimID = stim_dict[stim_type]
    idx = (abs(dhead) < maxvel) & (abs(dhead) > minvel) & (speed < maxspeed) & (stimid == stimID)
    wait_idx = int(np.ceil(wait_s / .3))
    minlen_idx = int(np.ceil(minlen_s / .3))
    point_sample =[]
    print 'idx_raw.sum() = %i' %idx.sum()
    inds = fc.get_contiguous_inds(idx, trim_left=wait_idx, keep_ends=True, min_contig_len=minlen_idx)#keep_ends=True

    std_in_stand = []
    mean_in_stand = []
    median_in_stand = []
    lastpoint_in_stand = []
    points = []
    for ind in inds:
        std_in_stand.append(fc.circstd(phase[ind]))
        if fc.circstd(phase[ind]) <= thr_std:
            mean_in_stand.append(fc.circmean(phase[ind]))
            lastpoint_in_stand.append(phase[ind[-1]+1])
            for index in ind:
                points.append(phase[index])

    print 'raw standing incidence number: %i' %len(std_in_stand)
    print 'standing incidence number: %i' %len(mean_in_stand)


    if fig == 1:
        rcParams['font.size'] = 15
        rcParams['xtick.labelsize'] = 'large'
        rcParams['ytick.labelsize'] = 'large'
        rcParams['axes.labelsize'] = 'large'
        bins = np.arange(-180, 180, binwidth)
        plt.figure(1, (20, 5))

        gs = gridspec.GridSpec(1, 2)
        ax = plt.subplot(gs[0, 0])
        ax.hist(std_in_stand, bins=np.arange(0, 70, 2), alpha=0.5)
        ax.set_xlabel('std in individual standing event')
        ax.set_ylabel('count')

        ax = plt.subplot(gs[0, 1])
        ax.hist(mean_in_stand, bins=bins)
        ax.set_xlabel('mean phase in each event')
        ax.set_ylabel('count')
        ax.set_xlim(-180, 180)
        ax.set_xticks(np.arange(-180, 225, 45))
    if fig == 2:
        mean_oneglo = np.array(mean_in_stand) % 45
        bins = np.arange(0, 45, binwidth)
        plt.figure(1, (20, 5))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])
        ax.hist(mean_oneglo, bins=bins)
        ax.set_xlabel('ini phase')
        ax.set_ylabel('count')
        ax.set_xlim(-2, 47)
        ax.set_xticks(np.arange(0, 45, 22.5))

    if fig == 3:
        bins = np.arange(-180, 180, binwidth)
        plt.figure(1, (20, 5))
        gs = gridspec.GridSpec(1, 1)

        ax = plt.subplot(gs[0, 0])
        ax.hist(points, bins=bins)
        ax.set_xlabel('phase')
        ax.set_ylabel('count')
        ax.set_xlim(-180, 180)
        ax.set_xticks(np.arange(-180, 180, 45))

    return mean_in_stand, lastpoint_in_stand, std_in_stand

def plot_ss_cali_ini_end(group, maxvel=15, maxspeed=4, minvel=0, glofluo_thresh=0.6, wait_s=0, minlen_s=1, stim_type='Dark', thr_std=10, binwidth=5, cl=0, fig=1):
    _, phase = get_group_data(group, 'pb.c1.phase')
    phase = fc.wrap(phase)
    _, dhead = get_group_data(group, 'pb.dhead')
    _, speed = get_group_data(group, 'pb.speed')
    _, glofluo = get_group_data(group, 'pb.c1.glo_peak_fluo')
    dt, stimid = get_group_data(group, 'abf.stimid', subsample=True)
    if cl == 0:
        stim_dict = {'Bar': 1, 'Dark': 2}
    else:
        stim_dict = {'Bar': 0, 'Dark': 1}
    stimID = stim_dict[stim_type]
    idx = (abs(dhead) < maxvel) & (abs(dhead) > minvel) & (speed < maxspeed) & (stimid == stimID) & (glofluo < glofluo_thresh)
    #idx = (stimid == stimID) & (glofluo < glofluo_thresh)
    wait_idx = int(np.ceil(wait_s / .19))
    minlen_idx = int(np.ceil(minlen_s / .19))
    print 'idx_raw.sum() = %i' %idx.sum()
    inds = fc.get_contiguous_inds(idx, trim_left=wait_idx, keep_ends=True, min_contig_len=minlen_idx)#keep_ends=True

    points = []
    ini_points = []
    end_points = []
    for ind in inds:
        ini_points.append(phase[(ind[0]-2):(ind[0])].mean())
        end_points.append(phase[(ind[-1]+1):(ind[-1]+3)].mean())
        for index in ind:
            points.append(phase[index])

    if fig == 1:
        bins = np.arange(-180, 180, binwidth)
        plt.figure(1, (20, 5))
        gs = gridspec.GridSpec(1, 3)

        ax = plt.subplot(gs[0, 0])
        ax.hist(ini_points, bins=bins)
        ax.set_xlabel('ini phase')
        ax.set_ylabel('count')
        ax.set_xlim(-180, 180)
        ax.set_xticks(np.arange(-180, 181, 90))

        ax = plt.subplot(gs[0, 1])
        ax.hist(end_points, bins=bins)
        ax.set_xlabel('end phase')
        ax.set_ylabel('count')
        ax.set_xlim(-180, 180)
        ax.set_xticks(np.arange(-180, 181, 90))

        ax = plt.subplot(gs[0, 2])
        ax.hist(np.array(end_points) - np.array(ini_points), bins=np.arange(-70, 70, 1))
        ax.set_xlabel('Delta phase')
        ax.set_ylabel('count')
        ax.set_xlim(-70, 70)

    if fig == 2:
        ini_oneglo = np.array(ini_points) % 45
        end_oneglo = np.array(end_points) % 45
        bins = np.arange(0, 45, binwidth)
        plt.figure(1, (20, 5))
        gs = gridspec.GridSpec(1, 2)
        ax = plt.subplot(gs[0, 0])
        ax.hist(ini_oneglo, bins=bins)
        ax.set_xlabel('ini phase')
        ax.set_ylabel('count')
        ax.set_xlim(-2, 47)
        ax.set_xticks(np.arange(0, 45, 22.5))

        ax = plt.subplot(gs[0, 1])
        ax.hist(end_oneglo, bins=bins)
        ax.set_xlabel('end phase')
        ax.set_ylabel('count')
        ax.set_xlim(-2, 47)
        ax.set_xticks(np.arange(0, 45, 22.5))

    if fig == 3:
        bins = np.arange(-180, 180, binwidth)
        plt.figure(1, (20, 5))
        gs = gridspec.GridSpec(1, 1)

        ax = plt.subplot(gs[0, 0])
        ax.hist(points, bins=bins)
        ax.set_xlabel('phase')
        ax.set_ylabel('count')
        ax.set_xlim(-180, 180)
        ax.set_xticks(np.arange(-180, 180, 45))

    return ini_points, end_points

def plot_ss_cali_running(group, maxvel=1000, maxspeed=1000, minvel=3.5, minspeed=15, wait_s=1, minlen_s=2, stim_type='Dark', binwidth=5, cl=1, fig=1):
    _, phase = get_group_data(group, 'pb.c1.phase')
    phase = fc.wrap(phase)
    _, dhead = get_group_data(group, 'pb.dhead')
    _, speed = get_group_data(group, 'pb.speed')
    _, glofluo = get_group_data(group, 'pb.c1.glo_peak_fluo')
    dt, stimid = get_group_data(group, 'abf.stimid', subsample=True)
    if cl == 0:
        stim_dict = {'Bar': 1, 'Dark': 2}
    else:
        stim_dict = {'Bar': 0, 'Dark': 1}
    stimID = stim_dict[stim_type]
    idx = ((abs(dhead) > minvel) | (speed > minspeed)) & (stimid == stimID)
    wait_idx = int(np.ceil(wait_s / .19))
    minlen_idx = int(np.ceil(minlen_s / .19))
    print 'idx_raw.sum() = %i' %idx.sum()
    inds = fc.get_contiguous_inds(idx, trim_left=wait_idx, keep_ends=True, min_contig_len=minlen_idx)#keep_ends=True

    points = []
    for ind in inds:
        for index in ind:
            points.append(phase[index])

    if fig == 1:
        rcParams['font.size'] = 15
        rcParams['xtick.labelsize'] = 'large'
        rcParams['ytick.labelsize'] = 'large'
        rcParams['axes.labelsize'] = 'large'
        bins = np.arange(-180, 180, binwidth)
        plt.figure(1, (20, 5))
        gs = gridspec.GridSpec(1, 1)

        ax = plt.subplot(gs[0, 0])
        ax.hist(points, bins=bins)
        ax.set_xlabel('phase')
        ax.set_ylabel('count')
        ax.set_xlim(-180, 180)
        ax.set_xticks(np.arange(-180, 225, 45))

    return points

def plot_ss_cali_fluoave(group, maxvel=15, maxspeed=3.5, minvel=0, wait_s=1, minlen_s=2, stim_type='Dark', thr_std=10, binwidth=5, cl=0, fig=1):
    _, phase = get_group_data(group, 'pb.c1.phase')
    glomfluo = get_group_data(group, 'pb.c1.acn')
    phase = fc.wrap(phase)
    _, dhead = get_group_data(group, 'pb.dhead')
    _, speed = get_group_data(group, 'pb.speed')
    dt, stimid = get_group_data(group, 'abf.stimid', subsample=True)
    if cl == 0:
        stim_dict = {'Bar': 1, 'Dark': 2}
    else:
        stim_dict = {'Bar': 0, 'Dark': 1}
    stimID = stim_dict[stim_type]
    idx = (abs(dhead) < maxvel) & (abs(dhead) > minvel) & (speed < maxspeed) & (stimid == stimID)
    wait_idx = int(np.ceil(wait_s / .19))
    minlen_idx = int(np.ceil(minlen_s / .19))
    print 'idx_raw.sum() = %i' %idx.sum()
    inds = fc.get_contiguous_inds(idx, trim_left=wait_idx, keep_ends=True, min_contig_len=minlen_idx)#keep_ends=True

    std_in_stand = []
    mean_in_stand = []
    lastpoint_in_stand = []
    points = []
    phases_from_fluoave = []
    for ind in inds:
        std_in_stand.append(fc.circstd(phase[ind]))
        if fc.circstd(phase[ind]) <= thr_std:
            mean_in_stand.append(fc.circmean(phase[ind]))
            lastpoint_in_stand.append(phase[ind[-1]+1])
            glomfluo_ave = glomfluo[:, ind].mean(axis=-1)
            glomfluo_ave = np.array([glomfluo_ave, glomfluo_ave])
            for index in ind:
                points.append(phase[index])
            phase_from_fluoave, _, _ = fc.powerspec_peak(glomfluo_ave, mode='constant', cval=8.435)
            phases_from_fluoave.append(-phase_from_fluoave[0])

    print 'raw standing incidence number: %i' %len(std_in_stand)
    print 'standing incidence number: %i' %len(mean_in_stand)

    if fig == 1:
        bins = np.arange(-180, 180, binwidth)
        plt.figure(1, (20, 5))
        gs = gridspec.GridSpec(1, 1)

        ax = plt.subplot(gs[0, 0])
        ax.hist(phases_from_fluoave, bins=bins)
        ax.set_xlabel('phase')
        ax.set_ylabel('count')
        ax.set_xlim(-180, 180)
        ax.set_xticks(np.arange(-180, 180, 45))

def plot_fluo_dist(group, phasearea=[0, 10], maxvel=60, minvel=0, erosion_ite=2, stim_type='Bar', cl=1, fig=1, ploterr=1):
    if cl == 1:
        stim_dict = {0: 'Bar', 1: 'Dark'}
    else:
        stim_dict = {1: 'Bar', 2: 'Dark'}
    fluo = [[]]
    for i in range(0, 15):
        fluo.append([])

    for ifly, fly in enumerate(group):
        for rec in fly:
            ts = get_trial_ts(rec)
            stimid = rec.subsample('stimid')
            xstim = fc.circ_moving_average(rec.pb.xstim, n=3)
            phase = rec.pb.c1.phase
            dhead = rec.subsample('dhead')
            speed = rec.subsample('speed')

            for i, (t0, t1) in enumerate(ts):
                idx = (rec.pb.t >= t0) & (rec.pb.t < t1) & (abs(speed) < maxvel) & (abs(speed) > minvel)
                istimid = int(stimid[idx].mean().round())
                if (istimid < 0) or (istimid > 1): continue
                if stim_dict[istimid] == stim_type:
                    if stim_type == 'Bar':
                        idx = idx & (abs(xstim) < 135)
                    idx = idx & (phase >= phasearea[0]) & (phase < phasearea[1])
                    if erosion_ite != 0:
                        idx = np.where(ndimage.binary_erosion(idx, iterations=erosion_ite))[0]
                        #idx = ndimage.binary_erosion(idx, iterations=erosion_ite)
                    else:
                        idx = np.where(idx)[0]
                    if idx.sum() > 0:
                        for time_slice in idx:
                            #if time_slice == True:
                            for glo_index in range(0, 16):
                                fluo[glo_index].append(rec.pb.c1.acn[time_slice][glo_index])

    mean_plot = np.zeros(16)
    std_plot = np.zeros(16)
    for glo_index in range(0, 16):
        mean_plot[glo_index] = np.mean(fluo[glo_index])
        std_plot[glo_index] = np.std(fluo[glo_index])
    if fig == 1:
        ax = plt.subplot(111)
        if ploterr == 1:
            plt.errorbar(np.arange(16)+1, mean_plot, std_plot)
        else:
            plt.plot(np.arange(16)+1, mean_plot)
        plt.xlim([0.5, 16.5])
        ax.set_xlabel('glomerulus number')
        ax.set_ylabel('fluorescence intensity (F - F0/(FM - F0)')
        #print 'number of data for ploting is: [%2.1f, %2.1f]--%i' %(phasearea[0], phasearea[1], len(fluo[0]))
    #return mean_plot, len(fluo[0])

def plot_subglom_corrcoef(r_mat, scale=1, save=0):
    rcParams['font.size'] = 15
    rcParams['xtick.labelsize'] = 'large'
    rcParams['ytick.labelsize'] = 'large'
    rcParams['axes.labelsize'] = 'large'
    
    num = r_mat.shape[0]
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])
    if scale:
        plt.pcolor(r_mat, cmap=plt.cm.Reds, vmin=0, vmax=1)
    else:
        plt.pcolor(r_mat, cmap=plt.cm.Reds)
    plt.colorbar()
    ax.set_xticks(np.arange(0.5, num + 0.5, 1))
    ax.set_yticks(np.arange(0.5, num + 0.5, 1))
    ax.set_xticklabels(np.arange(1, num + 1))
    ax.set_yticklabels(np.arange(1, num + 1))
    ax.set_xlabel('sub glomeruli')
    ax.set_ylabel('sub glomeruli')

    if save:
        figname = 'corr between subgloms'
        ph.save(figname, exts=['png'])
        #plt.savefig(figname, bbox_inches='tight')

def plot_period_phase_rela(group, minvel=0, minspeed=15, wait_s=0, minlen_s=2, stim_type='Bar', cl=0, fig=1):
    _, phase = get_group_data(group, 'pb.c1.phase')
    phase = fc.wrap(phase)
    _, dhead = get_group_data(group, 'pb.dhead')
    _, speed = get_group_data(group, 'pb.speed')
    glomfluo = get_group_data(group, 'pb.c1.acn')
    dt, stimid = get_group_data(group, 'abf.stimid', subsample=True)
    if cl == 0:
        stim_dict = {'Bar': 1, 'Dark': 2}
    else:
        stim_dict = {'Bar': 0, 'Dark': 1}
    stimID = stim_dict[stim_type]
    idx = ((abs(dhead) > minvel) | (speed > minspeed)) & (stimid == stimID)
    wait_idx = int(np.ceil(wait_s / .3))
    minlen_idx = int(np.ceil(minlen_s / .3))
    inds = fc.get_contiguous_inds(idx, trim_left=wait_idx, keep_ends=True, min_contig_len=minlen_idx)

    phase_final = []
    period_final = []
    for ind in inds:
        peak_phase, peak_period, peak_power = fc.powerspec_peak(glomfluo[:, ind].T, mode='max')
        for index in ind:
            phase_final.append(phase[index])
            period_final.append(peak_period[index-ind[0]])

    if fig == 1:
        rcParams['font.size'] = 15
        rcParams['xtick.labelsize'] = 'large'
        rcParams['ytick.labelsize'] = 'large'
        rcParams['axes.labelsize'] = 'large'
        ax = plt.subplot(111)
        ax.scatter(phase_final, period_final, s=10, c='blue', alpha=0.1)
        ax.set_xlabel('phase')
        ax.set_ylabel('period of two peaks')

def plot_offsets_frommss(group, fig=0, spenum=0):
    offsets_mean = []
    offsets_std = []
    for fly in group:
        offsets_mean.append([])
        offsets_std.append([])
        for r in fly:
            idx = (r.stimid == 0) & (r.xstim < 135) & (r.xstim > -135)
            offsets_mean[-1].append(fc.circmean(r.xstim[idx] - r.phase[idx]))
            offsets_std[-1].append(fc.circstd(r.xstim[idx] - r.phase[idx]))

    if fig == 1:
        ax = plt.subplot(111)
        ax.errorbar(np.arange(len(offsets_mean[spenum]))+1, offsets_mean[spenum], yerr=offsets_std[spenum], fmt='o')
        ax.set_xlim(0.8, len(offsets_mean[spenum]) + 0.2)

    if fig == 9:
        plt.figure(1, (50, 50))
        gs = gridspec.GridSpec(7, 7)
        for count in np.arange(0, 49):
            irow = count / 7
            icol = count % 7
            ax = plt.subplot(gs[irow, icol])
            ax.errorbar(np.arange(len(offsets_mean[count]))+1, offsets_mean[count], yerr=offsets_std[count], fmt='o')
            ax.set_xlim(0.8, len(offsets_mean[spenum]) + 0.2)

    return offsets_mean, offsets_std

def plot_scatter_dhead_dphase_onefly(rec, stimid=1, fig=1):
    # fig=1: scatter
    # fig=2: cross correlation
    ts = get_trial_ts(rec)
    dphase = []
    dhead = []
    for trial_repeat_index, (t0, t1) in enumerate(ts):
        idx = (rec.pb.t >= t0) & (rec.pb.t < t1)
        istimid = int(rec.subsample('stimid')[idx].mean().round())
        if istimid == stimid:
            dphase.append(rec.pb.c1.dphase[idx])
            dhead.append(rec.subsample('dhead')[idx])
    dphase = np.array(dphase)
    dhead = np.array(dhead)
    dt = rec.pb.sampling_period

    if fig == 1:
        rcParams['font.size'] = 20
        rcParams['xtick.labelsize'] = 'large'
        rcParams['ytick.labelsize'] = 'large'
        rcParams['axes.labelsize'] = 'large'
        plt.figure(1, (8, 5))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

        dphase.flatten()
        dhead.flatten()
        ax.scatter(dphase, dhead, facecolor='none', alpha=.15)

    if fig == 2:
        rcParams['font.size'] = 20
        rcParams['xtick.labelsize'] = 'large'
        rcParams['ytick.labelsize'] = 'large'
        rcParams['axes.labelsize'] = 'large'
        plt.figure(1, (8, 5))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])
        colorlist = {0:'red', 1:'blue', 2:'green', 3:'orange'}
        for index in range(len(dhead)):
            length = len(dhead[index])
            x = (np.arange(0, length) - length/2) * dt
            a = (dhead[index] - np.mean(dhead[index])) / (np.std(dhead[index]) * length)
            v = (dphase[index] - np.mean(dphase[index])) / np.std(dphase[index])
            ax.plot(x, np.correlate(a, v, 'same'), c=colorlist[index], lw=1)
            ax.set_xlim([-5, 5])

def plot_whole_trajectory(rec):
    x, y = rec.abf.get_trajectory()
    bj_xs = x[:-1][np.abs(np.diff(rec.abf.xstim_unwrap)) > 50]
    bj_ys = y[:-1][np.abs(np.diff(rec.abf.xstim_unwrap)) > 50]

    rcParams['font.size'] = 20
    rcParams['xtick.labelsize'] = 'large'
    rcParams['ytick.labelsize'] = 'large'
    rcParams['axes.labelsize'] = 'large'
    plt.figure(1, (9, 8))
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])

    ax.plot(x, y)
    ax.plot(x[0], y[0], marker='o', markersize=8, markeredgecolor='grey', markeredgewidth=2, markerfacecolor='none')
    ax.plot(x[-1], y[-1], marker='o', markersize=8, markeredgecolor='grey', markeredgewidth=2, markerfacecolor='grey')
    ax.plot(bj_xs, bj_ys, ls='none', marker='o', markersize=4, markeredgecolor='none', markeredgewidth=2, markerfacecolor='red')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_xticks(np.arange(np.floor(np.min(x)/100)*100, np.ceil(np.max(x)/100)*100, 100))
    ax.set_yticks(np.arange(np.floor(np.min(y)/100)*100, np.ceil(np.max(y)/100)*100, 100))
    ax.set_xticks(np.arange(np.floor(np.min(x)/100)*100, np.ceil(np.max(x)/100)*100, 25), minor=True)
    ax.set_yticks(np.arange(np.floor(np.min(y)/100)*100, np.ceil(np.max(y)/100)*100, 25), minor=True)
    ax.grid(which='minor', alpha=.5)
    ax.grid(which='major', alpha=1)

def plot_frame_unit(rec, ax=False, fig_size=(2,1), ch=1, i_frame=0, sigma=1, vm=[0,1], cm=plt.cm.Oranges,
               scalebar_micron=20, scalebar_lower=.9, scalebar_lefter=.3):
    if not ax:
        _ = plt.figure(1, fig_size)
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])

    _, apl = rec.get_signal('c%i.apl' % ch)
    img = apl[i_frame]
    if sigma:
        img = gaussian_filter(img, sigma=sigma)

    ax.pcolormesh(img, cmap=cm, vmin=vm[0], vmax=vm[1])
    ax.set_aspect('equal', 'box')
    xlim = [np.where(np.max(img, axis=0)>0)[0][0], np.where(np.max(img, axis=0)>0)[0][-1]]
    ylim = [np.where(np.max(img, axis=1) > 0)[0][0], np.where(np.max(img, axis=1) > 0)[0][-1]]
    ph.adjust_spines(ax, [], lw=.5, xlim=xlim, ylim=ylim, xticks=[], yticks=[])

    bar_length = scalebar_micron / rec.fm.micronsPerPixel
    ph.plot_x_scale_text(ax, bar_length=bar_length, text='', y_text_lower=scalebar_lower, x_text_lefter=scalebar_lefter)

def plot_frame(recs, fig_size=[3,3], wr=[], chs=[1,1], i_frames=[0,], sigma=1, vm=[0,1], cms=[plt.cm.Oranges,],
               scalebar_micron=20, sb_lower_list=[.9], sb_lefter_list=[.3],):
    # col: different location
    # row: different frames
    ncol = len(recs)
    nrow = len(i_frames)
    _ = plt.figure(1, fig_size)
    if len(wr):
        gs = gridspec.GridSpec(nrow, ncol, width_ratios=wr)
    else:
        gs = gridspec.GridSpec(nrow, ncol)

    for row in range(nrow):
        for col in range(ncol):
            ax = plt.subplot(gs[row, col])
            plot_frame_unit(recs[col], ax=ax, ch=chs[col], i_frame=i_frames[row], sigma=sigma, vm=vm, cm=cms[col],
                            scalebar_micron=scalebar_micron, scalebar_lower=sb_lower_list[col], scalebar_lefter=sb_lefter_list[col])

def plot_phase_subtracted_bolusshape_ave(group, axs=[], fig_size=(3,3), neuropil='fb', offset=180, channel=1, normalization='an',
                flight=False, exclude_standing=True, plot_standing=False, filters_com=[['dhead', [-100,100]],], filter_bin=['dforw', [0,10]], binnum=10,
                color=ph.blue, rainbow_color=True, colormap='viridis', show_sem=True, norm_curve=False, plotcosfit=True, fit_rectify=False, offset_fit=0,
                xs=np.arange(16)+1, xlim=[.5,16.5], ylim=[0, 4], y_barlength=1, lw=1.5, alpha_fill=0.1, fs=10, ylabel='', label_axeratio=-.1, print_data=False):

    # three types of filter: exclude_standing, filters_com, and filter_bin
    def filter_name_2_sig(filter_name):
        if filter_name == 'dhead':
            return dhead
        elif filter_name == 'dforw':
            return dforw
        elif filter_name == 'speed':
            return speed

    location_dict = {'pb':'pb', 'fb':'fb', 'ebproj':'pb', 'eb':'eb'}
    location = location_dict[neuropil]
    speed_standing = 1
    dhead_standing = 25


    # extracting data
    r = group[0][0]  # get the x axis for ploting
    _, _, x = r.get_objects('%s.c%i.xedges_interp' % (location, channel))
    _, _, celltype = r.get_objects('%s.c%i.celltype' % (location, channel))
    sigs = np.zeros((binnum, len(group), len(x)))
    sigs[:] = np.nan
    sigs_stand = np.zeros((len(group), len(x)))
    sigs_stand[:] = np.nan

    for ifly, fly in enumerate(group):
        sig_fly = np.zeros((binnum, len(fly), len(x)))
        sig_fly[:] = np.nan
        sig_fly_stand = np.zeros((len(fly), len(x)))
        sig_fly_stand[:] = np.nan

        for irec, rec in enumerate(fly):
            _, _, phase = rec.get_objects('%s.c%i.phase' % (location, channel))
            # _, _, phase = rec.get_objects('%s.c%i.phase' % (location, 2))
            im, ch, an_cubspl = rec.get_objects('%s.c%i.%s_cubspl' % (location, channel, normalization))
            a_nophase = im.cancel_phase(an_cubspl, phase, 10, ch.celltype, offset=offset)

            # apply three filters
            if flight:
                inds_flight = td.get_2p_ind_flight(rec)
                sig_fly[0][irec] = a_nophase[inds_flight].mean(axis=0)
            else:
                dhead = rec.subsample('dhead')
                speed = rec.subsample('speed')
                dforw = rec.subsample('dforw')
                inds = speed < 1e3
                if exclude_standing:
                    inds = inds & ((speed > speed_standing) | (np.abs(dhead) > dhead_standing))
                for filter_ in filters_com:
                    filter_name, filter_range = filter_
                    sig_filter = filter_name_2_sig(filter_name)
                    inds = inds & (sig_filter > filter_range[0]) & (sig_filter < filter_range[1])

                if binnum == 1:     # no additional filter, just one curve
                    sig_fly[0][irec] = a_nophase[inds].mean(axis=0)
                else:               # multiple bin filters
                    filter_name, filter_range = filter_bin
                    sig_filter = filter_name_2_sig(filter_name)
                    bin_edges = np.linspace(filter_range[0], filter_range[1], binnum + 1)
                    for ibin in range(binnum):
                        inds_bin = inds & (sig_filter >= bin_edges[ibin]) & (sig_filter < bin_edges[ibin+1])
                        sig_fly[ibin][irec] = a_nophase[inds_bin].mean(axis=0)

                if plot_standing:
                    inds = inds & ((speed < speed_standing) | (np.abs(dhead) < dhead_standing))
                    sig_fly_stand[irec] = a_nophase[inds].mean(axis=0)

        for ibin in range(binnum):
            sigs[ibin][ifly] = np.nanmean(sig_fly[ibin], axis=0)
            sigs_stand[ifly] = np.nanmean(sig_fly_stand, axis=0)

    # Subsample, mean, sem
    idx = np.arange(len(x)).reshape((len(x) / 10, 10))
    sigs_sub = np.nanmean(sigs[:, :, idx], axis=-1)
    sigs_mean = np.nanmean(sigs_sub, axis=1)
    n = (np.isnan(sigs_sub) == False).sum(axis=1)
    sigs_sem = np.nanstd(sigs_sub, axis=1) / np.sqrt(n)
    if plot_standing:
        idx = np.arange(len(x)).reshape((len(x) / 10, 10))
        sigs_sub_stand = np.nanmean(sigs_stand[:, idx], axis=-1)
        sigs_mean_stand = np.nanmean(sigs_sub_stand, axis=0)
        n = (np.isnan(sigs_sub_stand) == False).sum(axis=0)
        sigs_sem_stand = np.nanstd(sigs_sub_stand, axis=0) / np.sqrt(n)

    if neuropil == 'ebproj':
        eb_mean = []
        eb_sem = []
        for ibin in range(binnum):
            eb_mean.append(rec.pb.get_ebproj(sigs_mean[ibin], celltype, 'merge'))
            eb_sem.append(rec.pb.get_ebproj(sigs_sem[ibin], celltype, 'merge'))
        sigs_mean = np.array(eb_mean)
        sigs_sem = np.array(eb_sem)

    if norm_curve:
        norm_mean = []
        norm_sem = []
        for ibin in range(binnum):
            y_ = sigs_mean[ibin]
            y0, y1 = np.nanmin(y_), np.nanmax(y_)
            f_norm = 1. / (y1 - y0)
            norm_mean.append((y_ - y0) * f_norm)
            norm_sem.append(sigs_sem[ibin] * f_norm)
        sigs_mean = np.array(norm_mean)
        sigs_sem = np.array(norm_sem)


    # plot
    if not len(axs):
        _ = plt.figure(1, fig_size)
        if rainbow_color:
            gs = gridspec.GridSpec(1, 2, width_ratios=[30, 1])
            ax = plt.subplot(gs[0, 0])
            ax_cb = plt.subplot(gs[0, 1])
        else:
            gs = gridspec.GridSpec(1, 1)
            ax = plt.subplot(gs[0, 0])
    else:
        ax = axs[0]

    cm = matplotlib.cm.get_cmap(colormap)
    for ibin in range(binnum):
        c = cm(ibin * 1. / (binnum - 1)) if rainbow_color else color
        ys = sigs_mean[ibin]
        ax.plot(xs, ys, c=c, lw=lw)
        # ax.plot(xs, ys, c=c, lw=lw, marker='s', ms=7, mec='none', mfc=c)
        if show_sem:
            sem = sigs_sem[ibin]
            ax.fill_between(xs, ys-sem, ys+sem, color=c, alpha=alpha_fill, zorder=1)
        if print_data:
            ystr = ''
            for y in ys:
                ystr = ystr + '%.3f' % y + ', '
            print ystr

            ystr = ''
            for y in sem:
                ystr = ystr + '%.3f' % y + ', '
            print ystr
            
    if plot_standing:
        ys = sigs_mean_stand
        ax.plot(xs, ys, c='black', lw=lw, ls='--')
        if show_sem:
            sem = sigs_sem_stand
            ax.fill_between(xs, ys-sem, ys+sem, color='black', alpha=alpha_fill, zorder=1)

    if plotcosfit:
        pi = np.pi
        theta = np.linspace(-pi, pi, 256)
        x_cos = (theta - theta[0]) / (2 * pi) * (xlim[-1] - xlim[0]) + xlim[0]
        if fit_rectify:
            y_cos = np.cos(theta + np.radians(offset + offset_fit)) * 1
        else:
            y_cos = np.cos(theta + np.radians(offset + offset_fit)) * .5 + .5
        ax.plot(x_cos, y_cos, lw=3, c=ph.grey9, alpha=0.3, zorder=0)

    ax.set_ylabel(r'$\Delta$F/F0', rotation=0)

    # trim plot
    ph.adjust_spines(ax, ['bottom'], lw=.25, xlim=xlim, xticks=xs, ylim=ylim, yticks=[])
    ax.patch.set_facecolor('white')
    ax.grid(which='major', alpha=0)
    c_yscale = 'black' if rainbow_color else color
    ph.plot_y_scale_text(ax, bar_length=y_barlength, text='%1.1f' % y_barlength, x_text_right=-.1, y_text_upper=.05,
                         color=c_yscale, fontsize=fs, horizontalalignment='right', verticalalignment='center')
    ax.set_ylabel(ylabel, rotation='vertical', va='center')
    ax.get_yaxis().set_label_coords(label_axeratio, .5)

    if rainbow_color:
        ph.plot_colormap(axs[1], colormap=cm, reverse_cm=False, ylim=[0, 256], yticks=[0, 256],
                         yticklabels=['%1.1f' % filter_bin[1][0], '%.2f' % filter_bin[1][1]],
                         ylabel=filter_bin[0], label_rotation=0, label_pos="right", label_axeratio=1)

    return ys, sem, sigs_sub


def plot_phase_subtracted_bolusshape_ave_bridge_flight(group, ax, offset=180, channel=1,
            normalization='an', color=ph.blue, norm_curve=False, lw_mean=2, lw_fit=2, lw_indi=1, ms=7, alpha_sem=0.3,
            alpha_indi=0.6, print_mean=False):
    # three types of filter: exclude_standing, filters_com, and filter_bin
    def cosine_func_phaseonly(x, phase):
        global bsl
        global amp
        return amp * np.sin(x - phase) + bsl

    def get_PBsinusoidal_fit(F):
        # F needs to be 18 glomeruli
        # F = F_EPGs_flight
        dinterp = 0.1
        global bsl
        global amp
        # step 0: fill to 18 glomeruli
        F18 = copy.copy(F)
        if np.sum(np.isnan(F)):  # otherwise, it's Delta7, no need to fill
            if np.isnan(F[8]):  # PFN
                F18[8] = F[0]
                F18[9] = F[-1]
            else:  # EPG
                F18[0] = F[8]
                F18[-1] = F[9]

        params = []
        for side in range(2):
            F9 = F18[:9] if side == 0 else F18[9:]

            # step 1: interpolation
            x = np.arange(0, 8.1, 1)
            f = interp1d(x, F9, 'cubic')
            x_interp = np.arange(.1, 8, dinterp)
            F_interp = f(x_interp)

            # step 2: calculate baseline and amplitude
            bsl = (np.max(F_interp) + np.min(F_interp)) / 2.
            amp = (np.max(F_interp) - np.min(F_interp)) / 2.

            # step 3: fit to cosine
            x_fit = np.radians(np.linspace(0, 360, 9))
            params_fit, _ = optimize.curve_fit(cosine_func_phaseonly, x_fit, F9, p0=[1])

            params.append([np.min(F_interp), np.max(F_interp), bsl, amp, params_fit[0]])

        return params

    def plot_PBsinusoidal_fit(params, sigs, Fmean, Fsem,):
        # if not ax:
        #     ax = ph.large_ax(fig_size)

        # plot data
        xs = np.arange(18)
        ax.plot(xs, Fmean, c=color, lw=lw_mean, marker='s', ms=ms, mec='none', mfc=color, zorder=3)
        ax.fill_between(xs, Fmean - Fsem, Fmean + Fsem, facecolor=color, edgecolor='none', alpha=alpha_sem, zorder=2)
        for F in sigs:
            ax.plot(xs, F, c=color, lw=lw_indi, alpha=alpha_indi, zorder=1)

        # plot left fit
        fmin, fmax, bsl, amp, offset = params[0]
        x_fit_plot = np.radians(np.linspace(0, 360, 80))
        Ffit = amp * np.sin(x_fit_plot - offset) + bsl
        ax.plot(np.arange(0, 8., .1), Ffit, c=ph.grey4, ls='--', lw=lw_fit)
        ystr = ''
        for y in amp * np.sin(np.radians(np.linspace(0, 360, 9)) - offset) + bsl:
            ystr = ystr + '%.4f' % y + ', '
        print 'fit: %s' %ystr

        # plot right fit
        fmin, fmax, bsl, amp, offset = params[1]
        x_fit_plot = np.radians(np.linspace(0, 360, 80))
        Ffit = amp * np.sin(x_fit_plot - offset) + bsl
        ax.plot(np.arange(9, 17, .1), Ffit, c=ph.grey4, ls='--', lw=lw_fit)
        ystr = ''
        for y in amp * np.sin(np.radians(np.linspace(0, 360, 9)) - offset) + bsl:
            ystr = ystr + '%.4f' % y + ', '
        print 'fit: %s' %ystr

        ph.adjust_spines(ax, ['bottom'], lw=1, ylim=[fmin - amp * 0.2, fmax + amp * 0.4], yticks=[], xlim=[-.5, 17.5],
                         xticks=[], pad=0)
        ph.plot_y_scale_text(ax, bar_length=.5, text='.5', x_text_right=-.1, y_text_upper=.05,
                             color=color, fontsize=5, horizontalalignment='right', verticalalignment='center')

    location = 'pb'
    binnum = 1

    # extracting data
    r = group[0][0]  # get the x axis for ploting
    _, _, x = r.get_objects('%s.c%i.xedges_interp' % (location, channel))
    _, _, celltype = r.get_objects('%s.c%i.celltype' % (location, channel))
    sigs = np.zeros((binnum, len(group), len(x)))
    sigs[:] = np.nan
    sigs_stand = np.zeros((len(group), len(x)))
    sigs_stand[:] = np.nan

    for ifly, fly in enumerate(group):
        sig_fly = np.zeros((binnum, len(fly), len(x)))
        sig_fly[:] = np.nan
        sig_fly_stand = np.zeros((len(fly), len(x)))
        sig_fly_stand[:] = np.nan

        for irec, rec in enumerate(fly):
            _, _, phase = rec.get_objects('%s.c%i.phase' % (location, channel))
            # _, _, phase = rec.get_objects('%s.c%i.phase' % (location, 2))
            im, ch, an_cubspl = rec.get_objects('%s.c%i.%s_cubspl' % (location, channel, normalization))
            a_nophase = im.cancel_phase(an_cubspl, phase, 10, ch.celltype, offset=offset)

            # apply three filters
            inds_flight = td.get_2p_ind_flight(rec)
            sig_fly[0][irec] = a_nophase[inds_flight].mean(axis=0)

        for ibin in range(binnum):
            sigs[ibin][ifly] = np.nanmean(sig_fly[ibin], axis=0)

    # Subsample, mean, sem
    idx = np.arange(len(x)).reshape((len(x) / 10, 10))
    sigs_sub = np.nanmean(sigs[:, :, idx], axis=-1)
    sigs_mean = np.nanmean(sigs_sub, axis=1)
    n = (np.isnan(sigs_sub) == False).sum(axis=1)
    sigs_sem = np.nanstd(sigs_sub, axis=1) / np.sqrt(n)

    if norm_curve:
        norm_mean = []
        norm_sem = []
        for ibin in range(binnum):
            y_ = sigs_mean[ibin]
            y0, y1 = np.nanmin(y_), np.nanmax(y_)
            f_norm = 1. / (y1 - y0)
            norm_mean.append((y_ - y0) * f_norm)
            norm_sem.append(sigs_sem[ibin] * f_norm)
        sigs_mean = np.array(norm_mean)
        sigs_sem = np.array(norm_sem)

    # plot
    # plots
    sigs = sigs_sub[0]
    F_mean = np.nanmean(sigs, axis=0)
    sigs_var = np.var(sigs, axis=0)
    params = get_PBsinusoidal_fit(F_mean)
    plot_PBsinusoidal_fit(params, sigs, F_mean, sigs_sem[0],)

    if print_mean:
        ystr = ''
        for y in F_mean:
            ystr = ystr + '%.4f' % y + ', '
        print 'mean: %s' % ystr

        ystr = ''
        for y in sigs_var:
            ystr = ystr + '%.4f' % y + ', '
        print 'variance: %s' % ystr

def plot_phase_subtracted_bolusshape_ave_bridge_walk_chisquare_20210706(group, ax, offset=180, channel=1,
            normalization='an', color=ph.blue, norm_curve=False, lw_mean=2, lw_fit=2, lw_indi=1, ms=7, alpha_sem=0.3,
            alpha_indi=0.6, print_mean=False):
    # three types of filter: exclude_standing, filters_com, and filter_bin
    def cosine_func_phaseonly(x, phase):
        global bsl
        global amp
        return amp * np.sin(x - phase) + bsl

    # def FitErr(p, data, var, angles):
        # return np.sum((data - (p[0] * np.sin(np.radians(angles + p[1])) + p[2])) ** 2 / var)
        # return np.sum((data - (p[0] * np.sin(np.radians(angles + p[1])) + p[2])) ** 2)
    def my_square(p, data, var, angles):
        return np.sum((data - (p[0] * np.sin(np.radians(angles + p[1])) + p[2])) ** 2)
    def my_chisquare(p, data, var, angles):
        return np.sum((data - (p[0] * np.sin(np.radians(angles + p[1])) + p[2])) ** 2 / var)

    def get_PBsinusoidal_fit():
        angles = np.arange(-180, 180, 45)
        if np.isnan(F_mean[8]):  # PFN
            datas = [F_mean[:8], F_mean[-8:]]
            var = [sigs_var[:8], sigs_var[-8:]]
            dof = 8 - 3
        else:  # EPG
            datas = [F_mean[1:9], F_mean[-9:-1]]
            var = [sigs_var[1:9], sigs_var[-9:-1]]
            dof = 8 - 3

        params = []
        for i_data, data in enumerate(datas):
            p = optimize.fmin(my_square, [1, 1, 0], args=(data, var[i_data], angles), disp=False)
            params.append(p)
            print 'chi-square: %.4f' % (my_chisquare(p, data, var[i_data], angles)/dof)
            print 'p-value: %.6f' % stats.distributions.chi2.sf(my_chisquare(p, data, var[i_data], angles), dof)

        return params

    def plot_PBsinusoidal_fit():
        Fsem = sigs_sem[0]
        # plot data
        xs = np.arange(18)
        ax.plot(xs, F_mean, c=color, lw=lw_mean, marker='s', ms=ms, mec='none', mfc=color, zorder=3)
        ax.fill_between(xs, F_mean - Fsem, F_mean + Fsem, facecolor=color, edgecolor='none', alpha=alpha_sem, zorder=2)
        for F in sigs:
            ax.plot(xs, F, c=color, lw=lw_indi, alpha=alpha_indi, zorder=1)

        # plot left fit
        for side_ in [0,1]:
            if side_ == 0:  # Left bridge
                xs_plot = np.linspace(0, 8, 80)
                if np.isnan(F_mean[8]): # PFN
                    angle_plot = np.linspace(-180, 180, 80)
                else:   # EPG
                    angle_plot = np.linspace(-180-45,180-45, 80)
            else:  # right bridge
                xs_plot = np.linspace(9, 17, 80)
                if np.isnan(F_mean[8]):  # PFN
                    angle_plot = np.linspace(-180-45,180-45, 80)
                else:  # EPG
                    angle_plot = np.linspace(-180, 180, 80)

            p = params[side_]
            ax.plot(xs_plot, p[0] * np.sin(np.radians(angle_plot + p[1])) + p[2], c=ph.grey4, ls='--', lw=lw_fit)

        fmin = np.nanmin(p[0] * np.sin(np.radians(angle_plot + p[1])) + p[2])
        fmax = np.nanmax(p[0] * np.sin(np.radians(angle_plot + p[1])) + p[2])
        amp = (fmax-fmin)/2.
        ph.adjust_spines(ax, ['bottom'], lw=1, ylim=[fmin - amp * 0.4, fmax + amp * 0.4], yticks=[], xlim=[-.5, 17.5],
                         xticks=[], pad=0)
        ph.plot_y_scale_text(ax, bar_length=.5, text='.5', x_text_right=-.1, y_text_upper=.05,
                             color=color, fontsize=5, horizontalalignment='right', verticalalignment='center')

    location = 'pb'
    binnum = 1

    # extracting data
    r = group[0][0]  # get the x axis for ploting
    _, _, x = r.get_objects('%s.c%i.xedges_interp' % (location, channel))
    _, _, celltype = r.get_objects('%s.c%i.celltype' % (location, channel))
    sigs = np.zeros((binnum, len(group), len(x)))
    sigs[:] = np.nan
    sigs_stand = np.zeros((len(group), len(x)))
    sigs_stand[:] = np.nan

    for ifly, fly in enumerate(group):
        sig_fly = np.zeros((binnum, len(fly), len(x)))
        sig_fly[:] = np.nan
        sig_fly_stand = np.zeros((len(fly), len(x)))
        sig_fly_stand[:] = np.nan

        for irec, rec in enumerate(fly):
            _, _, phase = rec.get_objects('%s.c%i.phase' % (location, channel))
            im, ch, an_cubspl = rec.get_objects('%s.c%i.%s_cubspl' % (location, channel, normalization))
            a_nophase = im.cancel_phase(an_cubspl, phase, 10, ch.celltype, offset=offset)

            # apply three filters
            sig_fly[0][irec] = a_nophase.mean(axis=0)

        for ibin in range(binnum):
            sigs[ibin][ifly] = np.nanmean(sig_fly[ibin], axis=0)

    # Subsample, mean, sem
    idx = np.arange(len(x)).reshape((len(x) / 10, 10))
    sigs_sub = np.nanmean(sigs[:, :, idx], axis=-1)
    sigs_mean = np.nanmean(sigs_sub, axis=1)
    n = (np.isnan(sigs_sub) == False).sum(axis=1)
    sigs_sem = np.nanstd(sigs_sub, axis=1) / np.sqrt(n)

    if norm_curve:
        norm_mean = []
        norm_sem = []
        for ibin in range(binnum):
            y_ = sigs_mean[ibin]
            y0, y1 = np.nanmin(y_), np.nanmax(y_)
            f_norm = 1. / (y1 - y0)
            norm_mean.append((y_ - y0) * f_norm)
            norm_sem.append(sigs_sem[ibin] * f_norm)
        sigs_mean = np.array(norm_mean)
        sigs_sem = np.array(norm_sem)

    # plot
    # plots
    sigs = sigs_sub[0]
    F_mean = np.nanmean(sigs, axis=0)
    sigs_var = np.var(sigs, axis=0)

    params = get_PBsinusoidal_fit()
    plot_PBsinusoidal_fit()

    if print_mean:
        ystr = ''
        for y in F_mean:
            ystr = ystr + '%.4f' % y + ', '
        print 'mean: %s' % ystr

        ystr = ''
        for y in sigs_var:
            ystr = ystr + '%.4f' % y + ', '
        print 'variance: %s' % ystr

def plot_phase_subtracted_bolusshape_ave_bridge_flight_chisquare_20210706(group, ax, offset=180, channel=1,
            normalization='an', color=ph.blue, norm_curve=False, lw_mean=2, lw_fit=2, lw_indi=1, ms=7, alpha_sem=0.3,
            alpha_indi=0.6, print_mean=False):
    # three types of filter: exclude_standing, filters_com, and filter_bin
    def cosine_func_phaseonly(x, phase):
        global bsl
        global amp
        return amp * np.sin(x - phase) + bsl

    # def FitErr(p, data, var, angles):
        # return np.sum((data - (p[0] * np.sin(np.radians(angles + p[1])) + p[2])) ** 2 / var)
        # return np.sum((data - (p[0] * np.sin(np.radians(angles + p[1])) + p[2])) ** 2)
    def my_square(p, data, var, angles):
        return np.sum((data - (p[0] * np.sin(np.radians(angles + p[1])) + p[2])) ** 2)
    def my_chisquare(p, data, var, angles):
        return np.sum((data - (p[0] * np.sin(np.radians(angles + p[1])) + p[2])) ** 2 / var)

    def get_PBsinusoidal_fit():
        angles = np.arange(-180, 180, 45)
        if np.isnan(F_mean[8]):  # PFN
            datas = [F_mean[:8], F_mean[-8:]]
            var = [sigs_var[:8], sigs_var[-8:]]
            dof = 8 - 3
        else:  # EPG
            datas = [F_mean[1:9], F_mean[-9:-1]]
            var = [sigs_var[1:9], sigs_var[-9:-1]]
            dof = 8 - 3

        params = []
        for i_data, data in enumerate(datas):
            p = optimize.fmin(my_square, [1, 1, 0], args=(data, var[i_data], angles), disp=False)
            params.append(p)
            print 'chi-square: %.4f' % (my_chisquare(p, data, var[i_data], angles)/dof)
            print 'p-value: %.6f' % stats.distributions.chi2.sf(my_chisquare(p, data, var[i_data], angles), dof)

        return params

    def plot_PBsinusoidal_fit():
        Fsem = sigs_sem[0]
        # plot data
        xs = np.arange(18)
        ax.plot(xs, F_mean, c=color, lw=lw_mean, marker='s', ms=ms, mec='none', mfc=color, zorder=3)
        ax.fill_between(xs, F_mean - Fsem, F_mean + Fsem, facecolor=color, edgecolor='none', alpha=alpha_sem, zorder=2)
        for F in sigs:
            ax.plot(xs, F, c=color, lw=lw_indi, alpha=alpha_indi, zorder=1)

        # plot left fit
        for side_ in [0,1]:
            if side_ == 0:  # Left bridge
                xs_plot = np.linspace(0, 8, 80)
                if np.isnan(F_mean[8]): # PFN
                    angle_plot = np.linspace(-180, 180, 80)
                else:   # EPG
                    angle_plot = np.linspace(-180-45,180-45, 80)
            else:  # right bridge
                xs_plot = np.linspace(9, 17, 80)
                if np.isnan(F_mean[8]):  # PFN
                    angle_plot = np.linspace(-180-45,180-45, 80)
                else:  # EPG
                    angle_plot = np.linspace(-180, 180, 80)

            p = params[side_]
            ax.plot(xs_plot, p[0] * np.sin(np.radians(angle_plot + p[1])) + p[2], c=ph.grey4, ls='--', lw=lw_fit)

        fmin = np.nanmin(p[0] * np.sin(np.radians(angle_plot + p[1])) + p[2])
        fmax = np.nanmax(p[0] * np.sin(np.radians(angle_plot + p[1])) + p[2])
        amp = (fmax-fmin)/2.
        ph.adjust_spines(ax, ['bottom'], lw=1, ylim=[fmin - amp * 0.4, fmax + amp * 0.4], yticks=[], xlim=[-.5, 17.5],
                         xticks=[], pad=0)
        ph.plot_y_scale_text(ax, bar_length=.5, text='.5', x_text_right=-.1, y_text_upper=.05,
                             color=color, fontsize=5, horizontalalignment='right', verticalalignment='center')

    location = 'pb'
    binnum = 1

    # extracting data
    r = group[0][0]  # get the x axis for ploting
    _, _, x = r.get_objects('%s.c%i.xedges_interp' % (location, channel))
    _, _, celltype = r.get_objects('%s.c%i.celltype' % (location, channel))
    sigs = np.zeros((binnum, len(group), len(x)))
    sigs[:] = np.nan
    sigs_stand = np.zeros((len(group), len(x)))
    sigs_stand[:] = np.nan

    for ifly, fly in enumerate(group):
        sig_fly = np.zeros((binnum, len(fly), len(x)))
        sig_fly[:] = np.nan
        sig_fly_stand = np.zeros((len(fly), len(x)))
        sig_fly_stand[:] = np.nan

        for irec, rec in enumerate(fly):
            _, _, phase = rec.get_objects('%s.c%i.phase' % (location, channel))
            # _, _, phase = rec.get_objects('%s.c%i.phase' % (location, 2))
            im, ch, an_cubspl = rec.get_objects('%s.c%i.%s_cubspl' % (location, channel, normalization))
            a_nophase = im.cancel_phase(an_cubspl, phase, 10, ch.celltype, offset=offset)

            # apply three filters
            inds_flight = td.get_2p_ind_flight(rec)
            sig_fly[0][irec] = a_nophase[inds_flight].mean(axis=0)

        for ibin in range(binnum):
            sigs[ibin][ifly] = np.nanmean(sig_fly[ibin], axis=0)

    # Subsample, mean, sem
    idx = np.arange(len(x)).reshape((len(x) / 10, 10))
    sigs_sub = np.nanmean(sigs[:, :, idx], axis=-1)
    sigs_mean = np.nanmean(sigs_sub, axis=1)
    n = (np.isnan(sigs_sub) == False).sum(axis=1)
    sigs_sem = np.nanstd(sigs_sub, axis=1) / np.sqrt(n)

    if norm_curve:
        norm_mean = []
        norm_sem = []
        for ibin in range(binnum):
            y_ = sigs_mean[ibin]
            y0, y1 = np.nanmin(y_), np.nanmax(y_)
            f_norm = 1. / (y1 - y0)
            norm_mean.append((y_ - y0) * f_norm)
            norm_sem.append(sigs_sem[ibin] * f_norm)
        sigs_mean = np.array(norm_mean)
        sigs_sem = np.array(norm_sem)

    # plot
    # plots
    sigs = sigs_sub[0]
    F_mean = np.nanmean(sigs, axis=0)
    sigs_var = np.var(sigs, axis=0)

    params = get_PBsinusoidal_fit()
    plot_PBsinusoidal_fit()

    if print_mean:
        ystr = ''
        for y in F_mean:
            ystr = ystr + '%.4f' % y + ', '
        print 'mean: %s' % ystr

        ystr = ''
        for y in sigs_var:
            ystr = ystr + '%.4f' % y + ', '
        print 'variance: %s' % ystr




def plot_phasic_shape_summary(groups, neural_types=[''], neuropils=['fb'], loc_num=0, channel=1,
                              neural_offsets=[90,], offsets_fit=[0,0,0,0,180+22.5], neural_colors=[ph.purple,],
                              normalization='an', colormap='viridis', ylim_shape=[0,4], ylim_tuning=[0,7.5], lw_indi=1, lw_mean=3, fs=8,
                              xs_wedge=np.arange(16)+1, xlim_wedge=[.5,16.5],
                              label_axeratio=-0.1, sig_type='mean_an', plot_standing=False, **kwargs):
    nrow = len(groups)
    _ = plt.figure(1, (20, nrow*2))
    gs = gridspec.GridSpec(nrow, 8, width_ratios=[1,.02,.66,.66,1,1,1,1])

    for row, recss in enumerate(groups):

        # shape tuned by forward velocity
        axs = [plt.subplot(gs[row, 0]), plt.subplot(gs[row, 1])]
        plot_phase_subtracted_bolusshape_ave(recss, axs=axs, neuropil=neuropils[row], offset=neural_offsets[row], channel=channel,
                    normalization=normalization, filters_com=[['dhead', [-80, 80]], ], filter_bin=['dforw', [0, 8]],
                    color=neural_colors[row], rainbow_color=True, colormap=colormap, show_sem=True, plotcosfit=False,
                    xs=xs_wedge, xlim=xlim_wedge, ylim=ylim_shape, y_barlength=1, lw=lw_indi, alpha_fill=0.1, fs=fs,
                    ylabel=neural_types[row], label_axeratio=label_axeratio, plot_standing=plot_standing,)

        # 1d forward vel and turn vel tuning
        axs = [plt.subplot(gs[row, 2]), plt.subplot(gs[row, 3])]
        fbs.plot_intensity_velocity_walking_bouts(recss, axs=axs, walking_bouts=False, bins=20, indi_alpha=0.2,
                    lw_indi=lw_indi, lw=lw_mean, c=neural_colors[row], vel_range=[[-4, 10], [0, 200], [0, 3]],
                    xticks=[np.arange(-4, 8.1, 4), np.arange(0, 201, 50), [-.1, 1]], yticks=ylim_tuning, ylim=ylim_tuning,
                    loc_num=loc_num, ch_num=channel, sig_type=sig_type, subplot_num=2, plotcurvesonly=False, fs=fs, )

        # normalized shape, with half sinusoidal fit
        plot_phase_subtracted_bolusshape_ave(recss, axs=[plt.subplot(gs[row, 4]), ], neuropil=neuropils[row],
                                             offset=neural_offsets[row],
                                             normalization=normalization,  channel=channel,
                                             filters_com=[['dhead', [-80, 80]], ['dforw', [2, 6]]], filter_bin=[],
                                             binnum=1, color=neural_colors[row], rainbow_color=False, show_sem=True,
                                             norm_curve=True, plotcosfit=True, fit_rectify=True,
                                             offset_fit=offsets_fit[row],
                                             xs=xs_wedge, xlim=xlim_wedge, ylim=[-1.1, 1.1], y_barlength=1,
                                             lw=lw_indi, alpha_fill=0.1, fs=fs)

        # normalized shape, with half sinusoidal fit, 90 deg additional offset
        plot_phase_subtracted_bolusshape_ave(recss, axs=[plt.subplot(gs[row, 5]), ], neuropil=neuropils[row],
                                             offset=neural_offsets[row]+90,
                                             normalization=normalization,  channel=channel,
                                             filters_com=[['dhead', [-80, 80]], ['dforw', [2, 6]]], filter_bin=[],
                                             binnum=1, color=neural_colors[row], rainbow_color=False, show_sem=True,
                                             norm_curve=True, plotcosfit=True, fit_rectify=True,
                                             offset_fit=offsets_fit[row]+180,
                                             xs=xs_wedge, xlim=xlim_wedge, ylim=[-1.1, 1.1], y_barlength=1,
                                             lw=lw_indi, alpha_fill=0.1, fs=fs)

        # normalized shape, with full sinusoidal fit
        plot_phase_subtracted_bolusshape_ave(recss, axs=[plt.subplot(gs[row, 6]), ], neuropil=neuropils[row],
                                             offset=neural_offsets[row],
                                             normalization=normalization,  channel=channel,
                                             filters_com=[['dhead', [-80, 80]], ['dforw', [2, 6]]], filter_bin=[],
                                             binnum=1, color=neural_colors[row], rainbow_color=False, show_sem=True,
                                             norm_curve=True, plotcosfit=True, fit_rectify=False,
                                             offset_fit=offsets_fit[row],
                                             xs=xs_wedge, xlim=xlim_wedge, ylim=[-.1, 1.1], y_barlength=1,
                                             lw=lw_indi, alpha_fill=0.1, fs=fs)

        # normalized shape, with full sinusoidal fit, 90 deg additional offset
        plot_phase_subtracted_bolusshape_ave(recss, axs=[plt.subplot(gs[row, 7]), ], neuropil=neuropils[row],
                                             offset=neural_offsets[row] + 90,
                                             normalization=normalization,  channel=channel,
                                             filters_com=[['dhead', [-80, 80]], ['dforw', [2, 6]]], filter_bin=[],
                                             binnum=1, color=neural_colors[row], rainbow_color=False, show_sem=True,
                                             norm_curve=True, plotcosfit=True, fit_rectify=False,
                                             offset_fit=offsets_fit[row]+180,
                                             xs=xs_wedge, xlim=xlim_wedge, ylim=[-.1, 1.1], y_barlength=1,
                                             lw=lw_indi, alpha_fill=0.1, fs=fs)

def plot_phasic_shape_summary_2_20201113(groups, neural_types=[''], neuropils=['fb'], channel=1,
                              neural_offsets=[90,], offsets_fit=[0,0,0,0,180+22.5], neural_colors=[ph.purple,],
                              normalization='an', colormap='viridis', ylim_shape=[0,4], lw_indi=1, fs=8,
                              label_axeratio=-0.1, plot_standing=False, **kwargs):
    nrow = len(groups)
    _ = plt.figure(1, (13, nrow*2))
    gs = gridspec.GridSpec(nrow, 6, width_ratios=[1,.02,1,1,1,1])

    for row, recss in enumerate(groups):

        # shape tuned by forward velocity
        axs = [plt.subplot(gs[row, 0]), plt.subplot(gs[row, 1])]
        plot_phase_subtracted_bolusshape_ave(recss, axs=axs, neuropil=neuropils[row], offset=neural_offsets[row], channel=channel,
                    normalization=normalization, filters_com=[['dhead', [-80, 80]], ], filter_bin=['dforw', [0, 10]],
                    color=neural_colors[row], rainbow_color=True, colormap=colormap, show_sem=True, plotcosfit=False,
                    xs=np.arange(16) + 1, xlim=[.5, 16.5], ylim=ylim_shape, y_barlength=1, lw=lw_indi, alpha_fill=0.1, fs=fs,
                    ylabel=neural_types[row], label_axeratio=label_axeratio, plot_standing=plot_standing)

        # normalized shape, with half sinusoidal fit
        plot_phase_subtracted_bolusshape_ave(recss, axs=[plt.subplot(gs[row, 2]), ], neuropil=neuropils[row],
                                             offset=neural_offsets[row],
                                             normalization=normalization,  channel=channel,
                                             filters_com=[['dhead', [-80, 80]], ['dforw', [2, 6]]], filter_bin=[],
                                             binnum=1, color=neural_colors[row], rainbow_color=False, show_sem=True,
                                             norm_curve=True, plotcosfit=True, fit_rectify=True,
                                             offset_fit=offsets_fit[row],
                                             xs=np.arange(16) + 1, xlim=[.5, 16.5], ylim=[-1.1, 1.1], y_barlength=1,
                                             lw=lw_indi, alpha_fill=0.1, fs=fs)

        # normalized shape, with half sinusoidal fit, 90 deg additional offset
        plot_phase_subtracted_bolusshape_ave(recss, axs=[plt.subplot(gs[row, 3]), ], neuropil=neuropils[row],
                                             offset=neural_offsets[row]+90,
                                             normalization=normalization,  channel=channel,
                                             filters_com=[['dhead', [-80, 80]], ['dforw', [2, 6]]], filter_bin=[],
                                             binnum=1, color=neural_colors[row], rainbow_color=False, show_sem=True,
                                             norm_curve=True, plotcosfit=True, fit_rectify=True,
                                             offset_fit=offsets_fit[row]+180,
                                             xs=np.arange(16) + 1, xlim=[.5, 16.5], ylim=[-1.1, 1.1], y_barlength=1,
                                             lw=lw_indi, alpha_fill=0.1, fs=fs)

        # normalized shape, with full sinusoidal fit
        plot_phase_subtracted_bolusshape_ave(recss, axs=[plt.subplot(gs[row, 4]), ], neuropil=neuropils[row],
                                             offset=neural_offsets[row],
                                             normalization=normalization,  channel=channel,
                                             filters_com=[['dhead', [-80, 80]], ['dforw', [2, 6]]], filter_bin=[],
                                             binnum=1, color=neural_colors[row], rainbow_color=False, show_sem=True,
                                             norm_curve=True, plotcosfit=True, fit_rectify=False,
                                             offset_fit=offsets_fit[row],
                                             xs=np.arange(16) + 1, xlim=[.5, 16.5], ylim=[-.1, 1.1], y_barlength=1,
                                             lw=lw_indi, alpha_fill=0.1, fs=fs)

        # normalized shape, with full sinusoidal fit, 90 deg additional offset
        plot_phase_subtracted_bolusshape_ave(recss, axs=[plt.subplot(gs[row, 5]), ], neuropil=neuropils[row],
                                             offset=neural_offsets[row] + 90,
                                             normalization=normalization,  channel=channel,
                                             filters_com=[['dhead', [-80, 80]], ['dforw', [2, 6]]], filter_bin=[],
                                             binnum=1, color=neural_colors[row], rainbow_color=False, show_sem=True,
                                             norm_curve=True, plotcosfit=True, fit_rectify=False,
                                             offset_fit=offsets_fit[row]+180,
                                             xs=np.arange(16) + 1, xlim=[.5, 16.5], ylim=[-.1, 1.1], y_barlength=1,
                                             lw=lw_indi, alpha_fill=0.1, fs=fs)

def plot_phasic_shape_summary_3_20210219(groups, neural_types=[''], neuropils=['fb'], loc_num=0, channel=1,
                              neural_offsets=[90,], offsets_fit=[0,0,0,0,180+22.5], neural_colors=[ph.purple,],
                              normalization='an', colormap='viridis', ylim_shape=[0,4], ylims_tuning=[[0,2],[-.1,.2]], lw_indi=1, lw_mean=3, fs=8,
                              xs_wedge=np.arange(16)+1, xlim_wedge=[.5,16.5],
                              label_axeratio=-0.1, plot_standing=False, **kwargs):
    nrow = len(groups)
    _ = plt.figure(1, (18, nrow*2))
    gs = gridspec.GridSpec(nrow, 8, width_ratios=[1,.02,.8,.6,.8,.6,1,1])

    for row, recss in enumerate(groups):

        # shape tuned by forward velocity
        axs = [plt.subplot(gs[row, 0]), plt.subplot(gs[row, 1])]
        plot_phase_subtracted_bolusshape_ave(recss, axs=axs, neuropil=neuropils[row], offset=neural_offsets[row], channel=channel,
                    normalization=normalization, filters_com=[['dhead', [-80, 80]], ], filter_bin=['dforw', [0, 8]],
                    color=neural_colors[row], rainbow_color=True, colormap=colormap, show_sem=True, plotcosfit=False,
                    xs=xs_wedge, xlim=xlim_wedge, ylim=ylim_shape, y_barlength=1, lw=lw_indi, alpha_fill=0.1, fs=fs,
                    ylabel=neural_types[row], label_axeratio=label_axeratio, plot_standing=plot_standing,)

        # 1d forward vel and derivative of sig vel tuning
        fbs.plot_intensity_velocity_walking_bouts(recss, axs=[plt.subplot(gs[row, 2]), ], walking_bouts=False, bins=20, indi_alpha=0.2,
                    lw_indi=lw_indi, lw=lw_mean, c=neural_colors[row], vel_range=[[-4, 10],[0,400],[-2,5]],
                    xticks=[np.arange(-4, 8.1, 4),], yticks=ylims_tuning[0], ylim=ylims_tuning[0],
                    loc_num=loc_num, ch_num=channel, sig_type='peak_an', subplot_num=1, plotcurvesonly=False, fs=fs, )

        fbs.plot_intensity_locomotion_xcorr(recss, ax=plt.subplot(gs[row, 3]), ch=channel, behtype='forw', fs=fs,
                    color=neural_colors[row], sig_type='peak_an', ylim=[-0.2, 0.8], yticks=[0, 0.4, 0.8], )

        fbs.plot_intensity_velocity_walking_bouts(recss, axs=[plt.subplot(gs[row, 4]), ], walking_bouts=False, bins=20,
                    indi_alpha=0.2, lw_indi=lw_indi, lw=lw_mean, c=neural_colors[row], vel_range=[[-4, 10],[0,400],[-2,5]],
                    xticks=[np.arange(-4, 8.1, 4), ], yticks=ylims_tuning[1], ylim=ylims_tuning[1],
                    loc_num=loc_num, ch_num=channel, sig_type='d_peak_an', subplot_num=1, plotcurvesonly=False, fs=fs, )

        fbs.plot_intensity_locomotion_xcorr(recss, ax=plt.subplot(gs[row, 5]), ch=channel, behtype='forw', fs=fs,
                    color=neural_colors[row], sig_type='d_peak_an', ylim=[-0.2, 0.8], yticks=[0, 0.4, 0.8])

        # normalized shape, with half sinusoidal fit
        plot_phase_subtracted_bolusshape_ave(recss, axs=[plt.subplot(gs[row, 6]), ], neuropil=neuropils[row],
                                             offset=neural_offsets[row],
                                             normalization=normalization,  channel=channel,
                                             filters_com=[['dhead', [-80, 80]], ['dforw', [2, 6]]], filter_bin=[],
                                             binnum=1, color=neural_colors[row], rainbow_color=False, show_sem=True,
                                             norm_curve=True, plotcosfit=True, fit_rectify=True,
                                             offset_fit=offsets_fit[row],
                                             xs=xs_wedge, xlim=xlim_wedge, ylim=[-1.1, 1.1], y_barlength=1,
                                             lw=lw_indi, alpha_fill=0.1, fs=fs)

        # normalized shape, with full sinusoidal fit
        plot_phase_subtracted_bolusshape_ave(recss, axs=[plt.subplot(gs[row, 7]), ], neuropil=neuropils[row],
                                             offset=neural_offsets[row],
                                             normalization=normalization,  channel=channel,
                                             filters_com=[['dhead', [-80, 80]], ['dforw', [2, 6]]], filter_bin=[],
                                             binnum=1, color=neural_colors[row], rainbow_color=False, show_sem=True,
                                             norm_curve=True, plotcosfit=True, fit_rectify=False,
                                             offset_fit=offsets_fit[row],
                                             xs=xs_wedge, xlim=xlim_wedge, ylim=[-.1, 1.1], y_barlength=1,
                                             lw=lw_indi, alpha_fill=0.1, fs=fs)

def plot_phasic_shape_summary_flight_temp(groups,  neuropils=['fb'], neural_offsets=[90,], offsets_fit=[0,0,0,0,180+22.5], neural_colors=[ph.purple,],
                              normalization='an', lw_indi=1, fs=8,):
    nrow = len(groups)
    _ = plt.figure(1, (20, nrow*2))
    gs = gridspec.GridSpec(nrow, 8, width_ratios=[1,.02,.66,.66,1,1,1,1])

    for row, recss in enumerate(groups):
        # normalized shape, with half sinusoidal fit
        plot_phase_subtracted_bolusshape_ave(recss, axs=[plt.subplot(gs[row, 4]), ], neuropil=neuropils[row],
                                             offset=neural_offsets[row],
                                             normalization=normalization, flight=True,
                                             binnum=1, color=neural_colors[row], rainbow_color=False, show_sem=True,
                                             norm_curve=True, plotcosfit=True, fit_rectify=True,
                                             offset_fit=offsets_fit[row],
                                             xs=np.arange(16) + 1, xlim=[.5, 16.5], ylim=[-1.1, 1.1], y_barlength=1,
                                             lw=lw_indi, alpha_fill=0.1, fs=fs)

        # normalized shape, with half sinusoidal fit, 90 deg additional offset
        plot_phase_subtracted_bolusshape_ave(recss, axs=[plt.subplot(gs[row, 5]), ], neuropil=neuropils[row],
                                             offset=neural_offsets[row]+90,
                                             normalization=normalization, flight=True,
                                             binnum=1, color=neural_colors[row], rainbow_color=False, show_sem=True,
                                             norm_curve=True, plotcosfit=True, fit_rectify=True,
                                             offset_fit=offsets_fit[row]+180,
                                             xs=np.arange(16) + 1, xlim=[.5, 16.5], ylim=[-1.1, 1.1], y_barlength=1,
                                             lw=lw_indi, alpha_fill=0.1, fs=fs)

        # normalized shape, with full sinusoidal fit
        plot_phase_subtracted_bolusshape_ave(recss, axs=[plt.subplot(gs[row, 6]), ], neuropil=neuropils[row],
                                             offset=neural_offsets[row],
                                             normalization=normalization, flight=True,
                                             binnum=1, color=neural_colors[row], rainbow_color=False, show_sem=True,
                                             norm_curve=True, plotcosfit=True, fit_rectify=False,
                                             offset_fit=offsets_fit[row],
                                             xs=np.arange(16) + 1, xlim=[.5, 16.5], ylim=[-.1, 1.1], y_barlength=1,
                                             lw=lw_indi, alpha_fill=0.1, fs=fs)

        # normalized shape, with full sinusoidal fit, 90 deg additional offset
        plot_phase_subtracted_bolusshape_ave(recss, axs=[plt.subplot(gs[row, 7]), ], neuropil=neuropils[row],
                                             offset=neural_offsets[row] + 90,
                                             normalization=normalization, flight=True,
                                             binnum=1, color=neural_colors[row], rainbow_color=False, show_sem=True,
                                             norm_curve=True, plotcosfit=True, fit_rectify=False,
                                             offset_fit=offsets_fit[row]+180,
                                             xs=np.arange(16) + 1, xlim=[.5, 16.5], ylim=[-.1, 1.1], y_barlength=1,
                                             lw=lw_indi, alpha_fill=0.1, fs=fs)

def plot_bridge_phasediff(groups, ax, flight=False, exclude_stand=False, group_labels=['celltype',], ms_indi=2, fs=8,
                          yticks=np.arange(0,91,22.5), ylim=[0,90], color=ph.blue):
    def ave_weighted_angles(ws, angles):
        das = []
        for w in ws:
            x = np.dot(w, np.cos(angles))
            y = np.dot(w, np.sin(angles))
            z = complex(x, y)
            da = np.angle(z, deg=True)
            das.append(da)
        return das

    pb_angles = np.arange(-np.pi, np.pi, np.pi/4,)
    xss = []
    for recss in groups:
        xs = []
        for fly in recss:
            das = []
            for rec in fly:
                sigs = rec.ims[0].c1.al
                sigs__ = sigs[:, :9]
                sig_left = sigs__[~np.isnan(sigs__)].reshape(len(sigs__), 8)
                sigs__ = sigs[:, 9:]
                sig_right = sigs__[~np.isnan(sigs__)].reshape(len(sigs__), 8)

                if flight:
                    inds_flight = td.get_2p_ind_flight(rec)
                    sig_left = sig_left[inds_flight]
                    sig_right = sig_right[inds_flight]
                if exclude_stand:
                    inds_walk = ~td.detect_standing(rec,)
                    sig_left = sig_left[inds_walk]
                    sig_right = sig_right[inds_walk]

                das.extend(fc.circsubtract(ave_weighted_angles(sig_left, pb_angles), ave_weighted_angles(sig_right, pb_angles)))
            xs.append(fc.circmean(das))
        xss.append(xs)
    # print np.nanmean(xss, axis=1)
    group_info = [[i] for i in range(len(groups))]
    ph.errorbar_lc_xss(xss=xss, ax=ax, group_info=group_info, group_labels=group_labels, colors=[color,],
                       ylabel='', yticks=yticks, ylim=ylim, yticks_minor=[], fs=fs,
                       ro_x=45, rotate_ylabel=False, plot_bar=False, ms_indi=ms_indi,
                       subgroup_samefly=False, circstat=True, legend_list=group_labels, show_flynum=False,
                       show_nancol=False)

    for ytick in yticks:
        ax.axhline(ytick, c=ph.grey5, ls='--', lw=.6, alpha=.6, zorder=1)












 
































