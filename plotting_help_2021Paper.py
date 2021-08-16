from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.collections import LineCollection
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import functions_2021Paper as fc

from matplotlib import rcParams
import matplotlib
from matplotlib.ticker import StrMethodFormatter
from scipy import stats

import os, random, math

# redorange = (.96, .59, .2)
redorange = (244/255., 118/255., 33/255.)
ddorange = (102/255., 51/255., 0/255.)
borange = (1, .5, 0)
purple = (0.5, 0.2, .6)
lpurple = (0.6, 0.3, .8)
dpurple = (0.2, 0.2, .6)
dblue = (0, .2, .4)
nblue = (.11, .27, .49)
blue = (0.31, 0.47, 0.69)
bblue = (0.51, 0.67, 0.89)

dgreen = (0.3, 0.55, 0.22)
ddgreen = (0.1, 0.35, 0.02)
bgreen = (.34, .63, .56)
dred_pure = (.4, 0,0)
dblue_pure = (0,0,.5)
blue_green = (0,.3,.8)
blue_red = (0,0,1)
green_red = (.7,.7,0)
brown = '#D2691E'

yellow = (.95, .74, 22)
nyellow = (.3, .3, 0)

magenta = (175/255.,101/255.,168/255.)
dmagenta = (127/255.,22/255.,122/255.)
orange = (244/255., 135/255., 37/255.)
dorange = (232/255., 143/255., 70/255.)
red = (230/255., 0/255., 0/255.)
dred = (131/255., 35/255., 25/255.)
dimred = (221/255., 49/255., 49/255.)
blue = (37/255.,145/255.,207/255.)
dblue = (27/255.,117/255.,187/255.)
lgreen = (128/255.,189/255.,107/255.)
rose = (194/255.,52/255.,113/255.)
pink = (221/255., 35/255., 226/255.)
green = (0.5, 0.75, 0.42)
ngreen = (0.01, 0.3, 0.01)
brown = (71/255., 25/255., 2/255.)

black = (0., 0., 0.)
grey9 = (.1, .1, .1)
grey8 = (.2, .2, .2)
grey7 = (.3, .3, .3)
grey6 = (.4, .4, .4)
grey5 = (.5, .5, .5)
grey4 = (.6, .6, .6)
grey3 = (.7, .7, .7)
grey2 = (.8, .8, .8)
grey15 = (.85, .85, .85)
grey1 = (.9, .9, .9)
grey05 = (.95, .95, .95)
grey03 = (.97, .97, .97)
white = (1., 1., 1.)
lgrey = grey3
grey = grey5
dgrey = grey7

def create_linear_cm(RGB=[244,118,33]):
    N = 256
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(RGB[0]/255., 1, N)[::-1]
    vals[:, 1] = np.linspace(RGB[1]/255., 1, N)[::-1]
    vals[:, 2] = np.linspace(RGB[2]/255., 1, N)[::-1]
    newcmp = ListedColormap(vals)
    return newcmp

Magentas = create_linear_cm(RGB=[76,0,76])
Roses = create_linear_cm(RGB=[178,0,76])
# Pinks = create_linear_cm(RGB=[221,35,226])
Pinks = create_linear_cm(RGB=[203,45,211])
Reds = create_linear_cm(RGB=[204,30,30])
Browns = create_linear_cm(RGB=[71,25,2])

def seq_light_colours():

    return [(1,.4,.4), (.4,.7,1), (1,1,.4), (1,.4,1), (.4,1,.7),
            (.4,1,1), (1,.7,.4), (.7,.4,1), (.7,1,.4), (1,.4,.7)]

def seq_dark_colours():
    return [(.8,0,0), (0,.4,.8), (.6,.6,0), (.8,0,.8), (0,.8,.4),
            (.4,.8,0), (0,.8,.8), (.4,0,.8), (.8,0,.4), (.8,.4,0), ]

def seq_colours_high_contrast():
    return [[.4,0,0],[.2,.4,0],[0,.4,.4],[.2,0,.4],
            [.4,.2,0],[0,.4,0],[0,.2,.4],[.4,0,.4],
            [.4,.4,0],[0,.4,.2],[0,0,.4],[.4,0,.2],
            [1,0,0],[.5,1,0],[0,1,1],[.5,0,1],
            [1,.5,0],[0,1,0],[0,.5,1],[1,0,1],
            # [1,1,0],[0,1,.5],[0,0,1],[1,0,.5],
            [0, 1, .5], [0, 0, 1], [1, 0, .5],
            ]

def set_tickdir(direction):
    rcParams['xtick.direction'] = direction
    rcParams['ytick.direction'] = direction

def adjust_spines(ax, spines, lw=.5, xlim=None, ylim=None,
                  xticks='auto', xticks_minor='none', yticks='auto', yticks_minor='none',
                  xticklabels=[], yticklabels=[], color='black', pad=0, ticklength=2, ticklw=1.5):
    if not lw is 'none':
        rcParams['axes.linewidth'] = lw
        ax.tick_params('both', length=ticklength, width=lw, which='major')
        ax.tick_params('both', length=ticklength*.66, width=lw, which='minor')

    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', pad))  # outward by 10 points
            # spine.set_smart_bounds(True)
            spine.set_color(color)
        else:
            spine.set_color('none')  # don't draw spine
    
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)

    if 'bottom' in spines or 'top' in spines:
        if xticks == 'none':
            ax.set_xticks([])
        elif not xticks == 'auto':
            ax.set_xticks(xticks)
        if xticks_minor == 'none':
            ax.set_xticks([], minor=True)
        elif not xticks_minor == 'auto':
            ax.set_xticks(xticks_minor, minor=True)
        if len(xticklabels):
            ax.set_xticklabels(xticklabels)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])

    if 'left' in spines or 'right' in spines:
        if yticks =='none':
            ax.set_yticks([])
        elif not yticks == 'auto':
            ax.set_yticks(yticks)
        if yticks_minor == 'none':
            ax.set_yticks([], minor=True)
        elif not yticks_minor == 'auto':
            ax.set_yticks(yticks_minor, minor=True)
        if len(yticklabels):
            ax.set_yticklabels(yticklabels)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_label_position('left')
    elif 'right' in spines:
        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position('right')
    else:
        # no yaxis ticks
        for tic in ax.yaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_label_position('bottom')
    elif 'top' in spines:
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
    else:
        # no xaxis ticks
        for tic in ax.xaxis.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
            # ax.set_xticklabels([])

def draw_axes(ax, axes_to_keep=[], xlabel='', ylabel='', xticks='auto', yticks='auto',
              xlim='auto', ylim='auto', color='black'):
    all_axes = ['left', 'right', 'top', 'bottom']
    ax.xaxis.label.set_color(color)
    ax.tick_params(axis='x', colors=color)
    ax.yaxis.label.set_color(color)
    ax.tick_params(axis='y', colors=color)
    for axis in all_axes:
        if axis not in axes_to_keep:
            ax.spines[axis].set_color('none')
        else:
            ax.spines[axis].set_color(color)
            
    if 'left' in axes_to_keep:
        if yticks != 'auto':
            ax.set_yticks(yticks)
        ax.set_ylabel(ylabel, rotation='horizontal', horizontalalignment='right')
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_label_position('left') 
    elif 'right' in axes_to_keep:
        if yticks != 'auto':
            ax.set_yticks(yticks)
        ax.set_ylabel(ylabel, rotation='horizontal', horizontalalignment='right')
        ax.yaxis.set_ticks_position('right')
        ax.yaxis.set_label_position('right') 
    else:
        ax.yaxis.set_ticks([])

    if 'top' in axes_to_keep:
        if xticks != 'auto':
            ax.set_xticks(xticks)
        ax.set_xlabel(xlabel)
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top') 
    elif 'bottom' in axes_to_keep:
        if xticks != 'auto':
            ax.set_xticks(xticks)
        ax.set_xlabel(xlabel)
        ax.xaxis.set_ticks_position('bottom')
        ax.xaxis.set_label_position('bottom') 
    else:
        ax.xaxis.set_ticks([])
    
    if xlim != 'auto':
        ax.set_xlim(xlim)
    if ylim != 'auto':
        ax.set_ylim(ylim)

def make_segments(x, y):
    '''
    Create list of line segments from x and y coordinates, in the correct format for LineCollection:
    an array of the form   numlines x (points per line) x 2 (x and y) array
    '''

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    return segments

def colorline(x, y, z=None, ax=None, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0), lw=1, **kwargs):
    '''
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    '''
    
    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))
           
    # Special case if a single number:
    if not hasattr(z, "__iter__"):  # to check for numerical input -- this is a hack
        z = np.array([z])
        
    z = np.asarray(z)
    
    segments = make_segments(x, y)
    lc = LineCollection(segments, array=z, cmap=cmap, norm=norm, linewidth=lw, **kwargs)

    if ax is None:
        ax = plt.gca()
    ax.add_collection(lc)
    
    return lc

def circplot(x, y, circ='x', c=black, lw=1, ls='-', alpha=1, period=360, zorder=None):
    if circ == 'x':
        jumps = np.abs(np.diff(x))>period/2.
    elif circ == 'y':
        jumps = np.abs(np.diff(y))>period/2.
    
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    segments = segments[jumps==False]
    lc = LineCollection(segments, colors=c, linewidth=lw, linestyles=ls, alpha=alpha, zorder=zorder)
    ax = plt.gca()
    ax.add_collection(lc)

class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None, loc=4,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, width=0, color='black',
                 **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).

        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(Rectangle((0,0), sizex, width, fc=color, ec=color, lw=0))
        if sizey:
            bars.add_artist(Rectangle((0,0), width, sizey, fc=color, ec=color, lw=0))
 
        if sizex and labelx:
            bars = VPacker(children=[bars, TextArea(labelx, minimumdescent=False)],
                           align="center", pad=0, sep=sep)
        if sizey and labely:
            bars = HPacker(children=[TextArea(labely), bars],
                            align="center", pad=0, sep=sep)
 
        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=False, **kwargs)
 
def add_scalebar(ax, sbax=None, matchx=False, matchy=False, hidex=False, hidey=False, **kwargs):
    """ Add scalebars to axes

    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes

    - ax : the axis to attach ticks to
    - matchx,matchy : if True, set size of scale bars to spacing between ticks
                    if False, size should be set using sizex and sizey params
    - hidex,hidey : if True, hide x-axis and y-axis of parent
    - **kwargs : additional arguments passed to AnchoredScaleBars

    Returns created scalebar object
    """
    def f(axis):
        l = axis.get_majorticklocs()
        return len(l)>1 and (l[1] - l[0])
    
    if matchx:
        kwargs['sizex'] = f(ax.xaxis)
        kwargs['labelx'] = str(kwargs['sizex'])
    if matchy:
        kwargs['sizey'] = f(ax.yaxis)
        kwargs['labely'] = str(kwargs['sizey'])
        
    sb = AnchoredScaleBar(ax.transData, **kwargs)
    if sbax is None:
        ax.add_artist(sb)
    else:
        sbax.add_artist(sb)
 
    if hidex : ax.xaxis.set_visible(False)
    if hidey : ax.yaxis.set_visible(False)
 
    return sb

def set_legend(ax, facecolor='none', edgecolor='black', textcolor='black', loc='upper right',
               bbox_to_anchor=(0, 0)):
    handles, labels = ax.get_legend_handles_labels()
    legend = ax.legend(handles, labels, loc=loc, bbox_to_anchor=bbox_to_anchor)
    frame = legend.get_frame()
    frame.set_facecolor(facecolor)
    frame.set_edgecolor(edgecolor)
    for text in legend.get_texts():
        text.set_color(textcolor)
        
def save(suffix, rec=None, parentdir=True, exts=['png', 'pdf'], id=False, dpi=300,
         facecolor='none'):
    if parentdir:
        folder = './'
    else:
        folder = rec.folder
    directories = ['%s%s/' %(folder, ext) for ext in exts]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
    for ext in exts:
        if id and rec:
            flyid = rec.name
        else:
            flyid = ''
        plt.savefig('%s%s/%s%s.%s' %(folder, ext, flyid, suffix, ext), bbox_inches='tight',
                    transparent=True, dpi=dpi, facecolor=facecolor)


def set_xlim(ax, xlim):
    if len(xlim) == 3:
        xtickstep = xlim[2]
    else:
        xtickstep = xlim[1]
    ax.set_xlim(xlim[:2])
    ax.set_xticks(np.arange(xlim[0], xlim[1]+xtickstep/2., xtickstep))

def set_ylim(ax, ylim):
    if len(ylim) == 3:
        ytickstep = ylim[2]
    else:
        ytickstep = ylim[1]
    ax.set_ylim(ylim[:2])
    ax.set_yticks(np.arange(ylim[0], ylim[1]+ytickstep/2., ytickstep))

def seq_colours():
    redorange = (.96, .59, .2)
    dorange = (0.85, 0.54, 0.24)
    orange = (.96, .59, .2)
    purple = (0.5, 0.2, .6)
    dpurple = (0.2, 0.2, .6)
    nblue = (.11, .27, .49)
    blue = (0.31, 0.47, 0.69)
    bblue = (0.51, 0.67, 0.89)
    green = (0.5, 0.75, 0.42)
    dgreen = (0.3, 0.55, 0.22)
    ddgreen = (0.1, 0.35, 0.02)
    bgreen = (.34, .63, .56)
    red = (0.82, 0.32, 0.22)
    dred = (.52, .12, 0)
    yellow = (.95, .74, .22)
    brown = '#D2691E'
    return [redorange, dorange, orange, purple, dpurple, nblue, blue, bblue,
            green, dgreen, ddgreen, bgreen, red, dred, yellow, brown]

def set_fontsize(fontsize=20):
    rcParams['font.size'] = fontsize
    rcParams['xtick.labelsize'] = fontsize
    rcParams['ytick.labelsize'] = fontsize
    rcParams['axes.labelsize'] = fontsize
    # rcParams["font.family"] = "Helvetica Neue"
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Tahoma']

def large_ax(size=(20,20)):
    fig = plt.figure(1, size)
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])
    return ax

def plot_y_scale_text(ax, bar_length, text, x_text_right=None, y_text_upper=None, color='black',  **kwargs):
    ylim = ax.get_ylim()
    y_length = ylim[-1] - ylim[0]
    xlim = ax.get_xlim()
    x_length = xlim[-1] - xlim[0]
    if not x_text_right:
          x_text_right = 0.1
    if not y_text_upper:
        y_text_upper = 0.05
    x_text = xlim[0] + x_length * x_text_right
    y_text = ylim[0] + bar_length / 2.0 + y_length * y_text_upper
    x_bar = x_text + x_length * 0.01
    y0 = y_text - bar_length / 2.0
    y1 = y_text + bar_length / 2.0
    ax.plot([x_bar, x_bar], [y0, y1], c=color, lw=1, solid_capstyle='butt', clip_on=False)
    ax.text(x_text, y_text, text, color=color, clip_on=False, **kwargs)

def plot_x_scale_text(ax, bar_length, text, x_text_lefter=None, y_text_lower=None, color='black', **kwargs):
    ylim = ax.get_ylim()
    y_length = ylim[-1] - ylim[0]
    if not x_text_lefter:
        x_text_lefter = 0.1
    if not y_text_lower:
        y_text_lower = 0.1
    xlim = ax.get_xlim()
    x_length = xlim[-1] - xlim[0]
    x_text = xlim[-1] - bar_length/2.0 - x_length * x_text_lefter
    y_text = ylim[-1] - y_length * y_text_lower
    y_bar = y_text - y_length * 0.01
    x0 = x_text - bar_length / 2.0
    x1 = x_text + bar_length / 2.0
    ax.plot([x0, x1], [y_bar, y_bar], c=color, lw=1, solid_capstyle='butt', clip_on=False)
    ax.text(x_text, y_text, text, color=color, ha='center', va='bottom', clip_on=False, **kwargs)

def plot_xy_scalebar(ax, bar_length=1, fs=12, textlabel=True, unit='m', scale=1000, mode='corner', just_x=False):
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    text = str(bar_length*scale) + ' ' + unit
    xL = xlim[-1] - xlim[0]
    yL = ylim[-1] - ylim[0]

    if mode == 'corner':
        x0 = xlim[0] + xL * 0.05
        x1 = x0 + bar_length
        y0 = ylim[0] + yL * 0.05
        y1 = y0 + bar_length
        x_text = x0 + (x1 - x0) * .0
        y_text = y0 - (y1 - y0) * .01
        ha = 'left'
    elif mode == 'outside_corner':
        x0 = xlim[0] - xL * 0.05
        x1 = x0 + bar_length
        y0 = ylim[0] - yL * 0.05
        y1 = y0 + bar_length
        x_text = x0 + (x1 - x0) * .0
        y_text = y0 - (y1 - y0) * .01
        ha = 'left'
    elif mode == 'bottom_center_to_left':
        x0 = xlim[0] + xL / 2. - bar_length
        x1 = x0 + bar_length
        y0 = ylim[0] + yL * 0.03
        y1 = y0 + bar_length
        x_text = x0 + (x1 - x0) * .5
        y_text = y0 - (y1 - y0) * .03
        ha = 'center'
    elif mode == 'top_center_to_right':
        x0 = xlim[0] + xL * 0.8 - bar_length
        x1 = x0 + bar_length
        y0 = ylim[0] + yL * 0.7
        y1 = y0 + bar_length
        x_text = x0 + (x1 - x0) * .5
        y_text = y0 - (y1 - y0) * .1
        ha = 'center'

    ax.plot([x0, x1], [y0, y0], c='black', lw=2, solid_capstyle='butt', clip_on=False)
    if not just_x:
        ax.plot([x0, x0], [y0, y1], c='black', lw=2, solid_capstyle='butt', clip_on=False)
    if textlabel:
        ax.text(x_text, y_text, '%s' % text, ha=ha, va='top', fontsize=fs, clip_on=False)

def errorbar_lc_xsss(xsss=[], ax=False, xlabel_list=[], y_type='std', ylabel='', colors=[green], fs=15,
                     random_width=0.6, col_width=1.5, COL_width=1, plot_scale=0.5, margin_to_yaxis=1,
                     yticks=np.array([0, 1]), ylim=[0,1], marker="_", ms=5, legend=False, legend_list=[],
                     leg_loc='upper right', alternate_shading=0, seperating_lines=1, ave_in_fly=False,
                     plot_median=False, ro_x=45, label_axeratio=True, bbox_to_anchor=False, noyaxis=False,
                     connecting_means=False, rotate_ylabel=False, ytick_sci=False, show_ttest=False,):
    # 'x' represents each sample dot in the final plot
    # 'xs' represents each scenario/column in the final plot
    # 'xss' represents all columns from each genotype
    # 'xsss' represents different fly groups, usually different genotypes, i.e., different colors in the final plot

    num_col = len(xsss[0])
    num_COL = len(xsss)
    xlim = [0, col_width * num_col * num_COL + 1 + (num_COL - 1) * COL_width]
    y_xlabel = ylim[0] - (ylim[1] - ylim[0]) * 0.03
    x_step = COL_width + num_col * col_width

    set_fontsize(fs)
    if not ax:
        fig = plt.figure(1, ((num_col*num_COL*col_width+margin_to_yaxis+(num_COL-1)*COL_width)*plot_scale, 5))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])
    xmeanss = np.zeros((num_COL, num_col))          # for saving the xy coordinates of the means
    ymeanss = np.zeros((num_COL, num_col))          # for connecting_means plotting

    for i_col in range(num_col):
        x_sample_start = margin_to_yaxis + col_width * i_col
        x_mean_start = 1.7 + col_width * i_col
        x_sample_list = np.arange(x_sample_start, x_sample_start+x_step*num_COL, x_step)
        x_mean_list = np.arange(x_mean_start, x_mean_start+x_step*num_COL, x_step)

        for i_COL in range(num_COL):
            color = colors[i_col]
            x_mean = np.nanmean(xsss[i_COL][i_col])
            xmeanss[i_COL][i_col] = x_mean_list[i_COL]
            ymeanss[i_COL][i_col] = x_mean
            x_median = np.median(xsss[i_COL][i_col])
            x_std = np.nanstd(xsss[i_COL][i_col])
            x_sem = x_std / np.sqrt((np.isnan(xsss[i_COL][i_col]) == False).sum())
            x_ci95 = fc.confidence_interval(xsss[i_COL][i_col], confidence=0.95)
            for i in range(len(xsss[i_COL][i_col])):
                x_random = x_sample_list[i_COL] + (random.random() - .5) * random_width
                ax.plot(x_random, xsss[i_COL][i_col][i], 'o', mec=color, mew=1, mfc='white')

            if ave_in_fly:
                x_plot = x_mean
            elif plot_median:
                x_plot = x_median
            else:
                x_plot = x_mean

            if y_type == 'std':
                yerr = x_std
            elif y_type == 'sem':
                yerr = x_sem
            elif y_type == 'CI95':
                yerr = x_ci95
            else:
                raise NameError("Not recognized y_type '" + y_type + "'!")

            # print x_plot
            if legend and not i_COL:
                ax.errorbar(x_mean_list[i_COL], x_plot, yerr=yerr, c=color, marker=marker, mew=1, mec=color, mfc=color,
                            ms=ms, label=legend_list[i_col])
            else:
                ax.errorbar(x_mean_list[i_COL], x_plot, yerr=yerr, c=color, marker=marker, mew=1, mec=color, mfc=color,
                            ms=ms)

            if (len(xlabel_list) >= num_COL) and not i_col:
                ax.text((x_sample_list[i_COL]+x_mean_list[i_COL])/2.+col_width/2*(num_col-1), y_xlabel,
                        xlabel_list[i_COL], ha='center', va='top', rotation=ro_x)

    if alternate_shading:
        itv_shade = COL_width + num_col * col_width
        shade_start = margin_to_yaxis + itv_shade - (random_width + COL_width)/2.
        shade_start_list = np.arange(shade_start, shade_start + itv_shade * num_COL, itv_shade * 2)
        for shade_start in shade_start_list:
            ax.fill_between([shade_start, shade_start + itv_shade], ylim[0], ylim[-1], edgecolor='none', facecolor=grey3, alpha=.3)

    if seperating_lines:
        itv_shade = COL_width + num_col * col_width
        shade_start = margin_to_yaxis + itv_shade - (random_width + COL_width) / 2.
        shade_start_list = np.arange(shade_start, shade_start + itv_shade * num_COL, itv_shade)
        for shade_start in shade_start_list:
            ax.axvline(shade_start, ls='--', c=grey7, zorder=1, lw=.35, dashes=[10,10])

    if connecting_means:
        for i_COL in range(num_COL):
            for i_col in range(num_col-1):
                ax.plot([xmeanss[i_COL][i_col], xmeanss[i_COL][i_col+1]],
                        [ymeanss[i_COL][i_col], ymeanss[i_COL][i_col+1]], color=grey5, lw=1)

    if show_ttest:
        for i_COL in range(num_COL):
            x0 = 1.7 + col_width * 0 + x_step * i_COL
            x1 = 1.7 + col_width * 1 + x_step * i_COL
            y_bar = ylim[0] + (ylim[-1] - ylim[0]) * 0.83
            x_marker = (x0 + x1) / 2.
            y_marker = ylim[0] + (ylim[-1] - ylim[0]) * 0.84

            a = xsss[i_COL][0]
            b = xsss[i_COL][1]
            _, p2 = stats.ttest_ind(a, b)
            p_value = p2 / 2.
            if p_value >= 0.05:
                p_value_marker = 'ns'
            elif p_value >= 0.01:
                p_value_marker = '*'
            elif p_value >= 0.001:
                p_value_marker = '**'
            else:
                p_value_marker = '***'

            ax.plot([x0, x1], [y_bar, y_bar], lw=2, color='black')
            ax.text(x_marker, y_marker, p_value_marker, ha='center', va='bottom', fontsize=fs, color='black')

    if legend:
        if bbox_to_anchor:
            ax.legend(prop={'size': fs-5}, loc='upper left', bbox_to_anchor=bbox_to_anchor)
        else:
            ax.legend(prop={'size': fs-5}, loc=leg_loc)

    if noyaxis:
        adjust_spines(ax, ['left'], lw=1, xticks='none', yticks=[], xlim=xlim, ylim=ylim)
        ylabel = ''
    else:
        adjust_spines(ax, ['left'], lw=1, xticks='none', yticks=yticks, xlim=xlim, ylim=ylim)

    ax.patch.set_facecolor('white')
    ax.grid(which='major', alpha=0)
    if ylabel:
        if label_axeratio:
            if rotate_ylabel:
                ax.set_ylabel(ylabel, rotation = 'horizontal', ha = 'center')
            else:
                ax.set_ylabel(ylabel)
            if type(label_axeratio) is float:
                ax.get_yaxis().set_label_coords(label_axeratio, .5)
            else:
                ax.get_yaxis().set_label_coords(-.1, 0.5)
        else:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            tx = xlim[0] - 1.9
            ty = ylim[0] + (ylim[-1] - ylim[0]) * .5
            ax.text(tx, ty, ylabel, ha='center', va='center', fontsize=fs, rotation=90)

    if ytick_sci:
        ax.ticklabel_format(axis='y', style='sci', scilimits=(0,1))

    return ax


def errorbar_lc_xsss_2y(xsss=[], ylim1=[0,1], yticks1=np.array([0, 1]), ylim2=[1,3], yticks2=np.array([1,3.1]),
                        xlabel_list=[], y_type='std', ylabel='', colors=[green], fs=15,
                        random_width=0.6, col_width=1.5, COL_width=1, plot_scale=0.5, margin_to_yaxis=1,
                        marker="_", ms=5, legend=False, legend_list=[],
                        leg_loc='upper right', alternate_shading=0, seperating_lines=1, ave_in_fly=False,
                        plot_median=False, ro_x=45, label_axeratio=True, bbox_to_anchor=False, noyaxis=False,
                        connecting_means=False, rotate_ylabel=False, ytick_sci=False, show_ttest=False,):
    # ax1 is linear scale, ax2 is log scale
    #  'x' represents each sample dot in the final plot
    # 'xs' represents each scenario/column in the final plot
    # 'xss' represents all columns from each genotype
    # 'xsss' represents different fly groups, usually different genotypes, i.e., different colors in the final plot

    num_col = len(xsss[0])
    num_COL = len(xsss)
    xlim = [0, col_width * num_col * num_COL + 1 + (num_COL - 1) * COL_width]
    x_step = COL_width + num_col * col_width

    set_fontsize(fs)
    fig = plt.figure(1, ((num_col*num_COL*col_width+margin_to_yaxis+(num_COL-1)*COL_width)*plot_scale, 5+1))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 5])
    ax1 = plt.subplot(gs[1, 0])
    ax2 = plt.subplot(gs[0, 0])

    # plot linear scale, ax1
    errorbar_lc_xsss(xsss, ax=ax1, xlabel_list=xlabel_list, y_type=y_type, ylabel=ylabel, colors=colors, fs=fs,
                     yticks=yticks1, ylim=ylim1, marker='s', ms=8, legend=legend, legend_list=legend_list,
                     leg_loc=leg_loc, ro_x=ro_x, label_axeratio=label_axeratio, bbox_to_anchor=bbox_to_anchor,
                     connecting_means=connecting_means, rotate_ylabel=rotate_ylabel, ytick_sci=ytick_sci,
                     show_ttest=show_ttest)

    # plot log scale, ax2
    for i_col in range(num_col):
        x_sample_start = margin_to_yaxis + col_width * i_col
        x_sample_list = np.arange(x_sample_start, x_sample_start+x_step*num_COL, x_step)

        for i_COL in range(num_COL):
            color = colors[i_col]
            for i in range(len(xsss[i_COL][i_col])):
                if xsss[i_COL][i_col][i] > ylim2[0]:
                    x_random = x_sample_list[i_COL] + (random.random() - .5) * random_width
                    ax2.semilogy(x_random, xsss[i_COL][i_col][i], 'o', mec=color, mew=1, mfc='white')

    if seperating_lines:
        itv_shade = COL_width + num_col * col_width
        shade_start = margin_to_yaxis + itv_shade - (random_width + COL_width) / 2.
        shade_start_list = np.arange(shade_start, shade_start + itv_shade * num_COL, itv_shade)
        for shade_start in shade_start_list:
            ax2.axvline(shade_start, ls='--', c=grey7, zorder=1, lw=.35, dashes=[10,10])

    adjust_spines(ax2, ['left'], lw=1, xticks=[], xlim=xlim, ylim=ylim2, yticks=[1000])
    ax2.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax2.get_yaxis().set_minor_formatter(StrMethodFormatter('{x:0.0f}'))
    ax2.patch.set_facecolor('white')
    ax2.grid(which='major', alpha=0)


def plot_noline_between_lims(ax, x, y, threshold=135, **kwargs):
    # to deal with the annoying lines between -180 and +180
    inds = np.abs(np.gradient(y)) < threshold
    idxss = fc.get_contiguous_inds(inds, min_contig_len=1, keep_ends=True)
    for idxs in idxss:
        ax.plot(x[idxs], y[idxs], **kwargs)

def plot_noline_between_xlims(ax, x, y, threshold=135, **kwargs):
    # to deal with the annoying lines between -180 and +180
    inds = np.abs(np.gradient(x)) < threshold
    idxss = fc.get_contiguous_inds(inds, min_contig_len=1, keep_ends=True)
    for idxs in idxss:
        ax.plot(x[idxs], y[idxs], **kwargs)


def plot_colormap(ax, colormap=plt.cm.jet, reverse_cm=True, ylim=[0,256], yticks=[0,256], yticklabels=[], ylabel='ylabel', label_rotation=0, label_pos='right', label_axeratio=False, flipxy=False,):
    # plot colormap on a self-defined axis

    gradient = np.linspace(0, 1 - math.pow(10,-4), 256)
    gradient = np.vstack((gradient, gradient)).T[::-1,:] if reverse_cm else np.vstack((gradient, gradient)).T
    if not flipxy:
        ax.imshow(gradient, aspect='auto', cmap=colormap)
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.yaxis.tick_right()
        ax.set_ylabel(ylabel, rotation=label_rotation, va='center', ha='left')
        ax.yaxis.set_label_position(label_pos)
        if label_axeratio:
            ax.get_yaxis().set_label_coords(label_axeratio, .5)
        ax.set_ylim(ylim)

        ax.set_xticks([])
    else:
        ax.imshow(gradient.T, aspect='auto', cmap=colormap)
        ax.set_xticks(yticks)
        ax.set_xticklabels(yticklabels)
        ax.xaxis.tick_bottom()
        ax.set_xlabel(ylabel, rotation=label_rotation, va='top', ha='center')
        if label_axeratio:
            ax.get_xaxis().set_label_coords(.5, label_axeratio)
        ax.set_xlim(ylim)

        ax.set_yticks([])


    # for _, spine in ax.spines.items():
    #     spine.set_color('black')  # don't draw spine
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    ax.tick_params('both', length=0, width=0, which='major')

def plot_gradedcolors(ax, x, y, cm=plt.cm.viridis, nstep=10, reverse=True, **kwargs):
    n = len(x)
    step = int(n/nstep)
    color_seqs = np.linspace(1, 0, nstep) if reverse else np.linspace(0, 1, nstep)
    for i in range(nstep-1):
        idx0 = i*step
        idx1 = (i+1)*step+1
        c = cm(color_seqs[i])
        ax.plot(x[idx0:idx1], y[idx0:idx1], c=c, **kwargs)
    idx0 = (nstep-1)*step
    c = cm(1)
    ax.plot(x[idx0:], y[idx0:], c=c, **kwargs)


def special_color(special_color_nums, special_colors, i_col):
    if i_col in special_color_nums:
        if len(special_color_nums) == len(special_colors):
            color = special_colors[special_color_nums.index(i_col)]
        else:
            color = special_colors[0]
    else:
        color = False
    return color

def legend_group_number(group_info):
    # find out which group has the largest number of subgroups, use it to plot legend
    i_max = 0
    max_num = 0
    for igroup, sublist in enumerate(group_info):
        if len(sublist) > max_num:
            i_max = igroup
            max_num = len(sublist)
    return i_max

def errorbar_lc_xss(xss=[], ax=False, group_info=[[]],
                    col_width=0.4, subgroup_gap=0.1, group_gap=1, plot_scale=1, margin_to_yaxis=.5, figsize_height=5,
                    group_labels=[], colors=[grey2], special_color_nums=[], special_colors=[],
                    yerr_type='sem', ylabel='', yticks=np.array([0, 1]), ylim=[0,1], yticks_minor=[], fs=15, ms_indi=5,
                    alpha_indi=1,
                    noyaxis=False, ro_x=45, label_axeratio=True, rotate_ylabel=False, show_legend=False, legend_list=[],
                    legend_ncol=2, leg_loc='upper right', bbox_to_anchor=False, fs_legend=12, connecting_mean_list=[],
                    connecting_indi_list=[], ttest_list=[], test_type='student', plot_bar=True, alpha_bar=1,
                    subgroup_samefly=False, show_flynum=True, legend_frameon=False, yaxis_ratio=1, circstat=False,
                    circ_low=-180, circ_high=180, show_nancol=True, show_xlabel=True, hlines=[]):
    """
    second version of error bar plot, after function errorbar_lc_xsss.
    each plot is consisted of multiple groups, which is consisted of multiple subgroups (columns)
    group_labels only labels each group
    legend and color applies within each group, i.e, legends and colors are the same across groups, but different across subgroups
    :param xss:
        # 'x' represents each sample dot
        # 'xs' represents each subgroup, i.e, single columns
        # 'xss' represents all columns
    :param group_info: list of list, each sublist consists the seq # of each column to be in the same group
    :param colors: list, enumerate within each group columns
    :param special_color_nums: number of single column sequences
    :param special_colors: if len(special_colors) == len(special_color_nums), 1 to 1, otherwise, all use same spe color
    :param connecting_mean_list: list of list, each sublist consists the seq # of each column to be connected, one to next
    :param connecting_indi_list: same as connecting_means
    :return: ax
    """

    # calculate plotting parameters
    num_group = len(group_info)
    num_col = len([item for sublist in group_info for item in sublist])
    num_legend = len(legend_list)
    if num_col != len(xss):
        raise NameError("Error: length of xss does not match group_info, in function: ph.errorbar_lc_xss")
    xlim = [0, margin_to_yaxis + (num_col+2)*col_width + (num_group-1)*group_gap + (num_col-num_group)*subgroup_gap]
    y_xlabel = ylim[0] - (ylim[1] - ylim[0]) * .15
    y_xlabel_line = ylim[0] - (ylim[1] - ylim[0]) * .14
    y_n = ylim[0] - (ylim[1] - ylim[0]) * .1
    i_legend_group = legend_group_number(group_info)    # find out which group has the largest number of subgroups, use it to plot legend


    # setup plotting structure
    set_fontsize(fs)
    if not ax:
        fig = plt.figure(1, (xlim[-1]*plot_scale, figsize_height))
        gs = gridspec.GridSpec(1, 1)
        ax = plt.subplot(gs[0, 0])
    xmeans = []  # for saving the xy coordinates of the means
    ymeans = []  # for connecting_means
    indi_xpositions = [] # for saving the x coordinates of the individuals


    # plot individual fly and mean data
    for i_group in range(num_group):
        col_nums = group_info[i_group]

        for i_subgroup, i_col in enumerate(col_nums):

            # calculate plotting position, colors, etc.
            x_center = margin_to_yaxis + i_col*col_width + i_group*group_gap + (i_col-i_group)*subgroup_gap + .5*col_width
            _color = special_color(special_color_nums, special_colors, i_col)
            color = _color if _color else colors[i_subgroup % len(colors)]
            indi_xpositions.append([])

            # calculate plotting stats
            n_fly = len(xss[i_col])
            if not circstat:
                y_mean = np.nanmean(xss[i_col])
                if yerr_type == 'std':
                    yerr = np.nanstd(xss[i_col])
                elif yerr_type == 'sem':
                    yerr = np.nanstd(xss[i_col]) / np.sqrt(n_fly)
                elif yerr_type == 'CI95':
                    yerr = fc.confidence_interval(xss[i_col], confidence=0.95)
                else:
                    raise NameError("Not recognized y_type '" + y_type + "'!")
            else:
                # y_mean = fc.circmean(xss[i_col], low=circ_low, high=circ_high)
                y_mean = fc.circnanmean(xss[i_col], low=circ_low, high=circ_high, axis=0, nan_policy='omit')
                if yerr_type == 'std':
                    yerr = fc.circnanstd(xss[i_col], low=circ_low, high=circ_high, axis=0, nan_policy='omit')
                elif yerr_type == 'sem':
                    yerr = fc.circnanstd(xss[i_col], low=circ_low, high=circ_high, axis=0, nan_policy='omit') / np.sqrt(n_fly)
                else:
                    raise NameError("Not recognized y_type '" + y_type + "'!")
            xmeans.append(x_center)
            ymeans.append(y_mean)

            # plot individual fly marker
            for i in range(n_fly):
                x_random = x_center + (random.random() - .5) * col_width * 1.6
                indi_xpositions[-1].append(x_random)
                if plot_bar:
                    ax.plot(x_random, xss[i_col][i], 'o', ms=ms_indi, mec='black', mew=.6, mfc='none', alpha=alpha_indi)
                else:
                    if i == 0 and i_group == 0 and len(legend_list):
                        ax.plot(x_random, xss[i_col][i], 'o', ms=ms_indi, mec=color, mew=.6, mfc='none',
                                label=legend_list[i_subgroup], alpha=alpha_indi, zorder=1)
                    else:
                        ax.plot(x_random, xss[i_col][i], 'o', ms=ms_indi, mec=color, mew=.6, mfc='none', alpha=alpha_indi, zorder=1)

            # plot mean
            ax.errorbar(x_center, y_mean, yerr=yerr, color='black', fmt='_', ms=3, capsize=2, mew=.5,
                            ecolor='black', elinewidth=.5, zorder=2)
            # ax.errorbar(x_center, y_mean, yerr=yerr, color='black', fmt='_', ms=7, capsize=5, mew=1,
            #             ecolor='black', elinewidth=2, zorder=2)

            # plot bar
            if plot_bar:
                if show_legend and i_group == i_legend_group and i_subgroup < num_legend and len(legend_list):
                    ax.bar(x_center, y_mean, width=col_width, align='center', facecolor=color, edgecolor='none', linewidth=0,
                           label=legend_list[i_subgroup], alpha=alpha_bar, zorder=1)
                else:
                    ax.bar(x_center, y_mean, width=col_width, align='center', facecolor=color, edgecolor='none', linewidth=0,
                           alpha=alpha_bar, zorder=1)

            # plot number of flies
            if show_flynum:
                if subgroup_samefly:
                    if i_subgroup == 0:
                        x_center_nfly_start = margin_to_yaxis + group_info[i_group][0] * col_width + \
                                              i_group * group_gap + (group_info[i_group][0] - i_group) * subgroup_gap
                        x_center_nfly_length = len(group_info[i_group]) * col_width + \
                                               (len(group_info[i_group]) - 1) * subgroup_gap
                        x_center_nfly = x_center_nfly_start + x_center_nfly_length / 2.
                        ax.text(x_center_nfly, y_n, '%s' % len(xss[i_col]), ha='center', va='bottom')
                else:
                    ax.text(x_center, y_n, '%s' % len(xss[i_col]), ha='center', va='bottom')
                if i_col == 0:
                    ax.text(-subgroup_gap, y_n, 'n', ha='center', va='bottom')

        # plot xlabels
        if show_xlabel and (not np.isnan(y_mean) or show_nancol):
            x_xlabel_start = margin_to_yaxis + group_info[i_group][0] * col_width + i_group * group_gap \
                                 + (group_info[i_group][0] - i_group) * subgroup_gap
            x_xlabel_length = len(group_info[i_group]) * col_width + (len(group_info[i_group]) - 1) * subgroup_gap
            if ro_x == 0:
                x_xlabel_center = x_xlabel_start + x_xlabel_length / 2.
                ax.text(x_xlabel_center, y_xlabel, group_labels[i_group], ha='center', va='top', rotation=ro_x)
            else:
                x_xlabel_right = x_xlabel_start + x_xlabel_length
                ax.text(x_xlabel_right, y_xlabel, group_labels[i_group], ha='right', va='top', rotation=ro_x)
            x_xlabelline_left = x_xlabel_start + col_width/4.
            x_xlabelline_right = x_xlabel_start + x_xlabel_length - col_width/4.
            ax.plot([x_xlabelline_left, x_xlabelline_right], [y_xlabel_line, y_xlabel_line], color='black', lw=2, clip_on=False)

    # trim plot axes
    if noyaxis:
        adjust_spines(ax, [], lw=.4, xticks='none', yticks=[], yticks_minor=[], xlim=xlim, ylim=ylim)
        ylabel = ''
    elif yaxis_ratio == 1:
        adjust_spines(ax, ['left'], lw=.4, xticks='none', yticks=yticks, yticks_minor=yticks_minor, xlim=xlim, ylim=ylim)
    else:
        adjust_spines(ax, ['left'], lw=.4, xticks='none', yticks=yticks, yticks_minor=yticks_minor, xlim=xlim, ylim=ylim,
                      yticklabels=(np.array(yticks) * yaxis_ratio).astype('str'))

    ax.patch.set_facecolor('white')
    ax.grid(which='major', alpha=0)

    if ylabel:
        if label_axeratio:
            if rotate_ylabel:
                ax.set_ylabel(ylabel, rotation='horizontal', ha='center')
            else:
                ax.set_ylabel(ylabel)
            if type(label_axeratio) is float:
                ax.get_yaxis().set_label_coords(label_axeratio, .5)
            else:
                ax.get_yaxis().set_label_coords(-.1, 0.5)
        else:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            tx = xlim[0] - 1.9
            ty = ylim[0] + (ylim[-1] - ylim[0]) * .5
            ax.text(tx, ty, ylabel, ha='center', va='center', fontsize=fs, rotation=90)

    if show_legend:
        if bbox_to_anchor:
            ax.legend(prop={'size': fs_legend}, loc=leg_loc, bbox_to_anchor=bbox_to_anchor, ncol=legend_ncol, frameon=legend_frameon)
        else:
            ax.legend(prop={'size': fs_legend}, loc=leg_loc, ncol=legend_ncol, frameon=legend_frameon)

    # additional plot features
    if len(connecting_mean_list):
        for cols in connecting_mean_list:
            for _i in range(len(cols)-1):
                col0 = cols[_i]
                col1 = cols[_i + 1]
                ax.plot([xmeans[col0], xmeans[col1]], [ymeans[col0], ymeans[col1]], color=grey5, lw=2, alpha=1, zorder=1)

    if len(connecting_indi_list):
        for cols in connecting_indi_list:
            # check all cols have same number of flies
            if len(np.unique([len(xss[_col]) for _col in cols])) == 1:
                for _i in range(len(cols) - 1):
                    for _j in range(len(xss[cols[_i]])):
                        col0 = cols[_i]
                        col1 = cols[_i + 1]
                        ax.plot([indi_xpositions[col0][_j], indi_xpositions[col1][_j]], [xss[col0][_j], xss[col1][_j]], color=grey2, lw=1, alpha=.5, zorder=1)

    if len(ttest_list):
        for cols in ttest_list:
            for _i in range(len(cols) - 1):
                col0 = cols[_i]
                col1 = cols[_i + 1]
                x0, x1 = xmeans[col0], xmeans[col1]
                y_bar = ylim[0] + (ylim[-1] - ylim[0]) * 0.83
                x_marker = (x0 + x1) / 2.
                y_marker = ylim[0] + (ylim[-1] - ylim[0]) * 0.84

                ttest_a = np.array(xss[col0])[~np.isnan(xss[col0])]
                ttest_b = np.array(xss[col1])[~np.isnan(xss[col1])]
                if test_type == 'student':
                    _, p2 = stats.ttest_ind(ttest_a, ttest_b)
                    p_value = p2
                elif test_type == 'mannwhitneyu':
                    _, p_value = stats.mannwhitneyu(ttest_a, ttest_b)

                if p_value >= 0.05:
                    p_value_marker = 'ns'
                elif p_value >= 0.01:
                    p_value_marker = '*'
                elif p_value >= 0.001:
                    p_value_marker = '**'
                elif p_value >= 0:
                    p_value_marker = '***'
                else:
                    p_value_marker = 'err'
                # print p_value
                ax.plot([x0+col_width/4., x1-col_width/4.], [y_bar, y_bar], lw=2, color='black')
                ax.text(x_marker, y_marker, p_value_marker, ha='center', va='bottom', fontsize=fs, color='black')
                print p_value

    if len(hlines):
        for hline in hlines:
            ax.axhline(0, ls='--', lw=1, c=grey9, alpha=.5, zorder=1)




























