import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import math
import os, random, math

redorange = (.96, .59, .2)
ddorange = (102/255., 51/255., 0/255.)
dorange = (0.85, 0.54, 0.24)
purple = (0.5, 0.2, .6)
dpurple = (0.2, 0.2, .6)
nblue = (.11, .27, .49)
bblue = (0.51, 0.67, 0.89)
ngreen = (0.01, 0.3, 0.01)
green = (0.5, 0.75, 0.42)
dgreen = (0.3, 0.55, 0.22)
ddgreen = (0.1, 0.35, 0.02)
bgreen = (.34, .63, .56)
dred = (.52, .12, 0)
dred_pure = (.4, 0,0)
dblue_pure = (0,0,.5)
blue_green = (0,.3,.8)
blue_red = (0,0,1)
green_red = (.7,.7,0)


yellow = (.95, .74, .22)
nyellow = (.3, .3, 0)

magenta = (175/255.,101/255.,168/255.)
orange = (244/255., 135/255., 37/255.)
dred = (131/255., 35/255., 25/255.)
red = (230/255., 0/255., 0/255.)
blue = (37/255.,145/255.,207/255.)
dblue = (27/255.,117/255.,187/255.)
rose = (194/255.,52/255.,113/255.)
pink = (221/255., 35/255., 226/255.)
dimred = (221/255., 49/255., 49/255.)
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

def set_fontsize(fontsize=20):
    rcParams['font.size'] = fontsize
    rcParams['xtick.labelsize'] = fontsize
    rcParams['ytick.labelsize'] = fontsize
    rcParams['axes.labelsize'] = fontsize
    # rcParams["font.family"] = "Helvetica Neue"
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Tahoma']

def adjust_spines(ax, spines, lw=.5, xlim=None, ylim=None,
                  xticks='auto', xticks_minor='none', yticks='auto', yticks_minor='none',
                  xticklabels=[], yticklabels=[], color='black', pad=1, ticklength=2, ticklw=1.5):
    if not lw is 'none':
        rcParams['axes.linewidth'] = lw
        ax.tick_params('both', length=ticklength, width=lw, which='major')
        ax.tick_params('both', length=ticklength * .66, width=lw, which='minor')

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
        if yticks == 'none':
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

def plot_colormap(ax, colormap=plt.cm.jet, reverse_cm=True, ylim=[0,256], yticks=[0,256], yticklabels=[], ylabel='ylabel', label_rotation=0, label_pos='right', label_axeratio=False):
    # plot colormap on a self-defined axis

    gradient = np.linspace(0, 1 - math.pow(10,-4), 256)
    gradient = np.vstack((gradient, gradient)).T[::-1,:] if reverse_cm else np.vstack((gradient, gradient)).T
    ax.imshow(gradient, aspect='auto', cmap=colormap)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.yaxis.tick_right()
    ax.set_ylabel(ylabel, rotation=label_rotation, va='center', ha='left')
    ax.yaxis.set_label_position(label_pos)
    if label_axeratio:
        ax.get_yaxis().set_label_coords(label_axeratio, .5)
    ax.set_xticks([])
    ax.set_ylim(ylim)
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(0.5)
    ax.tick_params('both', length=0, width=0, which='major')

def large_ax(size=(20,20)):
    fig = plt.figure(1, size)
    gs = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs[0, 0])
    return ax

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