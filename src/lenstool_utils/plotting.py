#!/usr/bin/env python
"""Create a DY vs DX plot from chires.dat."""
import argparse
import numpy as np

from astropy.table import Table
from astropy import units
from matplotlib import pyplot as plt
from matplotlib import colors


def __chires2plots_argsHandler(options=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--color', '-c', type=str, metavar='COLOR', default='tab20',
        help='If a color is specified, then use the same color %(metavar)s for'
        'every family. %(metavar)s can be any color supported by matplotlib '
        '(eg. red, k, #ffbb00, etc.). It can also be any colormap supported '
        'by matplotlib (eg. tab20, jet, etc.). The default value is '
        '%(metavar)s=%(default)s'
    )

    parser.add_argument(
        '--style', '-s', type=str, metavar='MLP_STYLE', default=None,
        help='Set the style to use. %(metavar)s can be any matplotlib style or'
        'a list of styles (eg. "ggplot", "dark_background,presentation").'
    )

    parser.add_argument(
        '--vertical', action='store_true', default=False,
        help='Make plots using a vertical arrangement'
    )

    parser.add_argument(
        "chires_file", type=str, help='The chires file'
    )

    parser.add_argument(
        "out_filename", type=str,
        help='The name of the file where to save the figure'
    )

    return parser.parse_args(options)


def readChires(chires_file):
    with open(chires_file, 'r') as f:
        data_lines = f.readlines()[1:]

    tot_rmss = -1
    tot_rmsi = -1

    tbl_lines = []

    for line in data_lines:
        stripped_line = line.strip()
        if not (
            stripped_line.startswith('chi') or
            stripped_line.startswith('log')
        ):
            tbl_lines.append(line)
        elif stripped_line.startswith('chimul'):
            vals = stripped_line.split()
            tot_rmss = float(vals[5]) * units.arcsec
            tot_rmsi = float(vals[6]) * units.arcsec

    tbl_data = '\n'.join(tbl_lines)

    chires_tbl = Table.read(tbl_data, format='ascii')

    return chires_tbl, tot_rmss, tot_rmsi


def chires2plots(options=None):
    """
    Run the main program.

    Parameters
    ----------
    options : list
        list of cli parameters.

    Returns
    -------
    None.

    """
    args = __chires2plots_argsHandler(options)

    with open(args.chires_file, 'r') as f:
        data_lines = f.readlines()[1:]

    if args.style is not None:
        style_list = args.style.split(',')
        plt.style.use(style_list)

    chires_tbl, tot_rmss, tot_rmsi = readChires(args.chires_file)
    single_images = chires_tbl[chires_tbl['Narcs'] == 1]
    fam_stats = chires_tbl[chires_tbl['Narcs'] != 1]

    dx = [float(val) for val in single_images['dx']]
    dy = [float(val) for val in single_images['dy']]

    norm = colors.Normalize(vmin=0, vmax=np.max(single_images['N']))

    color_list = np.array([
        float(x) for x in single_images['N']
    ])

    try:
        cmap = plt.cm.get_cmap(args.color)
    except ValueError:
        color_list = args.color
        cmap = None

    fam_rmss = [
        float(x) for x in fam_stats['rmss']
    ]

    fam_rmsi = [
        float(x) for x in fam_stats['rmsi']
    ]

    fam_ids = [
        x['ID'][:-1] for x in fam_stats
    ]

    fam_idx = [
        int(x) for x in fam_stats['N']
    ]

    if args.vertical:
        fig = plt.figure(figsize=(5, 8))
        gs = fig.add_gridspec(4, 1)

        ax1 = fig.add_subplot(gs[:2, 0])
        ax2 = fig.add_subplot(gs[2, 0])
        ax3 = fig.add_subplot(gs[3, 0], sharex=ax2)
    else:
        fig = plt.figure(figsize=(10, 5))
        gs = fig.add_gridspec(2, 2)

        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1], sharex=ax2)

    ax1.scatter(
        dx, dy,
        marker='+',
        c=color_list,
        cmap=cmap,
        norm=norm,
    )
    ax1.set_title('True - Pred multiple images dy vs dx')
    ax1.set_xlabel('dx [arcsec]')
    ax1.set_ylabel('dy [arcsec]')

    ax1.annotate(
        f'Total RMSi: {tot_rmsi:.2f}\nTotal RMSs: {tot_rmss:.2f}',
        xy=(0, 1),
        xytext=(12, -12),
        va='top',
        xycoords='axes fraction',
        textcoords='offset points',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.5)
    )

    ax2_bars = ax2.bar(
        x=fam_idx,
        height=fam_rmsi,
        width=0.5,
        tick_label=fam_ids,
        zorder=2,
    )

    # Set the family color for each bar
    for bar, c in zip(ax2_bars, cmap(norm(fam_idx))):
        bar.set_color(c)

    ax2.set_ylabel('Image plane [arcsec]')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.xaxis.set_visible(False)
    ax2.set_title('Average family RMS')
    ax3_bars = ax3.bar(
        x=fam_idx,
        height=fam_rmss,
        width=0.5,
        tick_label=fam_ids,
        zorder=2
    )

    # Set the family color for each bar
    for bar, c in zip(ax3_bars, cmap(norm(fam_idx))):
        bar.set_color(c)

    l_colors = plt.rcParams['xtick.color']
    ax1.axvline(0, ls='--', lw=1, c=l_colors, alpha=0.9, zorder=0)
    ax1.axhline(0, ls='--', lw=1, c=l_colors, alpha=0.9, zorder=0)
    ax2.grid(axis='x')
    ax2.set_axisbelow(True)
    ax3.grid(axis='x')
    ax3.set_axisbelow(True)

    ax3.set_ylabel('Source plane [arcsec]')
    ax3.set_xlabel('Family/Clump ID')
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=90)
    ax3.tick_params(which='major', width=2.00, length=0)
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")

    plt.tight_layout(h_pad=0)
    plt.savefig(args.out_filename, bbox_inches='tight', dpi=150)
    plt.close(fig)
