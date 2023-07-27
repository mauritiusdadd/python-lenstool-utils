"""Provides classes and functions to handle the results stored in bayes.dat."""
import os
import sys
import time
import shutil
import argparse
import tempfile
import warnings

from multiprocessing import Pool

import numpy as np

from matplotlib import pyplot as plt

from astropy.table import Table
from astropy.io import fits
from astropy.wcs import WCS
from astropy import units

from . import datamodel
from . import lenstool_wrapper
from . import utils

BAYES_TO_POTFILE_PARAMS = {
    'rcut': 'cut',
    'sigma': 'sigma',
}

BAYES_TO_POT_PARAMS = {
    'x': 'x_centre',
    'y': 'y_centre',
    'emass': 'ellipticite',
    'theta': 'angle_pos',
    'rc': 'core_radius',
    'sigma': 'v_disp',
    'rcut': 'cut_radius'
}


class BayesDat():
    """Read, write and handle the file bayes.dat."""

    def __init__(self, fname=None, data=None):
        self.data = data

        if self.data is None and fname is not None:
            self.read(fname)

    @classmethod
    def read(cls, fname):
        """
        Read the content of bayes.dat.

        Parameters
        ----------
        fname : str
            The path of bayes.dat.

        Returns
        -------
        tbl : astropy.table.Table
            The table containing the actual data.

        """
        with open(fname, 'r') as f:
            datalines = f.readlines()

        # Read the header and count the columns
        colnames = []
        index = 0
        for j, header_line in enumerate(datalines):
            text = header_line.strip()
            if not text.startswith('#'):
                index = j
                break
            colnames.append(text[1:])

        n_cols = len(colnames)
        n_rows = len(datalines) - j

        myt = Table(names=colnames, data=np.zeros((n_rows, n_cols)))
        print(f"\nFound {n_cols:d} columns...", file=sys.stderr)
        print(f"Reading {n_rows:d} lines...", file=sys.stderr)

        for j, header_line in enumerate(datalines[index:]):
            prog = j / (len(datalines) - index)
            pbar = lenstool_wrapper.printpbar(prog, prog)

            sys.stderr.write(f'\r{pbar}\r')
            sys.stderr.flush()

            data = header_line.strip().split()
            try:
                myt[j] = data
            except Exception as exc:
                print(
                    f"ERROR on line {j + index + 1}:", str(exc),
                    file=sys.stderr
                )
                print(f"N COLS: {len(colnames)}", file=sys.stderr)
                print(f"N VALS: {len(data)}", file=sys.stderr)

        return cls(data=myt)

    def tail(self, n):
        """
        Select only the last entries of the current data.

        Parameters
        ----------
        n : float
            If greater than one, then only the last n lines are kept.
            If lesser than one, then only the last n-th fraction is kept.

        Returns
        -------
        trimmed : BayesDat
            The trimmed data

        """
        if n == 1:
            return
        elif n < 1:
            n = int(n * len(self.data))

        return BayesDat(data=self.data[int(-n):])

    def random_sample(self, n):
        """
        Extract a random sample from the current data.

        Parameters
        ----------
        n : float
            If greater than one, then only n random lines are extracted.
            If lesser than one, then only it is interpreted as a fraction.

        Returns
        -------
        sample : BayesDat
            The random sample.
        """
        if n == 1:
            return
        elif n < 1:
            n = int(n * len(self.data))

        random_mask = np.zeros(len(self.data), dtype=bool)
        random_mask[:int(n)] = True
        rng = np.random.default_rng()
        rng.shuffle(random_mask)
        return BayesDat(data=self.data[random_mask])

    def _write_dat(self, fname):
        if not fname.endswith('.dat'):
            fname = fname + '.dat'

    def write(self, fname, format='dat', overwrite=False, **kwargs):
        """
        Save bayes data to a file.

        Parameters
        ----------
        fname : str
            The output file name.
        format : str, optional
            The format of the output. Can be 'dat' or any format supported by
            astropy.table.Table. The default is 'dat'.

        Returns
        -------
        None.

        """
        if format == 'dat':
            self._write_dat(fname)
        else:
            self.data.write(
                fname,
                format=format,
                overwrite=overwrite,
                **kwargs
            )

    def updateParFile(self, in_par, bayes_index):
        """
        Update the values of a par file (ie. bestopt.par) with bayes.dat.

        Parameters
        ----------
        in_par : str
            The path of the input par file.
        bayes_index : int
            The row of the par file to use.

        Returns
        -------
        par : datamodel.ParametersDataModel
            The updated par file datamodel.

        """
        orig = datamodel.ParametersDataModel()
        orig.read(in_par)

        runmode = orig.get_runmode()

        for x in ['mass', 'ampli', 'shearfield', 'time']:
            try:
                x_vals = runmode.get_parameter_values(x)
            except KeyError:
                continue
            if x_vals[-1].lower().endswith('.fits'):
                base_name = x_vals[-1][:-5]
            else:
                base_name = x_vals[-1]
            x_vals[-1] = f"{base_name}_{bayes_index:06d}.fits"
            runmode.set_parameter_values(x, x_vals)

        for col in self.data.colnames:
            val = self.data[col][bayes_index]

            cname = col.lower().strip()

            if cname in ['nsample', 'ln(lhood)', 'chi2']:
                continue
            elif cname.startswith('pot'):
                pot_index = cname[4]
                param_name = BAYES_TO_POTFILE_PARAMS[cname.split()[1].strip()]
                ident = orig.get_identifier('potfile')
                new_val = [0, val, val]
            else:
                pot_index, other_data = cname.split(':')
                pot_index = pot_index.strip()[1]
                param_name = BAYES_TO_POT_PARAMS[other_data.split()[0].strip()]
                ident = orig.get_identifier('potentiel', pot_index)
                new_val = [val, ]

            if len(ident) == 1:
                ident[0].set_parameter_values(param_name, new_val)
            elif len(ident) > 1:
                warnings.warn("Multiple identifier matched: skipping...")

        return orig


def bayesMapPP(bayes_file, par_file, out_dir='bayesmap_out', tail=None,
               random_sample=None, n_jobs=4, no_tempfs=False, verbose=False):
    """
    Run lenstool using the given par file with data from bayes.dat.

    Parameters
    ----------
    bayes_file : str
        Path of the input par file.
    par_file : str
        Path of the input par file.

    Returns
    -------
    None.

    """
    def _loop(lt_process_list):
        to_be_removed = []
        time.sleep(0.1)
        for k, (j, j_tempd, j_lt_p, j_out_dir) in enumerate(lt_process_list):
            if j_lt_p._lenstool_popen.poll() is not None:
                to_be_removed.append(k)
            out = j_lt_p.parsedata()
            if out and verbose:
                print(out, file=sys.stderr)

        to_be_removed.sort()
        for k in to_be_removed[::-1]:
            j, j_tempd, j_lt_p, j_out_dir = lt_process_list.pop(k)
            # copy output files to their destination directory
            for file_to_copy in os.listdir(j_tempd.name):
                if not file_to_copy.lower().endswith('.fits'):
                    continue
                shutil.copy(
                    os.path.join(j_tempd.name, file_to_copy),
                    os.path.join(j_out_dir, file_to_copy)
                )
            time.sleep(0.1)
            j_tempd.cleanup()

    def _report_progress(j, tot, lt_process_list, state='Running'):
        running_prog = 0
        n_running_jobs = len(lt_process_list)
        for _, _, j_lt_p, _ in lt_process_list:
            running_prog += j_lt_p.global_progress

        total_prog = (j - n_running_jobs + running_prog) / tot

        pbar = lenstool_wrapper.printpbar(
            running_prog/n_running_jobs if n_running_jobs else 0,
            total_prog
        )
        print(
            f"\r{state}: {pbar} {total_prog: 6.2%} "
            f"[{n_running_jobs:d} | {j - n_running_jobs:d}/{tot}]",
            end='\r',
            file=sys.stderr
        )

    model_dir = os.path.dirname(os.path.realpath(par_file))
    map_out_dir = os.path.join(model_dir, out_dir)

    if not os.path.isdir(map_out_dir):
        os.makedirs(map_out_dir)

    bd = BayesDat.read(os.path.realpath(bayes_file))
    if tail:
        bd = bd.tail(tail)

    if random_sample:
        bd = bd.random_sample(random_sample)

    lt_max_processes = n_jobs
    lt_processes = []

    if no_tempfs:
        root_tmp_dir = tempfile.TemporaryDirectory(dir='.')
    else:
        root_tmp_dir = None

    print("\nRunning lenstool...", file=sys.stdout)
    n_entries = len(bd.data)
    try:
        for j in range(n_entries):
            while len(lt_processes) >= lt_max_processes:
                _loop(lt_processes)
                _report_progress(j, n_entries, lt_processes)

            tempd = tempfile.TemporaryDirectory(dir=root_tmp_dir.name)

            updated_par = bd.updateParFile(par_file, j)
            updated_par.get_runmode().set_parameter_values(
                'inverse', [0, 0.5, 1]
            )

            tmp_parfile = os.path.join(tempd.name, 'input.par')
            with open(tmp_parfile, 'w') as f:
                f.write(str(updated_par))

            # Copy auxiliary files to tempd
            for fid, idname in [('potfile', 'filein'), ('image', 'multfile')]:
                for fid in updated_par.get_identifier(fid):
                    file_name = fid.get_parameter_values(idname)[1]
                    shutil.copy(
                        os.path.join(model_dir, file_name),
                        os.path.join(tempd.name, file_name),
                    )

            lt_wrap = lenstool_wrapper.LensToolWrapper(n_threads=1)
            lt_wrap.run(tmp_parfile)

            lt_processes.append(
                (j, tempd, lt_wrap, map_out_dir)
            )

        print(
            "Waiting for the remaing processes to finish...", file=sys.stderr
        )

        while lt_processes:
            _loop(lt_processes)
            _report_progress(n_entries, n_entries, lt_processes)
    except KeyboardInterrupt:
        print(
            "\n\nStopping the remaining running processes...\n\n",
            file=sys.stderr
        )
        while lt_processes:
            j, j_tempd, j_lt_p, j_out_dir = lt_processes.pop()
            if j_lt_p._lenstool_popen.poll() is not None:
                continue
            j_lt_p.kill()
            _report_progress(n_entries, n_entries, lt_processes, 'Stopping')

    if root_tmp_dir is not None:
        root_tmp_dir.cleanup()


def bayesCubePP(inp_dir):
    """
    Combine multiple files generated by bayesMapPP into a single datacube.

    Parameters
    ----------
    inp_dir : str
        Path of the directory generated by bayesMapPP.

    Returns
    -------
    None.

    """
    base_name = os.path.basename(inp_dir)
    fits_list = [
        os.path.join(inp_dir, fname)
        for fname in os.listdir(inp_dir)
        if fname.endswith('.fits')
    ]

    data_cube = None
    header = None
    for j, fname in enumerate(fits_list):
        prog = (j + 1) / len(fits_list)
        pbar = lenstool_wrapper.printpbar(prog, prog)
        sys.stderr.write(f"\rGenerating datacube: {pbar} {prog: 6.2%}\r")
        sys.stderr.flush()
        with fits.open(fname) as hdul:
            data = hdul[0].data
            if data_cube is None:
                header = hdul[0].header.copy()
                data_cube = np.empty(
                    (len(fits_list), *data.shape),
                    dtype='float32'
                )
            data_cube[j, ...] = data

    print(
        "\nSaveing datacube...",
        file=sys.stderr
    )
    hdu = fits.PrimaryHDU(
        data=data_cube,
        header=header
    )
    hdu.writeto(base_name + '_cube.fits', overwrite=True)


def _einrWorker(fname):
    with fits.open(fname) as hdul:
        mywcs = WCS(hdul[0].header)
        e_rad_pix, e_rad_arcsec, poly = utils.getEffectiveEinsteinRadius(
            hdul[0].data,
            mywcs
        )
        crimg = utils.getCritic(hdul[0].data)
        return (e_rad_pix, e_rad_arcsec, poly, crimg, mywcs)


def bayesEinRadPP(inp_dir, outdir='.', verbose=False, overlay_img=None,
                  overlay_img_wcs=None, n_jobs=8):
    """
    Compute the Einstein radii for the files generated by bayesMapPP.

    Parameters
    ----------
    inp_dir : str
        Path of the directory generated by bayesMapPP.
    outdir : str, optional
        The path where to save the check images. If None then no image is
        generated. The default is '.'.
    verbose : bool, optional
        Whether to use an increased verbosity. The default is False.
    overlay_img: numpy.ndarray, optional
        An image used when generating the check images. The default is None.
    overlay_img_wcs: astropy.wcs.WCS, optional
        The WCS for overlay_img. The default is None.

    Returns
    -------
    None.

    """
    base_name = os.path.basename(inp_dir)
    fits_list = [
        os.path.join(inp_dir, fname)
        for fname in os.listdir(inp_dir)
        if fname.endswith('.fits')
    ]

    if outdir is not None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection=overlay_img_wcs)

        if overlay_img is not None:
            ax.imshow(
                overlay_img,
                origin='lower',
                cmap='gray_r',
                vmin=-0.005,
                vmax=0.05
            )
    else:
        ax = None

    e_rad_tbl = Table(names=['E_RAD_PIX', 'E_RAD_ARCSEC'])
    critics = None
    with Pool(n_jobs) as mypool:
        for j, res in enumerate(mypool.imap_unordered(_einrWorker, fits_list)):
            prog = (j + 1) / len(fits_list)
            pbar = lenstool_wrapper.printpbar(prog, prog)
            sys.stderr.write(
                f"\rComputing critical curves: {pbar} {prog: 6.2%}\r"
            )
            sys.stderr.flush()
            e_rad_pix, e_rad_arcsec, poly, critic, critic_wcs = res

            e_rad_tbl.add_row([e_rad_pix, e_rad_arcsec])

            if critics is None:
                critics = critic.astype('float32')
                critics_wcs = critic_wcs
            else:
                critics += critic.astype('float32')

            if verbose:
                print(
                    f"E rad. = {e_rad_arcsec:.2f} ({e_rad_pix} pixels)",
                    file=sys.stderr
                )
    print(
        "\nSaveing results...",
        file=sys.stderr
    )
    e_rad_tbl.write(base_name + '_einrad_tbl.fits', overwrite=True)

    if ax is not None:
        if overlay_img_wcs is not None:
            extent = utils.getReprojectionExtent(
                critics,
                critics_wcs,
                overlay_img_wcs
            )
        else:
            extent = None

        mappable = ax.imshow(
            critics,
            alpha=1.0 * (critics > 0),
            cmap='plasma',
            vmin=1,
            extent=extent
        )

        try:
            ra = ax.coords[0]
            dec = ax.coords[1]
        except AttributeError:
            pass
        else:
            ra.set_major_formatter('hh:mm:ss')
            dec.set_major_formatter('dd:mm:ss')

            ra.set_ticks(spacing=1 * units.arcmin, color='black')
            dec.set_ticks(spacing=1 * units.arcmin, color='black')
            ra.set_ticklabel(exclude_overlapping=True)
            dec.set_ticklabel(exclude_overlapping=True, rotation=90)

            ra.display_minor_ticks(True)
            dec.display_minor_ticks(True)

            ax.tick_params(
                axis='both',
                which='both',
                direction='in',
            )
            ax.set_xlabel('R.A.')
            ax.set_ylabel('Dec.')

        cbaxes = ax.inset_axes((0.65, 0.075, 0.3, 0.025))

        cb_fg_color = 'black'

        cbar = fig.colorbar(
            mappable,
            ticks=[1, np.max(critics)],
            ax=ax,
            cax=cbaxes,
            orientation='horizontal',
        )

        cbar.set_label(
            'N',
            color=cb_fg_color,
            labelpad=-10,
            y=0,
            fontsize=14
        )
        cbar.outline.set_edgecolor(cb_fg_color)
        cbar.ax.xaxis.set_tick_params(
            color=cb_fg_color,
            labelcolor=cb_fg_color
        )

        plt.tight_layout()
        fig.savefig(
            os.path.join(outdir, f'{base_name}_critics.png'),
            bbox_inches='tight',
            dpi=150
        )
        plt.close(fig)

        hdu = fits.PrimaryHDU(data=critics, header=critics_wcs.to_header())
        hdu.writeto(
            os.path.join(outdir, f'{base_name}_critics.fits'),
            overwrite=True
        )


def execBayesMapPP(options=None):
    """
    Execute bayesMapPP.

    Parameters
    ----------
    options : list, optional
        Program arguments. If note arguments are read from the command line.
        The default is None.

    Returns
    -------
    None.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'bayesfile', type=str, help='The path of a bayes.dat'
    )
    parser.add_argument(
        'parfile', type=str, help='The path of a par file (ie. best.par)'
    )
    parser.add_argument(
        '--outdir', '-o', type=str, metavar="OUT_DIR", default='bayesmap_out',
        help='Set the path where to save produced output files to %(metavar)s.'
        ' Te default value is %(metavar)s=%(default)s.'
    )

    parser.add_argument(
        '--tail', '-t', type=float, default=None, metavar='N',
        help='If %(metavar)s is greater than 1 then use only the last '
        '%(metavar)s rows of bayes.dat. If %(metavar)s is lesser than or equal'
        ' to one the it is interpreted as the fractional size.'
    )

    parser.add_argument(
        '--random-sample', '-r', type=float, default=None, metavar='N',
        help='If %(metavar)s is greater than 1 then use only %(metavar)s '
        'random entries from the input bayes.dat. If %(metavar)s is lesser '
        ' than or equal than one the it is interpreted as the fractional size.'
        ' If the parameter --tail is used, then the random sample is drawn '
        'from the trimmed version of bayes.dat'
    )

    parser.add_argument(
        '--n-jobs', '-j', type=int, default=4, metavar='N',
        help='Set the number of cuncurrent lenstool instances to run. '
        'The default value is %default)s=%(metavar)s.'
    )

    parser.add_argument(
        '--verbose', '-v', action='store_true', default=False,
        help='increase the verbosity of the output.'
    )

    parser.add_argument(
        '--no-tempfs', action='store_true', default=False,
        help='Do not use /tmp to store temporary files, instead use a '
        'temporary directory in the current work directory.'
    )

    args = parser.parse_args(options)

    bayesMapPP(
        args.bayesfile,
        args.parfile,
        args.outdir,
        verbose=args.verbose,
        tail=args.tail,
        random_sample=args.random_sample,
        n_jobs=args.n_jobs,
        no_tempfs=args.no_tempfs
    )


def execBayesCubePP(options=None):
    """
    Execute bayesCubePP.

    Parameters
    ----------
    options : list, optional
        List of arguments, if None argumets are read from command line.
        The default is None.

    Returns
    -------
    None.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'bayesmapdir', type=str,
        help='The path of a directory generated by bayesMapPP'
    )
    args = parser.parse_args(options)
    bayesCubePP(args.bayesmapdir)


def execBayesEinRadPP(options=None):
    """
    Execute bayesEinRadPP.

    Parameters
    ----------
    options : list, optional
        List of arguments, if None argumets are read from command line.
        The default is None.

    Returns
    -------
    None.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'bayesmapdir', type=str,
        help='The path of a directory generated by bayesMapPP'
    )

    parser.add_argument(
        '--verbose', '-v', action='store_true', default=False,
        help='Increase the verbosity of the output'
    )

    parser.add_argument(
        '--check-images', '-c', type=str, metavar='PATH', default=False,
        nargs='?', help='If specified a series of check images are generated.'
        '%(metavar)s is the path where the images are saved, it will be '
        'created if it does not exist. If %(metavar)s is not specified, the '
        'current working directory is used.'
    )

    parser.add_argument(
        '--overlay-image', '-o,', type=str, metavar='FITS_FILE', default=None,
        help='Specify an optional image to be overlayd with the plots.'
    )

    parser.add_argument(
        '--jobs', '-j', type=int, metavar='N_JOBS', default=4,
        help='Specify how many jobs to run concurrently. The default is '
        '%(metavar)s=%(default)s'
    )
    args = parser.parse_args(options)

    ov_img = None
    ov_img_wcs = None
    if args.overlay_image is not None:
        ov_img_path = os.path.realpath(args.overlay_image)
        if not os.path.isfile(args.overlay_image):
            print(
                f"WARINING: image {ov_img_path} does not exist or is invalid.",
                file=sys.stderr
            )
        else:
            with fits.open(ov_img_path) as hdul:
                ov_img = hdul[0].data
                ov_img_wcs = WCS(hdul[0].header)

    if args.check_images is False:
        outdir = None
    elif args.check_images is None:
        outdir = '.'
    else:
        outdir = os.path.realpath(args.check_images)
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

    bayesEinRadPP(
        args.bayesmapdir,
        outdir=outdir,
        verbose=args.verbose,
        overlay_img=ov_img,
        overlay_img_wcs=ov_img_wcs,
        n_jobs=args.jobs
    )
