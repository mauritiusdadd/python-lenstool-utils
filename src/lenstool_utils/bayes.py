"""Provides classes and functions to handle the results stored in bayes.dat."""
import os
import sys
import time
import shutil
import argparse
import tempfile
import warnings

import numpy as np

from tqdm import tqdm
from astropy.table import Table

from . import datamodel
from . import lenstool_wrapper

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

        for j, header_line in enumerate(tqdm(datalines[index:])):
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
               random_sample=None, verbose=False):
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
            j_tempd.cleanup()

    model_dir = os.path.dirname(os.path.realpath(par_file))
    map_out_dir = os.path.join(model_dir, out_dir)

    if not os.path.isdir(map_out_dir):
        os.makedirs(map_out_dir)

    bd = BayesDat.read(os.path.realpath(bayes_file))
    if tail:
        bd = bd.tail(tail)

    if random_sample:
        bd = bd.random_sample(random_sample)

    lt_max_processes = 10
    lt_processes = []

    print("\nRunning lenstool...", file=sys.stdout)
    for j in tqdm(range(len(bd.data))):
        while len(lt_processes) >= lt_max_processes:
            _loop(lt_processes)

        tempd = tempfile.TemporaryDirectory()

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

    print("Waiting for the remaing processes to finish...")
    while lt_processes:
        _loop(lt_processes)


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
        'bayesfile', type=str, help='The pat of a bayes.dat'
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
        '--verbose', '-v', action='store_true', default=False,
        help='increase the verbosity of the output.'
    )
    args = parser.parse_args(options)

    bayesMapPP(
        args.bayesfile,
        args.parfile,
        args.outdir,
        verbose=args.verbose,
        tail=args.tail,
        random_sample=args.random_sample
    )
