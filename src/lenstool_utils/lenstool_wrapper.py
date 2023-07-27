#!/usr/bin/env python
"""A simple wrapper for lenstool."""
import os
import sys
import time
import queue
import datetime
import argparse
import platform
import subprocess
from threading import Thread
from fcntl import fcntl, F_GETFL, F_SETFL

from . import datamodel


def find_prog(prog):
    """specex-zeropointinfo = "specex.zeropoints:main"
specex-cubestack = "specex.stack:cube_stack"
specex-cutout = "specex.cube:cutout_main"
specex-plot = "specex.plot:plot"

    Find the path of a pregram in your PATH.

    Parameters
    ----------
    prog : str
        Name of the program.

    Returns
    -------
    str
        The path of the program.

    """
    cmd = "where" if platform.system() == "Windows" else "which"
    try:
        return subprocess.check_output([cmd, prog]).strip().decode()
    except subprocess.CalledProcessError:
        return ""


def printpbar(partial, total=None, wid=32):
    """
    Return a nice text/unicode progress.

    eturn a nice text/unicode progress bar showing
    partial and total progress

    Parameters
    ----------
    partial : float
        Partial progress expressed as decimal value.
    total : float, optional
        Total progress expresses as decimal value.
        If it is not provided or it is None, than
        partial progress will be shown as total progress.
    wid : TYPE, optional
        Width in charachters of the progress bar.
        The default is 32.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    wid -= 2
    prog = int((wid)*partial)
    if total is None:
        total_prog = prog
        common_prog = prog
    else:
        total_prog = int((wid)*total)
        common_prog = min(total_prog, prog)
    pbar_full = '\u2588'*common_prog
    pbar_full += '\u2584'*(total_prog - common_prog)
    pbar_full += '\u2580'*(prog - common_prog)
    return (f"\u2595{{:<{wid}}}\u258F").format(pbar_full)


class LensToolWrapper():
    """A simple lenstool wrapper."""

    def __init__(self, lenstool_path=None, n_threads=-1):
        if lenstool_path is None:
            self.lenstool_exe = find_prog('lenstool')
        else:
            self.lenstool_exe = lenstool_path

        self._data_queue = None
        self._data_thread = None
        self._lenstool_popen = None
        self._start_time = time.time()
        self.old_state = None

        if n_threads <= 0:
            self.n_threads = os.cpu_count()
        else:
            self.n_threads = n_threads

        self.state = None
        self.global_progress = 0.0
        self.partial_progress = 0.0
        self.eta = '--'

    def parsedataline(self, line):
        """
        Parse a single data line.

        Parameters
        ----------
        line : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        values = line.split()
        out = ""
        status_string = values[0].lower().strip()
        if status_string == 'burn-in':
            self.state = 'Burn-in'
            self.global_progress = 0.5
            self.partial_progress = float(values[2])
        elif status_string == 'sampling':
            self.state = 'Sampling'
            cur, tot = values[3].split('/')
            self.global_progress = 1.0
            self.partial_progress = float(cur) / float(tot)
        elif (status_string == 'info:') and (values[1] == 'Compute'):
            self.state = 'Computing'
            cur, tot = values[-1].split('/')
            self.global_progress = float(cur) / float(tot)
            self.partial_progress = float(cur) / float(tot)
        else:
            out = line

        if self.partial_progress <= 0:
            self.eta = '--'
        else:
            eta = (time.time() - self._start_time) / self.partial_progress
            eta *= (1 - self.partial_progress)

            self.eta = f"{datetime.timedelta(seconds=eta)}".split('.')[0]

        if self.state != self.old_state:
            self._start_time = time.time()
            self.old_state = self.state
        return out

    def parsedata(self):
        """
        Parse data from lenstool output.

        Returns
        -------
        None.

        """
        if self._data_queue is None:
            return

        unparsed_out = []
        while not self._data_queue.empty():
            line = self._data_queue.get_nowait()
            out = self.parsedataline(line.decode().strip())
            if out:
                unparsed_out.append(out)

        return '\n'.join(unparsed_out)

    def run(self, param_file='default.par'):
        """
        Run lenstool and grab its output.

        See https://stackoverflow.com/questions/375427/a-non-blocking-read-on-a-subprocess-pipe-in-python

        Parameters
        ----------
        param_file : str, optional
            DESCRIPTION. The default is 'default.par'.

        Returns
        -------
        None.

        """
        ON_POSIX = 'posix' in sys.builtin_module_names

        # Check param file syntax
        pdata_model = datamodel.ParametersDataModel()
        pdata_model.read(param_file)

        def enqueue_output(p, queue):
            """
            Worker that reads data from an stdout/stderr.

            Parameters
            ----------
            p : TYPE
                DESCRIPTION.
            queue : TYPE
                DESCRIPTION.

            Returns
            -------
            line : TYPE
                DESCRIPTION.

            """
            def parse_buff(buff, stdfile):
                if p.poll() is not None:
                    return b""

                new_data = stdfile.read()

                if queue.full():
                    del new_data
                    return b""

                if new_data:
                    buff += new_data.replace(b'\r', b'\n')

                for line in buff.split(b'\n'):
                    line = line.strip()
                    if line:
                        queue.put(line)
                return line

            out_buff = b""
            err_buff = b""

            while p.poll() is None:

                if queue.full():
                    out_buff=b""
                    err_buff=b""

                out_buff += parse_buff(out_buff, p.stdout)
                err_buff += parse_buff(err_buff, p.stderr)
                time.sleep(0.5)

        my_env = os.environ.copy()
        my_env["EDITOR"] = ""
        my_env['OMP_NUM_THREADS'] = f"{self.n_threads:d}"
        param_file = os.path.realpath(param_file)
        model_dir = os.path.dirname(param_file)
        param_file = os.path.relpath(param_file, model_dir)
        self._lenstool_popen = subprocess.Popen(
            [self.lenstool_exe, param_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=32,
            close_fds=ON_POSIX,
            env=my_env,
            cwd=model_dir
        )

        flags = fcntl(self._lenstool_popen.stdout, F_GETFL)
        fcntl(self._lenstool_popen.stdout, F_SETFL, flags | os.O_NONBLOCK)

        flags = fcntl(self._lenstool_popen.stderr, F_GETFL)
        fcntl(self._lenstool_popen.stderr, F_SETFL, flags | os.O_NONBLOCK)

        # Thread to read program outputs
        self._data_queue = queue.Queue(maxsize=100)
        self._data_thread = Thread(
            target=enqueue_output,
            args=(
                self._lenstool_popen,
                self._data_queue
            )
        )

        # The thread dies with the program
        self._data_thread.daemon = True
        self._data_thread.start()

    def kill(self):
        """
        Kill the current lenstool process.

        Returns
        -------
        None.

        """
        while self._lenstool_popen is None:
            self._lenstool_popen.kill()
            time.sleep(0.1)


def main():
    """
    Run the main program.

    Returns
    -------
    None.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("paramfile", type=str)
    parser.add_argument(
        "--threads", '-t', type=int, default=os.cpu_count(), metavar='N',
        help="Set the number of threads to use. Default is "
        "%(metavar)s=%(default)d."
    )

    args = parser.parse_args()

    if not os.path.isfile(args.paramfile):
        print(f"\nError: input file {args.paramfile} does not exist!\n")
        sys.exit(1)

    lt_wrap = LensToolWrapper(n_threads=args.threads)
    if not os.path.exists(lt_wrap.lenstool_exe):
        print("\nError: cannot find lenstool executable in your PATH!\n")
        sys.exit(1)

    start_time = time.perf_counter()
    lt_wrap.run(args.paramfile)

    print("\n")
    while lt_wrap._lenstool_popen.poll() is None:
        out = lt_wrap.parsedata()
        if out:
            print(out)
        pbar = printpbar(lt_wrap.partial_progress, lt_wrap.global_progress)
        time.sleep(0.01)
        print(
            f"\r{lt_wrap.state}: {pbar} {lt_wrap.partial_progress: 6.2%} "
            f"   ETA: {lt_wrap.eta: <16s}",
            end='\r'
        )
    delta_t = time.perf_counter() - start_time
    print(f"\n--- done in {delta_t:.2} sec. ---")


if __name__ == '__main__':
    main()
