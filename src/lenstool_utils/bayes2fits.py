import numpy as np

from argparse import ArgumentParser
from tqdm import tqdm
from astropy.table import Table


def main(options=None):
    """
    Convert a bayes.dat file into a fits file.

    Parameters
    ----------
    options : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    parser = ArgumentParser()
    parser.add_argument(
        "bayesfile", metavar='BAYES_DAT', help='The input bayes dat file.'
    )

    args = parser.parse_args(options)

    with open(args.bayesfile, 'r') as f:
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
    print(f"Found {len(colnames)} columns...")

    for j, header_line in enumerate(tqdm(datalines[index:])):
        data = header_line.strip().split()
        try:
            myt[j] = data
        except Exception as exc:
            print(f"ERROR on line {j + index + 1}:", str(exc))
            print(f"N COLS: {len(colnames)}")
            print(f"N VALS: {len(data)}")

    myt.write(args.bayesfile + '.fits', overwrite=True)


if __name__ == '__main__':
    main(["/home/daddona/dottorato/Y1/muse_cubes/PLCKG287/lenstool/v2/v2.2_experiments/v2.2.7_gold_v11.0_3h_runmode3/results_runmode3/bayes.dat"])
