"""Convert bayes.dat to fits format."""
from argparse import ArgumentParser

from . import bayes


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

    bayes_tbl = bayes.BayesDat.read(args.bayesfile)
    bayes_tbl.write(args.bayesfile + '.fits', format='fits', overwrite=True)
