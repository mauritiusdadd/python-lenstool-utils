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

    myt = Table(names=colnames)
    print(colnames)
    print(myt)

    for j, header_line in enumerate(tqdm(datalines[index:])):
        data = header_line.strip().split()
        myt.add_row(data)

    myt.write(args.bayesfile + '.fits', overwrite=True)


if __name__ == '__main__':
    main()
