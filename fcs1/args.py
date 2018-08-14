import argparse
import os

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("This file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file

def get_args():
    """"""
    parser = argparse.ArgumentParser(
        description="FCS fitting check",
        epilog="Fits and shows the residuals with fitting parameters"
    )

    # either -s or -m required argument
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-s', action="store", required=False,
                       help='Single filename for analysis')
    group.add_argument('-m', action="store", required=False,
                       help='First file of Multiple files for analysis')
    parser.add_argument("--list",
                        help="Show file list available for analysis.",
                        action="store_true")
    # print(parser.parse_args().s)
    # print(parser.parse_args().m)
    return parser.parse_args()

