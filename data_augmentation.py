import argparse


def parse_command_line():
    parser = argparse.ArgumentParser('Yu-Gi-Oh! Dataset splitter testing parser', add_help=True)

    return parser.parse_args()