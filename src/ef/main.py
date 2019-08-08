#!/usr/bin/env python3

import argparse
import re
import sys
from argparse import ArgumentTypeError
from configparser import ConfigParser

import h5py

from ef.config.components import OutputFileConf
from ef.config.config import Config
from ef.field.solvers import pyamg
from ef.output.reader import Reader
from ef.runner import Runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_or_h5_file", help="config or h5 file", type=guess_input_type)
    parser.add_argument("--output-format", default="cpp", help="select output hdf5 format",
                        choices=["python", "cpp", "history", "none"])
    parser.add_argument("--prefix", help="customize output file prefix")
    parser.add_argument("--suffix", help="customize output file suffix")
    parser.add_argument("--solver", default="amg", help="select field solving library",
                        choices=["amg", "amgx"])

    args = parser.parse_args()

    is_config, parser_or_h5_filename = args.config_or_h5_file
    if args.solver == 'amg':
        solver_class = pyamg.FieldSolverPyamg
    elif args.solver == 'amgx':
        from ef.field.solvers import pyamgx
        solver_class = pyamgx.FieldSolverPyamgx
    if is_config:
        conf = read_conf(parser_or_h5_filename, args.prefix, args.suffix, args.output_format)
        sim = conf.make()
        writer = conf.output_file.make()
        Runner(sim, solver_class(sim.mesh, sim.inner_regions), writer).start()
    else:
        print("Continuing from h5 file:", parser_or_h5_filename)
        prefix, suffix = merge_h5_prefix_suffix(parser_or_h5_filename, args.prefix, args.suffix)
        print("Using output prefix and suffix:", prefix, suffix)
        with h5py.File(parser_or_h5_filename, 'r') as h5file:
            sim = Reader.read_simulation(h5file)
        writer = OutputFileConf(prefix, suffix, args.output_format).make()
        Runner(sim, solver_class(sim.mesh, sim.inner_regions), writer).continue_()
    del sim
    return 0


def guess_input_type(file_name):
    parser = ConfigParser()
    if file_name == '-':
        print("Reading config from stdin:", file_name)
        parser.read_file(sys.stdin)
        return True, parser

    print("Trying to guess input file type:", file_name)
    try:
        h5file = h5py.File(file_name, 'r')
        h5file.close()
        return False, file_name
    except Exception as err:
        h5_error = err
    try:
        with open(file_name, 'r') as file:
            parser.read_file(file)
        return True, parser
    except Exception as err:
        conf_error = err
    raise ArgumentTypeError("can't interpret file '{}' as either h5:\n\t{}\nor config:\n\t{}"
                            .format(file_name, h5_error, conf_error))


def read_conf(parser, prefix, suffix, format_):
    conf = Config.from_configparser(parser)
    if prefix:
        conf.output_file.prefix = prefix
    if suffix:
        conf.output_file.suffix = suffix
    conf.output_file.format_ = format_
    print(conf)
    return conf


def merge_h5_prefix_suffix(h5_filename, prefix, suffix):
    if prefix is None and suffix is None:
        try:
            prefix, suffix = extract_prefix_and_suffix_from_h5_filename(h5_filename)
        except ValueError:
            print("Can't extract output prefix and suffix from ", h5_filename)
            print("Using fallback")
            prefix = h5_filename
    if prefix is None:
        prefix = ""
    if suffix is None:
        suffix = ".h5"
    return prefix, suffix


def extract_prefix_and_suffix_from_h5_filename(h5_filename):
    rgx = r"\d{7}|history"  # search for timestep in filename (7 digits in a row)
    match = re.search(rgx, h5_filename)
    if match:
        return h5_filename[0:match.start()], h5_filename[match.end():]
    else:
        raise ValueError("Can't extract prefix and suffix from h5 filename.")


if __name__ == "__main__":
    main()
