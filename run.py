#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""This script will automatically run tests for the K-Means clustering program."""

__author__ = "Pei Xu, xuxx0884@umn.edu"
__copyright__ = "Copyright 2016, Pei Xu"
__license__ = "MIT"
__version__ = "1.0"
__date__ = "21:20:30, Nov. 21th, 2016"

import subprocess
import platform
import os
import time
import data as token

N_CLUSTERS = [20, 40, 60, 65, 80]

TOKEN_FOLDER = token.TARGET_FOLDER

DATA_FILE = [token.WORD_BAG_FILE] + \
    [token.N_GRAM_FILE.replace('<n>', str(n)) for n in token.N_GRAM]

CLASS_FILE = token.CLASS_FILE + token.CLASS_FILE_EXT

OUTPUT_FOLDER = "log"

OUTPUT_FILE_PREFIX = "result"

LOG_FILE_PREFIX = "log"

PROGRAM_NAME = "sphkmeans"


def run():
    token_dir = os.path.join(os.getcwd(), TOKEN_FOLDER)
    class_file = os.path.join(token_dir, CLASS_FILE)
    output_dir = os.path.join(os.getcwd(), OUTPUT_FOLDER)
    sys_platform = platform.system()
    if sys_platform == "Windows":
        program = os.path.join(os.getcwd(), PROGRAM_NAME + ".exe")
    else:
        program = os.path.join(os.getcwd(), PROGRAM_NAME)
    # subprocess.call(["make", "kmeans"])
    # print("Extracting tokens...")
    # os.system("python ./data.py")
    print("Runing test...")
    for k in N_CLUSTERS:
        for d in DATA_FILE:
            input_file = os.path.join(
                token_dir, d + token.TOKEN_FILE_EXT)
            outfile_suf = time.strftime("%H-%M-%S_%m-%d-%y", time.localtime())
            log_file = os.path.join(output_dir, "%s_%d_%s_%s" % (
                LOG_FILE_PREFIX, k, d, outfile_suf))
            out_file = os.path.join(output_dir, "%s_%d_%s_%s" % (
                OUTPUT_FILE_PREFIX, k, d, outfile_suf))
            f = open(log_file, 'w')
            p = subprocess.Popen([program, input_file, class_file, str(k),
                                  str(20), out_file])
            for line in p.stderr:
                print(line)
                f.write(str(line))
            break
        break
if __name__ == "__main__":
    introduction = "This is an autoscript to run a group of tests for the K-means program.\nDefaultly, it will test the following combaination of parameters:\n  Data Model: bag-of-words, 3-gram, 5-gram, 7-gram\n  # of Clusters: 20, 40, 60, 65, 80\n  Random Seed: odd numbers in the range of [0, 40]"
    print(introduction)
    run()
