##########################################################################
# Copyright (c) 2024 Reinhard Caspary     python clean.py                               #
# <reinhard.caspary@phoenixd.uni-hannover.de>                            #
# This program is free software under the terms of the MIT license.      #
##########################################################################

import glob
import shutil

globs = ["./build/", "./dist/", "./*.egg-info/", "./**/__pycache__/"]

for pattern in globs:
    for path in glob.iglob(pattern):
        shutil.rmtree(path)

