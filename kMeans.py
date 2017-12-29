#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
A K-means clustering program using MLlib.

This example requires NumPy (http://www.numpy.org/).
"""
from __future__ import print_function

import sys

import numpy as np
from pyspark import SparkContext
from pyspark.mllib.clustering import KMeans

#for giving us the time taken to execute the file
import time
start_time = time.time()



def parseVector(line):
    return np.array([float(x) for x in line.split(',')])


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: kmeans <file> <k>", file=sys.stderr)
        exit(-1)
    # initialize the SparkContext, the core of spark programming
    sc = SparkContext(appName="KMeans")
    # loading our data file given by the user
    lines = sc.textFile(sys.argv[1])
    # parsing the data in spark using map function
    data = lines.map(parseVector)
    # we read the number of clusters wanted by the user
    k = int(sys.argv[2])
    # now we can easily call the Kmeans function in MLlib
    model = KMeans.train(data, k)
    # outputting the cluster centers for us
    print("Final centers: " + str(model.clusterCenters))
    #job is done so we stop our SparkContext
    sc.stop()
print("---------%s seconds" %(time.time() - start_time))
