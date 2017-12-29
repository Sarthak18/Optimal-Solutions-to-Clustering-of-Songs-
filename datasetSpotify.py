import pandas as p
import numpy as np
from scipy.cluster.vq import vq,kmeans2,whiten

#import matplotlib.pyplot as plt
import matplotlib as mpl

import time
start_time = time.time()



dataf = p.read_csv('Now_with_Spotify_Final_new.csv')
dataf.head()

wanted_data = dataf.as_matrix(columns=['speechiness','liveness','loudness','danceability','acousticness','energy','tempo'])

numb_points = len(wanted_data)

white = whiten(wanted_data)

centroids, labels = kmeans2(white, 7, iter=20)

print numb_points
print(len(centroids))
print (len(np.unique(labels)))

# print(records)

# p.save_as(records=[records[0]], dest_file_name="finaldata.csv")
print time.time() - start_time
