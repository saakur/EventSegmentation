import numpy as np
from sys import argv, exit
from sklearn.metrics import jaccard_similarity_score, adjusted_rand_score, precision_score, f1_score, adjusted_rand_score, fowlkes_mallows_score
from scipy import signal
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join, isdir
from math import factorial
from collections import Counter
from math import log
from statsmodels.nonparametric.smoothers_lowess import lowess
import warnings

def warn(*args, **kwargs):
    pass
warnings.warn = warn

def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError, msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

predFilePath = argv[1]
gtFilePath = argv[2]


predFiles = sorted([join(predFilePath, f) for f in listdir(predFilePath) if isfile(join(predFilePath, f)) and f.endswith('.txt')])


fps=30
winSize = np.ceil(fps/2) // 2 * 2 + 1

numVideos=0
for predFile in predFiles:
	# if 'cereal' not in predFile:
	# 	continue
	numVideos += 1

	predFrames = []
	predFrames1 = []
	predErrors = []
	avgFr = []
	classNo = 0
	BGClass = []

	with open(predFile, 'rb') as file:
		lineNo = 0
		for line in file:
			data = line.replace('\n', '').split('\t')
			frameNo,predError = data
			frameNo, predError = int(frameNo), float(predError)
			predErrors.append(predError)
	# print predErrors

	
	predErrors_Ori = predErrors
	predErrors = movingaverage(predErrors, 80)
	predErrors = np.gradient(np.array(predErrors)).tolist()

	predBoundaries = signal.argrelextrema(np.array(predErrors), np.greater, order=int(0.57899*200))[0].tolist()
	predBoundaries.append(len(gtFrames)-1)

	outFile = predFile.replace('.txt', '_predBoundaries.txt')
	with open(outFile, 'w') as of:
		for p in predBoundaries:
			of.write('%d\n'%p)
print "Fin"