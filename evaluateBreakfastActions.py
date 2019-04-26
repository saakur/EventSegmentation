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

gtFiles = {}
for gtFilePath1 in listdir(gtFilePath):
	if not isdir(join(gtFilePath, gtFilePath1)):
		continue
	# print gtFilePath1
	for f in listdir(join(gtFilePath, gtFilePath1)):
		# print f
		if f.endswith('.coarse'):
			gtFiles[f.split('.')[0]] = join(gtFilePath, gtFilePath1, f)

avgFrames = []
avgBoundaries = []
avgClasses = []
VoI = []
ARI = []
avgIoD = []

fps=30
winSize = np.ceil(fps/2) // 2 * 2 + 1

print winSize
# print gtFiles
numVideos=0
for predFile in predFiles:
	# if 'cereal' not in predFile:
	# 	continue
	numVideos += 1
	vidKey = predFile.split('/')[-1].split('_')[:3]
	vidKey.pop(1)
	vidKey= '_'.join(vidKey)
	# print "\n\n", vidKey, predFile
	vidKey = vidKey.replace('salat', 'salad')
	vidKey = vidKey.replace('cereals', 'cereal')
	gtFile = gtFiles[vidKey]

	gtFrames = []
	predFrames = []
	predFrames1 = []
	predErrors = []
	avgFr = []
	classNo = 0
	BGClass = []
	actBoundary = []

	# print gtFile
	with open(gtFile, 'rb') as file:
		for line in file:
			# print line
			data, className = line.replace(' \n', '').replace('  ', '').split(' ')
			if className == 'SIL':
				BGClass.append(classNo)
			fromFrame,toFrame = [int(x) for x in data.split('-')]
			avgFr.append(toFrame - fromFrame)
			actBoundary.append(toFrame)
			# if  toFrame - fromFrame == 0:
			# 	print gtFile, line
			for i in range(fromFrame, toFrame+1):
				gtFrames.append(classNo)
			classNo += 1

	avgFrames.append(np.mean(avgFr))

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

	prevFrame = 0
	predClass = 0
	for nextFrame in predBoundaries:
		for i in range(prevFrame, nextFrame):
			predFrames.append(predClass)
		prevFrame = nextFrame
		predClass += 1

	if len(predFrames) < len(gtFrames):
		x = gtFrames[:len(predFrames)]
		y = predFrames
	else:
		x = gtFrames
		y = predFrames[:len(gtFrames)]


	TP = 0
	FP = 0
	TN = 0
	FN = 0
	ignoreClass = 0
	d = {x:0 for x in range(classNo)}
	for i in range(len(y)):
		if y[i] >= classNo:
			ignoreClass += 1
		if y[i] == x[i]:
			d[y[i]] += 1
			TP += 1
		else:
			FP += 1
	p = []
	for c,val in d.iteritems():
		detNo = sum([1 for i in y if i == c])
		if detNo == 0:
			continue
		p.append(1.0*val/detNo)
	p = np.mean(p)
	avgClasses.append(classNo)
	print predBoundaries, actBoundary, p
	
	avgIoD.append(p)


print "MoF", np.mean(avgIoD)
