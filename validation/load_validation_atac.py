import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sys import exit
from scipy import stats
from collections import Counter
import statsmodels.stats.multitest as multi
from math import e
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sklearn.linear_model

wd = '/pbld/mcg/lillianpetersen/ABC/'
wdvars = '/pbld/mcg/lillianpetersen/ABC/saved_variables/'
wdfigs = '/pbld/mcg/lillianpetersen/ABC/figures/'
MakePlots = False

#nSamples = 92

###################################################################
# Load ATAC
###################################################################
print('Load ATAC')
atacFile = np.swapaxes(np.array(pd.read_csv(wd+'data/validation_K562/atac_peaks_K562.rpm.narrowPeak', sep = '\t', header = None)),0,1)
chrATAC = atacFile[0]
positionATAC = np.array(atacFile[1:3],dtype=int)
atac = atacFile[3]
nPeaks = len(positionATAC[0])

peakName = np.zeros(shape=nPeaks, dtype=object)
for i in range(nPeaks):
	peakName[i] = '_'.join([chrATAC[i],str(positionATAC[0,i]),str(positionATAC[1,i])])

if MakePlots:
	plt.clf()
	n, bins, patches = plt.hist(np.amax(atacFull,axis=0), bins=100, range=[0,6])
	plt.title('Histogram of ATAC Peak Intensity (nPeaks = '+str(nPeaks)+')')
	plt.xlabel('rpm ATAC')
	plt.ylabel('Number of Peaks')
	#plt.xlim([0,3])
	#plt.ylim([0,600])
	plt.grid(True)
	plt.show()

# remove alt chromosomes
#keepAlt = chrATAC=='alt'
#keepFull = np.zeros(shape = (nSamples,len(keepAlt)), dtype = bool)
#for isample in range(nSamples):
#	keepFull[isample] = keepAlt
#keep2 = np.zeros(shape = (2,len(keepAlt)), dtype = bool)
#for isample in range(2):
#	keep2[isample] = keepAlt
#atac = np.ma.compress_cols(np.ma.masked_array(atacFull,keepFull))
#chrATAC = np.ma.compressed(np.ma.masked_array(chrATAC,keepAlt))
#peakName = np.ma.compressed(np.ma.masked_array(peakName,keepAlt))
#positionATAC = np.ma.compress_cols(np.ma.masked_array(positionATAC,keep2))
#lengthATAC = positionATAC[1] - positionATAC[0]
#nPeaks = len(chrATAC)

######### sort ATAC #########
# split into a different array for each chromosome, sorted by peak start within each
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y']:
	vars()['atac'+ichr] = atac[(chrATAC=='chr'+ichr)]
	vars()['chrATAC'+ichr] = chrATAC[(chrATAC=='chr'+ichr)]
	vars()['peakName'+ichr] = peakName[(chrATAC=='chr'+ichr)]
	vars()['positionATAC'+ichr] = positionATAC[:,(chrATAC=='chr'+ichr)]
	vars()['nPeaks'+ichr] = vars()['atac'+ichr].shape[0]

	sort = np.argsort(vars()['positionATAC'+ichr][0])
	vars()['atac'+ichr] = vars()['atac'+ichr][sort]
	vars()['chrATAC'+ichr] = vars()['chrATAC'+ichr][sort]
	vars()['peakName'+ichr] = vars()['peakName'+ichr][sort]
	vars()['positionATAC'+ichr] = vars()['positionATAC'+ichr][:,sort]

	if not os.path.exists(wdvars+'validation_K562/ATAC'):
		os.makedirs(wdvars+'validation_K562/ATAC')
	np.save(wdvars+'validation_K562/ATAC/atac'+ichr+'.npy', vars()['atac'+ichr])
	np.save(wdvars+'validation_K562/ATAC/chrATAC'+ichr+'.npy', vars()['chrATAC'+ichr])
	np.save(wdvars+'validation_K562/ATAC/peakName'+ichr+'.npy', vars()['peakName'+ichr])
	np.save(wdvars+'validation_K562/ATAC/positionATAC'+ichr+'.npy', vars()['positionATAC'+ichr])
	np.save(wdvars+'validation_K562/ATAC/nPeaks'+ichr+'.npy', vars()['nPeaks'+ichr])
############################

