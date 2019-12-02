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

# MCG B ALL Samples
MCGs = ['MCG001', 'MCG002', 'MCG003', 'MCG005', 'MCG006', 'MCG009', 'MCG010', 'MCG011', 'MCG012', 'MCG013', 'MCG016', 'MCG019', 'MCG020', 'MCG023', 'MCG024', 'MCG027', 'MCG028', 'MCG029', 'MCG034', 'MCG035', 'MCG036', 'MCG037', 'MCG038', 'MCG039']
nSamples = len(MCGs)

###################################################################
# Load ATAC
###################################################################
print('Load ATAC')
atacFile = np.swapaxes(np.array(pd.read_csv(wd+'data/B_ALL_merged_counts.rpkm.qn.txt', sep = '\t', header = None)),0,1)

peakName = atacFile[0]
nPeaks = len(peakName)
positionATAC = np.zeros(shape = (2,nPeaks))
chrATAC = []
for ipeak in range(nPeaks):
	postmp = peakName[ipeak].split('_')
	if np.amax(np.array(postmp)=='alt')==1:
		chrATAC.append('alt')
		continue
	chrATAC.append(postmp[0])
	positionATAC[:,ipeak] = postmp[1:]
chrATAC = np.array(chrATAC)

atacFull = np.zeros(shape = (nSamples,nPeaks)) # ATAC for every peak
for isample in range(nSamples):
	atacFull[isample] = atacFile[isample+1]

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
keepAlt = chrATAC=='alt'
keepFull = np.zeros(shape = (nSamples,len(keepAlt)), dtype = bool)
for isample in range(nSamples):
	keepFull[isample] = keepAlt
keep2 = np.zeros(shape = (2,len(keepAlt)), dtype = bool)
for isample in range(2):
	keep2[isample] = keepAlt
atac = np.ma.compress_cols(np.ma.masked_array(atacFull,keepFull))
chrATAC = np.ma.compressed(np.ma.masked_array(chrATAC,keepAlt))
peakName = np.ma.compressed(np.ma.masked_array(peakName,keepAlt))
positionATAC = np.ma.compress_cols(np.ma.masked_array(positionATAC,keep2))
lengthATAC = positionATAC[1] - positionATAC[0]
nPeaks = len(chrATAC)

######### sort ATAC #########
# split into a different array for each chromosome, sorted by peak start within each
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y']:
	vars()['atac'+ichr] = atac[:,(chrATAC=='chr'+ichr)]
	vars()['chrATAC'+ichr] = chrATAC[(chrATAC=='chr'+ichr)]
	vars()['peakName'+ichr] = peakName[(chrATAC=='chr'+ichr)]
	vars()['positionATAC'+ichr] = positionATAC[:,(chrATAC=='chr'+ichr)]
	vars()['nPeaks'+ichr] = vars()['atac'+ichr].shape[1]

	np.save(wdvars+'ATAC/atac'+ichr+'.npy', vars()['atac'+ichr])
	np.save(wdvars+'ATAC/chrATAC'+ichr+'.npy', vars()['chrATAC'+ichr])
	np.save(wdvars+'ATAC/peakName'+ichr+'.npy', vars()['peakName'+ichr])
	np.save(wdvars+'ATAC/positionATAC'+ichr+'.npy', vars()['positionATAC'+ichr])
	np.save(wdvars+'ATAC/nPeaks'+ichr+'.npy', vars()['nPeaks'+ichr])
############################

