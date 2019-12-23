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
# Load Chip
###################################################################
print('Load Chip')
chipFile = np.swapaxes(np.array(pd.read_csv(wd+'data/validation_K562/H3K27ac_K562_hg38.rpkm.txt', sep = '\t', header = None)),0,1)
chrChip = chipFile[0]
positionChip = np.array(chipFile[1:3],dtype=int)
chip = chipFile[3]
nPeaks = len(positionChip[0])

peakName = np.zeros(shape=nPeaks, dtype=object)
for i in range(nPeaks):
	peakName[i] = '_'.join([chrChip[i],str(positionChip[0,i]),str(positionChip[1,i])])

if MakePlots:
	plt.clf()
	n, bins, patches = plt.hist(chip, bins=100) #, range=[0,6])
	plt.title('Histogram of Chip Peak Intensity (nPeaks = '+str(nPeaks)+')')
	plt.xlabel('rpkm Chip')
	plt.ylabel('Number of Peaks')
	#plt.xlim([0,3])
	#plt.ylim([0,600])
	plt.grid(True)
	plt.show()

# remove alt chromosomes
#keepAlt = chrChip=='alt'
#keepFull = np.zeros(shape = (nSamples,len(keepAlt)), dtype = bool)
#for isample in range(nSamples):
#	keepFull[isample] = keepAlt
#keep2 = np.zeros(shape = (2,len(keepAlt)), dtype = bool)
#for isample in range(2):
#	keep2[isample] = keepAlt
#chip = np.ma.compress_cols(np.ma.masked_array(chipFull,keepFull))
#chrChip = np.ma.compressed(np.ma.masked_array(chrChip,keepAlt))
#peakName = np.ma.compressed(np.ma.masked_array(peakName,keepAlt))
#positionChip = np.ma.compress_cols(np.ma.masked_array(positionChip,keep2))
#lengthChip = positionChip[1] - positionChip[0]
#nPeaks = len(chrChip)

######### sort Chip #########
# split into a different array for each chromosome, sorted by peak start within each
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y']:
	vars()['chip'+ichr] = chip[(chrChip=='chr'+ichr)]
	vars()['chrChip'+ichr] = chrChip[(chrChip=='chr'+ichr)]
	vars()['peakName'+ichr] = peakName[(chrChip=='chr'+ichr)]
	vars()['positionChip'+ichr] = positionChip[:,(chrChip=='chr'+ichr)]
	vars()['nPeaks'+ichr] = vars()['chip'+ichr].shape[0]
	exit()

	sort = np.argsort(vars()['positionChip'+ichr][0])
	vars()['chip'+ichr] = vars()['chip'+ichr][sort]
	vars()['chrChip'+ichr] = vars()['chrChip'+ichr][sort]
	vars()['peakName'+ichr] = vars()['peakName'+ichr][sort]
	vars()['positionChip'+ichr] = vars()['positionChip'+ichr][:,sort]

	if not os.path.exists(wdvars+'validation_K562/Chip'):
		os.makedirs(wdvars+'validation_K562/Chip')
	np.save(wdvars+'validation_K562/Chip/chip'+ichr+'.npy', vars()['chip'+ichr])
	np.save(wdvars+'validation_K562/Chip/chrChip'+ichr+'.npy', vars()['chrChip'+ichr])
	np.save(wdvars+'validation_K562/Chip/peakName'+ichr+'.npy', vars()['peakName'+ichr])
	np.save(wdvars+'validation_K562/Chip/positionChip'+ichr+'.npy', vars()['positionChip'+ichr])
	np.save(wdvars+'validation_K562/Chip/nPeaks'+ichr+'.npy', vars()['nPeaks'+ichr])
############################

