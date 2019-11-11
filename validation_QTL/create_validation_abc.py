import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sys import exit
import sys
from scipy import stats
from collections import Counter
import statsmodels.stats.multitest as multi
from math import e
import os
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sklearn.linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

wd = '/pbld/mcg/lillianpetersen/ABC/'
wdvars = '/pbld/mcg/lillianpetersen/ABC/saved_variables/'
wdfigs = '/pbld/mcg/lillianpetersen/ABC/figures/'
wddata = '/pbld/mcg/lillianpetersen/ABC/data/'
wdfiles = '/pbld/mcg/lillianpetersen/ABC/written_files/'
MakePlots = False

nSamples = 92
nChr = 23

###################################################################
# Load ATAC
###################################################################
print('Load ATAC')
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	if not 'atac'+ichr in globals():
		vars()['atac'+ichr] = np.load(wdvars+'validation/ATAC/atac'+ichr+'.npy')
		vars()['chrATAC'+ichr] = np.load(wdvars+'validation/ATAC/chrATAC'+ichr+'.npy')
		vars()['peakName'+ichr] = np.load(wdvars+'validation/ATAC/peakName'+ichr+'.npy')
		vars()['positionATAC'+ichr] = np.load(wdvars+'validation/ATAC/positionATAC'+ichr+'.npy')
		vars()['nPeaks'+ichr] = np.load(wdvars+'validation/ATAC/nPeaks'+ichr+'.npy')

###################################################################
# Load HiC
###################################################################
print('load HiC')
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	if not 'hic'+ichr in globals():
		# Load arrays saved from load_validation_hic.npy
		vars()['hic'+ichr] = np.load( wdvars+'validation/HiC/hic'+ichr+'.npy')
		vars()['geneStart'+ichr] = np.load( wdvars+'validation/HiC/geneStart'+ichr+'.npy' )
		vars()['geneMatrix'+ichr] = np.load( wdvars+'validation/HiC/geneMatrix'+ichr+'.npy' )
		vars()['peakStart'+ichr] = np.load( wdvars+'validation/HiC/peakStart'+ichr+'.npy' )
		vars()['peakPos'+ichr] = np.load( wdvars+'validation/HiC/peakPos'+ichr+'.npy' )
		vars()['peakMatrix'+ichr] = np.load( wdvars+'validation/HiC/peakMatrix'+ichr+'.npy' )

		if np.amin(np.isin(vars()['peakMatrix'+ichr], vars()['peakName'+ichr])) == False:
			print 'ERROR: peaks dont match'
			exit()
		if np.amin(np.isin(vars()['peakName'+ichr], vars()['peakMatrix'+ichr])) == False:
			indices = np.isin(vars()['peakName'+ichr], vars()['peakMatrix'+ichr])
			vars()['peakName'+ichr] = vars()['peakName'+ichr][indices]
			vars()['atac'+ichr] = vars()['atac'+ichr][:,indices]
			vars()['chrATAC'+ichr] = vars()['chrATAC'+ichr][indices]
			vars()['positionATAC'+ichr] = vars()['positionATAC'+ichr][:,indices]
			vars()['nPeaks'+ichr] = len(vars()['peakName'+ichr])

############### Fix Indexing ###############
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	if np.amin(vars()['peakName'+ichr]==vars()['peakMatrix'+ichr]) == 0:
		print 'Fixing Indexing because 2 peaks have same start position: Chromosome '+ichr
		indices = np.where(vars()['peakName'+ichr]!=vars()['peakMatrix'+ichr])[0]
		argsortATAC = np.arange(len(vars()['peakName'+ichr]))
		argsortATAC[indices[0]] = indices[1]
		argsortATAC[indices[1]] = indices[0]

		vars()['atac'+ichr] = vars()['atac'+ichr][:,argsortATAC]
		vars()['chrATAC'+ichr] = vars()['chrATAC'+ichr][argsortATAC]
		vars()['peakName'+ichr] = vars()['peakName'+ichr][argsortATAC]
		vars()['positionATAC'+ichr] = vars()['positionATAC'+ichr][:,argsortATAC]

		np.save(wdvars+'validation/ATAC/atac'+ichr+'.npy', vars()['atac'+ichr])
		np.save(wdvars+'validation/ATAC/chrATAC'+ichr+'.npy', vars()['chrATAC'+ichr])
		np.save(wdvars+'validation/ATAC/peakName'+ichr+'.npy', vars()['peakName'+ichr])
		np.save(wdvars+'validation/ATAC/positionATAC'+ichr+'.npy', vars()['positionATAC'+ichr])
	
###################################################################
# Create ABC Matrix
###################################################################
print 'Create ABC'
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
#for ichr in ['21']:
	if not 'abc'+ichr in globals():
		print ichr,
		if np.amin(vars()['peakName'+ichr]==vars()['peakMatrix'+ichr]) == 0:
			print 'Error: Peak array sizes do not match'
			exit()

		nGenes = vars()['hic'+ichr].shape[0]
		nPeaks = vars()['hic'+ichr].shape[1]
		atacMean = np.mean(vars()['atac'+ichr],axis=0)
		vars()['abc'+ichr] = np.zeros(shape = (nGenes, nPeaks))
		peakPos = np.mean( vars()['positionATAC'+ichr][:,:],axis=0)
		for igene in np.arange(nGenes):
			genePos = vars()['geneStart'+ichr][igene]
			usePeak = np.abs(peakPos-genePos)<1500000 # True = good = within 1.5Mb
			Sum = np.sum( atacMean[usePeak] * vars()['hic'+ichr][igene,usePeak])
			vars()['abc'+ichr][igene,usePeak] = (atacMean[usePeak] * vars()['hic'+ichr][igene,usePeak]) / Sum
			if np.amax(vars()['abc'+ichr][igene,:]>1)==True: exit()
		np.save(wdvars+'validation/ABC/abc'+ichr+'.npy', vars()['abc'+ichr])
		vars()['abc'+ichr] = np.ma.masked_array(vars()['abc'+ichr],vars()['abc'+ichr]==0)

		if MakePlots:
			plt.clf()
			fig = plt.figure(figsize = (10,6))
			plt.imshow(vars()['abc'+ichr][0], cmap = 'hot_r', aspect='auto', interpolation='none',origin='lower', vmax=30)
			plt.title('ABC Matrix: MCG001 Chromosome '+ichr,fontsize=18)
			plt.xlabel('Peaks')
			plt.ylabel('Genes')
			plt.grid(True)
			plt.colorbar()
			plt.savefig(wdfigs+'abc_validation_MCG001_Chr'+ichr+'.pdf')
