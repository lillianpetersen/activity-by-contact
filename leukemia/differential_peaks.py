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

wd = '/pbld/mcg/lillianpetersen/ABC/'
wdvars = '/pbld/mcg/lillianpetersen/ABC/saved_variables/'
wddata = '/pbld/mcg/lillianpetersen/ABC/data/'
wdfigs = '/pbld/mcg/lillianpetersen/ABC/figures/'
wdfiles = '/pbld/mcg/lillianpetersen/ABC/written_files/'
MakePlots = False

# MCG B ALL Samples
MCGs = ['MCG001', 'MCG002', 'MCG003', 'MCG005', 'MCG006', 'MCG009', 'MCG010', 'MCG011', 'MCG012', 'MCG013', 'MCG016', 'MCG017', 'MCG019', 'MCG020', 'MCG023', 'MCG024', 'MCG027', 'MCG028', 'MCG034', 'MCG035', 'MCG036', 'MCG037', 'MCG038', 'MCG039']
nSamples = len(MCGs)
nChr = 23

subtypes = np.array(['ETV6-RUNX1', 'DUX4', 'Hyperdiploid', 'PAX5', 'Ph-like'])
typeNames = np.array(['ETVRUNX', 'DUX', 'Hyperdiploid', 'PAX', 'Phlike'])
typeNamesFile = np.array(['ETV6_RUNX1', 'DUX4', 'High_hyperdiploid', 'PAX5alt', 'Ph_like'])

sampleTypes = np.array(['PAX5', 'ETV6-RUNX1', 'PAX5alt', 'DUX4', 'ZNF384', 'PAX5alt', 'Hyperdiploid', 'DUX4', 'Hyperdiploid', 'Ph-like', 'Ph-like', 'ETV6-RUNX1', 'Hyperdiploid', 'Hyperdiploid', 'DUX4', 'ETV6-RUNX1', 'Hyperdiploid', 'Hyperdiploid', 'Hyperdiploid', 'ETV6-RUNX1', 'DUX4', 'Other', 'Ph-like', 'Ph-like'])
# 0=ETV6-RUNX1, 1=DUX4, 2=Hyperdiploid, 3=PAX5, 4=Ph-like, 5=Other
typesIndex = np.array([3, 0, 3, 1, 5, 3, 2, 1, 2, 4, 4, 0, 2, 2, 1, 0, 2, 2, 2, 0, 1, 5, 4, 4])

MCGtypeDict = {
	'MCG001': 'PAX5', # PAX5
	'MCG002': 'ETV6-RUNX1', # ETV6_RUNX1
	'MCG003': 'PAX5alt', # PAX5
	'MCG005': 'DUX4', # DUX4
	'MCG006': 'ZNF384', # not included
	'MCG009': 'PAX5alt', # PAX5
	'MCG010': 'Hyperdiploid', # Hyperdiploid
	'MCG011': 'DUX4', # Hyperdiploid
	'MCG012': 'Hyperdiploid', # Hyperdiploid
	'MCG013': 'Ph-like', # Ph_like
	'MCG016': 'Ph-like', # Ph_like
	'MCG017': 'ETV6-RUNX1', # ETV6_RUNX1
	'MCG019': 'Hyperdiploid', # Hyperdiploid
	'MCG020': 'Hyperdiploid', # Hyperdiploid
	'MCG023': 'DUX4', # DUX4
	'MCG024': 'ETV6-RUNX1', # ETV6_RUNX1
	'MCG027': 'Hyperdiploid', # Hyperdiploid
	'MCG028': 'Hyperdiploid', # not included
	'MCG034': 'Hyperdiploid', # not included
	'MCG035': 'ETV6-RUNX1', # ETV6_RUNX1
	'MCG036': 'DUX4', # DUX4
	'MCG037': 'Other', # not included
	'MCG038': 'Ph-like', # Ph_like
	'MCG039': 'Ph-like' # Ph_like
	}

indexTypeDict = {0  : 'PAX5', 1  : 'ETV6-RUNX1', 2  : 'PAX5alt', 3  : 'DUX4', 4  : 'ZNF384', 5  : 'PAX5alt', 6  : 'Hyperdiploid', 7  : 'DUX4', 8  : 'Hyperdiploid', 9  : 'Ph-like', 10 : 'Ph-like', 11 : 'ETV6-RUNX1', 12 : 'Hyperdiploid', 13 : 'Hyperdiploid', 14 : 'DUX4', 15 : 'ETV6-RUNX1', 16 : 'Hyperdiploid', 17 : 'Hyperdiploid', 18 : 'Hyperdiploid', 19 : 'ETV6-RUNX1', 20 : 'DUX4', 21 : 'Other', 22 : 'Ph-like', 23 : 'Ph-like' }


difPeaksFile = pd.read_csv(wddata+'differential_peaks.csv', sep=',', header=0)

difPeaksSig = difPeaksFile[difPeaksFile['padj']<0.05]

for itype in range(len(subtypes)):
	subtype = typeNames[itype]
	subtypeName = subtypes[itype]
	print('\n\n\n'+'Differential Peaks: '+subtypeName)

	vars()['difPeaks'+subtype] = difPeaksSig[difPeaksSig['label']==typeNamesFile[itype]].reset_index()

	vars()['difPeaks'+subtype]['chr'] = vars()['difPeaks'+subtype]['peakID'].str.split('_').str.get(0)
	vars()['difPeaks'+subtype]['start'] = vars()['difPeaks'+subtype]['peakID'].str.split('_').str.get(1)
	vars()['difPeaks'+subtype]['stop'] = vars()['difPeaks'+subtype]['peakID'].str.split('_').str.get(2) 
	vars()['difPeaks'+subtype]['convert'] = -9999*np.ones(shape=len(vars()['difPeaks'+subtype]))

	for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
		vars()['positionActivity'+ichr] = np.array(np.load(wdvars+'ATAC/positionATAC'+ichr+'.npy'),dtype=int)

		vars()['peakPos'+ichr] = np.zeros(shape=(2,np.sum(vars()['difPeaks'+subtype]['chr']=='chr'+ichr)), dtype=int)
		vars()['peakPos'+ichr][0,:] = vars()['difPeaks'+subtype][vars()['difPeaks'+subtype]['chr']=='chr'+ichr]['start']
		vars()['peakPos'+ichr][1,:] = vars()['difPeaks'+subtype][vars()['difPeaks'+subtype]['chr']=='chr'+ichr]['stop']

		# create 1D array of each bp and what peak it matches
		chrLen = int(np.amax( [np.amax(vars()['positionActivity'+ichr][1]), np.amax(vars()['peakPos'+ichr][1])] ))
		peakPos = -9999*np.ones(shape=chrLen,dtype=int)
		for ipeak in range( len(vars()['positionActivity'+ichr][0]) ):
			peakPos[ vars()['positionActivity'+ichr][0,ipeak]-500:vars()['positionActivity'+ichr][1,ipeak]+500 ] = ipeak
		
		vars()['peakIndex'+ichr] = np.zeros(shape=(vars()['peakPos'+ichr].shape[1]),dtype=int)
		for ival in range( len(vars()['peakIndex'+ichr]) ):
			ipeak = np.amax( peakPos[vars()['peakPos'+ichr][0,ival]:vars()['peakPos'+ichr][1,ival]] )
			if ival>-1:
				vars()['peakIndex'+ichr][ival] = ipeak
			else:
				vars()['peakIndex'+ichr][ival] = -9999

		print(subtype+' Chromosome '+ichr+': '+str(int(100*np.round( len(vars()['peakIndex'+ichr][vars()['peakIndex'+ichr]>-1])/float(len(np.unique(vars()['peakPos'+ichr][0]))),2)))+' % match out of'+str(len(np.unique(vars()['peakPos'+ichr][0])))+' peaks')
		vars()['difPeaks'+subtype]['convert'][vars()['difPeaks'+subtype]['chr']=='chr'+ichr] = vars()['peakIndex'+ichr]

	vars()['difPeaks'+subtype].to_csv(wddata+'differential_genes/differential_peaks_'+subtypeName+'.txt', sep = '\t', header=True)





















