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
#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

wd = '/pbld/mcg/lillianpetersen/ABC/'
wdvars = '/pbld/mcg/lillianpetersen/ABC/saved_variables/'
wdfigs = '/pbld/mcg/lillianpetersen/ABC/figures/'
wdfiles = '/pbld/mcg/lillianpetersen/ABC/written_files/'
MakePlots = False

# MCG B ALL Samples
MCGs = ['MCG001', 'MCG002', 'MCG003', 'MCG005', 'MCG006', 'MCG009', 'MCG010', 'MCG011', 'MCG012', 'MCG013', 'MCG016', 'MCG017', 'MCG019', 'MCG020', 'MCG023', 'MCG024', 'MCG027', 'MCG028', 'MCG034', 'MCG035', 'MCG036', 'MCG037', 'MCG038', 'MCG039']
nSamples = len(MCGs)
nChr = 23

subtypes = np.array(['ETV6-RUNX1', 'DUX4', 'Hyperdiploid', 'PAX5', 'Ph-like'])
typeNames = np.array(['ETVRUNX', 'DUX', 'Hyperdiploid', 'PAX', 'Phlike'])

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
indexTypeDict = {
	0  : 'PAX5', 1  : 'ETV6-RUNX1', 2  : 'PAX5alt', 3  : 'DUX4', 4  : 'ZNF384', 5  : 'PAX5alt', 6  : 'Hyperdiploid', 7  : 'DUX4', 8  : 'Hyperdiploid', 9  : 'Ph-like', 10 : 'Ph-like', 11 : 'ETV6-RUNX1', 12 : 'Hyperdiploid', 13 : 'Hyperdiploid', 14 : 'DUX4', 15 : 'ETV6-RUNX1', 16 : 'Hyperdiploid', 17 : 'Hyperdiploid', 18 : 'Hyperdiploid', 19 : 'ETV6-RUNX1', 20 : 'DUX4', 21 : 'Other', 22 : 'Ph-like', 23 : 'Ph-like' }

def top_k(numbers, k=2):
	c = Counter(numbers)
	most_common = [key for key, val in c.most_common(k)]
	return most_common

###################################################################
# Load RNA
###################################################################
print('Load RNA')
nGenes=0
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	if not 'expression'+ichr in globals():
		# Load arrays saved from load_rna.npy
		vars()['expression'+ichr] = np.load(wdvars+'RNA/expression'+ichr+'.npy')
		vars()['geneName'+ichr] = np.load(wdvars+'RNA/geneName'+ichr+'.npy')
		vars()['chrRNA'+ichr] = np.load(wdvars+'RNA/chrRNA'+ichr+'.npy')
		vars()['positionRNA'+ichr] = np.load(wdvars+'RNA/positionRNA'+ichr+'.npy')
		vars()['direction'+ichr] = np.load(wdvars+'RNA/direction'+ichr+'.npy')
		vars()['direction'+ichr][vars()['direction'+ichr]==1] = 0 # 0 = +
		vars()['direction'+ichr][vars()['direction'+ichr]==-1] = 1 # 1 = -

		vars()['tss'+ichr] = np.zeros(shape=(vars()['chrRNA'+ichr].shape))
		for i in range(len(vars()['chrRNA'+ichr])):
			vars()['tss'+ichr][i] = vars()['positionRNA'+ichr][:,i][vars()['direction'+ichr][i]]

###################################################################
# Load Activity
###################################################################
print('Load Activity')
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	if not 'activity'+ichr in globals():
		# Load arrays saved from load_atac.npy
		vars()['activity'+ichr] = np.load(wdvars+'ATAC/atac'+ichr+'.npy')
		vars()['chrActivity'+ichr] = np.load(wdvars+'ATAC/chrATAC'+ichr+'.npy')
		vars()['peakName'+ichr] = np.load(wdvars+'ATAC/peakName'+ichr+'.npy')
		vars()['positionActivity'+ichr] = np.load(wdvars+'ATAC/positionATAC'+ichr+'.npy')

	for itype in range(len(subtypes)):
		subtype = typeNames[itype]
		vars()['activity'+subtype+ichr] = np.mean( vars()['activity'+ichr][typesIndex==itype], axis=0)

###################################################################
# Load HiC
###################################################################
for itype in range(len(subtypes)):
	subtype = typeNames[itype]
	subtypeName = subtypes[itype]
	print('load HiC: '+subtypeName)
	for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
		if not 'hic'+subtype+ichr in globals():
			# Load arrays saved from load_hic.npy
			vars()['hic'+subtype+ichr] = np.load( wdvars+'HiC/'+subtypeName+'/hic'+ichr+'.npy')
			vars()['geneStart'+ichr] = np.load( wdvars+'HiC/general/geneStart'+ichr+'.npy' )
			vars()['geneMatrix'+ichr] = np.load( wdvars+'HiC/general/geneMatrix'+ichr+'.npy' )
			vars()['peakStart'+ichr] = np.load( wdvars+'HiC/general/peakStart'+ichr+'.npy' )
			vars()['peakMatrix'+ichr] = np.load( wdvars+'HiC/general/peakMatrix'+ichr+'.npy' )

			if np.amin(np.isin(vars()['peakMatrix'+ichr], vars()['peakName'+ichr])) == False:
				print 'ERROR: peaks dont match'
				exit()
			if np.amin(np.isin(vars()['peakMatrix'+ichr], vars()['peakName'+ichr])) == False:
				indices = np.isin(vars()['peakMatrix'+ichr], vars()['peakName'+ichr])
				vars()['hic'+subtype+ichr] = vars()['hic'+subtype+ichr][:,indices]
				vars()['peakStart'+ichr] = vars()['peakStart'+ichr][indices]
				vars()['peakMatrix'+ichr] = vars()['peakMatrix'+ichr][indices]
			if np.amin(np.isin(vars()['peakName'+ichr], vars()['peakMatrix'+ichr])) == False:
				indices = np.isin(vars()['peakName'+ichr], vars()['peakMatrix'+ichr])
				vars()['peakName'+ichr] = vars()['peakName'+ichr][indices]
				vars()['activity'+ichr] = vars()['activity'+ichr][:,indices]
				vars()['chrActivity'+ichr] = vars()['chrActivity'+ichr][indices]
				vars()['positionActivity'+ichr] = vars()['positionActivity'+ichr][:,indices]
				vars()['nPeaks'+ichr] = len(vars()['peakName'+ichr])
			if np.amin(np.isin(vars()['geneMatrix'+ichr], vars()['geneName'+ichr])) == False:
				print 'Fixing HiC Genes: Chromosome '+ichr
				indices = np.isin(vars()['geneMatrix'+ichr], vars()['geneName'+ichr])
				vars()['hic'+subtype+ichr] = vars()['hic'+ichr][indices,:]
				vars()['geneStart'+ichr] = vars()['geneStart'+ichr][indices]
				vars()['geneMatrix'+ichr] = vars()['geneMatrix'+ichr][indices]
			if np.amin(np.isin(vars()['geneName'+ichr], vars()['geneMatrix'+ichr])) == False:
				print 'Fixing RNA indexing: Chromosome '+ichr
				indices = np.isin(vars()['geneName'+ichr], vars()['geneMatrix'+ichr])
				vars()['expression'+ichr] = vars()['expression'+ichr][indices]
				vars()['geneName'+ichr] = vars()['geneName'+ichr][indices]
				vars()['tss'+ichr] = vars()['tss'+ichr][indices]
				vars()['positionRNA'+ichr] = vars()['positionRNA'+ichr][:,indices]
exit()
			
###################################################################
# Create ABC Matrix
###################################################################
for itype in range(len(subtypes)):
	subtype = typeNames[itype]
	subtypeName = subtypes[itype]
	print('Create ABC: '+subtypeName)
	for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	#for ichr in ['21']:
		if not 'abc'+ichr in globals():
			try:
				vars()['abc'+ichr] = np.load(wdvars+'ABC/abc'+ichr+'.npy')
				vars()['abc'+ichr] = np.ma.masked_array(vars()['abc'+ichr],vars()['abc'+ichr]==0)
			except:
				print ichr,
				peakMatch = np.isin( vars()['peakName'+ichr] , vars()['peakMatrix'+ichr])
				peakMask = np.array(1-peakMatch, dtype=bool)
				
				activityMean = np.mean(vars()['activity'+ichr],axis=0)
				#maskFull = np.zeros(shape = (nSamples,len(peakMask)), dtype = bool)
				#for isample in range(nSamples):
				#	maskFull[isample] = peakMask
				mask2 = np.zeros(shape = (2,len(peakMask)), dtype = bool)
				for isample in range(2):
					mask2[isample] = peakMask
				vars()['peakName'+ichr] = np.ma.compressed( np.ma.masked_array(vars()['peakName'+ichr],peakMask) )
				#vars()['activity'+ichr] = np.ma.compress_cols( np.ma.masked_array(vars()['activity'+ichr],maskFull) )
				activityMean = np.ma.compressed( np.ma.masked_array(activityMean,peakMask) )
				vars()['positionActivity'+ichr] = np.ma.compress_cols( np.ma.masked_array(vars()['positionActivity'+ichr],mask2) )
				if np.amin(vars()['peakName'+ichr]==vars()['peakMatrix'+ichr]) == 0: 
					print 'Error: Peak array sizes do not match'
					exit()
			
				nGenes = vars()['hic'+ichr].shape[0]
				nPeaks = vars()['hic'+ichr].shape[1]
				vars()['abc'+ichr] = np.zeros(shape = (nGenes, nPeaks))
				peakPos = np.mean( vars()['positionActivity'+ichr][:,:],axis=0)
				for igene in np.arange(nGenes):
					genePos = vars()['tss'+ichr][igene]
					usePeak1 = np.abs(peakPos-genePos)<1500000 # True = good = within 1.5Mb
					usePeak2 = np.abs(peakPos-genePos)>5000 # True = good = outside 5kb
					usePeak = usePeak1==usePeak2
					Sum = np.sum( activityMean[usePeak] * vars()['hic'+ichr][igene,usePeak])
					vars()['abc'+ichr][igene,usePeak] = (activityMean[usePeak] * vars()['hic'+ichr][igene,usePeak]) / Sum
				np.save(wdvars+'ABC/abc'+ichr+'.npy',vars()['abc'+ichr])
				mask = np.amax([vars()['abc'+ichr]==0, np.isnan(abcX)],axis=0)
				vars()['abc'+ichr] = np.ma.masked_array(vars()['abc'+ichr],mask)
			
				plt.clf()
				fig = plt.figure(figsize = (10,6))
				plt.imshow(vars()['abc'+ichr], cmap = 'hot_r', aspect='auto', interpolation='none',origin='lower', vmax=30)
				plt.title('ABC Matrix: Chromosome '+ichr,fontsize=18)
				plt.xlabel('Peaks')
				plt.ylabel('Genes')
				plt.grid(True)
				plt.colorbar()
				plt.savefig(wdfigs+'abc_Chr'+ichr+'.pdf')

