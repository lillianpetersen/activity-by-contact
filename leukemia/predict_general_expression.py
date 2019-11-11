import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sys import exit
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
wdfigs = '/pbld/mcg/lillianpetersen/ABC/figures/'
wdfiles = '/pbld/mcg/lillianpetersen/ABC/written_files/'
MakePlots = False

# MCG B ALL Samples
MCGs = ['MCG001', 'MCG002', 'MCG003', 'MCG005', 'MCG006', 'MCG009', 'MCG010', 'MCG011', 'MCG012', 'MCG013', 'MCG016', 'MCG017', 'MCG019', 'MCG020', 'MCG023', 'MCG024']
nSamples = len(MCGs)
nChr = 23

###################################################################
# Load RNA
###################################################################
print('Load RNA')
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	if not 'expression'+ichr in globals():
		# Load arrays saved from load_rna.npy
		vars()['expression'+ichr] = np.load(wdvars+'RNA/expression'+ichr+'.npy')
		vars()['geneName'+ichr] = np.load(wdvars+'RNA/geneName'+ichr+'.npy')
		vars()['chrRNA'+ichr] = np.load(wdvars+'RNA/chrRNA'+ichr+'.npy')
		vars()['positionRNA'+ichr] = np.load(wdvars+'RNA/positionRNA'+ichr+'.npy')
		vars()['nGenes'+ichr] = np.load(wdvars+'RNA/nGenes'+ichr+'.npy')
		vars()['direction'+ichr] = np.load(wdvars+'RNA/direction'+ichr+'.npy')

	if not 'expressionNorm'+ichr in globals():
		vars()['expressionNorm'+ichr] = np.zeros(shape = (vars()['expression'+ichr].shape))
		for igene in range(vars()['nGenes'+ichr]):
			vars()['expressionNorm'+ichr][:,igene] = sklearn.preprocessing.scale(vars()['expression'+ichr][:,igene])

###################################################################
# Load ATAC
###################################################################
print('Load ATAC')
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	if not 'atac'+ichr in globals():
		# Load arrays saved from load_atac.npy
		vars()['atac'+ichr] = np.load(wdvars+'ATAC/atac'+ichr+'.npy')
		vars()['chrATAC'+ichr] = np.load(wdvars+'ATAC/chrATAC'+ichr+'.npy')
		vars()['peakName'+ichr] = np.load(wdvars+'ATAC/peakName'+ichr+'.npy')
		vars()['positionATAC'+ichr] = np.load(wdvars+'ATAC/positionATAC'+ichr+'.npy')
		vars()['nPeaks'+ichr] = np.load(wdvars+'ATAC/nPeaks'+ichr+'.npy')

###################################################################
# Load HiC
###################################################################
print('load HiC')
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	if not 'hic'+ichr in globals():
		# Load arrays saved from load_hic.npy
		vars()['hic'+ichr] = np.load( wdvars+'HiC/hic'+ichr+'.npy')
    	vars()['geneStart'+ichr] = np.load( wdvars+'HiC/geneStart'+ichr+'.npy' )
    	vars()['geneMatrix'+ichr] = np.load( wdvars+'HiC/geneMatrix'+ichr+'.npy' )
    	vars()['peakStart'+ichr] = np.load( wdvars+'HiC/peakStart'+ichr+'.npy' )
    	vars()['peakMatrix'+ichr] = np.load( wdvars+'HiC/peakMatrix'+ichr+'.npy' )
    	vars()['peakMatrix'+ichr] = np.load( wdvars+'HiC/geneStartDict'+ichr+'.npy' )

##### Fix indexing for chromosome X #####
if len(geneMatrixX)!=len(geneNameX):
	keepGene = np.isin(geneNameX,geneMatrixX) # True = keep
	keepPeak = np.isin(peakNameX,peakMatrixX) # True = keep
	
	expressionX = expressionX[:,keepGene]
	geneNameX = geneNameX[keepGene]
	chrRNAX = chrRNAX[keepGene]
	directionX = directionX[keepGene]
	positionRNAX = positionRNAX[:,keepGene]
	nGenesX = expressionX.shape[1]
	
	atacX = atacX[:,keepPeak]
	chrATACX = chrATACX[keepPeak]
	peakNameX = peakNameX[keepPeak]
	positionATACX = positionATACX[:,keepPeak]
	nPeaksX = atacX.shape[1]
#########################################
	
###################################################################
# Create ABC Matrix
###################################################################
print 'Create ABC'
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
#for ichr in ['21']:
	if not 'abc'+ichr in globals():
		peakMatch = np.isin( vars()['peakName'+ichr] , vars()['peakMatrix'+ichr])
		peakMask = np.array(1-peakMatch, dtype=bool)
		
		maskFull = np.zeros(shape = (nSamples,len(peakMask)), dtype = bool)
		for isample in range(nSamples):
			maskFull[isample] = peakMask
		mask2 = np.zeros(shape = (2,len(peakMask)), dtype = bool)
		for isample in range(2):
			mask2[isample] = peakMask
		vars()['peakName'+ichr] = np.ma.compressed( np.ma.masked_array(vars()['peakName'+ichr],peakMask) )
		vars()['atac'+ichr] = np.ma.compress_cols( np.ma.masked_array(vars()['atac'+ichr],maskFull) )
		vars()['positionATAC'+ichr] = np.ma.compress_cols( np.ma.masked_array(vars()['positionATAC'+ichr],mask2) )
		if np.amin(vars()['peakName'+ichr]==vars()['peakMatrix'+ichr]) == 0: 
			print 'Error: Peak array sizes do not match'
			exit()
	
		vars()['abc'+ichr] = np.zeros(shape = (nSamples, vars()['hic'+ichr].shape[1], vars()['hic'+ichr].shape[2] ))
		for isample in range(nSamples):
			vars()['abc'+ichr][isample] = vars()['atac'+ichr][isample] * vars()['hic'+ichr][isample]
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
			plt.savefig(wdfigs+'abc_MCG001_Chr'+ichr+'.pdf')

###################################################################
# Sum ABC and ATAC Across All Peaks
###################################################################
print 'Sum ABC and ATAC across all peaks'
nClosest = 16
#for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
for ichr in ['21']:
	nGenes = vars()['nGenes'+ichr]
	## Sum ABC across all peaks in a gene
	#vars()['abcSum'+ichr] = np.sum( vars()['abc'+ichr], axis=2)
	# Sum ABC across all peaks within 300000bp of a gene
	vars()['abcSum'+ichr] = np.zeros( shape = (nSamples,nGenes))
	vars()['abcClosest'+ichr] = np.zeros( shape = (nSamples,nGenes,nClosest))
	for igene in range(nGenes):
		peakPos = np.mean( vars()['positionATAC'+ichr], axis=0 )
		genePos = np.mean( vars()['positionRNA'+ichr][:,igene])
		usePeak = np.abs( peakPos-genePos) < 300000
		vars()['abcSum'+ichr][:,igene] = np.sum( vars()['abc'+ichr][:,igene,usePeak], axis=1 )
		sort = np.argsort( np.abs(genePos-peakPos)[usePeak] )
		vars()['abcClosest'+ichr][:,igene] = vars()['abc'+ichr][:,igene,usePeak][:,sort[:nClosest]]
	vars()['abcClosestSum'+ichr] = np.sum(vars()['abcClosest'+ichr],axis=2)
	print 'abc', stats.spearmanr(np.ma.compressed(vars()['abcClosestSum'+ichr]),np.ma.compressed(vars()['expressionNorm'+ichr]))[0]

	# Sum ATAC across all peaks within 300000bp of a gene
	vars()['atacSum'+ichr] = np.zeros( shape = (nSamples,nGenes))
	vars()['atacClosest'+ichr] = np.zeros( shape = (nSamples,nGenes,nClosest))
	for igene in range(nGenes):
		peakPos = np.mean( vars()['positionATAC'+ichr], axis=0 )
		genePos = np.mean( vars()['positionRNA'+ichr][:,igene])
		usePeak = np.abs( peakPos-genePos) < 300000 # True = good = within 300kb
		tss = vars()['positionRNA'+ichr][ vars()['direction'+ichr][igene], igene]
		tssMask = np.abs(peakPos-tss) > 2000 # True = good = outside 2kb
		usePeak = np.amin([usePeak,tssMask],axis=0)

		vars()['atacSum'+ichr][:,igene] = np.sum( vars()['atac'+ichr][:,usePeak], axis=1 )
		sort = np.argsort( np.abs(genePos-peakPos)[usePeak] )
		vars()['atacClosest'+ichr][:,igene] = vars()['atac'+ichr][:,usePeak][:,sort[:nClosest]]
	vars()['atacClosestSum'+ichr] = np.sum(vars()['atacClosest'+ichr],axis=2)
	print 'atac', stats.spearmanr(np.ma.compressed(vars()['atacClosestSum'+ichr]),np.ma.compressed(vars()['expressionNorm'+ichr]))[0]

###################################################################
# Machine Learn and predict gene expression
###################################################################
print 'Predict Gene Expression'
targets = np.reshape(expressionNorm21,-1)

#### Set up testing and training set ####

for ichr in ['21']:

	testLen = int(round(0.2*len(targets)))
	vars()['r2MultiV_'+ichr] = np.zeros(shape = (20,testLen))
	vars()['predictMultiV_'+ichr] = np.zeros(shape = (20,testLen))
	vars()['errorMultiV_'+ichr] = np.zeros(shape = (20,testLen))

	features = np.zeros(shape=(len(targets),4))
	features[:,0] = np.reshape(vars()['atacSum'+ichr],-1)
	features[:,1] = np.reshape(vars()['abcSum'+ichr],-1)
	features[:,2] = np.reshape(vars()['atacClosestSum'+ichr],-1)
	features[:,3] = np.reshape(vars()['abcClosestSum'+ichr],-1)

	for itest in range(20):
		X_train, X_test, Y_train, Y_test = train_test_split(features, targets, test_size=0.20)
	
		########### Multivariate ###########
		clf=sklearn.linear_model.LinearRegression()
		clf.fit(X_train,Y_train)
		vars()['r2MultiV_'+ichr][itest,:] = clf.score(X_test,Y_test)
		predict = clf.predict(X_test)
		vars()['errorMultiV_'+ichr][itest,:] = np.abs((Y_test-predict)/Y_test) *100
		vars()['predictMultiV_'+ichr][itest,:] = predict
	exit()


exit()


















