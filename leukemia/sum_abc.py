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
		vars()['direction'+ichr] = np.load(wdvars+'RNA/direction'+ichr+'.npy')
		vars()['nGenes'+ichr] = np.load(wdvars+'RNA/nGenes'+ichr+'.npy')

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

###################################################################
# Sum ABC and ATAC Across All Peaks
###################################################################
print 'Sum ABC and ATAC across all peaks'
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	nGenes = vars()['nGenes'+ichr]
	# Sum ABC across all peaks in a gene
	vars()['abcSum'+ichr] = np.sum( vars()['abc'+ichr], axis=2)

	# Sum ATAC across all peaks within 1Mb of a gene
	vars()['atacSum'+ichr] = np.zeros( shape = (nSamples, nGenes))
	for igene in range(nGenes):
		peakPos = np.mean( vars()['positionATAC'+ichr], axis=0 )
		genePos = np.mean( vars()['positionRNA'+ichr][:,igene])
		usePeak = np.abs( peakPos-genePos) < 1000000

		vars()['atacSum'+ichr][:,igene] = np.sum( vars()['atac'+ichr][:,usePeak], axis=1 )

###################################################################
# Correlations: ABC Sum and ATAC Sum --> RNA
###################################################################
print 'Correlations: Sum ABC and ATAC --> RNA'
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	print ichr
	nGenes = vars()['nGenes'+ichr]

	vars()['corrABC'+ichr] = np.zeros(shape = (nGenes))
	vars()['corrATAC'+ichr] = np.zeros(shape = (nGenes))
	vars()['pValueABC'+ichr] = np.ones(shape = (nGenes))
	vars()['pValueATAC'+ichr] = np.ones(shape = (nGenes))
	vars()['pCorrectedABC'+ichr] = np.ones(shape = (nGenes))
	vars()['pCorrectedATAC'+ichr] = np.ones(shape = (nGenes))
	for igene in range(nGenes):
		vars()['corrABC'+ichr][igene], vars()['pValueABC'+ichr][igene] = stats.spearmanr( vars()['abcSum'+ichr][:,igene], vars()['expression'+ichr][:,igene] )
		vars()['corrATAC'+ichr][igene], vars()['pValueATAC'+ichr][igene] = stats.spearmanr( vars()['atacSum'+ichr][:,igene], vars()['expression'+ichr][:,igene] )
	
	vars()['pCorrectedABC'+ichr] = multi.multipletests( vars()['pValueABC'+ichr], alpha=0.05, method = 'fdr_bh')[1]
	vars()['pCorrectedATAC'+ichr] = multi.multipletests( vars()['pValueATAC'+ichr], alpha=0.05, method = 'fdr_bh')[1]

	vars()['nStrongGenesABC'+ichr] = np.sum(vars()['corrABC'+ichr]>np.sqrt(0.4))
	vars()['nStrongGenesATAC'+ichr] = np.sum(vars()['corrATAC'+ichr]>np.sqrt(0.4))
	
###################################################################
# Correlations: Sum of Significant
###################################################################

print('Correlations: Sum of Significant')

for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	if ichr!='21': continue
	nGenes = vars()['nGenes'+ichr]
	vars()['corr'+ichr] = np.load(wdvars+ 'ABC_stats/corr'+ichr+'.npy')
	vars()['pCorrected'+ichr] = np.load(wdvars+ 'ABC_stats/pCorrected'+ichr+'.npy')

	vars()['abcSigSum'+ichr] = np.zeros(shape = (nSamples,nGenes))
	vars()['atacSigSum'+ichr] = np.zeros(shape = (nSamples,nGenes))
	vars()['geneMask'+ichr] = np.zeros(shape = (nSamples,nGenes),dtype=bool)
	for igene in range(nGenes):
		if np.sum(vars()['corr'+ichr][igene] > np.sqrt(0.4))==0:
			continue
			vars()['geneMask'+ichr][:,igene] = 1
		peakIndices = np.where(vars()['corr'+ichr][igene] > np.sqrt(0.2))[0]
		vars()['abcSigSum'+ichr][:,igene] = np.sum( vars()['abc'+ichr][:,igene,peakIndices], axis=1)
		vars()['atacSigSum'+ichr][:,igene] = np.sum( vars()['atac'+ichr][:,peakIndices], axis=1)
	#vars()['abcSigSum'+ichr] = np.ma.compress_cols(np.ma.masked_array(vars()['abcSigSum'+ichr], vars()['geneMask'+ichr]))
	#vars()['expressionSig'+ichr] = np.ma.compress_cols(np.ma.masked_array(vars()['expression'+ichr], vars()['geneMask'+ichr]))

	vars()['corrSigABC'+ichr] = np.zeros(shape = (nGenes))
	vars()['corrSigATAC'+ichr] = np.zeros(shape = (nGenes))
	vars()['pValueSigABC'+ichr] = np.ones(shape = (nGenes))
	vars()['pValueSigATAC'+ichr] = np.ones(shape = (nGenes))
	vars()['pCorrectedSigABC'+ichr] = np.ones(shape = (nGenes))
	vars()['pCorrectedSigATAC'+ichr] = np.ones(shape = (nGenes))
	for igene in range(nGenes):
		if np.std( vars()['abcSigSum'+ichr][:,igene] ) == 0: continue
		vars()['corrSigABC'+ichr][igene], vars()['pValueSigABC'+ichr][igene] = stats.spearmanr( vars()['abcSigSum'+ichr][:,igene], vars()['expression'+ichr][:,igene] )
		vars()['corrSigATAC'+ichr][igene], vars()['pValueSigATAC'+ichr][igene] = stats.spearmanr( vars()['atacSigSum'+ichr][:,igene], vars()['expression'+ichr][:,igene] )
	
	vars()['pCorrectedSigABC'+ichr] = multi.multipletests( vars()['pValueSigABC'+ichr], alpha=0.05, method = 'fdr_bh')[1]
	vars()['pCorrectedSigATAC'+ichr] = multi.multipletests( vars()['pValueSigATAC'+ichr], alpha=0.05, method = 'fdr_bh')[1]

	vars()['nStrongGenesSigABC'+ichr] = np.sum(vars()['corrSigABC'+ichr]>np.sqrt(0.4))
	vars()['nStrongGenesSigATAC'+ichr] = np.sum(vars()['corrSigATAC'+ichr]>np.sqrt(0.4))

	vars()['maskSigABC'+ichr] = vars()['corrSigABC'+ichr]==0
	vars()['maskSigATAC'+ichr] = vars()['corrSigATAC'+ichr]==0

	vars()['corrSigABC'+ichr] = np.ma.masked_array( vars()['corrSigABC'+ichr], vars()['maskSigABC'+ichr] )
	vars()['corrSigATAC'+ichr] = np.ma.masked_array( vars()['corrSigATAC'+ichr], vars()['maskSigATAC'+ichr] )
	vars()['pValueSigABC'+ichr] = np.ma.masked_array( vars()['pValueSigABC'+ichr], vars()['maskSigABC'+ichr] )
	vars()['pValueSigATAC'+ichr] = np.ma.masked_array( vars()['pValueSigATAC'+ichr], vars()['maskSigATAC'+ichr] )
	vars()['pCorrectedSigABC'+ichr] = np.ma.masked_array( vars()['pCorrectedSigABC'+ichr], vars()['maskSigABC'+ichr] )
	vars()['pCorrectedSigATAC'+ichr] = np.ma.masked_array( vars()['pCorrectedSigATAC'+ichr], vars()['maskSigATAC'+ichr] )

###################################################################
# Regressions
###################################################################

nX = 4
nMethods = 3
nChr = 23

medianErrorAll = np.ones(shape = (nChr,nX,nMethods)) # [chromosomes], [abc,atac], [multivar,forest,lasso]
medianR2All = np.ones(shape = (nChr,nX,nMethods))
numStrongGenes = np.zeros(shape = (nChr,nX,nMethods))
numGenes = np.zeros(shape=(nChr,nX,nMethods))
pctGenes = np.zeros(shape=(nChr,nX,nMethods))

strongGenes = []
for ichr in range(nChr):
	emptyList = []
	strongGenes.append(emptyList)
	for ix in range(nX):
		emptyList = []
		strongGenes[ichr].append(emptyList)
		for imethod in range(nMethods):
			emptyList = []
			strongGenes[ichr][ix].append(emptyList)

Vars = ['abcSum','atacSum','abcSigSum','atacSigSum']
jchr = -1
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	jchr+=1
	if ichr!='21': continue
	nGenes = vars()['nGenes'+ichr]
	for ivar in range(len(Vars)):
		var = vars()[Vars[ivar]+ichr]

		vars()['r2MultiV_'+ichr] = np.zeros(shape = (20,nGenes) )
		vars()['r2Forest_'+ichr] = np.zeros(shape = (20,nGenes) )
		vars()['r2Lasso_'+ichr] =  np.zeros(shape = (20,nGenes) )
		vars()['medianErrorMultiV_'+ichr] = np.ones(shape = nGenes ) # cutoff, nGenes
		vars()['medianErrorForest_'+ichr] = np.ones(shape = nGenes ) # cutoff, nGenes
		vars()['medianErrorLasso_'+ichr] = np.ones(shape = nGenes ) # cutoff, nGenes

		for igene in range(nGenes):
			if vars()['maskSigABC'+ichr][igene]==1: continue
			features = np.array(var[:,igene])
			targets = vars()['expression'+ichr][:,igene]
			
			errorMultiV = np.ones(shape = (4, 20)) # 4 test samples, 20 loops
			errorForest = np.ones(shape = (4, 20))
			errorLasso = np.ones(shape = (4, 20))
			
			for itest in range(20):
				features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=0.25)
				features_train = features_train.reshape(-1,1)
				features_test = features_test.reshape(-1,1)
			
				########### Multivariate ###########
				clf=sklearn.linear_model.LinearRegression()
				clf.fit(features_train,targets_train)
				vars()['r2MultiV_'+ichr][itest,igene] = clf.score(features_test,targets_test)
				predict = clf.predict(features_test)
				errorMultiV[:,itest] = np.abs(predict-targets_test)/targets_test
			
				########### Random Forest ###########
				clf = sklearn.ensemble.RandomForestRegressor(n_estimators=30)
				clf.fit(features_train,targets_train)
				vars()['r2Forest_'+ichr][itest,igene] = clf.score(features_test,targets_test)
				predict = clf.predict(features_test)
				errorForest[:,itest] = np.abs(predict-targets_test)/targets_test
			
				########### Lasso ###########
				clf = sklearn.linear_model.Lasso()
				clf.fit(features_train,targets_train)
				vars()['r2Lasso_'+ichr][itest,igene] = clf.score(features_test,targets_test)
				predict = clf.predict(features_test)
				errorLasso[:,itest] = np.abs(predict-targets_test)/targets_test
			
			vars()['medianErrorMultiV_'+ichr][igene] = np.median(errorMultiV)
			vars()['medianErrorForest_'+ichr][igene] = np.median(errorForest)
			vars()['medianErrorLasso_'+ichr][igene] = np.median(errorLasso)

		###################################################################
		# Assign Summary Variables
		###################################################################

		medianErrorAll[jchr,ivar,0] = np.median( vars()['medianErrorMultiV_'+ichr][vars()['medianErrorMultiV_'+ichr]!=1] )
		medianErrorAll[jchr,ivar,1] = np.median( vars()['medianErrorForest_'+ichr][vars()['medianErrorForest_'+ichr]!=1] )
		medianErrorAll[jchr,ivar,2] = np.median( vars()['medianErrorLasso_'+ichr][vars()['medianErrorLasso_'+ichr]!=1] )
		medianR2All[jchr,ivar,0] = np.median( np.median( vars()['r2MultiV_'+ichr],axis=1)[np.median(vars()['r2MultiV_'+ichr],axis=1)!=0] )
		medianR2All[jchr,ivar,1] = np.median( np.median( vars()['r2Forest_'+ichr],axis=1)[np.median(vars()['r2Forest_'+ichr],axis=1)!=0] )
		medianR2All[jchr,ivar,2] = np.median( np.median( vars()['r2Lasso_'+ichr],axis=1)[np.median(vars()['r2Lasso_'+ichr],axis=1)!=0] )
		numStrongGenes[jchr,ivar,0] = len(np.where(np.median( vars()['r2MultiV_'+ichr][:,:],axis=0)>0.4)[0])
		numStrongGenes[jchr,ivar,1] = len(np.where(np.median( vars()['r2Forest_'+ichr][:,:],axis=0)>0.4)[0])
		numStrongGenes[jchr,ivar,2] = len(np.where(np.median( vars()['r2Lasso_'+ichr][:,:],axis=0)>0.4)[0])

		strongGenes[jchr][ivar][0].append( list( vars()['geneName'+ichr][ np.where(np.median(vars()['r2MultiV_'+ichr][:,:],axis=0)>0.4)[0] ]) )
		strongGenes[jchr][ivar][1].append( list( vars()['geneName'+ichr][ np.where(np.median(vars()['r2Forest_'+ichr][:,:],axis=0)>0.4)[0] ]) )
		strongGenes[jchr][ivar][2].append( list( vars()['geneName'+ichr][ np.where(np.median(vars()['r2Lasso_'+ichr][:,:],axis=0)>0.4)[0] ]) )

		numGenes[jchr,ivar,0] = vars()['medianErrorMultiV_'+ichr][vars()['medianErrorMultiV_'+ichr]!=1].shape[0]
		numGenes[jchr,ivar,1] = vars()['medianErrorForest_'+ichr][vars()['medianErrorForest_'+ichr]!=1].shape[0]
		numGenes[jchr,ivar,2] = vars()['medianErrorLasso_'+ichr][vars()['medianErrorLasso_'+ichr]!=1].shape[0]
		pctGenes[jchr,ivar,0] = vars()['medianErrorMultiV_'+ichr][vars()['medianErrorMultiV_'+ichr]!=1].shape[0]/float(vars()['nGenes'+ichr])
		pctGenes[jchr,ivar,1] = vars()['medianErrorForest_'+ichr][vars()['medianErrorForest_'+ichr]!=1].shape[0]/float(vars()['nGenes'+ichr])
		pctGenes[jchr,ivar,2] = vars()['medianErrorLasso_'+ichr][vars()['medianErrorLasso_'+ichr]!=1].shape[0]/float(vars()['nGenes'+ichr])

			
medianError = medianErrorAll[20]	
medianR2 = medianR2All[20]	
nStrongGenes = numStrongGenes[20]

np.save(wdvars+'Sum/medianError.npy',medianError)
np.save(wdvars+'Sum/medianR2.npy',medianR2)
np.save(wdvars+'Sum/nStrongGenes.npy',nStrongGenes)


plt.clf()
plt.plot(0,100*medianError[1,0], '^', color = 'tomato', markersize = 15) #label = 'ATAC Sum Multivariate', 
plt.plot(1,100*medianError[1,1], 'o', color = 'tomato', label = 'ATAC Sum', markersize = 15)
plt.plot(2,100*medianError[1,2], '+', color = 'tomato', mew=5, markersize = 15) #label = 'ATAC Sum Lasso', 

plt.plot(0,100*medianError[0,0], '^', color = 'aqua', markersize = 15) # label = 'ABC Sum Multivariate', 
plt.plot(1,100*medianError[0,1], 'o', color = 'aqua', label = 'ABC Sum', markersize = 15)
plt.plot(2,100*medianError[0,2], '+', color = 'aqua', mew=5, markersize = 15) #label = 'ABC Sum Lasso', 

plt.plot(0,100*medianError[3,0], '^', color = 'maroon', markersize = 15) #label = 'ATAC Sig Sum Multivariate', 
plt.plot(1,100*medianError[3,1], 'o', color = 'maroon', label = 'ATAC Sum of Significant', markersize = 15)
plt.plot(2,100*medianError[3,2], '+', color = 'maroon', mew=5, markersize = 15) #label = 'ATAC Sig Sum Lasso', 

plt.plot(0,100*medianError[2,0], '^', color = 'teal', markersize = 15) #label = 'ABC Sig Sum Multivariate', 
plt.plot(1,100*medianError[2,1], 'o', color = 'teal', label = 'ABC Sum of Significant', markersize = 15)
plt.plot(2,100*medianError[2,2], '+', color = 'teal', mew=5, markersize = 15) #label = 'ABC Sig Sum Lasso', 

plt.title('Median Error of Gene Expression Predictions: Summed ABC', fontsize=15)
plt.ylabel('Median Error (%) of Genes with Correlations')
plt.ylim([10,37])
plt.legend(loc='lower left',fontsize = 12)
plt.xlim([-0.5,2.5])
plt.xticks([0,1,2], ['Multivariate','Random Forest','Lasso'])
plt.grid(True)
plt.savefig(wdfigs+'summary/medianError_summed_abc.pdf')
plt.show()


plt.clf()
plt.plot(0,medianR2[1,0], '^', color = 'tomato', markersize = 15) #label = 'ATAC Sum Multivariate', 
plt.plot(1,medianR2[1,1], 'o', color = 'tomato', label = 'ATAC Sum', markersize = 15)
plt.plot(2,medianR2[1,2], '+', color = 'tomato', mew=5, markersize = 15) #label = 'ATAC Sum Lasso', 

plt.plot(0,medianR2[0,0], '^', color = 'aqua', markersize = 15) # label = 'ABC Sum Multivariate', 
plt.plot(1,medianR2[0,1], 'o', color = 'aqua', label = 'ABC Sum', markersize = 15)
plt.plot(2,medianR2[0,2], '+', color = 'aqua', mew=5, markersize = 15) #label = 'ABC Sum Lasso', 

plt.plot(0,medianR2[3,0], '^', color = 'maroon', markersize = 15) #label = 'ATAC Sig Sum Multivariate', 
plt.plot(1,medianR2[3,1], 'o', color = 'maroon', label = 'ATAC Sum of Significant', markersize = 15)
plt.plot(2,medianR2[3,2], '+', color = 'maroon', mew=5, markersize = 15) #label = 'ATAC Sig Sum Lasso', 

plt.plot(0,medianR2[2,0], '^', color = 'teal', markersize = 15) #label = 'ABC Sig Sum Multivariate', 
plt.plot(1,medianR2[2,1], 'o', color = 'teal', label = 'ABC Sum of Significant', markersize = 15)
plt.plot(2,medianR2[2,2], '+', color = 'teal', mew=5, markersize = 15) #label = 'ABC Sig Sum Lasso', 

plt.plot([-10,10],[0,0],'k-')
plt.title('Median R2 of Gene Expression Predictions: Summed ABC', fontsize=15)
plt.ylabel('Median R2 of Genes with Correlations')
plt.ylim([-0.5,1])
plt.legend(loc='upper left',fontsize = 12)
plt.xlim([-0.5,2.5])
plt.xticks([0,1,2], ['Multivariate','Random Forest','Lasso'])
plt.grid(True)
plt.savefig(wdfigs+'summary/medianR2_summed_abc.pdf')
plt.show()


plt.clf()
plt.plot(0,nStrongGenes[1,0], '^', color = 'tomato', markersize = 15) #label = 'ATAC Sum Multivariate', 
plt.plot(1,nStrongGenes[1,1], 'o', color = 'tomato', label = 'ATAC Sum', markersize = 15)
plt.plot(2,nStrongGenes[1,2], '+', color = 'tomato', mew=5, markersize = 15) #label = 'ATAC Sum Lasso', 

plt.plot(0,nStrongGenes[0,0], '^', color = 'aqua', markersize = 15) # label = 'ABC Sum Multivariate', 
plt.plot(1,nStrongGenes[0,1], 'o', color = 'aqua', label = 'ABC Sum', markersize = 15)
plt.plot(2,nStrongGenes[0,2], '+', color = 'aqua', mew=5, markersize = 15) #label = 'ABC Sum Lasso', 

plt.plot(0,nStrongGenes[3,0], '^', color = 'maroon', markersize = 15) #label = 'ATAC Sig Sum Multivariate', 
plt.plot(1,nStrongGenes[3,1], 'o', color = 'maroon', label = 'ATAC Sum of Significant', markersize = 15)
plt.plot(2,nStrongGenes[3,2], '+', color = 'maroon', mew=5, markersize = 15) #label = 'ATAC Sig Sum Lasso', 

plt.plot(0,nStrongGenes[2,0], '^', color = 'teal', markersize = 15) #label = 'ABC Sig Sum Multivariate', 
plt.plot(1,nStrongGenes[2,1], 'o', color = 'teal', label = 'ABC Sum of Significant', markersize = 15)
plt.plot(2,nStrongGenes[2,2], '+', color = 'teal', mew=5, markersize = 15) #label = 'ABC Sig Sum Lasso', 

plt.plot([-10,10],[0,0],'k-')
plt.title('Number of Genes with R2 > 0.4: Summed ABC', fontsize=15)
plt.ylabel('Number of Genes')
plt.ylim([0,40])
plt.legend(loc='upper left',fontsize = 12)
plt.xlim([-0.5,2.5])
plt.xticks([0,1,2], ['Multivariate','Random Forest','Lasso'])
plt.grid(True)
plt.savefig(wdfigs+'summary/numStrongGenes_summed_abc.pdf')
plt.show()
