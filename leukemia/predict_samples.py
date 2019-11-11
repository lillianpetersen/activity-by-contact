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
	
		#if MakePlots:
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
# Correlations: ABC and ATAC --> RNA
###################################################################
print('Correlations')

nX = 2
nMethods = 3

cutoff = np.array([1,2,3,4]) # Number of Peaks to use in regression
medianErrorAll = np.ones(shape = (nChr,nX,nMethods,len(cutoff))) # [chromosomes], [abc,atac], [multivar,forest], [cutoffs]
medianR2All = np.ones(shape = (nChr,nX,nMethods,len(cutoff)))
numStrongGenes = np.zeros(shape = (nChr,nX,nMethods,len(cutoff)))
numGenes = np.zeros(shape=(nChr,nX,nMethods,len(cutoff)))
pctGenes = np.zeros(shape=(nChr,nX,nMethods,len(cutoff)))

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
			for icutoff in range(len(cutoff)):
				emptyList = []
				strongGenes[ichr][ix][imethod].append(emptyList)

#xVars = ['abc','atac','abcPC','atacPC']
#xVarTitles = ['ABC','ATAC','ABC PC','ATAC PC']
xVars = ['abc','atac']
xVarTitles = ['ABC','ATAC']
jchr = -1
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	jchr+=1

	for ivar in range(len(xVars)):
		var = xVars[ivar]
		varTitle = xVarTitles[ivar]
	
		nGenes = vars()['nGenes'+ichr] = vars()['expression'+ichr].shape[1]
		nPeaks = vars()['nPeaks'+ichr] = vars()['atac'+ichr].shape[1]

		print '\nCorrelation '+varTitle+'-RNA: Chromosome',ichr

		try:
			vars()['corr'+ichr] = np.load(wdvars+'ABC_stats/'+ichr+'/corr'+ichr+'_'+var+'.npy')
			vars()['pCorrected'+ichr] = np.load(wdvars+'ABC_stats/'+ichr+'/pCorrected'+ichr+'_'+var+'.npy')
		except:
			print 'Calculating...'
			if var=='abcPC' or var=='atacPC': # compute principal components
				print '\nPrincipal Components '+varTitle+'-RNA: Chromosome',ichr
	
				vars()['PC'+ichr] = np.zeros(shape = (nSamples,nGenes,10))
				vars()['corr'+ichr] = np.zeros(shape = (nGenes,10 ))
				vars()['pValue'+ichr] = np.ones(shape = (nGenes,10 ))
				vars()['pCorrected'+ichr] = np.ones(shape = (nGenes ,10))
				vars()['corrMask'+ichr] = np.zeros(shape = vars()['corr'+ichr].shape, dtype = bool)
				
				#### Calculate Principal Components ####
				for igene in range(nGenes):
					if var=='atacPC':
						peakPos = np.mean( vars()['positionATAC'+ichr][:,:],axis=0)
						genePos = vars()['geneStart'+ichr][igene]
						usePeak = np.ones(shape = (nSamples,nPeaks))
						usePeak[:,:] = np.invert(np.abs(peakPos-genePos)<1000000) # False = Good = within 1Mb
						xdata = np.ma.compress_cols(np.ma.masked_array( vars()['atac'+ichr], usePeak ))
					if var=='abcPC':
						xdata = np.ma.compress_cols( vars()['abc'+ichr][:,igene,:])
	
					varScaled = StandardScaler().fit_transform(xdata)
					pca = PCA(n_components=10)
					vars()['PC'+ichr][:,igene,:] = pca.fit_transform(varScaled)
	
					#### Calculate Correlations ####
					for ipc in range(10):
						ydata = vars()['expression'+ichr][:,igene]
						x = vars()['PC'+ichr][:,igene,ipc]
						vars()['corr'+ichr][igene,ipc], vars()['pValue'+ichr][igene,ipc] = stats.spearmanr(x,ydata)
				
					###### Correct P Values (FDR) ######
					vars()['pCorrected'+ichr][igene] = multi.multipletests( vars()['pValue'+ichr][igene], alpha=0.05, method = 'fdr_bh')[1]
						
				vars()['corrMask'+ichr] = vars()['pCorrected'+ichr]>0.8
				vars()['corr'+ichr] = np.ma.masked_array( vars()['corr'+ichr], vars()['corrMask'+ichr] )
				vars()['pValue'+ichr] = np.ma.masked_array( vars()['pValue'+ichr], vars()['corrMask'+ichr] )
				vars()['pCorrected'+ichr] = np.ma.masked_array( vars()['pCorrected'+ichr], vars()['corrMask'+ichr] )
	
	
			if var=='abc' or var=='atac': # compute correlations
			
				vars()['corr'+ichr] = np.zeros(shape = (nGenes, nPeaks ))
				vars()['pValue'+ichr] = np.ones(shape = (nGenes, nPeaks ))
				vars()['pCorrected'+ichr] = np.ones(shape = (nGenes, nPeaks ))
				vars()['corrMask'+ichr] = np.zeros(shape = vars()['corr'+ichr].shape, dtype = bool)
				#vars()['predict'+ichr] = np.zeros(shape = (nSamples,nGenes, nPeaks ))
			
				for igene in range(nGenes):
					#print np.round(100*igene/float(nGenes),2), '%'
					if var=='atac':
						peakPos = np.mean( vars()['positionATAC'+ichr][:,:],axis=0)
						genePos = vars()['geneStart'+ichr][igene]
						usePeak = np.abs(peakPos-genePos)<2000000 # True = good = within 2Mb
						vars()['direction'+ichr][vars()['direction'+ichr]==1] = 0
						vars()['direction'+ichr][vars()['direction'+ichr]==-1] = 1
						tss = vars()['positionRNA'+ichr][ vars()['direction'+ichr][igene], igene]
						tssMask = np.abs(peakPos-tss) > 2000 # True = good = outside 2kb
						usePeak = np.amin([usePeak,tssMask],axis=0)
						vars()['corrMask'+ichr][igene,usePeak==False] = 1
					for ipeak in range(nPeaks):
						if var=='abc': x = vars()[var+ichr][:,igene,ipeak]
						elif var=='atac': 
							if usePeak[ipeak]==False: continue
							x = vars()[var+ichr][:,ipeak]
						if np.sum(x)==0 or np.ma.is_masked(np.sum(x)): 
							vars()['corrMask'+ichr][igene,ipeak] = 1
							continue
						if np.std(x)==0: 
							vars()['corrMask'+ichr][igene,ipeak] = 1
							continue
						ydata = vars()['expression'+ichr][:,igene]
						vars()['corr'+ichr][igene,ipeak], vars()['pValue'+ichr][igene,ipeak] = stats.spearmanr(x,ydata)
						#m,b = np.polyfit(x,ydata,1)
						#vars()['predict'+ichr][:,irna,igene,ipeak] = m*x+b
						if np.isnan(vars()['corr'+ichr][igene,ipeak]): exit()
				
					###### Correct P Values (FDR) ######
					vars()['pCorrected'+ichr][igene,:] = multi.multipletests( vars()['pValue'+ichr][igene], alpha=0.05, method = 'fdr_bh')[1]
				
				vars()['corr'+ichr] = np.ma.masked_array( vars()['corr'+ichr], vars()['corrMask'+ichr] )
				vars()['pValue'+ichr] = np.ma.masked_array( vars()['pValue'+ichr], vars()['corrMask'+ichr] )
				vars()['pCorrected'+ichr] = np.ma.masked_array( vars()['pCorrected'+ichr], vars()['corrMask'+ichr] )

				if not os.path.exists(wdvars+'ABC_stats/'+ichr):
					os.makedirs(wdvars+'ABC_stats/'+ichr)

				vars()['corr'+ichr].dump(wdvars+'ABC_stats/'+ichr+'/corr'+ichr+'_'+var+'.npy')
				vars()['pCorrected'+ichr].dump(wdvars+'ABC_stats/'+ichr+'/pCorrected'+ichr+'_'+var+'.npy')
		
		print 'Correlation Found: '+varTitle+'-RNA'
			
			#error = np.ones(shape = (nSamples,nGenes))
			#errorMedian = np.ones(shape = (nGenes))
			#topCorr = np.zeros(shape = (nGenes))
			#topR2 = np.zeros(shape = (nGenes))
			#numPeaks = np.zeros(shape = (nGenes))
			#genePeakDict = {}
			#for igene in range(nGenes):
			#	ipeak = np.where(vars()['corr'+ichr][igene,:]==np.amax(vars()['corr'+ichr][igene,:]))[0][0]
			#	
			#	if var=='abc': x = vars()[var+ichr][:,igene,ipeak]
			#	elif var=='atac': x = vars()[var+ichr][:,ipeak]
			#	ydata = vars()'expression'+ichr][:,igene]
			#	m,b = np.polyfit(x,ydata,1)
			#	yfit = m*x+b
			#	error[:,igene] = np.abs(yfit-ydata+1e-6)/(ydata+1e-6)
			#	errorMedian[igene] = np.median(error[:,igene])
			#	genePeakDict[igene] = ipeak
			#	topCorr[igene] = vars()['corr'+ichr][igene,ipeak]
			#	topR2[igene] = vars()['corr'+ichr][igene,ipeak]
			
				#plt.clf()
				#plt.plot(x, ydata, 'bo')
				#plt.plot(x, yfit, 'g-')
				#plt.title(geneName21[igene]+' by '+peakName21[ipeak]+
				#	'\nSpearmanR2 = '+str(np.round(corr21[1,igene,ipeak]**2,3))+', PearsonR2 = '+str(np.round(corr21[0,igene,ipeak]**2,3))+', KendallR2 = '+str(np.round(corr21[2,igene,ipeak]**2,3)),fontsize=12)
				#plt.xlabel('ABC')
				#plt.ylabel('Expression')
				#plt.grid(True)
				#plt.savefig(wdfigs+'lineplot_'+str(igene)+'_'+geneName21[igene]+'-'+peakName21[ipeak]+'.pdf')
			
			#if MakePlots:
				## Median Error for top peak
				#plt.clf()
				#n,bins,patches = plt.hist( errorMedian*100, bins=50)
				#plt.title(varTitle+': Median Error for Expression Prediction \nusing Top Peak in Each Gene')
				#plt.grid(True)
				#plt.xlabel('Percent Error')
				#plt.ylabel('Number of Genes')
				#plt.savefig(wdfigs+'avgError_distribution_'+var+'.pdf')
			
				##if MakePlots:
				## Array of corr
				#plt.clf()
				#fig = plt.figure(figsize = (8,6))
				#plt.imshow(corr21, cmap = 'hot_r', aspect='auto', vmin=0, interpolation='none',origin='lower')
				#plt.title('Corr '+varTitle+' -> Expression on Chr21',fontsize=15)
				#plt.xlabel('Peaks')
				#plt.ylabel('Genes')
				#plt.grid(True)
				#plt.colorbar()
				#plt.savefig(wdfigs+'expression_Corr_Chr21_'+var+'.pdf')
				#
				## Array of pValue
				#plt.clf()
				#fig = plt.figure(figsize = (9,6))
				#plt.imshow(-1*np.log2(pCorrected21), cmap = 'hot_r', aspect='auto', vmax=2, interpolation='none',origin='lower')
				#plt.title('P Corrected '+varTitle+' -> Expression on Chr21',fontsize=15)
				#plt.xlabel('Peaks')
				#plt.ylabel('Genes')
				#plt.grid(True)
				#plt.colorbar(label = '-log2(FDR P)')
				#plt.savefig(wdfigs+'expression_P_Chr21_'+var+'.pdf')
				#
				## hist of corrected P
				#plt.clf()
				#n,bins,patches = plt.hist( np.ma.compressed(pCorrected21), bins=100, range=[0,0.99])
				#plt.plot([0.05,0.05],[0,np.amax(n)*1.1],'r-')
				#plt.plot([0.01,0.01],[0,np.amax(n)*1.1],'r-')
				#plt.title('FDR P Values on Chr 21 between Expression and '+varTitle)
				#plt.grid(True)
				#plt.ylim([0,np.amax(n)*1.1])
				#plt.savefig(wd+'figures/FDR_pValues_all_'+var+'.pdf')
				#
				##sigPeaks = pCorrected21<0.01
				##sigPeaksPerGene = np.sum(sigPeaks,axis = 1)
				#
				##if MakePlots:
				##	stdDevExpression = np.std(expression21,axis=0)
				##	geneNums = np.where(stdDevExpression>10)[0]
				##	for geneNum in geneNums:
				##		plt.clf()
				##		plt.plot(geneStart21[geneNum], 0.1, 'g.', markersize = 50)
				##		plt.plot(peakStart21,pCorrected21[geneNum],'b*')
				##		plt.plot([0,np.amax(peakStart21)],[0.05,0.05],'r-')
				##		plt.plot([0,np.amax(peakStart21)],[0.01,0.01],'r-')
				##		plt.xlim([0,np.amax(peakStart21)])
				##		plt.title('FDR P Values between Gene '+geneName21[geneNum]+' and ABC Matrix')
				##		plt.xlabel('Position on Chromosome 21')
				##		plt.ylabel('FDR P Values')
				##		plt.savefig(wd+'figures/ABC_FDR_pValues_gene_'+geneName21[geneNum]+'.pdf')
				##		#plt.show()
		
		###################################################################
		# Multivariate Regression and Random Forest
		###################################################################
		print 'Multivariate Regression'
		
		cutoff = np.array([1,2,3,4]) # Number of Peaks to use in regression
		
		vars()['r2MultiV_'+ichr] = np.zeros(shape = (len(cutoff), 20, vars()['nGenes'+ichr]) )
		vars()['r2Forest_'+ichr] = np.zeros(shape = (len(cutoff), 20, vars()['nGenes'+ichr]) )
		vars()['r2Lasso_'+ichr] = np.zeros(shape = (len(cutoff), 20, vars()['nGenes'+ichr]) )
		vars()['medianErrorMultiV_'+ichr] = np.ones(shape = (len(cutoff), vars()['nGenes'+ichr]) ) # cutoff, nGenes
		vars()['medianErrorForest_'+ichr] = np.ones(shape = (len(cutoff), vars()['nGenes'+ichr]) ) # cutoff, nGenes
		vars()['medianErrorLasso_'+ichr] = np.ones(shape = (len(cutoff), vars()['nGenes'+ichr]) ) # cutoff, nGenes

		vars()['predictMultiV_'+ichr] = np.ones(shape = (4, 20, len(cutoff), vars()['nGenes'+ichr]) ) # 4 test samples, cutoff, 10 loops, nGenes
		vars()['predictForest_'+ichr] = np.ones(shape = (4, 20, len(cutoff), vars()['nGenes'+ichr]) ) # 4 test samples, cutoff, 10 loops, nGenes
		vars()['predictLasso_'+ichr] = np.ones(shape = (4, 20, len(cutoff), vars()['nGenes'+ichr]) ) # 4 test samples, cutoff, 10 loops, nGenes
		vars()['actual_'+ichr] = np.ones(shape = (4, 20, len(cutoff), vars()['nGenes'+ichr]) ) # 4 test samples, cutoff, 10 loops, nGenes
		
		for igene in range(vars()['nGenes'+ichr]):

			
			if var=='abc' or var=='atac':
				if var=='abc': xdata = vars()['abc'+ichr][:,igene]
				if var=='atac': 
					xdata = vars()['atac'+ichr]
				maxCut = np.sum( vars()['pCorrected'+ichr][igene]<0.98)
				if maxCut==0: continue
				for icut in range(len(cutoff)):
					cut = min(cutoff[icut], maxCut)
					#peakIndices = np.array(corr21[2,igene]).argsort()[-cut:]
					peakIndices = np.array( vars()['pCorrected'+ichr][igene]).argsort()[:cut]
					features = np.zeros(shape = (nSamples,len(peakIndices)) )
					targets = vars()['expression'+ichr][:,igene]
					for i in range(len(peakIndices)):
						features[:,i] = xdata[:,peakIndices[i]]

					errorMultiV = np.ones(shape = (4, 20)) # 4 test samples, 20 loops
					errorForest = np.ones(shape = (4, 20)) 
					errorLasso = np.ones(shape = (4, 20)) 
	
					for itest in range(20):
						features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=0.25)
						vars()['actual_'+ichr][:,itest,icut,igene] = targets_test
	
						########### Multivariate ###########
						clf=sklearn.linear_model.LinearRegression()
						clf.fit(features_train,targets_train)
						vars()['r2MultiV_'+ichr][icut,itest,igene] = clf.score(features_test,targets_test)
						predict = clf.predict(features_test)
						errorMultiV[:,itest] = np.abs(predict-targets_test)/targets_test
						vars()['predictMultiV_'+ichr][:,itest,icut,igene] = predict
	
						########### Random Forest ###########
						clf = sklearn.ensemble.RandomForestRegressor(n_estimators=30)
						clf.fit(features_train,targets_train)
						vars()['r2Forest_'+ichr][icut,itest,igene] = clf.score(features_test,targets_test)
						predict = clf.predict(features_test)
						errorForest[:,itest] = np.abs(predict-targets_test)/targets_test
						vars()['predictForest_'+ichr][:,itest,icut,igene] = predict
	
						########### Lasso ###########
						clf = sklearn.linear_model.Lasso()
						clf.fit(features_train,targets_train)
						vars()['r2Lasso_'+ichr][icut,itest,igene] = clf.score(features_test,targets_test)
						predict = clf.predict(features_test)
						errorLasso[:,itest] = np.abs(predict-targets_test)/targets_test
						vars()['predictLasso_'+ichr][:,itest,icut,igene] = predict
	
					vars()['medianErrorMultiV_'+ichr][icut,igene] = np.median(errorMultiV)
					vars()['medianErrorForest_'+ichr][icut,igene] = np.median(errorForest)
					vars()['medianErrorLasso_'+ichr][icut,igene] = np.median(errorLasso)

	
			if var=='abcPC' or var=='atacPC':
				maxCut = np.sum(1-vars()['corrMask'+ichr][igene])
				if maxCut==0: continue
				xdata = vars()['PC'+ichr][:,igene]
				for icut in range(len(cutoff)):
					cut = min(cutoff[icut], maxCut)
					peakIndices = np.argsort(vars()['pCorrected'+ichr][igene])[:cut]
					features = np.zeros(shape = (nSamples,len(peakIndices)) )
					targets = vars()['expression'][:,igene]
					for i in range(len(peakIndices)):
						features[:,i] = xdata[:,peakIndices[i]]

					errorMultiV = np.ones(shape = (4, 20)) # 4 test samples, 20 loops
					errorForest = np.ones(shape = (4, 20)) 
					errorLasso = np.ones(shape = (4, 20)) 
			
					for itest in range(20):
						features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=0.25)
	
						########### Multivariate ###########
						clf=sklearn.linear_model.LinearRegression()
						clf.fit(features_train,targets_train)
						vars()['r2MultiV_'+ichr][icut,itest,igene] = clf.score(features_test,targets_test)
						predict = clf.predict(features_test)
						errorMultiV[:,itest] = np.abs(predict-targets_test)/targets_test
	
						########### Random Forest ###########
						clf = sklearn.ensemble.RandomForestRegressor(n_estimators=30)
						clf.fit(features_train,targets_train)
						vars()['r2Forest_'+ichr][icut,itest,igene] = clf.score(features_test,targets_test)
						predict = clf.predict(features_test)
						errorForest[:,itest] = np.abs(predict-targets_test)/targets_test
	
						########### Lasso ###########
						clf = sklearn.linear_model.Lasso()
						clf.fit(features_train,targets_train)
						vars()['r2Lasso_'+ichr][icut,itest,igene] = clf.score(features_test,targets_test)
						predict = clf.predict(features_test)
						errorLasso[:,itest] = np.abs(predict-targets_test)/targets_test
	
					vars()['medianErrorMultiV_'+ichr][icut,igene] = np.median(errorMultiV)
					vars()['medianErrorForest_'+ichr][icut,igene] = np.median(errorForest)
					vars()['medianErrorLasso_'+ichr][icut,igene] = np.median(errorLasso)
		
		
				# Remake these plots with all predictions later
				if MakePlots:
					if not os.path.exists(wdfigs+'prediction_lineplots/chr21/'+geneName21[igene]):
						os.makedirs(wdfigs+'prediction_lineplots/chr21/'+geneName21[igene])
					plt.clf()
					plt.plot(predict,targets,'bo',markersize=8)
					m,b = np.polyfit(predict,targets,1)
					yfit = m*predict+b
					plt.plot(predict,yfit,'g-')
					plt.title(geneName21[igene]+' Expression Predictions: Top '+str(cut)+' Peaks \n Median Error = '+str(round(medianError*100,1))+', R2 = '+str(round(vars()['r2_'+ichr][icut,igene],2)),fontsize = 15)
					plt.xlabel('Predicted Gene Expression')
					plt.ylabel('Actual Gene Expression')
					plt.grid(True)
					plt.savefig(wdfigs+'prediction_lineplots/chr21/'+geneName21[igene]+'/prediction_top'+str(cut)+'.pdf')


		###################################################################
		# Assign Summary Variables
		###################################################################
		
		for icut in range(len(cutoff)):
			medianErrorAll[jchr,ivar,0,icut] = np.median( vars()['medianErrorMultiV_'+ichr][icut][vars()['medianErrorMultiV_'+ichr][icut]!=1] )
			medianErrorAll[jchr,ivar,1,icut] = np.median( vars()['medianErrorForest_'+ichr][icut][vars()['medianErrorForest_'+ichr][icut]!=1] )
			medianErrorAll[jchr,ivar,2,icut] = np.median( vars()['medianErrorLasso_'+ichr][icut][vars()['medianErrorLasso_'+ichr][icut]!=1] )
			medianR2All[jchr,ivar,0,icut] = np.median( np.median( vars()['r2MultiV_'+ichr],axis=1)[icut][np.median(vars()['r2MultiV_'+ichr],axis=1)[icut]!=0] )
			medianR2All[jchr,ivar,1,icut] = np.median( np.median( vars()['r2Forest_'+ichr],axis=1)[icut][np.median(vars()['r2Forest_'+ichr],axis=1)[icut]!=0] )
			medianR2All[jchr,ivar,2,icut] = np.median( np.median( vars()['r2Lasso_'+ichr],axis=1)[icut][np.median(vars()['r2Lasso_'+ichr],axis=1)[icut]!=0] )
			numStrongGenes[jchr,ivar,0,icut] = len(np.where(np.median( vars()['r2MultiV_'+ichr][icut,:,:],axis=0)>0.4)[0])
			numStrongGenes[jchr,ivar,1,icut] = len(np.where(np.median( vars()['r2Forest_'+ichr][icut,:,:],axis=0)>0.4)[0])
			numStrongGenes[jchr,ivar,2,icut] = len(np.where(np.median( vars()['r2Lasso_'+ichr][icut,:,:],axis=0)>0.4)[0])
		
			strongGenes[jchr][ivar][0][icut].append( list( vars()['geneName'+ichr][ np.where(np.median(vars()['r2MultiV_'+ichr][icut,:,:],axis=0)>0.4)[0] ]) )
			strongGenes[jchr][ivar][1][icut].append( list( vars()['geneName'+ichr][ np.where(np.median(vars()['r2Forest_'+ichr][icut,:,:],axis=0)>0.4)[0] ]) )
			strongGenes[jchr][ivar][2][icut].append( list( vars()['geneName'+ichr][ np.where(np.median(vars()['r2Lasso_'+ichr][icut,:,:],axis=0)>0.4)[0] ]) )
		
		numGenes[jchr,ivar,0] = vars()['medianErrorMultiV_'+ichr][1][vars()['medianErrorMultiV_'+ichr][1]!=1].shape[0]
		numGenes[jchr,ivar,1] = vars()['medianErrorForest_'+ichr][1][vars()['medianErrorForest_'+ichr][1]!=1].shape[0]
		numGenes[jchr,ivar,2] = vars()['medianErrorLasso_'+ichr][1][vars()['medianErrorLasso_'+ichr][1]!=1].shape[0]
		pctGenes[jchr,ivar,0] = vars()['medianErrorMultiV_'+ichr][1][vars()['medianErrorMultiV_'+ichr][1]!=1].shape[0]/float(vars()['nGenes'+ichr])
		pctGenes[jchr,ivar,1] = vars()['medianErrorForest_'+ichr][1][vars()['medianErrorForest_'+ichr][1]!=1].shape[0]/float(vars()['nGenes'+ichr])
		pctGenes[jchr,ivar,2] = vars()['medianErrorLasso_'+ichr][1][vars()['medianErrorLasso_'+ichr][1]!=1].shape[0]/float(vars()['nGenes'+ichr])

		np.save(wdvars+'ABC_stats/'+ichr+'/medianErrorMultiV_'+ichr+'.npy',vars()['medianErrorMultiV_'+ichr])
		np.save(wdvars+'ABC_stats/'+ichr+'/medianErrorForest_'+ichr+'.npy',vars()['medianErrorForest_'+ichr])
		np.save(wdvars+'ABC_stats/'+ichr+'/medianErrorLasso_'+ichr+'.npy',vars()['medianErrorLasso_'+ichr])

		np.save(wdvars+'ABC_stats/'+ichr+'/medianError'+ichr+'.npy',medianErrorAll[jchr])
		np.save(wdvars+'ABC_stats/'+ichr+'/medianR2_'+ichr+'.npy',medianR2All[jchr])
		np.save(wdvars+'ABC_stats/'+ichr+'/numStrongGenes'+ichr+'.npy',numStrongGenes[jchr])
		pickle.dump(strongGenes[jchr], open(wdvars+'ABC_stats/'+ichr+'/strongGenes'+ichr+'.npy','w'))
		np.save(wdvars+'ABC_stats/'+ichr+'/numGenes'+ichr+'.npy',numGenes[jchr])
		np.save(wdvars+'ABC_stats/'+ichr+'/pctGenes'+ichr+'.npy',pctGenes[jchr])

		vars()['predictMultiV_'+ichr] = vars()['predictMultiV_'+ichr].reshape(-1,*vars()['predictMultiV_'+ichr].shape[-2:]) # new shape: 80 samples x 4 cutoff x nGenes
		vars()['predictForest_'+ichr] = vars()['predictForest_'+ichr].reshape(-1,*vars()['predictForest_'+ichr].shape[-2:])
		vars()['predictLasso_'+ichr] = vars()['predictLasso_'+ichr].reshape(-1,*vars()['predictLasso_'+ichr].shape[-2:])
		vars()['actual_'+ichr] = vars()['actual_'+ichr].reshape(-1,*vars()['actual_'+ichr].shape[-2:])
		np.save(wdvars+'ABC_stats/'+ichr+'/predictMultiV_'+ichr+'.npy',vars()['predictMultiV_'+ichr])
		np.save(wdvars+'ABC_stats/'+ichr+'/predictForest_'+ichr+'.npy',vars()['predictForest_'+ichr])
		np.save(wdvars+'ABC_stats/'+ichr+'/predictLasso_'+ichr+'.npy',vars()['predictLasso_'+ichr])
		np.save(wdvars+'ABC_stats/'+ichr+'/actual_'+ichr+'.npy',vars()['actual_'+ichr])
		
np.save(wdvars+'ABC_stats/medianErrorAll',medianErrorAll)
np.save(wdvars+'ABC_stats/medianR2All',medianR2All)
np.save(wdvars+'ABC_stats/numStrongGenes',numStrongGenes)
np.save(wdvars+'ABC_stats/numGenes',numGenes)
np.save(wdvars+'ABC_stats/pctGenes',pctGenes)
exit()
###################################################################
# Average Chromosomes 
###################################################################
medianError = np.zeros(shape = (nX,nMethods,len(cutoff)))
medianR2 = np.zeros(shape = (nX,nMethods,len(cutoff)))
numStrongGenes = np.zeros(shape = (nX,nMethods,len(cutoff)))
numGenes = np.zeros(shape = (nX,nMethods,len(cutoff)))
pctGenes = np.zeros(shape = (nX,nMethods,len(cutoff)))

for jchr in range(23):
	medianError = np.mean(medianErrorAll[jchr]/numGenes[jchr])
	medianR2 = np.mean( medianR2All[jchr]/numGenes[jchr] )

			
# Number of Peaks controlling each gene
geneNum,count = np.unique(np.where(pCorrected21<0.2)[0], return_counts=True)
counts21 = np.zeros(shape=(nGenes21))
counts21[geneNum] = count
plt.clf()
n,bins,patches = plt.hist( counts21, bins=np.amax(count)+1, range=[0,np.amax(count)+1], align='left', normed=True)
plt.title(varTitle+': Number of Peaks Significantly Correlated per Gene')
plt.grid(True)
plt.xlim([-0.5,10.5])
plt.xlabel('Number of Peaks with FDP P Value < 0.2')
plt.ylabel('Genes')
plt.savefig(wdfigs+'peaks_per_gene_distribution_'+var+'.pdf')

# Number of Genes each peak controls
peakNum,count = np.unique(np.where(pCorrected21<0.2)[1], return_counts=True)
plt.clf()
n,bins,patches = plt.hist( count, bins=np.amax(count)+1, range=[0,np.amax(count)+1], align='left', normed=True)
plt.title(varTitle+': Number of Genes that Each Peak Controls')
plt.grid(True)
plt.xlim([0.5,25.5])
plt.xlabel('Number of Genes that each Peak Controls (FDP P Value < 0.2)')
plt.ylabel('Peaks')
plt.savefig(wdfigs+'genes_per_peak_distribution_'+var+'.pdf')

for icut in range(len(cutoff)):
	plt.clf()
	data = 100*medianErrorForest_21[icut][medianErrorForest_21[icut]!=1]
	n,bins,patches = plt.hist(data, bins=80, range=[0,200])
	plt.plot([np.median(data), np.median(data)],[0,20],'g-', linewidth=4)
	plt.title('Median Error for '+varTitle+' Expression Prediction \nusing Random Forests and Top '+str(cutoff[icut])+' Peaks in Each Gene')
	plt.grid(True)
	plt.ylim([0,16])
	plt.xlim([0,150])
	plt.savefig(wdfigs+'error_top'+str(cutoff[icut])+'_randomForest_distribution_'+var+'.pdf')

for icut in range(len(cutoff)):
	plt.clf()
	data = 100*medianErrorMultiV_21[icut][medianErrorMultiV_21[icut]!=1]
	n,bins,patches = plt.hist(data, bins=80, range=[0,200])
	plt.plot([np.median(data), np.median(data)],[0,20],'g-', linewidth=4)
	plt.title('Median Error for '+varTitle+' Expression Prediction \nusing Multivariate and Top '+str(cutoff[icut])+' Peaks in Each Gene')
	plt.grid(True)
	plt.ylim([0,16])
	plt.xlim([0,150])
	plt.savefig(wdfigs+'error_top'+str(cutoff[icut])+'_multivariate_distribution_'+var+'.pdf')

for icut in range(len(cutoff)):
	plt.clf()
	data = np.median(r2Forest_21,axis=1)[icut][np.median(r2Forest_21,axis=1)[icut]!=0]
	n,bins,patches = plt.hist(data, bins=60, range=[0,1])
	plt.plot([np.median(data), np.median(data)],[0,20],'g-', linewidth=4)
	plt.title('R2 for '+varTitle+' Expression Prediction \nusing Random Forests and Top '+str(cutoff[icut])+' Peaks in Each Gene')
	plt.grid(True)
	plt.ylim([0,10])
	plt.xlim([0,1])
	plt.savefig(wdfigs+'r2_top'+str(cutoff[icut])+'_randomForest_distribution_'+var+'.pdf')

for icut in range(len(cutoff)):
	plt.clf()
	data = np.median(r2MultiV_21,axis=1)[icut][np.median(r2MultiV_21,axis=1)[icut]!=0]
	n,bins,patches = plt.hist(data, bins=60, range=[0,1])
	plt.plot([np.median(data), np.median(data)],[0,20],'g-', linewidth=4)
	plt.title('R2 for '+varTitle+' Expression Prediction \nusing Multivariate and Top '+str(cutoff[icut])+' Peaks in Each Gene')
	plt.grid(True)
	plt.ylim([0,10])
	plt.xlim([0,1])
	plt.savefig(wdfigs+'r2_top'+str(cutoff[icut])+'_multivariate_distribution_'+var+'.pdf')
	

f = open(wdfiles+'expression_prediction_stats_180720_abc.csv','w')

f.write( 'X' +'\t'+ 'Method' +'\t'+ 'VarName' +'\t'+ '1' +'\t'+ '2' +'\t'+ '3' +'\t'+ '4' +'\n')

Xs = ['ABC','ATAC'] #,'ABC-PC','ATAC-PC']
Methods = ['Multivariate','Random Forest', 'Lasso']
VarNames = ['medianError','medianR2','pctGenes','numStrongGenes','strongGenes']
strongGenesVar = [strongGenesM,strongGenesF,strongGenesL]
Vars = [medianErrorAll,medianR2All,pctGenes,numStrongGenes]
for ix in range(len(Xs)):
	X = Xs[ix]
	for imethod in range(len(Methods)):
		Method = Methods[imethod]
		for ivar in range(len(VarNames)):
			if ivar==0 or ivar==1 or ivar==3:
				VarName = VarNames[ivar]
				Var = Vars[ivar]
				f.write( X +'\t'+ Method +'\t'+ VarName +'\t'+ str(Var[ix,imethod,0]) +'\t'+ str(Var[ix,imethod,1]) +'\t'+ str(Var[ix,imethod,2]) +'\t'+ str(Var[ix,imethod,3]) +'\n')
			elif ivar==2:
				VarName = VarNames[ivar]
				Var = Vars[ivar][:,:,0]
				f.write( X +'\t'+ Method +'\t'+ VarName +'\t'+ str(Var[ix,imethod]) +'\t'+ str(Var[ix,imethod]) +'\t'+ str(Var[ix,imethod]) +'\t'+ str(Var[ix,imethod]) +'\n')
			else:
				var = strongGenesVar[imethod]
				VarName = 'strongGenes'
				f.write( X +'\t'+ Method +'\t'+ VarName +'\t'+ str(var[ix][0]).replace('[','').replace(']','').replace("'",'') +'\t'+ str(var[ix][1]).replace('[','').replace(']','').replace("'",'') +'\t'+ str(var[ix][2]).replace('[','').replace(']','').replace("'",'') +'\t'+ str(var[ix][3]).replace('[','').replace(']','').replace("'",'') +'\n')
f.close()

#for icut in range(len(cutoff)):
#	avgR2All[irna,icut] = np.mean(np.array(r2M_21[icut]))
#	avgR2above0[irna,icut] = np.ma.mean(r2_21[irna,icut][r2_21[irna,icut]>0])
#	numGenesAbove0[irna,icut] = len(r2_21[irna,icut][r2_21[irna,icut]>0])
#
#	plt.clf()
#	plt.figure(figsize = (20,6))
#	#plt.bar(np.arange(len(r2_21[irna,icut])), r2_21[irna,icut], color='b', edgecolor='k', tick_label=geneMatrix21)
#	plt.bar(np.arange(len(r2_21[irna,icut])), r2_21[irna,icut], color='b', edgecolor='k')
#	plt.title('Explained Variance: ABC, pCorrected<'+str(cutoff[icut])+
#		'\n% Var Explained = '+str(np.round(percentExplainedVar[irna,icut],1))+
#		', Avg r2 = '+str(np.round(avgR2All[irna,icut],2))+
#		', Avg r2 (above 0) = '+str(np.round(avgR2above0[irna,icut],2))+
#		', Num Genes r2>0 = '+str(numGenesAbove0[irna,icut])+'/'+str(len(geneMatrix21)) )
#	plt.xlabel('Genes')
#	plt.ylabel('% Explained Variance')
#	plt.grid(True)
#	plt.savefig(wd+'figures/r2_by_gene_ABC_RNA'+rnaType+'_cut'+str(int(100*cutoff[icut]))+'.pdf')
#
#plt.clf()
#fig = plt.figure(figsize=(8,6))
#plt.plot(cutoff,percentExplainedVar[0],'-ob',markersize=10,label='ABC vs RNA')
#plt.plot(cutoff,percentExplainedVar[1],marker='o',markersize=10,color='orange',label='ABC vs log(RNA)')
#plt.plot(cutoff,percentExplainedVar[2],marker='o',markersize=10,color='red',label='ABC vs e^(RNA)')
#plt.title('Percent of Global Variance Explained by Model (Chr21)',fontsize=15)
#plt.xlabel('Number of Peaks')
#plt.ylabel('sum( variance * r^2 ) across genes')
#plt.legend(loc='upper left')
#plt.grid(True)
#plt.ylim([0,100])
#plt.savefig(wdfigs+'comparison_pct_variance_explained_chr21.pdf')
#plt.show()
#
#plt.clf()
#fig = plt.figure(figsize=(8,6))
#plt.plot(cutoff,100*avgR2All[0],'-ob',markersize=10,label='ABC vs RNA')
#plt.plot(cutoff,100*avgR2All[1],marker='o',markersize=10,color='orange',label='ABC vs log(RNA)')
#plt.plot(cutoff,100*avgR2All[2],marker='o',markersize=10,color='red',label='ABC vs e^(RNA)')
#plt.title('Average Explained Variance of Genes (Chr21)',fontsize=15)
#plt.xlabel('Number of Peaks')
#plt.ylabel('100 * r^2')
#plt.legend(loc='upper left')
#plt.grid(True)
#plt.ylim([0,100])
#plt.savefig(wdfigs+'comparison_avg_r2_all_chr21.pdf')
#plt.show()
#
#plt.clf()
#fig = plt.figure(figsize=(8,6))
#plt.plot(cutoff,100*avgR2above0[0],'-ob',markersize=10,label='ABC vs RNA')
#plt.plot(cutoff,100*avgR2above0[1],marker='o',markersize=10,color='orange',label='ABC vs log(RNA)')
#plt.plot(cutoff,100*avgR2above0[2],marker='o',markersize=10,color='red',label='ABC vs e^(RNA)')
#plt.title('Average Explained Variance of Affected Genes (Chr21)',fontsize=15)
#plt.xlabel('Numer of Peaks')
#plt.ylabel('100 * r^2')
#plt.legend(loc='lower left')
#plt.grid(True)
#plt.ylim([0,100])
#plt.savefig(wdfigs+'comparison_avg_r2_above0_chr21.pdf')
#plt.show()
#
#plt.clf()
#fig = plt.figure(figsize=(8,6))
#plt.plot(cutoff,100*numGenesAbove0[0]/135,'-ob',markersize=10,label='ABC vs RNA')
#plt.plot(cutoff,100*numGenesAbove0[1]/135,marker='o',markersize=10,color='orange',label='ABC vs log(RNA)')
#plt.plot(cutoff,100*numGenesAbove0[2]/135,marker='o',markersize=10,color='red',label='ABC vs e^(RNA)')
#plt.title('Percent of Genes with Correlations (Chr21)',fontsize=15)
#plt.xlabel('Numer of Peaks')
#plt.ylabel('# of Genes')
#plt.legend(loc='upper left')
#plt.grid(True)
#plt.ylim([0,100])
#plt.savefig(wdfigs+'comparison_num_genes_correlated_chr21.pdf')
#plt.show()









