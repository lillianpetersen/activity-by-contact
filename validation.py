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
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sns
#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

wd = '/pbld/mcg/lillianpetersen/ABC/'
wdvars = '/pbld/mcg/lillianpetersen/ABC/saved_variables/'
wdfigs = '/pbld/mcg/lillianpetersen/ABC/figures/'
wddata = '/pbld/mcg/lillianpetersen/ABC/data/'
wdfiles = '/pbld/mcg/lillianpetersen/ABC/written_files/'
MakePlots = False

# MCG B ALL Samples
nSamples = 92
nChr = 23

###################################################################
# Load RNA
###################################################################
print('Load RNA')
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	if not 'expression'+ichr in globals():
		# Load arrays saved from load_rna.npy
		vars()['expression'+ichr] = np.load(wdvars+'validation_K562/RNA/expression'+ichr+'.npy')
		vars()['geneName'+ichr] = np.load(wdvars+'validation_K562/RNA/geneName'+ichr+'.npy')
		vars()['chrRNA'+ichr] = np.load(wdvars+'validation_K562/RNA/chrRNA'+ichr+'.npy')
		vars()['positionRNA'+ichr] = np.load(wdvars+'validation_K562/RNA/positionRNA'+ichr+'.npy')
		vars()['direction'+ichr] = np.load(wdvars+'validation_K562/RNA/direction'+ichr+'.npy')

		#keep = vars()['expression'+ichr]>0.015
		#keep = np.sum(keep,axis=0)>46
		#vars()['expression'+ichr] = vars()['expression'+ichr][keep]
		#vars()['chrRNA'+ichr] = vars()['chrRNA'+ichr][keep]
		#vars()['geneName'+ichr] = vars()['geneName'+ichr][keep]
		#vars()['direction'+ichr] = vars()['direction'+ichr][keep]
		#vars()['positionRNA'+ichr] = vars()['positionRNA'+ichr][:,keep]

		directiontmp = np.array(vars()['direction'+ichr])
		directiontmp[directiontmp==1]=0 # +
		directiontmp[directiontmp==-1]=1 # -

		vars()['tss'+ichr] = np.zeros(shape=(vars()['chrRNA'+ichr].shape[0]))
		for i in range(len(vars()['chrRNA'+ichr])):
			vars()['tss'+ichr][i] = vars()['positionRNA'+ichr][:,i][directiontmp[i]]


		# Limit by standard dev / mean expression
		#stdMask = np.std(vars()['expression'+ichr],axis=0) / np.mean(vars()['expression'+ichr],axis=0) < 0.25 # True = bad
		#maskFull = np.zeros(shape = (nSamples,len(stdMask)), dtype = bool)
		#for isample in range(nSamples):
		#	maskFull[isample] = stdMask
		#mask2 = np.zeros(shape = (2,len(stdMask)), dtype = bool)
		#for isample in range(2):
		#	mask2[isample] = stdMask
		#vars()['expression'+ichr] = np.ma.compress_cols( np.ma.masked_array(vars()['expression'+ichr], maskFull) )
		#vars()['geneName'+ichr] = np.ma.compressed( np.ma.masked_array(vars()['geneName'+ichr], stdMask) )
		#vars()['chrRNA'+ichr] = np.ma.compressed( np.ma.masked_array(vars()['chrRNA'+ichr], stdMask) )
		#vars()['direction'+ichr] = np.ma.compressed( np.ma.masked_array(vars()['direction'+ichr], stdMask) )
		#vars()['positionRNA'+ichr] = np.ma.compress_cols( np.ma.masked_array(vars()['positionRNA'+ichr], mask2) )

	#if not 'expressionNorm'+ichr in globals():
	#	vars()['expressionNorm'+ichr] = np.zeros(shape = (vars()['expression'+ichr].shape))
	#	for igene in range(len(vars()['geneName'+ichr])):
	#		vars()['expressionNorm'+ichr][:,igene] = sklearn.preprocessing.scale(vars()['expression'+ichr][:,igene])

###################################################################
# Load Activity
###################################################################
print('Load Activity')
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	if not 'activity'+ichr in globals():
		vars()['activity'+ichr] = np.load(wdvars+'validation_K562/ATAC/activity'+ichr+'.npy')
		vars()['chrActivity'+ichr] = np.load(wdvars+'validation_K562/ATAC/chrATAC'+ichr+'.npy')
		vars()['peakName'+ichr] = np.load(wdvars+'validation_K562/ATAC/peakName'+ichr+'.npy')
		vars()['positionActivity'+ichr] = np.load(wdvars+'validation_K562/ATAC/positionATAC'+ichr+'.npy')
		vars()['nPeaks'+ichr] = np.load(wdvars+'validation_K562/ATAC/nPeaks'+ichr+'.npy')

###################################################################
# Load HiC
###################################################################
print('load HiC')
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	if not 'hic'+ichr in globals():
		# Load arrays saved from load_validation_hic.npy
		vars()['hic'+ichr] = np.load( wdvars+'validation_K562/HiC/hic'+ichr+'.npy')
		vars()['geneStart'+ichr] = np.load( wdvars+'validation_K562/HiC/geneStart'+ichr+'.npy' )
		vars()['geneMatrix'+ichr] = np.load( wdvars+'validation_K562/HiC/geneMatrix'+ichr+'.npy' )
		vars()['peakStart'+ichr] = np.load( wdvars+'validation_K562/HiC/peakStart'+ichr+'.npy' )
		vars()['peakPos'+ichr] = np.load( wdvars+'validation_K562/HiC/peakPos'+ichr+'.npy' )
		vars()['peakMatrix'+ichr] = np.load( wdvars+'validation_K562/HiC/peakMatrix'+ichr+'.npy' )

		if np.amin(np.isin(vars()['peakMatrix'+ichr], vars()['peakName'+ichr])) == False:
			indices = np.isin(vars()['peakMatrix'+ichr], vars()['peakName'+ichr])
			vars()['hic'+ichr] = vars()['hic'+ichr][:,indices]
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
			vars()['hic'+ichr] = vars()['hic'+ichr][indices,:]
			vars()['geneStart'+ichr] = vars()['geneStart'+ichr][indices]
			vars()['geneMatrix'+ichr] = vars()['geneMatrix'+ichr][indices]
		if np.amin(np.isin(vars()['geneName'+ichr], vars()['geneMatrix'+ichr])) == False:
			print 'Fixing RNA indexing: Chromosome '+ichr
			indices = np.isin(vars()['geneName'+ichr], vars()['geneMatrix'+ichr])
			vars()['expression'+ichr] = vars()['expression'+ichr][indices]
			vars()['geneName'+ichr] = vars()['geneName'+ichr][indices]
			vars()['tss'+ichr] = vars()['tss'+ichr][indices]
			vars()['positionRNA'+ichr] = vars()['positionRNA'+ichr][:,indices]

###################################################################
# Create ABC Matrix
###################################################################
print 'Create ABC'
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	if not 'abc'+ichr in globals():
		vars()['abc'+ichr] = np.load(wdvars+'validation_K562/ABC/abc'+ichr+'.npy')
		vars()['abc'+ichr] = np.ma.masked_array(vars()['abc'+ichr], vars()['abc'+ichr]==0)
	#	nGenes = vars()['hic'+ichr].shape[0]
	#	nPeaks = vars()['hic'+ichr].shape[1]
	#	activityMean = np.mean(vars()['activity'+ichr],axis=0)
	#	vars()['abc'+ichr] = np.zeros(shape = (nGenes, nPeaks))
	#	peakPos = np.mean( vars()['positionActivity'+ichr][:,:],axis=0)
	#	for igene in np.arange(nGenes):
	#		genePos = vars()['geneStart'+ichr][igene]
	#		usePeak = np.abs(peakPos-genePos)<1500000 # True = good = within 1.5Mb
	#		Sum = np.sum( activityMean[usePeak] * vars()['hic'+ichr][igene,usePeak])
	#		vars()['abc'+ichr][igene,usePeak] = (activityMean[usePeak] * vars()['hic'+ichr][igene,usePeak]) / Sum
	#		if np.amax(vars()['abc'+ichr][igene,:]>1)==True: exit()
	#		#vars()['abc'+ichr][igene,usePeak] = (activityMean[usePeak] * vars()['hic'+ichr][igene,usePeak])
	#	vars()['abc'+ichr] = np.ma.masked_array(vars()['abc'+ichr],vars()['abc'+ichr]==0)

###################################################################
# Create Distance Matrix
###################################################################
print '\nCompute Distance'
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
#for ichr in ['21']:
	if not 'dist'+ichr in globals():
		print ichr,
		#if np.amin(vars()['peakName'+ichr]==vars()['peakMatrix'+ichr]) == 0:
		#	print 'Error: Peak array sizes do not match'
		#	exit()

		nGenes = vars()['expression'+ichr].shape[0]
		nPeaks = vars()['activity'+ichr].shape[0]
		vars()['dist'+ichr] = np.zeros(shape = (nGenes, nPeaks))
		peakPos = np.mean( vars()['positionActivity'+ichr][:,:],axis=0)
		for igene in np.arange(nGenes):
			genePos = vars()['tss'+ichr][igene]
			usePeak1 = np.abs(peakPos-genePos)<1500000 # True = good = within 1.5Mb
			usePeak2 = np.abs(peakPos-genePos)>5000 # True = good = outside 5kb
			usePeak = usePeak1==usePeak2
			vars()['dist'+ichr][igene,usePeak] = np.abs(peakPos[usePeak]-genePos)
		vars()['dist'+ichr] = np.ma.masked_array(vars()['dist'+ichr],vars()['dist'+ichr]==0)

		#if MakePlots:
		plt.clf()
		fig = plt.figure(figsize = (10,6))
		plt.imshow(vars()['dist'+ichr], cmap = 'hot_r', aspect='auto', interpolation='none',origin='lower', vmax=1.7e6)
		plt.title('Dist Matrix: Chromosome '+ichr,fontsize=18)
		plt.xlabel('Peaks')
		plt.ylabel('Genes')
		plt.grid(True)
		plt.colorbar()
		plt.savefig(wdfigs+'dist_validationK562_Chr'+ichr+'.pdf')
		if ichr=='X': print '\n',

###################################################################
# Load Reference Peaks
###################################################################
print '\nLoad Reference Peaks'
validationDataFile = pd.read_csv(wddata+'validation_K562/known_connections/known_connections_full_hg38.txt', header=0, sep='\t')

peakChrV = np.array(validationDataFile['peakChr'])
peakPosV = np.array([validationDataFile['peakStart'],validationDataFile['peakStop']])
peakNameV = np.array(validationDataFile['peakName'])

for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22']:

	# Limit to only the current chromosome
	chrMask = peakChrV=='chr'+ichr
	vars()['peakPosV'+ichr] = peakPosV[:,chrMask]
	vars()['peakNameV'+ichr] = peakNameV[chrMask]

	# Sort variables by position
	sort = np.argsort(vars()['peakPosV'+ichr][0])
	vars()['peakPosV'+ichr] = vars()['peakPosV'+ichr][:,sort]
	vars()['peakNameV'+ichr] = vars()['peakNameV'+ichr][sort]

	# create 1D array of each bp and what peak it matches
	chrLen = np.amax( [np.amax(vars()['positionActivity'+ichr][1]), np.amax(vars()['peakPosV'+ichr][1])] )
	peakPos = -9999*np.ones(shape=chrLen,dtype=int)
	for ipeak in range( len(vars()['positionActivity'+ichr][0]) ):
		peakPos[ vars()['positionActivity'+ichr][0,ipeak]-2000:vars()['positionActivity'+ichr][1,ipeak]+2000 ] = ipeak

	vars()['peakIndex'+ichr] = np.zeros(shape=(vars()['peakPosV'+ichr].shape[1]),dtype=int)
	goodVal = 0
	noVal = 0
	for ival in range( len(vars()['peakIndex'+ichr]) ):
		ipeak = np.amax( peakPos[vars()['peakPosV'+ichr][0,ival]:vars()['peakPosV'+ichr][1,ival]] )
		if ival>-1:
			vars()['peakIndex'+ichr][ival] = ipeak
		else:
			vars()['peakIndex'+ichr][ival] = -9999

	print 'Chromosome', ichr+':', str(int(100*np.round( len(np.unique(vars()['peakIndex'+ichr][vars()['peakIndex'+ichr]>-1]))/float(len(np.unique(vars()['peakPosV'+ichr][0]))),2)))+'% match','out of', len(np.unique(vars()['peakPosV'+ichr][0])), 'peaks'

####################################################################
# Load Reference Genes
####################################################################
print '\nLoad Reference Genes'

geneNameV = np.array(validationDataFile['geneName'])
geneIDV = np.array(validationDataFile['geneID'])
directionV = np.array(validationDataFile['strand'])

pValue = np.array(validationDataFile['pValue_adjusted'])
beta = np.array(validationDataFile['beta'])
intercept = np.array(validationDataFile['intercept'])
fold_change = np.array(validationDataFile['fold_change'])

for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22']:
	# Limit to only the current chromosome
	chrMask = peakChrV=='chr'+ichr
	vars()['geneNameV'+ichr] = geneNameV[chrMask]
	vars()['directionV'+ichr] = directionV[chrMask]
	vars()['pValue1D'+ichr] = pValue[chrMask]
	vars()['beta1D'+ichr] = beta[chrMask]
	vars()['intercept1D'+ichr] = intercept[chrMask]
	vars()['fold_change1D'+ichr] = fold_change[chrMask]

	# Sort variables by position
	sort = np.argsort(vars()['peakPosV'+ichr][0])
	vars()['geneNameV'+ichr] = vars()['geneNameV'+ichr][sort]
	vars()['directionV'+ichr] = vars()['directionV'+ichr][sort]
	vars()['pValue1D'+ichr] = vars()['pValue1D'+ichr][sort]
	vars()['beta1D'+ichr] = vars()['beta1D'+ichr][sort]
	vars()['intercept1D'+ichr] = vars()['intercept1D'+ichr][sort]
	vars()['fold_change1D'+ichr] = vars()['fold_change1D'+ichr][sort]

	# geneIndex = array of reference genes to which rna gene they correpond to
	vars()['geneIndex'+ichr] = -9999*np.ones(shape=(vars()['geneNameV'+ichr].shape),dtype=int)
	for igene in range(len(vars()['geneNameV'+ichr])):
	    if np.amax(vars()['geneName'+ichr] == vars()['geneNameV'+ichr][igene])==True:
	   	 index = np.where( vars()['geneName'+ichr] == vars()['geneNameV'+ichr][igene])[0][0]
	   	 vars()['geneIndex'+ichr][igene] = index

	print 'Chromosome', ichr+':', str(int(100*np.round( len(np.unique(vars()['geneIndex'+ ichr][vars()['geneIndex'+ichr]>-1]))/float(len(np.unique(vars()['geneNameV'+ichr]))),2))) +'% match','out of', len(np.unique(vars()['geneNameV'+ichr])), 'genes'

###################################################################
# Create Arrays of Connections
###################################################################
print 'Create Arrays of Connections'

abcConnected = []
abcNotConnected= []
abcUnknown = []
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22']:
	nPeaks = len(vars()['expression'+ichr])
	nGenes = len(vars()['activity'+ichr])
	## connections = array of p-values, genes by peaks
	vars()['connections'+ichr] = np.ones(shape=(nPeaks,nGenes))
	vars()['beta'+ichr] = np.ones(shape=(nPeaks,nGenes))
	vars()['intercept'+ichr] = np.ones(shape=(nPeaks,nGenes))
	vars()['foldChange'+ichr] = np.ones(shape=(nPeaks,nGenes))
	for i in range(len(vars()['peakIndex'+ichr])):
		ipeak = vars()['peakIndex'+ichr][i]
		igene = vars()['geneIndex'+ichr][i]
		if ipeak<-1 or igene<-1: continue
		vars()['connections'+ichr][igene,ipeak] = vars()['pValue1D'+ichr][i]
		vars()['beta'+ichr][igene,ipeak] = vars()['beta1D'+ichr][i]
		vars()['intercept'+ichr][igene,ipeak] = vars()['intercept1D'+ichr][i]
		vars()['foldChange'+ichr][igene,ipeak] = vars()['fold_change1D'+ichr][i]

	exit()
	#vars()['connections'+ichr] = np.ma.masked_array(vars()['connections'+ichr],vars()['connections'+ichr]==1)
	abcConnected.append(vars()['abc'+ichr][vars()['connections'+ichr]<0.05])
	abcNotConnected.append(vars()['abc'+ichr][vars()['connections'+ichr][vars()['connections'+ichr]<1]>0.05])
	abcUnknown.append(vars()['abc'+ichr][vars()['connections'+ichr]==1])

#	meanABC = np.array(vars()['abc'+ichr])
#	activity = np.mean(np.array(vars()['activity'+ichr]),axis=0)
#	meanActivity = np.zeros(shape=(meanABC.shape))
#	for igene in range(len(meanABC[:,0])):
#		meanActivity[igene,:] = activity
#	rna = np.mean(np.array(vars()['expression'+ichr]),axis=0)
#	meanRNA = np.zeros(shape=(meanABC.shape))
#	for ipeak in range(len(meanABC[0,:])):
#		meanRNA[:,ipeak] = rna
#	meanDist = np.array(vars()['dist'+ichr])
#	meanCorr = np.array(vars()['corr'+ichr])
#
#	betaG = vars()['geneBeta'+ichr][vars()['geneConnections'+ichr]]
#	betaP = np.concatenate(vars()['peakBeta'+ichr][vars()['peakConnections'+ichr]])
#	betaPconnectedSep.append(betaP)
#	betaGconnectedSep.append(betaG)
#	abcConnectedSep.append( meanABC[vars()['geneConnections'+ichr],vars()['peakConnections'+ichr]] )
#	abcNotConnectedSep.append( np.ma.compressed(np.ma.masked_array(meanABC,notConnectedArray)))
#	abcUnknownSep.append( np.ma.compressed(np.ma.masked_array(meanABC,unknownArray)))
#
#	distConnectedSep.append( meanDist[vars()['geneConnections'+ichr],vars()['peakConnections'+ichr]] )
#	distNotConnectedSep.append( np.ma.compressed(np.ma.masked_array(meanDist,notConnectedArray)))
#	distUnknownSep.append( np.ma.compressed(np.ma.masked_array(meanDist,unknownArray)))
#
#	activityConnectedSep.append( meanActivity[vars()['geneConnections'+ichr],vars()['peakConnections'+ichr]] )
#	activityNotConnectedSep.append( np.ma.compressed(np.ma.masked_array(meanActivity,notConnectedArray)))
#	activityUnknownSep.append( np.ma.compressed(np.ma.masked_array(meanActivity,unknownArray)))
#
#	rnaConnectedSep.append( meanRNA[vars()['geneConnections'+ichr],vars()['peakConnections'+ichr]] )
#	rnaNotConnectedSep.append( np.ma.compressed(np.ma.masked_array(meanRNA,notConnectedArray)))
#	rnaUnknownSep.append( np.ma.compressed(np.ma.masked_array(meanRNA,unknownArray)))
#
#	corrConnectedSep.append( meanCorr[vars()['geneConnections'+ichr],vars()['peakConnections'+ichr]] )
#	corrNotConnectedSep.append( np.ma.compressed(np.ma.masked_array(meanCorr,notConnectedArray)))
#	corrUnknownSep.append( np.ma.compressed(np.ma.masked_array(meanCorr,unknownArray)) )
#
######## Define Masks #######
#abcConnected = np.array(np.concatenate(abcConnectedSep))
#distConnected = np.array(np.concatenate(distConnectedSep))
#mask = np.invert(np.amax([abcConnected==0,distConnected==0], axis=0))
#
#abcNotConnected = np.array(np.concatenate(abcNotConnectedSep))
#distNotConnected = np.array(np.concatenate(distNotConnectedSep))
#maskNotConnected = np.invert(np.amax([abcNotConnected==0,distNotConnected==0], axis=0))
#
#abcUnknown = np.array(np.concatenate(abcUnknownSep))
#distUnknown = np.array(np.concatenate(distUnknownSep))
#maskUnknown = np.invert(np.amax([abcUnknown==0,distUnknown==0], axis=0))
#
######## Concatenate Arrays #######
#distConnected = distConnected[mask]
#distNotConnected = distNotConnected[maskNotConnected]
#distUnknown = distUnknown[maskUnknown]
#
#activityConnected = np.array(np.concatenate(activityConnectedSep))
#activityConnected = activityConnected[mask]
#activityNotConnected = np.array(np.concatenate(activityNotConnectedSep))
#activityNotConnected = activityNotConnected[maskNotConnected]
#activityUnknown = np.array(np.concatenate(activityUnknownSep))
#activityUnknown = activityUnknown[maskUnknown]
#
#rnaConnected = np.array(np.concatenate(rnaConnectedSep))
#rnaConnected = rnaConnected[mask]
#rnaNotConnected = np.array(np.concatenate(rnaNotConnectedSep))
#rnaNotConnected = rnaNotConnected[maskNotConnected]
#rnaUnknown = np.array(np.concatenate(rnaUnknownSep))
#rnaUnknown = rnaUnknown[maskUnknown]
#
#corrConnected = np.array(np.concatenate(corrConnectedSep))
#corrConnected = corrConnected[mask]
#corrNotConnected = np.array(np.concatenate(corrNotConnectedSep))
#corrNotConnected = corrNotConnected[maskNotConnected]
#corrUnknown = np.array(np.concatenate(corrUnknownSep))
#corrUnknown = corrUnknown[maskUnknown]
#
#betaPconnect = np.array(np.concatenate(betaPconnectedSep))
#betaPconnect = betaPconnect[mask]
#betaGconnect = np.array(np.concatenate(betaGconnectedSep))
#betaGconnect = betaGconnect[mask]
#abcConnected = abcConnected[mask]
#abcNotConnected = abcNotConnected[maskNotConnected]
#abcUnknown = abcUnknown[maskUnknown]

###################################################################
# Distributions
###################################################################
############## ABC ##############
#logbins = np.logspace(np.log10(1e-6),np.log10(1),50)
connections1 = np.ma.masked_array(connections1,connections1==1)
abcConnected = connections1

logbins = np.logspace(np.log10(np.amin(abcUnknown)),np.log10(np.amax(abcConnected)),50)

Tabc,Pabc = stats.ttest_ind(abcConnected,abcUnknown,equal_var=False)

plt.clf()
fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)

#plt.hist(abcNotConnected, bins=logbins, range=[0,1], normed=True, color='b', alpha=0.8, label='No Connection')
axs[0].hist(abcConnected, bins=logbins, color='lime', alpha=0.9, label='caQTL-eQTL')
axs[0].plot([np.median(abcConnected),np.median(abcConnected)],[0,11],'k',linewidth=2)
axs[0].grid(True)
axs[0].legend(loc='upper left',fontsize=12)
#plt.ylim([0,13])
plt.xscale('log')

axs[1].hist(abcNotConnected, bins=logbins, color='b', alpha=0.9, label='caQTL, no eQTL')
axs[1].plot([np.median(abcNotConnected),np.median(abcNotConnected)],[0,2500],'k',linewidth=2)
axs[1].grid(True)
axs[1].legend(loc='upper left',fontsize=12)
#plt.ylim([0,13])
plt.xscale('log')

axs[2].hist(abcUnknown, bins=logbins, color='silver', alpha=0.8, label='no caQTL/eQTL')
axs[2].plot([np.median(abcUnknown),np.median(abcUnknown)],[0,50000],'k',linewidth=2)
plt.ylabel('Frequency')
plt.xlabel('ABC Score, log scale')
plt.xscale('log')
plt.grid(True)
plt.legend(loc='upper left',fontsize=12)
fig.suptitle('ABC Score of Known and Unknown Connections\nT-test = '+str(np.round(Tabc,2))+', P = '+str(np.format_float_scientific(Pabc,1)),fontsize=17)
plt.savefig(wdfigs+'known_unknown_abc_distributions.pdf')
plt.show()

plt.clf()
#plt.plot(distUnknown,abcUnknown,'o',color='silver',markersize=2.4)
plt.plot(distNotConnected,abcNotConnected,'bo',markersize=2.4)
plt.plot(distConnected,abcConnected,'go',markersize=10, label='caQTL-eQTL')
plt.plot(distConnected,abcConnected,'go',markersize=10, label='caQTL, no eQTL')
plt.title('Connections by Distance and ABC',fontsize = 17)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Distance (log scale)')
plt.ylabel('ABC (log scale)')
plt.xlim([5000,1500000])
plt.grid(True)
plt.ylim([np.amin(abcNotConnected),np.amax(abcConnected)])
plt.savefig(wdfigs+'abc_dist_loglog_validation.pdf')
plt.show()

############## Activity ##############
logbins = np.arange(0,100,100/70.)

Tactivity,Pactivity = stats.ttest_ind(activityConnected,activityUnknown,equal_var=False)

plt.clf()
fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)

axs[0].hist(activityConnected, bins=logbins, color='lime', alpha=0.9, label='caQTL-eQTL')
#axs[0].plot([np.median(activityConnected),np.median(activityConnected)],[0,45],'k',linewidth=2)
axs[0].grid(True)
axs[0].legend(loc='upper right',fontsize=12)

axs[1].hist(activityNotConnected, bins=logbins, color='b', alpha=0.8, label='caQTL, no eQTL')
#axs[1].plot([np.median(activityNotConnected),np.median(activityNotConnected)],[0,700],'k',linewidth=2)
axs[1].grid(True)
axs[1].legend(loc='upper right',fontsize=12)

axs[2].hist(activityUnknown, bins=logbins, color='silver', alpha=0.8, label='no caQTL/eQTL')
#axs[2].plot([np.median(activityUnknown),np.median(activityUnknown)],[0,15000],'k',linewidth=2)
#plt.ylim([0,400000])
plt.ylabel('Frequency')
plt.xlabel('Activity from Peak to Gene')
#plt.xscale('log')
plt.grid(True)
#plt.xlim([0,500])
plt.legend(loc='upper right',fontsize=12)
fig.suptitle('Activity of Known and Unknown Connections\nT-test = '+str(np.round(Tactivity,2))+', P = '+str(np.format_float_scientific(Pactivity,1)),fontsize=17)
plt.savefig(wdfigs+'known_unknown_activity_distributions.pdf')
plt.show()

############## RNA ##############
logbins = np.arange(0,3,3/70.)

Trna,Prna = stats.ttest_ind(rnaConnected,rnaUnknown,equal_var=False)

plt.clf()
fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)

axs[0].hist(rnaConnected, bins=logbins, color='lime', alpha=0.9, label='caQTL-eQTL')
#axs[0].plot([np.median(rnaConnected),np.median(rnaConnected)],[0,45],'k',linewidth=2)
axs[0].grid(True)
axs[0].legend(loc='upper right',fontsize=12)

axs[1].hist(rnaNotConnected, bins=logbins, color='b', alpha=0.8, label='caQTL, no eQTL')
#axs[1].plot([np.median(rnaNotConnected),np.median(rnaNotConnected)],[0,700],'k',linewidth=2)
axs[1].grid(True)
axs[1].legend(loc='upper right',fontsize=12)

axs[2].hist(rnaUnknown, bins=logbins, color='silver', alpha=0.8, label='no caQTL/eQTL')
#axs[2].plot([np.median(rnaUnknown),np.median(rnaUnknown)],[0,15000],'k',linewidth=2)
#plt.ylim([0,400000])
plt.ylabel('Frequency')
plt.xlabel('RNA from Peak to Gene')
#plt.xscale('log')
plt.grid(True)
#plt.xlim([0,500])
plt.legend(loc='upper right',fontsize=12)
fig.suptitle('RNA of Known and Unknown Connections\nT-test = '+str(np.round(Trna,2))+', P = '+str(np.format_float_scientific(Prna,1)),fontsize=17)
plt.savefig(wdfigs+'known_unknown_rna_distributions.pdf')
plt.show()

############## Dist ##############
#logbins = np.arange(0,1500000,1500000/70.)
logbins = np.logspace(np.log10(np.amin(distConnected)),np.log10(np.amax(distUnknown)),70.)

Tdist,Pdist = stats.ttest_ind(distConnected,distUnknown,equal_var=False)

plt.clf()
fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)

n,bins,patches = axs[0].hist(distConnected, bins=logbins, color='lime', alpha=0.9, label='caQTL-eQTL')
axs[0].plot([np.median(distConnected),np.median(distConnected)],[0,np.amax(n)],'k',linewidth=2)
axs[0].grid(True)
plt.xscale('log')
axs[0].legend(loc='upper right',fontsize=12)

n,bins,patches = axs[1].hist(distNotConnected, bins=logbins, color='b', alpha=0.8, label='caQTL, no eQTL')
axs[1].plot([np.median(distNotConnected),np.median(distNotConnected)],[0,np.amax(n)],'k',linewidth=2)
plt.xscale('log')
axs[1].grid(True)
axs[1].legend(loc='upper right',fontsize=12)

n,bins,patches = axs[2].hist(distUnknown, bins=logbins, color='silver', alpha=0.8, label='no caQTL/eQTL')
axs[2].plot([np.median(distUnknown),np.median(distUnknown)],[0,np.amax(n)],'k',linewidth=2)
#plt.ylim([0,400000])
plt.ylabel('Frequency')
plt.xlabel('Distance from Peak to Gene')
plt.xscale('log')
plt.grid(True)
plt.xlim([0,1500000])
plt.legend(loc='upper right',fontsize=12)
fig.suptitle('Distance of Known and Unknown Connections\nT-test = '+str(np.round(Tdist,2))+', P = '+str(np.format_float_scientific(Pdist,1)),fontsize=17)
plt.savefig(wdfigs+'known_unknown_dist_distributions.pdf')
plt.show()

############## Corr ##############
logbins = np.arange(-0.6,0.6,1.2/70)
#corrConnected = np.abs(corrConnected)
#corrNotConnected = np.abs(corrNotConnected)
#corrUnknown = np.abs(corrUnknown)

Tcorr,Pcorr = stats.ttest_ind(corrConnected,corrUnknown,equal_var=False)

plt.clf()
fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)

axs[0].hist(corrConnected, bins=logbins, color='lime', alpha=0.9, label='caQTL-eQTL')
axs[0].plot([np.median(corrConnected),np.median(corrConnected)],[0,8],'k',linewidth=2)
axs[0].grid(True)
axs[0].legend(loc='upper right',fontsize=12)

axs[1].hist(corrNotConnected, bins=logbins, color='b', alpha=0.9, label='caQTL, no eQTL')
axs[1].plot([np.median(corrNotConnected),np.median(corrNotConnected)],[0,1520],'k',linewidth=2)
axs[1].grid(True)
axs[1].legend(loc='upper right',fontsize=12)

axs[2].hist(corrUnknown, bins=logbins, color='silver', alpha=0.8, label='no caQTL/eQTL')
axs[2].plot([np.median(corrUnknown),np.median(corrUnknown)],[0,35000],'k',linewidth=2)
plt.ylabel('Frequency')
plt.xlabel('Correlation from Peak to Gene')
plt.grid(True)
plt.legend(loc='upper right',fontsize=12)
fig.suptitle('Correlation of Known and Unknown Connections\nT-test = '+str(np.round(Tcorr,2))+', P = '+str(np.format_float_scientific(Pcorr,1)),fontsize=17)
plt.savefig(wdfigs+'known_unknown_corr_distributions.pdf')
plt.show()

###################################################################
# Precision Recall
###################################################################
corrConnected = np.sort(corrConnected)[::-1]
distConnected = np.sort(distConnected)
abcConnected = np.sort(abcConnected)[::-1]
recallCorr = np.zeros(shape=(corrConnected.shape))
recallABC = np.zeros(shape=(abcConnected.shape))
recallDist = np.zeros(shape=(distConnected.shape))
precisionCorr = np.zeros(shape=(corrConnected.shape))
precisionABC = np.zeros(shape=(abcConnected.shape))
precisionDist = np.zeros(shape=(distConnected.shape))
for i in range(len(corrConnected)):
	corrCut = corrConnected[i]
	distCut = distConnected[i]
	recallCorr[i] = np.sum(corrConnected>=corrCut)/float(len(corrConnected))
	recallDist[i] = np.sum(distConnected<=distCut)/float(len(distConnected))
	precisionCorr[i] = np.sum(corrConnected>=corrCut)/float(np.sum(corrNotConnected>=corrCut)+np.sum(corrConnected>=corrCut))
	precisionDist[i] = np.sum(distConnected<=distCut)/float(np.sum(distNotConnected<=distCut)+np.sum(distConnected<=distCut))
	abcCut = abcConnected[i]
	recallABC[i] = np.sum(abcConnected>=abcCut)/float(len(abcConnected))
	precisionABC[i] = np.sum(abcConnected>=abcCut)/float(np.sum(abcNotConnected>=abcCut)+np.sum(abcConnected>=abcCut))


plt.clf()
plt.plot(100*recallCorr,100*precisionCorr,'g-',linewidth=3,label='Correlation')
plt.plot(100*recallDist,100*precisionDist,'b-',linewidth=3,label='Distance')
plt.plot(100*recallABC,100*precisionABC,'r-',linewidth=3,label='ABC')
plt.plot(100*recallABC[5],100*precisionABC[5],'ko',markersize=6)
plt.plot(100*recallABC[10],100*precisionABC[10],'ko',markersize=6)
plt.plot(100*recallABC[15],100*precisionABC[15],'ko',markersize=6)
plt.plot(100*recallABC[23],100*precisionABC[23],'ko',markersize=6)
plt.plot(100*recallABC[44],100*precisionABC[44],'ko',markersize=6)
plt.plot(100*recallABC[66],100*precisionABC[66],'ko',markersize=6)
#plt.plot([70,70],[0,100],'k',linewidth=3)
#plt.plot(100*recallABC,100*(precisionABC),'r-',linewidth=3,label='ABC')
plt.xlabel('Recall, %')
plt.ylabel('Precision, %')
#plt.xlim([0,100])
#plt.ylim([0.017,10.000001])
plt.ylim([np.amin(precisionDist)*20,100.00000001])
#plt.yscale('log')
plt.grid(True)
plt.legend(fontsize=12)
plt.title('Precision Recall Curve',fontsize=18)
plt.savefig(wdfigs+'precision_recall_curve.pdf')
plt.show()

np.save(wdvars+'validation_K562/PrecisionRecall/distConnected.npy',distConnected)
np.save(wdvars+'validation_K562/PrecisionRecall/corrConnected.npy',corrConnected)
np.save(wdvars+'validation_K562/PrecisionRecall/precisionDist.npy',precisionDist)
np.save(wdvars+'validation_K562/PrecisionRecall/precisionCorr.npy',precisionCorr)
np.save(wdvars+'validation_K562/PrecisionRecall/recallDist.npy',recallDist)
np.save(wdvars+'validation_K562/PrecisionRecall/recallCorr.npy',recallCorr)
exit()

x = np.zeros(shape=len(abcUnknown))
x[:] = 0.5
plt.clf()
plt.plot(abcConnected,betaGconnect,'bo')
plt.plot(abcConnected,betaPconnect,'ro')
plt.grid(True)
plt.title('ABC vs Beta')
plt.xscale('log')
plt.xlabel('ABC (log scale)')
plt.ylabel('beta')
plt.savefig(wdfigs+'abc_vs_beta.pdf')
plt.show()

###################################################################
# ROC Curve
###################################################################

y_true = np.zeros(shape=(len(abcConnected)+len(abcUnknown)))
y_true[len(abcUnknown):] = 1

abcAll = np.concatenate([abcUnknown,abcConnected])
distAll = np.concatenate([distUnknown,distConnected])
corrAll = np.concatenate([corrUnknown,corrConnected])

fprABC,tprABC,thresholds = roc_curve(y_true, abcAll)
fprDist,tprDist,thresholds = roc_curve(y_true, distAll)
fprCorr,tprCorr,thresholds = roc_curve(y_true, corrAll)

plt.clf()
plt.plot(fprABC, tprABC, color='r', lw=3, label='ABC')
plt.plot(1-fprDist, 1-tprDist, color='b', lw=3, label='Distance')
plt.plot(fprCorr, tprCorr, color='g', lw=3, label='Corr')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve',fontsize=18)
plt.grid(True)
plt.legend(loc="lower right")
plt.savefig(wdfigs+'roc_curve.pdf')
plt.show()
exit()

###################################################################
# Logistic Regression
###################################################################

targets = np.zeros(shape=(len(abcConnected)+len(abcUnknown)))
targets[len(abcUnknown):] = 1

abcAll = np.concatenate([abcUnknown,abcConnected])
distAll = np.concatenate([distUnknown,distConnected])
corrAll = np.concatenate([corrUnknown,corrConnected])

############ Just ABC ############
features = abcAll.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.25, random_state=0)

logisticRegr = LogisticRegression(class_weight={0: 0.001, 1: 5})
logisticRegr.fit(x_train, y_train)

prob = logisticRegr.predict_proba(x_test)
trueABC = y_test
probABC = prob[:,1]
fprABC,tprABC,thresholds = roc_curve(y_test, probABC)

## Confusion Matrix ##
predictions = np.array(prob[:,1]>0.5,dtype=int)
cm = np.array(metrics.confusion_matrix(y_test, predictions),dtype=float)
cm[0] = 100*cm[0]/float(np.sum(cm[0]))
cm[1] = 100*cm[1]/float(np.sum(cm[1]))
scoreABC = np.mean([cm[0,0],cm[1,1]])
print '\nABC only:',scoreABC
confusionMatrix = pd.DataFrame(cm, columns=['Not Connected','Connected'])

plt.clf()
plt.figure(figsize=(9,9))
sns.heatmap(confusionMatrix, vmin=0, vmax=100, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
plt.yticks([0.5,1.5],['Not Connected','Connected'])
all_sample_title = 'ABC Only: Accuracy Score: {0}'.format(np.round(scoreABC,3))
plt.title(all_sample_title, size = 18);
plt.savefig(wdfigs+'confusion_matrix_abc_only.pdf')
plt.show()

############ Just Dist ############
features = distAll.reshape(-1, 1)
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.25, random_state=0)

logisticRegr = LogisticRegression(class_weight={0: 0.005, 1: 5})
logisticRegr.fit(x_train, y_train)

prob = logisticRegr.predict_proba(x_test)
trueDist = y_test
probDist = prob[:,1]
fprDist,tprDist,thresholds = roc_curve(y_test, probDist)

## Confusion Matrix ##
predictions = np.array(prob[:,1]>0.5,dtype=int)
cm = np.array(metrics.confusion_matrix(y_test, predictions),dtype=float)
cm[0] = 100*cm[0]/float(np.sum(cm[0]))
cm[1] = 100*cm[1]/float(np.sum(cm[1]))
scoreDist = np.mean([cm[0,0],cm[1,1]])
print '\nDist only:',scoreDist
confusionMatrix = pd.DataFrame(cm, columns=['Not Connected','Connected'])

plt.clf()
plt.figure(figsize=(9,9))
sns.heatmap(confusionMatrix, vmin=0, vmax=100, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
plt.yticks([0.5,1.5],['Not Connected','Connected'])
all_sample_title = 'Dist Only: Accuracy Score: {0}'.format(np.round(scoreDist,3))
plt.title(all_sample_title, size = 18);
plt.savefig(wdfigs+'confusion_matrix_dist_only.pdf')
plt.show()

############ ABC + Dist ############
features = np.zeros(shape=(len(abcAll),2))
features[:,0] = abcAll
features[:,1] = distAll
#features[:,2] = corrAll
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.25, random_state=0)

logisticRegr = LogisticRegression(class_weight={0: 0.005, 1: 5})
logisticRegr.fit(x_train, y_train)

prob = logisticRegr.predict_proba(x_test)
trueBoth = y_test
probBoth = prob[:,1]
fprBoth,tprBoth,thresholds = roc_curve(y_test, probBoth)

## Concusion Marix ##
predictions = np.array(prob[:,1]>0.5,dtype=int)
cm = np.array(metrics.confusion_matrix(y_test, predictions),dtype=float)
cm[0] = 100*cm[0]/float(np.sum(cm[0]))
cm[1] = 100*cm[1]/float(np.sum(cm[1]))
scoreBoth = np.mean([cm[0,0],cm[1,1]])
print '\nABC + Dist:', scoreBoth
confusionMatrix = pd.DataFrame(cm, columns=['Not Connected','Connected'])

plt.clf()
plt.figure(figsize=(9,9))
sns.heatmap(confusionMatrix, vmin=0, vmax=100, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues');
plt.ylabel('Actual label');
plt.xlabel('Predicted label');
plt.yticks([0.5,1.5],['Not Connected','Connected'])
all_sample_title = 'ABC and Dist: Accuracy Score: {0}'.format(np.round(scoreBoth,3))
plt.title(all_sample_title, size = 18);
plt.savefig(wdfigs+'confusion_matrix_abc_and_dist.pdf')
plt.show()

############ ABC * Dist ############
features = distAll.reshape(-1, 1) * abcAll.reshape(-1,1)
x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.25, random_state=0)

logisticRegr = LogisticRegression(class_weight={0: 0.007, 1: 5})
logisticRegr.fit(x_train, y_train)

prob = logisticRegr.predict_proba(x_test)
trueMult = y_test
probMult = prob[:,1]
fprMult,tprMult,thresholds = roc_curve(y_test, probMult)

############ ABC + Dist ############
features = np.zeros(shape=(len(abcAll),2))
features[:,0] = abcAll
features[:,1] = distAll

logisticRegr = LogisticRegression(class_weight={0: 0.005, 1: 5})
logisticRegr.fit(features, targets)
pickle.dump(logisticRegr, open(wdvars+'validation_K562/results/logisticRegression.p','w'))


############ ROC Curve ############
plt.clf()
plt.plot(fprABC, tprABC, color='r', lw=3, label='ABC')
plt.plot(fprDist, tprDist, color='b', lw=3, label='Distance')
plt.plot(fprBoth, tprBoth, color='orange', lw=3, label='ABC + Dist + Corr')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve',fontsize=18)
plt.grid(True)
plt.legend(loc="lower right")
plt.savefig(wdfigs+'roc_curve.pdf')
plt.show()

############################################################
# Precision Recall with Prob
############################################################

bothConnected = probBoth[trueBoth==1]
bothNotConnected = probBoth[trueBoth==0]
distConnected = probDist[trueDist==1]
distNotConnected = probDist[trueDist==0]
multConnected = probMult[trueMult==1]
multNotConnected = probMult[trueMult==0]
abcConnected = probABC[trueABC==1]
abcNotConnected = probABC[trueABC==0]

bothConnected = np.sort(bothConnected)[::-1]
distConnected = np.sort(distConnected)[::-1]
multConnected = np.sort(multConnected)[::-1]
abcConnected = np.sort(abcConnected)[::-1]

bothCut = np.append(bothConnected,0)
distCut = np.append(distConnected,0)
multCut = np.append(multConnected,0)
abcCut = np.append(abcConnected,0)

recallBoth = np.zeros(shape=(bothCut.shape))
recallABC = np.zeros(shape=(abcCut.shape))
recallDist = np.zeros(shape=(distCut.shape))
recallMult = np.zeros(shape=(distCut.shape))
precisionBoth = np.zeros(shape=(bothCut.shape))
precisionABC = np.zeros(shape=(abcCut.shape))
precisionDist = np.zeros(shape=(distCut.shape))
precisionMult = np.zeros(shape=(distCut.shape))
for i in range(len(bothCut)):
	recallBoth[i] = np.sum(bothConnected>=bothCut[i])/float(len(bothConnected))
	recallDist[i] = np.sum(distConnected>=distCut[i])/float(len(distConnected))
	recallMult[i] = np.sum(multConnected>=multCut[i])/float(len(multConnected))
	recallABC[i] = np.sum(abcConnected>=abcCut[i])/float(len(abcConnected))
	precisionBoth[i] = np.sum(bothConnected>=bothCut[i])/float(np.sum(bothNotConnected>=bothCut[i])+np.sum(bothConnected>=bothCut[i]))
	precisionDist[i] = np.sum(distConnected>=distCut[i])/float(np.sum(distNotConnected>=distCut[i])+np.sum(distConnected>=distCut[i]))
	precisionMult[i] = np.sum(multConnected>=multCut[i])/float(np.sum(multNotConnected>=multCut[i])+np.sum(multConnected>=multCut[i]))
	precisionABC[i] = np.sum(abcConnected>=abcCut[i])/float(np.sum(abcNotConnected>=abcCut[i])+np.sum(abcConnected>=abcCut[i]))


plt.clf()
plt.plot(100*recallBoth,100*precisionBoth,'-',color='lime',linewidth=3,label='ABC + Dist')
plt.plot(100*recallDist,100*precisionDist,'b-',linewidth=3,label='Distance')
plt.plot(100*recallMult,100*precisionMult,'-',color='cyan',linewidth=3,label='ABC * Dist')
plt.plot(100*recallABC,100*precisionABC,'r-',linewidth=3,label='ABC')

plt.plot([100*recallABC[5],100*recallABC[5]],[0,100],'r',linewidth=2,linestyle='--',label='ABC: Prob > 50%')
#plt.plot([0,100],[100*precisionABC[4],100*precisionABC[4]],'r',linewidth=2,linestyle='--')
plt.plot([100*recallBoth[20]-1,100*recallBoth[20]-1],[0,100],color='lime',linewidth=2,linestyle='--',label='Both: Prob > 50%')
#plt.plot([0,100],[100*precisionBoth[20],100*precisionBoth[20]],color='lime',linewidth=2,linestyle='--')
plt.plot([100*recallDist[20],100*recallDist[20]],[0,100],color='b',linewidth=2,linestyle='--',label='Distance: Prob > 50%')
#plt.plot([0,100],[100*precisionDist[20],100*precisionDist[20]],color='b',linewidth=2,linestyle='--')

plt.xlabel('Recall, %')
plt.ylabel('Precision, %')
#plt.xlim([0,100])
#plt.ylim([0.017,10.000001])
plt.ylim([np.amin(precisionDist)*20,100.00000001])
#plt.yscale('log')
plt.grid(True)
plt.legend(fontsize=12)
plt.title('Precision Recall Curve',fontsize=18)
plt.savefig(wdfigs+'precision_recall_curve_logistical.pdf')
plt.show()

