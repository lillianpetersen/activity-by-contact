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
import matplotlib.colors as colors
#sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)

wd = '/pbld/mcg/lillianpetersen/ABC/'
wdvars = '/pbld/mcg/lillianpetersen/ABC/saved_variables/'
wddata = '/pbld/mcg/lillianpetersen/ABC/data/'
wdfigs = '/pbld/mcg/lillianpetersen/ABC/figures/'
wdfiles = '/pbld/mcg/lillianpetersen/ABC/written_files/'
MakePlots = False

# MCG B ALL Samples
MCGs = ['MCG001',         'MCG002',     'MCG003',  'MCG005','MCG006', 'MCG009',  'MCG010',       'MCG011', 'MCG012',       'MCG013',  'MCG016',  'MCG017',     'MCG019',       'MCG020',      'MCG023', 'MCG024']
nSamples = len(MCGs)
subGroups = ['PAX5_P80R', 'ETV6-RUNX1', 'PAX5alt', 'DUX4',  'ZNF384', 'PAX5alt', 'Hyperdiploid', 'DUX4',   'Hyperdiploid', 'Ph-like', 'Ph-like', 'ETV6-RUNX1', 'Hyperdiploid', 'Hyperdiploid', 'DUX4',  'ETV6-RUNX1']
subGroupDict = {
	'Hyperdiploid':0,
	'DUX4':1,
	'ETV6-RUNX1':2,
	'PAX5alt':3,
	'Ph-like':4,
	'PAX5_P80R':5,
	'ZNF384':6}
subGroupID = np.zeros(shape=nSamples,dtype=int)
for i in range(nSamples):
	subGroupID[i] = subGroupDict[subGroups[i]]
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
		vars()['direction'+ichr] = np.load(wdvars+'RNA/direction'+ichr+'.npy')

		# Limit by standard dev / mean expression
		stdMask = np.std(vars()['expression'+ichr],axis=0) / np.mean(vars()['expression'+ichr],axis=0) < 0.25 # True = bad
		maskFull = np.zeros(shape = (nSamples,len(stdMask)), dtype = bool)
		for isample in range(nSamples):
			maskFull[isample] = stdMask
		mask2 = np.zeros(shape = (2,len(stdMask)), dtype = bool)
		for isample in range(2):
			mask2[isample] = stdMask
		vars()['expression'+ichr] = np.ma.compress_cols( np.ma.masked_array(vars()['expression'+ichr], maskFull) )
		vars()['geneName'+ichr] = np.ma.compressed( np.ma.masked_array(vars()['geneName'+ichr], stdMask) )
		vars()['chrRNA'+ichr] = np.ma.compressed( np.ma.masked_array(vars()['chrRNA'+ichr], stdMask) )
		vars()['direction'+ichr] = np.ma.compressed( np.ma.masked_array(vars()['direction'+ichr], stdMask) )
		vars()['positionRNA'+ichr] = np.ma.compress_cols( np.ma.masked_array(vars()['positionRNA'+ichr], mask2) )
		directiontmp = np.array(vars()['direction'+ichr])
		directiontmp[directiontmp==1]=0
		directiontmp[directiontmp==-1]=1

		vars()['tss'+ichr] = np.zeros(shape=(vars()['chrRNA'+ichr].shape))
		for i in range(len(vars()['chrRNA'+ichr])):
			vars()['tss'+ichr][i] = vars()['positionRNA'+ichr][:,i][directiontmp[i]]

	if not 'expressionNorm'+ichr in globals():
		vars()['expressionNorm'+ichr] = np.zeros(shape = (vars()['expression'+ichr].shape))
		for igene in range(len(vars()['geneName'+ichr])):
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

		# Limit by standard dev / mean expression
		stdMask = np.std(vars()['atac'+ichr],axis=0) / np.mean(vars()['atac'+ichr],axis=0) < 0.25 # True = bad
		maskFull = np.zeros(shape = (nSamples,len(stdMask)), dtype = bool)
		for isample in range(nSamples):
			maskFull[isample] = stdMask
		mask2 = np.zeros(shape = (2,len(stdMask)), dtype = bool)
		for isample in range(2):
			mask2[isample] = stdMask
		vars()['atac'+ichr] = np.ma.compress_cols( np.ma.masked_array(vars()['atac'+ichr], maskFull) )
		vars()['peakName'+ichr] = np.ma.compressed( np.ma.masked_array(vars()['peakName'+ichr], stdMask) )
		vars()['chrATAC'+ichr] = np.ma.compressed( np.ma.masked_array(vars()['chrATAC'+ichr], stdMask) )
		vars()['positionATAC'+ichr] = np.ma.compress_cols( np.ma.masked_array(vars()['positionATAC'+ichr], mask2) )

	if not 'atacNorm'+ichr in globals():
		vars()['atacNorm'+ichr] = np.zeros(shape = (vars()['atac'+ichr].shape))
		for ipeak in range(len(vars()['peakName'+ichr])):
			vars()['atacNorm'+ichr][:,ipeak] = sklearn.preprocessing.scale(vars()['atac'+ichr][:,ipeak]) 
###################################################################
# Load HiC
###################################################################
print('load HiC')
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	if not 'hic'+ichr in globals():
		# Load arrays saved from load_hic.npy
		vars()['hic'+ichr] = np.load( wdvars+'HiC/merged/hic'+ichr+'.npy')
		vars()['geneStart'+ichr] = np.load( wdvars+'HiC/merged/geneStart'+ichr+'.npy' )
		vars()['geneMatrix'+ichr] = np.load( wdvars+'HiC/merged/geneMatrix'+ichr+'.npy' )
		vars()['peakStart'+ichr] = np.load( wdvars+'HiC/merged/peakStart'+ichr+'.npy' )
		vars()['peakMatrix'+ichr] = np.load( wdvars+'HiC/merged/peakMatrix'+ichr+'.npy' )

		if len(vars()['geneMatrix'+ichr])!=len(vars()['geneName'+ichr]):
			keepGene = np.isin(vars()['geneName'+ichr],vars()['geneMatrix'+ichr]) # True = keep
			if np.amin(keepGene)==0:
				vars()['expression'+ichr] = vars()['expression'+ichr][:,keepGene]
				vars()['expressionNorm'+ichr] = vars()['expressionNorm'+ichr][:,keepGene]
				vars()['geneName'+ichr] = vars()['geneName'+ichr][keepGene]
				vars()['chrRNA'+ichr] = vars()['chrRNA'+ichr][keepGene]
				vars()['direction'+ichr] = vars()['direction'+ichr][keepGene]
				vars()['positionRNA'+ichr] = vars()['positionRNA'+ichr][:,keepGene]
			
			keepPeak = np.isin(vars()['peakName'+ichr],vars()['peakMatrix'+ichr]) # True = keep
			if np.amin(keepPeak)==0:
				vars()['atac'+ichr] = vars()['atac'+ichr][:,keepPeak]
				vars()['chrATAC'+ichr] = vars()['chrATAC'+ichr][keepPeak]
				vars()['peakName'+ichr] = vars()['peakName'+ichr][keepPeak]
				vars()['positionATAC'+ichr] = vars()['positionATAC'+ichr][:,keepPeak]

			keepGene = np.isin(vars()['geneMatrix'+ichr],vars()['geneName'+ichr]) # True = keep
			if np.amin(keepGene)==0:
				vars()['hic'+ichr] = vars()['hic'+ichr][keepGene,:]
				vars()['geneStart'+ichr] = vars()['geneStart'+ichr][keepGene]
				vars()['geneMatrix'+ichr] = vars()['geneMatrix'+ichr][keepGene]

			keepPeak = np.isin(vars()['peakMatrix'+ichr],vars()['peakName'+ichr]) # True = keep
			if np.amin(keepPeak)==0:
				vars()['hic'+ichr] = vars()['hic'+ichr][:,keepPeak]
				vars()['peakStart'+ichr] = vars()['peakStart'+ichr][keepPeak]
				vars()['peakMatrix'+ichr] = vars()['peakMatrix'+ichr][keepPeak]
		
###################################################################
# Create ABC Matrix
###################################################################
print 'Create ABC'
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
			
			atacMean = np.mean(vars()['atac'+ichr],axis=0)
			#maskFull = np.zeros(shape = (nSamples,len(peakMask)), dtype = bool)
			#for isample in range(nSamples):
			#	maskFull[isample] = peakMask
			mask2 = np.zeros(shape = (2,len(peakMask)), dtype = bool)
			for isample in range(2):
				mask2[isample] = peakMask
			vars()['peakName'+ichr] = np.ma.compressed( np.ma.masked_array(vars()['peakName'+ichr],peakMask) )
			#vars()['atac'+ichr] = np.ma.compress_cols( np.ma.masked_array(vars()['atac'+ichr],maskFull) )
			atacMean = np.ma.compressed( np.ma.masked_array(atacMean,peakMask) )
			vars()['positionATAC'+ichr] = np.ma.compress_cols( np.ma.masked_array(vars()['positionATAC'+ichr],mask2) )
			if np.amin(vars()['peakName'+ichr]==vars()['peakMatrix'+ichr]) == 0: 
				print 'Error: Peak array sizes do not match'
				exit()
		
			nGenes = vars()['hic'+ichr].shape[0]
			nPeaks = vars()['hic'+ichr].shape[1]
			vars()['abc'+ichr] = np.zeros(shape = (nGenes, nPeaks))
			peakPos = np.mean( vars()['positionATAC'+ichr][:,:],axis=0)
			for igene in np.arange(nGenes):
				genePos = vars()['tss'+ichr][igene]
				usePeak1 = np.abs(peakPos-genePos)<1500000 # True = good = within 1.5Mb
				usePeak2 = np.abs(peakPos-genePos)>5000 # True = good = outside 5kb
				usePeak = usePeak1==usePeak2
				Sum = np.sum( atacMean[usePeak] * vars()['hic'+ichr][igene,usePeak])
				vars()['abc'+ichr][igene,usePeak] = (atacMean[usePeak] * vars()['hic'+ichr][igene,usePeak]) / Sum
			np.save(wdvars+'ABC/abc'+ichr+'.npy',vars()['abc'+ichr])
			mask = np.amax([vars()['abc'+ichr]==0, np.isnan(abcX)],axis=0)
			vars()['abc'+ichr] = np.ma.masked_array(vars()['abc'+ichr],mask)
		
	plt.clf()
	fig = plt.figure(figsize = (10,6))
	plt.imshow(vars()['abc'+ichr], cmap = 'hot_r', aspect='auto', interpolation='none',origin='lower', norm=colors.LogNorm(vmin = np.amin(vars()['abc'+ichr]),vmax=np.amax(vars()['abc'+ichr])))
	plt.title('ABC Matrix: Chromosome '+ichr,fontsize=18)
	plt.xlabel('Peaks')
	plt.ylabel('Genes')
	plt.grid(True)
	plt.colorbar()
	plt.savefig(wdfigs+'abc/abc_Chr'+ichr+'.pdf')
exit()

###################################################################
# Create Distance Matrix
###################################################################
print 'Compute Distance'
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
#for ichr in ['21']:
	if not 'dist'+ichr in globals():
		print ichr,
		#if np.amin(vars()['peakName'+ichr]==vars()['peakMatrix'+ichr]) == 0:
		#   print 'Error: Peak array sizes do not match'
		#   exit()

		nGenes = vars()['expression'+ichr].shape[1]
		nPeaks = vars()['atac'+ichr].shape[1]
		vars()['dist'+ichr] = np.zeros(shape = (nGenes, nPeaks))
		peakPos = np.mean( vars()['positionATAC'+ichr][:,:],axis=0)
		for igene in np.arange(nGenes):
			genePos = vars()['tss'+ichr][igene]
			usePeak1 = np.abs(peakPos-genePos)<1500000 # True = good = within 1.5Mb
			usePeak2 = np.abs(peakPos-genePos)>5000 # True = good = outside 5kb
			usePeak = usePeak1==usePeak2
			vars()['dist'+ichr][igene,usePeak] = np.abs(peakPos[usePeak]-genePos)
		mask = np.amax([vars()['dist'+ichr]==0,vars()['hic'+ichr]==0],axis=0)
		vars()['dist'+ichr] = np.ma.masked_array(vars()['dist'+ichr],mask)

		if MakePlots:
			plt.clf()
			fig = plt.figure(figsize = (10,6))
			plt.imshow(vars()['dist'+ichr][0], cmap = 'hot_r', aspect='auto', interpolation='none',origin='lower', vmax=1.7e6)
			plt.title('Dist Matrix: Chromosome '+ichr,fontsize=18)
			plt.xlabel('Peaks')
			plt.ylabel('Genes')
			plt.grid(True)
			plt.colorbar()
			plt.savefig(wdfigs+'dist_validation_Chr'+ichr+'.pdf')

###################################################################
# Define Peak-Gene Connections
###################################################################
print 'Define Peak-Gene Connections'
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	if not 'connections'+ichr in globals():
		nGenes = vars()['expression'+ichr].shape[1]
		nPeaks = vars()['atac'+ichr].shape[1]
		vars()['connections'+ichr] = np.zeros(shape=(nGenes,nPeaks), dtype=int)
		vars()['connections'+ichr][ vars()['abc'+ichr]>.028 ] = 2
		vars()['connections'+ichr][ vars()['abc'+ichr]>.057 ] = 3
		vars()['connections'+ichr][ vars()['abc'+ichr]>0.14 ] = 5
		vars()['connections'+ichr][ vars()['abc'+ichr]>0.20 ] = 6
		vars()['connections'+ichr][ vars()['abc'+ichr]>0.25 ] = 8
		vars()['connections'+ichr][ vars()['abc'+ichr]>0.45 ] = 15

if not 'connectDist' in globals():
	precisionToIndex = {2:0, 3:1, 5:2, 6:3, 8:4, 15:5}
	connectGenes = np.zeros(shape=(6,23,np.sum(connections1==2)),dtype=object)
	connectPeaks = np.zeros(shape=(6,23,np.sum(connections1==2)),dtype=object)
	connectDist = np.zeros(shape=(6,23,np.sum(connections1==2)))
	connectCorr = np.zeros(shape=(6,23,np.sum(connections1==2)))
	connectSlope= np.zeros(shape=(6,23,np.sum(connections1==2)))
	
	jchr = -1
	for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
		jpeak = -1*np.ones(shape=6,dtype=int)
		jchr+=1
		nGenes = vars()['expression'+ichr].shape[1]
		for igene in range(nGenes):
			peaks = np.where(vars()['connections'+ichr][igene]>0)[0]
			for ipeak in peaks:
				iconnect = precisionToIndex[vars()['connections'+ichr][igene,ipeak]]
				jpeak[iconnect]+=1
				connectGenes[iconnect,jchr,jpeak[iconnect]] = vars()['geneName'+ichr][igene]
				connectPeaks[iconnect,jchr,jpeak[iconnect]] = vars()['peakName'+ichr][ipeak]
				connectDist[iconnect,jchr,jpeak[iconnect]] = vars()['dist'+ichr][igene,ipeak]
	
				corr,p = stats.spearmanr( vars()['atac'+ichr][:,ipeak], vars()['expression'+ichr][:,igene])
				connectCorr[iconnect,jchr,jpeak[iconnect]] = corr
	
				m,b = np.polyfit(vars()['atacNorm'+ichr][:,ipeak],vars()['expressionNorm'+ichr][:,igene],1)
				connectSlope[iconnect,jchr,jpeak[iconnect]] = m
	
				if MakePlots:
					if not os.path.exists(wdfigs+'connected_peaks/'+ichr+'/'+vars()['geneName'+ichr][igene]):
						os.makedirs(wdfigs+'connected_peaks/'+ichr+'/'+vars()['geneName'+ichr][igene])
					x = vars()['atac'+ichr][:,ipeak]
					ydata = vars()['expression'+ichr][:,igene]
	
					m,b = np.polyfit(x,ydata,1)
					yfit = m*x+b
	
					abctmp = str(np.round(vars()['abc'+ichr][igene,ipeak],3))
	
					plt.clf()
					plt.plot(x, ydata, 'bo', markersize=10)
					plt.plot(x, yfit, 'g-', linewidth=2)
					plt.title( vars()['geneName'+ichr][igene]+' Expression and Connected Peak Intensity (ABC = '+abctmp+')\n Corr = '+str(round(corr,2)) )
					plt.xlabel('ATAC')
					plt.ylabel('Expression')
					plt.grid(True)
					plt.savefig(wdfigs+'connected_peaks/'+ichr+'/'+vars()['geneName'+ichr][igene]+'/'+vars()['geneName'+ichr][igene]+'_peak_'+str(abctmp)+'_corr.pdf')

###################################################################
# Find Variable Genes
###################################################################

leukGenes = pd.read_csv(wddata+'leukemia_genes.csv', header=None, sep='\t')
leukChr = np.array(leukGenes[0])
leukGeneName = np.array(leukGenes[1])
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	print ichr
	nGenes = vars()['expression'+ichr].shape[1]
	geneIndices = np.where(np.isin(vars()['geneName'+ichr],leukGeneName[leukChr==ichr]))[0]
	for igene in geneIndices:
		expressionMean = np.zeros(shape=(7))
		expressionStd = np.zeros(shape=(7))
		for igroup in range(7):
			expressionMean[igroup] = np.mean( vars()['expressionNorm'+ichr][:,igene][subGroupID==igroup], axis=0) # average of subgroups
			expressionStd[igroup] = np.std( vars()['expressionNorm'+ichr][:,igene][subGroupID==igroup], axis=0) # std dev within subgroups
	
		subGroupNames = ['Hyperdiploid (n=4)','DUX4 (n=3)','ETV6-RUNX1 (n=3)','PAX5alt (n=2)','Ph-like (n=2)','PAX5_P80R (n=1)','ZNF384 (n=1)']
		peakIndices = np.where( vars()['connections'+ichr][igene]>0 )[0]
		atacGroups = np.zeros(shape=(7,len(peakIndices)))
		for igroup in range(7):
			atacGroups[igroup,:] = np.mean( vars()['atac'+ichr][subGroupID==igroup][:,peakIndices], axis=0 )
		jpeak=-1
		for ipeak in peakIndices:
			jpeak+=1

			xdata = atacGroups[:,jpeak]
			ydata = expressionMean
			corr,p = stats.pearsonr(atacGroups[:,jpeak],expressionMean)
			m,b = np.polyfit(xdata,ydata,1)
			yfit = m*xdata+b
			if p<=0.05: 
				print vars()['geneName'+ichr][igene], vars()['abc'+ichr][igene,ipeak]

				if not os.path.exists(wdfigs+'leukemia_genes/significant'):
					os.makedirs(wdfigs+'leukemia_genes/significant')
				plt.clf()
				fig = plt.figure(figsize=(10,8))
				ax = fig.add_subplot(1,1,1)
				ax.scatter(xdata, ydata, color = 'b', s=150)
				for i,txt in enumerate(subGroupNames):
					ax.annotate(txt,(xdata[i],ydata[i]),fontsize=14)
				ax.plot(xdata,yfit,'g')
				plt.grid(True)
				ax.set_title(vars()['geneName'+ichr][igene]+' Expression Between Subgroups, by Connected Peak (ABC = '+str(np.round(vars()['abc'+ichr][igene,ipeak],2))+') \nCorr = '+str(np.round(corr,2))+', P = '+str(np.round(p,2)),fontsize=17)
				ax.set_xlabel('ATAC')
				ax.set_ylabel('Expression')
				plt.savefig(wdfigs+'leukemia_genes/significant/'+vars()['geneName'+ichr][igene]+'_peak_ABC_'+str(np.round(vars()['abc'+ichr][igene,ipeak],2))+'.pdf')

	#geneStd = np.std(expressionMean,axis=0) # std dev between subgroups
	#meanStd = np.mean(geneStd)
	#stdStd = np.std(geneStd)

	#geneIndices = np.where(geneStd > meanStd+2*stdStd )[0] # Standard deviations 2 standard deviations above the average standard deviation
	#print vars()['geneName'+ichr][geneIndices]

	#expressionGroup = expressionMean[:,geneIndices]

	#jgene=-1
	#for igene in geneIndices:
	#	jgene+=1























