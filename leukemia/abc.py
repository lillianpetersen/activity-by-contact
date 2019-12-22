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

subtypes = np.array(['DUX4', 'ETV6-RUNX1', 'Hyperdiploid', 'PAX5', 'Ph-like'])

typesName = np.array(['PAX5', 'ETV6-RUNX1', 'PAX5alt', 'DUX4', 'ZNF384', 'PAX5alt', 'Hyperdiploid', 'DUX4', 'Hyperdiploid', 'Ph-like', 'Ph-like', 'ETV6-RUNX1', 'Hyperdiploid', 'Hyperdiploid', 'DUX4', 'ETV6-RUNX1', 'Hyperdiploid', 'Hyperdiploid', 'Hyperdiploid', 'ETV6-RUNX1', 'DUX4', 'Other', 'Ph-like', 'Ph-like'])

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

for itype in range(len(subtypes)):
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
				plt.imshow(vars()['abc'+ichr], cmap = 'hot_r', aspect='auto', interpolation='none',origin='lower', vmax=30)
				plt.title('ABC Matrix: Chromosome '+ichr,fontsize=18)
				plt.xlabel('Peaks')
				plt.ylabel('Genes')
				plt.grid(True)
				plt.colorbar()
				plt.savefig(wdfigs+'abc_Chr'+ichr+'.pdf')
	
	###################################################################
	# Create Correlaiton Matrix
	###################################################################
	print 'Compute Correlations'
	#for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	##for ichr in ['21']:
	#	if not 'corr'+ichr in globals():
	#		try:
	#			vars()['corr'+ichr] = np.load(wdvars+'ABC_stats/corr'+ichr+'.npy')
	#			vars()['corr'+ichr] = np.ma.masked_array(vars()['corr'+ichr],vars()['corr'+ichr]==0)
	#		except:
	#			print ichr,
	#
	#			nGenes = vars()['expression'+ichr].shape[1]
	#			nPeaks = vars()['atac'+ichr].shape[1]
	#			vars()['corr'+ichr] = np.zeros(shape = (nGenes, nPeaks))
	#			peakPos = np.mean( vars()['positionATAC'+ichr][:,:],axis=0)
	#			for igene in np.arange(nGenes):
	#				genePos = vars()['tss'+ichr][igene]
	#				usePeak1 = np.abs(peakPos-genePos)<1500000 # True = good = within 1.5Mb
	#				usePeak2 = np.abs(peakPos-genePos)>5000 # True = good = outside 5kb
	#				usePeak = usePeak1==usePeak2
	#				for ipeak in np.arange(nPeaks)[usePeak]:
	#					vars()['corr'+ichr][igene,ipeak] = stats.spearmanr(vars()['expression'+ichr][:,igene],vars()['atac'+ichr][:,ipeak])[0]
	#			np.save(wdvars+'ABC_stats/corr'+ichr+'.npy',vars()['corr'+ichr])
	#			vars()['corr'+ichr] = np.ma.masked_array(vars()['corr'+ichr],vars()['corr'+ichr]==0)
		
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
	
	################################
	# Dist
	################################
	if MakePlots:
		print 'Make Plots'
		connectDist = np.ma.masked_array(connectDist,connectDist==0)
		bins = np.logspace(np.log10(5000),np.log10(np.amax(connectDist)),100)
		
		plt.clf()
		fig,axs= plt.subplots(6, 1, figsize=(9,11),sharex=True, sharey=False)
		labels = ['ABC>0.45','ABC>0.25','ABC>0.20','ABC>0.14','ABC>0.06','ABC>0.03']
		alpha = [1,0.8,0.6,0.5,0.3,0.2]
		j = 6
		for i in range(6):
			j-=1
			n,bins,patches = axs[i].hist(np.ma.compressed(connectDist[j]), bins=bins, color='b', alpha=alpha[i], label=labels[i]) 
			axs[i].plot([np.median(np.ma.compressed(connectDist[j])),np.median(np.ma.compressed(connectDist[j]))],[0,10000],'k',linewidth=3)
			axs[i].set_ylim([0,np.amax(n)*1.03])
			axs[i].legend()
			axs[i].grid(True)
		plt.xlim([5000,np.amax(connectDist)])
		axs[0].set_title('Distance between Peak-Gene Connections by ABC Score',fontsize=17)
		plt.ylabel('Number')
		plt.xlabel('Distance (log scale)')
		plt.xscale('log')
		plt.savefig(wdfigs+'final_statistics/distance.pdf')
		plt.show()
	
	################################
	# Corr
	################################
	if MakePlots:
		connectCorr = np.ma.masked_array(connectCorr,connectCorr==0)
		bins = np.arange(-1,1,2/100.)
		
		plt.clf()
		fig,axs= plt.subplots(6, 1, figsize=(9,11),sharex=True, sharey=False)
		labels = ['ABC>0.45','ABC>0.25','ABC>0.20','ABC>0.14','ABC>0.06','ABC>0.03']
		alpha = [1,0.8,0.6,0.5,0.3,0.2]
		j = 6
		for i in range(6):
			j-=1
			n,bins,patches = axs[i].hist(np.ma.compressed(connectCorr[j]), bins=bins, color='b', alpha=alpha[i], label=labels[i]) 
			axs[i].plot([0,0],[0,10000],'k',linewidth=1)
			axs[i].plot([np.mean(np.ma.compressed(connectCorr[j])),np.mean(np.ma.compressed(connectCorr[j]))],[0,10000],'k',linewidth=3)
			axs[i].set_ylim([0,np.amax(n)*1.03])
			axs[i].legend()
			axs[i].grid(True)
		#plt.xlim([5000,np.amax(connectCorr)])
		axs[0].set_title('Correlations between Peak-Gene Connections by ABC Score',fontsize=17)
		plt.ylabel('Number')
		plt.xlabel('Correlation')
		#plt.xscale('log')
		plt.savefig(wdfigs+'final_statistics/correlation.pdf')
		#plt.show()
	
	################################
	# Slope
	################################
	if MakePlots:
		connectSlope = np.ma.masked_array(connectSlope,connectSlope==0)
		bins = np.arange(-1,1,2/100.)
		
		plt.clf()
		fig,axs= plt.subplots(6, 1, figsize=(9,11),sharex=True, sharey=False)
		labels = ['ABC>0.45','ABC>0.25','ABC>0.20','ABC>0.14','ABC>0.06','ABC>0.03']
		alpha = [1,0.8,0.6,0.5,0.3,0.2]
		j = 6
		for i in range(6):
			j-=1
			n,bins,patches = axs[i].hist(np.ma.compressed(connectSlope[j]), bins=bins, color='b', alpha=alpha[i], label=labels[i]) 
			axs[i].plot([0,0],[0,10000],'k',linewidth=1)
			axs[i].plot([np.mean(np.ma.compressed(connectSlope[j])),np.mean(np.ma.compressed(connectSlope[j]))],[0,10000],'k',linewidth=3)
			axs[i].set_ylim([0,np.amax(n)*1.03])
			axs[i].legend(loc='upper left')
			axs[i].grid(True)
		#plt.xlim([5000,np.amax(connectSlope)])
		axs[0].set_title('Slope between Peak-Gene Connections by ABC Score',fontsize=17)
		plt.ylabel('Number')
		plt.xlabel('Slope')
		#plt.xscale('log')
		plt.savefig(wdfigs+'final_statistics/slope.pdf')
		#plt.show()
	
	################################
	# Peaks per Gene
	################################
	# Number of Peaks controlling each gene
	if MakePlots:
		nGenesFull = 10406
		
		plt.clf()
		fig,axs= plt.subplots(6, 1, figsize=(9,11),sharex=True, sharey=False)
		labels = ['ABC>0.45','ABC>0.25','ABC>0.20','ABC>0.14','ABC>0.06','ABC>0.03']
		alpha = [1,0.8,0.6,0.5,0.3,0.2]
		j = 6
		for i in range(6):
			j-=1
			gene,count = np.unique(connectGenes[j:][connectGenes[j:]!=0], return_counts=True)
			bins = np.arange(0,13,1)
			#for k in range(nGenesFull - len(count)):
			#	count = np.append(count,0)
			pctGenes = 100*np.round((nGenesFull-len(count))/float(nGenesFull),2)
			
			n,bins,patches = axs[i].hist( count, bins=bins, align='left', normed=True, alpha=alpha[i], color='b', label=labels[i])
			axs[i].text(0,np.amax(n)/1.2,str(int(pctGenes))+'% Genes', rotation=90, weight='bold')
			axs[i].grid(True)
			axs[i].plot([np.mean(count),np.mean(count)],[0,10000],'k',linewidth=3)
			axs[i].set_ylim([0,np.amax(n)*1.03])
			axs[i].legend(loc='upper right')
			
		axs[0].set_title('Number of Peaks per Gene',fontsize=17)
		plt.xlim([-0.5,12.5])
		plt.xlabel('Number of Peaks per Gene')
		plt.ylabel('Genes')
		plt.savefig(wdfigs+'final_statistics/peaks_per_gene_distribution.pdf')
		#plt.show()
	
	################################
	# Genes per Peak
	################################
	if MakePlots:
		nPeaksFull = 188044
		plt.clf()
		fig,axs= plt.subplots(6, 1, figsize=(9,11),sharex=True, sharey=False)
		labels = ['ABC>0.45','ABC>0.25','ABC>0.20','ABC>0.14','ABC>0.06','ABC>0.03']
		alpha = [1,0.8,0.6,0.5,0.3,0.2]
		j = 6
		for i in range(6):
			j-=1
			peak,count = np.unique(connectPeaks[j:][connectPeaks[j:]!=0], return_counts=True)
			bins = np.arange(0,13,1)
			#for k in range(nPeaksFull - len(count)):
			#	count = np.append(count,0)
			pctPeaks = 100*np.round((nPeaksFull-len(count))/float(nPeaksFull),2)
			
			n,bins,patches = axs[i].hist( count, bins=bins, align='left', normed=True, alpha=alpha[i], color='b', label=labels[i])
			axs[i].text(0,np.amax(n)/1.2,str(int(pctPeaks))+'% Peaks', rotation=90, weight='bold')
			axs[i].grid(True)
			axs[i].plot([np.mean(count),np.mean(count)],[0,10000],'k',linewidth=3)
			axs[i].set_ylim([0,np.amax(n)*1.03])
			axs[i].legend(loc='upper right')
			
		axs[0].set_title('Number of Genes per Peak',fontsize=17)
		plt.xlim([-0.5,12.5])
		plt.xlabel('Number of Genes per Peak')
		plt.ylabel('Peaks')
		plt.savefig(wdfigs+'final_statistics/genes_per_peak_distribution.pdf')
		#plt.show()
	
	###################################################################
	# Correlations: ABC and ATAC --> RNA
	###################################################################
	print 'Multivariate Regression'
	
	medianErrorAll = np.ones(shape = (nChr,2,6)) # [chromosomes], [multiV,randomforest], [cutoffs]
	medianR2All = np.ones(shape = (nChr,2,6))
	numGenes = np.zeros(shape=(nChr,2,6))
	pctGenes = np.zeros(shape=(nChr,2,6))
	jchr=-1
	for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20']:
		jchr+=1
		medianErrorAll[jchr] = np.load(wdvars+'ABC_stats/'+ichr+'/medianError'+ichr+'.npy',)
		medianR2All[jchr] = np.load(wdvars+'ABC_stats/'+ichr+'/medianR2_'+ichr+'.npy')
		numGenes[jchr] = np.load(wdvars+'ABC_stats/'+ichr+'/numGenes'+ichr+'.npy')
		pctGenes[jchr] = np.load(wdvars+'ABC_stats/'+ichr+'/pctGenes'+ichr+'.npy')
	
	jchr = -1
	for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','X']:
		print ichr
		jchr+=1
		if ichr!='21': continue
		print ichr,jchr
	
		nGenes = vars()['nGenes'+ichr] = vars()['expression'+ichr].shape[1]
		nPeaks = vars()['nPeaks'+ichr] = vars()['atac'+ichr].shape[1]
		
		###################################################################
		# Multivariate Regression and Random Forest
		###################################################################
		
		cutoff = np.array([1,2,3]) # Number of Peaks to use in regression
		
		vars()['r2MultiV_'+ichr] = np.zeros(shape = (6, 20, vars()['nGenes'+ichr]) )
		vars()['r2Forest_'+ichr] = np.zeros(shape = (6, 20, vars()['nGenes'+ichr]) )
		vars()['medianErrorMultiV_'+ichr] = np.ones(shape = (6, vars()['nGenes'+ichr]) ) # cutoff, nGenes
		vars()['medianErrorForest_'+ichr] = np.ones(shape = (6, vars()['nGenes'+ichr]) ) # cutoff, nGenes
	
		vars()['predictMultiV_'+ichr] = np.ones(shape = (4, 20, 6, vars()['nGenes'+ichr]) ) # 4 test samples, cutoff, 10 loops, nGenes
		vars()['predictForest_'+ichr] = np.ones(shape = (4, 20, 6, vars()['nGenes'+ichr]) ) # 4 test samples, cutoff, 10 loops, nGenes
		vars()['actual_'+ichr] = np.ones(shape = (4, 20, 6, vars()['nGenes'+ichr]) ) # 4 test samples, cutoff, 10 loops, nGenes
		
		for igene in range(vars()['nGenes'+ichr]):
			icut=-1
			for cut in [2,3,5,6,8,15]:
				icut+=1
				if np.sum(vars()['connections'+ichr][igene]>=cut)==0: continue
				peakIndices = np.where(vars()['connections'+ichr][igene]>=cut)[0]
				features = np.zeros(shape = (nSamples,len(peakIndices)) )
				targets = vars()['expression'+ichr][:,igene]
				for i in range(len(peakIndices)):
					features[:,i] = vars()['atac'+ichr][:,peakIndices[i]]
	
				errorMultiV = np.ones(shape = (4, 20)) # 4 test samples, 20 loops
				errorForest = np.ones(shape = (4, 20)) 
		
				for itest in range(20):
					features_train, features_test, targets_train, targets_test = train_test_split(features, targets, test_size=0.25)
					vars()['actual_'+ichr][:,itest,icut,igene] = targets_test
		
					########### Multivariate ###########
					clf=sklearn.linear_model.LinearRegression()
					clf.fit(features_train,targets_train)
					vars()['r2MultiV_'+ichr][icut,itest,igene] = clf.score(features_test,targets_test)
					predict = clf.predict(features_test)
					errorMultiV[:,itest] = np.abs(predict-targets_test)/abs(targets_test)
					vars()['predictMultiV_'+ichr][:,itest,icut,igene] = predict
		
					########### Random Forest ###########
					clf = sklearn.ensemble.RandomForestRegressor(n_estimators=30)
					clf.fit(features_train,targets_train)
					vars()['r2Forest_'+ichr][icut,itest,igene] = clf.score(features_test,targets_test)
					predict = clf.predict(features_test)
					errorForest[:,itest] = np.abs(predict-targets_test)/abs(targets_test)
					vars()['predictForest_'+ichr][:,itest,icut,igene] = predict
		
				vars()['medianErrorMultiV_'+ichr][icut,igene] = np.median(errorMultiV)
				vars()['medianErrorForest_'+ichr][icut,igene] = np.median(errorForest)
		
				if MakePlots:
					if not os.path.exists(wdfigs+'prediction_lineplots/chr'+ichr+'/'+vars()['geneName'+ichr][igene]):
						os.makedirs(wdfigs+'prediction_lineplots/chr'+ichr+'/'+vars()['geneName'+ichr][igene])
					fig = plt.figure(figsize = (10,8))
					predict = np.reshape(vars()['predictMultiV_'+ichr][:,:,icut,igene],-1)
					actual = np.reshape(vars()['actual_'+ichr][:,:,icut,igene],-1)
					plt.clf()
					plt.plot(predict,actual,'b*',markersize=15)
					m,b = np.polyfit(predict,actual,1)
					yfit = m*predict+b
					plt.plot(predict,yfit,'g-')
					plt.title('Expression Predictions for '+vars()['geneName'+ichr][igene]+': Multivariate, n='+str(cut)+'\n Median Error = '+str(round(vars()['medianErrorMultiV_'+ichr][icut,igene]*100,1)),fontsize = 18)
					plt.xlabel('Predicted Gene Expression')
					plt.ylabel('Actual Gene Expression')
					plt.grid(True)
					plt.savefig(wdfigs+'prediction_lineplots/chr'+ichr+'/'+vars()['geneName'+ichr][igene]+'/prediction_'+vars()['geneName'+ichr][igene]+'_multivariate_top'+str(cut)+'.pdf')
	
	
					predict = np.reshape(vars()['predictForest_'+ichr][:,:,icut,igene],-1)
					actual = np.reshape(vars()['actual_'+ichr][:,:,icut,igene],-1)
					plt.clf()
					plt.plot(predict,actual,'b*',markersize=15)
					m,b = np.polyfit(predict,actual,1)
					yfit = m*predict+b
					plt.plot(predict,yfit,'g-')
					plt.title('Expression Predictions for '+vars()['geneName'+ichr][igene]+': Random Forest, n='+str(cut)+'\n Median Error = '+str(round(vars()['medianErrorForest_'+ichr][icut,igene]*100,1)),fontsize = 18)
					plt.xlabel('Predicted Gene Expression')
					plt.ylabel('Actual Gene Expression')
					plt.grid(True)
					plt.savefig(wdfigs+'prediction_lineplots/chr'+ichr+'/'+vars()['geneName'+ichr][igene]+'/prediction_'+vars()['geneName'+ichr][igene]+'_randomforest_top'+str(cut)+'.pdf')
	
		###################################################################
		# Assign Summary Variables
		###################################################################
		
		for icut in range(6):
			medianErrorAll[jchr,0,icut] = np.median( vars()['medianErrorMultiV_'+ichr][icut][vars()['medianErrorMultiV_'+ichr][icut]!=1] )
			medianErrorAll[jchr,1,icut] = np.median( vars()['medianErrorForest_'+ichr][icut][vars()['medianErrorForest_'+ichr][icut]!=1] )
			medianR2All[jchr,0,icut] = np.median( np.median( vars()['r2MultiV_'+ichr],axis=1)[icut][np.median(vars()['r2MultiV_'+ichr],axis=1)[icut]!=0] )
			medianR2All[jchr,1,icut] = np.median( np.median( vars()['r2Forest_'+ichr],axis=1)[icut][np.median(vars()['r2Forest_'+ichr],axis=1)[icut]!=0] )
		
		numGenes[jchr,0] = vars()['medianErrorMultiV_'+ichr][1][vars()['medianErrorMultiV_'+ichr][1]!=1].shape[0]
		numGenes[jchr,1] = vars()['medianErrorForest_'+ichr][1][vars()['medianErrorForest_'+ichr][1]!=1].shape[0]
		pctGenes[jchr,0] = vars()['medianErrorMultiV_'+ichr][1][vars()['medianErrorMultiV_'+ichr][1]!=1].shape[0]/float(vars()['nGenes'+ichr])
		pctGenes[jchr,1] = vars()['medianErrorForest_'+ichr][1][vars()['medianErrorForest_'+ichr][1]!=1].shape[0]/float(vars()['nGenes'+ichr])
	
		if not os.path.exists(wdvars+'ABC_stats/'+ichr):
			os.makedirs(wdvars+'ABC_stats/'+ichr)
		np.save(wdvars+'ABC_stats/'+ichr+'/medianErrorMultiV_'+ichr+'.npy',vars()['medianErrorMultiV_'+ichr])
		np.save(wdvars+'ABC_stats/'+ichr+'/medianErrorForest_'+ichr+'.npy',vars()['medianErrorForest_'+ichr])
	
		np.save(wdvars+'ABC_stats/'+ichr+'/medianError'+ichr+'.npy',medianErrorAll[jchr])
		np.save(wdvars+'ABC_stats/'+ichr+'/medianR2_'+ichr+'.npy',medianR2All[jchr])
		np.save(wdvars+'ABC_stats/'+ichr+'/numGenes'+ichr+'.npy',numGenes[jchr])
		np.save(wdvars+'ABC_stats/'+ichr+'/pctGenes'+ichr+'.npy',pctGenes[jchr])
	
		vars()['predictMultiV_'+ichr] = vars()['predictMultiV_'+ichr].reshape(-1,*vars()['predictMultiV_'+ichr].shape[-2:]) # new shape: 80 samples x 4 cutoff x nGenes
		vars()['predictForest_'+ichr] = vars()['predictForest_'+ichr].reshape(-1,*vars()['predictForest_'+ichr].shape[-2:])
		vars()['actual_'+ichr] = vars()['actual_'+ichr].reshape(-1,*vars()['actual_'+ichr].shape[-2:])
		np.save(wdvars+'ABC_stats/'+ichr+'/predictMultiV_'+ichr+'.npy',vars()['predictMultiV_'+ichr])
		np.save(wdvars+'ABC_stats/'+ichr+'/predictForest_'+ichr+'.npy',vars()['predictForest_'+ichr])
		np.save(wdvars+'ABC_stats/'+ichr+'/actual_'+ichr+'.npy',vars()['actual_'+ichr])
		
	
	exit()
	###################################################################
	# Average Chromosomes 
	###################################################################
	medianError = np.zeros(shape = (nX,6))
	medianR2 = np.zeros(shape = (nX,6))
	numStrongGenes = np.zeros(shape = (nX,6))
	numGenes = np.zeros(shape = (nX,6))
	pctGenes = np.zeros(shape = (nX,6))
	
	for jchr in range(23):
		medianError = np.mean(medianErrorAll[jchr]/numGenes[jchr])
		medianR2 = np.mean( medianR2All[jchr]/numGenes[jchr] )
				
	for icut in range(6):
		plt.clf()
		data = 100*medianErrorForest_21[icut][medianErrorForest_21[icut]!=1]
		n,bins,patches = plt.hist(data, bins=80, range=[0,200])
		plt.plot([np.median(data), np.median(data)],[0,20],'g-', linewidth=4)
		plt.title('Median Error for '+varTitle+' Expression Prediction \nusing Random Forests and Top '+str(cutoff[icut])+' Peaks in Each Gene')
		plt.grid(True)
		plt.ylim([0,16])
		plt.xlim([0,150])
		plt.savefig(wdfigs+'error_top'+str(cutoff[icut])+'_randomForest_distribution_'+var+'.pdf')
	
	for icut in range(6):
		plt.clf()
		data = 100*medianErrorMultiV_21[icut][medianErrorMultiV_21[icut]!=1]
		n,bins,patches = plt.hist(data, bins=80, range=[0,200])
		plt.plot([np.median(data), np.median(data)],[0,20],'g-', linewidth=4)
		plt.title('Median Error for '+varTitle+' Expression Prediction \nusing Multivariate and Top '+str(cutoff[icut])+' Peaks in Each Gene')
		plt.grid(True)
		plt.ylim([0,16])
		plt.xlim([0,150])
		plt.savefig(wdfigs+'error_top'+str(cutoff[icut])+'_multivariate_distribution_'+var+'.pdf')
	
	for icut in range(6):
		plt.clf()
		data = np.median(r2Forest_21,axis=1)[icut][np.median(r2Forest_21,axis=1)[icut]!=0]
		n,bins,patches = plt.hist(data, bins=60, range=[0,1])
		plt.plot([np.median(data), np.median(data)],[0,20],'g-', linewidth=4)
		plt.title('R2 for '+varTitle+' Expression Prediction \nusing Random Forests and Top '+str(cutoff[icut])+' Peaks in Each Gene')
		plt.grid(True)
		plt.ylim([0,10])
		plt.xlim([0,1])
		plt.savefig(wdfigs+'r2_top'+str(cutoff[icut])+'_randomForest_distribution_'+var+'.pdf')
	
	for icut in range(6):
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









