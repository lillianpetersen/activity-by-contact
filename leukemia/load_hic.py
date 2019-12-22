import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sys import exit
import os

wd = '/pbld/mcg/lillianpetersen/ABC/'
wdhic = '/pbld/mcg/HiC/ABC_contact_vector/'
wdvars = '/pbld/mcg/lillianpetersen/ABC/saved_variables/'
wdfigs = '/pbld/mcg/lillianpetersen/ABC/figures/'
MakePlots = False

# MCG B ALL Samples
MCGs = ['MCG001', 'MCG002', 'MCG003', 'MCG005', 'MCG006', 'MCG009', 'MCG010', 'MCG011', 'MCG012', 'MCG013', 'MCG016', 'MCG017', 'MCG019', 'MCG020', 'MCG023', 'MCG024', 'MCG027', 'MCG028', 'MCG034', 'MCG035', 'MCG036', 'MCG037', 'MCG038', 'MCG039']
nSamples = len(MCGs)

fileNames = np.array(['DUX4_MCG005_MCG011_MCG023_MCG036', 'ETV6_RUNX1_MCG002_MCG017_MCG024_MCG031_MCG035', 'Hyperdiploid_MCG010_MCG012_MCG014_MCG019_MCG020_MCG027', 'PAX5_MCG001_MCG003_MCG009', 'Ph_like_MCG013_MCG016_MCG038_MCG039'])
subtypes = np.array(['DUX4', 'ETV6-RUNX1', 'Hyperdiploid', 'PAX5', 'Ph-like'])

typesName = np.array(['PAX5', 'ETV6-RUNX1', 'PAX5alt', 'DUX4', 'ZNF384', 'PAX5alt', 'Hyperdiploid', 'DUX4', 'Hyperdiploid', 'Ph-like', 'Ph-like', 'ETV6-RUNX1', 'Hyperdiploid', 'Hyperdiploid', 'DUX4', 'ETV6-RUNX1', 'Hyperdiploid', 'Hyperdiploid', 'Hyperdiploid', 'ETV6-RUNX1', 'DUX4', 'Other', 'Ph-like', 'Ph-like'])

subTypeDict = {
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

###################################################################
# Load HiC
###################################################################
print('Load HiC')
for itype in range(1,len(fileNames)):
	subtype = subtypes[itype]
	print '\n\n\n',subtype
	try:
		# Load saved variables to save time
		hic = np.load(wdvars+'HiC/'+subtype+'/hic.npy')
		hicGenes = np.load(wdvars+'HiC/'+subtype+'/hicGenes.npy')
		hicPeaks = np.load(wdvars+'HiC/'+subtype+'/hicPeaks.npy')
	except:
		# Code to calculate saved variables, if they cannot load
		print(subtype+': Try command failed: loading and masking HiC Data')
		
		hicFile = pd.read_csv(wdhic+'ABC_Cvector_'+fileNames[itype]+'_10kb.txt', sep = '\t', header = None, names=['gene','genePos','peak','merge','contact'])
		
		#hicFile[['gene','peak']] = hicFile[0].str.split(':',expand=True)
		
		#################### Remove Bad Genes ####################
		##### Reduce Genes to only the genes used in RNA (~14000) #####
		geneName = np.load(wdvars+'RNA/geneName.npy')
		hicGenes = np.array(hicFile['gene'])
		uniqueGenes = np.unique(hicGenes)
		goodGenes = np.isin(uniqueGenes,geneName,invert=True) # Array of whether to keep genes, shape of uniqueGenes
		
		geneDict = {}
		for i in range(len(uniqueGenes)):
			geneDict[uniqueGenes[i]] = goodGenes[i] # Dictionary of gene name to whether to keep it
		
		# calculate mask for each line in HiC (whether each gene is good)
		goodLine = np.ones(shape = hicFile.shape[0], dtype = bool)
		for line in range(hicFile.shape[0]):
			gene = hicGenes[line]
			goodLine[line] = geneDict[gene] #1=bad, 0=good
		
		hicArray = np.swapaxes(np.array(hicFile),0,1)
		hicPeaks = np.array(hicFile['peak'])
		hicAll = hicArray[4]
		
		# mask and compress all arrays to get rid of bad genes
		hicAll = np.ma.masked_array(hicAll,goodLine)
		hic = np.ma.compressed(hicAll)
		hicGenes = np.ma.compressed(np.ma.masked_array(hicGenes,goodLine))
		uniqueGenes = np.unique(hicGenes)
		hicPeaks = np.ma.compressed(np.ma.masked_array(hicPeaks,goodLine))
		
		# save variables
		if not os.path.exists(wdvars+'HiC/'+subtype):
			os.makedirs(wdvars+'HiC/'+subtype)
		np.save(wdvars+'HiC/'+subtype+'/hic.npy',hic)
		np.save(wdvars+'HiC/'+subtype+'/hicGenes.npy',hicGenes)
		np.save(wdvars+'HiC/'+subtype+'/hicUniqueGenes.npy',uniqueGenes)
		np.save(wdvars+'HiC/'+subtype+'/hicPeaks.npy',hicPeaks)
		np.save(wdvars+'HiC/'+subtype+'/hicLineMask.npy',goodLine)
	
	geneChrDict = pickle.load(open(wdvars+'geneChrDict.p','rb'))
	genePosDict = pickle.load(open(wdvars+'genePosDict.p','rb'))
	############ Sort HiC Lines by Gene Chromosome ###########
	hicChr = []
	hicPos = np.zeros(shape = (2,len(hicGenes)) )
	for i in range(len(hicGenes)):
		hicChr.append( geneChrDict[hicGenes[i]] )
		hicPos[:,i] = genePosDict[hicGenes[i]]
	hicStart = hicPos[0]
	hicEnd = hicPos[1]
	hicChr = np.array(hicChr)
	sortChr = np.argsort(hicChr)
	hChrSort = hicChr[sortChr]
	hGeneSort = hicGenes[sortChr]
	hPeakSort = hicPeaks[sortChr]
	hStartSort = hicStart[sortChr]
	hicSort = hic[sortChr]
	
	############ Sort HiC Lines by Gene Start Within each Chromosome ############
	for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
		print '\nSort HiC by Gene: Chromosome '+ichr
		vars()['hGeneStart'+ichr] = hStartSort[hChrSort==('chr'+ichr)]
		vars()['hGene'+ichr] = hGeneSort[hChrSort==('chr'+ichr)]
		vars()['hPeak'+ichr] = hPeakSort[hChrSort==('chr'+ichr)]
		vars()['hicFlat'+ichr] = hicSort[hChrSort==('chr'+ichr)]
		sortPos = np.argsort( vars()['hGeneStart'+ichr] )
		vars()['hGeneStart'+ichr] = vars()['hGeneStart'+ichr][sortPos]
		vars()['hGene'+ichr] = vars()['hGene'+ichr][sortPos]
		vars()['hPeak'+ichr] = vars()['hPeak'+ichr][sortPos]
		vars()['hicFlat'+ichr] = vars()['hicFlat'+ichr][sortPos]
	
		# Mask lines with peaks on wrong chromosomes
		peakMask = np.ones(shape=( len(vars()['hPeak'+ichr])),dtype=bool)
		for line in range(len( vars()['hPeak'+ichr] )):
			tmp = vars()['hPeak'+ichr][line].split('_')
			peakChr = tmp[0]
			if peakChr==('chr'+ichr):
				peakMask[line] = 0
		
		# Remove lines with peaks on wrong chromosomes
		vars()['hPeak'+ichr] = np.ma.compressed(np.ma.masked_array( vars()['hPeak'+ichr], peakMask ))
		vars()['hGene'+ichr] = np.ma.compressed(np.ma.masked_array( vars()['hGene'+ichr], peakMask ))
		vars()['hicFlat'+ichr] = np.ma.compressed(np.ma.masked_array( vars()['hicFlat'+ichr], peakMask ))
		vars()['hGeneStart'+ichr] = np.ma.compressed(np.ma.masked_array( vars()['hGeneStart'+ichr], peakMask ))
	
		# Dictionary geneName --> geneStart
		vars()['geneStartDict'+ichr] = {}
		for i in range(len(vars()['hGene'+ichr])):
			vars()['geneStartDict'+ichr][vars()['hGene'+ichr][i]] = vars()['hGeneStart'+ichr][i]
	
	############ Create 3D HiC Matrices for Each Chromosome ############
	for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
		print '\nCreate 3D HiC Matrices: Chromosome '+ichr 
		## Number Genes and Peaks
		print('Number Genes and Peaks')
		vars()['uGene'+ichr],gIndex = np.unique(vars()['hGene'+ichr], return_index=True)
		vars()['uPeak'+ichr],pIndex = np.unique(vars()['hPeak'+ichr], return_index=True)
		gIndex = np.argsort(gIndex)
		pIndex = np.argsort(pIndex)
		uGeneDict = {}
		for igene in range(len( vars()['uGene'+ichr] )):
			uGeneDict[ vars()['uGene'+ichr][igene] ] = gIndex[igene] # Dict of gene name to gene index
		uPeakDict = {}
		for ipeak in range(len( vars()['uPeak'+ichr] )):
			uPeakDict[ vars()['uPeak'+ichr][ipeak] ] = pIndex[ipeak] # Dict of peak name to peak index
		
		# Create 3D Matrices
		print('Create 3D Matrices')
		# HiC Matrix
		vars()['hic'+ichr] = np.zeros(shape = (len(uGeneDict), len(uPeakDict)) )
		for line in range(len( vars()['hicFlat'+ichr] )):
			igene = uGeneDict[ vars()['hGene'+ichr][line] ]
			ipeak = uPeakDict[ vars()['hPeak'+ichr][line] ]
			vars()['hic'+ichr][igene,ipeak] = vars()['hicFlat'+ichr][line]
		# HiC Genes
		vars()['geneMatrix'+ichr] = []
		geneDictReversed = dict(map(reversed, uGeneDict.items()))
		for igene in range(len( uGeneDict )):
			vars()['geneMatrix'+ichr].append( geneDictReversed[igene] )
		vars()['geneMatrix'+ichr] = np.array(vars()['geneMatrix'+ichr])
		# HiC Peaks
		vars()['peakMatrix'+ichr] = []
		peakDictReversed = dict(map(reversed, uPeakDict.items()))
		for ipeak in range(len( uPeakDict )):
			vars()['peakMatrix'+ichr].append( peakDictReversed[ipeak] )
		vars()['peakMatrix'+ichr] = np.array(vars()['peakMatrix'+ichr])
	
		### Sort HiC Matrix by peak position ###
		vars()['peakStart'+ichr] = np.zeros(shape = (len(vars()['peakMatrix'+ichr])), dtype=int)
		vars()['peakPos'+ichr] = np.zeros(shape = (2,len(vars()['peakMatrix'+ichr])), dtype=int)
		for i in range(len( vars()['peakMatrix'+ichr] )):
			tmp = vars()['peakMatrix'+ichr][i].split('_')
			vars()['peakStart'+ichr][i] = int(tmp[1])
			vars()['peakPos'+ichr][0,i] = int(tmp[1])
			vars()['peakPos'+ichr][1,i] = int(tmp[2])
	
		print('Sort Arrays According to Peak and Gene Position')
		peakSort = np.argsort(vars()['peakStart'+ichr])
		vars()['peakMatrix'+ichr] = vars()['peakMatrix'+ichr][peakSort]
		vars()['peakStart'+ichr] = vars()['peakStart'+ichr][peakSort]
		vars()['peakPos'+ichr] = vars()['peakPos'+ichr][:,peakSort]
		vars()['hic'+ichr] = vars()['hic'+ichr][:,peakSort]
	
		### Sort HiC Matrix by gene position ###
		vars()['geneStart'+ichr] = []
		for igene in range(len(vars()['geneMatrix'+ichr])):
			vars()['geneStart'+ichr].append( vars()['geneStartDict'+ichr][ vars()['geneMatrix'+ichr][igene] ] )
		vars()['geneStart'+ichr] = np.array(vars()['geneStart'+ichr])
		geneSort = np.argsort(vars()['geneStart'+ichr])
		vars()['geneMatrix'+ichr] = vars()['geneMatrix'+ichr][geneSort]
		vars()['geneStart'+ichr] = vars()['geneStart'+ichr][geneSort]
		vars()['hic'+ichr] = vars()['hic'+ichr][geneSort,:]
	
		## Plot final HiC Matrix on a chromosome
		plt.clf()
		fig = plt.figure(figsize = (8,6))
		plt.imshow(vars()['hic'+ichr], cmap = 'hot_r', aspect='auto', interpolation='none',origin='lower', vmax=30)
		plt.title(subtype+ ' HiC Matrix on Chromosome 21',fontsize=18)
		plt.xlabel('Peaks')
		plt.ylabel('Genes')
		plt.grid(True)
		plt.colorbar()
		plt.savefig(wdfigs+'HiC/HiC_'+subtype+'_Chr21.pdf')
	
		np.save( wdvars+'HiC/'+subtype+'/hic'+ichr+'.npy', vars()['hic'+ichr] )
		np.save( wdvars+'HiC/general/geneStart'+ichr+'.npy', vars()['geneStart'+ichr] )
		np.save( wdvars+'HiC/general/geneMatrix'+ichr+'.npy', vars()['geneMatrix'+ichr] )
		np.save( wdvars+'HiC/general/peakStart'+ichr+'.npy', vars()['peakStart'+ichr] )
		np.save( wdvars+'HiC/general/peakPos'+ichr+'.npy', vars()['peakPos'+ichr] )
		np.save( wdvars+'HiC/general/peakMatrix'+ichr+'.npy', vars()['peakMatrix'+ichr] )
		np.save( wdvars+'HiC/general/geneStartDict'+ichr+'.npy', vars()['peakMatrix'+ichr] )
		
