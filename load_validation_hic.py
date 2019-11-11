import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sys import exit

wd = '/pbld/mcg/lillianpetersen/ABC/'
wdvars = '/pbld/mcg/lillianpetersen/ABC/saved_variables/'
wdfigs = '/pbld/mcg/lillianpetersen/ABC/figures/'
MakePlots = False

###################################################################
# Load HiC
###################################################################
print('Load HiC')
try:
	# Load saved variables to save time
	hicSamples = np.load(wdvars+'validation_K562/HiC/hicSamples.npy')
	hicGenes = np.load(wdvars+'validation_K562/HiC/hicGenes.npy')
	hicPeaks = np.load(wdvars+'validation_K562/HiC/hicPeaks.npy')
except:
	# Code to calculate saved variables, if they cannot load
	print('Try command failed: loading and masking HiC Data')
	
	#hicFile = pd.read_csv(wd+'data/validation_K562/HiC_matrix_K562_500000.txt', sep = '\t', header = None)
	
	hicFile['gene'] = hicFile[0]
	hicFile['peak'] = hicFile[2]
	
	#################### Remove Bad Genes ####################
	##### Reduce Genes to only the genes used in RNA (~14000) #####
	geneName = np.load(wdvars+'geneNameAllFull.npy')
	hicGenes = np.array(hicFile['gene'])
	uniqueGenes = np.unique(hicGenes)
	goodGenes = np.isin(uniqueGenes,geneName,invert=True) # Array of whether to keep genes, shape of uniqueGenes
	
	hicPeaks = np.array(hicFile['peak'])
	hicSamplesAll = np.array( np.array(hicFile[4]) )

	geneDict = {}
	for i in range(len(uniqueGenes)):
		geneDict[uniqueGenes[i]] = goodGenes[i] # Dictionary of gene name to whether to keep it
	
	# calculate mask for each line in HiC (whether each gene is good)
	goodLine = np.ones(shape = hicFile.shape[0], dtype = bool)
	for line in range(hicFile.shape[0]):
		gene = hicGenes[line]
		goodLine[line] = geneDict[gene] #1=bad, 0=good
	
	# mask and compress all arrays to get rid of bad genes
	hicSamplesAll = np.ma.masked_array(hicSamplesAll,goodLine)
	hicSamples = np.ma.compressed(hicSamplesAll)
	hicGenes = np.ma.compressed(np.ma.masked_array(hicGenes,goodLine))
	uniqueGenes = np.unique(hicGenes)
	hicPeaks = np.ma.compressed(np.ma.masked_array(hicPeaks,goodLine))

	# calculate mask to remove alt chromosomes
	indices = [i for i, s in enumerate(hicPeaks) if 'alt' in s]
	mask = np.zeros(shape = (hicSamples.shape), dtype = bool)
	mask[indices] = 1

	# remove alt chromosomes
	hicSamples = np.ma.masked_array(hicSamples,mask)
	hicSamples = np.ma.compressed(hicSamples)
	hicGenes = np.ma.compressed(np.ma.masked_array(hicGenes,mask))
	uniqueGenes = np.unique(mask)
	hicPeaks = np.ma.compressed(np.ma.masked_array(hicPeaks,mask))
	
	# save variables
	np.save(wdvars+'validation_K562/HiC/hicSamples.npy',hicSamples)
	np.save(wdvars+'validation_K562/HiC/hicGenes.npy',hicGenes)
	np.save(wdvars+'validation_K562/HiC/hicUniqueGenes.npy',uniqueGenes)
	np.save(wdvars+'validation_K562/HiC/hicPeaks.npy',hicPeaks)
	#np.save(wdvars+'validation_K562/HiC/hicLineMask.npy',goodLine)

geneChrDict = pickle.load(open(wdvars+'geneChrAllDict.p','rb'))
genePosDict = pickle.load(open(wdvars+'genePosAllDict.p','rb'))
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
hPosSort = hicPos[:,sortChr]
hSamplesSort = hicSamples[sortChr]

############ Sort HiC Lines by Gene Start Within each Chromosome ############
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	print '\nSort HiC by Gene: Chromosome '+ichr
	vars()['hGeneStart'+ichr] = hStartSort[hChrSort==('chr'+ichr)]
	vars()['hGenePos'+ichr] = hPosSort[:,hChrSort==('chr'+ichr)]
	vars()['hGene'+ichr] = hGeneSort[hChrSort==('chr'+ichr)]
	vars()['hPeak'+ichr] = hPeakSort[hChrSort==('chr'+ichr)]
	vars()['hSamples'+ichr] = hSamplesSort[hChrSort==('chr'+ichr)]
	sortPos = np.argsort( vars()['hGeneStart'+ichr] )
	vars()['hGeneStart'+ichr] = vars()['hGeneStart'+ichr][sortPos]
	vars()['hGenePos'+ichr] = vars()['hGenePos'+ichr][:,sortPos]
	vars()['hGene'+ichr] = vars()['hGene'+ichr][sortPos]
	vars()['hPeak'+ichr] = vars()['hPeak'+ichr][sortPos]
	vars()['hSamples'+ichr] = vars()['hSamples'+ichr][sortPos]

	# Mask lines with peaks on wrong chromosomes
	peakMask = np.ones(shape=( len(vars()['hPeak'+ichr])),dtype=bool)
	for line in range(len( vars()['hPeak'+ichr] )):
		tmp = vars()['hPeak'+ichr][line].split('_')
		peakChr = tmp[0]
		if peakChr==('chr'+ichr):
			peakMask[line] = 0
	
	# Remove lines with peaks on wrong chromosomes
	mask2 = np.zeros(shape = (2,len(peakMask)), dtype = bool)
	for isample in range(2):
		mask2[isample] = peakMask
	vars()['hPeak'+ichr] = np.ma.compressed(np.ma.masked_array( vars()['hPeak'+ichr], peakMask ))
	vars()['hGene'+ichr] = np.ma.compressed(np.ma.masked_array( vars()['hGene'+ichr], peakMask ))
	vars()['hSamples'+ichr] = np.ma.compressed(np.ma.masked_array( vars()['hSamples'+ichr], peakMask ))
	vars()['hGeneStart'+ichr] = np.ma.compressed(np.ma.masked_array( vars()['hGeneStart'+ichr], peakMask ))
	vars()['hGenePos'+ichr] = np.ma.compress_cols(np.ma.masked_array( vars()['hGenePos'+ichr], mask2 ))

	# Dictionary geneName --> geneStart
	vars()['geneStartDict'+ichr] = {}
	vars()['geneEndDict'+ichr] = {}
	for i in range(len(vars()['hGene'+ichr])):
		vars()['geneStartDict'+ichr][vars()['hGene'+ichr][i]] = vars()['hGeneStart'+ichr][i]
		vars()['geneEndDict'+ichr][vars()['hGene'+ichr][i]] = vars()['hGenePos'+ichr][1,i]

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
	for line in range(len( vars()['hSamples'+ichr] )):
		igene = uGeneDict[ vars()['hGene'+ichr][line] ]
		ipeak = uPeakDict[ vars()['hPeak'+ichr][line] ]
		vars()['hic'+ichr][igene,ipeak] = vars()['hSamples'+ichr][line]
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

	vars()['genePos'+ichr] = np.zeros(shape=(2,len(vars()['geneStart'+ichr])),dtype='object')
	for igene in range(len(vars()['geneStart'+ichr])):
		vars()['genePos'+ichr][0,igene] = vars()['geneStartDict'+ichr][ vars()['geneMatrix'+ichr][igene] ]
		vars()['genePos'+ichr][1,igene] = vars()['geneEndDict'+ichr][ vars()['geneMatrix'+ichr][igene] ]

	geneSort = np.argsort(vars()['geneStart'+ichr])
	vars()['geneMatrix'+ichr] = vars()['geneMatrix'+ichr][geneSort]
	vars()['geneStart'+ichr] = vars()['geneStart'+ichr][geneSort]
	vars()['genePos'+ichr] = vars()['genePos'+ichr][:,geneSort]
	vars()['hic'+ichr] = vars()['hic'+ichr][geneSort,:]

	## Plot final HiC Matrix on a chromosome
	plt.clf()
	fig = plt.figure(figsize = (8,6))
	plt.imshow(vars()['hic'+ichr], cmap = 'hot_r', aspect='auto', interpolation='none',origin='lower', vmax=30)
	plt.title('HiC Matrix: Chromosome '+ichr,fontsize=18)
	plt.xlabel('Peaks')
	plt.ylabel('Genes')
	plt.grid(True)
	plt.colorbar()
	plt.savefig(wdfigs+'HiC/HiC_on_Chr'+ichr+'.pdf')

	np.save( wdvars+'validation_K562/HiC/hic'+ichr+'.npy', vars()['hic'+ichr] )
	np.save( wdvars+'validation_K562/HiC/geneStart'+ichr+'.npy', vars()['geneStart'+ichr] )
	np.save( wdvars+'validation_K562/HiC/genePos'+ichr+'.npy', vars()['genePos'+ichr] )
	np.save( wdvars+'validation_K562/HiC/geneMatrix'+ichr+'.npy', vars()['geneMatrix'+ichr] )
	np.save( wdvars+'validation_K562/HiC/peakStart'+ichr+'.npy', vars()['peakStart'+ichr] )
	np.save( wdvars+'validation_K562/HiC/peakPos'+ichr+'.npy', vars()['peakPos'+ichr] )
	np.save( wdvars+'validation_K562/HiC/peakMatrix'+ichr+'.npy', vars()['peakMatrix'+ichr] )
	np.save( wdvars+'validation_K562/HiC/geneStartDict'+ichr+'.npy', vars()['peakMatrix'+ichr] )
	
