import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sys import exit

wd = '/pbld/mcg/lillianpetersen/ABC/'
wdvars = '/pbld/mcg/lillianpetersen/ABC/saved_variables/'
wdfigs = '/pbld/mcg/lillianpetersen/ABC/figures/'
MakePlots = False

# MCG B ALL Samples
MCGs = ['MCG001', 'MCG002', 'MCG003', 'MCG005', 'MCG006', 'MCG009', 'MCG010', 'MCG011', 'MCG012', 'MCG013', 'MCG016', 'MCG017', 'MCG019', 'MCG020', 'MCG023', 'MCG024']
nSamples = len(MCGs)

###################################################################
# Load HiC
###################################################################
print('Load HiC')
try:
	# Load saved variables to save time
	hicSamples = np.load(wdvars+'HiC/hicSamples.npy')
	hicGenes = np.load(wdvars+'HiC/hicGenes.npy')
	hicPeaks = np.load(wdvars+'HiC/hicPeaks.npy')
except:
	# Code to calculate saved variables, if they cannot load
	print('Try command failed: loading and masking HiC Data')
	
	hicFile = pd.read_csv(wd+'data/HiC_matrix_500000_by_sample_quantile.txt', sep = '\t', header = None)
	
	hicFile[['gene','peak']] = hicFile[0].str.split(':',expand=True)
	
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
		#print(np.round(100*line/float(hicFile.shape[0]),2), '%') # counter
		gene = hicGenes[line]
		goodLine[line] = geneDict[gene] #1=bad, 0=good
	
	hicArray = np.swapaxes(np.array(hicFile),0,1)[1:]
	hicPeaks = np.array(hicFile['peak'])
	hicSamplesAll = hicArray[:nSamples]
	# reshape mask to have nSamples rows
	mask = np.zeros(shape = (nSamples,hicSamplesAll.shape[1]), dtype = bool)
	idx = np.where(goodLine==1)[0]
	mask[:,idx] = 1
	
	# mask and compress all arrays to get rid of bad genes
	hicSamplesAll = np.ma.masked_array(hicSamplesAll,mask)
	hicSamples = np.ma.compress_cols(hicSamplesAll)
	hicGenes = np.ma.compressed(np.ma.masked_array(hicGenes,goodLine))
	uniqueGenes = np.unique(hicGenes)
	hicPeaks = np.ma.compressed(np.ma.masked_array(hicPeaks,goodLine))
	
	# save variables
	np.save(wdvars+'hicSamples.npy',hicSamples)
	np.save(wdvars+'hicGenes.npy',hicGenes)
	np.save(wdvars+'hicUniqueGenes.npy',uniqueGenes)
	np.save(wdvars+'hicPeaks.npy',hicPeaks)
	np.save(wdvars+'hicLineMask.npy',goodLine)

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
hSamplesSort = hicSamples[:,sortChr]

############ Sort HiC Lines by Gene Start Within each Chromosome ############
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	print '\nSort HiC by Gene: Chromosome '+ichr
	vars()['hGeneStart'+ichr] = hStartSort[hChrSort==('chr'+ichr)]
	vars()['hGene'+ichr] = hGeneSort[hChrSort==('chr'+ichr)]
	vars()['hPeak'+ichr] = hPeakSort[hChrSort==('chr'+ichr)]
	vars()['hSamples'+ichr] = hSamplesSort[:,hChrSort==('chr'+ichr)]
	sortPos = np.argsort( vars()['hGeneStart'+ichr] )
	vars()['hGeneStart'+ichr] = vars()['hGeneStart'+ichr][sortPos]
	vars()['hGene'+ichr] = vars()['hGene'+ichr][sortPos]
	vars()['hPeak'+ichr] = vars()['hPeak'+ichr][sortPos]
	vars()['hSamples'+ichr] = vars()['hSamples'+ichr][:,sortPos]

	# Mask lines with peaks on wrong chromosomes
	peakMask = np.ones(shape=( len(vars()['hPeak'+ichr])),dtype=bool)
	for line in range(len( vars()['hPeak'+ichr] )):
		tmp = vars()['hPeak'+ichr][line].split('_')
		peakChr = tmp[0]
		if peakChr==('chr'+ichr):
			peakMask[line] = 0
	
	# Remove lines with peaks on wrong chromosomes
	mask = np.zeros(shape = (nSamples,len(peakMask)), dtype = bool)
	idx = np.where(peakMask==1)[0]
	mask[:,idx] = 1
	vars()['hPeak'+ichr] = np.ma.compressed(np.ma.masked_array( vars()['hPeak'+ichr], peakMask ))
	vars()['hGene'+ichr] = np.ma.compressed(np.ma.masked_array( vars()['hGene'+ichr], peakMask ))
	vars()['hSamples'+ichr] = np.ma.compress_cols(np.ma.masked_array( vars()['hSamples'+ichr], mask ))
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
	vars()['hic'+ichr] = np.zeros(shape = (nSamples, len(uGeneDict), len(uPeakDict)) )
	for line in range(len( vars()['hSamples'+ichr][0] )):
		igene = uGeneDict[ vars()['hGene'+ichr][line] ]
		ipeak = uPeakDict[ vars()['hPeak'+ichr][line] ]
		vars()['hic'+ichr][:,igene,ipeak] = vars()['hSamples'+ichr][:,line]
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
	vars()['hic'+ichr] = vars()['hic'+ichr][:,:,peakSort]

	### Sort HiC Matrix by gene position ###
	vars()['geneStart'+ichr] = []
	for igene in range(len(vars()['geneMatrix'+ichr])):
		vars()['geneStart'+ichr].append( vars()['geneStartDict'+ichr][ vars()['geneMatrix'+ichr][igene] ] )
	vars()['geneStart'+ichr] = np.array(vars()['geneStart'+ichr])
	geneSort = np.argsort(vars()['geneStart'+ichr])
	vars()['geneMatrix'+ichr] = vars()['geneMatrix'+ichr][geneSort]
	vars()['geneStart'+ichr] = vars()['geneStart'+ichr][geneSort]
	vars()['hic'+ichr] = vars()['hic'+ichr][:,geneSort,:]

	## Plot final HiC Matrix on a chromosome
	plt.clf()
	fig = plt.figure(figsize = (8,6))
	plt.imshow(vars()['hic'+ichr][0], cmap = 'hot_r', aspect='auto', interpolation='none',origin='lower', vmax=30)
	plt.title('HiC Matrix: MCG001 Chromosome 21',fontsize=18)
	plt.xlabel('Peaks')
	plt.ylabel('Genes')
	plt.grid(True)
	plt.colorbar()
	plt.savefig(wdfigs+'HiC/HiC_MCG001_Chr21.pdf')

	np.save( wdvars+'HiC/hic'+ichr+'.npy', vars()['hic'+ichr] )
	np.save( wdvars+'HiC/geneStart'+ichr+'.npy', vars()['geneStart'+ichr] )
	np.save( wdvars+'HiC/geneMatrix'+ichr+'.npy', vars()['geneMatrix'+ichr] )
	np.save( wdvars+'HiC/peakStart'+ichr+'.npy', vars()['peakStart'+ichr] )
	np.save( wdvars+'HiC/peakPos'+ichr+'.npy', vars()['peakPos'+ichr] )
	np.save( wdvars+'HiC/peakMatrix'+ichr+'.npy', vars()['peakMatrix'+ichr] )
	np.save( wdvars+'HiC/geneStartDict'+ichr+'.npy', vars()['peakMatrix'+ichr] )
	
