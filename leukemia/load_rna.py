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
# Load RNA
###################################################################
print('Load RNA')

rnaFile = np.swapaxes(np.array(pd.read_csv(wd+'data/MCG_RNA.rpkm.txt', sep = '\t', header = None)),0,1)

nGenes = rnaFile.shape[1]
chrRNA = rnaFile[0]
positionRNA = np.array(rnaFile[1:3], dtype = int) # start,stop for each gene
lengthRNA = positionRNA[1] - positionRNA[0] # length of gene
direction = np.array(rnaFile[3]) # + or -
direction[direction=='+']=1
direction[direction=='-']=-1
direction = np.array(direction,dtype=int)
geneNameFull = rnaFile[5]

# expression for all 60000 genes
expressionFull = np.zeros(shape = (nSamples,nGenes)) 
for isample in range(nSamples):
	expressionFull[isample] = rnaFile[isample+6]

####### filter genes #######
if MakePlots:
	plt.clf()
	n, bins, patches = plt.hist(np.amax(expressionFull,axis=0), bins=100, range=[0,100])
	plt.title('Histogram of Gene Expression (nGenes = '+str(nGenes)+')')
	plt.xlabel('RNA-seq Expression')
	plt.ylabel('Number of Genes')
	plt.xlim([0,100])
	plt.ylim([0,1000])
	plt.grid(True)
	plt.show()
	
	plt.clf()
	n, bins, patches = plt.hist(lengthRNA, bins=100, range=[0,3000])
	plt.title('Length of Genes (nGenes = '+str(nGenes)+')')
	plt.xlabel('Length of Gene')
	plt.ylabel('Number of Genes')
	#plt.xlim([0,100])
	#plt.ylim([0,1000])
	plt.grid(True)
	plt.show()

np.save(wdvars+'geneNameAllFull.npy',geneNameFull)
### Dictionary of geneName to chromosome and position
geneChrAllDict = {}
genePosAllDict = {}
for i in range(len(geneNameFull)):
	geneChrAllDict[geneNameFull[i]] = chrRNA[i]
	genePosAllDict[geneNameFull[i]] = positionRNA[:,i]
pickle.dump(geneChrAllDict, open(wdvars+'geneChrAllDict.p','wb'))
pickle.dump(genePosAllDict, open(wdvars+'genePosAllDict.p','wb'))

# all Samples have expressions > 0.1
keepRNAall = expressionFull>0.1
keepRNAsum = np.sum(keepRNAall,axis=0)
keepRNA = 1-(keepRNAsum==nSamples)
# length is > 200bp
keepLength = lengthRNA>200
keepLength = 1-keepLength
# combine
keep = np.amax([keepRNA,keepLength],axis=0)

keepFull = np.zeros(shape = (nSamples,len(keep)), dtype = bool)
for isample in range(nSamples):
	keepFull[isample] = keep
keep2 = np.zeros(shape = (2,len(keep)), dtype = bool)
for isample in range(2):
	keep2[isample] = keep
expression = np.ma.compress_cols(np.ma.masked_array(expressionFull,keepFull)) 
chrRNA = np.ma.compressed(np.ma.masked_array(chrRNA,keep))
lengthRNA = np.ma.compressed(np.ma.masked_array(lengthRNA,keep))
direction = np.ma.compressed(np.ma.masked_array(direction,keep))
geneName = np.ma.compressed(np.ma.masked_array(geneNameFull,keep))
positionRNA = np.ma.compress_cols(np.ma.masked_array(positionRNA,keep2))

# remove duplicate genes
names,counts = np.unique(geneName,return_counts=True)
badGenes = names[ np.where(counts>1)[0] ]
mask = np.isin(geneName,badGenes)
maskFull = np.zeros(shape = (nSamples,len(mask)), dtype = bool)
for isample in range(nSamples):
	maskFull[isample] = mask
mask2 = np.zeros(shape = (2,len(mask)), dtype = bool)
for isample in range(2):
	mask2[isample] = mask
expression = np.ma.compress_cols(np.ma.masked_array(expression,maskFull)) 
chrRNA = np.ma.compressed(np.ma.masked_array(chrRNA,mask))
lengthRNA = np.ma.compressed(np.ma.masked_array(lengthRNA,mask))
direction = np.ma.compressed(np.ma.masked_array(direction,mask))
geneName = np.ma.compressed(np.ma.masked_array(geneName,mask))
positionRNA = np.ma.compress_cols(np.ma.masked_array(positionRNA,mask2))

np.save(wdvars+'geneNameAll.npy',geneName)
### Dictionary of geneName to chromosome and position
geneChrAllDict = {}
genePosAllDict = {}
for i in range(len(geneName)):
	geneChrAllDict[geneName[i]] = chrRNA[i]
	genePosAllDict[geneName[i]] = positionRNA[:,i]
pickle.dump(geneChrAllDict, open(wdvars+'geneChrAllDict.p','wb'))
pickle.dump(genePosAllDict, open(wdvars+'genePosAllDict.p','wb'))

# remove genes that do not code for proteins
proteinCodingMask = np.load(wdvars+'proteinCodingMask.npy')
maskFull = np.zeros(shape = (nSamples,len(proteinCodingMask)), dtype = bool)
for isample in range(nSamples):
	maskFull[isample] = proteinCodingMask
mask2 = np.zeros(shape = (2,len(proteinCodingMask)), dtype = bool)
for isample in range(2):
	mask2[isample] = proteinCodingMask
expression = np.ma.compress_cols(np.ma.masked_array(expression,maskFull)) # expression for ~14000 genes
chrRNA = np.ma.compressed(np.ma.masked_array(chrRNA,proteinCodingMask))
lengthRNA = np.ma.compressed(np.ma.masked_array(lengthRNA,proteinCodingMask))
direction = np.ma.compressed(np.ma.masked_array(direction,proteinCodingMask))
geneName = np.ma.compressed(np.ma.masked_array(geneName,proteinCodingMask))
positionRNA = np.ma.compress_cols(np.ma.masked_array(positionRNA,mask2))

### Dictionary of geneName to chromosome and position
geneChrDict = {}
genePosDict = {}
for i in range(len(geneName)):
	geneChrDict[geneName[i]] = chrRNA[i]
	genePosDict[geneName[i]] = positionRNA[:,i]
pickle.dump(geneChrDict, open(wdvars+'geneChrDict.p','wb'))
pickle.dump(genePosDict, open(wdvars+'genePosDict.p','wb'))

nGenes = expression.shape[1]
# 10406 genes kept
############################

######### sort RNA #########
# split into a different array for each chromosome, sorted by gene start within each
# sort by chromosome
sortIndexChr = np.argsort(chrRNA)
chrRNA = chrRNA[sortIndexChr]
geneName = geneName[sortIndexChr]
np.save(wdvars+'RNA/geneName.npy',geneName)
positionRNA = positionRNA[:,sortIndexChr]
expression = expression[:,sortIndexChr]
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y']:
	# define different variables for each chromosome
	vars()['expression'+ichr] = expression[:,(chrRNA=='chr'+ichr)]
	vars()['geneName'+ichr] = geneName[(chrRNA=='chr'+ichr)]
	vars()['chrRNA'+ichr] = chrRNA[(chrRNA=='chr'+ichr)]
	vars()['direction'+ichr] = direction[(chrRNA=='chr'+ichr)]
	vars()['positionRNA'+ichr] = positionRNA[:,(chrRNA=='chr'+ichr)]

	# sort by gene position within chromosome
	sortIndexPos = np.argsort(vars()['positionRNA'+ichr][0,:])
	vars()['expression'+ichr] = vars()['expression'+ichr][:,sortIndexPos]
	vars()['geneName'+ichr] = vars()['geneName'+ichr][sortIndexPos]
	vars()['chrRNA'+ichr] = vars()['chrRNA'+ichr][sortIndexPos]
	vars()['direction'+ichr] = vars()['direction'+ichr][sortIndexPos]
	vars()['positionRNA'+ichr] = vars()['positionRNA'+ichr][:,sortIndexPos]
	vars()['nGenes'+ichr] = vars()['expression'+ichr].shape[1]
	if ichr=='X': exit()
	
	np.save(wdvars+'RNA/expression'+ichr+'.npy', vars()['expression'+ichr])
	np.save(wdvars+'RNA/geneName'+ichr+'.npy', vars()['geneName'+ichr])
	np.save(wdvars+'RNA/chrRNA'+ichr+'.npy', vars()['chrRNA'+ichr])
	np.save(wdvars+'RNA/positionRNA'+ichr+'.npy', vars()['positionRNA'+ichr])
	np.save(wdvars+'RNA/nGenes'+ichr+'.npy', vars()['nGenes'+ichr])
	np.save(wdvars+'RNA/direction'+ichr+'.npy', vars()['direction'+ichr])

############################

