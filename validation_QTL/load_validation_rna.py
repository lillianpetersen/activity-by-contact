import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sys import exit

wd = '/pbld/mcg/lillianpetersen/ABC/'
wdvars = '/pbld/mcg/lillianpetersen/ABC/saved_variables/'
wdfigs = '/pbld/mcg/lillianpetersen/ABC/figures/'
MakePlots = False

nSamples = 92

###################################################################
# Load RNA
###################################################################
print('Load RNA')

rnaFile = np.swapaxes(np.array(pd.read_csv(wd+'data/validation/genes/validation.rpkm.qn.txt', sep = '\t', header = None)),0,1)

nGenes = rnaFile.shape[1]
chrRNA = rnaFile[1]
positionRNA = np.array(rnaFile[2:4], dtype = int) # start,stop for each gene
lengthRNA = positionRNA[1] - positionRNA[0] # length of gene
direction = np.array(rnaFile[4]) # + or -
direction[direction=='+']=1
direction[direction=='-']=-1
direction = np.array(direction,dtype=int)
geneNameFull = rnaFile[0]

# expression for all 60000 genes
expressionFull = np.zeros(shape = (nSamples,nGenes)) 
for isample in range(nSamples):
	expressionFull[isample] = rnaFile[isample+5]*1000

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

# all Samples have expressions > 0.1
keepRNAall = expressionFull>0.001
keepRNAsum = np.sum(keepRNAall,axis=0)
keepRNA = 1-(keepRNAsum>31)
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

np.save(wdvars+'validation/RNA/geneNameAll.npy',geneName)
### Dictionary of geneName to chromosome and position
geneChrDict = {}
genePosDict = {}
for i in range(len(geneName)):
	geneChrDict[geneName[i]] = chrRNA[i]
	genePosDict[geneName[i]] = positionRNA[:,i]
pickle.dump(geneChrDict, open(wdvars+'validation/RNA/geneChrDict.p','wb'))
pickle.dump(genePosDict, open(wdvars+'validation/RNA/genePosDict.p','wb'))

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
	
	np.save(wdvars+'validation/RNA/expression'+ichr+'.npy', vars()['expression'+ichr])
	np.save(wdvars+'validation/RNA/geneName'+ichr+'.npy', vars()['geneName'+ichr])
	np.save(wdvars+'validation/RNA/chrRNA'+ichr+'.npy', vars()['chrRNA'+ichr])
	np.save(wdvars+'validation/RNA/positionRNA'+ichr+'.npy', vars()['positionRNA'+ichr])
	np.save(wdvars+'validation/RNA/nGenes'+ichr+'.npy', vars()['nGenes'+ichr])
	np.save(wdvars+'validation/RNA/direction'+ichr+'.npy', vars()['direction'+ichr])

############################

