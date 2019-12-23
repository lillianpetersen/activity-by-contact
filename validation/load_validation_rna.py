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

rnaFile = pd.read_csv(wd+'data/validation_K562/rna_K562.tsv', sep = '\t', header = 0)
rnaFile = rnaFile[rnaFile['FPKM']>0.1]
rnaFile = rnaFile[rnaFile['length']>200]
rnaFile = rnaFile.reset_index()

nGenes = len(rnaFile)
#chrRNA = rnaFile[1]
#positionRNA = np.array(rnaFile[2:4], dtype = int) # start,stop for each gene
#lengthRNA = positionRNA[1] - positionRNA[0] # length of gene
#direction = np.array(rnaFile[4]) # + or -
#direction[direction=='+']=1
#direction[direction=='-']=-1
#direction = np.array(direction,dtype=int)
geneID = rnaFile['gene_id']
#np.save(wd+'saved_variables/validation_K562/geneIDfull.npy',np.array(geneID))

inFileMask = np.load(wdvars+'validation_K562/inFileMask.npy')
proteinCodingMask = np.load(wdvars+'validation_K562/proteinCodingMask.npy')
geneName = np.load(wdvars+'validation_K562/geneName.npy')
chrRNA = np.load(wdvars+'validation_K562/geneChr.npy')
start = np.load(wdvars+'validation_K562/geneStart.npy')
stop = np.load(wdvars+'validation_K562/geneStop.npy')
direction = np.load(wdvars+'validation_K562/direction.npy')
positionRNA = np.zeros(shape=(2,len(start)))
positionRNA[0] = start
positionRNA[1] = stop
lengthRNA = positionRNA[1] - positionRNA[0] # length of gene
direction[direction=='+']=1
direction[direction=='-']=-1
direction = np.array(direction,dtype=int)

expression = rnaFile['FPKM']
expression = expression[inFileMask]

proteinCodingMask = np.invert(proteinCodingMask)
expression = expression[proteinCodingMask]
geneName = geneName[proteinCodingMask]
direction = direction[proteinCodingMask]
positionRNA = positionRNA[:,proteinCodingMask]
lengthRNA = lengthRNA[proteinCodingMask]
chrRNA = chrRNA[proteinCodingMask]
# expression for all 60000 genes
#expressionFull = np.zeros(shape = (nSamples,nGenes)) 
#for isample in range(nSamples):
#	expressionFull[isample] = rnaFile[isample+5]*1000

####### filter genes #######
if MakePlots:
	plt.clf()
	n, bins, patches = plt.hist(np.amax(expression,axis=0), bins=100, range=[0,100])
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

#######################
# Filter out genes of insufficient expression or length
#######################

if np.sum(np.isnan(expression))>0:
	print 'NANs'
	## all Samples have expressions > 0.1
	#keepRNAall = expression>0.001
	keepRNA = np.isnan(expression)
	#keepRNAsum = np.sum(keepRNAall,axis=0)
	#keepRNA = 1-(keepRNAsum>31)
	# length is > 200bp
	#keepLength = lengthRNA<200
	# combine
	keep = keepRNA
	
	#keepFull = np.zeros(shape = (nSamples,len(keep)), dtype = bool)
	#for isample in range(nSamples):
	#	keepFull[isample] = keep
	keep2 = np.zeros(shape = (2,len(keep)), dtype = bool)
	for isample in range(2):
		keep2[isample] = keep
	expression = np.ma.compressed(np.ma.masked_array(expression,keep)) 
	chrRNA = np.ma.compressed(np.ma.masked_array(chrRNA,keep))
	lengthRNA = np.ma.compressed(np.ma.masked_array(lengthRNA,keep))
	direction = np.ma.compressed(np.ma.masked_array(direction,keep))
	geneName = np.ma.compressed(np.ma.masked_array(geneName,keep))
	positionRNA = np.ma.compress_cols(np.ma.masked_array(positionRNA,keep2))

#np.save(wdvars+'validation/RNA/geneNameAll.npy',geneName)

### Dictionary of geneName to chromosome and position
geneChrDict = {}
genePosDict = {}
for i in range(len(geneName)):
	geneChrDict[geneName[i]] = chrRNA[i]
	genePosDict[geneName[i]] = positionRNA[:,i]
pickle.dump(geneChrDict, open(wdvars+'validation_K562/RNA/geneChrDict.p','wb'))
pickle.dump(genePosDict, open(wdvars+'validation_K562/RNA/genePosDict.p','wb'))

nGenes = expression.shape
# 10406 genes kept
############################

######### sort RNA #########
# split into a different array for each chromosome, sorted by gene start within each
# sort by chromosome
#sortIndexChr = np.argsort(chrRNA)
#chrRNA = chrRNA[sortIndexChr]
#geneName = geneName[sortIndexChr]
#np.save(wdvars+'RNA/geneName.npy',geneName)
#positionRNA = positionRNA[:,sortIndexChr]
#expression = expression[sortIndexChr]
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X','Y']:
	# define different variables for each chromosome
	vars()['expression'+ichr] = np.array(expression[(chrRNA=='chr'+ichr)])
	vars()['geneName'+ichr] = np.array(geneName[(chrRNA=='chr'+ichr)])
	vars()['chrRNA'+ichr] = np.array(chrRNA[(chrRNA=='chr'+ichr)])
	vars()['direction'+ichr] = np.array(direction[(chrRNA=='chr'+ichr)])
	vars()['positionRNA'+ichr] = np.array(positionRNA[:,(chrRNA=='chr'+ichr)])

	# sort by gene position within chromosome
	sortIndexPos = np.argsort(vars()['positionRNA'+ichr][0,:])
	vars()['expression'+ichr] = vars()['expression'+ichr][sortIndexPos]
	vars()['geneName'+ichr] = vars()['geneName'+ichr][sortIndexPos]
	vars()['chrRNA'+ichr] = vars()['chrRNA'+ichr][sortIndexPos]
	vars()['direction'+ichr] = vars()['direction'+ichr][sortIndexPos]
	vars()['positionRNA'+ichr] = vars()['positionRNA'+ichr][:,sortIndexPos]
	vars()['nGenes'+ichr] = vars()['expression'+ichr].shape[0]
	
	np.save(wdvars+'validation_K562/RNA/expression'+ichr+'.npy', vars()['expression'+ichr])
	np.save(wdvars+'validation_K562/RNA/geneName'+ichr+'.npy', vars()['geneName'+ichr])
	np.save(wdvars+'validation_K562/RNA/chrRNA'+ichr+'.npy', vars()['chrRNA'+ichr])
	np.save(wdvars+'validation_K562/RNA/positionRNA'+ichr+'.npy', vars()['positionRNA'+ichr])
	np.save(wdvars+'validation_K562/RNA/nGenes'+ichr+'.npy', vars()['nGenes'+ichr])
	np.save(wdvars+'validation_K562/RNA/direction'+ichr+'.npy', vars()['direction'+ichr])

############################

