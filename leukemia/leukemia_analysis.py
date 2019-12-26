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

wd = '/pbld/mcg/lillianpetersen/ABC/'
wdvars = '/pbld/mcg/lillianpetersen/ABC/saved_variables/'
wddata = '/pbld/mcg/lillianpetersen/ABC/data/'
wdfigs = '/pbld/mcg/lillianpetersen/ABC/figures/'
wdfiles = '/pbld/mcg/lillianpetersen/ABC/written_files/'
MakePlots = False

# MCG B ALL Samples
MCGs = ['MCG001', 'MCG002', 'MCG003', 'MCG005', 'MCG006', 'MCG009', 'MCG010', 'MCG011', 'MCG012', 'MCG013', 'MCG016', 'MCG017', 'MCG019', 'MCG020', 'MCG023', 'MCG024', 'MCG027', 'MCG028', 'MCG034', 'MCG035', 'MCG036', 'MCG037', 'MCG038', 'MCG039']
nSamples = len(MCGs)
nChr = 23

subtypes = np.array(['ETV6-RUNX1', 'DUX4', 'Hyperdiploid', 'PAX5', 'Ph-like'])
typeNames = np.array(['ETVRUNX', 'DUX', 'Hyperdiploid', 'PAX', 'Phlike'])

sampleTypes = np.array(['PAX5', 'ETV6-RUNX1', 'PAX5alt', 'DUX4', 'ZNF384', 'PAX5alt', 'Hyperdiploid', 'DUX4', 'Hyperdiploid', 'Ph-like', 'Ph-like', 'ETV6-RUNX1', 'Hyperdiploid', 'Hyperdiploid', 'DUX4', 'ETV6-RUNX1', 'Hyperdiploid', 'Hyperdiploid', 'Hyperdiploid', 'ETV6-RUNX1', 'DUX4', 'Other', 'Ph-like', 'Ph-like'])
# 0=ETV6-RUNX1, 1=DUX4, 2=Hyperdiploid, 3=PAX5, 4=Ph-like, 5=Other
typesIndex = np.array([3, 0, 3, 1, 5, 3, 2, 1, 2, 4, 4, 0, 2, 2, 1, 0, 2, 2, 2, 0, 1, 5, 4, 4])

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

###################################################################
# Load RNA
###################################################################
print('Load RNA')
nGenes=0
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	if not 'expression'+ichr in globals():
		# Load arrays saved from load_rna.npy
		vars()['expression'+ichr] = np.load(wdvars+'RNA/expression'+ichr+'.npy')
		vars()['geneName'+ichr] = np.load(wdvars+'RNA/geneName'+ichr+'.npy')
		vars()['geneID'+ichr] = np.load(wdvars+'RNA/geneID'+ichr+'.npy')
		vars()['chrRNA'+ichr] = np.load(wdvars+'RNA/chrRNA'+ichr+'.npy')
		vars()['positionRNA'+ichr] = np.load(wdvars+'RNA/positionRNA'+ichr+'.npy')
		vars()['direction'+ichr] = np.load(wdvars+'RNA/direction'+ichr+'.npy')

###################################################################
# Load Activity
###################################################################
print('Load Activity')
for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
	if not 'activity'+ichr in globals():
		# Load arrays saved from load_atac.npy
		vars()['activity'+ichr] = np.load(wdvars+'ATAC/atac'+ichr+'.npy')
		vars()['chrActivity'+ichr] = np.load(wdvars+'ATAC/chrATAC'+ichr+'.npy')
		vars()['peakName'+ichr] = np.load(wdvars+'ATAC/peakName'+ichr+'.npy')
		vars()['positionActivity'+ichr] = np.load(wdvars+'ATAC/positionATAC'+ichr+'.npy')

		for itype in range(len(subtypes)):
			subtype = typeNames[itype]
			vars()['activity'+subtype+ichr] = np.load(wdvars+'ATAC/atac'+subtype+ichr+'.npy')

###################################################################
# Load HiC and ABC
###################################################################
for itype in range(len(subtypes)):
	subtype = typeNames[itype]
	subtypeName = subtypes[itype]
	print('load HiC and ABC: '+subtypeName)

	for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
		if not 'hic'+subtype+ichr in globals():
			# Load arrays saved from load_hic.npy
			vars()['hic'+subtype+ichr] = np.load( wdvars+'HiC/'+subtypeName+'/hic'+ichr+'.npy')
			vars()['geneStart'+ichr] = np.load( wdvars+'HiC/general/geneStart'+ichr+'.npy' )
			vars()['geneMatrix'+ichr] = np.load( wdvars+'HiC/general/geneMatrix'+ichr+'.npy' )
			vars()['peakStart'+ichr] = np.load( wdvars+'HiC/general/peakStart'+ichr+'.npy' )
			vars()['peakMatrix'+ichr] = np.load( wdvars+'HiC/general/peakMatrix'+ichr+'.npy' )

			vars()['abc'+subtype+ichr] = np.load( wdvars+'ABC/abc'+subtype+ichr+'.npy')

###################################################################
# Differential Genes
###################################################################
chrOrder = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,'X','Y','M'])

for itype in range(len(subtypes)):
	subtype = typeNames[itype]
	subtypeName = subtypes[itype]
	print('Differential Genes: '+subtypeName)

	vars()['difGenes'+subtype] = pd.read_csv(wddata+'differential_genes/'+subtypeName+'_significant_genes.txt', sep = '\t', header=0)
	vars()['difGenes'+subtype]['geneID']=vars()['difGenes'+subtype]['genes']
	del vars()['difGenes'+subtype]['genes']
	vars()['difGenes'+subtype]['geneName'] = np.zeros( shape=len(vars()['difGenes'+subtype]), dtype=object)
	vars()['difGenes'+subtype]['chr'] = np.zeros( shape=len(vars()['difGenes'+subtype]), dtype=object)

	for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:

		genesIndex = np.isin(vars()['difGenes'+subtype]['geneID'], vars()['geneID'+ichr])

		for igene in range(sum(genesIndex)):
			geneIndexDif = np.where(genesIndex)[0][igene]
			geneID = vars()['difGenes'+subtype]['geneID'][geneIndexDif]
			geneIndexRNA = np.where( vars()['geneID'+ichr]==geneID )[0][0]
			vars()['difGenes'+subtype]['geneName'][geneIndexDif] = vars()['geneName'+ichr][geneIndexRNA]
			vars()['difGenes'+subtype]['chr'][geneIndexDif] = ichr

	mask = vars()['difGenes'+subtype]['geneName']!=0
	vars()['difGenes'+subtype] = vars()['difGenes'+subtype][mask].reset_index(drop=True)
	del vars()['difGenes'+subtype]['index']

	# Peaks
	vars()['difPeaks'+subtype] = pd.read_csv(wddata+'differential_genes/differential_peaks_'+subtypeName+'.txt', sep = '\t', header=0)
	del vars()['difPeaks'+subtype]['index']
	del vars()['difPeaks'+subtype]['Unnamed: 0']
	mask = vars()['difPeaks'+subtype]['convert']>0
	vars()['difPeaks'+subtype] = vars()['difPeaks'+subtype][mask].reset_index(drop=True)

###################################################################
# ABC between differential genes and peaks
###################################################################
abcCutoff = np.load(wdvars+'validation_K562/PrecisionRecall/atacContactConnected.npy')
abcPrecision = np.load(wdvars+'validation_K562/PrecisionRecall/precisionAtacContact.npy')
abcRecall = np.load(wdvars+'validation_K562/PrecisionRecall/recallAtacContact.npy')

difPeakGene = pd.DataFrame(columns=['subtype','chr','geneName','igene','geneP','geneLogFC','peakName','ipeak','peakP','peakLogFC','ABCscore'])

# small fix
abcHyperdiploid16[46,417] = 0.18537948

print('\n')
for itype in range(len(subtypes)):
	subtype = typeNames[itype]
	subtypeName = subtypes[itype]
	print('ABC analysis: '+subtypeName)

	for ichr in ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19','20','21','22','X']:
		vars()['difPeakGene'+subtype+ichr] = np.zeros(shape=vars()['abc'+subtype+ichr].shape, dtype=bool)

		genes = np.array(vars()['difGenes'+subtype]['geneName'][vars()['difGenes'+subtype]['chr']==ichr],dtype=object)
		geneIndices = np.where(np.isin(vars()['geneName'+ichr], genes))[0]

		peakIndices = np.unique(np.array(vars()['difPeaks'+subtype]['convert'][vars()['difPeaks'+subtype]['chr']=='chr'+ichr], dtype=int))

		for igene in range(len(geneIndices)):
			vars()['difPeakGene'+subtype+ichr][geneIndices[igene],peakIndices] = 1

		# at abc > 0.32, precision = 0.25 and recall = 0.10
		# at abc > 0.11, precision = 0.14 and recall = 0.40
		abcConnections = vars()['abc'+subtype+ichr][vars()['difPeakGene'+subtype+ichr]][vars()['abc'+subtype+ichr][vars()['difPeakGene'+subtype+ichr]]>0]
		if np.sum(abcConnections>0.11)>0:
			abcSig = abcConnections[abcConnections>0.11]
			for i in range(len(abcSig)):
				igene = np.where(vars()['abc'+subtype+ichr]==abcSig[i])[0][0]
				ipeak = np.where(vars()['abc'+subtype+ichr]==abcSig[i])[1][0]
				geneName = vars()['geneName'+ichr][igene]
				peakName = vars()['peakName'+ichr][ipeak]

				# retrieve peak and gene differential info
				geneIndex = np.where(vars()['difGenes'+subtype]['geneName']==geneName)[0][0]
				geneP = vars()['difGenes'+subtype]['p'][geneIndex]
				geneLogFC = vars()['difGenes'+subtype]['logFC'][geneIndex]

				peakTable = vars()['difPeaks'+subtype][vars()['difPeaks'+subtype]['chr']=='chr'+ichr].reset_index(drop=True)
				peakIndex = np.where(peakTable['convert']==ipeak)[0]
				peakP = np.amin(peakTable['padj'][peakIndex])
				peakLogFC = np.amax(peakTable['log2FoldChange'][peakIndex])

				difPeakGene = difPeakGene.append({'subtype':subtype,
									'chr':'chr'+ichr, 
									'geneName':geneName, 
									'igene':igene,  
									'geneP':geneP,  
									'geneLogFC':geneLogFC,  
									'peakName':peakName, 
									'ipeak':ipeak, 
									'peakP':geneP,  
									'peakLogFC':geneLogFC,  
									'ABCscore':abcSig[i]},ignore_index=True)










