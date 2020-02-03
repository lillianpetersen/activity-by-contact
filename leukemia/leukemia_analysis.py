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
import numpy as np 
import statsmodels.api as sm 
import pylab as py 
import statsmodels.stats.multitest as multitest

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

subtypes = np.array(['ETV6-RUNX1', 'DUX4', 'Hyperdiploid', 'PAX5alt', 'Ph-like'])
typeNames = np.array(['ETVRUNX', 'DUX', 'Hyperdiploid', 'PAX', 'Phlike'])

compactToSubtype = {'ETVRUNX':'ETV6-RUNX1', 
	'DUX':'DUX4',
	'Hyperdiploid':'Hyperdiploid',
	'PAX':'PAX5alt',
	'Phlike':'Ph-like'}

subtypeToCompact = {
	'ETV6-RUNX1':'ETVRUNX',
	'DUX4':'DUX',
	'Hyperdiploid':'Hyperdiploid',
	'PAX5alt':'PAX',
	'Ph-like':'Phlike'}

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

		for itype in range(len(subtypes)):
			subtype = typeNames[itype]
			vars()['expression'+subtype+ichr] = np.load(wdvars+'RNA/expression'+subtype+ichr+'.npy')

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

		vars()['positionActivity'+ichr] = np.array(vars()['positionActivity'+ichr],dtype=int)

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
# Differential Genes and Peaks
###################################################################
for itype in range(len(subtypes)):
	subtype = typeNames[itype]
	subtypeName = subtypes[itype]
	print('Differential Genes: '+subtypeName)

	vars()['allRgenes'+subtype] = pd.read_csv(wddata+'differential_genes/'+subtypeName+'_differential_genes.txt', sep = '\t', header = 0)[:-1]
	p = vars()['allRgenes'+subtype]['p']
	pMask = p<0.05

	vars()['difGenes'+subtype] = vars()['allRgenes'+subtype][pMask].reset_index(drop=True)

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

	# Peaks
	vars()['difPeaks'+subtype] = pd.read_csv(wddata+'differential_genes/differential_peaks_'+subtypeName+'.txt', sep = '\t', header=0)
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
abcHyperdiploid2[325,11777] = 0.07757214
abcHyperdiploid10[145,6927] = 0.0737588
abcHyperdiploid19[592,6894] = 0.0852113

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
		# at abc > 0.07, precision = 0.10 and recall = 0.50
		abcConnections = vars()['abc'+subtype+ichr][vars()['difPeakGene'+subtype+ichr]][vars()['abc'+subtype+ichr][vars()['difPeakGene'+subtype+ichr]]>0]
		if np.sum(abcConnections>0.07)>0:
			abcSig = abcConnections[abcConnections>0.05]
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

				difPeakGene = difPeakGene.append({'subtype':subtypeName,
									'chr':ichr, 
									'geneName':geneName, 
									'igene':igene,  
									'geneP':geneP,  
									'geneLogFC':geneLogFC,  
									'peakName':peakName, 
									'ipeak':ipeak, 
									'peakP':geneP,  
									'peakLogFC':geneLogFC,  
									'ABCscore':abcSig[i]},ignore_index=True)


difPeakGene.to_csv(wdfiles+'differential_peak_gene_connections.tsv', sep='\t', header=True, index=False)


###################################################################
# Plot differential peak/genes across subtypes
###################################################################
print '\n Identify Cancer Genes and Kinases, and Make Plots'

cancerGenes = pd.read_csv(wddata+'COSMIC_cancer_genes.tsv', header=0, sep='\t')
tissue_type = cancerGenes['Tissue Type']
tissue_type = tissue_type.str.split(',', expand=True)
leuk = np.sum(tissue_type=='L',axis=1).astype(bool)
cancerGenes = cancerGenes[leuk].reset_index(drop=True)

geneCategories = pd.read_csv(wddata+'gene_categories.tsv', header=0, sep='\t')
geneCategories['category'] = geneCategories['category'].str.title()

abcdata = np.zeros(shape=(len(difPeakGene),5))
expressiondata = np.zeros(shape=(len(difPeakGene),5))

corr = np.zeros(shape=len(difPeakGene))
p = np.zeros(shape=len(difPeakGene))
geneP = np.zeros(shape=len(difPeakGene))
geneCategory = np.zeros(shape=len(difPeakGene),dtype=object)
cancerGene = np.zeros(shape=len(difPeakGene),dtype=bool)

## Find Corr and P Value
for iline in range(len(difPeakGene)):
	line = difPeakGene.loc[[iline]].reset_index(drop=True)

	subtype = subtypeToCompact[line['subtype'][0]]
	ichr = line['chr'][0]
	geneName = line['geneName'][0]
	igene = line['igene'][0]
	peakName = line['peakName'][0]
	ipeak = line['ipeak'][0]
	geneP[iline] = line['geneP'][0]

	for itype in range(len(typeNames)):
		abcdata[iline,itype] = vars()['abc'+typeNames[itype]+ichr][igene,ipeak]
		expressiondata[iline,itype] = vars()['expression'+typeNames[itype]+ichr][igene]

	corr[iline], p[iline] = stats.pearsonr(abcdata[iline],expressiondata[iline])

## Correct P Value
pBool, pCorrected, tmp, tmp2 = multitest.multipletests(p, alpha=0.05, method='fdr_bh')

## Make Plots
for iline in range(len(difPeakGene)):
	line = difPeakGene.loc[[iline]].reset_index(drop=True)

	subtype = subtypeToCompact[line['subtype'][0]]
	ichr = line['chr'][0]
	geneName = line['geneName'][0]
	igene = line['igene'][0]
	peakName = line['peakName'][0]
	ipeak = line['ipeak'][0]

	cancerGene[iline] = np.amax(np.isin(cancerGenes['Gene Symbol'], geneName))
	HasCategory = np.amax(np.isin(geneCategories['entrez_gene_symbol'], geneName))
	if HasCategory:
		index = np.where(np.isin(geneCategories['entrez_gene_symbol'], geneName))[0][0]
		geneCategory[iline] = geneCategories['category'][index]
	
	#whichDif = np.where(subtype==typeNames)[0][0]
	#exit()

	if MakePlots:
		corrPrint = np.round(corr[iline],2)

		## Plot
		plt.clf()
		fig = plt.figure(figsize=(10,8))
		ax = fig.add_subplot(1,1,1)
		ax.scatter(abcdata[iline],expressiondata[iline],color='b',s=150)
		for i,txt in enumerate(subtypes):
			ax.annotate(txt,(abcdata[iline,i],expressiondata[iline,i]),fontsize=14)
		plt.grid(True)
		ax.set_xlabel('ABC')
		ax.set_ylabel('Gene Expression')
	
		if cancerGene[iline]:
			plt.title(geneName + ' (Leukemia Oncogene) with Peak '+peakName+'\n Corr = '+str(corrPrint)+', P = '+np.format_float_scientific(p[iline],precision=2),fontsize=18)
		elif HasCategory:
			plt.title(geneName + ' ('+geneCategory[iline]+') with Peak '+peakName+'\n Corr = '+str(corrPrint)+', P = '+np.format_float_scientific(p[iline],precision=2),fontsize=18)
		else:
			plt.title(geneName + ' with Peak '+peakName+'\n Corr = '+str(corrPrint)+', P = '+np.format_float_scientific(p[iline],precision=2),fontsize=18)
	
		if cancerGene[iline]:
			plt.savefig(wdfigs+'leukemia/differential_peakgenes/cancer_genes/'+subtype+'_'+str(corrPrint)+'_'+vars()['geneName'+ichr][igene]+'_peak_'+peakName+'.pdf')
		elif HasCategory and p[iline]<0.05:
			plt.savefig(wdfigs+'leukemia/differential_peakgenes/gene_categories/'+subtype+'_'+str(corrPrint)+'_'+vars()['geneName'+ichr][igene]+'_peak_'+peakName+'.pdf')
		elif p[iline]<0.05:
			plt.savefig(wdfigs+'leukemia/differential_peakgenes/no_cancer_high_p/'+subtype+'_'+str(corrPrint)+'_'+vars()['geneName'+ichr][igene]+'_peak_'+peakName+'.pdf')
		else:
			plt.savefig(wdfigs+'leukemia/differential_peakgenes/other/'+subtype+'_'+str(corrPrint)+'_'+vars()['geneName'+ichr][igene]+'_peak_'+peakName+'.pdf')


###### Make a QQ Plot ######
'''
# divide the diff gene p values into high and low corr (corr from ABC and expression)
geneP_corr = geneP[np.abs(corr)>0.85]
geneP_lowcorr = geneP[np.abs(corr)<0.6]

# calculate -log10(observed p) for high corr (qp stands for "p value for qq plot")
qp_high = -1*np.log10(np.sort(geneP_highcorr))
# calculate -log10(expectetd p) for high corr
qpExpected_high = np.arange(0.01,1.01,1./len(qp_high))
qpExpected_high = -1*np.log10(qpExpected_high)

# calculate -log10(observed p) for low corr
qp_low = -1*np.log10(np.sort(geneP_lowcorr))
qpExpected_low = np.arange(0.01,1.01,1./len(qp_low))
qpExpected_low = -1*np.log10(qpExpected_low)

maxqp = np.amax(qpExpected_low)

# make the qq plot
plt.clf()
plt.plot(qpExpected_low, qp_low, 'b.', markersize=15, mec='k', mew=1, label='Good Corr')
plt.plot(qpExpected_high, qp_high, 'g.', markersize=15, mec='k', mew=1, label='Low Corr')
plt.plot([0,maxqp],[0,maxqp],'r-')
plt.grid(True)
plt.axis('equal')
plt.ylim([0,maxqp])
plt.xlim([0,maxqp])
plt.title('QQ Plot of P Values of Differential Genes',fontsize=15)
plt.ylabel('-log10(observed p)',fontsize=13)
plt.xlabel('-log10(expected p)',fontsize=13)
plt.legend(loc='lower right')
plt.savefig(wdfigs+'leukemia/qq_plot_of_p_values_of_diff_genes.pdf')
'''

###################################################################
# Motif Analysis
###################################################################
print '\n Motif Analysis'

motifFull = pd.read_csv(wddata+'motifs_on_dif_peaks.txt', sep='\t', header=0)

clusters = pd.read_csv(wddata+'motif_clusters.tsv', header=0, sep='\t')
motifToCluster = {}
motifClusters = clusters['motifs'].str.upper()
for icluster in range(111):
	motifstmp = np.array(motifClusters[[icluster]].str.split(',',expand=True))[0]
	for imotif in range(len(motifstmp)):
		motifToCluster[motifstmp[imotif]] = icluster+1

motifs = pd.DataFrame(columns = ['subtype','chr','geneName','igene','geneP','geneLogFC','peakName','ipeak','PeakP','peakLogFC','ABCscore','motif','cluster','iline'])

for iline in range(len(difPeakGene)):
	line = difPeakGene.loc[[iline]].reset_index(drop=True)

	subtype = subtypeToCompact[line['subtype'][0]]
	ichr = line['chr'][0]
	geneName = line['geneName'][0]
	igene = line['igene'][0]
	peakName = line['peakName'][0]
	ipeak = line['ipeak'][0]
	geneP[iline] = line['geneP'][0]
	peakPos = vars()['positionActivity'+ichr][:,ipeak]

	motiftmp = motifFull[motifFull['seqnames']=='chr'+ichr]
	motiftmp = motiftmp[motiftmp['groups']==compactToSubtype[subtype]].reset_index(drop=True)

	motifPeakPos = np.zeros(shape=(2,len(motiftmp)),dtype=int)
	motifPeakPos[0] = motiftmp['start']
	motifPeakPos[1] = motiftmp['end']

	motifsAll = np.array(motiftmp['name'])

	whereMotifs = np.where((motifPeakPos[1]>peakPos[0])==(motifPeakPos[0]<peakPos[1]))[0]
	if len(whereMotifs)==0: 
		print subtype,'\t', geneName,'\t', peakName
		continue
	motifsThisLine = motifsAll[whereMotifs]

	motifsThisLine = pd.DataFrame(motifsThisLine,columns=['original'])
	motifsThisLine['original'] = motifsThisLine['original'].str.replace('\(var\.2\)','_var_2_')
	motifsThisLine = motifsThisLine['original'].str.split('.',expand=True)
	motifsThisLine.columns = ['ID','#','motif']
	motifsThisLine['motif'] = motifsThisLine['motif'].str.upper()
	clusterstmp = np.zeros(shape=len(motifsThisLine),dtype=int)
	for imotif in range(len(motifsThisLine)):
		try:
			clusterstmp[imotif] = motifToCluster[motifsThisLine['motif'][imotif]]
		except:
			clusterstmp[imotif] = -999
	motifsThisLine = motifsThisLine.assign(cluster = clusterstmp)
	motifsThisLine = motifsThisLine.drop(['ID','#'],axis=1)

	for imotif in range(len(motifsThisLine)):
		dftmp = line.assign(motif=motifsThisLine['motif'][imotif]).assign(cluster=motifsThisLine['cluster'][imotif]).assign(iline=iline)
		motifs = motifs.append( dftmp, ignore_index=True, sort=False)

for itype in range(len(subtypes)):
	subtype = typeNames[itype]
	subtypeName = subtypes[itype]

	vars()['peaksWithCluster'+subtype] = np.zeros(shape=111)
	for icluster in range(1,112):
		motifstmp = motifs[motifs['subtype']==subtypeName].reset_index(drop=True)
		whereCluster = np.where(motifstmp['cluster']==icluster)[0]
		#if len(whereCluster)>1 or len(whereCluster)==0:
		uniquePeaksCluster = np.unique(motifstmp['peakName'][whereCluster])
		#else:
		#	uniquePeaksCluster = np.unique(motifstmp['peakName'][whereCluster[0]])
		vars()['peaksWithCluster'+subtype][icluster-1] = len(uniquePeaksCluster)
	
peaksWithClusterDUX = peaksWithClusterDUX/np.sum(peaksWithClusterDUX)
peaksWithClusterETVRUNX = peaksWithClusterETVRUNX/np.sum(peaksWithClusterETVRUNX)
peaksWithClusterHyperdiploid = peaksWithClusterHyperdiploid/np.sum(peaksWithClusterHyperdiploid)

peaksWithClusterDUX = peaksWithClusterDUX[:102]
peaksWithClusterETVRUNX = peaksWithClusterETVRUNX[:102]
peaksWithClusterHyperdiploid = peaksWithClusterHyperdiploid[:102]

clusterNames = np.array(range(1,103),dtype=object)

clustersAll = np.array(np.sum([peaksWithClusterDUX,peaksWithClusterETVRUNX,peaksWithClusterHyperdiploid],axis=0),dtype=bool)

x = np.array(range(len(clusterNames)))

plt.clf()
fig, axs = plt.subplots(3, 1, sharex=True, sharey=False)

axs[0].bar(x, peaksWithClusterETVRUNX, color='cyan', alpha=0.9, label='ETV6-RUNX1')
axs[0].grid(True)
axs[0].set_ylim([0,0.09])
axs[0].legend(loc='upper right',fontsize=12)

axs[1].bar(x, peaksWithClusterDUX, color='lawngreen', alpha=0.8, label='DUX4')
axs[1].set_ylabel('Frequency')
axs[1].grid(True)
axs[1].set_ylim([0,0.09])
axs[1].legend(loc='upper right',fontsize=12)

axs[2].bar(x, peaksWithClusterHyperdiploid, color='orange', alpha=0.8, label='Hyperdiploid')
axs[2].set_ylim([0,0.09])
plt.xlabel('Cluster Category')
plt.xlim([0,102])
plt.grid(True)
plt.legend(loc='upper right',fontsize=12)
fig.suptitle('Motifs Present in Differential Peaks with \n High ABC Scores to Differential Genes',fontsize=17)
plt.savefig(wdfigs+'leukemia/motif_barchart.pdf')
plt.show()

































