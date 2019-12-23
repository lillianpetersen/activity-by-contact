import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sys import exit
from scipy import stats
from collections import Counter
import statsmodels.stats.multitest as multi
from math import e
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import sklearn.linear_model

wd = '/pbld/mcg/lillianpetersen/ABC/'
wdvars = '/pbld/mcg/lillianpetersen/ABC/saved_variables/'
wdfigs = '/pbld/mcg/lillianpetersen/ABC/figures/'
wdfiles = '/pbld/mcg/lillianpetersen/ABC/written_files/'
MakePlots = False
PC = False

if PC:
	Xs = ['ABC','ATAC','ABC-PC','ATAC-PC']
	nX = 4
	nMethods = 3
else:
	Xs = ['ABC','ATAC']
	nX = 2
	nMethods = 3

summaryFile = pd.read_csv(wdfiles+'expression_prediction_stats_190719.csv', sep = '\t')

medianError = np.zeros(shape = (nX, nMethods, 4)) # [abc,atac], [multivariate,randomforest], cutoffs
medianR2 = np.zeros(shape = (nX, nMethods, 4))
pctGenes = np.zeros(shape = (nX, nMethods, 4))
numStrongGenes = np.zeros(shape = (nX, nMethods, 4))

Methods = ['Multivariate','Random Forest','Lasso']
VarNames = ['medianError','medianR2','pctGenes','numStrongGenes','strongGenes']
#strongGenesVar = [strongGenesM,strongGenesF]
Vars = [medianError,medianR2,pctGenes,numStrongGenes]
for ix in range(len(Xs)):
	X = Xs[ix]
	for imethod in range(len(Methods)):
		Method = Methods[imethod]
		for ivar in range(len(VarNames)):
			if ivar!=4:
				VarName = VarNames[ivar]
				Var = Vars[ivar]
				Var[ix,imethod,:] = np.array( summaryFile[['1','2','3','4']][summaryFile['X']==X][summaryFile['Method']==Method][summaryFile['VarName']==VarName] )
			else:
				continue

errorSum = np.load(wdvars+'Sum/medianError.npy')
r2Sum = np.load(wdvars+'Sum/medianR2.npy')
nStrongGenesSum = np.load(wdvars+'Sum/nStrongGenes.npy')
error = np.amin(medianError,axis=2)
r2 = np.amax(medianR2,axis=2)
nSGenes = np.amax(numStrongGenes,axis=2)

######### Sum Plots #########
plt.clf()
plt.plot(0,100*errorSum[1,0], '^', color = '#800000', markersize = 15) #label = 'ATAC Sum Multivariate', 
plt.plot(1,100*errorSum[1,1], 'o', color = '#800000', label = 'ATAC Sum', markersize = 15)
plt.plot(2,100*errorSum[1,2], '+', color = '#800000', mew=5, markersize = 15) #label = 'ATAC Sum Lasso', 

plt.plot(0,100*errorSum[3,0], '^', color = '#f58231', markersize = 15) #label = 'ATAC Sig Sum Multivariate', 
plt.plot(1,100*errorSum[3,1], 'o', color = '#f58231', label = 'ATAC Sum of Significant', markersize = 15)
plt.plot(2,100*errorSum[3,2], '+', color = '#f58231', mew=5, markersize = 15) #label = 'ATAC Sig Sum Lasso', 

plt.plot(0,100*error[1,0], '^', color = '#e6194B', markersize = 15)
plt.plot(1,100*error[1,1], 'o', color = '#e6194B', label = 'ATAC All Peaks', markersize = 15)
plt.plot(2,100*error[1,2], '+', color = '#e6194B', mew=5, markersize = 15)

plt.plot(0,100*errorSum[0,0], '^', color = '#000075', markersize = 15) # label = 'ABC Sum Multivariate', 
plt.plot(1,100*errorSum[0,1], 'o', color = '#000075', label = 'ABC Sum', markersize = 15)
plt.plot(2,100*errorSum[0,2], '+', color = '#000075', mew=5, markersize = 15) #label = 'ABC Sum Lasso', 

plt.plot(0,100*errorSum[2,0], '^', color = '#42d4f4', markersize = 15) #label = 'ABC Sig Sum Multivariate', 
plt.plot(1,100*errorSum[2,1], 'o', color = '#42d4f4', label = 'ABC Sum of Significant', markersize = 15)
plt.plot(2,100*errorSum[2,2], '+', color = '#42d4f4', mew=5, markersize = 15) #label = 'ABC Sig Sum Lasso', 
                                                   
plt.plot(0,100*error[0,0], '^', color = '#4363d8', markersize = 15)
plt.plot(1,100*error[0,1], 'o', color = '#4363d8', label = 'ABC All Peaks', markersize = 15)
plt.plot(2,100*error[0,2], '+', color = '#4363d8', mew=5, markersize = 15)

plt.title('Median Error of Gene Expression Predictions', fontsize=15)
plt.ylabel('Median Error (%) of Genes with Correlations')
plt.ylim([10,37])
plt.legend(loc='lower left',fontsize = 12)
plt.xlim([-0.5,2.5])
plt.xticks([0,1,2], ['Multivariate','Random Forest','Lasso'])
plt.grid(True)
plt.savefig(wdfigs+'summary/medianError_all.pdf')
plt.show()


plt.clf()
plt.plot(0,r2Sum[1,0], '^', color = '#800000', markersize = 15) #label = 'ATAC Sum Multivariate', 
plt.plot(1,r2Sum[1,1], 'o', color = '#800000', label = 'ATAC Sum', markersize = 15)
plt.plot(2,r2Sum[1,2], '+', color = '#800000', mew=5, markersize = 15) #label = 'ATAC Sum Lasso', 

plt.plot(0,r2Sum[3,0], '^', color = '#f58231', markersize = 15) #label = 'ATAC Sig Sum Multivariate', 
plt.plot(1,r2Sum[3,1], 'o', color = '#f58231', label = 'ATAC Sum of Significant', markersize = 15)
plt.plot(2,r2Sum[3,2], '+', color = '#f58231', mew=5, markersize = 15) #label = 'ATAC Sig Sum Lasso', 

plt.plot(0,r2[1,0], '^', color = '#e6194B', markersize = 15)
plt.plot(1,r2[1,1], 'o', color = '#e6194B', label = 'ATAC All Peaks', markersize = 15)
plt.plot(2,r2[1,2], '+', color = '#e6194B', mew=5, markersize = 15)

plt.plot(0,r2Sum[0,0], '^', color = '#000075', markersize = 15) # label = 'ABC Sum Multivariate', 
plt.plot(1,r2Sum[0,1], 'o', color = '#000075', label = 'ABC Sum', markersize = 15)
plt.plot(2,r2Sum[0,2], '+', color = '#000075', mew=5, markersize = 15) #label = 'ABC Sum Lasso', 

plt.plot(0,r2Sum[2,0], '^', color = '#42d4f4', markersize = 15) #label = 'ABC Sig Sum Multivariate', 
plt.plot(1,r2Sum[2,1], 'o', color = '#42d4f4', label = 'ABC Sum of Significant', markersize = 15)
plt.plot(2,r2Sum[2,2], '+', color = '#42d4f4', mew=5, markersize = 15) #label = 'ABC Sig Sum Lasso', 
                                            
plt.plot(0,r2[0,0], '^', color = '#4363d8', markersize = 15)
plt.plot(1,r2[0,1], 'o', color = '#4363d8', label = 'ABC All Peaks', markersize = 15)
plt.plot(2,r2[0,2], '+', color = '#4363d8', mew=5, markersize = 15)

plt.plot([-10,10],[0,0],'k-')
plt.title('Median R2 of Gene Expression Predictions', fontsize=15)
plt.ylabel('Median R2 of Genes with Correlations')
plt.ylim([-0.5,1])
plt.legend(loc='upper left',fontsize = 12)
plt.xlim([-0.5,2.5])
plt.xticks([0,1,2], ['Multivariate','Random Forest','Lasso'])
plt.grid(True)
plt.savefig(wdfigs+'summary/medianR2_all.pdf')
plt.show()


plt.clf()
plt.plot(0,nStrongGenesSum[1,0], '^', color = '#800000', markersize = 15) #label = 'ATAC Sum Multivariate',
plt.plot(1,nStrongGenesSum[1,1], 'o', color = '#800000', label = 'ATAC Sum', markersize = 15)
plt.plot(2,nStrongGenesSum[1,2], '+', color = '#800000', mew=5, markersize = 15) #label = 'ATAC Sum Lasso',

plt.plot(0,nStrongGenesSum[3,0], '^', color = '#f58231', markersize = 15) #label = 'ATAC Sig Sum Multivariate',
plt.plot(1,nStrongGenesSum[3,1], 'o', color = '#f58231', label = 'ATAC Sum of Significant', markersize = 15)
plt.plot(2,nStrongGenesSum[3,2], '+', color = '#f58231', mew=5, markersize = 15) #label = 'ATAC Sig Sum Lasso',

plt.plot(0,nSGenes[1,0], '^', color = '#e6194B', markersize = 15)
plt.plot(1,nSGenes[1,1], 'o', color = '#e6194B', label = 'ATAC All Peaks', markersize = 15)
plt.plot(2,nSGenes[1,2], '+', color = '#e6194B', mew=5, markersize = 15)

plt.plot(0,nStrongGenesSum[0,0], '^', color = '#000075', markersize = 15) # label = 'ABC Sum Multivariate',
plt.plot(1,nStrongGenesSum[0,1], 'o', color = '#000075', label = 'ABC Sum', markersize = 15)
plt.plot(2,nStrongGenesSum[0,2], '+', color = '#000075', mew=5, markersize = 15) #label = 'ABC Sum Lasso',

plt.plot(0,nStrongGenesSum[2,0], '^', color = '#42d4f4', markersize = 15) #label = 'ABC Sig Sum Multivariate',
plt.plot(1,nStrongGenesSum[2,1], 'o', color = '#42d4f4', label = 'ABC Sum of Significant', markersize = 15)
plt.plot(2,nStrongGenesSum[2,2], '+', color = '#42d4f4', mew=5, markersize = 15) #label = 'ABC Sig Sum Lasso',

plt.plot(0,nSGenes[0,0], '^', color = '#4363d8', markersize = 15)
plt.plot(1,nSGenes[0,1], 'o', color = '#4363d8', label = 'ABC All Peaks', markersize = 15)
plt.plot(2,nSGenes[0,2], '+', color = '#4363d8', mew=5, markersize = 15)

plt.plot([-10,10],[0,0],'k-')
plt.title('Number of Genes with R2 > 0.4: Summed ABC', fontsize=15)
plt.ylabel('Number of Genes')
plt.ylim([0,40])
plt.legend(loc='upper left',fontsize = 12)
plt.xlim([-0.5,2.5])
plt.xticks([0,1,2], ['Multivariate','Random Forest','Lasso'])
plt.grid(True)
plt.savefig(wdfigs+'summary/numStrongGenes_all.pdf')
plt.show()
exit()

###########################################################
# Plots by cutoff
###########################################################


plt.clf()
if PC:
	plt.plot(np.arange(1,5), 100*medianError[3,0,:], '^-', color = 'darkorange', label = 'ATAC PC Multivariate', markersize = 10)
	plt.plot(np.arange(1,5), 100*medianError[3,1,:], 'o-', color = 'darkorange', label = 'ATAC PC Random Forest', markersize = 10)
	plt.plot(np.arange(1,5), 100*medianError[2,0,:], '^-', color = 'darkturquoise', label = 'ABC PC Multivariate', markersize = 10)
	plt.plot(np.arange(1,5), 100*medianError[2,1,:], 'o-', color = 'darkturquoise', label = 'ABC PC Random Forest', markersize = 10)
plt.plot(np.arange(1,5), 100*medianError[1,0,:], 'r^-', label = 'ATAC Multivariate', markersize = 10)
plt.plot(np.arange(1,5), 100*medianError[1,2,:], 'r+-', mew=5, ms=10, label = 'ATAC Lasso', markersize = 10)
plt.plot(np.arange(1,5), 100*medianError[1,1,:], 'ro-', label = 'ATAC Random Forest', markersize = 10)
plt.plot(np.arange(1,5), 100*medianError[0,0,:], 'b^-', label = 'ABC Multivariate', markersize = 10)
plt.plot(np.arange(1,5), 100*medianError[0,2,:], 'b+-', mew=5, ms=10, label = 'ABC Lasso', markersize = 10)
plt.plot(np.arange(1,5), 100*medianError[0,1,:], 'bo-', label = 'ABC Random Forest', markersize = 10)
plt.title('Median Error of Gene Expression Predictions', fontsize=15)
plt.xlabel('Number of Peaks Used to Predict')
plt.ylabel('Median Error (%) of Genes with Correlations')
plt.ylim([10,35])
plt.xlim([0.5,4.5])
plt.xticks([1,2,3,4])
plt.legend(loc='lower left',fontsize = 12)
plt.grid(True)
plt.savefig(wdfigs+'summary/medianError_with_Lasso.pdf')
plt.show()

plt.clf()
if PC:
	plt.plot(np.arange(1,5), medianR2[3,0,:], '^-', color = 'darkorange', label = 'ATAC PC Multivariate', markersize = 10)
	plt.plot(np.arange(1,5), medianR2[3,1,:], 'o-', color = 'darkorange', label = 'ATAC PC Random Forest', markersize = 10)
	plt.plot(np.arange(1,5), medianR2[2,0,:], '^-', color = 'darkturquoise', label = 'ABC PC Multivariate', markersize = 10)
	plt.plot(np.arange(1,5), medianR2[2,1,:], 'o-', color = 'darkturquoise', label = 'ABC PC Random Forest', markersize = 10)
plt.plot(np.arange(1,5), medianR2[1,0,:], 'r^-', label = 'ATAC Multivariate', markersize = 10)
plt.plot(np.arange(1,5), medianR2[1,2,:], 'r+-', mew=5, ms=10, label = 'ATAC Lasso', markersize = 10)
plt.plot(np.arange(1,5), medianR2[1,1,:], 'ro-', label = 'ATAC Random Forest', markersize = 10)
plt.plot(np.arange(1,5), medianR2[0,0,:], 'b^-', label = 'ABC Multivariate', markersize = 10)
plt.plot(np.arange(1,5), medianR2[0,2,:], 'b+-', mew=5, ms=10, label = 'ABC Lasso', markersize = 10)
plt.plot(np.arange(1,5), medianR2[0,1,:], 'bo-', label = 'ABC Random Forest', markersize = 10)
plt.plot([0,5],[0,0],'k-')
plt.title('Median R2 of Gene Expression Predictions', fontsize=15)
plt.xlabel('Number of Peaks Used to Predict')
plt.ylabel('Median R2 of Genes with Correlations')
plt.ylim([-0.5,1])
plt.xlim([0.5,4.5])
plt.xticks([1,2,3,4])
plt.legend(loc='upper left',fontsize=12)
plt.grid(True)
plt.savefig(wdfigs+'summary/medianR2_with_Lasso.pdf')
plt.show()

plt.clf()
if PC:
	plt.plot(np.arange(1,5), numStrongGenes[3,0,:], '^-', color = 'darkorange', label = 'ATAC PC Multivariate', markersize = 10)
	plt.plot(np.arange(1,5), numStrongGenes[3,1,:], 'o-', color = 'darkorange', label = 'ATAC PC Random Forest', markersize = 10)
	plt.plot(np.arange(1,5), numStrongGenes[2,0,:], '^-', color = 'darkturquoise', label = 'ABC PC Multivariate', markersize = 10)
	plt.plot(np.arange(1,5), numStrongGenes[2,1,:], 'o-', color = 'darkturquoise', label = 'ABC PC Random Forest', markersize = 10)
plt.plot(np.arange(1,5), numStrongGenes[1,0,:], 'r^-', label = 'ATAC Multivariate', markersize = 10)
plt.plot(np.arange(1,5), numStrongGenes[1,2,:], 'r+-', mew=5, ms=10, label = 'ATAC Lasso', markersize = 10)
plt.plot(np.arange(1,5), numStrongGenes[1,1,:], 'ro-', label = 'ATAC Random Forest', markersize = 10)
plt.plot(np.arange(1,5), numStrongGenes[0,0,:], 'b^-', label = 'ABC Multivariate', markersize = 10)
plt.plot(np.arange(1,5), numStrongGenes[0,2,:], 'b+-', mew=5, ms=10, label = 'ABC Lasso', markersize = 10)
plt.plot(np.arange(1,5), numStrongGenes[0,1,:], 'bo-', label = 'ABC Random Forest', markersize = 10)
plt.plot([0,5],[0,0],'k-')
plt.title('Number of Genes with R2 > 0.4', fontsize=15)
plt.xlabel('Number of Peaks Used to Predict')
plt.ylabel('Number of Genes')
plt.ylim([0,40])
plt.xlim([0.5,4.5])
plt.xticks([1,2,3,4])
plt.legend(loc = 'upper left',fontsize=12)
plt.grid(True)
plt.savefig(wdfigs+'summary/numStrongGenes_with_Lasso.pdf')
plt.show()

numGenes = (pctGenes*94)[:,0,0]
plt.clf()
if PC:
	plt.plot(np.arange(1,5), 100*numStrongGenes[3,0,:]/numGenes[3], '^-', color = 'darkorange', label = 'ATAC PC Multivariate, '+str(int(numGenes[3]))+' Genes', markersize = 10)
	plt.plot(np.arange(1,5), 100*numStrongGenes[3,1,:]/numGenes[3], 'o-', color = 'darkorange', label = 'ATAC PC Random Forest, '+str(int(numGenes[3]))+' Genes', markersize = 10)
	plt.plot(np.arange(1,5), 100*numStrongGenes[2,0,:]/numGenes[2], '^-', color = 'darkturquoise', label = 'ABC PC Multivariate, '+str(int(numGenes[2]))+' Genes', markersize = 10)
	plt.plot(np.arange(1,5), 100*numStrongGenes[2,1,:]/numGenes[2], 'o-', color = 'darkturquoise', label = 'ABC PC Random Forest, '+str(int(numGenes[2]))+' Genes', markersize = 10)
plt.plot(np.arange(1,5), 100*numStrongGenes[1,0,:]/numGenes[1], 'r^-', label = 'ATAC Multivariate, '+str(int(numGenes[1]))+' Genes', markersize = 10)
plt.plot(np.arange(1,5), 100*numStrongGenes[1,2,:]/numGenes[1], 'r+-', mew=5, ms=10, label = 'ATAC Lasso, '+str(int(numGenes[1]))+' Genes', markersize = 10)
plt.plot(np.arange(1,5), 100*numStrongGenes[1,1,:]/numGenes[1], 'ro-', label = 'ATAC Random Forest, '+str(int(numGenes[1]))+' Genes', markersize = 10)
plt.plot(np.arange(1,5), 100*numStrongGenes[0,0,:]/numGenes[0], 'b^-', label = 'ABC Multivariate, '+str(int(numGenes[0]))+' Genes', markersize = 10)
plt.plot(np.arange(1,5), 100*numStrongGenes[0,2,:]/numGenes[0], 'b+-', mew=5, ms=10, label = 'ABC Lasso, '+str(int(numGenes[0]))+' Genes', markersize = 10)
plt.plot(np.arange(1,5), 100*numStrongGenes[0,1,:]/numGenes[0], 'bo-', label = 'ABC Random Forest, '+str(int(numGenes[0]))+' Genes', markersize = 10)
plt.plot([0,5],[0,0],'k-')
plt.title('Percent of Correlated Genes with R2 > 0.4', fontsize=15)
plt.xlabel('Number of Peaks Used to Predict')
plt.ylabel('Percent of Genes with P Values < 1')
plt.ylim([0,100])
plt.xlim([0.5,4.5])
plt.xticks([1,2,3,4])
plt.legend(loc = 'upper left',fontsize=12)
plt.grid(True)
plt.savefig(wdfigs+'summary/pctStrongGenes_with_Lasso.pdf')
plt.show()

plt.clf()
if PC:
	plt.plot(np.arange(1,5), 100*pctGenes[2,0,:], '-', color = 'darkturquoise', label = 'ABC PC', linewidth = 5)
	plt.plot(np.arange(1,5), 100*pctGenes[3,0,:], '-', color = 'darkorange', label = 'ATAC PC', linewidth = 5)
plt.plot(np.arange(1,5), 100*pctGenes[0,0,:], 'b-', label = 'ABC', linewidth = 5)
plt.plot(np.arange(1,5), 100*pctGenes[1,0,:], 'r-', label = 'ATAC', linewidth = 5)
plt.plot([0,5],[0,0],'k-')
plt.title('Percent of Genes with P Values < 1', fontsize=15)
plt.xlabel('Number of Peaks Used to Predict')
plt.ylabel('Percent of Genes with P Values < 1')
plt.ylim([0,100])
plt.xlim([0.5,4.5])
plt.xticks([1,2,3,4])
plt.legend(loc = 'lower left',fontsize=12)
plt.grid(True)
plt.savefig(wdfigs+'summary/pctGenes_with_Lasso.pdf')
plt.show()


