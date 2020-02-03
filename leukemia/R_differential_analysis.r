#! /usr/local/bin/Rscript
library("limma")
library("edgeR")

exit <- function() {
  .Internal(.invokeRestart(list(NULL, NULL), NULL))
}


wd = '/pbld/mcg/lillianpetersen/ABC/'

# 1 = ETV6-RUNX1
# 2 = DUX4
# 3 = Hyperdiploid
# 4 = PAX5alt
# 5 = Ph-like
# 6 = PAX5
# 7 = ZNF384
# 8 = Other
samples = c('MCG001', 'MCG002', 'MCG003', 'MCG005', 'MCG006', 'MCG009', 'MCG010', 'MCG011', 'MCG012', 'MCG013', 'MCG016', 'MCG017', 'MCG019', 'MCG020', 'MCG023', 'MCG024', 'MCG027', 'MCG028', 'MCG034', 'MCG035', 'MCG036', 'MCG037', 'MCG038', 'MCG039')
typesName = c('PAX5', 'ETV6-RUNX1', 'PAX5alt', 'DUX4', 'ZNF384', 'PAX5alt', 'Hyperdiploid', 'DUX4', 'Hyperdiploid', 'Ph-like', 'Ph-like', 'ETV6-RUNX1', 'Hyperdiploid', 'Hyperdiploid', 'DUX4', 'ETV6-RUNX1', 'Hyperdiploid', 'Hyperdiploid', 'Hyperdiploid', 'ETV6-RUNX1', 'DUX4', 'Other', 'Ph-like', 'Ph-like')
types            = c(6, 1, 4, 2, 7, 4, 3, 2, 3, 5, 5, 1, 3, 3, 2, 1, 3, 3, 3, 1, 2, 8, 5, 5)
ETV6outgroup     = c(0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0)
DUX4outgroup     = c(0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)
HDoutgroup       = c(0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0) # hyperdiploid
PAX5altOutgroup  = c(0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
PhlikeOutgroup   = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1)
PAX5outgroup     = c(1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
ZNFoutgroup      = c(0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
OtherOutgroup    = c(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0)
#typeList = list(ETV6RUNX1 = ETV6outgroup, DUX4 = DUX4outgroup, Hyperdiploid = HDoutgroup, PAX5 = PAX5outgroup, Phlike = PhlikeOutgroup, Other = OtherOutgroup)
typeList = list(ETV6outgroup, DUX4outgroup, HDoutgroup, PAX5altOutgroup, PhlikeOutgroup, PAX5outgroup, ZNFoutgroup, OtherOutgroup)
typeString = c('ETV6RUNX1','DUX4','Hyperdiploid','PAX5alt','Phlike','PAX5','ZNF384','Other')

for (igroup in seq(length(typeList))){
	groupArray = typeList[[igroup]]
	print(typeString[igroup])
	
	gene_file = paste(wd,'data/',sep="")
	groups = types #16
	nSamples = 24
	
	rawdata = read.delim( paste(gene_file,"raw_gene_counts_B_ALL.txt",sep=""),header=FALSE) # edit input file name
	if (typeString[igroup]=='PAX5alt'){
		rawdata = subset(rawdata, select = -c(V2))
		groupArray = c(0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
		nSamples = 23
		}
	
	y = DGEList(counts=rawdata[,2:(nSamples+1)],genes=rawdata[,1],group=groupArray)
	keep = rowSums(y$counts>0.1)>=(nSamples/2) #half the samples have rpkm>0.1
	y = y[keep,] # keep only high enough genes
	y = DGEList(counts=y$counts,genes=y$genes)
	
	group = factor(groupArray) # outgroup one subtype
	design = model.matrix(~group)
	
	# estimate dispersion
	y = estimateGLMCommonDisp(y,design,verbose=TRUE)
	y = estimateGLMTrendedDisp(y, design)
	y = estimateGLMTagwiseDisp(y, design)
	#plotBCV(y)
	
	fit = glmQLFit(y,design)
	qlf = glmQLFTest(fit,coef=2)
	topTags(qlf)
	de = decideTestsDGE(qlf, p=0.05)
	summary(de)
	detags = rownames(y)[as.logical(de)]
	
	pdf( paste(wd,"figures/differential_genes_",typeString[igroup],".pdf",sep="") ) 
	plotSmear(qlf, de.tags=detags,
				xlab='Average log(RPM)',
				ylab='log(Fold Change)',
				main=paste(typeString[igroup],' Differential Genes') )
	abline(h=c(-2,2),col='blue')
	p = p.adjust(qlf$table$PValue,method="fdr")
	dev.off()
	
	write.table(cbind(y$genes,qlf$table,p),file=paste(gene_file,'differential_genes/',typeString[igroup],'_differential_genes.txt',sep=""),sep="\t",col.names=TRUE,row.names=FALSE, quote=FALSE) # write to a file
}
