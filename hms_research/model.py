#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 18:46:00 2023

@author: abhijitdey
"""

import pandas as pd
import scanpy as sc

sc.settings.verbosity = 3             # verbosity: errors (0), warnings (1), info (2), hints (3)
sc.logging.print_header()
sc.settings.set_figure_params(dpi=80, facecolor='white')

results_file = 'write/pbmc3k.h5ad'  # the file that will store the analysis results

# Read in the count matrix into an AnnData object, which holds many slots for 
# annotations and different representations of the data. It also comes with its own 
# HDF5-based file format: .h5ad.
adata = sc.read_10x_mtx(
    'data/filtered_gene_bc_matrices/hg19/',  # the directory with the `.mtx` file
    var_names='gene_symbols',                # use gene symbols for the variable names (variables-axis index)
    cache=True)      

adata.var_names_make_unique()  # this is unnecessary if using `var_names='gene_ids'` in `sc.read_10x_mtx`


adata

# Preprocessing

# Show those genes that yield the highest fraction of counts in each single cell, across all cells.

sc.pl.highest_expr_genes(adata, n_top=20, )

#Basic filtering:
#filtered out 19024 genes that are detected in less than 3 cells

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3) 

#Let's assemble some information about mitochondrial genes, which are important for quality control.

#High proportions are indicative of poor-quality cells (Islam et al. 2014; Ilicic et al. 2016), 
# possibly because of loss of cytoplasmic RNA from perforated cells. The reasoning is that 
# mitochondria are larger than individual transcript molecules and less likely to escape 
# through tears in the cell membrane.
# With pp.calculate_qc_metrics, we can compute many metrics very efficiently.

adata.var['mt'] = adata.var_names.str.startswith('MT-')  # annotate the group of mitochondrial genes as 'mt'
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

# A violin plot of some of the computed quality measures:
# the number of genes expressed in the count matrix
# the total counts per cell
# the percentage of counts in mitochondrial genes

sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'],
             jitter=0.4, multi_panel=True)

# Remove cells that have too many mitochondrial genes expressed or too many total counts:
    
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt')
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts')

# Actually do the filtering by slicing the AnnData object
adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]

#Total-count normalize (library-size correct) the data matrix  𝐗
# X to 10,000 reads per cell, so that counts become comparable among cells
sc.pp.normalize_total(adata, target_sum=1e4)

#Logarithmize the data
sc.pp.log1p(adata)

# Identify highly-variable genes
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
sc.pl.highly_variable_genes(adata)

#Set the .raw attribute of the AnnData object to the normalized and logarithmized 
# raw gene expression for later use in differential testing and visualizations of gene expression. 
# This simply freezes the state of the AnnData object
adata.raw = adata

#Actually do the filtering
adata = adata[:, adata.var.highly_variable]

#Regress out effects of total counts per cell and the percentage of mitochondrial genes expressed. 
# Scale the data to unit variance.
sc.pp.regress_out(adata, ['total_counts', 'pct_counts_mt'])

#Scale each gene to unit variance. Clip values exceeding standard deviation 10.
sc.pp.scale(adata, max_value=10)

#Principal component analysis¶
#Reduce the dimensionality of the data by running principal component analysis (PCA),
# which reveals the main axes of variation and denoises the data.
sc.tl.pca(adata, svd_solver='arpack')

#We can make a scatter plot in the PCA coordinates, but we will not use that later on.
sc.pl.pca(adata, color='CST3')

#Let us inspect the contribution of single PCs to the total variance in the data. 
#This gives us information about how many PCs we should consider in order to compute 
# the neighborhood relations of cells, e.g. used in the clustering function  
# sc.tl.louvain() or tSNE sc.tl.tsne(). In our experience, often a rough estimate of 
# the number of PCs does fine.
sc.pl.pca_variance_ratio(adata, log=True)

#Save the result.
adata.write(results_file)

adata

#Computing the neighborhood graph
#Let us compute the neighborhood graph of cells using the PCA representation 
#of the data matrix. You might simply use default values here. For the sake of reproducing 
#Seurat’s results, let’s take the following values
sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)

#Embedding the neighborhood graph
#sc.tl.paga(adata)
#sc.pl.paga(adata, plot=False)  # remove `plot=False` if you want to see the coarse-grained graph
#sc.tl.umap(adata, init_pos='paga')

sc.tl.umap(adata)

sc.pl.umap(adata, color=['CST3', 'NKG7', 'PPBP'])
#As we set the .raw attribute of adata, the previous plots showed the “raw” (normalized, 
#logarithmized, but uncorrected) gene expression. You can also plot the scaled and corrected 
#gene expression by explicitly stating that you don’t want to use .raw.
sc.pl.umap(adata, color=['CST3', 'NKG7', 'PPBP'], use_raw=False)

#Clustering the neighborhood graph

#Leiden graph-clustering method
sc.tl.leiden(adata)

sc.pl.umap(adata, color=['leiden', 'CST3', 'NKG7'])

adata.write(results_file)

#Finding marker genes
sc.tl.rank_genes_groups(adata, 'leiden', method='t-test')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

sc.settings.verbosity = 2  # reduce the verbosity

#The result of a Wilcoxon rank-sum (Mann-Whitney-U) test is very similar.
sc.tl.rank_genes_groups(adata, 'leiden', method='wilcoxon')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

adata.write(results_file)
#As an alternative, let us rank genes using logistic regression. For instance, 
#this has been suggested by Natranos et al. (2018). The essential difference is that
# here, we use a multi-variate appraoch whereas conventional differential tests are 
#uni-variate. Clark et al. (2014) has more details.
sc.tl.rank_genes_groups(adata, 'leiden', method='logreg')
sc.pl.rank_genes_groups(adata, n_genes=25, sharey=False)

#With the exceptions of IL7R, which is only found by the t-test and FCER1A, 
#which is only found by the other two appraoches, all marker genes are recovered in all approaches.

#Louvain Group	Markers	Cell Type
#0	IL7R	CD4 T cells
#1	CD14, LYZ	CD14+ Monocytes
#2	MS4A1	B cells
#3	CD8A	CD8 T cells
#4	GNLY, NKG7	NK cells
#5	FCGR3A, MS4A7	FCGR3A+ Monocytes
#6	FCER1A, CST3	Dendritic Cells
#7	PPBP	Megakaryocytes

#Let us also define a list of marker genes for later reference.
marker_genes = ['IL7R', 'CD79A', 'MS4A1', 'CD8A', 'CD8B', 'LYZ', 'CD14',
                'LGALS3', 'S100A8', 'GNLY', 'NKG7', 'KLRB1',
                'FCGR3A', 'MS4A7', 'FCER1A', 'CST3', 'PPBP']

adata = sc.read(results_file)

#Show the 10 top ranked genes per cluster 0, 1, …, 7 in a dataframe.

pd.DataFrame(adata.uns['rank_genes_groups']['names']).head(5)

#Get a table with the scores and groups.
result = adata.uns['rank_genes_groups']
groups = result['names'].dtype.names
pd.DataFrame(
    {group + '_' + key[:1]: result[key][group]
    for group in groups for key in ['names', 'pvals']}).head(5)
 
#Compare to a single cluster:
#sc.tl.rank_genes_groups(adata, 'leiden', groups=['0'], reference='1', method='wilcoxon')
#sc.pl.rank_genes_groups(adata, groups=['0'], n_genes=20)                                                    

#If we want a more detailed view for a certain group, use sc.pl.rank_genes_groups_violin.
sc.pl.rank_genes_groups_violin(adata, groups='0', n_genes=8)

#Reload the object with the computed differential expression (i.e. DE via a 
#comparison with the rest of the groups):
    
adata = sc.read(results_file)
sc.pl.rank_genes_groups_violin(adata, groups='0', n_genes=8)

#If you want to compare a certain gene across groups, use the following.
sc.pl.violin(adata, ['CST3', 'NKG7', 'PPBP'], groupby='leiden')

#Actually mark the cell types.
new_cluster_names = [
    'CD4 T', 'CD14 Monocytes',
    'B', 'CD8 T',
    'NK', 'FCGR3A Monocytes',
    'Dendritic', 'Megakaryocytes']
adata.rename_categories('leiden', new_cluster_names)

sc.pl.umap(adata, color='leiden', legend_loc='on data', title='', frameon=False, save='.pdf')

#Now that we annotated the cell types, let us visualize the marker genes.

sc.pl.dotplot(adata, marker_genes, groupby='leiden');

#There is also a very compact violin plot.
sc.pl.stacked_violin(adata, marker_genes, groupby='leiden', rotation=90);

#During the course of this analysis, the AnnData accumlated the following annotations.
adata
adata.write(results_file, compression='gzip')  # `compression='gzip'` saves disk space, but slows down writing and subsequent reading

adata.raw.to_adata().write('./write/pbmc3k_withoutX1.h5ad')

# Export single fields of the annotation of observations
# adata.obs[['n_counts', 'louvain_groups']].to_csv(
#     './write/pbmc3k_corrected_louvain_groups.csv')

# Export single columns of the multidimensional annotation
# adata.obsm.to_df()[['X_pca1', 'X_pca2']].to_csv(
#     './write/pbmc3k_corrected_X_pca.csv')

# Or export everything except the data using `.write_csvs`.
# Set `skip_data=False` if you also want to export the data.
# adata.write_csvs(results_file[:-5], )

  
















    
    
    
    
    
    
    
    
    
    