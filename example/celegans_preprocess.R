# need to install gdal before this: brew install gdal

# if (!requireNamespace("BiocManager", quietly = TRUE))
#   install.packages("BiocManager")
# BiocManager::install()

# BiocManager::install(c('BiocGenerics', 'DelayedArray', 'DelayedMatrixStats',
#                       'limma', 'lme4', 'S4Vectors', 'SingleCellExperiment',
#                       'SummarizedExperiment', 'batchelor', 'HDF5Array',
#                       'terra', 'ggrastr'))

# install.packages("devtools")
# devtools::install_github('cole-trapnell-lab/monocle3')
# install.packages("scales")
library(monocle3)

expression_matrix <- readRDS(url("https://depts.washington.edu:/trapnell-lab/software/monocle3/celegans/data/packer_embryo_expression.rds"))
cell_metadata <- readRDS(url("https://depts.washington.edu:/trapnell-lab/software/monocle3/celegans/data/packer_embryo_colData.rds"))
gene_annotation <- readRDS(url("https://depts.washington.edu:/trapnell-lab/software/monocle3/celegans/data/packer_embryo_rowData.rds"))

cds <- new_cell_data_set(expression_matrix,
                         cell_metadata = cell_metadata,
                         gene_metadata = gene_annotation)

cds <- preprocess_cds(cds, num_dim = 50)
cds <- align_cds(cds, alignment_group = "batch", residual_model_formula_str = "~ bg.300.loading + bg.400.loading + bg.500.1.loading + bg.500.2.loading + bg.r17.loading + bg.b01.loading + bg.b02.loading")

get_citations(cds)
X <- as.matrix(SingleCellExperiment::reducedDims(cds)[["Aligned"]])
embryo.time <- cell_metadata$embryo.time
cell.type <- cell_metadata$cell.type
size_factor <- cell_metadata$Size_Factor
unique(cell_metadata$lineage)
                           
library(scales)
library(ggplot2)
colormap <- scales::hue_pal()(27)
colormap
scales::show_col(colormap)

write.csv(X, "celegans_proccessed.csv")
write.csv(data.frame("embryo_time" = embryo.time, "cell_type" = cell.type, "size_factor" = size_factor), 
          "celegans_metadata.csv")
