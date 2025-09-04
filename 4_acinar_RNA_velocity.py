import scvelo as scv
import scanpy as sc
import harmonypy as hm
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import loompy
import os

# Load loom files-SAP
loom_dir = "/share/home/qlab/qlab_cdj/project/02_project_10x_yh_sap/matrix/loom/SAP"
loom_files = [os.path.join(loom_dir, f) for f in os.listdir(loom_dir) if f.endswith('.loom')]
loom_files = sorted(loom_files)

adata_list = []
for i, loom_file in enumerate(loom_files):
    # Load each loom file
    adata = scv.read_loom(loom_file, validate = False)
    
    # Ensure unique gene names by adding a sample identifier (if needed)
    adata.var_names_make_unique()  # This will handle gene name duplicates
    
    # Add sample ID to `obs` for future reference
    adata.obs['sample_id'] = f'SAP_{i+1}'
    
    # Append to list of adata objects
    adata_list.append(adata)
sap_sample = [f'SAP{i+1}' for i in range(len(adata_list))]

# Load loom files-NC
loom_dir = "/share/home/qlab/qlab_cdj/project/02_project_10x_yh_sap/matrix/loom/NC"
loom_files = [os.path.join(loom_dir, f) for f in os.listdir(loom_dir) if f.endswith('.loom')]
loom_files = sorted(loom_files)

for i, loom_file in enumerate(loom_files):
    # Load each loom file
    adata = scv.read_loom(loom_file, validate = False)
    
    # Ensure unique gene names by adding a sample identifier (if needed)
    adata.var_names_make_unique()  # This will handle gene name duplicates
    
    # Add sample ID to `obs` for future reference
    adata.obs['sample_id'] = f'NC_{i+1}'
    
    # Append to list of adata objects
    adata_list.append(adata)
nc_sample = ["NC1","NC10_PDAC","NC11_PDAC","NC1_PDAC","NC2","NC2_PDAC","NC3_PDAC","NC4_PDAC","NC5_PDAC","NC6_PDAC","NC7_PDAC","NC8_PDAC","NC9_PDAC"]

# Concatenate the datasets using scanpy's concatenate
adata_combined = sc.concat(adata_list, join = "outer", label = "sample_id", keys = (sap_sample + nc_sample))
adata_combined.obs_names
adata_combined.obs["sample_id"]

# subset acinar cells
barcode_file = "./acinar_barcodes_all.txt"
barcodes = pd.read_csv(barcode_file)
adata_subset = adata_combined[adata_combined.obs_names.isin(barcodes["barcode"])]

# add metadata
barcodes_in_adata = barcodes['barcode'].isin(adata_subset.obs_names)
metadata_to_add = barcodes.set_index('barcode').loc[adata_subset.obs_names]
for col in metadata_to_add.columns:
    adata_subset.obs[col] = metadata_to_add[col]

# check metadata
adata_subset.obs_names
adata_subset.obs


# ---- remove batch effect
# Step 1: Normalize and scale data
sc.pp.normalize_total(adata_subset, target_sum=1e4)
sc.pp.log1p(adata_subset)
sc.pp.highly_variable_genes(adata_subset, n_top_genes=3000)
adata_subset.var.highly_variable["PRSS1"] = True
adata_subset.var.highly_variable["CPA2"] = True
adata_subset.var.highly_variable["CTRB1"] = True
adata_subset.var.highly_variable["PRSS2"] = True
adata_subset.var.highly_variable["SYCN"] = True
adata_subset.var.highly_variable["PNLIP"] = True
adata_subset.var.highly_variable["SOX9"] = True
adata_subset.var.highly_variable["KRT19"] = True
adata_subset.var.highly_variable["KRT8"] = True
adata_subset.var.highly_variable["KRT18"] = True
adata_subset.var.highly_variable["SPP1"] = True
adata_subset.var.highly_variable["EPCAM"] = True
adata_subset = adata_subset[:, adata_subset.var.highly_variable]
sc.pp.scale(adata_subset, max_value=10)

# Step 2: Run PCA
sc.tl.pca(adata_subset, svd_solver='arpack')

# Step 3: Apply Harmony
# Harmony expects the batch key to be in `adata.obs`. Here, it's 'sample_id'
harmony_out = hm.run_harmony(adata_subset.obsm['X_pca'], adata_subset.obs, 'sample_id')

# Replace the original PCA coordinates with the batch-corrected ones
adata_subset.obsm['X_pca'] = harmony_out.Z_corr.T

# Step 4: Continue with UMAP or tSNE visualization
sc.pp.neighbors(adata_subset)


# ---- run scVelo
# Proceed with RNA velocity analysis
# scv.pp.filter_and_normalize(adata_subset, min_counts=20, n_top_genes=3000)
# scv.pp.moments(adata_subset, n_pcs=30, n_neighbors=30)
scv.tl.velocity(adata_subset)
scv.tl.velocity_graph(adata_subset)

# UMAP visualization
scv.tl.umap(adata_subset)
scv.tl.louvain(adata_subset,resolution = 0.15) # annotate clusters
adata_subset.obs
scv.pl.velocity_embedding_stream(adata_subset, basis = "umap", color = "louvain", legend_fontsize = 20, palette=['#3180B8', '#DBD99C', '#92CC92', '#986A60'])

# Save the plot
plt.savefig("../../figures/velocity_acinar_ADM.tiff", format = "TIFF")


# ---- add group info
conditions = [
    adata_subset.obs['sample_id'].str.contains('SAP', case=False, na=False),
    adata_subset.obs['sample_id'].str.contains('NC',  case=False, na=False)
]
choices = ['SAP', 'NC']
adata_subset.obs['Group'] = np.select(conditions, choices, default='Unknown')

print(adata_subset.obs['Group'].unique())
scv.pl.velocity_embedding_stream(adata_subset, basis = "umap", color = "Group", legend_fontsize = 20)
plt.savefig("../../figures/velocity_acinar_group.tiff", format = "TIFF")


# ---- sap only
sap_mask = adata_subset.obs["Group"] == "SAP"
adata_sap = adata_subset[sap_mask, :]
scv.pl.velocity_embedding_stream(adata_sap, basis = "umap", color = "Group", legend_fontsize = 20)
plt.savefig("../../figures/velocity_acinar_sap.tiff", format = "TIFF")


# ---- nc only
nc_mask = adata_subset.obs["Group"] == "NC"
adata_nc = adata_subset[nc_mask, :]
scv.pl.velocity_embedding_stream(adata_nc, basis = "umap", color = "Group", legend_fontsize = 20)
plt.savefig("../../figures/velocity_acinar_nc.tiff", format = "TIFF")


# ---- extract clusters
df = adata_subset.obs[['louvain']]
df.to_csv('./RNA_velocity_louvain.txt', sep='\t', index=True, header=False)


# ---- plot gene expression trend
# Acinar
# "PRSS1", "CPA2", "CTRB1", "PRSS2", "SYCN", "PNLIP",

# Ductal
# "SOX9", "KRT19", "KRT8", "KRT18", "SPP1", "EPCAM"

"PRSS1" in adata_subset.var_names
"CPA2" in adata_subset.var_names
"CTRB1" in adata_subset.var_names
"PRSS2" in adata_subset.var_names
"SYCN" in adata_subset.var_names
"PNLIP" in adata_subset.var_names

"SOX9" in adata_subset.var_names
"KRT19" in adata_subset.var_names
"KRT8" in adata_subset.var_names
"KRT18" in adata_subset.var_names
"SPP1" in adata_subset.var_names
"EPCAM" in adata_subset.var_names

scv.pl.velocity(adata_subset, ["PRSS1"], basis = "umap")
plt.savefig("../../figures/PRSS1_velocity_acinar.pdf", format = "pdf")
scv.pl.velocity(adata_subset, ["CPA2"], basis = "umap")
plt.savefig("../../figures/CPA2_velocity_acinar.pdf", format = "pdf")
scv.pl.velocity(adata_subset, ["CTRB1"], basis = "umap")
plt.savefig("../../figures/CTRB1_velocity_acinar.pdf", format = "pdf")
scv.pl.velocity(adata_subset, ["PRSS2"], basis = "umap")
plt.savefig("../../figures/PRSS2_velocity_acinar.pdf", format = "pdf")
scv.pl.velocity(adata_subset, ["SYCN"], basis = "umap")
plt.savefig("../../figures/SYCN_velocity_acinar.pdf", format = "pdf")
scv.pl.velocity(adata_subset, ["PNLIP"], basis = "umap")
plt.savefig("../../figures/PNLIP_velocity_acinar.pdf", format = "pdf")

scv.pl.velocity(adata_subset, ["SOX9"], basis = "umap")
plt.savefig("../../figures/SOX9_velocity_acinar.pdf", format = "pdf")
scv.pl.velocity(adata_subset, ["KRT19"], basis = "umap")
plt.savefig("../../figures/KRT19_velocity_acinar.pdf", format = "pdf")
scv.pl.velocity(adata_subset, ["KRT8"], basis = "umap")
plt.savefig("../../figures/KRT8_velocity_acinar.pdf", format = "pdf")
scv.pl.velocity(adata_subset, ["KRT18"], basis = "umap")
plt.savefig("../../figures/KRT18_velocity_acinar.pdf", format = "pdf")
scv.pl.velocity(adata_subset, ["SPP1"], basis = "umap")
plt.savefig("../../figures/SPP1_velocity_acinar.pdf", format = "pdf")
scv.pl.velocity(adata_subset, ["EPCAM"], basis = "umap")
plt.savefig("../../figures/EPCAM_velocity_acinar.pdf", format = "pdf")
