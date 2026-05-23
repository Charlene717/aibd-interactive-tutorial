"""
02_scgpt_zero_shot.py
=====================
Zero-shot cell embedding with scGPT, followed by UMAP + Leiden clustering.
Corresponds to: Ch06 (single-cell foundation models).

Requirements
------------
- scgpt
- scanpy
- anndata
- scgpt pretrained checkpoint (download separately):
    https://github.com/bowang-lab/scGPT  ->  whole-human pretrained

Hardware: ~16 GB RAM, GPU optional (greatly faster).
Runtime: a few minutes on GPU for a few k cells.

Usage
-----
$ python 02_scgpt_zero_shot.py path/to/sample.h5ad path/to/scgpt_whole_human
"""
import sys
import scanpy as sc
import scgpt


def main(h5ad_path: str, model_dir: str):
    adata = sc.read_h5ad(h5ad_path)

    # Standard preprocessing
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=True)

    # Embed with scGPT (zero-shot)
    emb = scgpt.tasks.embed_data(
        adata,
        model_dir=model_dir,
        gene_col="feature_name",   # or "var_names" depending on adata
        cell_type_key=None,
        max_length=1200,
        batch_size=64,
        return_new_adata=True,
    )

    # Neighbors + UMAP + Leiden
    sc.pp.neighbors(emb, use_rep="X_scGPT", n_neighbors=15)
    sc.tl.umap(emb)
    sc.tl.leiden(emb, resolution=0.6)

    # Save & summarize
    out = h5ad_path.replace(".h5ad", "_scgpt.h5ad")
    emb.write_h5ad(out)
    print(f"Saved {out}")
    print(f"Cells: {emb.n_obs}, Clusters: {emb.obs['leiden'].nunique()}")
    print(emb.obs["leiden"].value_counts().head(10))


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
