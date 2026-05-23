# 範例 Notebooks · Example Notebooks

> Companion code for **AI × 生醫大數據** interactive tutorial.

## 中文說明

這裡收錄了教程 20 章中代表性主題的可執行範例程式碼，採 Python（主）與 R（部分）雙語提供。每個檔案開頭都列出所需套件、預期執行時間與最低硬體需求。

> ⚠️ 部分範例會下載 Hugging Face 上的 gated 模型（如 `MahmoodLab/UNI`），執行前須先到 HF 申請存取權並 `huggingface-cli login`。

## English summary

Runnable example code for representative chapters of the **AI × Biomedical Big Data** tutorial — Python (primary) plus R (selected). Each file lists dependencies, expected runtime, and minimum hardware. Some examples pull gated Hugging Face models (e.g. `MahmoodLab/UNI`) — request access on HF and `huggingface-cli login` first.

## 檔案清單 / File list

| 檔案 | 章節對應 | 主題 |
|---|---|---|
| `01_esm2_embedding.py` / `01_esm2_embedding.R` | Ch01 / Ch11 | ESM-2 蛋白序列嵌入與相似度比較 |
| `02_scgpt_zero_shot.py` | Ch06 | scGPT 零樣本細胞嵌入 + UMAP |
| `03_medrag_minimal.py` | Ch16 | LangChain + FAISS 最小 MedRAG |
| `04_shap_xgboost.py` | Ch18 / Ch03 | XGBoost + SHAP 醫療表型解釋 |
| `05_deepsurv_tcga.py` | Ch14 | DeepSurv 在 TCGA-BRCA 的生存預測 |
| `06_calibration_decision_curve.py` | Ch17 | 校準曲線 + Decision Curve Analysis |

## 環境準備 / Environment

建議用 `uv` 或 `conda` 建立隔離環境：

```bash
# 用 uv
uv venv aibd && source aibd/bin/activate
uv pip install torch torchvision transformers scanpy anndata \
               fair-esm scgpt pycox lifelines dcurves shap xgboost \
               sentence-transformers langchain langchain-community faiss-cpu \
               openslide-python rdkit admet-ai monai

# 或用 conda
mamba env create -f environment.yml
```

R 端最低需求：

```r
install.packages(c("reticulate","Seurat","data.table","SummarizedExperiment",
                   "survival","survminer","pROC","PRROC","rmda","torch","luz"))
BiocManager::install(c("zellkonverter","SingleCellExperiment"))
```

## 執行建議 / Suggested order

1. 先跑 `01_*`（ESM-2）確認 PyTorch + transformers 安裝正常。
2. 跑 `04_shap_xgboost.py` 確認 XGBoost / SHAP 環境。
3. 跑 `06_calibration_*.py` 練習評估工具。
4. 進階：`02_scgpt_*`（需 ~16 GB RAM、權重需單獨下載）、`03_medrag_*`（需 ~5 GB 索引）、`05_deepsurv_*`。

## 引用

> Charlene (2026). *AI × Biomedical Big Data Interactive Tutorial* — Notebooks companion. E:\Charlene\Bioinformatics_Tutorials\AI_Meets_BioBigData.
