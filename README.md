# AI Meets BioBigData — Interactive Tutorial

**🌐 Live demo / 線上瀏覽**: <https://charlene717.github.io/aibd-interactive-tutorial/>

## 中文簡介

AI × 生醫大數據互動式教學。TCGA / GTEx、生物基礎模型、跨組學 AI 整合等主題，雙語並陳。

## English Description


A bilingual (English / 中文) interactive tutorial folder.

## Quick start

Open `index.html` in any modern browser. No build step required.

## Layout

- `index.html` — Hub page (20 cards across 5 blocks)
- `ch01-...html` → `ch20-...html` — 20 bilingual chapter pages
- `styles.css` — Shared design tokens & components
- `i18n.js` — Bilingual toggle + interactivity (i18n + quiz feedback + accordion + code tabs)
- `references/` — 100+ DOI-linked references
- `aibd-quiz/` — 200-question interactive quiz (4 modes: full, random-20, by-chapter, wrong-review)
- `notebooks/` — Example notebooks (Python + R): ESM-2, scGPT, MedRAG, SHAP+XGBoost, DeepSurv, Calibration

All sibling folders share this folder's `styles.css` and `i18n.js` via relative paths (`../styles.css`).
