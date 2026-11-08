# BT4222 Group 15: Personalised Product Search Re-Ranking (JDSearch)

Personalised product search re-ranking on the JDsearch dataset, using LightGBM (BCE and LambdaRank) and a DIN-style attention model, benchmarked against simple baselines and a re-run of Team 3's NCF.

**Module:** BT4222 Mining Web Data for Business Insights, NUS, AY2026/27 Term 1

**Group 15:** Aditya Vijay Kittur, Darren Chan Jun Rong, Fabian Lim Zi Yang, Lee Jia Sheng Linus, Saanvi Tondak, Wong Yao Feng Gareth

---

## The task

Each JD.com user's last search shows about 89 products, and only about 2% of them are engaged with. All of them already match the query, so query relevance alone cannot rank them. We re-rank each user's slate so that the products they click, add to cart or buy reach the top 10.

| | |
|---|---|
| **Data** | 171,728 users with at least one engaged product (2,103 with none are dropped) |
| **One row** | (user, query, candidate product) |
| **Target** | Binary engagement: click, cart or purchase = 1, ignored = 0 |
| **KPI** | Hit@10 (is any engaged product in the top 10?) |
| **Other metrics** | NDCG@10 (linear gain: click 1, cart 2, purchase 3), MRR, P@1 |
| **Split** | By user, 80 / 10 / 10, stratified by history length |
| **Training rows** | All positives + 20 random negatives per slate |
| **Validation / test rows** | Full slates (about 89 products each) |

Validation picks settings and the final model. The test set is scored once, after the final model is chosen.

---

## Results

### Test set (17,176 users, full slates)

| Model | Hit@10 | NDCG@10 | MRR | P@1 |
|---|---|---|---|---|
| Random | 0.2220 | 0.0812 | 0.0991 | 0.0296 |
| Query only | 0.2388 | 0.0931 | 0.1081 | 0.0367 |
| Team 3 NCF (as published) | 0.2538 | 0.1053 | 0.1189 | 0.0454 |
| Team 3 NCF (fixed: P(engaged)) | 0.4427 | 0.2134 | 0.2188 | 0.1124 |
| MF (truncated SVD) | 0.6017 | 0.3191 | 0.3210 | 0.1888 |
| Popularity | 0.6343 | 0.3395 | 0.3362 | 0.1982 |
| Repeat → popularity (best baseline) | 0.6762 | 0.3840 | 0.3875 | 0.2488 |
| Attention hybrid (DIN-style) | 0.7065 | 0.4038 | 0.4035 | 0.2613 |
| LightGBM binary (BCE) | 0.7179 | 0.4121 | 0.4125 | 0.2689 |
| **LightGBM LambdaRank (final)** | **0.7190** | **0.4146** | **0.4145** | **0.2707** |

**Final model: LightGBM LambdaRank**, picked on validation. On test it beats the best baseline (repeat → popularity) by +0.0428 Hit@10 (95% CI +0.0381 to +0.0479) and +0.0306 NDCG@10 (+0.0280 to +0.0331). Its edge over LightGBM BCE on test is proven for NDCG@10 only (+0.0025, CI +0.0010 to +0.0040; Hit@10 +0.0012, CI −0.0019 to +0.0041).

### Validation set (17,172 users, full slates)

| Model | Hit@10 | NDCG@10 | MRR | P@1 |
|---|---|---|---|---|
| Random | 0.2240 | 0.0826 | 0.1008 | 0.0302 |
| Query only | 0.2349 | 0.0912 | 0.1085 | 0.0368 |
| Team 3 NCF (as published) | 0.2533 | 0.1079 | 0.1214 | 0.0477 |
| Team 3 NCF (fixed: P(engaged)) | 0.4507 | 0.2187 | 0.2244 | 0.1165 |
| MF (truncated SVD) | 0.6029 | 0.3196 | 0.3181 | 0.1837 |
| Popularity | 0.6437 | 0.3448 | 0.3388 | 0.1972 |
| Repeat → popularity (best baseline) | 0.6813 | 0.3856 | 0.3869 | 0.2463 |
| Attention hybrid (DIN-style) | 0.7130 | 0.4065 | 0.4054 | 0.2617 |
| LightGBM binary (BCE) | 0.7188 | 0.4146 | 0.4155 | 0.2717 |
| LightGBM LambdaRank | 0.7231 | 0.4168 | 0.4169 | 0.2735 |

Paired comparisons (bootstrap 95% CIs over users) are in each notebook's Results section and in `step4_blend_and_segments.ipynb`.

---

## Repository structure

```
bt4222-group15-jdsearch/
├── README.md                        overview, results, data links, run instructions
├── requirements.txt                 packages beyond Colab defaults
├── step1_data_preprocess.ipynb      cleaning, user split, 65 features
├── step2_baselines.ipynb            non-personalised baselines + Team 3 NCF re-run
├── step3_lightgbm.ipynb             LightGBM binary (BCE)
├── step3_lambdarank.ipynb           LightGBM LambdaRank (final model)
├── step3_attention.ipynb            DIN-style attention hybrid + ablations
└── step4_blend_and_segments.ipynb   blend test, history-length segments, paired test comparisons
```

Data and trained models are too large for GitHub and are linked from Google Drive (see [Data](#data)).

---

## Pipeline

Run the notebooks in this order. Each later step reads the files the earlier steps save.

| Notebook | What it does | Reads | Writes | Colab runtime |
|---|---|---|---|---|
| `step1_data_preprocess.ipynb` | Drops no-engagement slates, shuffles candidates, drops actions logged 1 s before the search, splits users, builds the 65 features | JDSearch_raw | BT4222_artifacts | High-RAM CPU |
| `step2_baselines.ipynb` | Random, query only, popularity, repeat → popularity, MF, and Team 3's NCF re-run on full slates | BT4222_artifacts | baseline_scores.parquet, Team 3 model | GPU |
| `step3_lightgbm.ipynb` | LightGBM binary (BCE), early-stopped on validation NDCG@10 | BT4222_artifacts | a new results folder, lgbm_scores_val.parquet | CPU |
| `step3_lambdarank.ipynb` | Same model with a LambdaRank (listwise) loss; paired comparison with BCE | BT4222_artifacts | a new results folder | CPU |
| `step3_attention.ipynb` | DIN-style target attention over the user's history, joined with the 65 features; ablations and attention-weight analysis | BT4222_artifacts (incl. history_full.parquet) | BT4222_step3_attention (your My Drive) | GPU, High-RAM |
| `step4_blend_and_segments.ipynb` | Trains nothing: reads the frozen models' saved scores; blend test, history-length segments, paired test comparisons | the results folders below | a new results folder | CPU |

### Models

- **LightGBM binary (BCE):** scores each candidate on its own and sorts the slate by probability. Slate awareness comes from the `*_slate_pct` features, which give each candidate its percentile within its own slate.
- **LightGBM LambdaRank:** identical except for the loss, which pushes engaged products above ignored ones within each slate, weighted by the change in NDCG@10.
- **Attention hybrid:** for each candidate, weights the user's past actions by how similar they are to it (category embeddings, action type, time since the action), with a learned "none" slot, then joins the 65 features in one network.

---

## Data

The raw data and all intermediate files are on Google Drive with view-only access.

**Project Drive folder:** [BT4222-Group-15](https://drive.google.com/drive/folders/1YpCRQtsc8dqPi6Q4E9yTfc0wOkpCJ3Z8)

**Raw data:** JDsearch (Liu et al., 2023), https://github.com/rucliujn/JDsearch
Our copy: [JDSearch_raw](https://drive.google.com/drive/folders/1lverviGxkyR-0tBxU9DkUjWE8QhcigMH)

| File | Size | What it is | Made by |
|---|---|---|---|
| product_meta_data.txt | 4.84 GB | Product metadata: name terms, brand, 4-level category, shop | Raw |
| user_behavior_data.txt | 957 MB | Each user's last search, the slate shown with labels, and their history | Raw |

**Preprocessing artifacts:** [BT4222_artifacts](https://drive.google.com/drive/folders/1Yb4AGglIJV4Pdd8Sx6NozG8Klh6uweqH)

| File | Size | What it is | Made by |
|---|---|---|---|
| features.parquet | 302 MB | The 65 features + labels and split for every row used in modelling | step 1 |
| history_full.parquet | 450 MB | Full user histories, read by the attention model | step 1 |
| users.parquet | 3.5 MB | User split and user-level fields | step 1 |
| config.json | 2 KB | Feature list and settings (seed, leak guard, windows) | step 1 |
| team3_seq.npz, team3_meta.parquet | 40 MB, 13 MB | Inputs for the Team 3 NCF re-run | step 1 |
| baseline_scores.parquet | 115 MB | Validation and test scores for every baseline | step 2 |
| lgbm_scores_val.parquet | 23 MB | LightGBM BCE validation scores, the reference for paired tests | step 3 LightGBM |

**Results folders** (frozen 8 Nov models; each holds the model or scores and the result tables):

| Model | Validation run | Test run |
|---|---|---|
| LightGBM BCE | [BT4222_step3_lightgbm_val_08Nov_1257](https://drive.google.com/drive/folders/1EViXhlYkmQimuiweUwvYA9zfKTSSCplk) | [BT4222_step3_lightgbm_test_08Nov_1659](https://drive.google.com/drive/folders/1kwtdu8iMbjC9Onbe_LVxv0PedT10fiMG) |
| LightGBM LambdaRank (final) | [BT4222_step3_lambdarank_val_08Nov_1356](https://drive.google.com/drive/folders/1y1SQ2aQJikO7vesszwMkB9w9PuFhtab9) | [BT4222_step3_lambdarank_test_08Nov_1704](https://drive.google.com/drive/folders/1Ha380qwaIziWaLlDphdnoD3Hln1f-meV) |
| Attention hybrid | [BT4222_step3_attention_val_08Nov](https://drive.google.com/drive/folders/1kbxHt6ku79o8mJaxkRPllmm0bmLRQSLe) | [BT4222_step3_attention_test_08Nov](https://drive.google.com/drive/folders/1IIccC_KDGtqLy2wIv3OGsnBiOu30JOOU) |
| Step 4 (blend, segments, paired tests) | [BT4222_step4_blend_and_segments_08Nov_1808](https://drive.google.com/drive/folders/1n-H5OqDIlhuvufG1PA-tvxe-8KPNCmsI) | (same folder) |

---

## How to run in Colab

Every notebook is saved with its outputs, so results can be read without running anything.

| Notebook | Open | Runtime |
|---|---|---|
| step1_data_preprocess | [Open in Colab](https://colab.research.google.com/github/tyadollaz/bt4222-group15-jdsearch/blob/main/step1_data_preprocess.ipynb) | High-RAM CPU |
| step2_baselines | [Open in Colab](https://colab.research.google.com/github/tyadollaz/bt4222-group15-jdsearch/blob/main/step2_baselines.ipynb) | GPU |
| step3_lightgbm | [Open in Colab](https://colab.research.google.com/github/tyadollaz/bt4222-group15-jdsearch/blob/main/step3_lightgbm.ipynb) | CPU |
| step3_lambdarank | [Open in Colab](https://colab.research.google.com/github/tyadollaz/bt4222-group15-jdsearch/blob/main/step3_lambdarank.ipynb) | CPU |
| step3_attention | [Open in Colab](https://colab.research.google.com/github/tyadollaz/bt4222-group15-jdsearch/blob/main/step3_attention.ipynb) | GPU, High-RAM |
| step4_blend_and_segments | [Open in Colab](https://colab.research.google.com/github/tyadollaz/bt4222-group15-jdsearch/blob/main/step4_blend_and_segments.ipynb) | CPU |

To run a notebook:

1. Open it in Colab and set **Runtime → Change runtime type** as in the table.
2. Choose **Runtime → Run all**.
3. Load the data in one of two ways:
   - **Option A (automatic):** sign in to Google when the first cell asks. The notebook downloads the files it needs from the shared Drive folder.
   - **Option B (manual):** download the files listed under [Data](#data) into `/content/artifacts`, then set `DRIVE_ARTIFACT_FOLDER = ""` in the notebook's config cell so it reads the local copies.

### Key settings

| Setting | Value | Meaning |
|---|---|---|
| `MAX_USERS` | 0 | Use all users |
| `HISTORY_CAP` | None | Keep each user's full history |
| `LEAK_GUARD_SECONDS` | 1 | Drop history actions logged 1 s before the search (possible leakage) |
| `SEED` | 42 | Fixed seed for splits, sampling and training |
| `RUN_TUNING` | False | Use the chosen hyperparameters; set True to rerun the search |
| `RUN_TEST` | True in the saved step 3 notebooks | Test was scored once (8 Nov), after the final model was picked on validation; the saved outputs include it |
| `LOAD_MODEL_FROM` | the frozen 8 Nov validation run | LightGBM BCE and LambdaRank load the saved model from that results folder instead of retraining |

---

## References

- Liu, J., Dou, Z., Tang, G., & Xu, S. (2023). JDsearch: A Personalized Product Search Dataset with Real Queries and Full Interactions. *SIGIR 2023*. https://github.com/rucliujn/JDsearch
- Zhou, G. et al. (2018). Deep Interest Network for Click-Through Rate Prediction. *KDD 2018*.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS 2017*.
- Burges, C. J. C. (2010). From RankNet to LambdaRank to LambdaMART: An Overview. Microsoft Research Technical Report MSR-TR-2010-82.
- Team 3, BT4222 (past project): https://github.com/myathetchai/bt4222-team3-jdsearch
