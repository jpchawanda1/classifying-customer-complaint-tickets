# Classifying Customer Complaint Tickets

Routing free-text customer complaints from a Kenyan mobile-money and telecom provider to the
right department, automatically.

Coursework for **DSA 4020 — Natural Language Processing**, United States International
University – Africa.

---

## The problem

Agents currently read every incoming complaint and route it by hand to one of five desks:

| Category | Covers |
|---|---|
| `mpesa_transaction` | failed, reversed, duplicated or mis-sent transfers; wrong charges; agent and paybill/till problems |
| `network_data` | calls, SMS, data bundles, signal, fibre/router faults |
| `loan_credit` | mobile loans, overdrafts, limits, interest, penalties, credit-bureau listing |
| `account_access` | PIN, OTP, SIM swap, blocked accounts, login, profile updates |
| `fraud_scam` | impersonation, fake prize messages, hacking, unauthorised withdrawals, phishing |

The tickets are written the way customers actually write them — SMS shorthand, misspellings,
and English mixed with Kiswahili and Sheng:

> *"Nimenunua bundles za KES 15,000 lakini hazijaingia kwa simu yangu. My internet bundles
> finished within an hour even though I was not browsing. The 4G network is down at our
> office and technicians have not come on Friday."*

401 labelled training tickets, 99 unlabelled test tickets. The classes are imbalanced at
**3.48 : 1** — `mpesa_transaction` holds 34.7% of the data, `fraud_scam` just 10.0%.

## Results

Final model: **TF-IDF unigrams + logistic regression** (`class_weight='balanced'`, `C=10`),
305 features. Evaluated on a stratified 20% hold-out (81 tickets) never used for any
decision:

| Metric | Score |
|---|---|
| Accuracy | **0.914** |
| Macro F1 | **0.890** |
| Macro precision | 0.899 |
| Macro recall | 0.883 |
| Weighted F1 | 0.911 |
| *Majority-class baseline* | *0.347 accuracy / 0.103 macro F1* |

Per class:

| Category | F1 | Support |
|---|---|---|
| `loan_credit` | 1.000 | 11 |
| `mpesa_transaction` | 0.982 | 28 |
| `network_data` | 0.927 | 20 |
| `fraud_scam` | 0.800 | 8 |
| `account_access` | 0.741 | 14 |

`fraud_scam` is reported separately because it is the smallest class and the most expensive
to get wrong: **precision 0.857, recall 0.750** — it catches 6 of 8 fraud tickets.

## What the analysis found

**Accuracy did not choose the model.** Four TF-IDF models scored between 0.897 and 0.920
macro-F1 in 5-fold cross-validation, with fold-to-fold standard deviations of 0.025–0.033 —
the spread between them is smaller than the noise. MultinomialNB scored highest (0.920), but
logistic regression was chosen because it answers >99% confident on only 10% of tickets
against Naive Bayes's 65%, and separates right from wrong predictions by 0.217 confidence
against 0.148. That gap is what makes a low-confidence review queue possible at all.

**Three controlled experiments returned null results, reported as such.** Removing numbers
(which destroys `4G` → `g`) cost −0.0014 ± 0.0029 macro-F1; `class_weight='balanced'` made no
difference (p = 0.55); bigrams added 1,143 features for −0.0013. The corpus is redundant
enough that destroying one cue leaves another intact.

**Pre-trained embeddings underperformed sparse features.** GloVe maps `pin` to *lapel, pins,
chin, bow* (the clothing sense) and `pesa` into unrelated Italian vocabulary; `paybill`,
`fuliza`, `mkopo` and `sijarudishiwa` have no GloVe vector at all. Averaging also discards
word order entirely — two sentences describing *opposite* situations reduce to vectors with
cosine similarity 1.0000.

**A Transformer did not help** (bonus section). On the identical split:

| Approach | Macro F1 | Train time |
|---|---|---|
| TF-IDF + logistic regression | **0.890** | <0.01 s |
| Fine-tuned multilingual MiniLM (4 epochs) | 0.860 | 46 s |
| Frozen multilingual MiniLM + logistic regression | 0.855 | 2.5 s |

More telling than the scores: 6 of the classical model's 7 errors are also missed by the
fine-tuned Transformer, and **5 tickets are missed by every approach**. Those are genuinely
ambiguous tickets — the remaining errors are a labelling problem, not a modelling one.

**The dataset has a ceiling.** Undoing the injected typos, dates and amounts reduces all 401
tickets to roughly **59 sentence templates**, and no word type occurs only once. 30.9% of
tickets contain sentences pointing at more than one department while carrying a single label,
putting a ceiling near **0.895 accuracy** on what any classifier can achieve from these texts.
Two training tickets are the identical sentence under different labels.

## Repository contents

| Path | What it is |
|---|---|
| `nlp_complaint_classification.ipynb` | The full analysis — 36 code cells, runs top to bottom, outputs embedded |
| `Chawanda_670444_NLP_Assignment.docx` | Write-up: code, outputs, figures and written answers |
| `predictions.csv` | Test-set predictions (99 rows: `ticket_id`, `category`) |
| `assignment_outputs/` | Figures and `results.json` backing every number quoted above |
| `complaints_train.csv` / `complaints_test_unlabelled.csv` | Course-provided data (synthetic) |
| `NLP_Assignment_Complaint_Classification.docx` | The original assignment brief |

## Reproducing

```bash
python3 -m venv venv && source venv/bin/activate
pip install numpy==1.26.4 scipy==1.13.1 pandas==2.2.3 scikit-learn==1.5.2 \
            nltk==3.9.1 gensim==4.3.3 matplotlib==3.9.2 jupyter
# optional, for the bonus Transformer section
pip install torch==2.2.2 transformers==4.44.2

jupyter notebook nlp_complaint_classification.ipynb
```

The notebook downloads its own NLTK corpora and GloVe vectors on first run, and degrades
gracefully if either is unavailable. `random_state=42` is fixed throughout; Word2Vec uses
`workers=1` so gensim is deterministic. Every vectoriser is fitted inside the
cross-validation pipeline, and the test file is not read until the final prediction step.

Running it regenerates `predictions.csv`, all figures and `assignment_outputs/results.json`.

## A note on the data

`complaints_train.csv` and `complaints_test_unlabelled.csv` are **synthetic**, written for
teaching. They imitate real features of Kenyan customer text but contain no personal data —
no phone numbers, names, ID numbers or real transaction references, and only nine distinct
money amounts. Any production system built on this approach would need the privacy
safeguards discussed in the write-up.

## Academic integrity

This is coursework submitted for assessment, published for reference. If you are taking the
same course, **do not submit any of it as your own** — read it, disagree with it, and write
your own.
