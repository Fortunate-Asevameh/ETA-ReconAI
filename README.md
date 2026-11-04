# ETA-ReconAI - Transaction Classification & Reconciliation (prototype)

ETA Financial Technology | Prototype

Uploads a bank statement (and optionally a cash book), classifies every bank transaction, reconciles bank against books,
and produces a bank reconciliation statement and a cash-flow summary. All bundled data is **fictional**.

## Files (all must sit together in one folder / at the top level of the GitHub repository)

| File | Purpose |
|---|---|
| `app.py` | Streamlit user interface (6 tabs) |
| `recon_engine.py` | All logic: import, classification, reconciliation, summaries, evaluation |
| `sample_data.xlsx` | Fictional FY2025 dataset (bank statement, cash book, answer key) |
| `requirements.txt` | Python libraries to install |
| `eta_logo.png`, `eta_logo_small.png`, `eta_logo.svg` | ETA logo (app header, sidebar, browser tab) |
| `README.md` | This file |

## Run it on your computer

```bash
cd reconai
python -m venv venv
venv\Scripts\activate          # Windows   (Mac/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. Stop it with Ctrl + C. After replacing any file, stop and restart the app.

## Put it online (free): Streamlit Community Cloud

1. Create a public GitHub repository and upload **all files above** at the top level (not inside a subfolder).
2. Sign in at share.streamlit.io with GitHub, choose Create app, pick the repository, main file `app.py`, and Deploy.
3. To update the live site later: upload the changed files to GitHub (same names replace the old ones) and commit.
   If the page still looks old, use Manage app > Reboot app, then refresh with Ctrl+F5.
4. The site is public: use fictional data only. Never upload real client statements.

## How it works

**Classification (3 layers)**
1. Keyword/regex rules (48 ordered rules) for clear narrations.
2. Machine learning for the rest: TF-IDF + Logistic Regression. Choose what it learns from (sidebar, "Layer 2"):
   - *Bundled sample data* - trained on the sample's labels, then applied to your statement.
   - *This statement only (self-training)* - default for your own uploads. Rule-confirmed transactions plus your corrections teach it
     to classify the rest. Nothing from the sample is used.
   - *Off (rules only)*.
3. Human review queue. Anything the rules do not recognise and the model cannot classify with enough confidence (default 0.70)
   is labelled **Unclassified**; the model's best guess is shown as a *suggested category*. "Stamp Duty" can only come from a
   narration that actually says stamp duty (or EMTL). A guess is also blocked when the narration has no recognisable wording.

**Reconciliation (5 passes)**: exact amount within a date window; transposed digits; many bank lines to one book entry; one bank line
to many book entries; leftovers (bank only, deposit in transit, outstanding cheque, duplicate in books).

**Outputs**: bank reconciliation statement (difference must be zero), cash flow statement, monthly summary, Excel export.

## Using your own data

The bundled sample data is hidden by default, so the app opens empty and waits for your upload. To show the demo options again, set `SHOW_DEMO = True` near the top of `app.py`.

In the sidebar choose **Data source**:
1. *Bundled sample data*.
2. *Upload template workbook* - an .xlsx laid out like `sample_data.xlsx`.
3. *Upload my own bank + cash book files* - any CSV or Excel export:
   - Title rows above the headings are skipped automatically; check the column matching in the sidebar panels.
   - Use separate Debit/Credit columns, or one signed Amount column (negative = money out).
   - No Balance column? Type the opening balance in the sidebar. Newest-first files: tick the reverse box.
   - Dates like `2025-03-10` are read year-month-day; `10/03/2025` is read day-first.
   - Cash book: Receipts = money in, Payments = money out.
   - Use one financial year (12 months) at a time.
   - **No cash book?** Leave it empty. The app classifies and builds the cash flow summary, skips reconciliation, and runs bank-only
     checks (possible duplicates, running-balance breaks, unidentified items).
   - Not supported: PDFs/scans, a separate DR/CR indicator column, amounts with CR/DR suffixes, month-first dates, semicolon/tab files.

## Measured results on the sample data (ML trained Jan-Jun, tested Jul-Dec, 290 transactions)

| Setup | Accuracy | Items for review |
|---|---|---|
| ML only (rules off) | 90.7% | 25 |
| Rules + ML | 99.0% | 32 |

Reconciliation status matches the answer key on all 573 bank lines, and the reconciliation difference is zero.
Unclassified items count as not correct, so these figures are lower than a forced guess would give but are more honest.

## Limitations

- The rules were written with the sample (and one external statement) in view, so the hybrid figure is optimistic. Validate on a statement
  the system has never seen.
- Sample data is synthetic and cleaner than real bank text.
- Ambiguous items (capital vs loan, drawings vs expense) need business context: that is what the review queue is for.
- Review corrections last only for the current session.
- Single account, single currency, single financial year per run.
