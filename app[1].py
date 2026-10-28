"""
ETA-ReconAI - Transaction Classification & Reconciliation prototype (Streamlit).
Run:  streamlit run app.py
"""
import io
import os
import warnings
import pandas as pd
import streamlit as st
import recon_engine as E

warnings.filterwarnings("ignore")
LOGO = "eta_logo.png"
try:
    from PIL import Image
    _icon = Image.open("eta_logo_small.png") if os.path.exists("eta_logo_small.png") else "🧾"
except Exception:  # noqa
    _icon = "🧾"
st.set_page_config(page_title="ETA-ReconAI", page_icon=_icon, layout="wide")
SUBTITLE = "ETA Financial Technology | Prototype"


def show_header(title):
    """Logo + heading + subtitle line."""
    c1, c2 = st.columns([1, 7])
    if os.path.exists(LOGO):
        c1.image(LOGO, width=95)
    c2.title(title)
    c2.caption(SUBTITLE)


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ALL_CATS = sorted({c for _, c, _, _ in E.RULES} | {"Sales - Customer Transfers", "Inventory / Supplier Payments",
                                                  "Salaries & Wages", "Bank Charges", "Interest Income", "Unidentified - Suspense", E.UNCLASSIFIED})
money = lambda x: f"₦{x:,.2f}"


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load(file_bytes):
    return E.load_workbook_data(io.BytesIO(file_bytes))


def read_upload(up):
    buf = io.BytesIO(up.getvalue())
    buf.name = up.name
    return E.read_table(buf, up.name)


with open("sample_data.xlsx", "rb") as f:
    SAMPLE = f.read()

if os.path.exists(LOGO):
    st.sidebar.image(LOGO, width=110)
st.sidebar.title("ETA-ReconAI")
st.sidebar.caption(SUBTITLE)
st.sidebar.caption("Transaction classification & reconciliation prototype")
source = st.sidebar.radio("Data source", ["Bundled sample data", "Upload template workbook (.xlsx)",
                                          "Upload my own bank + cash book files"])
key = None
bank_only = False
if source.startswith("Bundled"):
    bank, book, key = load(SAMPLE)
    opening = E.opening_balance(bank)
    st.sidebar.info("Using the bundled sample data (fictional company, FY2025).")

elif source.startswith("Upload template"):
    up = st.sidebar.file_uploader("Workbook (.xlsx)", type=["xlsx"],
                                  help="Needs 'Bank Statement' and 'Cash Book' sheets in the same layout as the sample.")
    if up is None:
        show_header("ETA-ReconAI")
        st.info("Upload a workbook laid out like the sample, or choose another data source in the sidebar.")
        st.stop()
    try:
        bank, book, key = load(up.getvalue())
    except Exception as e:  # noqa
        st.error(f"Could not read the workbook: {e}")
        st.stop()
    opening = E.opening_balance(bank)

else:
    st.sidebar.caption("CSV or Excel files (first sheet is read). Title rows above the headers are skipped automatically.")
    ub = st.sidebar.file_uploader("Bank statement", type=["csv", "xlsx", "xls"], key="up_bank")
    uk = st.sidebar.file_uploader("Cash book (optional)", type=["csv", "xlsx", "xls"], key="up_book",
                                  help="No cash book? Leave this empty: you can still classify transactions and produce the cash flow summary.")
    if ub is None:
        show_header("ETA-ReconAI")
        st.info("Upload a bank statement in the sidebar (a cash book is optional), then confirm the column matching.")
        st.stop()
    try:
        rb = read_upload(ub)
        rk = read_upload(uk) if uk is not None else None
    except Exception as e:  # noqa
        st.error(f"Could not read the files: {e}")
        st.stop()
    newest_first = st.sidebar.checkbox("Bank rows are newest-first (reverse them)")
    if newest_first:
        rb = rb.iloc[::-1].reset_index(drop=True)
    gb = E.guess_columns(rb.columns, E.BANK_HINTS)
    gk = E.guess_columns(rk.columns, E.BOOK_HINTS) if rk is not None else {}
    bank_only = rk is None

    def pick(label, cols, guess, k):
        opts = ["(none)"] + list(cols)
        v = st.selectbox(label, opts, index=opts.index(guess) if guess in opts else 0, key=k)
        return None if v == "(none)" else v

    with st.sidebar.expander("Match bank columns", expanded=True):
        mb = {f: pick(lbl, rb.columns, gb.get(f), f"b_{f}_{ub.name}") for f, lbl in
              [("date", "Date"), ("narration", "Narration / description"), ("debit", "Debit / withdrawals"),
               ("credit", "Credit / deposits"), ("amount", "Single signed Amount (if no debit/credit)"),
               ("balance", "Balance (optional)"), ("reference", "Reference (optional)")]}
    mk = None
    if rk is not None:
        with st.sidebar.expander("Match cash book columns", expanded=True):
            mk = {f: pick(lbl, rk.columns, gk.get(f), f"k_{f}_{uk.name}") for f, lbl in
                  [("date", "Date"), ("description", "Description"), ("receipt", "Receipts (money in)"),
                   ("payment", "Payments (money out)"), ("amount", "Single signed Amount (if no receipt/payment)")]}
    if not (mb["date"] and mb["narration"]) or (mk is not None and not (mk["date"] and mk["description"])):
        show_header("ETA-ReconAI")
        st.warning("Choose the Date and Narration/Description columns for both files.")
        st.stop()
    try:
        bank = E.standardise_bank(rb, mb)
        book = E.standardise_book(rk, mk) if rk is not None else E.empty_book()
    except Exception as e:  # noqa
        st.error(f"Could not use that column matching: {e}")
        st.stop()
    if bank.empty or (rk is not None and book.empty):
        st.error("No valid rows found. Check the Date and amount columns.")
        st.stop()
    default_open = E.opening_balance(bank) if bank["balance"].notna().all() else 0.0
    opening = st.sidebar.number_input("Opening balance (NGN)", value=float(default_open), step=1000.0, format="%.2f",
                                      help="Balance before the first transaction. Filled automatically if the bank file has a Balance column.")
    if bank["date"].dt.to_period("M").nunique() > 12:
        st.sidebar.warning("More than 12 months of data. Monthly figures group by calendar month, so use one financial year at a time.")

st.sidebar.subheader("Settings")
use_rules = st.sidebar.checkbox("Layer 1: keyword rules", value=True)
OPT_SAMPLE = "Bundled sample data"
OPT_SELF = "This statement only (self-training)"
OPT_OFF = "Off (rules only)"
ml_source = st.sidebar.radio(
    "Layer 2: what should the ML layer learn from?", [OPT_SAMPLE, OPT_SELF, OPT_OFF],
    index=1 if source.startswith("Upload my own") else 0, key=f"ml_{source}",
    help="Self-training learns only from this statement's own narrations: transactions the rules recognise confidently "
         "(plus your corrections) teach the model how to classify the remaining ones. Nothing from the sample data is used.")
threshold = st.sidebar.slider("Review threshold (confidence below this goes to review)", 0.30, 0.95, 0.70, 0.05)
use_split = (key is not None) and ml_source == OPT_SAMPLE
if use_split:
    train_months = st.sidebar.slider("ML training window (first N months, needs Answer Key)", 3, 11, 6)
else:
    train_months = 12
window = st.sidebar.slider("Reconciliation date window (days)", 1, 14, 7)

# ----------------------------------------------------------------------------- model + classification
if "corrections" not in st.session_state:
    st.session_state.corrections = {}
if "extra_train" not in st.session_state:
    st.session_state.extra_train = False

model = None
if ml_source == OPT_SAMPLE:
    if key is not None:
        lab = bank.merge(key, on="sn")
        tr = lab[lab["month"] <= train_months]
    else:
        sample_bank, _, sample_key = load(SAMPLE)
        tr = sample_bank.merge(sample_key, on="sn")
        st.sidebar.info("The ML layer is trained on the bundled sample, then applied to your transactions.")
    model = E.train_model(tr, tr["true_category"])
elif ml_source == OPT_SELF:
    st.sidebar.caption("The ML layer learns only from this statement: rule-confirmed and human-corrected transactions teach it to classify the rest.")
    base = E.classify(bank, None, threshold, use_rules=use_rules)
    base = E.apply_corrections(base, st.session_state.corrections)
    model = E.self_train_model(base)
    if model is None:
        st.sidebar.warning("Not enough confidently classified transactions in this statement to train on yet (need 10+ across 2+ categories). "
                           "Turn on the keyword rules or correct some items in the Review queue.")
else:
    st.sidebar.caption("ML layer is off: only keyword rules classify. Everything else goes to the review queue.")

classified = E.classify(bank, model, threshold, use_rules=use_rules)
classified = E.apply_corrections(classified, st.session_state.corrections)

# optional: retrain using corrections + confident results (bundled-sample mode only; self-training already uses corrections)
if st.session_state.extra_train and st.session_state.corrections and ml_source == OPT_SAMPLE:
    conf = classified[(classified["confidence"] >= 0.9)]
    model2 = E.train_model(conf, conf["category"])
    classified = E.classify(bank, model2, threshold, use_rules=use_rules)
    classified = E.apply_corrections(classified, st.session_state.corrections)

if bank_only:
    bank_r, book_r = bank.copy(), book.copy()
    bank_r["recon_status"] = "Not reconciled (no cash book)"
    statement, diff = None, 0.0
else:
    bank_r, book_r = E.reconcile(bank, book, window)
    statement, diff = E.bank_rec_statement(bank_r, book_r, bank_r, book_r, opening)

# ----------------------------------------------------------------------------- header
show_header("ETA-ReconAI: Transaction Classification & Reconciliation")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Bank transactions", f"{len(bank):,}")
c2.metric("Cash book entries", "none" if bank_only else f"{len(book):,}")
c3.metric("Needs review", f"{int(classified['needs_review'].sum()):,}")
if bank_only:
    c4.metric("Reconciliation", "n/a", help="No cash book was uploaded.")
else:
    c4.metric("Reconciliation difference", money(diff), help="Adjusted bank balance minus adjusted cash book balance. Must be zero.")

t1, t2, t3, t4, t5, t6 = st.tabs(["1 Data", "2 Classification", "3 Review queue", "4 Reconciliation", "5 Financial summary", "6 Accuracy & export"])

# ----------------------------------------------------------------------------- tab 1
with t1:
    st.subheader("Bank statement")
    st.dataframe(bank[["sn", "date", "reference", "narration", "debit", "credit", "balance"]], use_container_width=True, height=300)
    st.subheader("Cash book")
    st.dataframe(book[["ref", "date", "description", "receipt", "payment"]], use_container_width=True, height=300)

# ----------------------------------------------------------------------------- tab 2
with t2:
    st.subheader("Classified transactions")
    f1, f2, f3 = st.columns(3)
    sel_cat = f1.multiselect("Category", sorted(classified["category"].unique()))
    sel_m = f2.multiselect("Method", ["Rule", "ML", "Human"], default=[])
    only_low = f3.checkbox("Only low confidence")
    v = classified.copy()
    if sel_cat: v = v[v["category"].isin(sel_cat)]
    if sel_m: v = v[v["method"].isin(sel_m)]
    if only_low: v = v[v["needs_review"]]
    st.dataframe(v[["sn", "date", "narration", "debit", "credit", "category", "suggested_category", "confidence", "method", "needs_review"]],
                 use_container_width=True, height=450)
    st.caption("Rule = keyword match; ML = TF-IDF + Logistic Regression prediction; Human = your correction.")
    mc = classified["method"].value_counts()
    st.bar_chart(mc)

# ----------------------------------------------------------------------------- tab 3
with t3:
    st.subheader("Review queue (items the system could not classify with confidence)")
    q = classified[classified["needs_review"]][["sn", "date", "narration", "debit", "credit", "category", "suggested_category", "confidence", "method"]].copy()
    st.write(f"{len(q)} transactions need a decision. Pick the correct category, then click **Apply corrections**.")
    if len(q):
        edited = st.data_editor(
            q, use_container_width=True, height=450, hide_index=True, key="review_editor",
            disabled=["sn", "date", "narration", "debit", "credit", "suggested_category", "confidence", "method"],
            column_config={"category": st.column_config.SelectboxColumn("category", options=ALL_CATS, required=True)})
        b1, b2 = st.columns(2)
        if b1.button("Apply corrections"):
            changed = edited.merge(q[["sn", "category"]], on="sn", suffixes=("", "_old"))
            changed = changed[changed["category"] != changed["category_old"]]
            for _, r in changed.iterrows():
                st.session_state.corrections[int(r["sn"])] = r["category"]
            st.rerun()
        if b2.button("Apply + retrain model on corrections"):
            changed = edited.merge(q[["sn", "category"]], on="sn", suffixes=("", "_old"))
            changed = changed[changed["category"] != changed["category_old"]]
            for _, r in changed.iterrows():
                st.session_state.corrections[int(r["sn"])] = r["category"]
            st.session_state.extra_train = True
            st.rerun()
    else:
        st.success("Nothing to review at this threshold.")
    st.caption(f"Corrections made this session: {len(st.session_state.corrections)}")

# ----------------------------------------------------------------------------- tab 4
with t4:
  if bank_only:
    st.subheader("Reconciliation needs a cash book")
    st.info("Reconciling means comparing the bank statement with your own records. With no cash book there is nothing to compare, "
            "so the bank reconciliation statement is not produced. Classification and the cash flow summary still work. "
            "Below are checks that need only the bank statement.")
    chk = E.bank_only_checks(bank)
    st.write("**Possible duplicate transactions (same date and amount)**")
    st.dataframe(chk["duplicates"], use_container_width=True, height=250)
    if "balance_breaks" in chk:
        st.write("**Running-balance breaks (balance does not follow from the previous line)**")
        if chk["balance_breaks"].empty:
            st.success("The running balance is consistent from start to finish.")
        else:
            st.dataframe(chk["balance_breaks"], use_container_width=True, height=250)
    st.write("**Items still unidentified**")
    st.dataframe(classified[classified["category"] == "Unidentified - Suspense"][["sn", "date", "narration", "debit", "credit"]], use_container_width=True)
  else:
    st.subheader("Reconciliation results")
    a, b = st.columns(2)
    a.write("**Bank lines by status**"); a.bar_chart(bank_r["recon_status"].value_counts())
    b.write("**Cash book entries by status**"); b.bar_chart(book_r["recon_status"].value_counts())
    st.subheader("Exceptions")
    ex_b = bank_r[~bank_r["recon_status"].isin(["Matched"])][["sn", "date", "narration", "net", "recon_status", "book_ref"]]
    ex_k = book_r[~book_r["recon_status"].isin(["Matched"])][["ref", "date", "description", "net", "recon_status", "bank_sn"]]
    st.write("Bank side"); st.dataframe(ex_b, use_container_width=True, height=300)
    st.write("Cash book side"); st.dataframe(ex_k, use_container_width=True, height=250)
    st.subheader("Bank reconciliation statement")
    st.dataframe(statement.assign(**{"Amount (NGN)": statement["Amount (NGN)"].map(lambda x: "" if pd.isna(x) else f"{x:,.2f}")}),
                 use_container_width=True, hide_index=True)
    if abs(diff) < 0.01:
        st.success("Reconciled: adjusted bank balance equals adjusted cash book balance.")
    else:
        st.error(f"Unreconciled difference of {money(diff)}. Check unmatched items.")

# ----------------------------------------------------------------------------- tab 5
with t5:
    st.subheader("Cash flow statement (from classified transactions)")
    cf = E.cash_flow_statement(classified, opening)
    st.dataframe(cf.assign(**{"Amount (NGN)": cf["Amount (NGN)"].map(lambda x: f"{x:,.2f}")}), use_container_width=True, hide_index=True)
    unresolved = classified[classified["category"].isin(["Unidentified - Suspense", E.UNCLASSIFIED])]["net"].sum()
    if abs(unresolved) > 0:
        st.warning(f"{money(unresolved)} is unclassified or in suspense. Resolve it before finalising the statements.")
    st.subheader("Monthly summary by category")
    piv = E.monthly_summary(classified)
    piv.columns = [MONTHS[c - 1] if isinstance(c, (int, float)) else c for c in piv.columns]
    st.dataframe(piv.round(0), use_container_width=True, height=500)
    st.subheader("Net cash movement by month")
    st.line_chart(classified.groupby("month")["net"].sum().rename(lambda m: MONTHS[m - 1]))

# ----------------------------------------------------------------------------- tab 6
with t6:
    st.subheader("Accuracy against the Answer Key")
    if key is None:
        st.info("No answer key is available for this data, so accuracy can't be scored. You can still review the classifications, the reconciliation exceptions, and the reports.")
    else:
        test_mask = classified["month"] > (train_months if use_split else 0)
        overall, by_clarity, by_method, merged = E.evaluate_classification(classified, key, test_mask)
        r_acc, _ = E.evaluate_reconciliation(bank_r, key)
        m1, m2, m3 = st.columns(3)
        m1.metric(f"Classification accuracy (held-out months {train_months + 1}-12)" if use_split else "Classification accuracy (all transactions)", f"{overall:.1%}")
        m2.metric("Reconciliation status accuracy", f"{r_acc:.1%}")
        auto = merged[~merged["sn"].isin(classified[classified["needs_review"]]["sn"])]
        m3.metric("Accuracy of auto-accepted items", f"{auto['correct'].mean():.1%}", help=f"{len(auto)} of {len(merged)} items")
        st.write("**By clarity**"); st.dataframe(by_clarity.style.format({"accuracy": "{:.1%}"}), use_container_width=True)
        st.write("**By method**"); st.dataframe(by_method.style.format({"accuracy": "{:.1%}"}), use_container_width=True)
        st.write("**Misclassified items**")
        st.dataframe(merged[~merged["correct"]][["sn", "narration", "category", "true_category", "confidence", "method", "clarity"]],
                     use_container_width=True)
        st.caption("Tip: untick 'keyword rules' in the sidebar to see how the ML layer performs on its own.")

    st.subheader("Export")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        classified.to_excel(xw, sheet_name="Classified", index=False)
        if not bank_only:
            bank_r.to_excel(xw, sheet_name="Bank recon", index=False)
            book_r.to_excel(xw, sheet_name="Book recon", index=False)
            statement.to_excel(xw, sheet_name="Bank rec statement", index=False)
        E.cash_flow_statement(classified, opening).to_excel(xw, sheet_name="Cash flow", index=False)
        piv.to_excel(xw, sheet_name="Monthly summary")
    st.download_button("Download results (.xlsx)", buf.getvalue(), file_name="eta_reconai_results.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
