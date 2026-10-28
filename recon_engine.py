"""
ReconAI engine - transaction classification + bank/cash-book reconciliation.

Classification : Layer 1 keyword/regex rules  ->  Layer 2 ML (TF-IDF + Logistic Regression)
                 ->  Layer 3 low-confidence items go to a human review queue.
Reconciliation : exact match -> transposition match -> many-to-one -> one-to-many
                 -> classify leftovers (deposit in transit, outstanding cheque, bank-only, duplicate).
"""
import re
import itertools
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# ----------------------------------------------------------------------------- loading
def load_workbook_data(path_or_file):
    """Read the sample workbook (Bank Statement, Cash Book, Answer Key)."""
    xl = pd.ExcelFile(path_or_file)
    bank = pd.read_excel(xl, sheet_name="Bank Statement", header=7)
    bank = bank[pd.to_numeric(bank["S/N"], errors="coerce").notna()].copy()
    bank = bank.rename(columns={"Trans Date": "date", "Narration": "narration", "Reference": "reference",
                                "Debit (NGN)": "debit", "Credit (NGN)": "credit", "Balance (NGN)": "balance", "S/N": "sn"})
    bank = bank[["sn", "date", "reference", "narration", "debit", "credit", "balance"]]
    book = pd.read_excel(xl, sheet_name="Cash Book", header=4)
    book = book[book["Book Ref"].astype(str).str.startswith("CB-")].copy()
    book = book.rename(columns={"Book Ref": "ref", "Date": "date", "Description": "description",
                                "Receipts (NGN)": "receipt", "Payments (NGN)": "payment"})[["ref", "date", "description", "receipt", "payment"]]
    try:
        key = pd.read_excel(xl, sheet_name="Answer Key", header=3)
        key = key[pd.to_numeric(key["S/N"], errors="coerce").notna()].copy()
        key = key.rename(columns={"S/N": "sn", "Category": "true_category", "Clarity": "clarity", "Recon Status": "true_status"})
        key = key[["sn", "true_category", "clarity", "true_status"]]
    except Exception:
        key = None
    return _clean_bank(bank), _clean_book(book), key


def _clean_bank(bank):
    b = bank.copy()
    b["date"] = pd.to_datetime(b["date"])
    for c in ("debit", "credit"):
        b[c] = pd.to_numeric(b[c], errors="coerce").fillna(0.0)
    b["sn"] = b["sn"].astype(int)
    b["net"] = (b["credit"] - b["debit"]).round(2)
    b["month"] = b["date"].dt.month
    return b.reset_index(drop=True)


def _clean_book(book):
    k = book.copy()
    k["date"] = pd.to_datetime(k["date"])
    for c in ("receipt", "payment"):
        k[c] = pd.to_numeric(k[c], errors="coerce").fillna(0.0)
    k["net"] = (k["receipt"] - k["payment"]).round(2)
    return k.reset_index(drop=True)


# ----------------------------------------------------------------------------- classification
UNCLASSIFIED = "Unclassified"   # shown when the system cannot decide; never forced into a real category
# (regex, category, confidence, direction)  direction: 'DR', 'CR' or None
RULES = [
    (r"REFUND CR NOTE", "Supplier Refund / Credit Note", 0.70, "CR"),
    (r"OVERPAYMENT REFUND", "Customer Refund", 0.90, "DR"),
    (r"REV/DUP", "Bank Reversal / Error", 0.75, "CR"),
    (r"POS SETTLEMENT", "Sales - POS Settlements", 0.97, "CR"),
    (r"ONLINE GATEWAY SETTLEMENT", "Sales - Online Gateway Settlements", 0.97, "CR"),
    (r"LOAN DISBURSEMENT", "Loan Proceeds", 0.98, "CR"),
    (r"LOAN MGT FEE|VAT ON LOAN MGT", "Loan Facility Fees", 0.97, "DR"),
    (r"LOAN REPAYMENT", "Loan Repayment - Principal + Interest", 0.98, "DR"),
    (r"PAYE", "PAYE Remittance", 0.98, "DR"),
    (r"PENSION", "Pension Remittance", 0.98, "DR"),
    (r"VAT REMITTANCE", "VAT Remittance", 0.98, "DR"),
    (r"WHT REMITTANCE", "WHT Remittance", 0.98, "DR"),
    (r"CIT PAYMENT", "Company Income Tax", 0.98, "DR"),
    (r"STAMP DUT(Y|IES)|\\bEMTL\\b|ELECTRONIC MONEY TRANSFER LEVY", "Stamp Duty", 0.98, "DR"),
    (r"UNPAID CHEQUE COMMISSION|ACCOUNT MAINTENANCE|SMS ALERT|NIP TRANSFER FEES|VAT ON BANK CHARGES|SWIFT CHARGES|SWIFT COMM|COMMISSION ON FX|FX COMMISSION|\\bCOT\\b|COMMISSION ON TURNOVER", "Bank Charges", 0.97, "DR"),
    (r"INTEREST ON CREDIT", "Interest Income", 0.97, "CR"),
    (r"RETURNED CHEQUE", "Returned Cheque (Dishonoured)", 0.80, "DR"),
    (r"CHEQUE DEPOSIT", "Sales - Cheque Deposits", 0.90, "CR"),
    (r"STAFF SALARIES|BONUS", "Salaries & Wages", 0.98, "DR"),
    (r"POWER DISCO|BROADBAND", "Utilities (Power & Internet)", 0.97, "DR"),
    (r"PETRO STATION|APEX LOGISTICS", "Fuel & Logistics", 0.90, "DR"),
    (r"SOCIAL MEDIA ADS|FLYERS", "Marketing & Advertising", 0.90, "DR"),
    (r"STATIONERY", "Office & Admin", 0.95, "DR"),
    (r"RENT", "Rent", 0.95, "DR"),
    (r"INSURANCE", "Insurance", 0.95, "DR"),
    (r"GENERATOR|LAGOS MOTORS", "Equipment & Vehicle Purchases", 0.95, "DR"),
    (r"AUDIT FEE|LEGAL FEES|CORPORATE AFFAIRS", "Professional Fees", 0.95, "DR"),
    (r"REPAIR", "Repairs & Maintenance", 0.80, "DR"),
    (r"FX PURCHASE", "Import Purchases (FX)", 0.70, "DR"),
    (r"CASH WITHDRAWAL", "Petty Cash Withdrawals", 0.65, "DR"),
    (r"CASH DEPOSIT|CDM DEPOSIT", "Sales - Cash Deposits", 0.65, "CR"),
    (r"ADEBAYO/FUNDING", "Owner Capital Injection", 0.60, "CR"),
    (r"ADEBAYO/ADVANCE", "Director Loan Received", 0.60, "CR"),
    (r"ADEBAYO/REFUND", "Director Loan Repaid", 0.60, "DR"),
    (r"ADEBAYO/(UPKEEP|PERSONAL)|SCHOOL FEES", "Owner Drawings", 0.70, "DR"),
    (r"SAVINGS ACC|TRF FROM 0245", "Transfer Between Own Accounts", 0.65, None),
    (r"MOB TRF-DR|MOBILE TRF CR REF|POS/WEB|TRF FRM 3PL", "Unidentified - Suspense", 0.50, None),
    # general Nigerian banking vocabulary
    (r"MTN|AIRTEL|GLO |9MOBILE|AIRTIME|FIBRE|INTERNET", "Utilities (Power & Internet)", 0.90, "DR"),
    (r"EKEDC|IKEDC|ELECTRICITY|PREPAID", "Utilities (Power & Internet)", 0.92, "DR"),
    (r"DIESEL|FUEL|PETROL", "Fuel & Logistics", 0.88, "DR"),
    (r"FREIGHT|HAULAGE|LOGISTICS|TRANSPORT", "Fuel & Logistics", 0.88, "DR"),
    (r"MAINTENANCE", "Repairs & Maintenance", 0.75, "DR"),
    (r"DIVIDEND", "Owner Drawings", 0.65, "DR"),
    (r"GRANT", "Grant / Other Income", 0.80, "CR"),
    (r"DISPOSAL|SALE OF (VEHICLE|ASSET|EQUIPMENT)", "Asset Disposal Proceeds", 0.80, "CR"),
    (r"(NIP CR|NIP TRF FROM|TRF FROM)[/ ].*(INV|PMT)", "Sales - Customer Transfers", 0.90, "CR"),
    (r"NIP TRF FROM .*/INV \d+", "Sales - Customer Transfers", 0.95, "CR"),
    (r"(NIP TRF TO|CHQ PAID \d+ -) .*(LTD|PALLETS).*", "Inventory / Supplier Payments", 0.92, "DR"),
]
_COMPILED = [(re.compile(p), c, conf, d) for p, c, conf, d in RULES]


def rule_classify(narration, debit, credit):
    direction = "DR" if debit > 0 else "CR"
    for rx, cat, conf, d in _COMPILED:
        if d and d != direction:
            continue
        if rx.search(narration.upper()):
            return cat, conf
    return None, 0.0


def _normalise(text, direction):
    t = re.sub(r"\d+", " ", str(text).upper())
    t = re.sub(r"[^A-Z& ]", " ", t)
    return f"{direction} " + re.sub(r"\s+", " ", t).strip()


def _features(df):
    return [_normalise(n, "DR" if d > 0 else "CR") for n, d in zip(df["narration"], df["debit"])]


def train_model(df, labels):
    """TF-IDF (words + char n-grams) + Logistic Regression on the narration text."""
    from sklearn.pipeline import FeatureUnion
    feats = FeatureUnion([
        ("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)),
    ])
    model = make_pipeline(feats, LogisticRegression(max_iter=2000, C=20, class_weight="balanced"))
    model.fit(_features(df), labels)
    return model



# Categories that may only come from an explicit keyword in the narration, never from an ML guess.
RULE_ONLY = {"Stamp Duty"}
_GENERIC = {"DR", "CR", "NIP", "TRF", "TRANSFER", "MOB", "MOBILE", "POS", "WEB", "FT", "PAYMENT", "PAY", "TO", "FROM", "REF",
            "REFERENCE", "NO", "NUM", "CHQ", "CHEQUE", "INWARD", "OUTWARD", "DEBIT", "CREDIT", "INV", "PMT", "TXN", "ONLINE"}


def _known_words(model):
    try:
        vec = model.steps[0][1].transformer_list[0][1]
        return {w for w in vec.vocabulary_ if " " not in w}
    except Exception:
        return None


def _ml_may_decide(narration, direction, guess, known):
    """An ML guess is only allowed when the narration has real wording the model has actually seen."""
    if guess in RULE_ONLY:
        return False
    toks = [t for t in _normalise(narration, direction).split()[1:] if len(t) >= 3 and t not in _GENERIC]
    if not toks:
        return False                       # nothing but numbers / generic transfer words
    if known is not None and not any(t.lower() in known for t in toks):
        return False                       # wording never seen in training
    return True


def classify(bank, model=None, threshold=0.70, use_rules=True):
    """Hybrid classification. Adds category, confidence, method and needs_review."""
    out = bank.copy()
    cats, confs, methods, sugg = [], [], [], []
    ml_pred, ml_conf = (None, None)
    known = _known_words(model) if model is not None else None
    if model is not None:
        proba = model.predict_proba(_features(out))
        ml_pred = model.classes_[proba.argmax(axis=1)]
        ml_conf = proba.max(axis=1)
    for i, row in out.iterrows():
        cat, conf = rule_classify(row["narration"], row["debit"], row["credit"]) if use_rules else (None, 0.0)
        if cat:
            cats.append(cat); confs.append(conf); methods.append("Rule"); sugg.append(cat)
        elif model is not None:
            guess, c = ml_pred[i], float(ml_conf[i])
            # below the confidence threshold the model's guess is NOT forced into a category
            ok_ = c >= threshold and _ml_may_decide(row["narration"], "DR" if row["debit"] > 0 else "CR", guess, known)
            cats.append(guess if ok_ else UNCLASSIFIED)
            confs.append(c if ok_ else min(c, max(threshold - 0.01, 0.0))); methods.append("ML"); sugg.append(guess)
        else:
            cats.append(UNCLASSIFIED); confs.append(0.0); methods.append("None"); sugg.append("")
    out["category"] = cats
    out["suggested_category"] = sugg
    out["confidence"] = np.round(confs, 3)
    out["method"] = methods
    out["needs_review"] = out["confidence"] < threshold
    return out


# ----------------------------------------------------------------------------- reconciliation
def _digits(x):
    return sorted(str(int(round(abs(x) * 100))))


def reconcile(bank, book, day_window=7):
    bank = bank.copy(); book = book.copy()
    bank["recon_status"] = ""; bank["book_ref"] = ""
    book["recon_status"] = ""; book["bank_sn"] = ""
    ub = set(bank.index); uk = set(book.index)

    def mark(bi_list, ki_list, status):
        for bi in bi_list:
            bank.at[bi, "recon_status"] = status
            bank.at[bi, "book_ref"] = ", ".join(book.at[k, "ref"] for k in ki_list)
        for ki in ki_list:
            book.at[ki, "recon_status"] = status
            book.at[ki, "bank_sn"] = ", ".join(str(bank.at[b, "sn"]) for b in bi_list)
        ub.difference_update(bi_list); uk.difference_update(ki_list)

    def timing(status, bis, kis):
        mb = {bank.at[b, "date"].month for b in bis}; mk = {book.at[k, "date"].month for k in kis}
        return "Timing difference" if status == "Matched" and mb != mk else status

    # Pass 1: exact amount within date window (closest date first)
    cands = []
    for bi in ub:
        for ki in uk:
            if abs(bank.at[bi, "net"] - book.at[ki, "net"]) < 0.005:
                dd = abs((bank.at[bi, "date"] - book.at[ki, "date"]).days)
                if dd <= day_window:
                    cands.append((dd, bi, ki))
    for dd, bi, ki in sorted(cands):
        if bi in ub and ki in uk:
            mark([bi], [ki], timing("Matched", [bi], [ki]))

    # Pass 2: transposed digits (same digits, different order)
    for bi in sorted(ub):
        for ki in sorted(uk):
            nb, nk = bank.at[bi, "net"], book.at[ki, "net"]
            if nb * nk > 0 and abs(nb - nk) > 0.005 and _digits(nb) == _digits(nk) \
                    and abs((bank.at[bi, "date"] - book.at[ki, "date"]).days) <= day_window:
                mark([bi], [ki], "Amount difference"); break

    # Pass 3: many bank lines -> one book entry
    for ki in sorted(uk):
        target = book.at[ki, "net"]
        pool = [b for b in ub if bank.at[b, "net"] * target > 0 and abs(bank.at[b, "net"]) < abs(target) + 0.005
                and abs((bank.at[b, "date"] - book.at[ki, "date"]).days) <= day_window]
        pool = sorted(pool, key=lambda b: abs((bank.at[b, "date"] - book.at[ki, "date"]).days))[:14]
        found = None
        for n in range(2, 7):
            for combo in itertools.combinations(pool, n):
                if abs(sum(bank.at[b, "net"] for b in combo) - target) < 0.005:
                    found = list(combo); break
            if found: break
        if found:
            mark(found, [ki], "Matched (grouped)")

    # Pass 4: one bank line -> many book entries
    for bi in sorted(ub):
        target = bank.at[bi, "net"]
        pool = [k for k in uk if book.at[k, "net"] * target > 0 and abs(book.at[k, "net"]) < abs(target) + 0.005
                and abs((book.at[k, "date"] - bank.at[bi, "date"]).days) <= day_window]
        pool = sorted(pool, key=lambda k: abs((book.at[k, "date"] - bank.at[bi, "date"]).days))[:12]
        found = None
        for n in range(2, 5):
            for combo in itertools.combinations(pool, n):
                if abs(sum(book.at[k, "net"] for k in combo) - target) < 0.005:
                    found = list(combo); break
            if found: break
        if found:
            mark([bi], found, "Matched (split)")

    # Pass 5: leftovers
    matched_sig = {(round(bank.at[b, "net"], 2), bank.at[b, "date"]) for b in bank.index if b not in ub}
    last_day = bank["date"].max()
    for bi in sorted(ub):
        hint = "Possible duplicate posting by bank" if (round(bank.at[bi, "net"], 2), bank.at[bi, "date"]) in matched_sig else ""
        bank.at[bi, "recon_status"] = "Bank only (not in cash book)"; bank.at[bi, "book_ref"] = hint
    matched_book_sig = {(round(book.at[k, "net"], 2), book.at[k, "date"]) for k in book.index if k not in uk}
    for ki in sorted(uk):
        net, d = round(book.at[ki, "net"], 2), book.at[ki, "date"]
        if (net, d) in matched_book_sig:
            book.at[ki, "recon_status"] = "Duplicate entry in cash book"
        elif (last_day - d).days <= 5:
            book.at[ki, "recon_status"] = "Deposit in transit" if net > 0 else "Outstanding cheque"
        else:
            book.at[ki, "recon_status"] = "In cash book only - investigate"
    return bank, book


def bank_rec_statement(bank, book, bank_recon, book_recon, opening):
    """Year-end bank reconciliation statement built from the engine output."""
    bank_close = opening + bank["net"].sum()
    book_close = opening + book["net"].sum()
    dit = book_recon.loc[book_recon.recon_status == "Deposit in transit", "net"].sum()
    oc = book_recon.loc[book_recon.recon_status == "Outstanding cheque", "net"].sum()
    bo = bank_recon[bank_recon.recon_status == "Bank only (not in cash book)"]
    bank_credits = bo.loc[bo.net > 0, "net"].sum(); bank_debits = bo.loc[bo.net < 0, "net"].sum()
    # amount errors: adjustment = bank amount - book amount (signed)
    amt = 0.0
    for _, r in bank_recon[bank_recon.recon_status == "Amount difference"].iterrows():
        ks = book_recon[book_recon.bank_sn == str(r["sn"])]
        amt += r["net"] - ks["net"].sum()
    dup = -book_recon.loc[book_recon.recon_status == "Duplicate entry in cash book", "net"].sum()
    rows = [
        ("BANK SIDE", None),
        ("Balance per bank statement", bank_close),
        ("Add: deposits in transit", dit),
        ("Less: outstanding cheques", oc),
        ("Adjusted bank balance", bank_close + dit + oc),
        ("CASH BOOK SIDE", None),
        ("Balance per cash book", book_close),
        ("Add: bank credits not in cash book", bank_credits),
        ("Less: bank debits not in cash book", bank_debits),
        ("Correct: amount errors in cash book", amt),
        ("Add back: duplicate entries in cash book", dup),
        ("Adjusted cash book balance", book_close + bank_credits + bank_debits + amt + dup),
    ]
    diff = round((bank_close + dit + oc) - (book_close + bank_credits + bank_debits + amt + dup), 2)
    return pd.DataFrame(rows, columns=["Item", "Amount (NGN)"]), diff


# ----------------------------------------------------------------------------- summaries
CLASS_OF = {}
_INV = {"Equipment & Vehicle Purchases", "Asset Disposal Proceeds"}
_FIN = {"Loan Proceeds", "Loan Repayment - Principal + Interest", "Loan Facility Fees", "Owner Capital Injection",
        "Director Loan Received", "Director Loan Repaid", "Owner Drawings"}
UNCLASSIFIED = "Unclassified"
_OTH = {"Transfer Between Own Accounts", "Bank Reversal / Error", "Unidentified - Suspense", UNCLASSIFIED}


def fs_class(cat):
    if cat in _INV: return "Investing"
    if cat in _FIN: return "Financing"
    if cat in _OTH: return "Other / Suspense"
    return "Operating"


def monthly_summary(classified):
    d = classified.copy()
    d["fs_class"] = d["category"].map(fs_class)
    piv = d.pivot_table(index=["fs_class", "category"], columns="month", values="net", aggfunc="sum", fill_value=0.0)
    piv["Total"] = piv.sum(axis=1)
    return piv


def cash_flow_statement(classified, opening):
    d = classified.copy(); d["fs_class"] = d["category"].map(fs_class)
    by = d.groupby("fs_class")["net"].sum()
    out = pd.DataFrame({"Item": ["Net cash from operating activities", "Net cash from investing activities",
                                 "Net cash from financing activities", "Other / suspense items (unresolved)",
                                 "Net movement in cash", "Opening cash balance", "Closing cash balance"],
                        "Amount (NGN)": [by.get("Operating", 0), by.get("Investing", 0), by.get("Financing", 0),
                                         by.get("Other / Suspense", 0), by.sum(), opening, opening + by.sum()]})
    return out


def evaluate_classification(classified, key, test_mask=None):
    m = classified.merge(key, on="sn")
    if test_mask is not None:
        m = m[m["sn"].isin(classified.loc[test_mask, "sn"])]
    m["correct"] = m["category"] == m["true_category"]
    overall = m["correct"].mean()
    by_clarity = m.groupby("clarity")["correct"].agg(["mean", "count"]).rename(columns={"mean": "accuracy", "count": "n"})
    by_method = m.groupby("method")["correct"].agg(["mean", "count"]).rename(columns={"mean": "accuracy", "count": "n"})
    return overall, by_clarity, by_method, m


def evaluate_reconciliation(bank_recon, key):
    m = bank_recon.merge(key[["sn", "true_status"]], on="sn")
    m["correct"] = m["recon_status"] == m["true_status"]
    return m["correct"].mean(), m


def opening_balance(bank):
    """Opening balance implied by the first statement line."""
    return float(round(bank["balance"].iloc[0] - bank["net"].iloc[0], 2))


def apply_corrections(classified, corrections):
    """Overwrite predictions with human decisions {sn: category}."""
    out = classified.copy()
    for sn, cat in corrections.items():
        m = out["sn"] == sn
        out.loc[m, "category"] = cat
        out.loc[m, "confidence"] = 1.0
        out.loc[m, "method"] = "Human"
        out.loc[m, "needs_review"] = False
    return out


# ----------------------------------------------------------------------------- importing your own data
BANK_HINTS = {
    "date": ["trans date", "transaction date", "posting date", "date", "value date"],
    "narration": ["narration", "description", "details", "particulars", "remarks", "transaction details"],
    "debit": ["debit", "withdrawal", "withdrawals", "money out", "paid out", "dr"],
    "credit": ["credit", "deposit", "deposits", "money in", "paid in", "cr"],
    "amount": ["amount"],
    "balance": ["balance", "running balance"],
    "reference": ["reference", "ref", "trans ref", "cheque no"],
}
BOOK_HINTS = {
    "date": ["date", "trans date", "transaction date"],
    "description": ["description", "narration", "details", "particulars"],
    "receipt": ["receipts", "receipt", "money in", "debit", "dr"],
    "payment": ["payments", "payment", "money out", "credit", "cr"],
    "amount": ["amount"],
}
_HEADER_WORDS = ("date", "narration", "description", "details", "particulars", "debit", "credit", "amount", "balance",
                 "receipt", "payment", "withdraw", "deposit")


def read_table(file, filename=""):
    """Read a CSV/Excel file and auto-detect the header row (skips title/preamble rows)."""
    name = (filename or getattr(file, "name", "")).lower()
    if name.endswith(".csv"):
        import csv, io
        data = file.read() if hasattr(file, "read") else open(file, "rb").read()
        if isinstance(data, bytes):
            data = data.decode("utf-8-sig", errors="replace")
        rows = list(csv.reader(io.StringIO(data)))
        width = max((len(r) for r in rows), default=0)
        raw = pd.DataFrame([r + [""] * (width - len(r)) for r in rows], dtype=str)
    else:
        raw = pd.read_excel(file, header=None, dtype=str)
    raw = raw.fillna("")
    hdr = 0
    for i in range(min(len(raw), 30)):
        cells = [str(c).strip().lower() for c in raw.iloc[i]]
        if sum(1 for c in cells if any(w in c for w in _HEADER_WORDS)) >= 2:
            hdr = i; break
    df = raw.iloc[hdr + 1:].copy()
    cols, seen = [], {}
    for j, c in enumerate(raw.iloc[hdr]):
        c = str(c).strip() or f"Column {j + 1}"
        seen[c] = seen.get(c, 0) + 1
        cols.append(c if seen[c] == 1 else f"{c} ({seen[c]})")
    df.columns = cols
    return df.reset_index(drop=True)


def guess_columns(columns, hints):
    """Best-guess mapping {field: column name or None} from header names."""
    low = {c: str(c).strip().lower() for c in columns}
    out = {}
    for field, words in hints.items():
        out[field] = None
        for w in words:                       # exact header match first
            hit = [c for c, l in low.items() if l == w]
            if hit: out[field] = hit[0]; break
        if out[field] is None:
            for w in words:                   # then 'contains'
                hit = [c for c, l in low.items() if w in l and len(w) > 3]
                if hit: out[field] = hit[0]; break
    return out


def _num(s):
    """Parse money text to floats without regular expressions (works on every pandas/pyarrow version).
    Handles 1,250,000.00 | N1,250.00 | NGN 500 | (500.00) as negative | blanks."""
    def conv(x):
        t = str(x).strip()
        if t == "" or t.lower() in ("nan", "none", "nat", "-"):
            return 0.0
        neg = (t.startswith("(") and t.endswith(")")) or t.startswith("-")
        t = "".join(ch for ch in t if ch.isdigit() or ch == ".")
        if t in ("", "."):
            return 0.0
        try:
            v = float(t)
        except ValueError:
            return 0.0
        return -v if neg else v
    return s.map(conv).astype(float)


def _dates(series):
    """Parse dates: ISO (2025-03-10) as year-month-day, everything else day-first (10/03/2025)."""
    x = series.astype(str).str.strip()
    iso = x.map(lambda t: len(t) >= 10 and t[:4].isdigit() and t[4] == "-" and t[7] == "-")
    out = pd.Series(pd.NaT, index=x.index, dtype="datetime64[ns]")
    if iso.any():
        out[iso] = pd.to_datetime(x[iso].str[:10], format="%Y-%m-%d", errors="coerce")
    if (~iso).any():
        out[~iso] = pd.to_datetime(x[~iso], dayfirst=True, errors="coerce", format="mixed")
    return out


def standardise_bank(raw, m):
    """Convert a raw bank table + column mapping into the engine's bank format."""
    df = pd.DataFrame({"date": _dates(raw[m["date"]]),
                       "narration": raw[m["narration"]].astype(str).str.strip()})
    if m.get("debit") and m.get("credit"):
        df["debit"] = _num(raw[m["debit"]]).abs(); df["credit"] = _num(raw[m["credit"]]).abs()
    elif m.get("amount"):
        a = _num(raw[m["amount"]]); df["credit"] = a.clip(lower=0); df["debit"] = (-a).clip(lower=0)
    else:
        raise ValueError("Map either Debit and Credit columns, or a single Amount column.")
    df["reference"] = raw[m["reference"]].astype(str) if m.get("reference") else ""
    df["balance"] = _num(raw[m["balance"]]) if m.get("balance") else np.nan
    df = df[df["date"].notna() & (df["narration"] != "") & ((df["debit"] > 0) | (df["credit"] > 0))].reset_index(drop=True)
    df.insert(0, "sn", range(1, len(df) + 1))
    return _clean_bank(df)


def standardise_book(raw, m):
    """Convert a raw cash book table + column mapping into the engine's cash book format."""
    df = pd.DataFrame({"date": _dates(raw[m["date"]]),
                       "description": raw[m["description"]].astype(str).str.strip()})
    if m.get("receipt") and m.get("payment"):
        df["receipt"] = _num(raw[m["receipt"]]).abs(); df["payment"] = _num(raw[m["payment"]]).abs()
    elif m.get("amount"):
        a = _num(raw[m["amount"]]); df["receipt"] = a.clip(lower=0); df["payment"] = (-a).clip(lower=0)
    else:
        raise ValueError("Map either Receipt and Payment columns, or a single Amount column.")
    df = df[df["date"].notna() & ((df["receipt"] > 0) | (df["payment"] > 0))].reset_index(drop=True)
    df.insert(0, "ref", [f"CB-{i:04d}" for i in range(1, len(df) + 1)])
    return _clean_book(df)


# ----------------------------------------------------------------------------- bank-only mode (no cash book)
def empty_book():
    """An empty cash book, used when the user has no cash book to upload."""
    return _clean_book(pd.DataFrame({"ref": [], "date": [], "description": [], "receipt": [], "payment": []}))


def bank_only_checks(bank):
    """Integrity checks that need no cash book: possible duplicates and running-balance breaks."""
    out = {}
    dup = bank[bank.duplicated(["date", "net"], keep=False)].sort_values(["date", "net", "sn"])
    out["duplicates"] = dup[["sn", "date", "narration", "debit", "credit"]]
    if "balance" in bank and bank["balance"].notna().all():
        expected = bank["balance"].iloc[0] + bank["net"].iloc[1:].cumsum().reindex(bank.index, fill_value=0.0)
        brk = bank[(expected - bank["balance"]).abs() > 0.01]
        out["balance_breaks"] = brk[["sn", "date", "narration", "debit", "credit", "balance"]]
    return out


# ----------------------------------------------------------------------------- self-training (learn only from the uploaded statement)
def self_train_model(base, min_conf=0.90, min_rows=10):
    """Train a text model on THIS statement's own confident classifications
    (keyword-rule hits and human corrections). Returns None if there is too little to learn from."""
    tr = base[base["confidence"] >= min_conf]
    if len(tr) < min_rows or tr["category"].nunique() < 2:
        return None
    return train_model(tr, tr["category"])
