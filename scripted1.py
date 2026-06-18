#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adaptive HR cleaner + per-zone reconciler.

Run ONCE PER ZONE, manually:

    python hr_clean_and_reconcile.py <master_file> <zone_file> [--zone NAME]
           [--no-clean] [--force-clean] [--chunksize N] [--parallel off|auto|N]

What it does
------------
Step 1 (adaptive clean) — runs ONLY when the master needs it:
    Keeps the 18 Required columns, drops rows with blank / '@'-less / 'noemail@'
    emails. This runs automatically the FIRST time (when the master has no
    'status' column yet). On every later zone run it is skipped, so re-running
    never wipes the 'status' column or earlier zones' results.

Step 2 (reconcile one zone) — runs every time:
    Reads which zone the dropped file is for (from its 'Zone' column). Zone-file
    rows are first filtered: rows with blank/'@'-less/'noemail@' emails are
    dropped, and — if the zone file has an 'Action' column — rows whose Action
    is anything other than 'ok' (case-insensitive) are dropped too. It then
    compares the master against the remaining zone rows on (Global Employee ID
    AND Employee Email):
      * master row of that zone found in the zone file -> status "validated"
      * master row of that zone NOT in the zone file   -> status "not validated"
      * zone-file person not anywhere in the master    -> appended to the master
                                                          with status "newly added"
    Rows of other zones are left untouched. The master is updated in place,
    with a timestamped backup taken before each change.
"""

import argparse
import os
import sys
import shutil
import tempfile
import time
from typing import List, Optional, Tuple

import pandas as pd

# -----------------------------
# Config: REQUIRED columns ONLY
# -----------------------------
REQUIRED_COLUMNS: List[str] = [
    "Zone",
    "Country",
    "Global Employee ID",
    "Local Employee ID",
    "Employee Name",
    "Employee Status",
    "Worker Type",
    "Employee Group",
    "Management Level",
    "First Hire Date",
    "Last Hire Date",
    "Position Name",
    "Job Family Group",
    "Job Family",
    "Job Profile Description",
    "ABI Entity 2",
    "Macro Entity Level 2 (Zone)",
    "text before Email",
    "Employee Email",
    "Band 4+",
    "Manager Employee ID Level 01",
    "Manager Name Level 01",
]

# Column / status constants
KEY_ID = "Global Employee ID"
KEY_EMAIL = "Employee Email"
ZONE_COL = "Zone"
ACTION_COL = "Action"          # zone-file column; rows kept only when its value is "ok"
ACTION_OK = "ok"
STATUS_COLUMN = "status"
STATUS_VALIDATED = "validated"
STATUS_NOT_VALIDATED = "not validated"
STATUS_NEWLY_ADDED = "newly added"

EXCEL_EXTS = {".xlsx", ".xlsm"}


# -----------------------------
# REQUIRED for multiprocessing
# -----------------------------
def chunk_processor(df_chunk: pd.DataFrame) -> pd.DataFrame:
    return filter_required_and_emails(df_chunk)


def identity(df: pd.DataFrame) -> pd.DataFrame:
    return df


def safe_workers(user_value: Optional[str]) -> int:
    if user_value is None or str(user_value).lower() in {"off", "false", "0"}:
        return 0
    if str(user_value).lower() == "auto":
        cpu = os.cpu_count() or 4
        return max(1, cpu - 2)
    try:
        n = int(user_value)
        return max(0, n)
    except Exception:
        cpu = os.cpu_count() or 4
        return max(1, cpu - 2)


# =========================================================
# Step 1: cleaning logic (carried over, unchanged behavior)
# =========================================================
def filter_required_and_emails(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [c for c in REQUIRED_COLUMNS if c in df.columns]
    if not keep_cols:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)

    df = df[keep_cols].copy()

    if KEY_EMAIL not in df.columns:
        return df.iloc[0:0]

    mask = valid_email_mask(df)
    return df[mask]


def valid_email_mask(df: pd.DataFrame) -> pd.Series:
    """Email-validity rule shared by Step 1 and the zone reconcile step.

    Keeps a row only if the email is non-blank, contains '@', and is not a
    'noemail@' placeholder (case-insensitive)."""
    if KEY_EMAIL not in df.columns:
        return pd.Series([False] * len(df), index=df.index)
    email = df[KEY_EMAIL].fillna("").astype(str).str.strip()
    return (
        email.ne("")
        & email.str.contains("@")
        & ~email.str.contains(r"noemail@", case=False)
    )


def find_action_column(df: pd.DataFrame) -> Optional[str]:
    """Return the zone file's action column name (case/space-insensitive), or
    None if the file has no such column."""
    for c in df.columns:
        if str(c).strip().lower() == ACTION_COL.lower():
            return c
    return None


def action_ok_mask(df: pd.DataFrame) -> Optional[pd.Series]:
    """Mask of rows whose action cell equals 'ok' (case-insensitive, trimmed).

    Returns None when the file has no action column, signalling 'no filter'."""
    col = find_action_column(df)
    if col is None:
        return None
    return df[col].fillna("").astype(str).str.strip().str.lower().eq(ACTION_OK)


def process_csv_inplace(
    input_path: str,
    chunksize: int = 200_000,
    parallel: int = 0,
) -> None:
    dirname, basename = os.path.split(input_path)
    root, ext = os.path.splitext(basename)
    backup_path = backup_file(input_path)

    tmp_fd, tmp_path = tempfile.mkstemp(prefix=f"{root}.tmp-", suffix=ext, dir=dirname)
    os.close(tmp_fd)

    read_kwargs = dict(sep=None, engine="python", encoding="utf-8-sig", chunksize=chunksize)

    if parallel > 0:
        from concurrent.futures import ProcessPoolExecutor

        first = True
        in_flight = {}
        next_write_idx = 0
        max_in_flight = max(2, parallel * 3)

        with pd.read_csv(input_path, **read_kwargs) as reader, \
                ProcessPoolExecutor(max_workers=parallel) as pool:

            for idx, df_chunk in enumerate(reader):
                while len(in_flight) >= max_in_flight:
                    done = [k for k, f in in_flight.items() if f.done()]
                    for k in sorted(done):
                        res = in_flight.pop(k).result()
                        if k == next_write_idx:
                            res.to_csv(tmp_path, mode="a", index=False, header=first)
                            first = False
                            next_write_idx += 1
                            while next_write_idx in in_flight and in_flight[next_write_idx].done():
                                res2 = in_flight.pop(next_write_idx).result()
                                res2.to_csv(tmp_path, mode="a", index=False, header=first)
                                first = False
                                next_write_idx += 1
                        else:
                            in_flight[k] = pool.submit(identity, res)

                in_flight[idx] = pool.submit(chunk_processor, df_chunk)

            while next_write_idx in in_flight:
                fut = in_flight.pop(next_write_idx)
                res = fut.result()
                res.to_csv(tmp_path, mode="a", index=False, header=first)
                first = False
                next_write_idx += 1

    else:
        first = True
        for df_chunk in pd.read_csv(input_path, **read_kwargs):
            cleaned = filter_required_and_emails(df_chunk)
            cleaned.to_csv(tmp_path, mode="a", index=False, header=first)
            first = False

    os.replace(tmp_path, input_path)
    print(f"   cleaned CSV written in-place. Backup saved as: {backup_path}")


def process_excel_inplace(input_path: str) -> None:
    backup_path = backup_file(input_path)
    df = pd.read_excel(input_path, engine="openpyxl")
    cleaned = filter_required_and_emails(df)
    write_table_inplace(cleaned, input_path)
    print(f"   cleaned Excel written in-place. Backup saved as: {backup_path}")


def run_clean(master_path: str, chunksize: int, parallel_arg: str) -> None:
    warn_missing_keep_columns(master_path)
    ext = os.path.splitext(master_path)[1].lower()
    if ext == ".csv":
        workers = safe_workers(parallel_arg)
        print(f"   processing CSV in chunks (chunksize={chunksize}, parallel_workers={workers}) ...")
        process_csv_inplace(master_path, chunksize=chunksize, parallel=workers)
    elif ext in EXCEL_EXTS:
        print("   processing Excel ...")
        process_excel_inplace(master_path)
    else:
        print(f"ERROR: Unsupported master file type: {ext}")
        sys.exit(2)


# =========================================================
# Shared file I/O helpers
# =========================================================
def backup_file(path: str) -> str:
    dirname, basename = os.path.split(path)
    root, ext = os.path.splitext(basename)
    ts = time.strftime("%Y%m%d-%H%M%S")
    backup_path = os.path.join(dirname, f"{root}.backup-{ts}{ext}")
    shutil.copy2(path, backup_path)
    return backup_path


def read_header_columns(path: str) -> List[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", nrows=0)
    elif ext in EXCEL_EXTS:
        df = pd.read_excel(path, engine="openpyxl", nrows=0)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return list(df.columns)


def warn_missing_keep_columns(path: str) -> None:
    """Warn if any expected keep-list column is absent from the file, so a
    typo'd / renamed header doesn't silently vanish from the cleaned output."""
    try:
        cols = set(read_header_columns(path))
    except Exception:
        return
    missing = [c for c in REQUIRED_COLUMNS if c not in cols]
    if missing:
        print("WARNING: these expected columns were NOT found in the master "
              "and will be absent from the cleaned output:")
        for c in missing:
            print(f"   - {c}")


def load_table(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".csv":
        # Read everything as strings to keep IDs/emails stable for matching.
        return pd.read_csv(
            path, sep=None, engine="python", encoding="utf-8-sig",
            dtype=str, keep_default_na=False,
        )
    elif ext in EXCEL_EXTS:
        return pd.read_excel(path, engine="openpyxl")
    raise ValueError(f"Unsupported file type: {ext}")


def write_table_inplace(df: pd.DataFrame, path: str) -> None:
    dirname, basename = os.path.split(path)
    root, ext = os.path.splitext(basename)
    ext_l = ext.lower()
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=f"{root}.tmp-", suffix=ext, dir=dirname)
    os.close(tmp_fd)
    if ext_l == ".csv":
        df.to_csv(tmp_path, index=False, encoding="utf-8-sig")
    elif ext_l in EXCEL_EXTS:
        df.to_excel(tmp_path, index=False, engine="openpyxl")
    else:
        os.remove(tmp_path)
        raise ValueError(f"Unsupported file type: {ext_l}")
    os.replace(tmp_path, path)


# =========================================================
# Matching helpers
# =========================================================
def _norm_id(v) -> str:
    """Normalize an employee id to a stable string.

    Handles the common Excel quirk where numeric ids load as floats
    (e.g. 12345.0) by stripping the trailing '.0'."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    s = str(v).strip()
    if s.endswith(".0") and s[:-2].isdigit():
        s = s[:-2]
    return s


def normalize_key(emp_id, email) -> Tuple[str, str]:
    """Match key: (normalized Global Employee ID, lower-cased Employee Email)."""
    sid = _norm_id(emp_id)
    if email is None:
        semail = ""
    else:
        try:
            semail = "" if pd.isna(email) else str(email).strip().lower()
        except (TypeError, ValueError):
            semail = str(email).strip().lower()
    return (sid, semail)


def make_keys(df: pd.DataFrame) -> pd.Series:
    if len(df) == 0:
        return pd.Series([], dtype=object, index=df.index)
    return df.apply(lambda r: normalize_key(r[KEY_ID], r[KEY_EMAIL]), axis=1)


def dedupe_by_key(df: pd.DataFrame, key_series: pd.Series, label: str):
    dup_mask = key_series.duplicated(keep="first")
    n = int(dup_mask.sum())
    if n:
        print(f"WARNING: {label}: {n} duplicate (Global Employee ID, Employee Email) "
              f"row(s) found; keeping first.")
    return df[~dup_mask].reset_index(drop=True), key_series[~dup_mask].reset_index(drop=True)


# =========================================================
# Step 1 trigger detection
# =========================================================
def is_master_prepared(path: str) -> bool:
    """A master is 'prepared' once it has a 'status' column (i.e. Step 1 + at
    least one reconcile have already run)."""
    try:
        cols = read_header_columns(path)
    except Exception:
        return False
    return STATUS_COLUMN in cols


# =========================================================
# Step 2: reconcile one zone into the master, in place
# =========================================================
def resolve_zone(zone_df: pd.DataFrame, zone_override: Optional[str]) -> str:
    if zone_override:
        return str(zone_override).strip()
    if ZONE_COL not in zone_df.columns:
        raise ValueError("Zone file has no 'Zone' column; pass --zone NAME.")
    vals = zone_df[ZONE_COL].fillna("").astype(str).str.strip()
    uniq = sorted({v for v in vals if v != ""})
    if len(uniq) == 0:
        raise ValueError("Zone file's 'Zone' column is empty; pass --zone NAME.")
    if len(uniq) > 1:
        raise ValueError(f"Zone file contains multiple zones {uniq}; pass --zone NAME.")
    return uniq[0]


def build_appended_rows(new_rows: pd.DataFrame, master_columns, resolved_zone: str) -> pd.DataFrame:
    """Align zone-file 'newly added' rows to the master's columns."""
    if new_rows.empty:
        return pd.DataFrame(columns=master_columns)

    data = {}
    for col in master_columns:
        if col in new_rows.columns:
            data[col] = list(new_rows[col].values)
        else:
            data[col] = [""] * len(new_rows)
    out = pd.DataFrame(data, columns=list(master_columns))

    # Clean up ids for newly-added rows so they don't carry a trailing '.0'.
    if KEY_ID in out.columns:
        out[KEY_ID] = out[KEY_ID].map(_norm_id)

    # Fill the Zone for any new row that didn't carry one.
    if ZONE_COL in out.columns:
        zser = out[ZONE_COL].fillna("").astype(str).str.strip()
        out.loc[zser.eq(""), ZONE_COL] = resolved_zone

    out[STATUS_COLUMN] = STATUS_NEWLY_ADDED
    return out


def reconcile_zone_inplace(master_path: str, zone_path: str, zone_override: Optional[str] = None) -> None:
    # 1) Back up the master before any change.
    backup_path = backup_file(master_path)

    # 2) Load master, ensure required columns + status column.
    master = load_table(master_path)
    for col in (KEY_ID, KEY_EMAIL, ZONE_COL):
        if col not in master.columns:
            raise ValueError(f"Master is missing required column: '{col}'")
    if STATUS_COLUMN not in master.columns:
        master[STATUS_COLUMN] = ""
    else:
        master[STATUS_COLUMN] = master[STATUS_COLUMN].fillna("")

    # 3) Load zone file; require id + email; drop invalid emails.
    zone = load_table(zone_path)
    for col in (KEY_ID, KEY_EMAIL):
        if col not in zone.columns:
            raise ValueError(f"Zone file is missing required column: '{col}'")

    zmask = valid_email_mask(zone)
    dropped_invalid = int((~zmask).sum())
    zone = zone[zmask].reset_index(drop=True)

    # 3b) Drop rows whose 'Action' is anything other than 'ok' (if the column
    #     exists; otherwise keep all rows).
    amask = action_ok_mask(zone)
    if amask is None:
        dropped_action = 0
    else:
        dropped_action = int((~amask).sum())
        zone = zone[amask].reset_index(drop=True)

    # 4) Resolve which zone this file is for.
    resolved_zone = resolve_zone(zone, zone_override)

    # 5) Keys on both sides; dedupe zone file by key.
    zone_key_all = make_keys(zone)
    zone, zone_key = dedupe_by_key(zone, zone_key_all, "Zone file")
    master_key = make_keys(master)

    zone_key_set = set(zone_key)
    master_key_set = set(master_key)

    # 6) Scope master to the current zone; set validated / not validated.
    zone_norm = str(resolved_zone).strip().lower()
    master_zone_mask = (
        master[ZONE_COL].fillna("").astype(str).str.strip().str.lower() == zone_norm
    )
    in_zone_file = master_key.isin(zone_key_set)

    validated_mask = master_zone_mask & in_zone_file
    not_validated_mask = master_zone_mask & ~in_zone_file

    master.loc[validated_mask, STATUS_COLUMN] = STATUS_VALIDATED
    master.loc[not_validated_mask, STATUS_COLUMN] = STATUS_NOT_VALIDATED

    # 7) Newly added: zone people not present anywhere in the master.
    new_rows_mask = ~zone_key.isin(master_key_set)
    new_rows = zone[new_rows_mask].reset_index(drop=True)
    appended = build_appended_rows(new_rows, master.columns, resolved_zone)
    if not appended.empty:
        master = pd.concat([master, appended], ignore_index=True)

    # 8) Write back in place + summary.
    write_table_inplace(master, master_path)

    n_validated = int(validated_mask.sum())
    n_not_validated = int(not_validated_mask.sum())
    n_added = int(len(appended))
    print(f"OK: reconciled zone '{resolved_zone}'. Backup saved as: {backup_path}")
    print(f"   validated      : {n_validated}")
    print(f"   not validated  : {n_not_validated}")
    print(f"   newly added    : {n_added}")
    if dropped_invalid:
        print(f"   (zone rows skipped for invalid/placeholder email: {dropped_invalid})")
    if dropped_action:
        print(f"   (zone rows skipped for Action != 'ok': {dropped_action})")


# =========================================================
# CLI
# =========================================================
def main():
    ap = argparse.ArgumentParser(
        description="Adaptive HR cleaner + per-zone reconciler (run once per zone)."
    )
    ap.add_argument("master_file", help="Master CSV/Excel (raw on first run; prepared after).")
    ap.add_argument("zone_file", help="The zone's CSV/Excel to reconcile against the master.")
    ap.add_argument("--zone", default=None,
                    help="Override the zone name (else read from the zone file's 'Zone' column).")
    ap.add_argument("--no-clean", action="store_true",
                    help="Never run Step 1 cleaning, even if the master looks raw.")
    ap.add_argument("--force-clean", action="store_true",
                    help="Force Step 1 cleaning even if the master already has a 'status' column.")
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--parallel", default="off")
    args = ap.parse_args()

    if args.no_clean and args.force_clean:
        print("ERROR: --no-clean and --force-clean are mutually exclusive.")
        sys.exit(2)

    master_path = os.path.abspath(args.master_file)
    zone_path = os.path.abspath(args.zone_file)
    if not os.path.exists(master_path):
        print(f"ERROR: Master not found: {master_path}")
        sys.exit(1)
    if not os.path.exists(zone_path):
        print(f"ERROR: Zone file not found: {zone_path}")
        sys.exit(1)

    prepared = is_master_prepared(master_path)

    # Decide whether Step 1 runs.
    if args.force_clean:
        do_clean = True
    elif args.no_clean:
        do_clean = False
    else:
        do_clean = not prepared

    if do_clean:
        print("-> Step 1: cleaning master (first-time preparation) ...")
        run_clean(master_path, chunksize=args.chunksize, parallel_arg=args.parallel)
    else:
        if not prepared and args.no_clean:
            print("-> Step 1 skipped (--no-clean): master has no 'status' column; "
                  "reconciling it as-is.")
        else:
            print("-> Step 1 skipped (master already prepared).")

    print("-> Step 2: reconciling zone file against master ...")
    try:
        reconcile_zone_inplace(master_path, zone_path, zone_override=args.zone)
    except ValueError as e:
        print(f"ERROR: {e}")
        sys.exit(3)


if __name__ == "__main__":
    main()
