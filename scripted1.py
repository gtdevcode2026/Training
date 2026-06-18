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
    Reads which zone the dropped file is for (from its 'Zone' column) and drops
    zone rows with blank/'@'-less/'noemail@' emails. Each remaining zone row is
    matched to the master with a CASCADE key — tried in priority order, stopping
    at the first that hits:
        Employee Email  ->  Global Employee ID  ->  Local Employee ID
    Matching is global (a zone person is found even if they live in a different
    zone in the master). The zone row's 'Action' column then drives the outcome:
      * Action == "ok" (case-insensitive):
          - matched   -> that master row's status becomes "validated"
                         (anywhere in the master, regardless of its Zone)
          - unmatched -> the person is appended to the master as "newly added"
      * Action is a non-blank value other than "ok":
          - matched   -> that master row is DELETED from the master
          - unmatched -> no-op
      * Action blank (or no Action column on a single row): no-op. If the zone
        file has NO Action column at all, every row is treated as "ok".
    Master rows of the reconciled zone that no "ok" row matched (and that were
    not deleted) become "not validated". Rows of other zones are left untouched
    unless a cross-zone match validated or deleted them.
    Finally, duplicate emails are removed from the master (keep first). The
    master is updated in place, with a timestamped backup taken before each
    change.
"""

import argparse
import os
import sys
import shutil
import tempfile
import time
from typing import List, Optional

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
LOCAL_ID = "Local Employee ID"
ZONE_COL = "Zone"
ACTION_COL = "Action"          # zone-file column; "ok" => validate/add, other => delete
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


def _norm_email(v) -> str:
    """Normalize an email for matching / dedupe: trimmed + lower-cased; '' if NA."""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v).strip().lower()


def _norm_local(v) -> str:
    """Normalize a Local Employee ID for matching: trimmed + lower-cased.

    Local ids are strings like 'EUR-600101', so (unlike Global IDs) they are not
    run through the float-stripping logic."""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v).strip().lower()


def build_index_map(norm_series: Optional[pd.Series]) -> dict:
    """Map each non-empty normalized value -> list of master index labels."""
    out: dict = {}
    if norm_series is None:
        return out
    for idx, key in norm_series.items():
        if key:
            out.setdefault(key, []).append(idx)
    return out


def cascade_match(z_email: str, z_gid: str, z_lid: str,
                  emap: dict, gmap: dict, lmap: dict) -> list:
    """Find matching master index labels for one zone row, trying keys in
    priority order and STOPPING at the first tier that yields any hit:
    Employee Email -> Global Employee ID -> Local Employee ID."""
    if z_email and z_email in emap:
        return emap[z_email]
    if z_gid and z_gid in gmap:
        return gmap[z_gid]
    if z_lid and z_lid in lmap:
        return lmap[z_lid]
    return []


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

    # 2) Load master, ensure required columns + status column. The native
    #    RangeIndex is kept stable: every match below works on these labels,
    #    and the index is only reset at the very end.
    master = load_table(master_path)
    for col in (KEY_ID, KEY_EMAIL, ZONE_COL):
        if col not in master.columns:
            raise ValueError(f"Master is missing required column: '{col}'")
    if STATUS_COLUMN not in master.columns:
        master[STATUS_COLUMN] = ""
    else:
        master[STATUS_COLUMN] = master[STATUS_COLUMN].fillna("")

    # 3) Load zone file; require id + email; drop invalid/placeholder emails.
    zone = load_table(zone_path)
    for col in (KEY_ID, KEY_EMAIL):
        if col not in zone.columns:
            raise ValueError(f"Zone file is missing required column: '{col}'")

    zmask = valid_email_mask(zone)
    dropped_invalid = int((~zmask).sum())
    zone = zone[zmask].reset_index(drop=True)

    # 4) Resolve which zone this file is for (uses the Zone column, independent
    #    of Action -> resolve from the full email-valid frame).
    resolved_zone = resolve_zone(zone, zone_override)

    # 5) Split zone rows by Action:
    #      "ok" (case-insensitive)                -> validate if matched, else add
    #      a non-blank value other than "ok"      -> delete the matched master row
    #      blank Action / no Action column        -> 'ok' if no column, else no-op
    ok_mask = action_ok_mask(zone)
    if ok_mask is None:
        ok_rows = zone
        nonok_rows = zone.iloc[0:0]
        blank_action = 0
    else:
        action_col = find_action_column(zone)
        action_vals = zone[action_col].fillna("").astype(str).str.strip()
        is_blank = action_vals.eq("")
        ok_rows = zone[ok_mask]
        nonok_rows = zone[(~ok_mask) & (~is_blank)]
        blank_action = int(((~ok_mask) & is_blank).sum())

    # 6) Build cascade lookups from the ORIGINAL master (value -> index labels).
    emap = build_index_map(master[KEY_EMAIL].map(_norm_email))
    gmap = build_index_map(master[KEY_ID].map(_norm_id))
    lmap = build_index_map(master[LOCAL_ID].map(_norm_local)) if LOCAL_ID in master.columns else {}
    has_zone_lid = LOCAL_ID in zone.columns

    def zkeys(row):
        e = _norm_email(row[KEY_EMAIL])
        g = _norm_id(row[KEY_ID])
        l = _norm_local(row[LOCAL_ID]) if has_zone_lid else ""
        return e, g, l

    # 7) Cascade-match. ok rows -> validate (if hit) else newly-added;
    #    non-ok rows -> delete the matched master row(s).
    validated_idx = set()
    newly_added_pos = []
    for pos, (_, row) in enumerate(ok_rows.iterrows()):
        e, g, l = zkeys(row)
        hits = cascade_match(e, g, l, emap, gmap, lmap)
        if hits:
            validated_idx.update(hits)
        else:
            newly_added_pos.append(pos)

    delete_idx = set()
    for _, row in nonok_rows.iterrows():
        e, g, l = zkeys(row)
        delete_idx.update(cascade_match(e, g, l, emap, gmap, lmap))

    # Deletion wins over validation if a master row is targeted by both.
    validated_idx -= delete_idx

    # 8) Apply statuses on the original index labels (before any drop/append).
    if validated_idx:
        master.loc[list(validated_idx), STATUS_COLUMN] = STATUS_VALIDATED

    zone_norm = str(resolved_zone).strip().lower()
    in_zone = master[ZONE_COL].fillna("").astype(str).str.strip().str.lower() == zone_norm
    nv_idx = (set(master.index[in_zone]) - validated_idx) - delete_idx
    if nv_idx:
        master.loc[list(nv_idx), STATUS_COLUMN] = STATUS_NOT_VALIDATED

    # 9) Delete the non-ok matched rows from the master.
    if delete_idx:
        master = master.drop(index=list(delete_idx))

    # 10) Append newly-added rows (ok rows that matched nothing) as-is.
    new_rows = ok_rows.iloc[newly_added_pos] if newly_added_pos else ok_rows.iloc[0:0]
    appended = build_appended_rows(new_rows, master.columns, resolved_zone)
    master = pd.concat([master, appended], ignore_index=True)

    # 11) FINAL step: remove duplicate emails from the master (keep first;
    #     blank emails are exempt so they don't all collapse into one row).
    email_norm = master[KEY_EMAIL].fillna("").astype(str).str.strip().str.lower()
    dup_mask = email_norm.duplicated(keep="first") & email_norm.ne("")
    dedupe_removed = int(dup_mask.sum())
    if dedupe_removed:
        master = master[~dup_mask].reset_index(drop=True)

    # 12) Write back in place + summary.
    write_table_inplace(master, master_path)

    print(f"OK: reconciled zone '{resolved_zone}'. Backup saved as: {backup_path}")
    print(f"   validated       : {len(validated_idx)}")
    print(f"   not validated   : {len(nv_idx)}")
    print(f"   newly added     : {int(len(appended))}")
    print(f"   deleted (master): {len(delete_idx)}")
    if dedupe_removed:
        print(f"   duplicate emails removed: {dedupe_removed}")
    if dropped_invalid:
        print(f"   (zone rows skipped for invalid/placeholder email: {dropped_invalid})")
    if blank_action:
        print(f"   (zone rows with blank Action skipped as no-op: {blank_action})")


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
