# references/utils_library.md
# Complete ssis_utils.py — one function per SSIS component
# Generated file location: utils/ssis_utils.py

---

## Table of Contents

1. [Sources](#sources) — OLE DB, Flat File, Excel, XML, ADO.NET
2. [Destinations](#destinations) — OLE DB, Flat File, Excel, Raw File
3. [Row Transforms](#row-transforms) — Derived Column, Data Conversion, Copy Column, Audit
4. [Join / Lookup Transforms](#join--lookup-transforms) — Lookup, Merge Join, Merge
5. [Set Operations](#set-operations) — Union All, Multicast
6. [Split / Route](#split--route) — Conditional Split
7. [Aggregation & Sort](#aggregation--sort) — Aggregate, Sort, Row Count
8. [Reshape](#reshape) — Pivot, Unpivot
9. [Slowly Changing Dimensions](#slowly-changing-dimensions)
10. [Script Component](#script-component)
11. [Fuzzy Operations](#fuzzy-operations)
12. [Connection Helpers](#connection-helpers)

---

## Full ssis_utils.py

```python
"""
utils/ssis_utils.py
====================
Reusable utility functions mirroring SSIS Data Flow components.
Each function is independently testable and stateless.

Design contract:
  - Inputs:  pd.DataFrame + explicit keyword args
  - Outputs: pd.DataFrame (or None for sinks)
  - No hidden globals; connections passed explicitly as SQLAlchemy engines
  - Docstrings reference the SSIS component they replace
"""

from __future__ import annotations
import logging
import os
import hashlib
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
from sqlalchemy import Engine, text

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. SOURCES
# ═══════════════════════════════════════════════════════════════════════════════

def source_oledb(
    engine: Engine,
    query: str,
    params: Optional[dict] = None,
    chunksize: Optional[int] = None,
) -> pd.DataFrame:
    """
    SSIS Component: OLE DB Source / ADO NET Source
    ------------------------------------------------
    Executes a SQL query and returns a DataFrame.

    Args:
        engine:    SQLAlchemy engine for the source connection manager.
        query:     SQL SELECT statement (maps to SSIS SQLCommand property).
        params:    Optional bind parameters dict.
        chunksize: If set, reads in chunks and concatenates (large tables).

    Returns:
        pd.DataFrame with all result rows.
    """
    log.debug("source_oledb: %s", query[:120])
    if chunksize:
        chunks = pd.read_sql(query, engine, params=params, chunksize=chunksize)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_sql(query, engine, params=params)
    log.info("source_oledb: %d rows, %d cols", len(df), len(df.columns))
    return df


def source_flatfile(
    file_path: str,
    delimiter: str = ",",
    encoding: str = "utf-8",
    header_row: bool = True,
    skip_rows: int = 0,
    column_names: Optional[List[str]] = None,
    dtype: Optional[dict] = None,
    null_values: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    SSIS Component: Flat File Source
    ----------------------------------
    Reads a delimited text file into a DataFrame.

    Args:
        file_path:    Full path to the file.
        delimiter:    Column delimiter (maps to ColumnDelimiter property).
        encoding:     File encoding (maps to CodePage property).
        header_row:   True if first row contains column names.
        skip_rows:    Number of rows to skip at top (maps to HeaderRowsToSkip).
        column_names: Override column names if no header.
        dtype:        Per-column dtypes.
        null_values:  Strings to treat as NaN.
    """
    log.debug("source_flatfile: %s", file_path)
    df = pd.read_csv(
        file_path,
        sep=delimiter,
        encoding=encoding,
        header=0 if header_row else None,
        skiprows=skip_rows,
        names=column_names,
        dtype=dtype,
        na_values=null_values,
        low_memory=False,
    )
    log.info("source_flatfile: %d rows from %s", len(df), file_path)
    return df


def source_excel(
    file_path: str,
    sheet_name: Union[str, int] = 0,
    header_row: int = 0,
    skip_rows: int = 0,
    dtype: Optional[dict] = None,
    usecols: Optional[Union[str, List]] = None,
) -> pd.DataFrame:
    """
    SSIS Component: Excel Source
    ------------------------------
    Reads an Excel worksheet into a DataFrame.
    """
    log.debug("source_excel: %s [%s]", file_path, sheet_name)
    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=header_row,
        skiprows=skip_rows,
        dtype=dtype,
        usecols=usecols,
    )
    log.info("source_excel: %d rows", len(df))
    return df


def source_xml(
    file_path: str,
    xpath: str = ".",
    namespaces: Optional[dict] = None,
) -> pd.DataFrame:
    """
    SSIS Component: XML Source
    ---------------------------
    Parses an XML file using the given XPath into a DataFrame.
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(file_path)
    root = tree.getroot()
    records = []
    for elem in root.findall(xpath, namespaces or {}):
        record = {child.tag: child.text for child in elem}
        record.update(elem.attrib)
        records.append(record)
    df = pd.DataFrame(records)
    log.info("source_xml: %d rows from %s", len(df), file_path)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 2. DESTINATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def dest_oledb(
    df: pd.DataFrame,
    engine: Engine,
    table: str,
    schema: Optional[str] = None,
    if_exists: str = "append",
    chunksize: int = 1000,
    fast_executemany: bool = True,
) -> int:
    """
    SSIS Component: OLE DB Destination / ADO NET Destination
    ----------------------------------------------------------
    Writes a DataFrame to a SQL table.

    Args:
        df:               DataFrame to write.
        engine:           SQLAlchemy engine for the destination.
        table:            Target table name.
        schema:           Target schema (None = default schema).
        if_exists:        'append' | 'replace' | 'fail'
        chunksize:        Rows per insert batch (maps to DefaultBufferMaxRows).
        fast_executemany: Use bulk insert if supported.

    Returns:
        Number of rows written.
    """
    if df.empty:
        log.warning("dest_oledb: empty DataFrame, nothing written to %s", table)
        return 0
    df.to_sql(
        name=table,
        con=engine,
        schema=schema,
        if_exists=if_exists,
        index=False,
        chunksize=chunksize,
    )
    log.info("dest_oledb: %d rows → %s.%s", len(df), schema or "dbo", table)
    return len(df)


def dest_flatfile(
    df: pd.DataFrame,
    file_path: str,
    delimiter: str = ",",
    encoding: str = "utf-8",
    include_header: bool = True,
    write_mode: str = "w",
) -> None:
    """
    SSIS Component: Flat File Destination
    ---------------------------------------
    Writes a DataFrame to a delimited text file.
    """
    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    df.to_csv(
        file_path,
        sep=delimiter,
        encoding=encoding,
        header=include_header,
        index=False,
        mode=write_mode,
    )
    log.info("dest_flatfile: %d rows → %s", len(df), file_path)


def dest_excel(
    df: pd.DataFrame,
    file_path: str,
    sheet_name: str = "Sheet1",
    include_header: bool = True,
    if_sheet_exists: str = "replace",
) -> None:
    """
    SSIS Component: Excel Destination
    -----------------------------------
    Writes a DataFrame to an Excel worksheet.
    """
    with pd.ExcelWriter(file_path, engine="openpyxl", mode="a" if os.path.exists(file_path) else "w",
                        if_sheet_exists=if_sheet_exists) as writer:
        df.to_excel(writer, sheet_name=sheet_name, index=False, header=include_header)
    log.info("dest_excel: %d rows → %s [%s]", len(df), file_path, sheet_name)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ROW TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════════

def derived_column(
    df: pd.DataFrame,
    expressions: Dict[str, Union[Any, Callable]],
    replace_existing: bool = False,
) -> pd.DataFrame:
    """
    SSIS Component: Derived Column
    --------------------------------
    Adds new columns or replaces existing ones based on expressions.

    Args:
        df:               Input DataFrame.
        expressions:      Dict of {output_col_name: value_or_callable}.
                          Callable receives the row (pd.Series) or the full df.
                          Use lambda df: df["col1"] + df["col2"] for vectorised ops.
        replace_existing: If True, overwrite existing columns (SSIS "replace" mode).

    Example:
        df = derived_column(df, {
            "FullName": lambda df: df["First"] + " " + df["Last"],
            "LoadDate": pd.Timestamp.now(),
            "IsActive": True,
        })
    """
    df = df.copy()
    for col, expr in expressions.items():
        if callable(expr):
            df[col] = expr(df)
        else:
            df[col] = expr
    return df


def data_conversion(
    df: pd.DataFrame,
    conversions: Dict[str, Union[str, type]],
    error_behavior: str = "fail",
) -> pd.DataFrame:
    """
    SSIS Component: Data Conversion
    ---------------------------------
    Converts column data types.

    Args:
        df:              Input DataFrame.
        conversions:     Dict of {column_name: target_dtype}.
                         Supported: 'str', 'int', 'float', 'datetime', 'bool',
                                    'int32', 'int64', 'float64', numpy dtypes.
        error_behavior:  'fail' | 'ignore' | 'coerce' (for numeric/datetime).

    Example:
        df = data_conversion(df, {
            "OrderDate": "datetime",
            "Amount":    "float64",
            "Qty":       "int32",
        })
    """
    df = df.copy()
    for col, dtype in conversions.items():
        try:
            if dtype in ("datetime", "date"):
                df[col] = pd.to_datetime(df[col], errors=error_behavior if error_behavior != "fail" else "raise")
            elif dtype in ("int", "int32", "int64"):
                df[col] = pd.to_numeric(df[col], errors=error_behavior if error_behavior != "fail" else "raise").astype(dtype)
            elif dtype in ("float", "float32", "float64"):
                df[col] = pd.to_numeric(df[col], errors=error_behavior if error_behavior != "fail" else "raise")
            elif dtype in ("bool",):
                df[col] = df[col].astype(bool)
            else:
                df[col] = df[col].astype(dtype)
        except Exception as e:
            if error_behavior == "fail":
                raise
            log.warning("data_conversion: col=%s dtype=%s error=%s", col, dtype, e)
    return df


def copy_column(
    df: pd.DataFrame,
    column_map: Dict[str, str],
) -> pd.DataFrame:
    """
    SSIS Component: Copy Column
    ----------------------------
    Creates copies of columns with new names.

    Args:
        column_map: {source_col: new_col_name}
    """
    df = df.copy()
    for src, tgt in column_map.items():
        df[tgt] = df[src]
    return df


def audit_transform(
    df: pd.DataFrame,
    package_name: str = "",
    task_name: str = "",
    machine_name: Optional[str] = None,
    user_name: Optional[str] = None,
    execution_id: Optional[str] = None,
) -> pd.DataFrame:
    """
    SSIS Component: Audit
    ----------------------
    Adds standard audit columns (package/machine/user metadata).
    Maps to SSIS Audit transform audit type properties.
    """
    df = df.copy()
    df["_PackageName"]   = package_name
    df["_TaskName"]      = task_name
    df["_MachineName"]   = machine_name or os.environ.get("COMPUTERNAME", "")
    df["_UserName"]      = user_name or os.environ.get("USERNAME", "")
    df["_ExecutionID"]   = execution_id or ""
    df["_LoadTimestamp"] = pd.Timestamp.utcnow()
    return df


def row_count(df: pd.DataFrame, variable_name: str = "rowCount") -> Tuple[pd.DataFrame, int]:
    """
    SSIS Component: Row Count
    --------------------------
    Counts rows passing through the pipeline.

    Returns:
        Tuple of (unchanged DataFrame, row count).
        Store the count in the equivalent PackageState variable.
    """
    count = len(df)
    log.info("row_count [%s]: %d", variable_name, count)
    return df, count


# ═══════════════════════════════════════════════════════════════════════════════
# 4. JOIN / LOOKUP TRANSFORMS
# ═══════════════════════════════════════════════════════════════════════════════

def lookup(
    df: pd.DataFrame,
    lookup_engine: Optional[Engine] = None,
    lookup_query: Optional[str] = None,
    lookup_df: Optional[pd.DataFrame] = None,
    join_cols: List[Tuple[str, str]] = None,
    output_cols: Optional[List[str]] = None,
    no_match_behavior: str = "redirect",
    cache_mode: str = "full",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    SSIS Component: Lookup Transform
    ----------------------------------
    Enriches a DataFrame by matching rows against a reference dataset.

    Args:
        df:                 Input DataFrame (the "main" pipeline rows).
        lookup_engine:      Engine to query for the lookup dataset.
        lookup_query:       SQL to fetch the lookup reference table.
        lookup_df:          Alternative: pass reference DataFrame directly.
        join_cols:          List of (input_col, lookup_col) pairs.
        output_cols:        Columns to bring back from the lookup table.
        no_match_behavior:  'redirect'  → return as separate no-match DataFrame
                            'ignore'    → drop no-match rows silently
                            'fail'      → raise on any no-match
        cache_mode:         'full' | 'partial' | 'none' (for logging only)

    Returns:
        Tuple: (matched_df, no_match_df)
        If no_match_behavior='ignore' or 'fail', no_match_df is empty.

    Example:
        matched, rejected = lookup(
            df,
            lookup_engine=dw_engine,
            lookup_query="SELECT CountryCode, CountryID FROM dim.Country",
            join_cols=[("CountryCode", "CountryCode")],
            output_cols=["CountryID"],
            no_match_behavior="redirect",
        )
    """
    # Build reference DataFrame
    if lookup_df is None:
        if lookup_engine is None or lookup_query is None:
            raise ValueError("Provide either lookup_df or (lookup_engine + lookup_query)")
        ref_df = pd.read_sql(lookup_query, lookup_engine)
        log.debug("lookup: reference has %d rows [%s]", len(ref_df), cache_mode)
    else:
        ref_df = lookup_df

    if not join_cols:
        raise ValueError("join_cols is required")

    left_keys  = [j[0] for j in join_cols]
    right_keys = [j[1] for j in join_cols]

    # Keep only needed columns from reference
    if output_cols:
        keep = list(set(right_keys + output_cols))
        ref_df = ref_df[keep]

    merged = df.merge(
        ref_df,
        left_on=left_keys,
        right_on=right_keys,
        how="left",
        indicator=True,
    )

    matched    = merged[merged["_merge"] == "both"].drop(columns=["_merge"])
    no_match   = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"] + (output_cols or []))

    log.info("lookup: %d matched, %d no-match", len(matched), len(no_match))

    if no_match_behavior == "fail" and len(no_match) > 0:
        raise ValueError(f"Lookup: {len(no_match)} rows had no match")

    return matched, no_match


def merge_join(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    left_keys: List[str],
    right_keys: List[str],
    join_type: str = "inner",
    sort_inputs: bool = True,
) -> pd.DataFrame:
    """
    SSIS Component: Merge Join
    ---------------------------
    Joins two sorted DataFrames (equivalent to SQL JOIN in data flow).
    SSIS requires both inputs to be sorted — this function optionally sorts.

    Args:
        left_df:     Left input DataFrame.
        right_df:    Right input DataFrame.
        left_keys:   Join key columns in left_df.
        right_keys:  Join key columns in right_df.
        join_type:   'inner' | 'left' | 'full' (maps to SSIS JoinType).
        sort_inputs: Sort both DataFrames on keys before joining.

    Returns:
        Joined DataFrame.
    """
    how_map = {"inner": "inner", "left": "left", "full": "outer"}
    how = how_map.get(join_type.lower(), "inner")

    if sort_inputs:
        left_df  = left_df.sort_values(left_keys)
        right_df = right_df.sort_values(right_keys)

    df = left_df.merge(
        right_df,
        left_on=left_keys,
        right_on=right_keys,
        how=how,
    )
    log.info("merge_join [%s]: %d rows", join_type, len(df))
    return df


def merge_transform(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    sort_col: str,
) -> pd.DataFrame:
    """
    SSIS Component: Merge
    ----------------------
    Combines two sorted, same-schema DataFrames (union preserving order).
    Both inputs must be sorted on the same column.
    """
    combined = pd.concat([df1, df2], ignore_index=True)
    combined = combined.sort_values(sort_col)
    log.info("merge_transform: %d rows", len(combined))
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# 5. SET OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def union_all(*dataframes: pd.DataFrame) -> pd.DataFrame:
    """
    SSIS Component: Union All
    --------------------------
    Combines multiple DataFrames with the same schema (no deduplication).
    Equivalent to SQL UNION ALL.
    """
    result = pd.concat(list(dataframes), ignore_index=True)
    log.info("union_all: %d input frames → %d rows", len(dataframes), len(result))
    return result


def multicast(df: pd.DataFrame, count: int = 2) -> List[pd.DataFrame]:
    """
    SSIS Component: Multicast
    --------------------------
    Produces N identical copies of the input DataFrame for separate downstream paths.

    Args:
        df:    Input DataFrame.
        count: Number of output copies.

    Returns:
        List of DataFrames (all shallow copies of the same data).
    """
    log.info("multicast: producing %d copies of %d rows", count, len(df))
    return [df.copy() for _ in range(count)]


# ═══════════════════════════════════════════════════════════════════════════════
# 6. SPLIT / ROUTE
# ═══════════════════════════════════════════════════════════════════════════════

def conditional_split(
    df: pd.DataFrame,
    conditions: Dict[str, Union[str, Callable]],
    default_output: str = "default",
) -> Dict[str, pd.DataFrame]:
    """
    SSIS Component: Conditional Split
    -----------------------------------
    Routes rows to named outputs based on conditions (evaluated in order).
    Unmatched rows go to the default output.

    Args:
        df:             Input DataFrame.
        conditions:     OrderedDict of {output_name: condition}.
                        Condition is either:
                          - A string column expression evaluated via df.eval()
                          - A callable: lambda df → boolean Series
        default_output: Name for rows not matching any condition.

    Returns:
        Dict of {output_name: pd.DataFrame}

    Example:
        outputs = conditional_split(df, {
            "HighValue":  lambda df: df["Amount"] > 1000,
            "ZeroAmount": "Amount == 0",
        })
        high    = outputs["HighValue"]
        zero    = outputs["ZeroAmount"]
        default = outputs["default"]
    """
    result    = {default_output: df.copy()}
    assigned  = pd.Series(False, index=df.index)

    for name, cond in conditions.items():
        if callable(cond):
            mask = cond(df) & ~assigned
        else:
            mask = df.eval(cond) & ~assigned
        result[name] = df[mask]
        assigned     = assigned | mask

    result[default_output] = df[~assigned]
    for name, out in result.items():
        log.info("conditional_split [%s]: %d rows", name, len(out))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 7. AGGREGATION & SORT
# ═══════════════════════════════════════════════════════════════════════════════

def aggregate(
    df: pd.DataFrame,
    group_by: List[str],
    aggregations: Dict[str, Union[str, Tuple[str, str]]],
) -> pd.DataFrame:
    """
    SSIS Component: Aggregate
    --------------------------
    Groups and aggregates a DataFrame.

    Args:
        df:           Input DataFrame.
        group_by:     Columns to group by (maps to SSIS GroupByType columns).
        aggregations: Dict of {output_col: (input_col, agg_func)} or
                              {output_col: agg_string_on_same_col}
                      agg_func: 'sum','mean','min','max','count','count_distinct',
                                'first','last','std','var'

    Example:
        df = aggregate(df,
            group_by=["Region", "Product"],
            aggregations={
                "TotalSales": ("Amount",  "sum"),
                "AvgQty":     ("Qty",     "mean"),
                "OrderCount": ("OrderID", "count"),
                "UniqueItems":("ItemSKU", "count_distinct"),
            }
        )
    """
    agg_dict: Dict[str, Any] = {}
    for out_col, spec in aggregations.items():
        if isinstance(spec, tuple):
            src_col, func = spec
        else:
            src_col, func = spec, spec  # same column, use directly

        if func == "count_distinct":
            agg_dict[out_col] = pd.NamedAgg(column=src_col, aggfunc="nunique")
        else:
            agg_dict[out_col] = pd.NamedAgg(column=src_col, aggfunc=func)

    result = df.groupby(group_by, as_index=False, dropna=False).agg(**agg_dict)
    log.info("aggregate: %d groups", len(result))
    return result


def sort_transform(
    df: pd.DataFrame,
    sort_cols: List[str],
    ascending: Union[bool, List[bool]] = True,
    remove_duplicates: bool = False,
    comparison_flags: Optional[Dict[str, bool]] = None,
) -> pd.DataFrame:
    """
    SSIS Component: Sort
    ---------------------
    Sorts a DataFrame and optionally removes duplicate rows.

    Args:
        df:                Input DataFrame.
        sort_cols:         Columns to sort by.
        ascending:         True/False or list of True/False per column.
        remove_duplicates: If True, drops duplicate rows (SSIS RemoveDuplicates).
        comparison_flags:  Dict of {col: case_sensitive} (informational only).
    """
    df = df.sort_values(sort_cols, ascending=ascending, na_position="last")
    if remove_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=sort_cols)
        log.info("sort_transform: removed %d duplicates", before - len(df))
    log.info("sort_transform: %d rows sorted on %s", len(df), sort_cols)
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. RESHAPE
# ═══════════════════════════════════════════════════════════════════════════════

def pivot_transform(
    df: pd.DataFrame,
    index_cols: List[str],
    pivot_col: str,
    value_col: str,
    agg_func: str = "sum",
    fill_value: Any = 0,
) -> pd.DataFrame:
    """
    SSIS Component: Pivot
    ----------------------
    Rotates rows into columns (wide format).

    Args:
        index_cols:  Columns to keep as rows (GroupBy in SSIS).
        pivot_col:   Column whose distinct values become new column headers.
        value_col:   Column to aggregate in the pivot cells.
        agg_func:    Aggregation function for duplicates.
        fill_value:  Value for missing combinations.

    Example:
        # SSIS: pivot Month (Jan,Feb,Mar) with Sum(Sales)
        df = pivot_transform(df, ["Product"], "Month", "Sales", "sum")
    """
    result = df.pivot_table(
        index=index_cols,
        columns=pivot_col,
        values=value_col,
        aggfunc=agg_func,
        fill_value=fill_value,
    ).reset_index()
    result.columns.name = None
    log.info("pivot_transform: %d rows, %d cols", len(result), len(result.columns))
    return result


def unpivot_transform(
    df: pd.DataFrame,
    id_vars: List[str],
    value_vars: List[str],
    var_name: str = "Attribute",
    value_name: str = "Value",
) -> pd.DataFrame:
    """
    SSIS Component: Unpivot
    ------------------------
    Rotates columns into rows (long format).

    Args:
        id_vars:    Columns to keep (PassThrough in SSIS).
        value_vars: Columns to unpivot into rows.
        var_name:   Name for the new "Attribute" column.
        value_name: Name for the new "Value" column.

    Example:
        # SSIS: unpivot Jan, Feb, Mar → Month, Sales
        df = unpivot_transform(df, ["Product"], ["Jan","Feb","Mar"],
                               var_name="Month", value_name="Sales")
    """
    result = df.melt(
        id_vars=id_vars,
        value_vars=value_vars,
        var_name=var_name,
        value_name=value_name,
    )
    log.info("unpivot_transform: %d rows", len(result))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SLOWLY CHANGING DIMENSIONS
# ═══════════════════════════════════════════════════════════════════════════════

def scd_type1_merge(
    incoming: pd.DataFrame,
    target_engine: Engine,
    target_table: str,
    pk_column: str,
    type1_cols: List[str],
    schema: Optional[str] = None,
) -> Dict[str, int]:
    """
    SSIS Component: Slowly Changing Dimension (Type 1 — Overwrite)
    ---------------------------------------------------------------
    Inserts new rows and updates changed attribute values in place.
    No history is kept; existing values are overwritten.

    Returns:
        Dict with counts: {'inserted': N, 'updated': N}
    """
    full_table = f"{schema}.{target_table}" if schema else target_table
    existing   = pd.read_sql(f"SELECT * FROM {full_table}", target_engine)

    merged = incoming.merge(existing[[pk_column]], on=pk_column, how="left", indicator=True)
    new_rows    = incoming[merged["_merge"] == "left_only"]
    update_rows = incoming[merged["_merge"] == "both"]

    # Insert new
    if not new_rows.empty:
        dest_oledb(new_rows, target_engine, target_table, schema=schema)

    # Update changed (Type 1: overwrite)
    if not update_rows.empty:
        with target_engine.begin() as conn:
            for _, row in update_rows.iterrows():
                set_clause  = ", ".join([f"{c} = :{c}" for c in type1_cols])
                where_clause = f"{pk_column} = :{pk_column}"
                stmt = text(f"UPDATE {full_table} SET {set_clause} WHERE {where_clause}")
                conn.execute(stmt, row[type1_cols + [pk_column]].to_dict())

    log.info("scd_type1: %d inserted, %d updated", len(new_rows), len(update_rows))
    return {"inserted": len(new_rows), "updated": len(update_rows)}


def scd_type2_merge(
    incoming: pd.DataFrame,
    target_engine: Engine,
    target_table: str,
    nk_column: str,
    sk_column: str,
    tracked_cols: List[str],
    effective_date_col: str = "EffectiveDate",
    expiry_date_col: str = "ExpiryDate",
    current_flag_col: str = "IsCurrent",
    schema: Optional[str] = None,
) -> Dict[str, int]:
    """
    SSIS Component: Slowly Changing Dimension (Type 2 — Historical)
    ----------------------------------------------------------------
    Inserts new rows, expires changed rows, inserts new version for changes.
    Adds surrogate key, effective/expiry dates, and current flag.

    Returns:
        Dict with counts: {'inserted': N, 'expired': N, 'new_versions': N}
    """
    import uuid
    full_table = f"{schema}.{target_table}" if schema else target_table
    existing   = pd.read_sql(f"SELECT * FROM {full_table} WHERE {current_flag_col} = 1", target_engine)
    now        = pd.Timestamp.utcnow()
    far_future = pd.Timestamp("9999-12-31")

    merged = incoming.merge(existing[[nk_column] + tracked_cols],
                            on=nk_column, how="left", indicator=True,
                            suffixes=("", "_existing"))

    new_rows     = incoming[merged["_merge"] == "left_only"].copy()
    matched      = merged[merged["_merge"] == "both"].copy()

    # Detect changed rows
    change_mask  = pd.Series(False, index=matched.index)
    for col in tracked_cols:
        change_mask |= (matched[col] != matched[f"{col}_existing"])
    changed_rows = incoming.loc[matched[change_mask].index].copy()

    # Insert new rows
    new_rows[sk_column]          = [str(uuid.uuid4()) for _ in range(len(new_rows))]
    new_rows[effective_date_col] = now
    new_rows[expiry_date_col]    = far_future
    new_rows[current_flag_col]   = 1
    dest_oledb(new_rows, target_engine, target_table, schema=schema)

    # Expire old versions
    if not changed_rows.empty:
        nk_values = changed_rows[nk_column].tolist()
        with target_engine.begin() as conn:
            placeholders = ",".join([f"'{v}'" for v in nk_values])
            conn.execute(text(
                f"UPDATE {full_table} SET {current_flag_col}=0, {expiry_date_col}=:now "
                f"WHERE {nk_column} IN ({placeholders}) AND {current_flag_col}=1"
            ), {"now": now})

        # Insert new versions
        changed_rows[sk_column]          = [str(uuid.uuid4()) for _ in range(len(changed_rows))]
        changed_rows[effective_date_col] = now
        changed_rows[expiry_date_col]    = far_future
        changed_rows[current_flag_col]   = 1
        dest_oledb(changed_rows, target_engine, target_table, schema=schema)

    log.info("scd_type2: %d new, %d expired, %d new_versions",
             len(new_rows), len(changed_rows), len(changed_rows))
    return {"inserted": len(new_rows), "expired": len(changed_rows),
            "new_versions": len(changed_rows)}


# ═══════════════════════════════════════════════════════════════════════════════
# 10. SCRIPT COMPONENT (STUB)
# ═══════════════════════════════════════════════════════════════════════════════

def script_component(
    df: pd.DataFrame,
    transform_fn: Callable[[pd.DataFrame], pd.DataFrame],
    component_name: str = "ScriptComponent",
) -> pd.DataFrame:
    """
    SSIS Component: Script Component (Transformation)
    --------------------------------------------------
    Applies a custom Python function to the DataFrame.
    Use this as a placeholder when translating C#/VB Script Components.

    Args:
        df:             Input DataFrame.
        transform_fn:   Your Python translation of the original script logic.
        component_name: For logging.

    Example:
        def my_script(df):
            df = df.copy()
            df["FullName"] = df["First"].str.strip() + " " + df["Last"].str.strip()
            return df

        df = script_component(df, my_script, "NameCleaner")
    """
    log.debug("script_component [%s]: %d rows in", component_name, len(df))
    result = transform_fn(df)
    log.info("script_component [%s]: %d rows out", component_name, len(result))
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 11. FUZZY OPERATIONS
# ═══════════════════════════════════════════════════════════════════════════════

def fuzzy_lookup(
    df: pd.DataFrame,
    reference_df: pd.DataFrame,
    input_col: str,
    reference_col: str,
    output_col: str = "FuzzyMatch",
    score_col: str = "MatchScore",
    min_score: float = 0.8,
    scorer: str = "ratio",
) -> pd.DataFrame:
    """
    SSIS Component: Fuzzy Lookup
    -----------------------------
    Matches input values to a reference table using fuzzy string matching.
    Requires: pip install rapidfuzz

    Args:
        df:            Input DataFrame.
        reference_df:  Reference table to match against.
        input_col:     Column in df to match.
        reference_col: Column in reference_df to match against.
        output_col:    New column for best match value.
        score_col:     New column for similarity score (0–1).
        min_score:     Minimum score to consider a match (0–1).
        scorer:        'ratio' | 'partial_ratio' | 'token_sort_ratio'
    """
    try:
        from rapidfuzz import process, fuzz
    except ImportError:
        raise ImportError("pip install rapidfuzz")

    scorer_map = {
        "ratio":            fuzz.ratio,
        "partial_ratio":    fuzz.partial_ratio,
        "token_sort_ratio": fuzz.token_sort_ratio,
    }
    score_fn   = scorer_map.get(scorer, fuzz.ratio)
    ref_values = reference_df[reference_col].tolist()

    matches, scores = [], []
    for val in df[input_col]:
        result = process.extractOne(val, ref_values, scorer=score_fn)
        if result and result[1] / 100 >= min_score:
            matches.append(result[0])
            scores.append(result[1] / 100)
        else:
            matches.append(None)
            scores.append(0.0)

    df = df.copy()
    df[output_col] = matches
    df[score_col]  = scores
    log.info("fuzzy_lookup: %d/%d matched above %.0f%%", sum(s > 0 for s in scores), len(df), min_score * 100)
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 12. CONNECTION HELPERS  (config/connections.py)
# ═══════════════════════════════════════════════════════════════════════════════

# Place this in config/connections.py (separate file):
#
# from sqlalchemy import create_engine
# from sqlalchemy.engine import Engine
# import os
#
# _ENGINES: dict = {}
#
# def get_engine(conn_name: str) -> Engine:
#     """
#     Returns a cached SQLAlchemy engine for the given connection manager name.
#     Connection strings are read from environment variables.
#
#     SSIS Connection Manager → Environment Variable mapping:
#       MySourceDB     → CONN_MYSOURCEDB
#       DW_Connection  → CONN_DW_CONNECTION
#
#     Supported URL formats (set in env var):
#       SQL Server:  mssql+pyodbc://user:pass@server/db?driver=ODBC+Driver+17+for+SQL+Server
#       Oracle:      oracle+cx_oracle://user:pass@host:1521/?service_name=svc
#       PostgreSQL:  postgresql+psycopg2://user:pass@host/db
#       MySQL:       mysql+pymysql://user:pass@host/db
#       SQLite:      sqlite:///path/to/file.db
#     """
#     if conn_name not in _ENGINES:
#         env_key = f"CONN_{conn_name.upper().replace(' ', '_').replace('-', '_')}"
#         url = os.environ.get(env_key)
#         if not url:
#             raise EnvironmentError(f"Missing env var: {env_key}")
#         _ENGINES[conn_name] = create_engine(url, fast_executemany=True)
#     return _ENGINES[conn_name]
```
