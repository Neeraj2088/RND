# Python Outline Templates by SSIS Package Type

---

## Type 1 — Simple ETL (Single Data Flow Task)

**Detected when:** One DFT, no loops, linear control flow.

```python
"""
Pipeline: {package_name}
Type: Simple ETL
"""
import logging
import os
import pandas as pd
from utils.ssis_utils import (
    source_oledb, dest_oledb,
    # ... add components used
)
from config.connections import get_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def run_pre_execute():
    """Maps to: OnPreExecute / Execute SQL tasks before DFT."""
    pass


def run_data_flow() -> pd.DataFrame:
    """Maps to: Data Flow Task — {dft_name}"""
    src_engine = get_engine("SOURCE_CONN")
    df = source_oledb(src_engine, query="{source_sql}")
    log.info("Source rows: %d", len(df))
    # ... transforms
    dest_engine = get_engine("DEST_CONN")
    dest_oledb(df, dest_engine, table="{dest_table}", schema="{dest_schema}")
    log.info("Loaded %d rows", len(df))
    return df


def run_post_execute():
    """Maps to: Execute SQL tasks after DFT / audit logging."""
    pass


def main():
    log.info("Pipeline START: {package_name}")
    try:
        run_pre_execute()
        run_data_flow()
        run_post_execute()
        log.info("Pipeline COMPLETE")
    except Exception as e:
        log.error("Pipeline FAILED: %s", e)
        raise


if __name__ == "__main__":
    main()
```

---

## Type 2 — Multi-Step Pipeline (Multiple DFTs with Precedence)

**Detected when:** 2+ DFTs connected by precedence constraints (success/failure/expression).

```python
"""
Pipeline: {package_name}
Type: Multi-Step Pipeline
Precedence: {ascii_tree}
"""
import logging
import pandas as pd
from utils.ssis_utils import *
from config.connections import get_engine

log = logging.getLogger(__name__)


class PackageState:
    """Mirrors SSIS package-level variables."""
    run_id: int = 0
    row_count: int = 0
    error_flag: bool = False
    # ... add variables from SSIS variable list


def task_truncate_staging(state: PackageState) -> bool:
    """Maps to: Execute SQL Task — TruncateStaging"""
    engine = get_engine("DW_CONN")
    with engine.connect() as conn:
        conn.execute("{truncate_sql}")
    return True  # success → next task proceeds


def dft_load_dimension(state: PackageState) -> bool:
    """Maps to: Data Flow Task — LoadDimension"""
    try:
        df = source_oledb(get_engine("SRC"), "{dim_sql}")
        # transforms...
        dest_oledb(df, get_engine("DW"), table="{dim_table}")
        state.row_count = len(df)
        return True
    except Exception as e:
        log.error("dft_load_dimension failed: %s", e)
        state.error_flag = True
        return False


def dft_load_fact(state: PackageState) -> bool:
    """Maps to: Data Flow Task — LoadFact"""
    try:
        df = source_oledb(get_engine("SRC"), "{fact_sql}")
        # transforms...
        dest_oledb(df, get_engine("DW"), table="{fact_table}")
        return True
    except Exception as e:
        log.error("dft_load_fact failed: %s", e)
        return False


def task_update_audit(state: PackageState):
    """Maps to: Execute SQL Task — UpdateAudit"""
    pass


def main():
    state = PackageState()
    log.info("START {package_name}")

    # Mirrors precedence constraints:
    # TruncateStaging →(Success)→ LoadDimension →(Success)→ LoadFact →(Success)→ UpdateAudit
    if not task_truncate_staging(state):
        log.error("Stopping: truncate failed")
        return

    if not dft_load_dimension(state):
        log.error("Stopping: dimension load failed")
        return

    if not dft_load_fact(state):
        log.error("Stopping: fact load failed")
        return

    task_update_audit(state)
    log.info("COMPLETE {package_name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```

---

## Type 3 — File/Record Iterator (ForEach Loop Container)

**Detected when:** `ForEachEnumerator` present — ForEach File, ForEach ADO, ForEach Item, etc.

```python
"""
Pipeline: {package_name}
Type: ForEach Loop — {enumerator_type}
Variable mapped: {variable_name}
"""
import os
import glob
import logging
import pandas as pd
from utils.ssis_utils import *
from config.connections import get_engine

log = logging.getLogger(__name__)

# SSIS ForEach mapping:
# ForEachFileEnumerator  → glob.glob(folder, pattern)
# ForEachADOEnumerator   → iterate rows of a DataFrame
# ForEachItemEnumerator  → iterate a hardcoded list
# ForEachSMOEnumerator   → iterate SQL Server objects
# ForEachNodeListEnumerator → iterate XML nodes

FOLDER_PATH = os.environ.get("INPUT_FOLDER", "{folder_path}")
FILE_PATTERN = "{file_pattern}"  # e.g. "*.csv"


def process_single_file(file_path: str) -> pd.DataFrame:
    """
    Maps to: inner DFT inside ForEach loop.
    SSIS variable: {loop_variable} = file_path
    """
    log.info("Processing: %s", file_path)
    df = source_flatfile(file_path, delimiter="{delimiter}", encoding="{encoding}")
    # transforms...
    return df


def foreach_file_loop():
    """Maps to: ForEach File Enumerator — {container_name}"""
    files = sorted(glob.glob(os.path.join(FOLDER_PATH, FILE_PATTERN)))
    if not files:
        log.warning("No files found in %s matching %s", FOLDER_PATH, FILE_PATTERN)
        return

    all_frames = []
    for file_path in files:
        try:
            df = process_single_file(file_path)
            all_frames.append(df)
        except Exception as e:
            log.error("Failed on %s: %s", file_path, e)
            # honour SSIS FailPackageOnFailure setting here

    if all_frames:
        combined = pd.concat(all_frames, ignore_index=True)
        dest_oledb(combined, get_engine("DEST_CONN"), table="{dest_table}")
        log.info("Total rows loaded: %d", len(combined))


def foreach_ado_loop(driver_df: pd.DataFrame):
    """Maps to: ForEach ADO Enumerator — iterates rows of driver_df."""
    for _, row in driver_df.iterrows():
        # SSIS variable assignments from row columns
        param_value = row["{column_name}"]
        process_record(param_value)


def main():
    log.info("START {package_name}")
    foreach_file_loop()
    log.info("COMPLETE")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```

---

## Type 4 — Batch / Cursor Loop (For Loop Container)

**Detected when:** `ForLoop` container present with InitExpression / EvalExpression / AssignExpression.

```python
"""
Pipeline: {package_name}
Type: For Loop
Init: {init_expression}   e.g. @counter = 0
Eval: {eval_expression}   e.g. @counter < @max_batches
Assign: {assign_expression} e.g. @counter = @counter + 1
"""
import logging
import pandas as pd
from utils.ssis_utils import *
from config.connections import get_engine

log = logging.getLogger(__name__)

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "{batch_size}"))
MAX_ROWS = int(os.environ.get("MAX_ROWS", "{max_rows}"))


def process_batch(offset: int, batch_size: int) -> int:
    """
    Maps to: DFT inside For Loop.
    Returns rows processed (0 = done).
    """
    query = f"""
        SELECT * FROM {"{source_table}"}
        ORDER BY {"{pk_column}"}
        OFFSET {offset} ROWS FETCH NEXT {batch_size} ROWS ONLY
    """
    df = source_oledb(get_engine("SRC"), query)
    if df.empty:
        return 0
    # transforms...
    dest_oledb(df, get_engine("DEST"), table="{dest_table}", if_exists="append")
    return len(df)


def for_loop_container():
    """
    Maps to: For Loop Container — {container_name}
    Equivalent: @i = 0; @i < @MaxBatches; @i++
    """
    offset = 0
    total = 0
    while True:
        rows = process_batch(offset, BATCH_SIZE)
        if rows == 0:
            break
        total += rows
        offset += BATCH_SIZE
        log.info("Processed %d rows so far (offset=%d)", total, offset)

    log.info("For Loop complete. Total rows: %d", total)


def main():
    log.info("START {package_name}")
    for_loop_container()
    log.info("COMPLETE")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```

---

## Type 5 — Hybrid / Scripted Package (Script Tasks + Complex Expressions)

**Detected when:** `ScriptTask`, `ScriptComponent`, or heavy use of SSIS expressions.

```python
"""
Pipeline: {package_name}
Type: Hybrid / Scripted
Contains Script Tasks: {script_task_names}
"""
import logging
import re
import pandas as pd
from utils.ssis_utils import *
from config.connections import get_engine

log = logging.getLogger(__name__)


# ── SSIS Expression Translation ────────────────────────────────────────────
# SSIS expressions use syntax like:
#   (DT_STR,50,1252)[FirstName] + " " + [LastName]
#   ISNULL([Column]) ? "Unknown" : [Column]
#   GETDATE()
#   SUBSTRING([Col],1,3)
# Python equivalents are emitted inline using .assign() or lambdas.

def translate_expression(ssis_expr: str, df: pd.DataFrame) -> pd.Series:
    """
    Generic SSIS expression → Python translation.
    For complex cases, this is manually expanded per column.
    """
    # Common pattern rewrites:
    expr = ssis_expr
    expr = re.sub(r'\(DT_STR,\d+,\d+\)', '', expr)   # cast removal
    expr = expr.replace("ISNULL(", "pd.isnull(")
    expr = expr.replace("GETDATE()", "pd.Timestamp.now()")
    expr = expr.replace("SUBSTRING(", "str_substring(")
    # ... emit a warning for unhandled expressions
    log.warning("Expression needs manual review: %s", ssis_expr)
    return pd.Series(dtype="object")


# ── Script Task Bodies ──────────────────────────────────────────────────────
# Each Script Task's C# or VB code is translated to a Python function.
# Preserve the original code in a comment block for reference.

def script_task_{task_name}(state) -> bool:
    """
    Maps to: Script Task — {task_name}
    Original language: {CSharp|VB}

    Original logic:
    // {original_code_summary}
    """
    # TODO: translate C# / VB logic here
    # Common patterns:
    #   Dts.Variables["varName"].Value  → state.var_name
    #   Dts.Connections["connName"]     → get_engine("connName")
    #   MessageBox.Show(...)            → log.info(...)
    #   Dts.TaskResult = Dts.Results.Success → return True
    return True


# ── Script Component (Data Flow) ───────────────────────────────────────────

def script_component_{component_name}(df: pd.DataFrame) -> pd.DataFrame:
    """
    Maps to: Script Component — {component_name}
    Type: {Source | Transformation | Destination}
    """
    # Row-by-row Script Component → vectorised Pandas
    # Original C#: Row.OutputColumn = Row.InputColumn.Trim().ToUpper();
    df = df.copy()
    # df["OutputColumn"] = df["InputColumn"].str.strip().str.upper()
    return df


def main():
    log.info("START {package_name}")
    # ... orchestrate tasks
    log.info("COMPLETE")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```

---

## Type 6 — Data Warehouse Load (SCD / Dimension / Fact)

**Detected when:** SCD component present, or DFT names suggest dimension/fact loading.

```python
"""
Pipeline: {package_name}
Type: Data Warehouse Load
Dimensions: {dim_list}
Facts: {fact_list}
"""
import logging
import pandas as pd
from datetime import datetime
from utils.ssis_utils import (
    source_oledb, dest_oledb,
    scd_type1_merge, scd_type2_merge,
    lookup, derived_column
)
from config.connections import get_engine

log = logging.getLogger(__name__)
RUN_DATE = datetime.utcnow()


def load_dimension_{dim_name}():
    """
    Maps to: DFT loading {dim_name} dimension.
    SCD Type: {1 | 2 | 3}
    """
    src = get_engine("SRC")
    dw = get_engine("DW")

    incoming = source_oledb(src, "{dim_source_query}")

    # SCD Type 1 — overwrite changed attributes
    scd_type1_merge(
        incoming=incoming,
        target_engine=dw,
        target_table="{schema}.{dim_table}",
        pk_column="{pk_col}",
        type1_cols=["{col1}", "{col2}"],
    )


def load_fact_{fact_name}():
    """
    Maps to: DFT loading {fact_name} fact table.
    """
    src = get_engine("SRC")
    dw  = get_engine("DW")

    df = source_oledb(src, "{fact_source_query}")

    # Lookup dimension keys
    df = lookup(
        df,
        lookup_engine=dw,
        lookup_query="SELECT {nk_col}, {sk_col} FROM {dim_table}",
        join_cols=[("{nk_col}", "{nk_col}")],
        output_cols=["{sk_col}"],
        no_match_behavior="redirect",   # or "fail" | "ignore"
    )

    df = derived_column(df, {
        "LoadDate": RUN_DATE,
        "RowHash":  lambda r: hash(tuple(r)),
    })

    dest_oledb(df, dw, table="{fact_table}", if_exists="append")
    log.info("Fact rows loaded: %d", len(df))


def main():
    log.info("START DW load — {package_name}")
    load_dimension_{dim_name}()
    load_fact_{fact_name}()
    log.info("DW load COMPLETE")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```
