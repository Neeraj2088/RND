---
name: ssis-to-python
description: >
  Convert SSIS (.dtsx) packages into clean, idiomatic Python ETL code. Use this
  skill whenever the user uploads or pastes an SSIS package XML, asks to migrate
  SSIS to Python/Pandas/SQLAlchemy, or needs to understand what an SSIS package
  does. Triggers on phrases like "convert SSIS", "migrate SSIS", "SSIS to Python",
  "translate dtsx", "rewrite SSIS pipeline", or any mention of SSIS components
  such as Data Flow Task, Control Flow, Lookup, Merge Join, Derived Column, etc.
  Always use this skill — do NOT attempt SSIS conversion from memory alone.
---

# SSIS → Python Conversion Skill

This skill converts SQL Server Integration Services (`.dtsx`) packages into
production-grade Python ETL scripts. Follow all four phases in order.

---

## PHASE 1 — Parse & Map the SSIS Package

### 1.1 What is a DTSX file?
A `.dtsx` file is XML. Key namespaces:
- `DTS:` — package-level metadata, variables, connections, control flow
- `pipeline:` — data flow components (sources, transforms, destinations)
- `SQLTask:` / `ScriptTask:` / `ForEachEnumerator:` — specialized task types

### 1.2 Parse Strategy
```python
import xml.etree.ElementTree as ET
tree = ET.parse("package.dtsx")
root = tree.getroot()
```

### 1.3 Build the Package Map

Extract and display these layers to the Python developer:

#### Layer A — Connection Managers
XPath: `.//DTS:ConnectionManager`
| Field | XML Attribute |
|-------|--------------|
| Name  | `DTS:ObjectName` |
| Type  | `DTS:CreationName` (OLEDB, FLATFILE, ADO.NET, etc.) |
| Connection String | `DTS:ConnectionString` property |

#### Layer B — Variables
XPath: `.//DTS:Variable`
| Field | Notes |
|-------|-------|
| Name  | `DTS:ObjectName` |
| DataType | `DTS:DataType` (3=int32, 8=string, 5=float64, 11=bool, 7=datetime) |
| Value | `DTS:VariableValue` |

#### Layer C — Control Flow Tree
XPath: `.//DTS:Executable` (recursive, nested = precedence constraints)

Render as an ASCII tree:
```
PACKAGE: MyPackage
├── [SEQ] Initialize
│   ├── [SQL] TruncateStagingTable
│   └── [EXPR] SetRunDate
├── [DFT] Load_Customers        ← Data Flow Task
│   ├── SOURCE: OLE DB (CustomerDB)
│   ├── TRANSFORM: Derived Column (FullName)
│   ├── TRANSFORM: Lookup (LookupCountryCode)
│   └── DEST: OLE DB (Staging.Customer)
├── [FELC] Loop_Files            ← ForEach Loop
│   └── [DFT] Load_FileData
└── [SQL] UpdateAuditLog
```

Component abbreviations:
- `[SEQ]` Sequence Container
- `[DFT]` Data Flow Task
- `[SQL]` Execute SQL Task
- `[EXPR]` Expression Task
- `[SCR]` Script Task
- `[FELC]` ForEach Loop Container
- `[FLC]` For Loop Container
- `[SEND]` Send Mail Task
- `[FTP]` FTP Task
- `[EXEC]` Execute Process Task

#### Layer D — Data Flow Component Map (for each DFT)
XPath: `.//pipeline:component`
For each component capture:
- `componentClassID` → maps to component type (see reference table below)
- `name` attribute
- Input/Output column names and data types
- Custom properties (SQL query, expression, join type, etc.)

#### SSIS componentClassID → Python mapping reference
| SSIS Class | Component Type | Python Approach |
|---|---|---|
| `Microsoft.OLEDBSource` | OLE DB Source | `pd.read_sql()` / SQLAlchemy |
| `Microsoft.FlatFileSource` | Flat File Source | `pd.read_csv()` |
| `Microsoft.ExcelSource` | Excel Source | `pd.read_excel()` |
| `Microsoft.OLEDBDestination` | OLE DB Destination | `df.to_sql()` / bulk insert |
| `Microsoft.FlatFileDestination` | Flat File Destination | `df.to_csv()` |
| `Microsoft.ExcelDestination` | Excel Destination | `df.to_excel()` |
| `Microsoft.DerivedColumn` | Derived Column | `df.assign()` / `df[col] =` |
| `Microsoft.Lookup` | Lookup | `pd.merge()` |
| `Microsoft.MergeJoin` | Merge Join | `pd.merge()` |
| `Microsoft.UnionAll` | Union All | `pd.concat()` |
| `Microsoft.ConditionalSplit` | Conditional Split | boolean mask filter |
| `Microsoft.DataConversion` | Data Conversion | `df[col].astype()` |
| `Microsoft.Sort` | Sort | `df.sort_values()` |
| `Microsoft.Aggregate` | Aggregate | `df.groupby().agg()` |
| `Microsoft.RowCount` | Row Count | `len(df)` |
| `Microsoft.MulticastTransformation` | Multicast | variable assignment copies |
| `Microsoft.ScriptComponent` | Script Component | inline Python function |
| `Microsoft.SlowlyChangingDimension` | SCD | custom SCD util function |
| `Microsoft.FuzzyLookup` | Fuzzy Lookup | `rapidfuzz` library |
| `Microsoft.TermExtraction` | Term Extraction | NLP / regex |
| `Microsoft.PivotTransformation` | Pivot | `df.pivot_table()` |
| `Microsoft.UnpivotTransformation` | Unpivot | `df.melt()` |

---

## PHASE 2 — Python Package Outline by Package Type

Based on the control flow map, identify the package type and use the matching
outline. See `references/outlines.md` for full boilerplate per type.

**Package type detection rules:**
- Single DFT, no loops → **Type 1: Simple ETL**
- Multiple DFTs with precedence → **Type 2: Multi-Step Pipeline**
- ForEach loop → **Type 3: File/Record Iterator**
- For Loop → **Type 4: Batch/Cursor Loop**
- Script tasks / complex expressions → **Type 5: Hybrid/Scripted**
- SCD / dimension loads → **Type 6: Data Warehouse Load**

---

## PHASE 3 — Component-to-Method Library (util.py)

Every SSIS component maps to a **reusable utility method**. When generating
code, always emit these into `utils/ssis_utils.py` and import from there.
**Never inline the logic in the main pipeline file.**

See `references/utils_library.md` for the complete utility function signatures
and implementations for all 25+ component types.

Key design rules for util methods:
1. Input is always a `pd.DataFrame` (or connection + query for sources)
2. Output is always a `pd.DataFrame` (or None for destinations/sinks)
3. All parameters are explicit — no hidden globals
4. Every function is independently testable
5. Docstring lists the equivalent SSIS component

---

## PHASE 4 — Stitch into Final Pipeline File

After building the map (Phase 1), selecting the outline (Phase 2), and
generating utils (Phase 3), produce the final `pipeline.py`:

```
output/
├── pipeline.py          ← main orchestration (mirrors control flow tree)
├── utils/
│   └── ssis_utils.py    ← all component utility functions
├── config/
│   └── connections.py   ← connection strings / env vars
└── tests/
    └── test_pipeline.py ← basic smoke tests per DFT
```

### Stitching Rules
1. One Python function per SSIS task/container in control flow
2. Precedence constraints → function call order or `if/else`
3. Failure constraints → `try/except` blocks
4. Package variables → Python variables or a `PackageState` dataclass
5. Expressions → Python f-strings or `eval()` with a context dict
6. Event handlers (OnError, OnPostExecute) → Python logging + decorators

### File Header Template
```python
"""
Pipeline: {package_name}
Converted from: {dtsx_filename}
Original description: {DTS:Description}
Generated: {date}

Control Flow:
{ascii_tree_from_phase1}
"""
```

---

## Execution Checklist

When converting a package, work through these steps and confirm each:

- [ ] Phase 1: XML parsed, connection managers listed, variable table built, ASCII tree rendered, all DFT component maps complete
- [ ] Phase 2: Package type identified, outline selected
- [ ] Phase 3: `ssis_utils.py` generated with all required component methods
- [ ] Phase 4: `pipeline.py` stitched, imports wired, connections parameterized
- [ ] `connections.py` uses env vars (never hardcoded credentials)
- [ ] All precedence constraints preserved as function order / conditionals
- [ ] Loop containers converted to Python loops
- [ ] Error paths (failure constraints) wrapped in try/except with logging

---

## Reference Files

- `references/outlines.md` — Full Python boilerplate for each of the 6 package types
- `references/utils_library.md` — Complete `ssis_utils.py` with all 25+ component functions
