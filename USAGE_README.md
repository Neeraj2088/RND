# Oracle Schema Auto-Discovery — Usage Guide

## Setup

```bash
pip install cx_Oracle python-dotenv
cp .env.template .env        # fill in your Oracle credentials
```

## Run Commands

```bash
# Discover ALL tables in your schema
python oracle_schema_discovery.py

# Discover specific tables only
python oracle_schema_discovery.py --tables POLICY_MASTER COVERAGE_DETAIL CLAIM_HEADER

# Custom schema owner + output file
python oracle_schema_discovery.py --schema FINANCE --output finance_schema.json

# Skip sample values (faster, smaller output)
python oracle_schema_discovery.py --no-samples

# More sample values per column
python oracle_schema_discovery.py --sample-limit 10
```

## What Gets Generated (oracle_schema.json)

```
schema_metadata        ← Discovery stats, version, timestamp
llm_instructions       ← Rules the LLM must follow when querying
tables
  └── TABLE_NAME
        ├── description          ← From Oracle comments (or TODO)
        ├── business_context     ← TODO: you fill this in
        ├── primary_key          ← Detected from constraints
        ├── unique_keys          ← Detected from constraints
        ├── indexes              ← All indexes on this table
        ├── approximate_row_count
        └── columns
              └── COLUMN_NAME
                    ├── data_type
                    ├── nullable
                    ├── primary_key (true/false)
                    ├── description       ← From Oracle col comments
                    ├── business_meaning  ← TODO: you fill this in
                    └── sample_values     ← Actual DB values (5 max)

relationships          ← Auto-detected by PK column name matching
  └── parent_table / parent_column → child_table / child_column
        ├── confidence   HIGH = column name exactly matches a PK
        ├── verified     false = needs your manual confirmation
        └── notes        reminder to verify cardinality

common_query_patterns  ← SQL templates for the LLM to follow
```

## After Running — Manual Enrichment Steps

Open oracle_schema.json and fill in the TODO fields:

1. `tables[*].description`       — what the table holds
2. `tables[*].business_context`  — business role of the table
3. `columns[*].business_meaning` — what the column means in business terms
4. `relationships[*].verified`   — set to true after you confirm each join
5. `relationships[*].cardinality`— correct ONE_TO_MANY / MANY_TO_ONE etc.

These enrichments are what make the LLM's query generation accurate.

## Relationship Detection Logic

Since no FK constraints exist, relationships are detected by:

  "Does any other table have a column with the SAME NAME as this table's PK?"

Example:
  POLICY_MASTER.POLICY_ID (PK)  →  found in COVERAGE_DETAIL.POLICY_ID
  ∴ Implicit relationship detected with HIGH confidence

## Feeding the Schema to Your LLM

```python
import json

with open("oracle_schema.json") as f:
    schema = json.load(f)

# Option A: full schema in system prompt (small DBs)
schema_context = json.dumps(schema, indent=2)

# Option B: chunk by table into your Vector DB (large DBs)
for table_name, table_meta in schema["tables"].items():
    chunk = {table_name: table_meta}
    vector_db.upsert(
        id=f"table_{table_name}",
        text=json.dumps(chunk),
        metadata={"type": "table_schema", "table": table_name}
    )
```
