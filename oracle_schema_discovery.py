"""
Oracle Database Schema Auto-Discovery Script
=============================================
Connects to Oracle DB and generates a detailed JSON schema skeleton
that an LLM can use to build dynamic queries.

Requirements:
    pip install cx_Oracle python-dotenv

Usage:
    python oracle_schema_discovery.py
    python oracle_schema_discovery.py --tables POLICY_MASTER COVERAGE_DETAIL
    python oracle_schema_discovery.py --schema HR --output my_schema.json
"""

import cx_Oracle
import json
import argparse
import os
import sys
from datetime import datetime
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()


# ─────────────────────────────────────────────
#  1. DATABASE CONNECTION
# ─────────────────────────────────────────────

def get_connection():
    """
    Reads credentials from environment variables.
    Set these in a .env file:

        ORACLE_USER=your_user
        ORACLE_PASSWORD=your_password
        ORACLE_DSN=host:port/service_name
        ORACLE_SCHEMA=YOUR_SCHEMA       (optional, defaults to ORACLE_USER)
    """
    user     = os.getenv("ORACLE_USER")
    password = os.getenv("ORACLE_PASSWORD")
    dsn      = os.getenv("ORACLE_DSN")

    if not all([user, password, dsn]):
        print("ERROR: Set ORACLE_USER, ORACLE_PASSWORD, ORACLE_DSN in your .env file.")
        sys.exit(1)

    try:
        conn = cx_Oracle.connect(user=user, password=password, dsn=dsn)
        print(f"✅ Connected to Oracle as '{user}' @ '{dsn}'")
        return conn
    except cx_Oracle.DatabaseError as e:
        print(f"❌ Connection failed: {e}")
        sys.exit(1)


# ─────────────────────────────────────────────
#  2. DISCOVERY FUNCTIONS
# ─────────────────────────────────────────────

def discover_tables(cursor, schema: str, table_filter: list = None) -> list:
    """
    Fetch all tables (or a filtered subset) from the schema.
    """
    if table_filter:
        placeholders = ", ".join([f":t{i}" for i in range(len(table_filter))])
        bind_vars    = {f"t{i}": t.upper() for i, t in enumerate(table_filter)}
        bind_vars["schema"] = schema.upper()
        sql = f"""
            SELECT table_name, num_rows, last_analyzed, comments
            FROM (
                SELECT t.table_name, t.num_rows, t.last_analyzed, c.comments
                FROM   all_tables t
                LEFT JOIN all_tab_comments c
                       ON c.table_name = t.table_name
                      AND c.owner      = t.owner
                WHERE  t.owner = :schema
                AND    t.table_name IN ({placeholders})
            )
            ORDER BY table_name
        """
        cursor.execute(sql, bind_vars)
    else:
        cursor.execute("""
            SELECT t.table_name, t.num_rows, t.last_analyzed, c.comments
            FROM   all_tables t
            LEFT JOIN all_tab_comments c
                   ON c.table_name = t.table_name
                  AND c.owner      = t.owner
            WHERE  t.owner = :schema
            ORDER  BY t.table_name
        """, schema=schema.upper())

    rows = cursor.fetchall()
    print(f"   Found {len(rows)} table(s)")
    return rows


def discover_columns(cursor, schema: str, table_name: str) -> list:
    """
    Fetch all columns for a table including type, nullability, defaults,
    and any column-level comments.
    """
    cursor.execute("""
        SELECT
            c.column_name,
            c.data_type,
            c.data_length,
            c.data_precision,
            c.data_scale,
            c.nullable,
            c.data_default,
            c.column_id,
            cc.comments
        FROM   all_tab_columns c
        LEFT JOIN all_col_comments cc
               ON cc.owner       = c.owner
              AND cc.table_name  = c.table_name
              AND cc.column_name = c.column_name
        WHERE  c.owner      = :schema
        AND    c.table_name = :table
        ORDER  BY c.column_id
    """, schema=schema.upper(), table=table_name.upper())
    return cursor.fetchall()


def discover_primary_keys(cursor, schema: str) -> dict:
    """
    Returns a dict: { TABLE_NAME: [PK_COL1, PK_COL2, ...] }
    """
    cursor.execute("""
        SELECT
            cc.table_name,
            cc.column_name,
            cc.position
        FROM   all_constraints  c
        JOIN   all_cons_columns cc
               ON cc.constraint_name = c.constraint_name
              AND cc.owner           = c.owner
        WHERE  c.owner           = :schema
        AND    c.constraint_type = 'P'
        ORDER  BY cc.table_name, cc.position
    """, schema=schema.upper())

    pk_map = defaultdict(list)
    for table, col, _ in cursor.fetchall():
        pk_map[table].append(col)
    return dict(pk_map)


def discover_unique_keys(cursor, schema: str) -> dict:
    """
    Returns a dict: { TABLE_NAME: [[UK_COL1, ...], [...]] }
    """
    cursor.execute("""
        SELECT
            cc.table_name,
            c.constraint_name,
            cc.column_name,
            cc.position
        FROM   all_constraints  c
        JOIN   all_cons_columns cc
               ON cc.constraint_name = c.constraint_name
              AND cc.owner           = c.owner
        WHERE  c.owner           = :schema
        AND    c.constraint_type = 'U'
        ORDER  BY cc.table_name, c.constraint_name, cc.position
    """, schema=schema.upper())

    uk_map = defaultdict(lambda: defaultdict(list))
    for table, con_name, col, _ in cursor.fetchall():
        uk_map[table][con_name].append(col)

    # Flatten: { TABLE: [ [col1, col2], [col3] ] }
    return {
        tbl: list(cons.values())
        for tbl, cons in uk_map.items()
    }


def discover_indexes(cursor, schema: str) -> dict:
    """
    Returns a dict: { TABLE_NAME: [ {index_name, columns, uniqueness} ] }
    """
    cursor.execute("""
        SELECT
            ic.table_name,
            ic.index_name,
            ic.column_name,
            ic.column_position,
            i.uniqueness
        FROM   all_ind_columns ic
        JOIN   all_indexes i
               ON i.index_name = ic.index_name
              AND i.owner      = ic.index_owner
        WHERE  ic.index_owner = :schema
        ORDER  BY ic.table_name, ic.index_name, ic.column_position
    """, schema=schema.upper())

    idx_map = defaultdict(lambda: defaultdict(lambda: {"columns": [], "uniqueness": ""}))
    for table, idx_name, col, _, uniqueness in cursor.fetchall():
        idx_map[table][idx_name]["columns"].append(col)
        idx_map[table][idx_name]["uniqueness"] = uniqueness

    return {
        tbl: [
            {"index_name": idx, "columns": meta["columns"], "unique": meta["uniqueness"] == "UNIQUE"}
            for idx, meta in idxs.items()
        ]
        for tbl, idxs in idx_map.items()
    }


def discover_implicit_relationships(cursor, schema: str, tables: list, pk_map: dict) -> list:
    """
    Core function: since you have NO foreign keys defined,
    this detects relationships by name-matching PK columns
    across tables — e.g. POLICY_MASTER.POLICY_ID → COVERAGE_DETAIL.POLICY_ID

    Logic:
    1. For every PK column in every table
    2. Search all other tables for a column with the SAME NAME
    3. If found, that's a likely implicit join relationship
    """
    relationships = []
    seen          = set()

    table_columns = {}
    for table in tables:
        cursor.execute("""
            SELECT column_name FROM all_tab_columns
            WHERE owner = :schema AND table_name = :table
        """, schema=schema.upper(), table=table.upper())
        table_columns[table] = {row[0] for row in cursor.fetchall()}

    for parent_table, pk_cols in pk_map.items():
        if parent_table not in table_columns:
            continue
        for pk_col in pk_cols:
            for child_table in tables:
                if child_table == parent_table:
                    continue
                if pk_col in table_columns.get(child_table, set()):
                    key = tuple(sorted([parent_table, child_table]) + [pk_col])
                    if key in seen:
                        continue
                    seen.add(key)
                    relationships.append({
                        "relationship_name"  : f"{parent_table.lower()}_to_{child_table.lower()}",
                        "detection_method"   : "implicit_column_name_match",
                        "confidence"         : "HIGH",
                        "parent_table"       : parent_table,
                        "parent_column"      : pk_col,
                        "child_table"        : child_table,
                        "child_column"       : pk_col,
                        "join_type"          : "LEFT JOIN",
                        "cardinality"        : "ONE_TO_MANY",
                        "description"        : (
                            f"{parent_table} is the parent. "
                            f"Join on {parent_table}.{pk_col} = {child_table}.{pk_col}"
                        ),
                        "verified"           : False,   # ← Set to True after manual review
                        "notes"              : "Auto-detected. Please verify cardinality and join type."
                    })

    print(f"   Detected {len(relationships)} implicit relationship(s)")
    return relationships


def discover_sample_values(cursor, schema: str, table: str, column: str, limit: int = 5) -> list:
    """
    Fetch distinct sample values for a column — helps LLM understand domain.
    Skips LOB / LONG types.
    """
    try:
        cursor.execute(f"""
            SELECT DISTINCT {column}
            FROM   {schema}.{table}
            WHERE  {column} IS NOT NULL
            AND    ROWNUM <= :lim
        """, lim=limit)
        return [str(row[0]) for row in cursor.fetchall()]
    except Exception:
        return []


def discover_row_counts(cursor, schema: str, table: str) -> int:
    """Fast approximate row count from Oracle stats."""
    try:
        cursor.execute("""
            SELECT num_rows FROM all_tables
            WHERE owner = :schema AND table_name = :table
        """, schema=schema.upper(), table=table.upper())
        row = cursor.fetchone()
        return row[0] if row and row[0] is not None else -1
    except Exception:
        return -1


# ─────────────────────────────────────────────
#  3. FORMAT COLUMN TYPE
# ─────────────────────────────────────────────

def format_data_type(data_type: str, length, precision, scale) -> str:
    if data_type == "NUMBER":
        if precision and scale:
            return f"NUMBER({precision},{scale})"
        elif precision:
            return f"NUMBER({precision})"
        return "NUMBER"
    elif data_type in ("VARCHAR2", "CHAR", "NVARCHAR2"):
        return f"{data_type}({length})" if length else data_type
    return data_type


# ─────────────────────────────────────────────
#  4. MAIN DISCOVERY ORCHESTRATOR
# ─────────────────────────────────────────────

def discover_schema(
    conn,
    schema       : str,
    table_filter : list  = None,
    sample_values: bool  = True,
    sample_limit : int   = 5
) -> dict:
    """
    Full discovery run. Returns the complete schema dict.
    """
    cursor = conn.cursor()
    schema = schema.upper()

    print(f"\n🔍 Discovering schema: {schema}")
    print("─" * 50)

    # ── Tables ────────────────────────────────
    print("📋 Fetching tables...")
    table_rows = discover_tables(cursor, schema, table_filter)
    table_names = [r[0] for r in table_rows]

    # ── PKs, UKs, Indexes ────────────────────
    print("🔑 Fetching primary keys...")
    pk_map = discover_primary_keys(cursor, schema)

    print("🔒 Fetching unique keys...")
    uk_map = discover_unique_keys(cursor, schema)

    print("📑 Fetching indexes...")
    idx_map = discover_indexes(cursor, schema)

    # ── Build table metadata ──────────────────
    print("📊 Fetching column details...")
    tables_dict = {}

    for table_name, num_rows, last_analyzed, table_comment in table_rows:
        print(f"   → {table_name}")
        col_rows = discover_columns(cursor, schema, table_name)
        columns  = {}

        for (col_name, data_type, length, precision, scale,
             nullable, default, col_id, col_comment) in col_rows:

            formatted_type = format_data_type(data_type, length, precision, scale)
            is_pk          = col_name in pk_map.get(table_name, [])

            col_entry = {
                "column_id"        : col_id,
                "data_type"        : formatted_type,
                "nullable"         : nullable == "Y",
                "primary_key"      : is_pk,
                "default_value"    : default.strip() if default else None,
                "description"      : col_comment or "TODO: add description",
                "business_meaning" : "TODO: add business meaning"
            }

            # Sample values (skip PKs to avoid noise)
            if sample_values and not is_pk and data_type not in ("BLOB", "CLOB", "LONG"):
                samples = discover_sample_values(cursor, schema, table_name, col_name, sample_limit)
                if samples:
                    col_entry["sample_values"] = samples

            columns[col_name] = col_entry

        tables_dict[table_name] = {
            "description"         : table_comment or "TODO: add table description",
            "business_context"    : "TODO: describe what this table represents",
            "schema_owner"        : schema,
            "primary_key"         : pk_map.get(table_name, []),
            "unique_keys"         : uk_map.get(table_name, []),
            "indexes"             : idx_map.get(table_name, []),
            "approximate_row_count": num_rows or -1,
            "last_analyzed"       : str(last_analyzed) if last_analyzed else None,
            "columns"             : columns
        }

    # ── Implicit Relationships ────────────────
    print("\n🔗 Detecting implicit relationships (no FK)...")
    relationships = discover_implicit_relationships(
        cursor, schema, table_names, pk_map
    )

    # ── Assemble final schema ─────────────────
    schema_doc = {
        "schema_metadata": {
            "generated_at"     : datetime.utcnow().isoformat() + "Z",
            "schema_owner"     : schema,
            "total_tables"     : len(tables_dict),
            "total_relationships": len(relationships),
            "discovery_method" : "oracle_data_dictionary",
            "fk_constraints_exist": False,
            "relationship_detection": "implicit_pk_column_name_matching",
            "version"          : "1.0"
        },
        "llm_instructions": {
            "purpose": (
                "This schema is provided to help an LLM build dynamic Oracle SQL queries. "
                "Use table descriptions and column descriptions to understand intent. "
                "Use the relationships section to determine correct JOIN conditions. "
                "Always use bind variables for user-supplied IDs. "
                "Never SELECT * — always pick specific columns relevant to the question."
            ),
            "join_rules": [
                "Use relationships[].parent_column and child_column for ON clauses",
                "Default to LEFT JOIN unless business logic requires INNER JOIN",
                "Always alias tables when joining more than 2 tables"
            ],
            "safety_rules": [
                "Only use tables listed in this schema",
                "Only use columns listed under each table",
                "Always filter by the anchor record ID passed by the user",
                "Do not use DML (INSERT/UPDATE/DELETE) — SELECT only"
            ]
        },
        "tables"       : tables_dict,
        "relationships": relationships,
        "common_query_patterns": [
            {
                "pattern_name" : "lookup_by_id",
                "description"  : "Fetch all relevant data for a single record by its primary key",
                "template"     : "SELECT <columns> FROM <anchor_table> LEFT JOIN <child_table> ON <join> WHERE <anchor_table>.<pk> = :record_id"
            },
            {
                "pattern_name" : "multi_table_analysis",
                "description"  : "Join parent and child tables to analyse calculated fields",
                "template"     : "SELECT p.<cols>, c.<cols> FROM <parent> p LEFT JOIN <child> c ON p.<pk> = c.<pk> WHERE p.<pk> = :record_id"
            }
        ]
    }

    return schema_doc


# ─────────────────────────────────────────────
#  5. OUTPUT HELPERS
# ─────────────────────────────────────────────

def save_schema(schema_doc: dict, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(schema_doc, f, indent=2, default=str)
    print(f"\n✅ Schema saved to: {output_path}")


def print_summary(schema_doc: dict):
    meta   = schema_doc["schema_metadata"]
    tables = schema_doc["tables"]
    rels   = schema_doc["relationships"]

    print("\n" + "═" * 55)
    print("  DISCOVERY SUMMARY")
    print("═" * 55)
    print(f"  Schema        : {meta['schema_owner']}")
    print(f"  Generated at  : {meta['generated_at']}")
    print(f"  Tables found  : {meta['total_tables']}")
    print(f"  Relationships : {meta['total_relationships']}")
    print()
    print("  TABLES:")
    for tbl, meta_t in tables.items():
        pk   = ", ".join(meta_t["primary_key"]) or "none"
        rows = meta_t["approximate_row_count"]
        cols = len(meta_t["columns"])
        print(f"    {tbl:<35} PK={pk:<20} cols={cols}  rows≈{rows}")

    if rels:
        print()
        print("  DETECTED RELATIONSHIPS:")
        for r in rels:
            print(f"    {r['parent_table']}.{r['parent_column']}"
                  f"  →  {r['child_table']}.{r['child_column']}"
                  f"  [{r['confidence']}]")
    print("═" * 55)


# ─────────────────────────────────────────────
#  6. CLI ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Auto-discover Oracle DB schema and generate LLM-ready JSON"
    )
    parser.add_argument(
        "--schema", default=None,
        help="Oracle schema/owner to discover (default: ORACLE_USER from .env)"
    )
    parser.add_argument(
        "--tables", nargs="*", default=None,
        help="Specific table names to include (default: all tables in schema)"
    )
    parser.add_argument(
        "--output", default="oracle_schema.json",
        help="Output JSON file path (default: oracle_schema.json)"
    )
    parser.add_argument(
        "--no-samples", action="store_true",
        help="Skip fetching sample column values"
    )
    parser.add_argument(
        "--sample-limit", type=int, default=5,
        help="Max sample values per column (default: 5)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args       = parse_args()
    conn       = get_connection()
    schema     = (args.schema or os.getenv("ORACLE_SCHEMA") or os.getenv("ORACLE_USER")).upper()

    schema_doc = discover_schema(
        conn         = conn,
        schema       = schema,
        table_filter = args.tables,
        sample_values= not args.no_samples,
        sample_limit = args.sample_limit
    )

    print_summary(schema_doc)
    save_schema(schema_doc, args.output)
    conn.close()
