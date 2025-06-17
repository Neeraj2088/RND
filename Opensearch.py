#!/usr/bin/env python3
"""
OpenSearch MCP Server for Kibana Query Generation
Generates various types of Kibana/OpenSearch queries based on natural language input
"""

import json
import logging
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass
from enum import Enum

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("opensearch-mcp")

class QueryType(Enum):
    MATCH = "match"
    TERM = "term"
    RANGE = "range"
    BOOL = "bool"
    WILDCARD = "wildcard"
    FUZZY = "fuzzy"
    AGGREGATION = "aggregation"
    MULTI_MATCH = "multi_match"
    REGEXP = "regexp"

@dataclass
class QueryContext:
    """Context for query generation"""
    index_pattern: str = "*"
    time_field: str = "@timestamp"
    default_size: int = 100
    time_range: Optional[Dict[str, str]] = None

class KibanaQueryGenerator:
    """Generates Kibana/OpenSearch queries from natural language"""
    
    def __init__(self):
        self.time_patterns = {
            r'last (\d+) (minute|hour|day|week|month)s?': self._parse_relative_time,
            r'past (\d+) (minute|hour|day|week|month)s?': self._parse_relative_time,
            r'(\d{4}-\d{2}-\d{2})': self._parse_date,
            r'today': lambda: self._get_today_range(),
            r'yesterday': lambda: self._get_yesterday_range(),
            r'this week': lambda: self._get_this_week_range(),
        }
        
        self.field_mappings = {
            'ip': 'client_ip',
            'user': 'user.name',
            'status': 'response.status_code',
            'method': 'request.method',
            'url': 'request.url',
            'host': 'host.name',
            'message': 'message',
            'level': 'log.level',
            'error': 'error.message',
        }
    
    def generate_query(self, user_query: str, context: QueryContext = None) -> Dict[str, Any]:
        """Generate a Kibana query from natural language input"""
        if context is None:
            context = QueryContext()
        
        user_query = user_query.lower().strip()
        
        # Determine query type and generate appropriate query
        if 'aggregate' in user_query or 'group by' in user_query or 'count' in user_query:
            return self._generate_aggregation_query(user_query, context)
        elif 'error' in user_query:
            return self._generate_error_query(user_query, context)
        elif any(op in user_query for op in ['>', '<', '>=', '<=', 'between']):
            return self._generate_range_query(user_query, context)
        elif '*' in user_query or '?' in user_query:
            return self._generate_wildcard_query(user_query, context)
        elif 'fuzzy' in user_query or 'similar' in user_query:
            return self._generate_fuzzy_query(user_query, context)
        elif ' and ' in user_query or ' or ' in user_query or ' not ' in user_query:
            return self._generate_bool_query(user_query, context)
        else:
            return self._generate_match_query(user_query, context)
    
    def _generate_match_query(self, user_query: str, context: QueryContext) -> Dict[str, Any]:
        """Generate a basic match query"""
        time_range = self._extract_time_range(user_query) or context.time_range
        
        # Extract field and value
        field, value = self._extract_field_value(user_query)
        
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "match": {
                                field: value
                            }
                        }
                    ]
                }
            },
            "size": context.default_size,
            "sort": [
                {context.time_field: {"order": "desc"}}
            ]
        }
        
        if time_range:
            query["query"]["bool"]["filter"] = [
                {
                    "range": {
                        context.time_field: time_range
                    }
                }
            ]
        
        return query
    
    def _generate_bool_query(self, user_query: str, context: QueryContext) -> Dict[str, Any]:
        """Generate a boolean query with multiple conditions"""
        time_range = self._extract_time_range(user_query) or context.time_range
        
        # Parse boolean conditions
        must_conditions = []
        should_conditions = []
        must_not_conditions = []
        
        # Split by boolean operators
        parts = re.split(r'\s+(and|or|not)\s+', user_query)
        current_operator = "and"
        
        for i, part in enumerate(parts):
            if part.lower() in ['and', 'or', 'not']:
                current_operator = part.lower()
                continue
            
            field, value = self._extract_field_value(part)
            condition = {"match": {field: value}}
            
            if current_operator == "and":
                must_conditions.append(condition)
            elif current_operator == "or":
                should_conditions.append(condition)
            elif current_operator == "not":
                must_not_conditions.append(condition)
        
        bool_query = {}
        if must_conditions:
            bool_query["must"] = must_conditions
        if should_conditions:
            bool_query["should"] = should_conditions
            bool_query["minimum_should_match"] = 1
        if must_not_conditions:
            bool_query["must_not"] = must_not_conditions
        
        query = {
            "query": {
                "bool": bool_query
            },
            "size": context.default_size,
            "sort": [
                {context.time_field: {"order": "desc"}}
            ]
        }
        
        if time_range:
            if "filter" not in query["query"]["bool"]:
                query["query"]["bool"]["filter"] = []
            query["query"]["bool"]["filter"].append(
                {
                    "range": {
                        context.time_field: time_range
                    }
                }
            )
        
        return query
    
    def _generate_range_query(self, user_query: str, context: QueryContext) -> Dict[str, Any]:
        """Generate a range query"""
        time_range = self._extract_time_range(user_query) or context.time_range
        
        # Extract range conditions
        range_conditions = []
        
        # Look for numeric ranges
        range_patterns = [
            r'(\w+)\s*>\s*(\d+)',
            r'(\w+)\s*<\s*(\d+)',
            r'(\w+)\s*>=\s*(\d+)',
            r'(\w+)\s*<=\s*(\d+)',
            r'(\w+)\s+between\s+(\d+)\s+and\s+(\d+)'
        ]
        
        for pattern in range_patterns:
            matches = re.findall(pattern, user_query)
            for match in matches:
                if len(match) == 2:  # Simple comparison
                    field, value = match
                    field = self._map_field(field)
                    
                    if '>=' in user_query:
                        range_conditions.append({
                            "range": {field: {"gte": int(value)}}
                        })
                    elif '>' in user_query:
                        range_conditions.append({
                            "range": {field: {"gt": int(value)}}
                        })
                    elif '<=' in user_query:
                        range_conditions.append({
                            "range": {field: {"lte": int(value)}}
                        })
                    elif '<' in user_query:
                        range_conditions.append({
                            "range": {field: {"lt": int(value)}}
                        })
                elif len(match) == 3:  # Between
                    field, min_val, max_val = match
                    field = self._map_field(field)
                    range_conditions.append({
                        "range": {
                            field: {
                                "gte": int(min_val),
                                "lte": int(max_val)
                            }
                        }
                    })
        
        query = {
            "query": {
                "bool": {
                    "must": range_conditions
                }
            },
            "size": context.default_size,
            "sort": [
                {context.time_field: {"order": "desc"}}
            ]
        }
        
        if time_range:
            if "filter" not in query["query"]["bool"]:
                query["query"]["bool"]["filter"] = []
            query["query"]["bool"]["filter"].append(
                {
                    "range": {
                        context.time_field: time_range
                    }
                }
            )
        
        return query
    
    def _generate_wildcard_query(self, user_query: str, context: QueryContext) -> Dict[str, Any]:
        """Generate a wildcard query"""
        time_range = self._extract_time_range(user_query) or context.time_range
        
        field, value = self._extract_field_value(user_query)
        
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "wildcard": {
                                field: value
                            }
                        }
                    ]
                }
            },
            "size": context.default_size,
            "sort": [
                {context.time_field: {"order": "desc"}}
            ]
        }
        
        if time_range:
            query["query"]["bool"]["filter"] = [
                {
                    "range": {
                        context.time_field: time_range
                    }
                }
            ]
        
        return query
    
    def _generate_fuzzy_query(self, user_query: str, context: QueryContext) -> Dict[str, Any]:
        """Generate a fuzzy query"""
        time_range = self._extract_time_range(user_query) or context.time_range
        
        field, value = self._extract_field_value(user_query)
        # Remove fuzzy indicators from value
        value = re.sub(r'\b(fuzzy|similar)\b', '', value).strip()
        
        query = {
            "query": {
                "bool": {
                    "must": [
                        {
                            "fuzzy": {
                                field: {
                                    "value": value,
                                    "fuzziness": "AUTO"
                                }
                            }
                        }
                    ]
                }
            },
            "size": context.default_size,
            "sort": [
                {context.time_field: {"order": "desc"}}
            ]
        }
        
        if time_range:
            query["query"]["bool"]["filter"] = [
                {
                    "range": {
                        context.time_field: time_range
                    }
                }
            ]
        
        return query
    
    def _generate_aggregation_query(self, user_query: str, context: QueryContext) -> Dict[str, Any]:
        """Generate an aggregation query"""
        time_range = self._extract_time_range(user_query) or context.time_range
        
        # Determine aggregation type and field
        agg_field = self._extract_aggregation_field(user_query)
        
        query = {
            "query": {
                "bool": {
                    "must": [
                        {"match_all": {}}
                    ]
                }
            },
            "size": 0,
            "aggs": {
                "grouped_data": {
                    "terms": {
                        "field": f"{agg_field}.keyword",
                        "size": 50
                    }
                }
            }
        }
        
        # Add date histogram if time-based aggregation
        if 'over time' in user_query or 'timeline' in user_query:
            query["aggs"]["timeline"] = {
                "date_histogram": {
                    "field": context.time_field,
                    "fixed_interval": "1h"
                }
            }
        
        if time_range:
            query["query"]["bool"]["filter"] = [
                {
                    "range": {
                        context.time_field: time_range
                    }
                }
            ]
        
        return query
    
    def _generate_error_query(self, user_query: str, context: QueryContext) -> Dict[str, Any]:
        """Generate a query specifically for errors"""
        time_range = self._extract_time_range(user_query) or context.time_range
        
        error_conditions = [
            {"range": {"response.status_code": {"gte": 400}}},
            {"match": {"log.level": "ERROR"}},
            {"exists": {"field": "error.message"}}
        ]
        
        query = {
            "query": {
                "bool": {
                    "should": error_conditions,
                    "minimum_should_match": 1
                }
            },
            "size": context.default_size,
            "sort": [
                {context.time_field: {"order": "desc"}}
            ]
        }
        
        if time_range:
            query["query"]["bool"]["filter"] = [
                {
                    "range": {
                        context.time_field: time_range
                    }
                }
            ]
        
        return query
    
    def _extract_field_value(self, query: str) -> tuple:
        """Extract field and value from query"""
        # Remove time-related parts
        query_clean = re.sub(r'\b(last|past|today|yesterday|this week|\d{4}-\d{2}-\d{2})\b.*', '', query).strip()
        
        # Look for field:value patterns
        field_value_match = re.search(r'(\w+):\s*([^\s]+)', query_clean)
        if field_value_match:
            field, value = field_value_match.groups()
            return self._map_field(field), value
        
        # Look for "field is/equals value" patterns
        equals_match = re.search(r'(\w+)\s+(is|equals?)\s+([^\s]+)', query_clean)
        if equals_match:
            field, _, value = equals_match.groups()
            return self._map_field(field), value
        
        # Default to message field
        return "message", query_clean
    
    def _extract_aggregation_field(self, query: str) -> str:
        """Extract field for aggregation"""
        # Look for "group by field" patterns
        group_by_match = re.search(r'group by (\w+)', query)
        if group_by_match:
            return self._map_field(group_by_match.group(1))
        
        # Look for "count field" patterns
        count_match = re.search(r'count (\w+)', query)
        if count_match:
            return self._map_field(count_match.group(1))
        
        # Default aggregation fields
        for field in ['status', 'user', 'host', 'method']:
            if field in query:
                return self._map_field(field)
        
        return "host.name"  # Default field
    
    def _extract_time_range(self, query: str) -> Optional[Dict[str, str]]:
        """Extract time range from query"""
        for pattern, parser in self.time_patterns.items():
            match = re.search(pattern, query)
            if match:
                if callable(parser):
                    return parser()
                else:
                    return parser(match.groups())
        return None
    
    def _parse_relative_time(self, groups: tuple) -> Dict[str, str]:
        """Parse relative time expressions"""
        amount, unit = groups
        amount = int(amount)
        
        unit_mapping = {
            'minute': 'minutes',
            'hour': 'hours', 
            'day': 'days',
            'week': 'weeks',
            'month': 'months'
        }
        
        now = datetime.utcnow()
        if unit in ['minute', 'minutes']:
            start_time = now - timedelta(minutes=amount)
        elif unit in ['hour', 'hours']:
            start_time = now - timedelta(hours=amount)
        elif unit in ['day', 'days']:
            start_time = now - timedelta(days=amount)
        elif unit in ['week', 'weeks']:
            start_time = now - timedelta(weeks=amount)
        elif unit in ['month', 'months']:
            start_time = now - timedelta(days=amount * 30)  # Approximate
        else:
            start_time = now - timedelta(hours=1)  # Default
        
        return {
            "gte": start_time.isoformat(),
            "lte": now.isoformat()
        }
    
    def _parse_date(self, groups: tuple) -> Dict[str, str]:
        """Parse specific date"""
        date_str = groups[0]
        start_date = datetime.strptime(date_str, "%Y-%m-%d")
        end_date = start_date + timedelta(days=1)
        
        return {
            "gte": start_date.isoformat(),
            "lt": end_date.isoformat()
        }
    
    def _get_today_range(self) -> Dict[str, str]:
        """Get today's date range"""
        now = datetime.utcnow()
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        
        return {
            "gte": start_of_day.isoformat(),
            "lte": now.isoformat()
        }
    
    def _get_yesterday_range(self) -> Dict[str, str]:
        """Get yesterday's date range"""
        now = datetime.utcnow()
        yesterday = now - timedelta(days=1)
        start_of_day = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        
        return {
            "gte": start_of_day.isoformat(),
            "lte": end_of_day.isoformat()
        }
    
    def _get_this_week_range(self) -> Dict[str, str]:
        """Get this week's date range"""
        now = datetime.utcnow()
        start_of_week = now - timedelta(days=now.weekday())
        start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
        
        return {
            "gte": start_of_week.isoformat(),
            "lte": now.isoformat()
        }
    
    def _map_field(self, field: str) -> str:
        """Map common field names to actual field names"""
        return self.field_mappings.get(field.lower(), field)

# Initialize the MCP server
server = Server("opensearch-mcp")
query_generator = KibanaQueryGenerator()

@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List available tools"""
    return [
        types.Tool(
            name="generate_kibana_query",
            description="Generate Kibana/OpenSearch queries from natural language",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language query to convert to Kibana query"
                    },
                    "index_pattern": {
                        "type": "string",
                        "description": "Index pattern to search (default: *)",
                        "default": "*"
                    },
                    "time_field": {
                        "type": "string",
                        "description": "Time field name (default: @timestamp)",
                        "default": "@timestamp"
                    },
                    "size": {
                        "type": "integer",
                        "description": "Number of results to return (default: 100)",
                        "default": 100
                    }
                },
                "required": ["query"]
            }
        ),
        types.Tool(
            name="explain_query",
            description="Explain what a Kibana query does in natural language",
            inputSchema={
                "type": "object",
                "properties": {
                    "kibana_query": {
                        "type": "string",
                        "description": "Kibana query JSON to explain"
                    }
                },
                "required": ["kibana_query"]
            }
        ),
        types.Tool(
            name="optimize_query",
            description="Optimize a Kibana query for better performance",
            inputSchema={
                "type": "object",
                "properties": {
                    "kibana_query": {
                        "type": "string",
                        "description": "Kibana query JSON to optimize"
                    }
                },
                "required": ["kibana_query"]
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Handle tool calls"""
    
    if name == "generate_kibana_query":
        user_query = arguments.get("query", "")
        index_pattern = arguments.get("index_pattern", "*")
        time_field = arguments.get("time_field", "@timestamp")
        size = arguments.get("size", 100)
        
        context = QueryContext(
            index_pattern=index_pattern,
            time_field=time_field,
            default_size=size
        )
        
        try:
            kibana_query = query_generator.generate_query(user_query, context)
            
            return [
                types.TextContent(
                    type="text",
                    text=f"Generated Kibana Query:\n\n```json\n{json.dumps(kibana_query, indent=2)}\n```\n\n"
                         f"Query Type: {_determine_query_type(kibana_query)}\n"
                         f"Description: {_describe_query(kibana_query, user_query)}"
                )
            ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=f"Error generating query: {str(e)}"
                )
            ]
    
    elif name == "explain_query":
        try:
            query_json = arguments.get("kibana_query", "")
            query_dict = json.loads(query_json)
            explanation = _explain_kibana_query(query_dict)
            
            return [
                types.TextContent(
                    type="text",
                    text=f"Query Explanation:\n\n{explanation}"
                )
            ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=f"Error explaining query: {str(e)}"
                )
            ]
    
    elif name == "optimize_query":
        try:
            query_json = arguments.get("kibana_query", "")
            query_dict = json.loads(query_json)
            optimized_query = _optimize_kibana_query(query_dict)
            
            return [
                types.TextContent(
                    type="text",
                    text=f"Optimized Query:\n\n```json\n{json.dumps(optimized_query, indent=2)}\n```\n\n"
                         f"Optimizations Applied:\n{_get_optimization_notes(query_dict, optimized_query)}"
                )
            ]
        except Exception as e:
            return [
                types.TextContent(
                    type="text",
                    text=f"Error optimizing query: {str(e)}"
                )
            ]
    
    else:
        raise ValueError(f"Unknown tool: {name}")

def _determine_query_type(query: Dict[str, Any]) -> str:
    """Determine the type of query"""
    query_part = query.get("query", {})
    
    if "bool" in query_part:
        return "Boolean Query"
    elif "match" in query_part:
        return "Match Query"
    elif "term" in query_part:
        return "Term Query"
    elif "range" in query_part:
        return "Range Query"
    elif "wildcard" in query_part:
        return "Wildcard Query"
    elif "fuzzy" in query_part:
        return "Fuzzy Query"
    elif query.get("aggs"):
        return "Aggregation Query"
    else:
        return "Unknown Query Type"

def _describe_query(query: Dict[str, Any], original_query: str) -> str:
    """Describe what the query does"""
    description = f"This query searches for: '{original_query}'\n\n"
    
    # Analyze query structure
    query_part = query.get("query", {})
    
    if "bool" in query_part:
        bool_query = query_part["bool"]
        if "must" in bool_query:
            description += f"- Must match: {len(bool_query['must'])} conditions\n"
        if "should" in bool_query:
            description += f"- Should match: {len(bool_query['should'])} conditions\n"
        if "must_not" in bool_query:
            description += f"- Must not match: {len(bool_query['must_not'])} conditions\n"
        if "filter" in bool_query:
            description += f"- Filtered by: {len(bool_query['filter'])} conditions\n"
    
    if query.get("aggs"):
        description += "- Includes aggregations for data analysis\n"
    
    if query.get("sort"):
        description += f"- Sorted by: {list(query['sort'][0].keys())[0]}\n"
    
    description += f"- Returns up to {query.get('size', 'all')} results"
    
    return description

def _explain_kibana_query(query: Dict[str, Any]) -> str:
    """Explain a Kibana query in natural language"""
    explanation = "This Kibana query performs the following operations:\n\n"
    
    # Analyze query section
    query_part = query.get("query", {})
    if query_part:
        explanation += "**Query Section:**\n"
        explanation += _explain_query_section(query_part)
        explanation += "\n"
    
    # Analyze aggregations
    aggs = query.get("aggs", {})
    if aggs:
        explanation += "**Aggregations:**\n"
        explanation += _explain_aggregations(aggs)
        explanation += "\n"
    
    # Analyze sorting
    sort = query.get("sort", [])
    if sort:
        explanation += "**Sorting:**\n"
        for sort_item in sort:
            for field, config in sort_item.items():
                order = config.get("order", "asc") if isinstance(config, dict) else "asc"
                explanation += f"- Sort by {field} in {order}ending order\n"
        explanation += "\n"
    
    # Analyze size
    size = query.get("size", 10)
    explanation += f"**Result Size:** Returns {size} documents"
    if size == 0:
        explanation += " (aggregation-only query)"
    
    return explanation

def _explain_query_section(query_part: Dict[str, Any]) -> str:
    """Explain the query section"""
    explanation = ""
    
    if "match_all" in query_part:
        explanation += "- Matches all documents\n"
    elif "match" in query_part:
        for field, value in query_part["match"].items():
            explanation += f"- Match field '{field}' with value '{value}'\n"
    elif "term" in query_part:
        for field, value in query_part["term"].items():
            explanation += f"- Exact match field '{field}' with value '{value}'\n"
    elif "bool" in query_part:
        bool_query = query_part["bool"]
        if "must" in bool_query:
            explanation += f"- Must satisfy {len(bool_query['must'])} conditions:\n"
            for condition in bool_query["must"]:
                explanation += f"  - {_explain_condition(condition)}\n"
        if "should" in bool_query:
            explanation += f"- Should satisfy at least one of {len(bool_query['should'])} conditions:\n"
            for condition in bool_query["should"]:
                explanation += f"  - {_explain_condition(condition)}\n"
        if "must_not" in bool_query:
            explanation += f"- Must not satisfy {len(bool_query['must_not'])} conditions:\n"
            for condition in bool_query["must_not"]:
                explanation += f"  - {_explain_condition(condition)}\n"
        if "filter" in bool_query:
            explanation += f"- Filtered by {len(bool_query['filter'])} conditions:\n"
            for condition in bool_query["filter"]:
                explanation += f"  - {_explain_condition(condition)}\n"
    
    return explanation

def _explain_condition(condition: Dict[str, Any]) -> str:
    """Explain a single query condition"""
    if "match" in condition:
        field, value = next(iter(condition["match"].items()))
        return f"Match '{field}' with '{value}'"
    elif "term" in condition:
        field, value = next(iter(condition["term"].items()))
        return f"Exact match '{field}' with '{value}'"
    elif "range" in condition:
        field, range_config = next(iter(condition["range"].items()))
        range_desc = []
        for op, value in range_config.items():
            op_desc = {"gte": ">=", "gt": ">", "lte": "<=", "lt": "<"}.get(op, op)
            range_desc.append(f"{op_desc} {value}")
        return f"Range on '{field}': {' and '.join(range_desc)}"
    elif "wildcard" in condition:
        field, value = next(iter(condition["wildcard"].items()))
        return f"Wildcard match '{field}' with pattern '{value}'"
    elif "fuzzy" in condition:
        field, config = next(iter(condition["fuzzy"].items()))
        if isinstance(config, dict):
            value = config.get("value", "")
            return f"Fuzzy match '{field}' with '{value}'"
        else:
            return f"Fuzzy match '{field}' with '{config}'"
    elif "exists" in condition:
        field = condition["exists"]["field"]
        return f"Field '{field}' exists"
    else:
        return f"Unknown condition: {condition}"

def _explain_aggregations(aggs: Dict[str, Any]) -> str:
    """Explain aggregations"""
    explanation = ""
    
    for agg_name, agg_config in aggs.items():
        if "terms" in agg_config:
            field = agg_config["terms"]["field"]
            size = agg_config["terms"].get("size", 10)
            explanation += f"- Group by '{field}' (top {size} values)\n"
        elif "date_histogram" in agg_config:
            field = agg_config["date_histogram"]["field"]
            interval = agg_config["date_histogram"].get("fixed_interval", "auto")
            explanation += f"- Time histogram on '{field}' with {interval} intervals\n"
        elif "histogram" in agg_config:
            field = agg_config["histogram"]["field"]
            interval = agg_config["histogram"]["interval"]
            explanation += f"- Numeric histogram on '{field}' with interval {interval}\n"
        elif "avg" in agg_config:
            field = agg_config["avg"]["field"]
            explanation += f"- Calculate average of '{field}'\n"
        elif "sum" in agg_config:
            field = agg_config["sum"]["field"]
            explanation += f"- Calculate sum of '{field}'\n"
        elif "max" in agg_config:
            field = agg_config["max"]["field"]
            explanation += f"- Find maximum value of '{field}'\n"
        elif "min" in agg_config:
            field = agg_config["min"]["field"]
            explanation += f"- Find minimum value of '{field}'\n"
        elif "cardinality" in agg_config:
            field = agg_config["cardinality"]["field"]
            explanation += f"- Count unique values in '{field}'\n"
        else:
            explanation += f"- {agg_name}: {list(agg_config.keys())[0]} aggregation\n"
    
    return explanation

def _optimize_kibana_query(query: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize a Kibana query for better performance"""
    optimized = query.copy()
    
    # Add _source filtering to reduce data transfer
    if "_source" not in optimized and optimized.get("size", 10) > 0:
        optimized["_source"] = ["@timestamp", "message", "host.name", "log.level"]
    
    # Add timeout to prevent long-running queries
    if "timeout" not in optimized:
        optimized["timeout"] = "30s"
    
    # Optimize bool queries
    if "query" in optimized and "bool" in optimized["query"]:
        bool_query = optimized["query"]["bool"]
        
        # Move range queries to filter context for better caching
        if "must" in bool_query:
            must_clauses = bool_query["must"].copy()
            filter_clauses = bool_query.get("filter", [])
            
            new_must = []
            for clause in must_clauses:
                if "range" in clause:
                    filter_clauses.append(clause)
                else:
                    new_must.append(clause)
            
            bool_query["must"] = new_must
            if filter_clauses:
                bool_query["filter"] = filter_clauses
    
    # Optimize aggregations
    if "aggs" in optimized:
        for agg_name, agg_config in optimized["aggs"].items():
            if "terms" in agg_config:
                # Limit terms aggregation size
                if agg_config["terms"].get("size", 10) > 1000:
                    agg_config["terms"]["size"] = 1000
                
                # Add execution hint for better performance
                if "execution_hint" not in agg_config["terms"]:
                    agg_config["terms"]["execution_hint"] = "map"
    
    # Set reasonable size limits
    if optimized.get("size", 10) > 10000:
        optimized["size"] = 10000
    
    return optimized

def _get_optimization_notes(original: Dict[str, Any], optimized: Dict[str, Any]) -> str:
    """Get notes about what optimizations were applied"""
    notes = []
    
    if "_source" in optimized and "_source" not in original:
        notes.append("- Added _source filtering to reduce data transfer")
    
    if "timeout" in optimized and "timeout" not in original:
        notes.append("- Added 30s timeout to prevent long-running queries")
    
    if original.get("size", 10) != optimized.get("size", 10):
        notes.append(f"- Limited result size from {original.get('size', 10)} to {optimized.get('size', 10)}")
    
    # Check for bool query optimizations
    orig_bool = original.get("query", {}).get("bool", {})
    opt_bool = optimized.get("query", {}).get("bool", {})
    
    if len(opt_bool.get("filter", [])) > len(orig_bool.get("filter", [])):
        notes.append("- Moved range queries to filter context for better caching")
    
    # Check aggregation optimizations
    orig_aggs = original.get("aggs", {})
    opt_aggs = optimized.get("aggs", {})
    
    for agg_name in orig_aggs:
        if agg_name in opt_aggs:
            orig_terms = orig_aggs[agg_name].get("terms", {})
            opt_terms = opt_aggs[agg_name].get("terms", {})
            
            if orig_terms.get("size", 10) != opt_terms.get("size", 10):
                notes.append(f"- Limited aggregation size for {agg_name}")
            
            if "execution_hint" in opt_terms and "execution_hint" not in orig_terms:
                notes.append(f"- Added execution hint for {agg_name} aggregation")
    
    if not notes:
        notes.append("- No optimizations needed - query is already well-optimized")
    
    return "\n".join(notes)

async def main():
    """Main entry point for the MCP server"""
    # Use stdin/stdout for communication
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="opensearch-mcp",
                server_version="1.0.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
