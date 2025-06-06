import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
import openai
from opensearchpy import OpenSearch

class TradingAnalyticsService:
    def __init__(self, opensearch_config: Dict, genai_config: Dict):
        """
        Initialize the service with OpenSearch and GenAI configurations
        """
        # OpenSearch client setup
        self.os_client = OpenSearch(
            hosts=[{'host': opensearch_config['host'], 'port': opensearch_config['port']}],
            http_auth=(opensearch_config['username'], opensearch_config['password']),
            use_ssl=opensearch_config.get('use_ssl', True),
            verify_certs=opensearch_config.get('verify_certs', False),
            ssl_assert_hostname=False,
            ssl_show_warn=False,
        )
        
        # GenAI API configuration
        self.genai_endpoint = genai_config['endpoint']
        self.genai_headers = {
            'Authorization': f"Bearer {genai_config['api_key']}",
            'Content-Type': 'application/json'
        }
        
        # Index name for trading data
        self.trading_index = 'trading_transactions'
    
    def generate_opensearch_query(self, user_question: str) -> Dict:
        """
        Use GenAI to convert natural language question to OpenSearch query
        """
        prompt = f"""
        Convert the following user question into an OpenSearch query JSON format.
        
        User Question: "{user_question}"
        
        Context: We have a trading transactions index with fields:
        - trader_id (keyword)
        - trader_name (text)
        - transaction_amount (double)
        - transaction_date (date)
        - transaction_type (keyword: buy/sell)
        - symbol (keyword)
        - quantity (long)
        - price_per_unit (double)
        
        Generate an OpenSearch query that:
        1. Filters for the last year
        2. Sorts by transaction_amount in descending order
        3. Returns top results
        4. Includes aggregations if needed
        
        Return only the JSON query without explanation.
        """
        
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are an expert in OpenSearch query generation. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }
        
        response = requests.post(self.genai_endpoint, headers=self.genai_headers, json=payload)
        
        if response.status_code == 200:
            query_text = response.json()['choices'][0]['message']['content']
            # Clean up the response to extract JSON
            query_text = query_text.strip()
            if query_text.startswith('```json'):
                query_text = query_text[7:-3]
            elif query_text.startswith('```'):
                query_text = query_text[3:-3]
            
            return json.loads(query_text)
        else:
            # Fallback query if GenAI fails
            return self.get_fallback_query()
    
    def get_fallback_query(self) -> Dict:
        """
        Fallback OpenSearch query for highest transactions in last year
        """
        one_year_ago = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        return {
            "query": {
                "range": {
                    "transaction_date": {
                        "gte": one_year_ago
                    }
                }
            },
            "sort": [
                {
                    "transaction_amount": {
                        "order": "desc"
                    }
                }
            ],
            "size": 10,
            "aggs": {
                "top_traders": {
                    "terms": {
                        "field": "trader_name.keyword",
                        "size": 5,
                        "order": {
                            "max_transaction": "desc"
                        }
                    },
                    "aggs": {
                        "max_transaction": {
                            "max": {
                                "field": "transaction_amount"
                            }
                        }
                    }
                }
            }
        }
    
    def search_transactions(self, query: Dict) -> Dict:
        """
        Execute the OpenSearch query
        """
        try:
            response = self.os_client.search(
                index=self.trading_index,
                body=query
            )
            return response
        except Exception as e:
            print(f"OpenSearch query failed: {str(e)}")
            return {"error": str(e)}
    
    def format_results_with_genai(self, search_results: Dict, original_question: str) -> str:
        """
        Use GenAI to format the search results into a human-readable response
        """
        prompt = f"""
        Original Question: "{original_question}"
        
        OpenSearch Results:
        {json.dumps(search_results, indent=2)}
        
        Format these results into a clear, professional response that:
        1. Directly answers the user's question
        2. Highlights the key findings
        3. Includes specific numbers and trader names
        4. Provides insights about the data
        5. Uses bullet points for clarity
        
        Make it conversational and informative.
        """
        
        payload = {
            "model": "gpt-4",
            "messages": [
                {"role": "system", "content": "You are a financial data analyst. Provide clear, accurate summaries of trading data."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 800
        }
        
        response = requests.post(self.genai_endpoint, headers=self.genai_headers, json=payload)
        
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
        else:
            return self.format_results_fallback(search_results)
    
    def format_results_fallback(self, search_results: Dict) -> str:
        """
        Fallback result formatting without GenAI
        """
        if "error" in search_results:
            return f"Error retrieving data: {search_results['error']}"
        
        hits = search_results.get('hits', {}).get('hits', [])
        
        if not hits:
            return "No transactions found for the specified criteria."
        
        response = "**Highest Transactions by Traders in the Last Year:**\n\n"
        
        for i, hit in enumerate(hits[:5], 1):
            source = hit['_source']
            response += f"{i}. **{source['trader_name']}**\n"
            response += f"   - Amount: ${source['transaction_amount']:,.2f}\n"
            response += f"   - Date: {source['transaction_date']}\n"
            response += f"   - Symbol: {source['symbol']}\n"
            response += f"   - Type: {source['transaction_type'].upper()}\n\n"
        
        # Add aggregation results if available
        aggs = search_results.get('aggregations', {})
        if 'top_traders' in aggs:
            response += "\n**Top Traders by Maximum Transaction:**\n"
            for bucket in aggs['top_traders']['buckets']:
                trader_name = bucket['key']
                max_amount = bucket['max_transaction']['value']
                response += f"- {trader_name}: ${max_amount:,.2f}\n"
        
        return response
    
    def process_user_query(self, user_question: str) -> str:
        """
        Main method to process user query end-to-end
        """
        print(f"Processing query: {user_question}")
        
        # Step 1: Generate OpenSearch query using GenAI
        print("Generating OpenSearch query...")
        os_query = self.generate_opensearch_query(user_question)
        print(f"Generated query: {json.dumps(os_query, indent=2)}")
        
        # Step 2: Execute OpenSearch query
        print("Executing OpenSearch query...")
        search_results = self.search_transactions(os_query)
        
        # Step 3: Format results using GenAI
        print("Formatting results...")
        formatted_response = self.format_results_with_genai(search_results, user_question)
        
        return formatted_response

# Example usage and configuration
def main():
    # Configuration
    opensearch_config = {
        'host': 'localhost',  # Your OpenSearch host
        'port': 9200,
        'username': 'admin',
        'password': 'admin',
        'use_ssl': True,
        'verify_certs': False
    }
    
    genai_config = {
        'endpoint': 'https://your-internal-genai-api.com/v1/chat/completions',  # Your internal GenAI endpoint
        'api_key': 'your-api-key-here'
    }
    
    # Initialize service
    service = TradingAnalyticsService(opensearch_config, genai_config)
    
    # Example queries
    sample_queries = [
        "Give me the highest transaction done by trader in last year",
        "Show me top 5 biggest trades from the past 12 months",
        "Which trader made the largest transaction recently?",
        "Find the highest value transactions in the last year"
    ]
    
    for query in sample_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        result = service.process_user_query(query)
        print(result)

# Additional utility functions for data ingestion
def create_sample_trading_data():
    """
    Create sample trading data for testing
    """
    import random
    from datetime import datetime, timedelta
    
    traders = [
        "John Smith", "Sarah Johnson", "Michael Chen", "Emily Davis", "Robert Wilson",
        "Lisa Anderson", "David Brown", "Jennifer Taylor", "Christopher Lee", "Amanda Martinez"
    ]
    
    symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "META", "NVDA", "NFLX", "ADBE", "CRM"]
    
    sample_data = []
    
    for i in range(1000):
        trader = random.choice(traders)
        symbol = random.choice(symbols)
        transaction_type = random.choice(["buy", "sell"])
        quantity = random.randint(1, 10000)
        price_per_unit = round(random.uniform(10, 500), 2)
        transaction_amount = quantity * price_per_unit
        
        # Generate random date within last year
        days_ago = random.randint(0, 365)
        transaction_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        sample_data.append({
            "trader_id": f"T{i+1:04d}",
            "trader_name": trader,
            "transaction_amount": transaction_amount,
            "transaction_date": transaction_date,
            "transaction_type": transaction_type,
            "symbol": symbol,
            "quantity": quantity,
            "price_per_unit": price_per_unit
        })
    
    return sample_data

def ingest_sample_data(service: TradingAnalyticsService):
    """
    Ingest sample data into OpenSearch
    """
    sample_data = create_sample_trading_data()
    
    # Create index mapping
    mapping = {
        "mappings": {
            "properties": {
                "trader_id": {"type": "keyword"},
                "trader_name": {"type": "text", "fields": {"keyword": {"type": "keyword"}}},
                "transaction_amount": {"type": "double"},
                "transaction_date": {"type": "date"},
                "transaction_type": {"type": "keyword"},
                "symbol": {"type": "keyword"},
                "quantity": {"type": "long"},
                "price_per_unit": {"type": "double"}
            }
        }
    }
    
    # Create index
    service.os_client.indices.create(index=service.trading_index, body=mapping, ignore=400)
    
    # Bulk insert data
    for i, doc in enumerate(sample_data):
        service.os_client.index(
            index=service.trading_index,
            body=doc,
            id=i+1
        )
    
    print(f"Ingested {len(sample_data)} sample trading records")

if __name__ == "__main__":
    main()