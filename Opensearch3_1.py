import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import asyncio
from datetime import datetime, timedelta
import numpy as np
from typing import Dict, List, Any
import re

class MockMCPClient:
    """Mock MCP client for testing without actual OpenSearch connection."""
    
    def __init__(self):
        self.sample_data = self._generate_sample_opensearch_response()
    
    def _generate_sample_opensearch_response(self) -> Dict[str, Any]:
        """Generate realistic OpenSearch aggregation response."""
        
        products = ["Laptop Pro", "Smartphone X", "Wireless Headphones", 
                   "Gaming Monitor", "Mechanical Keyboard", "Wireless Mouse",
                   "Tablet Ultra", "Smart Watch", "Bluetooth Speaker", "USB-C Hub"]
        
        months = ["2024-01", "2024-02", "2024-03", "2024-04", "2024-05", "2024-06",
                 "2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12"]
        
        # Generate realistic monthly buckets
        monthly_buckets = []
        
        for month in months:
            month_index = int(month.split('-')[1]) - 1
            product_buckets = []
            
            for product in products:
                # Create realistic sales patterns with seasonality and trends
                base_value = np.random.normal(5000, 1500)  # Base monthly sales
                seasonal_factor = 1 + 0.4 * np.sin(2 * np.pi * month_index / 12)  # Seasonal variation
                trend_factor = 1 + (month_index * 0.03)  # 3% monthly growth
                noise = np.random.normal(1, 0.1)  # Random noise
                
                # Special patterns for different products
                if "Gaming" in product and month_index in [10, 11]:  # Holiday boost for gaming
                    seasonal_factor *= 1.5
                elif "Smart Watch" in product and month_index in [0, 1]:  # New Year fitness boost
                    seasonal_factor *= 1.3
                
                sales_value = max(1000, base_value * seasonal_factor * trend_factor * noise)
                
                product_buckets.append({
                    "key": product,
                    "doc_count": int(sales_value / 100),  # Approximate number of transactions
                    "total_sales": {
                        "value": round(sales_value, 2)
                    }
                })
            
            monthly_buckets.append({
                "key_as_string": month,
                "key": int(month.replace("-", "")),
                "doc_count": sum(p["doc_count"] for p in product_buckets),
                "products": {
                    "buckets": product_buckets
                }
            })
        
        return {
            "aggregations": {
                "monthly_trends": {
                    "buckets": monthly_buckets
                }
            },
            "hits": {
                "total": {"value": 50000}
            }
        }
    
    async def search_products(self, query: Dict[str, Any], index: str = "products") -> Dict[str, Any]:
        """Mock search that returns sample data."""
        print(f"🔍 Mock search executed on index '{index}'")
        print(f"📊 Query processed: {json.dumps(query, indent=2)}")
        
        # Simulate some processing delay
        await asyncio.sleep(0.5)
        
        return self.sample_data

class SimpleNLQueryProcessor:
    """Simplified natural language query processor."""
    
    def parse_and_build_query(self, nl_query: str) -> Dict[str, Any]:
        """Parse natural language and build OpenSearch query."""
        
        # Extract year (default to 2024)
        year_match = re.search(r'(\d{4})', nl_query)
        year = int(year_match.group(1)) if year_match else 2024
        
        # Build the OpenSearch aggregation query
        query = {
            "query": {
                "bool": {
                    "filter": [
                        {
                            "range": {
                                "timestamp": {
                                    "gte": f"{year}-01-01",
                                    "lte": f"{year}-12-31",
                                    "format": "yyyy-MM-dd"
                                }
                            }
                        }
                    ]
                }
            },
            "aggs": {
                "monthly_trends": {
                    "date_histogram": {
                        "field": "timestamp",
                        "calendar_interval": "month",
                        "format": "yyyy-MM"
                    },
                    "aggs": {
                        "products": {
                            "terms": {
                                "field": "product_name.keyword",
                                "size": 20
                            },
                            "aggs": {
                                "total_sales": {
                                    "sum": {
                                        "field": "sales_amount"
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "size": 0
        }
        
        return query

class ProductTrendAnalyzer:
    """Analyze and visualize product trends."""
    
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
    
    def analyze_trends(self, opensearch_results: Dict[str, Any]) -> pd.DataFrame:
        """Convert OpenSearch results to pandas DataFrame for analysis."""
        
        data = []
        monthly_buckets = opensearch_results['aggregations']['monthly_trends']['buckets']
        
        for month_bucket in monthly_buckets:
            month = month_bucket['key_as_string']
            products = month_bucket['products']['buckets']
            
            for product in products:
                data.append({
                    'month': month,
                    'product': product['key'],
                    'sales': product['total_sales']['value'],
                    'transactions': product['doc_count']
                })
        
        df = pd.DataFrame(data)
        df['month'] = pd.to_datetime(df['month'])
        return df
    
    def create_trend_visualizations(self, df: pd.DataFrame):
        """Create comprehensive trend visualizations."""
        
        # Set up the plot grid
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle('📈 Product Trends Analysis 2024', fontsize=20, fontweight='bold')
        
        # Create subplots
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Line plot of top products over time
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_top_products_timeline(df, ax1)
        
        # 2. Total sales by product (bar chart)
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_total_sales_by_product(df, ax2)
        
        # 3. Growth analysis
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_growth_rates(df, ax3)
        
        # 4. Monthly heatmap
        ax4 = fig.add_subplot(gs[2, :])
        self._plot_monthly_heatmap(df, ax4)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_top_products_timeline(self, df: pd.DataFrame, ax):
        """Plot timeline for top 6 products by total sales."""
        
        # Get top products by total sales
        top_products = df.groupby('product')['sales'].sum().nlargest(6).index
        
        for product in top_products:
            product_data = df[df['product'] == product].sort_values('month')
            ax.plot(product_data['month'], product_data['sales'], 
                   marker='o', linewidth=2.5, markersize=6, label=product)
        
        ax.set_title('🔥 Top Products Sales Timeline', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Sales ($)', fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Format y-axis to show values in thousands
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    def _plot_total_sales_by_product(self, df: pd.DataFrame, ax):
        """Plot total sales by product."""
        
        total_sales = df.groupby('product')['sales'].sum().sort_values(ascending=True)
        
        bars = ax.barh(total_sales.index, total_sales.values, color='skyblue', alpha=0.8)
        ax.set_title('💰 Total Sales by Product', fontsize=14, fontweight='bold')
        ax.set_xlabel('Total Sales ($)', fontsize=12)
        
        # Add value labels
        for bar, value in zip(bars, total_sales.values):
            ax.text(value + value*0.01, bar.get_y() + bar.get_height()/2, 
                   f'${value/1000:.0f}K', va='center', fontsize=10)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    def _plot_growth_rates(self, df: pd.DataFrame, ax):
        """Plot growth rates from first to last month."""
        
        growth_data = []
        for product in df['product'].unique():
            product_df = df[df['product'] == product].sort_values('month')
            if len(product_df) >= 2:
                first_month = product_df.iloc[0]['sales']
                last_month = product_df.iloc[-1]['sales']
                if first_month > 0:
                    growth_rate = ((last_month - first_month) / first_month) * 100
                    growth_data.append({'product': product, 'growth_rate': growth_rate})
        
        growth_df = pd.DataFrame(growth_data).sort_values('growth_rate')
        
        colors = ['red' if x < 0 else 'green' for x in growth_df['growth_rate']]
        bars = ax.barh(growth_df['product'], growth_df['growth_rate'], color=colors, alpha=0.7)
        
        ax.set_title('📊 Growth Rate (Jan to Dec)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Growth Rate (%)', fontsize=12)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add percentage labels
        for bar, rate in zip(bars, growth_df['growth_rate']):
            ax.text(rate + (2 if rate > 0 else -2), bar.get_y() + bar.get_height()/2,
                   f'{rate:.1f}%', ha='left' if rate > 0 else 'right', va='center')
    
    def _plot_monthly_heatmap(self, df: pd.DataFrame, ax):
        """Plot monthly sales heatmap."""
        
        # Pivot data for heatmap
        heatmap_data = df.pivot_table(values='sales', index='product', columns='month', fill_value=0)
        
        # Format month columns for better display
        heatmap_data.columns = [col.strftime('%b') for col in heatmap_data.columns]
        
        # Create heatmap
        sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', 
                   cbar_kws={'label': 'Sales ($)'}, ax=ax)
        
        ax.set_title('🔥 Monthly Sales Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Product', fontsize=12)
    
    def print_trend_summary(self, df: pd.DataFrame):
        """Print a text summary of trends."""
        
        print("\n" + "="*60)
        print("📊 PRODUCT TRENDS SUMMARY 2024")
        print("="*60)
        
        # Top products by total sales
        top_products = df.groupby('product')['sales'].sum().nlargest(5)
        print("\n🏆 TOP 5 PRODUCTS BY TOTAL SALES:")
        for i, (product, sales) in enumerate(top_products.items(), 1):
            print(f"  {i}. {product}: ${sales:,.0f}")
        
        # Growth analysis
        print("\n📈 GROWTH ANALYSIS (Jan vs Dec):")
        growth_data = []
        for product in df['product'].unique():
            product_df = df[df['product'] == product].sort_values('month')
            if len(product_df) >= 2:
                first_month = product_df.iloc[0]['sales']
                last_month = product_df.iloc[-1]['sales']
                if first_month > 0:
                    growth_rate = ((last_month - first_month) / first_month) * 100
                    growth_data.append({'product': product, 'growth_rate': growth_rate})
        
        growth_df = pd.DataFrame(growth_data).sort_values('growth_rate', ascending=False)
        
        print("  Top Growers:")
        for _, row in growth_df.head(3).iterrows():
            print(f"    📊 {row['product']}: +{row['growth_rate']:.1f}%")
        
        print("  Declining Products:")
        declining = growth_df[growth_df['growth_rate'] < 0]
        if not declining.empty:
            for _, row in declining.head(3).iterrows():
                print(f"    📉 {row['product']}: {row['growth_rate']:.1f}%")
        else:
            print("    🎉 No declining products!")
        
        # Monthly insights
        monthly_totals = df.groupby('month')['sales'].sum()
        best_month = monthly_totals.idxmax().strftime('%B')
        worst_month = monthly_totals.idxmin().strftime('%B')
        
        print(f"\n📅 MONTHLY INSIGHTS:")
        print(f"  🔥 Best Month: {best_month} (${monthly_totals.max():,.0f})")
        print(f"  ❄️  Slowest Month: {worst_month} (${monthly_totals.min():,.0f})")
        
        # Overall trend
        total_sales = df['sales'].sum()
        print(f"\n💰 TOTAL SALES 2024: ${total_sales:,.0f}")
        print("="*60)

class ProductTrendClient:
    """Main client class that orchestrates the workflow."""
    
    def __init__(self, use_mock: bool = True):
        self.use_mock = use_mock
        self.query_processor = SimpleNLQueryProcessor()
        self.analyzer = ProductTrendAnalyzer()
        
        if use_mock:
            self.mcp_client = MockMCPClient()
        else:
            # Here you would initialize the real MCP client
            raise NotImplementedError("Real MCP client not implemented in this demo")
    
    async def analyze_trends(self, natural_query: str):
        """Main method to analyze trends from natural language query."""
        
        print("🚀 Starting Product Trend Analysis")
        print(f"📝 Query: '{natural_query}'")
        print("-" * 50)
        
        # Step 1: Process natural language query
        opensearch_query = self.query_processor.parse_and_build_query(natural_query)
        print("✅ Natural language query processed")
        
        # Step 2: Execute search through MCP
        print("🔍 Executing search through MCP...")
        results = await self.mcp_client.search_products(opensearch_query)
        print("✅ OpenSearch results received")
        
        # Step 3: Analyze trends
        print("📊 Analyzing trends...")
        df = self.analyzer.analyze_trends(results)
        print(f"✅ Processed {len(df)} data points across {df['product'].nunique()} products")
        
        # Step 4: Create visualizations
        print("📈 Creating visualizations...")
        self.analyzer.create_trend_visualizations(df)
        
        # Step 5: Print summary
        self.analyzer.print_trend_summary(df)
        
        return df

# Demo function
async def main():
    """Main demo function."""
    
    print("🎯 Product Trend Analysis Demo")
    print("Using Mock Data for OpenSearch via MCP")
    print("="*50)
    
    # Initialize client with mock data
    client = ProductTrendClient(use_mock=True)
    
    # Natural language query
    query = "Show me trend of all products in 2024"
    
    # Run analysis
    try:
        df = await client.analyze_trends(query)
        print("\n✅ Analysis completed successfully!")
        print(f"📊 Data shape: {df.shape}")
        
        # Show sample data
        print("\n📋 Sample Data:")
        print(df.head(10).to_string(index=False))
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

# Additional utility functions for real integration
def create_sample_opensearch_index_data():
    """Generate sample data that you can index into OpenSearch for testing."""
    
    products = ["Laptop Pro", "Smartphone X", "Wireless Headphones", 
               "Gaming Monitor", "Mechanical Keyboard", "Wireless Mouse",
               "Tablet Ultra", "Smart Watch", "Bluetooth Speaker", "USB-C Hub"]
    
    data = []
    
    for month in range(1, 13):  # Jan to Dec 2024
        for day in range(1, 31, 3):  # Sample every 3 days
            try:
                date = datetime(2024, month, day)
                
                for product in products:
                    # Generate realistic transaction data
                    num_transactions = np.random.poisson(10)  # Average 10 transactions per product per sample day
                    
                    for _ in range(num_transactions):
                        base_price = {
                            "Laptop Pro": 1200, "Smartphone X": 800, "Wireless Headphones": 150,
                            "Gaming Monitor": 300, "Mechanical Keyboard": 120, "Wireless Mouse": 50,
                            "Tablet Ultra": 500, "Smart Watch": 250, "Bluetooth Speaker": 80, "USB-C Hub": 30
                        }[product]
                        
                        # Add some price variation
                        price = base_price * np.random.normal(1, 0.1)
                        quantity = np.random.randint(1, 4)
                        
                        data.append({
                            "timestamp": date.isoformat(),
                            "product_name": product,
                            "sales_amount": round(price * quantity, 2),
                            "quantity": quantity,
                            "unit_price": round(price, 2),
                            "customer_id": f"cust_{np.random.randint(1000, 9999)}",
                            "transaction_id": f"txn_{len(data) + 1:06d}"
                        })
                        
            except ValueError:
                # Skip invalid dates (like Feb 30)
                continue
    
    return data

def print_integration_instructions():
    """Print instructions for integrating with real OpenSearch."""
    
    instructions = """
🔧 INTEGRATION WITH REAL OPENSEARCH:

1. Set up your OpenSearch MCP Server:
   - Use the server code from the previous artifact
   - Make sure OpenSearch is running and accessible
   - Configure proper authentication

2. Replace MockMCPClient with real MCP client:
   ```python
   from mcp.client.session import ClientSession
   from mcp.client.stdio import stdio_client
   
   class RealMCPClient:
       async def search_products(self, query, index="products"):
           async with stdio_client() as (read, write):
               async with ClientSession(read, write) as session:
                   await session.initialize()
                   result = await session.call_tool("search_documents", {
                       "index": index,
                       "query": query
                   })
                   return json.loads(result[0].text)
   ```

3. Index sample data into OpenSearch:
   ```python
   sample_data = create_sample_opensearch_index_data()
   # Use OpenSearch bulk API to index this data
   ```

4. Update the client initialization:
   ```python
   client = ProductTrendClient(use_mock=False)
   ```

5. Run your analysis:
   ```python
   df = await client.analyze_trends("trend of all products in 2024")
   ```
"""
    print(instructions)

if __name__ == "__main__":
    # Run the demo
    asyncio.run(main())
    
    # Print integration instructions
    print_integration_instructions()
    
    # Optionally generate sample data for indexing
    print("\n📝 Generating sample data for OpenSearch indexing...")
    sample_data = create_sample_opensearch_index_data()
    print(f"✅ Generated {len(sample_data)} sample records")
    print("💾 Save this data to 'sample_products.json' for indexing into OpenSearch")
    
    # Save sample data to file
    with open('sample_products.json', 'w') as f:
        for record in sample_data[:10]:  # Save first 10 records as example
            f.write(json.dumps(record) + '\n')
    
    print("📄 Sample data saved to 'sample_products.json'")
