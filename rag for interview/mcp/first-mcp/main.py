from fastmcp import FastMCP
import uuid

# Initialize the FastMCP server with a name
mcp = FastMCP("Store Database")

# Simulate an in-memory database with a dictionary
# Structure: {id: {field1: value1, field2: value2, ...}}
store_db = {}

@mcp.tool()
def add_item(name: str, price: float, category: str, stock: int = 0) -> dict:
    """Add a new item to the store
    
    Args:
        name: Item name
        price: Item price
        category: Item category
        stock: Available quantity (default: 0)
        
    Returns:
        Dictionary containing the added item with its generated ID
    """
    # Generate a unique ID
    item_id = str(uuid.uuid4())
    
    # Create the item data
    item_data = {
        "name": name,
        "price": price,
        "category": category,
        "stock": stock
    }
    
    # Store the data with the generated ID
    store_db[item_id] = item_data
    
    # Return the data with its ID
    result = item_data.copy()
    result["id"] = item_id
    return result

@mcp.tool()
def get_item(item_id: str) -> dict:
    """Get a specific item by its ID
    
    Args:
        item_id: The ID of the item to retrieve
        
    Returns:
        Dictionary containing the item data or error message
    """
    if item_id in store_db:
        result = store_db[item_id].copy()
        result["id"] = item_id
        return result
    return {"error": "Item not found"}

@mcp.tool()
def search_items(query: str, field: str = None) -> list:
    """Search for items in the store
    
    Args:
        query: The search query string
        field: Optional field name to limit the search to (name, category, etc.)
        
    Returns:
        List of matching items with their IDs
    """
    results = []
    
    for item_id, item_data in store_db.items():
        # If a field is specified, search only in that field
        if field is not None:
            if field in item_data and query.lower() in str(item_data[field]).lower():
                result = item_data.copy()
                result["id"] = item_id
                results.append(result)
        # Otherwise search in all fields
        else:
            for value in item_data.values():
                if query.lower() in str(value).lower():
                    result = item_data.copy()
                    result["id"] = item_id
                    results.append(result)
                    break
    
    return results

# Add some initial data to the database for testing
def populate_sample_data():
    """Add sample data to the database for testing purposes"""
    sample_items = [
        {"name": "Laptop", "price": 999.99, "category": "Electronics", "stock": 10},
        {"name": "Desk Chair", "price": 199.50, "category": "Furniture", "stock": 15},
        {"name": "Coffee Mug", "price": 12.99, "category": "Kitchen", "stock": 30},
        {"name": "Headphones", "price": 149.99, "category": "Electronics", "stock": 20},
        {"name": "Notebook", "price": 9.99, "category": "Office Supplies", "stock": 50}
    ]
    
    for item in sample_items:
        add_item(**item)
    
    print(f"Store database populated with {len(sample_items)} sample items")

if __name__ == "__main__":
    # Populate the database with sample data
    populate_sample_data()
    
    # Run the MCP server
    print("Starting Store Database MCP server...")
    mcp.run()