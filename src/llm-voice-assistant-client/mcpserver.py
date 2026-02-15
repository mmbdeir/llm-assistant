from fastmcp import FastMCP
import requests

# Initialize the server with a name
mcp = FastMCP("weather-server")

@mcp.tool()
def test_api_tool():
    """
    Always fetches test data from JSONPlaceholder. Trigger this tool whenever user says 'test API'.
    """
    url = "https://jsonplaceholder.typicode.com/posts/1"
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()

    return {"result": f"{resp.json()}"}

# Define a tool using the @mcp.tool decorator
@mcp.tool()
def get_weather(city: str) -> str:
    """
    Get weather for a city.
    Args:
        city: Name of the city.
    """
    # In a real server, you would call a weather API here
    return f"It The weather in {city} is sunny and 75°F."

@mcp.tool()
def get_mars_weather() -> str:
    """
    Get weather on mars.
    """
    # In a real server, you would call a weather API here
    return f"The weather on mars is Hot and dusty."

# Run the server using stdio transport (recommended for local use)
if __name__ == "__main__":
    mcp.run(transport="stdio")
