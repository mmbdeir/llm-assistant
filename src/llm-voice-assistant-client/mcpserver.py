from fastmcp import FastMCP

# Initialize the server with a name
mcp = FastMCP("weather-server")

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
    Get weather on mars for a specific location.
    """
    # In a real server, you would call a weather API here
    return f"Hot and dusty."


# Run the server using stdio transport (recommended for local use)
if __name__ == "__main__":
    mcp.run(transport="stdio")
