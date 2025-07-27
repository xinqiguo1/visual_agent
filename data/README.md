# Demo Data Documentation

This directory contains various demo datasets for testing the Visual Analytics Agent. Each dataset is designed to showcase different types of data analysis and visualization capabilities.

## Available Datasets

### 1. Sales Data (`sales_data.csv`)
- **Description**: Business sales data with multiple dimensions
- **Records**: 42 entries
- **Fields**: Date, Product, Category, Sales, Units, Region, Sales_Rep, Customer_Segment
- **Use Cases**: 
  - Sales performance analysis
  - Regional comparison
  - Product category analysis
  - Time series sales trends
  - Customer segmentation

### 2. Stock Data (`stock_data.csv`)
- **Description**: Financial market data for major tech stocks
- **Records**: 50 entries
- **Fields**: Date, Symbol, Open, High, Low, Close, Volume, Company
- **Companies**: AAPL, GOOGL, MSFT, TSLA, AMZN
- **Use Cases**:
  - Stock price analysis
  - Trading volume trends
  - Comparative stock performance
  - Financial candlestick charts
  - Market volatility analysis

### 3. User Activity Data (`user_activity.json`)
- **Description**: Web analytics and user behavior data
- **Records**: 5 users with session data
- **Fields**: user_id, demographics, sessions, actions, timestamps
- **Use Cases**:
  - User engagement analysis
  - Session duration tracking
  - Geographic user distribution
  - Subscription tier analysis
  - User journey mapping

### 4. Demographic Data (`demographic_data.csv`)
- **Description**: Survey data with demographic and satisfaction metrics
- **Records**: 50 respondents
- **Fields**: ID, Age, Gender, Education, Income, Employment_Status, Marital_Status, Children, Location, Satisfaction_Score, Product_Usage
- **Use Cases**:
  - Demographic analysis
  - Customer satisfaction correlation
  - Income distribution analysis
  - Geographic satisfaction mapping
  - Product usage patterns

### 5. Weather Data (`weather_data.csv`)
- **Description**: Environmental data for multiple cities
- **Records**: 50 entries across 5 cities
- **Fields**: Date, City, Temperature_C, Humidity_%, Pressure_hPa, Wind_Speed_kmh, Precipitation_mm, Weather_Condition
- **Cities**: New York, Los Angeles, Chicago, Miami, Seattle
- **Use Cases**:
  - Weather pattern analysis
  - Temperature trend comparison
  - Precipitation analysis
  - Multi-city climate comparison
  - Weather condition distribution

## Usage Examples

Each dataset can be used with the Visual Analytics Agent to:

1. **Upload Data**: Use the API endpoints to upload and process the data
2. **Generate Insights**: Automatically analyze patterns and trends
3. **Create Visualizations**: Generate charts, graphs, and dashboards
4. **Export Results**: Save analysis results and visualizations

## File Formats

- **CSV Files**: Standard comma-separated values for tabular data
- **JSON Files**: Structured data for complex nested information
- **All files**: UTF-8 encoded for universal compatibility

## Data Quality

All datasets are:
- ✅ Clean and formatted
- ✅ Realistic sample data
- ✅ Ready for immediate analysis
- ✅ Suitable for demonstration purposes

## Getting Started

1. Start the Visual Analytics Agent API
2. Upload any of these datasets via the web interface or API
3. Run analysis and generate visualizations
4. Explore the interactive dashboards created

For more information, see the main project README and API documentation. 