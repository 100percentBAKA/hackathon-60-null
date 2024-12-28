
# Stock Price Prediction and Future Trend Analysis

This project provides an interactive platform for analyzing stock prices, predicting future trends, and categorizing trends based on news sentiment and historical stock data. It leverages machine learning models, web scraping, and generative AI to offer comprehensive insights into stock market trends.

## Features

- **Historical Stock Data Visualization:**
  - Visualize the closing price trends of a given stock over time.

- **Stock Price Prediction:**
  - Predict future stock prices using a trained LSTM model.

- **News Sentiment Analysis:**
  - Scrape news articles from MoneyControl and Zerodha Pulse to analyze the sentiment around a stock.

- **Trend Classification:**
  - Use AI to classify the stock trend into one of three categories: `POSITIVE` (Buy), `NEGATIVE` (Sell), or `NEUTRAL` (Hold).

- **Interactive Gradio Interface:**
  - View visualizations, data summaries, predictions, and actionable insights in an easy-to-use web interface.

## Directory Structure

```
./
└── app
    ├── automation.py  # Web scraping utilities
    ├── genai.py       # Generative AI utilities
    └── prediction.py  # Prediction and analysis logic
```

## File Descriptions

### `/app/automation.py`

Handles web scraping tasks using Selenium:
- Scrapes news articles from MoneyControl and Zerodha Pulse.
- Filters and fetches relevant content based on stock tickers.

### `/app/genai.py`

Contains utilities to interact with generative AI models:
- Generates prompts to classify stock trends based on news sentiment and historical data.
- Invokes the AI model to predict trends and provide reasoning.

### `/app/prediction.py`

Combines historical data analysis, prediction, and visualization:
- Fetches historical stock data using Yahoo Finance.
- Predicts future stock prices using an LSTM model.
- Visualizes trends and compares predictions against historical data.
- Integrates scraping and AI-generated insights to provide actionable recommendations.

## Requirements

### Python Packages
Install the required Python packages using the following command:
```bash
pip install -r requirements.txt
```

### Additional Requirements
- **Selenium WebDriver**: Download and place the Chrome WebDriver in an accessible path (e.g., `C:\chromedriver.exe`).
- **Keras Model**: Include the pre-trained LSTM model file as `keras_model.h5` in the project root.
- **Gradio**: For interactive UI.

## Usage

### Running the Application
1. Clone the repository and navigate to the project directory.
2. Launch the application:
   ```bash
   python /app/prediction.py
   ```
3. Open the URL provided by Gradio to access the interactive interface.

### Inputs
- **Stock Ticker**: Enter the stock symbol (e.g., `INFY` for Infosys).
- **Company Name**: Enter the name of the company (e.g., `Infosys`).

### Outputs
- **Closing Price vs Time Chart**: Visualizes historical closing prices.
- **Prediction vs Original Chart**: Compares predicted and actual prices.
- **Data Summary**: Displays statistical summary of stock data.
- **Trend Output**: Classifies trend as `POSITIVE`, `NEGATIVE`, or `NEUTRAL`.
- **Recommended Action**: Displays actionable advice (`BUY`, `SELL`, or `HOLD`).

## Key Components

### Web Scraping
- Scrapes and parses news content using Selenium to gather stock-specific insights.

### Prediction Model
- Employs an LSTM model to forecast future stock prices based on historical data.

### Generative AI
- Leverages a GPT-based language model to classify trends and provide reasoning.

### Gradio Interface
- Simplifies interaction with visual plots, dataframes, and actionable insights.

## Example Output

### Input
- **Stock Ticker:** `TCS`
- **Company Name:** `Tata Consultancy Services`

### Output
1. Historical Price Chart
2. Predicted Price vs Actual Price Chart
3. Statistical Summary Table
4. Trend Output: `POSITIVE`
5. Recommended Action: `BUY`

## Screenshots

_Include screenshots of the Gradio interface, charts, and outputs._

## Dependencies

- Python 3.8+
- TensorFlow/Keras
- Selenium
- Gradio
- yFinance
- NumPy
- Pandas
- Matplotlib
- MinMaxScaler

## License
This project is licensed under the MIT License.

