import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from keras.models import load_model
import gradio as gr
from sklearn.preprocessing import MinMaxScaler
from genai import get_prediction_from_model
from automation import scrape_moneycontrol_news, scrape_zerodha_pulse_news, setup_driver

# Constants
start = '2010-01-01'
end = '2024-12-31'  # Extended end date
# start = '2024-01-01'
# end = '2024-10-31'  # Extended end date
model = load_model('keras_model.h5')

def predict_future_trend(ticker, days_to_predict=30):
    """
    Predict future stock price trend for the next specified number of days
    
    Args:
    ticker (str): Stock ticker symbol
    days_to_predict (int): Number of future days to predict
    
    Returns:
    tuple: Future price predictions, trend description
    """
    try:
        # Fetch historical stock data
        df = yf.download(ticker, start=start, end=end)
        
        if df.empty or 'Close' not in df:
            return None, "No data available"
        
        # Ensure data is numeric and handle any potential non-numeric values
        data = df['Close'].dropna()
        
        if len(data) < 100:
            return None, "Insufficient historical data"
        
        # Prepare data for prediction
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data.values.reshape(-1, 1))
        
        # Use last 100 days as input for prediction
        input_sequence = scaled_data[-100:]
        
        # Predict future prices
        future_predictions = []
        current_sequence = input_sequence.copy()
        
        for _ in range(days_to_predict):
            # Reshape the current sequence for prediction
            pred_sequence = current_sequence.reshape((1, 100, 1))
            
            # Predict next day's price
            next_pred = model.predict(pred_sequence)
            
            # Add prediction to future predictions
            future_predictions.append(next_pred[0, 0])
            
            # Update current sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = next_pred[0, 0]
        
        # Inverse transform predictions to original scale
        future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
        
        # Determine overall trend
        last_historical_price = float(data.iloc[-1])
        last_predicted_price = float(future_prices[-1])
        
        # Calculate the trend
        if last_predicted_price > last_historical_price:
            trend = "Upward 📈"
        else:
            trend = "Downward 📉"
        
        return future_prices, trend
    
    except Exception as e:
        return None, f"Error: {str(e)}"

def stock_prediction(ticker, company_name):
    # Set up Selenium WebDriver
    driver = setup_driver(chrome_driver_path="C:\\chromedriver.exe")

    try:
        # Scrape news from MoneyControl and Zerodha Pulse
        moneycontrol_news = scrape_moneycontrol_news(driver, company_name)
        zerodha_news = scrape_zerodha_pulse_news(driver, company_name)
        news_content = f"MoneyControl News:\n{moneycontrol_news}\n\nZerodha News:\n{zerodha_news}"
    finally:
        driver.quit()

    # Fetch stock data
    df = yf.download(ticker, start=start, end=end)

    if df.empty or 'Close' not in df:
        return None, None, None, f"No data available for the provided ticker: {company_name}"

    # Data summary
    data_summary = df.describe()
    
    # Flatten multi-level column headers if they exist
    if isinstance(data_summary.columns, pd.MultiIndex):
        data_summary.columns = [' '.join(col).strip() for col in data_summary.columns.values]
    
    # Reset index for Gradio compatibility
    data_summary = data_summary.reset_index()

    # Closing Price vs Time chart
    fig1, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(df['Close'], label=f'{company_name} Closing Price')
    ax1.set_title('Closing Price vs Time')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.legend()

    # Prepare training and testing data
    data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

    if data_training.empty:
        return None, None, None, "Insufficient data for training."

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_training_array = scaler.fit_transform(data_training)

    x_train, y_train = [], []
    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i - 100:i])
        y_train.append(data_training_array[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Prepare testing data
    past_100_days = data_training.tail(100)
    final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
    input_data = scaler.fit_transform(final_df)

    x_test, y_test = [], []
    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i - 100:i])
        y_test.append(input_data[i, 0])

    if not x_test or not y_test:
        return None, None, None, "Insufficient data for testing."

    x_test, y_test = np.array(x_test), np.array(y_test)

    # Prediction
    y_predicted = model.predict(x_test)
    scaler_scale = scaler.scale_

    scale_factor = 1 / scaler_scale[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor

    # Prediction vs Original chart
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    ax2.plot(y_test, 'b', label='Original Price')
    ax2.plot(y_predicted, 'r', label='Predicted Price')
    ax2.set_title('Prediction vs Original')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Price')
    ax2.legend()

    # Future trend prediction
    future_prices, trend = predict_future_trend(ticker)

    prediction, reasoning = get_prediction_from_model(news_content, data_summary, trend)

    return fig1, fig2, data_summary, prediction, reasoning, trend

# Update Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# Stock Price Prediction and Future Trend Analysis")
    gr.Markdown(
        "Enter a stock ticker symbol and company name to view its historical data, compare predictions, and see future trend analysis."
    )

    with gr.Row():
        ticker_input = gr.Textbox(
            label="Stock Ticker", placeholder="Enter ticker symbol (e.g., INFY)", lines=1
        )
        company_name_input = gr.Textbox(
            label="Company Name", placeholder="Enter company name (e.g., Infosys)", lines=1
        )

    with gr.Row():
        submit_button = gr.Button("Predict")

    with gr.Row():
        with gr.Column():
            plot1 = gr.Plot(label="Closing Price vs Time Chart")
            plot2 = gr.Plot(label="Prediction vs Original Chart")
        with gr.Column():
            data_summary = gr.DataFrame(label="Data Summary")
            trend_output = gr.Textbox(label="Trend Output")
            action_display = gr.HTML(label="Recommended Action")

    def display_action_button(prediction):
        prediction = str(prediction).upper()
        if "NEGATIVE" in prediction:
            return '<div style="background-color:red; color:white; padding:10px; text-align:center;">SELL</div>'
        elif "POSITIVE" in prediction:
            return '<div style="background-color:green; color:white; padding:10px; text-align:center;">BUY</div>'
        else:
            return '<div style="background-color:gray; color:white; padding:10px; text-align:center;">HOLD</div>'

    def update_ui(ticker, company_name):
        try:
            # Unpack the additional trend parameter
            fig1, fig2, data_summary, prediction, reasoning, trend = stock_prediction(ticker, company_name)
            
            # Combine prediction with reasoning and trend
            full_output = f"{prediction}: {reasoning} (Trend: {trend})"
            
            # Generate action HTML
            action_html = display_action_button(prediction)
            
            return fig1, fig2, data_summary, full_output, action_html
        except Exception as e:
            # Error handling
            error_html = f'<div style="background-color:red; color:white; padding:10px; text-align:center;">Error: {str(e)}</div>'
            return None, None, None, str(e), error_html

    submit_button.click(
        fn=update_ui,
        inputs=[ticker_input, company_name_input],
        outputs=[plot1, plot2, data_summary, trend_output, action_display]
    )

app.launch()