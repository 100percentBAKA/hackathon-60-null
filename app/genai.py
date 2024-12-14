from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

def create_prompt(news_content, data_summary, trend_analysis):
    """
    Generate the prompt template to classify stock trend.

    Args:
    - news_content (str): News about the stock.
    - data_summary (str): Stock data summary from the dataframe.
    - trend_analysis (str): Model's trend prediction.

    Returns:
    - str: Formatted prompt string.
    """
    prompt_template = PromptTemplate.from_template(
        "Classify the stock trend based on the given news, stock data summary, and the provided trend analysis into one of the following categories: "
        "'POSITIVE' (Buy), 'NEGATIVE' (Sell), or 'NEUTRAL' (Hold)."
        "\nProvide the classification with just one word ('POSITIVE', 'NEGATIVE', or 'NEUTRAL')."
        "\nAlso explain the reasoning behind the classification in 2-3 sentences."
        "\nHere is the news about the stock: {news}"
        "\nHere is the stock's data summary: {data_summary}"
        # "\nHere is the model's trend prediction: {trend_analysis}"
    )
    return prompt_template.format(news=news_content, data_summary=data_summary, trend_analysis=trend_analysis)

def get_prediction_from_model(news_content, data_summary, trend_analysis):
    """
    Interact with the language model to classify the stock trend.

    Args:
    - news_content (str): Scraped news about the stock.
    - data_summary (str): Stock data summary.
    - trend_analysis (str): Model's trend prediction.

    Returns:
    - tuple: (trend, reasoning)
    """
    prompt = create_prompt(news_content, data_summary, trend_analysis)

    # Initialize the language model
    model = ChatGroq(
        temperature=0,
        model_name="llama-3.1-8b-instant",
        api_key="use-your-own-api-key"
    )

    # Get the output from the model
    output = model.invoke(prompt)

    # Extract the prediction and reasoning
    prediction = output.content.split('\n')[0].strip()  # First line is the trend (POSITIVE/NEGATIVE/NEUTRAL)
    reasoning = '\n'.join(output.content.split('\n')[1:]).strip()  

    print('prediction')
    print(prediction)

    print('reasoning')
    print(reasoning)

    return prediction, reasoning
