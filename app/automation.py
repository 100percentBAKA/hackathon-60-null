import concurrent.futures
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

def setup_driver(chrome_driver_path):
    """Set up the Selenium WebDriver."""
    service = Service(chrome_driver_path)
    driver = webdriver.Chrome(service=service)
    return driver

def scrape_moneycontrol_news(driver, ticker):
    """Scrape news articles from MoneyControl for a specific ticker."""
    url = f"https://www.moneycontrol.com/news/tags/{ticker}.html"
    driver.get(url)
    driver.implicitly_wait(3)

    news_links = driver.find_elements(By.CSS_SELECTOR, "li.clearfix a")
    filtered_links = [link.get_attribute("href") for link in news_links if ticker.lower() in link.get_attribute("href").lower()]

    result = []
    for idx, link in enumerate(filtered_links[:2], 1):
        driver.get(link)
        driver.implicitly_wait(3)

        try:
            content_elements = driver.find_elements(By.CSS_SELECTOR, "div.content_wrapper.arti-flow p")
            content = "\n".join([element.text for element in content_elements if element.text.strip()])
            result.append(f"Content of News Article {idx} ///////\n{content}\n")
            
            print(result)
        except Exception as e:
            result.append(f"Could not extract content for article {idx}: {e}\n")

    return "\n".join(result)

def scrape_zerodha_pulse_news(driver, ticker):
    """Scrape news articles from Zerodha Pulse for a specific ticker."""
    url = "https://pulse.zerodha.com/"
    driver.get(url)
    driver.implicitly_wait(3)

    search_input = driver.find_element(By.ID, "q")
    search_input.send_keys(ticker)
    search_input.send_keys(Keys.RETURN)
    time.sleep(3)

    result = []
    try:
        links = WebDriverWait(driver, 8).until(
            EC.presence_of_all_elements_located((By.CSS_SELECTOR, "li.box.item h2.title a[rel='nofollow']"))
        )
        result.append(f"News links for '{ticker}':")
        for idx, link in enumerate(links, 1):
            href = link.get_attribute("href")
            result.append(f"{idx}. {href}")

            if "ndtvprofit" in href:
                driver.get(href)
                try:
                    content_elements = WebDriverWait(driver, 8).until(
                        EC.presence_of_all_elements_located((By.CSS_SELECTOR, "div.story-elements-m__story-element-ctx__tV2U1 p"))
                    )
                    content = "\n".join([element.text for element in content_elements if element.text.strip()])
                    result.append(f"Content from '{href}':\n{content}\n")

                    print(result)
                except Exception as e:
                    result.append(f"Could not extract content from '{href}': {e}\n")
    except Exception as e:
        result.append(f"Error while processing links: {e}\n")

    return "\n".join(result)

def run_with_timeout(func, *args, timeout=30):
    """Run a function with a timeout."""
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(func, *args)
        try:
            return future.result(timeout=timeout)  # Wait for the function to complete with a timeout
        except concurrent.futures.TimeoutError:
            return "Timeout reached: Partial results may have been fetched."

# Example usage
# if __name__ == "__main__":
#     chrome_driver_path = r"C:\\chromedriver.exe"
#     driver = setup_driver(chrome_driver_path)
#     ticker = "tcs"

#     try:
#         print("\nScraping MoneyControl (max 30 seconds)...")
#         moneycontrol_result = run_with_timeout(scrape_moneycontrol_news, driver, ticker, timeout=30)
#         print(moneycontrol_result)

#         print("\nScraping Zerodha Pulse (max 30 seconds)...")
#         zerodha_result = run_with_timeout(scrape_zerodha_pulse_news, driver, ticker, timeout=30)
#         print(zerodha_result)

#     finally:
#         driver.quit()
