# Soy Futures News Monitor & AI Analyzer

A Python-based system designed to monitor real-time news about the soy market, analyze articles using the Gemini AI, and generate structured data for probabilistic modeling of soy futures prices.

## 📋 Description

This application actively monitors news articles related to soy futures in key markets (Brazil, USA, Argentina) using the NewsAPI service. When a relevant article is found, it is scraped and sent to Google's Gemini AI for in-depth analysis. The AI's structured JSON response, which includes sentiment scores, market impact, and regional analysis, is then saved for use in quantitative models.

## 🚀 Features

  * **Real-time News Monitoring**: Tracks news for multiple soy-related keywords and tickers.
  * **AI-Powered Analysis**: Leverages Google's Gemini AI to perform expert-level analysis on news articles.
  * **Structured Data Output**: Generates clean, machine-readable JSON output for each analysis, perfect for data ingestion.
  * **Web Scraping**: Extracts full article text from URLs for comprehensive analysis.
  * **Comprehensive Logging**: Maintains detailed logs for both system events (`soy.log`) and AI analysis interactions (`geminianalysis.log`).
  * **Flexible Configuration**: Easily customize tickers, keywords, and the news refresh interval.

## ⚙️ Tech Stack

  * **Python 3.x**
  * **APIs & Libraries**:
      * `newsapi-python`: To fetch news articles.
      * `google-generativeai`: For AI-based text analysis with Gemini.
      * `requests` & `BeautifulSoup`: For web scraping article content.
      * `schedule`: To run the monitoring job at regular intervals.
      * `pandas`: For data handling (if extending the project).

## 💻 Installation & Setup

1.  **Clone the Repository**

    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```

2.  **Set Up a Virtual Environment**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    Create a `requirements.txt` file with the following content:

    ```txt
    newsapi-python
    google-generativeai
    requests
    beautifulsoup4
    schedule
    pandas
    ```

    Then, install the packages:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys**
    Create a file named `apikeys.py` in the root directory and add your API keys:

    ```python
    # apikeys.py
    news_api_key = "YOUR_NEWSAPI_KEY"
    gemini_api_key = "YOUR_GEMINI_API_KEY"
    ```

    ***Note***: Ensure `apikeys.py` is listed in your `.gitignore` file to keep your keys private.

## ⚡️ Usage

To start the monitor, run the `stock_news_monitor.py` script from your terminal. You can customize the refresh interval using the `--interval` flag.

```bash
# Run with a 5-minute (300 seconds) refresh interval
python stock_news_monitor.py --interval 300
```

The script will start searching for news based on the tickers and keywords defined in the `MonitorConfig` class.

## 📁 Project Output

The application generates three main output files:

  * **`responses.json`**: Appends the structured JSON output from every successful Gemini AI analysis. This file serves as your primary dataset.
  * **`soy.log`**: Records the main operational events of the script, such as when it starts, what articles it finds, and any errors encountered during news fetching.
  * **`geminianalysis.log`**: Contains logs specifically related to the AI analysis, including the request prompts and any errors from the Gemini API.

### Example JSON Output

Each analysis saved to `responses.json` follows this structure:

```json
{
    "summary": "China is delaying soybean purchases due to high Brazilian premiums and US-China trade tensions...",
    "overall_sentiment_score": -0.6,
    "key_factors": [
        "High Brazilian soy premiums",
        "US-China trade tensions",
        "Negative soy crushing margins in China"
    ],
    "market_impact_score": 9,
    "regional_impact": {
        "brazil": {
            "sentiment": -0.2,
            "reasoning": "Current high premiums are deterring Chinese buyers..."
        },
        "usa": {
            "sentiment": -0.8,
            "reasoning": "China is avoiding US soy purchases due to trade tensions..."
        },
        "argentina": {
            "sentiment": 0.6,
            "reasoning": "China made heavy purchases of Argentine grains..."
        }
    },
    "confidence_score": 0.9,
    "trading_recommendation": "sell"
}
```

## ⚠️ Error Handling

The system is designed to be resilient and logs errors gracefully:

  * **NewsAPI Errors**: If the query is too long or another API error occurs, it is logged in `soy.log`, and the script continues.
  * **Web Scraping Failures**: If an article URL is inaccessible (e.g., 403 Forbidden), the error is logged, and the analysis for that article is skipped.
  * **Gemini AI Errors**: If the AI model name is incorrect or the API call fails, the error is logged in both `soy.log` and `geminianalysis.log`.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.