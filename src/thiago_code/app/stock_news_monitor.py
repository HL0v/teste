import argparse
import time
import schedule
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional
from newsapi import NewsApiClient
import logging
import app.apikeys as apikeys
import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
import os
import warnings
import json
from typing import Dict, Any

# Suppress gRPC warnings
os.environ['GRPC_TRACE'] = 'none'
os.environ['GRPC_VERBOSITY'] = 'none'
warnings.filterwarnings('ignore', category=UserWarning)

#loggers
logging.basicConfig(
    filename='soy.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
soy_logger = logging.getLogger('soy')
gemini_logger = logging.getLogger('geminianalysis')
gemini_logger.setLevel(logging.INFO)
gemini_handler = logging.FileHandler('geminianalysis.log')
gemini_logger.addHandler(gemini_handler)

genai.configure(api_key=apikeys.gemini_api_key)

@dataclass
class NewsArticle:
    """Data structure for a news article."""
    title: str
    url: str
    published_at: str
    description: str
    matched_keywords: List[str] = field(default_factory=list)
    ticker: Optional[str] = None

@dataclass
class MonitorConfig:
    api_key: str = apikeys.news_api_key
    refresh_interval: int = 30
    tickers: List[str] = field(default_factory=list)
    price_moving_keywords: List[str] = field(default_factory=lambda: [

      #  """coringas"""
       'commodities', 

        'plantio', 'colheita',

     #   """fatores naturais"""
        'clima', 'chuva',
        'seca', 'pragas',
        'doenças', 'elninho',
        'geada', 'temperatura',
        'granizo', 'inundações',
     
        
       # """fatores de estrito mercado"""
        'oferta', 'demanda',

        'safra', 'exportação', 'importação',

        'especulação','tick de soja', 
        'cotação da soja','contrato futuro soja',
        

        #"""fatores geopolíticos"""
        'Guerra', 'sanções',
        'China', 'guerra comercial', 
        'política agrícola', 'subsídios', 
        'tarifas', 'acordo comercial',
        

    ])

class NewsMonitor:
    def _build_ticker_mapping(self) -> Dict[str, str]:
        """Build a mapping of tickers to company names for better search results."""
        # Basic mapping - could be extended with a comprehensive database
        return {
            
            'Brasil' : 'comercio de soja',
            'soja': 'soja',
            'B3': 'b3',
            'plantação de soja': 'plantio de soja',
            'colheita de soja': 'colheita de soja',
            'preço da soja': 'cotação da soja',
            'contrato futuro de soja': 'contrato futuro de soja',
            'exportação de soja': 'exportação de soja',
            'importação de soja': 'importação de soja',
            'mercado de soja': 'mercado de soja',
            'commodities agrícolas': 'commodities agrícolas',
            'clima para soja': 'clima para soja',
        }   
        # Return mapping for requested tickers
        
    def __init__(self, config: MonitorConfig):
        self.config = config
        self.newsapi = NewsApiClient(api_key=config.api_key)
        self.processed_articles: Set[str] = set()
        self.ticker_to_company = self._build_ticker_mapping()
        # Set tickers from the mapping
        self.config.tickers = list(self.ticker_to_company.keys())
    

    

    def _build_search_query(self, ticker: str) -> str:
        """Build search query for a specific ticker."""
        company_name = self.ticker_to_company.get(ticker, ticker)
        
        # Create query with ticker and company name
        base_query = f'"{ticker}" OR "{company_name}"'
        
        # Add keyword filters
        keyword_query = ' OR '.join([f'"{keyword}"' for keyword in self.config.price_moving_keywords])
        
        return f'({base_query}) AND ({keyword_query})'
    
    def _extract_matched_keywords(self, article_text: str) -> List[str]:
        """Extract which keywords matched in the article."""
        matched = []
        text_lower = article_text.lower()
        
        for keyword in self.config.price_moving_keywords:
            if keyword.lower() in text_lower:
                matched.append(keyword)
        
        return matched
    
    def fetch_news_for_ticker(self, ticker: str) -> List[NewsArticle]:
        """Fetch news articles for a specific ticker."""
        try:
            query = self._build_search_query(ticker)
            soy_logger.debug(f"Searching for ticker {ticker} with query: {query}")

            # Search for articles from the last 14 days
            end_date = datetime.now()
            start_date = end_date - timedelta(days=14)
            

            articles = self.newsapi.get_everything(
                q=query,
                language='pt',
                sort_by='publishedAt',
                from_param=start_date.strftime("%Y-%m-%d"),
                to=end_date.strftime('%Y-%m-%d'),
                page_size=20
            )
            
              # Add debug logging for API response
            soy_logger.info(f"Found {len(articles.get('articles', []))} total articles for {ticker}")

            news_articles = []
            
            for article in articles.get('articles', []):
                # Skip if already processed
                if article['url'] in self.processed_articles:
                    continue
                
                news_article = NewsArticle(
                    title=article.get('title', 'No title'),
                    url=article['url'],
                    published_at=article.get('publishedAt', ''),
                    description=article.get('description', ''),
                    matched_keywords=[],  # We'll fill this later
                    ticker=ticker
                )

                # Extract matched keywords from title and description
                article_text = f"{article.get('title', '')} {article.get('description', '')}"
                matched_keywords = self._extract_matched_keywords(article_text)

                # Debug log for keyword matching
                soy_logger.info(f"Article: {news_article.title}")
                soy_logger.info(f"Matched keywords: {matched_keywords}")
                
                # Only include articles with matched keywords
                news_article.matched_keywords = matched_keywords
                if matched_keywords:
                    news_articles.append(news_article)
                    self.processed_articles.add(article['url'])
            
            return news_articles
            
        except Exception as e:
            soy_logger.error(f"Error fetching news for {ticker}: {e}")
            return []
    
    def analyze_with_ai(self, url: str) -> Dict[str, Any]:
        gemini_logger.info(f"AI analysis requested for: {url}")
        ai_response = None
        response_text = None

        try:
            #scraping article text
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            article_text = "".join(soup.get_text().split())[:10000]
            #model selection and prompt
            model = genai.GenerativeModel('gemini-2.5-flash')
            prompt = f"""
                Act as an expert commodity market analyst specializing in agricultural futures, particularly soybeans. Your analysis will be used as input for a probabilistic model that predicts soy futures prices.

                    Analyze the following news article text. Your primary goal is to extract structured, quantifiable data.

                    **Article:**
                    {article_text}

                    **Task:**
                    Provide your analysis in a single JSON object. Do not include any explanatory text before or after the JSON. The JSON object must contain the following keys:

                    1.  `"summary"`: A concise, neutral summary of the article's main points in no more than three sentences.
                    2.  `"overall_sentiment_score"`: A single floating-point number from -1.0 (extremely negative for the soy market) to 1.0 (extremely positive for the soy market). 0.0 represents a neutral sentiment.
                    3.  `"key_factors"`: A JSON array of strings. List the primary economic, political, or environmental factors mentioned that could influence soy prices (e.g., "weather conditions in Brazil," "US trade policy," "demand from China").
                    4.  `"market_impact_score"`: An integer from 1 (minimal impact) to 10 (market-moving event) representing the potential significance of the news.
                    5.  `"regional_impact"`: A JSON object with keys for "brazil", "usa", and "argentina". For each country, provide an object with two keys:
                        * `"sentiment"`: A float from -1.0 to 1.0 for that specific region.
                        * `"reasoning"`: A brief string explaining the sentiment for that region based on the article. If the article does not mention a region, the sentiment should be 0.0 and the reasoning "Not mentioned in the article."
                    6.  `"confidence_score"`: A float from 0.0 to 1.0 indicating your confidence in the analysis based on the clarity and specificity of the provided article.
                    7. `"trading_recommendation"`: A string with one of the following values: "buy", "sell", "hold", or "monitor". This recommendation should be based solely on the content of the article and its potential impact on soy futures prices.
            """
            ai_response = model.generate_content(prompt)
            response_text = ai_response.text
            if response_text.startswith("```json"):
                response_text = response_text.strip("```json\n").strip("`")
            analysis_data = json.loads(response_text)
            gemini_logger.info(f"AI analysis for: {url}")

            filename = "responses.json"
            with open(filename, "a") as f:
                json.dump(analysis_data, f, indent=4)
            print(f"✅ Successfully saved analysis to {filename}")

            gemini_handler.setLevel(logging.INFO)

            return analysis_data
        except requests.exceptions.RequestException as e:
            gemini_logger.error(f"failed to fetch : {url}: {e}")
            return {"error": f"Failed to fetch URL: {e}"}
        except json.JSONDecodeError as e:
            gemini_logger.error(f"Failed to parse JSON from AI response for {url}: {e}")
            gemini_logger.error(f"Raw response was: {response_text}")
            return {"error": f"AI returned invalid JSON: {e}"}
        except Exception as e:
            gemini_logger.error(f"An unexpected error occurred during AI analysis for {url}: {e}")
            return {"error": f"An unexpected error ocurred: {e}"}
        
    def process_article(self, article: NewsArticle) -> None:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        soy_logger.info(f"Article found - {timestamp}")
        soy_logger.info(f"\n{'='*80}")
        soy_logger.info(f"{'='*80}")
        soy_logger.info(f"Trigger Word: {article.ticker}")
        soy_logger.info(f"Title: {article.title}")
        soy_logger.info(f"Published: {article.published_at}")
        soy_logger.info(f"URL: {article.url}")
        soy_logger.info(f"Tokens: {', '.join(article.matched_keywords)}")

        if article.description:
            desc = str(article.description)
            truncated_desc = desc[:200] + "..." if len(desc) > 200 else desc
            soy_logger.info(f"Description: {truncated_desc}\n")
        
        # Trigger AI analysis
        try:
            ai_result = self.analyze_with_ai(article.url)
            gemini_logger.info(f"AI Analysis: {ai_result.get('summary', 'No summary available')}\n")

        except Exception as e:
            gemini_logger.error(f"Error in AI analysis: {e}")
            gemini_logger.info(f"{'='*80}\n")


    def check_news(self) -> None:
        """Check news for all configured tickers."""
        soy_logger.info(f"Searching triggers: {', '.join(self.config.tickers)}")
        
        total_articles = 0
        
        for ticker in self.config.tickers:
            articles = self.fetch_news_for_ticker(ticker)
            
            for article in articles:
                self.process_article(article)
                total_articles += 1
        
        if total_articles == 0:
            soy_logger.info("No new relevant articles found")
        else:
            soy_logger.info(f"Found {total_articles} new relevant articles")
    
    def start_monitoring(self) -> None:
        """Start the continuous monitoring process."""
        soy_logger.info("\n\nEyes opened\n")
        soy_logger.info(f"Looking for triggers: {', '.join(self.config.tickers)}\n")
        soy_logger.info(f"Refresh interval: {self.config.refresh_interval} seconds\n")
        soy_logger.info(f"Keywords: {', '.join(self.config.price_moving_keywords[:5])}... (+{len(self.config.price_moving_keywords)-5} more)\n")
        soy_logger.info("Press Ctrl+C to stop\n")
        soy_logger.info(f"{'='*80}\n")
        
        
        # Schedule the job
        schedule.every(self.config.refresh_interval).seconds.do(self.check_news)
        
        # Run initial check
        self.check_news()
        
        # Keep running
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            soy_logger.info("\nClosing eyes\n")
            soy_logger.info("Eyes closed\n")

def main():
    """Main entry point."""
    reqs = argparse.ArgumentParser(description='Searching for price-moving events')
    reqs.add_argument(
        '--interval',
        type=int,
        default=300,
        help='Refresh interval in seconds (default: 300)'
    )

    init_reqs = reqs.parse_args()
    
    # Create configuration
    config =  MonitorConfig(
        api_key= apikeys.news_api_key,
        refresh_interval=init_reqs.interval
    )
    # Create and start monitor
    monitor = NewsMonitor(config)
    monitor.start_monitoring()

if __name__ == "__main__":
    main()