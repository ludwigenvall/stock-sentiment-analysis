"""
FinBERT sentiment analyzer for financial text
Uses pre-trained FinBERT model from HuggingFace
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinBERTAnalyzer:
    """
    Sentiment analysis using FinBERT
    Returns: positive, negative, neutral probabilities
    """

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        """
        Args:
            model_name: HuggingFace model identifier
        """
        logger.info(f"Loading FinBERT model: {model_name}")
        logger.info(
            "This may take 1-2 minutes on first run (downloading model)...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name)
        self.model.eval()  # Set to evaluation mode

        # Use GPU if available
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        logger.info(f"✓ Model loaded on {self.device}")

    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text

        Args:
            text: Financial text (news headline, article, Reddit post)

        Returns:
            Dict with keys: positive, negative, neutral, sentiment_score
            sentiment_score: -1 (very negative) to +1 (very positive)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # FinBERT outputs: [positive, negative, neutral]
        probs = probs.cpu().numpy()[0]

        positive_prob = float(probs[0])
        negative_prob = float(probs[1])
        neutral_prob = float(probs[2])

        # Calculate overall sentiment score (-1 to +1)
        sentiment_score = positive_prob - negative_prob

        # Determine label
        max_prob_idx = probs.argmax()
        labels = ['positive', 'negative', 'neutral']
        sentiment_label = labels[max_prob_idx]

        return {
            'positive': positive_prob,
            'negative': negative_prob,
            'neutral': neutral_prob,
            'sentiment_score': sentiment_score,
            'sentiment_label': sentiment_label
        }

    def analyze_batch(
        self,
        texts: List[str],
        batch_size: int = 16
    ) -> List[Dict[str, float]]:
        """
        Analyze sentiment for multiple texts (more efficient)

        Args:
            texts: List of texts
            batch_size: Process this many texts at once

        Returns:
            List of sentiment dicts
        """
        results = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)

            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

            probs = probs.cpu().numpy()

            # Process each result in batch
            for prob_dist in probs:
                positive_prob = float(prob_dist[0])
                negative_prob = float(prob_dist[1])
                neutral_prob = float(prob_dist[2])

                sentiment_score = positive_prob - negative_prob

                max_idx = prob_dist.argmax()
                labels = ['positive', 'negative', 'neutral']
                sentiment_label = labels[max_idx]

                results.append({
                    'positive': positive_prob,
                    'negative': negative_prob,
                    'neutral': neutral_prob,
                    'sentiment_score': sentiment_score,
                    'sentiment_label': sentiment_label
                })

        logger.info(f"✓ Analyzed {len(texts)} texts")
        return results

    def analyze_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = 'text'
    ) -> pd.DataFrame:
        """
        Add sentiment analysis to a DataFrame

        Args:
            df: DataFrame with text column
            text_column: Name of column containing text

        Returns:
            DataFrame with added sentiment columns
        """
        texts = df[text_column].fillna('').tolist()

        logger.info(f"Analyzing sentiment for {len(texts)} texts...")
        sentiments = self.analyze_batch(texts)

        # Add sentiment columns
        df['sentiment_positive'] = [s['positive'] for s in sentiments]
        df['sentiment_negative'] = [s['negative'] for s in sentiments]
        df['sentiment_neutral'] = [s['neutral'] for s in sentiments]
        df['sentiment_score'] = [s['sentiment_score'] for s in sentiments]
        df['sentiment_label'] = [s['sentiment_label'] for s in sentiments]

        return df


# Example usage
if __name__ == "__main__":
    # Initialize analyzer (takes 1-2 min first time)
    analyzer = FinBERTAnalyzer()

    # Test with sample financial texts
    sample_texts = [
        "Apple stock surges to all-time high on strong earnings",
        "Tesla faces regulatory challenges, shares decline sharply",
        "Market remains flat as investors await Fed decision",
        "NVIDIA announces breakthrough AI chip, stock soars 15%",
        "Company reports disappointing quarterly results"
    ]

    print("\n" + "="*70)
    print("🤖 TESTING FINBERT SENTIMENT ANALYSIS")
    print("="*70 + "\n")

    for text in sample_texts:
        result = analyzer.analyze_sentiment(text)

        # Color code based on sentiment
        if result['sentiment_score'] > 0.3:
            emoji = "📈 🟢"
        elif result['sentiment_score'] < -0.3:
            emoji = "📉 🔴"
        else:
            emoji = "➡️  🟡"

        print(f"{emoji} Text: {text}")
        print(
            f"   Sentiment: {result['sentiment_label']} (score: {result['sentiment_score']:+.3f})")
        print(
            f"   Probabilities: Pos={result['positive']:.2f} | Neg={result['negative']:.2f} | Neu={result['neutral']:.2f}")
        print()

    # Test batch processing
    print("="*70)
    print("🚀 BATCH PROCESSING TEST")
    print("="*70 + "\n")

    df = pd.DataFrame({'text': sample_texts})
    df = analyzer.analyze_dataframe(df)

    print(df[['text', 'sentiment_label', 'sentiment_score']].to_string(index=False))

    print("\n✅ FinBERT analyzer ready to use!")
