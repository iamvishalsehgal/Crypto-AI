"""Package setup for the AI Crypto Trading Bot."""

from setuptools import find_packages, setup

setup(
    name="crypto-ai-trading-bot",
    version="0.1.0",
    description="AI-powered cryptocurrency trading bot with ensemble models",
    author="Vishal Sehgal",
    python_requires=">=3.10",
    packages=find_packages(),
    install_requires=[
        "python-dotenv>=1.0.0",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "ccxt>=4.0.0",
        "pymongo>=4.5.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "torch>=2.0.0",
        "gymnasium>=0.29.0",
        "transformers>=4.30.0",
        "aiohttp>=3.9.0",
        "beautifulsoup4>=4.12.0",
        "matplotlib>=3.7.0",
        "prometheus-client>=0.17.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "ruff>=0.1.0",
        ],
        "social": [
            "praw>=7.7.0",
            "tweepy>=4.14.0",
            "newsapi-python>=0.2.7",
        ],
        "macro": [
            "fredapi>=0.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "crypto-bot=crypto_bot.main:main",
        ],
    },
)
