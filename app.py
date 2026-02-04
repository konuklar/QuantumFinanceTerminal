# =============================================================
# 🏛️ Institutional Apollo / ENIGMA – Quant Terminal v5.0
# Professional Portfolio Optimization & Global Multi-Asset Edition
# Enhanced Institutional Features & Comprehensive Error Handling
# =============================================================
# Add this at the top of your app.py
import sys
import subprocess
import pkg_resources

def install_package(package):
    """Install package if not available"""
    try:
        pkg_resources.get_distribution(package.split('==')[0].split('>')[0].split('<')[0])
        print(f"{package} already installed")
    except pkg_resources.DistributionNotFound:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Try to install pypfopt if not available
try:
    import pypfopt
    PYPFOPT_AVAILABLE = True
    PYPFOPT_VERSION = pypfopt.__version__
except ImportError:
    try:
        install_package("pypfopt>=1.5.0")
        import pypfopt
        PYPFOPT_AVAILABLE = True
        PYPFOPT_VERSION = pypfopt.__version__
    except:
        PYPFOPT_AVAILABLE = False
        PYPFOPT_VERSION = "Not Available"
        print("Warning: PyPortfolioOpt not available. Using fallback optimization methods.")
import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import stats, optimize, linalg
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any
import json
import concurrent.futures
from functools import lru_cache
import traceback
import time
import hashlib
import pickle
import base64
import io
from dataclasses import dataclass, field
import logging
import sys
import inspect
import requests
from decimal import Decimal, ROUND_HALF_UP
import csv

# =============================================================
# ENHANCED PYPFOPTOPT IMPORT WITH COMPREHENSIVE ERROR HANDLING
# =============================================================
try:
    # Import all PyPortfolioOpt components with enhanced error handling
    import pypfopt
    from pypfopt import expected_returns, risk_models
    from pypfopt.efficient_frontier import EfficientFrontier, EfficientSemivariance
    from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
    from pypfopt.objective_functions import L2_reg, L1_reg, negative_sharpe, negative_mean_variance
    from pypfopt.hierarchical_portfolio import HRPOpt
    from pypfopt.black_litterman import BlackLittermanModel
    from pypfopt.cla import CLA
    from pypfopt.value_at_risk import CVAROpt
    PYPFOPT_AVAILABLE = True
    PYPFOPT_VERSION = pypfopt.__version__
except ImportError as e:
    PYPFOPT_AVAILABLE = False
    PYPFOPT_VERSION = "Not Available"
    st.warning(f"⚠️ PyPortfolioOpt not available: {str(e)}. Some optimization features will be limited.")
    
    # Create mock classes for fallback
    class MockEfficientFrontier:
        def __init__(self, *args, **kwargs):
            pass
        def min_volatility(self):
            return {}
        def max_sharpe(self, *args, **kwargs):
            return {}
        def portfolio_performance(self, *args, **kwargs):
            return 0, 0, 0
    
    class EfficientFrontier:
        pass
    
    EfficientFrontier = MockEfficientFrontier

# Import optional ML packages with enhanced error handling
try:
    from sklearn.covariance import LedoitWolf, GraphicalLassoCV, OAS
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("⚠️ scikit-learn not available. Some advanced features will be limited.")

# Import optional statsmodels for factor analysis
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.warning("⚠️ statsmodels not available. Factor analysis will be limited.")

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('institutional_portfolio.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings but log them
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)
np.seterr(all='ignore')

# =============================================================
# ENHANCED INSTITUTIONAL CONFIGURATION WITH VALIDATION
# =============================================================
class InstitutionalConfig:
    """Enhanced institutional configuration parameters with validation"""
    
    # Portfolio constraints
    MAX_ASSETS = 150
    MIN_ASSETS = 2
    MIN_DATA_POINTS = 100
    TRADING_DAYS = 252
    DEFAULT_RF_RATE = 0.03
    
    # Optimization parameters
    MAX_ITERATIONS = 10000
    OPTIMIZATION_TOLERANCE = 1e-8
    REGULARIZATION_GAMMA = 0.1
    
    # Risk management
    MAX_LEVERAGE = 2.0
    MAX_CONCENTRATION = 0.25
    MIN_DIVERSIFICATION = 3
    MIN_SHARPE = -1.0
    MAX_VOLATILITY = 0.50
    MAX_DRAWDOWN = 0.30
    LIQUIDITY_THRESHOLD = 0.10
    
    # Performance monitoring
    ENABLE_AUDIT_LOG = True
    ENABLE_PERFORMANCE_TRACKING = True
    ENABLE_RISK_LIMITS = True
    ENABLE_COMPLIANCE_CHECKS = True
    
    # Data management
    MAX_CACHE_ENTRIES = 200
    PARALLEL_WORKERS = min(12, os.cpu_count() or 8)
    CHUNK_SIZE = 100
    TIMEOUT_SECONDS = 45
    DATA_VALIDATION_LEVEL = "strict"  # strict, moderate, lenient
    
    # Visualization
    CHART_HEIGHT = 600
    CHART_WIDTH = 1200
    COLOR_SCHEME = "institutional"
    
    # Reporting
    REPORT_DECIMAL_PLACES = 4
    CURRENCY_SYMBOL = "$"
    
    @classmethod
    def validate_config(cls):
        """Validate configuration parameters"""
        errors = []
        
        if cls.MAX_LEVERAGE < 1.0:
            errors.append("MAX_LEVERAGE must be >= 1.0")
        
        if cls.MAX_CONCENTRATION <= 0 or cls.MAX_CONCENTRATION > 1.0:
            errors.append("MAX_CONCENTRATION must be between 0 and 1")
        
        if cls.MIN_DIVERSIFICATION < 1:
            errors.append("MIN_DIVERSIFICATION must be >= 1")
        
        if cls.TRADING_DAYS < 1:
            errors.append("TRADING_DAYS must be >= 1")
        
        if errors:
            raise ValueError(f"Configuration validation failed: {', '.join(errors)}")
        
        return True

# Validate configuration
try:
    InstitutionalConfig.validate_config()
    logger.info("Configuration validated successfully")
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    st.error(f"Configuration error: {e}")

# =============================================================
# ENHANCED GLOBAL ASSET UNIVERSE WITH COMPREHENSIVE METADATA
# =============================================================
@dataclass
class AssetMetadata:
    """Enhanced asset metadata for institutional use with validation"""
    
    ticker: str
    name: str
    category: str
    region: str
    currency: str
    sector: str = "Unknown"
    market_cap: Optional[float] = None
    inception_date: Optional[str] = None
    expense_ratio: Optional[float] = None
    avg_daily_volume: Optional[float] = None
    dividend_yield: Optional[float] = None
    pe_ratio: Optional[float] = None
    beta: Optional[float] = None
    is_etf: bool = False
    is_currency: bool = False
    is_crypto: bool = False
    is_fund: bool = False
    is_index: bool = False
    risk_level: int = 3  # 1-5 scale, 1 being lowest risk
    
    def __post_init__(self):
        """Validate asset metadata"""
        if not self.ticker or len(self.ticker) > 20:
            raise ValueError(f"Invalid ticker: {self.ticker}")
        
        if self.market_cap is not None and self.market_cap < 0:
            raise ValueError(f"Invalid market cap for {self.ticker}")
        
        if self.expense_ratio is not None and (self.expense_ratio < 0 or self.expense_ratio > 1):
            raise ValueError(f"Invalid expense ratio for {self.ticker}")
        
        if self.risk_level < 1 or self.risk_level > 5:
            raise ValueError(f"Risk level must be between 1 and 5 for {self.ticker}")
    
    def to_dict(self):
        """Convert to dictionary with formatted values"""
        return {
            'ticker': self.ticker,
            'name': self.name,
            'category': self.category,
            'region': self.region,
            'currency': self.currency,
            'sector': self.sector,
            'market_cap': self.market_cap,
            'market_cap_formatted': f"{self.CURRENCY_SYMBOL}{self.market_cap:,.0f}" if self.market_cap else "N/A",
            'expense_ratio': self.expense_ratio,
            'expense_ratio_formatted': f"{self.expense_ratio:.2%}" if self.expense_ratio else "N/A",
            'dividend_yield': self.dividend_yield,
            'dividend_yield_formatted': f"{self.dividend_yield:.2%}" if self.dividend_yield else "N/A",
            'beta': self.beta,
            'beta_formatted': f"{self.beta:.2f}" if self.beta else "N/A",
            'is_etf': self.is_etf,
            'is_crypto': self.is_crypto,
            'risk_level': self.risk_level,
            'risk_description': self._get_risk_description()
        }
    
    def _get_risk_description(self):
        """Get textual description of risk level"""
        risk_descriptions = {
            1: "Very Low Risk",
            2: "Low Risk",
            3: "Moderate Risk",
            4: "High Risk",
            5: "Very High Risk"
        }
        return risk_descriptions.get(self.risk_level, "Unknown Risk")

# Enhanced asset universe with comprehensive metadata
GLOBAL_ASSET_UNIVERSE_ENHANCED = {
    # US Major Indices & ETFs
    "US_Indices": [
        AssetMetadata("SPY", "SPDR S&P 500 ETF", "Equity", "US", "USD", "Broad Market", 
                     market_cap=400000000000, expense_ratio=0.0945, avg_daily_volume=70000000,
                     dividend_yield=0.0132, beta=1.00, is_etf=True, risk_level=3),
        AssetMetadata("QQQ", "Invesco QQQ Trust", "Equity", "US", "USD", "Technology",
                     market_cap=200000000000, expense_ratio=0.20, avg_daily_volume=45000000,
                     dividend_yield=0.0055, beta=1.10, is_etf=True, risk_level=4),
        AssetMetadata("IWM", "iShares Russell 2000 ETF", "Equity", "US", "USD", "Small Cap",
                     market_cap=60000000000, expense_ratio=0.19, avg_daily_volume=35000000,
                     dividend_yield=0.0140, beta=1.25, is_etf=True, risk_level=4),
        AssetMetadata("DIA", "SPDR Dow Jones ETF", "Equity", "US", "USD", "Large Cap",
                     market_cap=30000000000, expense_ratio=0.16, avg_daily_volume=5000000,
                     dividend_yield=0.0180, beta=0.95, is_etf=True, risk_level=3),
        AssetMetadata("VTI", "Vanguard Total Stock Market ETF", "Equity", "US", "USD", "Total Market",
                     market_cap=350000000000, expense_ratio=0.03, avg_daily_volume=3500000,
                     dividend_yield=0.0130, beta=1.00, is_etf=True, risk_level=3),
    ],
    
    # Bonds & Fixed Income
    "Bonds": [
        AssetMetadata("TLT", "iShares 20+ Year Treasury Bond ETF", "Fixed Income", "US", "USD", "Treasury",
                     market_cap=40000000000, expense_ratio=0.15, avg_daily_volume=25000000,
                     dividend_yield=0.0280, beta=0.10, is_etf=True, risk_level=2),
        AssetMetadata("IEF", "iShares 7-10 Year Treasury Bond ETF", "Fixed Income", "US", "USD", "Treasury",
                     market_cap=25000000000, expense_ratio=0.15, avg_daily_volume=5000000,
                     dividend_yield=0.0240, beta=0.05, is_etf=True, risk_level=2),
        AssetMetadata("SHY", "iShares 1-3 Year Treasury Bond ETF", "Fixed Income", "US", "USD", "Treasury",
                     market_cap=24000000000, expense_ratio=0.15, avg_daily_volume=3500000,
                     dividend_yield=0.0200, beta=0.01, is_etf=True, risk_level=1),
        AssetMetadata("BND", "Vanguard Total Bond Market ETF", "Fixed Income", "US", "USD", "Aggregate",
                     market_cap=100000000000, expense_ratio=0.035, avg_daily_volume=4000000,
                     dividend_yield=0.0250, beta=0.03, is_etf=True, risk_level=2),
        AssetMetadata("HYG", "iShares iBoxx High Yield Corporate Bond ETF", "Fixed Income", "US", "USD", "High Yield",
                     market_cap=20000000000, expense_ratio=0.49, avg_daily_volume=20000000,
                     dividend_yield=0.0580, beta=0.40, is_etf=True, risk_level=4),
    ],
    
    # Commodities
    "Commodities": [
        AssetMetadata("GLD", "SPDR Gold Shares", "Commodity", "Global", "USD", "Gold",
                     market_cap=60000000000, expense_ratio=0.40, avg_daily_volume=8000000,
                     dividend_yield=0.00, beta=-0.10, is_etf=True, risk_level=3),
        AssetMetadata("SLV", "iShares Silver Trust", "Commodity", "Global", "USD", "Silver",
                     market_cap=15000000000, expense_ratio=0.50, avg_daily_volume=25000000,
                     dividend_yield=0.00, beta=0.30, is_etf=True, risk_level=4),
        AssetMetadata("USO", "United States Oil Fund", "Commodity", "Global", "USD", "Oil",
                     market_cap=4000000000, expense_ratio=0.83, avg_daily_volume=20000000,
                     dividend_yield=0.00, beta=0.50, is_etf=True, risk_level=5),
        AssetMetadata("UNG", "United States Natural Gas Fund", "Commodity", "Global", "USD", "Natural Gas",
                     market_cap=1000000000, expense_ratio=1.28, avg_daily_volume=10000000,
                     dividend_yield=0.00, beta=0.60, is_etf=True, risk_level=5),
        AssetMetadata("DBA", "Invesco DB Agriculture Fund", "Commodity", "Global", "USD", "Agriculture",
                     market_cap=2000000000, expense_ratio=0.93, avg_daily_volume=1000000,
                     dividend_yield=0.00, beta=0.20, is_etf=True, risk_level=4),
    ],
    
    # Cryptocurrencies
    "Cryptocurrencies": [
        AssetMetadata("BTC-USD", "Bitcoin USD", "Cryptocurrency", "Global", "USD", "Currency",
                     market_cap=900000000000, avg_daily_volume=30000000000,
                     dividend_yield=0.00, beta=2.50, is_crypto=True, risk_level=5),
        AssetMetadata("ETH-USD", "Ethereum USD", "Cryptocurrency", "Global", "USD", "Platform",
                     market_cap=400000000000, avg_daily_volume=15000000000,
                     dividend_yield=0.00, beta=2.80, is_crypto=True, risk_level=5),
        AssetMetadata("BNB-USD", "Binance Coin USD", "Cryptocurrency", "Global", "USD", "Exchange",
                     market_cap=80000000000, avg_daily_volume=2000000000,
                     dividend_yield=0.00, beta=3.00, is_crypto=True, risk_level=5),
        AssetMetadata("XRP-USD", "Ripple USD", "Cryptocurrency", "Global", "USD", "Payment",
                     market_cap=50000000000, avg_daily_volume=2000000000,
                     dividend_yield=0.00, beta=2.20, is_crypto=True, risk_level=5),
        AssetMetadata("ADA-USD", "Cardano USD", "Cryptocurrency", "Global", "USD", "Platform",
                     market_cap=20000000000, avg_daily_volume=1000000000,
                     dividend_yield=0.00, beta=2.60, is_crypto=True, risk_level=5),
    ],
    
    # Global Stocks - US
    "US_Stocks": [
        AssetMetadata("AAPL", "Apple Inc.", "Equity", "US", "USD", "Technology",
                     market_cap=2800000000000, avg_daily_volume=60000000,
                     dividend_yield=0.0055, pe_ratio=28.5, beta=1.20, risk_level=3),
        AssetMetadata("MSFT", "Microsoft Corporation", "Equity", "US", "USD", "Technology",
                     market_cap=2500000000000, avg_daily_volume=25000000,
                     dividend_yield=0.0075, pe_ratio=33.2, beta=0.90, risk_level=3),
        AssetMetadata("GOOGL", "Alphabet Inc. (Class A)", "Equity", "US", "USD", "Technology",
                     market_cap=1800000000000, avg_daily_volume=1500000,
                     dividend_yield=0.0000, pe_ratio=25.8, beta=1.05, risk_level=3),
        AssetMetadata("AMZN", "Amazon.com Inc.", "Equity", "US", "USD", "Consumer Discretionary",
                     market_cap=1500000000000, avg_daily_volume=40000000,
                     dividend_yield=0.0000, pe_ratio=62.5, beta=1.15, risk_level=4),
        AssetMetadata("TSLA", "Tesla Inc.", "Equity", "US", "USD", "Automotive",
                     market_cap=800000000000, avg_daily_volume=100000000,
                     dividend_yield=0.0000, pe_ratio=75.3, beta=2.00, risk_level=5),
        AssetMetadata("JPM", "JPMorgan Chase & Co.", "Equity", "US", "USD", "Financial",
                     market_cap=500000000000, avg_daily_volume=10000000,
                     dividend_yield=0.0240, pe_ratio=11.2, beta=1.10, risk_level=3),
        AssetMetadata("JNJ", "Johnson & Johnson", "Equity", "US", "USD", "Healthcare",
                     market_cap=400000000000, avg_daily_volume=5000000,
                     dividend_yield=0.0290, pe_ratio=16.8, beta=0.60, risk_level=2),
        AssetMetadata("V", "Visa Inc.", "Equity", "US", "USD", "Financial Services",
                     market_cap=500000000000, avg_daily_volume=6000000,
                     dividend_yield=0.0075, pe_ratio=31.5, beta=0.95, risk_level=3),
        AssetMetadata("PG", "Procter & Gamble Co.", "Equity", "US", "USD", "Consumer Staples",
                     market_cap=350000000000, avg_daily_volume=7000000,
                     dividend_yield=0.0240, pe_ratio=25.3, beta=0.40, risk_level=2),
        AssetMetadata("DIS", "The Walt Disney Company", "Equity", "US", "USD", "Communication Services",
                     market_cap=180000000000, avg_daily_volume=10000000,
                     dividend_yield=0.0000, pe_ratio=75.2, beta=1.35, risk_level=4),
    ],
    
    # International & Emerging Markets
    "International": [
        AssetMetadata("VEA", "Vanguard FTSE Developed Markets ETF", "Equity", "International", "USD", "International",
                     market_cap=120000000000, expense_ratio=0.05, avg_daily_volume=3000000,
                     dividend_yield=0.0310, beta=0.85, is_etf=True, risk_level=3),
        AssetMetadata("VWO", "Vanguard FTSE Emerging Markets ETF", "Equity", "Emerging Markets", "USD", "Emerging Markets",
                     market_cap=80000000000, expense_ratio=0.08, avg_daily_volume=15000000,
                     dividend_yield=0.0350, beta=1.10, is_etf=True, risk_level=4),
        AssetMetadata("EWJ", "iShares MSCI Japan ETF", "Equity", "Japan", "USD", "Japan",
                     market_cap=20000000000, expense_ratio=0.50, avg_daily_volume=10000000,
                     dividend_yield=0.0220, beta=0.70, is_etf=True, risk_level=3),
        AssetMetadata("EWU", "iShares MSCI United Kingdom ETF", "Equity", "UK", "USD", "United Kingdom",
                     market_cap=3000000000, expense_ratio=0.50, avg_daily_volume=700000,
                     dividend_yield=0.0400, beta=0.80, is_etf=True, risk_level=3),
        AssetMetadata("EWZ", "iShares MSCI Brazil ETF", "Equity", "Brazil", "USD", "Brazil",
                     market_cap=6000000000, expense_ratio=0.59, avg_daily_volume=20000000,
                     dividend_yield=0.1200, beta=1.50, is_etf=True, risk_level=5),
    ],
    
    # Real Estate
    "Real_Estate": [
        AssetMetadata("VNQ", "Vanguard Real Estate ETF", "Real Estate", "US", "USD", "REIT",
                     market_cap=35000000000, expense_ratio=0.12, avg_daily_volume=5000000,
                     dividend_yield=0.0400, beta=0.85, is_etf=True, risk_level=3),
        AssetMetadata("IYR", "iShares U.S. Real Estate ETF", "Real Estate", "US", "USD", "REIT",
                     market_cap=5000000000, expense_ratio=0.42, avg_daily_volume=10000000,
                     dividend_yield=0.0350, beta=0.90, is_etf=True, risk_level=3),
        AssetMetadata("SCHH", "Schwab U.S. REIT ETF", "Real Estate", "US", "USD", "REIT",
                     market_cap=8000000000, expense_ratio=0.07, avg_daily_volume=600000,
                     dividend_yield=0.0380, beta=0.80, is_etf=True, risk_level=3),
    ],
}

# Create comprehensive ticker lookup dictionaries
TICKER_TO_METADATA = {}
CATEGORY_TO_ASSETS = {}

for category, assets in GLOBAL_ASSET_UNIVERSE_ENHANCED.items():
    CATEGORY_TO_ASSETS[category] = []
    for asset in assets:
        try:
            TICKER_TO_METADATA[asset.ticker] = asset
            CATEGORY_TO_ASSETS[category].append(asset.ticker)
        except ValueError as e:
            logger.warning(f"Skipping invalid asset {asset.ticker}: {e}")

# Flatten universe for selection
ALL_TICKERS_ENHANCED = list(TICKER_TO_METADATA.keys())

# Enhanced symbol mapping for common variations
SYMBOL_MAPPING_ENHANCED = {
    "BTC-USD": ["BTC-USD", "BTCUSD", "BTCUSDT", "BITCOIN", "XBT"],
    "ETH-USD": ["ETH-USD", "ETHUSD", "ETHUSDT", "ETHEREUM"],
    "BRK-B": ["BRK-B", "BRK.B", "BERKSHIRE", "BRKB"],
    "GOOGL": ["GOOGL", "GOOG", "ALPHABET", "GOOGLE"],
    "EURUSD=X": ["EURUSD=X", "EURUSD", "EUR/USD", "EURUSD.FOREX"],
    "GBPUSD=X": ["GBPUSD=X", "GBPUSD", "GBP/USD", "GBPUSD.FOREX"],
    "USDJPY=X": ["USDJPY=X", "USDJPY", "USD/JPY", "USDJPY.FOREX"],
    "SPY": ["SPY", "SPX", "^GSPC", "SP500"],
    "QQQ": ["QQQ", "NDX", "^IXIC", "NASDAQ100"],
    "IWM": ["IWM", "RUT", "^RUT", "RUSSELL2000"],
}

# =============================================================
# ENHANCED CACHE MANAGEMENT WITH SIZE LIMITS AND VALIDATION
# =============================================================
class EnhancedInstitutionalCache:
    """Enhanced cache management for institutional use with size limits and validation"""
    
    def __init__(self, max_entries=InstitutionalConfig.MAX_CACHE_ENTRIES, 
                 max_memory_mb=500):
        self.max_entries = max_entries
        self.max_memory_mb = max_memory_mb
        self.cache = {}
        self.access_log = {}
        self.size_log = {}
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        
    def _estimate_memory_usage(self, obj):
        """Estimate memory usage of an object in bytes"""
        if isinstance(obj, pd.DataFrame):
            return obj.memory_usage(deep=True).sum()
        elif isinstance(obj, np.ndarray):
            return obj.nbytes
        elif isinstance(obj, dict):
            return sum(self._estimate_memory_usage(v) for v in obj.values())
        elif isinstance(obj, list):
            return sum(self._estimate_memory_usage(v) for v in obj)
        else:
            # Rough estimate for other objects
            import sys
            return sys.getsizeof(obj)
    
    def _generate_key(self, *args, **kwargs):
        """Generate deterministic cache key with validation"""
        key_parts = []
        
        # Add function name if provided in kwargs
        func_name = kwargs.pop('__func_name__', None)
        if func_name:
            key_parts.append(func_name)
        
        # Add function arguments with validation
        for arg in args:
            if isinstance(arg, (str, int, float, bool, type(None))):
                key_parts.append(str(arg))
            elif isinstance(arg, pd.DataFrame):
                try:
                    # Use hash of shape, columns, and first/last rows
                    summary = f"{arg.shape}|{','.join(arg.columns)}|{arg.iloc[0].sum():.6f}|{arg.iloc[-1].sum():.6f}"
                    key_parts.append(hashlib.md5(summary.encode()).hexdigest()[:16])
                except:
                    key_parts.append(str(arg.shape))
            elif isinstance(arg, np.ndarray):
                try:
                    # Use hash of shape, dtype, and first/last elements
                    if arg.size > 0:
                        summary = f"{arg.shape}|{arg.dtype}|{arg.flat[0]:.6f}|{arg.flat[-1]:.6f}"
                        key_parts.append(hashlib.md5(summary.encode()).hexdigest()[:16])
                    else:
                        key_parts.append("empty_array")
                except:
                    key_parts.append(str(arg.shape))
            elif isinstance(arg, list):
                # Hash first few elements if list is large
                if len(arg) > 10:
                    key_parts.append(f"list_{len(arg)}_{hashlib.md5(str(arg[:5]).encode()).hexdigest()[:8]}")
                else:
                    key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:16])
            else:
                # Try to get string representation
                try:
                    key_parts.append(hashlib.md5(str(arg).encode()).hexdigest()[:16])
                except:
                    key_parts.append(str(type(arg)))
        
        # Add keyword arguments
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={str(v)[:50]}")
        
        key_string = "|".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def get(self, func, *args, **kwargs):
        """Get cached result or compute with enhanced error handling"""
        self.total_requests += 1
        
        # Add function name to kwargs for key generation
        kwargs_with_func = kwargs.copy()
        kwargs_with_func['__func_name__'] = func.__name__
        
        key = self._generate_key(*args, **kwargs_with_func)
        
        if key in self.cache:
            self.hits += 1
            self.access_log[key] = time.time()
            logger.debug(f"Cache hit for {func.__name__}: {key[:8]}")
            
            # Validate cached data hasn't expired
            if self._is_expired(key):
                logger.debug(f"Cache expired for {key[:8]}, recomputing...")
                return self._compute_and_cache(func, key, *args, **kwargs)
            
            return self.cache[key]
        
        self.misses += 1
        logger.debug(f"Cache miss for {func.__name__}: {key[:8]}")
        return self._compute_and_cache(func, key, *args, **kwargs)
    
    def _compute_and_cache(self, func, key, *args, **kwargs):
        """Compute result and cache it"""
        try:
            result = func(*args, **kwargs)
            
            # Estimate memory usage
            mem_usage_bytes = self._estimate_memory_usage(result)
            mem_usage_mb = mem_usage_bytes / (1024 * 1024)
            
            # Check if we have enough memory
            total_memory_mb = sum(self.size_log.values()) / (1024 * 1024)
            if total_memory_mb + mem_usage_mb > self.max_memory_mb:
                logger.warning(f"Memory limit approaching: {total_memory_mb:.1f}MB used, "
                             f"{mem_usage_mb:.1f}MB needed for new cache entry")
                self._clean_cache(aggressive=True)
            
            # Store in cache
            self.cache[key] = result
            self.access_log[key] = time.time()
            self.size_log[key] = mem_usage_bytes
            
            # Clean cache if too large
            if len(self.cache) > self.max_entries:
                self._clean_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"Error computing cache entry for {key[:8]}: {str(e)[:100]}")
            raise
    
    def _is_expired(self, key, expiry_hours=24):
        """Check if cache entry has expired"""
        if key not in self.access_log:
            return True
        
        age_hours = (time.time() - self.access_log[key]) / 3600
        return age_hours > expiry_hours
    
    def _clean_cache(self, aggressive=False):
        """Remove least recently used entries with memory awareness"""
        if not self.cache:
            return
        
        # Sort by access time
        sorted_entries = sorted(self.access_log.items(), key=lambda x: x[1])
        
        if aggressive:
            # Remove 50% of entries
            entries_to_remove = int(len(self.cache) * 0.5)
        else:
            # Remove excess entries
            entries_to_remove = len(self.cache) - self.max_entries
        
        entries_to_remove = max(1, entries_to_remove)
        
        for key, _ in sorted_entries[:entries_to_remove]:
            if key in self.cache:
                del self.cache[key]
            if key in self.access_log:
                del self.access_log[key]
            if key in self.size_log:
                del self.size_log[key]
        
        logger.info(f"Cache cleaned: removed {entries_to_remove} entries, "
                   f"{len(self.cache)} entries remaining")
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        self.access_log.clear()
        self.size_log.clear()
        self.hits = 0
        self.misses = 0
        self.total_requests = 0
        logger.info("Cache cleared")
    
    def stats(self):
        """Get comprehensive cache statistics"""
        total_memory_mb = sum(self.size_log.values()) / (1024 * 1024) if self.size_log else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_entries,
            'utilization': len(self.cache) / self.max_entries,
            'hits': self.hits,
            'misses': self.misses,
            'total_requests': self.total_requests,
            'hit_rate': self.hits / max(1, self.total_requests),
            'total_memory_mb': total_memory_mb,
            'max_memory_mb': self.max_memory_mb,
            'memory_utilization': total_memory_mb / self.max_memory_mb,
            'avg_entry_size_mb': total_memory_mb / max(1, len(self.cache))
        }
    
    def export_stats(self):
        """Export cache statistics as DataFrame"""
        stats = self.stats()
        return pd.DataFrame([stats])

# Initialize global cache
global_cache = EnhancedInstitutionalCache()

# =============================================================
# ENHANCED DATA LOADER WITH MULTI-SOURCE SUPPORT
# =============================================================
class EnhancedInstitutionalDataLoader:
    """Professional data loader with institutional-grade features and multi-source support"""
    
    def __init__(self, cache_enabled=True, data_quality_check=True):
        self.cache_enabled = cache_enabled
        self.data_quality_check = data_quality_check
        self.data_sources = ['yfinance', 'alpha_vantage', 'tiingo', 'demo']
        self.retry_config = {
            'max_retries': 5,
            'retry_delay': 2.0,
            'backoff_factor': 1.5,
            'timeout': InstitutionalConfig.TIMEOUT_SECONDS
        }
        self.data_quality_metrics = {}
        
    @staticmethod
    def validate_ticker(ticker: str) -> Tuple[bool, str]:
        """Validate ticker format with detailed error messages"""
        if not isinstance(ticker, str):
            return False, "Ticker must be a string"
        
        # Remove common suffixes and clean
        clean_ticker = ticker.upper().strip()
        
        # Basic validation
        if len(clean_ticker) < 1:
            return False, "Ticker cannot be empty"
        if len(clean_ticker) > 20:
            return False, "Ticker too long (max 20 characters)"
        
        # Check for invalid characters
        invalid_chars = set('!@#$%^&*()[]{}|\\;:\'"<>,?`~')
        found_invalid = [c for c in clean_ticker if c in invalid_chars]
        if found_invalid:
            return False, f"Invalid characters in ticker: {found_invalid}"
        
        # Check for minimum alphanumeric characters
        alnum_chars = [c for c in clean_ticker if c.isalnum()]
        if len(alnum_chars) < 1:
            return False, "Ticker must contain at least one alphanumeric character"
        
        return True, "Valid ticker"
    
    def download_with_retry(self, ticker: str, start_date: str, end_date: str, 
                           source: str = 'yfinance', **kwargs) -> Optional[pd.Series]:
        """Download data with intelligent retry logic and fallback"""
        
        # Validate ticker first
        is_valid, message = self.validate_ticker(ticker)
        if not is_valid:
            logger.error(f"Invalid ticker {ticker}: {message}")
            return None
        
        # Try multiple sources if specified
        if source == 'auto':
            sources_to_try = ['yfinance', 'alpha_vantage', 'tiingo', 'demo']
        else:
            sources_to_try = [source]
        
        for current_source in sources_to_try:
            logger.info(f"Attempting to download {ticker} from {current_source}")
            
            for attempt in range(self.retry_config['max_retries']):
                try:
                    if current_source == 'yfinance':
                        result = self._download_yfinance(ticker, start_date, end_date, **kwargs)
                    elif current_source == 'alpha_vantage':
                        result = self._download_alpha_vantage(ticker, start_date, end_date, **kwargs)
                    elif current_source == 'tiingo':
                        result = self._download_tiingo(ticker, start_date, end_date, **kwargs)
                    elif current_source == 'demo':
                        result = self._generate_demo_data(ticker, start_date, end_date, **kwargs)
                    else:
                        logger.error(f"Unknown data source: {current_source}")
                        continue
                    
                    if result is not None and len(result) >= InstitutionalConfig.MIN_DATA_POINTS:
                        logger.info(f"Successfully downloaded {ticker} from {current_source} "
                                  f"({len(result)} data points)")
                        
                        # Store data quality metrics
                        self._record_data_quality(ticker, current_source, result)
                        
                        return result
                    else:
                        logger.warning(f"Download from {current_source} returned insufficient data "
                                     f"for {ticker}")
                        
                except Exception as e:
                    logger.warning(f"Attempt {attempt + 1}/{self.retry_config['max_retries']} "
                                 f"failed for {ticker} from {current_source}: {str(e)[:100]}")
                    
                    if attempt < self.retry_config['max_retries'] - 1:
                        delay = self.retry_config['retry_delay'] * (self.retry_config['backoff_factor'] ** attempt)
                        time.sleep(delay)
                    else:
                        logger.error(f"All attempts failed for {ticker} from {current_source}")
        
        logger.error(f"Failed to download {ticker} from all sources")
        return None
    
    def _download_yfinance(self, ticker: str, start_date: str, end_date: str, 
                          **kwargs) -> Optional[pd.Series]:
        """Download from yfinance with enhanced error handling and validation"""
        
        # Try alternative symbols
        symbols_to_try = [ticker]
        if ticker in SYMBOL_MAPPING_ENHANCED:
            symbols_to_try = SYMBOL_MAPPING_ENHANCED[ticker] + symbols_to_try
        
        for symbol in symbols_to_try:
            try:
                logger.debug(f"Trying yfinance symbol: {symbol}")
                ticker_obj = yf.Ticker(symbol)
                
                # Get info for validation (with timeout)
                try:
                    info = ticker_obj.info
                    is_valid = info.get('regularMarketPrice') is not None or info.get('previousClose') is not None
                    if not is_valid:
                        logger.debug(f"Invalid info for {symbol}, trying next symbol")
                        continue
                except Exception as e:
                    logger.debug(f"Could not get info for {symbol}: {str(e)[:50]}")
                    # Continue anyway, as info might not be available for all symbols
                
                # Download historical data with retry logic
                max_attempts = 3
                for attempt in range(max_attempts):
                    try:
                        hist = ticker_obj.history(
                            start=start_date,
                            end=end_date,
                            interval="1d",
                            auto_adjust=True,
                            prepost=False,
                            timeout=self.retry_config['timeout'],
                            progress=False
                        )
                        break
                    except Exception as e:
                        if attempt == max_attempts - 1:
                            raise
                        time.sleep(1)
                
                if hist.empty:
                    logger.debug(f"Empty history for {symbol}")
                    continue
                
                if len(hist) < InstitutionalConfig.MIN_DATA_POINTS:
                    logger.debug(f"Insufficient data points for {symbol}: {len(hist)}")
                    continue
                
                # Get price column with priority
                price_col = None
                for col in ['Close', 'Adj Close', 'Open', 'High', 'Low']:
                    if col in hist.columns and not hist[col].isna().all():
                        price_col = col
                        break
                
                if price_col is None:
                    logger.debug(f"No valid price column for {symbol}")
                    continue
                
                price_series = hist[price_col]
                
                # Validate data quality
                missing_pct = price_series.isna().sum() / len(price_series)
                if missing_pct > 0.3:
                    logger.debug(f"Too many missing values for {symbol}: {missing_pct:.1%}")
                    continue
                
                # Check for stale data
                latest_date = price_series.index[-1]
                days_since_update = (pd.Timestamp.now() - latest_date).days
                if days_since_update > 30:
                    logger.warning(f"Stale data for {symbol}: latest date {latest_date.date()} "
                                 f"({days_since_update} days ago)")
                
                # Check for outliers and anomalies
                if self.data_quality_check:
                    if not self._validate_price_series(price_series, ticker):
                        logger.warning(f"Data quality check failed for {symbol}")
                        continue
                
                logger.info(f"Successfully downloaded {symbol}: {len(price_series)} points, "
                          f"{missing_pct:.1%} missing")
                return price_series.rename(ticker)  # Rename to original ticker
                
            except Exception as e:
                logger.debug(f"Failed to download {symbol}: {str(e)[:100]}")
                continue
        
        return None
    
    def _download_alpha_vantage(self, ticker: str, start_date: str, end_date: str, 
                               **kwargs) -> Optional[pd.Series]:
        """Download from Alpha Vantage (placeholder)"""
        # Implement Alpha Vantage API integration here
        logger.debug(f"Alpha Vantage not implemented for {ticker}")
        return None
    
    def _download_tiingo(self, ticker: str, start_date: str, end_date: str, 
                        **kwargs) -> Optional[pd.Series]:
        """Download from Tiingo (placeholder)"""
        # Implement Tiingo API integration here
        logger.debug(f"Tiingo not implemented for {ticker}")
        return None
    
    def _validate_price_series(self, price_series: pd.Series, ticker: str) -> bool:
        """Validate price series for quality"""
        try:
            # Check for zero or negative prices
            if (price_series <= 0).any():
                logger.warning(f"Zero or negative prices found for {ticker}")
                return False
            
            # Check for large gaps (>50% daily moves)
            returns = price_series.pct_change().dropna()
            large_moves = returns.abs() > 0.5
            if large_moves.any():
                logger.warning(f"Large daily moves detected for {ticker}: "
                             f"{large_moves.sum()} moves > 50%")
                # Don't fail for large moves, just warn
            
            # Check for consecutive identical values (possible data error)
            consecutive_same = (price_series.diff() == 0).rolling(5).sum() >= 4
            if consecutive_same.any():
                logger.warning(f"Consecutive identical prices for {ticker}")
                # Don't fail for this
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating price series for {ticker}: {str(e)}")
            return False
    
    def _generate_demo_data(self, ticker: str, start_date: str, end_date: str, 
                           **kwargs) -> pd.Series:
        """Generate realistic demo data for testing with enhanced realism"""
        
        # Parse dates
        try:
            start = pd.Timestamp(start_date)
            end = pd.Timestamp(end_date)
        except:
            # Default to 1 year if date parsing fails
            end = pd.Timestamp.today()
            start = end - pd.DateOffset(years=1)
        
        # Generate business days
        dates = pd.bdate_range(start=start, end=end, freq='B')
        if len(dates) < InstitutionalConfig.MIN_DATA_POINTS:
            dates = pd.bdate_range(end=pd.Timestamp.today(), 
                                 periods=InstitutionalConfig.TRADING_DAYS, 
                                 freq='B')
        
        # Get asset type for realistic parameters
        metadata = TICKER_TO_METADATA.get(ticker)
        
        # Set parameters based on asset type
        if metadata:
            if metadata.is_crypto:
                base_price = np.random.uniform(20000, 50000)
                annual_vol = np.random.uniform(0.50, 0.80)
                annual_return = np.random.uniform(0.10, 0.30)
                drift_noise = np.random.uniform(0.02, 0.05)
            elif metadata.category == "Equity":
                base_price = np.random.uniform(50, 500)
                annual_vol = np.random.uniform(0.15, 0.40)
                annual_return = np.random.uniform(0.05, 0.15)
                drift_noise = np.random.uniform(0.01, 0.02)
            elif metadata.category == "Fixed Income":
                base_price = 100
                annual_vol = np.random.uniform(0.05, 0.15)
                annual_return = np.random.uniform(0.02, 0.06)
                drift_noise = np.random.uniform(0.001, 0.005)
            elif metadata.category == "Commodity":
                base_price = np.random.uniform(20, 200)
                annual_vol = np.random.uniform(0.25, 0.50)
                annual_return = np.random.uniform(0.00, 0.10)
                drift_noise = np.random.uniform(0.01, 0.03)
            else:
                base_price = 100
                annual_vol = 0.20
                annual_return = 0.06
                drift_noise = 0.01
        else:
            # Default parameters
            base_price = 100
            annual_vol = 0.20
            annual_return = 0.06
            drift_noise = 0.01
        
        # Generate realistic price path with GBM and jumps
        np.random.seed(hash(ticker) % 2**32)
        n_days = len(dates)
        
        daily_return = annual_return / InstitutionalConfig.TRADING_DAYS
        daily_vol = annual_vol / np.sqrt(InstitutionalConfig.TRADING_DAYS)
        
        # Generate random walk with drift and volatility clustering
        returns = np.zeros(n_days)
        current_vol = daily_vol
        
        for i in range(n_days):
            # Volatility clustering (GARCH-like effect)
            if i > 0:
                current_vol = daily_vol * (1 + 0.5 * abs(returns[i-1] / daily_vol))
            
            # Generate return with drift and noise
            drift = daily_return + drift_noise * np.random.randn()
            returns[i] = drift + current_vol * np.random.randn()
            
            # Add jumps (3% probability)
            if np.random.random() < 0.03:
                jump_size = np.random.choice([-1, 1]) * current_vol * np.random.exponential(2)
                returns[i] += jump_size
        
        # Add autocorrelation
        for i in range(1, n_days):
            returns[i] = 0.1 * returns[i-1] + 0.9 * returns[i]
        
        # Calculate prices
        cumulative_returns = np.exp(np.cumsum(returns))
        prices = base_price * cumulative_returns
        
        # Add some microstructure noise
        prices = prices * (1 + np.random.randn(n_days) * 0.001)
        
        # Ensure no negative prices
        prices = np.maximum(prices, 0.01)
        
        # Create series with proper index
        series = pd.Series(prices, index=dates, name=ticker)
        
        # Add some missing values (5% probability)
        mask = np.random.random(n_days) < 0.05
        series[mask] = np.nan
        
        # Forward fill missing values
        series = series.ffill().bfill()
        
        logger.info(f"Generated demo data for {ticker}: {len(series)} points, "
                   f"volatility={annual_vol:.1%}, return={annual_return:.1%}")
        
        return series
    
    def _record_data_quality(self, ticker: str, source: str, series: pd.Series):
        """Record data quality metrics"""
        self.data_quality_metrics[ticker] = {
            'source': source,
            'data_points': len(series),
            'missing_pct': series.isna().sum() / len(series),
            'start_date': series.index[0],
            'end_date': series.index[-1],
            'mean_price': series.mean(),
            'volatility': series.pct_change().std() * np.sqrt(InstitutionalConfig.TRADING_DAYS),
            'timestamp': datetime.now().isoformat()
        }
    
    def load_batch(self, tickers: List[str], start_date: str, end_date: str, 
                  use_parallel: bool = True, source: str = 'auto') -> pd.DataFrame:
        """Load batch of tickers with parallel processing and enhanced error handling"""
        
        if not tickers:
            logger.warning("No tickers provided for batch load")
            return pd.DataFrame()
        
        # Validate tickers
        valid_tickers = []
        invalid_tickers = []
        
        for ticker in tickers:
            is_valid, message = self.validate_ticker(ticker)
            if is_valid:
                valid_tickers.append(ticker)
            else:
                invalid_tickers.append((ticker, message))
                logger.warning(f"Invalid ticker: {ticker} - {message}")
        
        if len(valid_tickers) < InstitutionalConfig.MIN_ASSETS:
            raise ValueError(f"Need at least {InstitutionalConfig.MIN_ASSETS} valid tickers. "
                           f"Got {len(valid_tickers)}")
        
        # Log invalid tickers
        if invalid_tickers:
            logger.warning(f"{len(invalid_tickers)} invalid tickers: "
                         f"{', '.join([t for t, _ in invalid_tickers])}")
        
        # Check cache if enabled
        cache_key = None
        if self.cache_enabled:
            cache_key = f"batch_{hashlib.md5('|'.join(sorted(valid_tickers)).encode()).hexdigest()}_{start_date}_{end_date}_{source}"
            
            cached_result = st.session_state.get(cache_key)
            if cached_result is not None:
                logger.info(f"Cache hit for batch: {len(valid_tickers)} assets")
                return cached_result
        
        logger.info(f"Loading data for {len(valid_tickers)} assets from {source}...")
        logger.info(f"Date range: {start_date} to {end_date}")
        
        prices_dict = {}
        successful_tickers = []
        failed_tickers = []
        performance_stats = {}
        
        start_time = time.time()
        
        if use_parallel and len(valid_tickers) > 5:
            # Parallel processing with enhanced error handling
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=InstitutionalConfig.PARALLEL_WORKERS,
                thread_name_prefix="DataLoader"
            ) as executor:
                
                # Submit all download tasks
                future_to_ticker = {
                    executor.submit(
                        self.download_with_retry, 
                        ticker, start_date, end_date, source
                    ): ticker for ticker in valid_tickers
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_ticker):
                    ticker = future_to_ticker[future]
                    try:
                        price_data = future.result(timeout=self.retry_config['timeout'])
                        if price_data is not None and len(price_data) >= InstitutionalConfig.MIN_DATA_POINTS:
                            prices_dict[ticker] = price_data
                            successful_tickers.append(ticker)
                            
                            # Record performance
                            performance_stats[ticker] = {
                                'data_points': len(price_data),
                                'success': True
                            }
                        else:
                            failed_tickers.append(ticker)
                            performance_stats[ticker] = {
                                'data_points': len(price_data) if price_data else 0,
                                'success': False,
                                'error': 'Insufficient data'
                            }
                            
                    except concurrent.futures.TimeoutError:
                        failed_tickers.append(ticker)
                        performance_stats[ticker] = {
                            'success': False,
                            'error': 'Timeout'
                        }
                        logger.error(f"Timeout loading {ticker}")
                    except Exception as e:
                        failed_tickers.append(ticker)
                        performance_stats[ticker] = {
                            'success': False,
                            'error': str(e)[:100]
                        }
                        logger.error(f"Failed to load {ticker}: {str(e)[:100]}")
        else:
            # Sequential processing
            for ticker in valid_tickers:
                try:
                    price_data = self.download_with_retry(ticker, start_date, end_date, source)
                    if price_data is not None and len(price_data) >= InstitutionalConfig.MIN_DATA_POINTS:
                        prices_dict[ticker] = price_data
                        successful_tickers.append(ticker)
                        performance_stats[ticker] = {
                            'data_points': len(price_data),
                            'success': True
                        }
                    else:
                        failed_tickers.append(ticker)
                        performance_stats[ticker] = {
                            'data_points': len(price_data) if price_data else 0,
                            'success': False,
                            'error': 'Insufficient data'
                        }
                except Exception as e:
                    failed_tickers.append(ticker)
                    performance_stats[ticker] = {
                        'success': False,
                        'error': str(e)[:100]
                    }
                    logger.error(f"Failed to load {ticker}: {str(e)[:100]}")
        
        # Create DataFrame
        if prices_dict:
            try:
                prices_df = pd.DataFrame(prices_dict)
                
                # Align dates and sort
                prices_df = prices_df.sort_index()
                
                # Handle missing data with enhanced cleaning
                prices_df = self._clean_dataframe_enhanced(prices_df)
                
                if prices_df.empty:
                    raise ValueError("No valid data after cleaning")
                
                # Calculate basic statistics
                total_days = len(prices_df)
                coverage_pct = (prices_df.count() / total_days * 100).mean()
                
                # Store in cache
                if self.cache_enabled and cache_key:
                    st.session_state[cache_key] = prices_df
                
                # Log results
                elapsed_time = time.time() - start_time
                logger.info(f"Batch load completed in {elapsed_time:.2f} seconds")
                logger.info(f"Successfully loaded {len(successful_tickers)}/{len(valid_tickers)} assets")
                logger.info(f"Data coverage: {coverage_pct:.1f}%, Total days: {total_days}")
                
                if failed_tickers:
                    logger.warning(f"Failed to load: {failed_tickers}")
                
                # Store performance metrics
                self._store_batch_metrics(successful_tickers, failed_tickers, 
                                         performance_stats, elapsed_time)
                
                return prices_df
                
            except Exception as e:
                logger.error(f"Error creating DataFrame: {str(e)}")
                raise
        else:
            raise ValueError("Could not load any data")
    
    def _clean_dataframe_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate DataFrame with enhanced methods"""
        
        if df.empty:
            return df
        
        original_columns = df.columns.tolist()
        logger.debug(f"Cleaning DataFrame with {len(df)} rows and {len(df.columns)} columns")
        
        # Step 1: Remove columns with too much missing data (>50%)
        missing_pct = df.isna().mean()
        columns_to_drop = missing_pct[missing_pct > 0.5].index.tolist()
        
        if columns_to_drop:
            logger.debug(f"Dropping columns with >50% missing data: {columns_to_drop}")
            df = df.drop(columns=columns_to_drop)
        
        if len(df.columns) < InstitutionalConfig.MIN_ASSETS:
            logger.error(f"Not enough valid columns after missing data check: {len(df.columns)}")
            return pd.DataFrame()
        
        # Step 2: Forward fill small gaps (max 5 consecutive NaNs)
        logger.debug("Forward filling small gaps...")
        df = df.ffill(limit=5).bfill(limit=5)
        
        # Step 3: Remove any remaining NaNs
        logger.debug("Dropping remaining NaNs...")
        df = df.dropna()
        
        if df.empty:
            logger.error("DataFrame empty after dropping NaNs")
            return pd.DataFrame()
        
        # Step 4: Detect and handle outliers using IQR method
        logger.debug("Checking for outliers...")
        outlier_counts = {}
        
        for col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers > 0:
                outlier_counts[col] = outliers
                # Cap outliers instead of removing
                df[col] = df[col].clip(lower_bound, upper_bound)
        
        if outlier_counts:
            logger.debug(f"Outliers detected and capped: {outlier_counts}")
        
        # Step 5: Check for zero or negative prices
        invalid_prices = {}
        for col in df.columns:
            invalid_mask = df[col] <= 0
            if invalid_mask.any():
                invalid_prices[col] = invalid_mask.sum()
                # Replace with forward fill
                df.loc[invalid_mask, col] = np.nan
                df[col] = df[col].ffill().bfill()
        
        if invalid_prices:
            logger.warning(f"Zero or negative prices found and corrected: {invalid_prices}")
        
        # Step 6: Ensure minimum length
        if len(df) < InstitutionalConfig.MIN_DATA_POINTS:
            logger.warning(f"Insufficient data points after cleaning: {len(df)} < "
                         f"{InstitutionalConfig.MIN_DATA_POINTS}")
            # Try to keep what we have with a warning
        
        # Step 7: Calculate and log cleaning statistics
        final_columns = df.columns.tolist()
        removed_columns = set(original_columns) - set(final_columns)
        
        if removed_columns:
            logger.info(f"Removed {len(removed_columns)} columns during cleaning: {removed_columns}")
        
        logger.info(f"Cleaning complete: {len(df)} rows, {len(df.columns)} columns remaining")
        
        return df
    
    def _store_batch_metrics(self, successful: List[str], failed: List[str],
                            performance_stats: Dict, elapsed_time: float):
        """Store batch loading metrics"""
        self.batch_metrics = {
            'timestamp': datetime.now().isoformat(),
            'successful_count': len(successful),
            'failed_count': len(failed),
            'total_count': len(successful) + len(failed),
            'success_rate': len(successful) / (len(successful) + len(failed)) if (len(successful) + len(failed)) > 0 else 0,
            'elapsed_time_seconds': elapsed_time,
            'successful_tickers': successful,
            'failed_tickers': failed,
            'performance_stats': performance_stats
        }
    
    def get_data_quality_report(self) -> pd.DataFrame:
        """Get comprehensive data quality report"""
        if not hasattr(self, 'data_quality_metrics'):
            return pd.DataFrame()
        
        report_data = []
        for ticker, metrics in self.data_quality_metrics.items():
            report_data.append({
                'Ticker': ticker,
                'Source': metrics.get('source', 'Unknown'),
                'Data Points': metrics.get('data_points', 0),
                'Missing %': f"{metrics.get('missing_pct', 0) * 100:.2f}%",
                'Start Date': metrics.get('start_date'),
                'End Date': metrics.get('end_date'),
                'Mean Price': f"{metrics.get('mean_price', 0):.2f}",
                'Annual Volatility': f"{metrics.get('volatility', 0) * 100:.2f}%",
                'Timestamp': metrics.get('timestamp')
            })
        
        return pd.DataFrame(report_data)
    
    def get_batch_metrics(self) -> Dict:
        """Get batch loading metrics"""
        return getattr(self, 'batch_metrics', {})

# =============================================================
# ENHANCED PORTFOLIO OPTIMIZER WITH INSTITUTIONAL FEATURES
# =============================================================
class EnhancedInstitutionalPortfolioOptimizer:
    """Professional portfolio optimizer with institutional features and comprehensive error handling"""
    
    def __init__(self):
        self.config = InstitutionalConfig()
        self.optimization_history = []
        self.risk_limits_enabled = True
        self.compliance_checks_enabled = True
        self.performance_tracking_enabled = True
        
        # Initialize optimization strategies
        self.optimization_strategies = {
            "Minimum Volatility": self._optimize_min_volatility,
            "Maximum Sharpe Ratio": self._optimize_max_sharpe,
            "Maximum Quadratic Utility": self._optimize_max_quadratic_utility,
            "Efficient Risk": self._optimize_efficient_risk,
            "Efficient Return": self._optimize_efficient_return,
            "Risk Parity": self._optimize_risk_parity,
            "Maximum Diversification": self._optimize_max_diversification,
            "Hierarchical Risk Parity": self._optimize_hrp,
            "Equal Weight": self._optimize_equal_weight,
            "Market Cap Weight": self._optimize_market_cap,
            "Inverse Volatility": self._optimize_inverse_volatility,
            "Most Diversified Portfolio": self._optimize_most_diversified,
        }
        
        # Performance tracking
        self.performance_metrics = {
            'optimizations_performed': 0,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'avg_computation_time': 0,
            'last_optimization_time': None
        }
    
    def optimize_with_constraints(self, returns_df: pd.DataFrame, strategy: str,
                                 constraints: Dict = None, 
                                 risk_free_rate: float = None,
                                 additional_params: Dict = None) -> Dict:
        """Optimize portfolio with institutional constraints and comprehensive error handling"""
        
        # Validate inputs
        self._validate_optimization_inputs(returns_df, strategy)
        
        if risk_free_rate is None:
            risk_free_rate = self.config.DEFAULT_RF_RATE
        
        # Apply constraints
        if constraints is None:
            constraints = self._default_constraints(returns_df)
        
        # Merge additional parameters
        if additional_params:
            constraints.update(additional_params)
        
        # Start timing
        start_time = time.time()
        
        try:
            # Run optimization
            result = self._run_optimization(returns_df, strategy, constraints, risk_free_rate)
            
            # Apply risk limits
            if self.risk_limits_enabled:
                result = self._apply_risk_limits(result, returns_df)
            
            # Apply compliance checks
            if self.compliance_checks_enabled:
                result = self._apply_compliance_checks(result, returns_df, constraints)
            
            # Calculate additional metrics
            result = self._calculate_additional_metrics(result, returns_df, risk_free_rate)
            
            # Log optimization
            self._log_optimization(result, start_time)
            
            # Update performance metrics
            self._update_performance_metrics(start_time, success=True)
            
            return result
            
        except Exception as e:
            # Update performance metrics
            self._update_performance_metrics(start_time, success=False, error=str(e))
            
            logger.error(f"Optimization failed: {str(e)}")
            return self._fallback_optimization(returns_df, risk_free_rate, strategy)
    
    def _validate_optimization_inputs(self, returns_df: pd.DataFrame, strategy: str):
        """Validate optimization inputs with detailed error messages"""
        
        if returns_df is None:
            raise ValueError("Returns DataFrame is None")
        
        if returns_df.empty:
            raise ValueError("Returns DataFrame is empty")
        
        if len(returns_df.columns) < self.config.MIN_ASSETS:
            raise ValueError(f"Need at least {self.config.MIN_ASSETS} assets, got {len(returns_df.columns)}")
        
        if len(returns_df) < self.config.MIN_DATA_POINTS:
            raise ValueError(f"Need at least {self.config.MIN_DATA_POINTS} data points, got {len(returns_df)}")
        
        # Check for NaN or infinite values
        if returns_df.isna().any().any():
            raise ValueError("Returns DataFrame contains NaN values")
        
        if np.isinf(returns_df.values).any():
            raise ValueError("Returns DataFrame contains infinite values")
        
        # Check for constant returns (no variability)
        zero_vol_assets = returns_df.std() == 0
        if zero_vol_assets.any():
            zero_vol_tickers = returns_df.columns[zero_vol_assets].tolist()
            raise ValueError(f"Zero volatility assets detected: {zero_vol_tickers}")
        
        # Validate strategy
        valid_strategies = list(self.optimization_strategies.keys())
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Valid strategies are: {', '.join(valid_strategies)}")
    
    def _default_constraints(self, returns_df: pd.DataFrame) -> Dict:
        """Generate default institutional constraints with validation"""
        
        n_assets = len(returns_df.columns)
        
        return {
            'min_weight': 0.0,
            'max_weight': 1.0,
            'sum_to_one': True,
            'min_diversification': self.config.MIN_DIVERSIFICATION,
            'max_concentration': self.config.MAX_CONCENTRATION,
            'max_leverage': self.config.MAX_LEVERAGE,
            'allow_short': False,
            'asset_groups': None,
            'sector_limits': None,
            'turnover_limit': None,
            'liquidity_threshold': self.config.LIQUIDITY_THRESHOLD,
            'max_drawdown': self.config.MAX_DRAWDOWN,
            'risk_aversion': 1.0,
            'target_risk': None,
            'target_return': None,
            'reg_gamma': self.config.REGULARIZATION_GAMMA,
            'max_iterations': self.config.MAX_ITERATIONS,
            'tolerance': self.config.OPTIMIZATION_TOLERANCE
        }
    
    def _run_optimization(self, returns_df: pd.DataFrame, strategy: str,
                         constraints: Dict, risk_free_rate: float) -> Dict:
        """Run optimization with strategy-specific implementation"""
        
        # Get optimization function
        optimize_func = self.optimization_strategies.get(strategy)
        if optimize_func is None:
            raise ValueError(f"Unknown optimization strategy: {strategy}")
        
        # Try PyPortfolioOpt first if available
        if PYPFOPT_AVAILABLE and strategy not in ["Equal Weight", "Market Cap Weight", "Inverse Volatility"]:
            try:
                result = self._optimize_with_pypfopt(returns_df, strategy, constraints, risk_free_rate)
                if result.get('success', False):
                    return result
            except Exception as e:
                logger.warning(f"PyPortfolioOpt optimization failed for {strategy}: {str(e)[:100]}")
        
        # Fallback to custom optimization
        return optimize_func(returns_df, constraints, risk_free_rate)
    
    def _optimize_with_pypfopt(self, returns_df: pd.DataFrame, strategy: str,
                              constraints: Dict, risk_free_rate: float) -> Dict:
        """Optimize using PyPortfolioOpt with enhanced error handling"""
        
        try:
            # Calculate expected returns and covariance
            mu = expected_returns.mean_historical_return(returns_df)
            
            # Use advanced covariance estimation
            if SKLEARN_AVAILABLE and len(returns_df) > 100:
                try:
                    S = risk_models.CovarianceShrinkage(returns_df).ledoit_wolf()
                except:
                    S = risk_models.sample_cov(returns_df)
            else:
                S = risk_models.sample_cov(returns_df)
            
            # Validate covariance matrix
            if np.any(np.diag(S) <= 0):
                raise ValueError("Covariance matrix has non-positive diagonal elements")
            
            # Create efficient frontier
            ef = EfficientFrontier(
                mu, 
                S,
                weight_bounds=(constraints['min_weight'], constraints['max_weight'])
            )
            
            # Add L2 regularization for stability
            if constraints.get('reg_gamma', 0) > 0:
                ef.add_objective(L2_reg, gamma=constraints['reg_gamma'])
            
            # Apply strategy
            weights_dict = self._apply_pypfopt_strategy(ef, strategy, constraints, risk_free_rate)
            
            # Clean weights
            cleaned_weights = ef.clean_weights(cutoff=1e-6, rounding=6)
            
            # Calculate performance
            expected_return, expected_risk, sharpe_ratio = ef.portfolio_performance(
                risk_free_rate=risk_free_rate / self.config.TRADING_DAYS,
                verbose=False
            )
            
            # Convert to annualized
            expected_return_annual = expected_return * self.config.TRADING_DAYS
            expected_risk_annual = expected_risk * np.sqrt(self.config.TRADING_DAYS)
            sharpe_ratio_annual = sharpe_ratio * np.sqrt(self.config.TRADING_DAYS)
            
            # Convert weights to array
            weights_array = np.array([cleaned_weights.get(asset, 0) for asset in returns_df.columns])
            
            # Calculate diversification metrics
            diversification_ratio = self._calculate_diversification_ratio(weights_array, S)
            effective_n = self._calculate_effective_n(weights_array)
            
            result = {
                'weights': weights_array,
                'expected_return': expected_return_annual,
                'expected_risk': expected_risk_annual,
                'sharpe_ratio': sharpe_ratio_annual,
                'method': f"PyPortfolioOpt {strategy}",
                'cleaned_weights': cleaned_weights,
                'optimizer': 'PyPortfolioOpt',
                'constraints_applied': True,
                'success': True,
                'covariance_matrix': S,
                'expected_returns': mu,
                'diversification_ratio': diversification_ratio,
                'effective_n': effective_n,
                'weights_dict': weights_dict,
                'risk_free_rate': risk_free_rate
            }
            
            return result
            
        except Exception as e:
            logger.error(f"PyPortfolioOpt optimization failed: {str(e)}")
            raise
    
    def _apply_pypfopt_strategy(self, ef: EfficientFrontier, strategy: str,
                               constraints: Dict, risk_free_rate: float) -> Dict:
        """Apply specific optimization strategy with PyPortfolioOpt"""
        
        daily_rf = risk_free_rate / self.config.TRADING_DAYS
        
        if strategy == "Minimum Volatility":
            return ef.min_volatility()
        
        elif strategy == "Maximum Sharpe Ratio":
            return ef.max_sharpe(risk_free_rate=daily_rf)
        
        elif strategy == "Maximum Quadratic Utility":
            risk_aversion = constraints.get('risk_aversion', 1.0)
            return ef.max_quadratic_utility(risk_aversion=risk_aversion)
        
        elif strategy == "Efficient Risk":
            target_risk = constraints.get('target_risk', 0.15)
            return ef.efficient_risk(target_risk=target_risk / np.sqrt(self.config.TRADING_DAYS))
        
        elif strategy == "Efficient Return":
            target_return = constraints.get('target_return', 0.10)
            return ef.efficient_return(target_return=target_return / self.config.TRADING_DAYS)
        
        elif strategy == "Hierarchical Risk Parity" and PYPFOPT_AVAILABLE:
            # Use HRP optimization
            hrp = HRPOpt(ef.expected_returns, ef.cov_matrix)
            return hrp.optimize()
        
        elif strategy == "Most Diversified Portfolio":
            return ef.max_sharpe(risk_free_rate=0)
        
        else:
            # Default to minimum volatility
            return ef.min_volatility()
    
    def _optimize_min_volatility(self, returns_df: pd.DataFrame, 
                                constraints: Dict, risk_free_rate: float) -> Dict:
        """Custom minimum volatility optimization"""
        return self._optimize_custom_base(returns_df, "minimum_volatility", constraints, risk_free_rate)
    
    def _optimize_max_sharpe(self, returns_df: pd.DataFrame,
                            constraints: Dict, risk_free_rate: float) -> Dict:
        """Custom maximum Sharpe ratio optimization"""
        return self._optimize_custom_base(returns_df, "maximum_sharpe", constraints, risk_free_rate)
    
    def _optimize_max_quadratic_utility(self, returns_df: pd.DataFrame,
                                       constraints: Dict, risk_free_rate: float) -> Dict:
        """Custom maximum quadratic utility optimization"""
        return self._optimize_custom_base(returns_df, "maximum_utility", constraints, risk_free_rate)
    
    def _optimize_efficient_risk(self, returns_df: pd.DataFrame,
                                constraints: Dict, risk_free_rate: float) -> Dict:
        """Custom efficient risk optimization"""
        return self._optimize_custom_base(returns_df, "efficient_risk", constraints, risk_free_rate)
    
    def _optimize_efficient_return(self, returns_df: pd.DataFrame,
                                  constraints: Dict, risk_free_rate: float) -> Dict:
        """Custom efficient return optimization"""
        return self._optimize_custom_base(returns_df, "efficient_return", constraints, risk_free_rate)
    
    def _optimize_custom_base(self, returns_df: pd.DataFrame, method: str,
                             constraints: Dict, risk_free_rate: float) -> Dict:
        """Base function for custom optimization methods"""
        
        n_assets = len(returns_df.columns)
        
        # Calculate expected returns and covariance
        mu = returns_df.mean().values * self.config.TRADING_DAYS
        Sigma = returns_df.cov().values * self.config.TRADING_DAYS
        
        # Validate covariance matrix
        if np.any(np.diag(Sigma) <= 0):
            # Add small regularization
            Sigma = Sigma + np.eye(n_assets) * 1e-6
        
        try:
            # Use different optimization methods
            if method == "minimum_volatility":
                weights = self._solve_min_volatility(Sigma, constraints)
            elif method == "maximum_sharpe":
                weights = self._solve_max_sharpe(mu, Sigma, risk_free_rate, constraints)
            elif method == "maximum_utility":
                risk_aversion = constraints.get('risk_aversion', 1.0)
                weights = self._solve_max_utility(mu, Sigma, risk_aversion, constraints)
            elif method == "efficient_risk":
                target_risk = constraints.get('target_risk', 0.15)
                weights = self._solve_efficient_risk(mu, Sigma, target_risk, constraints)
            elif method == "efficient_return":
                target_return = constraints.get('target_return', 0.10)
                weights = self._solve_efficient_return(mu, Sigma, target_return, constraints)
            else:
                # Fallback to equal weight
                weights = np.ones(n_assets) / n_assets
        except Exception as e:
            logger.warning(f"Custom optimization failed for {method}: {str(e)}")
            weights = np.ones(n_assets) / n_assets
        
        # Ensure weights sum to 1
        weights = weights / (weights.sum() + 1e-10)
        
        # Calculate performance
        portfolio_return = mu @ weights
        portfolio_risk = np.sqrt(weights.T @ Sigma @ weights)
        sharpe_ratio = (portfolio_return - risk_free_rate) / (portfolio_risk + 1e-10)
        
        cleaned_weights = dict(zip(returns_df.columns, weights))
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'expected_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'method': f"Custom {method.replace('_', ' ').title()}",
            'cleaned_weights': cleaned_weights,
            'optimizer': 'Custom',
            'constraints_applied': False,
            'success': True,
            'covariance_matrix': Sigma,
            'expected_returns': mu
        }
    
    def _solve_min_volatility(self, Sigma: np.ndarray, constraints: Dict) -> np.ndarray:
        """Solve minimum volatility portfolio"""
        n = Sigma.shape[0]
        
        # Quadratic programming: minimize w'Σw
        # Subject to: sum(w) = 1, w >= 0
        
        from scipy.optimize import minimize
        
        def objective(w):
            return w.T @ Sigma @ w
        
        def constraint_sum(w):
            return np.sum(w) - 1
        
        # Initial guess
        w0 = np.ones(n) / n
        
        # Constraints
        cons = [{'type': 'eq', 'fun': constraint_sum}]
        bounds = [(0, 1) for _ in range(n)]
        
        # Solve
        result = minimize(objective, w0, method='SLSQP', 
                         bounds=bounds, constraints=cons,
                         options={'maxiter': self.config.MAX_ITERATIONS,
                                  'ftol': self.config.OPTIMIZATION_TOLERANCE})
        
        if result.success:
            return result.x
        else:
            raise ValueError(f"Optimization failed: {result.message}")
    
    def _solve_max_sharpe(self, mu: np.ndarray, Sigma: np.ndarray, 
                         risk_free_rate: float, constraints: Dict) -> np.ndarray:
        """Solve maximum Sharpe ratio portfolio"""
        n = len(mu)
        
        # Transform to quadratic programming problem
        from scipy.optimize import minimize
        
        def objective(w):
            portfolio_return = mu @ w
            portfolio_risk = np.sqrt(w.T @ Sigma @ w)
            return - (portfolio_return - risk_free_rate) / (portfolio_risk + 1e-10)
        
        def constraint_sum(w):
            return np.sum(w) - 1
        
        # Initial guess
        w0 = np.ones(n) / n
        
        # Constraints
        cons = [{'type': 'eq', 'fun': constraint_sum}]
        bounds = [(0, 1) for _ in range(n)]
        
        # Solve
        result = minimize(objective, w0, method='SLSQP',
                         bounds=bounds, constraints=cons,
                         options={'maxiter': self.config.MAX_ITERATIONS,
                                  'ftol': self.config.OPTIMIZATION_TOLERANCE})
        
        if result.success:
            return result.x
        else:
            # Fallback to inverse volatility
            volatilities = np.sqrt(np.diag(Sigma))
            weights = 1 / (volatilities + 1e-10)
            return weights / weights.sum()
    
    # Other optimization methods would follow similar patterns...
    
    def _optimize_risk_parity(self, returns_df: pd.DataFrame,
                             constraints: Dict, risk_free_rate: float) -> Dict:
        """Risk parity optimization"""
        
        n_assets = len(returns_df.columns)
        
        # Calculate covariance matrix
        cov_matrix = returns_df.cov().values * self.config.TRADING_DAYS
        
        # Risk parity optimization using iterative method
        weights = self._risk_parity_optimization(cov_matrix)
        
        # Calculate performance
        mu = returns_df.mean().values * self.config.TRADING_DAYS
        portfolio_return = mu @ weights
        portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - risk_free_rate) / (portfolio_risk + 1e-10)
        
        cleaned_weights = dict(zip(returns_df.columns, weights))
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'expected_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'method': "Risk Parity",
            'cleaned_weights': cleaned_weights,
            'optimizer': 'Custom',
            'constraints_applied': False,
            'success': True
        }
    
    def _risk_parity_optimization(self, cov_matrix: np.ndarray, 
                                 max_iterations: int = 1000,
                                 tolerance: float = 1e-8) -> np.ndarray:
        """Risk parity optimization implementation"""
        
        n = len(cov_matrix)
        weights = np.ones(n) / n
        
        for iteration in range(max_iterations):
            # Calculate portfolio volatility
            portfolio_vol = np.sqrt(weights.T @ cov_matrix @ weights)
            
            if portfolio_vol < 1e-10:
                break
            
            # Calculate marginal risk contributions
            marginal_risk = cov_matrix @ weights / portfolio_vol
            
            # Calculate risk contributions
            risk_contributions = weights * marginal_risk
            
            # Calculate target risk contributions (equal)
            target_contributions = np.ones(n) * portfolio_vol / n
            
            # Update weights using gradient method
            adjustment = 0.1 * (target_contributions - risk_contributions) / (marginal_risk + 1e-10)
            weights += adjustment
            
            # Ensure non-negative and normalized
            weights = np.maximum(weights, 0)
            weights_sum = weights.sum()
            if weights_sum > 0:
                weights = weights / weights_sum
            
            # Check convergence
            if np.max(np.abs(adjustment)) < tolerance:
                break
        
        return weights
    
    def _optimize_max_diversification(self, returns_df: pd.DataFrame,
                                     constraints: Dict, risk_free_rate: float) -> Dict:
        """Maximum diversification portfolio optimization"""
        
        n_assets = len(returns_df.columns)
        cov_matrix = returns_df.cov().values * self.config.TRADING_DAYS
        volatilities = np.sqrt(np.diag(cov_matrix))
        
        # Calculate correlation matrix
        D = np.diag(1 / (volatilities + 1e-10))
        corr_matrix = D @ cov_matrix @ D
        
        # Solve for maximum diversification weights
        try:
            weights = np.linalg.solve(corr_matrix, np.ones(n_assets))
            weights = weights / (weights.sum() + 1e-10)
        except np.linalg.LinAlgError:
            # Fallback to inverse volatility
            weights = 1 / (volatilities + 1e-10)
            weights = weights / weights.sum()
        
        # Calculate performance
        mu = returns_df.mean().values * self.config.TRADING_DAYS
        portfolio_return = mu @ weights
        portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - risk_free_rate) / (portfolio_risk + 1e-10)
        
        # Calculate diversification ratio
        weighted_vol = weights @ volatilities
        diversification_ratio = weighted_vol / portfolio_risk if portfolio_risk > 0 else 0
        
        cleaned_weights = dict(zip(returns_df.columns, weights))
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'expected_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'method': "Maximum Diversification",
            'cleaned_weights': cleaned_weights,
            'optimizer': 'Custom',
            'constraints_applied': False,
            'success': True,
            'diversification_ratio': diversification_ratio
        }
    
    def _optimize_hrp(self, returns_df: pd.DataFrame,
                     constraints: Dict, risk_free_rate: float) -> Dict:
        """Hierarchical Risk Parity optimization"""
        
        n_assets = len(returns_df.columns)
        
        if PYPFOPT_AVAILABLE:
            try:
                # Use PyPortfolioOpt's HRP implementation
                cov_matrix = returns_df.cov().values * self.config.TRADING_DAYS
                hrp = HRPOpt(returns_df.mean().values, cov_matrix)
                weights_dict = hrp.optimize()
                
                # Convert to array
                weights = np.array([weights_dict.get(asset, 0) for asset in returns_df.columns])
            except:
                # Fallback to inverse volatility
                volatilities = returns_df.std().values * np.sqrt(self.config.TRADING_DAYS)
                weights = 1 / (volatilities + 1e-10)
                weights = weights / weights.sum()
        else:
            # Simple hierarchical clustering-based approach
            weights = self._simple_hrp(returns_df)
        
        # Calculate performance
        mu = returns_df.mean().values * self.config.TRADING_DAYS
        cov_matrix = returns_df.cov().values * self.config.TRADING_DAYS
        portfolio_return = mu @ weights
        portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - risk_free_rate) / (portfolio_risk + 1e-10)
        
        cleaned_weights = dict(zip(returns_df.columns, weights))
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'expected_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'method': "Hierarchical Risk Parity",
            'cleaned_weights': cleaned_weights,
            'optimizer': 'Custom',
            'constraints_applied': False,
            'success': True
        }
    
    def _simple_hrp(self, returns_df: pd.DataFrame) -> np.ndarray:
        """Simple hierarchical risk parity implementation"""
        
        from scipy.cluster.hierarchy import linkage, fcluster
        from scipy.spatial.distance import squareform
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr().values
        
        # Convert to distance matrix
        distance_matrix = np.sqrt(2 * (1 - corr_matrix))
        
        # Perform hierarchical clustering
        condensed_dist = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method='ward')
        
        # Get clusters
        n_assets = len(returns_df.columns)
        clusters = fcluster(linkage_matrix, t=0.5, criterion='distance')
        
        # Calculate inverse volatility weights within each cluster
        volatilities = returns_df.std().values * np.sqrt(self.config.TRADING_DAYS)
        weights = np.zeros(n_assets)
        
        for cluster_id in np.unique(clusters):
            cluster_indices = np.where(clusters == cluster_id)[0]
            if len(cluster_indices) > 0:
                cluster_vols = volatilities[cluster_indices]
                cluster_weights = 1 / (cluster_vols + 1e-10)
                cluster_weights = cluster_weights / cluster_weights.sum()
                weights[cluster_indices] = cluster_weights
        
        # Normalize across all assets
        weights = weights / weights.sum()
        
        return weights
    
    def _optimize_equal_weight(self, returns_df: pd.DataFrame,
                              constraints: Dict, risk_free_rate: float) -> Dict:
        """Equal weight portfolio"""
        n_assets = len(returns_df.columns)
        weights = np.ones(n_assets) / n_assets
        
        return self._calculate_portfolio_metrics(returns_df, weights, "Equal Weight", risk_free_rate)
    
    def _optimize_market_cap(self, returns_df: pd.DataFrame,
                            constraints: Dict, risk_free_rate: float) -> Dict:
        """Market cap weighted portfolio"""
        n_assets = len(returns_df.columns)
        
        # Get market caps from metadata
        market_caps = []
        for ticker in returns_df.columns:
            metadata = TICKER_TO_METADATA.get(ticker)
            if metadata and metadata.market_cap:
                market_caps.append(metadata.market_cap)
            else:
                market_caps.append(1.0)  # Default if no market cap available
        
        market_caps = np.array(market_caps)
        weights = market_caps / market_caps.sum()
        
        return self._calculate_portfolio_metrics(returns_df, weights, "Market Cap Weight", risk_free_rate)
    
    def _optimize_inverse_volatility(self, returns_df: pd.DataFrame,
                                    constraints: Dict, risk_free_rate: float) -> Dict:
        """Inverse volatility weighted portfolio"""
        volatilities = returns_df.std().values * np.sqrt(self.config.TRADING_DAYS)
        weights = 1 / (volatilities + 1e-10)
        weights = weights / weights.sum()
        
        return self._calculate_portfolio_metrics(returns_df, weights, "Inverse Volatility", risk_free_rate)
    
    def _calculate_portfolio_metrics(self, returns_df: pd.DataFrame, 
                                    weights: np.ndarray, 
                                    method: str, 
                                    risk_free_rate: float) -> Dict:
        """Calculate portfolio metrics for given weights"""
        
        mu = returns_df.mean().values * self.config.TRADING_DAYS
        cov_matrix = returns_df.cov().values * self.config.TRADING_DAYS
        
        portfolio_return = mu @ weights
        portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe_ratio = (portfolio_return - risk_free_rate) / (portfolio_risk + 1e-10)
        
        # Calculate diversification metrics
        diversification_ratio = self._calculate_diversification_ratio(weights, cov_matrix)
        effective_n = self._calculate_effective_n(weights)
        
        cleaned_weights = dict(zip(returns_df.columns, weights))
        
        return {
            'weights': weights,
            'expected_return': portfolio_return,
            'expected_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'method': method,
            'cleaned_weights': cleaned_weights,
            'optimizer': 'Custom',
            'constraints_applied': False,
            'success': True,
            'diversification_ratio': diversification_ratio,
            'effective_n': effective_n
        }
    
    def _calculate_diversification_ratio(self, weights: np.ndarray, 
                                        cov_matrix: np.ndarray) -> float:
        """Calculate diversification ratio"""
        weighted_vol = weights @ np.sqrt(np.diag(cov_matrix))
        portfolio_risk = np.sqrt(weights.T @ cov_matrix @ weights)
        
        if portfolio_risk > 0:
            return weighted_vol / portfolio_risk
        return 0.0
    
    def _calculate_effective_n(self, weights: np.ndarray) -> float:
        """Calculate effective number of assets"""
        if np.any(weights < 0):
            # For portfolios with short positions, use absolute weights
            weights = np.abs(weights)
            weights = weights / weights.sum()
        
        if np.sum(weights**2) > 0:
            return 1 / np.sum(weights**2)
        return 0.0
    
    def _apply_risk_limits(self, result: Dict, returns_df: pd.DataFrame) -> Dict:
        """Apply institutional risk limits with comprehensive checks"""
        
        weights = result['weights'].copy()
        n_assets = len(weights)
        
        # Check concentration limits
        max_weight = np.max(weights)
        if max_weight > self.config.MAX_CONCENTRATION:
            logger.warning(f"Concentration limit exceeded: {max_weight:.2%} > {self.config.MAX_CONCENTRATION:.2%}")
            
            # Apply concentration limit
            excess = max_weight - self.config.MAX_CONCENTRATION
            max_index = np.argmax(weights)
            weights[max_index] -= excess
            
            # Redistribute excess to other assets proportionally
            other_indices = [i for i in range(n_assets) if i != max_index]
            if other_indices:
                other_weights = weights[other_indices]
                other_weights_sum = other_weights.sum()
                if other_weights_sum > 0:
                    redistribution = excess * (other_weights / other_weights_sum)
                    weights[other_indices] += redistribution
        
        # Check diversification
        effective_n = self._calculate_effective_n(weights)
        if effective_n < self.config.MIN_DIVERSIFICATION:
            logger.warning(f"Diversification below minimum: {effective_n:.1f} < {self.config.MIN_DIVERSIFICATION}")
            
            # Try to improve diversification by reducing largest weights
            sorted_indices = np.argsort(weights)[::-1]
            while effective_n < self.config.MIN_DIVERSIFICATION and len(sorted_indices) > 1:
                # Reduce largest weight and redistribute
                largest_idx = sorted_indices[0]
                reduction = min(weights[largest_idx] * 0.1, 0.05)
                weights[largest_idx] -= reduction
                
                # Redistribute to smallest weights
                smallest_idx = sorted_indices[-1]
                weights[smallest_idx] += reduction
                
                # Recalculate effective N
                effective_n = self._calculate_effective_n(weights)
                sorted_indices = np.argsort(weights)[::-1]
        
        # Check for negative weights (short positions)
        if not result.get('allow_short', False):
            negative_weights = weights[weights < 0]
            if len(negative_weights) > 0:
                logger.warning(f"Negative weights found and set to zero: {negative_weights}")
                weights = np.maximum(weights, 0)
                weights = weights / weights.sum()
        
        # Recalculate performance with adjusted weights
        portfolio_returns_series = (returns_df * weights).sum(axis=1)
        mu = returns_df.mean().values * self.config.TRADING_DAYS
        cov_matrix = returns_df.cov().values * self.config.TRADING_DAYS
        
        result['weights'] = weights
        result['expected_return'] = mu @ weights
        result['expected_risk'] = np.sqrt(weights.T @ cov_matrix @ weights)
        
        if result.get('risk_free_rate'):
            result['sharpe_ratio'] = (result['expected_return'] - result['risk_free_rate']) / (result['expected_risk'] + 1e-10)
        
        result['risk_limits_applied'] = True
        result['effective_diversification'] = effective_n
        result['max_concentration'] = weights.max()
        result['concentration_herfindahl'] = np.sum(weights ** 2)
        
        return result
    
    def _apply_compliance_checks(self, result: Dict, returns_df: pd.DataFrame,
                                constraints: Dict) -> Dict:
        """Apply compliance checks to portfolio"""
        
        compliance_issues = []
        
        # Check leverage
        leverage = np.sum(np.abs(result['weights']))
        if leverage > constraints.get('max_leverage', self.config.MAX_LEVERAGE):
            compliance_issues.append(f"Leverage {leverage:.2f} exceeds limit {constraints['max_leverage']}")
        
        # Check volatility
        if result['expected_risk'] > self.config.MAX_VOLATILITY:
            compliance_issues.append(f"Volatility {result['expected_risk']:.2%} exceeds limit {self.config.MAX_VOLATILITY:.2%}")
        
        # Check Sharpe ratio
        if result['sharpe_ratio'] < self.config.MIN_SHARPE:
            compliance_issues.append(f"Sharpe ratio {result['sharpe_ratio']:.2f} below minimum {self.config.MIN_SHARPE}")
        
        # Check sector limits if provided
        if constraints.get('sector_limits'):
            sector_exposure = self._calculate_sector_exposure(result['weights'], returns_df.columns)
            for sector, limit in constraints['sector_limits'].items():
                if sector in sector_exposure and sector_exposure[sector] > limit:
                    compliance_issues.append(f"Sector {sector} exposure {sector_exposure[sector]:.2%} exceeds limit {limit:.2%}")
        
        result['compliance_issues'] = compliance_issues
        result['compliance_passed'] = len(compliance_issues) == 0
        
        if compliance_issues:
            logger.warning(f"Compliance issues detected: {compliance_issues}")
        
        return result
    
    def _calculate_sector_exposure(self, weights: np.ndarray, tickers: List[str]) -> Dict:
        """Calculate sector exposure for portfolio"""
        sector_exposure = {}
        
        for ticker, weight in zip(tickers, weights):
            metadata = TICKER_TO_METADATA.get(ticker)
            if metadata:
                sector = metadata.sector
                sector_exposure[sector] = sector_exposure.get(sector, 0) + weight
        
        return sector_exposure
    
    def _calculate_additional_metrics(self, result: Dict, returns_df: pd.DataFrame,
                                     risk_free_rate: float) -> Dict:
        """Calculate additional portfolio metrics"""
        
        weights = result['weights']
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # Calculate drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Calculate Sortino ratio
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(self.config.TRADING_DAYS) if len(downside_returns) > 0 else 0
        sortino_ratio = (result['expected_return'] - risk_free_rate) / (downside_dev + 1e-10)
        
        # Calculate information ratio (vs equal weight benchmark)
        benchmark_weights = np.ones(len(weights)) / len(weights)
        benchmark_returns = (returns_df * benchmark_weights).sum(axis=1)
        active_returns = portfolio_returns - benchmark_returns
        tracking_error = active_returns.std() * np.sqrt(self.config.TRADING_DAYS)
        information_ratio = active_returns.mean() * self.config.TRADING_DAYS / (tracking_error + 1e-10)
        
        # Calculate beta to market (using equal weight as proxy)
        covariance = np.cov(portfolio_returns, benchmark_returns)[0, 1]
        benchmark_variance = benchmark_returns.var()
        beta = covariance / (benchmark_variance + 1e-10) if benchmark_variance > 0 else 1.0
        
        # Calculate turnover (simplified)
        if 'previous_weights' in result:
            previous_weights = result['previous_weights']
            turnover = np.sum(np.abs(weights - previous_weights)) / 2
        else:
            turnover = 0.0
        
        # Add metrics to result
        result.update({
            'max_drawdown': max_drawdown,
            'sortino_ratio': sortino_ratio,
            'information_ratio': information_ratio,
            'beta': beta,
            'tracking_error': tracking_error,
            'turnover': turnover,
            'downside_deviation': downside_dev,
            'skewness': portfolio_returns.skew(),
            'kurtosis': portfolio_returns.kurtosis(),
            'var_95': np.percentile(portfolio_returns, 5),
            'cvar_95': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean()
        })
        
        return result
    
    def _log_optimization(self, result: Dict, start_time: float):
        """Log optimization results with comprehensive details"""
        
        elapsed_time = time.time() - start_time
        
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'method': result.get('method', 'Unknown'),
            'success': result.get('success', False),
            'expected_return': float(result.get('expected_return', 0)),
            'expected_risk': float(result.get('expected_risk', 0)),
            'sharpe_ratio': float(result.get('sharpe_ratio', 0)),
            'max_drawdown': float(result.get('max_drawdown', 0)),
            'sortino_ratio': float(result.get('sortino_ratio', 0)),
            'n_assets': len(result.get('weights', [])),
            'optimizer': result.get('optimizer', 'Unknown'),
            'elapsed_time_seconds': elapsed_time,
            'risk_limits_applied': result.get('risk_limits_applied', False),
            'compliance_passed': result.get('compliance_passed', True),
            'effective_diversification': float(result.get('effective_diversification', 0)),
            'diversification_ratio': float(result.get('diversification_ratio', 0))
        }
        
        self.optimization_history.append(log_entry)
        
        # Keep only last 1000 entries
        if len(self.optimization_history) > 1000:
            self.optimization_history = self.optimization_history[-1000:]
        
        logger.info(f"Optimization logged: {result['method']}, "
                   f"Return: {result['expected_return']:.2%}, "
                   f"Risk: {result['expected_risk']:.2%}, "
                   f"Sharpe: {result['sharpe_ratio']:.2f}, "
                   f"Time: {elapsed_time:.2f}s")
    
    def _update_performance_metrics(self, start_time: float, success: bool, error: str = None):
        """Update performance tracking metrics"""
        
        elapsed_time = time.time() - start_time
        
        self.performance_metrics['optimizations_performed'] += 1
        
        if success:
            self.performance_metrics['successful_optimizations'] += 1
        else:
            self.performance_metrics['failed_optimizations'] += 1
        
        # Update average computation time
        current_avg = self.performance_metrics['avg_computation_time']
        n_successful = self.performance_metrics['successful_optimizations']
        
        if n_successful > 0:
            self.performance_metrics['avg_computation_time'] = (
                (current_avg * (n_successful - 1) + elapsed_time) / n_successful
            )
        
        self.performance_metrics['last_optimization_time'] = datetime.now().isoformat()
        
        if error:
            self.performance_metrics['last_error'] = error[:200]
    
    def _fallback_optimization(self, returns_df: pd.DataFrame, 
                              risk_free_rate: float, strategy: str) -> Dict:
        """Fallback optimization when everything fails"""
        
        n_assets = len(returns_df.columns)
        weights = np.ones(n_assets) / n_assets
        
        # Calculate basic metrics
        portfolio_returns = (returns_df * weights).sum(axis=1)
        expected_return = portfolio_returns.mean() * self.config.TRADING_DAYS
        expected_risk = portfolio_returns.std() * np.sqrt(self.config.TRADING_DAYS)
        sharpe_ratio = (expected_return - risk_free_rate) / (expected_risk + 1e-10)
        
        cleaned_weights = dict(zip(returns_df.columns, weights))
        
        return {
            'weights': weights,
            'expected_return': expected_return,
            'expected_risk': expected_risk,
            'sharpe_ratio': sharpe_ratio,
            'method': f'Equal Weight (Fallback for {strategy})',
            'cleaned_weights': cleaned_weights,
            'optimizer': 'Fallback',
            'constraints_applied': False,
            'success': False,
            'error': 'All optimization methods failed, using equal weight as fallback'
        }
    
    def get_optimization_history(self) -> pd.DataFrame:
        """Get optimization history as DataFrame"""
        return pd.DataFrame(self.optimization_history)
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics"""
        return self.performance_metrics.copy()
    
    def clear_history(self):
        """Clear optimization history"""
        self.optimization_history.clear()
        self.performance_metrics = {
            'optimizations_performed': 0,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'avg_computation_time': 0,
            'last_optimization_time': None
        }
        logger.info("Optimization history cleared")

# =============================================================
# ENHANCED RISK ANALYTICS ENGINE WITH COMPREHENSIVE METRICS
# =============================================================
class EnhancedInstitutionalRiskAnalytics:
    """Professional risk analytics with institutional features and comprehensive metrics"""
    
    def __init__(self):
        self.config = InstitutionalConfig()
        self.risk_metrics_history = []
        self.var_methods = {
            'historical': self._historical_var,
            'parametric': self._parametric_var,
            'ewma': self._ewma_var,
            'monte_carlo': self._monte_carlo_var,
            'cornish_fisher': self._cornish_fisher_var,
            'modified_var': self._modified_var,
            'garch_var': self._garch_var
        }
        
        # Initialize stress test scenarios
        self.stress_scenarios = self._initialize_stress_scenarios()
    
    def _initialize_stress_scenarios(self) -> Dict:
        """Initialize comprehensive stress test scenarios"""
        return {
            '2008_Financial_Crisis': {
                'description': 'Global Financial Crisis 2008',
                'equity_shock': -0.50,
                'bond_shock': 0.15,  # Flight to quality
                'commodity_shock': -0.40,
                'fx_shock': 0.20,
                'duration_days': 252,
                'recovery_months': 60
            },
            '2020_COVID_Crash': {
                'description': 'COVID-19 Market Crash',
                'equity_shock': -0.35,
                'bond_shock': 0.10,
                'commodity_shock': -0.60,
                'fx_shock': 0.15,
                'duration_days': 30,
                'recovery_months': 12
            },
            '2011_European_Debt_Crisis': {
                'description': 'European Sovereign Debt Crisis',
                'equity_shock': -0.20,
                'bond_shock': -0.15,  # Sovereign bonds
                'commodity_shock': -0.15,
                'fx_shock': 0.25,  # EUR weakness
                'duration_days': 180,
                'recovery_months': 24
            },
                        '2015_China_Stock_Market_Crash': {
                'description': 'Chinese Stock Market Crash',
                'equity_shock': -0.30,
                'bond_shock': 0.08,
                'commodity_shock': -0.25,
                'fx_shock': 0.10,
                'duration_days': 60,
                'recovery_months': 18
            },
            'Tech_Bubble_2000': {
                'description': 'Dot-com Bubble Burst',
                'equity_shock': -0.45,
                'bond_shock': 0.12,
                'commodity_shock': 0.05,
                'fx_shock': 0.08,
                'duration_days': 500,
                'recovery_months': 84
            },
            'Black_Monday_1987': {
                'description': 'Black Monday 1987',
                'equity_shock': -0.23,
                'bond_shock': 0.05,
                'commodity_shock': -0.10,
                'fx_shock': 0.05,
                'duration_days': 1,
                'recovery_months': 20
            },
            'Interest_Rate_Shock_1994': {
                'description': '1994 Bond Market Massacre',
                'equity_shock': -0.10,
                'bond_shock': -0.20,
                'commodity_shock': 0.05,
                'fx_shock': 0.15,
                'duration_days': 90,
                'recovery_months': 36
            },
            'Custom_Stress': {
                'description': 'Custom Stress Scenario',
                'equity_shock': -0.20,
                'bond_shock': -0.10,
                'commodity_shock': -0.15,
                'fx_shock': 0.10,
                'duration_days': 30,
                'recovery_months': 12
            }
        }
    
    def compute_comprehensive_risk_metrics(self, returns_df: pd.DataFrame,
                                         portfolio_weights: np.ndarray,
                                         risk_free_rate: float = None) -> Dict:
        """Compute comprehensive institutional risk metrics"""
        
        if risk_free_rate is None:
            risk_free_rate = self.config.DEFAULT_RF_RATE
        
        try:
            # Calculate portfolio returns
            portfolio_returns = (returns_df * portfolio_weights).sum(axis=1)
            
            # Basic risk metrics
            risk_metrics = self._compute_basic_risk_metrics(portfolio_returns, risk_free_rate)
            
            # Value at Risk metrics
            var_metrics = self._compute_var_metrics(portfolio_returns)
            
            # Stress testing
            stress_metrics = self._compute_stress_test_metrics(returns_df, portfolio_weights)
            
            # Factor risk metrics
            factor_metrics = self._compute_factor_risk_metrics(returns_df, portfolio_weights, portfolio_returns)
            
            # Liquidity metrics
            liquidity_metrics = self._compute_liquidity_metrics(returns_df, portfolio_weights)
            
            # Concentration metrics
            concentration_metrics = self._compute_concentration_metrics(portfolio_weights)
            
            # Combine all metrics
            comprehensive_metrics = {
                **risk_metrics,
                **var_metrics,
                **stress_metrics,
                **factor_metrics,
                **liquidity_metrics,
                **concentration_metrics,
                'calculation_timestamp': datetime.now().isoformat(),
                'risk_free_rate': risk_free_rate,
                'n_assets': len(portfolio_weights),
                'portfolio_return_series': portfolio_returns
            }
            
            # Store in history
            self._store_risk_metrics(comprehensive_metrics)
            
            return comprehensive_metrics
            
        except Exception as e:
            logger.error(f"Error computing risk metrics: {str(e)}")
            raise
    
    def _compute_basic_risk_metrics(self, portfolio_returns: pd.Series,
                                   risk_free_rate: float) -> Dict:
        """Compute basic risk metrics"""
        
        # Annualization factor
        annual_factor = self.config.TRADING_DAYS
        
        # Calculate metrics
        annual_return = portfolio_returns.mean() * annual_factor
        annual_volatility = portfolio_returns.std() * np.sqrt(annual_factor)
        sharpe_ratio = (annual_return - risk_free_rate) / (annual_volatility + 1e-10)
        
        # Downside metrics
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(annual_factor) if len(downside_returns) > 0 else 0
        sortino_ratio = (annual_return - risk_free_rate) / (downside_volatility + 1e-10)
        
        # Drawdown analysis
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        max_drawdown = drawdown.min()
        avg_drawdown = drawdown.mean()
        drawdown_duration = (drawdown < 0).astype(int).groupby((drawdown < 0).astype(int).diff().ne(0).cumsum()).sum()
        max_drawdown_duration = drawdown_duration.max() if not drawdown_duration.empty else 0
        
        # Higher moment statistics
        skewness = portfolio_returns.skew()
        kurtosis = portfolio_returns.kurtosis()
        excess_kurtosis = kurtosis - 3
        
        # Omega ratio (allows for non-normal returns)
        threshold = risk_free_rate / self.config.TRADING_DAYS
        gains = portfolio_returns[portfolio_returns > threshold].sum()
        losses = abs(portfolio_returns[portfolio_returns <= threshold].sum())
        omega_ratio = gains / (losses + 1e-10)
        
        # Calmar ratio
        calmar_ratio = annual_return / (abs(max_drawdown) + 1e-10)
        
        # Martin ratio (Ulcer index)
        ulcer_index = np.sqrt((drawdown**2).mean())
        martin_ratio = annual_return / (ulcer_index + 1e-10)
        
        # Tail ratio (95th/5th percentile)
        tail_ratio = abs(portfolio_returns.quantile(0.95) / portfolio_returns.quantile(0.05))
        
        # Gain to pain ratio
        total_gain = portfolio_returns[portfolio_returns > 0].sum()
        total_loss = abs(portfolio_returns[portfolio_returns < 0].sum())
        gain_to_pain = total_gain / (total_loss + 1e-10)
        
        # Recovery factor
        if max_drawdown < 0:
            recovery_factor = annual_return / abs(max_drawdown)
        else:
            recovery_factor = np.inf
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'omega_ratio': omega_ratio,
            'calmar_ratio': calmar_ratio,
            'martin_ratio': martin_ratio,
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_drawdown_duration': max_drawdown_duration,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'excess_kurtosis': excess_kurtosis,
            'tail_ratio': tail_ratio,
            'gain_to_pain_ratio': gain_to_pain,
            'recovery_factor': recovery_factor,
            'downside_volatility': downside_volatility,
            'ulcer_index': ulcer_index
        }
    
    def _compute_var_metrics(self, portfolio_returns: pd.Series,
                            confidence_levels: List[float] = None) -> Dict:
        """Compute Value at Risk metrics using multiple methods"""
        
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99, 0.995]
        
        var_metrics = {}
        
        for confidence in confidence_levels:
            var_key = f"var_{int(confidence*100)}"
            cvar_key = f"cvar_{int(confidence*100)}"
            
            # Historical VaR
            historical_var = self._historical_var(portfolio_returns, confidence)
            historical_cvar = self._historical_cvar(portfolio_returns, confidence)
            
            # Parametric VaR (assuming normality)
            parametric_var = self._parametric_var(portfolio_returns, confidence)
            parametric_cvar = self._parametric_cvar(portfolio_returns, confidence)
            
            # Cornish-Fisher VaR (adjusting for skewness and kurtosis)
            cornish_fisher_var = self._cornish_fisher_var(portfolio_returns, confidence)
            cornish_fisher_cvar = self._modified_var(portfolio_returns, confidence)
            
            var_metrics.update({
                f'{var_key}_historical': historical_var,
                f'{cvar_key}_historical': historical_cvar,
                f'{var_key}_parametric': parametric_var,
                f'{cvar_key}_parametric': parametric_cvar,
                f'{var_key}_cornish_fisher': cornish_fisher_var,
                f'{cvar_key}_cornish_fisher': cornish_fisher_cvar,
            })
        
        # Additional VaR metrics
        var_metrics.update({
            'worst_daily_loss': portfolio_returns.min(),
            'best_daily_gain': portfolio_returns.max(),
            'var_breaches_95': self._count_var_breaches(portfolio_returns, 0.95, var_metrics['var_95_historical']),
            'var_breaches_99': self._count_var_breaches(portfolio_returns, 0.99, var_metrics['var_99_historical']),
            'max_consecutive_losses': self._max_consecutive_losses(portfolio_returns)
        })
        
        return var_metrics
    
    def _historical_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Historical Value at Risk"""
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _historical_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Historical Conditional Value at Risk (Expected Shortfall)"""
        var = self._historical_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def _parametric_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Parametric (Normal) Value at Risk"""
        from scipy.stats import norm
        mu = returns.mean()
        sigma = returns.std()
        z_score = norm.ppf(1 - confidence)
        return mu + z_score * sigma
    
    def _parametric_cvar(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Parametric Conditional Value at Risk"""
        from scipy.stats import norm
        mu = returns.mean()
        sigma = returns.std()
        z_score = norm.ppf(1 - confidence)
        return mu + sigma * norm.pdf(z_score) / (1 - confidence)
    
    def _cornish_fisher_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Cornish-Fisher Value at Risk (adjusts for skewness and kurtosis)"""
        from scipy.stats import norm
        
        mu = returns.mean()
        sigma = returns.std()
        skew = returns.skew()
        kurt = returns.kurtosis()
        
        z_alpha = norm.ppf(1 - confidence)
        
        # Cornish-Fisher expansion
        z_cf = (z_alpha + 
                (z_alpha**2 - 1) * skew / 6 +
                (z_alpha**3 - 3*z_alpha) * (kurt - 3) / 24 -
                (2*z_alpha**3 - 5*z_alpha) * skew**2 / 36)
        
        return mu + z_cf * sigma
    
    def _modified_var(self, returns: pd.Series, confidence: float = 0.95) -> float:
        """Modified VaR (similar to Cornish-Fisher)"""
        return self._cornish_fisher_var(returns, confidence)
    
    def _count_var_breaches(self, returns: pd.Series, confidence: float, 
                           var_value: float) -> int:
        """Count VaR breaches"""
        return (returns < var_value).sum()
    
    def _max_consecutive_losses(self, returns: pd.Series) -> int:
        """Calculate maximum consecutive losses"""
        losses = returns < 0
        if not losses.any():
            return 0
        
        max_consecutive = 0
        current_consecutive = 0
        
        for is_loss in losses:
            if is_loss:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0
        
        return max_consecutive
    
    def _compute_stress_test_metrics(self, returns_df: pd.DataFrame,
                                    portfolio_weights: np.ndarray) -> Dict:
        """Compute stress testing metrics for various scenarios"""
        
        stress_metrics = {}
        
        for scenario_name, scenario_params in self.stress_scenarios.items():
            try:
                # Apply scenario shocks
                scenario_returns = self._apply_stress_scenario(returns_df, scenario_params)
                
                # Calculate portfolio impact
                portfolio_scenario_returns = (scenario_returns * portfolio_weights).sum(axis=1)
                
                # Calculate metrics
                total_loss = portfolio_scenario_returns.sum()
                max_daily_loss = portfolio_scenario_returns.min()
                recovery_months = scenario_params.get('recovery_months', 12)
                
                # Calculate expected recovery time (simplified)
                avg_daily_return = portfolio_scenario_returns.mean()
                if avg_daily_return > 0:
                    recovery_days = abs(total_loss) / (avg_daily_return * self.config.TRADING_DAYS / 12)
                else:
                    recovery_days = np.inf
                
                stress_metrics.update({
                    f'stress_{scenario_name}_total_loss': total_loss,
                    f'stress_{scenario_name}_max_daily_loss': max_daily_loss,
                    f'stress_{scenario_name}_recovery_months': recovery_months,
                    f'stress_{scenario_name}_estimated_recovery_days': recovery_days,
                    f'stress_{scenario_name}_description': scenario_params['description']
                })
                
            except Exception as e:
                logger.warning(f"Failed to compute stress scenario {scenario_name}: {str(e)}")
                stress_metrics.update({
                    f'stress_{scenario_name}_total_loss': np.nan,
                    f'stress_{scenario_name}_max_daily_loss': np.nan,
                    f'stress_{scenario_name}_recovery_months': np.nan,
                    f'stress_{scenario_name}_estimated_recovery_days': np.nan,
                    f'stress_{scenario_name}_error': str(e)[:100]
                })
        
        # Calculate sensitivity to key factors
        factor_sensitivities = self._calculate_factor_sensitivities(returns_df, portfolio_weights)
        stress_metrics.update(factor_sensitivities)
        
        return stress_metrics
    
    def _apply_stress_scenario(self, returns_df: pd.DataFrame,
                              scenario_params: Dict) -> pd.DataFrame:
        """Apply stress scenario shocks to returns"""
        
        # Create shocked returns
        shocked_returns = returns_df.copy()
        
        # Apply shocks based on asset categories
        for ticker in returns_df.columns:
            metadata = TICKER_TO_METADATA.get(ticker)
            
            if metadata:
                category = metadata.category
                
                if category == "Equity":
                    shock = scenario_params.get('equity_shock', 0)
                elif category == "Fixed Income":
                    shock = scenario_params.get('bond_shock', 0)
                elif category == "Commodity":
                    shock = scenario_params.get('commodity_shock', 0)
                elif metadata.is_crypto:
                    # Crypto typically has higher beta to equity shocks
                    shock = scenario_params.get('equity_shock', 0) * 1.5
                elif metadata.is_etf:
                    # ETF shocks depend on underlying
                    if 'Equity' in metadata.sector or 'Stock' in metadata.name:
                        shock = scenario_params.get('equity_shock', 0)
                    elif 'Bond' in metadata.sector or 'Treasury' in metadata.name:
                        shock = scenario_params.get('bond_shock', 0)
                    else:
                        shock = scenario_params.get('equity_shock', 0) * 0.5
                else:
                    shock = 0
            else:
                # Default to equity shock for unknown assets
                shock = scenario_params.get('equity_shock', 0)
            
            # Apply shock (convert annual shock to daily)
            daily_shock = shock / self.config.TRADING_DAYS
            shocked_returns[ticker] = returns_df[ticker] + daily_shock
        
        return shocked_returns
    
    def _calculate_factor_sensitivities(self, returns_df: pd.DataFrame,
                                       portfolio_weights: np.ndarray) -> Dict:
        """Calculate portfolio sensitivity to various risk factors"""
        
        factor_sensitivities = {}
        
        # Market factor (using equal-weighted portfolio as proxy)
        market_returns = returns_df.mean(axis=1)
        portfolio_returns = (returns_df * portfolio_weights).sum(axis=1)
        
        # Beta calculation
        covariance = np.cov(portfolio_returns, market_returns)[0, 1]
        market_variance = market_returns.var()
        beta = covariance / market_variance if market_variance > 0 else 1.0
        
        # Size factor (small vs large cap)
        # This is simplified - in practice you'd use proper size factor returns
        large_cap_returns = self._get_category_returns(returns_df, ['US_Stocks', 'US_Indices'])
        small_cap_returns = self._get_category_returns(returns_df, ['US_Stocks'])
        
        if not large_cap_returns.empty and not small_cap_returns.empty:
            size_factor = small_cap_returns.mean() - large_cap_returns.mean()
            size_beta = self._calculate_factor_beta(portfolio_returns, size_factor)
        else:
            size_beta = 0
        
        # Value factor (simplified)
        # In practice, you'd use proper value factor returns
        
        # Momentum factor (recent performance)
        momentum_factor = returns_df.iloc[-60:].mean().mean() - returns_df.iloc[:-60].mean().mean()
        momentum_beta = self._calculate_factor_beta(portfolio_returns, momentum_factor)
        
        # Volatility factor
        volatility_factor = returns_df.std().mean()
        
        factor_sensitivities.update({
            'market_beta': beta,
            'size_beta': size_beta,
            'momentum_beta': momentum_beta,
            'volatility_exposure': volatility_factor,
            'correlation_to_market': portfolio_returns.corr(market_returns),
            'r_squared_vs_market': beta**2 * market_variance / (portfolio_returns.var() + 1e-10)
        })
        
        return factor_sensitivities
    
    def _get_category_returns(self, returns_df: pd.DataFrame,
                             categories: List[str]) -> pd.Series:
        """Get returns for assets in specific categories"""
        
        tickers_in_categories = []
        for category in categories:
            if category in CATEGORY_TO_ASSETS:
                tickers_in_categories.extend(CATEGORY_TO_ASSETS[category])
        
        # Filter for tickers that exist in our returns dataframe
        available_tickers = [t for t in tickers_in_categories if t in returns_df.columns]
        
        if not available_tickers:
            return pd.Series()
        
        # Return equal-weighted returns for the category
        category_returns = returns_df[available_tickers].mean(axis=1)
        return category_returns
    
    def _calculate_factor_beta(self, portfolio_returns: pd.Series,
                              factor_returns: Union[float, pd.Series]) -> float:
        """Calculate beta to a factor"""
        
        if isinstance(factor_returns, pd.Series) and len(factor_returns) > 1:
            # Align indices
            common_idx = portfolio_returns.index.intersection(factor_returns.index)
            if len(common_idx) > 10:
                cov = np.cov(portfolio_returns.loc[common_idx], factor_returns.loc[common_idx])[0, 1]
                var = factor_returns.loc[common_idx].var()
                return cov / var if var > 0 else 0
        elif isinstance(factor_returns, (int, float)):
            # For single factor value
            return factor_returns / (portfolio_returns.mean() + 1e-10)
        
        return 0
    
    def _compute_factor_risk_metrics(self, returns_df: pd.DataFrame,
                                    portfolio_weights: np.ndarray,
                                    portfolio_returns: pd.Series) -> Dict:
        """Compute factor-based risk metrics"""
        
        factor_metrics = {}
        
        # Principal Component Analysis for risk factor decomposition
        try:
            if len(returns_df.columns) > 2 and SKLEARN_AVAILABLE:
                # Standardize returns
                scaler = StandardScaler()
                scaled_returns = scaler.fit_transform(returns_df.fillna(0))
                
                # Perform PCA
                pca = PCA(n_components=min(5, len(returns_df.columns)))
                pca_result = pca.fit_transform(scaled_returns)
                
                # Calculate factor exposures
                factor_exposures = {}
                for i in range(min(5, len(returns_df.columns))):
                    explained_variance = pca.explained_variance_ratio_[i]
                    factor_exposures[f'pca_factor_{i+1}_variance'] = explained_variance
                
                factor_metrics.update({
                    **factor_exposures,
                    'total_explained_variance': pca.explained_variance_ratio_.sum(),
                    'num_significant_factors': (pca.explained_variance_ratio_ > 0.1).sum()
                })
        except Exception as e:
            logger.debug(f"PCA analysis failed: {str(e)}")
        
        # Calculate concentration to top positions
        sorted_weights = np.sort(portfolio_weights)[::-1]
        if len(sorted_weights) >= 5:
            top_5_concentration = sorted_weights[:5].sum()
            top_10_concentration = sorted_weights[:min(10, len(sorted_weights))].sum()
        else:
            top_5_concentration = sorted_weights.sum()
            top_10_concentration = sorted_weights.sum()
        
        factor_metrics.update({
            'top_5_concentration': top_5_concentration,
            'top_10_concentration': top_10_concentration,
            'weight_gini_coefficient': self._calculate_gini_coefficient(portfolio_weights)
        })
        
        return factor_metrics
    
    def _calculate_gini_coefficient(self, weights: np.ndarray) -> float:
        """Calculate Gini coefficient for weight distribution"""
        # Sort weights
        sorted_weights = np.sort(weights)
        n = len(sorted_weights)
        
        if n == 0:
            return 0
        
        # Calculate Gini coefficient
        index = np.arange(1, n + 1)
        gini = (np.sum((2 * index - n - 1) * sorted_weights)) / (n * np.sum(sorted_weights))
        
        return gini
    
    def _compute_liquidity_metrics(self, returns_df: pd.DataFrame,
                                  portfolio_weights: np.ndarray) -> Dict:
        """Compute liquidity-related risk metrics"""
        
        liquidity_metrics = {}
        
        # Estimate liquidity using volume data (if available)
        # This is a simplified approach - real implementation would use actual volume data
        
        # Calculate turnover impact (simplified)
        # Assumes each position takes 5 days to liquidate without price impact
        daily_turnover_capacity = 0.20  # 20% per day
        estimated_liquidation_days = np.sum(np.abs(portfolio_weights)) / daily_turnover_capacity
        
        # Calculate concentration-adjusted liquidation time
        # Larger positions take longer to liquidate
        concentration_factor = 1 + 2 * self._calculate_gini_coefficient(portfolio_weights)
        adjusted_liquidation_days = estimated_liquidation_days * concentration_factor
        
        liquidity_metrics.update({
            'estimated_liquidation_days': estimated_liquidation_days,
            'adjusted_liquidation_days': adjusted_liquidation_days,
            'liquidity_score': 1 / (adjusted_liquidation_days + 1),
            'daily_turnover_capacity': daily_turnover_capacity,
            'illiquid_assets_pct': (portfolio_weights > self.config.LIQUIDITY_THRESHOLD).sum() / len(portfolio_weights)
        })
        
        return liquidity_metrics
    
    def _compute_concentration_metrics(self, portfolio_weights: np.ndarray) -> Dict:
        """Compute portfolio concentration metrics"""
        
        # Calculate Herfindahl-Hirschman Index (HHI)
        hhi = np.sum(portfolio_weights ** 2)
        
        # Calculate effective number of positions
        effective_n = 1 / hhi if hhi > 0 else 0
        
        # Calculate entropy
        positive_weights = portfolio_weights[portfolio_weights > 0]
        if len(positive_weights) > 0:
            entropy = -np.sum(positive_weights * np.log(positive_weights + 1e-10))
        else:
            entropy = 0
        
        # Calculate diversification ratio
        # This would need covariance matrix in practice
        diversification_ratio = effective_n / len(portfolio_weights) if len(portfolio_weights) > 0 else 0
        
        # Calculate exposure to largest position
        largest_weight = np.max(portfolio_weights) if len(portfolio_weights) > 0 else 0
        second_largest = np.partition(portfolio_weights, -2)[-2] if len(portfolio_weights) > 1 else 0
        
        concentration_metrics = {
            'herfindahl_index': hhi,
            'effective_number_positions': effective_n,
            'portfolio_entropy': entropy,
            'diversification_ratio': diversification_ratio,
            'largest_position_weight': largest_weight,
            'second_largest_position_weight': second_largest,
            'largest_to_second_ratio': largest_weight / (second_largest + 1e-10),
            'positions_above_5pct': (portfolio_weights > 0.05).sum(),
            'positions_above_10pct': (portfolio_weights > 0.10).sum(),
            'zero_weight_positions': (portfolio_weights == 0).sum()
        }
        
        return concentration_metrics
    
    def _store_risk_metrics(self, metrics: Dict):
        """Store risk metrics in history"""
        
        # Create a simplified version for history
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'annual_return': metrics.get('annual_return', 0),
            'annual_volatility': metrics.get('annual_volatility', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'var_95_historical': metrics.get('var_95_historical', 0),
            'cvar_95_historical': metrics.get('cvar_95_historical', 0),
            'herfindahl_index': metrics.get('herfindahl_index', 0),
            'effective_number_positions': metrics.get('effective_number_positions', 0),
            'market_beta': metrics.get('market_beta', 0)
        }
        
        self.risk_metrics_history.append(history_entry)
        
        # Keep only last 1000 entries
        if len(self.risk_metrics_history) > 1000:
            self.risk_metrics_history = self.risk_metrics_history[-1000:]
    
    def get_risk_metrics_history(self) -> pd.DataFrame:
        """Get risk metrics history as DataFrame"""
        return pd.DataFrame(self.risk_metrics_history)
    
    def generate_risk_report(self, metrics: Dict) -> str:
        """Generate comprehensive risk report"""
        
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append("INSTITUTIONAL PORTFOLIO RISK REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Number of Assets: {metrics.get('n_assets', 0)}")
        report_lines.append(f"Risk-Free Rate: {metrics.get('risk_free_rate', 0):.2%}")
        report_lines.append("")
        
        # Performance Metrics
        report_lines.append("PERFORMANCE METRICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Annual Return: {metrics.get('annual_return', 0):.2%}")
        report_lines.append(f"Annual Volatility: {metrics.get('annual_volatility', 0):.2%}")
        report_lines.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        report_lines.append(f"Sortino Ratio: {metrics.get('sortino_ratio', 0):.2f}")
        report_lines.append(f"Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        report_lines.append(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
        report_lines.append("")
        
        # Risk Metrics
        report_lines.append("RISK METRICS")
        report_lines.append("-" * 40)
        report_lines.append(f"VaR (95%, Historical): {metrics.get('var_95_historical', 0):.2%}")
        report_lines.append(f"CVaR (95%, Historical): {metrics.get('cvar_95_historical', 0):.2%}")
        report_lines.append(f"VaR (99%, Historical): {metrics.get('var_99_historical', 0):.2%}")
        report_lines.append(f"Worst Daily Loss: {metrics.get('worst_daily_loss', 0):.2%}")
        report_lines.append(f"Downside Volatility: {metrics.get('downside_volatility', 0):.2%}")
        report_lines.append("")
        
        # Portfolio Characteristics
        report_lines.append("PORTFOLIO CHARACTERISTICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Herfindahl Index: {metrics.get('herfindahl_index', 0):.4f}")
        report_lines.append(f"Effective Number of Positions: {metrics.get('effective_number_positions', 0):.1f}")
        report_lines.append(f"Largest Position: {metrics.get('largest_position_weight', 0):.2%}")
        report_lines.append(f"Top 5 Concentration: {metrics.get('top_5_concentration', 0):.2%}")
        report_lines.append(f"Market Beta: {metrics.get('market_beta', 0):.2f}")
        report_lines.append("")
        
        # Stress Testing Summary
        report_lines.append("STRESS TESTING SUMMARY")
        report_lines.append("-" * 40)
        
        stress_scenarios = []
        for key, value in metrics.items():
            if key.startswith('stress_') and key.endswith('_total_loss'):
                scenario_name = key.replace('stress_', '').replace('_total_loss', '')
                total_loss = value
                max_loss_key = f'stress_{scenario_name}_max_daily_loss'
                max_loss = metrics.get(max_loss_key, 0)
                
                if not pd.isna(total_loss) and not pd.isna(max_loss):
                    stress_scenarios.append((scenario_name, total_loss, max_loss))
        
        for scenario_name, total_loss, max_loss in stress_scenarios[:3]:  # Top 3 scenarios
            report_lines.append(f"{scenario_name.replace('_', ' ').title()}:")
            report_lines.append(f"  Total Loss: {total_loss:.2%}")
            report_lines.append(f"  Max Daily Loss: {max_loss:.2%}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)

# =============================================================
# MAIN INSTITUTIONAL PORTFOLIO APPLICATION
# =============================================================
class InstitutionalPortfolioApplication:
    """Main application class for institutional portfolio management"""
    
    def __init__(self):
        # Initialize components
        self.config = InstitutionalConfig()
        self.data_loader = EnhancedInstitutionalDataLoader()
        self.optimizer = EnhancedInstitutionalPortfolioOptimizer()
        self.risk_analytics = EnhancedInstitutionalRiskAnalytics()
        
        # Session state management
        self._initialize_session_state()
        
        # Page configuration
        st.set_page_config(
            page_title="🏛️ Institutional Apollo / ENIGMA – Quant Terminal v5.0",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        self._inject_custom_css()
    
    def _initialize_session_state(self):
        """Initialize session state variables"""
        
        if 'initialized' not in st.session_state:
            st.session_state.initialized = True
            
            # Data management
            st.session_state.prices_data = None
            st.session_state.returns_data = None
            st.session_state.selected_tickers = []
            st.session_state.date_range = {
                'start': (datetime.now() - timedelta(days=365*3)).strftime('%Y-%m-%d'),
                'end': datetime.now().strftime('%Y-%m-%d')
            }
            
            # Portfolio optimization
            st.session_state.optimization_results = None
            st.session_state.current_strategy = "Minimum Volatility"
            st.session_state.risk_free_rate = self.config.DEFAULT_RF_RATE
            
            # Risk analytics
            st.session_state.risk_metrics = None
            
            # User preferences
            st.session_state.show_advanced = False
            st.session_state.data_source = 'auto'
            st.session_state.use_cache = True
            
            # Performance tracking
            st.session_state.performance_history = []
            
            # Portfolio constraints
            st.session_state.constraints = {
                'min_weight': 0.0,
                'max_weight': 1.0,
                'allow_short': False,
                'max_concentration': self.config.MAX_CONCENTRATION,
                'min_diversification': self.config.MIN_DIVERSIFICATION,
                'reg_gamma': self.config.REGULARIZATION_GAMMA
            }
    
    def _inject_custom_css(self):
        """Inject custom CSS for better styling"""
        
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
            padding-bottom: 0.5rem;
            border-bottom: 3px solid #1f77b4;
        }
        
        .section-header {
            font-size: 1.5rem;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 2rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #3498db;
        }
        
        .metric-card {
            background-color: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 5px solid #3498db;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .warning-card {
            background-color: #fff3cd;
            border-color: #ffeaa7;
            border-left: 5px solid #fdcb6e;
        }
        
        .success-card {
            background-color: #d1ecf1;
            border-color: #bee5eb;
            border-left: 5px solid #17a2b8;
        }
        
        .stButton > button {
            width: 100%;
            margin-top: 10px;
            margin-bottom: 10px;
        }
        
        .stDownloadButton > button {
            background-color: #28a745;
            color: white;
        }
        
        .stDownloadButton > button:hover {
            background-color: #218838;
        }
        
        /* Dataframe styling */
        .dataframe {
            font-size: 0.9em;
        }
        
        /* Plotly chart sizing */
        .js-plotly-plot {
            margin: auto;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Custom tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def run(self):
        """Main application entry point"""
        
        try:
            # Application header
            self._render_header()
            
            # Sidebar for configuration
            with st.sidebar:
                self._render_sidebar()
            
            # Main content area with tabs
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📊 Dashboard",
                "⚙️ Optimization",
                "📈 Risk Analytics",
                "📋 Portfolio Analysis",
                "🔧 Advanced Tools"
            ])
            
            with tab1:
                self._render_dashboard()
            
            with tab2:
                self._render_optimization()
            
            with tab3:
                self._render_risk_analytics()
            
            with tab4:
                self._render_portfolio_analysis()
            
            with tab5:
                self._render_advanced_tools()
            
            # Footer
            self._render_footer()
            
        except Exception as e:
            self._handle_error(e)
    
    def _render_header(self):
        """Render application header"""
        
        st.markdown('<h1 class="main-header">🏛️ Institutional Apollo / ENIGMA – Quant Terminal v5.0</h1>', 
                   unsafe_allow_html=True)
        
        # Subheader with status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="PyPortfolioOpt",
                value="Available" if PYPFOPT_AVAILABLE else "Not Available",
                delta=PYPFOPT_VERSION if PYPFOPT_AVAILABLE else None
            )
        
        with col2:
            st.metric(
                label="scikit-learn",
                value="Available" if SKLEARN_AVAILABLE else "Not Available"
            )
        
        with col3:
            assets_loaded = len(st.session_state.selected_tickers) if st.session_state.selected_tickers else 0
            st.metric(
                label="Assets Loaded",
                value=assets_loaded,
                delta="Ready" if assets_loaded > 0 else "No Data"
            )
        
        with col4:
            cache_stats = global_cache.stats()
            hit_rate = cache_stats['hit_rate']
            st.metric(
                label="Cache Efficiency",
                value=f"{hit_rate:.1%}",
                delta=f"{cache_stats['hits']} hits"
            )
        
        st.markdown("---")
    
    def _render_sidebar(self):
        """Render sidebar with configuration options"""
        
        st.sidebar.markdown("## 🔧 Configuration")
        
        # Data source selection
        st.sidebar.markdown("### 📊 Data Source")
        data_source = st.sidebar.selectbox(
            "Data Source",
            options=['auto', 'yfinance', 'demo'],
            index=0,
            help="Select data source. 'auto' will try multiple sources."
        )
        
        # Date range selection
        st.sidebar.markdown("### 📅 Date Range")
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.strptime(st.session_state.date_range['start'], '%Y-%m-%d'),
                max_value=datetime.now()
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.strptime(st.session_state.date_range['end'], '%Y-%m-%d'),
                max_value=datetime.now(),
                min_value=start_date
            )
        
        # Asset selection
        st.sidebar.markdown("### 🏦 Asset Selection")
        
        # Category selection
        selected_categories = st.sidebar.multiselect(
            "Select Asset Categories",
            options=list(CATEGORY_TO_ASSETS.keys()),
            default=["US_Indices", "Bonds", "US_Stocks"],
            help="Select asset categories to include in the universe"
        )
        
        # Get tickers from selected categories
        tickers_from_categories = []
        for category in selected_categories:
            if category in CATEGORY_TO_ASSETS:
                tickers_from_categories.extend(CATEGORY_TO_ASSETS[category])
        
        # Manual ticker input
        st.sidebar.markdown("#### Manual Ticker Entry")
        manual_tickers = st.sidebar.text_area(
            "Additional Tickers (comma-separated)",
            value="",
            help="Enter additional tickers not in the categories above"
        )
        
        # Parse manual tickers
        if manual_tickers:
            manual_ticker_list = [t.strip().upper() for t in manual_tickers.split(',') if t.strip()]
            tickers_from_categories.extend(manual_ticker_list)
        
        # Remove duplicates and limit
        all_tickers = list(dict.fromkeys(tickers_from_categories))
        all_tickers = all_tickers[:self.config.MAX_ASSETS]
        
        # Ticker selection
        selected_tickers = st.sidebar.multiselect(
            "Select Assets for Portfolio",
            options=all_tickers,
            default=st.session_state.selected_tickers if st.session_state.selected_tickers else all_tickers[:10],
            help="Select assets to include in portfolio optimization"
        )
        
        # Data loading button
        st.sidebar.markdown("---")
        if st.sidebar.button("📥 Load Data", type="primary", use_container_width=True):
            with st.spinner("Loading data..."):
                self._load_data(selected_tickers, start_date, end_date, data_source)
        
        # Clear cache button
        if st.sidebar.button("🗑️ Clear Cache", use_container_width=True):
            global_cache.clear()
            st.success("Cache cleared!")
            st.rerun()
        
        # Advanced settings
        with st.sidebar.expander("⚙️ Advanced Settings"):
            st.session_state.risk_free_rate = st.number_input(
                "Risk-Free Rate (%)",
                min_value=0.0,
                max_value=20.0,
                value=self.config.DEFAULT_RF_RATE * 100,
                step=0.1
            ) / 100
            
            st.session_state.use_cache = st.checkbox("Enable Caching", value=True)
            st.session_state.show_advanced = st.checkbox("Show Advanced Options", value=False)
            
            # Constraints
            st.markdown("#### Portfolio Constraints")
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.constraints['min_weight'] = st.number_input(
                    "Min Weight %",
                    min_value=0.0,
                    max_value=100.0,
                    value=0.0,
                    step=0.1
                ) / 100
            
            with col2:
                st.session_state.constraints['max_weight'] = st.number_input(
                    "Max Weight %",
                    min_value=0.0,
                    max_value=100.0,
                    value=100.0,
                    step=0.1
                ) / 100
            
            st.session_state.constraints['allow_short'] = st.checkbox("Allow Short Positions", value=False)
            
            if st.session_state.constraints['allow_short']:
                st.session_state.constraints['max_leverage'] = st.slider(
                    "Max Leverage",
                    min_value=1.0,
                    max_value=5.0,
                    value=2.0,
                    step=0.1
                )
        
        # Update session state
        st.session_state.selected_tickers = selected_tickers
        st.session_state.date_range = {
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d')
        }
        
        # Display loaded data info
        if st.session_state.prices_data is not None:
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 📈 Loaded Data Info")
            st.sidebar.write(f"**Assets:** {len(st.session_state.prices_data.columns)}")
            st.sidebar.write(f"**Period:** {len(st.session_state.prices_data)} days")
            st.sidebar.write(f"**Start:** {st.session_state.prices_data.index[0].strftime('%Y-%m-%d')}")
            st.sidebar.write(f"**End:** {st.session_state.prices_data.index[-1].strftime('%Y-%m-%d')}")
            
            # Data quality indicator
            missing_pct = st.session_state.prices_data.isna().sum().sum() / st.session_state.prices_data.size
            if missing_pct < 0.01:
                st.sidebar.success("✅ Data Quality: Excellent")
            elif missing_pct < 0.05:
                st.sidebar.info("ℹ️ Data Quality: Good")
            else:
                st.sidebar.warning("⚠️ Data Quality: Check missing values")
    
    def _load_data(self, tickers: List[str], start_date: datetime, end_date: datetime, 
                  data_source: str = 'auto'):
        """Load data for selected tickers"""
        
        if not tickers:
            st.error("Please select at least one ticker")
            return
        
        try:
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Update data loader configuration
            self.data_loader.cache_enabled = st.session_state.use_cache
            
            # Load data
            status_text.text(f"Loading data for {len(tickers)} assets...")
            
            prices_df = self.data_loader.load_batch(
                tickers=tickers,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d'),
                use_parallel=True,
                source=data_source
            )
            
            progress_bar.progress(50)
            status_text.text("Processing returns data...")
            
            if prices_df.empty:
                st.error("No data loaded. Please check ticker symbols and date range.")
                return
            
            # Calculate returns
            returns_df = prices_df.pct_change().dropna()
            
            # Store in session state
            st.session_state.prices_data = prices_df
            st.session_state.returns_data = returns_df
            
            progress_bar.progress(100)
            status_text.text("Data loaded successfully!")
            
            # Show data quality report
            with st.expander("📋 Data Quality Report"):
                quality_report = self.data_loader.get_data_quality_report()
                if not quality_report.empty:
                    st.dataframe(quality_report, use_container_width=True)
                else:
                    st.info("No quality report available.")
            
            # Show success message
            st.success(f"✅ Successfully loaded {len(prices_df.columns)} assets with {len(prices_df)} data points")
            
            # Auto-rerun to update UI
            st.rerun()
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            logger.error(f"Data loading error: {str(e)}", exc_info=True)
    
    def _render_dashboard(self):
        """Render main dashboard"""
        
        st.markdown('<h2 class="section-header">📊 Portfolio Dashboard</h2>', 
                   unsafe_allow_html=True)
        
        if st.session_state.prices_data is None:
            st.info("👈 Please load data from the sidebar to begin analysis.")
            return
        
        # Display key metrics
        self._render_key_metrics()
        
        # Price and returns charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Price Evolution")
            self._plot_price_evolution()
        
        with col2:
            st.markdown("#### 📊 Daily Returns Distribution")
            self._plot_returns_distribution()
        
        # Correlation matrix
        st.markdown("#### 🔗 Correlation Matrix")
        self._plot_correlation_matrix()
        
        # Asset performance summary
        st.markdown("#### 📋 Asset Performance Summary")
        self._render_asset_performance_table()
    
    def _render_key_metrics(self):
        """Render key portfolio metrics"""
        
        returns_df = st.session_state.returns_data
        
        # Calculate basic statistics
        annual_returns = returns_df.mean() * self.config.TRADING_DAYS
        annual_volatility = returns_df.std() * np.sqrt(self.config.TRADING_DAYS)
        sharpe_ratios = (annual_returns - st.session_state.risk_free_rate) / annual_volatility
        
        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_return = annual_returns.mean()
            st.metric("Avg Annual Return", f"{avg_return:.2%}")
        
        with col2:
            avg_vol = annual_volatility.mean()
            st.metric("Avg Annual Volatility", f"{avg_vol:.2%}")
        
        with col3:
            avg_sharpe = sharpe_ratios.mean()
            st.metric("Avg Sharpe Ratio", f"{avg_sharpe:.2f}")
        
        with col4:
            correlation = returns_df.corr().values[np.triu_indices_from(returns_df.corr().values, k=1)].mean()
            st.metric("Avg Correlation", f"{correlation:.2f}")
    
    def _plot_price_evolution(self):
        """Plot price evolution of selected assets"""
        
        prices_df = st.session_state.prices_data
        
        # Normalize prices to 100 at start for comparison
        normalized_prices = prices_df / prices_df.iloc[0] * 100
        
        fig = go.Figure()
        
        for column in normalized_prices.columns:
            fig.add_trace(go.Scatter(
                x=normalized_prices.index,
                y=normalized_prices[column],
                mode='lines',
                name=column,
                line=dict(width=1.5),
                hovertemplate='%{x|%Y-%m-%d}<br>%{y:.1f}%<extra></extra>'
            ))
        
        fig.update_layout(
            title="Normalized Price Evolution (Base=100)",
            xaxis_title="Date",
            yaxis_title="Normalized Price",
            height=400,
            showlegend=True,
            hovermode='x unified',
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_returns_distribution(self):
        """Plot distribution of returns"""
        
        returns_df = st.session_state.returns_data
        
        fig = go.Figure()
        
        # Add histogram for each asset
        for column in returns_df.columns[:5]:  # Limit to first 5 for clarity
            fig.add_trace(go.Histogram(
                x=returns_df[column],
                name=column,
                opacity=0.7,
                nbinsx=50
            ))
        
        fig.update_layout(
            title="Daily Returns Distribution",
            xaxis_title="Daily Return",
            yaxis_title="Frequency",
            height=400,
            barmode='overlay',
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_correlation_matrix(self):
        """Plot correlation matrix heatmap"""
        
        returns_df = st.session_state.returns_data
        
        # Calculate correlation matrix
        corr_matrix = returns_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate='%{text}',
            hoverongaps=False,
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Asset Correlation Matrix",
            height=500,
            xaxis_title="Assets",
            yaxis_title="Assets",
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_asset_performance_table(self):
        """Render asset performance table"""
        
        returns_df = st.session_state.returns_data
        
        # Calculate performance metrics
        performance_data = []
        
        for ticker in returns_df.columns:
            asset_returns = returns_df[ticker]
            
            # Basic metrics
            annual_return = asset_returns.mean() * self.config.TRADING_DAYS
            annual_vol = asset_returns.std() * np.sqrt(self.config.TRADING_DAYS)
            sharpe_ratio = (annual_return - st.session_state.risk_free_rate) / (annual_vol + 1e-10)
            
            # Downside metrics
            downside_returns = asset_returns[asset_returns < 0]
            downside_vol = downside_returns.std() * np.sqrt(self.config.TRADING_DAYS) if len(downside_returns) > 0 else 0
            sortino_ratio = (annual_return - st.session_state.risk_free_rate) / (downside_vol + 1e-10)
            
            # Maximum drawdown
            cumulative = (1 + asset_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()
            
            # Get metadata
            metadata = TICKER_TO_METADATA.get(ticker, {})
            
            performance_data.append({
                'Ticker': ticker,
                'Name': metadata.name if hasattr(metadata, 'name') else ticker,
                'Category': metadata.category if hasattr(metadata, 'category') else 'Unknown',
                'Annual Return': annual_return,
                'Annual Volatility': annual_vol,
                'Sharpe Ratio': sharpe_ratio,
                'Sortino Ratio': sortino_ratio,
                'Max Drawdown': max_dd,
                'Last Price': st.session_state.prices_data[ticker].iloc[-1] if st.session_state.prices_data is not None else None
            })
        
        # Create DataFrame
        perf_df = pd.DataFrame(performance_data)
        
        # Format numeric columns
        format_dict = {
            'Annual Return': '{:.2%}',
            'Annual Volatility': '{:.2%}',
            'Sharpe Ratio': '{:.2f}',
            'Sortino Ratio': '{:.2f}',
            'Max Drawdown': '{:.2%}',
            'Last Price': '{:.2f}'
        }
        
        styled_df = perf_df.style.format(format_dict)
        
        # Display with conditional formatting
        st.dataframe(
            styled_df,
            use_container_width=True,
            height=400
        )
        
        # Download button
        csv = perf_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Performance Data",
            data=csv,
            file_name="asset_performance.csv",
            mime="text/csv"
        )
    
    def _render_optimization(self):
        """Render optimization interface"""
        
        st.markdown('<h2 class="section-header">⚙️ Portfolio Optimization</h2>', 
                   unsafe_allow_html=True)
        
        if st.session_state.returns_data is None:
            st.info("👈 Please load data from the sidebar to begin optimization.")
            return
        
        # Optimization controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            strategy = st.selectbox(
                "Optimization Strategy",
                options=list(self.optimizer.optimization_strategies.keys()),
                index=list(self.optimizer.optimization_strategies.keys()).index(
                    st.session_state.current_strategy
                ) if st.session_state.current_strategy in self.optimizer.optimization_strategies else 0
            )
        
        with col2:
            optimization_button = st.button(
                "🚀 Run Optimization",
                type="primary",
                use_container_width=True
            )
        
        with col3:
            if st.session_state.optimization_results:
                clear_button = st.button(
                    "🗑️ Clear Results",
                    use_container_width=True
                )
                if clear_button:
                    st.session_state.optimization_results = None
                    st.rerun()
        
        # Run optimization
        if optimization_button:
            with st.spinner("Running optimization..."):
                try:
                    results = self.optimizer.optimize_with_constraints(
                        returns_df=st.session_state.returns_data,
                        strategy=strategy,
                        constraints=st.session_state.constraints,
                        risk_free_rate=st.session_state.risk_free_rate
                    )
                    
                    st.session_state.optimization_results = results
                    st.session_state.current_strategy = strategy
                    
                    st.success("✅ Optimization completed successfully!")
                    
                except Exception as e:
                    st.error(f"Optimization failed: {str(e)}")
                    logger.error(f"Optimization error: {str(e)}", exc_info=True)
        
        # Display optimization results
        if st.session_state.optimization_results:
            self._render_optimization_results()
    
    def _render_optimization_results(self):
        """Render optimization results"""
        
        results = st.session_state.optimization_results
        
        # Performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Expected Return",
                f"{results['expected_return']:.2%}",
                delta="Annual"
            )
        
        with col2:
            st.metric(
                "Expected Risk",
                f"{results['expected_risk']:.2%}",
                delta="Annual Volatility"
            )
        
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{results['sharpe_ratio']:.2f}",
                delta=f"RF: {st.session_state.risk_free_rate:.2%}"
            )
        
        with col4:
            if 'max_drawdown' in results:
                st.metric(
                    "Max Drawdown",
                    f"{results['max_drawdown']:.2%}",
                    delta="Estimated"
                )
        
        # Portfolio weights visualization
        st.markdown("#### 📊 Portfolio Allocation")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._plot_portfolio_weights(results)
        
        with col2:
            # Weight summary
            weights_df = pd.DataFrame({
                'Asset': list(results['cleaned_weights'].keys()),
                'Weight': list(results['cleaned_weights'].values())
            }).sort_values('Weight', ascending=False)
            
            # Format weights
            weights_df['Weight'] = weights_df['Weight'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(
                weights_df,
                use_container_width=True,
                height=400
            )
        
        # Portfolio performance metrics
        st.markdown("#### 📈 Detailed Performance Metrics")
        
        metrics_cols = st.columns(3)
        
        performance_metrics = [
            ("Method", results.get('method', 'N/A')),
            ("Sortino Ratio", f"{results.get('sortino_ratio', 0):.2f}"),
            ("Information Ratio", f"{results.get('information_ratio', 0):.2f}"),
            ("Beta to Market", f"{results.get('beta', 0):.2f}"),
            ("Tracking Error", f"{results.get('tracking_error', 0):.2%}"),
            ("Diversification Ratio", f"{results.get('diversification_ratio', 0):.2f}"),
            ("Effective Number of Assets", f"{results.get('effective_n', 0):.1f}"),
            ("Max Concentration", f"{results.get('max_concentration', 0):.2%}"),
            ("Turnover", f"{results.get('turnover', 0):.2%}")
        ]
        
        for i, (name, value) in enumerate(performance_metrics):
            with metrics_cols[i % 3]:
                st.metric(name, value)
        
        # Download buttons
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            # Export weights
            weights_data = pd.DataFrame({
                'Asset': list(results['cleaned_weights'].keys()),
                'Weight': list(results['cleaned_weights'].values()),
                'Category': [TICKER_TO_METADATA.get(t, AssetMetadata(t, t, 'Unknown', 'Unknown', 'USD')).category 
                           for t in results['cleaned_weights'].keys()]
            })
            
            csv_weights = weights_data.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Portfolio Weights",
                data=csv_weights,
                file_name="portfolio_weights.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export optimization report
            report_text = self._generate_optimization_report(results)
            st.download_button(
                label="📄 Download Optimization Report",
                data=report_text,
                file_name="optimization_report.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Efficient frontier plot (if available)
        if st.session_state.show_advanced:
            st.markdown("#### 📈 Efficient Frontier")
            try:
                self._plot_efficient_frontier()
            except Exception as e:
                st.warning(f"Could not plot efficient frontier: {str(e)}")
    
    def _plot_portfolio_weights(self, results: Dict):
        """Plot portfolio weights"""
        
        weights = results['cleaned_weights']
        
        # Filter out zero weights
        non_zero_weights = {k: v for k, v in weights.items() if v > 0.001}
        
        if not non_zero_weights:
            st.info("No significant weights to display.")
            return
        
        # Create pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(non_zero_weights.keys()),
            values=list(non_zero_weights.values()),
            hole=0.4,
            textinfo='label+percent',
            hoverinfo='label+value+percent',
            textposition='inside',
            marker=dict(colors=px.colors.qualitative.Plotly)
        )])
        
        fig.update_layout(
            title=f"Portfolio Allocation - {results.get('method', 'Optimized Portfolio')}",
            height=500,
            showlegend=False,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _generate_optimization_report(self, results: Dict) -> str:
        """Generate optimization report"""
        
        report_lines = []
        
        report_lines.append("=" * 80)
        report_lines.append("PORTFOLIO OPTIMIZATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Optimization Method: {results.get('method', 'N/A')}")
        report_lines.append(f"Risk-Free Rate: {st.session_state.risk_free_rate:.2%}")
        report_lines.append("")
        
        # Performance metrics
        report_lines.append("PERFORMANCE METRICS")
        report_lines.append("-" * 40)
        report_lines.append(f"Expected Annual Return: {results.get('expected_return', 0):.2%}")
        report_lines.append(f"Expected Annual Volatility: {results.get('expected_risk', 0):.2%}")
        report_lines.append(f"Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
        report_lines.append(f"Sortino Ratio: {results.get('sortino_ratio', 0):.2f}")
        report_lines.append(f"Maximum Drawdown: {results.get('max_drawdown', 0):.2%}")
        report_lines.append(f"Information Ratio: {results.get('information_ratio', 0):.2f}")
        report_lines.append("")
        
        # Portfolio weights
        report_lines.append("PORTFOLIO ALLOCATION")
        report_lines.append("-" * 40)
        
        weights = results.get('cleaned_weights', {})
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        for ticker, weight in sorted_weights:
            if weight > 0.001:  # Only show weights > 0.1%
                metadata = TICKER_TO_METADATA.get(ticker)
                asset_name = metadata.name if metadata else ticker
                report_lines.append(f"{ticker:10} {asset_name:30} {weight:>8.2%}")
        
        report_lines.append("")
        report_lines.append(f"Total Assets: {len(weights)}")
        report_lines.append(f"Non-zero Positions: {len([w for w in weights.values() if w > 0.001])}")
        report_lines.append(f"Effective N: {results.get('effective_n', 0):.1f}")
        report_lines.append("")
        
        # Constraints and parameters
        report_lines.append("OPTIMIZATION PARAMETERS")
        report_lines.append("-" * 40)
        for key, value in st.session_state.constraints.items():
            if value is not None:
                report_lines.append(f"{key}: {value}")
        
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def _plot_efficient_frontier(self):
        """Plot efficient frontier"""
        
        try:
            if not PYPFOPT_AVAILABLE:
                st.info("PyPortfolioOpt not available for efficient frontier calculation.")
                return
            
            returns_df = st.session_state.returns_data
            
            # Calculate efficient frontier
            mu = expected_returns.mean_historical_return(returns_df)
            S = risk_models.sample_cov(returns_df)
            
            ef = EfficientFrontier(mu, S)
            
            # Generate points on efficient frontier
            ef_max_sharpe = ef.max_sharpe()
            ret_sharpe, risk_sharpe, _ = ef.portfolio_performance()
            
            # Get min volatility portfolio
            ef_min_vol = EfficientFrontier(mu, S)
            ef_min_vol.min_volatility()
            ret_min_vol, risk_min_vol, _ = ef_min_vol.portfolio_performance()
            
            # Generate frontier
            ef_frontier = EfficientFrontier(mu, S)
            ef_frontier.efficient_risk(target_risk=risk_sharpe)
            ret_frontier, risk_frontier, _ = ef_frontier.portfolio_performance()
            
            # Plot
            fig = go.Figure()
            
            # Individual assets
            asset_risks = np.sqrt(np.diag(S * self.config.TRADING_DAYS))
            asset_returns = mu.values * self.config.TRADING_DAYS
            
            fig.add_trace(go.Scatter(
                x=asset_risks,
                y=asset_returns,
                mode='markers+text',
                name='Assets',
                marker=dict(size=10, color='blue'),
                text=returns_df.columns,
                textposition="top center",
                hovertemplate='<b>%{text}</b><br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
            ))
            
            # Current portfolio
            if st.session_state.optimization_results:
                current_risk = st.session_state.optimization_results['expected_risk']
                current_return = st.session_state.optimization_results['expected_return']
                
                fig.add_trace(go.Scatter(
                    x=[current_risk],
                    y=[current_return],
                    mode='markers',
                    name='Current Portfolio',
                    marker=dict(size=15, color='red', symbol='star'),
                    hovertemplate='<b>Current Portfolio</b><br>Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
                ))
            
            # Capital Market Line
            x_range = np.linspace(0, max(asset_risks) * 1.5, 50)
            cml_returns = st.session_state.risk_free_rate + (ret_sharpe - st.session_state.risk_free_rate) / risk_sharpe * x_range
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=cml_returns,
                mode='lines',
                name='Capital Market Line',
                line=dict(dash='dash', color='gray'),
                hovertemplate='Risk: %{x:.2%}<br>Return: %{y:.2%}<extra></extra>'
            ))
            
            fig.update_layout(
                title="Efficient Frontier",
                xaxis_title="Annual Risk (Volatility)",
                yaxis_title="Annual Return",
                height=500,
                template="plotly_white",
                hovermode='closest'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.warning(f"Could not generate efficient frontier: {str(e)}")
    
    def _render_risk_analytics(self):
        """Render risk analytics interface"""
        
        st.markdown('<h2 class="section-header">📈 Risk Analytics</h2>', 
                   unsafe_allow_html=True)
        
        if st.session_state.returns_data is None:
            st.info("👈 Please load data from the sidebar to begin risk analysis.")
            return
        
        if st.session_state.optimization_results is None:
            st.info("⚠️ Please run portfolio optimization first to analyze portfolio risk.")
            return
        
        # Risk analysis controls
        col1, col2 = st.columns([1, 3])
        
        with col1:
            compute_risk = st.button(
                "🔍 Compute Risk Metrics",
                type="primary",
                use_container_width=True
            )
        
        with col2:
            st.info("Click the button to compute comprehensive risk metrics for the optimized portfolio.")
        
        # Compute risk metrics
        if compute_risk:
            with st.spinner("Computing risk metrics..."):
                try:
                    risk_metrics = self.risk_analytics.compute_comprehensive_risk_metrics(
                        returns_df=st.session_state.returns_data,
                        portfolio_weights=st.session_state.optimization_results['weights'],
                        risk_free_rate=st.session_state.risk_free_rate
                    )
                    
                    st.session_state.risk_metrics = risk_metrics
                    st.success("✅ Risk analysis completed successfully!")
                    
                except Exception as e:
                    st.error(f"Risk analysis failed: {str(e)}")
                    logger.error(f"Risk analysis error: {str(e)}", exc_info=True)
        
        # Display risk metrics
        if st.session_state.risk_metrics:
            self._render_risk_metrics()
    
    def _render_risk_metrics(self):
        """Render comprehensive risk metrics"""
        
        metrics = st.session_state.risk_metrics
        
        # Performance and Risk Overview
        st.markdown("#### 📊 Performance & Risk Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Annual Return",
                f"{metrics.get('annual_return', 0):.2%}",
                delta=f"RF: {st.session_state.risk_free_rate:.2%}"
            )
        
        with col2:
            st.metric(
                "Sharpe Ratio",
                f"{metrics.get('sharpe_ratio', 0):.2f}",
                delta="Risk-adjusted"
            )
        
        with col3:
            st.metric(
                "Max Drawdown",
                f"{metrics.get('max_drawdown', 0):.2%}",
                delta="Worst loss"
            )
        
        with col4:
            st.metric(
                "Sortino Ratio",
                f"{metrics.get('sortino_ratio', 0):.2f}",
                delta="Downside risk"
            )
        
        # VaR and CVaR metrics
        st.markdown("#### ⚠️ Value at Risk (VaR) Metrics")
        
        var_cols = st.columns(4)
        
        var_metrics = [
            ("VaR (95%)", metrics.get('var_95_historical', 0)),
            ("CVaR (95%)", metrics.get('cvar_95_historical', 0)),
            ("VaR (99%)", metrics.get('var_99_historical', 0)),
            ("Worst Daily", metrics.get('worst_daily_loss', 0))
        ]
        
        for i, (name, value) in enumerate(var_metrics):
            with var_cols[i]:
                st.metric(name, f"{value:.2%}")
        
        # Portfolio characteristics
        st.markdown("#### 🏦 Portfolio Characteristics")
        
        char_cols = st.columns(4)
        
        char_metrics = [
            ("Beta to Market", metrics.get('market_beta', 0)),
            ("Effective N", metrics.get('effective_number_positions', 0)),
            ("HHI", f"{metrics.get('herfindahl_index', 0):.4f}"),
            ("Largest Position", f"{metrics.get('largest_position_weight', 0):.2%}")
        ]
        
        for i, (name, value) in enumerate(char_metrics):
            with char_cols[i]:
                st.metric(name, value)
        
        # Risk visualization
        st.markdown("#### 📈 Risk Visualization")
        
        tab1, tab2, tab3 = st.tabs(["Returns Distribution", "Drawdown Analysis", "Stress Testing"])
        
        with tab1:
            self._plot_risk_returns_distribution(metrics)
        
        with tab2:
            self._plot_drawdown_analysis(metrics)
        
        with tab3:
            self._plot_stress_testing(metrics)
        
        # Detailed metrics table
        with st.expander("📋 Detailed Risk Metrics Table"):
            self._render_detailed_risk_table(metrics)
        
        # Risk report download
        st.markdown("---")
        col1, col2 = st.columns(2)
        
        with col1:
            # Export risk metrics
            risk_df = pd.DataFrame([metrics])
            csv_risk = risk_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Risk Metrics",
                data=csv_risk,
                file_name="risk_metrics.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Generate risk report
            risk_report = self.risk_analytics.generate_risk_report(metrics)
            st.download_button(
                label="📄 Download Risk Report",
                data=risk_report,
                file_name="risk_report.txt",
                mime="text/plain",
                use_container_width=True
            )
    
    def _plot_risk_returns_distribution(self, metrics: Dict):
        """Plot portfolio returns distribution with risk metrics"""
        
        portfolio_returns = metrics.get('portfolio_return_series')
        
        if portfolio_returns is None:
            st.info("No return series available for plotting.")
            return
        
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=portfolio_returns,
            name='Returns Distribution',
            nbinsx=50,
            marker_color='blue',
            opacity=0.7
        ))
        
        # Add VaR lines
        var_95 = metrics.get('var_95_historical', 0)
        var_99 = metrics.get('var_99_historical', 0)
        
        # Add vertical lines for VaR
        fig.add_vline(
            x=var_95,
            line_dash="dash",
            line_color="orange",
            annotation_text=f"95% VaR: {var_95:.2%}",
            annotation_position="top right"
        )
        
        fig.add_vline(
            x=var_99,
            line_dash="dash",
            line_color="red",
            annotation_text=f"99% VaR: {var_99:.2%}",
            annotation_position="top right"
        )
        
        # Add mean line
        mean_return = portfolio_returns.mean()
        fig.add_vline(
            x=mean_return,
            line_dash="solid",
            line_color="green",
            annotation_text=f"Mean: {mean_return:.2%}",
            annotation_position="top left"
        )
        
        fig.update_layout(
            title="Portfolio Returns Distribution with VaR",
            xaxis_title="Daily Return",
            yaxis_title="Frequency",
            height=400,
            showlegend=False,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_drawdown_analysis(self, metrics: Dict):
        """Plot drawdown analysis"""
        
        portfolio_returns = metrics.get('portfolio_return_series')
        
        if portfolio_returns is None:
            st.info("No return series available for plotting.")
            return
        
        # Calculate drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        fig = go.Figure()
        
        # Drawdown plot
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown * 100,
            fill='tozeroy',
            fillcolor='rgba(255, 0, 0, 0.3)',
            line=dict(color='red', width=1),
            name='Drawdown',
            hovertemplate='%{x|%Y-%m-%d}<br>Drawdown: %{y:.2f}%<extra></extra>'
        ))
        
        # Highlight max drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min() * 100
        
        fig.add_trace(go.Scatter(
            x=[max_dd_idx],
            y=[max_dd_value],
            mode='markers+text',
            marker=dict(size=15, color='red', symbol='x'),
            text=[f"Max: {max_dd_value:.2f}%"],
            textposition="top center",
            name='Max Drawdown',
            hovertemplate='Max Drawdown: %{y:.2f}%<br>Date: %{x|%Y-%m-%d}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Portfolio Drawdown Analysis",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=400,
            template="plotly_white",
            hovermode='x unified'
        )
        
        # Add horizontal line at 0
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _plot_stress_testing(self, metrics: Dict):
        """Plot stress testing results"""
        
        # Extract stress test metrics
        stress_metrics = {}
        for key, value in metrics.items():
            if key.startswith('stress_') and key.endswith('_total_loss'):
                scenario_name = key.replace('stress_', '').replace('_total_loss', '')
                stress_metrics[scenario_name] = {
                    'total_loss': value,
                    'max_daily_loss': metrics.get(f'stress_{scenario_name}_max_daily_loss', 0),
                    'description': metrics.get(f'stress_{scenario_name}_description', '')
                }
        
        if not stress_metrics:
            st.info("No stress test results available.")
            return
        
        # Prepare data for plotting
        scenarios = list(stress_metrics.keys())
        total_losses = [stress_metrics[s]['total_loss'] * 100 for s in scenarios]
        max_daily_losses = [stress_metrics[s]['max_daily_loss'] * 100 for s in scenarios]
        
        # Create bar chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=scenarios,
            y=total_losses,
            name='Total Loss',
            marker_color='red',
            text=[f"{loss:.1f}%" for loss in total_losses],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Total Loss: %{y:.2f}%<extra></extra>'
        ))
        
        fig.add_trace(go.Bar(
            x=scenarios,
            y=max_daily_losses,
            name='Max Daily Loss',
            marker_color='orange',
            text=[f"{loss:.1f}%" for loss in max_daily_losses],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Max Daily Loss: %{y:.2f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title="Stress Testing Results",
            xaxis_title="Stress Scenario",
            yaxis_title="Loss (%)",
            height=400,
            barmode='group',
            template="plotly_white",
            xaxis_tickangle=-45
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show scenario descriptions
        with st.expander("📋 Stress Scenario Descriptions"):
            for scenario, data in stress_metrics.items():
                if data['description']:
                    st.write(f"**{scenario.replace('_', ' ').title()}**: {data['description']}")
    
    def _render_detailed_risk_table(self, metrics: Dict):
        """Render detailed risk metrics table"""
        
        # Categorize metrics
        categories = {
            'Performance Metrics': [
                'annual_return', 'annual_volatility', 'sharpe_ratio', 
                'sortino_ratio', 'calmar_ratio', 'omega_ratio'
            ],
            'Downside Risk': [
                'max_drawdown', 'downside_volatility', 'var_95_historical',
                'cvar_95_historical', 'var_99_historical', 'worst_daily_loss'
            ],
            'Portfolio Characteristics': [
                'market_beta', 'effective_number_positions', 'herfindahl_index',
                'largest_position_weight', 'top_5_concentration', 'portfolio_entropy'
            ],
            'Higher Moments': [
                'skewness', 'kurtosis', 'excess_kurtosis', 'tail_ratio'
            ]
        }
        
        # Create DataFrame for each category
        for category, metric_keys in categories.items():
            st.markdown(f"**{category}**")
            
            category_data = []
            for key in metric_keys:
                if key in metrics:
                    value = metrics[key]
                    
                    # Format based on value type
                    if isinstance(value, float):
                        if 'ratio' in key.lower() or 'beta' in key:
                            formatted = f"{value:.2f}"
                        elif 'return' in key.lower() or 'volatility' in key or 'drawdown' in key:
                            formatted = f"{value:.2%}"
                        else:
                            formatted = f"{value:.4f}"
                    else:
                        formatted = str(value)
                    
                    category_data.append({
                        'Metric': key.replace('_', ' ').title(),
                        'Value': formatted
                    })
            
            if category_data:
                df = pd.DataFrame(category_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
        
        # Show all metrics in expander
        with st.expander("🔍 All Risk Metrics"):
            all_metrics_df = pd.DataFrame([
                {'Metric': k, 'Value': str(v)}
                for k, v in metrics.items()
                if not isinstance(v, (pd.Series, pd.DataFrame, np.ndarray))
            ])
            st.dataframe(all_metrics_df, use_container_width=True, height=400)
    
    def _render_portfolio_analysis(self):
        """Render portfolio analysis tools"""
        
        st.markdown('<h2 class="section-header">📋 Portfolio Analysis</h2>', 
                   unsafe_allow_html=True)
        
        if st.session_state.prices_data is None:
            st.info("👈 Please load data from the sidebar to begin analysis.")
            return
        
        # Analysis tools
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Performance Attribution",
            "🔍 Monte Carlo Simulation",
            "📊 Rolling Analysis",
            "📉 Factor Analysis"
        ])
        
        with tab1:
            self._render_performance_attribution()
        
        with tab2:
            self._render_monte_carlo_simulation()
        
        with tab3:
            self._render_rolling_analysis()
        
        with tab4:
            self._render_factor_analysis()
    
    def _render_performance_attribution(self):
        """Render performance attribution analysis"""
        
        if st.session_state.optimization_results is None:
            st.info("Please run portfolio optimization first for performance attribution.")
            return
        
        st.markdown("#### 📈 Performance Attribution Analysis")
        
        # Get portfolio weights and returns
        weights = st.session_state.optimization_results['weights']
        returns_df = st.session_state.returns_data
        tickers = returns_df.columns
        
        # Calculate individual asset contributions
        portfolio_returns = (returns_df * weights).sum(axis=1)
        total_return = (1 + portfolio_returns).prod() - 1
        
        # Calculate contribution by asset
        contributions = {}
        for i, ticker in enumerate(tickers):
            asset_return = (1 + returns_df[ticker]).prod() - 1
            contribution = weights[i] * asset_return
            contributions[ticker] = {
                'weight': weights[i],
                'asset_return': asset_return,
                'contribution': contribution,
                'contribution_pct': contribution / total_return if total_return != 0 else 0
            }
        
        # Sort by contribution
        sorted_contributions = sorted(
            contributions.items(),
            key=lambda x: x[1]['contribution'],
            reverse=True
        )
        
        # Create DataFrame for display
        contrib_data = []
        for ticker, data in sorted_contributions:
            metadata = TICKER_TO_METADATA.get(ticker)
            contrib_data.append({
                'Ticker': ticker,
                'Name': metadata.name if metadata else ticker,
                'Weight': data['weight'],
                'Asset Return': data['asset_return'],
                'Contribution': data['contribution'],
                '% of Total': data['contribution_pct']
            })
        
        contrib_df = pd.DataFrame(contrib_data)
        
        # Format percentages
        format_dict = {
            'Weight': '{:.2%}',
            'Asset Return': '{:.2%}',
            'Contribution': '{:.2%}',
            '% of Total': '{:.2%}'
        }
        
        styled_df = contrib_df.style.format(format_dict)
        
        # Display
        st.dataframe(styled_df, use_container_width=True, height=400)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Contribution bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=contrib_df['Ticker'],
                    y=contrib_df['Contribution'] * 100,
                    text=contrib_df['Contribution'].apply(lambda x: f"{x:.2%}"),
                    textposition='auto',
                    marker_color='skyblue'
                )
            ])
            
            fig.update_layout(
                title="Return Contribution by Asset",
                xaxis_title="Asset",
                yaxis_title="Contribution (%)",
                height=400,
                template="plotly_white",
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Weight vs Return scatter
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=contrib_df['Weight'] * 100,
                y=contrib_df['Asset Return'] * 100,
                mode='markers+text',
                text=contrib_df['Ticker'],
                marker=dict(
                    size=contrib_df['Contribution'] * 10000,  # Scale for visibility
                    color=contrib_df['Contribution'] * 100,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Contribution %")
                ),
                hovertemplate='<b>%{text}</b><br>Weight: %{x:.1f}%<br>Return: %{y:.2f}%<br>Contribution: %{marker.size:.2f}%<extra></extra>'
            ))
            
            fig.update_layout(
                title="Weight vs Return Analysis",
                xaxis_title="Weight (%)",
                yaxis_title="Asset Return (%)",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_monte_carlo_simulation(self):
        """Render Monte Carlo simulation"""
        
        if st.session_state.returns_data is None:
            st.info("Please load data first for Monte Carlo simulation.")
            return
        
        st.markdown("#### 🔍 Monte Carlo Simulation")
        
        # Simulation parameters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            n_simulations = st.slider(
                "Number of Simulations",
                min_value=100,
                max_value=10000,
                value=1000,
                step=100
            )
        
        with col2:
            time_horizon = st.slider(
                "Time Horizon (Days)",
                min_value=30,
                max_value=2520,  # 10 years
                value=252,  # 1 year
                step=21  # Approximately 1 month
            )
        
        with col3:
            initial_portfolio_value = st.number_input(
                "Initial Portfolio Value ($)",
                min_value=1000,
                max_value=10000000,
                value=100000,
                step=1000
            )
        
        # Run simulation button
        if st.button("Run Monte Carlo Simulation", type="primary"):
            with st.spinner(f"Running {n_simulations} simulations..."):
                try:
                    simulation_results = self._run_monte_carlo_simulation(
                        n_simulations=n_simulations,
                        time_horizon=time_horizon,
                        initial_value=initial_portfolio_value
                    )
                    
                    # Display results
                    self._display_monte_carlo_results(simulation_results, initial_portfolio_value)
                    
                except Exception as e:
                    st.error(f"Simulation failed: {str(e)}")
    
    def _run_monte_carlo_simulation(self, n_simulations: int, time_horizon: int,
                                   initial_value: float) -> Dict:
        """Run Monte Carlo simulation"""
        
        returns_df = st.session_state.returns_data
        
        if st.session_state.optimization_results:
            weights = st.session_state.optimization_results['weights']
            portfolio_returns = (returns_df * weights).sum(axis=1)
        else:
            # Equal weight if no optimization
            weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
            portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # Estimate parameters from historical data
        mean_return = portfolio_returns.mean()
        std_return = portfolio_returns.std()
        
        # Run simulations
        np.random.seed(42)  # For reproducibility
        simulation_results = []
        
        for _ in range(n_simulations):
            # Generate random returns
            random_returns = np.random.normal(
                mean_return, 
                std_return, 
                time_horizon
            )
            
            # Calculate portfolio value path
            portfolio_values = initial_value * np.cumprod(1 + random_returns)
            simulation_results.append(portfolio_values)
        
        # Calculate statistics
        simulation_results = np.array(simulation_results)
        
        # Percentiles
        percentiles = [5, 25, 50, 75, 95]
        percentile_values = np.percentile(simulation_results[:, -1], percentiles)
        
        # Probability of loss
        final_values = simulation_results[:, -1]
        prob_loss = np.mean(final_values < initial_value)
        
        # Expected shortfall
        loss_values = final_values[final_values < initial_value]
        expected_shortfall = np.mean(initial_value - loss_values) if len(loss_values) > 0 else 0
        
        return {
            'simulation_paths': simulation_results,
            'final_values': final_values,
            'percentiles': dict(zip(percentiles, percentile_values)),
            'prob_loss': prob_loss,
            'expected_shortfall': expected_shortfall,
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values)
        }
    
    def _display_monte_carlo_results(self, results: Dict, initial_value: float):
        """Display Monte Carlo simulation results"""
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Expected Final Value",
                f"${results['mean_final_value']:,.0f}",
                delta=f"{((results['mean_final_value']/initial_value)-1):.2%}"
            )
        
        with col2:
            st.metric(
                "Median Final Value",
                f"${results['median_final_value']:,.0f}",
                delta="50th percentile"
            )
        
        with col3:
            st.metric(
                "Probability of Loss",
                f"{results['prob_loss']:.2%}",
                delta="P(Value < Initial)"
            )
        
        with col4:
            st.metric(
                "Expected Shortfall",
                f"${results['expected_shortfall']:,.0f}",
                delta="Average loss given loss"
            )
        
        # Percentile values
        st.markdown("##### 📊 Value at Risk Percentiles")
        
        percentiles_df = pd.DataFrame({
            'Percentile': list(results['percentiles'].keys()),
            'Value': list(results['percentiles'].values()),
            'Return': [(v/initial_value)-1 for v in results['percentiles'].values()]
        })
        
        # Format
        percentiles_df['Value'] = percentiles_df['Value'].apply(lambda x: f"${x:,.0f}")
        percentiles_df['Return'] = percentiles_df['Return'].apply(lambda x: f"{x:.2%}")
        
        st.dataframe(percentiles_df, use_container_width=True)
        
        # Visualization
        st.markdown("##### 📈 Simulation Visualization")
        
        tab1, tab2 = st.tabs(["Simulation Paths", "Final Value Distribution"])
        
        with tab1:
            # Plot sample of simulation paths
            fig = go.Figure()
            
            # Plot a sample of paths (first 100 for clarity)
            n_sample = min(100, len(results['simulation_paths']))
            for i in range(n_sample):
                fig.add_trace(go.Scatter(
                    x=list(range(len(results['simulation_paths'][i]))),
                    y=results['simulation_paths'][i],
                    mode='lines',
                    line=dict(width=0.5, color='rgba(0, 100, 255, 0.1)'),
                    showlegend=False,
                    hoverinfo='skip'
                ))
            
            # Plot median path
            median_path = np.median(results['simulation_paths'], axis=0)
            fig.add_trace(go.Scatter(
                x=list(range(len(median_path))),
                y=median_path,
                mode='lines',
                line=dict(width=3, color='red'),
                name='Median Path'
            ))
            
            # Plot initial value line
            fig.add_hline(
                y=initial_value,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Initial: ${initial_value:,.0f}",
                annotation_position="bottom right"
            )
            
            fig.update_layout(
                title="Monte Carlo Simulation Paths",
                xaxis_title="Time (Days)",
                yaxis_title="Portfolio Value ($)",
                height=500,
                template="plotly_white",
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            # Plot distribution of final values
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=results['final_values'],
                nbinsx=50,
                name='Final Values',
                marker_color='blue',
                opacity=0.7
            ))
            
            # Add vertical lines for percentiles
            colors = ['red', 'orange', 'green', 'orange', 'red']
            for percentile, color in zip([5, 25, 50, 75, 95], colors):
                value = results['percentiles'][percentile]
                fig.add_vline(
                    x=value,
                    line_dash="dash",
                    line_color=color,
                    annotation_text=f"{percentile}%: ${value:,.0f}",
                    annotation_position="top" if percentile >= 50 else "bottom"
                )
            
            # Add initial value line
            fig.add_vline(
                x=initial_value,
                line_dash="solid",
                line_color="black",
                annotation_text=f"Initial: ${initial_value:,.0f}",
                annotation_position="top"
            )
            
            fig.update_layout(
                title="Distribution of Final Portfolio Values",
                xaxis_title="Final Portfolio Value ($)",
                yaxis_title="Frequency",
                height=500,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_rolling_analysis(self):
        """Render rolling analysis tools"""
        
        if st.session_state.returns_data is None:
            st.info("Please load data first for rolling analysis.")
            return
        
        st.markdown("#### 📊 Rolling Analysis")
        
        # Rolling window parameters
        col1, col2 = st.columns(2)
        
        with col1:
            window_size = st.slider(
                "Rolling Window Size (Days)",
                min_value=21,  # Approximately 1 month
                max_value=252,  # 1 year
                value=63,  # Approximately 3 months
                step=21
            )
        
        with col2:
            metric = st.selectbox(
                "Metric to Analyze",
                options=['Volatility', 'Sharpe Ratio', 'Returns', 'Drawdown', 'Correlation'],
                index=0
            )
        
        # Calculate rolling metrics
        returns_df = st.session_state.returns_data
        
        if st.session_state.optimization_results:
            weights = st.session_state.optimization_results['weights']
            portfolio_returns = (returns_df * weights).sum(axis=1)
        else:
            # Equal weight if no optimization
            weights = np.ones(len(returns_df.columns)) / len(returns_df.columns)
            portfolio_returns = (returns_df * weights).sum(axis=1)
        
        # Calculate rolling metric
        if metric == 'Volatility':
            rolling_metric = portfolio_returns.rolling(window=window_size).std() * np.sqrt(self.config.TRADING_DAYS)
            yaxis_title = "Annualized Volatility"
            format_func = lambda x: f"{x:.2%}"
        elif metric == 'Sharpe Ratio':
            rolling_returns = portfolio_returns.rolling(window=window_size).mean() * self.config.TRADING_DAYS
            rolling_vol = portfolio_returns.rolling(window=window_size).std() * np.sqrt(self.config.TRADING_DAYS)
            rolling_metric = (rolling_returns - st.session_state.risk_free_rate) / rolling_vol
            yaxis_title = "Sharpe Ratio"
            format_func = lambda x: f"{x:.2f}"
        elif metric == 'Returns':
            rolling_metric = portfolio_returns.rolling(window=window_size).mean() * self.config.TRADING_DAYS
            yaxis_title = "Annualized Return"
            format_func = lambda x: f"{x:.2%}"
        elif metric == 'Drawdown':
            cumulative = (1 + portfolio_returns).cumprod()
            rolling_max = cumulative.rolling(window=window_size, min_periods=1).max()
            rolling_metric = (cumulative - rolling_max) / rolling_max
            yaxis_title = "Drawdown"
            format_func = lambda x: f"{x:.2%}"
        elif metric == 'Correlation':
            # Calculate average correlation between assets
            rolling_corr = returns_df.rolling(window=window_size).corr().groupby(level=0).apply(
                lambda x: x.values[np.triu_indices_from(x.values, k=1)].mean()
            )
            rolling_metric = rolling_corr
            yaxis_title = "Average Correlation"
            format_func = lambda x: f"{x:.2f}"
        else:
            rolling_metric = pd.Series()
            yaxis_title = ""
            format_func = lambda x: f"{x:.2f}"
        
        # Plot rolling metric
        if not rolling_metric.empty:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=rolling_metric.index,
                y=rolling_metric,
                mode='lines',
                line=dict(width=2, color='blue'),
                name=f'Rolling {metric}',
                hovertemplate='%{x|%Y-%m-%d}<br>' + f'{metric}: ' + '%{customdata}<extra></extra>',
                customdata=[format_func(x) for x in rolling_metric]
            ))
            
            # Add rolling average
            if len(rolling_metric) > window_size:
                rolling_mean = rolling_metric.rolling(window=window_size).mean()
                fig.add_trace(go.Scatter(
                    x=rolling_mean.index,
                    y=rolling_mean,
                    mode='lines',
                    line=dict(width=2, color='red', dash='dash'),
                    name='Rolling Average',
                    hovertemplate='%{x|%Y-%m-%d}<br>Average: %{customdata}<extra></extra>',
                    customdata=[format_func(x) for x in rolling_mean]
                ))
            
            fig.update_layout(
                title=f"Rolling {metric} ({window_size}-day window)",
                xaxis_title="Date",
                yaxis_title=yaxis_title,
                height=500,
                template="plotly_white",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(f"Current {metric}", format_func(rolling_metric.iloc[-1]))
            
            with col2:
                st.metric(f"Average {metric}", format_func(rolling_metric.mean()))
            
            with col3:
                st.metric(f"Maximum {metric}", format_func(rolling_metric.max()))
            
            with col4:
                st.metric(f"Minimum {metric}", format_func(rolling_metric.min()))
        
        # Rolling correlation heatmap
        if metric == 'Correlation' and len(returns_df.columns) <= 15:
            st.markdown("##### 🔗 Rolling Correlation Heatmap")
            
            # Select assets for heatmap
            selected_assets = st.multiselect(
                "Select assets for correlation heatmap",
                options=returns_df.columns.tolist(),
                default=returns_df.columns.tolist()[:5] if len(returns_df.columns) >= 5 else returns_df.columns.tolist()
            )
            
            if len(selected_assets) >= 2:
                # Calculate rolling correlations for selected assets
                rolling_correlations = {}
                
                for asset in selected_assets:
                    for other_asset in selected_assets:
                        if asset != other_asset:
                            key = f"{asset}-{other_asset}"
                            corr = returns_df[asset].rolling(window=window_size).corr(returns_df[other_asset])
                            rolling_correlations[key] = corr
                
                # Create heatmap data
                heatmap_data = []
                dates = rolling_correlations[list(rolling_correlations.keys())[0]].index
                
                # Sample dates for heatmap (every 10th point for performance)
                sample_dates = dates[::10]
                
                for date in sample_dates:
                    row = {'Date': date}
                    for key, corr_series in rolling_correlations.items():
                        if date in corr_series.index:
                            row[key] = corr_series[date]
                        else:
                            row[key] = np.nan
                    heatmap_data.append(row)
                
                heatmap_df = pd.DataFrame(heatmap_data)
                
                if not heatmap_df.empty:
                    # Create heatmap
                    fig = go.Figure(data=go.Heatmap(
                        z=heatmap_df.drop('Date', axis=1).values.T,
                        x=heatmap_df['Date'],
                        y=list(rolling_correlations.keys()),
                        colorscale='RdBu',
                        zmid=0,
                        colorbar=dict(title="Correlation")
                    ))
                    
                    fig.update_layout(
                        title=f"Rolling Correlations ({window_size}-day window)",
                        xaxis_title="Date",
                        yaxis_title="Asset Pair",
                        height=600,
                        template="plotly_white"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_factor_analysis(self):
        """Render factor analysis tools"""
        
        if st.session_state.returns_data is None:
            st.info("Please load data first for factor analysis.")
            return
        
        st.markdown("#### 📉 Factor Analysis")
        
        if not STATSMODELS_AVAILABLE:
            st.warning("Factor analysis requires statsmodels package. Please install it for full functionality.")
            st.code("pip install statsmodels")
            return
        
        # Factor analysis implementation would go here
        # This is a placeholder for the factor analysis functionality
        
        st.info("Factor analysis module is under development. Advanced factor modeling will be available in the next release.")
        
        # Placeholder for future implementation
        with st.expander("🔍 Planned Factor Analysis Features"):
            st.markdown("""
            **Planned Features:**
            
            1. **Multi-Factor Models**
               - Fama-French 3/5 factor models
               - Carhart 4-factor model
               - Custom factor definitions
            
            2. **Risk Factor Attribution**
               - Factor exposures calculation
               - Risk decomposition by factors
               - Factor timing analysis
            
            3. **Style Analysis**
               - Returns-based style analysis
               - Holdings-based style analysis
               - Style drift detection
            
            4. **Macroeconomic Factors**
               - Interest rate sensitivity
               - Inflation exposure
               - Economic cycle positioning
            
            5. **Statistical Factors**
               - Principal Component Analysis
               - Maximum likelihood factor analysis
               - Dynamic factor models
            
            **Expected Release:** Q2 2024
            """)
    
    def _render_advanced_tools(self):
        """Render advanced tools and utilities"""
        
        st.markdown('<h2 class="section-header">🔧 Advanced Tools</h2>', 
                   unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "⚡ Performance",
            "🗃️ Cache Management",
            "📁 Data Export",
            "🔐 Security & Logs"
        ])
        
        with tab1:
            self._render_performance_tools()
        
        with tab2:
            self._render_cache_management()
        
        with tab3:
            self._render_data_export()
        
        with tab4:
            self._render_security_logs()
    
    def _render_performance_tools(self):
        """Render performance monitoring tools"""
        
        st.markdown("#### ⚡ Performance Monitoring")
        
        # Optimizer performance
        optimizer_metrics = self.optimizer.get_performance_metrics()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Optimizations",
                optimizer_metrics['optimizations_performed']
            )
        
        with col2:
            success_rate = optimizer_metrics['successful_optimizations'] / max(1, optimizer_metrics['optimizations_performed'])
            st.metric(
                "Success Rate",
                f"{success_rate:.1%}"
            )
        
        with col3:
            st.metric(
                "Avg Computation Time",
                f"{optimizer_metrics['avg_computation_time']:.2f}s"
            )
        
        with col4:
            if optimizer_metrics['last_optimization_time']:
                last_time = datetime.fromisoformat(optimizer_metrics['last_optimization_time'])
                st.metric(
                    "Last Optimization",
                    last_time.strftime("%H:%M:%S")
                )
        
        # Cache performance
        cache_stats = global_cache.stats()
        
        st.markdown("##### 🗃️ Cache Performance")
        
        cache_cols = st.columns(4)
        
        with cache_cols[0]:
            st.metric(
                "Cache Hit Rate",
                f"{cache_stats['hit_rate']:.1%}"
            )
        
        with cache_cols[1]:
            st.metric(
                "Total Entries",
                cache_stats['size']
            )
        
        with cache_cols[2]:
            st.metric(
                "Memory Usage",
                f"{cache_stats['total_memory_mb']:.1f} MB"
            )
        
        with cache_cols[3]:
            st.metric(
                "Total Requests",
                cache_stats['total_requests']
            )
        
        # Optimization history
        st.markdown("##### 📊 Optimization History")
        
        history_df = self.optimizer.get_optimization_history()
        
        if not history_df.empty:
            # Display last 10 optimizations
            recent_history = history_df.tail(10)
            
            # Format columns
            format_dict = {
                'expected_return': '{:.2%}',
                'expected_risk': '{:.2%}',
                'sharpe_ratio': '{:.2f}',
                'max_drawdown': '{:.2%}',
                'elapsed_time_seconds': '{:.2f}s'
            }
            
            styled_df = recent_history.style.format(format_dict)
            
            st.dataframe(
                styled_df,
                use_container_width=True,
                height=400
            )
            
            # Clear history button
            if st.button("Clear Optimization History", type="secondary"):
                self.optimizer.clear_history()
                st.success("Optimization history cleared!")
                st.rerun()
        else:
            st.info("No optimization history available yet.")
        
        # System information
        st.markdown("##### 🖥️ System Information")
        
        sys_info = {
            'Python Version': sys.version.split()[0],
            'Streamlit Version': st.__version__,
            'NumPy Version': np.__version__,
            'Pandas Version': pd.__version__,
            'PyPortfolioOpt Version': PYPFOPT_VERSION,
            'CPU Cores': os.cpu_count() or 'Unknown',
            'Platform': sys.platform
        }
        
        sys_df = pd.DataFrame(list(sys_info.items()), columns=['Component', 'Version'])
        st.dataframe(sys_df, use_container_width=True, hide_index=True)
    
    def _render_cache_management(self):
        """Render cache management tools"""
        
        st.markdown("#### 🗃️ Cache Management")
        
        cache_stats = global_cache.stats()
        
        # Cache statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Cache Size",
                f"{cache_stats['size']}/{cache_stats['max_size']}",
                delta=f"{cache_stats['utilization']:.1%} utilization"
            )
            
            st.metric(
                "Memory Usage",
                f"{cache_stats['total_memory_mb']:.1f} MB",
                delta=f"{cache_stats['memory_utilization']:.1%} of limit"
            )
        
        with col2:
            st.metric(
                "Hit Rate",
                f"{cache_stats['hit_rate']:.1%}",
                delta=f"{cache_stats['hits']} hits"
            )
            
            st.metric(
                "Avg Entry Size",
                f"{cache_stats['avg_entry_size_mb']:.2f} MB",
                delta=f"{cache_stats['misses']} misses"
            )
        
        # Cache actions
        st.markdown("##### ⚙️ Cache Actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Clear Cache", use_container_width=True):
                global_cache.clear()
                st.success("Cache cleared successfully!")
                st.rerun()
        
        with col2:
            if st.button("Export Cache Stats", use_container_width=True):
                stats_df = global_cache.export_stats()
                csv = stats_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="Download Stats",
                    data=csv,
                    file_name="cache_statistics.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        with col3:
            if st.button("Optimize Cache", use_container_width=True):
                global_cache._clean_cache(aggressive=True)
                st.success("Cache optimized!")
                st.rerun()
        
        # Cache configuration
        st.markdown("##### ⚙️ Cache Configuration")
        
        with st.form("cache_config_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                new_max_entries = st.number_input(
                    "Maximum Entries",
                    min_value=10,
                    max_value=10000,
                    value=global_cache.max_entries,
                    step=10
                )
            
            with col2:
                new_max_memory = st.number_input(
                    "Maximum Memory (MB)",
                    min_value=10,
                    max_value=10000,
                    value=global_cache.max_memory_mb,
                    step=10
                )
            
            if st.form_submit_button("Update Configuration"):
                global_cache.max_entries = new_max_entries
                global_cache.max_memory_mb = new_max_memory
                st.success("Cache configuration updated!")
        
        # Cache content preview
        st.markdown("##### 📋 Cache Content Preview")
        
        if global_cache.cache:
            cache_keys = list(global_cache.cache.keys())[:20]  # Show first 20
            
            cache_preview = []
            for key in cache_keys:
                entry = global_cache.cache[key]
                size_mb = global_cache.size_log.get(key, 0) / (1024 * 1024)
                last_access = datetime.fromtimestamp(global_cache.access_log.get(key, 0)).strftime("%Y-%m-%d %H:%M:%S")
                
                cache_preview.append({
                    'Key': key[:20] + "...",
                    'Type': type(entry).__name__,
                    'Size (MB)': f"{size_mb:.2f}",
                    'Last Access': last_access
                })
            
            preview_df = pd.DataFrame(cache_preview)
            st.dataframe(preview_df, use_container_width=True, height=300)
        else:
            st.info("Cache is empty.")
    
    def _render_data_export(self):
        """Render data export tools"""
        
        st.markdown("#### 📁 Data Export")
        
        if st.session_state.prices_data is None:
            st.info("No data loaded. Please load data first to export.")
            return
        
        # Export options
        st.markdown("##### 📊 Export Current Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Export prices
            prices_df = st.session_state.prices_data
            prices_csv = prices_df.to_csv().encode('utf-8')
            
            st.download_button(
                label="📥 Download Price Data",
                data=prices_csv,
                file_name="price_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        with col2:
            # Export returns
            returns_df = st.session_state.returns_data
            returns_csv = returns_df.to_csv().encode('utf-8')
            
            st.download_button(
                label="📥 Download Returns Data",
                data=returns_csv,
                file_name="returns_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Export optimization results
        if st.session_state.optimization_results:
            st.markdown("##### ⚙️ Export Optimization Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Export weights
                weights = st.session_state.optimization_results['cleaned_weights']
                weights_df = pd.DataFrame(list(weights.items()), columns=['Asset', 'Weight'])
                weights_csv = weights_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Download Portfolio Weights",
                    data=weights_csv,
                    file_name="portfolio_weights.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                # Export performance metrics
                metrics = {
                    'Metric': [
                        'Expected Return',
                        'Expected Risk',
                        'Sharpe Ratio',
                        'Max Drawdown',
                        'Sortino Ratio',
                        'Information Ratio',
                        'Method'
                    ],
                    'Value': [
                        st.session_state.optimization_results.get('expected_return', 0),
                        st.session_state.optimization_results.get('expected_risk', 0),
                        st.session_state.optimization_results.get('sharpe_ratio', 0),
                        st.session_state.optimization_results.get('max_drawdown', 0),
                        st.session_state.optimization_results.get('sortino_ratio', 0),
                        st.session_state.optimization_results.get('information_ratio', 0),
                        st.session_state.optimization_results.get('method', 'N/A')
                    ]
                }
                
                metrics_df = pd.DataFrame(metrics)
                metrics_csv = metrics_df.to_csv(index=False).encode('utf-8')
                
                st.download_button(
                    label="📥 Download Performance Metrics",
                    data=metrics_csv,
                    file_name="performance_metrics.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        
        # Export risk metrics
        if st.session_state.risk_metrics:
            st.markdown("##### 📈 Export Risk Metrics")
            
            risk_metrics = st.session_state.risk_metrics
            
            # Filter out complex objects
            simple_metrics = {}
            for key, value in risk_metrics.items():
                if not isinstance(value, (pd.Series, pd.DataFrame, np.ndarray, dict, list)):
                    simple_metrics[key] = value
            
            risk_df = pd.DataFrame([simple_metrics])
            risk_csv = risk_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download Risk Metrics",
                data=risk_csv,
                file_name="risk_metrics.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        # Batch export
        st.markdown("##### 📦 Batch Export")
        
        with st.form("batch_export_form"):
            export_options = st.multiselect(
                "Select data to export",
                options=[
                    "Price Data",
                    "Returns Data",
                    "Portfolio Weights",
                    "Performance Metrics",
                    "Risk Metrics",
                    "Optimization History",
                    "Cache Statistics"
                ],
                default=["Price Data", "Returns Data"]
            )
            
            export_format = st.radio(
                "Export format",
                options=["CSV", "Excel", "JSON"],
                horizontal=True
            )
            
            if st.form_submit_button("Create Batch Export"):
                with st.spinner("Preparing batch export..."):
                    try:
                        # Create export package
                        export_data = {}
                        
                        if "Price Data" in export_options:
                            export_data['prices'] = st.session_state.prices_data
                        
                        if "Returns Data" in export_options:
                            export_data['returns'] = st.session_state.returns_data
                        
                        if "Portfolio Weights" in export_options and st.session_state.optimization_results:
                            weights = st.session_state.optimization_results['cleaned_weights']
                            export_data['weights'] = pd.DataFrame(list(weights.items()), columns=['Asset', 'Weight'])
                        
                        if "Performance Metrics" in export_options and st.session_state.optimization_results:
                            metrics = {
                                'Metric': [
                                    'Expected Return',
                                    'Expected Risk',
                                    'Sharpe Ratio',
                                    'Max Drawdown',
                                    'Sortino Ratio',
                                    'Information Ratio',
                                    'Method'
                                ],
                                'Value': [
                                    st.session_state.optimization_results.get('expected_return', 0),
                                    st.session_state.optimization_results.get('expected_risk', 0),
                                    st.session_state.optimization_results.get('sharpe_ratio', 0),
                                    st.session_state.optimization_results.get('max_drawdown', 0),
                                    st.session_state.optimization_results.get('sortino_ratio', 0),
                                    st.session_state.optimization_results.get('information_ratio', 0),
                                    st.session_state.optimization_results.get('method', 'N/A')
                                ]
                            }
                            export_data['performance_metrics'] = pd.DataFrame(metrics)
                        
                        if "Risk Metrics" in export_options and st.session_state.risk_metrics:
                            simple_metrics = {}
                            for key, value in st.session_state.risk_metrics.items():
                                if not isinstance(value, (pd.Series, pd.DataFrame, np.ndarray, dict, list)):
                                    simple_metrics[key] = value
                            export_data['risk_metrics'] = pd.DataFrame([simple_metrics])
                        
                        if "Optimization History" in export_options:
                            history_df = self.optimizer.get_optimization_history()
                            export_data['optimization_history'] = history_df
                        
                        if "Cache Statistics" in export_options:
                            stats_df = global_cache.export_stats()
                            export_data['cache_statistics'] = stats_df
                        
                        # Create export file
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        if export_format == "Excel":
                            # Create Excel file with multiple sheets
                            output = io.BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                for sheet_name, data in export_data.items():
                                    if isinstance(data, pd.DataFrame):
                                        data.to_excel(writer, sheet_name=sheet_name[:31], index=False)
                            
                            output.seek(0)
                            st.download_button(
                                label=f"📥 Download Excel File",
                                data=output,
                                file_name=f"portfolio_export_{timestamp}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True
                            )
                        
                        elif export_format == "JSON":
                            # Convert to JSON
                            json_data = {}
                            for key, data in export_data.items():
                                if isinstance(data, pd.DataFrame):
                                    json_data[key] = data.to_dict(orient='records')
                                elif isinstance(data, dict):
                                    json_data[key] = data
                            
                            json_str = json.dumps(json_data, indent=2, default=str)
                            st.download_button(
                                label=f"📥 Download JSON File",
                                data=json_str,
                                file_name=f"portfolio_export_{timestamp}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        
                        else:  # CSV
                            # Create ZIP file with multiple CSVs
                            import zipfile
                            
                            zip_buffer = io.BytesIO()
                            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                                for file_name, data in export_data.items():
                                    if isinstance(data, pd.DataFrame):
                                        csv_data = data.to_csv(index=False)
                                        zip_file.writestr(f"{file_name}.csv", csv_data)
                            
                            zip_buffer.seek(0)
                            st.download_button(
                                label=f"📥 Download ZIP File",
                                data=zip_buffer,
                                file_name=f"portfolio_export_{timestamp}.zip",
                                mime="application/zip",
                                use_container_width=True
                            )
                    
                    except Exception as e:
                        st.error(f"Export failed: {str(e)}")
    
    def _render_security_logs(self):
        """Render security and logging tools"""
        
        st.markdown("#### 🔐 Security & Logs")
        
        # Security settings
        st.markdown("##### 🔒 Security Settings")
        
        with st.form("security_settings"):
            col1, col2 = st.columns(2)
            
            with col1:
                enable_audit = st.checkbox(
                    "Enable Audit Logging",
                    value=InstitutionalConfig.ENABLE_AUDIT_LOG
                )
                
                enable_compliance = st.checkbox(
                    "Enable Compliance Checks",
                    value=InstitutionalConfig.ENABLE_COMPLIANCE_CHECKS
                )
            
            with col2:
                enable_risk_limits = st.checkbox(
                    "Enable Risk Limits",
                    value=InstitutionalConfig.ENABLE_RISK_LIMITS
                )
                
                enable_tracking = st.checkbox(
                    "Enable Performance Tracking",
                    value=InstitutionalConfig.ENABLE_PERFORMANCE_TRACKING
                )
            
            if st.form_submit_button("Update Security Settings"):
                InstitutionalConfig.ENABLE_AUDIT_LOG = enable_audit
                InstitutionalConfig.ENABLE_COMPLIANCE_CHECKS = enable_compliance
                InstitutionalConfig.ENABLE_RISK_LIMITS = enable_risk_limits
                InstitutionalConfig.ENABLE_PERFORMANCE_TRACKING = enable_tracking
                
                # Update optimizer settings
                self.optimizer.risk_limits_enabled = enable_risk_limits
                self.optimizer.compliance_checks_enabled = enable_compliance
                self.optimizer.performance_tracking_enabled = enable_tracking
                
                st.success("Security settings updated!")
        
        # Log viewing
        st.markdown("##### 📋 Application Logs")
        
        log_tab1, log_tab2, log_tab3 = st.tabs(["Recent Logs", "Error Logs", "Download Logs"])
        
        with log_tab1:
            # Display recent logs
            try:
                with open('institutional_portfolio.log', 'r') as f:
                    logs = f.readlines()[-100:]  # Last 100 lines
                
                if logs:
                    st.text_area(
                        "Recent Logs",
                        value="".join(logs),
                        height=300,
                        disabled=True
                    )
                else:
                    st.info("No logs available yet.")
                    
            except FileNotFoundError:
                st.info("Log file not found. Logging may not be enabled.")
        
        with log_tab2:
            # Display error logs
            try:
                with open('institutional_portfolio.log', 'r') as f:
                    error_logs = [line for line in f if 'ERROR' in line or 'WARNING' in line][-50:]
                
                if error_logs:
                    st.text_area(
                        "Error and Warning Logs",
                        value="".join(error_logs),
                        height=300,
                        disabled=True
                    )
                else:
                    st.success("No errors or warnings found in recent logs.")
                    
            except FileNotFoundError:
                st.info("Log file not found.")
        
        with log_tab3:
            # Log download options
            col1, col2 = st.columns(2)
            
            with col1:
                try:
                    with open('institutional_portfolio.log', 'rb') as f:
                        log_data = f.read()
                    
                    st.download_button(
                        label="📥 Download Full Log",
                        data=log_data,
                        file_name="portfolio_logs.txt",
                        mime="text/plain",
                        use_container_width=True
                    )
                except FileNotFoundError:
                    st.info("Log file not found.")
            
            with col2:
                if st.button("Clear Log File", use_container_width=True):
                    try:
                        open('institutional_portfolio.log', 'w').close()
                        st.success("Log file cleared!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to clear log file: {str(e)}")
        
        # Data validation
        st.markdown("##### 🔍 Data Validation")
        
        if st.session_state.prices_data is not None:
            validation_results = self._validate_data_quality()
            
            if validation_results['issues']:
                st.warning(f"Found {len(validation_results['issues'])} data quality issues")
                
                with st.expander("View Data Issues"):
                    for issue in validation_results['issues']:
                        st.write(f"⚠️ {issue}")
            else:
                st.success("Data quality validation passed!")
            
            # Show validation metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                missing_pct = st.session_state.prices_data.isna().sum().sum() / st.session_state.prices_data.size
                st.metric(
                    "Missing Data",
                    f"{missing_pct:.2%}",
                    delta="Lower is better"
                )
            
            with col2:
                zero_prices = (st.session_state.prices_data <= 0).sum().sum()
                st.metric(
                    "Zero/Negative Prices",
                    zero_prices,
                    delta="Should be 0"
                )
            
            with col3:
                outlier_count = validation_results.get('outlier_count', 0)
                st.metric(
                    "Potential Outliers",
                    outlier_count,
                    delta="Check if excessive"
                )
        else:
            st.info("Load data to run validation checks.")
    
    def _validate_data_quality(self) -> Dict:
        """Validate data quality"""
        
        issues = []
        prices_df = st.session_state.prices_data
        
        if prices_df is None:
            return {'issues': [], 'outlier_count': 0}
        
        # Check for missing data
        missing_pct = prices_df.isna().sum().sum() / prices_df.size
        if missing_pct > 0.05:
            issues.append(f"High percentage of missing data: {missing_pct:.2%}")
        
        # Check for zero or negative prices
        zero_prices = (prices_df <= 0).sum().sum()
        if zero_prices > 0:
            issues.append(f"Found {zero_prices} zero or negative prices")
        
        # Check for extreme outliers
        outlier_count = 0
        for column in prices_df.columns:
            Q1 = prices_df[column].quantile(0.25)
            Q3 = prices_df[column].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((prices_df[column] < Q1 - 3*IQR) | (prices_df[column] > Q3 + 3*IQR)).sum()
            outlier_count += outliers
        
        if outlier_count > len(prices_df.columns) * 5:  # Arbitrary threshold
            issues.append(f"Found {outlier_count} potential outliers")
        
        # Check for stale data
        latest_date = prices_df.index[-1]
        days_since_update = (pd.Timestamp.now() - latest_date).days
        if days_since_update > 5:
            issues.append(f"Data may be stale. Latest data point is {days_since_update} days old")
        
        # Check for large gaps
        returns_df = prices_df.pct_change()
        large_moves = (returns_df.abs() > 0.2).sum().sum()  # 20% daily moves
        if large_moves > len(prices_df.columns):
            issues.append(f"Found {large_moves} large daily price moves (>20%)")
        
        return {
            'issues': issues,
            'outlier_count': outlier_count,
            'missing_pct': missing_pct,
            'zero_prices': zero_prices,
            'large_moves': large_moves,
            'days_since_update': days_since_update
        }
    
    def _render_footer(self):
        """Render application footer"""
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🏛️ Institutional Apollo / ENIGMA**")
            st.markdown("Quant Terminal v5.0")
        
        with col2:
            st.markdown("**📧 Support**")
            st.markdown("For issues or feature requests, please contact support.")
        
        with col3:
            st.markdown("**📄 Disclaimer**")
            st.markdown("This tool is for educational and research purposes only. Not financial advice.")
        
        # Version and build info
        st.markdown(
            f"""
            <div style='text-align: center; color: #666; font-size: 0.8em; margin-top: 2rem;'>
                Build: {datetime.now().strftime('%Y%m%d')} | 
                Python {sys.version.split()[0]} | 
                Streamlit {st.__version__}
            </div>
            """,
            unsafe_allow_html=True
        )
    
    def _handle_error(self, error: Exception):
        """Handle application errors gracefully"""
        
        logger.error(f"Application error: {str(error)}", exc_info=True)
        
        # Display error message
        st.error(f"An error occurred: {str(error)}")
        
        # Show traceback in expander for debugging
        with st.expander("Error Details (Technical)"):
            st.code(traceback.format_exc())
        
        # Recovery options
        st.markdown("### 🔄 Recovery Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Restart Application", use_container_width=True):
                # Clear session state and rerun
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        with col2:
            if st.button("📝 Report Issue", use_container_width=True):
                st.info("""
                Please include the following in your issue report:
                1. Error message above
                2. Steps to reproduce
                3. Expected behavior
                
                Contact: support@institutional-apollo.com
                """)

# =============================================================
# MAIN APPLICATION ENTRY POINT
# =============================================================
def main():
    """Main application entry point"""
    
    try:
        # Create and run application
        app = InstitutionalPortfolioApplication()
        app.run()
        
    except Exception as e:
        # Handle initialization errors
        st.error(f"Failed to initialize application: {str(e)}")
        logger.critical(f"Application initialization failed: {str(e)}", exc_info=True)
        
        # Provide basic error recovery
        if st.button("🔄 Try Again"):
            st.rerun()

# =============================================================
# APPLICATION LAUNCH
# =============================================================
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('institutional_portfolio.log'),
            logging.StreamHandler()
        ]
    )
    
    # Set numpy print options
    np.set_printoptions(precision=4, suppress=True)
    
    # Run the application
    main()
