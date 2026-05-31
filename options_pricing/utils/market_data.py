import numpy as np


class MarketData:
    def __init__(self, ticker: str = None):
        self.ticker = ticker
        self.spot = None
        self.chain = None

    def fetch(self):
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("Run: pip install yfinance")

        tk = yf.Ticker(self.ticker)
        hist = tk.history(period="5d")
        self.spot = float(hist["Close"].iloc[-1])
        return self.spot

    def fetch_chain(self, expiry: str = None):
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("Run: pip install yfinance")

        tk = yf.Ticker(self.ticker)

        if expiry is None:
            expiry = tk.options[0]

        chain = tk.option_chain(expiry)
        self.chain = chain
        return chain

    def calls(self):
        if self.chain is None:
            raise ValueError("Call fetch_chain() first")
        return self.chain.calls[["strike", "lastPrice", "bid", "ask", "impliedVolatility", "openInterest"]]

    def puts(self):
        if self.chain is None:
            raise ValueError("Call fetch_chain() first")
        return self.chain.puts[["strike", "lastPrice", "bid", "ask", "impliedVolatility", "openInterest"]]

    def mid_prices(self, option_type: str = "calls"):
        df = self.calls() if option_type == "calls" else self.puts()
        df = df.copy()
        df["mid"] = (df["bid"] + df["ask"]) / 2
        return df[["strike", "mid", "impliedVolatility"]].dropna()

    @staticmethod
    def historical_vol(prices: np.ndarray, window: int = 21):
        log_returns = np.diff(np.log(prices))
        rolling_vol = []
        for i in range(window, len(log_returns) + 1):
            vol = np.std(log_returns[i - window:i], ddof=1) * np.sqrt(252)
            rolling_vol.append(vol)
        return np.array(rolling_vol)

    @staticmethod
    def risk_free_rate():
        try:
            import yfinance as yf
            tnx = yf.Ticker("^TNX")
            rate = tnx.history(period="1d")["Close"].iloc[-1] / 100
            return float(rate)
        except Exception:
            return 0.05
