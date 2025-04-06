# pca_risk_dashboard.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import yfinance as yf
import os 
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis
from scipy.linalg import pinv
import plotly.graph_objects as go
import plotly.express as px
import time

# --- Streamlit Page Configuration ---
st.set_page_config(layout="wide", page_title="PCA Risk Factor Analysis")

# --- Define Cache Filenames ---
DATA_DIR = "data"

STOCK_CACHE_CSV = os.path.join(DATA_DIR, "sp500_data_cache.csv")
METADATA_CACHE_CSV = os.path.join(DATA_DIR, "sp500_metadata.csv")
ETF_CACHE_CSV = os.path.join(DATA_DIR, "etf_data_cache.csv")
TICKER_BLACKLIST_CSV = os.path.join(DATA_DIR, "ticker_blacklist.csv")

# --- Load Ticker Blacklist ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def load_ticker_blacklist(filename: str = TICKER_BLACKLIST_CSV) -> list:
    """Loads blacklisted tickers from CSV file."""
    if not os.path.exists(filename):
        return []
    
    try:
        df = pd.read_csv(filename, header=None)
        # Flatten the DataFrame to a list, split by comma, strip whitespace, and remove empty strings
        blacklist = []
        for row in df.values.flatten():
            blacklist.extend([ticker.strip() for ticker in row.split(',') if ticker.strip()])
        return blacklist
    except Exception as e:
        st.warning(f"Failed to load ticker blacklist from {filename}: {e}")
        return []

# --- Caching Functions ---
# Use Streamlit's caching decorators for performance

@st.cache_data(ttl=3600) # Cache for 1 hour
def get_sp500_symbols():
    """Gets S&P 500 symbols from Wikipedia and filters out blacklisted tickers."""
    try:
        table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
        symbols = [sym.replace(".", "-") for sym in table.Symbol.tolist()]
        
        # Filter out blacklisted tickers
        blacklist = load_ticker_blacklist()
        if blacklist:
            filtered_symbols = [sym for sym in symbols if sym not in blacklist]
            skipped = set(symbols) - set(filtered_symbols)
            if skipped:
                st.info(f"Filtered out {len(skipped)} blacklisted tickers: {', '.join(skipped)}")
            return filtered_symbols
        return symbols
    except Exception as e:
        st.error(f"Failed to fetch S&P 500 symbols: {e}")
        return []

# Modified get_stock_metadata to use persistent CSV cache
# Note: Cache decorator removed to avoid CacheReplayClosureError with download_log UI elements
def get_stock_metadata_cached(symbols: frozenset, filename: str = METADATA_CACHE_CSV) -> pd.DataFrame:
    """
    Collects metadata, using a persistent CSV cache.
    Downloads only missing symbols and updates the cache.
    """
    symbols_list = list(symbols) # Convert frozenset back to list for processing
    cached_metadata = pd.DataFrame()
    symbols_to_fetch = symbols_list[:] # Assume all need fetching initially

    if os.path.exists(filename):
        try:
            cached_metadata = pd.read_csv(filename)
            # Check which requested symbols are already in the cache
            symbols_in_cache = set(cached_metadata['Symbol'].unique())
            symbols_needed = set(symbols_list)
            symbols_to_fetch = list(symbols_needed - symbols_in_cache)
            download_log.info(f"Metadata Cache: Found {len(symbols_in_cache)} symbols. Need to fetch {len(symbols_to_fetch)}.")
        except Exception as e:
            download_log.warning(f"Could not read metadata cache '{filename}': {e}. Will download all.")
            cached_metadata = pd.DataFrame() # Ensure it's empty if read failed
            symbols_to_fetch = symbols_list[:]

    else:
        download_log.info(f"Metadata cache '{filename}' not found. Will download for {len(symbols_to_fetch)} symbols.")

    new_metadata_list = []
    if symbols_to_fetch:
        failed_symbols = []
        progress_bar = download_log.progress(0, text=f"Downloading Metadata for {len(symbols_to_fetch)} symbols...")
        total_to_fetch = len(symbols_to_fetch)

        for i, symbol in enumerate(symbols_to_fetch):
            retry_count = 0
            max_retries = 3
            retry_delay = 2  # seconds

            while retry_count < max_retries:
                try:
                    ticker = yf.Ticker(symbol)
                    info = ticker.info
                    stock_info = {
                        'Symbol': symbol,
                        'Name': info.get('longName', 'Unknown'),
                        'Sector': info.get('sector', 'Unknown'),
                        'Industry': info.get('industry', 'Unknown'),
                        'MarketCap': info.get('marketCap', None),
                        'Country': info.get('country', 'Unknown'),
                        'Exchange': info.get('exchange', 'Unknown')
                    }
                    new_metadata_list.append(stock_info)
                    break  # Success, exit retry loop
                except Exception as e:
                    retry_count += 1
                    if retry_count < max_retries:
                        download_log.info(f"Retry {retry_count}/{max_retries} for {symbol} metadata: {str(e)}")
                        time.sleep(retry_delay)
                    else:
                        failed_symbols.append(symbol)
                        download_log.warning(f"Failed all retries for {symbol} metadata: {str(e)}")

            progress_bar.progress((i + 1) / total_to_fetch, text=f"Downloading Metadata: {symbol}")

        progress_bar.empty()
        if failed_symbols:
            download_log.warning(f"Failed to download metadata for {len(failed_symbols)} symbols after retries: {', '.join(failed_symbols)}")

    if new_metadata_list:
        new_metadata_df = pd.DataFrame(new_metadata_list)
        # Combine new data with cached data (if any)
        combined_metadata = pd.concat([cached_metadata, new_metadata_df], ignore_index=True).drop_duplicates(subset=['Symbol'], keep='last')

        # Save the updated combined data back to CSV
        try:
            combined_metadata.to_csv(filename, index=False)
            download_log.success(f"Updated metadata cache '{filename}'.")
        except Exception as e:
            download_log.error(f"Failed to write metadata cache '{filename}': {e}")

        # Return the relevant subset for the originally requested symbols
        return combined_metadata[combined_metadata['Symbol'].isin(symbols_list)].reset_index(drop=True)
    else:
        # Return the relevant subset from the cache if no new data was fetched
        return cached_metadata[cached_metadata['Symbol'].isin(symbols_list)].reset_index(drop=True)

@st.cache_data(ttl=3600) # Cache for 1 hour
def get_stock_metadata(symbols: list) -> pd.DataFrame:
    """Collects sector and industry metadata for stocks."""
    metadata = []
    failed_symbols = []
    progress_bar = st.progress(0, text="Downloading Metadata...")
    total_symbols = len(symbols)

    for i, symbol in enumerate(symbols):
        retry_count = 0
        max_retries = 3
        retry_delay = 2  # seconds

        while retry_count < max_retries:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                stock_info = {
                    'Symbol': symbol,
                    'Name': info.get('longName', 'Unknown'),
                    'Sector': info.get('sector', 'Unknown'),
                    'Industry': info.get('industry', 'Unknown'),
                    'MarketCap': info.get('marketCap', None),
                    'Country': info.get('country', 'Unknown'),
                    'Exchange': info.get('exchange', 'Unknown')
                }
                metadata.append(stock_info)
                break  # Success, exit retry loop
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    st.info(f"Retry {retry_count}/{max_retries} for {symbol} metadata")
                    time.sleep(retry_delay)
                else:
                    failed_symbols.append(symbol)
                    # st.warning(f"Failed to get metadata for {symbol} after retries: {e}")

        # Update progress bar
        progress_bar.progress((i + 1) / total_symbols, text=f"Downloading Metadata: {symbol}")

    progress_bar.empty() # Clear progress bar
    df = pd.DataFrame(metadata)

    if failed_symbols:
        st.warning(f"Failed to download metadata for {len(failed_symbols)} stocks after retries: {', '.join(failed_symbols)}")

    return df

# Modified load_stock_prices to use persistent CSV cache
# Note: Cache decorator removed to avoid CacheReplayClosureError with download_log UI elements
def load_stock_prices_cached(symbols: frozenset, start: date, end: date, filename: str = STOCK_CACHE_CSV) -> pd.DataFrame:
    """
    Loads stock prices, prioritizing a persistent CSV cache.
    Downloads data if cache is missing, outdated, or doesn't contain all symbols.
    """
    symbols_list = list(symbols)
    requested_start_dt = pd.Timestamp(start)
    requested_end_dt = pd.Timestamp(end)
    download_end_str = (end + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Track failed tickers for potential blacklisting
    failed_tickers = []

    # --- Cache Check ---
    cached_data = pd.DataFrame()
    if os.path.exists(filename):
        try:
            download_log.info(f"Reading stock price cache: '{filename}'")
            cached_data = pd.read_csv(filename)
            cached_data['Date'] = pd.to_datetime(cached_data['Date'])
            cached_data.set_index('Date', inplace=True)

            cached_start_date = cached_data.index.min()
            cached_end_date = cached_data.index.max()
            cached_symbols = set(cached_data.columns)
            requested_symbols_set = set(symbols_list)

            cache_covers_dates = (requested_start_dt >= cached_start_date) and (requested_end_dt <= cached_end_date)
            cache_covers_symbols = requested_symbols_set.issubset(cached_symbols)

            if cache_covers_dates and cache_covers_symbols:
                download_log.success(f"Cache Hit: Using '{filename}' for requested symbols.")
                return cached_data.loc[requested_start_dt:requested_end_dt, symbols_list]
            
            # If we get here, we need to download at least some data
            if not cache_covers_symbols:
                # Only download symbols not in cache
                symbols_to_download = list(requested_symbols_set - cached_symbols)
                download_log.info(f"Partial Cache Hit: Need to download {len(symbols_to_download)} new symbols.")
            else:
                # If dates aren't covered, we'd need to download all symbols for the new date range
                # Currently not handling date extension (would need to merge with careful date handling)
                symbols_to_download = symbols_list
                download_log.warning(f"Cache Miss on dates: Need to download fresh data for date range.")
        except Exception as e:
            download_log.warning(f"Could not read stock cache: {e}. Will download all symbols.")
            cached_data = pd.DataFrame()
            symbols_to_download = symbols_list
    else:
        download_log.info(f"Stock price cache '{filename}' not found. Will download all symbols.")
        symbols_to_download = symbols_list

    # --- Download Fresh Data (only for needed symbols) ---
    if not symbols_to_download:
        # This shouldn't happen given the logic above, but just in case
        return cached_data.loc[requested_start_dt:requested_end_dt, symbols_list]
        
    download_log.info(f"Downloading fresh stock price data for {len(symbols_to_download)} symbols...")
    data_dict = {}
    failed_symbols = []
    progress_bar = download_log.progress(0, text="Downloading Stock Prices...")

    for i, symbol in enumerate(symbols_to_download):
        retry_count = 0
        max_retries = 3
        retry_delay = 2  # seconds
        
        while retry_count < max_retries:
            try:
                progress_bar.progress((i + 1) / len(symbols_to_download), text=f"Downloading {symbol}...")
                
                # Use same parameters as in pca.py (no auto_adjust parameter)
                data = yf.download(
                    symbol, 
                    start=start.strftime('%Y-%m-%d'), 
                    end=download_end_str, 
                    progress=False
                )
                
                if not data.empty and 'Close' in data.columns:
                    # Filter to the exact end date requested
                    data = data[data.index <= requested_end_dt]
                    if not data.empty:
                        data_dict[symbol] = data['Close']
                        break  # Success, exit retry loop
                    else:
                        retry_count += 1
                        if retry_count < max_retries:
                            download_log.info(f"Retry {retry_count}/{max_retries} for {symbol} - empty after date filtering")
                            time.sleep(retry_delay)
                        else:
                            failed_symbols.append(symbol)
                            failed_tickers.append(symbol)  # Add to potential blacklist candidates
                            download_log.warning(f"Failed all retries for {symbol} - empty after date filtering")
                else:
                    retry_count += 1
                    if retry_count < max_retries:
                        download_log.info(f"Retry {retry_count}/{max_retries} for {symbol} - empty download")
                        time.sleep(retry_delay)
                    else:
                        failed_symbols.append(symbol)
                        failed_tickers.append(symbol)  # Add to potential blacklist candidates
                        download_log.warning(f"Failed all retries for {symbol} - empty download")
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    download_log.info(f"Retry {retry_count}/{max_retries} for {symbol}: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    failed_symbols.append(symbol)
                    failed_tickers.append(symbol)  # Add to potential blacklist candidates
                    download_log.warning(f"Failed all retries for {symbol}: {str(e)}")

        progress_bar.progress((i + 1) / len(symbols_to_download), text=f"Downloading: {symbol}")

    progress_bar.empty()
    
    # Store failed tickers in session state for blacklist management
    if failed_tickers:
        if 'recent_failed_tickers' not in st.session_state:
            st.session_state.recent_failed_tickers = set()
        st.session_state.recent_failed_tickers.update(failed_tickers)

    if not data_dict and cached_data.empty:
        download_log.error("No stock data could be downloaded. Check symbols or date range.")
        return pd.DataFrame()

    # Create dataframe with newly downloaded data
    if data_dict:
        new_prices_df = pd.concat(data_dict.values(), axis=1)
        new_prices_df.columns = data_dict.keys()
        
        if failed_symbols:
            download_log.warning(f"Failed to download price data for {len(failed_symbols)} symbols.")
    else:
        new_prices_df = pd.DataFrame()

    # --- Update Cache File with both existing and new data ---
    try:
        if not cached_data.empty and not new_prices_df.empty:
            # Combine new data with existing cache without losing any data
            # First ensure they have compatible indexes (dates)
            combined_prices = cached_data.copy()
            # Add new columns from new_prices_df
            for col in new_prices_df.columns:
                combined_prices[col] = new_prices_df[col]
            
            download_log.success(f"Combined {len(cached_data.columns)} cached symbols with {len(new_prices_df.columns)} newly downloaded symbols.")
            
            # Save the updated combined data back to CSV
            combined_prices.to_csv(filename)
            download_log.success(f"Updated stock price cache '{filename}' with combined data.")
            
            # Return only the requested symbols for the requested date range
            return combined_prices.loc[requested_start_dt:requested_end_dt, symbols_list]
        elif not new_prices_df.empty:
            # If no existing cache, just save the new data
            new_prices_df.to_csv(filename)
            download_log.success(f"Created new stock price cache '{filename}'.")
            return new_prices_df
        else:
            # If no new data but we have cache, filter the cache for the request
            filtered_cache = cached_data.loc[requested_start_dt:requested_end_dt, symbols_list]
            download_log.info(f"Using filtered cache data for {len(symbols_list)} symbols.")
            return filtered_cache
    except Exception as e:
        download_log.error(f"Failed to update stock price cache: {e}")
        
        # Even if cache update fails, try to return the data we have
        if not new_prices_df.empty:
            available_symbols = [s for s in symbols_list if s in new_prices_df.columns]
            return new_prices_df[available_symbols]
        elif not cached_data.empty:
            available_symbols = [s for s in symbols_list if s in cached_data.columns]
            return cached_data.loc[requested_start_dt:requested_end_dt, available_symbols]
        else:
            return pd.DataFrame()

# Note: Cache decorator removed to avoid CacheReplayClosureError with download_log UI elements
def load_etf_returns_cached(symbols: list, start: date, end: date, filename: str = ETF_CACHE_CSV) -> pd.DataFrame:
    """
    Loads ETF returns data with persistent CSV caching.
    Downloads new data only if cache is missing, outdated, or doesn't have requested symbols.
    """
    requested_start_dt = pd.Timestamp(start)
    requested_end_dt = pd.Timestamp(end)
    download_end_str = (end + timedelta(days=1)).strftime('%Y-%m-%d') # For yfinance
    
    # Track failed ETFs for potential blacklisting
    failed_tickers = []
    
    # --- Check Cache ---
    if os.path.exists(filename):
        try:
            download_log.info(f"Reading ETF cache: '{filename}'")
            cached_data = pd.read_csv(filename)
            cached_data['Date'] = pd.to_datetime(cached_data['Date'])
            cached_data.set_index('Date', inplace=True)
            
            cached_start_date = cached_data.index.min()
            cached_end_date = cached_data.index.max()
            cached_symbols = set(cached_data.columns)
            requested_symbols_set = set(symbols)
            
            # Conditions for using cache
            cache_covers_dates = (requested_start_dt >= cached_start_date) and (requested_end_dt <= cached_end_date)
            cache_covers_symbols = requested_symbols_set.issubset(cached_symbols)
            
            if cache_covers_dates and cache_covers_symbols:
                download_log.success(f"ETF Cache Hit: Using '{filename}' for requested ETFs")
                # Get returns from price data
                etf_prices = cached_data.loc[requested_start_dt:requested_end_dt, symbols]
                etf_returns = etf_prices.pct_change().dropna()
                return etf_returns
            else:
                download_log.warning("ETF Cache Miss: Will download fresh ETF data")
                
        except Exception as e:
            download_log.warning(f"Error reading ETF cache: {e}. Will download fresh ETF data.")
    else:
        download_log.info(f"ETF cache '{filename}' not found. Will download ETF data.")
    
    # --- Download Fresh ETF Data ---
    download_log.info(f"Downloading ETF price data for {len(symbols)} ETFs...")
    
    data_dict = {}
    failed_symbols = []
    
    # Use simpler approach like in pca.py - download one by one
    progress_bar = download_log.progress(0, text="Downloading ETFs...")
    
    for i, symbol in enumerate(symbols):
        retry_count = 0
        max_retries = 3
        retry_delay = 2  # seconds
        
        while retry_count < max_retries:
            try:
                progress_bar.progress((i + 1) / len(symbols), text=f"Downloading {symbol}...")
                
                # Use same parameters as in pca.py (no auto_adjust parameter)
                data = yf.download(
                    symbol, 
                    start=start.strftime('%Y-%m-%d'), 
                    end=download_end_str, 
                    progress=False
                )
                
                if not data.empty and len(data) > 0:
                    data_dict[symbol] = data['Close']
                    break  # Success, exit retry loop
                else:
                    retry_count += 1
                    if retry_count < max_retries:
                        download_log.info(f"Retry {retry_count}/{max_retries} for ETF {symbol} - empty download")
                        time.sleep(retry_delay)
                    else:
                        failed_symbols.append(symbol)
                        failed_tickers.append(symbol)  # Add to potential blacklist candidates
                        download_log.warning(f"Failed all retries for ETF {symbol} - empty download")
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    download_log.info(f"Retry {retry_count}/{max_retries} for ETF {symbol}: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    failed_symbols.append(symbol)
                    failed_tickers.append(symbol)  # Add to potential blacklist candidates
                    download_log.warning(f"Failed all retries for ETF {symbol}: {str(e)}")
        
        progress_bar.progress((i + 1) / len(symbols), text=f"Downloading: {symbol}")

    progress_bar.empty()
    
    # Store failed ETFs in session state for blacklist management
    if failed_tickers:
        if 'recent_failed_tickers' not in st.session_state:
            st.session_state.recent_failed_tickers = set()
        st.session_state.recent_failed_tickers.update(failed_tickers)
    
    # Check if we got any data
    if not data_dict:
        download_log.error("Failed to download any ETF data.")
        return pd.DataFrame()
    
    # Create combined dataframe
    etf_prices = pd.concat(data_dict.values(), axis=1)
    etf_prices.columns = data_dict.keys()
    
    # Filter to requested date range 
    etf_prices = etf_prices[etf_prices.index <= requested_end_dt]
    
    # Report on success/failure
    successful_symbols = list(data_dict.keys())
    download_log.success(f"Successfully downloaded {len(successful_symbols)} out of {len(symbols)} ETFs.")
    if failed_symbols:
        download_log.warning(f"Failed to download data for ETFs: {', '.join(failed_symbols)}")
    
    # --- Update Cache File ---
    try:
        etf_prices.to_csv(filename)
        download_log.success(f"Updated ETF cache '{filename}'.")
    except Exception as e:
        download_log.error(f"Failed to write ETF cache '{filename}': {e}")
    
    # Calculate returns
    etf_returns = etf_prices.pct_change().dropna()
    return etf_returns

def filter_stocks_by_data_completeness(prices: pd.DataFrame, min_completeness: float = 0.9) -> pd.DataFrame:
    """Filters stocks based on data completeness and calculates returns."""
    if prices.empty:
        return pd.DataFrame()

    completeness = prices.notna().mean() # Use mean for percentage
    stocks_to_keep = completeness[completeness >= min_completeness].index
    filtered_prices = prices[stocks_to_keep]

    n_original = len(prices.columns)
    n_filtered = len(stocks_to_keep)
    download_log.write(f"Filtered stocks by completeness (>={min_completeness*100:.0f}%): Kept {n_filtered} out of {n_original} stocks.")

    # Calculate returns and drop days with any NAs (common practice)
    returns = filtered_prices.pct_change().dropna()

    if returns.empty:
        download_log.warning("Returns data is empty after filtering and dropping NA.")
    else:
         download_log.write(f"Resulting returns matrix shape: {returns.shape} (Trading Days x Stocks)")

    return returns

@st.cache_data # Cache the PCA fitting process
def fit_pca_risk_model(returns: pd.DataFrame, n_components: int = 5, standardize: bool = False):
    """Fits PCA to returns matrix."""
    if returns.empty or n_components <= 0:
        return None

    X = returns.values
    # Optionally standardize
    if standardize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    try:
        # Ensure n_components is not larger than available features/samples
        effective_n_components = min(n_components, X.shape[0], X.shape[1])
        if effective_n_components != n_components:
             st.warning(f"Reduced n_components from {n_components} to {effective_n_components} due to data dimensions.")
             n_components = effective_n_components
             if n_components == 0: # Should not happen if returns is not empty, but check
                 return None

        pca = PCA(n_components=n_components)
        pca.fit(X)

        components = pca.components_  # shape: [K x N]
        factor_returns = X @ components.T
        reconstructed = factor_returns @ components
        residuals = X - reconstructed

        factor_names = [f"f{i+1}" for i in range(n_components)]
        exposures = pd.DataFrame(components.T, index=returns.columns, columns=factor_names)
        factor_returns = pd.DataFrame(factor_returns, index=returns.index, columns=factor_names)
        residuals = pd.DataFrame(residuals, index=returns.index, columns=returns.columns)

        return {
            "exposures": exposures,
            "factor_returns": factor_returns,
            "residuals": residuals,
            "explained_variance": pca.explained_variance_ratio_,
            "pca_object": pca # Optionally return the fitted object
        }
    except Exception as e:
        st.error(f"Error during PCA fitting: {e}")
        return None


# --- Plotting Functions (Modified to return figures) ---

def plot_pca_explained_variance_plotly(pct):
    """Creates Plotly bar/line chart for explained variance."""
    cum_pct = np.cumsum(pct)
    x = np.arange(1, len(pct) + 1)

    fig = go.Figure()

    # Bar chart for individual contribution
    fig.add_trace(go.Bar(x=x, y=pct * 100, name='Individual % Variance'))

    # Line chart for cumulative contribution
    fig.add_trace(go.Scatter(x=x, y=cum_pct * 100, mode='lines+markers', name='Cumulative % Variance', yaxis='y2'))

    fig.update_layout(
        title='PCA Explained Variance',
        xaxis_title='Principal Component',
        yaxis_title='Individual Variance (%)',
        yaxis2=dict(
            title='Cumulative Variance (%)',
            overlaying='y',
            side='right',
            range=[0, 105],
            showgrid=False
        ),
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
        template="plotly_white",
         height=400
    )
    fig.update_xaxes(tickvals=x, ticktext=[f'PC {i}' for i in x])
    return fig

@st.cache_data
def identify_factor_outliers(factor_returns: pd.DataFrame, percentile: float = 95) -> pd.DataFrame:
    """
    Identifies outlier dates in factor returns based on Mahalanobis distance.
    Uses first 3 factors to identify unusual market movements.
    """
    if factor_returns.shape[1] < 3:
        st.warning("Need at least 3 PCA components to identify outliers.")
        return pd.DataFrame()
        
    # Calculate Mahalanobis distance for each point
    data = factor_returns.iloc[:, :3].values
    
    # Calculate mean and covariance
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    cov_inv = pinv(cov)
    
    # Calculate Mahalanobis distance for each point
    distances = np.array([mahalanobis(x, mean, cov_inv) for x in data])
    
    # Find threshold for outliers
    threshold = np.percentile(distances, percentile)
    
    # Get outlier indices
    outlier_indices = np.where(distances > threshold)[0]
    
    # Create DataFrame with outlier information
    outliers = pd.DataFrame({
        'Date': factor_returns.index[outlier_indices],
        'Distance': distances[outlier_indices],
        'Factor1': data[outlier_indices, 0],
        'Factor2': data[outlier_indices, 1],
        'Factor3': data[outlier_indices, 2]
    })
    
    # Sort by distance
    outliers = outliers.sort_values('Distance', ascending=False)
    
    return outliers

def plot_3d_factors_plotly(factor_returns: pd.DataFrame):
    """Shows interactive 3D scatter plot of first 3 factor returns with outliers highlighted."""
    if factor_returns.shape[1] < 3:
        st.warning("Need at least 3 PCA components to generate 3D plot.")
        return None

    # Get outlier information
    outliers = identify_factor_outliers(factor_returns)
    
    # Create a plotly figure with two subplots
    fig = go.Figure()

    # Prepare data for visualization
    data = factor_returns.iloc[:, :3].values
    
    # Regular points (non-outliers)
    regular_mask = ~factor_returns.index.isin(outliers['Date'])
    regular_dates = factor_returns.index[regular_mask].strftime('%Y-%m-%d').tolist()
    
    # Add regular points
    fig.add_trace(go.Scatter3d(
        x=data[regular_mask, 0],
        y=data[regular_mask, 1],
        z=data[regular_mask, 2],
        mode='markers',
        marker=dict(
            size=4,
            color='blue',
            opacity=0.3
        ),
        name='Regular Points',
        text=regular_dates,
        hovertemplate="Date: %{text}<br>" +
                      "Factor 1: %{x:.4f}<br>" +
                      "Factor 2: %{y:.4f}<br>" +
                      "Factor 3: %{z:.4f}<extra></extra>"
    ))
    
    # Outlier points
    if not outliers.empty:
        outlier_mask = factor_returns.index.isin(outliers['Date'])
        outlier_dates = factor_returns.index[outlier_mask].strftime('%Y-%m-%d').tolist()
        outlier_distances = outliers['Distance'].tolist()
        
        # Create custom hover text with distances
        hover_texts = [f"Date: {d}<br>Distance: {dist:.2f}" for d, dist in zip(outlier_dates, outlier_distances)]
        
        fig.add_trace(go.Scatter3d(
            x=data[outlier_mask, 0],
            y=data[outlier_mask, 1],
            z=data[outlier_mask, 2],
            mode='markers',
            marker=dict(
                size=10,
                color='red',
                opacity=0.8
            ),
            name='Outliers',
            text=hover_texts,
            hovertemplate="%{text}<br>" +
                          "Factor 1: %{x:.4f}<br>" +
                          "Factor 2: %{y:.4f}<br>" +
                          "Factor 3: %{z:.4f}<extra></extra>"
        ))

    # Update layout
    fig.update_layout(
        title='3D Factor Returns with Outliers Highlighted',
        scene=dict(
            xaxis_title='Factor 1',
            yaxis_title='Factor 2',
            zaxis_title='Factor 3'
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        ),
        margin=dict(l=0, r=0, b=0, t=40), 
        height=600
    )

    return fig

@st.cache_data
def compute_risk_decomposition(returns: pd.DataFrame, _model: dict):
    """Calculates total, systematic, and idiosyncratic vol."""
    if not _model or returns.empty:
        return pd.DataFrame()

    F = _model["factor_returns"]
    B = _model["exposures"]
    e = _model["residuals"]

    total_vol = returns.std() * np.sqrt(252) # Annualized
    residual_vol = e.std() * np.sqrt(252)    # Annualized

    # Reconstruct systematic returns to calculate systematic vol
    # Ensure index alignment if factor_returns/exposures were modified
    common_index = F.index.intersection(returns.index) # Should be identical if derived correctly
    common_cols = B.index.intersection(returns.columns) # Should be identical

    F_aligned = F.loc[common_index]
    B_aligned = B.loc[common_cols]
    returns_aligned = returns.loc[common_index, common_cols]

    if F_aligned.empty or B_aligned.empty:
        st.error("Alignment issue in risk decomposition.")
        return pd.DataFrame()

    # Calculate systematic returns: R_systematic = F @ B.T
    # Need to handle potential shape mismatches if columns/indices don't align perfectly
    # Assuming alignment holds:
    reconstructed_returns = pd.DataFrame(F_aligned.values @ B_aligned.T.values, index=F_aligned.index, columns=B_aligned.index)

    # Align reconstructed returns with original returns for std calculation
    reconstructed_aligned, returns_aligned = reconstructed_returns.align(returns, join='inner', axis=0)
    reconstructed_aligned, returns_aligned = reconstructed_aligned.align(returns_aligned, join='inner', axis=1)

    systematic_vol = reconstructed_aligned.std() * np.sqrt(252) # Annualized

    risk_table = pd.DataFrame({
        "TotalVol (Ann.)": total_vol,
        "SystematicVol (Ann.)": systematic_vol,
        "ResidualVol (Ann.)": residual_vol,
    })

    # Calculate percentages after ensuring alignment
    risk_table["Systematic%"] = (risk_table["SystematicVol (Ann.)"]**2 / risk_table["TotalVol (Ann.)"]**2).fillna(0)
    risk_table["Residual%"] = (risk_table["ResidualVol (Ann.)"]**2 / risk_table["TotalVol (Ann.)"]**2).fillna(0)

    # Add back metadata if available (for plotting)
    # This join should happen outside the cached function if metadata changes independently
    return risk_table


def plot_risk_contributions_plotly(risk_table_merged: pd.DataFrame, group_by: str = "Industry"):
    """Plots residual vs. total risk scatter using Plotly."""
    if risk_table_merged.empty:
        st.warning("Risk table is empty, cannot generate plot.")
        return None

    if group_by not in risk_table_merged.columns:
         st.error(f"Grouping column '{group_by}' not found in the risk table. Available: {risk_table_merged.columns.tolist()}")
         # Fallback to a default or skip coloring
         color_col = None
         title_suffix = ""
    else:
        color_col = group_by
        title_suffix = f" by {group_by}"


    hover_data = ['Name', 'Symbol']
    if 'Sector' in risk_table_merged.columns: hover_data.append('Sector')
    if 'Industry' in risk_table_merged.columns: hover_data.append('Industry')
    if 'Dominant Factor (Corr)' in risk_table_merged.columns: hover_data.append('Dominant Factor (Corr)')


    fig = px.scatter(
        risk_table_merged,
        x="TotalVol (Ann.)",
        y="ResidualVol (Ann.)",
        color=color_col, # Color by selected group
        hover_data=hover_data,
        # text="Symbol", # Optionally show symbol on plot
        title=f"Residual Risk vs. Total Risk{title_suffix}",
        labels={
            "TotalVol (Ann.)": "Total Volatility (Annualized)",
            "ResidualVol (Ann.)": "Residual Volatility (Annualized)",
            "color": group_by # Legend title
        }
    )

    fig.update_traces(
        marker=dict(size=8, opacity=0.7),
        # textposition='top center' # If showing text
    )
    fig.update_layout(
        height=600,
        template="plotly_white",
        hovermode="closest",
        legend_title_text=group_by
    )
    return fig

@st.cache_data
def calculate_factor_correlations(stock_returns: pd.DataFrame, etf_returns: pd.DataFrame,
                                  factor_etfs: list):
    """Calculates correlation of each stock with factor ETFs."""
    if stock_returns.empty or etf_returns.empty or not factor_etfs:
        return pd.DataFrame()

    # Check which requested ETFs are actually available in the data
    valid_etfs = [etf for etf in factor_etfs if etf in etf_returns.columns]
    if not valid_etfs:
        st.warning("None of the specified factor ETFs were found in the loaded ETF data.")
        st.write(f"Requested ETFs: {factor_etfs}")
        st.write(f"Available ETFs: {etf_returns.columns.tolist()}")
        return pd.DataFrame()

    # Align indices
    common_index = stock_returns.index.intersection(etf_returns.index)
    if common_index.empty:
        st.warning("No common dates between stock and ETF returns for factor correlation.")
        return pd.DataFrame()

    aligned_stocks = stock_returns.loc[common_index]
    aligned_etfs = etf_returns.loc[common_index, valid_etfs]

    # Calculate correlations
    corr_matrix = pd.concat([aligned_stocks, aligned_etfs], axis=1).corr()
    stock_etf_corr = corr_matrix.loc[aligned_stocks.columns, valid_etfs]

    # Determine dominant factor
    dominant_factor = stock_etf_corr.abs().idxmax(axis=1)
    
    # Combine results
    factor_corr_df = pd.DataFrame({
        'Dominant Factor': dominant_factor,
         # Look up the correlation corresponding to the dominant factor ETF
        'Dominant Correlation': [stock_etf_corr.loc[idx, factor] for idx, factor in dominant_factor.items()]
    })

    # Add individual correlations
    factor_corr_df = factor_corr_df.join(stock_etf_corr)

    return factor_corr_df


@st.cache_data
def apply_factor_shock(_model: dict, shock_vector: np.ndarray):
    """Applies factor shock and returns predicted impacts."""
    if not _model: return pd.Series(dtype=float)
    B = _model["exposures"] # N x K
    n_factors = B.shape[1]

    # Ensure shock_vector has the correct length
    if len(shock_vector) != n_factors:
        st.error(f"Shock vector length ({len(shock_vector)}) must match number of factors ({n_factors}).")
        # Pad or truncate shock vector (use with caution)
        # Example: Pad with zeros
        padded_shock = np.zeros(n_factors)
        len_to_use = min(len(shock_vector), n_factors)
        padded_shock[:len_to_use] = shock_vector[:len_to_use]
        shock_vector = padded_shock
        # Or return empty series: return pd.Series(dtype=float)

    shock = pd.Series(shock_vector, index=B.columns)
    shocked_returns = B @ shock # (N x K) @ (K x 1) -> N x 1
    shocked_returns.name = "ShockImpact"
    return shocked_returns.sort_values() # Sort ascending for plot


def plot_shock_impact_plotly(shocked_returns_merged: pd.DataFrame, group_by: str = "Industry"):
    """Creates interactive bar plot of shock impacts using Plotly."""
    if shocked_returns_merged.empty:
        st.warning("Shocked returns data is empty, cannot generate plot.")
        return None

    if group_by not in shocked_returns_merged.columns:
         st.error(f"Grouping column '{group_by}' not found for shock plot. Available: {shocked_returns_merged.columns.tolist()}")
         color_col = None
         title_suffix = ""
    else:
        color_col = group_by
        title_suffix = f" by {group_by}"

    # Ensure sorting by impact for correct bar order
    df_plot = shocked_returns_merged.sort_values('ShockImpact')

    hover_data = ['Name', 'Symbol']
    if 'Sector' in df_plot.columns: hover_data.append('Sector')
    if 'Industry' in df_plot.columns: hover_data.append('Industry')
    if 'Dominant Factor (Corr)' in df_plot.columns: hover_data.append('Dominant Factor (Corr)')


    fig = px.bar(
        df_plot,
        x='Symbol', # Use Symbol for x-axis to avoid super long names
        y='ShockImpact',
        color=color_col,
        hover_data=hover_data,
        title=f"Predicted Return Impact from Factor Shock{title_suffix}",
        labels={
            "Symbol": "Stock Symbol",
            "ShockImpact": "Predicted Return Impact (%)", # Assuming shock implies %
             "color": group_by
        }
    )

    # Enforce the order based on sorted impact
    fig.update_layout(
        height=600,
        template="plotly_white",
        hovermode="closest",
        xaxis={'categoryorder':'array', 'categoryarray': df_plot['Symbol'].tolist()},
        legend_title_text=group_by,
        yaxis_tickformat=".2%" # Format y-axis as percentage
    )

    # Add horizontal line at y=0
    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="black")

    return fig

@st.cache_data
def analyze_factor_etf_correlation(factor_returns: pd.DataFrame, etf_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the correlation between PCA factors and ETF returns.
    Returns a DataFrame with the correlation matrix.
    """
    if etf_returns.empty:
        return pd.DataFrame()
        
    # Align indices - crucial step!
    common_index = factor_returns.index.intersection(etf_returns.index)
    if common_index.empty:
        return pd.DataFrame()

    aligned_factors = factor_returns.loc[common_index]
    aligned_etfs = etf_returns.loc[common_index]
    
    # Combine for correlation calculation
    combined_data = pd.concat([aligned_factors, aligned_etfs], axis=1).corr()
    
    # Extract the relevant part (Factors vs ETFs)
    factor_etf_corr = combined_data.loc[aligned_factors.columns, aligned_etfs.columns]
    
    return factor_etf_corr

@st.cache_data
def calculate_all_etf_correlations(stock_returns: pd.DataFrame, etf_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates correlation of each stock with ALL available ETFs.
    Returns a matrix of stock-ETF correlations.
    """
    if stock_returns.empty or etf_returns.empty:
        return pd.DataFrame()
        
    # Align indices
    common_index = stock_returns.index.intersection(etf_returns.index)
    if common_index.empty:
        return pd.DataFrame()
        
    aligned_stocks = stock_returns.loc[common_index]
    aligned_etfs = etf_returns.loc[common_index]
    
    # Calculate correlations
    corr_matrix = pd.concat([aligned_stocks, aligned_etfs], axis=1).corr()
    stock_etf_corr = corr_matrix.loc[aligned_stocks.columns, aligned_etfs.columns]
    
    return stock_etf_corr

def create_outlier_table_display(model: dict, outliers: pd.DataFrame, metadata: pd.DataFrame, top_n: int = 10):
    """
    Creates a Streamlit display of outlier dates and their top contributing stocks.
    """
    if outliers.empty:
        st.warning("No outliers found in the factor returns.")
        return
    
    st.write(f"Found {len(outliers)} outlier dates (top 5% most unusual factor movements)")
    
    # Create a dictionary for quick metadata lookup
    meta_dict = {row['Symbol']: row for _, row in metadata.iterrows()}
    
    # Show summary info for top outliers
    date_dfs = []
    for date in outliers['Date'].head(top_n):  # Show top N outlier dates
        # Get the factor values for this date
        date_factors = model["factor_returns"].loc[date]
        
        # Create a summary dictionary for this date
        date_data = {
            'Date': date.strftime('%Y-%m-%d'),
            'Distance': outliers[outliers['Date'] == date]['Distance'].values[0],
            'Factor1': date_factors['f1'],
            'Factor2': date_factors['f2'],
            'Factor3': date_factors['f3']
        }
        date_dfs.append(date_data)
    
    # Display summary table
    date_summary = pd.DataFrame(date_dfs)
    st.subheader("Top Outlier Dates")
    st.dataframe(date_summary.style.format({
        'Distance': '{:.2f}',
        'Factor1': '{:.4f}',
        'Factor2': '{:.4f}',
        'Factor3': '{:.4f}'
    }), use_container_width=True)
    
    # For the top outlier date, show contributing stocks
    if not outliers.empty:
        top_date = outliers['Date'].iloc[0]
        st.subheader(f"Stock Contributions to Top Outlier ({top_date.strftime('%Y-%m-%d')})")
        
        date_factors = model["factor_returns"].loc[top_date]
        
        # For each factor, collect top contributing stocks
        factor_summary = []
        for i, factor in enumerate(['f1', 'f2', 'f3']):
            if i >= model["exposures"].shape[1]:
                continue
                
            factor_value = date_factors[factor]
            top_stocks = model["exposures"][factor].sort_values(ascending=False).head(5)
            
            for stock, exposure in top_stocks.items():
                stock_meta = meta_dict.get(stock, {'Name': 'Unknown', 'Sector': 'Unknown', 'Industry': 'Unknown'})
                factor_summary.append({
                    'Factor': factor,
                    'Factor Value': factor_value,
                    'Symbol': stock,
                    'Name': stock_meta['Name'],
                    'Exposure': exposure,
                    'Sector': stock_meta['Sector'],
                    'Industry': stock_meta['Industry']
                })
        
        # Display as table
        factor_summary_df = pd.DataFrame(factor_summary)
        st.dataframe(factor_summary_df.style.format({
            'Factor Value': '{:.4f}',
            'Exposure': '{:.4f}'
        }), use_container_width=True)

# --- Streamlit App Layout ---

st.title("📈 PCA Risk Factor Analysis (with S&P 500 stocks)")

# Create an expander for download and cache-related messages
download_log = st.expander("💾 Data Download & Cache Log", expanded=False)

# --- Sidebar Inputs ---
st.sidebar.header("Analysis Controls")

# Blacklist Management
with st.sidebar.expander("Ticker Blacklist Management", expanded=False):
    st.write("Manage tickers to exclude from analysis (e.g., tickers with persistent download issues)")
    
    # Display current blacklist in a dropdown
    current_blacklist = load_ticker_blacklist()
    
    # Force refresh of blacklist by clearing cache for the load function
    load_ticker_blacklist.clear()
    current_blacklist = load_ticker_blacklist()
    
    if current_blacklist:
        st.write("**Current Blacklist:**")
        # Use a dropdown to view blacklisted tickers
        st.selectbox("Select ticker to view details:", current_blacklist, key="view_blacklist")
        # Also show all as comma-separated list
        st.write(", ".join(current_blacklist))
    else:
        st.info("**Blacklist is empty**")
    
    # Display recent failed tickers
    if 'recent_failed_tickers' in st.session_state and st.session_state.recent_failed_tickers:
        failed_tickers = list(st.session_state.recent_failed_tickers)
        st.write("**Recent Failed Tickers:**")
        st.write(", ".join(failed_tickers))
        
        # Add button to bulk-add failed tickers to blacklist
        new_blacklist_tickers = [t for t in failed_tickers if t not in current_blacklist]
        if new_blacklist_tickers:
            if st.button(f"Add {len(new_blacklist_tickers)} Failed Tickers to Blacklist"):
                try:
                    # Append to blacklist file
                    with open(TICKER_BLACKLIST_CSV, "a+") as f:
                        # Move to beginning of file to check if we need to add a comma
                        f.seek(0)
                        content = f.read().strip()
                        if content and not content.endswith(","):
                            f.write(", ")
                        elif not content:  # File is empty
                            pass
                        f.write(", ".join(new_blacklist_tickers))
                    st.success(f"Added {len(new_blacklist_tickers)} tickers to blacklist.")
                    # Clear the cache for blacklist function
                    load_ticker_blacklist.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to update blacklist: {e}")
    
    # Add to blacklist
    new_ticker = st.text_input("Add ticker to blacklist:", key="add_ticker").strip().upper()
    if st.button("Add to Blacklist") and new_ticker:
        if new_ticker in current_blacklist:
            st.warning(f"{new_ticker} is already in the blacklist.")
        else:
            try:
                # Append to blacklist file
                with open(TICKER_BLACKLIST_CSV, "a+") as f:
                    # Move to beginning of file to check if we need to add a comma
                    f.seek(0)
                    content = f.read().strip()
                    if content and not content.endswith(","):
                        f.write(", ")
                    elif not content:  # File is empty
                        pass  # Just write the ticker
                    f.write(new_ticker)
                st.success(f"Added {new_ticker} to blacklist.")
                # Clear the cache for blacklist function
                load_ticker_blacklist.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Failed to update blacklist: {e}")
    
    # Remove from blacklist
    if current_blacklist:
        ticker_to_remove = st.selectbox("Select ticker to remove:", current_blacklist, key="remove_ticker")
        if st.button("Remove from Blacklist") and ticker_to_remove:
            try:
                # Read current blacklist
                updated_blacklist = [t for t in current_blacklist if t != ticker_to_remove]
                
                # Write updated blacklist back to file
                with open(TICKER_BLACKLIST_CSV, "w") as f:
                    f.write(", ".join(updated_blacklist))
                
                st.success(f"Removed {ticker_to_remove} from blacklist.")
                # Clear the cache for blacklist function
                load_ticker_blacklist.clear()
                st.rerun()
            except Exception as e:
                st.error(f"Failed to update blacklist: {e}")

    # Clear failed tickers history
    if 'recent_failed_tickers' in st.session_state and st.session_state.recent_failed_tickers:
        if st.button("Clear Failed Tickers History"):
            st.session_state.recent_failed_tickers = set()
            st.success("Cleared failed tickers history.")
            st.rerun()

# Date Range
default_end_date = date.today()
# Adjust if weekend
if default_end_date.weekday() >= 5: # 5=Saturday, 6=Sunday
    default_end_date -= timedelta(days=default_end_date.weekday() - 4) # Go back to Friday

default_start_date = default_end_date - timedelta(days=3*365) # Default 3 years back
start_dt = st.sidebar.date_input("Start Date", default_start_date, max_value=default_end_date - timedelta(days=1))
end_dt = st.sidebar.date_input("End Date", default_end_date, min_value=start_dt, max_value=date.today())


# Stock Universe
all_symbols = get_sp500_symbols()
if not all_symbols:
    st.sidebar.error("Could not load S&P 500 list.")
    num_stocks = st.sidebar.slider("Number of S&P 500 Stocks", 10, 100, 50, 10, disabled=True)
    selected_symbols = []
else:
    # Sort symbols by market cap if metadata is available
    try:
        # Load metadata CSV for market cap information
        if os.path.exists(METADATA_CACHE_CSV):
            metadata_df = pd.read_csv(METADATA_CACHE_CSV)
            # Convert MarketCap to numeric and handle NaN values
            metadata_df['MarketCap'] = pd.to_numeric(metadata_df['MarketCap'], errors='coerce').fillna(0)
            
            # Filter to only include symbols that are in our S&P 500 list
            metadata_df = metadata_df[metadata_df['Symbol'].isin(all_symbols)]
            
            # Sort by MarketCap descending
            sorted_metadata = metadata_df.sort_values('MarketCap', ascending=False)
            
            # Get the sorted symbols list
            market_cap_sorted_symbols = sorted_metadata['Symbol'].tolist()
            
            # Add any symbols from all_symbols that might be missing in metadata to the end
            missing_symbols = [sym for sym in all_symbols if sym not in market_cap_sorted_symbols]
            market_cap_sorted_symbols.extend(missing_symbols)
            
            # Use the sorted list instead of the original
            all_symbols = market_cap_sorted_symbols
        else:
            st.sidebar.warning("Metadata file not found. Stocks will not be sorted by Market Cap.")
    except Exception as e:
        st.sidebar.warning(f"Could not sort stocks by Market Cap: {e}")
    
    num_stocks = st.sidebar.slider("Number of S&P 500 Stocks (Top N by Market Cap)", min_value=10, max_value=len(all_symbols), value=503, step=10)
    selected_symbols = all_symbols[:num_stocks]


# Data Filtering
min_completeness = st.sidebar.slider("Minimum Data Completeness", min_value=0.5, max_value=1.0, value=0.9, step=0.05, format="%.2f")

# PCA Settings
n_components = st.sidebar.slider("Number of PCA Components", min_value=2, max_value=20, value=5, step=1)
standardize = st.sidebar.checkbox("Standardize Returns Before PCA", value=False)

# Ranking / Display Settings
top_n_rank = st.sidebar.number_input("Top/Bottom N Stocks in Rankings", min_value=3, max_value=25, value=10, step=1)

# Stress Test Settings
with st.sidebar.expander("Stress Test Shocks (Std Dev)", expanded=False):
    shock_inputs = []
    for i in range(n_components):
        default_value = -1.0 if i == 0 else 0.0  # Set Factor 1 Shock default to -1.0
        shock = st.number_input(f"Factor {i+1} Shock", value=default_value, step=0.1, key=f"shock_{i}")
        shock_inputs.append(shock)
shock_vector = np.array(shock_inputs)

# Plot Grouping
st.sidebar.subheader("Plot Grouping")
plot_group = st.sidebar.radio("Group Scatter/Bar Plots By:", ["Industry", "Sector", "Dominant Factor"], index=0)


# --- Main Analysis Area ---

# Add a button to trigger the main analysis
if st.sidebar.button("🚀 Run Analysis"):

    # Use frozenset for symbols list passed to cached functions
    symbols_frozen = frozenset(selected_symbols)

    # 1. Load Data using cached functions
    # Metadata first, as it's simpler to update incrementally
    with download_log.container():
        st.spinner("Loading Stock Metadata (using cache)...")
        metadata = get_stock_metadata_cached(symbols_frozen, filename=METADATA_CACHE_CSV)
        if metadata.empty:
             st.error("Failed to load any metadata. Cannot proceed.")
             st.stop()
        # Ensure we only proceed with symbols for which we got metadata
        symbols_with_metadata = frozenset(metadata['Symbol'].unique())
        if len(symbols_with_metadata) < len(symbols_frozen):
             download_log.warning(f"Could only retrieve metadata for {len(symbols_with_metadata)} out of {len(symbols_frozen)} requested symbols. Proceeding with available data.")
        meta_dict = {row['Symbol']: row for _, row in metadata.iterrows()} # For lookups


    with download_log.container():
        st.spinner("Loading Stock Prices (using cache)...")
        # Pass only symbols for which metadata was loaded successfully
        stock_prices = load_stock_prices_cached(symbols_with_metadata, start_dt, end_dt, filename=STOCK_CACHE_CSV)

    if stock_prices.empty:
        st.error("Failed to load sufficient stock price data. Stopping analysis.")
        st.stop()

    # Symbols might have changed again if price download failed for some
    final_symbols_in_prices = stock_prices.columns.tolist()
    if len(final_symbols_in_prices) < len(symbols_with_metadata):
        download_log.warning(f"Price data available for only {len(final_symbols_in_prices)} symbols after download/cache check.")
        # Filter metadata again to match final price data
        metadata = metadata[metadata['Symbol'].isin(final_symbols_in_prices)].reset_index(drop=True)
        meta_dict = {row['Symbol']: row for _, row in metadata.iterrows()}


    # 2. Filter Data & Calculate Returns
    with download_log.container():
        st.spinner("Filtering Data & Calculating Returns...")
        portfolio_returns = filter_stocks_by_data_completeness(stock_prices, min_completeness)

    if portfolio_returns.empty:
        st.error("No data remaining after filtering. Check completeness threshold or data source. Stopping analysis.")
        st.stop()

    # Update metadata one last time to match the final set of stocks after filtering
    final_symbols = portfolio_returns.columns.tolist()
    metadata = metadata[metadata['Symbol'].isin(final_symbols)].reset_index(drop=True)
    meta_dict = {row['Symbol']: row for _, row in metadata.iterrows()}


    # 3. Fit PCA Model
    with st.spinner(f"Fitting PCA Model ({n_components} components)..."):
        model = fit_pca_risk_model(portfolio_returns, n_components, standardize)

    if not model:
        st.error("Failed to fit PCA model. Stopping analysis.")
        st.stop()

    # 4. Load ETF Data (for interpretation)
    etf_symbols = [
        # S&P 500 Factor ETFs
        'IVE',   # iShares S&P 500 Value ETF
        'IVW',   # iShares S&P 500 Growth ETF
        'SPLV',  # Invesco S&P 500 Low Volatility ETF
        
        # Macro ETFs
        'TLT',   # US Treasury Bonds (Interest Rate)
        'GLD',   # Gold
        'USO',   # Oil
        'HYG',   # High Yield Bonds (Credit Risk)
    ]
    with download_log.container():
        # Filter ETF symbols through blacklist
        blacklist = load_ticker_blacklist()
        if blacklist:
            filtered_etf_symbols = [sym for sym in etf_symbols if sym not in blacklist]
            skipped = set(etf_symbols) - set(filtered_etf_symbols)
            if skipped:
                download_log.info(f"Filtered out {len(skipped)} blacklisted ETFs: {', '.join(skipped)}")
            etf_symbols = filtered_etf_symbols
         
        # Ensure ETF data aligns with the actual date range of portfolio returns
        actual_start_dt = portfolio_returns.index.min().date()
        actual_end_dt = portfolio_returns.index.max().date()
        etf_returns = load_etf_returns_cached(etf_symbols, actual_start_dt, actual_end_dt, filename=ETF_CACHE_CSV)

    # Add a progress message after data loading
    st.success("✅ Data loaded successfully! Check the 'Data Download & Cache Log' expander for details.")


    # --- Display Results ---
    st.header("📊 PCA Model Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Number of Stocks Analyzed", portfolio_returns.shape[1])
        st.metric("Number of Trading Days", portfolio_returns.shape[0])
        st.metric("Number of PCA Factors", n_components)
        st.write(f"Date Range: {portfolio_returns.index.min().strftime('%Y-%m-%d')} to {portfolio_returns.index.max().strftime('%Y-%m-%d')}")
        st.write(f"Standardized Returns: {'Yes' if standardize else 'No'}")

    with col2:
        # Explained Variance Plot
        st.subheader("Explained Variance")
        fig_var = plot_pca_explained_variance_plotly(model['explained_variance'])
        st.plotly_chart(fig_var, use_container_width=True)

    # Factor Returns 3D Plot
    if n_components >= 3:
        st.subheader("Factor Returns (First 3 Components)")
        with st.spinner("Generating 3D Factor Plot..."):
            fig_3d = plot_3d_factors_plotly(model['factor_returns'])
            if fig_3d:
                st.plotly_chart(fig_3d, use_container_width=True)

    # --- Factor Exposure Analysis ---
    st.header("🔍 Factor Exposure Analysis")
    with st.expander("View Factor Exposure Rankings"):
        exposures = model['exposures'] # N x K
        st.write(f"Showing Top/Bottom {top_n_rank} stocks for each factor based on exposure:")
        for factor in exposures.columns:
            st.subheader(f"{factor.upper()} Exposure")
            ranked = exposures[factor].sort_values(ascending=False)
            ranked_df = pd.DataFrame(ranked).reset_index()
            ranked_df.columns = ['Symbol', 'Exposure']
            # Merge with metadata
            ranked_df = ranked_df.merge(metadata[['Symbol', 'Name', 'Sector', 'Industry']], on='Symbol', how='left')

            st.write(f"Top {top_n_rank} (Highest Exposure)")
            st.dataframe(ranked_df.head(top_n_rank).style.format({'Exposure': '{:.4f}'}), use_container_width=True)

            st.write(f"Bottom {top_n_rank} (Lowest Exposure / Highest Negative)")
            st.dataframe(ranked_df.tail(top_n_rank).sort_values('Exposure', ascending=True).style.format({'Exposure': '{:.4f}'}), use_container_width=True)

    # Add Outlier Analysis Table below Factor Exposure
    with st.expander("View Outlier Factor Returns Analysis"):
        st.subheader("Outlier Analysis")
        with st.spinner("Identifying factor return outliers..."):
            outliers = identify_factor_outliers(model['factor_returns'])
            create_outlier_table_display(model, outliers, metadata, top_n=10)

    # Define factor ETF mapping dictionary for interpretation
    factor_etf_map = {
        'IVE': 'Value',
        'IVW': 'Growth',
        'SPLV': 'Low Vol',
        'TLT': 'Bonds',
        'GLD': 'Gold',
        'USO': 'Oil',
        'HYG': 'High Yield'
    }

    # --- Factor Interpretation Table ---
    st.header("🔗 Factor Interpretation")
    if not etf_returns.empty:
        with st.spinner("Analyzing Factor-ETF Correlations..."):
            factor_returns = model['factor_returns']
            factor_etf_corr = analyze_factor_etf_correlation(factor_returns, etf_returns)
            
            if not factor_etf_corr.empty:
                st.subheader("Factor vs ETF Return Correlation Matrix")
                
                # Add ETF descriptions to column names for better readability
                renamed_columns = {}
                for col in factor_etf_corr.columns:
                    renamed_columns[col] = f"{col} ({factor_etf_map.get(col, col)})"
                
                factor_etf_corr_display = factor_etf_corr.copy()
                factor_etf_corr_display.columns = [renamed_columns.get(col, col) for col in factor_etf_corr.columns]
                
                # Display with formatting
                st.dataframe(factor_etf_corr_display.style.format("{:.3f}").background_gradient(cmap='coolwarm', axis=None), 
                            use_container_width=True)

                # Find most correlated ETF for each factor
                most_corr_etf = factor_etf_corr.abs().idxmax(axis=1)
                summary_corr = []
                for factor in factor_etf_corr.index:
                    etf = most_corr_etf[factor]
                    corr_val = factor_etf_corr.loc[factor, etf]
                    summary_corr.append({
                        "Factor": factor,
                        "Most Correlated ETF": f"{etf} ({factor_etf_map.get(etf, etf)})",
                        "Correlation": corr_val,
                        "Abs Correlation": abs(corr_val)
                    })
                summary_df = pd.DataFrame(summary_corr)
                st.subheader("Most Correlated ETF for each Factor")
                st.dataframe(summary_df.style.format({'Correlation': '{:.3f}', 'Abs Correlation': '{:.3f}'}), 
                            use_container_width=True)
            else:
                st.warning("No overlapping dates between factor returns and ETF returns for correlation analysis.")
    else:
        st.warning("ETF returns data not loaded. Cannot analyze factor correlations.")

    # --- Stock ETF Correlations --- (MOVED ABOVE RISK DECOMPOSITION)
    st.header("🔗 Most Correlated ETF for Each Stock")
    if not etf_returns.empty:
        with st.spinner("Analyzing Stock-ETF Correlations..."):
            # Calculate stock-ETF correlations
            with st.expander("View All Stock-ETF Correlations"):
                all_etf_corr = calculate_all_etf_correlations(portfolio_returns, etf_returns)
                
                if not all_etf_corr.empty:
                    # Calculate the most correlated ETF for each stock
                    most_corr_etf_per_stock = all_etf_corr.abs().idxmax(axis=1)
                    max_corr_values = pd.Series([all_etf_corr.loc[stock, etf] for stock, etf in most_corr_etf_per_stock.items()], 
                                               index=most_corr_etf_per_stock.index)
                    
                    # Create summary table
                    etf_corr_summary = pd.DataFrame({
                        'Stock': most_corr_etf_per_stock.index,
                        'Most Correlated ETF': [f"{etf} ({factor_etf_map.get(etf, etf)})" for etf in most_corr_etf_per_stock.values],
                        'Correlation': max_corr_values
                    })
                    
                    # Display the correlation table
                    st.dataframe(etf_corr_summary.sort_values('Correlation', ascending=False).style.format({'Correlation': '{:.3f}'}), 
                                use_container_width=True)
                    
                    # Allow download of full correlation matrix
                    st.download_button(
                        label="Download Full Stock-ETF Correlation Matrix",
                        data=all_etf_corr.to_csv(),
                        file_name="stock_etf_correlations.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("Could not calculate stock-ETF correlations.")
    else:
        st.warning("ETF returns data not loaded, skipping stock-ETF correlation analysis.")

    # --- Risk Decomposition --- (NOW AFTER FACTOR INTERPRETATION VS ETFS)
    st.header("📉 Risk Decomposition Analysis")

    # Calculate Factor Correlations first (if ETF data available)
    factor_corr_df = pd.DataFrame() # Initialize empty
    if not etf_returns.empty:
        with st.spinner("Calculating Stock-ETF Factor Correlations..."):
            # Use only the key factor ETFs for main categorization
            factor_etfs_for_classification = ['IVE', 'IVW', 'SPLV'] 
            factor_corr_df = calculate_factor_correlations(portfolio_returns, etf_returns, factor_etfs_for_classification)
            if not factor_corr_df.empty:
                 # Create a readable label for plotting/tables
                 factor_corr_df['Dominant Factor (Corr)'] = factor_corr_df.apply(
                     lambda row: f"{factor_etf_map.get(row['Dominant Factor'], row['Dominant Factor'])} ({row['Dominant Correlation']:.2f})", axis=1
                 )
                 st.success("Calculated stock correlations with Value, Growth, Low Vol ETFs.")
            else:
                 st.warning("Could not calculate factor correlations.")


    with st.spinner("Calculating Risk Decomposition..."):
        risk_table = compute_risk_decomposition(portfolio_returns, model)

    if not risk_table.empty:
        # Merge risk table with metadata and factor correlations for plotting
        risk_table_merged = risk_table.merge(metadata[['Symbol', 'Name', 'Sector', 'Industry']], left_index=True, right_on='Symbol', how='left')
        if not factor_corr_df.empty:
             risk_table_merged = risk_table_merged.merge(factor_corr_df[['Dominant Factor (Corr)']], left_on='Symbol', right_index=True, how='left')
             risk_table_merged['Dominant Factor (Corr)'] = risk_table_merged['Dominant Factor (Corr)'].fillna('N/A') # Handle stocks potentially missing factor corr


        # Add the plot grouping column based on user selection
        if plot_group == "Dominant Factor" and 'Dominant Factor (Corr)' in risk_table_merged.columns:
            plot_col = 'Dominant Factor (Corr)'
        elif plot_group == "Sector" and 'Sector' in risk_table_merged.columns:
             plot_col = 'Sector'
        else: # Default to Industry
             plot_col = 'Industry'

        fig_risk = plot_risk_contributions_plotly(risk_table_merged, group_by=plot_col)
        if fig_risk:
            st.plotly_chart(fig_risk, use_container_width=True)

        with st.expander("View Risk Decomposition Table"):
             display_cols = ['Symbol', 'Name', 'Sector', 'Industry', 'TotalVol (Ann.)', 'SystematicVol (Ann.)', 'ResidualVol (Ann.)', 'Systematic%', 'Residual%']
             if 'Dominant Factor (Corr)' in risk_table_merged.columns:
                  display_cols.insert(4, 'Dominant Factor (Corr)') # Add factor column
             st.dataframe(risk_table_merged[display_cols].style.format({
                'TotalVol (Ann.)': '{:.2%}',
                'SystematicVol (Ann.)': '{:.2%}',
                'ResidualVol (Ann.)': '{:.2%}',
                'Systematic%': '{:.1%}',
                'Residual%': '{:.1%}'
             }), use_container_width=True)
    else:
        st.warning("Could not compute risk decomposition.")


    # --- Stress Testing ---
    st.header("⚡ Stress Testing")
    st.write(f"Applying Shock Vector: {shock_vector.round(2).tolist()}")

    if np.any(shock_vector): # Only run if shock is non-zero
        with st.spinner("Applying Factor Shocks..."):
            shocked_returns = apply_factor_shock(model, shock_vector)

        if not shocked_returns.empty:
             # Merge with metadata and factor correlations for plotting
             shocked_returns_df = pd.DataFrame(shocked_returns).reset_index()
             shocked_returns_df.columns = ['Symbol', 'ShockImpact']
             shocked_returns_merged = shocked_returns_df.merge(metadata[['Symbol', 'Name', 'Sector', 'Industry']], on='Symbol', how='left')
             if not factor_corr_df.empty:
                 shocked_returns_merged = shocked_returns_merged.merge(factor_corr_df[['Dominant Factor (Corr)']], left_on='Symbol', right_index=True, how='left')
                 shocked_returns_merged['Dominant Factor (Corr)'] = shocked_returns_merged['Dominant Factor (Corr)'].fillna('N/A')

             # Determine plot grouping column
             if plot_group == "Dominant Factor" and 'Dominant Factor (Corr)' in shocked_returns_merged.columns:
                 plot_col_shock = 'Dominant Factor (Corr)'
             elif plot_group == "Sector" and 'Sector' in shocked_returns_merged.columns:
                 plot_col_shock = 'Sector'
             else: # Default to Industry
                 plot_col_shock = 'Industry'

             fig_shock = plot_shock_impact_plotly(shocked_returns_merged, group_by=plot_col_shock)
             if fig_shock:
                 st.plotly_chart(fig_shock, use_container_width=True)

             with st.expander("View Stress Test Impact Table"):
                 display_cols_shock = ['Symbol', 'Name', 'Sector', 'Industry', 'ShockImpact']
                 if 'Dominant Factor (Corr)' in shocked_returns_merged.columns:
                      display_cols_shock.insert(4, 'Dominant Factor (Corr)')
                 st.dataframe(shocked_returns_merged[display_cols_shock].sort_values('ShockImpact').style.format({'ShockImpact': '{:.2%}'}), use_container_width=True)
        else:
             st.warning("Could not compute shock impacts.")
    else:
        st.info("Shock vector is all zeros. No stress test plot generated.")

else:
    st.info("Adjust settings in the sidebar and click 'Run Analysis' to generate the report.")

st.sidebar.markdown("---")
with st.sidebar.expander("Session Information", expanded=False):
    # Display session state information
    st.write("**Current Settings:**")
    st.write(f"• Date Range: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
    st.write(f"• Number of Stocks: {num_stocks}")
    st.write(f"• Data Completeness: {min_completeness:.0%}")
    st.write(f"• PCA Components: {n_components}")
    st.write(f"• Standardize Returns: {'Yes' if standardize else 'No'}")
    st.write(f"• Plot Grouping: {plot_group}")
    
    # Show shock vector if any non-zero values
    if np.any(shock_vector):
        st.write("**Active Shock Vector:**")
        for i, shock in enumerate(shock_vector):
            if shock != 0:
                st.write(f"• Factor {i+1}: {shock:.1f} std")
