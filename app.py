import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import time

try:
    from scipy.stats import ttest_1samp, wilcoxon
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ==========================================================================================
# 0. CẤU HÌNH TRANG & GIAO DIỆN
# ==========================================================================================
st.set_page_config(
    page_title="Tối ưu hóa Danh mục HOSE | MACD + RSI + PSO",
    layout="wide",
    page_icon="📈"
)

# Custom CSS for Premium Dark UI
st.markdown("""
<style>
    /* Styling headers */
    .main-header {
        font-size: 36px;
        font-weight: 800;
        background: linear-gradient(135deg, #10b981 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        text-align: center;
    }
    .sub-header {
        font-size: 16px;
        color: #9ca3af;
        text-align: center;
        margin-bottom: 25px;
    }
    /* Cards and metrics */
    .stCard {
        background-color: #1f2937;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #374151;
        margin-bottom: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .metric-value {
        font-size: 28px;
        font-weight: 800;
        color: #10b981;
    }
    .metric-label {
        font-size: 14px;
        color: #9ca3af;
        font-weight: 500;
        margin-bottom: 5px;
    }
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 16px;
        background-color: #1f2937;
        border-radius: 8px 8px 0px 0px;
        color: #9ca3af;
        border: 1px solid #374151;
    }
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6 !important;
        color: white !important;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Ứng Dụng Tối Ưu Hóa Danh Mục Đầu Tư Chứng Khoán</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Chiến lược kết hợp MACD + RSI & Tối ưu hóa tham số Bầy đàn (PSO) trên HOSE</div>', unsafe_allow_html=True)

# Hằng số mặc định
DEFAULT_DATA_PATH = "HOSE_2020_2023.csv"
TRADING_DAYS     = 252
SEED             = 42

# Miền tìm kiếm tham số cho PSO
PARAM_BOUNDS = [
    ("fast",        5,   30,  True),   # EMA nhanh của MACD
    ("slow",       20,   60,  True),   # EMA chậm của MACD
    ("sign",        5,   20,  True),   # EMA đường tín hiệu MACD
    ("rsi_window",  7,   25,  True),   # chu kỳ RSI
    ("lower",      20,   50,  False),  # ngưỡng RSI dưới (lọc khi mua)
    ("upper",      65,   90,  False),  # ngưỡng RSI trên (chốt lời khi quá mua)
]

# ==========================================================================================
# 1. SIDEBAR CẤU HÌNH THAM SỐ
# ==========================================================================================
st.sidebar.header("⚙️ Cấu hình Tham số")

# Tải file dữ liệu
uploaded_file = st.sidebar.file_uploader("📁 Tải lên file CSV dữ liệu HOSE:", type=["csv"])

# Vốn & Phí giao dịch
st.sidebar.subheader("💰 Cấu hình Vốn & Giao dịch")
initial_capital = st.sidebar.number_input(
    "Vốn đầu tư ban đầu (VND):",
    min_value=10_000_000,
    max_value=100_000_000_000,
    value=1_000_000_000,
    step=50_000_000,
    format="%d"
)
fee_rate = st.sidebar.slider("Phí giao dịch mỗi chiều (%):", min_value=0.0, max_value=1.0, value=0.15, step=0.01) / 100
rf_annual = st.sidebar.slider("Lãi suất phi rủi ro năm (%):", min_value=0.0, max_value=15.0, value=4.0, step=0.5) / 100
rf_daily = rf_annual / TRADING_DAYS

# Phương thức Tối ưu hóa Tham số
st.sidebar.subheader("🤖 Tối ưu hóa Tham số (MACD + RSI)")
opt_method = st.sidebar.radio(
    "Chọn phương thức cấu hình tham số:",
    options=["Sử dụng tham số tối ưu sẵn", "Chạy thuật toán PSO live", "Điều chỉnh thủ công"]
)

# Biến lưu trữ tham số
best_p = {}

if opt_method == "Điều chỉnh thủ công":
    st.sidebar.markdown("**Thông số MACD & RSI:**")
    best_p["fast"] = st.sidebar.slider("MACD Fast Span (nhanh):", 5, 30, 10)
    best_p["slow"] = st.sidebar.slider("MACD Slow Span (chậm):", 20, 60, 20)
    best_p["sign"] = st.sidebar.slider("MACD Signal Span (tín hiệu):", 5, 20, 15)
    best_p["rsi_window"] = st.sidebar.slider("RSI Window (chu kỳ):", 7, 25, 15)
    best_p["lower"] = st.sidebar.slider("RSI Lower Limit (ngưỡng dưới):", 20.0, 50.0, 31.3)
    best_p["upper"] = st.sidebar.slider("RSI Upper Limit (ngưỡng trên):", 65.0, 90.0, 85.0)
    
    # Sửa lỗi chéo logic
    if best_p["slow"] <= best_p["fast"]:
        best_p["slow"] = best_p["fast"] + 1
    if best_p["upper"] <= best_p["lower"]:
        best_p["upper"] = best_p["lower"] + 5
        
elif opt_method == "Chạy thuật toán PSO live":
    pso_particles = st.sidebar.slider("Số lượng hạt trong bầy (Particles):", 10, 50, 24)
    pso_iters = st.sidebar.slider("Số vòng lặp (Iterations):", 10, 100, 35)
    pso_seed = st.sidebar.number_input("Hạt giống ngẫu nhiên (Seed):", value=SEED)
else:
    # Sử dụng bộ tham số tối ưu hóa mặc định trong báo cáo
    best_p = {
        "fast": 10,
        "slow": 20,
        "sign": 15,
        "rsi_window": 15,
        "lower": 31.3,
        "upper": 85.0
    }
    st.sidebar.info(
        f"Mặc định (In-sample 2020):\n"
        f"- MACD: {best_p['fast']}/{best_p['slow']}/{best_p['sign']}\n"
        f"- RSI: {best_p['rsi_window']} (Ngưỡng: {best_p['lower']:.1f} - {best_p['upper']:.1f})"
    )

# Cấu hình danh mục
st.sidebar.subheader("🧺 Cấu hình Danh mục Out-of-Sample")
portfolio_size = st.sidebar.slider("Số lượng cổ phiếu trong rổ (K):", 3, 8, 5)
weight_scheme = st.sidebar.selectbox("Cách phân bổ tỷ trọng:", options=["equal", "performance"], format_func=lambda x: "Tỷ trọng Đều (Equal)" if x=="equal" else "Tỷ trọng theo Hiệu suất (Performance)")
rebalance_freq = st.sidebar.selectbox("Tần suất tái cân bằng:", options=["none", "monthly", "quarterly", "annual"], format_func=lambda x: {"none": "Không tái cân bằng", "monthly":"Hằng tháng (Monthly)", "quarterly":"Hằng quý (Quarterly)", "annual":"Hằng năm (Annual)"}[x])


# ==========================================================================================
# 2. ĐỌC DỮ LIỆU & CHỈ BÁO KỸ THUẬT (Đã tối ưu/cached)
# ==========================================================================================
@st.cache_data
def load_data(source):
    """
    Đọc dữ liệu từ file path hoặc file upload.
    Trả về DataFrame close và open.
    """
    try:
        if isinstance(source, str):
            df = pd.read_csv(source, low_memory=False)
        else:
            # File upload
            df = pd.read_csv(source, low_memory=False)
            
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
        df["date"]   = pd.to_datetime(df["date"])
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
        df = df.sort_values(["ticker", "date"])
        
        close = df.pivot(index="date", columns="ticker", values="adj_close").sort_index()
        open_ = df.pivot(index="date", columns="ticker", values="adj_open").sort_index()
        return close, open_
    except Exception as e:
        st.error(f"Lỗi khi đọc file dữ liệu: {e}")
        return None, None

# Chọn nguồn dữ liệu
data_source = uploaded_file if uploaded_file is not None else DEFAULT_DATA_PATH

close_all, open_all = load_data(data_source)

if close_all is None or open_all is None:
    st.warning("⚠️ Không thể tải dữ liệu. Vui lòng kiểm tra lại file CSV hoặc tải lên file hợp lệ.")
    st.stop()

# Tách dữ liệu VNINDEX và các cổ phiếu
STOCKS = [c for c in close_all.columns if c != "VNINDEX"]
is_mask = (close_all.index <= "2020-12-31")
close_is, open_is = close_all.loc[is_mask, STOCKS], open_all.loc[is_mask, STOCKS]
close_oos, open_oos = close_all.loc[~is_mask, STOCKS], open_all.loc[~is_mask, STOCKS]

# ==========================================================================================
# 3. ĐỊNH NGHĨA PHƯƠNG THỨC TÍNH TOÁN
# ==========================================================================================
def compute_macd(close, fast, slow, sign):
    macd_line   = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=sign, adjust=False).mean()
    return macd_line, signal_line

def compute_rsi(close, window):
    delta = close.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs  = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def rsi_matrix(close_df, window):
    delta = close_df.diff()
    gain  = delta.clip(lower=0)
    loss  = -delta.clip(upper=0)
    ag = gain.rolling(window=window, min_periods=window).mean()
    al = loss.rolling(window=window, min_periods=window).mean()
    rs = ag / al.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)

def desired_pos_matrix(close_df, p):
    macd = close_df.ewm(span=p["fast"], adjust=False).mean() - close_df.ewm(span=p["slow"], adjust=False).mean()
    sig  = macd.ewm(span=p["sign"], adjust=False).mean()
    rsi  = rsi_matrix(close_df, p["rsi_window"])
    macd_above = macd > sig
    bear_cross = (~macd_above) & macd_above.shift(1).fillna(False)
    buy  = macd_above & (rsi > p["lower"])
    sell = bear_cross | (rsi > p["upper"])
    pos  = pd.DataFrame(np.nan, index=close_df.index, columns=close_df.columns)
    pos  = pos.mask(buy, 1.0).mask(sell, 0.0).ffill().fillna(0.0)
    return pos

def macd_rsi_signals(close_series, p):
    macd_line, signal_line = compute_macd(close_series, p["fast"], p["slow"], p["sign"])
    rsi = compute_rsi(close_series, p["rsi_window"])
    macd_above = macd_line > signal_line
    bear_cross = (~macd_above) & macd_above.shift(1).fillna(False)
    buy  = macd_above & (rsi > p["lower"])
    sell = bear_cross | (rsi > p["upper"])
    pos  = pd.Series(np.nan, index=close_series.index)
    pos  = pos.mask(buy, 1.0).mask(sell, 0.0).ffill().fillna(0.0)
    return pos, buy, sell

def backtest_single(open_arr, close_arr, desired_pos, capital=initial_capital, fee=fee_rate):
    n = len(close_arr)
    hold = np.zeros(n)
    hold[1:] = desired_pos[:-1]
    prev = np.zeros(n); prev[1:] = hold[:-1]
    o, c = open_arr, close_arr
    Cprev = np.roll(c, 1)
    enter = (hold == 1) & (prev == 0)
    both  = (hold == 1) & (prev == 1)
    exit_ = (hold == 0) & (prev == 1)
    factor = np.ones(n)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor[enter] = (c[enter] / o[enter]) * (1 - fee)
        factor[both]  = c[both] / Cprev[both]
        factor[exit_] = (o[exit_] / Cprev[exit_]) * (1 - fee)
    factor[0] = 1.0
    equity   = capital * np.cumprod(factor)
    n_trades = int(enter.sum() + exit_.sum())
    return equity, n_trades

def backtest_matrix(open_df, close_df, dp_df, capital=initial_capital, fee=fee_rate):
    O, C, DP = open_df.values, close_df.values, dp_df.values
    hold = np.zeros_like(C); hold[1:] = DP[:-1]
    prev = np.zeros_like(C); prev[1:] = hold[:-1]
    Cprev = np.roll(C, 1, axis=0)
    enter = (hold == 1) & (prev == 0)
    both  = (hold == 1) & (prev == 1)
    exit_ = (hold == 0) & (prev == 1)
    factor = np.ones_like(C)
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = np.where(enter, C / O * (1 - fee), factor)
        factor = np.where(both,  C / Cprev,         factor)
        factor = np.where(exit_, O / Cprev * (1 - fee), factor)
    factor[0, :] = 1.0
    equity = capital * np.cumprod(factor, axis=0)
    ntr = (enter.sum(axis=0) + exit_.sum(axis=0)).astype(int)
    return equity, ntr

def perf_metrics(equity, n_trades=None, initial=initial_capital,
                 rf_daily=rf_daily, periods=TRADING_DAYS):
    equity = np.asarray(equity, dtype=float)
    ret = equity[1:] / equity[:-1] - 1.0
    total_return = equity[-1] / initial - 1.0
    n = len(equity); years = n / periods
    cagr = (equity[-1] / initial) ** (1 / years) - 1.0 if equity[-1] > 0 else -1.0
    vol  = ret.std(ddof=1) * np.sqrt(periods) if len(ret) > 1 else np.nan
    excess = ret - rf_daily
    sd = ret.std(ddof=1)
    sharpe = np.sqrt(periods) * excess.mean() / sd if sd > 0 else np.nan
    downside = np.minimum(excess, 0.0)
    dd_dev = np.sqrt((downside ** 2).mean())
    sortino = np.sqrt(periods) * excess.mean() / dd_dev if dd_dev > 0 else np.nan
    run_max = np.maximum.accumulate(equity)
    mdd = (equity / run_max - 1.0).min()
    calmar = cagr / abs(mdd) if mdd < 0 else np.nan
    out = {
        "Total Return [%]": total_return * 100, "CAGR [%]": cagr * 100,
        "Volatility [%]": vol * 100, "Sharpe": sharpe, "Sortino": sortino,
        "Max Drawdown [%]": mdd * 100, "Calmar": calmar, "Final Value": equity[-1],
    }
    if n_trades is not None:
        out["Trades"] = n_trades
    return out

def sharpe_vector(equity, rf_daily=rf_daily, periods=TRADING_DAYS):
    ret = equity[1:] / equity[:-1] - 1.0
    excess = ret - rf_daily
    sd = ret.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(periods) * excess.mean(axis=0) / sd

def decode(vec):
    p = {}
    for val, (name, lo, hi, is_int) in zip(vec, PARAM_BOUNDS):
        v = lo + (hi - lo) * val
        p[name] = int(round(v)) if is_int else float(v)
    if p["slow"]  <= p["fast"]:  p["slow"]  = p["fast"]  + 1
    if p["upper"] <= p["lower"]: p["upper"] = p["lower"] + 5
    return p

# ==========================================================================================
# 4. THIẾT LẬP TABS TRÊN WEBSITE
# ==========================================================================================
tab_overview, tab_opt, tab_backtest, tab_stat, tab_signal = st.tabs([
    "📊 Dữ liệu & Tổng quan",
    "⚙️ Tối ưu hóa PSO",
    "📈 Backtest & Hiệu suất",
    "🔬 Kiểm định Thống kê",
    "🔍 Biểu đồ Tín hiệu kỹ thuật"
])

# ------------------------------------------------------------------------------------------
# TAB 1: DỮ LIỆU & TỔNG QUAN
# ------------------------------------------------------------------------------------------
with tab_overview:
    st.markdown("### 📊 Tổng quan tập dữ liệu")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class="stCard">
            <div class="metric-label">Tổng số cổ phiếu phân tích</div>
            <div class="metric-value">{len(STOCKS)} mã</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stCard">
            <div class="metric-label">Số ngày giao dịch (Toàn kỳ)</div>
            <div class="metric-value">{len(close_all.index)} phiên</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="stCard">
            <div class="metric-label">Thời gian dữ liệu</div>
            <div class="metric-value" style="font-size: 20px; padding-top: 5px;">
                {close_all.index[0].strftime('%d/%m/%Y')} → {close_all.index[-1].strftime('%d/%m/%Y')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("#### Xem trước bảng giá đóng cửa đã điều chỉnh (Adj Close Price)")
    st.dataframe(close_all.head(10), use_container_width=True)
    
    # Biểu đồ mô tả phân bổ số lượng bản ghi hoặc biến động VN-Index
    if "VNINDEX" in close_all.columns:
        st.markdown("#### Diễn biến VN-Index toàn kỳ (2020 - 2023)")
        fig_vnindex = px.line(
            close_all, 
            y="VNINDEX", 
            title="Biến động VN-Index",
            labels={"date": "Ngày", "VNINDEX": "Điểm số"},
            color_discrete_sequence=["#3b82f6"]
        )
        fig_vnindex.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_vnindex, use_container_width=True)

# ------------------------------------------------------------------------------------------
# TAB 2: TỐI ƯU HÓA PSO & XẾP HẠNG
# ------------------------------------------------------------------------------------------
with tab_opt:
    st.markdown("### 🤖 Giai đoạn Học (In-sample 2020) & Thuật toán PSO")
    
    # Xếp hạng in-sample
    @st.cache_data
    def rank_stocks_in_sample(p_params):
        dp = desired_pos_matrix(close_is, p_params)
        eqM, ntrM = backtest_matrix(open_is, close_is, dp)
        rows = []
        for j, tk in enumerate(STOCKS):
            m = perf_metrics(eqM[:, j], int(ntrM[j]), initial=initial_capital)
            rows.append({
                "ticker": tk, 
                "IS_Return%": m["Total Return [%]"], 
                "IS_Sharpe": m["Sharpe"],
                "IS_Sortino": m["Sortino"], 
                "IS_MaxDD%": m["Max Drawdown [%]"], 
                "IS_Trades": int(ntrM[j])
            })
        return pd.DataFrame(rows).sort_values("IS_Sharpe", ascending=False).reset_index(drop=True)

    if opt_method == "Chạy thuật toán PSO live":
        st.markdown("#### 🚀 Chạy Tối ưu hóa PSO Live")
        st.write("Thuật toán PSO sẽ tìm kiếm bộ tham số tối ưu cho chỉ báo MACD & RSI nhằm cực đại hóa Sharpe Ratio trung bình trên toàn thị trường trong năm 2020.")
        
        run_pso = st.button("▶️ Bắt đầu Tối ưu hóa PSO")
        
        if run_pso:
            progress_bar = st.progress(0)
            log_container = st.empty()
            
            # Hàm thích nghi
            def fitness(p):
                dp = desired_pos_matrix(close_is, p)
                eq, _ = backtest_matrix(open_is, close_is, dp)
                sh = sharpe_vector(eq)
                sh = sh[np.isfinite(sh)]
                return float(sh.mean()) if len(sh) else -9.0
            
            # PSO implementation
            rng = np.random.default_rng(pso_seed)
            dim = len(PARAM_BOUNDS)
            X = rng.random((pso_particles, dim))
            V = rng.uniform(-0.1, 0.1, (pso_particles, dim))
            pbest = X.copy()
            
            pbest_val = np.array([fitness(decode(x)) for x in X])
            g = int(np.argmax(pbest_val))
            gbest = pbest[g].copy()
            gbest_val = pbest_val[g]
            
            pso_logs = []
            
            for it in range(pso_iters):
                r1 = rng.random((pso_particles, dim))
                r2 = rng.random((pso_particles, dim))
                V = 0.7 * V + 1.5 * r1 * (pbest - X) + 1.5 * r2 * (gbest - X)
                V = np.clip(V, -0.3, 0.3)
                X = np.clip(X + V, 0.0, 1.0)
                
                vals = np.array([fitness(decode(x)) for x in X])
                imp = vals > pbest_val
                pbest[imp] = X[imp]
                pbest_val[imp] = vals[imp]
                
                if pbest_val.max() > gbest_val:
                    g = int(np.argmax(pbest_val))
                    gbest = pbest[g].copy()
                    gbest_val = pbest_val[g]
                
                progress = (it + 1) / pso_iters
                progress_bar.progress(progress)
                
                log_text = f"Vòng lặp {it+1:2d}/{pso_iters}: Sharpe trung bình tốt nhất = {gbest_val:.4f}"
                pso_logs.append(log_text)
                log_container.text("\n".join(pso_logs[-10:]))  # Show last 10 lines
                time.sleep(0.02)
                
            best_p = decode(gbest)
            st.session_state["best_p"] = best_p
            st.session_state["best_v"] = gbest_val
            st.success("🎉 Đã hoàn thành thuật toán PSO!")
            
        if "best_p" in st.session_state:
            best_p = st.session_state["best_p"]
            gbest_val = st.session_state["best_v"]
            
            col_res1, col_res2 = st.columns(2)
            with col_res1:
                st.markdown("##### 📌 Bộ tham số tối ưu tìm được:")
                st.json(best_p)
            with col_res2:
                st.metric("Sharpe trung bình (In-sample 2020)", f"{gbest_val:.4f}")
                
    else:
        st.markdown("#### 📌 Cấu hình Tham số hiện tại")
        col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
        col_m1.metric("EMA nhanh (Fast)", best_p["fast"])
        col_m2.metric("EMA chậm (Slow)", best_p["slow"])
        col_m3.metric("Tín hiệu (Signal)", best_p["sign"])
        col_m4.metric("Chu kỳ RSI", best_p["rsi_window"])
        col_m5.metric("RSI ngưỡng dưới", f"{best_p['lower']:.1f}")
        col_m6.metric("RSI ngưỡng trên", f"{best_p['upper']:.1f}")

    # Xếp hạng in-sample và hiển thị
    if len(best_p) > 0:
        st.markdown("---")
        st.markdown("#### 🏆 Xếp hạng cổ phiếu HOSE theo Sharpe Ratio (In-sample 2020)")
        
        rank_df = rank_stocks_in_sample(best_p)
        top5_list = rank_df.head(portfolio_size)["ticker"].tolist()
        
        st.write(f"👉 **Rổ {portfolio_size} cổ phiếu tốt nhất được chọn:** {', '.join(top5_list)}")
        
        fig_rank = px.bar(
            rank_df.head(20),
            x="ticker",
            y="IS_Sharpe",
            title=f"Top 20 cổ phiếu có Sharpe Ratio in-sample cao nhất",
            labels={"ticker": "Mã cổ phiếu", "IS_Sharpe": "Sharpe Ratio 2020"},
            color="IS_Sharpe",
            color_continuous_scale=px.colors.sequential.Plotly3
        )
        fig_rank.update_layout(template="plotly_dark", height=400)
        st.plotly_chart(fig_rank, use_container_width=True)
        
        st.dataframe(rank_df.round(3), use_container_width=True)

# ------------------------------------------------------------------------------------------
# TAB 3: BACKTEST & HIỆU SUẤT OUT-OF-SAMPLE
# ------------------------------------------------------------------------------------------
with tab_backtest:
    st.markdown("### 📈 Hiệu suất Danh mục đầu tư Out-of-sample (2021 - 2023)")
    
    if len(best_p) == 0:
        st.info("💡 Vui lòng thiết lập hoặc chạy tối ưu hóa PSO ở Tab 'Tối ưu hóa PSO' trước.")
    else:
        # Xếp hạng in-sample để chọn rổ
        rank_df = rank_stocks_in_sample(best_p)
        basket = rank_df.head(portfolio_size)["ticker"].tolist()
        is_ret_dict = {row["ticker"]: row["IS_Return%"] for _, row in rank_df.iterrows()}
        
        # Hàm chạy danh mục
        def rebalance_indices(dates, freq, oos_start_pos):
            if freq == "none":
                return set()
            idx = []
            for pos in range(oos_start_pos + 1, len(dates)):
                d, prev = dates[pos], dates[pos - 1]
                if freq == "monthly"   and (d.year, d.month) != (prev.year, prev.month):
                    idx.append(pos)
                elif freq == "quarterly" and (d.year, (d.month - 1)//3) != (prev.year, (prev.month - 1)//3):
                    idx.append(pos)
                elif freq == "annual"  and d.year != prev.year:
                    idx.append(pos)
            return set(idx)

        def run_portfolio(close_df, open_df, stocks, params, weight_scheme, rebalance,
                          init_perf, capital=initial_capital, fee=fee_rate, oos_start="2021-01-04"):
            dates = close_df.index
            N = len(stocks)
            dp = desired_pos_matrix(close_df[stocks], params).values
            C  = close_df[stocks].values
            O  = open_df[stocks].values
            
            # Check if oos_start exists in index, find closest if not
            if pd.Timestamp(oos_start) in dates:
                oos_pos = list(dates).index(pd.Timestamp(oos_start))
            else:
                oos_pos = np.searchsorted(dates, pd.Timestamp(oos_start))
                
            rebal   = rebalance_indices(dates, rebalance, oos_pos)

            init_perf_vec = np.array([init_perf[tk] for tk in stocks], dtype=float)
            w0 = compute_weights(weight_scheme, init_perf_vec, N)
            cash   = w0 * capital
            shares = np.zeros(N)
            inpos  = np.zeros(N, dtype=bool)
            last_rebal_value = w0 * capital

            eq_dates, eq_vals = [], []
            total_trades = 0
            total_fee_paid = 0.0

            for pos in range(oos_pos, len(dates)):
                o_t, c_t = O[pos], C[pos]
                regime_prev = dp[pos - 1]

                # (1) Khớp lệnh theo tín hiệu
                for i in range(N):
                    if regime_prev[i] == 1 and not inpos[i]:
                        if cash[i] > 0:
                            f = cash[i] * fee
                            shares[i] = (cash[i] - f) / o_t[i]
                            total_fee_paid += f; cash[i] = 0.0; total_trades += 1
                        inpos[i] = True
                    elif regime_prev[i] == 0 and inpos[i]:
                        proceeds = shares[i] * o_t[i]
                        f = proceeds * fee
                        cash[i] = proceeds - f
                        total_fee_paid += f; shares[i] = 0.0; inpos[i] = False; total_trades += 1

                # (2) Tái cân bằng định kỳ
                if pos in rebal:
                    sleeve_val = cash + shares * c_t
                    V = sleeve_val.sum()
                    if weight_scheme == "performance":
                        growth = sleeve_val / np.where(last_rebal_value > 0, last_rebal_value, np.nan) - 1.0
                        growth = np.nan_to_num(growth, nan=0.0)
                        w = compute_weights("performance", growth, N)
                    else:
                        w = compute_weights("equal", None, N)
                    tgt = w * V
                    reb_fee = sum(fee * abs(tgt[i] - shares[i] * c_t[i]) for i in range(N) if inpos[i])
                    scale = (V - reb_fee) / V if V > 0 else 1.0
                    for i in range(N):
                        if inpos[i]:
                            shares[i] = (tgt[i] * scale) / c_t[i]; cash[i] = 0.0
                        else:
                            cash[i] = tgt[i] * scale; shares[i] = 0.0
                    total_fee_paid += reb_fee
                    last_rebal_value = tgt * scale

                eq_dates.append(dates[pos])
                eq_vals.append(cash.sum() + (shares * c_t).sum())

            equity = pd.Series(eq_vals, index=eq_dates)
            m = perf_metrics(equity.values, total_trades, initial=capital)
            m["Fees Paid"] = total_fee_paid
            return equity, m

        def compute_weights(scheme, perf_vector, n):
            if scheme == "equal":
                return np.full(n, 1.0 / n)
            pos = np.clip(perf_vector, 0.0, None)
            return np.full(n, 1.0 / n) if pos.sum() <= 0 else pos / pos.sum()

        def buy_hold_basket(close_df, open_df, stocks, weights=None, capital=initial_capital,
                            fee=fee_rate, oos_start="2021-01-04"):
            dates = close_df.index
            if pd.Timestamp(oos_start) in dates:
                oos_pos = list(dates).index(pd.Timestamp(oos_start))
            else:
                oos_pos = np.searchsorted(dates, pd.Timestamp(oos_start))
                
            N = len(stocks)
            if weights is None:
                weights = np.full(N, 1.0 / N)
            o0 = open_df[stocks].values[oos_pos]
            shares = (weights * capital * (1 - fee)) / o0
            C = close_df[stocks].values[oos_pos:]
            equity = pd.Series((C * shares).sum(axis=1), index=dates[oos_pos:])
            return equity, perf_metrics(equity.values, 1, initial=capital)

        def buy_hold_index(close_df, ticker, capital=initial_capital, oos_start="2021-01-04"):
            dates = close_df.index
            if pd.Timestamp(oos_start) in dates:
                oos_pos = list(dates).index(pd.Timestamp(oos_start))
            else:
                oos_pos = np.searchsorted(dates, pd.Timestamp(oos_start))
                
            s = close_df[ticker].values[oos_pos:]
            equity = pd.Series(capital * s / s[0], index=dates[oos_pos:])
            return equity, perf_metrics(equity.values, 0, initial=capital)

        # Chạy giả lập
        eq_strat, m_strat = run_portfolio(close_all, open_all, basket, best_p, weight_scheme, rebalance_freq, is_ret_dict)
        eq_bh, m_bh = buy_hold_basket(close_all, open_all, basket)
        eq_idx, m_idx = buy_hold_index(close_all, "VNINDEX")
        eq_eqw, m_eqw = buy_hold_basket(close_all, open_all, STOCKS) # ETF proxy
        
        # Biểu đồ Plotly Equity Curve
        st.markdown("#### 📊 So sánh Đường cong tài sản (Equity Curve)")
        
        fig_eq = go.Figure()
        fig_eq.add_trace(go.Scatter(x=eq_strat.index, y=eq_strat.values, name="Chiến lược MACD+RSI", line=dict(color="#10b981", width=3)))
        fig_eq.add_trace(go.Scatter(x=eq_bh.index, y=eq_bh.values, name="Buy & Hold rổ Top-K", line=dict(color="#fbbf24", width=2)))
        fig_eq.add_trace(go.Scatter(x=eq_eqw.index, y=eq_eqw.values, name="Equal-Weight 99 mã (ETF Proxy)", line=dict(color="#a78bfa", width=2)))
        fig_eq.add_trace(go.Scatter(x=eq_idx.index, y=eq_idx.values, name="VN-Index", line=dict(color="#ef4444", width=2, dash='dot')))
        
        fig_eq.update_layout(
            template="plotly_dark",
            title="Diễn biến Giá trị tài sản ròng (Equity Curve) giai đoạn 2021 - 2023",
            xaxis_title="Ngày",
            yaxis_title="Giá trị tài sản (VND)",
            height=500,
            hovermode="x unified"
        )
        st.plotly_chart(fig_eq, use_container_width=True)
        
        # Bảng chỉ số tài chính
        st.markdown("#### 📊 Các chỉ số tài chính so sánh")
        
        metric_names = [
            "Total Return [%]", "CAGR [%]", "Volatility [%]", 
            "Sharpe", "Sortino", "Max Drawdown [%]", "Calmar", "Trades"
        ]
        
        df_metrics = pd.DataFrame({
            "Chiến lược MACD+RSI": [m_strat.get(k) for k in metric_names],
            "Buy & Hold rổ Top-K": [m_bh.get(k) for k in metric_names],
            "Equal-Weight 99 mã": [m_eqw.get(k) for k in metric_names],
            "VN-Index": [m_idx.get(k) for k in metric_names]
        }, index=[
            "Tổng lợi nhuận (%)", "Lợi nhuận kép CAGR (%)", "Độ biến động Volatility (%)",
            "Tỷ số Sharpe", "Tỷ số Sortino", "Mức sụt giảm cực đại MDD (%)", "Tỷ số Calmar", "Tổng số giao dịch"
        ])
        
        st.table(df_metrics.round(3))
        
        st.info(
            f"💰 **Giá trị cuối kỳ Chiến lược:** {m_strat['Final Value']:,.0f} VND | "
            f"💸 **Tổng chi phí giao dịch đã trả:** {m_strat['Fees Paid']:,.0f} VND"
        )
        
        # Phân tích theo từng năm
        st.markdown("---")
        st.markdown("#### 📅 Phân tích Hiệu suất chi tiết theo từng năm")
        
        def submetrics(equity, label):
            out = {"label": label, "full": perf_metrics(equity.values, initial=initial_capital)}
            for yr in [2021, 2022, 2023]:
                seg  = equity[equity.index.year == yr]
                prev = equity[equity.index.year < yr]
                start_val = prev.values[-1] if len(prev) else initial_capital
                e = np.concatenate([[start_val], seg.values])
                out[str(yr)] = perf_metrics(e, initial=start_val)
            return out

        S_sub = submetrics(eq_strat, "Chiến lược")
        B_sub = submetrics(eq_bh, "Buy & Hold Top-K")
        V_sub = submetrics(eq_idx, "VN-Index")
        
        years_df_list = []
        for yr in ["2021", "2022", "2023"]:
            years_df_list.append({
                "Năm": yr,
                "Chiến lược Lợi nhuận (%)": S_sub[yr]['Total Return [%]'],
                "Chiến lược Max Drawdown (%)": S_sub[yr]['Max Drawdown [%]'],
                "Buy&Hold Lợi nhuận (%)": B_sub[yr]['Total Return [%]'],
                "Buy&Hold Max Drawdown (%)": B_sub[yr]['Max Drawdown [%]'],
                "VN-Index Lợi nhuận (%)": V_sub[yr]['Total Return [%]']
            })
        
        st.dataframe(pd.DataFrame(years_df_list).round(2), use_container_width=True)

# ------------------------------------------------------------------------------------------
# TAB 4: KIỂM ĐỊNH THỐNG KÊ
# ------------------------------------------------------------------------------------------
with tab_stat:
    st.markdown("### 🔬 Kiểm định Thống kê (Statistical Validation)")
    
    if len(best_p) == 0:
        st.info("💡 Vui lòng thiết lập hoặc chạy tối ưu hóa PSO ở Tab 'Tối ưu hóa PSO' trước.")
    elif not HAVE_SCIPY:
        st.error("Không tìm thấy thư viện Scipy để thực hiện kiểm định thống kê.")
    else:
        r_s = eq_strat.values[1:]/eq_strat.values[:-1]-1
        r_b = eq_bh.values[1:]/eq_bh.values[:-1]-1
        r_i = eq_idx.values[1:]/eq_idx.values[:-1]-1
        
        t1 = ttest_1samp(r_s - rf_daily, 0, alternative="greater")
        wb = wilcoxon(r_s, r_b, alternative="greater")
        wi = wilcoxon(r_s, r_i, alternative="greater")
        
        st.markdown("#### Kết quả kiểm định thống kê lợi nhuận theo ngày:")
        
        col_s1, col_s2, col_s3 = st.columns(3)
        
        with col_s1:
            is_sig_t = t1.pvalue < 0.05
            st.markdown(f"""
            <div class="stCard" style="border-top: 4px solid {'#10b981' if is_sig_t else '#ef4444'};">
                <div class="metric-label">T-test (Lợi nhuận vượt RF > 0)</div>
                <div class="metric-value" style="color: {'#10b981' if is_sig_t else '#ef4444'}; font-size: 20px;">
                    p-value = {t1.pvalue:.2e}
                </div>
                <div style="font-size: 13px; margin-top: 10px; color: #9ca3af;">
                    {'✅ Có ý nghĩa thống kê (p < 0.05)' if is_sig_t else '❌ Không có ý nghĩa thống kê (p >= 0.05)'}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_s2:
            is_sig_wi = wi.pvalue < 0.05
            st.markdown(f"""
            <div class="stCard" style="border-top: 4px solid {'#10b981' if is_sig_wi else '#ef4444'};">
                <div class="metric-label">Wilcoxon Test vs VN-Index</div>
                <div class="metric-value" style="color: {'#10b981' if is_sig_wi else '#ef4444'}; font-size: 20px;">
                    p-value = {wi.pvalue:.2e}
                </div>
                <div style="font-size: 13px; margin-top: 10px; color: #9ca3af;">
                    {'✅ Có ý nghĩa thống kê (p < 0.05)' if is_sig_wi else '❌ Không có ý nghĩa thống kê (p >= 0.05)'}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_s3:
            is_sig_wb = wb.pvalue < 0.05
            st.markdown(f"""
            <div class="stCard" style="border-top: 4px solid {'#10b981' if is_sig_wb else '#ef4444'};">
                <div class="metric-label">Wilcoxon Test vs Buy & Hold Rổ</div>
                <div class="metric-value" style="color: {'#10b981' if is_sig_wb else '#ef4444'}; font-size: 20px;">
                    p-value = {wb.pvalue:.4f}
                </div>
                <div style="font-size: 13px; margin-top: 10px; color: #9ca3af;">
                    {'✅ Có ý nghĩa thống kê (p < 0.05)' if is_sig_wb else '❌ Không có ý nghĩa thống kê theo ngày (p >= 0.05)'}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("""
        > **Giải thích khoa học:**
        > - **t-test**: Kiểm tra xem lợi nhuận vượt trội hằng ngày của Chiến lược so với lãi suất phi rủi ro có thực sự lớn hơn 0 hay không. Kết quả có ý nghĩa thống kê chứng minh chiến lược tạo ra alpha thực sự.
        > - **Wilcoxon (vs VN-Index / Buy & Hold)**: Là kiểm định phi tham số so sánh sự phân bố lợi nhuận ngày của Chiến lược vs Benchmark. Nếu p-value < 0.05, điều này chứng minh sự vượt trội của Chiến lược không phải là do may mắn ngẫu nhiên.
        > - *Lưu ý*: Lợi thế của Chiến lược so với Buy & Hold theo ngày thường ít có ý nghĩa thống kê trực tiếp trên từng phiên lẻ, mà chủ yếu phát huy hiệu ứng lãi kép thông qua quản trị rủi ro cắt lỗ và thoát hàng sớm tại các nhịp sụp gãy lớn (ví dụ năm 2022).
        """)

# ------------------------------------------------------------------------------------------
# TAB 5: BIỂU ĐỒ TÍN HIỆU KỸ THUẬT
# ------------------------------------------------------------------------------------------
with tab_signal:
    st.markdown("### 🔍 Biểu đồ Tín hiệu Kỹ thuật chi tiết từng Cổ phiếu")
    
    selected_ticker = st.selectbox("Chọn cổ phiếu hiển thị tín hiệu:", options=STOCKS)
    
    if selected_ticker:
        # Lấy thông tin giá, MACD, RSI
        close_series = close_all[selected_ticker]
        open_series = open_all[selected_ticker]
        
        # Tính toán tín hiệu
        pos, buy, sell = macd_rsi_signals(close_series, best_p)
        macd_line, signal_line = compute_macd(close_series, best_p["fast"], best_p["slow"], best_p["sign"])
        rsi = compute_rsi(close_series, best_p["rsi_window"])
        
        # Chỉ lấy phần Out-of-Sample từ 2021 trở đi để vẽ biểu đồ cho rõ ràng hoặc toàn bộ tùy ý
        # Hãy cho phép người dùng chọn xem In-sample (2020) hay Out-of-sample (2021-2023)
        timeline_choice = st.radio("Chọn khoảng thời gian hiển thị đồ thị:", options=["Giai đoạn Học (2020)", "Giai đoạn Đầu tư (2021 - 2023)", "Toàn bộ (2020 - 2023)"], horizontal=True)
        
        if timeline_choice == "Giai đoạn Học (2020)":
            mask = close_series.index <= "2020-12-31"
        elif timeline_choice == "Giai đoạn Đầu tư (2021 - 2023)":
            mask = close_series.index > "2020-12-31"
        else:
            mask = pd.Series(True, index=close_series.index)
            
        c_plot = close_series[mask]
        o_plot = open_series[mask]
        pos_plot = pos[mask]
        macd_plot = macd_line[mask]
        sig_plot = signal_line[mask]
        rsi_plot = rsi[mask]
        
        # Xác định điểm mua bán thực tế trong khoảng được plot
        # Điểm mua: Vị thế chuyển từ 0 sang 1
        # Điểm bán: Vị thế chuyển từ 1 sang 0
        actual_buy = (pos_plot == 1) & (pos_plot.shift(1).fillna(0) == 0)
        actual_sell = (pos_plot == 0) & (pos_plot.shift(1).fillna(0) == 1)
        
        buy_dates = c_plot.index[actual_buy]
        buy_prices = c_plot[actual_buy]
        
        sell_dates = c_plot.index[actual_sell]
        sell_prices = c_plot[actual_sell]
        
        # Tạo Subplots
        fig_sub = make_subplots(
            rows=3, cols=1, 
            shared_xaxes=True, 
            vertical_spacing=0.05,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # 1. Price chart + Buy/Sell markers
        fig_sub.add_trace(go.Scatter(x=c_plot.index, y=c_plot.values, name="Giá điều chỉnh", line=dict(color="#3b82f6", width=2)), row=1, col=1)
        
        fig_sub.add_trace(go.Scatter(
            x=buy_dates, y=buy_prices, 
            mode='markers', 
            name='Điểm MUA', 
            marker=dict(symbol='triangle-up', size=12, color='#10b981', line=dict(width=1, color='black'))
        ), row=1, col=1)
        
        fig_sub.add_trace(go.Scatter(
            x=sell_dates, y=sell_prices, 
            mode='markers', 
            name='Điểm BÁN', 
            marker=dict(symbol='triangle-down', size=12, color='#ef4444', line=dict(width=1, color='black'))
        ), row=1, col=1)
        
        # 2. MACD Chart
        fig_sub.add_trace(go.Scatter(x=macd_plot.index, y=macd_plot.values, name="MACD Line", line=dict(color="#f59e0b", width=1.5)), row=2, col=1)
        fig_sub.add_trace(go.Scatter(x=sig_plot.index, y=sig_plot.values, name="Signal Line", line=dict(color="#3b82f6", width=1.5)), row=2, col=1)
        
        macd_hist = macd_plot - sig_plot
        colors_hist = ['#10b981' if val >= 0 else '#ef4444' for val in macd_hist.values]
        fig_sub.add_trace(go.Bar(x=macd_hist.index, y=macd_hist.values, name="Histogram", marker_color=colors_hist), row=2, col=1)
        
        # 3. RSI Chart
        fig_sub.add_trace(go.Scatter(x=rsi_plot.index, y=rsi_plot.values, name="RSI", line=dict(color="#8b5cf6", width=1.5)), row=3, col=1)
        # Ngưỡng rsi
        fig_sub.add_hline(y=best_p["lower"], line_dash="dash", line_color="#ef4444", annotation_text="Ngưỡng dưới", row=3, col=1)
        fig_sub.add_hline(y=best_p["upper"], line_dash="dash", line_color="#10b981", annotation_text="Ngưỡng trên", row=3, col=1)
        
        # Layout settings
        fig_sub.update_layout(
            template="plotly_dark",
            title=f"Tín hiệu Kỹ thuật cho cổ phiếu {selected_ticker}",
            height=700,
            hovermode="x unified",
            xaxis3_title="Ngày",
            yaxis1_title="Giá (VND)",
            yaxis2_title="MACD",
            yaxis3_title="RSI",
            showlegend=True
        )
        st.plotly_chart(fig_sub, use_container_width=True)
