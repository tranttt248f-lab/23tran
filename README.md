# 📈 Streamlit Web App: Tối Ưu Hóa Danh Mục Đầu Tư HOSE (2020 - 2023)

Ứng dụng web được xây dựng bằng **Streamlit** giúp mô phỏng và tối ưu hóa danh mục đầu tư chứng khoán trên sàn HOSE giai đoạn 2020–2023, kết hợp hai chỉ báo kỹ thuật phổ biến là **MACD** và **RSI**, cùng với thuật toán tối ưu hóa bầy đàn **PSO (Particle Swarm Optimization)**.

Ý tưởng và thuật toán dựa trên nghiên cứu chiến lược giao dịch kết hợp tránh thiên kiến nhìn trước tương lai (look-ahead bias):
- **Giai đoạn Học (In-sample: 2020)**: Sử dụng PSO để tìm bộ tham số tối ưu cho chỉ báo MACD & RSI trên toàn thị trường, sau đó xếp hạng và chọn ra các cổ phiếu tốt nhất (Top 3, 4, hoặc 5) theo tỷ số Sharpe.
- **Giai đoạn Đầu tư (Out-of-sample: 2021–2023)**: Áp dụng chiến lược lên rổ cổ phiếu đã chọn với các kịch bản phân bổ tỷ trọng (Đều hoặc theo Hiệu suất) và tần suất tái cân bằng (Không, Hằng tháng, Hằng quý, Hằng năm), so sánh với các chỉ số thị trường (VN-Index, rổ Buy & Hold, ETF proxy).

---

## 📁 Cấu Trúc Thư Mục Dự Án

```text
├── app.py                # File chạy chính của ứng dụng Streamlit
├── HOSE_2020_2023.csv    # Tập dữ liệu lịch sử giá cổ phiếu HOSE
├── requirements.txt      # Khai báo các thư viện Python cần thiết
└── README.md             # Hướng dẫn sử dụng dự án (file này)
```

---

## 🛠️ Cài Đặt và Chạy Dưới Local

### 1. Yêu cầu hệ thống
Hãy chắc chắn rằng máy tính của bạn đã cài đặt **Python 3.8+**.

### 2. Cài đặt các thư viện cần thiết
Mở Terminal/Command Prompt tại thư mục dự án và chạy lệnh sau để cài đặt các package phụ thuộc:
```bash
pip install -r requirements.txt
```

### 3. Khởi chạy ứng dụng Streamlit
Chạy lệnh bên dưới để khởi động server cục bộ:
```bash
streamlit run app.py
```
Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ mặc định: `http://localhost:8501`.

---

## 🚀 Hướng Dẫn Deploy Lên Streamlit Cloud

Để đưa ứng dụng này lên mạng (Public Web App) hoàn toàn miễn phí, hãy làm theo các bước sau:

1. **Đưa code lên GitHub**:
   - Tạo một repository mới trên GitHub.
   - Commit và push toàn bộ các file trong thư mục này (`app.py`, `requirements.txt`, `README.md`, `HOSE_2020_2023.csv`) lên repository đó.
2. **Deploy trên Streamlit**:
   - Truy cập trang [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
   - Nhấn vào nút **"New app"**.
   - Chọn Repository, Branch (thường là `main` hoặc `master`), và chỉ định Main file path là `app.py`.
   - Nhấn **"Deploy!"**.
   - Chờ một vài phút để Streamlit cấu hình môi trường và khởi chạy ứng dụng của bạn.

---

## 📊 Các Tính Năng Chính Trên Giao Diện Web App

- **Bảng điều khiển (Sidebar)**: Cấu hình linh hoạt nguồn dữ liệu (tải file lên hoặc dùng dữ liệu mặc định), vốn khởi tạo, phí giao dịch, lãi suất phi rủi ro, và số lượng hạt/vòng lặp cho thuật toán PSO.
- **Tối ưu hóa tham số (PSO)**: Cho phép chạy trực tiếp thuật toán PSO trên web hoặc sử dụng bộ tham số được tối ưu sẵn nhằm tiết kiệm thời gian chạy. Hiển thị log huấn luyện trực quan.
- **Xếp hạng & Chọn rổ cổ phiếu**: Xem danh sách các cổ phiếu HOSE được xếp hạng theo Sharpe in-sample.
- **Hiệu suất Danh mục Out-of-sample (2021-2023)**:
  - So sánh đồ thị tăng trưởng tài sản (Equity Curve) trực quan qua **Plotly** giữa Chiến lược và 3 Benchmark.
  - Xem bảng so sánh các chỉ số tài chính quan trọng: CAGR, Volatility, Sharpe, Sortino, Max Drawdown, Calmar, tổng số lệnh giao dịch, và tổng phí giao dịch đã trả.
  - Phân tích chi tiết hiệu suất theo từng năm (2021, 2022, 2023).
- **Kiểm định Thống kê**: Tích hợp các kiểm định giả thuyết thống kê (t-test, Wilcoxon) để đánh giá độ tin cậy của chiến lược đầu tư.
- **Biểu đồ Tín hiệu Kỹ thuật**: Chọn bất kỳ mã chứng khoán nào để xem chi tiết biểu đồ giá cùng các điểm Mua/Bán trực quan đi kèm đồ thị MACD và RSI.
