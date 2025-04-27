Để ý phần hdfs_data ý

# Trend analytics
- Phần này tập trung vào phân tích mấy cái chỉ số môi trường, khí hậu, ... (nên và có thể lấy thêm nhé)
- File mẫu xử lí đưa về parquet t làm ở phần code\trend_analytics\data_pipeline\preprocess.py (mới có co2 thoi), and up lên ggc nên để csv cũng k sao (chủ yếu làm cho biết parquet)
- Phần này thì đơn giản thoi, đưa về bài toán time series data analytics như hồi học PTDLDB là đc, and visualize là xong (đáp lên ggc mà dùng qua gg big query)

# UHI Index Analytics
- Phần này data t xử lí hết ròi, dùng ở phần tabular ý, thạm thời dùng cái 1x1 để train model nhé, mấy cái còn lại t đang xem, nếu cần thì cho zô cx được
- Tất cá các file ở folder 1x1 đều có số samples như nhau, tọa dộ tương đương, nên merge theo tọa độ là được nha
- Làm visualize như phần trên
- Train model thì thử XGB, ExtraTree, LightGBM, target nằm ở file này nha hdfs_data\uhi_index_analytics\tabular\long_lat_uhiindex.csv
- Sau đó thì nhớ trích xuất features importance nha