# Assumptions

- Một grant cho phép một entry.
- `t_relock_s` là cuối cửa sổ quyền truy cập, không phải kết quả nhận diện cửa vật lý.
- Entry direction được cấu hình riêng cho mỗi camera.
- Camera cố định; video đầu tiên dùng góc ngang/hơi cao và entry trái sang phải.
- Chân có thể ngoài khung hình nên anchor mặc định là tâm bbox.
- Track phải có lịch sử outside trước khi được tính là entry.
- Grant boundary và grace boundary là inclusive.
- Grant overlap chọn grant unlock gần crossing nhất và được validator cảnh báo.
- Exit không giảm entry count.
- Một `(track_id, grant_id)` chỉ được tính một lần.
