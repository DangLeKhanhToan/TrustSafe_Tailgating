# Thuật toán phát hiện tailgating

## 1. Mô hình nghiệp vụ

Một `grant` là một lần hệ thống cấp quyền mở cửa sau khi quẹt thẻ hoặc mở khóa
bằng điện thoại. Trong bài tập, grant được mô phỏng bằng một khoảng thời gian:

```csv
grant_id,t_unlock_s,t_relock_s
g01,10.0,15.0
```

Hệ thống không xác định người trong ảnh có đúng là chủ thẻ hay không. Nó chỉ so
sánh một quyền vào cửa với số người thực sự đi qua ngưỡng cửa theo hướng `in`.

| Số entry trong grant | Kết quả |
|---:|---|
| 0 | Grant không được dùng, `none` |
| 1 | Hợp lệ, `none` |
| 2+ | `tailgating` |
| Entry ngoài grant | `unauthorised_entry` |

## 2. Luồng xử lý

```text
video frame
  -> YOLOX person detections
  -> tracker gán ID
  -> lấy anchor của mỗi track
  -> crossing state machine
  -> crossing event
  -> ghép với grant
  -> deduplicate
  -> events.json + annotated.mp4
```

Detector, tracker và business logic được tách nhau. Vì vậy có thể thay YOLOX
bằng detector khác hoặc thay tracker baseline bằng ByteTrack mà không sửa logic
grant/crossing.

## 3. Anchor cho góc quay ngang

Video đầu tiên quay ngang hoặc hơi từ trên xuống, người đi vào từ trái sang
phải và có thể không nhìn thấy chân. Anchor mặc định là tâm bounding box:

```text
anchor_x = (x1 + x2) / 2
anchor_y = (y1 + y2) / 2
```

Với threshold gần thẳng đứng, chuyển động theo trục ngang quyết định crossing.
Không dùng mép trước bbox vì nó phát crossing quá sớm; không đợi toàn bộ bbox qua
line vì bbox rung hoặc bị cửa che có thể làm crossing không bao giờ hoàn tất.

## 4. Xác định phía trong và phía ngoài

Với threshold từ `A` đến `B`, signed distance của điểm `P` là:

```text
d(P) = cross(B - A, P - A) / |B - A|
```

Dấu của biểu thức phụ thuộc thứ tự A/B. User click thêm một điểm nằm phía
`INSIDE`; dấu của điểm tham chiếu được dùng để chuẩn hóa mọi khoảng cách. Vì vậy
đảo A và B không làm đổi nghĩa inside/outside.

Hysteresis tạo ba vùng:

```text
d < -h       OUTSIDE
|d| <= h     BAND
d > h        INSIDE
```

Band ngăn bbox rung quanh threshold tạo nhiều crossing giả.

## 5. Crossing state machine

Một entry chỉ được xác nhận khi track có lịch sử ổn định ở ngoài, đi qua band,
sau đó ổn định ở trong:

```text
UNKNOWN -> OUTSIDE_STABLE -> BAND -> INSIDE_CANDIDATE -> CONFIRMED
```

Điều kiện xác nhận:

1. Có ít nhất `stable_frames` observation ở outside.
2. Anchor đi về phía inside.
3. Có ít nhất `stable_frames` observation ở inside.
4. Projected displacement tối thiểu đạt `min_displacement_px`.
5. Track chưa phát entry tương ứng.

Track xuất hiện lần đầu ở inside không được đếm, vì hệ thống chưa quan sát thấy
nó đi từ ngoài vào. Người đi vào band rồi lùi lại outside cũng không được đếm.

Thời điểm crossing được nội suy giữa hai observation quanh line:

```text
r = -d_previous / (d_current - d_previous)
t_cross = t_previous + r * (t_current - t_previous)
```

Nhờ vậy timestamp không bị trễ đến lúc state machine đủ số frame xác nhận.

## 6. Tracking

Tracker baseline dự đoán tâm tiếp theo bằng vận tốc gần nhất. Các detection được
ghép với track bằng khoảng cách tâm kết hợp IoU. Track được giữ trong
`max_lost_seconds` để vượt qua một đoạn che khuất ngắn.

Đây là baseline có chủ ý: khi hai người chồng lấp hoàn toàn, ID có thể đổi hoặc
hai người có thể bị detector gộp thành một bbox. Failure này phải xuất hiện trong
video phân tích, không được che bằng heuristic giả định luôn có hai người.

## 7. Ghép crossing với grant

Effective grant window là:

```text
[t_unlock - pre_grace, t_relock + post_grace]
```

Boundary là inclusive. Nếu nhiều grant overlap, chọn grant đã unlock gần nhất
nhưng không sau crossing. Validator vẫn cảnh báo overlap vì đây thường là lỗi dữ
liệu hoặc nghiệp vụ mơ hồ.

Entry trong cùng grant được deduplicate bằng:

```text
(track_id, grant_id)
```

Do đó cùng người `in -> out -> in` trong một grant chỉ đóng góp một entry. Cùng
track có thể được tính lại ở grant sau.

## 8. Metrics

Với mỗi grant:

```text
ground_truth_tailgating = gt_count > 1
predicted_tailgating = predicted_count > 1
```

Hệ thống báo precision, recall, F1, exact count accuracy, count MAE và confusion
table. FPS phải đo toàn pipeline, gồm decode, inference, tracking và rendering.

## 9. Các failure mode dự kiến

- Detector gộp hai người đang chồng lấp thành một bbox: false negative.
- Tracker đổi ID trong lúc crossing: có thể đếm hai lần.
- Người xuất hiện lần đầu ngay bên trong: không đủ bằng chứng để kết luận entry.
- Cánh cửa che người lâu hơn `max_lost_seconds`: track mới được tạo.
- Threshold đặt quá gần mép ảnh: không đủ outside/inside history.
- Hysteresis quá rộng: crossing xác nhận chậm hoặc bị bỏ sót.
- Hysteresis quá hẹp: bbox jitter tạo transition không ổn định.
- Grace period quá dài: unauthorized entry có thể bị gán nhầm vào grant cũ.

Các tham số phải được chọn trên development/validation session rồi khóa trước
khi chạy final test.
