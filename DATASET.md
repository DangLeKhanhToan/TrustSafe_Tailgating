# Dataset Recording Protocol

Quay camera cố định và ghi lại resolution, FPS, độ cao, góc nghiêng, ánh sáng,
actor, quần áo và props cho mỗi clip. Entry chính đi trái sang phải nhưng phải có
exit phải sang trái và các negative case.

## Checklist

- 10 lần một người vào bình thường.
- 6 lần người thứ hai theo sát dưới 1.5 giây.
- 4 lần hai người song song hoặc overlapping.
- 2 lần ba người trở lên.
- 3 lần một người ra khi người khác vào.
- 4 lần đi gần cửa nhưng không vào.
- 3 lần do dự/lùi lại rồi vào.
- 3 lần mang túi, hộp hoặc cart.
- 2 lần vào khi không có grant.
- 3 lần ánh sáng xấu hoặc mặc rain gear.

Trộn thứ tự scenario trong mỗi buổi quay. Chia development/validation/test theo
session hoặc tổ hợp actor, không chia theo frame gần nhau.
