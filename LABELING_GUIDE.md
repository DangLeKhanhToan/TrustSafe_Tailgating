# Pilot labeling workflow

The camera YAML stores geometry only: the threshold line, which side is inside,
and crossing tolerances. It does not contain grants or ground truth. Create a
new YAML whenever the camera position, crop, resolution, or door line changes.

## 1. Configure the pilot camera

```powershell
python configure_camera.py `
  --video recordings\pilot_01.mp4 `
  --out configs\pilot_01.yaml
```

Click the two endpoints of the door threshold, then click a point on the inside
side. Press `S` to save.

## 2. Label the long video

```powershell
python label_session.py `
  --video recordings\pilot_01.mp4 `
  --camera-config configs\pilot_01.yaml `
  --out annotations\pilot_01.json
```

Important keys:

- `Space`: play or pause.
- `A` / `D`: previous or next frame.
- `J` / `L`: jump backward or forward five seconds.
- `B` / `E`: begin or end one benchmark event.
- `G` / `R`: unlock or relock one grant.
- `1`, `2`, `3`: choose the physical actor being labeled.
- `I` / `O`: add an entry or exit crossing at the current frame.
- `+` / `-`: change `scenario_id`.
- `U`: undo the latest edit.
- `S`: save. Open grants and events must be closed first.
- `Q` or `Esc`: quit.

Suggested order for each take: press `B`, press `G` if the door is granted,
mark every crossing with `I` or `O`, press `R`, then press `E`. For an
unauthorized entry, omit `G/R`. A green on-screen LED means the current video
time is inside a grant.

The app assigns `g001`, `c001`, and `e001` automatically. An entry is linked to
the single active grant. Exits are never assigned to grants. Reusing person
`p001` for `in -> out -> in` under one grant lets evaluation treat this as one
unique person rather than tailgating.

## 3. Cut labeled events

```powershell
python cut_labeled_session.py `
  --video recordings\pilot_01.mp4 `
  --annotations annotations\pilot_01.json `
  --out clips\pilot_01
```

Each `B/E` event produces an MP4, a grants CSV, a ground-truth crossings CSV,
and a metadata JSON. Five seconds of context are added before and after the
event by default. Use `--padding-s 3` to change this.

The canonical source is `annotations\pilot_01.json`. Generated CSV files should
not be edited by hand.
