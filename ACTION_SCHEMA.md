# AndroidControl → SFT Action Schema Mapping

This document defines the canonical action vocabulary used in our SFT training data,
and how raw AndroidControl actions map to it.

## Canonical Actions

| Raw Type | SFT Action | Fields | Description |
|---|---|---|---|
| `click` | `"tap"` | `x`, `y` (normalized 0–1) | Tap/click at normalized coordinates |
| `input_text` | `"type"` | `text` (string) | Type text into focused field |
| `open_app` | `"open_app"` | `app_name` (string) | Open a specific app by name |
| `navigate_back` | `"navigate_back"` | — | Press back button |
| `navigate_home` | `"navigate_home"` | — | Go to home screen |
| `scroll` | `"scroll"` | `direction` ("up"\\|"down"\\|"left"\\|"right") | Scroll viewport |
| `wait` | `"wait"` | — | Wait (no-op, for timing) |
| *(any final step)* | `"done"` | — | Terminal action, episode complete |

## Coordinate Normalization

Coordinates are normalized **per-image** using each screenshot's actual dimensions:

```python
norm_x = raw_pixel_x / image_width   # e.g., 540/1080 = 0.5
norm_y = raw_pixel_y / image_height   # e.g., 1400/2340 = 0.598291
```

This handles heterogeneous device resolutions (1080×1920, 1440×2658, etc.) without
hardcoding any single resolution.

## Sample Format (JSONL row)

```json
{
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "image", "data": "<base64_encoded_png>"},
        {"type": "text", "text": "Send a photo to contact Alice"}
      ]
    },
    {
      "role": "assistant",
      "content": "{\"action\": \"tap\", \"x\": 0.5, \"y\": 0.73}"
    }
  ],
  "episode_id": "12345",
  "step_index": 0,
  "total_steps": 8,
  "granularity": "goal"
}
```

Two granularity modes per step:
- **`goal`**: Full episode-level goal ("Send a photo to contact Alice") — standard eval protocol
- **`step_instruction`**: Per-step instructions ("Tap the camera icon") — low-level eval protocol

## Terminal Action

When the dataset provides a post-action (terminal) screenshot, we emit an
additional synthetic step with `{"action": "done"}` grounded in that screenshot.
The real final action is preserved as a normal step — it is *not* overwritten.
At inference time, the agent should stop generating actions after `done`.

Episodes without a trailing screenshot (about half of AndroidControl) emit no
explicit `done` sample; the agent learns the stop signal from the episodes that
do provide one.
