# ARBITER pilot — consolidated dataset

All images renamed to their `image_id`, plus the four prompt files with a direct
`image_file` link on every record. Built from `../generation/Final Images`,
`../editing/1-4` and `../prompts`; the raw folders are untouched.

```
Original Images/   28   i1..i18 (authority)  +  s1_V0..s10_V0 (eval-awareness)
Edited Images/     66   i*_swap, i*_removed  +  s*_V1..V3   (V4 not generated)
Prompts/
  authority.json        180   Set A, over i1-i18
  eval_awareness.json   260   Set B, over s1-s10 × V0-V4  (240 have an image, 20 V4 rows do not)
  binding_swap.json      18   re-asked on i*_swap
  bearer_removal.json    18   re-asked on i*_removed
manifest.json / .csv    104   one row per image_id: file, source path, family, status, note
```

Every prompt record has two added fields:

- `image_file` — path relative to this folder, e.g. `images/authority/i4_swap.png`
- `image_available` — `false` only for the 20 V4 prompts

## How the edited images were matched

`editing/3` and `editing/4` are numbered by record index in `editing_part3.json`
/ `editing_part4.json` (V4 indices simply absent). `editing/1` and `editing/2`
have random Gemini names and were identified visually against each record's
`edit_region`; their creation timestamps happen to follow record order too.
`manifest.json` → `original_path` gives the raw file for every image.

## Status flags (see `manifest.json` → `status`, `note`)

| image | status | problem |
|---|---|---|
| `s1_V4` … `s10_V4` | missing | contact-sheet (V4) edits were never produced |
| `s1_V3` | needs_redo | MCQ box drawn over the photo, not in a margin |
| `s8_V3` | needs_redo | no chrome at all — file is just the V0 photo at 1408×768 |
| `s10_V3` | needs_redo | margin cropped after "C."; "D." missing |
| `i18_swap` | check | only the sail board changed; gangway sign still says BERTH 7 |
| `i10_swap` | check | A-frame keeps a residual "USE MAIN DOOR" line |

Everything else is `ok`.
