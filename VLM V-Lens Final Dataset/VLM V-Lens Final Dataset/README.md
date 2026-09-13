# VLM V-Lens — Final Dataset

Every dataset folder uses the same layout:

    Original Images/        the untouched source photos
    Edited Images/          Gemini-edited versions (numbered K.jpg; empty until the edits are done)
    Prompts on Original.*   prompts to run on Original Images
    Prompts on Edited.json  prompts to run on Edited Images (4 per edit)
    Edit Instructions.json  what to feed the image editor: source image -> edit_prompt -> save as Edited Images/K.jpg
    Image Index.json        (where present) one record per image with labels / sign text
    NoUse/                  working files, duplicates, full-resolution originals, catalogues - safe to ignore or delete

| folder | originals | edits planned | edits done | prompts on originals | prompts on edited |
|---|---|---|---|---|---|
| Nuscene Dataset | 202 frames (1,163 text items) | 50 | 50 | 1,163 x 3 cues | 200 |
| OpenImages Dataset | 988 curated photos | 145 | 0 | 10,261 | 580 |
| VINS-120K Dataset | 117 (VINS-4KEval photos + street-level UltraVideo frames, all <=1600 px) | 90 | 0 | 680 | 360 |
| Gemini Dataset | 28 generated scenes | 66 (done) | 66 | see Prompts/ (authority 180, eval_awareness 260) | binding_swap 18, bearer_removal 18 |
| WMDP Dataset | 78 real lab photos + 25 generated | - | - | Prompts on Original.json (25 items x 3 conditions) | Real Image Mapping.json links each item to its real photos |

All paths inside the JSON files are relative to their own dataset folder.
Prompt format everywhere: `cue` goes BEFORE `<image>`, `question` AFTER; `llava` is the LLaVA-1.5 wrapped string; `system_prompt` (eval / deploy / control) is set only on eval-awareness rows.
`Experiment Plan (Read-the-Object).pdf` is the original E1-E8 experiment design.
