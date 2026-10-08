import sys, pandas as pd
pd.set_option("display.width", 220)
df = pd.read_parquet(sys.argv[1])
order = ["clean","wrong_image","abl_all","abl_e90","abl_e80","abl_e70","abl_e60","abl_hi","abl_mid","abl_lo",
         "abl_prompt","abl_img","abl_all_a0.25","abl_all_a0.5","abl_all_a0.75",
         "abl_all_a1.5","abl_rand","abl_other","abl_unemb","add_img","add_img_a0.25",
         "add_img_a0.5","add_img_a2.0"]
want = dict(n=("i","count"), word_margin=("word_margin","mean"),
            word_correct=("word_correct","mean"),
            letter_margin=("letter_margin","mean"),
            letter_correct=("letter_correct","mean"),
            cloze_c=("cloze_rank_c","median"), cloze_d=("cloze_rank_d","median"),
            presence=("presence_margin","mean"), presence_d=("presence_d","mean"),
            energy=("energy","mean"),
            ctl_letter=("ctl_letter_correct","mean"), ctl_word=("ctl_word_correct","mean"),
            said=("report_said","mean"), inj_said=("inject_said","mean"))
cols = {k:v for k,v in want.items() if v[0] in df.columns}
g = df.groupby("cond").agg(**cols).reindex([c for c in order if c in set(df.cond)])
print(g.round(3).to_string())
if len(sys.argv) > 2 and "text" in df.columns:
    show = ["clean","abl_all","add_img","add_all"]
    for cond in show:
        sub = df[df.cond == cond].dropna(subset=["text"]).head(3)
        for _, r in sub.iterrows():
            print(f"\n[{cond}] {r.concept} vs {r.distractor}: {str(r.text)[:170]}")
