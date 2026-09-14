# SPDX-License-Identifier: Apache-2.0
"""Eval-awareness worked-example gallery: same photo, same question, two framings.

No model, no GPU.  Maps are read from results/evalaware/qwen2b_v2_cases.json,
which pairs each item's eval-framed prompt (an exam item: "Question ... Options
A) B) C)") with a deploy-framed prompt (a casual user request).  The photograph
is a fixed, unrelated backdrop -- nothing in the image changes between the two
framings.  The logit lens and the J-lens read image tokens, so their two
readings are byte-identical; VJ-Lens differentiates the live answer and moves
with the framing.
"""
import sys, os, json, collections, textwrap
sys.path.insert(0,"/workspace/vlm/code"); sys.path.insert(0,"/workspace/vlm/code/exp")
import numpy as np
from jvlm import style
style.use(1.0)
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from _gallov import load_rgb, ramp

d=json.load(open("/workspace/vlm/results/evalaware/qwen2b_v2_cases.json"))
byp=collections.defaultdict(dict)
for c in d: byp[c['pair_id']][c['is_eval']]=c
PAIRS=['4598','1162','10514']
out="/workspace/vlm/figures/f133_eval_cards.png"
grid=14

def short(s,n=88):
    s=" ".join(s.split()).replace("$",r"\$")
    s=s if len(s)<=n else s[:n].rsplit(" ",1)[0]+" ..."
    return textwrap.fill(s,22)

def diverge():
    ra,ga,ba=to_rgb(style.BLUE); rb,gb,bb=to_rgb(style.RED)
    return LinearSegmentedColormap.from_list("d",
        [(rb,gb,bb,0.9),(rb,gb,bb,0.28),(1,1,1,0.0),(ra,ga,ba,0.28),(ra,ga,ba,0.9)])

def heat(ax,img,m,hue,title,q=0.55,vmax=None):
    ax.imshow(img,extent=[0,1,0,1],origin="upper",interpolation="lanczos")
    a=np.array(m,float).reshape(grid,grid); a=a-a.min()
    mx=vmax if vmax is not None else a.max()
    if mx>0: a=a/mx
    a=np.clip((a-q)/(1-q),0,1)
    ax.imshow(a,extent=[0,1,0,1],origin="upper",cmap=ramp(hue),vmin=0,vmax=1,interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(title,loc="left",color=style.INK,fontsize=15,pad=4)

def heat_diff(ax,img,a,b,title):
    ax.imshow(img,extent=[0,1,0,1],origin="upper",interpolation="lanczos")
    a=np.array(a,float).reshape(grid,grid); b=np.array(b,float).reshape(grid,grid)
    dif=a-b; s=np.abs(dif).max() or 1.0; dif=dif/s
    ax.imshow(dif,extent=[0,1,0,1],origin="upper",cmap=diverge(),vmin=-1,vmax=1,interpolation="bilinear")
    ax.set_xticks([]); ax.set_yticks([])
    for s2 in ax.spines.values(): s2.set_visible(False)
    ax.set_title(title,loc="left",color=style.INK,fontsize=15,pad=4)

fig=plt.figure(figsize=(15.5,3.05*len(PAIRS)+0.55))
gs=fig.add_gridspec(len(PAIRS),6,wspace=0.06,hspace=0.30,left=0.008,right=0.995,top=0.925,bottom=0.02)
for r,pid in enumerate(PAIRS):
    g=byp[pid]; e,dp=g[True],g[False]; img=load_rgb(e['image_path'],392)
    ax=fig.add_subplot(gs[r,0]); ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title("photo (unrelated backdrop)",loc="left",fontsize=14,color=style.INK,pad=4)
    axt=fig.add_subplot(gs[r,1]); axt.axis("off")
    # the truncated, wrapped prompt can still run to several lines; place
    # "deploy framing" from the actual line count so a long eval prompt
    # cannot run into it
    ep = short(e['prompt'])
    axt.text(0,0.95,"eval framing",fontsize=14.5,color=style.BLUE,va="top")
    axt.text(0,0.845,'"'+ep+'"',fontsize=13.5,color=style.INK,va="top")
    y_dep = 0.845 - (ep.count("\n")+1)*0.10 - 0.05
    axt.text(0,y_dep,"deploy framing",fontsize=14.5,color=style.RED,va="top")
    dpp = short(dp['prompt'])
    axt.text(0,y_dep-0.105,'"'+dpp+'"',fontsize=13.5,color=style.INK,va="top")
    heat(fig.add_subplot(gs[r,2]),img,e['contrast_map']['logit'],style.ORANGE,"logit lens  (eval = deploy)")
    heat(fig.add_subplot(gs[r,3]),img,e['contrast_map']['jlens'],style.VIOLET,"J-lens  (eval = deploy)")
    ve=e['contrast_map']['vj']; vd=dp['contrast_map']['vj']
    vmax=max(max(ve)-min(ve),max(vd)-min(vd))
    heat(fig.add_subplot(gs[r,4]),img,ve,style.BLUE,"VJ-Lens | eval",vmax=vmax)
    heat_diff(fig.add_subplot(gs[r,5]),img,ve,vd,"VJ-Lens shift  (blue eval, red deploy)")
fig.text(0.5,0.982,"Same photo, same question, two framings: only VJ-Lens registers eval vs deploy",
         fontsize=18,color=style.INK,va="top",ha="center")
fig.savefig(out,dpi=200); print("wrote",out)
