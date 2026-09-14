# SPDX-License-Identifier: Apache-2.0
"""CLEVR reference-resolution gallery: who does each reading credit for the pointer.

No model, no GPU.  Region attributions and occlusion drops are read from
results/pointer/qwen2b_near_rows.parquet, boxes from qwen2b_near_items.json.
The relational question ("...nearest the large matte ball") forces the model to
resolve a pointer: the landmark is not the answer, only the route to it.  The
occlusion column is the ground truth -- the fall in the answer logit when each
region's activations are covered.  VJ-Lens's region attribution ranks referent >
landmark > distractor, matching occlusion; the logit lens and the J-lens hand
the same-shape distractor as much or more credit than the landmark pointer.
Bars are each reading's share, clipped at zero and scaled to its own maximum.
"""
import sys, os, json, textwrap
sys.path.insert(0,"/workspace/vlm/code")
import numpy as np, pandas as pd
from jvlm import style
style.use(1.0)
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

items=json.load(open("/workspace/vlm/results/pointer/qwen2b_near_items.json"))
by={it['scene_index']:it for it in items}
df=pd.read_parquet("/workspace/vlm/results/pointer/qwen2b_near_rows.parquet")
df=df[(df.baseline=='clevr')&(df.question=='main')].rename(columns={'drop':'dropv'})
SCENES=[2279,9192,10172]
out="/workspace/vlm/figures/f132_clevr_cards.png"
roles=['referent','landmark','distractor']
RC={'referent':style.BLUE,'landmark':style.ORANGE,'distractor':style.MUTED}
READ=[('logit lens','logit lens_sum'),('J-lens','J-lens_sum'),('VJ-Lens','VJ-Lens_sum'),('occlusion (truth)','dropv')]

def vals(band,col):
    v=np.array([band[band.role==r][col].mean() for r in roles],float)
    v=np.clip(v,0,None); m=v.max() or 1.0
    return v/m

fig=plt.figure(figsize=(14.5,3.15*len(SCENES)+0.6))
gs=fig.add_gridspec(len(SCENES),3,width_ratios=[1.25,1.15,1.7],wspace=0.12,hspace=0.30,
                    left=0.006,right=0.985,top=0.915,bottom=0.03)
for r,sc in enumerate(SCENES):
    it=by[sc]; band=df[(df.scene==sc)&(df.layer.between(3,11))]
    arr=np.asarray(Image.open(it['path']).convert('RGB')); H,W=arr.shape[:2]
    ax=fig.add_subplot(gs[r,0]); ax.imshow(arr); ax.set_aspect('equal')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    for role,ls in [('referent','-'),('landmark','-'),('distractor',(0,(3,2)))]:
        x,y,w,h=it[f'box_{role}']
        ax.add_patch(Rectangle((x*W,y*H),w*W,h*H,fill=False,edgecolor=RC[role],lw=2.6,linestyle=ls))
    ax.set_title("scene",loc="left",fontsize=15,color=style.INK,pad=4)
    axt=fig.add_subplot(gs[r,1]); axt.axis("off")
    axt.text(0,0.97,"relational question",fontsize=14,color=style.INK2,va="top")
    q = textwrap.fill('"'+it['question_main']+'"', 26)
    axt.text(0,0.885,q,fontsize=15,color=style.INK,va="top",linespacing=1.3)
    # the question can wrap to 1-3 lines; "model answers" is placed from the
    # actual line count so a long question cannot run into it
    y_ans = min(0.60, 0.885 - (q.count("\n")+1)*0.12 - 0.04)
    axt.text(0,y_ans,"model answers  ",fontsize=14,color=style.INK2,va="top")
    axt.text(0.42,y_ans,it['answer_main'],fontsize=15,color=style.BLUE,va="top")
    axt.text(0,0.36,"referent",fontsize=14,color=RC['referent'],va="top")
    axt.text(0,0.24,"landmark  = the pointer",fontsize=14,color=RC['landmark'],va="top")
    axt.text(0,0.12,"distractor  = same-shape decoy",fontsize=14,color=style.INK2,va="top")
    axb=fig.add_subplot(gs[r,2]); axb.axis("off")
    colx=[0.28,0.54,0.80]; barmax=0.20; rowy=[0.80,0.585,0.37,0.155]; bh=0.075
    for j,role in enumerate(roles):
        axb.text(colx[j],0.955,role,fontsize=13.5,color=RC[role],ha="left",va="bottom")
    for i,(nm,col) in enumerate(READ):
        v=vals(band,col); yc=rowy[i]
        axb.text(0.24,yc,nm,ha="right",va="center",fontsize=14,color=style.INK)
        for j,role in enumerate(roles):
            bw=barmax*v[j]
            axb.add_patch(Rectangle((colx[j],yc-bh/2),max(bw,0.004),bh,color=RC[role],
                          alpha=0.95 if v[j]==max(v) else 0.5))
    axb.set_xlim(0.0,1.02); axb.set_ylim(0.06,1.0)
    axb.set_title("who each reading credits  (bar = share of the reading)",loc="left",
                  fontsize=14,color=style.INK,pad=8)
fig.text(0.5,0.978,"Resolving “nearest the landmark”: only VJ-Lens credits the pointer, matching occlusion",
         fontsize=18,color=style.INK,va="top",ha="center")
fig.savefig(out,dpi=200); print("wrote",out)
