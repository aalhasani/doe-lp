import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from math import comb
import doelp, problems as pr
B,O,A,INK,INK2="#2a78d6","#eb6834","#1baf7a","#0b0b0b","#52514e"
plt.rcParams.update({"font.family":"serif","font.size":8,"axes.edgecolor":INK2,"axes.labelcolor":INK,
 "xtick.color":INK2,"ytick.color":INK2,"axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
 "grid.color":"#e4e3df","grid.linewidth":0.6,"lines.linewidth":1.6,"legend.frameon":False,"savefig.dpi":600})
# Fig 1 Reddy Mikks
p=pr.get("P1 Reddy Mikks (Ex. 2.1-1)"); c=p["c"]
ev=doelp.evd_solve(c,p["G"],p["h"]); V=ev["vertices"]
ang=np.arctan2(V[:,1]-V[:,1].mean(),V[:,0]-V[:,0].mean()); P=V[np.argsort(ang)]
fig,ax=plt.subplots(figsize=(3.5,3.4))
xx,yy=np.meshgrid(np.linspace(0,4.5,200),np.linspace(0,3,200))
cs=ax.contour(xx,yy,5*xx+4*yy,levels=[5,10,15,21],colors="#b9b8b2",linewidths=0.6,linestyles="--")
ax.clabel(cs,fmt="z=%g",fontsize=6)
ax.fill(P[:,0],P[:,1],color="#cde2fb",zorder=0,label="Feasible region")
pv=[];doelp.csa_solve(c,p["G"],p["h"],path=pv);pv=np.array(pv)
pc=[];doelp.csa_solve(c,p["G"],p["h"],x0=np.array([.5,.5]),path=pc);pc=np.array(pc)
ax.plot(pv[:,0],pv[:,1],"-o",color=B,ms=4,label="Steepest ascent from vertex (= simplex path)")
ax.plot(pc[:,0],pc[:,1],"-s",color=O,ms=4,label="Steepest ascent from interior point")
ax.scatter(V[:,0],V[:,1],marker="D",s=22,facecolor="white",edgecolor=INK,zorder=5,label="Extreme-vertices design points")
ax.scatter(*V.mean(0),marker="*",s=60,color=INK,zorder=5,label="Centroid run")
ax.annotate("Optimum (3, 1.5)\nz = 21",xy=(3,1.5),xytext=(3.2,2.2),fontsize=7,color=INK,arrowprops=dict(arrowstyle="-",color=INK2,lw=.6))
ax.set_xlabel("$x_1$ (tons of exterior paint)");ax.set_ylabel("$x_2$ (tons of interior paint)")
ax.set_xlim(0,4.5);ax.set_ylim(0,3);ax.legend(fontsize=6,loc="upper center",bbox_to_anchor=(0.5,-0.2),ncol=1)
fig.tight_layout();fig.savefig("results/fig1_reddy.png",bbox_inches="tight");fig.savefig("results/fig1_reddy.pdf",bbox_inches="tight")
# Fig 2 scaling
e2=json.load(open("results/e2.json")); sq=[d for d in e2 if d["m"]==d["n"]]
n=[d["n"] for d in sq]
fig,axs=plt.subplots(1,2,figsize=(7.16,2.4))
ax=axs[0]
ax.plot(n,[d["pv"] for d in sq],"-o",color=B,ms=4,label="Simplex pivots (Dantzig rule)")
ax.plot(n,[d["st"] for d in sq],"none",marker="x",color=INK,ms=6,label="DOE-CSA stages, vertex start")
ax.plot(n,[d["stc"] for d in sq],"-s",color=O,ms=4,label="DOE-CSA stages, interior start")
ax.set_xlabel("Number of decision variables n (m = n)");ax.set_ylabel("Mean steps to optimum");ax.legend(fontsize=6.5)
ax.set_title("(a) Steps along the ascent path",fontsize=8,loc="left")
ax=axs[1]
nn=np.arange(2,21);ax.semilogy(nn,[comb(3*k,k) for k in nn],"-",color=A,label="Extreme-vertices design: candidate systems")
done=[d for d in sq if d["evd_done"]]
ax.semilogy([d["n"] for d in done],[d["runs"] for d in done],"o",color=INK,ms=4,label="Feasible vertices actually run")
sq20=[d for d in sq if d["n"]<=20]; ax.semilogy([d["n"] for d in sq20],[d["pv"]+1 for d in sq20],"-o",color=B,ms=4,label="Simplex pivots + 1")
ax.set_xlabel("Number of decision variables n (m = n)");ax.set_ylabel("Count (log scale)");ax.legend(fontsize=6.5)
ax.set_title("(b) Cost of exhaustive vertex design",fontsize=8,loc="left")
fig.tight_layout();fig.savefig("results/fig2_scaling.png");fig.savefig("results/fig2_scaling.pdf")
# Fig 3 noisy
e3=json.load(open("results/e3.json"))
names=["P1 Reddy Mikks (Ex. 2.1-1)","P8 Production (Ex. 2.4-2)","P7 Bank loan (Ex. 2.4-1)"]
fig,axs=plt.subplots(1,3,figsize=(7.16,2.3),sharey=False)
for ax,nm in zip(axs,names):
    d=[x for x in e3 if x["problem"]==nm and x["r"]==2]
    s=[x["sigma"]*100 for x in d]
    for q,(col,mk,lab) in enumerate([(O,"s","Pick-the-winner"),(B,"o","DOE model (EVD + centroid)"),(A,"^","DOE model (D-optimal subset)")]):
        ax.plot(s,[x["loss_mean"][q] for x in d],"-",marker=mk,color=col,ms=4,label=lab)
    ax.set_title(nm.split(" (")[0],fontsize=8,loc="left");ax.set_xlabel("Noise SD (% of response range)")
axs[0].set_ylabel("Mean optimality loss (%)");axs[0].legend(fontsize=6.5)
fig.tight_layout();fig.savefig("results/fig3_noise.png");fig.savefig("results/fig3_noise.pdf")
print("ok", pv.tolist(), pc.tolist())
