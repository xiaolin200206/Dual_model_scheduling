import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
R=Path('/home/claude/github'); OUT=Path('/home/claude/paper/figs')
plt.rcParams.update({"font.family":"serif","font.serif":["DejaVu Serif"],"font.size":9,"axes.linewidth":0.8,"savefig.bbox":"tight","savefig.pad_inches":0.05})
C_RPI,C_JET="#4a6fa5","#c4703c"; MODES=['sequential','parallel','staggered']; DUTY=['duty','nonduty']
def load(p):
    df=pd.read_csv(p)
    for c in df.columns:
        if c not in ('Timestamp','Schedule_Mode','Batt_State','Leaf_Detections','Pest_Detections'): df[c]=pd.to_numeric(df[c],errors='coerce')
    t=pd.to_datetime(df['Timestamp'],format='%H:%M:%S.%f',errors='coerce').ffill(); df['sec']=(t-t.iloc[0]).dt.total_seconds().values.copy(); return df
def per_inf(s):
    s=s.values; s=s[s>0]; return s[np.r_[True,s[1:]!=s[:-1]]]
runs={(plat,m,d):load(R/sub/'data'/f'{pre}_{m}_{d}.csv') for plat,sub,pre in (('RPi5','raspberry-pi','raspberry'),('Jetson','jetson','jetson')) for m in MODES for d in DUTY}
# Fig 1 boxplots
fig,axes=plt.subplots(1,2,figsize=(7.2,3.6))
for ax,title,key in zip(axes,['Small model $M_S$ (YOLO11n)','Large model $M_L$ (YOLO11s)'],['Pest_Lat_ms','Leaf_Lat_ms']):
    data,labels,colors=[],[],[]
    for plat in ('RPi5','Jetson'):
        for d in DUTY:
            for m in MODES:
                data.append(per_inf(runs[(plat,m,d)][key])); labels.append(f"{m[:4]}\n{'duty' if d=='duty' else 'cont'}"); colors.append(C_RPI if plat=='RPi5' else C_JET)
    bp=ax.boxplot(data,whis=(5,95),showfliers=False,widths=0.62,patch_artist=True,medianprops=dict(color='0.15',lw=1.1),showmeans=True,meanprops=dict(marker='D',markerfacecolor='white',markeredgecolor='0.2',markersize=3.2))
    for p,c in zip(bp['boxes'],colors): p.set_facecolor(c); p.set_alpha(0.55); p.set_edgecolor('0.25'); p.set_linewidth(0.8)
    for el in ('whiskers','caps'):
        for ln in bp[el]: ln.set_color('0.35'); ln.set_linewidth(0.8)
    ax.set_yscale('log'); ax.set_xticklabels(labels,fontsize=6.4); ax.set_ylabel('Per-inference latency (ms, log scale)',fontsize=8.5); ax.set_title(title,fontsize=9.5,pad=6)
    ax.grid(axis='y',ls=':',lw=0.5,color='0.75',alpha=0.7); ax.set_axisbelow(True); ax.axvline(6.5,color='0.5',lw=0.9,ls='--')
    ax.text(3.5,ax.get_ylim()[1]*0.72,'Raspberry Pi 5',ha='center',fontsize=7.5,color=C_RPI,fontweight='bold'); ax.text(9.5,ax.get_ylim()[1]*0.72,'Jetson Orin Nano Super',ha='center',fontsize=7.5,color=C_JET,fontweight='bold')
fig.tight_layout(); fig.savefig(OUT/'Fig3_latency_distributions.png',dpi=300); plt.close(fig)
# Fig 2 throughput vs latency
fig,axes=plt.subplots(1,2,figsize=(7.2,3.2)); mk={'sequential':'o','parallel':'s','staggered':'^'}
for (plat,m,d),df in runs.items():
    c=C_RPI if plat=='RPi5' else C_JET; h=(df['sec'].iloc[-1]-df['sec'].iloc[0])/3600
    for ax,key in zip(axes,['Pest_Lat_ms','Leaf_Lat_ms']):
        x=per_inf(df[key]); ax.scatter(x.mean(),len(x)/h,marker=mk[m],s=34,facecolor=c if d=='duty' else 'white',edgecolor=c,lw=1.1,zorder=3)
for ax,t in zip(axes,['Small model $M_S$','Large model $M_L$']):
    ax.set_xscale('log'); ax.set_xlabel('Mean per-inference latency (ms, log)',fontsize=8); ax.set_ylabel('Achieved throughput (inferences h$^{-1}$)',fontsize=8); ax.set_title(t,fontsize=9); ax.grid(ls=':',lw=0.5,color='0.8'); ax.set_axisbelow(True); ax.tick_params(labelsize=7)
h=[Line2D([],[],marker='o',ls='',color='0.3',label='sequential'),Line2D([],[],marker='s',ls='',color='0.3',label='parallel'),Line2D([],[],marker='^',ls='',color='0.3',label='staggered'),Line2D([],[],marker='o',ls='',markerfacecolor='0.3',markeredgecolor='0.3',label='duty-cycled'),Line2D([],[],marker='o',ls='',markerfacecolor='white',markeredgecolor='0.3',label='continuous'),Line2D([],[],marker='s',ls='',color=C_RPI,label='Raspberry Pi 5'),Line2D([],[],marker='s',ls='',color=C_JET,label='Jetson Orin Nano Super')]
axes[0].legend(handles=h,fontsize=6,frameon=False,loc='lower left',ncol=2)
fig.tight_layout(); fig.savefig(OUT/'Fig4_throughput_latency.png',dpi=300); plt.close(fig)
# Fig 3 battery 4-panel
rp=load(R/'raspberry-pi/data/raspberry_battery.csv'); jt=load(R/'jetson/data/jetson_battery.csv')
fig,axes=plt.subplots(2,2,figsize=(7.2,5.0))
for df,lab,c in ((rp,'Raspberry Pi 5',C_RPI),(jt,'Jetson Orin Nano Super',C_JET)):
    m=df['sec']/60
    axes[0,0].plot(m,df['Batt_Percent'],lw=1.2,color=c,label=lab); axes[0,1].plot(m,df['Temp_C'].rolling(120,min_periods=1).mean(),lw=1.2,color=c)
    axes[1,0].plot(m,df['Batt_Current_mA'].abs().rolling(120,min_periods=1).mean(),lw=1.2,color=c)
    P=(df['Batt_Voltage_mV']/1000*df['Batt_Current_mA'].abs()/1000).fillna(0); dt=np.clip(np.diff(df['sec'],prepend=df['sec'].iloc[0]),0,5); E=(P*dt).cumsum()/3600
    p=df['Pest_Lat_ms'].values; chg=np.r_[False,p[1:]!=p[:-1]]&(p>0); axes[1,1].plot(np.cumsum(chg),E,lw=1.2,color=c)
axes[0,0].set_ylabel('Pack state of charge (%)',fontsize=8); axes[0,0].set_title('(a) Pack discharge',fontsize=9); axes[0,0].legend(fontsize=6.5,frameon=False)
axes[0,1].axhline(82,color='0.35',ls='--',lw=0.9); axes[0,1].text(5,83,'82 °C software cut-off',fontsize=6.5,color='0.35'); axes[0,1].set_ylabel('Die temperature (°C)',fontsize=8); axes[0,1].set_title('(b) Thermal trajectory (60-sample mean)',fontsize=9); axes[0,1].set_ylim(40,90)
axes[1,0].set_ylabel('Discharge current (mA)',fontsize=8); axes[1,0].set_title('(c) Pack-side current (60-sample mean)',fontsize=9)
axes[1,1].set_xlabel('Cycles completed (one inference of each model)',fontsize=8); axes[1,1].set_ylabel('Cumulative energy (W h)',fontsize=8); axes[1,1].set_title('(d) Energy against work done',fontsize=9)
for ax in (axes[0,0],axes[0,1],axes[1,0]): ax.set_xlabel('Elapsed time (min)',fontsize=8)
for ax in axes.flat: ax.grid(ls=':',lw=0.5,color='0.8'); ax.set_axisbelow(True); ax.tick_params(labelsize=7)
fig.tight_layout(); fig.savefig(OUT/'Fig1_battery_trials.png',dpi=300); plt.close(fig)
# Fig 4 energy floor
k=np.logspace(0,np.log10(32),200); fig,ax=plt.subplots(figsize=(4.6,3.0))
for lab,c,idle,inf,over,comp in (('Raspberry Pi 5',C_RPI,7.50,9.66,1.213,0.622),('Jetson Orin Nano Super',C_JET,11.36,15.37,1.245,0.121)):
    ax.plot(k,(idle*over+inf*comp/k)/3.6,color=c,lw=1.4,label=lab); ax.axhline(idle*over/3.6,color=c,ls=':',lw=0.9)
ax.axvline(1.89,color='0.4',ls='--',lw=0.8); ax.text(1.95,4.47,'Jetson crosses\nRPi 5 at 1.9×',fontsize=6.5,color='0.3')
ax.set_xscale('log'); ax.set_xlabel('Inference-runtime speed-up (×)',fontsize=8); ax.set_ylabel('Energy per cycle (mW h)',fontsize=8); ax.set_xticks([1,2,4,8,16,32]); ax.set_xticklabels(['1','2','4','8','16','32']); ax.legend(fontsize=6.5,frameon=False); ax.grid(ls=':',lw=0.5,color='0.8'); ax.tick_params(labelsize=7)
fig.tight_layout(); fig.savefig(OUT/'Fig2_energy_floor.png',dpi=300); plt.close(fig); print('ok')
