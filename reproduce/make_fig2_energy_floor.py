"""Regenerates Fig. 2 (two panels) from the two-state parameters of Table 2.
(a) energy per cycle vs inference-runtime speed-up, non-inferring interval fixed;
(b) energy per cycle vs non-inferring interval, inference time fixed.
Place in reproduce/ and run from the repository root."""
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family":"serif","font.size":9})
J=dict(Pi=11.36,Pf=15.37,ti=0.121,tidle=1.245)   # Jetson, three-hour battery trial
R=dict(Pi=7.50,Pf=9.66,ti=0.622,tidle=1.213)      # RPi 5, gauge-resolved floor
R2=dict(R,Pi=3.9)                                 # RPi 5, ready-state baseline
e=lambda p,t:(p['Pi']*t+p['Pf']*p['ti'])/3.6
cross=lambda a,b:(b['Pf']*b['ti']-a['Pf']*a['ti'])/(a['Pi']-b['Pi'])
t=np.linspace(0,2,401); s=np.logspace(0,5,300)
fig,(a1,a2)=plt.subplots(1,2,figsize=(9,3.6))
for p,c,l in [(R,'#4c72b0','Raspberry Pi 5'),(J,'#dd8452','Jetson Orin Nano Super')]:
    a1.plot(s,(p['Pi']*p['tidle']+p['Pf']*p['ti']/s)/3.6,color=c,lw=2,label=l); a1.axhline(p['Pi']*p['tidle']/3.6,color=c,ls=':',lw=1)
a1.plot(s,(R2['Pi']*R2['tidle']+R2['Pf']*R2['ti']/s)/3.6,color='#4c72b0',lw=1,ls='--',label='RPi 5, 3.9 W ready-state floor')
a1.set_xscale('log',base=2); a1.set_xlim(1,32); a1.set_xlabel('Inference-runtime speed-up (×)'); a1.set_ylabel('Energy per cycle (mW h)')
a1.set_title('(a) Non-inferring interval held at measured value',loc='left',fontsize=9); a1.grid(alpha=.3); a1.legend(fontsize=7,loc='lower left')
a1.axvline(1.9,color='grey',ls='--',lw=1); a1.text(1.95,4.45,'1.9×',fontsize=7,color='grey')
a2.plot(t,e(J,t),color='#dd8452',lw=2,label='Jetson Orin Nano Super')
a2.plot(t,e(R,t),color='#4c72b0',lw=2,label='Raspberry Pi 5, gauge floor 7.50 W')
a2.plot(t,e(R2,t),color='#4c72b0',lw=1,ls='--',label='Raspberry Pi 5, ready-state floor 3.9 W')
c1,c2=cross(J,R2),cross(J,R)
a2.axvspan(c1,c2,color='grey',alpha=.2); a2.text((c1+c2)/2,0.25,f'crossover\n{c1:.2f}–{c2:.2f} s',ha='center',fontsize=7)
a2.axvline(1.2,color='k',ls=':',lw=1); a2.text(1.23,0.9,'this study\n(1.2 s timer)',fontsize=7)
a2.set_xlabel('Non-inferring interval per cycle $t_{idle}$ (s)'); a2.set_ylabel('Energy per cycle (mW h)'); a2.set_xlim(0,2); a2.set_ylim(0,6)
a2.set_title('(b) Inference time held at measured value',loc='left',fontsize=9); a2.grid(alpha=.3); a2.legend(fontsize=7,loc='upper left')
fig.tight_layout(); fig.savefig('figures/Fig2_energy_floor.png',dpi=300); fig.savefig('figures/Fig2_energy_floor.pdf')
print(f'crossover t_idle: {c1:.2f}-{c2:.2f} s; at t_idle=0 Jetson {e(J,0):.2f} vs RPi {e(R,0):.2f} mWh')
