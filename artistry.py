"""Geometric calligraphy effects; pressure is a dimensionless artistic command, not measured force."""
import numpy as np
from scipy.interpolate import PchipInterpolator
from PIL import Image, ImageDraw

def plan(strokes,speed,style):
    if not np.isfinite(speed) or speed <= 0: raise ValueError('speed must be finite and positive')
    targets=[]; flags=[]; pressure=[]; ids=[]; prev=np.array([.45,0,.25])
    def append(points,ink,pres,sid):
        targets.extend(points); flags.extend([ink]*len(points)); pressure.extend(pres); ids.extend([sid]*len(points))
    def move(dest):
        nonlocal prev
        n=max(2,int(np.ceil(np.linalg.norm(dest-prev)/max(speed,.025)/.02)))
        u=np.linspace(0,1,n+1)[1:]; h=10*u**3-15*u**4+6*u**5
        append(prev+(dest-prev)*h[:,None],False,np.zeros(n),-1); prev=dest.copy()
    for sid,stroke in enumerate(strokes):
        xy=np.asarray(stroke,float)
        if xy.ndim!=2 or xy.shape[1]!=2 or len(xy)<2 or not np.isfinite(xy).all() or np.any((xy<0)|(xy>1)):
            raise ValueError('Each stroke needs >=2 finite normalized XY points')
        xy=xy[np.r_[True,np.linalg.norm(np.diff(xy,axis=0),axis=1)>1e-8]]
        if len(xy)<2: raise ValueError('Zero-length stroke')
        pts=np.column_stack([.35+.2*xy[:,0],.1-.2*xy[:,1],np.full(len(xy),.2)])
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(pts,axis=0),axis=1))]
        curve=PchipInterpolator(arc,pts,axis=0)
        dense=curve(np.linspace(0,arc[-1],1200)); length=np.r_[0,np.cumsum(np.linalg.norm(np.diff(dense,axis=0),axis=1))]
        n=max(10,int(np.ceil(length[-1]/speed/.02)))
        u=np.linspace(0,1,n); timing=10*u**3-15*u**4+6*u**5
        path=np.column_stack([np.interp(timing*length[-1],length,dense[:,j]) for j in range(3)])
        move(path[0]+[0,0,.04]); move(path[0])
        # A deliberate start, fuller middle, tapered exit. Not a contact-force model.
        strength={'classic':.8,'bold':1.0,'fine':.38}[style]
        pres=strength*(.15+.85*np.sin(np.pi*u)**.65)*(1-.32*u)
        append(path,True,pres,sid); prev=path[-1]; move(prev+[0,0,.04])
    if not targets: raise ValueError('No strokes')
    return np.asarray(targets),flags,pressure,ids

def export_art(rows,pressure,ids,out,style):
    size=1400; rng=np.random.default_rng(42)
    paper=np.clip(rng.normal(244,1.2,(size,size,1))+np.array([3,0,-11]),0,255).astype('uint8')
    im=Image.fromarray(paper); draw=ImageDraw.Draw(im)
    for row,p,sid in zip(rows,pressure,ids):
        if sid<0 or abs(row[6]-.2)>=.003: continue
        x=(row[4]-.31)/.28*size; y=(.14-row[5])/.28*size
        r=(.0006+.0024*p)/.28*size
        draw.ellipse((x-r,y-r,x+r,y+r),fill=(25,23,21))
    draw.rectangle((1150,1150,1260,1260),outline=(160,45,37),width=6)
    draw.text((1165,1190),'UR5e',fill=(160,45,37))
    im.save(out/'artwork.png')
    np.savetxt(out/'brush_commands.csv',np.column_stack([rows[:,0],ids,pressure]),delimiter=',',header='time,stroke_id,virtual_pressure',comments='')
    import matplotlib.pyplot as plt
    err=np.linalg.norm(rows[:,1:4]-rows[:,4:7],axis=1)*1000
    fig,ax=plt.subplots(2,1,figsize=(10,6),sharex=True)
    ax[0].plot(rows[:,0],err); ax[0].axhline(3,color='r',ls='--'); ax[0].set_ylabel('Tip error (mm)')
    ax[1].plot(rows[:,0],pressure,color='black'); ax[1].set_ylabel('Virtual pressure'); ax[1].set_xlabel('Simulation time (s)')
    fig.tight_layout(); fig.savefig(out/'diagnostics.png',dpi=150); plt.close(fig)
