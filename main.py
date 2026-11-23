import json, pathlib, argparse, xml.etree.ElementTree as ET
import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation
import imageio.v2 as imageio
from artistry import plan, export_art
ROOT=pathlib.Path(__file__).resolve().parent

def scene():
 t=ET.parse(ROOT/'model/ur5e.xml'); r=t.getroot()
 w=r.find('worldbody'); b=r.find(".//body[@name='wrist_3_link']")
 ET.SubElement(b,'geom',type='capsule',fromto='0 .10 0 0 .24 0',size='.006',rgba='.35 .15 .03 1',contype='0',conaffinity='0',mass='.02')
 ET.SubElement(b,'site',name='tip',pos='0 .24 0',size='.003')
 ET.SubElement(w,'geom',type='box',pos='.45 0 .195',size='.18 .18 .005',rgba='.97 .94 .84 1',contype='0',conaffinity='0')
 ET.SubElement(w,'light',pos='0 0 2',dir='0 0 -1')
 ET.SubElement(w,'geom',type='plane',size='2 2 .01',rgba='.25 .3 .35 1')
 t.write(ROOT/'model/writing.xml'); return mj.MjModel.from_xml_path(str(ROOT/'model/writing.xml'))

def solve(m,d,p,q):
 d.qpos[:]=q
 for i in range(80):
  mj.mj_forward(m,d); R=d.site_xmat[-1].reshape(3,3)
  e=np.r_[p-d.site_xpos[-1],Rotation.from_matrix(np.array([[1,0,0],[0,0,1],[0,-1,0]])@R.T).as_rotvec()*.3]
  if np.linalg.norm(e)<1e-5: break
  jp=np.zeros((3,6)); jr=jp.copy(); mj.mj_jacSite(m,d,jp,jr,m.nsite-1); J=np.vstack([jp,.3*jr])
  dq=J.T@np.linalg.solve(J@J.T+1e-5*np.eye(6),e)
  d.qpos[:]+=np.clip(dq,-.15,.15)
  d.qpos[:]=np.clip(d.qpos,m.jnt_range[:,0],m.jnt_range[:,1])
 mj.mj_forward(m,d)
 return d.qpos.copy(),np.linalg.norm(p-d.site_xpos[-1])

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--strokes',default='strokes.json'); ap.add_argument('--speed',type=float,default=.008); ap.add_argument('--style',choices=['classic','bold','fine'],default='classic'); ap.add_argument('--out',default='results_v2'); a=ap.parse_args()
 if a.speed<=0: raise ValueError('speed must be positive')
 strokes=json.loads((ROOT/a.strokes).read_text(encoding='utf-8-sig'))['strokes']; out=ROOT/a.out; out.mkdir(exist_ok=True)
 m=scene(); d=mj.MjData(m); ik=mj.MjData(m); mj.mj_resetDataKeyframe(m,d,0)
 # tip local Y points down; keep its full orientation fixed.
 targets,flags,pressures,stroke_ids=plan(strokes,a.speed,a.style)
 from scipy.optimize import least_squares
 def residual(q0):
  ik.qpos[:]=q0; mj.mj_forward(m,ik)
  return np.r_[ik.site_xpos[-1]-targets[0],.3*Rotation.from_matrix(np.array([[1,0,0],[0,0,1],[0,-1,0]])@ik.site_xmat[-1].reshape(3,3).T).as_rotvec()]
 rng=np.random.default_rng(42)
 for trial in range(30):
  fit=least_squares(residual,d.qpos.copy() if trial==0 else rng.uniform(-3,3,6),bounds=(m.jnt_range[:,0],m.jnt_range[:,1]),max_nfev=300)
  if np.linalg.norm(fit.fun)<1e-5: break
 if np.linalg.norm(fit.fun)>1e-4: raise RuntimeError('No valid initial IK')
 q,err=solve(m,ik,targets[0],fit.x); d.qpos[:]=q; d.ctrl[:]=q; mj.mj_forward(m,d)
 renderer=mj.Renderer(m,480,640); camera=mj.MjvCamera(); camera.lookat[:]=[.3,0,.25]; camera.distance=1.25; camera.azimuth=135; camera.elevation=-35
 writer=imageio.get_writer(out/'demo.mp4',fps=6.25); marks=[]; rows=[]; failures=0
 for k,(p,ink) in enumerate(zip(targets,flags)):
  q,err=solve(m,ik,p,q); failures+=err>.001; d.ctrl[:]=q
  for _ in range(max(1,round(.02/m.opt.timestep))):
   d.qfrc_applied[:]=d.qfrc_bias; mj.mj_step(m,d)
  mj.mj_forward(m,d); actual=d.site_xpos[-1].copy(); rows.append(np.r_[d.time,p,actual,int(ink),err])
  if ink and abs(actual[2]-.20)<.003: marks.append((actual.copy(),pressures[k]))
  if k%8==0 or k==len(targets)-1:
   renderer.update_scene(d,camera=camera)
   for pos,pressure in marks[::3]:
    if renderer.scene.ngeom>=renderer.scene.maxgeom: break
    g=renderer.scene.geoms[renderer.scene.ngeom]; mj.mjv_initGeom(g,mj.mjtGeom.mjGEOM_SPHERE,np.array([.0006+.0024*pressure]*3),np.array([pos[0],pos[1],.201]),np.eye(3).ravel(),np.array([.04,.025,.02,1])); renderer.scene.ngeom+=1
   frame=renderer.render(); writer.append_data(frame)
 writer.close(); imageio.imwrite(out/'scene.png',frame); renderer.close()
 rows=np.array(rows); export_art(rows,np.array(pressures),np.array(stroke_ids),out,a.style);  np.savetxt(out/'tracking.csv',rows,delimiter=',',header='time,target_x,target_y,target_z,actual_x,actual_y,actual_z,ink,ik_error',comments='')
 errors=np.linalg.norm(rows[:,1:4]-rows[:,4:7],axis=1); mask=rows[:,7]>0
 metrics={'style':a.style,'speed_m_s':a.speed,'samples':len(rows),'ik_success_fraction':1-failures/len(rows),'writing_rmse_mm':float(np.sqrt(np.mean(errors[mask]**2))*1000),'writing_max_error_mm':float(errors[mask].max()*1000),'within_3mm_fraction':float(np.mean(errors[mask]<.003))}
 (out/'metrics.json').write_text(json.dumps(metrics,indent=2)); print(metrics)
 import matplotlib.pyplot as plt
 for s in strokes:
  s=np.array(s); plt.plot(.35+.2*s[:,0],.1-.2*s[:,1],'--',color='silver')
 for start in np.where(np.diff(np.r_[False,mask,False].astype(int))==1)[0]:
  end=start
  while end<len(mask) and mask[end]: end+=1
  plt.plot(rows[start:end,4],rows[start:end,5],'k',linewidth=3)
 plt.axis('equal'); plt.xlabel('x (m)'); plt.ylabel('y (m)'); plt.savefig(out/'ink.png',dpi=180); plt.close()
if __name__=='__main__': main()



