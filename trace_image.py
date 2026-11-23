"""人工描点：左键添加点，Enter结束当前笔画，S保存；关闭窗口结束。"""
import argparse,json,pathlib
import matplotlib.pyplot as plt
from PIL import Image
p=argparse.ArgumentParser(); p.add_argument('image'); p.add_argument('--output',default='custom.json'); a=p.parse_args()
im=Image.open(a.image); w,h=im.size; strokes=[]; current=[]
fig,ax=plt.subplots(); ax.imshow(im); ax.set_title('Click points; Enter: next stroke; S: save')
def click(e):
 if e.inaxes is ax and e.button==1:
  current.append([e.xdata/w,e.ydata/h]); ax.plot(e.xdata,e.ydata,'r.'); fig.canvas.draw_idle()
def key(e):
 if e.key=='enter' and len(current)>1: strokes.append(current.copy()); current.clear()
 if e.key=='s':
  result=strokes+([current.copy()] if len(current)>1 else [])
  pathlib.Path(a.output).write_text(json.dumps({'source':str(a.image),'strokes':result}),encoding='utf-8'); print('Saved',a.output)
fig.canvas.mpl_connect('button_press_event',click); fig.canvas.mpl_connect('key_press_event',key); plt.show()
