"""Incrementally receive Gaussian packets; no training checkpoint is loaded."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser(description='Render a progressive stream without loading its checkpoint')
p.add_argument('--assets',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--duration',type=float,default=12)
p.add_argument('--fps',type=int,default=30)
p.add_argument('--load-duration',type=float,default=10)
p.add_argument('--growth',type=float,default=3)
p.add_argument('--font',default='DejaVuSans.ttf')
a=p.parse_args()
repo=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(repo))
from diff_gaussian_rasterization_density import GaussianRasterizationSettings,GaussianRasterizer
from utils.graphics_utils import compute_face_orientation
from roma import rotmat_to_unitquat,quat_xyzw_to_wxyz,quat_wxyz_to_xyzw,quat_product
D=a.assets.resolve()
O=a.output.resolve()
O.mkdir(parents=True,exist_ok=False)
manifest=json.loads((D/'manifest.json').read_text())
header=torch.load(D/'header.pt',map_location='cpu',weights_only=False)
a.subject=header['subject']
base=header['base_vertices'];total_vertices=header['total_vertices'];nframes=int(a.duration*a.fps)
animation=header['animation_vertices'].cuda();cameras=header['cameras']
for c in cameras:
 for key in ['world_view_transform','full_proj_transform','camera_center']:c[key]=c[key].cuda()
keys=['ids','faces','xyz','scaling','rotation','opacity','features']
loaded={};geometry=[];cursor=0;bytes_read=manifest['header_bytes'];events=[]
white=torch.ones(3,device='cuda')

def receive(packet_index):
 global loaded,bytes_read
 info=manifest['packets'][packet_index]
 packet=torch.load(D/info['file'],map_location='cpu')
 assert info['bytes']==(D/info['file']).stat().st_size
 values={key:packet[key].cuda() for key in keys}
 loaded=values if not loaded else {key:torch.cat([loaded[key],values[key]]) for key in keys}
 order=torch.argsort(loaded['ids']);loaded={key:value[order] for key,value in loaded.items()}
 geometry.append({key:packet[key].cuda() for key in ['vertex_ids','vertex_parents','vertex_beta']})
 bytes_read+=info['bytes']
 events.append(dict(packet=packet_index,points=len(loaded['ids']),bytes_read=bytes_read,level=info['level']))

def world(frame):
 # (V,3): only received subdivision vertices are reconstructed.
 vertices=torch.zeros((total_vertices,3),device='cuda');lo=int(math.floor(frame));hi=int(math.ceil(frame));mix=frame-lo
 vertices[:base]=animation[lo]*(1-mix)+animation[hi]*mix
 for g in geometry:
  vertices[g['vertex_ids']]=torch.bmm(g['vertex_beta'].unsqueeze(1),vertices[g['vertex_parents']]).squeeze(1)
 faces=loaded['faces'];orient,scale=compute_face_orientation(vertices,faces,return_scale=True)
 centers=vertices[faces].mean(1)
 xyz=torch.bmm(orient,loaded['xyz'][...,None]).squeeze(-1)*scale+centers
 q=quat_xyzw_to_wxyz(rotmat_to_unitquat(orient));q=torch.nn.functional.normalize(q,dim=-1)
 rot=torch.nn.functional.normalize(loaded['rotation'],dim=-1)
 rot=quat_xyzw_to_wxyz(quat_product(quat_wxyz_to_xyzw(q),quat_wxyz_to_xyzw(rot)))
 return xyz,loaded['scaling'].exp()*scale,rot

def render_view(view,params):
 c=cameras[view]
 if not loaded:return torch.ones((3,c['image_height'],c['image_width']),device='cuda')
 xyz,scale,rot=params
 settings=GaussianRasterizationSettings(image_height=c['image_height'],image_width=c['image_width'],
  tanfovx=math.tan(c['FoVx']/2),tanfovy=math.tan(c['FoVy']/2),bg=white,scale_modifier=1.,
  viewmatrix=c['world_view_transform'],projmatrix=c['full_proj_transform'],sh_degree=header['sh_degree'],
  campos=c['camera_center'],prefiltered=False,debug=False)
 image,_=GaussianRasterizer(raster_settings=settings)(means3D=xyz,means2D=torch.zeros((len(xyz),4),device='cuda'),
  shs=loaded['features'],colors_precomp=None,opacities=loaded['opacity'].sigmoid(),scales=scale,rotations=rot,cov3D_precomp=None)
 return image.clamp(0,1)

font=a.font
titlefont=ImageFont.truetype(font,25);small=ImageFont.truetype(font,16);label=ImageFont.truetype(font,18)
width,height=1128,720
cmd=['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(a.fps),'-i','-',
 '-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(O/f'avatar_{a.subject}_incremental.mp4')]
encoder=subprocess.Popen(cmd,stdin=subprocess.PIPE)
trace=[]
with torch.no_grad():
 for f in range(nframes):
  target_count=header['points']*min(f/(int(a.load_duration*a.fps)-1),1)**a.growth
  count=len(loaded['ids']) if loaded else 0
  while cursor<len(manifest['packets']) and count+manifest['packets'][cursor]['points']<=target_count:
   receive(cursor);cursor+=1
   count=len(loaded['ids'])
  # Resample one forward expression segment to 360 frames, without looping.
  frame=f*(len(animation)-1)/(nframes-1)
  params=world(frame) if loaded else None
  canvas=Image.new('RGB',(width,height),(246,247,250));draw=ImageDraw.Draw(canvas)
  draw.text((18,12),f'ProgressiveAvatars  /  Subject {a.subject}',font=titlefont,fill=(25,31,43))
  for v in range(3):
   image=render_view(v,params)
   rgb=(image.permute(1,2,0).cpu().numpy()*255+.5).astype(np.uint8)
   panel=Image.fromarray(rgb).resize((360,525),Image.Resampling.LANCZOS)
   x=12+v*372;canvas.paste(panel,(x,66))
   draw.text((x+8,43),['View A','View B','View C'][v],font=small,fill=(75,85,105))
  count=len(loaded['ids']) if loaded else 0;ratio=count/header['points']
  draw.text((18,606),f'3DGS loading  {ratio*100:5.1f}%    |    {count:,} / {header["points"]:,} Gaussians',font=label,fill=(25,31,43))
  draw.rounded_rectangle((18,640,width-18,653),radius=6,fill=(217,223,233))
  draw.rounded_rectangle((18,640,18+int((width-36)*ratio),653),radius=6,fill=(39,112,219))
  draw.text((18,672),f'{bytes_read/1e6:.1f} / {manifest["total_bytes"]/1e6:.1f} MB   |   Level-first, contribution-ranked   |   Count-based loading / simulated arrival',font=small,fill=(75,85,105))
  encoder.stdin.write(np.asarray(canvas).tobytes())
  trace.append(dict(frame=f,animation_frame=frame,loaded_chunks=cursor,loaded_points=count,loaded_bytes=bytes_read))
  if f in [0,nframes//6,nframes//2,int(a.load_duration*a.fps)-1,nframes-1]:canvas.save(O/f'preview_{f:03d}.png')
  if f%60==0:print('RENDER',a.subject,f,cursor,count,flush=True)
 encoder.stdin.close();assert encoder.wait()==0
 assert cursor==len(manifest['packets']) and len(loaded['ids'])==header['points']
 assert bytes_read==manifest['total_bytes']
 # Independent receiver vs source checkpoint at three poses and three cameras.
 references=torch.load(D/'references.pt',map_location='cpu');checks=[]
 for ref in references:
  image=render_view(ref['view'],world(ref['frame'])).cpu();error=(image-ref['image']).abs()
  result=dict(frame=ref['frame'],view=ref['view'],max_error=float(error.max()),mean_error=float(error.mean()),mse=float(error.square().mean()))
  assert result['max_error']<2e-4 and result['mse']<1e-10,result
  checks.append(result)
(O/'load_trace.json').write_text(json.dumps(trace))
(O/'verification.json').write_text(json.dumps(dict(status='passed',subject=a.subject,checks=checks,read_events=events,
 points=header['points'],loaded_bytes=bytes_read,total_chunks=cursor,frames=nframes),indent=2))
print('DEMO_COMPLETE',a.subject,flush=True)
