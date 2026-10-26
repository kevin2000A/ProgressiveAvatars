"""Export contribution-ranked packets from an accepted checkpoint; never train."""
import argparse
import json
import os
from pathlib import Path
import sys
import torch

p=argparse.ArgumentParser(description='Export a contribution-ranked progressive avatar stream')
p.add_argument('--model',type=Path,required=True,help='Training output with run_config.json')
p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--source',type=Path,help='Override the dataset path in the saved configuration')
p.add_argument('--subject',default='avatar')
p.add_argument('--score-views',type=int,default=48)
p.add_argument('--animation-sequence',default='EXP-3')
p.add_argument('--animation-frames',type=int,default=120)
a=p.parse_args()
run=a.model.resolve();out=a.output.resolve()
out.mkdir(parents=True,exist_ok=False)
repo=Path(__file__).resolve().parents[1]
config=json.loads((run/'run_config.json').read_text())
os.chdir(repo);sys.path.insert(0,str(repo))
from arguments import ModelParams,OptimizationParams,PipelineParams
from scene import Scene,AdaptiveFlameGaussianModel,TriangleSplitMode
from gaussian_renderer import render
cfg=argparse.Namespace(**config['args'])
cfg.model_path=str(run)
if a.source is not None:cfg.source_path=str(a.source.resolve())
mp=ModelParams(argparse.ArgumentParser());op=OptimizationParams(argparse.ArgumentParser());pp=PipelineParams(argparse.ArgumentParser())
dataset,opt,pipe=mp.extract(cfg),op.extract(cfg),pp.extract(cfg)
model=AdaptiveFlameGaussianModel(dataset.sh_degree,dataset.disable_flame_static_offset,dataset.not_finetune_flame_params,split_mode=TriangleSplitMode.ONE_TO_THREE,max_level=5)
model.set_lod_include_descendants(True);scene=Scene(dataset,model,shuffle=False)
state,step=torch.load(a.checkpoint.resolve(),weights_only=False)
model.restore(state,opt);model.set_lod_include_descendants(True)
N=len(model._xyz);levels=model.geo.node_levels
assert torch.equal(model.binding.long(),torch.arange(N,device='cuda'))
train=scene.getTrainCameras();black=torch.zeros(3,device='cuda');white=torch.ones(3,device='cuda')
score=torch.zeros(N,device='cuda');samples=[];checks=[]
for k in range(a.score_views):
 idx=k*len(train)//a.score_views;cam=train[idx]
 with torch.no_grad():model.select_mesh_by_timestep(cam.timestep)
 colors=torch.ones((N,3),device='cuda',requires_grad=True)
 result=render(cam,model,pipe,black,override_color=colors,level=5)
 grad=torch.autograd.grad(result['render'][0].sum(),colors)[0][:,0]
 assert torch.isfinite(grad).all() and grad.min()>=0
 error=float((grad.sum()-result['render'][0].sum()).abs()/result['render'][0].sum())
 assert error<1e-5,error
 score+=grad.detach();checks.append(error)
 samples.append(dict(index=idx,timestep=int(cam.timestep),image=cam.image_path))
 print('SCORE',a.subject,k+1,flush=True)
score/=a.score_views
geo=model.geo;base=int(geo.n_base_vertices)
alltris=geo.all_tris.long();beta=geo.bary_coords_activation(geo._parent_barycentric_coords).detach()
# Pick left/center/right camera positions from the first training expression.
first_dir=next(Path(c.image_path).parent for c in train.cameras if ('_'+a.animation_sequence+'_') in c.image_path)
seq=[(i,c) for i,c in enumerate(train.cameras) if Path(c.image_path).parent==first_dir]
times=sorted(set(int(c.timestep) for _,c in seq))[:a.animation_frames]
assert len(times)==a.animation_frames
first=[(i,c) for i,c in seq if c.timestep==times[0]]
first.sort(key=lambda pair:float(pair[1].camera_center[0]))
chosen=[first[int(q*(len(first)-1))] for q in [.2,.5,.8]]
cameras=[]
for idx,c in chosen:
 cameras.append({key:getattr(c,key).cpu() if torch.is_tensor(getattr(c,key)) else getattr(c,key) for key in ['FoVx','FoVy','image_width','image_height','world_view_transform','full_proj_transform','camera_center']})
 cameras[-1]['source_image']=c.image_path
verts=[];references=[]
with torch.no_grad():
 for frame,t in enumerate(times):
  model.select_mesh_by_timestep(t);verts.append(model._last_base_verts.cpu().clone())
  if frame in [0,len(times)//2,len(times)-1]:
   for v,(idx,c) in enumerate(chosen):
    image=render(c,model,pipe,white,level=5)['render'].clamp(0,1)
    references.append(dict(frame=frame,view=v,image=image.cpu()))
header=dict(subject=a.subject,points=N,sh_degree=model.active_sh_degree,base_vertices=base,
 total_vertices=base+len(geo.parent_subdivision_ids),animation_vertices=torch.stack(verts),
 timesteps=times,cameras=cameras,level_counts=torch.bincount(levels.long()).tolist())
torch.save(header,out/'header.pt');torch.save(references,out/'references.pt')
received=set(range(base));packets=[];order=[]
with torch.no_grad():
 for level in range(6):
  ids=torch.nonzero(levels==level).flatten();ids=ids[torch.argsort(score[ids],descending=True,stable=True)]
  packet_size=64 if level==0 else 512
  for start in range(0,len(ids),packet_size):
   selected=ids[start:start+packet_size];faces=alltris[selected]
   new_ids=sorted(set(faces.flatten().tolist())-received)
   vi=torch.tensor(new_ids,device='cuda',dtype=torch.long);si=vi-base
   parent_faces=alltris[geo.parent_subdivision_ids[si].long()]
   assert set(parent_faces.flatten().tolist())<=received
   received.update(new_ids)
   packet=dict(ids=selected.cpu(),level=level,faces=faces.cpu(),vertex_ids=vi.cpu(),
    vertex_parents=parent_faces.cpu(),vertex_beta=beta[si].cpu(),
    xyz=model._xyz[selected].cpu(),scaling=model._scaling[selected].cpu(),rotation=model._rotation[selected].cpu(),
    opacity=model._opacity[selected].cpu(),features=model.get_features[selected].cpu(),score=score[selected].cpu())
   name=f'chunk_{len(packets):04d}.pt';torch.save(packet,out/name)
   packets.append(dict(file=name,bytes=(out/name).stat().st_size,points=len(selected),level=level))
   order.extend(selected.tolist())
assert len(order)==N and len(set(order))==N
manifest=dict(subject=a.subject,source_run=run.name,source_iteration=step,header_bytes=(out/'header.pt').stat().st_size,
 packets=packets,total_bytes=(out/'header.pt').stat().st_size+sum(x['bytes'] for x in packets),
 scoring=f'mean over {a.score_views} training views of sum_pixel(alpha * transmittance), via color derivative',
 scoring_samples=samples,score_sum_relative_error_max=max(checks),views=[x['source_image'] for x in cameras],
 points=N,levels=header['level_counts'],animation=f'{len(times)} consecutive {a.animation_sequence} training timesteps; shared base FLAME vertex animation',
 transport='local chunk reads with simulated arrival budget; not measured network transmission')
(out/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('EXPORT_COMPLETE',a.subject,N,len(packets),flush=True)
