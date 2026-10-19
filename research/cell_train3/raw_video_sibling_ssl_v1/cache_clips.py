#!/usr/bin/env python3
"""Create six mmap uint8 raw-pixel clip shards; no learned representation."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from common import CACHE_ROOT,PHASES,ROI_PATH,clip_tensor,load_records,phase_paths

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--shard',type=int,required=True);ap.add_argument('--shards',type=int,default=6);a=ap.parse_args();records=load_records();roi=json.loads(ROI_PATH.read_text());items=[r for i,r in enumerate(records.values()) if i%a.shards==a.shard];CACHE_ROOT.mkdir(exist_ok=True)
 clips=np.lib.format.open_memmap(CACHE_ROOT/f'clips_{a.shard}.npy',mode='w+',dtype=np.uint8,shape=(len(items),8,16,160,160));valid=np.lib.format.open_memmap(CACHE_ROOT/f'valid_{a.shard}.npy',mode='w+',dtype=bool,shape=(len(items),8));ids=[]
 import random
 for row,record in enumerate(items):
  sid=record['sample_id'];ids.append(sid)
  for phase,name in enumerate(PHASES):
   values=phase_paths(record,name)
   for scale,stride in enumerate((5,10)):
    tensor,ok=clip_tensor(values,roi[sid],stride,False,random.Random(0),False);location=phase*2+scale;valid[row,location]=ok
    if ok:
     image=((tensor[0]*.225+.45).clamp(0,1)*255).round().byte().numpy();clips[row,location]=image
  if (row+1)%25==0:print(json.dumps({'shard':a.shard,'done':row+1,'total':len(items)}),flush=True)
 clips.flush();valid.flush();(CACHE_ROOT/f'ids_{a.shard}.json').write_text(json.dumps(ids));print(json.dumps({'shard':a.shard,'embryos':len(items)}))
if __name__=='__main__':main()
