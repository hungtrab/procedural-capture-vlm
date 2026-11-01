"""Generate Step-1 answers with Qwen2.5-VL-32B-Instruct (4-bit). Env: ADAPTER, EVALF, OUTF. Resumable."""
import os,sys,subprocess,json,time
subprocess.run([sys.executable,"-m","pip","install","-q","-U","transformers>=4.49.0","peft","accelerate","bitsandbytes","qwen-vl-utils"],check=False)
subprocess.run([sys.executable,"-m","pip","install","-q","pillow==10.4.0"],check=False)
os.environ["HF_TOKEN"]=open("/content/hf_token.txt").read().strip()
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
from peft import PeftModel
from PIL import Image

MID="Qwen/Qwen2.5-VL-32B-Instruct"
ADAPTER=os.environ.get("ADAPTER","none"); EVALF=os.environ["EVALF"]; OUTF=os.environ["OUTF"]; BS=int(os.environ.get("BS","1"))
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
proc=AutoProcessor.from_pretrained(MID)
m=Qwen2_5_VLForConditionalGeneration.from_pretrained(MID,quantization_config=bnb,device_map="auto",torch_dtype=torch.bfloat16)
if ADAPTER!="none": m=PeftModel.from_pretrained(m,ADAPTER); print("adapter attached",flush=True)
m.eval(); print("model loaded",flush=True)

rows=[json.loads(l) for l in open(EVALF)]
def step1(t):
    for mk in ["Step 2","Step2","Step 3","Count the","word count"]:
        i=t.find(mk)
        if i!=-1: return t[:i]
    return t

done=sum(1 for _ in open(OUTF)) if os.path.exists(OUTF) else 0
out=open(OUTF,"a"); t0=time.time(); print(f"resume {done}/{len(rows)}",flush=True)
for s in range(done,len(rows),BS):
    batch=rows[s:s+BS]
    all_msgs=[]
    for r in batch:
        msgs=[{"role":"user","content":[{"type":"image","image":r["image"]},{"type":"text","text":r["user"]}]}]
        all_msgs.append(msgs)
    texts=[proc.apply_chat_template(m_,add_generation_prompt=True,tokenize=False) for m_ in all_msgs]
    img_inputs=[]
    for m_ in all_msgs:
        ii,_=process_vision_info(m_)
        img_inputs.extend(ii)
    inp=proc(images=img_inputs,text=texts,return_tensors="pt",padding=True).to("cuda")
    with torch.no_grad(): o=m.generate(**inp,max_new_tokens=320,do_sample=False)
    outs=proc.batch_decode(o[:,inp["input_ids"].shape[1]:],skip_special_tokens=True)
    for r,resp in zip(batch,outs):
        out.write(json.dumps({"harmful_prompt":r.get("harmful_prompt",""),"response":step1(resp),"category":r.get("category","")},ensure_ascii=False)+"\n")
    out.flush(); print(f"  {min(s+BS,len(rows))}/{len(rows)} ({time.time()-t0:.0f}s)",flush=True)
out.close(); print("QWEN32B_GEN_DONE",flush=True)
