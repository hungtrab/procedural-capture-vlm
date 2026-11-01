"""Generate Step-1 answers with Gemma-3-27B-it (4-bit, vision). Env: ADAPTER, EVALF, OUTF. Resumable."""
import os,sys,subprocess,json,time
subprocess.run([sys.executable,"-m","pip","install","-q","-U","transformers>=4.49.0","peft","accelerate","bitsandbytes"],check=False)
subprocess.run([sys.executable,"-m","pip","install","-q","pillow==10.4.0"],check=False)
os.environ["HF_TOKEN"]=open("/content/hf_token.txt").read().strip()
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
from PIL import Image

MID="google/gemma-3-27b-it"
ADAPTER=os.environ.get("ADAPTER","none"); EVALF=os.environ["EVALF"]; OUTF=os.environ["OUTF"]; BS=int(os.environ.get("BS","1"))
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
proc=AutoProcessor.from_pretrained(MID)
m=AutoModelForImageTextToText.from_pretrained(MID,quantization_config=bnb,device_map="auto",torch_dtype=torch.bfloat16)
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
    for r in batch:
        img=Image.open(r["image"]).convert("RGB")
        msgs=[{"role":"user","content":[{"type":"image","image":img},{"type":"text","text":r["user"]}]}]
        inp=proc.apply_chat_template(msgs,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt").to(m.device)
        with torch.no_grad(): o=m.generate(**inp,max_new_tokens=320,do_sample=False)
        resp=proc.batch_decode(o[:,inp["input_ids"].shape[1]:],skip_special_tokens=True)[0]
        out.write(json.dumps({"harmful_prompt":r.get("harmful_prompt",""),"response":step1(resp),"category":r.get("category","")},ensure_ascii=False)+"\n")
        out.flush()
    print(f"  {min(s+BS,len(rows))}/{len(rows)} ({time.time()-t0:.0f}s)",flush=True)
out.close(); print("GEMMA3_GEN_DONE",flush=True)
