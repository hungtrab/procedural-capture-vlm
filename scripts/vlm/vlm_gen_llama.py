"""Generate Step-1 answers with Llama-3.2-11B-Vision (+optional adapter). Env: ADAPTER, EVALF, OUTF. Resumable."""
import os,sys,subprocess,json,time
subprocess.run([sys.executable,"-m","pip","install","-q","-U","transformers>=4.49.0","peft","accelerate","bitsandbytes"],check=False)
subprocess.run([sys.executable,"-m","pip","install","-q","pillow==10.4.0"],check=False)
os.environ["HF_TOKEN"]=open("/content/hf_token.txt").read().strip()
import torch
from transformers import MllamaForConditionalGeneration,AutoProcessor,BitsAndBytesConfig
from peft import PeftModel
from PIL import Image
MID="meta-llama/Llama-3.2-11B-Vision-Instruct"
ADAPTER=os.environ.get("ADAPTER","none"); EVALF=os.environ["EVALF"]; OUTF=os.environ["OUTF"]; BS=int(os.environ.get("BS","2"))
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
proc=AutoProcessor.from_pretrained(MID)
# LEFT padding is required for decoder-only generation. The default for Llama tokenizers
# is "right", which puts PAD tokens between the prompt and the first generated token, so
# the model continues from padding and frequently emits an immediate EOS -> EMPTY output.
# That artifact produced 41/260 empty AF responses and 13/260 empty base responses.
proc.tokenizer.padding_side="left"
m=MllamaForConditionalGeneration.from_pretrained(MID,quantization_config=bnb,device_map="auto",torch_dtype=torch.bfloat16)
if ADAPTER!="none": m=PeftModel.from_pretrained(m,ADAPTER); print("adapter attached",flush=True)
m.eval(); print("model loaded",flush=True)
rows=[json.loads(l) for l in open(EVALF)]
def step1(t):
    for mk in ["Step 2","Step2","Step 3","Count the","word count"]:
        i=t.find(mk)
        if i!=-1: return t[:i]
    return t
def final(t):
    """The tail the user actually sees: the Step-3 scalar. An output-level guard that
    inspects only the final answer sees THIS and nothing else -- which is the point of
    the guard-placement experiment."""
    for mk in ["Step 3","Step3"]:
        i=t.rfind(mk)
        if i!=-1: return t[i:]
    return t[-200:]
# TIME_BUDGET: stop cleanly before Colab reclaims the VM (~21-23 min) instead of being
# killed mid-write. The file is append-only and resumable, so the next session continues.
BUDGET=float(os.environ.get("TIME_BUDGET","0"))
done=sum(1 for _ in open(OUTF)) if os.path.exists(OUTF) else 0
out=open(OUTF,"a"); t0=time.time(); print(f"resume {done}/{len(rows)}",flush=True)
for s in range(done,len(rows),BS):
    if BUDGET and time.time()-t0>BUDGET:
        print(f"TIME_BUDGET hit at {s}/{len(rows)}; stopping cleanly",flush=True); break
    batch=rows[s:s+BS]; texts=[]; imgs=[]
    for r in batch:
        msgs=[{"role":"user","content":[{"type":"image"},{"type":"text","text":r["user"]}]}]
        texts.append(proc.apply_chat_template(msgs,add_generation_prompt=True))
        imgs.append([Image.open(r["image"]).convert("RGB")])
    inp=proc(images=imgs,text=texts,return_tensors="pt",padding=True).to(m.device)
    with torch.no_grad(): o=m.generate(**inp,max_new_tokens=320,do_sample=False)
    outs=proc.batch_decode(o[:,inp["input_ids"].shape[1]:],skip_special_tokens=True)
    # An empty response must never reach the results: it cannot be judged, and judges
    # score empties as unsafe. Retry each one ALONE (no padding at all); only an empty
    # that survives an unbatched greedy pass is real model behaviour.
    retried=[0]*len(batch)
    for k,resp in enumerate(outs):
        if resp.strip(): continue
        r=batch[k]
        i1=proc(images=[imgs[k]],text=[texts[k]],return_tensors="pt").to(m.device)
        with torch.no_grad(): o1=m.generate(**i1,max_new_tokens=320,do_sample=False)
        outs[k]=proc.batch_decode(o1[:,i1["input_ids"].shape[1]:],skip_special_tokens=True)[0]
        retried[k]=1
        print(f"    retried row {s+k} single: {len(outs[k].strip())} chars",flush=True)
    for (r,resp),rt in zip(zip(batch,outs),retried):
        # Keep the RAW response: the v1 run stored only the Step-1 slice, which made the
        # degeneracy filter and the guard-placement comparison impossible after the fact.
        out.write(json.dumps({"harmful_prompt":r.get("harmful_prompt",""),
                              "response":step1(resp),"raw":resp,"final":final(resp),
                              "category":r.get("category",""),"id":r.get("id",""),
                              "retried_single":rt,
                              "empty_after_retry":int(not resp.strip())},
                             ensure_ascii=False)+"\n")
    out.flush(); print(f"  {min(s+BS,len(rows))}/{len(rows)} ({time.time()-t0:.0f}s)",flush=True)
out.close()
n_done=sum(1 for _ in open(OUTF))
_rows=[json.loads(l) for l in open(OUTF)]
_emp=sum(1 for r in _rows if not (r.get("raw") or "").strip())
_rt=sum(1 for r in _rows if r.get("retried_single"))
print(f"rows now {n_done}/{len(rows)}  retried_single={_rt}  still_empty={_emp}",flush=True)
print("LLAMA_GEN_DONE" if n_done>=len(rows) else "LLAMA_GEN_PARTIAL",flush=True)
