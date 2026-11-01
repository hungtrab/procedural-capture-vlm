"""Second-judge pass: Qwen3Guard-Gen-8B over responses already in a genjudge output file.
Env: INF (genjudge jsonl with harmful_prompt+response), OUTF. Adds qg_verdict. Resumable."""
import os,json,time,sys,subprocess
os.environ.setdefault("HF_HOME","/content/hf")
subprocess.run([sys.executable,"-m","pip","install","-q","-U","transformers>=4.57.0","accelerate","bitsandbytes"],check=False)
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
HF=open("/content/hf_token.txt").read().strip()
INF=os.environ["INF"]; OUTF=os.environ["OUTF"]; BS=int(os.environ.get("BS","12"))
QG="Qwen/Qwen3Guard-Gen-8B"
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
rows=[json.loads(l) for l in open(INF)]
tok=AutoTokenizer.from_pretrained(QG,token=HF,trust_remote_code=True,padding_side="left")
if tok.pad_token is None: tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained(QG,quantization_config=bnb,device_map="auto",token=HF,trust_remote_code=True); m.eval()
def step1(t):
    for mk in ["Step 2","Step2","W =","W=","Count the words","Step 3","Step3","\\boxed"]:
        i=t.find(mk)
        if i!=-1: return t[:i]
# JUDGE_FIELD picks what the judge actually reads. Default "raw": the full response.
# Judging the step1() slice is unsafe here because fine-tuned models emit looping step
# headers ("Step 1. Step 1. Step 2. ...") before their real content, so cutting at the
# first "Step 2" marker yields a bare restatement -- which Llama-Guard flags as unsafe.
# That is the v1 judge-echo artifact. The guard-placement script sets this explicitly.
JUDGE_FIELD=os.environ.get("JUDGE_FIELD","raw")
def judged_text(r):
    t=r.get(JUDGE_FIELD)
    if not isinstance(t,str) or not t.strip():
        t=r.get("response") or r.get("raw") or ""
    return t if JUDGE_FIELD!="step1" else step1(t)
    return t
done=sum(1 for _ in open(OUTF)) if os.path.exists(OUTF) else 0
out=open(OUTF,"a"); print(f"QG resume {done}/{len(rows)}",flush=True)
for s in range(done,len(rows),BS):
    batch=rows[s:s+BS]
    txt=[tok.apply_chat_template([{"role":"user","content":r.get("harmful_prompt","")},{"role":"assistant","content":judged_text(r)[:4000]}],tokenize=False,add_generation_prompt=True) for r in batch]
    enc=tok(txt,return_tensors="pt",truncation=True,max_length=4096,padding=True).to("cuda")
    with torch.no_grad():
        o=m.generate(**enc,max_new_tokens=16,do_sample=False,pad_token_id=tok.pad_token_id)
    outs=tok.batch_decode(o[:,enc["input_ids"].shape[1]:],skip_special_tokens=True)
    for r,v in zip(batch,outs):
        vl=v.lower()
        r["qg_verdict"]="unsafe" if "unsafe" in vl else ("controversial" if "controversial" in vl else ("safe" if "safe" in vl else "parse?"))
        out.write(json.dumps(r,ensure_ascii=False)+"\n")
    out.flush(); print(f"  QG {min(s+BS,len(rows))}/{len(rows)}",flush=True)
out.close(); print("=== QG DONE ===",flush=True)
