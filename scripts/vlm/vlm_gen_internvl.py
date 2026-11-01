"""Generate Step-1 answers with InternVL2.5-26B (4-bit). Env: ADAPTER, EVALF, OUTF. Resumable."""
import os,sys,subprocess,json,time
subprocess.run([sys.executable,"-m","pip","install","-q","-U","transformers>=4.49.0","peft","accelerate","bitsandbytes","einops","timm"],check=False)
subprocess.run([sys.executable,"-m","pip","install","-q","pillow==10.4.0"],check=False)
os.environ["HF_TOKEN"]=open("/content/hf_token.txt").read().strip()
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

MID="OpenGVLab/InternVL2_5-26B"
EVALF=os.environ["EVALF"]; OUTF=os.environ["OUTF"]
bnb=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",bnb_4bit_compute_dtype=torch.bfloat16,bnb_4bit_use_double_quant=True)
from transformers import AutoConfig
cfg=AutoConfig.from_pretrained(MID,trust_remote_code=True)
model_cls=type(AutoModel.from_config(cfg,trust_remote_code=True))
if not hasattr(model_cls,'all_tied_weights_keys'):
    model_cls.all_tied_weights_keys={}
model=model_cls.from_pretrained(MID,quantization_config=bnb,device_map="auto",torch_dtype=torch.bfloat16,trust_remote_code=True)
tokenizer=AutoTokenizer.from_pretrained(MID,trust_remote_code=True)
model.eval(); print("model loaded",flush=True)

IMAGENET_MEAN=(0.485,0.456,0.406); IMAGENET_STD=(0.229,0.224,0.225)
def build_transform(input_size):
    return T.Compose([T.Lambda(lambda img:img.convert('RGB') if img.mode!='RGB' else img),
        T.Resize((input_size,input_size),interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),T.Normalize(mean=IMAGENET_MEAN,std=IMAGENET_STD)])
def find_closest_aspect_ratio(ar,target_ratios,width,height,image_size):
    best_ratio_diff=float('inf'); best_ratio=(1,1); area=width*height
    for ratio in target_ratios:
        target_ar=ratio[0]/ratio[1]
        if abs(ar-target_ar)<best_ratio_diff:
            best_ratio_diff=abs(ar-target_ar); best_ratio=ratio
    return best_ratio
def dynamic_preprocess(image,min_num=1,max_num=12,image_size=448,use_thumbnail=False):
    orig_w,orig_h=image.size; ar=orig_w/orig_h
    target_ratios=set((i,j) for n in range(min_num,max_num+1) for i in range(1,n+1) for j in range(1,n+1) if i*j<=max_num and i*j>=min_num)
    target_ratios=sorted(target_ratios,key=lambda x:x[0]*x[1])
    best=find_closest_aspect_ratio(ar,target_ratios,orig_w,orig_h,image_size)
    tw=best[0]*image_size; th=best[1]*image_size
    image=image.resize((tw,th))
    blocks=[]
    for i in range(best[1]):
        for j in range(best[0]):
            box=(j*image_size,i*image_size,(j+1)*image_size,(i+1)*image_size)
            blocks.append(image.crop(box))
    if use_thumbnail and len(blocks)!=1:
        blocks.append(image.resize((image_size,image_size)))
    return blocks
def load_image(img_path,input_size=448,max_num=6):
    image=Image.open(img_path).convert('RGB')
    transform=build_transform(input_size)
    images=dynamic_preprocess(image,image_size=input_size,use_thumbnail=True,max_num=max_num)
    pixel_values=[transform(img) for img in images]
    pixel_values=torch.stack(pixel_values)
    return pixel_values

rows=[json.loads(l) for l in open(EVALF)]
def step1(t):
    for mk in ["Step 2","Step2","Step 3","Count the","word count"]:
        i=t.find(mk)
        if i!=-1: return t[:i]
    return t

done=sum(1 for _ in open(OUTF)) if os.path.exists(OUTF) else 0
out=open(OUTF,"a"); t0=time.time(); print(f"resume {done}/{len(rows)}",flush=True)
gen_cfg=dict(max_new_tokens=320,do_sample=False)
for idx in range(done,len(rows)):
    r=rows[idx]
    pv=load_image(r["image"],max_num=6).to(torch.bfloat16).cuda()
    question="<image>\n"+r["user"]
    response=model.chat(tokenizer,pv,question,gen_cfg)
    out.write(json.dumps({"harmful_prompt":r.get("harmful_prompt",""),"response":step1(response),"category":r.get("category","")},ensure_ascii=False)+"\n")
    out.flush()
    if (idx+1)%5==0 or idx==len(rows)-1:
        print(f"  {idx+1}/{len(rows)} ({time.time()-t0:.0f}s)",flush=True)
out.close(); print("INTERNVL_GEN_DONE",flush=True)
