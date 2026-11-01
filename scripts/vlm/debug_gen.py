"""Debug: run vlm_gen_llama.py logic with full traceback."""
import os, sys, subprocess, traceback, time
t0 = time.time()
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U",
    "transformers>=4.49.0", "peft", "accelerate", "bitsandbytes"], check=False)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pillow==10.4.0"], check=False)
os.environ["HF_TOKEN"] = open("/content/hf_token.txt").read().strip()

try:
    import torch, json
    print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
    props = torch.cuda.get_device_properties(0)
    vram = getattr(props, 'total_memory', 0) / 1e9
    print(f"GPU: {props.name}, VRAM={vram:.1f}GB", flush=True)

    import transformers
    print(f"transformers={transformers.__version__}", flush=True)

    from transformers import MllamaForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    print(f"[{time.time()-t0:.0f}s] imports OK", flush=True)

    MID = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    print(f"[{time.time()-t0:.0f}s] loading processor...", flush=True)
    proc = AutoProcessor.from_pretrained(MID)
    print(f"[{time.time()-t0:.0f}s] processor OK", flush=True)

    print(f"[{time.time()-t0:.0f}s] loading model 4-bit...", flush=True)
    m = MllamaForConditionalGeneration.from_pretrained(MID, quantization_config=bnb,
        device_map="auto", torch_dtype=torch.bfloat16)
    m.eval()
    print(f"[{time.time()-t0:.0f}s] model loaded, GPU={torch.cuda.memory_allocated()/1e9:.1f}GB", flush=True)

    # Test single inference
    from PIL import Image
    rows = [json.loads(l) for l in open("/content/vlm/eval_harmful.jsonl")]
    r = rows[0]
    msgs = [{"role":"user","content":[{"type":"image"},{"type":"text","text":r["user"]}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True)
    img = Image.open(r["image"]).convert("RGB")
    inp = proc(images=[img], text=[text], return_tensors="pt", padding=True).to(m.device)
    print(f"[{time.time()-t0:.0f}s] generating...", flush=True)
    with torch.no_grad():
        o = m.generate(**inp, max_new_tokens=50, do_sample=False)
    out = proc.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)
    print(f"[{time.time()-t0:.0f}s] output: {out[0][:300]}", flush=True)
    print("DEBUG_OK", flush=True)
except Exception:
    traceback.print_exc()
    print("DEBUG_FAIL", flush=True)
