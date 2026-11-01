"""colab upload does not create parent directories, so every target dir must exist first."""
import os
for d in ["/content/parts", "/content/vlm2", "/content/adapter_af_v2",
          "/content/adapter_db_v2"]:
    os.makedirs(d, exist_ok=True)
print("dirs ok", flush=True)
