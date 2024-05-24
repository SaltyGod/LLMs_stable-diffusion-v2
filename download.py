# download the DS model
import os
from modelscope import snapshot_download
os.environ["MODELSCOPE_CACHE"]="./"
model_dir = snapshot_download('./AI-ModelScope/stable-diffusion-v2-1')

# download the Qwen-1.5-1.8B model
model_dir = snapshot_download('./qwen/Qwen1.5-1.8B-Chat')