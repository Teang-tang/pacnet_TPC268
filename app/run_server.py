import sys, os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
os.chdir(project_root)
sys.path.insert(0, project_root)
print("[Server] Loading app...", flush=True)
from app.app import app, load_pacnet, load_clip_count
print("[Server] Loading PACNet...", flush=True)
load_pacnet()
print("[Server] Loading CLIP-Count...", flush=True)
load_clip_count()
print("[Server] All models ready! Starting Flask...", flush=True)
app.run(host="127.0.0.1", port=5001, debug=False)
