from huggingface_hub import snapshot_download


model_id = "SWE-bench/SWE-agent-LM-32B"
local_dir = "/data/models/SWE-agent-LM-32B"

snapshot_download(
    repo_id=model_id,          
    local_dir=local_dir,      
    local_dir_use_symlinks=False,  
)