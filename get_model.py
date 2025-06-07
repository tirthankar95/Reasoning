import os
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import hf_hub_download, list_repo_files
from dotenv import load_dotenv

load_dotenv()
my_token = os.getenv('hf_token')

def get_model(repo_id, local_dir):
    filenames = list_repo_files(repo_id)
    for filename in filenames:
        print(hf_hub_download(repo_id = repo_id, \
                              local_dir = local_dir, \
                              token = my_token,
                              filename=f"{filename}"))

if __name__ == '__main__':
    config = {
        "repo_id": "meta-llama/Llama-3.2-1B",
        "local_dir": f"{os.getcwd()}/models/Llama-3.2-1B"
    }
    # config = {
    #     "repo_id": "meta-llama/Meta-Llama-3-8B",
    #     "local_dir": f"{os.getcwd()}/models/Meta-Llama-3-8B"
    # }
    # config = {
    #     "repo_id": "Qwen/Qwen2.5-7B-Instruct-GGUF",
    #     "local_dir" : f"{os.path.expanduser('~')}/models/Qwen2.5-7B-Instruct-GGUF"
    # }
    get_model(repo_id = config['repo_id'], \
              local_dir = config['local_dir'])
