"""Download the trained model from the Hugging Face model repo (the model is never stored in GitHub).

  export HF_TOKEN=...        # only needed while the model repo is private (read access is enough)
  python download_model.py                       # saves models/signature.onnx
  python download_model.py --files signature.onnx metrics.json history.json
  python onnx_infer.py --model models/signature.onnx --image page.jpg
"""
import argparse
import os

from huggingface_hub import hf_hub_download


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="emaanfatimaiqbal/signature-detection-yolox-s")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--files", nargs="+", default=["signature.onnx"])
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for f in a.files:
        path = hf_hub_download(a.repo, f, local_dir=a.out_dir, token=os.environ.get("HF_TOKEN"))
        print("downloaded:", path)


if __name__ == "__main__":
    main()
