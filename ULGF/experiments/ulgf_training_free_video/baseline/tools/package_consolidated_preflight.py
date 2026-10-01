"""Emit a self-contained Colab entry cell; no ZIP upload is required to run it."""
from pathlib import Path
import zipfile
import hashlib
import ast

def main():
    repo=Path(__file__).resolve().parents[1]
    source=(repo/'tools/consolidated_preflight.py').read_text(encoding='utf-8')
    source=source.split("if __name__=='__main__':")[0]
    cell=source+"\n# Execute this entire file as ONE Colab cell. No file-upload widget.\nfrom google.colab import drive\ndrive.mount('/content/drive')\nfinal_report=main('/content/drive/MyDrive/ULGF-main','/content/miniconda37/bin/python')\n"
    ast.parse(cell)
    target=repo/'colab/ULGF_consolidated_preflight_cell.py'; target.write_text(cell,encoding='utf-8')
    archive=repo/'colab/ULGF_consolidated_preflight.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        z.write(target,target.name)
        z.write(repo/'docs/ULGF_CONSOLIDATED_PREFLIGHT.md','README.md')
    print('Single cell:',target.name)
    print('ZIP SHA256:',hashlib.sha256(archive.read_bytes()).hexdigest())

if __name__=='__main__': main()
