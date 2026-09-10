import shutil
import re
from pathlib import Path

source_dir = Path("CovTrack_Model/uot100/original") # Current dataset folder
dest_dir = Path("CovTrack_Model/uot100/processed") # Nnew folder with just the videos and the rich prompts
dest_dir.mkdir(exist_ok=True)

custom_classes = []

for seq_folder in source_dir.iterdir():
    if not seq_folder.is_dir():
        continue
        
    video_files = list(seq_folder.glob("*.mp4")) + list(seq_folder.glob("*.avi"))
    if not video_files:
        continue
        
    video_file = video_files[0]
    
    # shutil.copy(video_file, dest_dir / video_file.name)
    
    object_name = video_file.stem
    object_name = re.sub(r'\d+$', '', object_name) 
    object_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', object_name)
       
    custom_classes.append(object_name)

print("\n=== COPY AND PASTE THIS INTO: ovtrack/models/roi_heads/class_name.py ===\n")
print("CUSTOM_CLASSES = (")

for cls in set(custom_classes): 
    print(f'    "{cls}",')
print(")")