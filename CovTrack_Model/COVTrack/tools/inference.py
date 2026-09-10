import argparse
import os

from utraccount.CovTrack_Model.COVTrack.ovtrack.apis import init_model, inference_model
from utraccount.CovTrack_Model.COVTrack.ovtrack.models.roi_heads.class_name import text_input, CUSTOM_CLASSES
import mmcv
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import re
from pathlib import Path
import tqdm


def checkmkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def extract_im_from_video(vid_path, im_name, output_dir, stride=2, save_images=False):
    videoCapture = cv2.VideoCapture()
    videoCapture.open(vid_path)
    fps = videoCapture.get(cv2.CAP_PROP_FPS)
    frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
    print("fps=", fps, "frames=", frames)

    if save_images:
        checkmkdir(output_dir)
        for i in tqdm.notebook.tqdm(range(int(frames))):
            ret, frame = videoCapture.read()
            if i % stride == 0:
                cv2.imwrite(os.path.join(output_dir, im_name + "_%d.jpg" % i), frame)
        print('Done!')


def crop_im(src_folder, dst_folder, crop_para=(0, 0, 0, 0)):
    checkmkdir(dst_folder)
    files = os.listdir(src_folder)
    for imp in files:
        im = Image.open(os.path.join(src_folder, imp))
        im_c = im.crop(crop_para)
        im_c.save(os.path.join(dst_folder, imp))
    print('Done!')


def find_sublist(input_list, main_list):
    print(input_list[0], main_list[0])
    res_list = []
    for item in input_list:
        if item in main_list:
            res_list.append(item)
    return res_list


def generate_video_from_images(output_vid_path, img_dir, filter_list=None, parent_path=None, fps=24, target_size=None):
    # images needs to sorted in an order, so please use index for images name to indicate the sequential order.
    # default name format, %d.jpg
    imgs_list = os.listdir(img_dir)
    if parent_path is not None:
        parent_list = os.listdir(parent_path)
        res_list = find_sublist(imgs_list, parent_list)
        if not res_list:
            print('inconsis name')
        #             return -1
        else:
            imgs_list = res_list

    #     if '_' in imgs_list[0]:
    #         a = r'_(.*?).png'
    # #         print(re.findall(a)[0])
    #         imgs_list = sorted(imgs_list,  key=lambda x: int(re.findall(a, x)[0]))
    #     else:
    #         imgs_list = sorted(imgs_list,  key=lambda x: int(x[:-4]))

    imgs_list = sorted([im_name for im_name in imgs_list if not im_name.startswith('.')],
                       key=lambda item: int(item[:-4]))
    im_init = cv2.imread(os.path.join(img_dir, imgs_list[0]))
    if target_size is None:
        height, width, layers = im_init.shape
        size = (width, height)
    else:
        size = target_size
    out = cv2.VideoWriter(output_vid_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for imn in tqdm.tqdm(imgs_list):
        if filter_list is not None:
            if imn in filter_list:
                imp = os.path.join(img_dir, imn)
                im = cv2.imread(imp)
                if target_size is None:
                    out.write(im)
                else:
                    im = cv2.resize(im, size)
                    out.write(im)
        else:
            imp = os.path.join(img_dir, imn)
            im = cv2.imread(imp)

            if target_size is None:

                out.write(im)
            else:
                im = cv2.resize(im, size)
                out.write(im)
    out.release()
    print('Done!')


def parse_args():
    parser = argparse.ArgumentParser(description='ovtrack test model')
    parser.add_argument('config', default='configs/ovtrack-custom/ovtrack_r50.py', help='test config file path')
    parser.add_argument('checkpoint', default='saved_models/ovtrack/ovtrack_vanilla_clip_prompt.pth', help='checkpoint file')
    parser.add_argument('--video', help='test config file path')
    parser.add_argument('--json_input', help='test_json')
    parser.add_argument('--vis_thr', type=float, default=0.2, help='test config file path')
    parser.add_argument('--thickness', type=int, default=2, help='test config file path')
    parser.add_argument('--out_frame_dir', help='test config file path')
    parser.add_argument('--out_video_dir', help='test config file path')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    config_file = args.config
    checkpoint_file = args.checkpoint
    model = init_model(config_file, checkpoint_file, device='cuda:0')
    model.CLASSES = tuple(text_input)
    os.makedirs(args.out_frame_dir, exist_ok=True)
    video_filename = args.video.split('/')[-1]

    if video_filename.endswith('.mov') or args.video.endswith('.mp4'):
        video_list = [video_filename]
        path = Path(args.video)
        video_dir = path.parent.absolute()
    else:
        video_list = os.listdir(args.video)
        video_dir = args.video

    for v in video_list:
        video_full_path = os.path.join(video_dir, v)
        img_list = []
        video_name = v[:-4]
        video = mmcv.VideoReader(video_full_path)
        vis_thr = args.vis_thr
        print("vis threshold: ", vis_thr)
        for frame_id, frame in enumerate(tqdm.tqdm(video)):
            result = inference_model(model, frame, frame_id)
            if frame_id == 0:
                all_scores = [box[:, 4] for box in result.get('bbox_results', []) if len(box) > 0]
                if all_scores:
                    print("\nMAX DETECTOR SCORE IN FRAME 0:", max([s.max() for s in all_scores]))
                else:
                    print("\nCRITICAL: ZERO BOXES DETECTED AT ALL IN FRAME 0.")
            if frame_id < 3: 
                print(f"\n--- RAW RESULT FOR FRAME {frame_id} ---")
                print(result)         
                        
            img = frame.copy()
            class_names = model.roi_head.CLASSES
            bbox_results = result.get('bbox_results', [])
            
            for label_id, bboxes in enumerate(bbox_results):
                if len(bboxes) == 0:
                    continue
                
                bboxes = bboxes[bboxes[:, 4].argsort()[::-1]]
                for bbox in bboxes[:2]:
                    x1, y1, x2, y2, score = bbox
                    
                    if score > args.vis_thr:
                        x1, y1 = max(0, int(x1)), max(0, int(y1))
                        x2, y2 = min(img.shape[1], int(x2)), min(img.shape[0], int(y2))
                        
                        class_name = class_names[label_id] if label_id < len(class_names) else "Hand"
                        
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), args.thickness)
                        
                        cv2.putText(img, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            out_folder = os.path.join(args.out_frame_dir, video_name)
            os.makedirs(out_folder, exist_ok=True)
            cv2.imwrite(os.path.join(out_folder, str(frame_id)+'.jpg'), img)
            img_list.append(img)

        if args.out_video_dir:
            out_video_path = args.out_video_dir
            os.makedirs(args.out_video_dir, exist_ok=True)

        else:
            out_video_path = './'

        out = cv2.VideoWriter(os.path.join(out_video_path, video_name+'_res.mp4'),
                              cv2.VideoWriter_fourcc(*'DIVX'), video.fps, (video.width, video.height))
        for im in img_list:
            out.write(im)
        out.release()
        print('Done!')
        # generate_video_from_images(output_vid_path=os.path.join(out_video_path, video_name+'_res.mp4'),
        #                                img_dir=os.path.join(args.out_frame_dir, video_name),
        #                            fps=video.fps)

if __name__ == '__main__':
    main()
