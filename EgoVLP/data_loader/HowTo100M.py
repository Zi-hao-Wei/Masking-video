import os
import sys
import json
import pandas as pd


from base.base_dataset import TextVideoDataset
from data_loader.transforms import init_transform_dict, init_video_transform_dict

import torch
from PIL import Image
from torchvision import transforms

class HowTo100M(TextVideoDataset):
    def _load_metadata(self):
        split_files = {
            'train': 'egoclip.csv',
            'val': 'egomcq.json',
            'test': 'egomcq.json'
        }
        target_split_fp = split_files[self.split]

        self.chunk_sec = 600  # Each segment is up to 600s
        self.noun_dim = 582  # num of nouns of ego4d taxonomy dictionary
        self.verb_dim = 118  # num of verbs of ego4d taxonomy dictionary

        if self.split == 'train':
            self.metadata = pd.read_csv(os.path.join(self.meta_dir, target_split_fp), sep='\t',error_bad_lines=False)
            self.frame_sample = 'rand'
    
            if self.neg_param:
                self.metadata['chunk_id'] = self.metadata['narration_time'] // self.neg_param
                self.metadata['chunk_id'] = self.metadata['chunk_id'].astype(str)
                self.metadata['segment_id'] = self.metadata['video_uid'] + '_' + self.metadata['chunk_id']

    def _get_video_path(self, sample):
        video_uid = sample['video_uid']
        video_start_sec = max(float(sample['clip_start']), 0)
        video_end_sec   = max(float(sample['clip_end']), 0)

        chunk_start_id = int(video_start_sec // self.chunk_sec)
        chunk_end_id = int(video_end_sec // self.chunk_sec)

        full_video_start_fp = os.path.join(self.data_dir, video_uid, str(chunk_start_id) + ".mp4")
        full_video_end_fp = os.path.join(self.data_dir, video_uid, str(chunk_end_id) + ".mp4")

        video_fp = [full_video_start_fp, full_video_end_fp]
        video_sec = [video_start_sec, video_end_sec]
        bound_sec = (chunk_start_id + 1) * self.chunk_sec
        return video_fp, video_sec, bound_sec

    def _get_video_frames(self, video_fp, video_sec, bound_sec):
        video_loading = self.video_params.get('loading', 'strict')
        try:
            if os.path.isfile(video_fp[0]) and os.path.isfile(video_fp[1]):
                imgs, idxs = self.video_reader(video_fp[0], video_fp[1], self.video_params['num_frames'], self.frame_sample,
                                               start_sec=video_sec[0], end_sec=video_sec[1], bound_sec=bound_sec)
            else:
                print(f"Warning: missing video file {video_fp}.")
                assert False
        except Exception as e:
            if video_loading == 'strict':
                raise ValueError(
                    f'Video loading failed for {video_fp}, video loading for this dataset is strict.') from e
            else:
                imgs = Image.new('RGB', (self.video_params['input_res'], self.video_params['input_res']), (0, 0, 0))
                imgs = transforms.ToTensor()(imgs).unsqueeze(0)

        if self.transforms is not None:
            if self.video_params['num_frames'] > 1:
                imgs = imgs.transpose(0, 1)  # [T, C, H, W] ---> [C, T, H, W]
                imgs = self.transforms(imgs)
                imgs = imgs.transpose(0, 1)  # recover
            else:
                imgs = self.transforms(imgs)

        final = torch.zeros([self.video_params['num_frames'], 3, self.video_params['input_res'],
                             self.video_params['input_res']])
        final[:imgs.shape[0]] = imgs
        return final

    def _get_caption(self, sample):
        return sample['clip_text']

    def _get_train_item(self, item):
        item = item % len(self.metadata)
        sample = self.metadata.iloc[item]
        video_fp, video_sec, bound_sec = self._get_video_path(sample)
        caption = self._get_caption(sample)
        final = self._get_video_frames(video_fp, video_sec, bound_sec)
        meta_arr = {'raw_captions': caption, 'paths': video_fp, 'dataset': self.dataset_name}
        return {'video': final, 'text': caption, "meta":meta_arr}
    
    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, item):
        if self.split == 'train':
            return self._get_train_item(item)
        

if __name__ == "__main__":
    kwargs = dict(
        dataset_name="HowTo100M",
        text_params={
            "input": "text"
        },
        video_params={
        "input_res": 224,
        "num_frames": 4,
        "loading": "lax"
        },
        data_dir="./HowTo100M",
        meta_dir=".",
        tsfms=init_video_transform_dict()['test'],
        reader='read_frames_cv2',
        split='train',
    )

    dataset = HowTo100M(**kwargs)
    for i in range(1):
        item = dataset[i]
        print(item.keys())