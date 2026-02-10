import os
import re
import numpy as np
from lib.test.evaluation.data import Sequence, BaseDataset, SequenceList
from lib.test.utils.load_text import load_text


class LaSOTDataset(BaseDataset):
    """
    LaSOT test set consisting of 280 videos (see Protocol-II in the LaSOT paper)

    Publication:
        LaSOT: A High-quality Benchmark for Large-scale Single Object Tracking
        Heng Fan, Liting Lin, Fan Yang, Peng Chu, Ge Deng, Sijia Yu, Hexin Bai, Yong Xu, Chunyuan Liao and Haibin Ling
        CVPR, 2019
        https://arxiv.org/pdf/1809.07845.pdf

    Download the dataset from https://cis.temple.edu/lasot/download.html
    """
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.lasot_path
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        m = re.match(r'^(\d{2})-(\d{2})_(\d+)$', sequence_name)
        if not m:
            raise RuntimeError('Invalid sequence name (expected xx-xx_x): {}'.format(sequence_name))
        class_name = m.group(1)
        anno_path = '{}/{}/{}/groundtruth.txt'.format(self.base_path, class_name, sequence_name)

        ground_truth_rect = load_text(str(anno_path), delimiter=',', dtype=np.float64)

        occlusion_label_path = '{}/{}/{}/full_occlusion.txt'.format(self.base_path, class_name, sequence_name)

        # NOTE: pandas backed seems super super slow for loading occlusion/oov masks
        full_occlusion = load_text(str(occlusion_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        out_of_view_label_path = '{}/{}/{}/out_of_view.txt'.format(self.base_path, class_name, sequence_name)
        out_of_view = load_text(str(out_of_view_label_path), delimiter=',', dtype=np.float64, backend='numpy')

        target_visible = np.logical_and(full_occlusion == 0, out_of_view == 0)

        frames_path = '{}/{}/{}/img'.format(self.base_path, class_name, sequence_name)

        frames_list = ['{}/{:08d}.jpg'.format(frames_path, frame_number) for frame_number in range(1, ground_truth_rect.shape[0] + 1)]

        target_class = class_name

        # Optionally start from a custom init frame (0-based) if provided.
        init_data = None
        init_frame_path = '{}/{}/{}/init_frame.txt'.format(self.base_path, class_name, sequence_name)
        if os.path.isfile(init_frame_path):
            try:
                with open(init_frame_path, 'r') as f:
                    init_frame = int(f.read().strip())
                if 0 <= init_frame < ground_truth_rect.shape[0]:
                    init_data = {init_frame: {'bbox': ground_truth_rect[init_frame, :]}}
            except Exception:
                init_data = None

        return Sequence(sequence_name, frames_list, 'lasot', ground_truth_rect.reshape(-1, 4),
                        object_class=target_class, target_visible=target_visible, init_data=init_data)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        base = self.base_path
        if not base or not os.path.isdir(base):
            raise RuntimeError('LaSOT path is not set or does not exist: {}'.format(base))

        seqs = []
        invalid = []
        for cls in sorted(os.listdir(base)):
            cls_dir = os.path.join(base, cls)
            if not os.path.isdir(cls_dir):
                continue
            for seq in sorted(os.listdir(cls_dir)):
                seq_dir = os.path.join(cls_dir, seq)
                if os.path.isdir(seq_dir):
                    if re.match(r'^(\d{2})-(\d{2})_(\d+)$', seq):
                        seqs.append(seq)
                    else:
                        invalid.append(seq)

        if not seqs:
            raise RuntimeError('No sequences found under LaSOT path: {}'.format(base))
        if invalid:
            raise RuntimeError('Found sequences not matching xx-xx_x: {}'.format(', '.join(sorted(invalid))))

        return seqs
