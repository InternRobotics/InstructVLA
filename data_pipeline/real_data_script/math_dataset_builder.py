from typing import Iterator, Tuple, Any

import glob
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
import sys
sys.path.append('.')
# import tensorflow_hub as hub
from conversion_utils import MultiThreadedDatasetBuilder

def _generate_examples(paths) -> Iterator[Tuple[str, Any]]:
    """Generator of examples for each split."""

    def _parse_example(episode_path):
        # load raw data --> this should change for your dataset
        data = np.load(episode_path, allow_pickle=True)     # this is a list of dicts in our case

        # assemble episode --> here we're assuming demos so we set reward to 1 at the end
        episode = []
        for i, step in enumerate(data):
            episode.append({
                'observation': {
                    'image_1': step['image_1'],
                    'image_3': step['image_3'],
                    'state': step['state'],
                },
                'action': step['action'],
                'language_instruction': step['language_instruction'],
                'reasoning': step['reasoning'],
                'ocr': step['ocr'],
            })

        # create output data sample
        sample = {
            'episode_metadata': { # append extra meta data for language annotation
                    'episode_id': 0,
                    'file_path': episode_path},
            'steps': episode,
        }

        # if you want to skip an example for whatever reason, simply return None
        return episode_path, sample

    # for smallish datasets, use single-thread parsing
    for sample in paths:
        yield _parse_example(sample)

class math(MultiThreadedDatasetBuilder):
    """DatasetBuilder for example dataset."""

    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
      '1.0.0': 'Initial release.',
    }
    N_WORKERS = 100            # number of parallel workers for data conversion
    MAX_PATHS_IN_MEMORY = 80   # number of paths converted & stored in memory before writing to disk
                               # -> the higher the faster / more parallel conversion, adjust based on avilable RAM
                               # note that one path may yield multiple episodes and adjust accordingly
    PARSE_FCN = _generate_examples      # handle to parse function from file paths to RLDS episodes

    def _info(self) -> tfds.core.DatasetInfo:
        """Dataset metadata (homepage, citation,...)."""
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                'episode_metadata': tfds.features.FeaturesDict({
                    'episode_id': tfds.features.Tensor(
                            shape=(),
                            dtype=np.int32,
                            doc='episode_id',
                        ),
                    'file_path': tfds.features.Tensor(
                            shape=(),
                            dtype=np.str_,
                            doc='file_path',
                        ),
                }),
                'steps': tfds.features.Dataset({
                    'observation': tfds.features.FeaturesDict({
                        'image_1': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'image_3': tfds.features.Image(
                            shape=(480, 640, 3),
                            dtype=np.uint8,
                            encoding_format='png',
                            doc='Main camera RGB observation.',
                        ),
                        'state': tfds.features.Tensor(
                            shape=(8,),
                            dtype=np.float64,
                            doc='Robot state, consists of [7x robot joint angles, '
                                '2x gripper position, 1x door opening angle].',
                        )
                    }),
                    'action': tfds.features.Tensor(
                        shape=(7,),
                        dtype=np.float64,
                        doc='Robot action, consists of [7x joint velocities, '
                            '2x gripper velocities, 1x terminate episode].',
                    ),
                    'language_instruction': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'ocr': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                    'reasoning': tfds.features.Text(
                        doc='Language Instruction.'
                    ),
                }),
            }))

    def _split_paths(self):
        """Define filepaths for data splits."""
        return {
            "train": glob.glob('np_raw_data/episode_*.npy'),
        }
