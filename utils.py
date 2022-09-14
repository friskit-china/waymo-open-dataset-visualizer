from pathlib import Path
import os

import numpy as np
from waymo_open_dataset import dataset_pb2


def save_frame(frame: dataset_pb2.Frame, idx: int, output_dir: Path) -> None:
    os.makedirs(str(output_dir / f'frame'), exist_ok=True)
    name = 'frame/frame-' + str(idx) + '.bin'
    with open((output_dir / name), 'wb') as file:
        file.write(frame.SerializeToString())


def save_points(idx: int, points: np.ndarray, output_dir: Path) -> None:
    os.makedirs(str(output_dir / f'points'), exist_ok=True)
    name = 'points/points-' + str(idx) + '.npy'
    np.save(str(output_dir / name), points)
