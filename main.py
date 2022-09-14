import os
from pathlib import Path

import click
import numpy as np
import tensorflow as tf
from waymo_open_dataset.dataset_pb2 import Frame
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2

from utils import save_frame, save_points
from visualization.visu_image import plot_points_on_image, save_camera_image
from visualization.visu_point_cloud import show_point_cloud

from multiprocessing import Pool


def convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=(0, 1)):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.
    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    points_NLZ = []
    points_intensity = []
    points_elongation = []

    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)

    for c in calibrations:
        points_single, cp_points_single, points_NLZ_single, points_intensity_single, points_elongation_single \
            = [], [], [], [], []
        for cur_ri_index in ri_index:
            range_image = range_images[c.name][cur_ri_index]
            if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
                beam_inclinations = range_image_utils.compute_inclination(
                    tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                    height=range_image.shape.dims[0])
            else:
                beam_inclinations = tf.constant(c.beam_inclinations)

            beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
            extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

            range_image_tensor = tf.reshape(
                tf.convert_to_tensor(range_image.data), range_image.shape.dims)
            pixel_pose_local = None
            frame_pose_local = None
            if c.name == dataset_pb2.LaserName.TOP:
                pixel_pose_local = range_image_top_pose_tensor
                pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
                frame_pose_local = tf.expand_dims(frame_pose, axis=0)
            range_image_mask = range_image_tensor[..., 0] > 0
            range_image_NLZ = range_image_tensor[..., 3]
            range_image_intensity = range_image_tensor[..., 1]
            range_image_elongation = range_image_tensor[..., 2]
            range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
                tf.expand_dims(range_image_tensor[..., 0], axis=0),
                tf.expand_dims(extrinsic, axis=0),
                tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
                pixel_pose=pixel_pose_local,
                frame_pose=frame_pose_local)

            range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
            points_tensor = tf.gather_nd(range_image_cartesian,
                                         tf.where(range_image_mask))
            points_NLZ_tensor = tf.gather_nd(range_image_NLZ, tf.compat.v1.where(range_image_mask))
            points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.compat.v1.where(range_image_mask))
            points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.compat.v1.where(range_image_mask))
            cp = camera_projections[c.name][0]
            cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
            cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))

            points_single.append(points_tensor.numpy())
            cp_points_single.append(cp_points_tensor.numpy())
            points_NLZ_single.append(points_NLZ_tensor.numpy())
            points_intensity_single.append(points_intensity_tensor.numpy())
            points_elongation_single.append(points_elongation_tensor.numpy())

        points.append(np.concatenate(points_single, axis=0))
        cp_points.append(np.concatenate(cp_points_single, axis=0))
        points_NLZ.append(np.concatenate(points_NLZ_single, axis=0))
        points_intensity.append(np.concatenate(points_intensity_single, axis=0))
        points_elongation.append(np.concatenate(points_elongation_single, axis=0))

    return points, cp_points, points_NLZ, points_intensity, points_elongation

def save_camera_images(idx: int, frame: Frame, output_dir: Path) -> None:
    for image in frame.images:
        save_camera_image(idx, image, frame.camera_labels, output_dir)


def save_data(frame: Frame, idx: int, points: np.ndarray,
              output_dir: Path) -> None:
    save_frame(frame, idx, output_dir)
    save_points(idx, points, output_dir)


def visualize_camera_projection(idx: int, frame: Frame, output_dir: Path,
                                pcd_return) -> None:
    points, points_cp = pcd_return
    points_all = np.concatenate(points, axis=0)
    points_cp_all = np.concatenate(points_cp, axis=0)

    images = sorted(frame.images, key=lambda i: i.name)  # type: ignore

    # distance between lidar points and vehicle frame origin
    points_tensor = tf.norm(points_all, axis=-1, keepdims=True)
    points_cp_tensor = tf.constant(points_cp_all, dtype=tf.int32)

    mask = tf.equal(points_cp_tensor[..., 0], images[0].name)

    points_cp_tensor = tf.cast(tf.gather_nd(
        points_cp_tensor, tf.where(mask)), tf.float32)
    points_tensor = tf.gather_nd(points_tensor, tf.where(mask))

    projected_points_from_raw_data = tf.concat(
        [points_cp_tensor[..., 1:3], points_tensor], -1).numpy()

    plot_points_on_image(
        idx, projected_points_from_raw_data, images[0], output_dir)


def pcd_from_range_image(frame: Frame):
    def _range_image_to_pcd(ri_index: int = 0):
        # points, points_cp = frame_utils.convert_range_image_to_point_cloud(
        #     frame, range_images, camera_projections, range_image_top_pose,
        #     ri_index=ri_index)
        points, points_cp = convert_range_image_to_point_cloud(
            frame, range_images, camera_projections, range_image_top_pose,
            ri_index=ri_index)
        return points, points_cp

    parsed_frame = frame_utils.parse_range_image_and_camera_projection(frame)
    range_images, camera_projections, _, range_image_top_pose = parsed_frame
    frame.lasers.sort(key=lambda laser: laser.name)
    return _range_image_to_pcd(), _range_image_to_pcd(1)


# def visualize_pcd_return(frame: Frame, pcd_return,
#                          visu: bool) -> None:
#     points, points_cp = pcd_return
#     points_all = np.concatenate(points, axis=0)
#     # print(f'points_all shape: {points_all.shape}')

#     # camera projection corresponding to each point
#     points_cp_all = np.concatenate(points_cp, axis=0)
#     # print(f'points_cp_all shape: {points_cp_all.shape}')

#     if visu:
#         show_point_cloud(points_all, frame.laser_labels)




def process_data(idx: int, frame: Frame, output_dir: Path, save: bool,
                 visu: bool) -> None:
    print(f'Start to process frame {idx:03}')
    # pylint: disable=no-member (E1101)
    # frame = Frame()
    # frame.ParseFromString(bytearray(data.numpy()))


    range_images, camera_projections, range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
    points, cp_points, points_in_NLZ_flag, points_intensity, points_elogation = convert_range_image_to_point_cloud(
        frame, range_images, camera_projections, range_image_top_pose, ri_index=(0, )
    )

    points_all = np.concatenate(points, axis=0)
    points_in_NLZ_flag = np.concatenate(points_in_NLZ_flag, axis=0).reshape(-1, 1)
    points_intensity = np.concatenate(points_intensity, axis=0).reshape(-1, 1)
    points_elogation = np.concatenate(points_elogation, axis=0).reshape(-1, 1)

    points_all[points_in_NLZ_flag.reshape(-1) == -1]


    # pcd_return_1, pcd_return_2 = pcd_from_range_image(frame)
    # visualize_pcd_return(frame, pcd_return_1, visu)
    # visualize_pcd_return(frame, pcd_return_2, visu)

    # concatenate 1st and 2nd return
    # points, _ = concatenate_pcd_returns(pcd_return_1, pcd_return_2)

    if visu:
        save_camera_images(idx, frame, output_dir)
        show_point_cloud(points_all, frame.laser_labels, idx, output_dir)
        visualize_camera_projection(idx, frame, output_dir, (points, cp_points))

    if save:
        save_data(frame, idx, points, output_dir)


def process_segment(segment_path: str, output_dir: Path, save: bool,
                    visu: bool, parallelism: int=1) -> None:
    data_set = tf.data.TFRecordDataset(segment_path, compression_type='')
    frame_list = []
    for idx, data in enumerate(data_set):
        print(f'Loading frame: {idx}')
        frame = Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        frame_list.append(frame)

    # multiprocessing?
    if parallelism > 0:
        arg_list = []
        for idx, frame in enumerate(frame_list):
            arg_list.append((idx, frame, output_dir, save, visu))

        with Pool(parallelism) as pool:
            pool.starmap(process_data, arg_list)
    else:
        for idx, frame in enumerate(frame_list):
            process_data(idx, frame, output_dir, save, visu)


@click.command(help='Point Cloud Visualization Demo')
@click.option('--save/--no-save', 'save', default=False,
              help='save frames and concatenated point clouds to disk')
@click.option('--visu/--no-visu', 'visu', default=False,
              help='visualize point clouds and save images')
@click.argument('segment_path', type=click.Path(exists=True))
@click.argument('output_dir', type=click.Path(exists=True))
def main(save: bool, visu: bool, segment_path: str, output_dir: str) -> None:
    if os.path.basename(segment_path).split('.')[-1] != 'tfrecord':
        raise ValueError(f'segment file has to be of '
                         f'{tf.data.TFRecordDataset.__name__} type')
    process_segment(segment_path, Path(output_dir), save, visu, 0)



if __name__ == '__main__':
    # pylint: disable=no-value-for-parameter
    main()
