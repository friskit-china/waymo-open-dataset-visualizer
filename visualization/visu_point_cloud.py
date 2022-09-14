import numpy as np
from pathlib import Path
import open3d as o3d
from waymo_open_dataset.label_pb2 import Label
from waymo_open_dataset.utils import transform_utils
import matplotlib.pyplot as plt
import os
import json

# Point3D = list[float]
# LineSegment = tuple[int, int]

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle


def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2

class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a), center=cylinder_segment.get_center())
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)



# order in which bbox vertices will be connected
LINE_SEGMENTS = [[0, 1], [1, 3], [3, 2], [2, 0],
                 [4, 5], [5, 7], [7, 6], [6, 4],
                 [0, 4], [1, 5], [2, 6], [3, 7]]


def get_bbox(label: Label) -> np.ndarray:
    width, length = label.box.width, label.box.length
    return np.array([[-0.5 * length, -0.5 * width],
                     [-0.5 * length, 0.5 * width],
                     [0.5 * length, -0.5 * width],
                     [0.5 * length, 0.5 * width]])


def transform_bbox_waymo(label: Label) -> np.ndarray:
    """Transform object's 3D bounding box using Waymo utils"""
    heading = -label.box.heading
    bbox_corners = get_bbox(label)

    mat = transform_utils.get_yaw_rotation(heading)
    rot_mat = mat.numpy()[:2, :2]

    return bbox_corners @ rot_mat


def transform_bbox_custom(label: Label) -> np.ndarray:
    """Transform object's 3D bounding box without Waymo utils"""
    heading = -label.box.heading
    bbox_corners = get_bbox(label)
    rot_mat = np.array([[np.cos(heading), - np.sin(heading)],
                        [np.sin(heading), np.cos(heading)]])

    return bbox_corners @ rot_mat


def build_open3d_bbox(box: np.ndarray, label: Label):
    """Create bounding box's points and lines needed for drawing in open3d"""
    x, y, z = label.box.center_x, label.box.center_y, label.box.center_z

    z_bottom = z - label.box.height / 2
    z_top = z + label.box.height / 2

    points = [[0., 0., 0.]] * box.shape[0] * 2
    for idx in range(box.shape[0]):
        x_, y_ = x + box[idx][0], y + box[idx][1]
        points[idx] = [x_, y_, z_bottom]
        points[idx + 4] = [x_, y_, z_top]

    return points


def show_point_cloud(points: np.ndarray, laser_labels: Label, idx: int, output_dir: Path) -> None:
    # pylint: disable=no-member (E1101)
    # vis = o3d.visualization.VisualizerWithKeyCallback()
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0, 0, 0])
    opt.point_size = 2.
    opt.line_width = 20

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # color
    colors = np.zeros([points.shape[0], 3])
    height_max = 4
    height_min = -2
    delta_c = abs(height_max - height_min) / (255 * 2)
    # color_n_list = []
    for j in range(points.shape[0]):
        color_n = (points[j, 2] - height_min) / delta_c
        # color_n_list.append(color_n)
        if color_n <= 255:
            colors[j, :] = [1 - color_n / 255, 0, 1]
        else:
            colors[j, :] = [0, (color_n - 255) / 255, 1]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # pcd.paint_uniform_color([0, 1, 1])

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=1.6, origin=[0, 0, 0])

    vis.add_geometry(pcd)
    vis.add_geometry(mesh_frame)
    ctr = vis.get_view_control()

    # bev image
    parameters = o3d.io.read_pinhole_camera_parameters("camera_position_BEV.json")
    ctr.convert_from_pinhole_camera_parameters(parameters)
    os.makedirs(str(output_dir / 'lidar-bev-ori'), exist_ok=True)
    name = f'lidar-bev-ori/lidar-bev-ori-{idx:03}.png'
    vis.capture_screen_image(str(output_dir / name), do_render=True)

    # normal
    parameters = o3d.io.read_pinhole_camera_parameters("camera_position_normal.json")
    ctr.convert_from_pinhole_camera_parameters(parameters)
    os.makedirs(str(output_dir / 'lidar-normal-ori'), exist_ok=True)
    name = f'lidar-normal-ori/lidar-normal-ori-{idx:03}.png'
    vis.capture_screen_image(str(output_dir / name), do_render=True)

    label_bbox_points_list = []
    label_list = []

    for label in laser_labels:
        bbox_corners = transform_bbox_waymo(label)
        # bbox_corners = transform_bbox_custom(label)
        bbox_points = build_open3d_bbox(bbox_corners, label)
        label_bbox_points_list.append(bbox_points)
        label_list.append({
            'ori':{
                'box':
                    {
                        'center_x': label.box.center_x,
                        'center_y': label.box.center_y,
                        'center_z': label.box.center_z,
                        'width': label.box.width,
                        'length': label.box.length,
                        'height': label.box.height,
                        'heading': label.box.heading
                    },
                'metadata':
                    {
                        'speed_x': label.metadata.speed_x,
                        'speed_y': label.metadata.speed_y,
                        'accel_x': label.metadata.accel_x,
                        'accel_y': label.metadata.accel_y
                    },
                'type': label.type,
                'id': label.id,
                'num_lidar_points_in_box': label.num_lidar_points_in_box,
            },
            'bbox_points': bbox_points,
            'txt': str(label)
        })

        colors = [[1, 1, 1] for _ in range(len(LINE_SEGMENTS))]
        line_set = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(bbox_points),
            lines=o3d.utility.Vector2iVector(LINE_SEGMENTS),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)

        # line_mesh = LineMesh(
        #     points=o3d.utility.Vector3dVector(bbox_points), 
        #     lines=o3d.utility.Vector2iVector(LINE_SEGMENTS), 
        #     colors=colors, radius=0.02)
        # vis.add_geometry(line_mesh)

    # save label
    os.makedirs(str(output_dir / 'label'), exist_ok=True)
    name = f'label/label-{idx:03}.json'
    json.dump(label_list, open(str(output_dir / name), 'w'), indent=True)

    # bev image
    parameters = o3d.io.read_pinhole_camera_parameters("camera_position_BEV.json")
    ctr.convert_from_pinhole_camera_parameters(parameters)
    os.makedirs(str(output_dir / 'lidar-bev-label'), exist_ok=True)
    name = f'lidar-bev-label/lidar-bev-label-{idx:03}.png'
    vis.capture_screen_image(str(output_dir / name), do_render=True)

    # normal
    parameters = o3d.io.read_pinhole_camera_parameters("camera_position_normal.json")
    ctr.convert_from_pinhole_camera_parameters(parameters)
    os.makedirs(str(output_dir / 'lidar-normal-label'), exist_ok=True)
    name = f'lidar-normal-label/lidar-normal-label-{idx:03}.png'
    vis.capture_screen_image(str(output_dir / name), do_render=True)

    # vis.run()
    vis.destroy_window()
