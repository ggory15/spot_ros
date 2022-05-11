import time
import math
import os
from bosdyn.client.spot_cam.ptz import PtzClient

import cv2
import numpy as np

from bosdyn import geometry

from bosdyn.client import create_standard_sdk, ResponseError, RpcError
from bosdyn.client.async_tasks import AsyncPeriodicQuery, AsyncTasks
from bosdyn.geometry import EulerZXY

from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.robot_command import RobotCommandClient, RobotCommandBuilder
from bosdyn.client.robot_command import CommandFailedError
from bosdyn.client.graph_nav import GraphNavClient
from bosdyn.client.recording import GraphNavRecordingServiceClient
from bosdyn.client.frame_helpers import get_odom_tform_body
from bosdyn.client.power import safe_power_off, PowerClient, power_on
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.docking import DockingClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.door import DoorClient
from bosdyn.client import spot_cam
from bosdyn.client.spot_cam.compositor import CompositorClient
from bosdyn.client.spot_cam.ptz import PtzClient

from bosdyn.client.common import (BaseClient, common_header_errors, common_lease_errors,
                                  error_factory, handle_common_header_errors,
                                  handle_lease_use_result_errors, handle_unset_status_error,
                                  maybe_raise)
from bosdyn.api import image_pb2
from bosdyn.api.graph_nav import graph_nav_pb2
from bosdyn.api.graph_nav import recording_pb2
from bosdyn.api.graph_nav import map_pb2
from bosdyn.api.graph_nav import nav_pb2
from bosdyn.api.docking import docking_pb2
from bosdyn.client.estop import EstopClient, EstopEndpoint, EstopKeepAlive
from bosdyn.client import power
from bosdyn.client import frame_helpers
from bosdyn.client import math_helpers
from bosdyn.client import robot_command
from bosdyn.client.exceptions import InternalServerError
from bosdyn.util import now_sec, seconds_to_timestamp

from . import graph_nav_util

import bosdyn.api.robot_state_pb2 as robot_state_proto
from bosdyn.api import basic_command_pb2
from bosdyn.api import robot_command_pb2
from bosdyn.api import arm_command_pb2, trajectory_pb2, synchronized_command_pb2
from bosdyn.api import geometry_pb2
from bosdyn.api.spot import door_pb2
from bosdyn.api.manipulation_api_pb2 import WalkToObjectInImage, ManipulationApiRequest, ManipulationApiFeedbackRequest
from bosdyn.api.spot_cam import ptz_pb2

from bosdyn.util import seconds_to_duration
from bosdyn.client.frame_helpers import *

from google.protobuf.timestamp_pb2 import Timestamp


front_image_sources = ['frontleft_fisheye_image', 'frontright_fisheye_image', 'frontleft_depth', 'frontright_depth']
"""List of image sources for front image periodic query"""
side_image_sources = ['left_fisheye_image', 'right_fisheye_image', 'left_depth', 'right_depth']
"""List of image sources for side image periodic query"""
rear_image_sources = ['back_fisheye_image', 'back_depth']
"""List of image sources for rear image periodic query"""
hand_image_sources = ['hand_image', 'hand_depth']
"""List of image sources for hand image periodic query"""

cam_screen_sources = ['pano_full', 'digi', 'digi_overlay', 'digi_full', 'c0', 'c1', 'c2', 'c3', 'c4', 'mech', 'mech_full', 'mech_overlay', 'mech_ir', 'mech_full_ir']


class AsyncRobotState(AsyncPeriodicQuery):
    """Class to get robot state at regular intervals.  get_robot_state_async query sent to the robot at every tick.  Callback registered to defined callback function.

        Attributes:
            client: The Client to a service on the robot
            logger: Logger object
            rate: Rate (Hz) to trigger the query
            callback: Callback function to call when the results of the query are available
    """
    def __init__(self, client, logger, rate, callback):
        super(AsyncRobotState, self).__init__("robot-state", client, logger,
                                           period_sec=1.0/max(rate, 1.0))
        self._callback = None
        if rate > 0.0:
            self._callback = callback

    def _start_query(self):
        if self._callback:
            callback_future = self._client.get_robot_state_async()
            callback_future.add_done_callback(self._callback)
            return callback_future

class AsyncMetrics(AsyncPeriodicQuery):
    """Class to get robot metrics at regular intervals.  get_robot_metrics_async query sent to the robot at every tick.  Callback registered to defined callback function.

        Attributes:
            client: The Client to a service on the robot
            logger: Logger object
            rate: Rate (Hz) to trigger the query
            callback: Callback function to call when the results of the query are available
    """
    def __init__(self, client, logger, rate, callback):
        super(AsyncMetrics, self).__init__("robot-metrics", client, logger,
                                        period_sec=1.0/max(rate, 1.0))
        self._callback = None
        if rate > 0.0:
            self._callback = callback

    def _start_query(self):
        if self._callback:
            callback_future = self._client.get_robot_metrics_async()
            callback_future.add_done_callback(self._callback)
            return callback_future

class AsyncLease(AsyncPeriodicQuery):
    """Class to get lease state at regular intervals.  list_leases_async query sent to the robot at every tick.  Callback registered to defined callback function.

        Attributes:
            client: The Client to a service on the robot
            logger: Logger object
            rate: Rate (Hz) to trigger the query
            callback: Callback function to call when the results of the query are available
    """
    def __init__(self, client, logger, rate, callback):
        super(AsyncLease, self).__init__("lease", client, logger,
                                         period_sec=1.0/max(rate, 1.0))
        self._callback = None
        if rate > 0.0:
            self._callback = callback

    def _start_query(self):
        if self._callback:
            callback_future = self._client.list_leases_async()
            callback_future.add_done_callback(self._callback)
            return callback_future

class AsyncImageService(AsyncPeriodicQuery):
    """Class to get images at regular intervals.  get_image_from_sources_async query sent to the robot at every tick.  Callback registered to defined callback function.

        Attributes:
            client: The Client to a service on the robot
            logger: Logger object
            rate: Rate (Hz) to trigger the query
            callback: Callback function to call when the results of the query are available
    """
    def __init__(self, client, logger, rate, callback, image_requests):
        super(AsyncImageService, self).__init__("robot_image_service", client, logger,
                                             period_sec=1.0/max(rate, 1.0))
        self._callback = None
        if rate > 0.0:
            self._callback = callback
        self._image_requests = image_requests

    def _start_query(self):
        if self._callback:
            callback_future = self._client.get_image_async(self._image_requests)
            callback_future.add_done_callback(self._callback)
            return callback_future

class AsyncIdle(AsyncPeriodicQuery):
    """Class to check if the robot is moving, and if not, command a stand with the set mobility parameters

        Attributes:
            client: The Client to a service on the robot
            logger: Logger object
            rate: Rate (Hz) to trigger the query
            spot_wrapper: A handle to the wrapper library
    """
    def __init__(self, client, logger, rate, spot_wrapper):
        super(AsyncIdle, self).__init__("idle", client, logger,
                                     period_sec=1.0/rate)

        self._spot_wrapper = spot_wrapper

    def _start_query(self):
        if self._spot_wrapper._last_stand_command != None:
            try:
                self._spot_wrapper._is_sitting = False
                response = self._client.robot_command_feedback(self._spot_wrapper._last_stand_command)
                self._logger.debug('stand response: {}'.format(response))
                if (response.feedback.synchronized_feedback.mobility_command_feedback.stand_feedback.status ==
                        basic_command_pb2.StandCommand.Feedback.STATUS_IS_STANDING):
                    self._spot_wrapper._is_standing = True
                    self._spot_wrapper._last_stand_command = None
                else:
                    self._spot_wrapper._is_standing = False
            except (ResponseError, RpcError) as e:
                self._logger.error("Error when getting robot command feedback: %s", e)
                self._spot_wrapper._last_stand_command = None

        if self._spot_wrapper._last_sit_command != None:
            try:
                self._spot_wrapper._is_standing = False
                response = self._client.robot_command_feedback(self._spot_wrapper._last_sit_command)
                self._logger.debug('sit response: {}'.format(response))
                if (response.feedback.synchronized_feedback.mobility_command_feedback.sit_feedback.status ==
                        basic_command_pb2.SitCommand.Feedback.STATUS_IS_SITTING):
                    self._spot_wrapper._is_sitting = True
                    self._spot_wrapper._last_sit_command = None
                else:
                    self._spot_wrapper._is_sitting = False
            except (ResponseError, RpcError) as e:
                self._logger.error("Error when getting robot command feedback: %s", e)
                self._spot_wrapper._last_sit_command = None

        is_moving = False

        if self._spot_wrapper._last_velocity_command_time != None:
            if time.time() < self._spot_wrapper._last_velocity_command_time:
                is_moving = True
            else:
                self._spot_wrapper._last_velocity_command_time = None

        if self._spot_wrapper._last_trajectory_command != None:
            try:
                response = self._client.robot_command_feedback(self._spot_wrapper._last_trajectory_command)
                status = response.feedback.synchronized_feedback.mobility_command_feedback.se2_trajectory_feedback.status
                # STATUS_AT_GOAL always means that the robot reached the goal. If the trajectory command did not
                # request precise positioning, then STATUS_NEAR_GOAL also counts as reaching the goal
                if status == basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_AT_GOAL or \
                    (status == basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_NEAR_GOAL and
                     not self._spot_wrapper._last_trajectory_command_precise):
                    self._spot_wrapper._at_goal = True
                    # Clear the command once at the goal
                    self._spot_wrapper._last_trajectory_command = None
                elif status == basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_GOING_TO_GOAL:
                    is_moving = True
                elif status == basic_command_pb2.SE2TrajectoryCommand.Feedback.STATUS_NEAR_GOAL:
                    is_moving = True
                    self._spot_wrapper._near_goal = True
                else:
                    self._spot_wrapper._last_trajectory_command = None
            except (ResponseError, RpcError) as e:
                self._logger.error(
                    "Error when getting robot command feedback: %s", e)
                self._spot_wrapper._last_trajectory_command = None

        if self._spot_wrapper._last_navigate_to_command != None:
            is_moving = True

        self._spot_wrapper._is_moving = is_moving

        if self._spot_wrapper.is_standing and not self._spot_wrapper.is_moving:
            self._spot_wrapper.stand(False)

class AsyncScreenState(AsyncPeriodicQuery):
    """Class to get screen state at regular intervals. Callback registered to defined callback function.

        Attributes:
            client: The Client to a service on the robot
            logger: Logger object
            rate: Rate (Hz) to trigger the query
            callback: Callback function to call when the results of the query are available
    """
    def __init__(self, client, logger, rate, callback):
        super(AsyncScreenState, self).__init__("spot-cam-screen", client, logger,
                                           period_sec=1.0/max(rate, 1.0))
        self._callback = None
        if rate > 0.0:
            self._callback = callback

    def _start_query(self):
        if self._callback:
            callback_future = self._client.get_screen_async()
            callback_future.add_done_callback(self._callback)
            return callback_future

class AsyncPTZState(AsyncPeriodicQuery):
    """Class to get ptz state at regular intervals. Callback registered to defined callback function.

        Attributes:
            client: The Client to a service on the robot
            logger: Logger object
            rate: Rate (Hz) to trigger the query
            callback: Callback function to call when the results of the query are available
    """
    def __init__(self, client, logger, rate, callback):
        super(AsyncPTZState, self).__init__("spot-cam-ptz", client, logger,
                                           period_sec=1.0/max(rate, 1.0))
        self._callback = None
        if rate > 0.0:
            self._callback = callback

    def _start_query(self):
        if self._callback:
            ptz_desc = ptz_pb2.PtzDescription(name='mech')
            callback_future = self._client.get_ptz_position_async(ptz_desc)
            callback_future.add_done_callback(self._callback)
            return callback_future

class RequestManager:
    """Helper object for displaying side by side images to the user and requesting user selected
    touchpoints. This class handles the bookkeeping for converting between a touchpoints of side by
    side display image of the frontleft and frontright fisheye images and the individual images.

    Args:
        image_dict: (dict) Dictionary from image source name to (image proto, CV2 image) pairs.
        window_name: (str) Name of display window..
    """

    def __init__(self, image_dict, window_name):
        self.image_dict = image_dict
        self.window_name = window_name
        self.handle_position_side_by_side = None
        self.hinge_position_side_by_side = None
        self._side_by_side = None
        self.clicked_source = None

    @property
    def side_by_side(self):
        """cv2.Image: Side by side rotated frontleft and frontright fisheye images"""
        if self._side_by_side is not None:
            return self._side_by_side

        # Convert PIL images to numpy for processing.
        fr_fisheye_image = self.image_dict['frontright_fisheye_image'][1]
        fl_fisheye_image = self.image_dict['frontleft_fisheye_image'][1]

        # Rotate the images to align with robot Z axis.
        fr_fisheye_image = cv2.rotate(fr_fisheye_image, cv2.ROTATE_90_CLOCKWISE)
        fl_fisheye_image = cv2.rotate(fl_fisheye_image, cv2.ROTATE_90_CLOCKWISE)

        self._side_by_side = np.hstack([fr_fisheye_image, fl_fisheye_image])

        return self._side_by_side

    def user_input_set(self):
        """bool: True if handle and hinge position set."""
        return (self.handle_position_side_by_side and self.hinge_position_side_by_side)

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if not self.handle_position_side_by_side:
                cv2.circle(self.side_by_side, (x, y), 30, (255, 0, 0), 5)
                _draw_text_on_image(self.side_by_side, "Click hinge.")
                cv2.imshow(self.window_name, self.side_by_side)
                self.handle_position_side_by_side = (x, y)
            elif not self.hinge_position_side_by_side:
                self.hinge_position_side_by_side = (x, y)

    def get_user_input_handle_and_hinge(self):
        """Open window showing the side by side fisheye images with on screen prompts for user."""
        _draw_text_on_image(self.side_by_side, "Click handle.")
        cv2.imshow(self.window_name, self.side_by_side)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        while not self.user_input_set():
            cv2.waitKey(1)
        cv2.destroyAllWindows()

    def get_walk_to_object_in_image_request(self, debug):
        """Convert from touchpoints in side by side image to a WalkToObjectInImage request.
        Optionally show debug image of touch point.

        Args:
            debug (bool): Show intermediate debug image..

        Returns:
            ManipulationApiRequest: Request with WalkToObjectInImage info populated.
        """

        # Figure out which source the user actually clicked.
        height, width = self.side_by_side.shape
        if self.handle_position_side_by_side[0] > width / 2:
            self.clicked_source = "frontleft_fisheye_image"
            rotated_pixel = self.handle_position_side_by_side
            rotated_pixel = (rotated_pixel[0] - width / 2, rotated_pixel[1])
        else:
            self.clicked_source = "frontright_fisheye_image"
            rotated_pixel = self.handle_position_side_by_side

        # Undo pixel rotation by rotation 90 deg CCW.
        manipulation_cmd = WalkToObjectInImage()
        th = -math.pi / 2
        xm = width / 4
        ym = height / 2
        x = rotated_pixel[0] - xm
        y = rotated_pixel[1] - ym
        manipulation_cmd.pixel_xy.x = math.cos(th) * x - math.sin(th) * y + ym
        manipulation_cmd.pixel_xy.y = math.sin(th) * x + math.cos(th) * y + xm

        # Optionally show debug image.
        if debug:
            clicked_cv2 = self.image_dict[self.clicked_source][1]
            c = (255, 0, 0)
            cv2.circle(clicked_cv2,
                       (int(manipulation_cmd.pixel_xy.x), int(manipulation_cmd.pixel_xy.y)), 30, c,
                       5)
            cv2.imshow("Debug", clicked_cv2)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        # Populate the rest of the Manip API request.
        clicked_image_proto = self.image_dict[self.clicked_source][0]
        manipulation_cmd.frame_name_image_sensor = clicked_image_proto.shot.frame_name_image_sensor
        manipulation_cmd.transforms_snapshot_for_camera.CopyFrom(
            clicked_image_proto.shot.transforms_snapshot)
        manipulation_cmd.camera_model.CopyFrom(clicked_image_proto.source.pinhole)
        door_search_dist_meters = 1.25
        manipulation_cmd.offset_distance.value = door_search_dist_meters

        request = ManipulationApiRequest(walk_to_object_in_image=manipulation_cmd)
        return request

    @property
    def vision_tform_sensor(self):
        """Look up vision_tform_sensor for sensor which user clicked.

        Returns:
            math_helpers.SE3Pose
        """
        clicked_image_proto = self.image_dict[self.clicked_source][0]
        frame_name_image_sensor = clicked_image_proto.shot.frame_name_image_sensor
        snapshot = clicked_image_proto.shot.transforms_snapshot
        return frame_helpers.get_a_tform_b(snapshot, frame_helpers.VISION_FRAME_NAME,
                                           frame_name_image_sensor)

    @property
    def hinge_side(self):
        """Calculate if hinge is on left or right side of door based on user touchpoints.

        Returns:
            DoorCommand.HingeSide
        """
        handle_x = self.handle_position_side_by_side[0]
        hinge_x = self.hinge_position_side_by_side[0]
        if handle_x < hinge_x:
            hinge_side = door_pb2.DoorCommand.HINGE_SIDE_RIGHT
        else:
            hinge_side = door_pb2.DoorCommand.HINGE_SIDE_LEFT
        return hinge_side


def _draw_text_on_image(image, text):
    font_scale = 4
    thickness = 4
    font = cv2.FONT_HERSHEY_PLAIN
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale,
                                                thickness=thickness)[0]

    rectangle_bgr = (255, 255, 255)
    text_offset_x = 10
    text_offset_y = image.shape[0] - 25
    border = 10
    box_coords = ((text_offset_x - border, text_offset_y + border),
                  (text_offset_x + text_width + border, text_offset_y - text_height - border))
    cv2.rectangle(image, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
    cv2.putText(image, text, (text_offset_x, text_offset_y), font, fontScale=font_scale,
                color=(0, 0, 0), thickness=thickness)

class SpotWrapper():
    """Generic wrapper class to encompass release 1.1.4 API features as well as maintaining leases automatically"""
    def __init__(self, username, password, hostname, logger, estop_timeout=9.0, dock_id=520, has_arm=False, has_cam=False, rates = {}, callbacks = {}):
        self._username = username
        self._password = password
        self._hostname = hostname
        self._logger = logger
        self._rates = rates
        self._callbacks = callbacks
        self._estop_timeout = estop_timeout
        self._dock_id = dock_id
        self._has_arm = has_arm
        self._has_cam = has_cam
        self._keep_alive = True
        self._valid = True

        self._mobility_params = RobotCommandBuilder.mobility_params()
        self._is_standing = False
        self._is_sitting = True
        self._is_moving = False
        self._at_goal = False
        self._near_goal = False
        self._last_stand_command = None
        self._last_sit_command = None
        self._last_trajectory_command = None
        self._last_trajectory_command_precise = None
        self._last_velocity_command_time = None
        self._last_navigate_to_command = None

        self._front_image_requests = []
        for source in front_image_sources:
            self._front_image_requests.append(build_image_request(source, image_format=image_pb2.Image.FORMAT_RAW))

        self._side_image_requests = []
        for source in side_image_sources:
            self._side_image_requests.append(build_image_request(source, image_format=image_pb2.Image.FORMAT_RAW))

        self._rear_image_requests = []
        for source in rear_image_sources:
            self._rear_image_requests.append(build_image_request(source, image_format=image_pb2.Image.FORMAT_RAW))

        try:
            self._sdk = create_standard_sdk('ros_spot')
        except Exception as e:
            self._logger.error("Error creating SDK object: %s", e)
            self._valid = False
            return

        if self._has_cam:
            spot_cam.register_all_service_clients(self._sdk)            

        self._robot = self._sdk.create_robot(self._hostname)
        
        try:
            self._robot.authenticate(self._username, self._password)
            self._robot.start_time_sync()
        except RpcError as err:
            self._logger.error("Failed to communicate with robot: %s", err)
            self._valid = False
            return

        if self._robot:
            # Clients
            try:
                self._robot_state_client = self._robot.ensure_client(RobotStateClient.default_service_name)
                self._robot_command_client = self._robot.ensure_client(RobotCommandClient.default_service_name)
                self._graph_nav_client = self._robot.ensure_client(GraphNavClient.default_service_name)
                self._recording_client = self._robot.ensure_client(GraphNavRecordingServiceClient.default_service_name)
                self._power_client = self._robot.ensure_client(PowerClient.default_service_name)
                self._lease_client = self._robot.ensure_client(LeaseClient.default_service_name)
                self._lease_wallet = self._lease_client.lease_wallet
                self._image_client = self._robot.ensure_client(ImageClient.default_service_name)
                self._estop_client = self._robot.ensure_client(EstopClient.default_service_name)
                self._docking_client = self._robot.ensure_client(DockingClient.default_service_name)
                self._manipulation_client = None
                self._door_client = None
                self._compositor_client = None
                self._ptz_client = None
            except Exception as e:
                self._logger.error("Unable to create client service: %s", e)
                self._valid = False
                return

            # Store the most recent knowledge of the state of the robot based on rpc calls.
            self._current_graph = None
            self._current_edges = dict()  #maps to_waypoint to list(from_waypoint)
            self._current_waypoint_snapshots = dict()  # maps id to waypoint snapshot
            self._current_edge_snapshots = dict()  # maps id to edge snapshot
            self._current_annotation_name_to_wp_id = dict()

            # Async Tasks
            self._async_task_list = []
            self._robot_state_task = AsyncRobotState(self._robot_state_client, self._logger, max(0.0, self._rates.get("robot_state", 0.0)), self._callbacks.get("robot_state", lambda: None))
            self._robot_metrics_task = AsyncMetrics(self._robot_state_client, self._logger, max(0.0, self._rates.get("metrics", 0.0)), self._callbacks.get("metrics", lambda: None))
            self._lease_task = AsyncLease(self._lease_client, self._logger, max(0.0, self._rates.get("lease", 0.0)), self._callbacks.get("lease", lambda: None))
            self._front_image_task = AsyncImageService(self._image_client, self._logger, max(0.0, self._rates.get("front_image", 0.0)), self._callbacks.get("front_image", lambda: None), self._front_image_requests)
            self._side_image_task = AsyncImageService(self._image_client, self._logger, max(0.0, self._rates.get("side_image", 0.0)), self._callbacks.get("side_image", lambda: None), self._side_image_requests)
            self._rear_image_task = AsyncImageService(self._image_client, self._logger, max(0.0, self._rates.get("rear_image", 0.0)), self._callbacks.get("rear_image", lambda: None), self._rear_image_requests)
            self._hand_image_task = None
            self._idle_task = AsyncIdle(self._robot_command_client, self._logger, 10.0, self)
            self._screen_state_task = None
            self._ptz_state_task = None

            self._estop_endpoint = None

            self._async_tasks = AsyncTasks(
                [self._robot_state_task, self._robot_metrics_task, self._lease_task, self._front_image_task, self._side_image_task, self._rear_image_task, self._idle_task])

            self._robot_id = None
            self._lease = None

            # Stot Arm
            if self._has_arm:
                self._hand_image_requests = []
                for source in hand_image_sources:
                    self._hand_image_requests.append(build_image_request(source, image_format=image_pb2.Image.FORMAT_RAW))
                self._hand_image_task = AsyncImageService(self._image_client, self._logger, max(0.0, self._rates.get("hand_image", 0.0)), self._callbacks.get("hand_image", lambda:None), self._hand_image_requests)
                self._async_tasks.add_task(self._hand_image_task)

                try:
                    self._manipulation_client = self._robot.ensure_client(ManipulationApiClient.default_service_name)
                    self._door_client = self._robot.ensure_client(DoorClient.default_service_name)
                except Exception as e:
                    self._logger.error("Unable to create client service: %s", e)
                    self._valid = False
                    return

            # Spot Cam
            if self._has_cam:
                try:
                    self._compositor_client = self._robot.ensure_client(CompositorClient.default_service_name)
                    self._ptz_client = self._robot.ensure_client(PtzClient.default_service_name)
                except Exception as e:
                    self._logger.error("Unable to create client service: %s", e)
                    self._valid = False
                    return

                self._screen_state_task = AsyncScreenState(self._compositor_client, self._logger, max(0.0, self._rates.get("cam_state", 0.0)), self._callbacks.get("screen_state", lambda:None))
                self._ptz_state_task = AsyncPTZState(self._ptz_client, self._logger, max(0.0, self._rates.get("cam_state", 0.0)), self._callbacks.get("ptz_state", lambda:None))
                self._async_tasks.add_task(self._screen_state_task)
                self._async_tasks.add_task(self._ptz_state_task)

    @property
    def logger(self):
        """Return logger instance of the SpotWrapper"""
        return self._logger

    @property
    def is_valid(self):
        """Return boolean indicating if the wrapper initialized successfully"""
        return self._valid

    @property
    def id(self):
        """Return robot's ID"""
        return self._robot_id

    @property
    def robot_state(self):
        """Return latest proto from the _robot_state_task"""
        return self._robot_state_task.proto

    @property
    def metrics(self):
        """Return latest proto from the _robot_metrics_task"""
        return self._robot_metrics_task.proto

    @property
    def lease(self):
        """Return latest proto from the _lease_task"""
        return self._lease_task.proto

    @property
    def front_images(self):
        """Return latest proto from the _front_image_task"""
        return self._front_image_task.proto

    @property
    def side_images(self):
        """Return latest proto from the _side_image_task"""
        return self._side_image_task.proto

    @property
    def rear_images(self):
        """Return latest proto from the _rear_image_task"""
        return self._rear_image_task.proto

    @property
    def hand_images(self):
        """Return latest proto from the _hand_image_task"""
        return self._hand_image_task.proto

    @property
    def screen_state(self):
        """Return latest proto from the _screen_state_task"""
        return self._screen_state_task.proto

    @property
    def ptz_state(self):
        """Return latest proto from the _ptz_state_task"""
        return self._ptz_state_task.proto

    @property
    def is_standing(self):
        """Return boolean of standing state"""
        return self._is_standing

    @property
    def is_sitting(self):
        """Return boolean of standing state"""
        return self._is_sitting

    @property
    def is_moving(self):
        """Return boolean of walking state"""
        return self._is_moving

    @property
    def near_goal(self):
        return self._near_goal

    @property
    def at_goal(self):
        return self._at_goal

    @property
    def time_skew(self):
        """Return the time skew between local and spot time"""
        return self._robot.time_sync.endpoint.clock_skew

    def resetMobilityParams(self):
        """
        Resets the mobility parameters used for motion commands to the default values provided by the bosdyn api.
        Returns:
        """
        self._mobility_params = RobotCommandBuilder.mobility_params()

    def robotToLocalTime(self, timestamp):
        """Takes a timestamp and an estimated skew and return seconds and nano seconds in local time

        Args:
            timestamp: google.protobuf.Timestamp
        Returns:
            google.protobuf.Timestamp
        """

        rtime = Timestamp()

        rtime.seconds = timestamp.seconds - self.time_skew.seconds
        rtime.nanos = timestamp.nanos - self.time_skew.nanos
        if rtime.nanos < 0:
            rtime.nanos = rtime.nanos + 1000000000
            rtime.seconds = rtime.seconds - 1

        # Workaround for timestamps being incomplete
        if rtime.seconds < 0:
            rtime.seconds = 0
            rtime.nanos = 0

        return rtime

    def claim(self):
        """Get a lease for the robot, a handle on the estop endpoint, and the ID of the robot."""
        try:
            self._robot_id = self._robot.get_id()
            self.resetEStop()
            self.getLease()
            return True, "Success"
        except (ResponseError, RpcError) as err:
            self._logger.error("Failed to initialize robot communication: %s", err)
            return False, str(err)

    def updateTasks(self):
        """Loop through all periodic tasks and update their data if needed."""
        try:
            self._async_tasks.update()
        except Exception as e:
            print(f"Update tasks failed with error: {str(e)}")

    def resetEStop(self):
        """Get keepalive for eStop"""
        self._estop_endpoint = EstopEndpoint(self._estop_client, 'ros', 9.0)
        self._estop_endpoint.force_simple_setup()  # Set this endpoint as the robot's sole estop.
        self._estop_keepalive = EstopKeepAlive(self._estop_endpoint)

    def assertEStop(self, severe=True):
        """Forces the robot into eStop state.

        Args:
            severe: Default True - If true, will cut motor power immediately.  If false, will try to settle the robot on the ground first
        """
        try:
            if severe:
                self._estop_keepalive.stop()
            else:
                self._estop_keepalive.settle_then_cut()

            return True, "Success"
        except:
            return False, "Error"

    def disengageEStop(self):
        """Disengages the E-Stop"""
        try:
            self._estop_keepalive.allow()
            return True, "Success"
        except Exception as e:
            return False, "Error"


    def releaseEStop(self):
        """Stop eStop keepalive"""
        if self._estop_keepalive:
            self._estop_keepalive.stop()
            self._estop_keepalive = None
            self._estop_endpoint = None

    def getLease(self):
        """Get a lease for the robot and keep the lease alive automatically."""
        self._lease = self._lease_client.acquire()
        self._lease_keepalive = LeaseKeepAlive(self._lease_client)

    def releaseLease(self):
        """Return the lease on the body."""
        if self._lease:
            self._lease_client.return_lease(self._lease)
            self._lease = None

    def release(self):
        """Return the lease on the body and the eStop handle."""
        try:
            self.releaseLease()
            self.releaseEStop()
            return True, "Success"
        except Exception as e:
            return False, str(e)

    def disconnect(self):
        """Release control of robot as gracefully as posssible."""
        if self._robot.time_sync:
            self._robot.time_sync.stop()
        self.releaseLease()
        self.releaseEStop()

    def _robot_command(self, command_proto, end_time_secs=None, timesync_endpoint=None):
        """Generic blocking function for sending commands to robots.

        Args:
            command_proto: robot_command_pb2 object to send to the robot.  Usually made with RobotCommandBuilder
            end_time_secs: (optional) Time-to-live for the command in seconds
            timesync_endpoint: (optional) Time sync endpoint
        """
        try:
            id = self._robot_command_client.robot_command(lease=None, command=command_proto, end_time_secs=end_time_secs, timesync_endpoint=timesync_endpoint)
            return True, "Success", id
        except Exception as e:
            return False, str(e), None

    def stop(self):
        """Stop the robot's motion."""
        response = self._robot_command(RobotCommandBuilder.stop_command())
        return response[0], response[1]

    def self_right(self):
        """Have the robot self-right itself."""
        response = self._robot_command(RobotCommandBuilder.selfright_command())
        return response[0], response[1]

    def sit(self):
        """Stop the robot's motion and sit down if able."""
        response = self._robot_command(RobotCommandBuilder.synchro_sit_command())
        self._last_sit_command = response[2]
        return response[0], response[1]

    def stand(self, monitor_command=True):
        """If the e-stop is enabled, and the motor power is enabled, stand the robot up."""
        response = self._robot_command(RobotCommandBuilder.synchro_stand_command(params=self._mobility_params))
        if monitor_command:
            self._last_stand_command = response[2]
        return response[0], response[1]

    def safe_power_off(self):
        """Stop the robot's motion and sit if possible.  Once sitting, disable motor power."""
        response = self._robot_command(RobotCommandBuilder.safe_power_off_command())
        return response[0], response[1]

    def clear_behavior_fault(self, id):
        """Clear the behavior fault defined by id."""
        try:
            rid = self._robot_command_client.clear_behavior_fault(behavior_fault_id=id, lease=None)
            return True, "Success", rid
        except Exception as e:
            return False, str(e), None

    def power_on(self):
        """Enble the motor power if e-stop is enabled."""
        try:
            power.power_on(self._power_client)
            return True, "Success"
        except Exception as e:
            return False, str(e)

    def set_mobility_params(self, mobility_params):
        """Set Params for mobility and movement

        Args:
            mobility_params: spot.MobilityParams, params for spot mobility commands.
        """
        self._mobility_params = mobility_params

    def get_mobility_params(self):
        """Get mobility params
        """
        return self._mobility_params

    def velocity_cmd(self, v_x, v_y, v_rot, cmd_duration=0.125):
        """Send a velocity motion command to the robot.

        Args:
            v_x: Velocity in the X direction in meters
            v_y: Velocity in the Y direction in meters
            v_rot: Angular velocity around the Z axis in radians
            cmd_duration: (optional) Time-to-live for the command in seconds.  Default is 125ms (assuming 10Hz command rate).
        """
        end_time=time.time() + cmd_duration
        response = self._robot_command(RobotCommandBuilder.synchro_velocity_command(
                                      v_x=v_x, v_y=v_y, v_rot=v_rot, params=self._mobility_params),
                                      end_time_secs=end_time, timesync_endpoint=self._robot.time_sync.endpoint)
        self._last_velocity_command_time = end_time
        return response[0], response[1]

    def trajectory_cmd(self, goal_x, goal_y, goal_heading, cmd_duration, frame_name='odom', precise_position=False):
        """Send a trajectory motion command to the robot.

        Args:
            goal_x: Position X coordinate in meters
            goal_y: Position Y coordinate in meters
            goal_heading: Pose heading in radians
            cmd_duration: Time-to-live for the command in seconds.
            frame_name: frame_name to be used to calc the target position. 'odom' or 'vision'
            precise_position: if set to false, the status STATUS_NEAR_GOAL and STATUS_AT_GOAL will be equivalent. If
            true, the robot must complete its final positioning before it will be considered to have successfully
            reached the goal.
        """
        self._at_goal = False
        self._near_goal = False
        self._last_trajectory_command_precise = precise_position
        self._logger.info("got command duration of {}".format(cmd_duration))
        end_time=time.time() + cmd_duration
        if frame_name == 'vision':
            vision_tform_body = frame_helpers.get_vision_tform_body(
                    self._robot_state_client.get_robot_state().kinematic_state.transforms_snapshot)
            body_tform_goal = math_helpers.SE3Pose(x=goal_x, y=goal_y, z=0, rot=math_helpers.Quat.from_yaw(goal_heading))
            vision_tform_goal = vision_tform_body * body_tform_goal
            response = self._robot_command(
                            RobotCommandBuilder.synchro_se2_trajectory_point_command(
                                goal_x=vision_tform_goal.x,
                                goal_y=vision_tform_goal.y,
                                goal_heading=vision_tform_goal.rot.to_yaw(),
                                frame_name=frame_helpers.VISION_FRAME_NAME,
                                params=self._mobility_params),
                            end_time_secs=end_time
                            )
        elif frame_name == 'odom':
            odom_tform_body = frame_helpers.get_odom_tform_body(
                    self._robot_state_client.get_robot_state().kinematic_state.transforms_snapshot)
            body_tform_goal = math_helpers.SE3Pose(x=goal_x, y=goal_y, z=0, rot=math_helpers.Quat.from_yaw(goal_heading))
            odom_tform_goal = odom_tform_body * body_tform_goal
            response = self._robot_command(
                            RobotCommandBuilder.synchro_se2_trajectory_point_command(
                                goal_x=odom_tform_goal.x,
                                goal_y=odom_tform_goal.y,
                                goal_heading=odom_tform_goal.rot.to_yaw(),
                                frame_name=frame_helpers.ODOM_FRAME_NAME,
                                params=self._mobility_params),
                            end_time_secs=end_time
                            )
        else:
            raise ValueError('frame_name must be \'vision\' or \'odom\'')
        if response[0]:
            self._last_trajectory_command = response[2]
        return response[0], response[1]

    def list_graph(self):
        """List waypoint ids of garph_nav
        Args:
        """
        ids, eds = self._list_graph_waypoint_and_edge_ids()
        # skip waypoint_ for v2.2.1, skip waypiont for < v2.2
        return [v for k, v in sorted(ids.items(), key=lambda id : int(id[0].replace('waypoint_','')))]

    def clear_graph(self):
        try:
            self._clear_graph()
            return True, 'Success'
        except Exception as e:
            return False, 'Error: {}'.format(e)

    def start_recording(self):
        return self._start_recording()

    def stop_recording(self):
        return self._stop_recording()

    def download_graph(self, download_filepath):
        return self._download_full_graph(download_filepath)

    def upload_graph(self, upload_path):
        """List waypoint ids of garph_nav
        Args:
          upload_path : Path to the root directory of the map.
        """
        try:
            self._clear_graph()
            self._upload_graph_and_snapshots(upload_path)
            return True, 'Success'
        except Exception as e:
            return False, 'Error: {}'.format(e)

    def set_localization_fiducial(self):
        try:
            self._set_initial_localization_fiducial()
            return True, 'Success'
        except Exception as e:
            return False, 'Error: {}'.format(e)

    def set_localization_waypoint(self, waypoint_id):
        try:
            resp = self._set_initial_localization_waypoint(waypoint_id)
            return resp[0], resp[1]
        except Exception as e:
            return False, 'Error: {}'.format(e)

    def cancel_navigate_to(self):
        self._cancel_navigate_to()

    def navigate_to(self,
                    id_navigate_to):
        """ navigate with graph nav.

        Args:
           id_navigate_to : Waypont id string for where to goal
        """
        self._get_localization_state()
        resp = self._start_navigate_to(id_navigate_to)
        return resp

    ## copy from spot-sdk/python/examples/graph_nav_command_line/graph_nav_command_line.py
    def _get_localization_state(self, *args):
        """Get the current localization and state of the robot."""
        state = self._graph_nav_client.get_localization_state()
        self._logger.info('Got localization: \n%s' % str(state.localization))
        odom_tform_body = get_odom_tform_body(state.robot_kinematics.transforms_snapshot)
        self._logger.info('Got robot state in kinematic odometry frame: \n%s' % str(odom_tform_body))
        return state

    def _set_initial_localization_fiducial(self, *args):
        """Trigger localization when near a fiducial."""
        current_odom_tform_body = get_odom_tform_body(self._robot_state_client.get_robot_state().kinematic_state.transforms_snapshot).to_proto()
        # Create an empty instance for initial localization since we are asking it to localize
        # based on the nearest fiducial.
        localization = nav_pb2.Localization()
        self._graph_nav_client.set_localization(initial_guess_localization=localization,
                                                ko_tform_body=current_odom_tform_body)

    def _set_initial_localization_waypoint(self, waypoint_id):
        """Trigger localization to a waypoint."""
        destination_waypoint = graph_nav_util.find_unique_waypoint_id(waypoint_id, self._current_graph, self._current_annotation_name_to_wp_id, self._logger)
        if not destination_waypoint:
            return False, 'Failed to find the unique waypoint id.'

        robot_state = self._robot_state_client.get_robot_state()
        current_odom_tform_body = get_odom_tform_body(robot_state.kinematic_state.transforms_snapshot).to_proto()
        # Create an initial localization to the specified waypoint as the identity.
        localization = nav_pb2.Localization()
        localization.waypoint_id = destination_waypoint
        localization.waypoint_tform_body.rotation.w = 1.0
        self._graph_nav_client.set_localization(
            initial_guess_localization=localization,
            # It's hard to get the pose perfect, search +/-20 deg and +/-20cm (0.2m).
            max_distance = 0.2,
            max_yaw = 20.0 * math.pi / 180.0,
            fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NO_FIDUCIAL,
            ko_tform_body=current_odom_tform_body)

        return True, 'Success'

    def _list_graph_waypoint_and_edge_ids(self, *args):
        """List the waypoint ids and edge ids of the graph currently on the robot."""

        # Download current graph
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            self._logger.error("Empty graph.")
            return
        self._current_graph = graph

        localization_id = self._graph_nav_client.get_localization_state().localization.waypoint_id

        # Update and print waypoints and edges
        self._current_annotation_name_to_wp_id, self._current_edges = graph_nav_util.update_waypoints_and_edges(
            graph, localization_id, self._logger)
        return self._current_annotation_name_to_wp_id, self._current_edges


    def _upload_graph_and_snapshots(self, upload_filepath):
        """Upload the graph and snapshots to the robot."""
        self._logger.info("Loading the graph from disk into local storage...")
        with open(upload_filepath + "/graph", "rb") as graph_file:
            # Load the graph from disk.
            data = graph_file.read()
            self._current_graph = map_pb2.Graph()
            self._current_graph.ParseFromString(data)
            self._logger.info("Loaded graph has {} waypoints and {} edges".format(
                len(self._current_graph.waypoints), len(self._current_graph.edges)))
        for waypoint in self._current_graph.waypoints:
            # Load the waypoint snapshots from disk.
            with open(upload_filepath + "/waypoint_snapshots/{}".format(waypoint.snapshot_id),
                      "rb") as snapshot_file:
                waypoint_snapshot = map_pb2.WaypointSnapshot()
                waypoint_snapshot.ParseFromString(snapshot_file.read())
                self._current_waypoint_snapshots[waypoint_snapshot.id] = waypoint_snapshot
        for edge in self._current_graph.edges:
            # Load the edge snapshots from disk.
            with open(upload_filepath + "/edge_snapshots/{}".format(edge.snapshot_id),
                      "rb") as snapshot_file:
                edge_snapshot = map_pb2.EdgeSnapshot()
                edge_snapshot.ParseFromString(snapshot_file.read())
                self._current_edge_snapshots[edge_snapshot.id] = edge_snapshot
        # Upload the graph to the robot.
        self._logger.info("Uploading the graph and snapshots to the robot...")
        response = self._graph_nav_client.upload_graph(lease=self._lease.lease_proto,
                                                       graph=self._current_graph)
        # Upload the snapshots to the robot.
        for index, waypoint_snapshot_id in enumerate(response.unknown_waypoint_snapshot_ids):
            waypoint_snapshot = self._current_waypoint_snapshots[waypoint_snapshot_id]
            self._graph_nav_client.upload_waypoint_snapshot(waypoint_snapshot)
            self._logger.debug("Uploaded waypont snapshot {}/{} {}".format(index+1,len(self._current_waypoint_snapshots),waypoint_snapshot.id))
        for index, edge_snapshot_id in enumerate(response.unknown_edge_snapshot_ids):
            edge_snapshot = self._current_edge_snapshots[edge_snapshot_id]
            self._graph_nav_client.upload_edge_snapshot(edge_snapshot)
            self._logger.debug("Uploaded edge snapshot {}/{} {}".format(index+1,len(self._current_edge_snapshots),edge_snapshot.id))

        # The upload is complete! Check that the robot is localized to the graph,
        # and it if is not, prompt the user to localize the robot before attempting
        # any navigation commands.
        localization_state = self._graph_nav_client.get_localization_state()
        if not localization_state.localization.waypoint_id:
            # The robot is not localized to the newly uploaded graph.
            self._logger.warn(
                "Upload complete! The robot is currently not localized to the map; please localize")

    def _cancel_navigate_to(self):
        self._navigate_to_valid = False

    def _start_navigate_to(self, waypoint_id):
        """Navigate to a specific waypoint."""
        # Take the first argument as the destination waypoint.

        self._lease = self._lease_wallet.get_lease()
        destination_waypoint = graph_nav_util.find_unique_waypoint_id(
            waypoint_id,
            self._current_graph,
            self._current_annotation_name_to_wp_id,
            self._logger)
        if not destination_waypoint:
            # Failed to find the appropriate unique waypoint id for the navigation command.
            return
        else:
            self._logger.info('navigate to {}'.format(destination_waypoint))

        self._lease = self._lease_wallet.advance()
        sublease = self._lease.create_sublease()
        self._lease_keepalive.shutdown()

        self._navigate_to_valid = True
        nav_to_cmd_id = None
        while self._navigate_to_valid:
            # Sleep for half a second to allow for command execution.
            time.sleep(0.5)
            try:
                nav_to_cmd_id = self._graph_nav_client.navigate_to(destination_waypoint, 1.0,
                                                                   leases=[sublease.lease_proto],
                                                                   command_id=nav_to_cmd_id)
                self._last_navigate_to_command = nav_to_cmd_id
            except ResponseError as e:
                self._logger.error("Error while navigating {}".format(e))
                break
            if self._check_success(nav_to_cmd_id):
                self._last_navigate_to_command = None
                break

        self._lease = self._lease_wallet.advance()
        self._lease_keepalive = LeaseKeepAlive(self._lease_client)

        if self._navigate_to_valid == False:
            return False, 'Navigate to is canceled.', 'preempted'
        status = self._graph_nav_client.navigation_feedback(nav_to_cmd_id)
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            return True, "Successfully completed the navigation commands!", None
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            return False, "Robot got lost when navigating the route, the robot will now sit down.", 'fail'
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            return False, "Robot got stuck when navigating the route, the robot will now sit down.", 'fail'
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            return False, "Robot is impaired.", 'fail'
        else:
            return False, "Navigation command is not complete yet.", 'fail'

    def _navigate_route(self, *args):
        """Navigate through a specific route of waypoints."""
        if len(args) < 1:
            # If no waypoint ids are given as input, then return without requesting navigation.
            self._logger.error("No waypoints provided for navigate route.")
            return
        waypoint_ids = args[0]
        for i in range(len(waypoint_ids)):
            waypoint_ids[i] = graph_nav_util.find_unique_waypoint_id(
                waypoint_ids[i], self._current_graph, self._current_annotation_name_to_wp_id, self._logger)
            if not waypoint_ids[i]:
                # Failed to find the unique waypoint id.
                return

        edge_ids_list = []
        all_edges_found = True
        # Attempt to find edges in the current graph that match the ordered waypoint pairs.
        # These are necessary to create a valid route.
        for i in range(len(waypoint_ids) - 1):
            start_wp = waypoint_ids[i]
            end_wp = waypoint_ids[i + 1]
            edge_id = self._match_edge(self._current_edges, start_wp, end_wp)
            if edge_id is not None:
                edge_ids_list.append(edge_id)
            else:
                all_edges_found = False
                self._logger.error("Failed to find an edge between waypoints: ", start_wp, " and ", end_wp)
                self._logger.error(
                    "List the graph's waypoints and edges to ensure pairs of waypoints has an edge."
                )
                break

        self._lease = self._lease_wallet.get_lease()
        if all_edges_found:
            if not self.toggle_power(should_power_on=True):
                self._logger.error("Failed to power on the robot, and cannot complete navigate route request.")
                return

            # Stop the lease keepalive and create a new sublease for graph nav.
            self._lease = self._lease_wallet.advance()
            sublease = self._lease.create_sublease()
            self._lease_keepalive.shutdown()

            # Navigate a specific route.
            route = self._graph_nav_client.build_route(waypoint_ids, edge_ids_list)
            is_finished = False
            while not is_finished:
                # Issue the route command about twice a second such that it is easy to terminate the
                # navigation command (with estop or killing the program).
                nav_route_command_id = self._graph_nav_client.navigate_route(
                    route, cmd_duration=1.0, leases=[sublease.lease_proto])
                time.sleep(.5)  # Sleep for half a second to allow for command execution.
                # Poll the robot for feedback to determine if the route is complete. Then sit
                # the robot down once it is finished.
                is_finished = self._check_success(nav_route_command_id)

            self._lease = self._lease_wallet.advance()
            self._lease_keepalive = LeaseKeepAlive(self._lease_client)

    def _clear_graph(self, *args):
        """Clear the state of the map on the robot, removing all waypoints and edges."""
        return self._graph_nav_client.clear_graph(lease=self._lease.lease_proto)

    def _check_success(self, command_id=-1):
        """Use a navigation command id to get feedback from the robot and sit when command succeeds."""
        if command_id == -1:
            # No command, so we have not status to check.
            return False
        status = self._graph_nav_client.navigation_feedback(command_id)
        if status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL:
            # Successfully completed the navigation commands!
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST:
            self._logger.error("Robot got lost when navigating the route, the robot will now sit down.")
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK:
            self._logger.error("Robot got stuck when navigating the route, the robot will now sit down.")
            return True
        elif status.status == graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED:
            self._logger.error("Robot is impaired.")
            return True
        else:
            # Navigation command is not complete yet.
            return False

    def _match_edge(self, current_edges, waypoint1, waypoint2):
        """Find an edge in the graph that is between two waypoint ids."""
        # Return the correct edge id as soon as it's found.
        for edge_to_id in current_edges:
            for edge_from_id in current_edges[edge_to_id]:
                if (waypoint1 == edge_to_id) and (waypoint2 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint2, to_waypoint=waypoint1)
                elif (waypoint2 == edge_to_id) and (waypoint1 == edge_from_id):
                    # This edge matches the pair of waypoints! Add it the edge list and continue.
                    return map_pb2.Edge.Id(from_waypoint=waypoint1, to_waypoint=waypoint2)
        return None

    def _start_recording(self):
        """Start recording a map."""
        should_start_recording = self._should_we_start_recording()
        if not should_start_recording:
            message =  "The system is not in the proper state to start recording." + \
                "Try using the graph_nav_command_line to either clear the map or " + \
                "attempt to localize to the map."
            self._logger.error(message)
            return False, message
        try:
            status = self._recording_client.start_recording()
            self._logger.info("Successfully started recording a map.")
            return True, "Successfully started recording a map."
        except Exception as err:
            self._logger.error("Start recording failed: {}".format(err))
            return False, "Start recording failed: {}".format(err)

    def _stop_recording(self):
        """Stop or pause recording a map."""
        try:
            status = self._recording_client.stop_recording()
            return True, "Successfully stopped recording a map."
        except Exception as err:
            return False, "Stop recording failed: {}".format(err)

    def _get_recording_status(self):
        """Get the recording service's status."""
        status = self._recording_client.get_record_status()
        if status.is_recording:
            return True, "The recording service is on."
        else:
            return False, "The recording service is off."

    def _create_default_waypoint(self):
        """Create a default waypoint at the robot's current location."""
        resp = self._recording_client.create_waypoint(waypoint_name="default")
        if resp.status == recording_pb2.CreateWaypointResponse.STATUS_OK:
            return True, "Successfully created a waypoint."
        else:
            return False, "Could not create a waypoint."

    def _should_we_start_recording(self):
        graph = self._graph_nav_client.download_graph()
        if graph is not None:
            if len(graph.waypoints) > 0:
                localization_state = self._graph_nav_client.get_localization_state()
                if not localization_state.localization.waypoint_id:
                    return False
        return True

    def _write_bytes(self, filepath, filename, data):
        """Write data to a file."""
        os.makedirs(filepath, exist_ok=True)
        with open(filepath + filename, 'wb+') as f:
            f.write(data)
            f.close()

    def _download_full_graph(self, download_path):
        """Download the graph and snapshots from the robot."""
        graph = self._graph_nav_client.download_graph()
        if graph is None:
            return False, "Failed to download the graph."
        graph_bytes = graph.SerializeToString()
        self._write_bytes(download_path, '/graph', graph_bytes)
        # Download the waypoint and edge snapshots.
        for waypoint in graph.waypoints:
            try:
                waypoint_snapshot = self._graph_nav_client.download_waypoint_snapshot(
                    waypoint.snapshot_id)
            except Exception:
                print("Failed to download waypoint snapshot: " +
                      waypoint.snapshot_id)
                continue
            self._write_bytes(
                download_path + '/waypoint_snapshots',
                '/' + waypoint.snapshot_id,
                waypoint_snapshot.SerializeToString())
        for edge in graph.edges:
            try:
                edge_snapshot = self._graph_nav_client.download_edge_snapshot(
                    edge.snapshot_id)
            except Exception:
                # Failure in downloading edge snapshot. Continue to next snapshot.
                print("Failed to download edge snapshot: " + edge.snapshot_id)
                continue
            self._write_bytes(
                download_path + '/edge_snapshots',
                '/' + edge.snapshot_id,
                edge_snapshot.SerializeToString())
        return True, 'Success'


    def dock(self, dock_id):
        "Dock the robot to dockid"
        try:
            # おまじない.これをするとLease Errorがでなくなる
            self.sit()
            self.stand()
            # Dock the robot
            self._blocking_dock_robot(dock_id)
            # Sit after Dock
            self.sit()
            return True, "Success"
        except Exception as e:
            return False, str(e)


    def undock(self, timeout=20):
        "Power motor on and undock the robot from the station"
        try:
            # Undock the robot
            self._blocking_undock(timeout)
            # Stand after Undock
            self.stand()
            return True, "Success"
        except Exception as e:
            return False, str(e)

    def stow(self):
        """Stow the arm."""
        if self._robot.has_arm():
            response = self._robot_command(RobotCommandBuilder.arm_stow_command())
            return response[0], response[1]
        else:
            return False, "Robot don't have the arm."

    def unstow(self):
        """Unstow the arm."""
        if self._robot.has_arm():
            response = self._robot_command(RobotCommandBuilder.arm_ready_command())
            return response[0], response[1]
        else:
            return False, "Robot don't have the arm."

    def gipper_open(self):
        """Gripper is opened."""
        if self._robot.has_arm():
            response = self._robot_command(RobotCommandBuilder.claw_gripper_open_command())
            return response[0], response[1]
        else:
            return False, "Robot don't have the arm."

    def gipper_close(self):
        """Gripper is closeed."""
        if self._robot.has_arm():
            response = self._robot_command(RobotCommandBuilder.claw_gripper_close_command())
            return response[0], response[1]
        else:
            return False, "Robot don't have the arm."

    def pitch_up(self):
        """Pitch robot up to look for door handle."""
        self._logger.info("Pitching robot up...")
        footprint_R_body = geometry.EulerZXY(0.0, 0.0, -1 * math.pi / 6.0)
        response = self._robot_command(RobotCommandBuilder.synchro_stand_command(footprint_R_body=footprint_R_body))
        timeout_sec = 10.0
        end_time = time.time() + timeout_sec
        while time.time() < end_time:
            command_feedback = self._robot_command_client.robot_command_feedback(response[2])
            synchronized_feedback = command_feedback.feedback.synchronized_feedback
            status = synchronized_feedback.mobility_command_feedback.stand_feedback.status
            if (status == basic_command_pb2.StandCommand.Feedback.STATUS_IS_STANDING):
                self._logger.info("Robot pitched.")
                return
            time.sleep(1.0)
        raise Exception("Failed to pitch robot.")

    def get_images_as_cv2(self, sources):
        """Request image sources from robot. Decode and store as OpenCV image as well as proto.
        Args:
            sources: (list) String names of image sources.

        Returns:
            dict: Dictionary from image source name to (image proto, CV2 image) pairs.
        """
        image_responses = self._image_client.get_image_from_sources(sources)
        image_dict = dict()
        for response in image_responses:
            # Convert image proto to CV2 image, for display later.
            image = np.frombuffer(response.shot.image.data, dtype=np.uint8)
            image = cv2.imdecode(image, -1)
            image_dict[response.source.name] = (response, image)
        return image_dict

    def walk_to_object_in_image(self, request_manager, debug):
        """Command the robot to walk toward user selected point. The WalkToObjectInImage feedback
        reports a raycast result, converting the 2D touchpoint to a 3D location in space.

        Args:
            request_manager: (RequestManager) Object for bookkeeping user touch points.
            debug (bool): Show intermediate debug image..

        Returns:
            ManipulationApiResponse: Feedback from WalkToObjectInImage request.
        """
        manipulation_api_request = request_manager.get_walk_to_object_in_image_request(debug)

        # Send a manipulation API request. Using the points selected by the user, the robot will
        # walk toward the door handle.
        self._logger.info("Walking toward door...")
        response = self._manipulation_client.manipulation_api_command(manipulation_api_request)

        # Check feedback to verify the robot walks to the handle. The service will also return a
        # FrameTreeSnapshot that contain a walkto_raycast_intersection point.
        command_id = response.manipulation_cmd_id
        feedback_request = ManipulationApiFeedbackRequest(manipulation_cmd_id=command_id)
        timeout_sec = 15.0
        end_time = time.time() + timeout_sec
        while time.time() < end_time:
            response = self._manipulation_client.manipulation_api_feedback_command(feedback_request)
            assert response.manipulation_cmd_id == command_id, "Got feedback for wrong command."
            if (response.current_state == manipulation_api_pb2.MANIP_STATE_DONE):
                return response
        raise Exception("Manipulation command timed out. Try repositioning the robot.")

    def open_door(self, request_manager, snapshot):
        """Command the robot to automatically open a door via the door service API.

        Args:
            request_manager: (RequestManager) Object for bookkeeping user touch points.
            snapshot: (TransformSnapshot) Snapshot from the WalkToObjectInImage command which contains
                the 3D location reported from a raycast based on the user hinge touch point.
        """
        self._logger.info("Opening door...")

        # Using the raycast intersection point and the
        vision_tform_raycast = frame_helpers.get_a_tform_b(snapshot, frame_helpers.VISION_FRAME_NAME,
                                                        frame_helpers.RAYCAST_FRAME_NAME)
        vision_tform_sensor = request_manager.vision_tform_sensor
        raycast_point_wrt_vision = vision_tform_raycast.get_translation()
        ray_from_camera_to_object = raycast_point_wrt_vision - vision_tform_sensor.get_translation()
        ray_from_camera_to_object_norm = np.sqrt(np.sum(ray_from_camera_to_object**2))
        ray_from_camera_normalized = ray_from_camera_to_object / ray_from_camera_to_object_norm

        auto_cmd = door_pb2.DoorCommand.AutoGraspCommand()
        auto_cmd.frame_name = frame_helpers.VISION_FRAME_NAME
        search_dist_meters = 0.25
        search_ray = search_dist_meters * ray_from_camera_normalized
        search_ray_start_in_frame = raycast_point_wrt_vision - search_ray
        auto_cmd.search_ray_start_in_frame.CopyFrom(geometry_pb2.Vec3(x=search_ray_start_in_frame[0], 
                                                    y=search_ray_start_in_frame[1], z=search_ray_start_in_frame[2]))

        search_ray_end_in_frame = raycast_point_wrt_vision + search_ray
        auto_cmd.search_ray_end_in_frame.CopyFrom(geometry_pb2.Vec3(x=search_ray_end_in_frame[0], 
                                                    y=search_ray_end_in_frame[1], z=search_ray_end_in_frame[2]))

        auto_cmd.hinge_side = request_manager.hinge_side
        auto_cmd.swing_direction = door_pb2.DoorCommand.SWING_DIRECTION_UNKNOWN

        door_command = door_pb2.DoorCommand.Request(auto_grasp_command=auto_cmd)
        request = door_pb2.OpenDoorCommandRequest(door_command=door_command)

        # Command the robot to open the door.
        response = self._door_client.open_door(request)

        feedback_request = door_pb2.OpenDoorFeedbackRequest()
        feedback_request.door_command_id = response.door_command_id

        timeout_sec = 60.0
        end_time = time.time() + timeout_sec
        while time.time() < end_time:
            feedback_response = self._door_client.open_door_feedback(feedback_request)
            if (feedback_response.status !=
                    basic_command_pb2.RobotCommandFeedbackStatus.STATUS_PROCESSING):
                raise Exception("Door command reported status ")
            if (feedback_response.feedback.status == door_pb2.DoorCommand.Feedback.STATUS_COMPLETED):
                self._logger.info("Opened door.")
                return
            time.sleep(0.5)
        raise Exception("Door command timed out. Try repositioning the robot.")
 
    def execute_open_door(self, debug=False):
        """High level behavior sequence for commanding the robot to open a door."""
        # Pitch the robot up.
        try:
            self.pitch_up()
        except Exception as e:
            return False, str(e)

        # Capture images from the two from cameras.
        image_dict = self.get_images_as_cv2(['frontleft_fisheye_image', 'frontright_fisheye_image'])

        # Get handle and hinge locations from user input.
        window_name = "Open Door Example"
        request_manager = RequestManager(image_dict, window_name)
        request_manager.get_user_input_handle_and_hinge()
        assert request_manager.user_input_set(), "Failed to get user input for handle and hinge."

        try:
            # Tell the robot to walk toward the door.
            manipulation_feedback = self.walk_to_object_in_image(request_manager, debug)
            time.sleep(3.0)
        except Exception as e:
            return False, str(e)

        # The ManipulationApiResponse for the WalkToObjectInImage command returns a transform snapshot
        # that contains where user clicked door handle point intersects the world. We use this
        # intersection point to execute the door command.
        snapshot = manipulation_feedback.transforms_snapshot_manipulation_data

        try:
            self.open_door(request_manager, snapshot)
            time.sleep(3.0)
        except Exception as e:
            return False, str(e)
    
    def open_door_main(self):
        """Opening the door"""
        try:
            with LeaseKeepAlive(self._lease_client):
                self.execute_open_door()
                return True, "Success"
        except Exception as e:
            return False, str(e)

    def set_screen(self, screen):
        """Set the cam screen"""
        if self._has_cam:
            if screen in cam_screen_sources:
                self._compositor_client.set_screen(screen)
                return True, "Success"
            else:
                return False, "Invalid screen. " + "Screen list is " + str(cam_screen_sources)
        else:
            return False, "Robot don't have the spot cam."

    def control_cam_ptz(self, pan, tilt, zoom):
        """Control the cam ptz"""
        if self._has_cam:
            if pan < 0 or pan > 360 or tilt < -90 or tilt > 90 or zoom < 1 or zoom > 30:
                return False, "Invalid command value"
            else:
                ptz_desc = ptz_pb2.PtzDescription(name='mech')
                self._ptz_client.set_ptz_position(ptz_desc, pan, tilt, zoom)
                return True, "Success"
        else:
            return False, "Robot don't have the spot cam."

    def arm_cartesian(self, goal):
        """Move the arm to the commanded cartesian position"""
        robot_command.blocking_stand(self._robot_command_client)

        pose = geometry_pb2.Vec3(x = goal.target.position.x,
                                    y = goal.target.position.y,
                                    z = goal.target.position.z)
        quat = geometry_pb2.Quaternion(x = goal.target.orientation.x,
                                 y = goal.target.orientation.y,
                                 z = goal.target.orientation.z,
                                 w = goal.target.orientation.w)
        flat_body_T_hand = geometry_pb2.SE3Pose(position=pose, rotation=quat)

        robot_state = self._robot_state_client.get_robot_state()
        odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                         ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)
        odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_obj(flat_body_T_hand) 

        try:
            response = self._robot_command(RobotCommandBuilder.arm_pose_command(
                odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, 
                odom_T_hand.rot.w, odom_T_hand.rot.x, odom_T_hand.rot.y, odom_T_hand.rot.z, 
                ODOM_FRAME_NAME, 3))
            return response[0], response[1], response[2]
        except Exception as e:
            return False, str(e)

    def _blocking_dock_robot(self, dock_id, num_retries=4, timeout=30):

        attempt_number = 0
        docking_success = False

        # Try to dock the robot
        while attempt_number < num_retries and not docking_success:
            attempt_number += 1
            converter = self._robot.time_sync.get_robot_time_converter()
            start_time = converter.robot_seconds_from_local_seconds(now_sec())
            cmd_end_time = start_time + timeout
            cmd_timeout = cmd_end_time + 10  # client side buffer

            prep_pose = (docking_pb2.PREP_POSE_USE_POSE if
                         (attempt_number % 2) else docking_pb2.PREP_POSE_SKIP_POSE)

            cmd_id = self._docking_client.docking_command(dock_id, self._robot.time_sync.endpoint.clock_identifier,
                                                    seconds_to_timestamp(cmd_end_time), prep_pose)

            while converter.robot_seconds_from_local_seconds(now_sec()) < cmd_timeout:
                feedback = self._docking_client.docking_command_feedback_full(cmd_id)
                maybe_raise(common_lease_errors(feedback))
                status = feedback.status
                if status == docking_pb2.DockingCommandFeedbackResponse.STATUS_IN_PROGRESS:
                    # keep waiting/trying
                    time.sleep(1)
                elif status == docking_pb2.DockingCommandFeedbackResponse.STATUS_DOCKED:
                    docking_success = True
                    break
                elif (status in [
                        docking_pb2.DockingCommandFeedbackResponse.STATUS_MISALIGNED,
                        docking_pb2.DockingCommandFeedbackResponse.STATUS_ERROR_COMMAND_TIMED_OUT,
                ]):
                    # Retry
                    break
                else:
                    raise CommandFailedError(
                        "Docking Failed, status: '%s'" %
                        docking_pb2.DockingCommandFeedbackResponse.Status.Name(status))

        if docking_success:
            return attempt_number - 1

        # Try and put the robot in a safe position
        try:
            self._blocking_go_to_prep_pose(robot, dock_id)
        except CommandFailedError:
            pass

        # Raise error on original failure to dock
        raise CommandFailedError("Docking Failed, too many attempts")


    def _blocking_go_to_prep_pose(self, dock_id, timeout=20):
        converter = self._robot.time_sync.get_robot_time_converter()
        start_time = converter.robot_seconds_from_local_seconds(now_sec())
        cmd_end_time = start_time + timeout
        cmd_timeout = cmd_end_time + 10  # client side buffer

        cmd_id = self._docking_client.docking_command(dock_id, self._robot.time_sync.endpoint.clock_identifier,
                                                seconds_to_timestamp(cmd_end_time),
                                                docking_pb2.PREP_POSE_ONLY_POSE)

        while converter.robot_seconds_from_local_seconds(now_sec()) < cmd_timeout:
            feedback = self._docking_client.docking_command_feedback_full(cmd_id)
            maybe_raise(common_lease_errors(feedback))
            status = feedback.status
            if status == docking_pb2.DockingCommandFeedbackResponse.STATUS_IN_PROGRESS:
                # keep waiting/trying
                time.sleep(1)
            elif status == docking_pb2.DockingCommandFeedbackResponse.STATUS_AT_PREP_POSE:
                return
            else:
                raise CommandFailedError("Failed to go to the prep pose, status: '%s'" %
                                         docking_pb2.DockingCommandFeedbackResponse.Status.Name(status))

        raise CommandFailedError("Error going to the prep pose, timeout exceeded.")


    def _blocking_undock(self, timeout=20):
        converter = self._robot.time_sync.get_robot_time_converter()
        start_time = converter.robot_seconds_from_local_seconds(now_sec())
        cmd_end_time = start_time + timeout
        cmd_timeout = cmd_end_time + 10  # client side buffer

        cmd_id = self._docking_client.docking_command(0, self._robot.time_sync.endpoint.clock_identifier,
                                                seconds_to_timestamp(cmd_end_time),
                                                docking_pb2.PREP_POSE_UNDOCK)

        while converter.robot_seconds_from_local_seconds(now_sec()) < cmd_timeout:
            feedback = self._docking_client.docking_command_feedback_full(cmd_id)
            maybe_raise(common_lease_errors(feedback))
            status = feedback.status
            if status == docking_pb2.DockingCommandFeedbackResponse.STATUS_IN_PROGRESS:
                # keep waiting/trying
                time.sleep(1)
            elif status == docking_pb2.DockingCommandFeedbackResponse.STATUS_AT_PREP_POSE:
                return
            else:
                raise CommandFailedError("Failed to undock the robot, status: '%s'" %
                                         docking_pb2.DockingCommandFeedbackResponse.Status.Name(status))

        raise CommandFailedError("Error undocking the robot, timeout exceeded.")
