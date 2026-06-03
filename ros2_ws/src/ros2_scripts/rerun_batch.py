from pathlib import Path
import time
import rerun as rr
import argparse

import trio
import sys
import termios
import tty

import numpy as np

from xacrodoc import XacroDoc
from pytransform3d.urdf import UrdfTransformManager
from scipy.spatial.transform import Rotation

import cv2

SCRIPT_DESCRIPTION=\
"""This script allows the visualization of large mcap files in rerun by reading them sequentially
"""

def get_time_diff(decoded_msg, curr_time: float | None) -> tuple[float, float]:
    if curr_time is None:
        curr_time = decoded_msg.header.stamp.sec + decoded_msg.header.stamp.nanosec * 1e-9
        return (0, curr_time)
    else:
        last_time: float = curr_time
        curr_time: float = decoded_msg.header.stamp.sec + decoded_msg.header.stamp.nanosec * 1e-9
        elapsed: float = curr_time - last_time
        return (1/elapsed, curr_time)

curr_rgb_time = None
curr_depth_time = None
rgb_image_count = 0
depth_image_count = 0
def log_image(decoded_msg, channel):
    global rgb_image_count
    global depth_image_count

    height = decoded_msg.height
    width = decoded_msg.width
    encoding = decoded_msg.encoding

    if encoding == "16UC1":
        depth_image_count += 1
        raw_data = np.frombuffer(decoded_msg.data, dtype=np.uint16)
        img_tensor = raw_data.reshape((height, width))

        # Downsampling
        width = int(width / 4)
        height = int(height / 4)
        dim = (width, height)

        img_tensor = cv2.resize(img_tensor, dim, interpolation=cv2.INTER_NEAREST)
        
        rr.log(channel.topic + '/image', rr.DepthImage(img_tensor, meter=1000))

        # Log FPS
        global curr_depth_time
        (fps, time) = get_time_diff(decoded_msg, curr_depth_time)
        curr_depth_time = time

        rr.log('/stats/fps/depth', rr.Scalars(scalars=[fps]))
    else:
        raw_data = np.frombuffer(decoded_msg.data, dtype=np.uint8)
        if encoding in ("rgb8", "bgr8"):
            global curr_rgb_time
            rgb_image_count += 1
            img_tensor = raw_data.reshape((height, width, 3))

            if encoding == "bgr8":
                img_tensor = img_tensor[:, :, ::-1]

            rr.log(channel.topic + '/image', rr.Image(img_tensor))
            
            # Log FPS
            (fps, time) = get_time_diff(decoded_msg, curr_rgb_time)
            curr_rgb_time = time

            rr.log('/stats/fps/color', rr.Scalars(scalars=[fps]))
        elif encoding in ("mono8", "8UC1"):
            img_tensor = raw_data.reshape((height, width))
            rr.log(channel.topic + '/image', rr.Image(img_tensor))
        else:
            return


    rr.log(
        "stats/image_loss",
        rr.Scalars(scalars=[abs(rgb_image_count - depth_image_count)])
    )

def log_gnss(decoded_msg, channel):
    latlon = [decoded_msg.latitude,decoded_msg.longitude]

    # Color = i32 RGBA. Blue (R=0, G=0, B=255, A=255)
    # Opacity can be skipped in array form
    color_scheme = {
        -1: [100,100,100], # gray
        0: [255,0,0], # red
        1: [0,0,255], # blue
        2: [0,255,0], # green
    }

    status = decoded_msg.status.status
    rr.log(
        "/reach" + channel.topic,
        rr.GeoPoints(
            lat_lon=latlon,
            radii=rr.Radius.ui_points(5.0),
            colors=[color_scheme[status]],
        )
    )

def log_imu(decoded_msg, channel, options):
    # rr.log(
    #     channel.topic + "/angular_velocity",
    #     rr.Arrows3D(
    #         vectors=[
    #             decoded_msg.angular_velocity.x,
    #             decoded_msg.angular_velocity.y,
    #             decoded_msg.angular_velocity.z,
    #         ],
    #         labels="Angular velocity"
    #     )
    # )

    match channel.topic:
        case "/reach_1/imu":
            t_baselink_imu = options['t/reach_imu1']
            q_baselink_imu = options['q/reach_imu1']
        case "/reach_2/imu":
            t_baselink_imu = options['t/reach_imu2']
            q_baselink_imu = options['q/reach_imu2']
        case "/reach_3/imu":
            t_baselink_imu = options['t/reach_imu3']
            q_baselink_imu = options['q/reach_imu3']
        case _:
            return

    rr.log(
        channel.topic + "/linear_acceleration",
        rr.InstancePoses3D(
            translations=[t_baselink_imu],
            quaternions=[q_baselink_imu],
        )
    )
    
    x = decoded_msg.linear_acceleration.x
    y = decoded_msg.linear_acceleration.y
    z = decoded_msg.linear_acceleration.z
    vector = np.array([x,y,z])
    rr.log(
        channel.topic + "/linear_acceleration",
        rr.Arrows3D(
            vectors=vector,
            origins=[0,0,0],
            labels="Linear acceleration"
        )
    )
    
    # rotate vector
    quat = Rotation.from_quat(q_baselink_imu)
    v_rot = quat.apply(vector)
    rr.log(
        channel.topic + "/stats/x",
        rr.Scalars(
            scalars=[v_rot[0]]
        )
    )
    rr.log(
        channel.topic + "/stats/y",
        rr.Scalars(
            scalars=[v_rot[1]]
        )
    )
    rr.log(
        channel.topic + "/stats/z",
        rr.Scalars(
            scalars=[v_rot[2]]
        )
    )

def log_odometry(decoded_msg, channel):
    pos = decoded_msg.pose.pose.position
    ori = decoded_msg.pose.pose.orientation

    rr.log(
        channel.topic + "/path",
        rr.Points3D(
            [[pos.x, pos.y, pos.z]],
            radii=rr.Radius.ui_points(5.0),
            colors=[[0, 255, 0]]
        )
    )
    rr.log(
        channel.topic + "/linear",
        rr.InstancePoses3D(
            translations=[[pos.x, pos.y, pos.z]],
            quaternions=[[ori.x, ori.y, ori.z, ori.w]]
        )
    )
    rr.log(
        channel.topic + "/linear",
        rr.Arrows3D(
            vectors=[
                decoded_msg.twist.twist.linear.x,
                decoded_msg.twist.twist.linear.y,
                decoded_msg.twist.twist.linear.z,
            ],
            origins=[0.0,0.0,0.0],
            labels="Linear acceleration"
        )
    )

def set_time(options, decoded_msg, msg):
    time = msg.log_time
    if options['header_timestamp'] and hasattr(msg, 'header'):
        time = to_ns(decoded_msg.header.stamp)

    rr.set_time("time", timestamp=np.datetime64(time, "ns"))

def to_ns(stamp):
    return stamp.sec * int(1e9) + stamp.nanosec

send_channel, receive_channel = trio.open_memory_channel(10)
should_exit = False

async def handle_input():
    print("Handling input. Press q to quit")

    global receive_channel
    async with receive_channel:
        async for key in receive_channel:
            print(f"\r\nLatest key: {key}", flush=True)
            if key == 'q' or key == 'Q':
                global should_exit
                should_exit = True
                break

    print("\r\nFinish handle_input")

# catches the keys pressed
def run_listener(trio_token):
    global send_channel

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setraw(fd)
        print("Raw mode active. Press any key (q to exit)...", end="", flush=True)

        while True:
            key = sys.stdin.read(1)

            try:
                trio.from_thread.run_sync(
                    send_channel.send_nowait,
                    key,
                    trio_token=trio_token
                )
            except trio.WouldBlock:
                print("Would block: ignoring")
                pass # ignore the input even if we couldn't handle it
            except trio.RunFinishedError:
                # Trio loop is done.
                return False

            try:
                if key == "q":
                    break
            except:
                pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    print("\r\nFinished listening")

async def stream_mcap(mcap_path: Path, options):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    rr.init("batch_example")
    rr.spawn(memory_limit=options['memory_limit'])
    # rr.send_blueprint(blueprint=options['initial_blueprint'])
    
    print(f"Opening {mcap_path} for sequential streaming");

    message_count = 0
    start_time = time.time()
    play_all = options['play_all']

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        options['initial_time'] = reader.get_summary().statistics.message_start_time
        options['final_time'] = reader.get_summary().statistics.message_end_time
        options['time_diff'] = options['final_time'] - options['initial_time']

        doc = XacroDoc.from_file(options['urdf'])
        urdf_str = doc.to_urdf_string()
        utm = UrdfTransformManager()
        utm.load_urdf(urdf_str)

        for imu in ["reach_imu1", "reach_imu2", "reach_imu3"]:
            T_baselink_imu = utm.get_transform(imu, "base_link")
            t_baselink_imu = T_baselink_imu[0:3, 3]
            q_baselink_imu = Rotation.from_matrix(T_baselink_imu[0:3, 0:3]).as_quat()
            options["t/" + imu] = t_baselink_imu
            options["q/" + imu] = q_baselink_imu

        for imu in ["/reach_1/imu/stats", "/reach_2/imu/stats", "/reach_3/imu/stats"]:
            for (val, color) in [("/x", [200,50,0]), ("/y",[0,200,75]), ("/z",[0,75,220])]:
                rr.log(
                    imu + val,
                    rr.SeriesLines(colors=color),
                    static=True
                )

        rr.log(
            "/realsense/depth/image_rect_raw",
            rr.Pinhole(
                resolution=[1280 / 4, 720 / 4],
                focal_length=[645.4064 / 4, 648.5756 / 4],
                principal_point=[648.7339 / 4, 349.0376 / 4]
            ),
            static=True
        )

        rr.log(
            "/realsense/color/image_raw",
            rr.Pinhole(
                resolution=[1280, 720],
                focal_length=[890.4202, 895.5269],
                principal_point=[633.5761, 375.3947]
            ),
            static=True
        )

        for schema, channel, msg, decoded_msg in reader.iter_decoded_messages():
            await trio.sleep(0)
            if should_exit:
                break

            if schema is None: continue

            set_time(options, decoded_msg, msg)
            match schema.name:
                case "sensor_msgs/msg/Image":
                    log_image(decoded_msg, channel)
                case "sensor_msgs/msg/NavSatFix":
                    log_gnss(decoded_msg, channel)
                case "sensor_msgs/msg/Imu":
                    log_imu(decoded_msg, channel, options)
                case "nav_msgs/msg/Odometry":
                    log_odometry(decoded_msg, channel)
                case _:
                    continue

            message_count += 1
            if message_count % 10000 == 0:
                elapsed = time.time() - start_time
                print(f"Streamed {message_count} messages... ({elapsed:.2f}s elapsed)")
                if not play_all and message_count % 20000 == 0:
                    res = input("Stream paused. Do you want to continue? (q/Q to quit) ")
                    if res.strip().lower() == "q":
                        break

async def main():
    parser = argparse.ArgumentParser(description=SCRIPT_DESCRIPTION)
    parser.add_argument(
        '-b', '--bag_path', type=Path, required=True,
        help='Path to rosbag fle to read and extract info from.'
    )
    parser.add_argument(
        '-m', '--memory_limit', type=str, required=False, default="50%",
        help='Memory limit before rerun garbage collects the old messages'
    )
    parser.add_argument(
        '-a', '--play_all', action='store_true',
        help='Play the whole mcap without pause'
    )
    parser.add_argument(
        '--header_timestamp', action='store_true',
        help='Use the message timestamp information instead of the log time in Ros'
    )
    # parser.add_argument(
    #     '-V', '--initial_blueprint', type=Path, required=False, default='./batch_blueprint.rbl',
    #     help='Initial view to start the recording'
    # )
    parser.add_argument(
        '--urdf', type=Path, required=False, default='../../../data/config/rosario_v2.urdf.xacro',
        help='URDF file to use for the transforms'
    )

    args = parser.parse_args()

    options = {}

    options['memory_limit'] = args.memory_limit
    options['play_all'] = args.play_all
    options['header_timestamp'] = args.header_timestamp
    # options['initial_blueprint'] = args.initial_blueprint
    options['urdf'] = args.urdf

    trio_token = trio.lowlevel.current_trio_token()

    print("Starting stream")
    async with trio.open_nursery() as nursery:
        nursery.start_soon(stream_mcap, args.bag_path, options)
        nursery.start_soon(handle_input)

        await trio.to_thread.run_sync(
            run_listener, trio_token
        )

    print("Exiting main")

if __name__ == '__main__':
    trio.run(main)
