from pathlib import Path
import time
import rerun as rr
import argparse

import trio
import sys
import curses
import tty

import numpy as np

from xacrodoc import XacroDoc
from pytransform3d.urdf import UrdfTransformManager
from scipy.spatial.transform import Rotation

import cv2

SCRIPT_DESCRIPTION=\
"""This script allows the visualization of large mcap files in rerun by reading them sequentially
"""

CONTROLS_DESCRIPTION=\
"""Controls: 'q' to exit, 'p' to pause, 'c' to toggle blueprints.
"""

image_stats = {
    "curr_rgb_time": 0.0,
    "curr_depth_time": 0.0,
    "rgb_image_count": 0,
    "depth_image_count": 0,
}

# send_channel, receive_channel = trio.open_memory_channel(10)
app = {
    "should_exit": False,
    "pause_event": trio.Event(),
    "curr_blueprint": 0,
    "blueprints": [],
}

def toggle_blueprint():
    global app
    app['curr_blueprint'] = (app['curr_blueprint'] + 1) % len(app['blueprints'])
    rr.log_file_from_path(app['blueprints'][app['curr_blueprint']])

class CursesStdoutRedirect:
    def __init__(self, log_path: Path = Path("output.log")):
        self.log_file = log_path.open("a", encoding="utf-8")
        self.log_path = log_path
        self.buffer = []

    def write(self, msg):
        if msg:
            self.log_file.write(msg)
            self.log_file.flush()

            lines = msg.splitlines()
            for line in lines:
                if line.strip() or line == "":
                    self.buffer.append(line)

    def flush(self):
        self.log_file.flush()

    def close(self):
        self.log_file.close()
        self.log_path.unlink(missing_ok=True)
        

async def handle_input(stdscr, cancel_scope):
    while True:
        # doesnt block because of stdscr.nodelay(True)
        try:
            key = stdscr.getch()
        except:
            key = -1

        if key == ord('q') or key == ord('Q'):
            app['should_exit'] = True
            cancel_scope.cancel() # Stop the nursery
            return
        elif key == ord('p'):
            if app['pause_event'].is_set():
                app['pause_event'] = trio.Event()
            else:
                app['pause_event'].set()
        elif key == ord('c'):
            toggle_blueprint()

        await trio.sleep(0.05)

async def draw_loop(stdscr, control_win, stdout_win, redirect):
    while True:
        height, width = stdscr.getmaxyx()
        control_height, _ = control_win.getmaxyx()
        stdout_height, _ = stdout_win.getmaxyx()

        # control window
        control_win.erase()
        control_win.box()
        control_win.addstr(1, 2, "CONTROLS", curses.A_BOLD)
        control_win.addstr(2, 2, "Press 'p' to pause | Press 'c' to toggle blueprints | Press 'q' to exit")
        control_win.noutrefresh()

        # stdout window
        stdout_win.erase()
        stdout_win.box()
        stdout_win.addstr(0, 2, " Standard Output ", curses.A_BOLD)

        max_visible_lines = stdout_height - 2
        visible_buffer = redirect.buffer[-max_visible_lines:]

        for idx, line in enumerate(visible_buffer):
            truncated_line = line[:width - 4]
            stdout_win.addstr(idx + 1, 2, truncated_line)

        stdout_win.noutrefresh()

        # render both windows
        curses.doupdate()

        await trio.sleep(0.033)

# sets up curses to handle input and display
async def run_curses(stdscr, bag_path: Path, options):
    curses.curs_set(False)
    stdscr.nodelay(True)
    stdscr.clear()

    height, width = stdscr.getmaxyx()

    # control window
    control_height = 5
    control_win = curses.newwin(control_height, width, 0, 0)

    # stdout window
    stdout_height = height - control_height
    stdout_win = curses.newwin(stdout_height, width, control_height, 0)

    # redirect stdout to logfile
    redirect = CursesStdoutRedirect()
    original_stdout = sys.stdout
    sys.stdout = redirect

    try:
        with trio.CancelScope() as cancel_scope:
            async with trio.open_nursery() as nursery:
                nursery.start_soon(handle_input, stdscr, cancel_scope)
                nursery.start_soon(draw_loop, stdscr, control_win, stdout_win, redirect)

                nursery.start_soon(stream_mcap, bag_path, options)

    finally:
        sys.stdout = original_stdout
        redirect.close()

def get_time_diff(decoded_msg, curr_time: float | None) -> tuple[float, float]:
    if curr_time is None:
        curr_time = decoded_msg.header.stamp.sec + decoded_msg.header.stamp.nanosec * 1e-9
        return (0, curr_time)
    else:
        last_time: float = curr_time
        curr_time: float = decoded_msg.header.stamp.sec + decoded_msg.header.stamp.nanosec * 1e-9
        elapsed: float = curr_time - last_time
        return (1/elapsed, curr_time)


def log_image(decoded_msg, channel):
    global image_stats

    height = decoded_msg.height
    width = decoded_msg.width
    encoding = decoded_msg.encoding

    if encoding == "16UC1":
        image_stats["depth_image_count"] += 1
        raw_data = np.frombuffer(decoded_msg.data, dtype=np.uint16)
        img_tensor = raw_data.reshape((height, width))

        # Downsampling
        width = int(width / 4)
        height = int(height / 4)
        dim = (width, height)

        img_tensor = cv2.resize(img_tensor, dim, interpolation=cv2.INTER_NEAREST)
        
        rr.log(channel.topic + '/image', rr.DepthImage(img_tensor, meter=1000))

        # Log FPS
        (fps, time) = get_time_diff(decoded_msg, image_stats['curr_depth_time'])
        image_stats['curr_depth_time'] = time

        rr.log('/stats/fps/depth', rr.Scalars(scalars=[fps]))
    else:
        raw_data = np.frombuffer(decoded_msg.data, dtype=np.uint8)
        if encoding in ("rgb8", "bgr8"):
            image_stats['rgb_image_count'] += 1
            img_tensor = raw_data.reshape((height, width, 3))

            if encoding == "bgr8":
                img_tensor = img_tensor[:, :, ::-1]

            rr.log(channel.topic + '/image', rr.Image(img_tensor))
            
            # Log FPS
            (fps, time) = get_time_diff(decoded_msg, image_stats['curr_rgb_time'])
            image_stats['curr_rgb_time'] = time

            rr.log('/stats/fps/color', rr.Scalars(scalars=[fps]))
        elif encoding in ("mono8", "8UC1"):
            img_tensor = raw_data.reshape((height, width))
            rr.log(channel.topic + '/image', rr.Image(img_tensor))
        else:
            return


    rr.log(
        "stats/image_loss",
        rr.Scalars(scalars=[abs(image_stats['rgb_image_count'] - image_stats['depth_image_count'])])
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

async def stream_mcap(mcap_path: Path, options):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    rr.init("batch_example")
    rr.spawn(memory_limit=options['memory_limit'])

    toggle_blueprint()
    
    print(f"Opening {mcap_path} for sequential streaming");

    message_count = 0
    start_time = time.time()
    # play_all = options['play_all']

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

        global app
        app['streamer'] = reader.iter_decoded_messages()

        while True:
            schema, channel, msg, decoded_msg = next(app['streamer'])
            
            if app['should_exit']:
                break

            # If pause is set, waits. Else, returns immediately
            await app['pause_event'].wait()

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
                # if not play_all and message_count % 20000 == 0:
                #     res = input("Stream paused. Do you want to continue? (q/Q to quit) ")
                #     if res.strip().lower() == "q":
                #         break

def main():
    parser = argparse.ArgumentParser(description=SCRIPT_DESCRIPTION)
    parser.add_argument(
        '-b', '--bag_path', type=Path, required=True,
        help='Path to rosbag fle to read and extract info from.'
    )
    parser.add_argument(
        '-m', '--memory_limit', type=str, required=False, default="50%",
        help='Memory limit before rerun garbage collects the old messages'
    )
    # parser.add_argument(
    #     '-a', '--play_all', action='store_true',
    #     help='Play the whole mcap without pause'
    # )
    parser.add_argument(
        '--header_timestamp', action='store_true',
        help='Use the message timestamp information instead of the log time in Ros'
    )
    # parser.add_argument(
    #     '-V', '--initial_blueprint', type=Path, required=False, default='./batch_blueprint.rbl',
    #     help='Initial view to start the recording'
    # )
    parser.add_argument(
        '--urdf', type=Path, required=False, default='../../../../data/config/rosario_v2.urdf.xacro',
        help='URDF file to use for the transforms'
    )
    parser.add_argument(
        '--blueprints', type=Path, required=False, default='./blueprints',
        help='Path to blueprints to toggle between'
    )

    args = parser.parse_args()

    options = {}

    options['memory_limit'] = args.memory_limit
    # options['play_all'] = args.play_all
    options['header_timestamp'] = args.header_timestamp
    # options['initial_blueprint'] = args.initial_blueprint
    options['urdf'] = args.urdf

    global app
    app['blueprints'] = [str(file) for file in args.blueprints.iterdir() if file.is_file()]
    app['pause_event'].set()

    print("Starting stream")
    curses.wrapper(run_trio_inside_wrapper, args.bag_path, options)

def run_trio_inside_wrapper(stdscr, bag_path: Path, options):
    trio.run(run_curses, stdscr, bag_path, options)

if __name__ == '__main__':
    main()
