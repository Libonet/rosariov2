from pathlib import Path
import time
import rerun as rr
import argparse

import numpy as np
from collections import deque

SCRIPT_DESCRIPTION=\
"""This script allows the visualization of large mcap files in rerun by reading them sequentially
"""

# import rerun.blueprint as rrb
# 
# blueprint = rrb.Blueprint(
#     rrb.TimeSeriesView(
#         origin=
#     )
# )

def log_image(decoded_msg, channel):
    height = decoded_msg.height
    width = decoded_msg.width
    encoding = decoded_msg.encoding

    if encoding == "16UC1":
        raw_data = np.frombuffer(decoded_msg.data, dtype=np.uint16)
        img_tensor = raw_data.reshape((height, width))

        rr.log(channel.topic, rr.DepthImage(img_tensor))
    else:
        raw_data = np.frombuffer(decoded_msg.data, dtype=np.uint8)
        if encoding in ("rgb8", "bgr8"):
            img_tensor = raw_data.reshape((height, width, 3))

            if encoding == "bgr8":
                img_tensor = img_tensor[:, :, ::-1]
        elif encoding in ("mono8", "8UC1"):
            img_tensor = raw_data.reshape((height, width))
        else:
            return

        rr.log(channel.topic, rr.Image(img_tensor))


def log_gnss(decoded_msg, channel):
    latlon = [decoded_msg.latitude,decoded_msg.longitude]

    # time = decoded_msg.header.stamp.sec * int(1e9) + decoded_msg.header.stamp.nanosec
    # progress = (time - options['initial_time']) / options['time_diff']

    # Color = i32 RGBA. Blue (R=0, G=0, B=255, A=255)
    # Hex = 0x0000FFFF
    color_scheme = {
        '/reach_1/fix': 0xFF0000FF,
        '/reach_2/fix': 0x00FF00FF,
        '/reach_3/fix': 0x0000FFFF,
    }

    rr.log(
        "/reach" + channel.topic,
        rr.GeoPoints(
            lat_lon=latlon,
            radii=rr.Radius.ui_points(5.0),
            colors=[color_scheme[channel.topic]],
        )
    )

'''
Imu(header=Header(stamp=Time(sec=1703261659, nanosec=191286087), frame_id=reach_2_imu), orientation=Quaternion(x=0.568870544, y=-0.418870896, z=-0.578819871, w=-0.407309562), orientation_covariance=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], angular_velocity=Vector3(x=-0.0063853506, y=-0.0117064761, z=0.0117064761), angular_velocity_covariance=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], linear_acceleration=Vector3(x=-9.8569282042175, y=0.181959326171875, z=-0.026336218310752), linear_acceleration_covariance=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
'''
def log_imu(decoded_msg, channel, options):
    # options['imu_orientation'].append(decoded_msg.orientation)
    # options['imu_angular_velocity'].append(decoded_msg.angular_velocity)
    # options['imu_linear_acceleration'].append(decoded_msg.linear_acceleration) 
        
    rr.log(
        "imu/angular_velocity",
        rr.Scalars(
            
        )
    )
    rr.log(
        "imu/linear_acceleration",
        rr.Scalars()
    )

def set_time(options, decoded_msg, msg):
    time = msg.log_time
    if options['header_timestamp'] and hasattr(msg, 'header'):
        time = to_ns(decoded_msg.header.stamp)

    rr.set_time("time", timestamp=np.datetime64(time, "ns"))

def to_ns(stamp):
    return stamp.sec * int(1e9) + stamp.nanosec

def stream_mcap(mcap_path: Path, options):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    rr.init("batch_example")
    rr.spawn(memory_limit=options['memory_limit'])
    
    print(f"Opening {mcap_path} for sequential streaming");

    message_count = 0
    start_time = time.time()
    play_all = options['play_all']

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        options['initial_time'] = reader.get_summary().statistics.message_start_time
        options['final_time'] = reader.get_summary().statistics.message_end_time
        options['time_diff'] = options['final_time'] - options['initial_time']

        # persist imu values to display them all in a 'window'
        options['imu_orientation'] = deque(maxlen=1000)
        options['imu_angular_velocity'] = deque(maxlen=1000)
        options['imu_linear_acceleration'] = deque(maxlen=1000)

        for schema, channel, msg, decoded_msg in reader.iter_decoded_messages():
            if schema is None: continue

            match schema.name:
                case "sensor_msgs/msg/Image":
                    set_time(options, decoded_msg, msg)
                    log_image(decoded_msg, channel)
                case "sensor_msgs/msg/NavSatFix":
                    set_time(options, decoded_msg, msg)
                    log_gnss(decoded_msg, channel)
                # case "sensor_msgs/msg/Imu":
                #     log_imu(decoded_msg, channel, options)
                case _:
                    continue

            message_count += 1
            if message_count % 1000 == 0:
                elapsed = time.time() - start_time
                print(f"Streamed {message_count} messages... ({elapsed:.2f}s elapsed)")
                if not play_all and message_count % 5000 == 0:
                    res = input("Stream paused. Do you want to continue? (q/Q to quit) ")
                    if res.lower() == "q":
                        break


def stream_mcap_with_rerun(mcap_path: Path, options):
    from rerun.experimental import McapReader, send_chunks

    rr.init("batch_example")
    rr.spawn(memory_limit=options['memory_limit'])

    print(f"Opening {mcap_path} for sequential streaming");

    reader = McapReader(mcap_path)

    message_count = 0
    start_time = time.time()

    read_iter = reader.stream()
    for chunk in read_iter:
        if chunk.entity_path != "/realsense/color/image_raw":
            # print(chunk.entity_path)
            continue

        send_chunks(chunk)
        print("sent chunk of ", chunk.entity_path)

        # print(chunk)
        # print("num columns =", chunk.num_columns)
        # print("num rows=", chunk.num_rows)

        message_count += 1
        if message_count % 10000 == 0:
            elapsed = time.time() - start_time
            print(f"Streamed {message_count} messages... ({elapsed:.2f}s elapsed)")

if __name__ == '__main__':
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

    args = parser.parse_args()

    options = {}

    options['memory_limit'] = args.memory_limit
    options['play_all'] = args.play_all
    options['header_timestamp'] = args.header_timestamp

    stream_mcap(args.bag_path, options)
    # stream_mcap_with_rerun(args.bag_path, options)
