from pathlib import Path
import time
import rerun as rr
import argparse

import numpy as np

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

    rr.log(
        channel.topic,
        rr.GeoPoints(
            lat_lon=latlon,
            radii=rr.Radius.ui_points(5.0),
            colors=decoded_msg.header.stamp.sec,
        )
    )
    # rr.log(
    #     "pointTest",
    #     rr.Points3D(
    #         positions=[decoded_msg.latitude,decoded_msg.longitude,decoded_msg.altitude],
    #         radii=rr.Radius.ui_points(5.0),
    #         colors=decoded_msg.header.stamp.sec
    #     )
    # )

def stream_mcap(mcap_path: Path, options):
    from mcap.reader import make_reader
    from mcap_ros2.decoder import DecoderFactory
    rr.init("batch_example")
    rr.spawn(memory_limit=options['memory_limit'])
    
    print(f"Opening {mcap_path} for sequential streaming");

    message_count = 0
    start_time = time.time()

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])

        for schema, channel, msg, decoded_msg in reader.iter_decoded_messages():
            rr.set_time("time", timestamp=np.datetime64(msg.log_time, "ns"))

            if schema is None: continue

            match schema.name:
                case "sensor_msgs/msg/Image":
                    log_image(decoded_msg, channel)
                case "sensor_msgs/msg/NavSatFix":
                    log_gnss(decoded_msg, channel)
                case _:
                    pass

            message_count += 1
            if message_count % 10000 == 0:
                elapsed = time.time() - start_time
                print(f"Streamed {message_count} messages... ({elapsed:.2f}s elapsed)")

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
        '-m', '--memory_limit', type=str, required=False, default="4GB",
        help='Memory limit before rerun garbage collects the old messages'
    )

    args = parser.parse_args()

    options = {}

    options['memory_limit'] = args.memory_limit

    stream_mcap(args.bag_path, options)
    # stream_mcap_with_rerun(args.bag_path, options)
