from pathlib import Path
import time
import rerun as rr
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
import argparse

import numpy as np

SCRIPT_DESCRIPTION=\
"""This script allows the visualization of large mcap files by reading them sequentially
"""

def stream_mcap(mcap_path: Path):
    rr.init("batch_example")
    rr.spawn(memory_limit='2GB')
    
    print(f"Opening {mcap_path} for sequential streaming");

    with open(mcap_path, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])

        message_count = 0
        start_time = time.time()

        for schema, channel, msg, decoded_msg in reader.iter_decoded_messages():
            rr.set_time("log_time", timestamp=np.datetime64(msg.log_time, 'ns'))

            match channel.topic:
                case "/realsense/color/image_raw" | "/realsense/infra1/image_rect_raw" | "/realsense/infra2/image_rect_raw":
                    height = decoded_msg.height
                    width = decoded_msg.width
                    encoding = decoded_msg.encoding
                    
                    raw_data = np.frombuffer(decoded_msg.data, dtype=np.uint8)

                    if encoding in ("rgb8", "bgr8"):
                        img_tensor = raw_data.reshape((height, width, 3))
                        
                        if encoding == "bgr8":
                            img_tensor = img_tensor[:, :, ::-1]
                            
                    elif encoding in ("mono8", "8UC1"):
                        img_tensor = raw_data.reshape((height, width))
                        
                    else:
                        continue

                    entity_path = channel.topic.lstrip("/")

                    rr.log(entity_path, rr.Image(img_tensor))
                case _:
                    pass

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

    args = parser.parse_args()

    stream_mcap(args.bag_path)
