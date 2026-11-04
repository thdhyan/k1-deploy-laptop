"""ROS2 multi-robot camera relay node.

Architecture:
  - Parent process runs in domain 0 (laptop domain, where RViz lives)
  - For each robot, spawns a child process in domain 0
  - Child subscribes to /booster_video_stream (CompressedImage from robot)
  - Decompresses JPEG/PNG and republishes as raw Image under /{robot_ns}/...

Usage (launched by fleet_rviz.launch.py, do not run directly):
    ros2 run k1_fleet_rviz domain_relay --ros-args \
        -p robot_ns:=k1_0 -p source_domain:=0
"""
import os
import sys
import struct
import pickle
import signal
import multiprocessing
from multiprocessing import Process, Pipe

# Use 'spawn' to avoid inheriting parent's rclpy context via fork()
multiprocessing.set_start_method('spawn', force=True)

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
import numpy as np
import cv2

# ── Topics to relay ──
# The robot's active camera stream is /booster_video_stream (CompressedImage)
# We decompress and republish as raw Image for RViz

COMPRESSED_TOPIC = '/booster_video_stream'

# Additional raw topics (may not be publishing if camera not connected)
RAW_TOPICS = [
    ('/boostercamera/head/raw/rgb',            Image),
    ('/boostercamera/head/raw/right/rgb',      Image),
    ('/boostercamera/head/depth',              Image),
]


def _decompress_image(compressed_msg):
    """Decompress CompressedImage to Image."""
    try:
        np_arr = np.frombuffer(compressed_msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if cv_image is None:
            return None
        # Convert BGR to RGB
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        img_msg = Image()
        img_msg.header = compressed_msg.header
        img_msg.height = cv_image.shape[0]
        img_msg.width = cv_image.shape[1]
        img_msg.encoding = 'rgb8'
        img_msg.is_bigendian = False
        img_msg.step = cv_image.shape[1] * 3
        img_msg.data = cv_image.tobytes()
        return img_msg
    except Exception as e:
        return None


def _child_relay(conn, robot_ns, source_domain):
    """Child process: subscribe to camera topics, send messages over pipe."""
    os.environ['ROS_DOMAIN_ID'] = str(source_domain)

    import rclpy
    from rclpy.node import Node

    rclpy.init(args=None)
    node = Node(f'relay_child_{robot_ns}')

    # Match robot camera QoS
    qos_reliable = QoSProfile(
        depth=5,
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.VOLATILE,
    )

    qos_best = QoSProfile(
        depth=5,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )

    def compressed_cb(msg):
        """Handle CompressedImage: decompress and forward."""
        img_msg = _decompress_image(msg)
        if img_msg is not None:
            try:
                data = pickle.dumps(img_msg)
                conn.send(('msg', 'compressed_rgb', data))
            except Exception:
                pass

    def raw_cb(topic_base):
        def cb(msg):
            try:
                data = pickle.dumps(msg)
                conn.send(('msg', topic_base, data))
            except Exception:
                pass
        return cb

    # Subscribe to the active camera stream (CompressedImage)
    node.create_subscription(
        CompressedImage, COMPRESSED_TOPIC, compressed_cb, qos_best
    )

    # Also subscribe to raw topics (may not be publishing)
    for topic_base, msg_type in RAW_TOPICS:
        node.create_subscription(
            msg_type, topic_base, raw_cb(topic_base), qos_reliable
        )

    node.get_logger().info(
        f'Child [{robot_ns}] subscribing in domain {source_domain}: '
        f'compressed + {len(RAW_TOPICS)} raw topics'
    )

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)
    except Exception:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


# ── Parent process: runs in domain 0, republishes from pipe ──────────

class DomainRelay(Node):
    """Parent relay: republishes messages received from child processes."""

    def __init__(self):
        super().__init__('domain_relay_parent')

        self.declare_parameter('robot_ns', 'k1_0')
        self.declare_parameter('source_domain', 0)

        self._ns = self.get_parameter('robot_ns').value
        self._src_domain = self.get_parameter('source_domain').value

        qos = QoSProfile(
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # Create publishers in this domain (domain 0)
        self._pubs = {}

        # Compressed stream -> decompressed raw Image
        self._pubs['compressed_rgb'] = self.create_publisher(
            Image, f'/{self._ns}/boostercamera/head/rgb', qos
        )

        # Raw topics pass through
        for topic_base, msg_type in RAW_TOPICS:
            pub_topic = f'/{self._ns}{topic_base}'
            self._pubs[topic_base] = self.create_publisher(
                msg_type, pub_topic, qos
            )

        # Start child process
        parent_conn, child_conn = Pipe()
        self._child = Process(
            target=_child_relay,
            args=(child_conn, self._ns, self._src_domain),
            daemon=True,
        )
        self._child.start()
        child_conn.close()

        # Poll pipe for messages from child
        self.create_timer(0.001, self._poll_pipe)
        self._conn = parent_conn
        self._msg_count = 0

        self.get_logger().info(
            f'Relay [{self._ns}] domain {self._src_domain} -> 0: '
            f'compressed + {len(RAW_TOPICS)} raw topics bridged'
        )

    def _poll_pipe(self):
        """Read all pending messages from child and republish."""
        while self._conn.poll(0):
            try:
                kind, topic_base, data = self._conn.recv()
                if kind == 'msg':
                    msg = pickle.loads(data)
                    pub = self._pubs.get(topic_base)
                    if pub:
                        pub.publish(msg)
                        self._msg_count += 1
            except Exception:
                break

    def destroy_node(self):
        """Clean up child process."""
        if self._child and self._child.is_alive():
            self._child.terminate()
            self._child.join(timeout=2)
        if self._conn:
            self._conn.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DomainRelay()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
