"""ROS2 integration skeleton for Turing-Radar."""

from __future__ import annotations

import argparse

try:
    import rclpy
    from rclpy.node import Node

    ROS2_AVAILABLE = True
except ImportError:  # pragma: no cover
    rclpy = None
    Node = object
    ROS2_AVAILABLE = False


class TuringRadarNode(Node):  # pragma: no cover
    """Minimal ROS2 node skeleton for future ANIMA integration."""

    def __init__(self) -> None:
        super().__init__("turing_radar_node")
        self.get_logger().info("TuringRadarNode started")

    def process_pdws(self, pulses):
        """Hook for future subscriber callback -> deinterleaver -> publisher pipeline."""
        return {"labels": [], "n_clusters": 0, "n_pulses": len(pulses)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ROS2 node entrypoint for Turing-Radar")
    _ = parser.parse_args(argv)

    if not ROS2_AVAILABLE:
        print("ROS2 is not available in this environment. Install rclpy on the target runtime.")
        return 0

    assert rclpy is not None
    rclpy.init(args=argv)
    node = TuringRadarNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
