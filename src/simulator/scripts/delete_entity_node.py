#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from gazebo_msgs.srv import DeleteEntity


class DeleteEntityNode(Node):
    def __init__(self):
        super().__init__('delete_entity_node')
        self.client = self.create_client(DeleteEntity, '/delete_entity')

        while not self.client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info('Waiting for /delete_entity service...')

        req = DeleteEntity.Request()
        req.name = 'm3_gtr'  # entity name to delete

        self.future = self.client.call_async(req)
        self.future.add_done_callback(self.done_callback)

    def done_callback(self, future):
        try:
            result = future.result()
            if result.success:
                self.get_logger().info(f"Entity 'm3_gtr' deleted successfully.")
            else:
                self.get_logger().warn(f"Failed to delete entity: {result.status_message}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DeleteEntityNode()
    rclpy.spin_once(node, timeout_sec=5.0)  # give it time to complete
    rclpy.shutdown()


if __name__ == '__main__':
    main()
