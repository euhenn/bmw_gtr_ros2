#pragma once

#include <gazebo/gazebo.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/common.hh>
#include <gazebo/physics/physics.hh>

#include <rclcpp/rclcpp.hpp>
#include <msgs/msg/imu.hpp>

namespace gazebo
{
    namespace bno055
    {   
        class BNO055: public ModelPlugin
        {
        private: 
            physics::ModelPtr m_model;
            event::ConnectionPtr update_connection_;
            
            // ROS2 components
            rclcpp::Node::SharedPtr ros_node_;
            rclcpp::Publisher<::msgs::msg::IMU>::SharedPtr imu_publisher_;
            
            // IMU message
            ::msgs::msg::IMU imu_msg_;

        public: 
            BNO055();
            void Load(physics::ModelPtr, sdf::ElementPtr);
            void OnUpdate(const common::UpdateInfo&);
        };
    } // namespace bno055
} // namespace gazebo