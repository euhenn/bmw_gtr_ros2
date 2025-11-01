#pragma once

#include <gazebo/gazebo.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/common/common.hh>
#include <gazebo/physics/physics.hh>

#include <rclcpp/rclcpp.hpp>
#include <deque>
#include <random>

#include "msgs/msg/localisation.hpp"

namespace gazebo
{
    namespace gps
    {   
        class GPS: public ModelPlugin
        {
        private: 
            physics::ModelPtr m_model;
            event::ConnectionPtr update_connection_;
            
            rclcpp::Node::SharedPtr ros_node_;
            rclcpp::Publisher<::msgs::msg::Localisation>::SharedPtr gps_publisher_;
            
            ::msgs::msg::Localisation gps_msg_;
            
            std::deque<::msgs::msg::Localisation> gps_history_;
            
            std::default_random_engine random_generator_;
            std::normal_distribution<double> gps_noise_distribution_;
            std::uniform_real_distribution<double> uniform_distribution_;
            
            double last_pub_;
            double packet_loss_time_;

            bool losing_pkg_ = false;
            
            // Config parameters
            double prob_of_starting_losing_pkgs_;
            double max_time_pkg_loss_;
            int gps_delay_;
            
        public: 
            GPS();
            void Load(physics::ModelPtr, sdf::ElementPtr);
            void OnUpdate(const common::UpdateInfo&);
        };
    }  // namespace gps
}  // namespace gazebo