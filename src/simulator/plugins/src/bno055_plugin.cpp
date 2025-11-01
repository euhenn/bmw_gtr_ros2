#include "bno055_plugin.hpp"

#define DEBUG false

using namespace std::chrono_literals;

namespace gazebo
{
    namespace bno055
    {
        BNO055::BNO055() : ModelPlugin()
        {
        }

        void BNO055::Load(physics::ModelPtr model_ptr, sdf::ElementPtr sdf_ptr)
        {
            this->m_model = model_ptr;

            if (!rclcpp::ok())
            {
                rclcpp::init(0, nullptr);
            }

            this->ros_node_ = rclcpp::Node::make_shared("bno055_plugin_node");

            std::string topic_name = "/automobile/IMU";
            this->imu_publisher_ = this->ros_node_->create_publisher<::msgs::msg::IMU>(topic_name, 10);

            // Set up an update connection
            this->update_connection_ = event::Events::ConnectWorldUpdateBegin(
                std::bind(&BNO055::OnUpdate, this, std::placeholders::_1));

            if (DEBUG)
            {
                std::cerr << "\n\n";
                RCLCPP_INFO(ros_node_->get_logger(), "====================================================================");
                RCLCPP_INFO(ros_node_->get_logger(), "[bno055_plugin] attached to: %s", this->m_model->GetName().c_str());
                RCLCPP_INFO(ros_node_->get_logger(), "[bno055_plugin] publish to: %s", topic_name.c_str());
                RCLCPP_INFO(ros_node_->get_logger(), "[bno055_plugin] Usefull data: orientation and position");
                RCLCPP_INFO(ros_node_->get_logger(), "====================================================================");
            }
        }

        void BNO055::OnUpdate(const common::UpdateInfo &info)
        {

            this->imu_msg_.roll         = this->m_model->RelativePose().Rot().Roll();
            this->imu_msg_.pitch        = this->m_model->RelativePose().Rot().Pitch();
            this->imu_msg_.yaw          = this->m_model->RelativePose().Rot().Yaw();
            this->imu_msg_.posx         = this->m_model->RelativePose().Pos().X();
            this->imu_msg_.posy         = abs(this->m_model->RelativePose().Pos().Y());
            this->imu_msg_.timestamp    = this->m_model->GetWorld()->SimTime().Float();

            this->imu_publisher_->publish(this->imu_msg_);
        }
    } // namespace bno055
} // namespace gazebo

GZ_REGISTER_MODEL_PLUGIN(gazebo::bno055::BNO055)