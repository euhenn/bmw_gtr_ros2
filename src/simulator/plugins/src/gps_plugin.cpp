#include "gps_plugin.hpp"

#define DEBUG false
#define NOISE_ON_MEASUREMENTS false

using namespace std::chrono_literals;

namespace gazebo
{
    namespace gps
    {
        GPS::GPS() : ModelPlugin(),
                     gps_noise_distribution_(0.0, 0.07), // Default STD of 0.07 [m]
                     uniform_distribution_(0.0, 1.0),
                     prob_of_starting_losing_pkgs_(0.0),
                     max_time_pkg_loss_(0.0),
                     gps_delay_(0)
        {
        }

        void GPS::Load(physics::ModelPtr model_ptr, sdf::ElementPtr sdf_ptr)
        {

            this->m_model = model_ptr;

            if (!rclcpp::ok())
            {
                rclcpp::init(0, nullptr);
            }

            this->ros_node_ = rclcpp::Node::make_shared("gps_plugin_node");
            std::string topic_name = "/automobile/localisation";
            this->gps_publisher_ = this->ros_node_->create_publisher<::msgs::msg::Localisation>(topic_name, 10);

            random_generator_.seed(std::chrono::system_clock::now().time_since_epoch().count());

            // Set up update connection
            this->update_connection_ = event::Events::ConnectWorldUpdateBegin(
                std::bind(&GPS::OnUpdate, this, std::placeholders::_1));

            if (DEBUG)
            {
                RCLCPP_INFO(ros_node_->get_logger(), "====================================================================");
                RCLCPP_INFO(ros_node_->get_logger(), "[gps_plugin] attached to: %s", this->m_model->GetName().c_str());
                RCLCPP_INFO(ros_node_->get_logger(), "[gps_plugin] publish to: %s", topic_name.c_str());
                RCLCPP_INFO(ros_node_->get_logger(), "[gps_plugin] PROB_OF_STARTING_LOSING_PKGS: %f", prob_of_starting_losing_pkgs_);
                RCLCPP_INFO(ros_node_->get_logger(), "[gps_plugin] MAX_TIME_PKG_LOSS: %f", max_time_pkg_loss_);
                RCLCPP_INFO(ros_node_->get_logger(), "[gps_plugin] GPS_DELAY: %d", gps_delay_);
                RCLCPP_INFO(ros_node_->get_logger(), "====================================================================");
            }
        }

        void GPS::OnUpdate(const common::UpdateInfo &info)
        {
            auto current_time = this->m_model->GetWorld()->SimTime();
            double time_stamp = current_time.Double();

            // Add noise
            double true_x = this->m_model->RelativePose().Pos().X();
            double true_y = std::abs(this->m_model->RelativePose().Pos().Y());

            if (NOISE_ON_MEASUREMENTS)
            {
                // Simulate GPS noise
                this->gps_msg_.pos_x = true_x + gps_noise_distribution_(random_generator_);
                this->gps_msg_.pos_y = true_y + gps_noise_distribution_(random_generator_);
                this->gps_msg_.yaw = this->m_model->RelativePose().Rot().Yaw();
            }
            else
            {
                // No noise on measurements
                this->gps_msg_.pos_x = true_x;
                this->gps_msg_.pos_y = true_y;
                this->gps_msg_.yaw = this->m_model->RelativePose().Rot().Yaw();
            }
            this->gps_msg_.timestamp = time_stamp;

            // Add to history
            this->gps_history_.push_back(this->gps_msg_);

            // If history is longer than GPS_DELAY, pop first element
            if (this->gps_history_.size() > static_cast<size_t>(gps_delay_))
            {
                ::msgs::msg::Localisation gps_to_pub = this->gps_history_.front();
                this->gps_history_.pop_front();

                // Package loss sim
                if (this->losing_pkg_)
                {
                    double time_diff = time_stamp - this->last_pub_;
                    if ((time_diff > this->packet_loss_time_) || (time_diff <= 0.0))
                    {
                        this->losing_pkg_ = false;
                    }
                }
                else
                {
                    bool start_losing_pkgs = uniform_distribution_(random_generator_) < prob_of_starting_losing_pkgs_;
                    if (start_losing_pkgs)
                    {
                        this->packet_loss_time_ = max_time_pkg_loss_ * uniform_distribution_(random_generator_);
                        this->losing_pkg_ = true;
                    }
                }

                bool inside_no_signal_region = false; // region check logic if needed
                bool can_publish = (!inside_no_signal_region) && (!this->losing_pkg_);

                if (can_publish)
                {
                    this->gps_publisher_->publish(gps_to_pub);
                    this->last_pub_ = time_stamp;
                }
            }
        }
    } // namespace gps
} // namespace gazebo
GZ_REGISTER_MODEL_PLUGIN(gazebo::gps::GPS)