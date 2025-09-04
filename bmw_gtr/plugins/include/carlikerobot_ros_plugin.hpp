#pragma once

#include <gazebo/gazebo.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/physics/physics.hh>
#include <memory>
#include <string>
#include <thread>
#include "carlikerobot.hpp"
#include "rapidjson/document.h"
// ROS 2
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors.hpp>
#include <std_msgs/msg/string.hpp>
#include <rclcpp/qos.hpp>

namespace gazebo {
namespace carlikerobot {

class IRobotCommandSetter {
public:
    virtual void setCommand() = 0;
    virtual ~IRobotCommandSetter() = default;
    float f_steer;
    float f_speed;
};

typedef std::shared_ptr<IRobotCommandSetter> IRobotCommandSetterPtr;

class CMessageHandler {
public:
    CMessageHandler(std::string, IRobotCommandSetter*);
    ~CMessageHandler();
    void OnMsgCommand(const std_msgs::msg::String::SharedPtr msg);
    //void OnSpeedUpdate(const std_msgs::msg::Float32::SharedPtr msg);
    //void OnSteerUpdate(const std_msgs::msg::Float32::SharedPtr msg);
    rclcpp::Node::SharedPtr GetNode() const;

private:
    void unknownMessage();
    void brakeMessage(float val);
    void spedMessage(float val);
    void sterMessage(float val);
    void moveMessage(float speed, float steer);
    
    IRobotCommandSetter* _robotSetter;
    rclcpp::Node::SharedPtr _rosNode;
    
    std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> _executor;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr _commandSubscriber;
    //rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr speed_sub_;
    //rclcpp::Subscription<std_msgs::msg::Float32>::SharedPtr steer_sub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr _feedbackPublisher;
    std::thread _spinThread;
};

class CCarLikeRobotRosPlugin : public ModelPlugin, public IRobotCommandSetter {
public:
    bool LoadParameterJoints(sdf::ElementPtr _sdf);
    void Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf) override;
    void setCommand() override;

private:
    IWheelsSpeedPtr _rearWheelsSpeedPtr;
    IWheelsSpeedPtr _frontWheelSpeedPtr;
    ISteerWheelsPtr _steerWheelsAnglePtr;
    physics::ModelPtr _model;
    std::shared_ptr<CMessageHandler> _messageHandler;
};

}  // namespace carlikerobot
}  // namespace gazebo