// carlikerobot_ros_plugin.cpp - Fixed version with proper registration

#include "carlikerobot_ros_plugin.hpp"
#include <rclcpp/qos.hpp>

#define DEBUG false

namespace gazebo {
namespace carlikerobot {

// CMessageHandler implementation (same as before)
CMessageHandler::CMessageHandler(std::string _modelName, IRobotCommandSetter* _setter) {
    this->_robotSetter = _setter;

    std::string topicName = "/automobile/command";
    std::string listen_topicName = "/automobile/feedback";

    // Initialize ROS2 node only if not already done
    if (!rclcpp::ok()) {
        rclcpp::init(0, nullptr);
    }
    
    // Use model name in node name to make it unique
    std::string nodeName = _modelName + "_node";
    this->_rosNode = std::make_shared<rclcpp::Node>(nodeName);

    this->_commandSubscriber = _rosNode->create_subscription<std_msgs::msg::String>(
        topicName,
        10,  
        std::bind(&CMessageHandler::OnMsgCommand, this, std::placeholders::_1)
    );

    this->_feedbackPublisher = _rosNode->create_publisher<std_msgs::msg::String>(
        listen_topicName, 
        10
        //rclcpp::QoS(1000).reliable()  // Increase from 10 to 100
    );

    // Create executor and spin in background thread
    _executor = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    _executor->add_node(_rosNode);
    
    _spinThread = std::thread([this]() {
        _executor->spin();
    });

    if (DEBUG) {
        //std::cerr << "\n\n";
        RCLCPP_INFO(_rosNode->get_logger(), "===========================================================");
        RCLCPP_INFO(_rosNode->get_logger(), "[carlikerobot_ros_plugin] attached to: %s", _modelName.c_str());
        RCLCPP_INFO(_rosNode->get_logger(), "[carlikerobot_ros_plugin] subscribe to: %s", topicName.c_str());
        RCLCPP_INFO(_rosNode->get_logger(), "[carlikerobot_ros_plugin] publish to: %s", listen_topicName.c_str());
        RCLCPP_INFO(_rosNode->get_logger(), "Node name: %s", nodeName.c_str());
        RCLCPP_INFO(_rosNode->get_logger(), "===========================================================");
    }
}

rclcpp::Node::SharedPtr gazebo::carlikerobot::CMessageHandler::GetNode() const {
    return _rosNode;
}

CMessageHandler::~CMessageHandler() {
    // Stop the executor first
    if (_executor != nullptr) {
        _executor->cancel();
    }
    
    // Join the thread
    if (_spinThread.joinable()) {
        _spinThread.join();
    }
    
    // Clean up
    if (_executor != nullptr) {
        _executor->remove_node(_rosNode);
        _executor.reset();
    }
    
    if (_rosNode != nullptr) {
        _rosNode.reset(); 
    }
}

void CMessageHandler::OnMsgCommand(const std_msgs::msg::String::SharedPtr msg) {
    if (DEBUG) {
        RCLCPP_INFO(_rosNode->get_logger(), "Received command: %s", msg->data.c_str());
    }
    
    rapidjson::Document doc;
    doc.Parse(msg->data.c_str());

    if (doc.HasParseError()) {
        RCLCPP_ERROR(_rosNode->get_logger(), "JSON parse error");
        unknownMessage();
        return;
    }

    if (!doc.HasMember("action")) {
        RCLCPP_WARN(_rosNode->get_logger(), "Missing 'action' field in command");
        unknownMessage();
        return;
    }

    std::string command = doc["action"].GetString();

    if (command == "1" && doc.HasMember("speed")) {
        spedMessage(doc["speed"].GetFloat());
    } else if (command == "2" && doc.HasMember("steerAngle")) {
        sterMessage(doc["steerAngle"].GetFloat());
    } else if (command == "3" && doc.HasMember("steerAngle")) {
        brakeMessage(doc["steerAngle"].GetFloat());
    } else if (command == "4" && doc.HasMember("speed") && doc.HasMember("steerAngle")) {
        moveMessage(doc["speed"].GetFloat(), doc["steerAngle"].GetFloat());
    } else {
        RCLCPP_WARN(_rosNode->get_logger(), "Unknown command or missing parameters: %s", command.c_str());
        unknownMessage();
    }
}

void CMessageHandler::unknownMessage() {
    auto msg = std_msgs::msg::String();
    msg.data = "@MESS:err;;";
    _feedbackPublisher->publish(msg);
    
    if (DEBUG) {
        RCLCPP_INFO(_rosNode->get_logger(), "Sent error feedback");
    }
}

void CMessageHandler::brakeMessage(float val) {
    _robotSetter->f_speed = 0;
    _robotSetter->f_steer = val;
    _robotSetter->setCommand();

    auto msg = std_msgs::msg::String();
    msg.data = "@3:ack;;";
    _feedbackPublisher->publish(msg);
    
    if (DEBUG) {
        RCLCPP_INFO(_rosNode->get_logger(), "Brake command executed, steer: %f", val);
    }
}

void CMessageHandler::spedMessage(float val) {
    _robotSetter->f_speed = val;
    _robotSetter->setCommand();

    auto msg = std_msgs::msg::String();
    msg.data = "@1:ack;;";
    _feedbackPublisher->publish(msg);
    
    if (DEBUG) {
        RCLCPP_INFO(_rosNode->get_logger(), "Speed command executed: %f", val);
    }
}

void CMessageHandler::sterMessage(float val) {
    _robotSetter->f_steer = val;
    _robotSetter->setCommand();

    auto msg = std_msgs::msg::String();
    msg.data = "@2:ack;;";
    _feedbackPublisher->publish(msg);
    
    if (DEBUG) {
        RCLCPP_INFO(_rosNode->get_logger(), "Steer command executed: %f", val);
    }
}

void CMessageHandler::moveMessage(float speed, float steer) {
    _robotSetter->f_speed = speed;
    _robotSetter->f_steer = steer;
    _robotSetter->setCommand();

    auto msg = std_msgs::msg::String();
    msg.data = "@4:ack;;";
    _feedbackPublisher->publish(msg);
    
    if (DEBUG) {
        RCLCPP_INFO(_rosNode->get_logger(), "Move command executed - speed: %f, steer: %f", speed, steer);
    }
}

// CCarLikeRobotRosPlugin implementation
void CCarLikeRobotRosPlugin::Load(physics::ModelPtr _parent, sdf::ElementPtr _sdf) {
    //std::cout << "=== CCarLikeRobotRosPlugin::Load() called ===" << std::endl;
    
    // Store the model pointer
    this->_model = _parent;
    
    if (DEBUG) {
        std::cout << "Loading CCarLikeRobotRosPlugin for model: " << _model->GetName() << std::endl;
    }
    
    // Initialize ROS2 if not already done
    if (!rclcpp::ok()) {
        rclcpp::init(0, nullptr);
    }
    
    // Create the message handler (moved before load parameters to call the _rosNode)
    this->_messageHandler = std::make_shared<CMessageHandler>(
        this->_model->GetName(), 
        static_cast<IRobotCommandSetter*>(this)
    );

    // Load joint parameters from SDF
    if (!LoadParameterJoints(_sdf)) {
        std::cerr << "Failed to load joint parameters" << std::endl;
        return;
    }
    
    // Initialize speed and steering values
    this->f_speed = 0.0;
    this->f_steer = 0.0;

    //std::cout << "=== CCarLikeRobotRosPlugin loaded successfully ===" << std::endl;
}


bool CCarLikeRobotRosPlugin::LoadParameterJoints(sdf::ElementPtr f_sdf)
        {

            // start [wheelbase] [axletrack] [wheelradius] handling
            double l_wheelbase   = 0;
            double l_axletrack   = 0;  
            double l_wheelradius = 0; 

            auto _rosNode = _messageHandler->GetNode();

            if(DEBUG)
            {
                std::cerr << "\n\n";    
                RCLCPP_INFO(_rosNode->get_logger(),"====================================================================");
            }

            if(f_sdf->HasElement("wheelbase"))
            {
                l_wheelbase = f_sdf->Get<double>("wheelbase");

                if(DEBUG)
                {
                    RCLCPP_INFO(_rosNode->get_logger(),"OK [wheelbase]   = %f", l_wheelbase);
                }
            }
            else
            {
                if(DEBUG)
                {
                    RCLCPP_INFO(_rosNode->get_logger(),"WARNING: [wheelbase] = 0 DEFAULT");
                }
            }
            
            if(f_sdf->HasElement("axletrack"))
            {
                l_axletrack = f_sdf->Get<double>("axletrack");

                if(DEBUG)
                {
                    RCLCPP_INFO(_rosNode->get_logger(),"OK [axletrack]   = %f", l_axletrack);
                }
            }
            else
            {
                RCLCPP_INFO(_rosNode->get_logger(),"WARNING: [axletrack] = 0 DEFAULT");
            }
            
            if(f_sdf->HasElement("wheelradius"))
            {
                l_wheelradius = f_sdf->Get<double>("wheelradius");

                if(DEBUG)
                {
                    RCLCPP_INFO(_rosNode->get_logger(),"OK [wheelradius] = %f", l_wheelradius);
                }
            }
            else
            {
                std::cerr << "WARNING: [wheelradius] = 0 DEFAULT\n";
                std::cerr << "CRITICAL: Invalid plugin parameters, wrong wheel radius.\n\
                              CarLikeRobotPlugin plugin is not loaded.\n";
                return false;
            }
            // end [wheelbase] [axletrack] [wheelradius] handling

            // start [speed_wheel_joints] [front_wheel_joints] [rear_wheel_joints]
            sdf::ElementPtr l_steering_joints_sdf   = NULL;
            sdf::ElementPtr l_speed_wheel_sdf       = NULL; 
            sdf::ElementPtr l_front_wheel_sdf       = NULL;
            sdf::ElementPtr l_rear_wheel_sdf        = NULL;
            
            if(f_sdf->HasElement("speed_wheel_joints"))
            {
    
                double l_kp_speed = 0;
                double l_ki_speed = 0;
                double l_kd_speed = 0;
    
                l_speed_wheel_sdf = f_sdf->GetElement("speed_wheel_joints");

                if(DEBUG)
                {
                    RCLCPP_INFO(_rosNode->get_logger(),"====================================================================");
                    std::cerr << "\n\n";
                    RCLCPP_INFO(_rosNode->get_logger(),"====================================================================");
                    RCLCPP_INFO(_rosNode->get_logger(),"FOUND: [speed_wheel_joints]");
                }

                // start [kp] [ki] [kd] 
                if(l_speed_wheel_sdf->HasElement("kp"))
                {
                    l_kp_speed = l_speed_wheel_sdf->Get<double>("kp");
                    
                    if(DEBUG)
                    {
                        RCLCPP_INFO(_rosNode->get_logger(),"OK [kp] = %f", l_kp_speed);
                    } 
                }
                else
                {
                    std::cerr << "<kp> under <speed_wheel_joints> not found. 0 DEFAULT\n";
                }

                if(l_speed_wheel_sdf->HasElement("kd"))
                {
                    l_kd_speed = l_speed_wheel_sdf->Get<double>("kd");
                    
                    if(DEBUG)
                    {
                        RCLCPP_INFO(_rosNode->get_logger(),"OK [kd] = %f", l_kd_speed);
                    }
                }
                else
                {
                    std::cerr << "<kd> under <speed_wheel_joints> not found. 0 DEFAULT\n";
                }

                if(l_speed_wheel_sdf->HasElement("ki"))
                {
                    l_ki_speed = l_speed_wheel_sdf->Get<double>("ki");
                    
                    if(DEBUG)
                    {
                        RCLCPP_INFO(_rosNode->get_logger(),"OK [ki] = %f", l_ki_speed);
                    }
                }
                else
                {
                    std::cerr << "<ki> under <speed_wheel_joints> not found. 0 DEFAULT\n";
                }// end [kp] [ki] [kd] 


                // start HasElement("front_wheel_joints")
                if(l_speed_wheel_sdf->HasElement("front_wheel_joints"))
                {
                    if(DEBUG)
                    {
                        RCLCPP_INFO(_rosNode->get_logger(),"====================================================================");
                        std::cerr << "\n\n";
                        RCLCPP_INFO(_rosNode->get_logger(),"====================================================================");
                        RCLCPP_INFO(_rosNode->get_logger(),"FOUND: [front_wheel_joints]");
                    }

                    l_front_wheel_sdf = l_speed_wheel_sdf->GetElement("front_wheel_joints");

                    // START FRONT WHEELS CONTROLLER FOR SPINNING
                    std::string l_left  = l_front_wheel_sdf->Get<std::string>("leftjoint");
                    if(l_left == " ")
                    {
                        std::cerr << "CRITICAL: empty front [leftjoint] name. Plugin WAS NOT LOADED. exitting...\n";
                        return false;
                    }
                    else
                    {
                        if(DEBUG)
                        {
                            RCLCPP_INFO(_rosNode->get_logger(),"OK front [leftjoint]  name = %s", l_left.c_str());
                        }
                    }

                    std::string l_right = l_front_wheel_sdf->Get<std::string>("rightjoint");
                    if(l_right == " ")
                    {
                        std::cerr << "CRITICAL: empty front [rightjoint] name. Plugin WAS NOT LOADED. exitting...\n";
                        return false;
                    }
                    else
                    {
                        if(DEBUG)
                        {
                            RCLCPP_INFO(_rosNode->get_logger(),"OK front [rightjoint] name = %s", l_right.c_str());
                        }
                    }

                    physics::JointPtr l_rightJoint  = this->_model->GetJoint(l_right);
                    if(l_rightJoint == NULL)
                    {
                        std::cerr << "CRITICAL: front [rightjoint] name MISMACH. Check model's joints names\
                                      Plugin WAS NOT LOADED. exitting...\n";
                        return false;
                    }
                    else
                    {
                        if(DEBUG)
                        {
                            RCLCPP_INFO(_rosNode->get_logger(),"OK front [rightjoint] was found in model");
                        }
                    }

                    physics::JointPtr l_leftJoint   = this->_model->GetJoint(l_left);
                    if(l_leftJoint == NULL)
                    {
                        std::cerr << "CRITICAL: front [leftjoint] name MISMACH. Check model's joints names\
                                      Plugin WAS NOT LOADED. exitting...\n";
                        return false;
                    }
                    else
                    {
                        if(DEBUG)
                        {
                            RCLCPP_INFO(_rosNode->get_logger(),"OK front [leftjoint]  was found in model");
                        }
                    }

                    // PID
                    common::PID l_rightPID  = common::PID(l_kp_speed, l_ki_speed, l_kd_speed);
                    common::PID l_leftPID   = common::PID(l_kp_speed, l_ki_speed, l_kd_speed);

                    // FrontWheelsSpeed
                    this->_frontWheelSpeedPtr = IWheelsSpeedPtr(new CFrontWheelsSpeed(l_axletrack,
                                                                                      l_wheelbase,
                                                                                      l_wheelradius,
                                                                                      l_rightJoint,
                                                                                      l_leftJoint,
                                                                                      l_rightPID,
                                                                                      l_leftPID,
                                                                                      this->_model));
                    // END FRONT WHEELS CONTROLLER FOR SPINNING
                }// END HasElement("front_wheel_joints")


                // start HasElement("rear_wheel_joints") 
                if(l_speed_wheel_sdf->HasElement("rear_wheel_joints"))
                {
                    if(DEBUG)
                    {
                        RCLCPP_INFO(_rosNode->get_logger(),"====================================================================");
                        std::cerr << "\n\n";
                        RCLCPP_INFO(_rosNode->get_logger(),"====================================================================");
                        RCLCPP_INFO(_rosNode->get_logger(),"FOUND: [rear_wheel_joints]");
                    }

                    l_rear_wheel_sdf = l_speed_wheel_sdf->GetElement("rear_wheel_joints");

                    // START REAR WHEELS CONTROLLER FOR SPINNING
                    std::string l_left  = l_rear_wheel_sdf->Get<std::string>("leftjoint");
                    if(l_left == " ")
                    {
                        std::cerr << "CRITICAL: empty rear [leftjoint] name. Plugin WAS NOT LOADED. exitting...\n";
                        return false;
                    }
                    else
                    {
                        if(DEBUG)
                        {
                            RCLCPP_INFO(_rosNode->get_logger(),"OK rear [leftjoint]  name = %s", l_left.c_str());
                        }
                    }

                    std::string l_right = l_rear_wheel_sdf->Get<std::string>("rightjoint");
                    if(l_right == " ")
                    {
                        std::cerr << "CRITICAL: empty rear [rightjoint] name. Plugin WAS NOT LOADED. exitting...\n";
                        return false;
                    }
                    else
                    {
                       if(DEBUG)
                       {
                            RCLCPP_INFO(_rosNode->get_logger(),"OK rear [rightjoint] name = %s", l_right.c_str());
                       }
                    }

                    physics::JointPtr l_rightJoint  = this->_model->GetJoint(l_right);
                    if(l_rightJoint == NULL)
                    {
                        std::cerr << "CRITICAL: rear [rightjoint] name MISMACH. Check model's joints names\
                                      Plugin WAS NOT LOADED. exitting...\n";
                        return false; 
                    }
                    else
                    {
                        if(DEBUG)
                        {
                            RCLCPP_INFO(_rosNode->get_logger(),"OK rear [rightjoint] was found in model");
                        }
                    }

                    physics::JointPtr l_leftJoint   = this->_model->GetJoint(l_left);
                    if(l_leftJoint == NULL)
                    {
                        std::cerr << "CRITICAL: rear [leftjoint] name MISMACH. Check model's joints names\
                                      Plugin WAS NOT LOADED. exitting...\n";
                        return false;
                    }
                    else
                    {
                        if(DEBUG)
                        {
                            RCLCPP_INFO(_rosNode->get_logger(),"OK rear [leftjoint] was found in model");
                        }
                    }

                    // PID
                    common::PID l_rightPID = common::PID(l_kp_speed,l_ki_speed,l_kd_speed);
                    common::PID l_leftPID  = common::PID(l_kp_speed,l_ki_speed,l_kd_speed);
                    
                    // RearWheelsSpeed
                    this->_rearWheelsSpeedPtr = IWheelsSpeedPtr(new CRearWheelsSpeed( l_axletrack
                                                                    ,l_wheelbase
                                                                    ,l_wheelradius
                                                                    ,l_rightJoint
                                                                    ,l_leftJoint
                                                                    ,l_rightPID
                                                                    ,l_leftPID
                                                                    ,this->_model));
                    // END REAR WHEELS CONTROLLER FOR SPINNING
                }// END HasElement("rear_wheel_joints")                
            
            }// END [speed_wheel_joints] [front_wheel_joints] [rear_wheel_joints]
            

            if(f_sdf->HasElement("steer_wheel_joints"))
            {

                l_steering_joints_sdf = f_sdf->GetElement("steer_wheel_joints");
                
                if(DEBUG)
                {
                    RCLCPP_INFO(_rosNode->get_logger(),"====================================================================");
                    std::cerr << "\n\n";
                    RCLCPP_INFO(_rosNode->get_logger(),"====================================================================");    
                    RCLCPP_INFO(_rosNode->get_logger(),"FOUND: [steer_wheel_joints]");
                }
                
                double l_kp_position = 0;
                double l_ki_position = 0;
                double l_kd_position = 0;

                // start [kp] [ki] [kd] 
                if(l_steering_joints_sdf->HasElement("kp"))
                {
                    l_kp_position = l_steering_joints_sdf->Get<double>("kp");
                    if(DEBUG)
                    {
                        RCLCPP_INFO(_rosNode->get_logger(),"OK [kp] = %f", l_kp_position);
                    }
                }
                else
                {
                    std::cerr << "<kp> under <speed_wheel_joints> not found. 0 DEFAULT\n";
                }

                if(l_steering_joints_sdf->HasElement("kd"))
                {
                    l_kd_position = l_steering_joints_sdf->Get<double>("kd");
                    if(DEBUG)
                    {
                        RCLCPP_INFO(_rosNode->get_logger(),"OK [kd] = %f", l_kd_position);
                    }
                }
                else
                {
                    std::cerr << "<kd> under <speed_wheel_joints> not found. 0 DEFAULT\n";
                }

                if(l_steering_joints_sdf->HasElement("ki"))
                {
                    l_ki_position = l_steering_joints_sdf->Get<double>("ki");
                    
                    if(DEBUG)
                    {
                        RCLCPP_INFO(_rosNode->get_logger(),"OK [ki] = %f", l_ki_position);
                    }
                }
                else
                {
                    std::cerr << "<ki> under <speed_wheel_joints> not found. 0 DEFAULT\n";
                }// end [kp] [ki] [kd] 
                

                // Steering angle pid
                std::string l_left  = l_steering_joints_sdf->Get<std::string>("leftjoint");
                std::string l_right = l_steering_joints_sdf->Get<std::string>("rightjoint");
                
                if(l_left=="" || l_right==""){
                    std::cerr << "CRITICAL: Invalid steering joints. CarLikeRobotPlugin plugin is not loaded.\n";
                    return false;
                }


                physics::JointPtr l_rightJoint = this->_model->GetJoint(l_right);
                physics::JointPtr l_leftJoint  = this->_model->GetJoint(l_left);
                
                if(l_leftJoint==NULL || l_rightJoint==NULL){
                    std::cerr<<"Invalid steering joints. CarLikeRobotPlugin plugin is not loaded.\n";
                    return false;
                }

                // PID
                common::PID l_rightPID = common::PID(l_kp_position,l_ki_position,l_kd_position);
                common::PID l_leftPID  = common::PID(l_kp_position,l_ki_position,l_kd_position);
                // A steering angle calculator
                this->_steerWheelsAnglePtr = ISteerWheelsPtr(new CSteerWheelsAngle( l_axletrack
                                                                    ,l_wheelbase
                                                                    ,l_rightJoint
                                                                    ,l_leftJoint
                                                                    ,l_rightPID
                                                                    ,l_leftPID
                                                                    ,this->_model));
                // END HasElement("steer_wheel_joints")
            }

            return true;
        }


void CCarLikeRobotRosPlugin::setCommand() {
    if (DEBUG) {
        std::cout << "setCommand called - speed: " << this->f_speed 
                  << ", steer: " << this->f_steer << std::endl;
    }
             
    if(this->_rearWheelsSpeedPtr!=NULL){
       this->_rearWheelsSpeedPtr->update(this->f_steer,this->f_speed); 
    }
    
    if(this->_frontWheelSpeedPtr!=NULL){
        this->_frontWheelSpeedPtr->update(this->f_steer,this->f_speed);    
    }
    
    if(this->_steerWheelsAnglePtr!=NULL){
        this->_steerWheelsAnglePtr->update(this->f_steer);               
    }
}

} // namespace carlikerobot
} // namespace gazebo

// CRITICAL: Plugin registration MUST be outside all namespaces
GZ_REGISTER_MODEL_PLUGIN(gazebo::carlikerobot::CCarLikeRobotRosPlugin)