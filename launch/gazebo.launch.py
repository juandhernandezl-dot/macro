import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                             TimerAction, SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():

    pkg_share = get_package_share_directory('modelo_robot')
    pkg_ros_gz_sim = get_package_share_directory('ros_gz_sim')

    urdf_path = os.path.join(pkg_share, 'urdf', 'URDF_TOBAR.urdf')
    controllers_yaml = os.path.join(pkg_share, 'config', 'controllers.yaml')

    # Reemplaza el placeholder con la ruta absoluta real
    with open(urdf_path, 'r') as f:
        robot_desc = f.read()
    robot_desc = robot_desc.replace('RUTA_CONTROLLERS_YAML', controllers_yaml)

    use_sim_time = LaunchConfiguration('use_sim_time', default='true')

    resource_path = os.path.dirname(pkg_share)
    set_ign_resource_path = SetEnvironmentVariable(
        name='IGN_GAZEBO_RESOURCE_PATH',
        value=resource_path
    )

    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, 'launch', 'gz_sim.launch.py')
        ),
        launch_arguments={'gz_args': '-r empty.sdf'}.items(),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'robot_description': robot_desc,
            'use_sim_time': use_sim_time,
        }]
    )

    spawn_robot = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='ros_gz_sim',
                executable='create',
                name='spawn_modelo_robot',
                output='screen',
                arguments=[
                    '-name', 'modelo_robot',
                    '-topic', '/robot_description',
                    '-x', '0.0',
                    '-y', '0.0',
                    '-z', '1.2',
                ],
            )
        ]
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        output='screen',
        arguments=[
            '/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock',
        ],
    )

    # Los spawners esperan a que el controller_manager levante
    # El controller_manager lo crea el plugin de Gazebo al spawnear el robot
    load_joint_state_broadcaster = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_state_broadcaster',
                           '--controller-manager', '/controller_manager'],
                output='screen',
            )
        ]
    )

    load_joint_trajectory_controller = TimerAction(
        period=7.0,
        actions=[
            Node(
                package='controller_manager',
                executable='spawner',
                arguments=['joint_trajectory_controller',
                           '--controller-manager', '/controller_manager'],
                output='screen',
            )
        ]
    )

    joint_state_publisher_gui = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='joint_state_publisher_gui',
                executable='joint_state_publisher_gui',
                name='joint_state_publisher_gui',
                parameters=[{'use_sim_time': use_sim_time}],
                output='screen',
            )
        ]
    )

    rviz = TimerAction(
        period=4.0,
        actions=[
            Node(
                package='rviz2',
                executable='rviz2',
                name='rviz2',
                output='screen',
                parameters=[{'use_sim_time': use_sim_time}],
            )
        ]
    )
    
    joint_gui_bridge = TimerAction(
        period=8.0,
        actions=[
            Node(
                package='modelo_robot',
                executable='joint_gui_bridge',
                name='joint_gui_bridge',
                output='screen',
            )
        ]
    )
    
    return LaunchDescription([
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Usar reloj de simulación'
        ),
        set_ign_resource_path,
        robot_state_publisher,
        gz_sim,
        spawn_robot,
        bridge,
        load_joint_state_broadcaster,
        load_joint_trajectory_controller,
        rviz,
        joint_gui_bridge,
    ])