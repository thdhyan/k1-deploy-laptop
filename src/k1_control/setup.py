from setuptools import setup, find_packages

package_name = 'k1_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/k1_control']),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Booster K1',
    author_email='th.dhyan.us@gmail.com',
    description='ROS2 control bridge for Booster K1',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'sim_bridge = k1_control.sim_bridge_node:main',
            'joint_state_pub = k1_control.joint_state_publisher:main',
        ],
    },
)
