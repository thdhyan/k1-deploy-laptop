from setuptools import setup

package_name = 'k1_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=[],
    py_modules=[],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/sim_gazebo.launch.py',
            'launch/sim_isaac.launch.py',
            'launch/real.launch.py',
            'launch/fleet.launch.py',
        ]),
    ],
    install_requires=['setuptools'],
    author='thakk100',
    author_email='th.dhyan.us@gmail.com',
    maintainer='thakk100',
    maintainer_email='th.dhyan.us@gmail.com',
    url='https://github.com/booster-robotics/booster_ws',
    download_url='',
    keywords=['ROS2', 'K1', 'humanoid', 'robot'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.12',
    ],
    description='Top-level launch entry points for K1 humanoid robot',
    long_description='Launch bringup for K1: gazebo/isaac simulation, real robot, and multi-robot fleet',
    license='Apache-2.0',
)
