from glob import glob

from setuptools import setup, find_packages

package_name = 'k1_locomotion'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/k1_locomotion']),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Booster K1',
    author_email='th.dhyan.us@gmail.com',
    description='ROS2 locomotion policy node for Booster K1',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'locomotion_node = k1_locomotion.locomotion_node:main',
        ],
    },
)
