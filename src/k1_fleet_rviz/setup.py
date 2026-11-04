from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'k1_fleet_rviz'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.rviz') + glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='thakk100',
    maintainer_email='thakk100@umn.edu',
    description='Fleet RViz: 18 camera streams + TF for 6x K1',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'domain_relay = k1_fleet_rviz.domain_relay:main',
        ],
    },
)
