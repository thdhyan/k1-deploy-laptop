from setuptools import setup, find_packages

package_name = 'k1_wbc'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/k1_wbc']),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Booster K1',
    author_email='th.dhyan.us@gmail.com',
    description='ROS2 whole-body controller for Booster K1',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'wbc_node = k1_wbc.wbc_node:main',
        ],
    },
)
