from setuptools import setup
from glob import glob
import os

package_name = 'brain'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        # REQUIRED: register with ament index
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),

        # install package.xml
        ('share/' + package_name, ['package.xml']),

        # install launch files if they exist
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Eugen Hulub',
    maintainer_email='root@todo.todo',
    description='Python MPC brain node for BMW GTR.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main_brain = brain.main_brain:main',
        ],
    },
)
