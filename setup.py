from setuptools import find_packages
from setuptools import setup

VERSION = '1.0.0'

REQUIRED_PACKAGES = [
    'numpy',
    'Image',
    'pypng',
    'matplotlib',
    'scipy',
]

if __name__ == '__main__':
    setup(
        name='flownet2',
        version=VERSION,
        description='FlowNet2 for tensorflow',
        url='https://github.com/LoSealL/flownet2-tf',
        packages=find_packages(),
        install_requires=REQUIRED_PACKAGES,
        license='MIT',
        author='sampepose, Wenyi Tang',
        author_email='wenyitang@outlook.com'
    )
