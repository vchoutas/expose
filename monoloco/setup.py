from setuptools import setup

# extract version from __init__.py
with open('monoloco/__init__.py', 'r') as f:
    VERSION_LINE = [l for l in f if l.startswith('__version__')][0]
    VERSION = VERSION_LINE.split('=')[1].strip()[1:-1]

setup(
    name='monoloco',
    version=VERSION,
    packages=[
        'monoloco',
        'monoloco.network',
        'monoloco.eval',
        'monoloco.train',
        'monoloco.prep',
        'monoloco.visuals',
        'monoloco.utils'
    ],
    license='GNU AGPLv3',
    description='MonoLoco: Monocular 3D Pedestrian Localization and Uncertainty Estimation',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Lorenzo Bertoni',
    author_email='lorenzo.bertoni@epfl.ch',
    url='https://github.com/vita-epfl/monoloco',
    zip_safe=False,

    install_requires=[
        'Pillow<=6.3',
        'tabulate<=0.8.3',   # For evaluation
    ],
    extras_require={
        'test': [
            'pylint<=2.4.2',
            'pytest<=4.6.3',
        ],
        'prep': [
            'nuscenes-devkit<=1.0.2',
        ],
    },
)
