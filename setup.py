# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['crying_vegetable', 'crying_vegetable.mattoolbox']

package_data = \
{'': ['*']}

install_requires = \
['Pillow==9.0.0',
 'SciencePlots>=1.0.9,<2.0.0',
 'SoundFile>=0.10.3,<0.11.0',
 'fire>=0.4.0,<0.5.0',
 'ipython>=8.0.1,<9.0.0',
 'japanize-matplotlib>=1.1.3,<2.0.0',
 'matplotlib>=3.5.1,<4.0.0',
 'numpy>=1.22.2,<2.0.0',
 'pandas>=1.4.0,<2.0.0',
 'scipy>=1.8.0,<2.0.0']

entry_points = \
{'console_scripts': ['crying_vegetable_jikken = '
                     'crying_vegetable.crying_vegetable_jikken:main']}

setup_kwargs = {
    'name': 'crying-vegetable',
    'version': '0.2.2',
    'description': 'create vocoded vegetable cry sound from time variant NDI values',
    'author': 'Kotaro SONODA',
    'author_email': 'kotaro@nagasaki-u.ac.jp',
    'maintainer': None,
    'maintainer_email': None,
    'url': None,
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'entry_points': entry_points,
    'python_requires': '>=3.10,<3.11',
}


setup(**setup_kwargs)