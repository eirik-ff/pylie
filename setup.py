from setuptools import setup

setup(
    name='pylie',
    version='0.5.0',
    description='A small Lie library for Python',
    author='Trym Vegard Haavardsholm',
    license='BSD-3-Clause',
    packages=['pylie', 'pylie.util'],
    install_requires=['numpy']
)
