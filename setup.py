from setuptools import setup

setup(
    name='textgenlstm',
    url='https://github.com/SIDI3128/textgenlstm',
    author='NUS CS5242 final project',
    author_email='martinautier@hotmail.fr',
    packages=['textgenlstm'],
    install_requires=['numpy', 'pandas', 'keras.models', 'keras.layers'],
    version='0.1',
    license='NUS',
    description='text generator using rnn networks',
)