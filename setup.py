from setuptools import setup

setup(name='gym_malware',
      version='0.0.1',
      install_requires=['gym','numpy','sklearn','requests','keras-rl']  # And any other dependencies the package needs
)  

# note: must install https://github.com/lief-project/LIEF/releases/download/0.7.0/linux_lief-0.7.0_py3.6.tar.gz [modify for your platform] manually
# pip install https://github.com/lief-project/LIEF/releases/download/0.7.0/linux_lief-0.7.0_py3.6.tar.gz [modify for your platform]

