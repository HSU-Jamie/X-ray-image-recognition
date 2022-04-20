from configparser import ConfigParser
from pprint import pprint
config_path = './config_ini.ini'
config = {}
f = ConfigParser()
f.read(config_path)
for section in f.sections():
    config[section] = {}
    for option in f.options(section):
        config[section][option] = f.get(section, option)
#pprint(config)
pprint(config['FrameCapture']['capture0_state'])
