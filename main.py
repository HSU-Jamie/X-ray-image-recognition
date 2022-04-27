import Function
import time

if __name__ == '__main__':
    START = time.time()
    config = Function.Config()
    config.load_config('./config_ini.ini')
    config.load_model()
    camera0, camera1 = Function.open_camera(config.capture0, config.capture1)
    camera0, camera1 = Function.set_window(camera0, camera1, config.showpic, config.video_col, config.video_row)
    Function.take_suitcase(config, camera0, camera1, START)
    print('finish\n')
    END = time.time()
    print('total time {} sec'.format(END-START))

