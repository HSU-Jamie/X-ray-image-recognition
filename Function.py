import os
import time
import cv2
import numpy as np


class Config:
    def __init__(self, capture0='', capture1='', LtoR=0, bottom_limit=0, thr_line=0, showpic=0, video_col=0,
                 video_row=0, maxVal_thr=0,
                 min_dist=0, capture_min_time=0, folder_path='', duration=60, xraytype=0):  # 宣告的config相對應的變數
        self.capture0 = capture0
        self.capture1 = capture1
        self.LtoR = LtoR
        self.folder_path = folder_path
        self.duration = duration
        self.xraytype = xraytype
        self.bottom_limit = bottom_limit
        self.thr_line = thr_line
        self.showpic = showpic
        self.video_col = video_col
        self.video_row = video_row
        self.maxVal_thr = maxVal_thr
        self.min_dist = min_dist
        self.capture_min_time = capture_min_time

    def load_config(self):  # 將config檔案讀取並放到相對應的變數
        path = './config.txt'
        try:
            with open(path, 'r') as f:
                for line in f.readlines():
                    temp = line.strip().split('=')  # 去掉空白並以=分割
                    if temp[0].find('LtoR') != -1:
                        self.LtoR = eval(temp[1].strip())
                    elif temp[0].find('capture0') != -1:
                        self.capture0 = temp[1].strip()
                    elif temp[0].find('capture1') != -1:
                        self.capture1 = temp[1].strip()
                    elif temp[0].find('folder_path') != -1:
                        self.folder_path = temp[1]
                    elif temp[0].find('duration') != -1:
                        self.duration = eval(temp[1])
                    elif temp[0].find('xraytype') != -1:
                        self.xraytype = eval(temp[1])
                    elif temp[0].find('bottom_limit') != -1:
                        self.bottom_limit = eval(temp[1])
                    elif temp[0].find('thr_line') != -1:
                        self.thr_line = eval(temp[1])
                    elif temp[0].find('showpic') != -1:
                        self.showpic = eval(temp[1])
                    elif temp[0].find('video_col') != -1:
                        self.video_col = eval(temp[1])
                    elif temp[0].find('video_row') != -1:
                        self.video_row = eval(temp[1])
                    elif temp[0].find('maxVal_thr') != -1:
                        self.maxVal_thr = eval(temp[1])
                    elif temp[0].find('min_dist') != -1:
                        self.min_dist = eval(temp[1])
                    elif temp[0].find('capture_min_time') != -1:
                        self.capture_min_time = eval(temp[1])
        except:
            print('load config error')  # 讀取失敗


def open_camera(capture0, capture1):  # 讀取鏡頭並回傳給主程式
    if capture0 == '0':
        cam0 = cv2.VideoCapture(0)
    elif capture0 != '0' or capture0 != '1':
        cam0 = cv2.VideoCapture(capture0)
    if capture1 == '1':
        cam1 = cv2.VideoCapture(1)
    elif capture0 != '0' or capture0 != '1':
        cam1 = cv2.VideoCapture(capture1)
    if not cam0.isOpened():
        print('capture0 read video Failed !\n')
        return
    if not cam1.isOpened():
        print('capture1 read video Failed !\n')
        return
    return cam0, cam1


def set_window(cam0, cam1, showpic, video_col, video_row):  # 設定視窗大小
    if showpic != 0:
        cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('frame', video_col // 2, video_row // 2)
        cv2.namedWindow('bin', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('bin', video_col // 2, video_row // 2)
    cam0.set(cv2.CAP_PROP_FRAME_WIDTH, video_col)
    cam0.set(cv2.CAP_PROP_FRAME_HEIGHT, video_row)
    cam1.set(cv2.CAP_PROP_FRAME_WIDTH, video_col)
    cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, video_row)
    return cam0, cam1


def get_accument(binary, thr_line, mode):  # mode為LtoR
    s = np.full([binary.shape[1]], 9999)  # 初始陣列
    if not mode:
        for i in range(thr_line, binary.shape[1], 1):
            s[i] = sum(binary[:, i]) / 255  # col加總
        index_col = np.argmin(s)  # 找出最小索引值
    else:
        for i in range(thr_line, 0, -1):
            s[i] = sum(binary[:, i]) / 255  # col加總
        index_col = np.argmin(s)  # 找出最小索引值
    return index_col


def take_picture(*args):  # [pic1, path] or [pic1, pic2, path]
    str_filedate = time.strftime('%Y%m%d%H%M-%S', time.localtime())
    if args[-1] == '':
        path = 'C:/frameCapture2/pic'
    else:
        path = args[-1]
    if not os.path.exists(path):
        os.mkdir(path)
    if len(args) == 2:  # 單射源
        picfilename = str_filedate + '.jpg'
        cv2.imwrite(path + '/' + picfilename, args[0])

    elif len(args) == 3:  # 雙射源
        picfilename1 = str_filedate + '_A.jpg'
        picfilename2 = str_filedate + '_B.jpg'
        cv2.imwrite(path + '/' + picfilename1, args[0])
        cv2.imwrite(path + '/' + picfilename2, args[1])

    else:  # 進來的東西是錯的
        print('take_picture error')


def start_work(config, cam0, cam1, start_time):
    frame_num = 0
    input_target = np.array([])

    if config.xraytype:  # 雙射源
        while True:
            ret0, frame0 = cam0.read()
            ret1, frame1 = cam1.read()
            if not (ret0 or ret1):
                print('camera error\n')
                break
            # 去掉下方Bar
            frame0 = frame0[0:config.bottom_limit, 0:config.video_col]
            frame1 = frame1[0:config.bottom_limit, 0:config.video_col]
            # 指定禎數
            if (frame_num % config.duration) == 0:
                roi_start = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
                origin0 = frame0.copy()
                origin1 = frame1.copy()

                if config.showpic:
                    cv2.line(frame0, (config.thr_line, 0), (config.thr_line, config.bottom_limit), (255, 0, 0), 4)
                # 轉灰階做二值化
                gray = cv2.cvtColor(origin0, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

                if not config.LtoR:  # LtoR = 0 物件右到左

                    if not input_target.size:  # 倘使無初始相片可供matchTemplate比對，則自行找
                        col = get_accument(binary, config.thr_line, config.LtoR)

                        if config.showpic:
                            cv2.line(frame0, (col, 0), (col, config.bottom_limit), (0, 0, 255), 4)

                        res0 = origin0[0:config.bottom_limit, config.thr_line:col]

                        input_target = origin0[0:config.bottom_limit, col - 100: col]
                        input_target = cv2.cvtColor(input_target, cv2.COLOR_BGR2GRAY)
                    #比對input_target 在 gray的位置
                    result = cv2.matchTemplate(gray, input_target, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, loc = cv2.minMaxLoc(result)


                    if max_val == 1:
                        if config.showpic:
                            print('max_val = 1, match error\n')
                        max_val = 0

                    if max_val > config.maxVal_thr: # 根據比對結果框選邊框，並決定擷取條件。
                        cv2.rectangle(frame0, loc, (loc[0] + input_target.shape[1], loc[1] + input_target.shape[0]),
                                      (0, 255, 0), 4)

                        if (config.video_col - (loc[0] + input_target.shape[1])) > config.min_dist:  # 大於指定寬度開始搜尋
                            temp = sum(binary[:, config.video_col - 10]) / 255
                            if temp < 10:
                                capture_time = time.time() - start_time
                                if capture_time < config.capture_min_time:
                                    print('截圖費時:' + '{:.2f}'.format(capture_time) + 'time too short so pass\n')
                                else:
                                    print('截圖費時: ', '{:.2f}\n'.format(capture_time))
                                    start_time = time.time()
                                    if config.showpic:
                                        print('找到最低點截圖\n')
                                    res0 = origin0[0:config.bottom_limit,
                                           (loc[0] + input_target.shape[1]):config.video_col]
                                    res1 = origin1[0:config.bottom_limit,
                                           (loc[0] + input_target.shape[1]):config.video_col]
                                    take_picture(res0, res1, config.folder_path)
                                    input_target = origin0[0:config.bottom_limit,
                                                   config.video_col - 100:config.video_col]
                                    input_target = cv2.cvtColor(input_target, cv2.COLOR_BGR2GRAY)

                        if (loc[0] + input_target.shape[1]) <= config.thr_line: # 如果有過長的寬度則截圖
                            capture_time = time.time() - start_time
                            if capture_time < config.capture_min_time:
                                print('截圖費時:' + '{:.2f}'.format(capture_time) + 'time too short so pass\n')
                            else:
                                print('截圖費時: ', '{:.2f}\n'.format(capture_time))
                                start_time = time.time()
                                if config.showpic:
                                    print('未找到最低點，但物件過長向後尋找可能的物件邊界並截圖\n')
                                col = get_accument(binary, loc[0] + input_target.shape[1] + config.min_dist,

                                                   config.LtoR)
                                res0 = origin0[0:config.bottom_limit, loc[0] + input_target.shape[1]:col]
                                res1 = origin1[0:config.bottom_limit, loc[0] + input_target.shape[1]:col]
                                take_picture(res0, res1, config.folder_path)
                                input_target = origin0[0:config.bottom_limit, (col - 100):col]
                                input_target = cv2.cvtColor(input_target, cv2.COLOR_BGR2GRAY)

                    else:  # 遺失邊框 尋找新框
                        col = get_accument(binary, config.thr_line, config.LtoR)
                        if config.showpic:
                            print('遺失邊框 找新框\n')
                            cv2.rectangle(frame0, (col - 100, 0), (col, config.bottom_limit), (0, 0, 255), 4)
                        input_target = origin0[0:config.bottom_limit, col - 100:col]
                        input_target = cv2.cvtColor(input_target, cv2.COLOR_BGR2GRAY)

                else:  # LtoR = 1 物件由左到右
                    if not input_target.size:  # 倘使無初始相片可供matchTemplate比對，則自行找
                        col = get_accument(binary, config.thr_line, config.LtoR)

                        if config.showpic:
                            cv2.line(frame0, (col, 0), (col, config.bottom_limit), (0, 0, 255), 4)

                        res0 = origin0[0:config.bottom_limit, col:config.thr_line]

                        input_target = origin0[0:config.bottom_limit, col:col + 100]
                        input_target = cv2.cvtColor(input_target, cv2.COLOR_BGR2GRAY)
                    # 比對input_target 在 gray的位置
                    result = cv2.matchTemplate(gray, input_target, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, loc = cv2.minMaxLoc(result)

                    if max_val == 1:
                        if config.showpic:
                            print('max_val = 1, match error\n')
                        max_val = 0

                    if max_val > config.maxVal_thr: # 根據比對結果框選邊框，並決定擷取條件。
                        cv2.rectangle(frame0, loc, (loc[0] + input_target.shape[1], loc[1] + input_target.shape[0]),
                                      (0, 255, 0), 4)

                        if loc[0] > config.min_dist:  # 大於指定寬度開始搜尋
                            temp = sum(binary[:, 10]) / 255
                            if temp < 10:
                                capture_time = time.time() - start_time
                                if capture_time < config.capture_min_time:
                                    print('截圖費時:' + '{:.2f}'.format(capture_time) + 'time too short so pass\n')
                                else:
                                    print('截圖費時: ', '{:.2f}\n'.format(capture_time))
                                    start_time = time.time()
                                    if config.showpic:
                                        print('找到最低點截圖\n')
                                    res0 = origin0[0:config.bottom_limit, 0:loc[0]]
                                    res1 = origin1[0:config.bottom_limit, 0:loc[0]]
                                    take_picture(res0, res1, config.folder_path)
                                    input_target = origin0[0:config.bottom_limit, 0:100]
                                    input_target = cv2.cvtColor(input_target, cv2.COLOR_BGR2GRAY)

                        if loc[0] > config.thr_line: # 如果有過長的寬度則截圖
                            capture_time = time.time() - start_time
                            if capture_time < config.capture_min_time:
                                print('截圖費時:' + '{:.2f}'.format(capture_time) + 'time too short so pass\n')
                            else:
                                print('截圖費時: ', '{:.2f}\n'.format(capture_time))
                                start_time = time.time()
                                if config.showpic:
                                    print('未找到最低點，但物件過長向後尋找可能的物件邊界並截圖\n')
                                col = get_accument(binary, loc[0] - config.min_dist, config.LtoR)
                                res0 = origin0[0:config.bottom_limit, col:loc[0]]
                                res1 = origin1[0:config.bottom_limit, col:loc[0]]
                                take_picture(res0, res1, config.folder_path)
                                input_target = origin0[0:config.bottom_limit, col:col + 100]
                                input_target = cv2.cvtColor(input_target, cv2.COLOR_BGR2GRAY)

                    else:  # 遺失邊框 尋找新框
                        col = get_accument(binary, config.thr_line, config.LtoR)
                        if config.showpic:
                            print('遺失邊框 找新框\n')
                            cv2.rectangle(frame0, (col, 0), (col + 100, config.bottom_limit), (0, 0, 255), 4)
                        input_target = origin0[0:config.bottom_limit, col:col + 100]
                        input_target = cv2.cvtColor(input_target, cv2.COLOR_BGR2GRAY)

                if config.showpic:

                    cv2.imshow('res', res0)
                    cv2.imshow('frame', frame0)
                    cv2.imshow('bin', binary)
                    cv2.waitKey(1)

            elif (frame_num % config.duration) == (config.duration - 1):
                roi_end = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
                roi_diff = cv2.absdiff(roi_start, roi_end)
                _, roi_diffbin = cv2.threshold(roi_diff, 100, 255, cv2.THRESH_BINARY)
                mean = cv2.mean(roi_diffbin)
                if mean[0] == 0.0:
                    frame_num = 0
                    if config.showpic:
                        print('畫面沒有移動，不動作\n')

            frame_num += 1

    else:  # 單射源
        while True:
            ret, frame = cam0.read()
            if not ret:
                print('camera error\n')
                break

            frame = frame[0:config.bottom_limit, 0:config.video_col]
            if (frame_num % config.duration) == 0:
                roi_start = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                origin = frame.copy()
                if config.showpic:
                    cv2.line(frame, (config.thr_line, 0), (config.thr_line, config.bottom_limit), (255, 0, 0), 4)

                gray = cv2.cvtColor(origin, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

                if not config.LtoR: #LtoR = 0物件由右至左 <--
                    if not input_target.size:  # 倘使無初始相片可供matchTemplate比對，則自行找
                        col = get_accument(binary, config.thr_line, config.LtoR)

                        if config.showpic:
                            cv2.line(frame, (col, 0), (col, config.bottom_limit), (0, 0, 255), 4)

                        res = origin[0:config.bottom_limit, config.thr_line:col]
                        input_target = origin[0: config.bottom_limit, col - 100:col]
                        input_target = cv2.cvtColor(input_target, cv2.COLOR_BGR2GRAY)

                    result = cv2.matchTemplate(gray, input_target, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, loc = cv2.minMaxLoc(result)

                    if max_val == 1:
                        if config.showpic:
                            print('max_val = 1, match error\n')
                        max_val = 0
                    if max_val > config.maxVal_thr: #根據比對結果框選邊框，並決定擷取條件。
                        cv2.rectangle(frame, loc,
                                      (loc[0] + input_target.shape[1], loc[1] + input_target.shape[0]), (0, 255, 0), 4)

                        if (config.video_col - (loc[0] + input_target.shape[1])) > config.min_dist:
                            temp = sum(binary[:, config.video_col - 10]) / 255
                            if temp < 10:
                                capture_time = time.time() - start_time

                                if capture_time < config.capture_min_time:
                                    print('截圖費時:' + '{:.2f}'.format(capture_time) + 'time too short so pass\n')
                                else:
                                    print('截圖費時: ', '{:.2f}\n'.format(capture_time))
                                    start_time = time.time()
                                    if config.showpic:
                                        print('找到最低點截圖\n')
                                    res = origin[0:config.bottom_limit,
                                          (loc[0] + input_target.shape[1]):config.video_col]
                                    take_picture(res, config.folder_path)
                                    input_target = origin[0:config.bottom_limit,
                                                   (config.video_col - 100):config.video_col]
                                    input_target = cv2.cvtColor(input_target, cv2.COLOR_BGR2GRAY)

                        if (loc[0] + input_target.shape[1]) <= config.thr_line: # 如果有過長的寬度則截圖
                            capture_time = time.time() - start_time
                            if capture_time < config.capture_min_time:
                                print('截圖費時:' + '{:.2f}'.format(capture_time) + 'time too short so pass\n')
                            else:
                                print('截圖費時: ', '{:.2f}\n'.format(capture_time))
                                start_time = time.time()
                                if config.showpic:
                                    print('未找到最低點，但物件過長向後尋找可能的物件邊界並截圖\n')
                                col = get_accument(binary, loc[0] + input_target.shape[1] + config.min_dist,
                                                   config.LtoR)
                                res = origin[0:config.bottom_limit, loc[0] + input_target.shape[1]:col]
                                take_picture(res, config.folder_path)
                                input_target = origin[0:config.bottom_limit, (col - 100):col]
                                input_target = cv2.cvtColor(input_target, cv2.COLOR_BGR2GRAY)

                    else: # 遺失邊框 找新框
                        col = get_accument(binary, config.thr_line, config.LtoR)
                        if config.showpic:
                            print('遺失邊框 找新框\n')
                            cv2.rectangle(frame, (col - 100, 0), (col, config.bottom_limit), (0, 0, 255), 4)
                        input_target = origin[0:config.bottom_limit, col - 100:col]
                        input_target = cv2.cvtColor(input_target, cv2.COLOR_BGR2GRAY)

                else: # LtoR = 1 物件由左至右 -->
                    if not input_target.size:  # 倘使無初始相片可供matchTemplate比對，則自行找
                        col = get_accument(binary, config.thr_line, config.LtoR)

                        if config.showpic:
                            cv2.line(frame, (col, 0), (col, config.bottom_limit), (0, 0, 255), 4)

                        res = origin[0:config.bottom_limit, col:config.thr_line]
                        input_target = origin[0: config.bottom_limit, col:col + 100]
                        input_target = cv2.cvtColor(input_target, cv2.COLOR_BGR2GRAY)

                    result = cv2.matchTemplate(gray, input_target, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, loc = cv2.minMaxLoc(result)

                    if max_val == 1:
                        if config.showpic:
                            print('max_val = 1, match error\n')
                        max_val = 0
                    if max_val > config.maxVal_thr: # 根據比對結果框選邊框，並決定擷取條件。
                        cv2.rectangle(frame, loc,
                                      (loc[0] + input_target.shape[1], loc[1] + input_target.shape[0], (0, 255, 0), 4))

                        if loc[0] > config.min_dist: # 大於一定寬度開始找尋
                            temp = sum(binary[:, 10]) / 255
                            if temp < 10:
                                capture_time = time.time() - start_time

                                if capture_time < config.capture_min_time:
                                    print('截圖費時:' + '{:.2f}'.format(capture_time) + 'time too short so pass\n')
                                else:
                                    print('截圖費時: ', '{:.2f}\n'.format(capture_time))
                                    start_time = time.time()
                                    if config.showpic:
                                        print('找到最低點截圖\n')
                                    res = origin[0:config.bottom_limit, 0:loc[0]]
                                    take_picture(res, config.folder_path)
                                    input_target = origin[0:config.bottom_limit, 0:100]
                                    input_target = cv2.cvtColor(input_target, cv2.COLOR_BGR2GRAY)

                        if loc[0] >= config.thr_line: # 如果有過長的寬度則截圖
                            capture_time = time.time() - start_time
                            if capture_time < config.capture_min_time:
                                print('截圖費時:' + '{:.2f}'.format(capture_time) + 'time too short so pass\n')
                            else:
                                print('截圖費時: ', '{:.2f}\n'.format(capture_time))
                                start_time = time.time()
                                if config.showpic:
                                    print('未找到最低點，但物件過長向後尋找可能的物件邊界並截圖\n')
                                col = get_accument(binary, loc[0] - config.min_dist, config.LtoR)
                                res = origin[0:config.bottom_limit, col:loc[0]]
                                take_picture(res, config.folder_path)
                                input_target = origin[0:config.bottom_limit, col:col + 100]
                                input_target = cv2.cvtColor(input_target, cv2.COLOR_BGR2GRAY)

                    else: # 遺失邊框 找新框
                        col = get_accument(binary, config.thr_line, config.LtoR)
                        if config.showpic:
                            print('遺失邊框 找新框\n')
                            cv2.rectangle(frame, (col + 100, 0), (col, config.bottom_limit), (0, 0, 255), 4)
                        input_target = origin[0:config.bottom_limit, col:col + 100]
                        input_target = cv2.cvtColor(input_target, cv2.COLOR_BGR2GRAY)

                if config.showpic:
                    cv2.imshow('res', res)
                    cv2.imshow('frame', frame)
                    cv2.imshow('bin', binary)
                    cv2.waitKey(1)

            elif (frame_num % config.duration) == (config.duration - 1):
                roi_end = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                roi_diff = cv2.absdiff(roi_start, roi_end)
                _, roi_diffbin = cv2.threshold(roi_diff, 100, 255, cv2.THRESH_BINARY)
                mean = cv2.mean(roi_diffbin)
                if mean[0] == 0.0:
                    frame_num = 0
                    if config.showpic:
                        print('畫面沒有移動，不動作\n')

            frame_num += 1
