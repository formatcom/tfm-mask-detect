import uos
import lcd
import sensor
import image
import utime
import machine
import time

import Maix
import KPU as kpu
import ulab as np

from Maix import GPIO
from board import board_info
from fpioa_manager import fm
from modules import amg88xx

class Detector(object):

    LED_WARNING = 0
    LED_INFO    = 1

    SET_LED_ON  = 0
    SET_LED_OFF = 1

    GC_HEAD_SIZE = 0x180000 # 1.5 MB

    YOLOv2_ANCHOR = (
                        11.89594595, 13.20592966,
                         6.22881356,      10.556,
                         4.26829268,  5.84536082,
                        10.56704981,  9.68973747,
                         8.10810811, 12.69041667,)

    YOLOv2_N_ANCHOR  = 5
    YOLOv2_THRESHOLD = 0.5
    YOLOv2_NMS       = 0.3

    YOLOv2_RATIO     = 240/224

    ZOOM = 2
    CELL = 30 # 240/8   lcd.width / amg88xx width

    TEMPERATURE_N_SAMPLES = 6
    TEMPERATURE_RESOLUTION = [
                                0.25, # ℃
                                0.29, # ℃
    ]

    # temperature distance calibration
    TEMPERATURE_DISTANCE_OFFSET =  4.5 # ℃
    TEMPERATURE_DISTANCE_BASE   =  160
    TEMPERATURE_DISTANCE_REF    =   80

    STATE_INIT         = 0
    STATE_DETECT       = 1
    STATE_MENU         = 2
    STATE_CALIBRATION  = 3
    STATE_PAUSE        = 4
    STATE_INDEX        = [0, 1, 2, 3, 4]

    def _set_enviroment(self):
        if Maix.utils.gc_heap_size() != self.GC_HEAD_SIZE:
            Maix.utils.gc_heap_size(self.GC_HEAD_SIZE)
            machine.reset()

    def __init__(self,
                        model=0x300000,
                        cam_off_x=10,
                        cam_off_y=-5):

        print('\n')
        print('init')

        kpu.memtest()
        print()

        self._set_enviroment()

        self.state = self.STATE_INIT

        fm.fpioa.set_function(board_info.GROVE1, fm.fpioa.I2C0_SCLK)
        fm.fpioa.set_function(board_info.GROVE2, fm.fpioa.I2C0_SDA)

        fm.register(board_info.LED_R, fm.fpioa.GPIO0)
        fm.register(board_info.LED_G, fm.fpioa.GPIO1)

        fm.register(board_info.KEY2,     fm.fpioa.GPIOHS0)
        fm.register(board_info.KEY1,     fm.fpioa.GPIOHS1)
        fm.register(board_info.BOOT_KEY, fm.fpioa.GPIOHS2)

        self.key_left   = GPIO(GPIO.GPIOHS0, GPIO.IN)
        self.key_select = GPIO(GPIO.GPIOHS1, GPIO.IN)
        self.key_right  = GPIO(GPIO.GPIOHS2, GPIO.IN)

        sensor.reset(dual_buff=True)
        sensor.set_pixformat(sensor.RGB565)
        sensor.set_framesize(sensor.B240X240)
        sensor.set_windowing((240, 240))
        sensor.set_vflip(1)
        sensor.set_hmirror(1)

        self.dev = amg88xx(0)
        self.camera = sensor

        lcd.init()

        self.cam_off_x = cam_off_x
        self.cam_off_y = cam_off_y

        self._ir_calc_index_x = ( (self.CELL * self.ZOOM) - cam_off_x ) / self.CELL
        self._ir_calc_index_y = ( (self.CELL * self.ZOOM) - cam_off_y ) / self.CELL

        # init leds
        self.led = [
                GPIO(GPIO.GPIO0, GPIO.OUT),
                GPIO(GPIO.GPIO1, GPIO.OUT),
        ]

        self.led_off()

        print()
        # test memory
        kpu.memtest()

        print()
        print('create buffers')

        # display screen front/back
        self.canvas      = image.Image()
        self.back        = image.Image()
        self._capture    = image.Image() # for take photo
        self._capture_ui = image.Image()

        self.ibuf = image.Image()

        # test memory
        kpu.memtest()
        print()
        print('load kmodel')

        # init neural network processor
        self.task = kpu.load(model)

        # info memory
        kpu.memtest()
        print()

        kpu.init_yolo2(self.task,
                    self.YOLOv2_THRESHOLD,
                    self.YOLOv2_NMS,
                    self.YOLOv2_N_ANCHOR,
                    self.YOLOv2_ANCHOR)


        # map state
        self._run = [
                        self.step,                   # STATE_INIT
                        self.section_detect,         # STATE_DETECT
                        lambda : None,               # STATE_MENU
                        self.section_calibration,    # STATE_CALIBRATION
                        self.section_pause,          # STATE_PAUSE
                        ]




        # UI CALIBRATION EXAMPLE
        #
        #  [ 1, 0, 0, 0, 0]   + 00.00 ℃
        #  [ 1, 1, 2, 0, 5]   + 12.05 ℃
        #  [-1, 0, 2, 0, 0]   - 02.00 ℃

        self._calibration = [1, 0, 0, 0, 0]
        self._calibration_index = 0
        self._calibration_select = -1

        # load calibration
        self.calibration = 0

        try:
            f = open('/flash/ir_calibration', 'r')
            _txt_calibration = f.read()
            f.close()

            self.calibration = float(_txt_calibration)

            self._calibration[0] = -1 if _txt_calibration[0] == '-' else 1
            self._calibration[1] =   int(_txt_calibration[1])
            self._calibration[2] =   int(_txt_calibration[2])
            self._calibration[3] =   int(_txt_calibration[4])
            self._calibration[4] =   int(_txt_calibration[5])

        except:
            pass

        '''
        table mode

        mode  0.25   0.29   calibration   dist off
        -------------------------------------------
        1| 0     x
        2| 1            x
        -------------------------------------------
        3| 2     x               x
        4| 3     x                               x
        5| 4     x               x               x
        -------------------------------------------
        6| 5            x        x
        7| 6            x                        x
        8| 7            x        x               x
        -------------------------------------------
        '''

        self.mode = 2

        print('ready')

    def led_off(self):
        for l in self.led:
            l.value(self.SET_LED_OFF)

    def snapshot(self):
        snapshot = self.camera.snapshot()

        return snapshot

    def get_ide_framebuffer(self):
        return self.snapshot()

    def detect_face(self, img):
        return kpu.run_yolo2(self.task, img)

    def temperature_get_addr(self, point):
        x = int((point[0] / self.CELL ) + self._ir_calc_index_x) - 1
        y = int((point[1] / self.CELL ) + self._ir_calc_index_y) - 1

        return (x, y)

    def detect_temperature(self, with_buff=False):

        temperature = np.zeros((8, 8)) + np.array(self.dev.temperature()).reshape((8, 8))

        '''
        for _ in range(self.TEMPERATURE_N_SAMPLES):
            temperature = temperature + np.array(self.dev.temperature()).reshape((8, 8))

        temperature = temperature/self.TEMPERATURE_N_SAMPLES
        '''

        '''
        flipped horizontally
        > np.flip(temperature, axis=1)

        flipped vertically
        > np.flip(temperature, axis=0)

        flipped horizontally+vertically
        > np.flip(temperature)
        '''

        temperature = np.flip(temperature)

        if not with_buff:
            return temperature, None

        mn, mx, _, _ = self.dev.min_max()

        buf = self.dev.to_image(mn, mx, self.CELL, self.dev.METHOD_NEAREST)
        buf = buf.replace(buf, hmirror=True, vflip=True)

        buf = buf.rotation_corr(
                x_translation=self.cam_off_x, y_translation=self.cam_off_y, zoom=self.ZOOM)

        buf = buf.to_rainbow(1)

        return temperature, buf

    def section_calibration(self):

        if self.key_select.value() == 0:
            while(self.key_select.value() == 0):
                pass

            # change calibration sign
            if self._calibration_index == 0:
                self._calibration[0] *= -1

            # select digit
            elif self._calibration_index < 5 and self._calibration_select == -1:
                self._calibration_select = self._calibration_index

            # unselect digit
            elif self._calibration_select > -1:
                self._calibration_select = -1

            # save calibration
            elif self._calibration_index == 5:

                _txt_calibration = '{}{}{}.{}{}'.format(
                                        '-' if self._calibration[0] < 0 else '+',
                                                                    *self._calibration[1:])
                self.calibration = float(_txt_calibration)

                try:
                    f = open('/flash/ir_calibration', 'w')
                    f.write(_txt_calibration)
                    f.close()
                except:
                    pass

                self.state = self.STATE_DETECT

            utime.sleep_ms(150)


        elif self.key_left.value() == 0:
            while(self.key_left.value() == 0):
                pass

            # change select digit
            if self._calibration_select == -1 and self._calibration_index > 0:
                self._calibration_index -= 1

            # change digit val
            if self._calibration_select > 0 and self._calibration_select < 5:
                self._calibration[self._calibration_select] -= 1
                if self._calibration[self._calibration_select] < 0:
                    self._calibration[self._calibration_select] = 9

            utime.sleep_ms(150)

        elif self.key_right.value() == 0:
            while(self.key_right.value() == 0):
                pass

            # change select digit
            if self._calibration_select == -1 and self._calibration_index < 5:
                self._calibration_index += 1

            # change digit val
            if self._calibration_select > 0 and self._calibration_select < 5:
                self._calibration[self._calibration_select] = \
                                    ( self._calibration[self._calibration_select] + 1 ) \
                                    % 10

            utime.sleep_ms(150)

        self.back.clear()

        for i, v in enumerate(self._calibration):
            if i == 0:
                v = '-' if v < 0 else '+'

            if   self._calibration_select == i:
                c = (0xFF, 0xFF, 0x00)
            elif self._calibration_index == i:
                c = (0x00, 0xFF, 0x00)
            else:
                c = (0xFF, 0xFF, 0xFF)

            self.back.draw_string((i * 30) + 40, (lcd.height()//2) - 30, str(v),
                                                            color=c, scale=4)

            if i == 2:
                self.back.draw_string((i * 30) + 60, (lcd.height()//2) - 4, ',',
                                                        color=(0xFF, 0xFF, 0xFF), scale=2)

        if self._calibration_index == 5:
            c = (0x00, 0xFF, 0x00)
        else:
            c = (0xFF, 0xFF, 0xFF)

        self.back.draw_string(
                                lcd.width() - 40,
                                (lcd.height()//2) - 10,
                                'OK',
                                color=c, scale=2)

        buf = self.get_ide_framebuffer() # only for diplay in IDE
        buf.draw_image(self.back, 0, 0)

    def section_detect(self):
        if self.key_select.value() == 0:
            ti = utime.ticks_ms()
            while(self.key_select.value() == 0):
                pass
            tf = utime.ticks_ms()

            # open menu calibration
            if (tf - ti) > 1000:
                self.state = self.STATE_CALIBRATION

            # take photo
            else:
                self.state = self.STATE_PAUSE

            self._calibration_select = -1
            self._calibration_index  =  0

            utime.sleep_ms(150)
            return None

        # change mode resolution
        elif self.key_left.value() == 0:
            while(self.key_right.value() == 0):
                pass
            utime.sleep_ms(150)
            self.mode -= 1
            if self.mode < 0:
                self.mode = 7

        elif self.key_right.value() == 0:
            while(self.key_right.value() == 0):
                pass
            utime.sleep_ms(150)
            self.mode = (self.mode + 1) % 8

        snap = self.snapshot()

        self._capture.draw_image(snap, 0, 0)

        # input IA 224 x 224
        self.ibuf.draw_image(snap, 0, 0)

        _ibuf = self.ibuf.resize(224, 224)
        _ibuf.pix_to_ai()

        scope = self.detect_face(_ibuf)

        self.led_off()
        if scope:

            warning = False
            for item in scope:

                face = [round(v * self.YOLOv2_RATIO) for v in item.rect()]

                cid = item.classid()
                val = item.value()

                if val < 0.45:
                    continue

                p = [face[0],                  # x
                     face[1],                  # y
                     face[2],                  # w
                     face[3],]                 # h

                pir = [
                    p[0] + 10 + (face[2] // 2),
                    p[1] + 45,
                    8, 8,
                ]

                temp, _ = self.detect_temperature(with_buff=False)

                for i in range(self.TEMPERATURE_N_SAMPLES-1):
                    _temp, _ = self.detect_temperature(with_buff=False)
                    temp = temp + _temp

                temp = temp / self.TEMPERATURE_N_SAMPLES

                addr = self.temperature_get_addr(pir)

                # draw point for ir
                # snap.draw_rectangle(pir, color=(0xFF, 0xFF, 0xFF), fill=True)

                try:

                    resolution = self.TEMPERATURE_RESOLUTION[0]
                    if self.mode not in [0, 2, 3, 4]:
                        resolution = self.TEMPERATURE_RESOLUTION[1]

                    val = temp[addr[0], addr[1]][0] * resolution

                    face_h = p[3] if p[3] < self.TEMPERATURE_DISTANCE_BASE else \
                                                            self.TEMPERATURE_DISTANCE_BASE

                    off = (self.TEMPERATURE_DISTANCE_BASE-face_h)/self.TEMPERATURE_DISTANCE_REF * \
                                        self.TEMPERATURE_DISTANCE_OFFSET

                    if self.mode not in [3, 4, 6, 7]:
                        off = 0

                    calibration = self.calibration
                    if self.mode not in [2, 4, 5, 7]:
                        calibration = 0

                    val = val + off + calibration

                    # draw face and detect fever
                    if cid == 1 or val > 38:
                        warning = True
                        snap.draw_rectangle(p, color=(0xFF, 0, 0), thickness=3)
                    else:
                        self.led[self.LED_INFO].value(self.SET_LED_ON)
                        snap.draw_rectangle(p, color=(0, 0xFF, 0), thickness=3)

                    # draw info
                    snap.draw_rectangle(p[0], p[1] + p[3] - 20, 115, 15, color=(0, 0, 0), fill=True)
                    snap.draw_string(p[0] + 4, p[1] + p[3] - 20,
                                        '{:02.2f} C  {:02.2f}  {:04}'.format(val, off, face_h),
                                        color=(0xFF, 0xFF, 0xFF), scale=1.2)
                except:
                    pass

            if warning:
                self.led_off()
                self.led[self.LED_WARNING].value(self.SET_LED_ON)

        # draw mode resolution ir
        snap.draw_rectangle(0, 0, 20, 20, color=0, fill=True)
        snap.draw_string(10, 3, str(self.mode + 1), color=(0xFF, 0xFF, 0xFF))

        # save last capture
        self._capture_ui.draw_image(snap, 0, 0)

        self.back.draw_image(snap, 0, 0)

    def section_pause(self):

        # confirm save or exit
        if self.key_select.value() == 0:
            while(self.key_select.value() == 0):
                pass

            if self._calibration_select == 1:
                self._calibration_select = 2
            else:
                self.state = self.STATE_DETECT

        # select exit
        elif self.key_left.value() == 0:
            while(self.key_left.value() == 0):
                pass
            self._calibration_select = -1

        # select save
        elif self.key_right.value() == 0:
            while(self.key_right.value() == 0):
                pass
            self._calibration_select = 1


        # save photo process
        if self._calibration_select == 3:
            for i in range(2):
                try:
                    n = len(uos.listdir('/sd'))
                    self._capture.save('/sd/{}-photo.jpg'.format(n))
                    self._capture_ui.save('/sd/{}-detect.jpg'.format(n))

                    # success
                    self.state = self.STATE_DETECT
                    break
                except:
                    if i == 1: # err
                        self._calibration_select = 4
                    else:
                        machine.SDCard.remount()

        t = 'exit'
        ct = (0xFF, 0xFF, 0xFF)
        c = (0, 0, 0)
        off = 0

        # save: code 1
        if self._calibration_select == 1:
            t = 'save'

        elif self._calibration_select in [2, 3]:
            t = 'writing'
            off = -10
            self._calibration_select = 3

        # no found sd: code 2
        elif self._calibration_select == 4:
            t = 'err: sd no found'
            c = (0xFF, 0, 0)
            off = -60

        self.back.draw_rectangle(0, 0, lcd.width(), 30, color=c, fill=True)
        self.back.draw_string(100+off, 3, t, color=ct, scale=2)

        buf = self.get_ide_framebuffer() # only for diplay in IDE
        buf.draw_image(self.back, 0, 0)

    def step(self):
        if self.state == self.STATE_INIT or self.state not in self.STATE_INDEX:
            self.state = self.STATE_DETECT

        self._run[self.state]()

    def flip(self):
        self.canvas.draw_image(self.back, 0, 0)
        lcd.display(self.canvas)


# 0x200000 MobileNet7_5   1.9M
# 0x400000 TinyYolo       2.2M

o = Detector(model=0x200000)

clock = time.clock()
while True:
    clock.tick()
    try:
        o.step()

        # draw FPS
        o.back.draw_string(210, 3, '{:02}'.format(round(clock.fps())),
                                                color=(0xFF, 0, 0), scale=2)

        buf = o.get_ide_framebuffer() # only for diplay in IDE
        buf.draw_image(o.back, 0, 0)
        del buf

        o.flip()
    except Exception as e:
        print('INFO', e)
        kpu.memtest()
