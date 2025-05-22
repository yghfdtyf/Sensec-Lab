

# import RPi.GPIO as GPIO
import time

# 初始化GPIO
# GPIO.setwarnings(False)   # 不打印 warning 信息
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(20, GPIO.OUT)
# GPIO.setup(21, GPIO.OUT)
# GPIO.output(20, 1)        # 关闭夹爪电磁阀

def pump_on():
    '''
    开启夹爪
    '''
    print('    开启夹爪')
    # GPIO.output(20, 0)

def pump_off():
    '''
    关闭夹爪，夹爪放气，释放物体
    '''
    print('    关闭夹爪')
    # GPIO.output(20, 1)   # 关闭夹爪电磁阀
    # time.sleep(0.05)
    # GPIO.output(21, 0)   # 打开泄气阀门
    # time.sleep(0.2)
    # GPIO.output(21, 1)
    # time.sleep(0.05)
    # GPIO.output(21, 0)   # 再一次泄气，确保物体释放
    # time.sleep(0.2)
    # GPIO.output(21, 1)
    # time.sleep(0.05)