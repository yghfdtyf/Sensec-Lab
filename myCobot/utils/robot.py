from pymycobot.mycobot import MyCobot
from pymycobot import PI_PORT, PI_BAUD
import cv2
import numpy as np
import time
from utils.pump import *

# 连接机械臂
mc = MyCobot("/dev/ttyUSB0", 1000000)
#mc = MyCobot(PI_PORT, PI_BAUD)
# 设置运动模式为插补
#mc.set_fresh_mode(0)

# import RPi.GPIO as GPIO
# # 初始化GPIO
# GPIO.setwarnings(False)   # 不打印 warning 信息
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(20, GPIO.OUT)
# GPIO.setup(21, GPIO.OUT)
# GPIO.output(20, 1)        # 关闭夹爪电磁阀

def back_zero():
    '''
    机械臂归零
    '''
    print('机械臂归零')
    mc.send_angles([0, 0, 0, 0, 0, 0], 40)
    time.sleep(2)

def relax_arms():
    print('放松机械臂关节')
    mc.release_all_servos()

# 左右摆头
def head_shake():
    mc.send_angles([0.87,(-50.44),47.28,0.35,(-0.43),(-0.26)],70)
    time.sleep(1)
    for count in range(2):
        mc.send_angle(5, 30, 80)
        time.sleep(0.5)
        mc.send_angle(5, -30,80)
        time.sleep(0.5)
    # mc.send_angles([0.87,(-50.44),47.28,0.35,(-0.43),(-0.26)],70)
    # time.sleep(1)
    mc.send_angles([0, 0, 0, 0, 0, 0], 40)
    time.sleep(2)

# 跳舞
def head_dance():
    mc.send_angles([0.87,(-50.44),47.28,0.35,(-0.43),(-0.26)],70)
    time.sleep(1)
    for count in range(1):
        mc.send_angles([(-0.17),(-94.3),118.91,(-39.9),59.32,(-0.52)],80)
        time.sleep(1.2)
        mc.send_angles([67.85,(-3.42),(-116.98),106.52,23.11,(-0.52)],80)
        time.sleep(1.7)
        mc.send_angles([(-38.14),(-115.04),116.63,69.69,3.25,(-11.6)],80)
        time.sleep(1.7)
        mc.send_angles([2.72,(-26.19),140.27,(-110.74),(-6.15),(-11.25)],80)
        time.sleep(1)
        mc.send_angles([0,0,0,0,0,0],80)

# 点头
def head_nod():
    mc.send_angles([0.87,(-50.44),47.28,0.35,(-0.43),(-0.26)],70)
    for count in range(2):
        mc.send_angle(4, 13, 70)
        time.sleep(0.5)
        mc.send_angle(4, -20, 70)
        time.sleep(1)
        mc.send_angle(4,13,70)
        time.sleep(0.5)
    mc.send_angles([0.87,(-50.44),47.28,0.35,(-0.43),(-0.26)],70)

def move_to_coords(X=150, Y=-130, HEIGHT_SAFE=230):
    print('移动至指定坐标：X {} Y {}'.format(X, Y))
    mc.send_coords([X, Y, HEIGHT_SAFE, 0, 180, 90], 20, 0)
    time.sleep(2)

def single_joint_move(joint_index, angle):
    print('关节 {} 旋转至 {} 度'.format(joint_index, angle))
    mc.send_angle(joint_index, angle, 40)
    time.sleep(1)

def move_to_top_view():
    print('移动至俯视姿态并开启夹爪')
    # mc.send_angles([-62.13, 8.96, -87.71, -14.41, 2.54, -16.34], 10)

    mc.send_angles([27.77, -44.12, 0.08, -44.2, 2.1, -16.69], 10)
    # 开启夹爪
    mc.set_gripper_value(100, 10)
    time.sleep(2)


def top_view_shot(check=True):
    '''
    拍摄一张图片并保存
    check：是否需要人工看屏幕确认拍照成功，再在键盘上按q键确认继续
    '''
    move_to_top_view()

    # 初始化摄像头（0为默认摄像头，外接摄像头可尝试1）
    cap = cv2.VideoCapture(0)  # 
    time.sleep(1) 

    if not cap.isOpened():
        print("摄像头无法打开")
        exit()

    save_path = "temp/vl_now.jpg"  # 保存路径

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法获取画面")
            break
        
        frame = cv2.flip(frame, 1)  # 1 表示沿x轴翻转
        # 显示实时画面
        # cv2.imshow("picture", frame)

        cv2.imwrite(save_path, frame)
        print(f"照片已保存至：{save_path}")
        break  # 拍照后退出循环
        
        # 按键检测
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord('s') or key == ord('S'):  # 按下S键拍照 
        #     cv2.imwrite(save_path, frame)
        #     print(f"照片已保存至：{save_path}")
        #     photo_taken = True
        #     break  # 拍照后退出循环
        # elif key == ord('q') or key == ord('Q'):  # 按下Q键退出 
        #     break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

    
    # # 获取摄像头，传入0表示获取系统默认摄像头
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # if not cap.isOpened():
    #     print("摄像头无法打开")
    #     exit()

    # # 打开cap
    # cap.open(0)
    # time.sleep(0.3)
    # success, img_bgr = cap.read()
    # img_bgr = cv2.flip(img_bgr, 1)  # 1 表示沿x轴翻转
    
    # # 保存图像
    # print('保存至temp/vl_now.jpg')
    # cv2.imwrite('temp/vl_now.jpg', img_bgr)
    
    # # 屏幕上展示图像
    # cv2.imshow('zihao_vlm', img_bgr) 
    
    # if check:
    #     print('请确认拍照成功，按c键继续，按q键退出')
    #     while(True):
    #         key = cv2.waitKey(10) & 0xFF
    #         if key == ord('c'): # 按c键继续
    #             break
    #         if key == ord('q'): # 按q键退出
    #             # exit()
    #             cv2.destroyAllWindows()   # 关闭所有opencv窗口
    #             raise NameError('按q退出')
    # else:
    #     if cv2.waitKey(10) & 0xFF == None:
    #         pass
        
    # # 关闭摄像头
    # cap.release()
    # # 关闭图像窗口
    # cv2.destroyAllWindows()

def eye2hand(X_im=160, Y_im=120):
    '''
    输入目标点在图像中的像素坐标，转换为机械臂坐标
    '''

    # 整理两个标定点的坐标
    cali_1_im = [17, 470]                       # 左下角，第一个标定点的像素坐标，要手动填！
    cali_1_mc = [136.3, 57.1]                  # 左下角，第一个标定点的机械臂坐标，要手动填！
    cali_2_im = [637, 3]                         # 右上角，第二个标定点的像素坐标
    cali_2_mc = [273.9, -54.6]                    # 右上角，第二个标定点的机械臂坐标，要手动填！
    
    X_cali_im = [cali_1_im[0], cali_2_im[0]]     # 像素坐标
    X_cali_mc = [cali_1_mc[0], cali_2_mc[0]]     # 机械臂坐标
    Y_cali_im = [cali_2_im[1], cali_1_im[1]]     # 像素坐标，先小后大
    Y_cali_mc = [cali_2_mc[1], cali_1_mc[1]]     # 机械臂坐标，先大后小

    # X差值
    X_mc = int(np.interp(X_im, X_cali_im, X_cali_mc))

    # Y差值
    Y_mc = int(np.interp(Y_im, Y_cali_im, Y_cali_mc))

    return X_mc, Y_mc

# 夹爪抓取并移动物体
def pump_move(mc, XY_START=[230,-50], HEIGHT_START=97, XY_END=[100,220], HEIGHT_END=133, HEIGHT_SAFE=220):

    '''
    用夹爪，将物体从起点抓取移动至终点

    mc：机械臂实例
    XY_START：起点机械臂坐标
    HEIGHT_START：起点高度，方块用90，药盒子用70
    XY_END：终点机械臂坐标
    HEIGHT_END：终点高度
    HEIGHT_SAFE：搬运途中安全高度
    '''
    
    # 初始化GPIO
    # GPIO.setmode(GPIO.BCM)
    # GPIO.setup(20, GPIO.OUT)
    # GPIO.setup(21, GPIO.OUT)

    # 设置运动模式为插补
    # mc.set_fresh_mode(0)
    
    # # 机械臂归零
    # print('    机械臂归零')
    # mc.send_angles([0, 0, 0, 0, 0, 0], 40)
    # time.sleep(4)
    
    # 夹爪移动至物体上方
    print('    夹爪移动至物体上方')
    mc.send_coords([XY_START[0], XY_START[1], HEIGHT_SAFE, 0, 180, 90], 20, 0)
    time.sleep(1)

    # 开启夹爪
    # pump_on()
    
    # 夹爪向下抓取物体
    print('    夹爪向下抓取物体')
    mc.send_coords([XY_START[0], XY_START[1], HEIGHT_START, 0, 180, 90], 15, 0)
    time.sleep(2)

    # 夹爪夹紧
    mc.set_gripper_value(20, 10)
    time.sleep(2)

    # 升起物体
    print('    升起物体')
    mc.send_coords([XY_START[0], XY_START[1], HEIGHT_SAFE, 0, 180, 90], 15, 0)
    time.sleep(2)

    # 搬运物体至目标上方
    print('    搬运物体至目标上方')
    mc.send_coords([XY_END[0], XY_END[1], HEIGHT_SAFE, 0, 180, 90], 15, 0)
    time.sleep(2)

    # 向下放下物体
    print('    向下放下物体')
    mc.send_coords([XY_END[0], XY_END[1], HEIGHT_END, 0, 180, 90], 20, 0)
    time.sleep(2)

    # 关闭夹爪
    # pump_off()
    # 夹爪放松
    mc.set_gripper_value(100, 10)
    time.sleep(1)


    # 机械臂归零
    print('    机械臂归零')
    mc.send_angles([0, 0, 0, 0, 0, 0], 40)
    time.sleep(1)
