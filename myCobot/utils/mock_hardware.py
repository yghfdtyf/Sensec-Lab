import time
import numpy as np
import cv2
import os

class MockMyCobot:
    """模拟 MyCobot 机械臂类"""
    def __init__(self, port=None, baudrate=None, debug=False):
        self.angles = [0, 0, 0, 0, 0, 0]
        self.coords = [160, 0, 200, 0, 0, 0]  # 默认坐标
        self.speed = 50
        self.debug = debug
        print("初始化模拟 MyCobot 机械臂")
    
    def send_angles(self, angles, speed):
        self.angles = angles
        self.speed = speed
        if self.debug:
            print(f"机械臂移动关节到: {angles}, 速度: {speed}")
        # 模拟移动时间
        time.sleep(0.5)
        return True
    
    def send_angle(self, joint_id, angle, speed):
        if 0 <= joint_id < len(self.angles):
            self.angles[joint_id] = angle
        if self.debug:
            print(f"关节 {joint_id} 移动到: {angle}, 速度: {speed}")
        # 模拟移动时间
        time.sleep(0.2)
        return True
    
    def send_coords(self, coords, speed, mode=0):
        self.coords = coords
        if self.debug:
            print(f"机械臂移动到坐标: {coords}, 速度: {speed}, 模式: {mode}")
        # 模拟移动时间
        time.sleep(0.5)
        return True
    
    def get_angles(self):
        # 模拟读取可能的延迟
        time.sleep(0.05)
        return self.angles
    
    def get_coords(self):
        # 模拟读取可能的延迟
        time.sleep(0.05)
        return self.coords
    
    def get_encoders(self):
        # 模拟编码器值 (与 get_angles 类似但可能有不同的单位)
        return [int(a * 100) for a in self.angles]
    
    def set_encoders(self, encoders, speed):
        # 模拟设置编码器值
        if self.debug:
            print(f"设置编码器值: {encoders}, 速度: {speed}")
        time.sleep(0.1)
        return True
    
    def release_all_servos(self):
        if self.debug:
            print("放松所有伺服电机")
        return True
    
    def set_fresh_mode(self, mode):
        if self.debug:
            print(f"设置刷新模式: {mode}")
        return True

class MockCamera:
    """模拟摄像头类"""
    def __init__(self):
        self.test_images = self._find_test_images()
        self.current_img_index = 0
        print(f"初始化模拟摄像头，找到 {len(self.test_images)} 张测试图片")
    
    def _find_test_images(self):
        """查找可用的测试图片"""
        test_dirs = ['temp', 'asset', 'test_result']
        images = []
        
        for dir_name in test_dirs:
            if os.path.exists(dir_name):
                for file in os.listdir(dir_name):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        images.append(os.path.join(dir_name, file))
        
        # 如果没找到图片，创建一个简单的测试图像
        if not images:
            test_img = np.ones((480, 640, 3), dtype=np.uint8) * 255  # 白色背景
            # 画一些几何图形来模拟物体
            cv2.rectangle(test_img, (100, 100), (200, 200), (0, 0, 255), -1)  # 红色方块
            cv2.rectangle(test_img, (300, 300), (400, 400), (0, 255, 0), -1)  # 绿色方块
            cv2.circle(test_img, (450, 150), 50, (255, 0, 0), -1)  # 蓝色圆形
            
            # 保存测试图像
            os.makedirs('temp', exist_ok=True)
            cv2.imwrite('temp/mock_test_image.jpg', test_img)
            images.append('temp/mock_test_image.jpg')
        
        return images
    
    def read(self):
        """模拟摄像头读取图像"""
        if not self.test_images:
            # 如果没有测试图片，返回简单的彩色测试图像
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            return True, img
        
        # 从测试图片中读取
        img = cv2.imread(self.test_images[self.current_img_index])
        # 循环使用测试图片
        self.current_img_index = (self.current_img_index + 1) % len(self.test_images)
        return True, img
    
    def release(self):
        """释放摄像头资源"""
        pass

# 模拟 GPIO 接口
class MockGPIO:
    """模拟 GPIO 接口"""
    BCM = 'BCM'
    OUT = 'OUT'
    IN = 'IN'
    
    def __init__(self):
        self.pins = {}
        print("初始化模拟 GPIO")
    
    def setmode(self, mode):
        self.mode = mode
    
    def setup(self, pin, mode):
        self.pins[pin] = {'mode': mode, 'value': 0}
    
    def output(self, pin, value):
        if pin in self.pins:
            self.pins[pin]['value'] = value
            print(f"设置 GPIO 引脚 {pin} 为 {value}")
    
    def input(self, pin):
        return self.pins.get(pin, {}).get('value', 0)
    
    def setwarnings(self, flag):
        pass
    
    def cleanup(self):
        self.pins = {}

# 导出模拟类
mock_mc = MockMyCobot(debug=True)
mock_camera = MockCamera()
mock_gpio = MockGPIO()