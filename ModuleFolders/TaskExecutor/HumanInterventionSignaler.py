from PyQt5.QtCore import QObject, pyqtSignal, Qt

class HumanInterventionSignaler(QObject):
    """
    用于在工作线程和GUI线程之间传递人工介入请求的信号类
    """
    # 信号：请求人工介入 (task_type, task_data) -> result
    # 我们不能直接通过信号返回值，所以需要一种同步机制
    # 这里我们使用信号通知主线程，主线程处理完后通过另一个信号或共享变量返回结果
    
    request_signal = pyqtSignal(str, dict, object)  # task_type, task_data, result_container

