import concurrent.futures
import time

class ComputationWithTimeout:
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds

    def run(self, mission=None):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            if mission:
                future = executor.submit(mission)
            else:
                future = executor.submit(self.long_computation)
            try:
                # 等待计算完成，设置超时时间
                future.result(timeout=self.timeout_seconds)
            except concurrent.futures.TimeoutError:
                raise TimeoutException("Calculation took too long!")

    def long_computation(self):
        # 模拟一个计算复杂度高的步骤
        time.sleep(5)
 
class TimeoutException(Exception):
    pass