from src.trainer_package_gdamms.trainer_progress import TrainProgress
import time

tp = TrainProgress(20, 10, 10, 10)

with tp:
    for i in range((20 * (10 + 10) + 10)):
        tp.step()
        time.sleep(0.1)
