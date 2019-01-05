import time
from utils import *
import config

logger = build_logger("model", config.file2write)


for i in range(100):
	logger.info(str(i))
	time.sleep(2)