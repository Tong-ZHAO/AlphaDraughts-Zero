import logging
import os

def build_logger(name):
	log_directory = "../logs"
	file_path = os.path.join(log_directory, "logger_" + name + ".txt")
	if not os.path.exists(log_directory):
		os.makedirs(log_directory)

	logger = logging.getLogger(name)
	logger.setLevel(logging.INFO)
	handler = logging.FileHandler(file_path)
	fmt = "%(asctime)s %(name)s: %(message)s"
	handler.setFormatter(fmt)
	logging.basicConfig(filename=file_path, filemode='w', format=fmt)
	return logger

'''
Example:

from utils import *

logger = build_logger("model")
logger.info(".......")

'''