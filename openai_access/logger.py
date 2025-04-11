"""
Logging
"""
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from pythonjsonlogger import jsonlogger

# 确保 logs 目录存在
logs_dir = './logs'
if not os.path.exists(logs_dir):
    os.makedirs(logs_dir)

# 配置 readable log handler
readable_log_handler = logging.StreamHandler()
readable_log_handler.setFormatter(
    logging.Formatter("%(asctime)-15s %(levelname)-8s %(message)s %(pathname)s")
)

# 配置 file handler
file_handler = TimedRotatingFileHandler(os.path.join(logs_dir, "log"), when="H")
file_handler.setFormatter(
    jsonlogger.JsonFormatter("%(asctime)-15s %(levelname)-8s %(message)s %(pathname)s", json_ensure_ascii=False)
)

# 配置基本的日志设置
logging.basicConfig(
    level="INFO",
    handlers=[readable_log_handler, file_handler]
)

# 初始化日志记录器
logging.info("获取logger")

def get_logger(name: str):
    """get sub-loggers"""
    print(f"获取logger: {name}")
    logger = logging.getLogger(name)
    return logger