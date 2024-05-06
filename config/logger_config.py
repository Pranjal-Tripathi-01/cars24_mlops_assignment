import logging

fmt = '%(asctime)s : %(funcName)s : %(message)s'

info = logging.FileHandler("logs/info.log", mode='a')
info.setLevel(logging.INFO)

error = logging.FileHandler("logs/error.log", mode='a')
error.setLevel(logging.ERROR)

console_error_handler = logging.StreamHandler()
console_error_handler.setLevel(logging.ERROR)

console_info_handler = logging.StreamHandler()
console_info_handler.setLevel(logging.INFO) 

formatter = logging.Formatter(fmt)
console_error_handler.setFormatter(formatter)
console_info_handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO, format=fmt, handlers=[info, error, 
                                                    console_info_handler, console_error_handler])

logger = logging.getLogger()
