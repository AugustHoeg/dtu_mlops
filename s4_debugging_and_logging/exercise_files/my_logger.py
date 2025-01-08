import os
import sys
from loguru import logger

logger.remove()
#logger.add(sys.stdout, level="WARNING") # Warning and above will be printed to the console
logger.add("s4_debugging_and_logging/exercise_files/my_log.log", level="WARNING", rotation="100 MB") # Warning and above will be printed to my_log.log

logger.debug("Used for debugging your code.")
logger.info("Informative messages from your code.")
logger.warning("Everything works but there is something to be aware of.")
logger.error("There's been a mistake with the process.")
logger.critical("There is something terribly wrong and process may terminate.")
