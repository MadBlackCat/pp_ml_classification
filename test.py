import re
import tensorflow as tf
from tensorboardX import SummaryWriter

text = "[@START#]"
print(re.search(r"\[@start#\]", text.lower()))
