import sys
import pickle
import os
import glob
import time
from generate_labels import predict_task

output_path = sys.argv[1]
input_path = sys.argv[2]

predict_task(input_path, output_path)
