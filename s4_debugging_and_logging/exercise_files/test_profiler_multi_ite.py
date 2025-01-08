import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity

model = models.resnet18()
inputs = torch.randn(5, 3, 224, 224)

''' # Multi-ite profiling
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True) as prof:
    for i in range(10):
        model(inputs)
        prof.step()

print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))

prof.export_chrome_trace("s4_debugging_and_logging/exercise_files/trace.json")
'''

from torch.profiler import profile, tensorboard_trace_handler
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True, record_shapes=True, on_trace_ready=tensorboard_trace_handler("./s4_debugging_and_logging/exercise_files/log/resnet18")) as prof:
    for i in range(10):
        model(inputs)
        prof.step()

print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))

# Now run this command in terminal to see the tensorboard
# tensorboard --logdir=./s4_debugging_and_logging/exercise_files/log

# Open this link:
# http://localhost:6006/#pytorch_profiler