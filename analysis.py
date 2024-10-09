import pandas as pd
from datetime import timedelta

# Reading in log data
power_log = pd.read_csv("./Outputs/power_log.csv")
util_log = pd.read_csv("./Outputs/util_log.csv")
memory_log = pd.read_csv("./Outputs/memory_log.csv")
timestamps = pd.read_csv("./Outputs/model_timestamps.csv")

# Getting BP and FF timestamps
# BP_start_time = timestamps["BP"][0]
# BP_end_time = timestamps["BP"][1]
FF_start_time = timestamps["FF"][0]
FF_end_time = timestamps["FF"][1]

# runtime
# BP_runtime = BP_end_time - BP_start_time
FF_runtime = FF_end_time - FF_start_time
print("="*50)
# print(f"BP runtime: {BP_runtime}")
print(f"FF runtime: {timedelta(seconds=FF_runtime)}")
print("="*50)

## Getting Average BP power
# BP_power = power_log[(power_log["Timestamp"] >= BP_start_time) & (power_log["Timestamp"] <= BP_end_time)]["Value"].mean()
# print(f"Average BP power: {BP_power}")
# print("Total BP power consumed: ", (BP_power * BP_runtime))

## Getting Average FF power
FF_power = power_log[(power_log["Timestamp"] >= FF_start_time) & (power_log["Timestamp"] <= FF_end_time)]["Value"].mean()
print(f"Average FF power: {FF_power}")
print("Total FF power consumed: ", (FF_power * FF_runtime))

print("="*50)

# ## Getting Average BP utilization
# BP_util = util_log[(util_log["Timestamp"] >= BP_start_time) & (util_log["Timestamp"] <= BP_end_time)]["Value"].mean()
# print(f"Average BP utilization: {BP_util}")
# print("Total BP utilization: ", (BP_util * BP_runtime))


## Getting Average FF utilization
FF_util = util_log[(util_log["Timestamp"] >= FF_start_time) & (util_log["Timestamp"] <= FF_end_time)]["Value"].mean()
print(f"Average FF utilization: {FF_util}")
print("Total FF utilization: ", (FF_util * FF_runtime))

print("="*50)

# ## Getting Average BP memory usage
# BP_mem = memory_log[(memory_log["Timestamp"] >= BP_start_time) & (memory_log["Timestamp"] <= BP_end_time)]["Value"].mean()
# print(f"Average BP memory usage: {BP_mem}")
# print("Total BP memory usage: ", (BP_mem * BP_runtime))

## Getting Average FF memory usage
FF_mem = memory_log[(memory_log["Timestamp"] >= FF_start_time) & (memory_log["Timestamp"] <= FF_end_time)]["Value"].mean()
print(f"Average FF memory usage: {FF_mem}")
print("Total FF memory usage: ", (FF_mem * FF_runtime))

print("="*50)


