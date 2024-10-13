import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# Reading in log data
power_log = pd.read_csv("./Outputs/power_log.csv")
util_log = pd.read_csv("./Outputs/util_log.csv")
memory_log = pd.read_csv("./Outputs/memory_log.csv")
timestamps = pd.read_csv("./Outputs/model_timestamps.csv")
BP_epoch_data = pd.read_csv("./Outputs/BP_epoch_data.csv")
FF_epoch_data = pd.read_csv("./Outputs/FF_epoch_data.csv")

# Getting BP and FF timestamps
BP_start_time = timestamps["BP"][0]
BP_end_time = timestamps["BP"][1]
FF_start_time = timestamps["FF"][0]
FF_end_time = timestamps["FF"][1]

# runtime
BP_runtime = BP_end_time - BP_start_time
FF_runtime = FF_end_time - FF_start_time
print("="*50)
print(f"BP runtime: {timedelta(seconds=BP_runtime)}")
print(f"FF runtime: {timedelta(seconds=FF_runtime)}")
print("="*50)

##### Avg pwoer #####
# BP
# BP_power = power_log[(power_log["Timestamp"] >= BP_start_time) & (power_log["Timestamp"] <= BP_end_time)]["Value"].mean()
# print(f"Average BP power: {BP_power}")
# print("Total BP power consumed: ", (BP_power * BP_runtime))
# FF
# FF_power = power_log[(power_log["Timestamp"] >= FF_start_time) & (power_log["Timestamp"] <= FF_end_time)]["Value"].mean()
# print(f"Average FF power: {FF_power}")
# print("Total FF power consumed: ", (FF_power * FF_runtime))
print("="*50)

##### Memory Util #####
# BP 
BP_util = util_log[(util_log["Timestamp"] >= BP_start_time) & (util_log["Timestamp"] <= BP_end_time)]["Value"].mean()
print(f"Average BP utilization: {BP_util}")
print("Total BP utilization: ", (BP_util * BP_runtime))
# FF 
FF_util = util_log[(util_log["Timestamp"] >= FF_start_time) & (util_log["Timestamp"] <= FF_end_time)]["Value"].mean()
print(f"Average FF utilization: {FF_util}")
print("Total FF utilization: ", (FF_util * FF_runtime))
print("="*50)

##### Memory usage #####
# BP
BP_mem = memory_log[(memory_log["Timestamp"] >= BP_start_time) & (memory_log["Timestamp"] <= BP_end_time)]["Value"].mean()
print(f"Average BP memory usage: {BP_mem}")
print("Total BP memory usage: ", (BP_mem * BP_runtime))
# FF
FF_mem = memory_log[(memory_log["Timestamp"] >= FF_start_time) & (memory_log["Timestamp"] <= FF_end_time)]["Value"].mean()
print(f"Average FF memory usage: {FF_mem}")
print("Total FF memory usage: ", (FF_mem * FF_runtime))
print("="*50)

#### Power Usage ####
# BP
BP_power = power_log[(power_log["Timestamp"] >= BP_start_time) & (power_log["Timestamp"] <= BP_end_time)]["Value"].mean()
print(f"Average BP power: {BP_power}")
print("Total BP power consumed: ", (BP_power * BP_runtime))
# FF
FF_power = power_log[(power_log["Timestamp"] >= FF_start_time) & (power_log["Timestamp"] <= FF_end_time)]["Value"].mean()
print(f"Average FF power: {FF_power}")
print("Total FF power consumed: ", (FF_power * FF_runtime))
print("="*50)


#### Plots ####
# # Function to print the power log
# plt.plot(power_log["Timestamp"], power_log["Value"])
# plt.title('Power Draw Comparison')
# plt.xlabel('Timestamp (UTC)')
# plt.ylabel('Power')
# plt.axvline(x=BP_start_time, color="r", linestyle="--", label="BP Start")
# plt.axvline(x=BP_end_time, color="r", linestyle="--", label="BP End")
# plt.axvline(x=FF_start_time, color="g", linestyle="--", label="FF Start")
# plt.axvline(x=FF_end_time, color="g", linestyle="--", label="FF End")
# plt.legend()
# plt.savefig("./images/power_log.png")
# plt.clf()

# Memory Utilization
plt.plot(util_log["Timestamp"],util_log["Value"])
plt.title('Memory Utilisation')
plt.xlabel('Timestamp (UTC)')
plt.ylabel('Memory Utilisation (%)')
plt.axvline(x=BP_start_time, color="r", linestyle="--", label="BP Start")
plt.axvline(x=BP_end_time, color="r", linestyle="--", label="BP End")
plt.axvline(x=FF_start_time, color="g", linestyle="--", label="FF Start")
plt.axvline(x=FF_end_time, color="g", linestyle="--", label="FF End")
plt.legend()
plt.savefig("./images/util_log.png")
plt.clf()

# Memory usage
plt.plot(memory_log["Timestamp"], memory_log["Value"])
plt.title('Memory Usage')
plt.xlabel('Timestamp (UTC)')
plt.ylabel('Memory Usage (MB)')
plt.axvline(x=BP_start_time, color="r", linestyle="--", label="BP Start")
plt.axvline(x=BP_end_time, color="r", linestyle="--", label="BP End")
plt.axvline(x=FF_start_time, color="g", linestyle="--", label="FF Start")
plt.axvline(x=FF_end_time, color="g", linestyle="--", label="FF End")
plt.legend()
plt.savefig("./images/memory_log.png")
plt.clf()

# Power usage
plt.plot(power_log["Timestamp"], power_log["Value"])
plt.title('Power Usage')
plt.xlabel('Timestamp (UTC)')
plt.ylabel('Power Usage (W)')
plt.axvline(x=BP_start_time, color="r", linestyle="--", label="BP Start")
plt.axvline(x=BP_end_time, color="r", linestyle="--", label="BP End")
plt.axvline(x=FF_start_time, color="g", linestyle="--", label="FF Start")
plt.axvline(x=FF_end_time, color="g", linestyle="--", label="FF End")
plt.legend()
plt.savefig("./images/power_log.png")
plt.clf()

# Plotting  memory data for each epoch
plt.plot(BP_epoch_data["Epoch"], BP_epoch_data["Memory"], label="BP")
plt.plot(FF_epoch_data["Epoch"], FF_epoch_data["Memory"], label="FF")
plt.title('Memory Usage per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Memory Usage (MB)')
plt.legend()
plt.savefig("./images/memory_epoch.png")
plt.clf()

# Plotting power data for each epoch
plt.plot(BP_epoch_data["Epoch"], BP_epoch_data["Power"], label="BP")
plt.plot(FF_epoch_data["Epoch"], FF_epoch_data["Power"], label="FF")
plt.title('Power Usage per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Power Usage (W)')
plt.legend()
plt.savefig("./images/power_epoch.png")
plt.clf()

# Plotting utilization data for each epoch
plt.plot(BP_epoch_data["Epoch"], BP_epoch_data["Utilization"], label="BP")
plt.plot(FF_epoch_data["Epoch"], FF_epoch_data["Utilization"], label="FF")
plt.title('Utilization per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Utilization (%)')
plt.legend()
plt.savefig("./images/util_epoch.png")
plt.clf()

# Plotting ratio of FF power usage to BP power usage
plt.plot(BP_epoch_data["Epoch"], FF_epoch_data["Power"]/BP_epoch_data["Power"])
plt.axhline(y=1, color='r', linestyle='--')
plt.title('FF/BP Power Usage Ratio')
plt.xlabel('Epoch')
plt.ylabel('FF/BP Power Usage Ratio')
plt.savefig("./images/power_ratio.png")
plt.clf()

# Plotting ratio of FF memory usage to BP memory usage
plt.plot(BP_epoch_data["Epoch"], FF_epoch_data["Memory"]/BP_epoch_data["Memory"])
plt.axhline(y=1, color='r', linestyle='--')
plt.title('FF/BP Memory Usage Ratio')
plt.xlabel('Epoch')
plt.ylabel('FF/BP Memory Usage Ratio')
plt.savefig("./images/memory_ratio.png")
plt.clf()

# Plotting ratio of FF utilization to BP utilization
plt.plot(BP_epoch_data["Epoch"], FF_epoch_data["Utilization"]/BP_epoch_data["Utilization"])
plt.axhline(y=1, color='r', linestyle='--')
plt.title('FF/BP Utilization Ratio')
plt.xlabel('Epoch')
plt.ylabel('FF/BP Utilization Ratio')
plt.savefig("./images/util_ratio.png")
plt.clf()
