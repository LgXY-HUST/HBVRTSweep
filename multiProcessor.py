#read netlist
import os
import numpy as np
import matplotlib.pyplot as plt
import PyLTSpice
from PyLTSpice import SimRunner, RawRead, SimCommander, LTspice

# 触发并发的地方：脚本对每个 C1 值调用了 runner.run(net, run_filename=...) 多次，将多个仿真任务提交给同一个 SimRunner 实例，从而实现批量仿真调度。
# 并发执行由谁负责：并不是 Python 的 multiprocessing，而由 PyLTSpice.SimRunner 内部管理并行（它会以独立 LTspice 进程或线程方式启动多个 LTspice 实例并行运行）。
# 结果区分与文件安全：使用 run_filename（或 SimRunner 自动加后缀）确保每次仿真生成不同的 .net/.raw 文件，避免覆盖和冲突。
# 同步与等待：调用 runner.wait_completion() 会阻塞，直到 SimRunner 管理的所有仿真子进程完成。
# 控制并发数：可以在创建 SimRunner 时传入并发限制参数（例如 parallel_sims=4）来限制同时运行的 LTspice 实例数量。如果你的 PyLTSpice 版本支持，示例：
# runner = SimRunner(simulator=simulator, output_folder=WORKING_DIR, parallel_sims=4)



# 1. 配置路径
# 请将此路径修改为你电脑上 LTspice 的实际安装位置
LTSPICE_EXE = r"D:\software\simulation\hardware\LtSpice\LTspice.exe"
WORKING_DIR = r"./temp"  # 仿真文件存放目录
ASC_FILE = "./simulation_folder/UniversalOpAmp2.asc"          # 你的电路图文件名

def run_ltspice_simulation():
    # 确保工作目录存在
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)

    # 配置 LTspice 模拟器对象u
    simulator = LTspice.create_from(LTSPICE_EXE)

    # 这是旧版本，SimCommander 已经不再需要手动创建
    # # 2. 初始化 SimCommander (用于修改电路参数)
    # # 这会创建一个 .net 网表文件进行操作
    # meater = SimCommander(ASC_FILE, simulator=simulator)
    # # 动态修改参数：将电阻 R1 的值改为 2k (可选)
    # meater.set_component_value('R1', '2k')
    # meater.set_parameters(V_input=5.0)  # 如果电路中有参数变量 {V_input}
    

     # 这是旧版本，新版本使用  SpiceEditor 来加载和修改 netlist
    net = PyLTSpice.SpiceEditor(ASC_FILE)  # 加载网表
    # net.set_parameters(res=0, cap=100e-6)  # 更新参数 res 和 cap
    net.set_component_value('R1', '2k')    # Updating the value of R2 to 2k
    # 我们将把这里固定的 C1 设值改为在后面用循环进行参数扫描
    # net.set_component_value('C1', '10n')    # 这一行被我们抛弃，交给下面的循环处理
    #net.set_element_model('V3', "SINE(0 1 0.3Meg 0 0 0)")  # changing the behaviour of V3
    net.set_element_model('V3', "PULSE(-50m 50m 250n 1n 1n 1u 2u)")  # changing the behaviour of V3
     
    
    # 设置仿真命令，例如 transient analysis
    # 尝试移除原有的 .tran 指令 (如果存在)
    net.remove_instruction(".tran")
    # 添加新的 .tran 指令
    net.add_instructions(".tran 1.5u")
    # 3. 初始化 SimRunner (用于执行并行仿真)
    runner = SimRunner(simulator=simulator, output_folder=WORKING_DIR, parallel_sims=16)
    
    print("正在启动 LTspice 多进程参数扫描仿真...")
    
    # 采用 初值-终值-步长 形式定义扫描范围：从 100e-12 (100p) 到 1e-9 (1n)，步长为 200e-12 (200p)
    c1_start = 100e-12
    c1_stop = 1e-9
    c1_step = 100e-12
    
    # 使用 numpy.arange 生成扫描值数组
    c1_sweep_values = np.arange(c1_start, c1_stop + c1_step/2, c1_step)
    
    for c_val in c1_sweep_values:
        # 将浮点数转换为便于阅读的工程单位字符串 (例如: 100p, 1n) 进行赋值
        if c_val < 1e-9:
            c_val_str = f"{c_val*1e12:.0f}p"
        else:
            c_val_str = f"{c_val*1e9:.1f}n"
            
        net.set_component_value('C1', c_val_str)  # 动态修改 C1
        # 设置不同的 run_filename 可以方便后续处理不同的文件
        runner.run(net, run_filename=f"C1_sweep_{c_val_str}.net")
    
    # 阻塞等待所有仿真任务完成
    runner.wait_completion()
    print("扫描仿真全部完成！")

    # 4. 读取仿真结果并绘制折线图
    plt.figure(1, figsize=(12, 7))
    found_result = False
    
    c1_values_list = []
    rise_times = []
    import re

    # === 在这里设定计算上升时间的电压阈值范围 (V) ===
    V_THRESHOLD_LOW = -0.047   # 起始阈值 (例如 10% 对应的电压)
    V_THRESHOLD_HIGH = 0.047  # 结束阈值 (例如 90% 对应的电压)
    # ================================================

    # === 在这里设定查找上升沿的时间窗口范围 (秒) ===
    # 用来避开仿真早期的不稳定状态或只针对特定的脉冲沿进行测量
    # 如果不想限制，可以设 T_WINDOW_START = 0，T_WINDOW_END = float('inf')
    T_WINDOW_START = 0.0e-6          # 查找时间窗口起点
    T_WINDOW_END = 1.4e-6         # 查找时间窗口终点 (设为1.5us)
    # ================================================

    # runner 迭代器会遍历出所有的 (raw文件对象, log文件对象)
    for raw, log in runner:
        raw_file_path = str(raw)
        
        if raw_file_path and os.path.exists(raw_file_path):
            found_result = True
            raw_data = RawRead(raw_file_path)
            
            # 获取时间和电压数据
            time = raw_data.get_trace('time').get_wave()
            v_out = raw_data.get_trace('V(out)').get_wave()
            
            # 使用用户设定的阈值来计算上升时间
            v_10 = V_THRESHOLD_LOW
            v_90 = V_THRESHOLD_HIGH
            
            # 获取在指定时间窗口内，电压首次超过设定阈值的索引
            idx_10_arr = np.where((v_out <= v_10) & (time >= T_WINDOW_START) & (time <= T_WINDOW_END))[0]
            idx_90_arr = np.where((v_out <= v_90) & (time >= T_WINDOW_START) & (time <= T_WINDOW_END))[0]
            
            # --- Debug 打印 ---
            # 你可以在终端里查看每个文件被抓取到的数组长度和具体的时间点
            label_name = os.path.basename(raw_file_path).replace('.raw', '').replace('C1_sweep_', 'C1=')
            print(f"[{label_name}] idx_10_arr length: {len(idx_10_arr)}, idx_90_arr length: {len(idx_90_arr)}")
            if len(idx_10_arr) > 0:
                print(f"    -> First passed V_10 at time: {time[idx_10_arr[0]]}")
            if len(idx_90_arr) > 0:
                print(f"    -> First passed V_90 at time: {time[idx_90_arr[0]]}")
            # ------------------
            
            if len(idx_10_arr) > 0 and len(idx_90_arr) > 0:
                idx_10 = idx_10_arr[0]
                idx_90 = idx_90_arr[0]
                rise_time = time[idx_90] - time[idx_10]
            else:
                rise_time = 0
            
            match = re.search(r'C1_sweep_([0-9.]+)([pn])', raw_file_path)
            if match:
                val = float(match.group(1))
                unit = match.group(2)
                c1_val = val * 1e-12 if unit == 'p' else val * 1e-9
                
                c1_values_list.append(c1_val)
                rise_times.append(rise_time)
            
            # 提取之前设定的文件名，用来作为画图的图例（Legend）
            label_name = os.path.basename(raw_file_path).replace('.raw', '').replace('C1_sweep_', 'C1=')
            
            plt.figure(1)
            plt.plot(time, v_out, label=label_name)

    if found_result:
        plt.figure(1)
        plt.title('Transient Response vs C1 Values')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.grid(True)
        # plt.legend() # 若有多条曲线可能会遮挡，这里先保留，或者可以在实际中注释掉
        
        if len(c1_values_list) > 0:
            # 根据 C1 的容值排序，确保画图时线不乱交错
            sorted_indices = np.argsort(c1_values_list)
            c1_values_sorted = np.array(c1_values_list)[sorted_indices]
            rise_times_sorted = np.array(rise_times)[sorted_indices]
            
            plt.figure(2, figsize=(10, 6))
            plt.plot(c1_values_sorted * 1e12, rise_times_sorted * 1e9, marker='o')
            plt.title('Rise Time (10% to 90%) vs C1 Capacitance')
            plt.xlabel('C1 Capacitance (pF)')
            plt.ylabel('Rise Time (ns)')
            plt.grid(True)
            
        plt.show()
    else:
        print("未找到结果文件!")

if __name__ == "__main__":
    run_ltspice_simulation()


