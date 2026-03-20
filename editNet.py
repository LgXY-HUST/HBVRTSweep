#read netlist
import os
import matplotlib.pyplot as plt
import PyLTSpice
from PyLTSpice import SimRunner, RawRead, SimCommander, LTspice


# 1. 配置路径
# 请将此路径修改为你电脑上 LTspice 的实际安装位置
LTSPICE_EXE = r"D:\tools\SPICE\LtSpice\LTspice.exe"
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
    net.set_component_value('C1', '10n')    # Updating the value of R1 to 4k
    #net.set_element_model('V3', "SINE(0 1 0.3Meg 0 0 0)")  # changing the behaviour of V3
    net.set_element_model('V3', "PULSE(-50m 50m 250n 1n 1n 1u 2u)")  # changing the behaviour of V3
     
    
    # 设置仿真命令，例如 transient analysis
    # 尝试移除原有的 .tran 指令 (如果存在)
    net.remove_instruction(".tran")
    # 添加新的 .tran 指令
    net.add_instructions(".tran 2u")

    # 3. 初始化 SimRunner (用于执行仿真)
    runner = SimRunner(simulator=simulator, output_folder=WORKING_DIR)
    
    print("正在启动 LTspice 仿真...")
    # 运行仿真
    runner.run(net)
    
    # 阻塞等待所有仿真任务完成
    runner.wait_completion()
    print("仿真完成！")

    # 4. 读取仿真结果 (.raw 文件)
    # PyLTSpice 的 SimRunner 为了避免覆盖，会自动为输出文件添加后缀 (例如 _1, _2 等)
    # 通过遍历 runner 可以获取最新生成的 raw_file
    raw_file_path = None
    for raw, log in runner:
        raw_file_path = str(raw)

    if raw_file_path and os.path.exists(raw_file_path):
        raw_data = RawRead(raw_file_path)
        
        # 获取时间轴的真实数值数组 (如果有step参数扫描这里会有多维，这里默认取第一组)
        time = raw_data.get_trace('time').get_wave()
        # 获取电压/电流信号的真实数值数组
        v_out = raw_data.get_trace('V(out)').get_wave()
        
        # 5. 绘图显示
        plt.figure(figsize=(10, 6))
        plt.plot(time, v_out, label='V(out)')
        plt.title('LTspice Simulation Result')
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.grid(True)
        plt.legend()
        plt.show()
    else:
        print(f"未找到结果文件: {raw_file_path}")

if __name__ == "__main__":
    run_ltspice_simulation()


