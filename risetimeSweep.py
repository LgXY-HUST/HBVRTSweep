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
ASC_FILE = "./sweep_trise/sweep.asc"          # 你的电路图文件名

def plot_waveforms(results_data):
    """
    绘制所有仿真的2D关键波形抓取图，标注关键时间点（仅抓取成功时标注）。
    """
    if not results_data:
        print("没有可用于绘图的仿真结果数据")
        return

    # 独立画出每一次仿真的Vd_DUT和Vs_DUT电压之差，并标注关键时间点
    for idx, res in enumerate(results_data):
        plt.figure(f"Simulation_{idx+1}", figsize=(10, 6))
        
        time = res['time']
        v_out = res['v_out']
        w_start, w_end = res['window']
        
        # 限制绘图时间范围以便于观察细节 (单位 us)
        mask = (time >= w_start) & (time <= w_end)
        time_us = time[mask] * 1e6
        v_out_window = v_out[mask]
        
        plt.plot(time_us, v_out_window, label=r'$V_{d\_DUT} - V_{s\_DUT}$', color='b')
        
        # 获取标定的阈值
        v_10 = res['v_10']
        v_90 = res['v_90']
        
        if res['t_10'] is not None and res['t_90'] is not None:
            t_10_us = res['t_10'] * 1e6
            t_90_us = res['t_90'] * 1e6
            
            plt.plot(t_10_us, v_10, 'ro', label=f'10% threshold ({v_10:.1f}V)')
            plt.plot(t_90_us, v_90, 'go', label=f'90% threshold ({v_90:.1f}V)')
            
            # 画辅助线 (垂直与水平)
            plt.vlines(t_10_us, plt.ylim()[0], v_10, colors='r', linestyles='dashed', alpha=0.5)
            plt.vlines(t_90_us, plt.ylim()[0], v_90, colors='g', linestyles='dashed', alpha=0.5)
            plt.hlines(v_10, plt.xlim()[0], t_10_us, colors='r', linestyles='dashed', alpha=0.5)
            plt.hlines(v_90, plt.xlim()[0], t_90_us, colors='g', linestyles='dashed', alpha=0.5)
        
        plt.title(f"Voltage Rise at Turn-off (VLoad={res['vload']}V, I={res['i_target']}A)\nRise Time (10%~90%) = {res['rise_time']*1e9:.2f} ns")
        plt.xlabel(r'Time ($\mu$s)')
        plt.ylabel('Voltage (V)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.tight_layout()

def plot_rise_time_3d(results_data):
    """
    用一张三维图绘制上升时间 (X=目标电流, Y=VLoad, Z=上升时间)。
    """
    if not results_data:
        print("没有可用于绘图的仿真结果数据")
        return

    fig_3d = plt.figure("3D_Rise_Time_Analysis", figsize=(10, 8))
    ax = fig_3d.add_subplot(111, projection='3d')
    
    X = [res['i_target'] for res in results_data]
    Y = [res['vload'] for res in results_data]
    Z = [res['rise_time'] * 1e9 for res in results_data] # 转为 ns
    
    # 画离散点
    scatter = ax.scatter(X, Y, Z, c=Z, cmap='coolwarm', s=100, marker='o', depthshade=False)
    fig_3d.colorbar(scatter, ax=ax, label='Rise Time (ns)', pad=0.1)
    
    # 至少3个点时绘制曲面
    if len(X) >= 3:
        try:
            ax.plot_trisurf(X, Y, Z, cmap='coolwarm', alpha=0.6, edgecolor='none')
        except Exception:
            pass
            
    ax.set_xlabel('Target Current (A)', labelpad=10)
    ax.set_ylabel('VLoad (V)', labelpad=10)
    ax.set_zlabel('Rise Time (ns)', labelpad=10)
    ax.set_title('Rise Time vs Target Current & VLoad', fontweight='bold')
    
    # 展示所有生成的图表
    # 如果单独调用这两个函数，需在外部保证最后调用 plt.show()
    # plt.show()

def run_ltspice_simulation():
    # 确保工作目录存在
    if not os.path.exists(WORKING_DIR):
        os.makedirs(WORKING_DIR)
        
    # 将依赖的 .lib 文件拷贝到 temp 文件夹下以便 LTspice 找到
    import shutil
    lib_files = [
        "sweep_trise/1EDI3031AS.lib",
        "../DPT_MODEL/C3M0040120K.lib"
    ]
    for lib in lib_files:
        if os.path.exists(lib):
            shutil.copy(lib, WORKING_DIR)
        else:
            print(f"Warning: Missing library {lib}")

    # 配置 LTspice 模拟器对象u
    simulator = LTspice.create_from(LTSPICE_EXE)

 
    

    # 初始化 SimRunner (用于执行并行仿真)
    net = PyLTSpice.SpiceEditor(ASC_FILE)  # 加载网表
    
    # 强制沿用之前配置的仿真和求解器设置，防止多线程批量仿真退化到默认求解器而导致速度极慢
    net.add_instructions(
        ".options method=gear"   # 沿用 Gear 积分方法，提升收敛性
    )
    
    runner = SimRunner(simulator=simulator, output_folder=WORKING_DIR, parallel_sims=8 , timeout=1200)  # 设置并行数和超时时间（秒）
    
    print("正在启动 LTspice 多进程参数扫描仿真...")
    
    # 配置要扫描的 VLoad 和目标电流 I_target
    vload_list = np.arange(150, 400 + 100/2, 100)
    i_target_list = np.arange(2, 30 + 10/2, 20)
    
    # 假设 Lload = 100uH
    Lload_val = 100e-6 

    for vload in vload_list:
        for i_target in i_target_list:
            # 根据 VLoad 和 I_target 计算 OnTime1
            # I = (VLoad / Lload) * OnTime1  => OnTime1 = I * Lload / VLoad
            ontime1 = i_target * Lload_val / vload
            ontime1_str = f"{ontime1*1e6:.2f}u"
            
            # 动态修改 VLoad 和 OnTime1 参数
            net.set_parameters(VLoad=vload, OnTime1=ontime1_str)
            
            run_name = f"Vload_{vload}V_I_{i_target}A.net"
            runner.run(net, run_filename=run_name, switches=['-alt'])
    
    # 阻塞等待所有仿真任务完成
    runner.wait_completion()
    print("扫描仿真全部完成！")

    # 4. 读取仿真结果并提取上升时间(通过读取 raw 文件波形插值)
    import re
    from PyLTSpice import RawRead
    
    results_data = []  # 新增一个列表保存画图所需波形与特征点

    # 遍历 runner 的所有结果，恢复原有的原始波形抓取方式
    for raw, log in runner:
        raw_file_path = str(raw)
        
        if raw_file_path and os.path.exists(raw_file_path):
            # 过滤掉非瞬态仿真的结果 (比如 .op.raw)
            if '.op.raw' in raw_file_path:
                continue
                
            # 从文件名解析 Vload 和 I_target，以确定当前的动态阈值
            match = re.search(r'Vload_([0-9\.]+)V_I_([0-9\.]+)A', raw_file_path)
            if not match:
                continue
                
            vload_val = float(match.group(1))
            i_target_val = float(match.group(2))
            
            # 读取日志文件，获取 LTspice 计算并打印的 .meas 参数
            # 因为之前我们根据 TStart=10u 做了固定估算，实际上网表的 TStart 被修改成了 1u，导致窗口不对。
            # 直接通过计算准确的 T5 值：T5 = TStart + Ttrans + Tdead + Ttrans + OnTime1
            # 原网表中 TStart=1u, Ttrans=10n, Tdead=150n.
            t_start = 1e-6
            t_trans = 10e-9
            t_dead = 150e-9
            ontime1 = i_target_val * Lload_val / vload_val
            t5_accurate = t_start + t_trans + t_dead + t_trans + ontime1
            
            # 为保险起见也可以尝试从 log 中的 eoff 指令（如果测量成功）内提取 T5
            # log_file_path = raw_file_path.replace('.raw', '.log')
            # 如果不提取，直接根据物理公式准确计算出来的 t5_accurate 也是极度精确的
            
            # 指定时间窗口：从 T5 附近开始，留一定的余量
            T_WINDOW_START = t5_accurate - 0.2e-6
            T_WINDOW_END = t5_accurate + 1.5e-6
            
            raw_data = RawRead(raw_file_path)
            time = raw_data.get_trace('time').get_wave()
            
            try:
                # 获取 dut 的电压差 (Vd_DUT - Vs_DUT)
                vd = raw_data.get_trace('V(vd_dut)').get_wave()
                vs = raw_data.get_trace('V(vs_dut)').get_wave()
                v_out = vd - vs
                
                # 获取母线电压 VBUS 作为阈值判据依据
                vbus_wave = raw_data.get_trace('V(vbus)').get_wave()
                vbus_val = vbus_wave[0] # 取第一个点即可假设恒定，或者也可以使用 np.mean(vbus_wave)
                
            except Exception as e:
                print(f"[{os.path.basename(raw_file_path)}] 找不到波形数据: {e}")
                continue
            
            # 使用母线电压 VBUS 作为该组仿真的 10% 和 90% 阈值
            v_10 = 0.1 * vbus_val
            v_90 = 0.9 * vbus_val
            
            # 获取在指定时间窗口内的点
            in_window = (time >= T_WINDOW_START) & (time <= T_WINDOW_END)
            
            # 寻找上升沿
            idx_10_arr = np.where(in_window & (v_out >= v_10))[0]
            idx_90_arr = np.where(in_window & (v_out >= v_90))[0]
            
            # 线性插值函数
            def get_interpolated_time(idx_arr, target_v):
                if len(idx_arr) == 0:
                    return None
                i = idx_arr[0]
                if i == 0 or not in_window[i-1]:
                    return time[i]
                
                t1, v1 = time[i-1], v_out[i-1]
                t2, v2 = time[i], v_out[i]
                
                if v2 == v1:
                    return t2
                # 线性插值公式： t = t1 + (v_target - v1) * (t2 - t1) / (v2 - v1)
                return t1 + (target_v - v1) * (t2 - t1) / (v2 - v1)
            
            t_10 = get_interpolated_time(idx_10_arr, v_10)
            t_90 = get_interpolated_time(idx_90_arr, v_90)
            
            if t_10 is not None and t_90 is not None:
                rise_time = t_90 - t_10
                print(f"VLoad={vload_val}V, I_Target={i_target_val}A : Voltage Rise Time (Raw) = {rise_time*1e9:.2f} ns")
            else:
                rise_time = 0.0
                print(f"VLoad={vload_val}V, I_Target={i_target_val}A : 未能在窗口区 {T_WINDOW_START*1e6:.2f}us~{T_WINDOW_END*1e6:.2f}us 抓取到 {v_10}V~{v_90}V 上升沿")
                
            # 保存数据供画图使用 (无论是否抓取到边沿都保存，以便 3D 图以 0 展示)
            results_data.append({
                'vload': vload_val,
                'i_target': i_target_val,
                'time': time,
                'v_out': v_out,
                't_10': t_10,
                't_90': t_90,
                'v_10': v_10,
                'v_90': v_90,
                'rise_time': rise_time,
                'window': (T_WINDOW_START, T_WINDOW_END)
            })

    # 根据提取的结果选择调用画图函数
    plot_waveforms(results_data)
    plot_rise_time_3d(results_data)
    plt.show()

if __name__ == "__main__":
    run_ltspice_simulation()


