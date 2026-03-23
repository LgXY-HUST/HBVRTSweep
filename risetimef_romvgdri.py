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
        fig, ax1 = plt.subplots(figsize=(10, 6), num=f"Simulation_{idx+1}")
        
        time = res['time']
        v_out = res['v_out']
        v_driver = res.get('v_driver', None)
        w_start, w_end = res['window']
        
        # 限制绘图时间范围以便于观察细节 (单位 us)
        mask = (time >= w_start) & (time <= w_end)
        time_us = time[mask] * 1e6
        v_out_window = v_out[mask]
        
        ax1.plot(time_us, v_out_window, label=r'Vds ($V_{d\_DUT} - V_{s\_DUT}$)', color='b')
        ax1.set_xlabel(r'Time ($\mu$s)')
        ax1.set_ylabel('Vds Voltage (V)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # 获取标定的阈值
        v_start_thresh = res['v_10']
        v_end_thresh = res['v_90']
        
        if v_driver is not None:
            ax2 = ax1.twinx()
            v_driver_window = v_driver[mask]
            ax2.plot(time_us, v_driver_window, label=r'Vdriver ($V_{g\_dri} - V_{ks}$)', color='orange', linestyle='--')
            ax2.set_ylabel('Driver Voltage (V)', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
        
        if res['t_10'] is not None and res['t_90'] is not None:
            t_10_us = res['t_10'] * 1e6
            t_90_us = res['t_90'] * 1e6
            
            if v_driver is not None:
                ax2.plot(t_10_us, v_start_thresh, 'ro', label=f'0.9 VCC2 ({v_start_thresh:.1f}V)')
                ax2.hlines(v_start_thresh, plt.xlim()[0], t_10_us, colors='r', linestyles='dashed', alpha=0.5)
            else:
                ax1.plot(t_10_us, v_start_thresh, 'ro', label=f'Start Thresh ({v_start_thresh:.1f}V)')
                
            ax1.plot(t_90_us, v_end_thresh, 'go', label=f'0.9 VBUS ({v_end_thresh:.1f}V)')
            
            # 画辅助线 (垂直与水平)
            ax1.axvline(t_10_us, color='r', linestyle='dashed', alpha=0.5)
            ax1.axvline(t_90_us, color='g', linestyle='dashed', alpha=0.5)
            ax1.hlines(v_end_thresh, plt.xlim()[0], t_90_us, colors='g', linestyles='dashed', alpha=0.5)
        
        plt.title(f"Voltage Rise at Turn-off (VLoad={res['vload']}V, I={res['i_target']}A)\nRise Time = {res['rise_time']*1e9:.2f} ns")
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # 合并图例
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        if v_driver is not None:
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc=0)
        else:
            ax1.legend(loc=0)
            
        fig.tight_layout()

def plot_rise_time_3d(results_data):
    """
    用一张三维图绘制上升时间 (X=实际电流, Y=VLoad, Z=上升时间)。使用插值产生平滑曲面。
    """
    if not results_data:
        print("没有可用于绘图的仿真结果数据")
        return

    fig_3d = plt.figure("3D_Rise_Time_Analysis", figsize=(10, 8))
    ax = fig_3d.add_subplot(111, projection='3d')
    
    # 获取实际测量的电流和电压数据
    X = np.array([res['i_actual'] for res in results_data])
    Y = np.array([res['vload'] for res in results_data])
    Z = np.array([res['rise_time'] * 1e9 for res in results_data]) # 转为 ns
    
    # 画离散点
    scatter = ax.scatter(X, Y, Z, c=Z, cmap='coolwarm', s=20, marker='o', depthshade=False, label='Simulated Points')
    fig_3d.colorbar(scatter, ax=ax, label='Rise Time (ns)', pad=0.1)
    
    # 当有足够且多维数据点时（X和Y的独立值均大于等于2），通过网格插值绘制连续曲面以便填充无数据点区域
    if len(X) >= 4 and len(np.unique(X)) >= 2 and len(np.unique(Y)) >= 2:
        try:
            from scipy.interpolate import griddata
            
            # 创建更密集的网格用于插值
            grid_x, grid_y = np.mgrid[min(X):max(X):100j, min(Y):max(Y):100j]
            
            # 使用 linear 方法进行插值
            grid_z = griddata((X, Y), Z, (grid_x, grid_y), method='linear')
            
            # 绘制插值生成的曲面
            ax.plot_surface(grid_x, grid_y, grid_z, cmap='coolwarm', alpha=0.5, edgecolor='none')
        except Exception as e:
            print(f"曲面插值失败: {e}")
            try:
                # 回退到简单的三角曲面
                ax.plot_trisurf(X, Y, Z, cmap='coolwarm', alpha=0.6, edgecolor='none')
            except Exception as e2:
                print(f"三角曲面绘制失败: {e2}")
    else:
        print("X或Y维度的独立数据点不足以生成曲面（比如扫描固定VLoad），仅显示散点。")
            
    ax.set_xlabel('Actual Current I(Rshunt) (A)', labelpad=10)
    ax.set_ylabel('VLoad (V)', labelpad=10)
    ax.set_zlabel('Rise Time (ns)', labelpad=10)
    ax.set_title('Rise Time vs Actual Current & VLoad', fontweight='bold')
    
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
    
    runner = SimRunner(simulator=simulator, output_folder=WORKING_DIR, parallel_sims=4 , timeout=1200)  # 设置并行数和超时时间（秒）
    
    print("正在启动 LTspice 多进程参数扫描仿真...")
    
    # 配置要扫描的 VLoad 和目标电流 I_target
    vload_list = np.arange(20, 380 , 20)
    i_target_list = np.arange(0.1, 3.2, 0.15)
    
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
            # runner.run(net, run_filename=run_name, switches=['-alt'])
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
            T_WINDOW_START = t5_accurate - 0.0e-6
            T_WINDOW_END = t5_accurate + 1.5e-6
            
            raw_data = RawRead(raw_file_path)
            time = raw_data.get_trace('time').get_wave()
            
            try:
                # 获取 dut 的电压差 (Vds: Vd_DUT - Vs_DUT)
                vd = raw_data.get_trace('V(vd_dut)').get_wave()
                vs = raw_data.get_trace('V(vs_dut)').get_wave()
                vds = vd - vs
                
                # 获取驱动的电压差 (vg_dri - vks) 作为提取起点的判据
                vg_dri = raw_data.get_trace('V(vg_dri)').get_wave()
                vks = raw_data.get_trace('V(vks)').get_wave()
                v_driver = vg_dri - vks
                
                # 获取在准确时刻（T5）通过采样电阻上的电流（取绝对值避免负方向符号影响）
                try:
                    i_rshunt = raw_data.get_trace('I(Rshunt)').get_wave()
                except Exception:
                    i_rshunt = raw_data.get_trace('I(rshunt)').get_wave()
                i_actual_val = abs(np.interp(t5_accurate, time, i_rshunt))
                
                # 获取母线电压 VBUS 作为 Vds 阈值判据依据
                vbus_wave = raw_data.get_trace('V(vbus)').get_wave()
                vbus_val = vbus_wave[0] # 取第一个点即可假设恒定
                vbus_val = 400.0 # 直接使用理论值，仿真中母线电压基本没有降落，可以直接用理论值
                # VCC2 从仿真文件中定义为 18V
                vcc2_val = 18.0
                
            except Exception as e:
                print(f"[{os.path.basename(raw_file_path)}] 找不到波形数据: {e}")
                continue
            
            # 使用母线电压 VBUS 的 90% 作为终点阈值，驱动电压 0.9VCC2 作为起点阈值
            v_start_thresh = 0.9 * vcc2_val
            v_end_thresh = 0.9 * vbus_val
            
            # 获取在指定时间窗口内的点
            in_window = (time >= T_WINDOW_START) & (time <= T_WINDOW_END)
            
            # 寻找关断边沿：驱动电压 v_driver 下降到 v_start_thresh，而 vds 上升到 v_end_thresh
            idx_start_arr = np.where(in_window & (v_driver <= v_start_thresh))[0]
            idx_end_arr = np.where(in_window & (vds >= v_end_thresh))[0]
            
            # 线性插值函数需要传入其对应的波形数组
            def get_interpolated_time(idx_arr, target_v, wave_arr):
                if len(idx_arr) == 0:
                    return None
                i = idx_arr[0]
                if i == 0 or not in_window[i-1]:
                    return time[i]
                
                t1, v1 = time[i-1], wave_arr[i-1]
                t2, v2 = time[i], wave_arr[i]
                
                if v2 == v1:
                    return t2
                return t1 + (target_v - v1) * (t2 - t1) / (v2 - v1)
            
            t_10 = get_interpolated_time(idx_start_arr, v_start_thresh, v_driver)
            t_90 = get_interpolated_time(idx_end_arr, v_end_thresh, vds)
            
            if t_10 is not None and t_90 is not None:
                # 应对可能的提前跨越异常状况，保证 t_90 在 t_10 后才算正确
                if t_90 > t_10:
                    rise_time = t_90 - t_10
                else:
                    rise_time = 0.0
                print(f"VLoad={vload_val}V, I_Target={i_target_val}A : Voltage Rise Time (Raw) = {rise_time*1e9:.2f} ns")
            else:
                rise_time = 0.0
                print(f"VLoad={vload_val}V, I_Target={i_target_val}A : 未能在窗口区 {T_WINDOW_START*1e6:.2f}us~{T_WINDOW_END*1e6:.2f}us 提取到上升时间")
                
            # 保存数据供画图使用 (无论是否抓取到边沿都保存，以便 3D 图以 0 展示)
            results_data.append({
                'vload': vload_val,
                'i_target': i_target_val,
                'i_actual': i_actual_val,
                'time': time,
                'v_out': vds, # 将 Vds 作为主展示波形
                'v_driver': v_driver, # 带上驱动波形
                't_10': t_10,
                't_90': t_90,
                'v_10': v_start_thresh,
                'v_90': v_end_thresh,
                'rise_time': rise_time,
                'window': (T_WINDOW_START, T_WINDOW_END)
            })

    # 根据提取的结果选择调用画图函数
    # plot_waveforms(results_data)
    plot_rise_time_3d(results_data)
    plt.show()

if __name__ == "__main__":
    run_ltspice_simulation()


