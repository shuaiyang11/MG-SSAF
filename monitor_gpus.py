import subprocess
import time

def get_gpu_memory_usage():
    """
    使用nvidia-smi命令获取显卡的显存使用情况
    返回一个列表，包含每张显卡的显存使用量（单位：MB）
    """
    try:
        # 执行nvidia-smi命令获取显存使用信息
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], 
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode != 0:
            raise Exception(f"Error executing nvidia-smi: {result.stderr}")
        
        # 解析输出结果
        gpu_memory_usages = [int(line.strip()) for line in result.stdout.strip().split('\n')]
        return gpu_memory_usages
    except Exception as e:
        print(f"Error: {e}")
        return []

def main():
    threshold = 1000  # 显存使用量阈值（单位：MB）
    check_interval = 60  # 检查间隔时间（单位：秒）
    executed = False  # 标志变量，用于标记是否已执行test.py脚本

    while not executed:
        gpu_memory_usages = get_gpu_memory_usage()
        if len(gpu_memory_usages) != 4:
            print("未检测到4张显卡，请检查显卡配置！")
            time.sleep(check_interval)
            continue

        # 检查所有显卡的显存使用量是否都低于阈值
        if all(memory < threshold for memory in gpu_memory_usages):
            print(f"所有显卡的显存使用量均低于{threshold}MB，执行test.py脚本...")
            try:
                # 执行当前路径下的test.py脚本
                command = ("CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port 13891 main.py --cfg ./configs/DRFormer/DRFormer-Tiny-BigWSize.yaml --data-path ../imagenet/ --batch-size 128")
                # command = ("python monitor_gpus_test.py")
                # 执行命令
                subprocess.run(command, shell=True, check=True)
                print("test.py脚本执行完成！")
                executed = True  # 设置标志变量为True，表示已执行test.py脚本
            except subprocess.CalledProcessError as e:
                print(f"执行test.py脚本时出错：{e}")
        else:
            print(f"当前显卡显存使用情况：{gpu_memory_usages}MB，未达到执行条件。")

        time.sleep(check_interval)

if __name__ == "__main__":
    main()