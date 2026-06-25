import os
import shutil
import random
from datetime import datetime  # 用于打印时间戳


def log(message):
    """自定义日志函数，带时间戳"""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{now}] {message}")


def main():
    # ------------------- 配置参数（根据实际情况修改） -------------------
    # 原始数据根目录（包含 test_list.txt、train_val_list.txt 和 images 文件夹）
    DATA_ROOT = "D:/PycharmProjects/PythonProject1/TB_dataset"  # 修改为实际路径
    # 输出目录（划分后的训练/验证/测试集存放位置）
    OUTPUT_ROOT = os.path.join(DATA_ROOT, "dataset")  # 输出目录设为原始数据下的dataset文件夹
    # 影像文件存放的父目录（包含多个子文件夹如images_001）
    IMAGE_PARENT_DIR = os.path.join(DATA_ROOT, "images")  # 指向包含子文件夹的images目录

    # 随机种子（保证划分可复现）
    RANDOM_SEED = 42

    # ------------------- 步骤1：创建输出目录 -------------------
    log("正在创建输出目录...")
    try:
        os.makedirs(OUTPUT_ROOT, exist_ok=True)
        train_dir = os.path.join(OUTPUT_ROOT, "train")
        val_dir = os.path.join(OUTPUT_ROOT, "val")
        test_dir = os.path.join(OUTPUT_ROOT, "test")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        log(f"输出目录创建成功：{OUTPUT_ROOT}")
    except Exception as e:
        log(f"创建输出目录失败！错误：{e}")
        return

    # ------------------- 步骤2：读取文件列表 -------------------
    log("\n正在读取测试集和训练验证集文件列表...")
    try:
        # 读取测试集文件列表（test_list.txt）
        test_list_path = os.path.join(DATA_ROOT, "test_list.txt")
        if not os.path.exists(test_list_path):
            log(f"错误：测试集列表文件 {test_list_path} 不存在！")
            return
        with open(test_list_path, "r", encoding="utf-8") as f:
            test_files = [line.strip() for line in f.readlines() if line.strip()]
        log(f"读取到测试集文件数量：{len(test_files)}")

        # 读取训练+验证集文件列表（train_val_list.txt）
        train_val_list_path = os.path.join(DATA_ROOT, "train_val_list.txt")
        if not os.path.exists(train_val_list_path):
            log(f"错误：训练验证集列表文件 {train_val_list_path} 不存在！")
            return
        with open(train_val_list_path, "r", encoding="utf-8") as f:
            train_val_files = [line.strip() for line in f.readlines() if line.strip()]
        log(f"读取到训练验证集文件数量：{len(train_val_files)}")

    except Exception as e:
        log(f"读取文件列表失败！错误：{e}")
        return

    # ------------------- 步骤2.5：构建文件名到完整路径的映射（处理子文件夹） -------------------
    # 扫描子文件夹（简化日志）
    log("\n正在扫描图像子文件夹...")
    file_path_map = {file: os.path.join(root, file)
                        for root, _, files in os.walk(IMAGE_PARENT_DIR)
                        for file in files}
    log(f"扫描到 {len(file_path_map)} 个图像文件")
    # 转换文件列表为完整路径（处理可能包含子文件夹路径的文件名）
    def convert_to_full_path(file_list):
        full_paths = []
        for file in file_list:
            # 优先使用带路径的文件名（如images_001/file.jpg）
            if os.path.isfile(os.path.join(IMAGE_PARENT_DIR, file)):
                full_paths.append(os.path.join(IMAGE_PARENT_DIR, file))
            # 否则通过文件名映射查找（适用于纯文件名场景）
            elif file in file_path_map:
                full_paths.append(file_path_map[file])
            else:
                log(f"警告：未找到文件路径 → {file}")
        return full_paths

    test_full_paths = convert_to_full_path(test_files)
    train_val_full_paths = convert_to_full_path(train_val_files)

    # ------------------- 步骤3：划分训练集和验证集 -------------------
    log("\n正在划分训练集和验证集（8:2）...")
    try:
        random.seed(RANDOM_SEED)
        random.shuffle(train_val_full_paths)
        total = len(train_val_full_paths)
        train_num = int(total * 0.8)
        train_files = train_val_full_paths[:train_num]
        val_files = train_val_full_paths[train_num:]
        log(f"训练集文件数量：{len(train_files)}，验证集文件数量：{len(val_files)}")
    except Exception as e:
        log(f"划分训练验证集失败！错误：{e}")
        return

    # ------------------- 步骤4：复制文件到输出目录 -------------------
    log("\n开始复制文件到输出目录...")

    def copy_files(file_list, target_dir):
        """通用文件复制函数（处理完整路径文件）"""
        success_count = 0
        fail_count = 0
        for src_path in file_list:
            file_name = os.path.basename(src_path)  # 提取文件名
            dst_path = os.path.join(target_dir, file_name)

            # 检查原始文件是否存在
            if not os.path.exists(src_path):
                log(f"警告：原始文件不存在，跳过复制 → {src_path}")
                fail_count += 1
                continue

            # 检查目标路径是否有写入权限（可选）
            if not os.access(os.path.dirname(dst_path), os.W_OK):
                log(f"警告：目标目录无写入权限，跳过复制 → {dst_path}")
                fail_count += 1
                continue

            # 执行复制
            try:
                shutil.copy(src_path, dst_path)
                success_count += 1
                #log(f"复制成功 → {src_path} → {dst_path}")
            except Exception as e:
                log(f"复制失败！错误：{e} → {src_path}")
                fail_count += 1
        return success_count, fail_count

    # 复制测试集（使用完整路径列表）
    log("\n===== 复制测试集 =====")
    test_success, test_fail = copy_files(test_full_paths, test_dir)
    log(f"测试集复制完成：成功 {test_success} 个，失败 {test_fail} 个")

    # 复制训练集
    log("\n===== 复制训练集 =====")
    train_success, train_fail = copy_files(train_files, train_dir)
    log(f"训练集复制完成：成功 {train_success} 个，失败 {train_fail} 个")

    # 复制验证集
    log("\n===== 复制验证集 =====")
    val_success, val_fail = copy_files(val_files, val_dir)
    log(f"验证集复制完成：成功 {val_success} 个，失败 {val_fail} 个")

    log("\n数据集划分完成！")


if __name__ == "__main__":
    main()