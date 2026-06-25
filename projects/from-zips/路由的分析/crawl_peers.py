import re
import csv

def parse_document(file_path):
    """解析文档内容并提取信息"""
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    data = []
    current_collector = None

    for line in lines:
        line = line.strip()
        # 跳过空行或分隔符
        if not line or line.startswith('===='):
            continue

        # 跳过表格标题行（不提取收集器名称）
        if line.startswith('ROUTEVIEWS COLLECTOR'):
            continue

        # 匹配数据行并提取字段
        match = re.match(r'(\S+)\s+(\d+)\s+([\w.:][\w.:/]*)\s+(\d+)\s+\|\s+(\w+)\s+\|\s+(\w+)\s+\|\s+(.+)', line)
        if match:
            collector = match.group(1)          # ROUTEVIEWS COLLECTOR（数据行中的收集器名称）
            as_number = match.group(2)         # AS编号
            peering_address = match.group(3)   # 对接地址（IPv4或IPv6）
            prefixes = match.group(4)          # 前缀数量
            cc = match.group(5)                # 国家代码
            region = match.group(6)            # 区域
            asname = match.group(7)            # AS名称

            # 如果 current_collector 尚未设置或发生变化，更新为当前行的 collector
            if current_collector != collector:
                current_collector = collector
                print(f"更新收集器名称为：{current_collector}")

            # 添加数据到列表
            data.append([current_collector, as_number, peering_address, prefixes, cc, region, asname])
        else:
            print(f"警告：无法匹配数据行：{line}")

    return data

def save_to_csv(data, csv_file):
    """将提取的数据保存到CSV文件"""
    with open(csv_file, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(['ROUTEVIEWS COLLECTOR', 'AS NUMBER', 'PEERING ADDRESS', 'PREFIXES', 'CC', 'REGION', 'ASNAME'])
        # 写入数据行
        writer.writerows(data)

if __name__ == '__main__':
    # 输入和输出文件路径
    file_path = '1.txt'              # 假设文档内容存储在1.txt中
    csv_file = 'routeviews_data.csv' # 输出CSV文件
    # 解析文档并保存到CSV
    data = parse_document(file_path)
    save_to_csv(data, csv_file)
    print(f'数据已保存到 {csv_file}')