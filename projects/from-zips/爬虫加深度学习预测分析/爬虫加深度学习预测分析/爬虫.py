import requests
import pandas as pd
from bs4 import BeautifulSoup
import os
import re
import time
from datetime import datetime
from io import StringIO
import traceback

# 配置参数：多个公司信息
companies = [
    {"stock_code": "002039", "company_name": "黔源电力"},
    {"stock_code": "600868", "company_name": "梅雁吉祥"},
    {"stock_code": "600900", "company_name": "长江电力"},
    {"stock_code": "600025", "company_name": "华能水电"},
    {"stock_code": "600674", "company_name": "川投能源"},
    {"stock_code": "600236", "company_name": "桂冠电力"},
    {"stock_code": "600886", "company_name": "国投电力"},
    {"stock_code": "000601", "company_name": "韶能股份"},
    {"stock_code": "000993", "company_name": "闽东电力"},
    {"stock_code": "000722", "company_name": "湖南发展"}
]

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3',
    'Referer': 'https://s.askci.com/'
}


def fetch_with_retry(url, retries=3):
    for i in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            return response.text
        except Exception as e:
            if i == retries - 1:
                raise
            print(f"请求失败，正在重试 ({i + 1}/{retries})...")
            time.sleep(3)
    return None


def clean_numeric(value):
    """增强型数据清洗"""
    try:
        cleaned = str(value).strip().replace('--', '').replace(',', '').replace('－', '-')
        if cleaned == '':
            return None
        return float(cleaned)
    except Exception as e:
        print(f"数据清洗失败: {value} | 错误：{str(e)}")
        return None


def parse_financial_table(html):
    soup = BeautifulSoup(html, 'html.parser')
    main_table = soup.find('table', {'class': 'tb'}) or soup.find('table')

    if not main_table:
        return pd.DataFrame()

    try:
        headers = []
        for row in main_table.find_all('tr'):
            th_cells = row.find_all(['th', 'td'])
            if th_cells:
                headers = [cell.get_text(strip=True) for cell in th_cells]
                break

        data = []
        for row in main_table.find_all('tr')[1:]:
            cells = row.find_all('td')
            if len(cells) == len(headers):
                data.append([cell.get_text(strip=True) for cell in cells])

        df = pd.DataFrame(data, columns=headers)
        indicator_col = df.columns[0]
        df.columns = [col.replace('\n', '').replace('\t', '').strip() for col in df.columns]

        numeric_cols = [col for col in df.columns if col != indicator_col]
        for col in numeric_cols:
            df[col] = df[col].apply(clean_numeric)

        result = []
        for _, row in df.iterrows():
            indicator = row[indicator_col]
            for date_col in numeric_cols:
                result.append({
                    '报告日期': date_col,
                    '指标名称': indicator,
                    '数值': row[date_col]
                })

        return pd.DataFrame(result)

    except Exception as e:
        print(f"表格解析失败: {str(e)}")
        return pd.DataFrame()


# 更新后的指标映射表（新增“市值”和“支付的股利”）
INDICATOR_MAP = {
    # 现金流量表
    r'经营.*现金流': '经营活动现金流净额',
    r'分配.*股利|股息支付': '股息支付',
    r'支付给职工以及为职工支付的现金': '职工薪酬现金',

    # 利润表
    r'营业总?收入': '营业收入',
    r'净利.*母': '净利润',
    r'基本每股收益': '每股收益',
    r'营业成本': '营业成本',

    # 资产负债表
    r'资产合计': '总资产',
    r'负债合计': '总负债',
    r'所有者权益合计': '净资产/市值',
    r'应付股利': '支付的股利',  # 新增
    # 新增
}


def normalize_indicator(name):
    """增强型指标名称标准化"""
    name = re.sub(r'\*|\s', '', str(name))
    for pattern, new_name in INDICATOR_MAP.items():
        if re.search(pattern, name, re.IGNORECASE):
            return new_name
    return name


def transform_final_data(raw_df, stock_code, company_name):
    raw_df['指标名称'] = raw_df['指标名称'].apply(normalize_indicator)
    filtered = raw_df[raw_df['指标名称'].isin(INDICATOR_MAP.values())]

    wide_df = filtered.pivot_table(
        index='报告日期',
        columns='指标名称',
        values='数值',
        aggfunc='first'
    ).reset_index()

    numeric_cols = wide_df.select_dtypes(include=[float]).columns
    wide_df[numeric_cols] = wide_df[numeric_cols] / 10000

    wide_df.columns = [
        f"{col}（元）" if col == '每股收益' else f"{col}（亿元）" if col != '报告日期' else col
        for col in wide_df.columns
    ]

    wide_df.insert(0, '公司代码', stock_code)
    wide_df.insert(0, '公司名称', company_name)

    return wide_df


def main():
    for company in companies:
        stock_code = company["stock_code"]
        company_name = company["company_name"]

        try:
            urls = {
                'cash_flow': f'https://s.askci.com/StockInfo/FinancialReport/CashFlow/?stockCode={stock_code}&dateRange=,6',
                'income': f'https://s.askci.com/StockInfo/FinancialReport/Profit/?stockCode={stock_code}&dateRange=,6',
                'balance': f'https://s.askci.com/StockInfo/FinancialReport/BalanceSheet/?stockCode={stock_code}&dateRange=,6'
            }

            all_data = []

            for report_type, url in urls.items():
                print(f'正在抓取 {company_name} - {report_type} 报表...')
                try:
                    html = fetch_with_retry(url)
                    if not html:
                        continue

                    df = parse_financial_table(html)
                    print(f"\n{report_type} 报表解析到 {len(df)} 行数据")

                    all_data.append(df)

                    os.makedirs('html_backup', exist_ok=True)
                    with open(f'html_backup/{stock_code}_{report_type}.html', 'w', encoding='utf-8') as f:
                        f.write(html)

                    time.sleep(1)

                except Exception as e:
                    print(f"{report_type} 处理失败：{str(e)}")
                    continue

            if not all_data:
                raise ValueError("未获取到有效数据")

            raw_df = pd.concat(all_data, ignore_index=True)
            final_df = transform_final_data(raw_df, stock_code, company_name)

            output_file = f"{stock_code}_财务数据_{datetime.now().strftime('%Y%m%d')}.xlsx"
            final_df.to_excel(output_file, index=False)

            print(f"\n数据已成功保存至：{output_file}")
            print("最新数据预览：")
            print(final_df.tail(5).to_string(index=False))

        except Exception as e:
            print(f"程序运行失败 {company_name} ({stock_code}): {str(e)}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
