# BOM 符合性验证工具

## 技术方向

Excel 数据校验、BOM 对比、桌面 GUI 工具。

## 技术栈

- Python
- tkinter
- openpyxl
- pandas
- Excel 报告输出

## 工作链路

1. 读取 BOM Excel 文件。
2. 按规则对物料、数量、层级或字段一致性进行校验。
3. 生成差异结果和对比报告。
4. 通过 GUI 提供文件选择、校验执行和结果提示。
5. 输出 `BOM_对比报告.xlsx` 便于人工复核。

## 关键内容

- `bom_validator_gui.py`：GUI 和校验主逻辑。
- `bom.xlsx`：示例 BOM。
- `BOM_对比报告.xlsx`：校验结果样例。
- `使用说明.docx`：工具使用说明。

## 后续整理

- 将校验规则抽到配置文件。
- 增加异常输入处理和单元测试。

