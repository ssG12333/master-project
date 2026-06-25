from graphviz import Digraph

# 创建一个有向图
dot = Digraph(comment='障碍物生成流程')

# 添加节点
dot.node('A', '开始')
dot.node('B', '输入：区域大小、训练阶段、配置参数')
dot.node('C', '根据阶段设置参数（密度、高度、半径、间距等）')
dot.node('D', '计算期望障碍物数：λ = ρ × A')
dot.node('E', '从泊松分布采样：N ~ Poisson(λ)')
dot.node('F', '对每个障碍物 i = 1 to N')
dot.node('G', '生成随机位置 (x, y) ~ Uniform')
dot.node('H', '从幂律分布生成半径：r ~ PowerLaw(α_r, r_min, r_max)')
dot.node('I', '从幂律分布生成高度：h ~ PowerLaw(α_h, h_min, h_max)')
dot.node('J', '检查边界约束：是否在区域内？')
dot.node('K', '检查间隙约束')
dot.node('L', '重试')
dot.node('M', '达到最大尝试次数？')
dot.node('N', '放弃该障碍物')
dot.node('O', '所有gap满足')
dot.node('P', '添加到障碍物列表')
dot.node('Q', 'i++')
dot.node('R', '继续下一个')
dot.node('S', '返回障碍物列表')
dot.node('T', '结束')

# 添加边
dot.edges(['AB', 'BC', 'CD', 'DE', 'EF', 'FG', 'GH', 'HI', 'IJ', 'JK', 'KL', 'LM', 'MN', 'KO', 'OP', 'PQ', 'QR', 'RS', 'ST'])

# 添加条件分支
dot.edge('J', 'K', label='是')
dot.edge('J', 'L', label='否')
dot.edge('M', 'G', label='否')
dot.edge('M', 'N', label='是')
dot.edge('K', 'L', label='不满足')
dot.edge('K', 'O', label='满足')

# 保存并渲染图表
dot.render('obstacle_generation', format='png', view=True)