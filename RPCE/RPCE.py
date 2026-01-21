import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

# 配置中文+英文兼容字体
matplotlib.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class RPCE_Node:
    def __init__(self, id, x, y, energy, node_type, region_id):
        self.id = id
        self.x = x
        self.y = y
        self.energy = energy
        self.initial_energy = energy
        self.alive = True
        self.node_type = node_type  # 'cooperative' 或 'receiver'
        self.region_id = region_id
        self.energy_consumed = 0  # 记录能量消耗
        self.hops = 0  # 记录跳数

class RPCE_Simulation:
    def __init__(self, width, height, num_coop_nodes, num_recv_nodes, initial_energy, sink_x, sink_y, energy_transfer_range):
        self.width = width
        self.height = height
        self.num_coop_nodes = num_coop_nodes
        self.num_recv_nodes = num_recv_nodes
        self.total_nodes = num_coop_nodes + num_recv_nodes
        self.initial_energy = initial_energy
        self.sink_x = sink_x
        self.sink_y = sink_y
        self.energy_transfer_range = energy_transfer_range
        self.nodes = []
        self.initialize_nodes()
        
        # 能量参数（根据论文表3.1）
        self.E_elec = 50e-9  # 电子能量消耗 (J/bit)
        self.epsilon_amp = 100e-12  # 能量传输参数 (J/bit/m²)
        self.tau = 2  # 能量收集系数
        self.E_min = 4.0  # 最小能量 (J)
        self.E_max = 10.0  # 最大能量 (J)
        self.d_min = 1.0  # 最小传输距离 (m)
        self.d_max = 10.0  # 最大传输距离 (m)
        self.R = 20.0  # 能量补充范围 (m)
        self.lambda_param = 0.5  # 路径损耗指数
        self.eta = 0.7  # 能量转换效率
        self.packet_size = 1024  # 数据包大小 (bit)
        self.packet_rate = 1  # 数据包发送速率 (packet/s)
        
        # 太阳能收集参数
        self.solar_power = 1000  # 太阳辐射功率 (W/m²)
        self.panel_area = 0.1  # 太阳能板面积 (m²)
        self.solar_efficiency = 0.15  # 太阳能转换效率
        
        # 路由参数
        self.transmission_range = 50  # 数据传输范围 (m)
    
    def initialize_nodes(self):
        # 1. 初始化能量协作节点（正三角形网格分布）
        node_id = 0
        region_id = 1
        grid_size = int(np.sqrt(self.num_coop_nodes))
        delta_x = self.width / (grid_size + 1)
        delta_y = self.height / (grid_size + 1)
        
        for i in range(grid_size):
            for j in range(grid_size):
                if node_id >= self.num_coop_nodes:
                    break
                # 正三角形网格分布
                x = (i + 1) * delta_x
                y = (j + 1) * delta_y + (i % 2) * delta_y / 2
                node = RPCE_Node(node_id, x, y, self.initial_energy, 'cooperative', region_id)
                self.nodes.append(node)
                node_id += 1
        
        # 2. 初始化能量接收节点（随机分布）
        for i in range(self.num_recv_nodes):
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            node = RPCE_Node(node_id, x, y, self.initial_energy, 'receiver', region_id)
            self.nodes.append(node)
            node_id += 1
    
    def distance(self, node1, node2):
        return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
    
    def energy_consumption(self, d):
        """计算数据传输能量消耗"""
        E_tx = self.E_elec * self.packet_size + self.epsilon_amp * self.packet_size * d**2
        E_rx = self.E_elec * self.packet_size
        return E_tx + E_rx
    
    def energy_collection(self, node, time_step):
        """计算节点收集的太阳能"""
        # 简化的太阳能收集模型，基于时间和节点位置
        time_factor = np.sin((time_step / 100) * np.pi)  # 模拟一天中的太阳辐射变化
        position_factor = 0.8 + 0.2 * np.random.random()  # 位置影响因子
        collected_energy = self.solar_power * self.panel_area * self.solar_efficiency * time_factor * position_factor * 0.01
        return collected_energy
    
    def wireless_energy_transfer(self, sender, receiver, d):
        """无线能量传输"""
        if d > self.energy_transfer_range:
            return 0
        
        # 计算可传输的能量
        available_energy = sender.energy - self.E_min
        if available_energy <= 0:
            return 0
        
        # 能量传输损耗
        transfer_loss = 1 / (d**self.lambda_param)
        transfer_energy = min(available_energy, (self.E_max - receiver.energy) / self.eta)
        
        # 执行能量传输
        sender.energy -= transfer_energy
        receiver.energy += transfer_energy * self.eta
        
        # 确保能量不超过最大值
        if receiver.energy > self.E_max:
            receiver.energy = self.E_max
        
        return transfer_energy
    
    def get_neighbors(self, node):
        """获取节点的邻居节点"""
        neighbors = []
        for other in self.nodes:
            if other.id != node.id and other.alive:
                d = self.distance(node, other)
                if d <= self.transmission_range and d >= self.d_min:
                    neighbors.append((other, d))
        return neighbors
    
    def get_neighbors_towards_sink(self, node):
        """获取朝向汇聚节点的邻居节点"""
        neighbors = []
        sink_node = RPCE_Node(-1, self.sink_x, self.sink_y, 0, 'sink', 0)
        for other in self.nodes:
            if other.id != node.id and other.alive:
                d_node_sink = self.distance(node, sink_node)
                d_other_sink = self.distance(other, sink_node)
                if d_other_sink < d_node_sink:  # 朝向汇聚节点
                    d = self.distance(node, other)
                    if d <= self.transmission_range and d >= self.d_min:
                        neighbors.append((other, d))
        return neighbors
    
    def route_data(self, source):
        """路由数据到汇聚节点"""
        current = source
        hops = 0
        max_hops = 20
        
        while hops < max_hops:
            sink_dist = self.distance(current, RPCE_Node(-1, self.sink_x, self.sink_y, 0, 'sink', 0))
            if sink_dist <= self.transmission_range:
                # 直接传输到汇聚节点
                energy_cost = self.energy_consumption(sink_dist)
                if current.energy >= energy_cost:
                    current.energy -= energy_cost
                    current.energy_consumed += energy_cost
                    hops += 1
                    return True, hops
                else:
                    return False, hops
            
            # 获取朝向汇聚节点的邻居
            neighbors = self.get_neighbors_towards_sink(current)
            if not neighbors:
                return False, hops
            
            # 选择下一跳节点（基于距离和剩余能量）
            min_cost = float('inf')
            next_node = None
            for neighbor, d in neighbors:
                cost = self.energy_consumption(d) / neighbor.energy
                if cost < min_cost:
                    min_cost = cost
                    next_node = neighbor
            
            if not next_node:
                return False, hops
            
            # 传输数据
            energy_cost = self.energy_consumption(self.distance(current, next_node))
            if current.energy >= energy_cost:
                current.energy -= energy_cost
                current.energy_consumed += energy_cost
                current = next_node
                hops += 1
            else:
                return False, hops
        
        return False, hops
    
    def non_uniform_charging(self, time_step):
        """执行非均匀充电策略"""
        sink_node = RPCE_Node(-1, self.sink_x, self.sink_y, 0, 'sink', 0)
        
        # 为每个能量协作节点执行充电
        for coop_node in [node for node in self.nodes if node.node_type == 'cooperative' and node.alive]:
            # 获取充电范围内的节点
            charging_candidates = []
            for other in self.nodes:
                if other.id != coop_node.id and other.alive:
                    d = self.distance(coop_node, other)
                    if d <= self.R:
                        # 计算充电优先级
                        dist_to_sink = self.distance(other, sink_node)
                        energy_usage_rate = other.energy_consumed / (time_step + 1) if time_step > 0 else 0
                        # 优先级 = 1/(距离目的节点的距离) + 能量消耗率
                        priority = (1.0 / (dist_to_sink + 1)) + energy_usage_rate
                        charging_candidates.append((other, d, priority))
            
            # 按优先级排序
            charging_candidates.sort(key=lambda x: x[2], reverse=True)
            
            # 为候选节点充电
            for receiver, d, priority in charging_candidates:
                self.wireless_energy_transfer(coop_node, receiver, d)
    
    def get_alive_nodes(self):
        return [node for node in self.nodes if node.alive]
    
    def get_are(self):
        """计算剩余能量平均值"""
        alive_nodes = self.get_alive_nodes()
        if not alive_nodes:
            return 0
        return sum(node.energy for node in alive_nodes) / len(alive_nodes)
    
    def get_ahc(self, time_step):
        """计算节点平均跳数"""
        alive_nodes = self.get_alive_nodes()
        if not alive_nodes:
            return 0
        
        total_hops = 0
        success_count = 0
        
        for node in alive_nodes:
            success, hops = self.route_data(node)
            if success:
                total_hops += hops
                success_count += 1
        
        return total_hops / success_count if success_count > 0 else 0
    
    def reset(self):
        """重置模拟"""
        self.nodes = []
        self.initialize_nodes()

def run_rpce_simulation():
    # 参数设置（根据论文表3.1）
    width = 450.0  # 部署区域 (m)
    height = 450.0
    num_coop_nodes = 241  # 能量协作节点数量
    num_recv_nodes = 241  # 能量接收节点数量
    initial_energy = 4.0  # 节点初始能量 (J)
    sink_x = width  # 汇聚节点位置
    sink_y = height / 2
    energy_transfer_range = 20.0  # 能量传输范围 (m)
    simulation_time = 60  # 模拟时间 (分钟)
    time_steps_per_minute = 10  # 每分钟的时间步
    total_time_steps = simulation_time * time_steps_per_minute
    
    # 不同节点数量的模拟
    node_counts = [144, 256, 400]
    time_points = [10, 30, 50]  # 分钟
    
    # 存储结果
    are_results = {count: [] for count in node_counts}
    ahc_results = {count: [] for count in node_counts}
    
    # 运行不同节点数量的模拟
    for count in node_counts:
        # 计算能量协作节点和能量接收节点的数量
        coop_count = int(count / 2) + 1
        recv_count = int(count / 2)
        
        for time_point in time_points:
            sim = RPCE_Simulation(width, height, coop_count, recv_count, initial_energy, sink_x, sink_y, energy_transfer_range)
            
            # 运行模拟到指定时间点
            for t in range(time_point * time_steps_per_minute):
                # 收集能量
                for node in sim.nodes:
                    if node.alive:
                        collected_energy = sim.energy_collection(node, t)
                        node.energy += collected_energy
                        if node.energy > sim.E_max:
                            node.energy = sim.E_max
                
                # 执行非均匀充电
                sim.non_uniform_charging(t)
                
                # 发送数据包
                for _ in range(sim.packet_rate):
                    alive_nodes = sim.get_alive_nodes()
                    if alive_nodes:
                        source = random.choice(alive_nodes)
                        sim.route_data(source)
            
            # 记录性能指标
            are_results[count].append(sim.get_are())
            ahc_results[count].append(sim.get_ahc(time_point * time_steps_per_minute))

    # 修复后的代码，移除position参数，使用正确的分组柱状图绘制方法
    # 绘制剩余能量平均值ARE随时间和节点数量变化示意图
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red']
    labels = ['RPCE10', 'RPCE30', 'RPCE50']

    # 将节点数量转换为数值型x轴位置
    x = np.arange(len(node_counts))
    width = 0.25  # 柱状图宽度

    for i, time_point in enumerate(time_points):
        are_values = [are_results[count][i] for count in node_counts]
        plt.bar(x + i * width, are_values, color=colors[i], label=labels[i], width=width)

    plt.xlabel('节点数量')
    plt.ylabel('平均剩余能量 (J)')
    plt.title('剩余能量平均值ARE随时间和节点数量变化示意图')
    plt.xticks(x + width, [str(count) for count in node_counts])  # 设置x轴标签
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('are_node_count.png')
    plt.show()

    # 绘制节点平均跳数AHC随时间和节点数量变化示意图
    plt.figure(figsize=(10, 6))

    for i, time_point in enumerate(time_points):
        ahc_values = [ahc_results[count][i] for count in node_counts]
        plt.bar(x + i * width, ahc_values, color=colors[i], label=labels[i], width=width)

    plt.xlabel('节点数量')
    plt.ylabel('平均跳数')
    plt.title('节点平均跳数AHC随时间和节点数量变化示意图')
    plt.xticks(x + width, [str(count) for count in node_counts])  # 设置x轴标签
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('ahc_node_count.png')
    plt.show()
    
    # 绘制节点剩余能量分布三维图（节点数量为144，运行60分钟）
    count = 144
    coop_count = int(count / 2) + 1
    recv_count = int(count / 2)
    
    sim = RPCE_Simulation(width, height, coop_count, recv_count, initial_energy, sink_x, sink_y, energy_transfer_range)
    
    for t in range(60 * time_steps_per_minute):
        # 收集能量
        for node in sim.nodes:
            if node.alive:
                collected_energy = sim.energy_collection(node, t)
                node.energy += collected_energy
                if node.energy > sim.E_max:
                    node.energy = sim.E_max
        
        # 执行非均匀充电
        sim.non_uniform_charging(t)
        
        # 发送数据包
        for _ in range(sim.packet_rate):
            alive_nodes = sim.get_alive_nodes()
            if alive_nodes:
                source = random.choice(alive_nodes)
                sim.route_data(source)
    
    # 绘制三维图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    node_ids = [node.id for node in sim.nodes]
    energies = [node.energy for node in sim.nodes]
    
    # 为每个节点创建一个时间序列（简化为60分钟）
    times = np.array([60] * len(sim.nodes))
    
    # 绘制三维柱状图
    ax.bar3d(node_ids, times, np.zeros(len(sim.nodes)), 1, 1, energies, shade=True, color=plt.cm.jet(np.array(energies)/sim.E_max))
    
    ax.set_xlabel('节点ID')
    ax.set_ylabel('时间 (分钟)')
    ax.set_zlabel('剩余能量 (J)')
    ax.set_title('节点数量为144时各个中继节点剩余能量分布情况示意图')
    
    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=plt.cm.jet, norm=plt.Normalize(vmin=0, vmax=sim.E_max))
    sm._A = []
    fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5)
    
    plt.tight_layout()
    plt.savefig('energy_distribution_3d.png')
    plt.show()

if __name__ == '__main__':
    run_rpce_simulation()
