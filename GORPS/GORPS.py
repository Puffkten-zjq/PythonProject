import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

# 配置中文+英文兼容字体
matplotlib.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class Node:
    def __init__(self, id, x, y, energy, is_coop=False):
        self.id = id
        self.x = x
        self.y = y
        self.energy = energy
        self.initial_energy = energy
        self.alive = True
        self.is_coop = is_coop  # 是否为能量协作节点
        self.energy_consumed = 0  # 记录能量消耗
        self.hops = 0  # 记录跳数

class GORPS_Simulation:
    def __init__(self, width, height, num_nodes, initial_energy, sink_x, sink_y, energy_transfer_range):
        self.width = width
        self.height = height
        self.num_nodes = num_nodes
        self.initial_energy = initial_energy
        self.sink_x = sink_x
        self.sink_y = sink_y
        self.energy_transfer_range = energy_transfer_range
        self.nodes = []
        
        # 能量参数（根据论文表6.1）
        self.E_elec = 50e-9  # 电子能量消耗 (J/bit)
        self.epsilon_amp = 100e-12  # 能量传输参数 (J/bit/m²)
        self.B = 1024  # 数据包大小 (bit)
        self.d_min = 4.5  # 最小传输距离 (m)
        self.d_max = 9.0  # 最大传输距离 (m)
        self.packet_rate = 1  # 数据包发送速率 (packet/s)
        self.E_max = 10.0  # 节点最大能量 (J)
        self.E_min = 0.0  # 节点最小能量 (J)
        
        # 太阳能收集参数
        self.solar_power = 1000  # 太阳辐射功率 (W/m²)
        self.panel_area = 0.1  # 太阳能板面积 (m²)
        self.solar_efficiency = 0.22  # 太阳能转换效率 (22%)
        
        # 能量传输参数
        self.energy_transfer_efficiency = 0.5  # 无线能量传输效率 (50%)
        
        # 时间参数
        self.total_time = 24  # 仿真时间 (小时)
        self.time_step = 1  # 时间步长 (分钟)
        self.time_steps_per_hour = 60  # 每小时的时间步数
        
    def initialize_nodes_graphene(self):
        """基于石墨烯结构的节点部署策略"""
        self.nodes = []
        node_id = 0
        
        # 计算石墨烯结构的参数
        num_layers = int(np.sqrt(self.num_nodes / 6)) + 1
        r0 = 5.0  # 初始半径
        
        for layer in range(num_layers):
            n_nodes = 6 * (layer + 1)  # 每层的节点数
            angle_step = 2 * np.pi / n_nodes
            radius = r0 * (layer + 1)
            
            for i in range(n_nodes):
                if node_id >= self.num_nodes:
                    break
                    
                angle = i * angle_step + layer * np.pi / 6  # 每层旋转一定角度
                x = self.width / 2 + radius * np.cos(angle)
                y = self.height / 2 + radius * np.sin(angle)
                
                # 交替设置为能量协作节点和能量接收节点
                is_coop = (node_id % 2 == 0)
                node = Node(node_id, x, y, self.initial_energy, is_coop)
                self.nodes.append(node)
                node_id += 1
        
        # 如果节点数不足，添加随机分布的节点
        while node_id < self.num_nodes:
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            is_coop = (node_id % 2 == 0)
            node = Node(node_id, x, y, self.initial_energy, is_coop)
            self.nodes.append(node)
            node_id += 1
    
    def initialize_nodes_triangle(self):
        """基于正三角形结构的节点部署策略（用于EHOR算法）"""
        self.nodes = []
        node_id = 0
        
        # 计算正三角形网格参数
        grid_size = int(np.sqrt(self.num_nodes)) + 1
        spacing = self.width / (grid_size + 1)
        
        for i in range(grid_size):
            for j in range(grid_size):
                if node_id >= self.num_nodes:
                    break
                    
                x = (i + 0.5) * spacing
                y = (j + 0.5) * spacing + (i % 2) * spacing / 2
                
                # 交替设置为能量协作节点和能量接收节点
                is_coop = (node_id % 2 == 0)
                node = Node(node_id, x, y, self.initial_energy, is_coop)
                self.nodes.append(node)
                node_id += 1
        
        # 如果节点数不足，添加随机分布的节点
        while node_id < self.num_nodes:
            x = random.uniform(0, self.width)
            y = random.uniform(0, self.height)
            is_coop = (node_id % 2 == 0)
            node = Node(node_id, x, y, self.initial_energy, is_coop)
            self.nodes.append(node)
            node_id += 1
    
    def distance(self, node1, node2):
        return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
    
    def energy_consumption(self, d):
        """计算数据传输能量消耗"""
        E_tx = self.E_elec * self.B + self.epsilon_amp * self.B * d**2
        E_rx = self.E_elec * self.B
        return E_tx + E_rx
    
    def energy_collection(self, node, time_step):
        """计算节点收集的太阳能，基于论文中的太阳能辐射曲线"""
        if not node.is_coop:
            return 0
        
        # 模拟太阳能辐射曲线（基于时间的正弦曲线）
        hour = (time_step / self.time_steps_per_hour) % 24
        
        # 基于论文中描述的太阳能辐射曲线
        if 6 <= hour < 20:
            # 太阳能辐射活跃期
            t = hour - 6
            solar_factor = np.sin((t / 14) * np.pi)  # 从6点到20点的正弦曲线
            collected_energy = self.solar_power * self.panel_area * self.solar_efficiency * solar_factor * (self.time_step / 60)
        else:
            # 夜晚无太阳能
            collected_energy = 0
        
        return collected_energy
    
    def energy_transfer(self, source, target, energy_amount):
        """能量协作节点向能量接收节点传输能量"""
        if not source.is_coop or source.id == target.id or not source.alive or not target.alive:
            return False
        
        d = self.distance(source, target)
        if d > self.energy_transfer_range:
            return False
        
        # 计算实际可传输的能量
        available_energy = source.energy - self.E_min
        if available_energy <= 0:
            return False
        
        actual_transfer = min(energy_amount, available_energy)
        received_energy = actual_transfer * self.energy_transfer_efficiency
        
        # 更新节点能量
        source.energy -= actual_transfer
        target.energy += received_energy
        
        # 限制能量上限
        if target.energy > self.E_max:
            target.energy = self.E_max
        
        return True
    
    def energy_balance_management(self, time_step):
        """GORPS算法的能量均衡管理机制"""
        coop_nodes = [node for node in self.nodes if node.is_coop and node.alive]
        recv_nodes = [node for node in self.nodes if not node.is_coop and node.alive]
        
        if not coop_nodes or not recv_nodes:
            return
        
        # 按剩余能量排序
        recv_nodes.sort(key=lambda x: x.energy)
        coop_nodes.sort(key=lambda x: x.energy, reverse=True)
        
        # 能量均衡分配
        for recv_node in recv_nodes:
            # 计算该节点需要的能量
            needed_energy = self.E_max - recv_node.energy
            
            if needed_energy <= 0:
                continue
            
            # 寻找最佳能量源
            for coop_node in coop_nodes:
                if coop_node.energy <= self.E_min:
                    continue
                
                # 计算可传输的能量
                transfer_energy = min(needed_energy / self.energy_transfer_efficiency, 
                                    coop_node.energy - self.E_min)
                
                if transfer_energy <= 0:
                    continue
                
                # 执行能量传输
                if self.energy_transfer(coop_node, recv_node, transfer_energy):
                    needed_energy -= transfer_energy * self.energy_transfer_efficiency
                    
                    if needed_energy <= 0:
                        break
    
    def non_uniform_charging(self, time_step):
        """EHOR算法的非均匀充电策略"""
        coop_nodes = [node for node in self.nodes if node.is_coop and node.alive]
        recv_nodes = [node for node in self.nodes if not node.is_coop and node.alive]
        
        if not coop_nodes or not recv_nodes:
            return
        
        # 按剩余能量排序，优先为能量低的节点充电
        recv_nodes.sort(key=lambda x: x.energy)
        
        for recv_node in recv_nodes:
            # 计算该节点需要的能量
            needed_energy = self.E_max - recv_node.energy
            
            if needed_energy <= 0:
                continue
            
            # 寻找最近的协作节点
            nearest_coop = None
            min_distance = float('inf')
            
            for coop_node in coop_nodes:
                d = self.distance(coop_node, recv_node)
                if d < min_distance and coop_node.energy > self.E_min:
                    min_distance = d
                    nearest_coop = coop_node
            
            if nearest_coop is not None:
                # 传输能量
                self.energy_transfer(nearest_coop, recv_node, needed_energy)
    
    def get_neighbors(self, node):
        """获取节点的邻居节点"""
        neighbors = []
        for other in self.nodes:
            if other.id != node.id and other.alive:
                d = self.distance(node, other)
                if self.d_min <= d <= self.d_max:
                    neighbors.append((other, d))
        return neighbors
    
    def get_neighbors_towards_sink(self, node):
        """获取朝向汇聚节点的邻居节点"""
        neighbors = []
        sink_node = Node(-1, self.sink_x, self.sink_y, 0)
        for other in self.nodes:
            if other.id != node.id and other.alive:
                d_node_sink = self.distance(node, sink_node)
                d_other_sink = self.distance(other, sink_node)
                if d_other_sink < d_node_sink:  # 朝向汇聚节点
                    d = self.distance(node, other)
                    if self.d_min <= d <= self.d_max:
                        neighbors.append((other, d))
        return neighbors
    
    def gorps_route(self, source):
        """GORPS路由算法"""
        current = source
        hops = 0
        
        while True:
            # 检查是否到达汇聚节点
            sink_node = Node(-1, self.sink_x, self.sink_y, 0)
            if self.distance(current, sink_node) <= self.d_max:
                # 直接传输到汇聚节点
                energy_cost = self.energy_consumption(self.distance(current, sink_node))
                current.energy -= energy_cost
                current.energy_consumed += energy_cost
                hops += 1
                break

            # 获取朝向汇聚节点的邻居
            neighbors = self.get_neighbors_towards_sink(current)
            if not neighbors:
                return False, hops  # 路由失败

            # GORPS选择策略
            best_node = None
            best_score = float('-inf')

            for neighbor, d in neighbors:
                # 考虑距离、剩余能量和能量均衡的综合得分
                distance_factor = 1.0 / (d + 1)
                energy_factor = neighbor.energy / neighbor.initial_energy
                
                # 能量均衡因子（基于与平均能量的差异）
                avg_energy = self.get_are()
                balance_factor = 1.0 - abs(neighbor.energy - avg_energy) / (self.E_max - self.E_min)
                
                # 路由贡献因子
                sink_node = Node(-1, self.sink_x, self.sink_y, 0)
                d_neighbor_sink = self.distance(neighbor, sink_node)
                progress_factor = (self.distance(current, sink_node) - d_neighbor_sink) / self.distance(current, sink_node)
                
                score = distance_factor * energy_factor * balance_factor * progress_factor

                if score > best_score:
                    best_score = score
                    best_node = neighbor

            # 传输数据
            energy_cost = self.energy_consumption(self.distance(current, best_node))
            current.energy -= energy_cost
            current.energy_consumed += energy_cost
            current = best_node
            hops += 1

            if current.energy <= self.E_min:
                current.alive = False
                return False, hops

        return True, hops
    
    def ehor_route(self, source):
        """EHOR路由算法"""
        current = source
        hops = 0
        
        while True:
            # 检查是否到达汇聚节点
            sink_node = Node(-1, self.sink_x, self.sink_y, 0)
            if self.distance(current, sink_node) <= self.d_max:
                # 直接传输到汇聚节点
                energy_cost = self.energy_consumption(self.distance(current, sink_node))
                current.energy -= energy_cost
                current.energy_consumed += energy_cost
                hops += 1
                break

            # 获取朝向汇聚节点的邻居
            neighbors = self.get_neighbors_towards_sink(current)
            if not neighbors:
                return False, hops  # 路由失败

            # EHOR选择策略
            best_node = None
            min_energy_cost = float('inf')

            for neighbor, d in neighbors:
                # 基于最小能量消耗选择
                energy_cost = self.energy_consumption(d)
                if energy_cost < min_energy_cost:
                    min_energy_cost = energy_cost
                    best_node = neighbor

            # 传输数据
            energy_cost = self.energy_consumption(self.distance(current, best_node))
            current.energy -= energy_cost
            current.energy_consumed += energy_cost
            current = best_node
            hops += 1

            if current.energy <= self.E_min:
                current.alive = False
                return False, hops

        return True, hops
    
    def get_alive_nodes(self):
        return [node for node in self.nodes if node.alive]
    
    def get_are(self):
        """计算剩余能量平均值"""
        alive_nodes = self.get_alive_nodes()
        if not alive_nodes:
            return 0
        return sum(node.energy for node in alive_nodes) / len(alive_nodes)
    
    def get_sre(self):
        """计算剩余能量标准差"""
        alive_nodes = self.get_alive_nodes()
        if len(alive_nodes) <= 1:
            return 0
        are = self.get_are()
        variance = sum((node.energy - are)**2 for node in alive_nodes) / len(alive_nodes)
        return np.sqrt(variance)
    
    def get_ahc(self, time_step, route_func):
        """计算平均跳数"""
        if not self.get_alive_nodes():
            return 0
        
        # 随机选择10个节点计算平均跳数
        sample_nodes = random.sample(self.get_alive_nodes(), min(10, len(self.get_alive_nodes())))
        total_hops = 0
        success_count = 0
        
        for node in sample_nodes:
            success, hops = route_func(node)
            if success:
                total_hops += hops
                success_count += 1
        
        return total_hops / success_count if success_count > 0 else 0

def run_gorps_simulation():
    """运行GORPS算法仿真"""
    # 参数设置（根据论文表6.1）
    width = 100.0  # m
    height = 100.0  # m
    num_nodes = 484  # 总节点数（241个能量协作节点 + 241个能量接收节点）
    initial_energy = 10.0  # J
    sink_x = width / 2  # 汇聚节点位于中心
    sink_y = height / 2
    energy_transfer_range = 20.0  # m
    
    # 时间设置
    total_hours = 24  # 24小时仿真
    time_steps_per_hour = 60
    total_time_steps = total_hours * time_steps_per_hour
    
    # 创建GORPS仿真对象
    sim_gorps = GORPS_Simulation(width, height, num_nodes, initial_energy, sink_x, sink_y, energy_transfer_range)
    sim_gorps.initialize_nodes_graphene()
    
    # 创建EHOR仿真对象
    sim_ehor = GORPS_Simulation(width, height, num_nodes, initial_energy, sink_x, sink_y, energy_transfer_range)
    sim_ehor.initialize_nodes_triangle()
    
    # 记录性能指标
    are_values_gorps = []  # GORPS剩余能量平均值
    sre_values_gorps = []  # GORPS剩余能量标准差
    ahc_values_gorps = []  # GORPS平均跳数
    
    are_values_ehor = []  # EHOR剩余能量平均值
    sre_values_ehor = []  # EHOR剩余能量标准差
    ahc_values_ehor = []  # EHOR平均跳数
    
    # 运行仿真
    for t in range(total_time_steps):
        # GORPS仿真
        # 1. 收集能量
        for node in sim_gorps.nodes:
            if node.alive:
                collected_energy = sim_gorps.energy_collection(node, t)
                node.energy += collected_energy
                if node.energy > sim_gorps.E_max:
                    node.energy = sim_gorps.E_max
        
        # 2. 执行能量均衡管理
        sim_gorps.energy_balance_management(t)
        
        # 3. 发送数据包
        for _ in range(sim_gorps.packet_rate):
            alive_nodes = sim_gorps.get_alive_nodes()
            if alive_nodes:
                source = random.choice(alive_nodes)
                sim_gorps.gorps_route(source)
        
        # 4. 记录性能指标
        are_values_gorps.append(sim_gorps.get_are())
        sre_values_gorps.append(sim_gorps.get_sre())
        ahc_values_gorps.append(sim_gorps.get_ahc(t, sim_gorps.gorps_route))
        
        # EHOR仿真
        # 1. 收集能量
        for node in sim_ehor.nodes:
            if node.alive:
                collected_energy = sim_ehor.energy_collection(node, t)
                node.energy += collected_energy
                if node.energy > sim_ehor.E_max:
                    node.energy = sim_ehor.E_max
        
        # 2. 执行非均匀充电
        sim_ehor.non_uniform_charging(t)
        
        # 3. 发送数据包
        for _ in range(sim_ehor.packet_rate):
            alive_nodes = sim_ehor.get_alive_nodes()
            if alive_nodes:
                source = random.choice(alive_nodes)
                sim_ehor.ehor_route(source)
        
        # 4. 记录性能指标
        are_values_ehor.append(sim_ehor.get_are())
        sre_values_ehor.append(sim_ehor.get_sre())
        ahc_values_ehor.append(sim_ehor.get_ahc(t, sim_ehor.ehor_route))
        
        # 输出进度
        if t % (time_steps_per_hour * 2) == 0:
            hour = t // time_steps_per_hour
            print(f"仿真时间: {hour}小时, GORPS存活节点数: {len(sim_gorps.get_alive_nodes())}, EHOR存活节点数: {len(sim_ehor.get_alive_nodes())}")
    
    # 绘制仿真结果
    time_hours = np.arange(total_time_steps) / time_steps_per_hour
    
    # 1. 剩余能量平均值ARE随时间变化
    plt.figure(figsize=(10, 6))
    plt.plot(time_hours, are_values_gorps, 'r-', label='GORPS-Graphene')
    plt.plot(time_hours, are_values_ehor, 'b--', label='EHOR-Triangle')
    plt.xlabel('时间 (小时)')
    plt.ylabel('剩余能量平均值 (J)')
    plt.title('剩余能量平均值ARE随时间变化示意图')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('are_time_gorps_vs_ehor.png')
    plt.show()
    
    # 2. 剩余能量标准差SRE随时间变化
    plt.figure(figsize=(10, 6))
    plt.plot(time_hours, sre_values_gorps, 'r-', label='GORPS-Graphene')
    plt.plot(time_hours, sre_values_ehor, 'b--', label='EHOR-Triangle')
    plt.xlabel('时间 (小时)')
    plt.ylabel('剩余能量标准差 (J)')
    plt.title('剩余能量标准差SRE随时间变化示意图')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('sre_time_gorps_vs_ehor.png')
    plt.show()
    
    # 3. 平均跳数AHC随时间变化
    plt.figure(figsize=(10, 6))
    plt.plot(time_hours, ahc_values_gorps, 'r-', label='GORPS-Graphene')
    plt.plot(time_hours, ahc_values_ehor, 'b--', label='EHOR-Triangle')
    plt.xlabel('时间 (小时)')
    plt.ylabel('平均跳数')
    plt.title('平均跳数AHC随时间变化示意图')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('ahc_time_gorps_vs_ehor.png')
    plt.show()
    
    # 4. 节点22的剩余能量变化情况
    plt.figure(figsize=(10, 6))
    node_id = 22
    plt.plot(time_hours[:60], [sim_gorps.nodes[node_id].energy for _ in range(60)], 'r-', label='GORPS-Graphene at 10 am')
    plt.plot(time_hours[:60], [sim_ehor.nodes[node_id].energy for _ in range(60)], 'b--', label='EHOR-Triangle at 10 am')
    plt.plot(time_hours[120:180], [sim_gorps.nodes[node_id].energy for _ in range(60)], 'g-', label='GORPS-Graphene at 12 am')
    plt.plot(time_hours[120:180], [sim_ehor.nodes[node_id].energy for _ in range(60)], 'y--', label='EHOR-Triangle at 12 am')
    plt.xlabel('时间 (分钟)')
    plt.ylabel('剩余能量 (J)')
    plt.title('节点22的剩余能量变化情况')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('node22_energy_change.png')
    plt.show()
    
    # 5. 节点240的剩余能量变化情况
    plt.figure(figsize=(10, 6))
    node_id = 240
    plt.plot(time_hours[:60], [sim_gorps.nodes[node_id].energy for _ in range(60)], 'r-', label='GORPS-Graphene at 10 am')
    plt.plot(time_hours[:60], [sim_ehor.nodes[node_id].energy for _ in range(60)], 'b--', label='EHOR-Triangle at 10 am')
    plt.plot(time_hours[120:180], [sim_gorps.nodes[node_id].energy for _ in range(60)], 'g-', label='GORPS-Graphene at 12 am')
    plt.plot(time_hours[120:180], [sim_ehor.nodes[node_id].energy for _ in range(60)], 'y--', label='EHOR-Triangle at 12 am')
    plt.xlabel('时间 (分钟)')
    plt.ylabel('剩余能量 (J)')
    plt.title('节点240的剩余能量变化情况')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('node240_energy_change.png')
    plt.show()
    
    # 6. GORPS节点剩余能量三维分布（10 am）
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for node in sim_gorps.nodes:
        color = 'green' if node.alive else 'red'
        ax.scatter(node.x, node.y, node.energy, c=color, s=50, alpha=0.6)
    
    ax.set_xlabel('X坐标 (m)')
    ax.set_ylabel('Y坐标 (m)')
    ax.set_zlabel('剩余能量 (J)')
    ax.set_title('GORPS算法在10 am时的节点剩余能量分布情况')
    plt.tight_layout()
    plt.savefig('energy_distribution_gorps_10am.png')
    plt.show()
    
    # 7. EHOR节点剩余能量三维分布（10 am）
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for node in sim_ehor.nodes:
        color = 'green' if node.alive else 'red'
        ax.scatter(node.x, node.y, node.energy, c=color, s=50, alpha=0.6)
    
    ax.set_xlabel('X坐标 (m)')
    ax.set_ylabel('Y坐标 (m)')
    ax.set_zlabel('剩余能量 (J)')
    ax.set_title('EHOR算法在10 am时的节点剩余能量分布情况')
    plt.tight_layout()
    plt.savefig('energy_distribution_ehor_10am.png')
    plt.show()

if __name__ == '__main__':
    run_gorps_simulation()
