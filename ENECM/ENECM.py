import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib

# 配置中文+英文兼容字体
matplotlib.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class Node:
    def __init__(self, id, x, y, energy):
        self.id = id
        self.x = x
        self.y = y
        self.energy = energy
        self.initial_energy = energy
        self.alive = True
        self.energy_consumed = 0  # 记录能量消耗

class ENECM_Simulation:
    def __init__(self, width, height, num_nodes, initial_energy, sink_x, sink_y, transmission_range, has_eh=False, has_ecm=False):
        self.width = width
        self.height = height
        self.num_nodes = num_nodes
        self.initial_energy = initial_energy
        self.sink_x = sink_x
        self.sink_y = sink_y
        self.transmission_range = transmission_range
        self.has_eh = has_eh  # 是否具有能量收集功能
        self.has_ecm = has_ecm  # 是否具有能量均衡管理机制
        self.nodes = []
        self.initialize_nodes()
        
        # 能量参数（根据论文表5.1）
        self.E_elec = 50e-9  # 电子能量消耗 (J/bit)
        self.epsilon_amp = 100e-12  # 能量传输参数 (J/bit/m²)
        self.tau = 2  # 能量收集系数
        self.B = 1024  # 数据包大小 (bit)
        self.R = 50.0  # 能量补充范围 (m)
        self.d_min = 5.0  # 最小传输距离 (m)
        self.d_max = 17.0  # 最大传输距离 (m)
        self.packet_rate = 1  # 数据包发送速率 (packet/s)
        
        # 太阳能收集参数
        self.solar_power = 1000  # 太阳辐射功率 (W/m²)
        self.panel_area = 0.1  # 太阳能板面积 (m²)
        self.solar_efficiency = 0.15  # 太阳能转换效率
        self.E_max = 7.2  # 节点最大能量 (J)，与初始能量相同
        self.E_min = 0.0  # 节点最小能量 (J)
    
    def initialize_nodes(self):
        # 一维网络拓扑：节点均匀分布在一条直线上
        for i in range(self.num_nodes):
            x = i * (self.width / (self.num_nodes - 1)) if self.num_nodes > 1 else self.width / 2
            y = self.height / 2
            node = Node(i, x, y, self.initial_energy)
            self.nodes.append(node)
    
    def distance(self, node1, node2):
        return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)
    
    def energy_consumption(self, d):
        """计算数据传输能量消耗"""
        E_tx = self.E_elec * self.B + self.epsilon_amp * self.B * d**2
        E_rx = self.E_elec * self.B
        return E_tx + E_rx
    
    def energy_collection(self, node, time_step):
        """计算节点收集的太阳能"""
        if not self.has_eh:
            return 0
        
        # 简化的太阳能收集模型，基于时间和节点位置
        time_factor = np.sin((time_step / 100) * np.pi)  # 模拟一天中的太阳辐射变化
        position_factor = 0.8 + 0.2 * np.random.random()  # 位置影响因子
        collected_energy = self.solar_power * self.panel_area * self.solar_efficiency * time_factor * position_factor * 0.01
        return collected_energy
    
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
        sink_node = Node(-1, self.sink_x, self.sink_y, 0)
        for other in self.nodes:
            if other.id != node.id and other.alive:
                d_node_sink = self.distance(node, sink_node)
                d_other_sink = self.distance(other, sink_node)
                if d_other_sink < d_node_sink:  # 朝向汇聚节点
                    d = self.distance(node, other)
                    if d <= self.transmission_range and d >= self.d_min:
                        neighbors.append((other, d))
        return neighbors
    
    def ens_or_route(self, source):
        """ENS_OR能量感知地理路由算法"""
        current = source
        hops = 0
        
        while True:
            # 检查是否到达汇聚节点
            sink_node = Node(-1, self.sink_x, self.sink_y, 0)
            if self.distance(current, sink_node) <= self.transmission_range:
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

            # ENS_OR选择策略
            best_node = None
            best_score = float('-inf')

            for neighbor, d in neighbors:
                # 考虑距离和剩余能量的综合得分
                distance_factor = 1.0 / (d + 1)
                energy_factor = neighbor.energy / neighbor.initial_energy
                score = distance_factor * energy_factor

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
    
    def enecm_route(self, source):
        """ENECM能量均衡管理路由算法"""
        current = source
        hops = 0
        
        while True:
            # 检查是否到达汇聚节点
            sink_node = Node(-1, self.sink_x, self.sink_y, 0)
            if self.distance(current, sink_node) <= self.transmission_range:
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

            # ENECM选择策略（考虑能量均衡管理）
            best_node = None
            best_score = float('-inf')

            for neighbor, d in neighbors:
                # 考虑距离、剩余能量和能量均衡的综合得分
                distance_factor = 1.0 / (d + 1)
                energy_factor = neighbor.energy / neighbor.initial_energy
                
                # 能量均衡因子（如果启用了能量均衡管理）
                if self.has_ecm:
                    # 计算节点的能量均衡度
                    avg_energy = self.get_are()
                    balance_factor = 1.0 - abs(neighbor.energy - avg_energy) / (self.E_max - self.E_min)
                    score = distance_factor * energy_factor * balance_factor
                else:
                    score = distance_factor * energy_factor

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
    
    def energy_balance_management(self):
        """能量均衡管理机制"""
        if not self.has_ecm:
            return
        
        # 计算网络平均能量
        avg_energy = self.get_are()
        
        # 能量传输范围
        transfer_range = self.R
        
        # 寻找能量过剩和能量不足的节点
        surplus_nodes = []
        deficit_nodes = []
        
        for node in self.nodes:
            if not node.alive:
                continue
            
            if node.energy > avg_energy + 0.5:
                surplus_nodes.append(node)
            elif node.energy < avg_energy - 0.5:
                deficit_nodes.append(node)
        
        # 执行能量传输
        for surplus_node in surplus_nodes:
            for deficit_node in deficit_nodes:
                d = self.distance(surplus_node, deficit_node)
                if d <= transfer_range:
                    # 计算可传输的能量
                    transfer_amount = min(surplus_node.energy - avg_energy, avg_energy - deficit_node.energy)
                    
                    if transfer_amount > 0:
                        surplus_node.energy -= transfer_amount
                        deficit_node.energy += transfer_amount
    
    def get_alive_nodes(self):
        return [node for node in self.nodes if node.alive]
    
    def get_are(self):
        """计算剩余能量平均值"""
        alive_nodes = self.get_alive_nodes()
        if not alive_nodes:
            return 0
        return sum(node.energy for node in alive_nodes) / len(alive_nodes)
    
    def get_fdn(self):
        """获取第一个死亡节点的时间"""
        for i, node in enumerate(self.nodes):
            if not node.alive:
                return i
        return -1  # 没有死亡节点
    
    def reset(self):
        """重置模拟"""
        self.nodes = []
        self.initialize_nodes()

def run_one_dimensional_simulation():
    # 参数设置（根据论文表5.1）
    width = 2500  # m
    height = 50  # m
    num_nodes_list = [25, 50, 75, 100, 125]  # 节点总数
    d_min_list = [5, 7, 9, 11, 13, 15, 17]  # 节点间距
    initial_energy = 7.2  # J
    sink_x = width  # 汇聚节点在最右端
    sink_y = height / 2
    transmission_range = 50  # m
    simulation_time = 80  # s
    packet_rate = 1  # packet/s
    
    # 定义三种算法
    algorithms = [
        {'name': 'ENS_OR', 'has_eh': False, 'has_ecm': False},
        {'name': 'ENS_OR_EH', 'has_eh': True, 'has_ecm': False},
        {'name': 'ENECM', 'has_eh': True, 'has_ecm': True}
    ]
    
    # 1. 节点数量对ARE的影响（图5.4）
    print("运行节点数量对ARE的影响仿真...")
    are_node_count_results = {}
    
    for algo in algorithms:
        are_values = []
        for num_nodes in num_nodes_list:
            sim = ENECM_Simulation(width, height, num_nodes, initial_energy, sink_x, sink_y, transmission_range, 
                                  algo['has_eh'], algo['has_ecm'])
            
            for t in range(simulation_time):
                # 收集能量
                if algo['has_eh']:
                    for node in sim.nodes:
                        if node.alive:
                            collected_energy = sim.energy_collection(node, t)
                            node.energy += collected_energy
                            if node.energy > sim.E_max:
                                node.energy = sim.E_max
                
                # 执行能量均衡管理
                if algo['has_ecm']:
                    sim.energy_balance_management()
                
                # 发送数据包
                alive_nodes = sim.get_alive_nodes()
                if alive_nodes:
                    source = alive_nodes[0]  # 源节点固定为第一个节点
                    if algo['name'] == 'ENS_OR':
                        sim.ens_or_route(source)
                    else:
                        sim.enecm_route(source)
            
            are_values.append(sim.get_are())
        are_node_count_results[algo['name']] = are_values
    
    # 绘制节点数量对ARE的影响
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']
    
    for i, algo in enumerate(algorithms):
        plt.errorbar(num_nodes_list, are_node_count_results[algo['name']], 
                    yerr=0.2, fmt=f'{markers[i]}-', color=colors[i], label=algo['name'], capsize=5)
    
    plt.xlabel('Number of Nodes')
    plt.ylabel('Average Residual Energy(J)')
    plt.title('剩余能量平均值ARE随节点数量变化示意图')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('are_node_count_enecm.png')
    plt.show()
    
    # 2. 节点间距对ARE的影响（图5.5）
    print("运行节点间距对ARE的影响仿真...")
    num_nodes = 50  # 固定节点数量为50
    are_dmin_results = {}
    
    for algo in algorithms:
        are_values = []
        for d_min in d_min_list:
            # 重新计算网络宽度以保持节点间距
            new_width = (num_nodes - 1) * d_min
            sim = ENECM_Simulation(new_width, height, num_nodes, initial_energy, new_width, sink_y, transmission_range, 
                                  algo['has_eh'], algo['has_ecm'])
            sim.d_min = d_min
            sim.d_max = d_min + 2
            
            for t in range(simulation_time):
                # 收集能量
                if algo['has_eh']:
                    for node in sim.nodes:
                        if node.alive:
                            collected_energy = sim.energy_collection(node, t)
                            node.energy += collected_energy
                            if node.energy > sim.E_max:
                                node.energy = sim.E_max
                
                # 执行能量均衡管理
                if algo['has_ecm']:
                    sim.energy_balance_management()
                
                # 发送数据包
                alive_nodes = sim.get_alive_nodes()
                if alive_nodes:
                    source = alive_nodes[0]  # 源节点固定为第一个节点
                    if algo['name'] == 'ENS_OR':
                        sim.ens_or_route(source)
                    else:
                        sim.enecm_route(source)
            
            are_values.append(sim.get_are())
        are_dmin_results[algo['name']] = are_values
    
    # 绘制节点间距对ARE的影响
    plt.figure(figsize=(10, 6))
    
    for i, algo in enumerate(algorithms):
        plt.errorbar(d_min_list, are_dmin_results[algo['name']], 
                    yerr=0.2, fmt=f'{markers[i]}-', color=colors[i], label=algo['name'], capsize=5)
    
    plt.xlabel('The distance between two nearest nodes(m)')
    plt.ylabel('Average Residual Energy(J)')
    plt.title('剩余能量平均值ARE随节点间距变化示意图')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('are_dmin_enecm.png')
    plt.show()
    
    # 3. ARE随时间变化（图5.6）
    print("运行ARE随时间变化仿真...")
    num_nodes = 50
    d_min = 7
    new_width = (num_nodes - 1) * d_min
    are_time_results = {}
    
    for algo in algorithms:
        sim = ENECM_Simulation(new_width, height, num_nodes, initial_energy, new_width, sink_y, transmission_range, 
                              algo['has_eh'], algo['has_ecm'])
        sim.d_min = d_min
        sim.d_max = d_min + 2
        are_values = []
        
        for t in range(simulation_time):
            # 收集能量
            if algo['has_eh']:
                for node in sim.nodes:
                    if node.alive:
                        collected_energy = sim.energy_collection(node, t)
                        node.energy += collected_energy
                        if node.energy > sim.E_max:
                            node.energy = sim.E_max
            
            # 执行能量均衡管理
            if algo['has_ecm']:
                sim.energy_balance_management()
            
            # 发送数据包
            alive_nodes = sim.get_alive_nodes()
            if alive_nodes:
                source = alive_nodes[0]  # 源节点固定为第一个节点
                if algo['name'] == 'ENS_OR':
                    sim.ens_or_route(source)
                else:
                    sim.enecm_route(source)
            
            are_values.append(sim.get_are())
        are_time_results[algo['name']] = are_values
    
    # 绘制ARE随时间变化
    plt.figure(figsize=(10, 6))
    
    for i, algo in enumerate(algorithms):
        plt.plot(range(simulation_time), are_time_results[algo['name']], 
                marker=markers[i], color=colors[i], label=algo['name'])
    
    plt.xlabel('Time(s)')
    plt.ylabel('Average Residual Energy(J)')
    plt.title('节点总数为50以及节点间距为7m的情况下ARE值随时间变化过程示意图')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('are_time_enecm.png')
    plt.show()
    
    # 4. 节点剩余能量分布（图5.7）
    print("运行节点剩余能量分布仿真...")
    num_nodes = 50
    d_min = 7
    new_width = (num_nodes - 1) * d_min
    time_point = 30  # 30秒时的能量分布
    energy_dist_results = {}
    
    for algo in algorithms:
        sim = ENECM_Simulation(new_width, height, num_nodes, initial_energy, new_width, sink_y, transmission_range, 
                              algo['has_eh'], algo['has_ecm'])
        sim.d_min = d_min
        sim.d_max = d_min + 2
        
        for t in range(time_point):
            # 收集能量
            if algo['has_eh']:
                for node in sim.nodes:
                    if node.alive:
                        collected_energy = sim.energy_collection(node, t)
                        node.energy += collected_energy
                        if node.energy > sim.E_max:
                            node.energy = sim.E_max
            
            # 执行能量均衡管理
            if algo['has_ecm']:
                sim.energy_balance_management()
            
            # 发送数据包
            alive_nodes = sim.get_alive_nodes()
            if alive_nodes:
                source = alive_nodes[0]  # 源节点固定为第一个节点
                if algo['name'] == 'ENS_OR':
                    sim.ens_or_route(source)
                else:
                    sim.enecm_route(source)
        
        # 记录节点能量
        energy_dist = [node.energy if node.alive else 0 for node in sim.nodes]
        energy_dist_results[algo['name']] = energy_dist
    
    # 绘制节点剩余能量分布
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for i, algo in enumerate(algorithms):
        axes[i].bar(range(num_nodes), energy_dist_results[algo['name']], color=colors[i])
        axes[i].set_title(f'{algo["name"]}')
        axes[i].set_xlabel('Relay node')
        axes[i].set_ylabel('Residual Energy(J)')
        axes[i].set_ylim(0, 10)
    
    plt.suptitle('时间为30s时三个算法的各个节点剩余能量分布情况示意图')
    plt.tight_layout()
    plt.savefig('energy_distribution_enecm.png')
    plt.show()
    
    # 5. 节点数量对FDN的影响（图5.8）
    print("运行节点数量对FDN的影响仿真...")
    fdn_node_count_results = {}
    simulation_time = 1800  # 1800秒
    
    for algo in algorithms:
        fdn_values = []
        for num_nodes in num_nodes_list:
            sim = ENECM_Simulation(width, height, num_nodes, initial_energy, sink_x, sink_y, transmission_range, 
                                  algo['has_eh'], algo['has_ecm'])
            
            fdn = -1
            for t in range(simulation_time):
                # 收集能量
                if algo['has_eh']:
                    for node in sim.nodes:
                        if node.alive:
                            collected_energy = sim.energy_collection(node, t)
                            node.energy += collected_energy
                            if node.energy > sim.E_max:
                                node.energy = sim.E_max
                
                # 执行能量均衡管理
                if algo['has_ecm']:
                    sim.energy_balance_management()
                
                # 发送数据包
                alive_nodes = sim.get_alive_nodes()
                if alive_nodes:
                    source = alive_nodes[0]  # 源节点固定为第一个节点
                    if algo['name'] == 'ENS_OR':
                        sim.ens_or_route(source)
                    else:
                        sim.enecm_route(source)
                
                # 检查是否有节点死亡
                current_fdn = sim.get_fdn()
                if current_fdn != -1 and fdn == -1:
                    fdn = t
                    break
            
            fdn_values.append(fdn if fdn != -1 else simulation_time)
        fdn_node_count_results[algo['name']] = fdn_values
    
    # 绘制节点数量对FDN的影响
    plt.figure(figsize=(10, 6))
    
    for i, algo in enumerate(algorithms):
        plt.bar(np.array(range(len(num_nodes_list))) + i*0.3, fdn_node_count_results[algo['name']], 
                width=0.3, color=colors[i], label=algo['name'])
    
    plt.xlabel('Number of Nodes')
    plt.ylabel('First Dead Node(s)')
    plt.title('第一个死亡节点FDN随节点数量变化示意图')
    plt.xticks(np.array(range(len(num_nodes_list))) + 0.3, num_nodes_list)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('fdn_node_count_enecm.png')
    plt.show()

if __name__ == '__main__':
    run_one_dimensional_simulation()
