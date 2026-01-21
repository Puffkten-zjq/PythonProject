import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib

# 新增：配置中文+英文兼容字体（Windows系统）
matplotlib.rcParams['font.family'] = ['SimHei', 'DejaVu Sans']  # 先中文黑体，后默认英文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

# 后续的 Node 类、Simulation 类等代码不变...
class Node:
    def __init__(self, id, x, y, energy):
        self.id = id
        self.x = x
        self.y = y
        self.energy = energy
        self.initial_energy = energy
        self.alive = True

class Simulation:
    def __init__(self, width, height, num_nodes, initial_energy, sink_x, sink_y, transmission_range):
        self.width = width
        self.height = height
        self.num_nodes = num_nodes
        self.initial_energy = initial_energy
        self.sink_x = sink_x
        self.sink_y = sink_y
        self.transmission_range = transmission_range
        self.nodes = []
        self.initialize_nodes()

        # 能量消耗参数
        self.E_elec = 50e-9  # 电子能量消耗 (J/bit)
        self.epsilon_fs = 10e-12  # 自由空间模型参数
        self.epsilon_mp = 0.0013e-12  # 多路径衰减模型参数
        self.d0 = 87  # 距离阈值
        self.packet_size = 120  # bit

    def initialize_nodes(self):
        # 节点线性均匀分布
        for i in range(self.num_nodes):
            x = i * (self.width / (self.num_nodes - 1)) if self.num_nodes > 1 else self.width / 2
            y = self.height / 2
            node = Node(i, x, y, self.initial_energy)
            self.nodes.append(node)

    def distance(self, node1, node2):
        return np.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

    def energy_consumption(self, d):
        """计算传输能量消耗"""
        if d <= self.d0:
            E_tx = self.E_elec * self.packet_size + self.epsilon_fs * self.packet_size * d**2
        else:
            E_tx = self.E_elec * self.packet_size + self.epsilon_mp * self.packet_size * d**4
        E_rx = self.E_elec * self.packet_size
        return E_tx + E_rx

    def get_neighbors(self, node, d_min=None):
        """获取节点的邻居节点"""
        neighbors = []
        for other in self.nodes:
            if other.id != node.id and other.alive:
                d = self.distance(node, other)
                if d <= self.transmission_range:
                    if d_min is None or d >= d_min:
                        neighbors.append((other, d))
        return neighbors

    def get_neighbors_towards_sink(self, node, d_min=None):
        """获取朝向汇聚节点的邻居节点"""
        neighbors = []
        for other in self.nodes:
            if other.id != node.id and other.alive:
                d_node_sink = self.distance(node, Node(-1, self.sink_x, self.sink_y, 0))
                d_other_sink = self.distance(other, Node(-1, self.sink_x, self.sink_y, 0))
                if d_other_sink < d_node_sink:  # 朝向汇聚节点
                    d = self.distance(node, other)
                    if d <= self.transmission_range:
                        if d_min is None or d >= d_min:
                            neighbors.append((other, d))
        return neighbors

    def mte_route(self, source, d_min=None):
        """最小传输能量路由算法"""
        current = source
        hops = 0
        while True:
            if self.distance(current, Node(-1, self.sink_x, self.sink_y, 0)) <= self.transmission_range:
                # 直接传输到汇聚节点
                energy_cost = self.energy_consumption(self.distance(current, Node(-1, self.sink_x, self.sink_y, 0)))
                current.energy -= energy_cost
                hops += 1
                break

            neighbors = self.get_neighbors(current, d_min)
            if not neighbors:
                return False, hops  # 路由失败

            # 选择最小传输能量的邻居
            min_energy = float('inf')
            next_node = None
            for neighbor, d in neighbors:
                energy_cost = self.energy_consumption(d)
                if energy_cost < min_energy:
                    min_energy = energy_cost
                    next_node = neighbor

            current.energy -= min_energy
            current = next_node
            hops += 1

            if current.energy <= 0:
                current.alive = False
                return False, hops

        return True, hops

    def geraf_route(self, source, d_min=None):
        """基于地理位置的随机转发路由算法"""
        current = source
        hops = 0
        while True:
            if self.distance(current, Node(-1, self.sink_x, self.sink_y, 0)) <= self.transmission_range:
                # 直接传输到汇聚节点
                energy_cost = self.energy_consumption(self.distance(current, Node(-1, self.sink_x, self.sink_y, 0)))
                current.energy -= energy_cost
                hops += 1
                break

            neighbors = self.get_neighbors_towards_sink(current, d_min)
            if not neighbors:
                return False, hops  # 路由失败

            # 随机选择一个朝向汇聚节点的邻居
            next_node, d = random.choice(neighbors)
            energy_cost = self.energy_consumption(d)
            current.energy -= energy_cost
            current = next_node
            hops += 1

            if current.energy <= 0:
                current.alive = False
                return False, hops

        return True, hops

    def ens_or_route(self, source, d_min=None):
        """ENS_OR能量感知地理路由算法"""
        current = source
        hops = 0
        while True:
            if self.distance(current, Node(-1, self.sink_x, self.sink_y, 0)) <= self.transmission_range:
                # 直接传输到汇聚节点
                energy_cost = self.energy_consumption(self.distance(current, Node(-1, self.sink_x, self.sink_y, 0)))
                current.energy -= energy_cost
                hops += 1
                break

            neighbors = self.get_neighbors_towards_sink(current, d_min)
            if not neighbors:
                return False, hops  # 路由失败

            # 计算最优传输距离
            d_opt = np.sqrt(self.epsilon_fs / self.epsilon_mp)

            # ENS_OR选择策略
            best_node = None
            best_score = float('-inf')

            for neighbor, d in neighbors:
                # 考虑距离和剩余能量的综合得分
                distance_factor = 1.0 / (1.0 + abs(d - d_opt))
                energy_factor = neighbor.energy / neighbor.initial_energy
                score = distance_factor * energy_factor

                if score > best_score:
                    best_score = score
                    best_node = neighbor

            energy_cost = self.energy_consumption(self.distance(current, best_node))
            current.energy -= energy_cost
            current = best_node
            hops += 1

            if current.energy <= 0:
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

    def reset(self):
        """重置模拟"""
        self.nodes = []
        self.initialize_nodes()

def run_simulation():
    # 参数设置
    width = 500  # m
    height = 50  # m
    num_nodes = 50
    initial_energy = 0.5  # J
    sink_x = 0  # 汇聚节点在最左端
    sink_y = height / 2
    transmission_range = 100  # m
    simulation_time = 350  # s
    packet_rate = 1  # packet/s

    # 运行三种算法的模拟
    algorithms = ['MTE', 'GeRaF', 'ENS_OR']
    results = {}

    for algo in algorithms:
        sim = Simulation(width, height, num_nodes, initial_energy, sink_x, sink_y, transmission_range)
        are_values = []
        sre_values = []

        for t in range(simulation_time):
            # 每个时间步发送数据包
            alive_nodes = sim.get_alive_nodes()
            if not alive_nodes:
                break

            # 选择源节点（随机选择一个活着的节点）
            source = random.choice(alive_nodes)

            # 执行路由算法
            if algo == 'MTE':
                sim.mte_route(source)
            elif algo == 'GeRaF':
                sim.geraf_route(source)
            elif algo == 'ENS_OR':
                sim.ens_or_route(source)

            # 记录性能指标
            are_values.append(sim.get_are())
            sre_values.append(sim.get_sre())

        results[algo] = {'are': are_values, 'sre': sre_values}

    # 绘制ARE随时间变化
    plt.figure(figsize=(10, 6))
    for algo in algorithms:
        plt.plot(results[algo]['are'], label=algo)
    plt.xlabel('Time(s)')
    plt.ylabel('Average Residual Energy(J)')
    plt.title('剩余能量平均值ARE随时间变化示意图')
    plt.legend()
    plt.grid(True)
    plt.savefig('are_time.png')
    plt.show()

    # 绘制SRE随时间变化
    plt.figure(figsize=(10, 6))
    for algo in algorithms:
        plt.plot(results[algo]['sre'], label=algo)
    plt.xlabel('Time(s)')
    plt.ylabel('Standard Deviation of Residual Energy(J)')
    plt.title('剩余能量的标准差SRE随时间变化示意图')
    plt.legend()
    plt.grid(True)
    plt.savefig('sre_time.png')
    plt.show()

    # 不同d_min下的RPR、FDN、NL比较
    d_min_values = [5, 10, 15, 20, 25]
    rpr_results = {}
    fdn_results = {}
    nl_results = {}

    for algo in algorithms:
        rpr_values = []
        fdn_values = []
        nl_values = []

        for d_min in d_min_values:
            sim = Simulation(width, height, num_nodes, initial_energy, sink_x, sink_y, transmission_range)
            success_count = 0
            total_count = 0
            fdn_time = -1
            nl_time = -1

            for t in range(500):  # 最长模拟时间
                alive_nodes = sim.get_alive_nodes()
                if not alive_nodes:
                    if nl_time == -1:
                        nl_time = t
                    break

                # 检查第一个死亡节点
                if fdn_time == -1 and len(alive_nodes) < num_nodes:
                    fdn_time = t

                # 发送数据包
                source = random.choice(alive_nodes)
                total_count += 1

                if algo == 'MTE':
                    success, _ = sim.mte_route(source, d_min)
                elif algo == 'GeRaF':
                    success, _ = sim.geraf_route(source, d_min)
                elif algo == 'ENS_OR':
                    success, _ = sim.ens_or_route(source, d_min)

                if success:
                    success_count += 1

            # 计算数据包接收率
            rpr = success_count / total_count if total_count > 0 else 0
            rpr_values.append(rpr)

            # 记录FDN和NL
            fdn_values.append(fdn_time if fdn_time != -1 else 500)
            nl_values.append(nl_time if nl_time != -1 else 500)

        rpr_results[algo] = rpr_values
        fdn_results[algo] = fdn_values
        nl_results[algo] = nl_values

    # 绘制RPR随d_min变化
    plt.figure(figsize=(10, 6))
    for algo in algorithms:
        plt.plot(d_min_values, rpr_results[algo], label=algo, marker='o')
    plt.xlabel('The distance between two nearest nodes(m)')
    plt.ylabel('Receiving Packets Ratio')
    plt.title('数据包接收率RPR随d_min变化示意图')
    plt.legend()
    plt.grid(True)
    plt.savefig('rpr_dmin.png')
    plt.show()

    # 绘制FDN随d_min变化
    plt.figure(figsize=(10, 6))
    for algo in algorithms:
        plt.plot(d_min_values, fdn_results[algo], label=algo, marker='o')
    plt.xlabel('The distance between two nearest nodes(m)')
    plt.ylabel('First Dead Node(s)')
    plt.title('第一个死亡节点FDN随d_min变化示意图')
    plt.legend()
    plt.grid(True)
    plt.savefig('fdn_dmin.png')
    plt.show()

    # 绘制NL随d_min变化
    plt.figure(figsize=(10, 6))
    for algo in algorithms:
        plt.plot(d_min_values, nl_results[algo], label=algo, marker='o')
    plt.xlabel('The distance between two nearest nodes(m)')
    plt.ylabel('Network Lifetime(s)')
    plt.title('网络生命周期NL随d_min变化示意图')
    plt.legend()
    plt.grid(True)
    plt.savefig('nl_dmin.png')
    plt.show()

if __name__ == '__main__':
    run_simulation()