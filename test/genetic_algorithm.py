"""
群体智能算法迭代流程
包含选择、交叉、变异等操作的完整实现
"""

import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import copy


@dataclass
class GAConfig:
    """遗传算法配置参数"""
    population_size: int = 50  # 种群大小
    max_generations: int = 100  # 最大迭代次数
    crossover_rate: float = 0.8  # 交叉概率
    mutation_rate: float = 0.1  # 变异概率
    tournament_size: int = 3  # 锦标赛选择大小
    elite_size: int = 2  # 精英保留数量
    selection_pressure: float = 2.0  # 选择压力


class GeneticAlgorithm:
    """遗传算法核心类"""
    
    def __init__(self, config: GAConfig):
        self.config = config
        self.population = []
        self.generation = 0
        self.best_individual = None
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
    def initialize_population(self, feature_pool: List[Dict], 
                            population_size: Optional[int] = None) -> List[List[str]]:
        """初始化种群
        
        Args:
            feature_pool: 特征池，包含所有可用特征
            population_size: 种群大小，如果为None则使用配置中的值
            
        Returns:
            初始化的种群，每个个体是一个特征名称列表
        """
        if population_size is None:
            population_size = self.config.population_size
            
        population = []
        feature_names = [f['feat_name'] for f in feature_pool]
        
        # 生成随机个体
        for i in range(population_size):
            # 随机选择特征数量（最少选5个，最多选所有特征的80%）
            min_features = min(5, len(feature_names))
            max_features = max(min_features, int(len(feature_names) * 0.8))
            num_features = random.randint(min_features, max_features)
            
            # 随机选择特征
            selected_features = random.sample(feature_names, num_features)
            population.append(selected_features)
            
        self.population = population
        return population
    
    def evaluate_fitness(self, individual: List[str], feature_dict: Dict[str, Dict]) -> float:
        """评估个体适应度
        
        Args:
            individual: 个体（特征名称列表）
            feature_dict: 特征字典，key为特征名，value为特征信息
            
        Returns:
            适应度值
        """
        if not individual:
            return 0.0
            
        # 获取选中特征的信息
        selected_features = []
        for feat_name in individual:
            if feat_name in feature_dict:
                selected_features.append(feature_dict[feat_name])
        
        if not selected_features:
            return 0.0
            
        # 计算适应度（这里可以根据实际需求调整）
        # 策略1：平均FeatureScore
        avg_score = np.mean([f.get('FeatureScore', 0) for f in selected_features])
        
        # 策略2：考虑特征数量（避免选择过多特征）
        feature_count_penalty = len(individual) * 0.01  # 特征数量惩罚
        
        # 策略3：考虑特征多样性（避免选择相似特征）
        diversity_bonus = self._calculate_diversity_bonus(selected_features)
        
        fitness = avg_score - feature_count_penalty + diversity_bonus
        
        return max(fitness, 0.0)  # 确保适应度非负
    
    def _calculate_diversity_bonus(self, features: List[Dict]) -> float:
        """计算特征多样性奖励"""
        if len(features) < 2:
            return 0.0
            
        # 简单的多样性计算：基于FeatureScore的标准差
        scores = [f.get('FeatureScore', 0) for f in features]
        diversity = np.std(scores) if len(scores) > 1 else 0.0
        
        return diversity * 0.1  # 多样性奖励系数
    
    def tournament_selection(self, fitness_scores: List[float]) -> int:
        """锦标赛选择
        
        Args:
            fitness_scores: 所有个体的适应度分数
            
        Returns:
            被选中的个体索引
        """
        tournament_indices = random.sample(range(len(fitness_scores)), 
                                         self.config.tournament_size)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        # 返回适应度最高的个体
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return winner_idx
    
    def crossover(self, parent1: List[str], parent2: List[str]) -> Tuple[List[str], List[str]]:
        """交叉操作
        
        Args:
            parent1: 父代个体1
            parent2: 父代个体2
            
        Returns:
            两个子代个体
        """
        if random.random() > self.config.crossover_rate:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
            
        # 确保父代有足够多的特征进行交叉
        if len(parent1) < 2 or len(parent2) < 2:
            return copy.deepcopy(parent1), copy.deepcopy(parent2)
            
        # 单点交叉
        crossover_point1 = random.randint(1, len(parent1) - 1)
        crossover_point2 = random.randint(1, len(parent2) - 1)
        
        # 创建子代
        child1 = parent1[:crossover_point1] + parent2[crossover_point2:]
        child2 = parent2[:crossover_point2] + parent1[crossover_point1:]
        
        # 去重并保持特征顺序
        child1 = list(dict.fromkeys(child1))
        child2 = list(dict.fromkeys(child2))
        
        return child1, child2
    
    def mutate(self, individual: List[str], all_features: List[str]) -> List[str]:
        """变异操作
        
        Args:
            individual: 待变异的个体
            all_features: 所有可用特征列表
            
        Returns:
            变异后的个体
        """
        if random.random() > self.config.mutation_rate:
            return copy.deepcopy(individual)
            
        mutated = copy.deepcopy(individual)
        available_features = [f for f in all_features if f not in mutated]
        
        # 随机选择变异类型
        mutation_type = random.choice(['add', 'remove', 'replace'])
        
        if mutation_type == 'add' and available_features:
            # 添加新特征
            new_feature = random.choice(available_features)
            mutated.append(new_feature)
            
        elif mutation_type == 'remove' and len(mutated) > 1:
            # 移除特征
            remove_idx = random.randint(0, len(mutated) - 1)
            mutated.pop(remove_idx)
            
        elif mutation_type == 'replace' and available_features and mutated:
            # 替换特征
            replace_idx = random.randint(0, len(mutated) - 1)
            new_feature = random.choice(available_features)
            mutated[replace_idx] = new_feature
            
        return mutated
    
    def evolve_generation(self, feature_dict: Dict[str, Dict], 
                         all_features: List[str]) -> List[List[str]]:
        """进化一代
        
        Args:
            feature_dict: 特征字典
            all_features: 所有可用特征列表
            
        Returns:
            新一代种群
        """
        # 评估当前种群适应度
        fitness_scores = []
        for individual in self.population:
            fitness = self.evaluate_fitness(individual, feature_dict)
            fitness_scores.append(fitness)
            
        # 记录统计信息
        best_fitness = max(fitness_scores)
        avg_fitness = np.mean(fitness_scores)
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        
        # 找到最佳个体
        best_idx = np.argmax(fitness_scores)
        self.best_individual = copy.deepcopy(self.population[best_idx])
        
        # 创建新一代种群
        new_population = []
        
        # 精英保留
        elite_indices = np.argsort(fitness_scores)[-self.config.elite_size:]
        for idx in elite_indices:
            new_population.append(copy.deepcopy(self.population[idx]))
            
        # 生成剩余个体
        while len(new_population) < self.config.population_size:
            # 选择父代
            parent1_idx = self.tournament_selection(fitness_scores)
            parent2_idx = self.tournament_selection(fitness_scores)
            
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # 交叉
            child1, child2 = self.crossover(parent1, parent2)
            
            # 变异
            child1 = self.mutate(child1, all_features)
            child2 = self.mutate(child2, all_features)
            
            new_population.extend([child1, child2])
            
        # 截断到指定大小
        self.population = new_population[:self.config.population_size]
        self.generation += 1
        
        return self.population
    
    def run_evolution(self, feature_pool: List[Dict], 
                     max_generations: Optional[int] = None) -> Dict:
        """运行完整的进化过程
        
        Args:
            feature_pool: 特征池
            max_generations: 最大迭代次数，如果为None则使用配置中的值
            
        Returns:
            进化结果字典
        """
        if max_generations is None:
            max_generations = self.config.max_generations
            
        # 构建特征字典
        feature_dict = {f['feat_name']: f for f in feature_pool}
        all_features = list(feature_dict.keys())
        
        # 初始化种群
        self.initialize_population(feature_pool)
        
        print(f"开始进化，种群大小：{self.config.population_size}，最大代数：{max_generations}")
        print(f"特征池大小：{len(feature_pool)}")
        
        # 进化循环
        for generation in range(max_generations):
            self.evolve_generation(feature_dict, all_features)
            
            # 打印进度
            if (generation + 1) % 10 == 0:
                current_best = self.best_fitness_history[-1]
                current_avg = self.avg_fitness_history[-1]
                print(f"第 {generation + 1:3d} 代：最佳适应度 = {current_best:.4f}，平均适应度 = {current_avg:.4f}")
        
        # 返回最终结果
        result = {
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness_history[-1],
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'total_generations': self.generation,
            'selected_features': self.best_individual,
            'selected_feature_count': len(self.best_individual)
        }
        
        return result
    
    def save_evolution_history(self, filepath: str) -> None:
        """保存进化历史"""
        history = {
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'generations': self.generation,
            'best_individual': self.best_individual
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False)


def create_genetic_algorithm(config: Optional[GAConfig] = None) -> GeneticAlgorithm:
    """创建遗传算法实例"""
    if config is None:
        config = GAConfig()
    return GeneticAlgorithm(config)


def run_feature_selection_ga(feature_pool: List[Dict], 
                           config: Optional[GAConfig] = None,
                           max_generations: int = 100) -> Dict:
    """运行特征选择的遗传算法
    
    Args:
        feature_pool: 特征池
        config: 算法配置，如果为None则使用默认配置
        max_generations: 最大迭代次数
        
    Returns:
        选择结果
    """
    ga = create_genetic_algorithm(config)
    result = ga.run_evolution(feature_pool, max_generations)
    return result


if __name__ == "__main__":
    # 示例用法
    print("遗传算法迭代流程设计完成！")
    print("主要功能：")
    print("1. 初始化种群")
    print("2. 评估适应度")
    print("3. 锦标赛选择")
    print("4. 交叉操作")
    print("5. 变异操作")
    print("6. 精英保留")
    print("7. 进化历史记录")