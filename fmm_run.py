import os
import re
import sys
import argparse
import yaml
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime

from pathlib import Path
from pdb import set_trace
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mutual_info_score
from sklearn.metrics import f1_score, precision_score, recall_score
from imblearn.metrics import geometric_mean_score
from scipy.stats import skew, kurtosis
from scipy.stats import entropy
from glob import glob
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))


def resolve_project_path(path_value):
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str(BASE_DIR / path)

from operators.feature_scoring import FeatureScoring
from operators.feature_selector import FeatureSelector
from operators.feature_generation import generate_new_feature_value
from operators.meta_crossover import JsonCrossover
from operators.meta_mutation import JsonMutation
from utils.model_voice import print_buddha_finish
from utils.log import get_logger, add_file_handler

def safe_multiclass_metrics(y_true, feature_values):
    """为多分类问题计算特征质量指标"""
    
    # --- 核心修复函数：确保数据不是 MaskedArray ---
    def to_clean_array(arr):
        # 如果是 MaskedArray，填充 mask 并转为普通 array
        if np.ma.is_masked(arr):
            # 使用 NaN 填充被 mask 的值（或者你可以根据业务逻辑选其他值）
            return np.array(arr.filled(np.nan))
        # 强制转换为普通 array (np.asarray 会保留 MaskedArray 属性，所以用 np.array)
        return np.array(arr)

    # 1. 清洗输入数据
    y_true_clean = to_clean_array(y_true)
    feature_values_clean = to_clean_array(feature_values)
    
    # 移除 NaN 值以防止计算错误（如果 fillna(med) 后仍有极端情况）
    # 注意：这步是可选的，取决于你是否允许 NaN 进入计算
    valid_mask = ~np.isnan(feature_values_clean) & ~np.isnan(y_true_clean)
    if not valid_mask.all():
        y_true_clean = y_true_clean[valid_mask]
        feature_values_clean = feature_values_clean[valid_mask]

    mi = np.nan
    normalized_gain = np.nan

    # 计算互信息（作为特征重要性指标）
    try:
        if len(np.unique(feature_values_clean)) > 20:
            bins = min(10, len(np.unique(feature_values_clean)))
            # 使用分位数离散化
            # 修复点：这里传入的一定是普通 ndarray，不会再报 partition 警告
            quantiles = np.quantile(feature_values_clean, np.linspace(0, 1, bins + 1)[1:-1])
            # np.digitize 如果传入空数组会报错，加个判断
            if len(quantiles) > 0:
                feature_discrete = np.digitize(feature_values_clean, quantiles)
            else:
                feature_discrete = np.zeros_like(feature_values_clean)
        else:
            feature_discrete = feature_values_clean
        
        mi = mutual_info_score(y_true_clean, feature_discrete)
    except Exception as e:
        # print(f"MI Error: {e}") # Debug only
        mi = np.nan
    
    # 计算信息增益
    try:
        # 复用上面的离散化逻辑或 feature_discrete
        # 如果上面没有计算成功，这里重新计算
        f_discrete = feature_discrete

        # 创建特征值到类别分布的映射
        value_to_class_probs = {}
        unique_f_vals = np.unique(f_discrete)
        
        for val in unique_f_vals:
            mask = f_discrete == val
            mask_sum = mask.sum()
            if mask_sum > 0:
                class_counts = {}
                # 只切片对应的 y
                y_subset = y_true_clean[mask]
                unique_sub_y, counts_sub_y = np.unique(y_subset, return_counts=True)
                for cls, count in zip(unique_sub_y, counts_sub_y):
                    class_counts[cls] = count / mask_sum
                value_to_class_probs[val] = class_counts
        
        # 计算整体熵
        unique_classes, class_counts = np.unique(y_true_clean, return_counts=True)
        overall_probs = class_counts / len(y_true_clean)
        overall_probs = np.clip(overall_probs, 1e-10, 1) # 避免 log(0)
        overall_probs = overall_probs / overall_probs.sum()
        overall_entropy = entropy(overall_probs)
        
        # 计算条件熵
        conditional_entropies = []
        weights = []
        
        # 使用所有的唯一类别确保对齐
        all_unique_classes = unique_classes 

        for val, probs in value_to_class_probs.items():
            val_weight = (f_discrete == val).sum() / len(f_discrete)
            weights.append(val_weight)
            
            # 向量化获取概率，避免循环
            val_probs = np.array([probs.get(c, 0) for c in all_unique_classes])
            val_probs = np.clip(val_probs, 1e-10, 1) # 避免 log(0)
            if val_probs.sum() > 0:
                val_probs = val_probs / val_probs.sum()
            
            conditional_entropies.append(entropy(val_probs))
        
        if overall_entropy > 0:
            information_gain = overall_entropy - np.sum(np.array(weights) * np.array(conditional_entropies))
            normalized_gain = information_gain / overall_entropy
        else:
            normalized_gain = 0
            
    except Exception as e:
        # print(f"IG Error: {e}") # Debug only
        normalized_gain = np.nan
    
    return mi, normalized_gain


# def single_numeric_quality(s: pd.Series, y: pd.Series):
#     """计算数值型特征在多分类问题中的质量指标"""
#     med = s.median()
#     feature_values = s.fillna(med).values
    
#     # 使用多分类特征评估函数
#     mi, info_gain = safe_multiclass_metrics(y.values, feature_values)
    
#     return mi, info_gain



def build_FMM(train_raw: pd.DataFrame, label_col:str='label'):
    """
    Build FMM from the training dataset for multi-class classification.
    """
    assert label_col in train_raw.columns, f"label_col {label_col} not in train_raw.columns"

    y = train_raw[label_col] # y is a Series object, with one column being the index and the other being the actual values.
    features = [c for c in train_raw.columns if c != label_col]  # get names
    print(f"[INFO] 类别分布：{dict(y.value_counts())}")
    y_encoded = y.tolist()
    print(f"[INFO] 多分类模式：检测到 {len(np.unique(y_encoded))} 个类别")
    # le = LabelEncoder()
    # y_encoded = le.fit_transform(y)  
    # print(f"original y: {y}")
    # print(f"encoded y: {y_encoded}")
    # set_trace()
    # y_bin, pos_label = get_binary_target(y, positive_class)
    # print(f"[INFO] AUPRC use positive label: {pos_label}")

    rows = []

    for feat in features:
        s = train_raw[feat]
        ftype = 'numeric'  # 所有特征都是数字类型，无需推断

        n = len(s)
        n_missing = int(s.isna().sum())
        missing_rate = n_missing / n if n > 0 else np.nan
        n_unique = s.nunique(dropna=True)  # A method in 'pandas' used to count the number of unique values in a Series
        
        mean_v = std_v = sk_v = ku_v = min_v = max_v = np.nan
        q01 = q05 = q25 = q50 = q75 = q95 = q99 = np.nan
        mi = info_gain = np.nan  # mi: mutual information, info_gain: information gain

        # all features are numeric type, just process them
        x = s.astype(float)
        non_na = x.dropna()

        # 修复点：显式调用 .values 获取 numpy array，避免传入 Pandas Series 可能引发的类型转换问题
        # 虽然 Pandas Series 传给 np.quantile 通常没问题，但 .values 是最保险的
        non_na_values = non_na.values

        if len(non_na_values) >= 2:
            mean_v = float(non_na_values.mean())
            std_v = float(non_na_values.std(ddof=1)) if len(non_na_values) > 1 else 0.0
            sk_v = float(skew(non_na_values)) if len(non_na_values) > 2 else 0.0
            ku_v = float(kurtosis(non_na_values)) if len(non_na_values) > 3 else 0.0
            min_v = float(non_na_values.min())
            max_v = float(non_na_values.max())
            # 使用 values 计算分位数
            q01 = float(np.quantile(non_na_values, 0.01))
            q05 = float(np.quantile(non_na_values, 0.05))
            q25 = float(np.quantile(non_na_values, 0.25))
            q50 = float(np.quantile(non_na_values, 0.50))
            q75 = float(np.quantile(non_na_values, 0.75))
            q95 = float(np.quantile(non_na_values, 0.95))
            q99 = float(np.quantile(non_na_values, 0.99))

        med = x.median()
        feature_values = x.fillna(med).values
        label_value = y.values
        
        # 使用多分类特征评估函数（函数内部会处理离散化）
        mi, info_gain = safe_multiclass_metrics(label_value, feature_values)

        row = {
            "feat_name": feat,
            "feat_type": ftype,
            "mean": mean_v,
            "std": std_v,
            "skew": sk_v,
            "kurtosis": ku_v,
            "min": min_v,
            "max": max_v,
            "q01": q01,
            "q05": q05,
            "q25": q25,
            "q50": q50,
            "q75": q75,
            "q95": q95,
            "q99": q99,
            "n_unique": n_unique,
            "missing_rate": missing_rate,
            "mi_to_y": mi,  # 互信息
            "info_gain": info_gain,  # 信息增益（多分类特征质量指标）
            "FeatureScore": 0,
            "lineage": "raw",
            "ops_chain": [],
            "gen_round": 0
        }

        rows.append(row)

    return pd.DataFrame(rows)


def save_metrics_to_csv(metrics_dict, file_path, round_num):
    """将分类性能指标保存到CSV文件"""
    # 添加轮次信息
    metrics_dict['round'] = round_num
    metrics_dict['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 转换为DataFrame
    df = pd.DataFrame([metrics_dict])
    
    # 如果文件不存在，创建新文件并写入表头；如果存在，追加数据
    if not os.path.exists(file_path):
        df.to_csv(file_path, index=False, mode='w', encoding='utf-8')
    else:
        df.to_csv(file_path, index=False, mode='a', header=False, encoding='utf-8')


def calculate_average_metrics(results_dir, summary_file):
    """计算所有数据集的平均指标并保存到汇总文件"""
    # 查找所有数据集的指标文件
    dataset_dirs = [d for d in os.listdir(results_dir) if d.startswith('dataset_')]
    
    all_metrics = []
    
    # 读取每个数据集的指标文件
    for dataset_dir in dataset_dirs:
        metrics_file = os.path.join(results_dir, dataset_dir, 'classification_metrics.csv')
        if os.path.exists(metrics_file):
            df = pd.read_csv(metrics_file)
            all_metrics.append(df)
    
    if not all_metrics:
        logger.warning("没有找到任何指标文件")
        return
    
    # 合并所有指标
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    
    # 按轮次分组计算平均值
    avg_metrics = combined_metrics.groupby('round').agg({
        'test_accuracy': 'mean',
        'test_balanced_accuracy': 'mean',
        'f1_score': 'mean',
        'precision': 'mean',
        'recall': 'mean',
        'geometric_mean': 'mean',
        'num_features': 'mean'
    }).reset_index()
    
    # 添加数据集数量信息
    dataset_counts = combined_metrics.groupby('round')['dataset_index'].nunique().reset_index()
    dataset_counts.columns = ['round', 'num_datasets']
    avg_metrics = pd.merge(avg_metrics, dataset_counts, on='round')
    
    # 保存汇总指标
    avg_metrics.to_csv(summary_file, index=False, encoding='utf-8')
    logger.info(f"平均指标已保存到: {summary_file}")
    logger.info(f"共处理了 {len(dataset_dirs)} 个数据集的指标")


# def calculate_average_metrics_for_folds(results_dir, summary_file):
#     """计算所有数据集所有折的平均指标并保存到汇总文件（用于HD模式）"""
#     all_metrics = []
    
#     # 遍历所有数据集目录
#     dataset_dirs = sorted([d for d in os.listdir(results_dir) if d.startswith('dataset_')])
    
#     for dataset_dir in dataset_dirs:
#         dataset_path = os.path.join(results_dir, dataset_dir)
#         # 查找所有折的指标文件
#         fold_files = [f for f in os.listdir(dataset_path) if f.startswith('fold_') and f.endswith('_classification_metrics.csv')]
        
#         for fold_file in fold_files:
#             metrics_file = os.path.join(dataset_path, fold_file)
#             if os.path.exists(metrics_file):
#                 # 读取每个折的指标文件
#                 df = pd.read_csv(metrics_file)
#                 all_metrics.append(df)
    
#     if all_metrics:
#         # 合并所有折的指标
#         combined_metrics = pd.concat(all_metrics, ignore_index=True)
        
#         # 按轮次计算平均指标（跨所有折）
#         avg_metrics = combined_metrics.groupby('round').agg({
#             'test_accuracy': 'mean',
#             'test_balanced_accuracy': 'mean',
#             'f1_score': 'mean',
#             'precision': 'mean',
#             'recall': 'mean',
#             'geometric_mean': 'mean',
#             'num_features': 'mean'
#         }).reset_index()
        
#         # 保存汇总结果
#         avg_metrics.to_csv(summary_file, index=False, encoding='utf-8')
#         logger.info(f"交叉验证平均指标已保存到: {summary_file}")
        
#         # 计算按轮次和折数的详细平均指标
#         fold_avg_metrics = combined_metrics.groupby(['round', 'fold_index']).agg({
#             'test_accuracy': 'mean',
#             'test_balanced_accuracy': 'mean',
#             'f1_score': 'mean',
#             'precision': 'mean',
#             'recall': 'mean',
#             'geometric_mean': 'mean',
#             'num_features': 'mean'
#         }).reset_index()
        
#         # 保存按折的详细指标
#         fold_summary_file = summary_file.replace('.csv', '_by_fold.csv')
#         fold_avg_metrics.to_csv(fold_summary_file, index=False, encoding='utf-8')
#         logger.info(f"按折详细指标已保存到: {fold_summary_file}")
#     else:
#         logger.warning("未找到任何折的指标文件，无法计算平均指标")


def build_label_meta(train_raw: pd.DataFrame, label_col: str):
    assert label_col in train_raw.columns

    y = train_raw[label_col]
    total = len(y)
    rows = []

    for cls,cnt in y.value_counts().items():
        idx_list = y.index[y==cls].tolist()
        row = {
            "class_label": cls,
            "count": int(cnt),
            "ratio": float(cnt / total),
            "indices": idx_list,
            "tags": []
        }
        rows.append(row)

    return pd.DataFrame(rows)


REQUIRED_GENERATED_FEATURE_KEYS = {
    'feat_name', 'feat_type', 'mean', 'std', 'skew', 'kurtosis', 'min', 'max',
    'q01', 'q05', 'q25', 'q50', 'q75', 'q95', 'q99', 'n_unique',
    'missing_rate', 'mi_to_y', 'info_gain', 'FeatureScore', 'lineage',
    'ops_chain', 'gen_round'
}


def _is_valid_generated_feature(feature: Dict) -> bool:
    if not isinstance(feature, dict) or feature.get('error'):
        return False
    if not REQUIRED_GENERATED_FEATURE_KEYS.issubset(feature):
        return False
    return isinstance(feature.get('ops_chain'), list)


def _generate_valid_feature(crossover,
                            mutation,
                            current_file_path: str,
                            use_crossover: bool,
                            logger,
                            max_attempts: int = 3):
    operation_type = "交叉" if use_crossover else "变异"
    for attempt in range(1, max_attempts + 1):
        if use_crossover:
            result = crossover.perform_crossover(current_file_path, num_pairs=1)
        else:
            result = mutation.perform_mutation(current_file_path, num_mutations=1)

        if _is_valid_generated_feature(result):
            return result, operation_type

        logger.warning(f"{operation_type}生成的特征无效，重试 {attempt}/{max_attempts}")

    logger.warning(f"{operation_type}连续生成无效特征，跳过本轮生成")
    return None, operation_type


def score_features(json_file_path: str, output_file_path: str, number: int) -> None:
    scorer = FeatureScoring()
    scorer.process_json_file(json_file_path, output_file_path, number=number)


def select_features(features: List[Dict]) -> List[Dict]:
    selector = FeatureSelector()
    return selector.select_by_knee_point(features)
    


if __name__ == '__main__':
    try:
        start_time = datetime.now()
        
        parser = argparse.ArgumentParser(description='fmm framework evolution paradigm')
        parser.add_argument('--config',default='configs/base.yaml',help='fmm 配置文件路径')
        parser.add_argument('--mode', choices=['industry', 'test', 'HD'], default='HD', help='industry: 执行行业数据集；test: 执行测试数据集；HD: 执行UCI数据集')
        parser.add_argument('--dataset-name', default=None,
                            help='Run only datasets whose file stem contains this text.')
        parser.add_argument('--max-rounds', type=int, default=100,
                            help='Maximum evolution rounds per dataset/fold.')
        args = parser.parse_args()

        if args.mode == 'industry':
            with open(args.config, encoding='utf-8', mode='r') as f:
                config = yaml.safe_load(f)

            train_dir = resolve_project_path(config['industry_data']['train_path'])
            test_dir = resolve_project_path(config['industry_data']['test_path'])
            results_path = resolve_project_path(config['output']['results_path'])
            meta_info_path = resolve_project_path(config['output']['meta_info_path'])
            log_path = resolve_project_path(config['output']['log_path'])

            # 获取当前时间戳
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 给路径添加时间戳后缀（去掉末尾的/）
            results_file_name = f"{config['output']['results_path'].rstrip('/')}_{timestamp}"
            meta_info_file_name = f"{config['output']['meta_info_path'].rstrip('/')}_{timestamp}"
            log_file_name = f"{config['output']['log_path'].rstrip('/')}_{timestamp}"

            # 创建日志目录和文件
            log_timestamp_dir = os.path.join(log_path, f"fmm_run_{timestamp}")
            os.makedirs(log_timestamp_dir, exist_ok=True)
            log_file_path = os.path.join(log_timestamp_dir, "fmm_run.log")

            # 初始化日志系统
            logger = get_logger("FMM_Run", emoji="🚀")
            # 添加文件处理器，将日志写入文件，这个函数只需要调用一次，整个系统（包括现有的模块和未来加载的模块）的日志都会自动流向log_file_path
            # 使用 "FMM" 作为过滤器，这样所有以 "FMM" 开头的日志器都会被捕获
            add_file_handler(log_file_path, filter="FMM", level="DEBUG")

            # 创建结果目录（与meta_info目录使用相同的时间戳）
            results_timestamp_dir = os.path.join(results_path, results_file_name)
            os.makedirs(results_timestamp_dir, exist_ok=True)
            logger.info(f"创建结果时间戳目录: {results_timestamp_dir}")

            logger.info(f"FMM运行开始，时间戳: {timestamp}")
            logger.info(f"日志文件路径: {log_file_path}")
            logger.info(f"配置文件: {args.config}")
            logger.info(f"运行模式: {args.mode}")

            # print(f"{meta_info_file_name}")

            # 创建meta_info目录下的时间戳文件夹
            meta_info_timestamp_dir = os.path.join(meta_info_path, meta_info_file_name)
            os.makedirs(meta_info_timestamp_dir, exist_ok=True)
            logger.info(f"创建meta_info时间戳目录: {meta_info_timestamp_dir}")

            train_files = sorted(glob(os.path.join(train_dir,'*_Train.csv')))
            test_files = sorted(glob(os.path.join(test_dir,'*_Test.csv')))
            if args.dataset_name:
                train_files = [
                    path for path in train_files
                    if args.dataset_name in os.path.splitext(os.path.basename(path))[0].replace('_Train', '')
                ]
                test_files = [
                    path for path in test_files
                    if args.dataset_name in os.path.splitext(os.path.basename(path))[0].replace('_Test', '')
                ]
            # print(train_files)
            # set_trace()

            length = len(train_files)
            max_rounds = args.max_rounds if args.max_rounds is not None else 2
            logger.info(f"训练数据集数量：{length}")

            for i in range(length):
                train_path = train_files[i]
                test_path = test_files[i]
                logger.debug(f"训练文件路径: {train_path}")
                logger.debug(f"测试文件路径: {test_path}")

                # 提取数据集名称（除去后缀）
                dataset_name = os.path.splitext(os.path.basename(train_path))[0].replace('_Train', '')
                logger.info(f"开始处理数据集 {dataset_name} ({i+1}/{length})")
                logger.info(f"数据集名称: {dataset_name}")

                # reading without headers
                train_raw = pd.read_csv(train_path, header=None)
                test_raw = pd.read_csv(test_path, header=None)
                logger.info(f"数据集形状 - 训练集: {train_raw.shape}, 测试集: {test_raw.shape}")

                LABEL_COL = 'label'
                POSITIVE_CLASS = None   # If unset, the class with the fewest samples is selected automatically.

                # The first column is label, and the remaining columns are f1, f2, f3...
                n_cols = train_raw.shape[1]
                train_raw.columns = [LABEL_COL] + [f'f{i}' for i in range(1, n_cols)]
                test_raw.columns = [LABEL_COL] + [f'f{i}' for i in range(1, n_cols)]
                test_size = len(test_raw)
                combined_raw = pd.concat([train_raw, test_raw], ignore_index=True)
                test_index = combined_raw.sample(n=test_size, random_state=42).index
                test_raw = combined_raw.loc[test_index].reset_index(drop=True)
                train_raw = combined_raw.drop(test_index).reset_index(drop=True)
                logger.debug(f"列名设置完成，共 {n_cols} 列")

                logger.info("开始构建FMM特征元信息")
                fmm = build_FMM(train_raw, label_col=LABEL_COL)
                fmm_json = fmm.to_json(orient='records', indent=2, force_ascii=False)
                file_name = os.path.join(meta_info_timestamp_dir, f'FMM_train_{dataset_name}.json')
                meta_info_iter_dir = os.path.join(meta_info_timestamp_dir, dataset_name)
                results_iter_dir = os.path.join(results_timestamp_dir, dataset_name)
                os.makedirs(meta_info_iter_dir, exist_ok=True)
                os.makedirs(results_iter_dir, exist_ok=True)
                logger.debug(f"FMM元信息将保存到: {file_name}")
                with open(file_name, 'w', encoding='utf-8') as f:  # 以写入模式打开文件。如果文件不存在，Python会自动创建它；如果文件已存在，则会覆盖它。
                    f.write(fmm_json)
                logger.info(f"FMM元信息已保存，共 {len(fmm)} 个特征")

                # 上面已经存好当前数据集的元信息，接下来要做的就是读取该数据集计算特征得分（FeatureScoring()）
                logger.info("开始计算特征得分")
                score_features(file_name, meta_info_iter_dir, number=0)

                # 从os.path.join(output_file_path, f'iter_{number}.json')读取特征得分
                file_name = os.path.join(meta_info_iter_dir, f'iter_0.json')
                with open(file_name, 'r', encoding='utf-8') as f:
                    features = json.load(f)
                logger.info(f"初始特征数量: {len(features)}")
                crossover = JsonCrossover()
                mutation = JsonMutation()

                # 初始化未更新计数器，用于30次未更新则结束当前数据集
                no_update_count = 0
                current_best_balanced_accuracy = -1.0
                
                for j in range(1, max_rounds + 1):
                    # 记录每轮开始时间
                    round_start_time = datetime.now()
                    # 接下来是从features里面选择两个父代进行交叉，或者选择一个父代进行变异
                    current_file_path = os.path.join(meta_info_iter_dir, f'iter_{j-1}.json')
                    output_file = os.path.join(meta_info_iter_dir, f'iter_{j}.json')

                    # 设置交叉和变异的概率
                    crossover_prob = 0.7  # 70%概率进行交叉
                    mutation_prob = 0.3   # 30%概率进行变异

                    # 读取当前文件内容
                    with open(current_file_path, 'r', encoding='utf-8') as f:
                        current_features = json.load(f)

                    # 随机选择交叉或变异
                    result, operation_type = _generate_valid_feature(
                        crossover,
                        mutation,
                        current_file_path,
                        use_crossover=random.random() < crossover_prob,
                        logger=logger
                    )

                    if result is None:
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(current_features, f, indent=2, ensure_ascii=False)
                    else:
                        current_n = len(current_features)
                        result['feat_name'] = f'f{current_n+1}'
                        current_features.append(result)

                    if result:
                        # 保存单个结果为列表格式
                        with open(output_file, 'w', encoding='utf-8') as f:
                            json.dump(current_features, f, indent=2, ensure_ascii=False)
                        score_features(output_file, meta_info_iter_dir, number=j)
                        logger.info(f"第{j}轮进化完成，执行了{operation_type}操作")
                    else:
                        logger.warning(f"第{j}轮{operation_type}操作失败")

                    # 读取新文件内容
                    with open(output_file, 'r', encoding='utf-8') as f:
                        current_features = json.load(f)

                    logger.info(f"第{j}轮进化后特征数量: {len(current_features)}")
                    selected_features = select_features(current_features)
                    logger.info(f"选择了 {len(selected_features)} 个特征")
                    # 搜集所有选择的特征列名
                    selected_feature_names = []

                    # print("选择的selector_features:", selected_features)

                    # set_trace()

                    # 现在我选择出来了部分特征的元信息，我要逐个遍历这些特征的特征名字（一般都是f1, f2, f3...）
                    for f in selected_features:
                        feat_name = f['feat_name']
                        # print(f"当前特征名字：{feat_name}")
                        selected_feature_names.append(feat_name)

                        # 检查特征是否为通过进化操作生成的新特征（非原始特征）
                        if f['ops_chain'] and f['lineage'] != 'raw' and feat_name not in train_raw.columns:
                            # 说明这个特征是后面交叉变异得来的，我需要取出ops_chain里面的父代特征名字（从左往右第一个fA），根据父代特征值以及该新特征的元信息在train_raw新开的第feat_index列新建一列特征值
                            # 从ops_chain的第一个字符串中提取第一个f后面的数字

                            ops_string = f['ops_chain'][0]
                            match = re.search(r'f(\d+)', ops_string)
                            if match:
                                parent_feat_name = f'f{match.group(1)}'  # 直接构造父特征名字符串
                                child_feat_name = feat_name  # 子特征名已经是字符串格式

                                logger.debug(f"train_raw 创建新特征前的形状: {train_raw.shape}")
                                logger.debug(f"生成新特征: 父特征={parent_feat_name}, 子特征={child_feat_name}")
                                # 使用新的特征生成函数来创建新特征值
                                train_raw = generate_new_feature_value(
                                    train_raw=train_raw,
                                    parent_feat_name=parent_feat_name,
                                    child_feat_name=child_feat_name,
                                    child_meta_info=f,
                                    output_file_path=output_file
                                )

                                test_raw = generate_new_feature_value(
                                    train_raw=test_raw,
                                    parent_feat_name=parent_feat_name,
                                    child_feat_name=child_feat_name,
                                    child_meta_info=f,
                                    output_file_path=output_file
                                )

                                logger.debug(f"train_raw 创建新特征后的形状: {train_raw.shape}")
                                logger.debug(f"当前选择的特征名字是 {feat_name}")
                                # set_trace()
                    # 检查特征列名是否存在
                    for feat_name in selected_feature_names:
                        if feat_name not in train_raw.columns:
                            logger.error(f"错误：特征 {feat_name} 不存在于数据集中")
                            set_trace()

                    # 现在已经搜集到了所有的特征名，我需要根据这些特征名从train_raw中取出所有的特征列进行分类
                    selected_features_df = train_raw[selected_feature_names]
                    # 取出标签列
                    labels = train_raw[LABEL_COL]
                    # 用随机森林算法进行分类
                    classifier = RandomForestClassifier(
                        n_estimators=200,
                        random_state=42,
                        class_weight='balanced',
                        n_jobs=-1,
                    )
                    classifier.fit(selected_features_df, labels)
                    # 用随机森林算法进行预测
                    predictions = classifier.predict(selected_features_df)
                    # 训练集计算准确率
                    # accuracy = accuracy_score(labels, predictions)
                    # balanced_accuracy = balanced_accuracy_score(labels, predictions)
                    # print(f"随机森林 平衡准确率: {balanced_accuracy:.4f}")
                    # print(f"随机森林 分类准确率: {accuracy:.4f}")

                    # 应该要对测试集进行预测
                    test_features = test_raw[selected_feature_names]
                    test_predictions = classifier.predict(test_features)
                    # 取出测试集标签列
                    test_labels = test_raw[LABEL_COL]
                    test_accuracy = accuracy_score(test_labels, test_predictions)
                    test_balanced_accuracy = balanced_accuracy_score(test_labels, test_predictions)
                    # 再多算一些分类性能指标
                    f1 = f1_score(test_labels, test_predictions, average='weighted')
                    precision = precision_score(test_labels, test_predictions, average='weighted')
                    recall = recall_score(test_labels, test_predictions, average='weighted')
                    gmean = geometric_mean_score(test_labels, test_predictions, average='weighted')

                    
                    # 保存指标到CSV文件
                    metrics_file = os.path.join(results_iter_dir, 'classification_metrics.csv')
                    
                    # 读取历史平衡准确率值，如果文件存在的话
                    best_balanced_accuracy = -1.0
                    best_metrics = None
                    if os.path.exists(metrics_file):
                        try:
                            history_df = pd.read_csv(metrics_file)
                            if 'test_balanced_accuracy' in history_df.columns:
                                best_balanced_accuracy_idx = history_df['test_balanced_accuracy'].idxmax()
                                best_balanced_accuracy = history_df['test_balanced_accuracy'].max()
                                if not pd.isna(best_balanced_accuracy_idx):
                                    best_metrics = history_df.iloc[best_balanced_accuracy_idx].to_dict()
                        except Exception as e:
                            logger.warning(f"读取历史指标文件失败: {e}")
                    
                    # 更新当前最佳平衡准确率和未更新计数器
                    if test_balanced_accuracy > current_best_balanced_accuracy:
                        current_best_balanced_accuracy = test_balanced_accuracy
                        no_update_count = 0  # 重置计数器
                    else:
                        no_update_count += 1  # 未更新计数器递增
                    
                    # 检查是否连续30次未更新，如果是则提前结束当前数据集
                    if no_update_count >= 30:
                        logger.info(f"数据集{dataset_name}在第{j}轮提前结束：连续{no_update_count}次未更新最高平衡准确率 (当前: {test_balanced_accuracy:.4f}, 最佳: {current_best_balanced_accuracy:.4f})")
                        break
                    
                    # 只有当新的平衡准确率比历史最佳更高时才更新指标文件
                    if test_balanced_accuracy > best_balanced_accuracy:
                        metrics_dict = {
                            'dataset_index': dataset_name,
                            'test_accuracy': test_accuracy,
                            'test_balanced_accuracy': test_balanced_accuracy,
                            'f1_score': f1,
                            'precision': precision,
                            'recall': recall,
                            'geometric_mean': gmean,
                            'num_features': len(selected_features),
                            'operation_type': operation_type
                        }
                        save_metrics_to_csv(metrics_dict, metrics_file, j)
                        # logger.info(f"第{j}轮指标已保存到: {metrics_file}")
                    else:
                        # 如果新的平衡准确率不优于历史最佳，保存上一轮的最佳指标
                        if best_metrics:
                            save_metrics_to_csv(best_metrics, metrics_file, j)
                            # logger.info(f"第{j}轮保存了历史最佳指标")
                        # else:
                            # logger.info(f"第{j}轮指标未保存，且无历史指标可保存")

                    # print(f"测试集分类准确率: {test_accuracy:.4f}")

                    # print_buddha_finish()
                    # set_trace()

                    # set_trace()

                    # 这个位置还能访问到current_features吗

                    # 记录本轮结束时间
                    end_time = datetime.now()
                    duration = end_time - round_start_time
                    logger.info(f"第{j}轮进化完成，耗时: {duration.total_seconds():.2f}秒")

                # 记录所有进化轮次结束
                logger.info("所有进化轮次完成")
                logger.info(f"最终结果保存在: {results_path}")

            # 计算所有数据集的平均指标并保存到汇总文件
            # summary_file = os.path.join(results_timestamp_dir, 'average_metrics_summary.csv')
            # calculate_average_metrics(results_timestamp_dir, summary_file)

            # 记录程序总运行时间
            total_time = datetime.now() - start_time
            logger.info(f"程序总运行时间: {total_time.total_seconds():.2f}秒")
        elif args.mode == 'HD':
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            train_dir = resolve_project_path(config['HD_data']['path'])
            results_path = resolve_project_path(config['output']['results_path'])
            meta_info_path = resolve_project_path(config['output']['meta_info_path'])
            log_path = resolve_project_path(config['output']['log_path'])
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            results_file_name = f"{config['output']['results_path'].rstrip('/')}_{timestamp}"
            meta_info_file_name = f"{config['output']['meta_info_path'].rstrip('/')}_{timestamp}"
            log_file_name = f"{config['output']['log_path'].rstrip('/')}_{timestamp}"
            
            # 创建日志目录和文件
            log_timestamp_dir = os.path.join(log_path, f"fmm_run_{timestamp}")
            os.makedirs(log_timestamp_dir, exist_ok=True)
            log_file_path = os.path.join(log_timestamp_dir, "fmm_run.log")
            
            # 初始化日志系统
            logger = get_logger("FMM_Run", emoji="🚀")
            # 添加文件处理器，将日志写入文件
            add_file_handler(log_file_path, filter="FMM", level="DEBUG")
            
            # 创建结果目录
            results_timestamp_dir = os.path.join(results_path, results_file_name)
            os.makedirs(results_timestamp_dir, exist_ok=True)
            logger.info(f"创建结果时间戳目录: {results_timestamp_dir}")
            
            logger.info(f"FMM运行开始，时间戳: {timestamp}")
            logger.info(f"日志文件路径: {log_file_path}")
            logger.info(f"配置文件: {args.config}")
            logger.info(f"运行模式: {args.mode}")
            
            # 创建meta_info目录下的时间戳文件夹
            meta_info_timestamp_dir = os.path.join(meta_info_path, meta_info_file_name)
            os.makedirs(meta_info_timestamp_dir, exist_ok=True)
            logger.info(f"创建meta_info时间戳目录: {meta_info_timestamp_dir}")
            
            # 获取所有数据集文件
            dataset_files = sorted(glob(os.path.join(train_dir,'*.csv')))
            if args.dataset_name:
                dataset_files = [
                    path for path in dataset_files
                    if args.dataset_name in os.path.splitext(os.path.basename(path))[0]
                ]
            length = len(dataset_files)
            max_rounds = args.max_rounds if args.max_rounds is not None else 3
            logger.info(f"数据集数量：{length}")
            
            # 五折交叉验证
            n_folds = 5
            
            for i in range(length):
                dataset_path = dataset_files[i]
                

                logger.debug(f"数据集文件路径: {dataset_path}")
                
                # 提取数据集名称（除去后缀）
                dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
                logger.info(f"开始处理数据集 {dataset_name} ({i+1}/{length})")
                logger.info(f"数据集名称: {dataset_name}")
                
                # 读取数据集
                data_raw = pd.read_csv(dataset_path, header=None)
                logger.info(f"数据集形状: {data_raw.shape}")
                
                LABEL_COL = 'label'
                POSITIVE_CLASS = None   # if not filled in, it will automatically find the class with the fewest samples
                
                # naming columns: the first column is label, and the rest are f1, f2, f3...
                n_cols = data_raw.shape[1]
                data_raw.columns = [f'f{i}' for i in range(1, n_cols)] + [LABEL_COL]
                logger.debug(f"列名设置完成，共 {n_cols} 列")
                
                # 为每个数据集创建目录，使用真实的数据集名称
                meta_info_iter_dir = os.path.join(meta_info_timestamp_dir, dataset_name)
                results_iter_dir = os.path.join(results_timestamp_dir, dataset_name)
                os.makedirs(meta_info_iter_dir, exist_ok=True)
                os.makedirs(results_iter_dir, exist_ok=True)
                
                # 初始化交叉验证
                kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                
                # 存储每一折的结果
                fold_results = []
                
                # 跟踪上一轮的最佳指标
                last_best_metrics = None
                
                # 对每一折进行处理
                for fold_idx, (train_idx, test_idx) in enumerate(kf.split(data_raw.drop(LABEL_COL, axis=1), data_raw[LABEL_COL])):
                    logger.info(f"开始处理第 {fold_idx+1}/{n_folds} 折交叉验证")
                    
                    # 分割训练集和测试集
                    train_raw = data_raw.iloc[train_idx].copy()
                    test_raw = data_raw.iloc[test_idx].copy()
                    logger.info(f"第{fold_idx+1}折数据集形状 - 训练集: {train_raw.shape}, 测试集: {test_raw.shape}")
                    
                    # construct FMM & LabelMeta (只在第一折构建，后续折复用)
                    if fold_idx == 0:
                        logger.info("开始构建FMM特征元信息")
                        fmm = build_FMM(train_raw, label_col=LABEL_COL)
                        # Get JSON string first
                        fmm_json = fmm.to_json(orient='records', indent=2, force_ascii=False)
                        # Write to file
                        file_name = os.path.join(meta_info_iter_dir, f'FMM_train_{dataset_name}.json')
                        logger.debug(f"FMM元信息将保存到: {file_name}")
                        with open(file_name, 'w', encoding='utf-8') as f:
                            f.write(fmm_json)
                        logger.info(f"FMM元信息已保存，共 {len(fmm)} 个特征")
                        
                        # 上面已经存好当前数据集的元信息，接下来要做的就是读取该数据集计算特征得分（FeatureScoring()）
                        logger.info("开始计算特征得分")
                        score_features(file_name, meta_info_iter_dir, number=0)
                        
                        # 从os.path.join(output_file_path, f'iter_{number}.json')读取特征得分
                        file_name = os.path.join(meta_info_iter_dir, f'iter_0.json')
                        with open(file_name, 'r', encoding='utf-8') as f:
                            features = json.load(f)
                        logger.info(f"初始特征数量: {len(features)}")
                        crossover = JsonCrossover()
                        mutation = JsonMutation()
                    
                    # 每一折都重新初始化特征数据
                    with open(os.path.join(meta_info_iter_dir, f'iter_0.json'), 'r', encoding='utf-8') as f:
                        features = json.load(f)
                    
                    # 每一折的进化过程
                    # 初始化未更新计数器，用于30次未更新则结束当前折
                    no_update_count = 0
                    current_best_balanced_accuracy = -1.0
                    
                    for j in range(1, max_rounds + 1):
                        # 记录每轮开始时间
                        round_start_time = datetime.now()
                        # 接下来是从features里面选择两个父代进行交叉，或者选择一个父代进行变异
                        current_file_path = os.path.join(meta_info_iter_dir, f'iter_{j-1}.json')
                        output_file = os.path.join(meta_info_iter_dir, f'iter_{j}.json')
                        
                        # 设置交叉和变异的概率
                        crossover_prob = 0.7  # 70%概率进行交叉
                        mutation_prob = 0.3   # 30%概率进行变异
                        
                        # 读取当前文件内容
                        with open(current_file_path, 'r', encoding='utf-8') as f:
                            current_features = json.load(f)
                        
                        # 随机选择交叉或变异
                        result, operation_type = _generate_valid_feature(
                            crossover,
                            mutation,
                            current_file_path,
                            use_crossover=random.random() < crossover_prob,
                            logger=logger
                        )
                        
                        if result is None:
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(current_features, f, indent=2, ensure_ascii=False)
                        else:
                            current_n = len(current_features)
                            result['feat_name'] = f'f{current_n+1}'
                            current_features.append(result)
                        
                        if result:
                            # 保存单个结果为列表格式
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(current_features, f, indent=2, ensure_ascii=False)
                            score_features(output_file, meta_info_iter_dir, number=j)
                            logger.info(f"第{j}轮进化完成，执行了{operation_type}操作")
                        else:
                            logger.warning(f"第{j}轮{operation_type}操作失败")
                        
                        # 读取新文件内容
                        with open(output_file, 'r', encoding='utf-8') as f:
                            current_features = json.load(f)
                        
                        logger.info(f"第{j}轮进化后特征数量: {len(current_features)}")
                        selected_features = select_features(current_features)
                        logger.info(f"选择了 {len(selected_features)} 个特征")
                        # 搜集所有选择的特征列名
                        selected_feature_names = []
                        
                        # 现在我选择出来了部分特征的元信息，我要逐个遍历这些特征的特征名字（一般都是f1, f2, f3...）
                        for f in selected_features:
                            feat_name = f['feat_name']
                            selected_feature_names.append(feat_name)
                            
                            # 检查特征是否为通过进化操作生成的新特征（非原始特征）
                            if f['ops_chain'] and f['lineage'] != 'raw' and feat_name not in train_raw.columns:
                                # 说明这个特征是后面交叉变异得来的，我需要取出ops_chain里面的父代特征名字（从左往右第一个fA），根据父代特征值以及该新特征的元信息在train_raw新开的第feat_index列新建一列特征值
                                # 从ops_chain的第一个字符串中提取第一个f后面的数字
                                
                                ops_string = f['ops_chain'][0]
                                match = re.search(r'f(\d+)', ops_string)
                                if match:
                                    parent_feat_name = f'f{match.group(1)}'  # 直接构造父特征名字符串
                                    child_feat_name = feat_name  # 子特征名已经是字符串格式
                                    
                                    logger.debug(f"train_raw 创建新特征前的形状: {train_raw.shape}")
                                    logger.debug(f"生成新特征: 父特征={parent_feat_name}, 子特征={child_feat_name}")
                                    # 使用新的特征生成函数来创建新特征值
                                    train_raw = generate_new_feature_value(
                                        train_raw=train_raw,
                                        parent_feat_name=parent_feat_name,
                                        child_feat_name=child_feat_name,
                                        child_meta_info=f,
                                        output_file_path=output_file
                                    )
                                    
                                    test_raw = generate_new_feature_value(
                                        train_raw=test_raw,
                                        parent_feat_name=parent_feat_name,
                                        child_feat_name=child_feat_name,
                                        child_meta_info=f,
                                        output_file_path=output_file
                                    )
                                    
                                    logger.debug(f"train_raw 创建新特征后的形状: {train_raw.shape}")
                                    logger.debug(f"当前选择的特征名字是 {feat_name}")
                        
                        # 检查特征列名是否存在
                        for feat_name in selected_feature_names:
                            if feat_name not in train_raw.columns:
                                logger.error(f"错误：特征 {feat_name} 不存在于数据集中")
                                set_trace()
                        
                        # 现在已经搜集到了所有的特征名，我需要根据这些特征名从train_raw中取出所有的特征列进行分类
                        selected_features_df = train_raw[selected_feature_names]
                        # 取出标签列
                        labels = train_raw[LABEL_COL]
                        # 用随机森林算法进行分类
                        classifier = RandomForestClassifier(
                            n_estimators=200,
                            random_state=42,
                            class_weight='balanced',
                            n_jobs=-1,
                        )
                        classifier.fit(selected_features_df, labels)
                        
                        # 应该要对测试集进行预测
                        test_features = test_raw[selected_feature_names]
                        test_predictions = classifier.predict(test_features)
                        # 取出测试集标签列
                        test_labels = test_raw[LABEL_COL]
                        test_accuracy = accuracy_score(test_labels, test_predictions)
                        test_balanced_accuracy = balanced_accuracy_score(test_labels, test_predictions)
                        # 再多算一些分类性能指标
                        f1 = f1_score(test_labels, test_predictions, average='weighted')
                        precision = precision_score(test_labels, test_predictions, average='weighted')
                        recall = recall_score(test_labels, test_predictions, average='weighted')
                        gmean = geometric_mean_score(test_labels, test_predictions, average='weighted')
                        
                        
                        
                        # 保存指标到CSV文件
                        metrics_file = os.path.join(results_iter_dir, f'fold_{fold_idx+1}_classification_metrics.csv')
                        
                        # 读取历史平衡准确率值，如果文件存在的话
                        best_balanced_accuracy = -1.0
                        best_metrics = None
                        if os.path.exists(metrics_file):
                            try:
                                history_df = pd.read_csv(metrics_file)
                                if 'test_balanced_accuracy' in history_df.columns:
                                    best_balanced_accuracy_idx = history_df['test_balanced_accuracy'].idxmax()
                                    best_balanced_accuracy = history_df['test_balanced_accuracy'].max()
                                    if not pd.isna(best_balanced_accuracy_idx):
                                        best_metrics = history_df.iloc[best_balanced_accuracy_idx].to_dict()
                            except Exception as e:
                                logger.warning(f"读取历史指标文件失败: {e}")
                        
                        # 更新当前最佳平衡准确率和未更新计数器
                        if test_balanced_accuracy > current_best_balanced_accuracy:
                            current_best_balanced_accuracy = test_balanced_accuracy
                            no_update_count = 0  # 重置计数器
                        else:
                            no_update_count += 1  # 未更新计数器递增
                        
                        # 检查是否连续30次未更新，如果是则提前结束当前折
                        if no_update_count >= 30:
                            logger.info(f"第{fold_idx+1}折在第{j}轮提前结束：连续{no_update_count}次未更新最高平衡准确率 (当前: {test_balanced_accuracy:.4f}, 最佳: {current_best_balanced_accuracy:.4f})")
                            break
                        
                        # 只有当新的平衡准确率比历史最佳更高时才更新指标文件
                        if test_balanced_accuracy > best_balanced_accuracy:
                            metrics_dict = {
                                'dataset_index': dataset_name,
                                'fold_index': fold_idx+1,
                                'test_accuracy': test_accuracy,
                                'test_balanced_accuracy': test_balanced_accuracy,
                                'f1_score': f1,
                                'precision': precision,
                                'recall': recall,
                                'geometric_mean': gmean,
                                'num_features': len(selected_features),
                                'operation_type': operation_type
                            }
                            save_metrics_to_csv(metrics_dict, metrics_file, j)
                            # logger.info(f"第{fold_idx+1}折第{j}轮指标已保存到: {metrics_file}")
                        else:
                            if best_metrics:
                                save_metrics_to_csv(best_metrics, metrics_file, j)
                                # logger.info(f"第{fold_idx+1}折第{j}轮保存了历史最佳指标")
                            # else:
                                # logger.info(f"第{fold_idx+1}折第{j}轮指标未保存，且无历史指标可保存")
                        
                        # 记录本轮结束时间
                        end_time = datetime.now()
                        duration = end_time - round_start_time
                        logger.info(f"第{fold_idx+1}折第{j}轮进化完成，耗时: {duration.total_seconds():.2f}秒")
                    
                    # 记录当前折结束
                    logger.info(f"第{fold_idx+1}折进化完成")
                
                # 计算所有折的平均指标并保存到汇总文件
                # summary_file = os.path.join(results_iter_dir, 'average_metrics_summary.csv')
                # calculate_average_metrics_for_folds(results_iter_dir, summary_file)
                # logger.info(f"数据集{i}的所有折平均指标已保存到: {summary_file}")
                
                # 记录当前数据集处理完成
                logger.info(f"数据集{dataset_name}处理完成")
            
            # 计算所有数据集的平均指标并保存到汇总文件
            # summary_file = os.path.join(results_timestamp_dir, 'average_metrics_summary.csv')
            # calculate_average_metrics_for_folds(results_timestamp_dir, summary_file)
            # logger.info(f"所有数据集的平均指标已保存到: {summary_file}")
            
            # 记录程序总运行时间
            total_time = datetime.now() - start_time
            logger.info(f"程序总运行时间: {total_time.total_seconds():.2f}秒")


    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}")
        import traceback
        logger.error(f"错误详情: {traceback.format_exc()}")
        raise


        # label_meta = build_label_meta(train_raw, label_col=LABEL_COL)

        # # save it to the local device to take a look
        # fmm.to_csv(os.path.join(meta_info_timestamp_dir, f'FMM_train_{i}.csv'), index=False)
        # label_meta.to_csv(os.path.join(meta_info_timestamp_dir, f'LabelMeta_train_{i}.csv'), index=False)

        # # Check the information
        # print("[INFO] read the file:", train_path)
        # print("[INFO] the dimension of the dataset:", train_raw.shape)
        # print("[INFO] FMM has already been saved to FMM_train.csv, its shape: ", fmm.shape)
        # print("[INFO] LabelMeta has already been saved to LabelMeta_train.csv, its shape: ", label_meta.shape)
