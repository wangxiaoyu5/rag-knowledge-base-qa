# -*- coding: utf-8 -*-
"""
下载重排序模型脚本 (Cross-Encoder)
模型: BAAI/bge-reranker-base
大小: 约 1GB
"""
import os
import sys

def download_reranker_model():
    """从 ModelScope 下载重排序模型"""
    try:
        from modelscope import snapshot_download
    except ImportError:
        print("请先安装 modelscope: pip install modelscope")
        sys.exit(1)
    
    # 创建模型保存目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(current_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    print('=' * 60)
    print('开始下载重排序模型 (Cross-Encoder)')
    print('模型: BAAI/bge-reranker-base')
    print('=' * 60)
    print(f'模型将保存到: {model_dir}')
    print('模型大小: 约 1GB')
    print('下载时间: 取决于网络速度（通常 5-15 分钟）')
    print('=' * 60)
    
    try:
        # 从 modelscope 下载模型
        downloaded_path = snapshot_download(
            'BAAI/bge-reranker-base',
            cache_dir=model_dir,
            revision='master'
        )
        print('\n' + '=' * 60)
        print('[OK] 重排序模型下载完成！')
        print(f'保存路径: {downloaded_path}')
        print('=' * 60)
        
        # 创建模型路径配置文件
        config_path = os.path.join(current_dir, 'reranker_config.txt')
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(f'RERANKER_MODEL_PATH={downloaded_path}\n')
        print(f'\n模型路径已保存到: {config_path}')
        print('\n现在可以在 Web 界面中使用重排序功能了！')
        
        return downloaded_path
        
    except Exception as e:
        print(f'\n[ERROR] 下载失败: {e}')
        print('\n可能的解决方案:')
        print('1. 检查网络连接')
        print('2. 使用 VPN 或代理')
        print('3. 手动从 https://modelscope.cn/models/BAAI/bge-reranker-base 下载')
        sys.exit(1)


if __name__ == "__main__":
    download_reranker_model()
