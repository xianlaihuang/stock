"""
V4 激进：与 engine_v2 / dynamic_weight_engine（V2/V3）完全隔离的后端分支。
后续仅改本包内文件，勿在 engine_v2 中增加 v4 逻辑。
"""
from v4_aggressive.engine import analyze_signals_v4_aggressive, get_conditions

__all__ = ['analyze_signals_v4_aggressive', 'get_conditions']
