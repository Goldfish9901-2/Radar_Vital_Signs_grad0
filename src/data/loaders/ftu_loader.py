"""4TU.ResearchD数据集加载器"""

from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

from .base_loader import BaseDataLoader


class FTUDataLoader(BaseDataLoader):
    """4TU.ResearchD数据集加载器。
    
    该加载器负责读取4TU数据集的雷达二进制文件和参考CSV文件，
    并将其转换为标准格式供后续处理使用。
    
    数据格式说明：
        - 雷达数据: .bin文件，包含IQ复数数据
        - 参考心率: .CSV文件，包含时间序列心率值
        - 参考呼吸率: .ods文件，包含呼吸率统计值
    
    Attributes:
        NUM_ADC (int): ADC采样点数，默认250
        NUM_RX (int): 接收天线数，默认4
        NUM_FRAMES (int): 帧数，默认1200
        NUM_CHIRPS (int): 每帧chirp数，默认128
        SCENARIO_MAP (dict): 场景名称映射
    
    Example:
        >>> loader = FTUDataLoader('Dataset/4TU.ResearchD')
        >>> radar_data = loader.load_radar_data(1, 'Distance', '80 cm', 1)
        >>> ref_data = loader.load_reference_data(1, 'Distance', '80 cm', 1)
        >>> print(f"Radar shape: {radar_data.shape}")
        Radar shape: (4, 250, 153600)
    """
    
    # 雷达配置常量
    NUM_ADC = 250
    NUM_RX = 4
    NUM_FRAMES = 1200
    NUM_CHIRPS = 128
    
    # 场景映射
    SCENARIO_MAP = {
        'Distance': '1. Distance Scenario',
        'Orientation': '2. Orientation Scenario',
        'Angle': '3. Angle Scenario',
        'Elevated': '4. Elevated'
    }
    
    def __init__(self, dataset_path: str):
        """初始化4TU数据加载器。
        
        Args:
            dataset_path: 数据集根目录路径
        
        Raises:
            FileNotFoundError: 如果数据集路径不存在或结构不完整
        """
        super().__init__(dataset_path)
        self.radar_data_path = self.dataset_path / 'Radar data'
        self.hr_ref_path = self.dataset_path / 'HR_Ref_Values'
        self.br_ref_path = self.dataset_path / 'BR_Ref_Values'
        
        self._validate_dataset_structure()
    
    def _validate_dataset_structure(self) -> None:
        """验证数据集目录结构是否完整。
        
        Raises:
            FileNotFoundError: 如果缺少必要的目录
        """
        required_dirs = [
            self.radar_data_path,
            self.hr_ref_path,
            self.br_ref_path
        ]
        
        for dir_path in required_dirs:
            if not dir_path.exists():
                raise FileNotFoundError(
                    f"数据集目录结构不完整，缺少: {dir_path}"
                )
    
    def load_radar_data(
        self,
        participant_id: int,
        scenario: str,
        distance: str,
        repeat: int
    ) -> np.ndarray:
        """加载雷达原始数据。
        
        Args:
            participant_id: 参与者ID (1-10)
            scenario: 场景类型 ('Distance', 'Orientation', 'Angle', 'Elevated')
            distance: 距离 (如 '80 cm')
            repeat: 重复次数 (1-4)
        
        Returns:
            雷达数据立方体，shape为(num_rx, num_adc, num_chirps)
            - num_rx: 4 (接收天线数)
            - num_adc: 250 (ADC采样点数)
            - num_chirps: 153600 (1200帧 × 128 chirps/帧)
        
        Raises:
            ValueError: 如果参数无效
            FileNotFoundError: 如果数据文件不存在
            IOError: 如果文件读取失败
        """
        # 参数验证
        self._validate_parameters(participant_id, scenario, repeat)
        
        # 构建文件路径
        file_path = self._build_radar_file_path(
            participant_id, scenario, distance, repeat
        )
        
        # 读取二进制文件
        bin_file = file_path / 'data_Raw_0.bin'
        try:
            raw_data = np.fromfile(bin_file, dtype=np.int16)
        except Exception as e:
            raise IOError(f"读取雷达数据失败: {bin_file}, 错误: {e}")
        
        # 验证数据大小
        expected_size = (
            self.NUM_ADC * self.NUM_RX * 
            self.NUM_FRAMES * self.NUM_CHIRPS * 2  # IQ
        )
        if len(raw_data) != expected_size:
            raise ValueError(
                f"数据大小不匹配。期望: {expected_size}, 实际: {len(raw_data)}"
            )
        
        # 重塑数据为数据立方体
        return self._reshape_radar_data(raw_data)
    
    def load_reference_data(
        self,
        participant_id: int,
        scenario: str,
        distance: str,
        repeat: int
    ) -> Dict[str, Any]:
        """加载参考心率数据。
        
        Args:
            participant_id: 参与者ID (1-10)
            scenario: 场景类型
            distance: 距离
            repeat: 重复次数 (1-4)
        
        Returns:
            包含以下键的字典:
                - 'heart_rate': 心率时间序列 (bpm)
                - 'timestamps': 时间戳 (秒)
        
        Raises:
            FileNotFoundError: 如果参考文件不存在
            IOError: 如果文件读取失败
        """
        # 构建参考文件路径
        hr_file = self._build_hr_file_path(
            participant_id, scenario, distance, repeat
        )
        
        # 读取CSV文件（跳过前3行元数据）
        try:
            df = pd.read_csv(hr_file, skiprows=3, header=None)
        except Exception as e:
            raise IOError(f"读取参考数据失败: {hr_file}, 错误: {e}")
        
        # 提取时间和心率列
        timestamps = df.iloc[:, 1].values  # 第2列是时间
        heart_rate = df.iloc[:, 2].values  # 第3列是心率
        
        # 转换时间格式 (HH:MM:SS -> 秒)
        timestamps_sec = self._convert_timestamps(timestamps)
        
        return {
            'heart_rate': heart_rate,
            'timestamps': timestamps_sec
        }
    
    def _validate_parameters(
        self,
        participant_id: int,
        scenario: str,
        repeat: int
    ) -> None:
        """验证输入参数的有效性。
        
        Args:
            participant_id: 参与者ID
            scenario: 场景类型
            repeat: 重复次数
        
        Raises:
            ValueError: 如果参数无效
        """
        if not 1 <= participant_id <= 10:
            raise ValueError(
                f"participant_id必须在1-10之间，得到: {participant_id}"
            )
        
        if scenario not in self.SCENARIO_MAP:
            raise ValueError(
                f"scenario必须是{list(self.SCENARIO_MAP.keys())}之一，"
                f"得到: {scenario}"
            )
        
        if not 1 <= repeat <= 4:
            raise ValueError(f"repeat必须在1-4之间，得到: {repeat}")
    
    def _build_radar_file_path(
        self,
        participant_id: int,
        scenario: str,
        distance: str,
        repeat: int
    ) -> Path:
        """构建雷达数据文件路径。
        
        Args:
            participant_id: 参与者ID
            scenario: 场景类型
            distance: 距离
            repeat: 重复次数
        
        Returns:
            雷达数据文件所在目录路径
        
        Raises:
            FileNotFoundError: 如果路径不存在
        """
        scenario_folder = self.SCENARIO_MAP[scenario]
        
        path = (
            self.radar_data_path /
            f'Participant {participant_id}' /
            scenario_folder /
            distance /
            str(repeat)
        )
        
        if not path.exists():
            raise FileNotFoundError(f"数据路径不存在: {path}")
        
        return path
    
    def _build_hr_file_path(
        self,
        participant_id: int,
        scenario: str,
        distance: str,
        repeat: int
    ) -> Path:
        """构建心率参考文件路径。
        
        Args:
            participant_id: 参与者ID
            scenario: 场景类型
            distance: 距离
            repeat: 重复次数
        
        Returns:
            心率参考文件路径
        
        Raises:
            FileNotFoundError: 如果文件不存在
        """
        scenario_folder = self.SCENARIO_MAP[scenario]
        
        path = (
            self.hr_ref_path /
            f'Participant {participant_id}' /
            scenario_folder /
            distance /
            f'R{repeat}.CSV'
        )
        
        if not path.exists():
            raise FileNotFoundError(f"参考文件不存在: {path}")
        
        return path
    
    def _reshape_radar_data(self, raw_data: np.ndarray) -> np.ndarray:
        """按 chirp 展开顺序重排为 [RX, ADC, Frames*Chirps] 复数数据立方体。

        对应顺序（与图示一致）：
        1) 整体文件：chirp1 -> chirp2 -> ... -> chirpM
        2) 每个 chirp 内：RX0 -> RX1 -> RX2 -> RX3
        3) 每个 RX 内：I1 I2 Q1 Q2 I3 I4 Q3 Q4 ... I(N-1) I(N) Q(N-1) Q(N)
        """
        total_chirps = self.NUM_FRAMES * self.NUM_CHIRPS
        if self.NUM_ADC % 2 != 0:
            raise ValueError(
                f"当前实现要求 NUM_ADC 为偶数，得到: {self.NUM_ADC}"
            )

        # 每个 chirp 中，单个 RX 占用的 int16 数：NUM_ADC 个 I + NUM_ADC 个 Q
        words_per_rx_per_chirp = self.NUM_ADC * 2
        words_per_chirp = self.NUM_RX * words_per_rx_per_chirp
        expected_words = total_chirps * words_per_chirp
        if raw_data.size != expected_words:
            raise ValueError(
                f"原始数据长度与配置不匹配。期望 int16 数: {expected_words}, 实际: {raw_data.size}"
            )

        # 先按 chirp、RX、(I1 I2 Q1 Q2) 四元组拆分
        # grouped 形状: [total_chirps, NUM_RX, NUM_ADC/2, 4]
        grouped = raw_data.reshape(
            total_chirps,
            self.NUM_RX,
            self.NUM_ADC // 2,
            4
        ).astype(np.float32)

        # 还原每个 RX 的连续 I/Q 序列
        i_data = np.empty((total_chirps, self.NUM_RX, self.NUM_ADC), dtype=np.float32)
        q_data = np.empty((total_chirps, self.NUM_RX, self.NUM_ADC), dtype=np.float32)
        i_data[:, :, 0::2] = grouped[:, :, :, 0]  # I1, I3, ...
        i_data[:, :, 1::2] = grouped[:, :, :, 1]  # I2, I4, ...
        q_data[:, :, 0::2] = grouped[:, :, :, 2]  # Q1, Q3, ...
        q_data[:, :, 1::2] = grouped[:, :, :, 3]  # Q2, Q4, ...

        # 转为复数，并整理成 [RX, ADC, total_chirps]
        complex_data = i_data + 1j * q_data  # [total_chirps, rx, samples]
        frame_major = complex_data.reshape(
            self.NUM_FRAMES,
            self.NUM_CHIRPS,
            self.NUM_RX,
            self.NUM_ADC
        )
        return np.transpose(frame_major, (0, 2, 1, 3)).astype(np.complex64)

    def _convert_timestamps(self, timestamps: np.ndarray) -> np.ndarray:
        """将时间戳从HH:MM:SS格式转换为秒。

        Args:
            timestamps: 时间戳字符串数组，格式为 "00:00:01"

        Returns:
            时间戳秒数数组
        """
        seconds = []
        for ts in timestamps:
            if isinstance(ts, str):
                parts = ts.split(':')
                if len(parts) == 3:
                    h, m, s = map(int, parts)
                    seconds.append(h * 3600 + m * 60 + s)
                else:
                    seconds.append(0)
            else:
                seconds.append(0)

        return np.array(seconds)

    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集详细信息。

        Returns:
            包含数据集信息的字典
        """
        return {
            'dataset_name': '4TU.ResearchD',
            'dataset_path': str(self.dataset_path),
            'num_participants': 10,
            'scenarios': list(self.SCENARIO_MAP.keys()),
            'radar_config': {
                'num_adc': self.NUM_ADC,
                'num_rx': self.NUM_RX,
                'num_frames': self.NUM_FRAMES,
                'num_chirps': self.NUM_CHIRPS,
                'frequency_range': '77-81 GHz',
                'frame_rate': '20 fps'
            }
        }

