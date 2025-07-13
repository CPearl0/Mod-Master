import configparser
from pathlib import Path
from typing import Dict


class config_manager:
    def __init__(self) -> None:
        self.config_path = Path("configs/config.ini")
        self.config = configparser.ConfigParser()

    def default_config(self) -> Dict[str, Dict[str, str]]:
        return {
            "langchain": {
                "api-key": "输入LangSmith API Key",
            },
            "zhipu": {
                "api-key": "输入智谱API Key",
                "model": "glm-4-plus"
            },
        }

    def config_exists(self) -> bool:
        return self.config_path.exists()

    def create_default_config(self) -> None:
        default = self.default_config()
        for section, options in default.items():
            self.config[section] = options
        self.save_config()

    def save_config(self) -> None:
        try:
            config_dir = self.config_path.parent
            config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                self.config.write(f)
        except IOError as e:
            print(f"⚠️ 保存配置文件时出错: {e}")

    def load_config(self) -> configparser.ConfigParser:
        if not self.config_exists():
            self.create_default_config()
            return self.config

        try:
            self.config.read(self.config_path)
        except configparser.Error as e:
            print(f"⚠️ 加载配置文件时出错: {e}")
            self.create_default_config()
        return self.config
