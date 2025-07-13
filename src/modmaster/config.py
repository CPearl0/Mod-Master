import configparser
from pathlib import Path
from typing import Dict
import os


class config_manager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self) -> None:
        if self._initialized:
            return
        self.config_path = Path("configs/config.ini")
        self.config = configparser.ConfigParser()
        self.load_config()
        self._initialized = True

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
            print("已生成配置文件，请在config/config.ini中进行配置")
            exit()
        try:
            self.config.read(self.config_path)
        except configparser.Error as e:
            print(f"⚠️ 加载配置文件时出错: {e}")
            self.create_default_config()

        # Set the environment
        api_key = self.get("zhipu", "api-key")
        os.environ["ZHIPUAI_API_KEY"] = api_key

        langsmith_api_key = self.get("langchain", "api-key")
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_END_POINT"] = "https://api.smith.langchain.com"
        os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = "modmaster"

        return self.config

    def get(self, section: str, option: str) -> str:
        return self.config.get(section, option)


config = config_manager()
