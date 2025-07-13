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
        self.properties_config = configparser.ConfigParser()
        self.load_config()
        self._initialized = True

    def default_config(self) -> Dict[str, Dict[str, str]]:
        return {
            "langchain": {
                "api-key": "Input LangSmith API Key here",
            },
            "zhipu": {
                "api-key": "Input Zhipu API Key here",
                "model": "glm-4-plus"
            },
            "project": {
                "project-dir": "Input your project path here",
            }
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
            print(f"⚠️ Error saving config file: {e}")

    def load_config(self) -> None:
        if not self.config_exists():
            self.create_default_config()
            print(
                "Config file generated. Please configure it in config/config.ini.")
            exit()
        try:
            self.config.read(self.config_path)
        except configparser.Error as e:
            print(f"⚠️ Error loading config file: {e}")
            self.create_default_config()

        print("Config file loaded successfully")

        # Set the environment
        api_key = self.get("zhipu", "api-key")
        os.environ["ZHIPUAI_API_KEY"] = api_key

        langsmith_api_key = self.get("langchain", "api-key")
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_END_POINT"] = "https://api.smith.langchain.com"
        os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = "modmaster"

        project_dir = self.get("project", "project-dir")
        project_path = Path(project_dir)
        if not project_path.is_dir():
            print("Project path not found!")
            exit()

        try:
            with open(project_path / "gradle.properties") as stream:
                content = "[dummy_section]\n" + stream.read()
            self.properties_config.read_string(content)
        except configparser.Error as e:
            print(f"⚠️ Error loading gradle.properties file: {e}")
            exit()

        print("Project loaded successfully")
        print(f"Mod name: {self.get_project_property("mod_name")}")
        print(f"Mod version: {self.get_project_property("mod_version")}")
        print(
            f"Hello, mod developers! {self.get_project_property("mod_authors")}!")

    def get(self, section: str, option: str) -> str:
        return self.config.get(section, option)

    def get_project_property(self, option: str) -> str:
        return self.properties_config.get("dummy_section", option)


config = config_manager()
