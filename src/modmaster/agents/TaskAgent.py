import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, TypedDict, Literal, TypedDict, Callable
from langgraph.graph import StateGraph, END
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.embeddings import ZhipuAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores import InMemoryVectorStore
from bs4.filter import SoupStrainer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import ChatPromptTemplate
import modmaster.prompts

from modmaster.config import config


class AgentState(TypedDict):
    task: str
    src_path: Path
    directory_structure: str
    directory_actions: List[str]
    needed_files: Dict[str, str]
    retrieved_info: str
    analysis_result: Dict[str, List[Dict]]
    modification_report: List[str]


class TaskAgent:
    def __init__(self):
        self.proj_path = Path(config.get("project", "project-dir"))
        self.src_path = self.proj_path / "src"
        self.workflow = self._build_workflow()

        self.chat_model = ChatZhipuAI(
            model=config.get("zhipu", "model"),
            temperature=0.95,
        )

        self.embeddings = ZhipuAIEmbeddings(
            api_key=config.get("zhipu", "api-key"),
            model="embedding-3",
        )

        self.vector_store = InMemoryVectorStore(self.embeddings)

        self._init_knowledge_base()

    def _load_documents(self) -> List[Document]:
        loader = WebBaseLoader(
            (
                "https://docs.neoforged.net/docs/1.21.1/concepts/registries",
                "https://docs.neoforged.net/docs/1.21.1/items/",
                "https://docs.neoforged.net/docs/1.21.1/blocks/",
                "https://docs.neoforged.net/docs/1.21.1/blocks/states",
                "https://docs.neoforged.net/docs/1.21.1/resources/",
                "https://docs.neoforged.net/docs/1.21.1/concepts/events",
            ),
            bs_kwargs=dict(
                parse_only=SoupStrainer(
                    name=("h2", "h3", "p", "li", "div")
                )
            )
        )
        return loader.load()

    def _init_knowledge_base(self):
        docs = self._load_documents()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=300)
        all_splits = text_splitter.split_documents(docs)

        BATCH_SIZE = 64
        for i in range(0, len(all_splits), BATCH_SIZE):
            batch = all_splits[i:i+BATCH_SIZE]
            self.vector_store.add_documents(batch)

    def _generate_directory_structure(self, path: Path, indent: int = 0) -> str:
        """生成目录结构树"""
        result = []
        prefix = "│   " * (indent - 1) + "├── " if indent > 0 else ""

        if path.is_dir():
            result.append(f"{prefix}{path.name}/")
            children = sorted(path.iterdir())
            for i, item in enumerate(children):
                child_result = self._generate_directory_structure(
                    item, indent + 1)
                result.append(child_result)
        else:
            result.append(f"{prefix}{path.name}")

        return "\n".join(result)

    def directory_operation_node(self, state: AgentState) -> AgentState:
        """使用大模型分析任务描述并执行目录操作"""

        # 1. 准备大模型提示
        directory_prompt = ChatPromptTemplate.from_messages([
            ("system", modmaster.prompts.PROMPT_ROLE),
            ("user", """
             用户希望完成如下任务：
             {task}
             现在需要你分析一下完成这一任务是否需要调整现有目录结构。
             现有文件结构（src文件夹内）：
             {directory_structure}
             --------
             注意事项：
             1. 只需要分析是否需要调整目录！不需要输出是否调整文件！
             2. 注意代码的模块性，合理分配代码所在的目录。例如，若需要注册一些东西，而尚没有registry包，则应建立。若需要建立时间处理函数，而尚没有events包，则应在合适位置建立。
             --------
             请分析是否需要执行以下操作：
             create_directory: 创建新目录，需要 target_path
             move_directory: 移动目录（包含所有内容），需要 source_path 和 target_path"
             rename_directory: 重命名目录，需要 source_path 和 target_path
             输出格式：JSON数组，每个元素包含 action, source_path(仅移动/重命名), target_path
             所有目录均相对于项目根目录下的src目录
             --------
             如果需要操作，请提供JSON数组格式的操作计划，包含操作类型和路径。
             如果不需要操作，请输出空JSON数组。
             除了JSON数组，不要输出其他内容。
             """)
        ])

        # 2. 创建解析器
        class DirectoryOperation(TypedDict):
            action: Literal["create_directory",
                            "move_directory", "rename_directory"]
            source_path: Optional[str]
            target_path: str

        parser = JsonOutputParser(pydantic_object=DirectoryOperation)

        # 3. 构建完整链
        chain = directory_prompt | self.chat_model | parser

        # 4. 调用大模型获取操作建议
        try:
            response = chain.invoke({
                "directory_structure": state["directory_structure"],
                "task": state["task"]
            })

            # 验证并转换响应为 List[DirectoryOperation]
            operations: List[DirectoryOperation] = []
            if isinstance(response, list):
                for op in response:
                    if (isinstance(op, dict) and "action" in op and "target_path" in op):
                        # 创建符合DirectoryOperation类型的操作项
                        valid_op = DirectoryOperation(
                            action=op["action"],
                            target_path=op["target_path"],
                            source_path=op.get("source_path")
                        )
                        operations.append(valid_op)
        except Exception as e:
            print(f"目录操作分析失败: {e}")
            operations = []

        # 5. 执行目录操作并记录
        actions = []
        for op in operations:
            action_type = op["action"]
            target_path = op["target_path"]
            try:
                if action_type == "create_directory":
                    full_path = self.src_path / target_path
                    if not full_path.exists():
                        full_path.mkdir(parents=True, exist_ok=True)
                        actions.append(f"创建目录: {target_path}")

                elif action_type in ("move_directory", "rename_directory"):
                    source_path = op.get("source_path")
                    if not source_path:
                        continue  # 跳过缺少源路径的操作

                    src = self.src_path / source_path
                    dest = self.src_path / target_path

                    # 确保源目录存在
                    if not src.exists() or not src.is_dir():
                        print(f"源目录不存在: {source_path}")
                        continue

                    # 确保目标目录不存在
                    if dest.exists():
                        print(f"目标目录已存在: {target_path}")
                        continue

                    # 移动/重命名目录
                    shutil.move(str(src), str(dest))

                    if action_type == "move_directory":
                        actions.append(
                            f"移动目录: {source_path} -> {target_path}"
                        )
                    else:
                        actions.append(
                            f"重命名目录: {source_path} -> {target_path}"
                        )

            except Exception as e:
                print(f"执行目录操作失败: {op} - {e}")
                actions.append(f"操作失败: {action_type} {target_path}")

        # 6. 更新目录结构
        dir_structure = self._generate_directory_structure(self.src_path)

        return {
            "directory_actions": actions,
            "directory_structure": dir_structure
        }  # type: ignore

    def code_view_node(self, state: AgentState) -> AgentState:
        """使用大模型识别并读取相关代码文件"""
        # 1. 准备代码识别提示模板
        code_view_prompt = ChatPromptTemplate.from_messages([
            ("system", modmaster.prompts.PROMPT_ROLE),
            ("user",
             """
             用户希望完成如下任务：
             {task}
             现在需要你分析一下列出需要查看的Java文件路径（所有路径均相对于项目根目录下的src目录）。
             现有文件结构（src文件夹内）：
             {directory_structure}
             只查看你认为有必要的源文件。
             输出格式：JSON数组
             例如：["main/java/com/example/App.java",...]
             如果任务不需要查看任何文件，输出空数组 []。
             不要返回JSON数组之外的任何内容。
             """
             )
        ])

        # 2. 创建解析器
        parser = JsonOutputParser()

        # 3. 构建链
        chain = code_view_prompt | self.chat_model | parser

        # 4. 调用大模型获取文件路径建议
        try:
            response = chain.invoke({
                "directory_structure": state["directory_structure"],
                "task": state["task"]
            })

            # 验证并提取文件路径列表
            file_paths: List[str] = []
            if isinstance(response, list):
                for path in response:
                    if isinstance(path, str) and path.endswith(".java"):
                        file_paths.append(path)
        except Exception as e:
            print(f"Error getting file: {e}")
            file_paths = []

        # 5. 读取文件内容
        needed_files = {}
        for rel_path in file_paths:
            try:
                file_path = self.src_path / rel_path
                if file_path.exists() and file_path.is_file():
                    content = file_path.read_text(encoding='utf-8')
                    needed_files[rel_path] = content
                else:
                    print(f"File don't exist: {rel_path}")
            except Exception as e:
                print(f"Error reading file: {rel_path} - {str(e)}")

        return {"needed_files": needed_files}  # type: ignore

    def information_retrieval_node(self, state: AgentState) -> AgentState:
        """使用RAG从知识库检索相关信息"""
        retrieved_docs = self.vector_store.similarity_search(
            state["task"], 12)

        return {"retrieved_info": retrieved_docs}  # type: ignore

    class FileOperation(TypedDict):
        """文件操作类型定义"""
        operation: Literal["create", "modify", "delete"]
        file_path: str
        content: str
        reason: str

    def analysis_node(self, state: AgentState) -> AgentState:
        """使用大模型分析需要进行的文件操作"""
        task = state["task"]
        needed_files = state["needed_files"]
        retrieved_info = state.get("retrieved_info", "")
        directory_structure = state.get("directory_structure", "")

        # 1. 准备代码分析提示模板
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", modmaster.prompts.PROMPT_ROLE),
            ("user",
             """
             用户希望完成如下任务：
             {task}
             现在需要你具体完成文件级别的操作。
             --------
             现有文件结构（src文件夹内）：
             {directory_structure}
             --------
             一些源代码如下：
             {file_contents}
             --------
             检索到的相关信息：（注意，这些信息不一定都有用，请自行参考）
             {retrieved_info}
             --------
             注意事项：
             1. 只进行你认为有必要的操作。
             2. 注意代码的模块性，合理分配代码所在的目录。
             3. 注意补全所有的import，检查是否有遗漏的import。不要删除原有的import。
             4. 注意我们用的是neoforge，注意MC版本。不要写成其他版本的写法。
                例如，对于注册来说，一般而言应该使用DeferredRegister，并将结果保存在Supplier中。
             --------
             输出格式：
             1. 使用JSON数组格式，每个元素代表一个文件操作
             2. 每个操作包含以下字段：
                - operation: 操作类型（create/modify/delete）
                - file_path: 文件相对路径（所有路径均相对于项目根目录下的src目录，如：main/java/com/example/App.java）
                - content: 文件完整内容（create/modify操作需要，delete可为空）
             3. 操作类型说明：
                - create: 创建新文件，提供完整文件内容
                - modify: 修改文件，提供修改后的完整文件内容
                - delete: 删除文件，content字段可为空
             4. 对于modify操作，必须提供完整的修改后文件内容
             5. 如果没有需要操作的文件，返回空JSON数组[]
             不要返回JSON数组之外的任何内容。
             """
             )
        ])

        # 2. 准备文件内容上下文
        file_contents = "\n\n".join(
            [f"文件: {path}\n```java\n{content}\n```"
             for path, content in needed_files.items()]
        )

        # 3. 创建解析器
        parser = JsonOutputParser(pydantic_object=None)

        # 4. 构建链
        chain = analysis_prompt | self.chat_model | parser

        # 5. 调用大模型获取操作计划
        try:
            # 调用大模型获取修改建议
            response = chain.invoke({
                "task": task,
                "directory_structure": directory_structure,
                "retrieved_info": retrieved_info,
                "file_contents": file_contents
            })

            # 验证并转换响应为 List[FileOperation]
            operations: List[dict] = []
            if isinstance(response, list):
                for op in response:
                    if (isinstance(op, dict) and
                        "operation" in op and
                        "file_path" in op and
                            "content" in op):

                        # 验证操作类型
                        if op["operation"] not in ("create", "modify", "delete"):
                            continue

                        # 验证文件路径
                        if not op["file_path"].endswith(".java"):
                            continue

                        # 对于删除操作，内容可以为空
                        if op["operation"] == "delete":
                            op["content"] = ""

                        operations.append(op)

            # 将操作列表转换为文件路径为键的字典
            analysis_result = {
                op["file_path"]: {
                    "operation": op["operation"],
                    "content": op["content"]
                } for op in operations
            }
        except Exception as e:
            print(f"文件操作分析失败: {e}")
            analysis_result = {}

        return {"analysis_result": analysis_result}  # type: ignore

    def execution_node(self, state: AgentState) -> AgentState:
        """执行文件操作计划"""
        analysis_result = state["analysis_result"]
        modification_report = []

        for file_path, operation_info in analysis_result.items():
            operation = operation_info["operation"]  # type: ignore
            content = operation_info["content"]  # type: ignore
            full_path = self.src_path / file_path

            try:
                if operation == "create":
                    # 确保目录存在
                    full_path.parent.mkdir(parents=True, exist_ok=True)

                    # 写入文件内容
                    full_path.write_text(content, encoding="utf-8")
                    modification_report.append(
                        f"Created file: {file_path}"
                    )

                elif operation == "modify":
                    # 确保文件存在
                    if not full_path.exists():
                        modification_report.append(
                            f"Failed to modify: {full_path} (File not found)"
                        )
                        continue

                    # 写入修改后的内容
                    full_path.write_text(content, encoding="utf-8")
                    modification_report.append(
                        f"Modified file: {file_path}"
                    )

                elif operation == "delete":
                    # 确保文件存在
                    if not full_path.exists():
                        modification_report.append(
                            f"Failed to delete: {file_path} (File not found)"
                        )
                        continue

                    # 删除文件
                    os.remove(full_path)
                    modification_report.append(
                        f"Deleted file: {file_path}"
                    )

            except Exception as e:
                modification_report.append(
                    f"Operation failed: {operation} {file_path} - {str(e)}"
                )

        return {"modification_report": modification_report}  # type: ignore

    def _build_workflow(self) -> Callable:
        """构建LangGraph工作流"""
        workflow = StateGraph(AgentState)

        # 添加节点
        workflow.add_node("directory_operation", self.directory_operation_node)
        workflow.add_node("code_view", self.code_view_node)
        workflow.add_node("info_retrieval", self.information_retrieval_node)
        workflow.add_node("analysis", self.analysis_node)
        workflow.add_node("execution", self.execution_node)

        # 设置工作流
        workflow.set_entry_point("directory_operation")
        workflow.add_edge("directory_operation", "code_view")
        workflow.add_edge("code_view", "info_retrieval")
        workflow.add_edge("info_retrieval", "analysis")
        workflow.add_edge("analysis", "execution")
        workflow.add_edge("execution", END)

        return workflow.compile()  # type: ignore

    def run(self, task: str) -> AgentState:
        dir_structure = self._generate_directory_structure(self.src_path)

        initial_state: AgentState = {
            "task": task,
            "src_path": self.src_path,
            "directory_structure": dir_structure,
            "directory_actions": [],
            "needed_files": {},
            "retrieved_info": "",
            "analysis_result": {},
            "modification_report": []
        }

        return self.workflow.invoke(initial_state)  # type: ignore
