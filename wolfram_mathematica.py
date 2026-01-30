# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "mcp[cli]",
#     "wolframclient",
# ]
# ///
import os
from typing import Any, Dict, Optional

from mcp.server.fastmcp import FastMCP
from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wlexpr

from animalid import AnimalIdGenerator

# --- 初始化 ---

# 1. 初始化 MCP 服务器
mcp = FastMCP("mathematica")

# 2. 从环境变量加载密钥并初始化 AnimalID 生成器
#    请设置 'ANIMALID_SECRET_KEY' 环境变量以确保安全
secret_key = os.getenv("ANIMALID_SECRET_KEY", "default-secret-key-for-dev")
if secret_key == "default-secret-key-for-dev":
    print(
        "Warning: Using default secret key. Please set ANIMALID_SECRET_KEY environment variable for production."
    )
id_generator = AnimalIdGenerator(secret_key=secret_key)

# 3. 用于存储活动会话的字典
#    键是 animal_id，值是 WolframLanguageSession 对象
sessions: Dict[str, WolframLanguageSession] = {}


# --- Kernel Path Handling (NO auto-search) ---


def get_kernel_path_from_env() -> Optional[str]:
    """
    If WOLFRAM_KERNEL_PATH is set, use it.
    If it is set but invalid, raise a clear error.
    If it is not set, return None (wolframclient will attempt its own autodetection).
    """
    raw = os.getenv("WOLFRAM_KERNEL_PATH", "").strip()
    if not raw:
        return None

    # Expand ~ and env vars
    path = os.path.expandvars(os.path.expanduser(raw))

    if not os.path.exists(path):
        raise RuntimeError(f"WOLFRAM_KERNEL_PATH was set but does not exist: '{path}'")
    if not os.path.isfile(path):
        raise RuntimeError(f"WOLFRAM_KERNEL_PATH was set but is not a file: '{path}'")
    if not os.access(path, os.X_OK):
        raise RuntimeError(
            f"WOLFRAM_KERNEL_PATH was set but is not executable: '{path}'"
        )

    print(f"Using Wolfram Kernel from WOLFRAM_KERNEL_PATH: {path}")
    return path


# Evaluate once at startup (no searching, just env or None)
KERNEL_PATH = get_kernel_path_from_env()


# --- 工具定义 ---


@mcp.tool()
def create_mathematica_session() -> str:
    """
    Creates and initializes a new, isolated Wolfram Language session.

    This tool is the first step for any Mathematica-related task. It returns a unique,
    secure session identifier (e.g., 'fox-wolf-bear-lion') that you MUST use in
    subsequent calls to 'execute_code' and 'close_session'.

    Each session is completely independent and maintains its own state (variables,
    function definitions, etc.).

    Returns:
        A string containing a success message and the unique session ID.
        Example: "Session created successfully. Your session ID is: bee-sloth-auk-mole"
    """
    session_id = id_generator.generate()
    try:
        # If KERNEL_PATH is provided, force that kernel.
        # Otherwise, let wolframclient attempt its own autodetection.
        if KERNEL_PATH:
            session = WolframLanguageSession(kernel=KERNEL_PATH)
        else:
            session = WolframLanguageSession()

        sessions[session_id] = session
        return f"Session created successfully. Your session ID is: {session_id}"
    except Exception as e:
        raise RuntimeError(f"Failed to create Wolfram Language session: {e}") from e


@mcp.tool()
async def execute_mathematica_code(session_id: str, code: str) -> Any:
    """
    Executes a string of Wolfram Language code within a specific, active session.

    To use this tool, you must provide a valid 'session_id' obtained from a
    previous call to 'create_session'. The code will be executed in the
    context of that session, meaning it can access variables and functions
    defined in previous calls within the same session.

    Args:
        session_id: The unique identifier for an active session, provided by
                    'create_session'. Example: 'bee-sloth-auk-mole'.
        code: A string containing the Wolfram Language code to be executed.
              The code should be syntactically correct.
              Example 1 (simple calculation): 'Total[Range[100]]'
              Example 2 (symbolic computation): 'Solve[x^2 - 5x + 6 == 0, x]'
              Example 3 (data visualization): 'Plot[Sin[x], {x, 0, 2 Pi}]'

    Returns:
        The direct result of the code execution from the Wolfram Engine. The data
        type can vary (e.g., integer, list, string, or a complex expression).
        For plots, it may return a representation of the graphics object.
    """
    # 验证 session_id 格式和校验和
    if not id_generator.verify(session_id):
        raise ValueError("Invalid session ID. It might be malformed or tampered with.")

    # 查找会话
    session = sessions.get(session_id)
    if not session:
        raise ValueError(
            f"Session with ID '{session_id}' not found or has been closed."
        )

    try:
        # 将字符串代码转换为 wlexpr 对象并执行
        result = session.evaluate(wlexpr(code))
        return result
    except Exception as e:
        raise RuntimeError(
            f"An error occurred during execution in session '{session_id}': {e}"
        ) from e


@mcp.tool()
def close_mathematica_session(session_id: str) -> str:
    """
    Terminates a specific Wolfram Language session and releases all associated resources.

    It is good practice to call this tool when you are finished with a session
    to free up system memory and kernel licenses. Once a session is closed, its
    ID can no longer be used.

    Args:
        session_id: The unique identifier of the session you wish to close.
                    This must be an ID from an active, open session.
                    Example: 'bee-sloth-auk-mole'.

    Returns:
        A confirmation message indicating that the session was successfully closed.
    """
    # 验证 session_id
    if not id_generator.verify(session_id):
        raise ValueError("Invalid session ID.")

    session = sessions.get(session_id)
    if not session:
        raise ValueError(f"Session with ID '{session_id}' not found or already closed.")

    try:
        # 终止会话并从字典中移除
        session.terminate()
        del sessions[session_id]
        return f"Session '{session_id}' closed successfully."
    except Exception as e:
        raise RuntimeError(
            f"An error occurred while closing session '{session_id}': {e}"
        ) from e


# --- 运行服务器 ---

if __name__ == "__main__":
    # 检查 wolframclient 是否已安装
    try:
        import wolframclient  # noqa: F401
    except ImportError:
        print("Error: 'wolframclient' is not installed.")
        print("Please install it using: pip install wolframclient")
        raise SystemExit(1)

    if not KERNEL_PATH:
        print(
            "Note: WOLFRAM_KERNEL_PATH is not set. "
            "wolframclient will attempt automatic kernel detection."
        )

    mcp.run(transport="stdio")
