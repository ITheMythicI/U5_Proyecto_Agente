"""
Módulo de Ejecución Controlada de Código Analítico (Agente Programador).
Permite generar y ejecutar código Python y SQL con validación por AST,
lista blanca de librerías, bloqueo de llamadas del sistema y captura segura de salidas.
"""

import ast
import io
import sys
import time
from typing import Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import duckdb
import pandas as pd
import numpy as np
import polars as pl
import math

from .config import config

# Módulos analíticos estrictamente autorizados
ALLOWED_MODULES = {
    "math", "numpy", "np", "pandas", "pd", "polars", "pl",
    "duckdb", "scipy", "sklearn", "matplotlib", "plt",
    "seaborn", "sns", "datetime", "collections", "itertools"
}

# Funciones y atributos intrínsecamente peligrosos
FORBIDDEN_CALLS = {
    "open", "eval", "exec", "compile", "__import__", "input",
    "breakpoint", "help", "quit", "exit"
}

FORBIDDEN_ATTRIBUTES = {
    "__class__", "__bases__", "__subclasses__", "__globals__",
    "__code__", "__reduce__", "__reduce_ex__", "__builtins__"
}


class CodeSecurityValidator(ast.NodeVisitor):
    """Inspecciona el Árbol de Sintaxis Abstracta (AST) para garantizar inocuidad."""

    def __init__(self):
        self.errors: List[str] = []

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            base_module = alias.name.split(".")[0]
            if base_module not in ALLOWED_MODULES:
                self.errors.append(f"Módulo '{alias.name}' bloqueado. Solo se permiten librerías analíticas aprobadas.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            base_module = node.module.split(".")[0]
            if base_module not in ALLOWED_MODULES:
                self.errors.append(f"Módulo '{node.module}' bloqueado. Solo se permiten librerías analíticas aprobadas.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Bloquear llamadas a funciones como open(), eval(), etc.
        if isinstance(node.func, ast.Name):
            if node.func.id in FORBIDDEN_CALLS:
                self.errors.append(f"Llamada a función prohibida '{node.func.id}()'.")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        # Bloquear introspección de atributos dunder peligrosos
        if node.attr in FORBIDDEN_ATTRIBUTES:
            self.errors.append(f"Acceso a atributo restringido '{node.attr}'.")
        self.generic_visit(node)


class ControlledCodeExecutor:
    """Entorno de ejecución controlado y aislado para scripts analíticos."""

    def __init__(self, timeout_seconds: int = 15):
        self.timeout_seconds = timeout_seconds
        self.execution_logs: List[Dict[str, Any]] = []

    def validate_code(self, code_str: str) -> List[str]:
        """Verifica la seguridad sintáctica y estática del código."""
        try:
            tree = ast.parse(code_str)
        except SyntaxError as e:
            return [f"Error de sintaxis en el código: {e}"]

        validator = CodeSecurityValidator()
        validator.visit(tree)
        return validator.errors

    def execute_python(self, code_str: str) -> Dict[str, Any]:
        """
        Ejecuta código analítico Python en un entorno restringido.
        Por defecto permanece deshabilitado por política de seguridad y mínimo privilegio.
        """
        if not config.enable_code_execution:
            return {
                "status": "disabled",
                "message": "La ejecución arbitraria de código dinámico está deshabilitada por defecto por política de seguridad y aislamiento del sistema. Las consultas y cálculos analíticos deben realizarse mediante las herramientas específicas del proyecto (query_duckdb, predict_match, get_team_profile).",
                "code": code_str
            }

        start_time = time.time()
        validation_errors = self.validate_code(code_str)

        if validation_errors:
            entry = {
                "status": "security_violation",
                "code": code_str,
                "errors": validation_errors,
                "duration_seconds": round(time.time() - start_time, 4)
            }
            self.execution_logs.append(entry)
            return entry

        def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
            base_module = name.split(".")[0]
            if base_module not in ALLOWED_MODULES:
                raise ImportError(f"Módulo '{name}' no permitido. Solo se permiten librerías analíticas aprobadas.")
            return __import__(name, globals, locals, fromlist, level)

        def query_db(sql_query: str):
            """Ejecuta una consulta SQL segura en DuckDB y devuelve un DataFrame de pandas."""
            from .tools import query_duckdb
            res = query_duckdb(sql_query)
            if res["status"] != "success":
                raise RuntimeError(res.get("message", "Error ejecutando consulta SQL."))
            return pd.DataFrame(res["rows"])

        # Preparar espacio de nombres seguro
        safe_globals = {
            "__builtins__": {
                "abs": abs, "all": all, "any": any, "bool": bool,
                "dict": dict, "enumerate": enumerate, "float": float,
                "int": int, "len": len, "list": list, "map": map,
                "max": max, "min": min, "pow": pow, "print": print,
                "range": range, "reversed": reversed, "round": round,
                "set": set, "sorted": sorted, "str": str, "sum": sum,
                "tuple": tuple, "zip": zip, "__import__": safe_import
            },
            "math": math,
            "np": np,
            "numpy": np,
            "pd": pd,
            "pandas": pd,
            "pl": pl,
            "polars": pl,
            "duckdb": duckdb,
            "query_db": query_db,
            "LAKEHOUSE_PATH": str(config.lakehouse_db_path)
        }
        safe_locals: Dict[str, Any] = {}

        # Redirigir stdout para capturar impresiones
        stdout_capture = io.StringIO()
        original_stdout = sys.stdout

        def _run():
            sys.stdout = stdout_capture
            try:
                exec(code_str, safe_globals, safe_locals)
                return True, None
            except Exception as e:
                return False, str(e)
            finally:
                sys.stdout = original_stdout

        try:
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_run)
                success, error_msg = future.result(timeout=self.timeout_seconds)
        except TimeoutError:
            success = False
            error_msg = f"Tiempo de ejecución excedido (límite: {self.timeout_seconds} segundos)."
        except Exception as e:
            success = False
            error_msg = f"Fallo imprevisto en ejecución: {str(e)}"
        finally:
            sys.stdout = original_stdout

        captured_output = stdout_capture.getvalue()
        duration = round(time.time() - start_time, 4)

        # Serializar variables locales resultantes filtrando tipos complejos
        serializable_locals = {}
        for k, v in safe_locals.items():
            if not k.startswith("_"):
                if isinstance(v, (int, float, str, bool, list, dict)):
                    serializable_locals[k] = v
                elif hasattr(v, "to_dict"):
                    try:
                        serializable_locals[k] = v.to_dict()
                    except Exception:
                        serializable_locals[k] = str(v)
                else:
                    serializable_locals[k] = str(type(v))

        result_payload = {
            "status": "success" if success else "runtime_error",
            "stdout": captured_output.strip(),
            "returned_variables": serializable_locals,
            "error": error_msg,
            "duration_seconds": duration
        }

        self.execution_logs.append({
            "status": result_payload["status"],
            "code": code_str,
            "output_summary": captured_output[:200],
            "duration_seconds": duration
        })

        return result_payload
