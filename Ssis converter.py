import xml.etree.ElementTree as ET
from typing import Dict, List, Any
import os

class SSISConverter:
    def __init__(self, ssis_file_path: str):
        """Initialize the converter with an SSIS package file."""
        self.file_path = ssis_file_path
        self.tree = ET.parse(ssis_file_path)
        self.root = self.tree.getroot()
        self.namespaces = {
            'dt': 'www.microsoft.com/SqlServer/Dts',
            'dts': 'www.microsoft.com/SqlServer/Dts',
            'xs': 'http://www.w3.org/2001/XMLSchema'
        }
        self.python_code = []
        self.imports = set()
        self.variables = {}
        
    def get_elements(self, path: str) -> List:
        """Helper to find elements with namespace handling."""
        for ns_key, ns_url in self.namespaces.items():
            try:
                elements = self.root.findall(f'.//{{{ns_url}}}{path}')
                if elements:
                    return elements
            except:
                pass
        return []
    
    def extract_variables(self):
        """Extract package variables."""
        self.add_import("os")
        self.python_code.append("# Package Variables")
        self.python_code.append("variables = {}")
        
        var_elements = self.get_elements('Variable')
        for var in var_elements:
            var_name = var.get('ObjectName', 'unknown')
            var_type = var.get('DataType', 'string')
            self.variables[var_name] = var_type
            self.python_code.append(f"variables['{var_name}'] = None  # Type: {var_type}")
        
        self.python_code.append("")
    
    def extract_connections(self) -> Dict[str, str]:
        """Extract connection strings."""
        self.add_import("pyodbc")
        connections = {}
        self.python_code.append("# Connections")
        
        conn_elements = self.get_elements('ConnectionManager')
        for conn in conn_elements:
            conn_name = conn.get('ObjectName', 'unknown')
            conn_type = conn.get('CreationName', 'unknown')
            
            connections[conn_name] = {
                'name': conn_name,
                'type': conn_type
            }
            
            self.python_code.append(f"\n# Connection: {conn_name}")
            self.python_code.append(f"# Type: {conn_type}")
            self.python_code.append(f"{conn_name}_conn = None  # Configure connection string")
        
        self.python_code.append("")
        return connections
    
    def extract_tasks(self):
        """Extract and convert tasks."""
        self.python_code.append("\n# Tasks")
        
        tasks = self.get_elements('Executable')
        for task in tasks:
            task_name = task.get('ObjectName', 'unknown')
            task_type = task.get('ExecutableType', 'unknown')
            
            self.python_code.append(f"\ndef {self._sanitize_name(task_name)}():")
            self.python_code.append(f"    \"\"\"Task: {task_name} (Type: {task_type})\"\"\"")
            self.python_code.append("    try:")
            
            if 'SQL' in task_type:
                self._convert_sql_task(task)
            elif 'Script' in task_type:
                self._convert_script_task(task)
            elif 'File' in task_type:
                self._convert_file_task(task)
            elif 'DataFlow' in task_type:
                self._convert_dataflow_task(task)
            else:
                self.python_code.append(f"        print(f'Executing task: {task_name}')")
            
            self.python_code.append("    except Exception as e:")
            self.python_code.append(f"        print(f'Error in {task_name}: {{str(e)}}')")
            self.python_code.append("        raise")
    
    def _convert_sql_task(self, task):
        """Convert SQL Execute Task."""
        self.add_import("pyodbc")
        sql_statement = task.get('SqlStatementSource', '')
        self.python_code.append(f"        # SQL Statement: {sql_statement[:50]}...")
        self.python_code.append("        conn = pyodbc.connect('your_connection_string')")
        self.python_code.append("        cursor = conn.cursor()")
        self.python_code.append(f"        cursor.execute('''")
        self.python_code.append(f"            {sql_statement}")
        self.python_code.append(f"        ''')")
        self.python_code.append("        conn.commit()")
        self.python_code.append("        cursor.close()")
        self.python_code.append("        conn.close()")
    
    def _convert_script_task(self, task):
        """Convert Script Task."""
        self.python_code.append("        # Script Task - Custom logic here")
        self.python_code.append("        pass")
    
    def _convert_file_task(self, task):
        """Convert File System Task."""
        self.add_import("shutil")
        self.python_code.append("        # File operation")
        self.python_code.append("        # shutil.copy(source, destination)")
        self.python_code.append("        pass")
    
    def _convert_dataflow_task(self, task):
        """Convert Data Flow Task."""
        self.add_import("pandas")
        self.python_code.append("        # Data Flow Task")
        self.python_code.append("        # df = pd.read_sql(query, conn)")
        self.python_code.append("        pass")
    
    def _sanitize_name(self, name: str) -> str:
        """Convert SSIS object names to valid Python function names."""
        name = name.replace(' ', '_').replace('-', '_')
        name = ''.join(c for c in name if c.isalnum() or c == '_')
        return name.lower()
    
    def add_import(self, import_name: str):
        """Add an import statement."""
        self.imports.add(import_name)
    
    def generate_main_execution(self):
        """Generate main execution flow."""
        self.python_code.append("\ndef main():")
        self.python_code.append("    \"\"\"Main execution flow.\"\"\"")
        self.python_code.append("    try:")
        
        tasks = self.get_elements('Executable')
        for task in tasks:
            task_name = self._sanitize_name(task.get('ObjectName', 'unknown'))
            self.python_code.append(f"        {task_name}()")
        
        self.python_code.append("        print('Package executed successfully')")
        self.python_code.append("    except Exception as e:")
        self.python_code.append("        print(f'Package execution failed: {str(e)}')")
        self.python_code.append("        raise")
        self.python_code.append("")
        self.python_code.append("if __name__ == '__main__':")
        self.python_code.append("    main()")
    
    def convert(self) -> str:
        """Convert SSIS package to Python code."""
        # Add imports at the beginning
        import_section = ['"""', 'Auto-generated Python code from SSIS Package', '"""', '']
        import_section.extend([f'import {imp}' for imp in sorted(self.imports)])
        import_section.append('')
        
        # Extract package components
        self.extract_variables()
        self.extract_connections()
        self.extract_tasks()
        self.generate_main_execution()
        
        # Combine all sections
        full_code = import_section + self.python_code
        return '\n'.join(full_code)
    
    def save_to_file(self, output_path: str):
        """Save converted Python code to a file."""
        code = self.convert()
        with open(output_path, 'w') as f:
            f.write(code)
        print(f"Python code saved to {output_path}")


def convert_ssis_package(ssis_file: str, output_file: str = None):
    """
    Convenience function to convert SSIS package.
    
    Args:
        ssis_file: Path to the SSIS package file (.dtsx)
        output_file: Output path for Python file (optional)
    """
    if not os.path.exists(ssis_file):
        raise FileNotFoundError(f"SSIS file not found: {ssis_file}")
    
    if output_file is None:
        output_file = ssis_file.replace('.dtsx', '.py')
    
    converter = SSISConverter(ssis_file)
    converter.save_to_file(output_file)
    
    return converter.convert()


# Example usage
if __name__ == '__main__':
    # Replace with your SSIS file path
    ssis_file = 'path/to/your/package.dtsx'
    output_file = 'converted_package.py'
    
    try:
        python_code = convert_ssis_package(ssis_file, output_file)
        print("Conversion successful!")
        print("\nGenerated Python code:")
        print(python_code)
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
