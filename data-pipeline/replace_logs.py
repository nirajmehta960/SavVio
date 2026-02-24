import ast
import os

files = [
    'dags/src/validation/run_validation.py',
    'dags/src/validation/validate/raw_validator.py',
    'dags/src/validation/validate/feature_validator.py',
    'dags/src/validation/validate/processed_validator.py',
    'dags/src/validation/anomaly/anomaly_validator.py',
    'dags/src/preprocess/run_preprocessing.py',
    'dags/src/ingestion/gcs_loader.py',
    'dags/src/database/db_connection.py',
    'dags/src/database/upload_to_db.py',
    'dags/src/database/run_database.py',
    'dags/src/database/vector_embed.py',
    'dags/src/ingestion/run_ingestion.py',
    'dags/src/ingestion/api_loader.py'
]

base_dir = '/Users/nirajmehta/Documents/SavVio/data-pipeline/'

class LoggingCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.calls = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and getattr(node.func.value, 'id', '') == 'logging':
                if getattr(node.func, 'attr', '') == 'basicConfig':
                    self.calls.append(node)
        self.generic_visit(node)

for f in files:
    filepath = os.path.join(base_dir, f)
    with open(filepath, 'r') as file:
        content = file.read()
        
    tree = ast.parse(content)
    visitor = LoggingCallVisitor()
    visitor.visit(tree)
    
    if not visitor.calls:
        continue
        
    lines = content.split('\n')
    
    # Process from bottom up to preserve line numbers
    calls = sorted(visitor.calls, key=lambda n: n.lineno, reverse=True)
    
    for call in calls:
        start_line = call.lineno - 1
        end_line = call.end_lineno - 1
        
        # Determine indentation from start line
        original_start = lines[start_line]
        indent_len = len(original_start) - len(original_start.lstrip(' \t'))
        indent = original_start[:indent_len]
        
        # Replace the first line of the call with our replacement lines
        lines[start_line] =  f"{indent}from src.utils import setup_logging\n{indent}setup_logging()"
        
        # Blank out the rest of the lines spanned by this AST node
        for i in range(start_line + 1, end_line + 1):
            lines[i] = None
            
    # Filter out blanked lines
    new_lines = [line for line in lines if line is not None]
    
    with open(filepath, 'w') as file:
        file.write('\n'.join(new_lines))
        
    print(f"Updated {f}")
