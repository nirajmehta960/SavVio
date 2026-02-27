import os
import ast

directory = '/Users/nirajmehta/Documents/SavVio/data-pipeline/dags/src'

class LoggingCallVisitor(ast.NodeVisitor):
    def __init__(self):
        self.calls = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Attribute):
            if isinstance(node.func.value, ast.Name) and getattr(node.func.value, 'id', '') == 'logging':
                if getattr(node.func, 'attr', '') == 'basicConfig':
                    self.calls.append(node)
        self.generic_visit(node)

for root, dirs, files in os.walk(directory):
    for file in files:
        if file.endswith('.py') and file != 'utils.py':
            filepath = os.path.join(root, file)
            with open(filepath, 'r') as f:
                content = f.read()

            try:
                tree = ast.parse(content)
            except Exception as e:
                print(f"Error parsing {filepath}: {e}")
                continue

            visitor = LoggingCallVisitor()
            visitor.visit(tree)

            if visitor.calls:
                print(f"Found calls in {filepath}")
