# transpiler.py (b1.0.0)
import re
import sys
import os
import argparse
import ast
from typing import List, Dict, Any, Optional, Tuple
from textwrap import dedent
from dataclasses import dataclass

# ==============================================================================
# --- Runtime Code Definition ---
# ==============================================================================

_RUNTIME_SOURCE = '''
# --- Sugarcoat Runtime ---

class SugarcoatRuntimeError(Exception):
    """Custom exception for Sugarcoat runtime errors"""
    pass

_SC_ALIAS_MAP = {
    'trim': 'strip',
    'upcase': 'upper',
    'downcase': 'lower',
    'keys': 'keys',
    'values': 'values',
    'items': 'items',
    'reverse': lambda seq: (seq.reverse() or seq) if isinstance(seq, list) else seq[::-1],
    
    'append': lambda seq, item: (seq.append(item) or seq) if isinstance(seq, list) else (seq + str(item)) if isinstance(seq, str) else seq,
    'push':   lambda seq, item: (seq.append(item) or seq) if isinstance(seq, list) else (seq + str(item)) if isinstance(seq, str) else seq,

    'pop': 'pop',
    'shift': lambda x: x.pop(0) if x else None,
    'unshift': lambda x, item: x.insert(0, item) or x,
    'length': len,
    'size': len,
    'empty?': lambda x: len(x) == 0 if hasattr(x, '__len__') else x is None,
    'nil?': lambda x: x is None,
    'include?': lambda x, item: item in x if hasattr(x, '__contains__') else False,
}

def _sc_call(value, func_name_str, *args, **kwargs):
    """The main dispatcher for Sugarcoat calls with a robust separation
       between method calls and global function calls
    """
    import builtins
    import inspect

    if value is None and func_name_str not in ['nil?', 'empty?']:
        if not args:
            raise SugarcoatRuntimeError(f"Cannot call '{func_name_str}' on None value")

    # Priority 1: Check for a method on the value itself
    if value is not None and hasattr(value, func_name_str):
        method = getattr(value, func_name_str)
        if callable(method):
            try:
                # Methods are called directly on the object
                # The piped value is implicit `self`, so we only pass `*args`
                result = method(*args, **kwargs)
                return value if result is None and func_name_str in ['append', 'extend', 'sort', 'reverse'] else result
            except TypeError as e:
                raise SugarcoatRuntimeError(f"Error calling method '{func_name_str}': {e}")

    # Priority 2: Check for a string alias that maps to a method
    if value is not None and func_name_str in _SC_ALIAS_MAP:
        alias_target_name = _SC_ALIAS_MAP[func_name_str]
        if isinstance(alias_target_name, str) and hasattr(value, alias_target_name):
            method = getattr(value, alias_target_name)
            if callable(method):
                try:
                    # Same logic as above: aliased methods are called directly.
                    result = method(*args, **kwargs)
                    return value if result is None and alias_target_name in ['append', 'extend', 'sort', 'reverse'] else result
                except TypeError as e:
                    raise SugarcoatRuntimeError(f"Error calling aliased method '{alias_target_name}': {e}")

    # Priority 3: Check for global functions, function aliases, and built-ins
    # For these, the piped `value` becomes the FIRST argument
    target_func = None
    if func_name_str in globals() and callable(globals()[func_name_str]):
        target_func = globals()[func_name_str]
    elif func_name_str in _SC_ALIAS_MAP and callable(_SC_ALIAS_MAP[func_name_str]):
        target_func = _SC_ALIAS_MAP[func_name_str]
    elif hasattr(builtins, func_name_str) and callable(getattr(builtins, func_name_str)):
        target_func = getattr(builtins, func_name_str)

    if target_func:
        # Construct the final argument list, with the piped value first
        final_args = (value, *args) if value is not None else args
        
        # Before calling, check if the function can actually accept the arguments
        try:
            sig = inspect.signature(target_func)
            # This will raise a TypeError if the arguments don't match the signature
            sig.bind(*final_args, **kwargs)
        except TypeError:
            # If binding fails and the function takes 0 args (like celebrate), call it with none.
            try:
                sig_check_no_args = inspect.signature(target_func)
                if len(sig_check_no_args.parameters) == 0:
                    final_args = ()
            except (ValueError, TypeError):
                pass # Proceed with original args if inspect fails
        
        try:
            return target_func(*final_args, **kwargs)
        except TypeError as e:
            raise SugarcoatRuntimeError(f"Error calling function '{func_name_str}': {e}")

    raise SugarcoatRuntimeError(f"Could not find function, method, or alias '{func_name_str}' for object of type {type(value).__name__}")

def _sc_smart_cast(value, target_func_name):
    """Attempts to convert value to expected type based on function signature"""
    if value is None:
        return value
        
    try:
        import inspect
        target_func = globals().get(target_func_name)
        if not callable(target_func):
            return value
            
        try:
            sig = inspect.signature(target_func)
            params = list(sig.parameters.values())
            if params:
                first_param = params[0]
                if first_param.annotation != inspect.Parameter.empty:
                    target_type = first_param.annotation
                    if callable(target_type):
                        try:
                            return target_type(value)
                        except (ValueError, TypeError):
                            pass
        except (ValueError, TypeError, AttributeError):
            pass
    except Exception:
        pass
    
    # Smart casting for strings
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            # Try int first
            try:
                return int(stripped)
            except ValueError:
                pass
            # Try float
            try:
                return float(stripped)
            except ValueError:
                pass
    
    return value

def _sc_get_attr(obj, key):
    """Intelligently gets an attribute or key from an object
       Tries attribute access first then key access for dicts
    """
    # 1. Try standard attribute access (for objects)
    if hasattr(obj, key):
        return getattr(obj, key)
    # 2. Try key access (for dictionaries)
    try:
        if isinstance(obj, dict) and key in obj:
            return obj[key]
    except TypeError:
        # This handles cases where `obj` is not a dict and doesn't support `in`
        pass
    # 3. If all else fails raise an error
    raise SugarcoatRuntimeError(f"Could not access property or key '{key}' on object of type {type(obj).__name__}")
'''

# ==============================================================================
# --- Transpiler Class ---
# ==============================================================================

@dataclass
class ParsedExpression:
    """Represents a parsed expression with metadata"""
    original: str
    processed: str
    has_assignment: bool = False
    assignment_var: Optional[str] = None

class Transpiler:
    """Sugarcoat transpiler class"""
    
    TERMINAL_FUNCTIONS = {'print', 'announce', 'save', 'log', 'raise', 'exit', 'quit'}
    PROPERTY_ALIASES = {'length', 'size', 'empty?', 'nil?', 'include?'}
    CONTROL_KEYWORDS = {'if', 'elif', 'else', 'match', 'case', 'for', 'while', 'try', 'except', 'finally'}
    
    def __init__(self):
        self.output: List[str] = []
        self.temp_var_count = 0
        self.line_number = 0
        self.current_indent = ""
        
    def get_temp_var(self) -> str:
        """Generate a unique temporary variable name"""
        self.temp_var_count += 1
        return f"_sc_val_{self.temp_var_count}"
    
    def _validate_python_syntax(self, code: str) -> bool:
        """Validate that generated Python code has valid syntax"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _safe_split_on_operators(self, line: str, operators: List[str]) -> List[str]:
        """Safely split line on operators"""
        
        # Sort operators from longest to shortest to ensure correct matching (e.g., '->>' before '->')
        operators = sorted(operators, key=len, reverse=True)
        
        parts = []
        current = ""
        in_string = None
        escape = False
        paren_depth = 0
        bracket_depth = 0
        brace_depth = 0
        
        i = 0
        while i < len(line):
            char = line[i]
            
            if escape:
                current += char
                escape = False
                i += 1
                continue
                
            if char == '\\':
                escape = True
                current += char
                i += 1
                continue
                
            if in_string:
                current += char
                if char == in_string:
                    in_string = None
                i += 1
                continue
                
            if char in "\"'":
                in_string = char
                current += char
                i += 1
                continue
                
            if char == '(': paren_depth += 1
            elif char == ')': paren_depth -= 1
            elif char == '[': bracket_depth += 1
            elif char == ']': bracket_depth -= 1
            elif char == '{': brace_depth += 1
            elif char == '}': brace_depth -= 1
                
            if paren_depth == 0 and bracket_depth == 0 and brace_depth == 0:
                found_op = None
                for op in operators:
                    if line[i:i+len(op)] == op:
                        found_op = op
                        break
                        
                if found_op:
                    parts.append(current.strip())
                    parts.append(found_op)
                    current = ""
                    i += len(found_op)
                    continue
                    
            current += char
            i += 1
            
        if current.strip():
            parts.append(current.strip())
            
        return parts

    def _has_operator(self, line: str, operators: List[str]) -> bool:
        """Check if line contains any of the specified operators outside of strings"""
        in_string = None
        escape = False
        paren_depth = 0
        
        i = 0
        while i < len(line):
            char = line[i]
            
            if escape:
                escape = False
                i += 1
                continue
                
            if char == '\\':
                escape = True
                i += 1
                continue
                
            if in_string:
                if char == in_string:
                    in_string = None
                i += 1
                continue
                
            if char in "\"'":
                in_string = char
                i += 1
                continue
                
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
                
            # Only check for operators at top level
            if paren_depth == 0:
                for op in operators:
                    if line[i:i+len(op)] == op:
                        return True
                        
            i += 1
            
        return False

    def _preprocess(self, code: str) -> str:
        """Preprocessing"""

        def f_string_replacer(match):
            quote = match.group(1)
            content = match.group(2)
            # Replace all instances of #{...} with {...}
            f_content = re.sub(r'#\{([^}]+)\}', r'{\1}', content)
            return f'f{quote}{f_content}{quote}'
        
        code = re.sub(r'(["\'])((?:(?!\1).)*?#\{.*?(?:(?!\1).)*)\1', f_string_replacer, code)

        # Constructor sugar
        code = re.sub(r'\.new\s*\(', '(', code)
    
        code = re.sub(
            r'^(\s*)(\w+)\s*\{([^}]+)\}\s*\{([^}]+)\}\s*$',
            r'\1if \4:\n\1    \2(\3)',
            code,
            flags=re.MULTILINE
        )
        
        new_lines = []
        in_class = False
        class_indent_level = -1
        
        for line_num, line in enumerate(code.splitlines(), 1):
            try:
                stripped = line.lstrip()
                indent = len(line) - len(stripped)
                
                if stripped:
                    if in_class and indent <= class_indent_level:
                        in_class = False
                    if stripped.startswith('class '):
                        in_class = True
                        class_indent_level = indent
                
                if in_class:
                    line = re.sub(r'(?<!\w)@(\w+)', r'self.\1', line)
                    line = re.sub(r'(\s*)def\s+initialize\b', r'\1def __init__', line)
                    
                    method_match = re.match(r'(\s*def\s+[a-zA-Z_]\w*)(.*)', line.rstrip())
                    if method_match:
                        prefix, rest = method_match.groups()
                        # Clean up the parameters part removing potential colons and parens
                        params_str = rest.strip().strip(':').strip('()')
                        params = [p.strip() for p in params_str.split(',') if p.strip()]
                        
                        # Make sure 'self' is the first parameter
                        if not params or params[0] != 'self':
                            params.insert(0, 'self')
                            
                        # Rebuild the line with correct syntax
                        line = f"{prefix}({', '.join(params)}):"

                new_lines.append(line)
                
            except Exception as e:
                print(f"Warning: Error processing line {line_num}: {e}", file=sys.stderr)
                new_lines.append(line)
        
        code = '\n'.join(new_lines)
        
        # Property aliases
        alias_pattern = '|'.join(re.escape(alias) for alias in self.PROPERTY_ALIASES)
        code = re.sub(
            fr'\b([a-zA-Z_][\w.\[\]\'\"]*?)\.({alias_pattern})\b',
            r'_sc_call(\1, "\2")',
            code
        )
        
        return code

    def _parse_ternary(self, expr: str) -> str:
        """Parse ternary expressions with proper precedence"""
        expr = expr.strip()
        if not expr:
            return expr
            
        # Find the question mark at the top level
        parts = self._safe_split_on_operators(expr, ['?'])
        if len(parts) < 3:
            return expr
            
        # Should be: condition ? true_expr : false_expr
        if parts[1] != '?':
            return expr
            
        condition = parts[0].strip()
        rest = ''.join(parts[2:]).strip()
        
        # Find the colon
        colon_parts = self._safe_split_on_operators(rest, [':'])
        if len(colon_parts) < 3 or colon_parts[1] != ':':
            return expr
            
        true_expr = colon_parts[0].strip()
        false_expr = ''.join(colon_parts[2:]).strip()
        
        # Recursively parse nested ternaries
        return f"({self._parse_ternary(true_expr)}) if ({self._parse_ternary(condition)}) else ({self._parse_ternary(false_expr)})"

    def _parse_expression(self, expr: str) -> ParsedExpression:
        """Parse an expression and extract assignment information"""
        expr = expr.strip()
        assignment_match = re.match(r'^\s*([a-zA-Z_]\w*)\s*=\s*(.*)', expr, re.DOTALL)
        
        if assignment_match:
            var_name, value_expr = assignment_match.groups()
            return ParsedExpression(
                original=expr,
                processed=value_expr.strip(),
                has_assignment=True,
                assignment_var=var_name.strip()
            )
        else:
            return ParsedExpression(
                original=expr,
                processed=expr,
                has_assignment=False
            )

    def transpile(self, code: str) -> str:
        """Main transpilation method"""
        try:
            code = self._preprocess(code)
            lines = code.splitlines()
            
            i = 0
            while i < len(lines):
                self.line_number = i + 1
                line = lines[i]
                stripped = line.lstrip()
                indent = line[:len(line) - len(stripped)]
                self.current_indent = indent
                
                if not stripped or stripped.startswith('#'):
                    self.output.append(line)
                    i += 1
                    continue

                # Priority 1: Check for and process ternary operators on a single line
                if self._has_operator(stripped, ['?']):
                    assignment_match = re.match(r'^(\s*\w+\s*=\s*)(.*)', line)
                    if assignment_match:
                        prefix, expr = assignment_match.groups()
                        self.output.append(f"{prefix}{self._parse_ternary(expr)}")
                    else:
                        self.output.append(self._parse_ternary(line))
                    
                    i += 1
                    continue # We've handled this line, move to the next.

                # Priority 2: Check if this line is the START of a control flow block
                if re.search(r'->\s*(if|match|for|while)', stripped):
                    block_lines = [line]
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j]
                        if not next_line.strip():
                            block_lines.append(next_line)
                            j += 1
                            continue
                            
                        next_stripped = next_line.lstrip()
                        next_indent_len = len(next_line) - len(next_stripped)
                        
                        if next_indent_len > len(indent) or \
                        (next_indent_len == len(indent) and next_stripped.startswith(('->', '->>', '~>'))):
                            block_lines.append(next_line)
                            j += 1
                        else:
                            break
                    
                    self._transpile_control_flow_block(block_lines, indent, self.line_number)
                    i = j

                # Priority 3: Check for a simple belt chain.
                elif self._has_operator(stripped, ['->', '->>', '~>']):
                    block_lines = [line]
                    j = i + 1
                    while j < len(lines):
                        next_line = lines[j]
                        if not next_line.strip():
                            block_lines.append(next_line)
                            j += 1
                            continue
                            
                        next_stripped = next_line.lstrip()
                        next_indent_len = len(next_line) - len(next_stripped)
                        last_meaningful_line = next((l.strip() for l in reversed(block_lines) if l.strip()), None)

                        if next_indent_len > len(indent) or \
                        (last_meaningful_line and next_indent_len == len(indent) and last_meaningful_line.endswith(('->', '->>', '~>'))):
                            block_lines.append(next_line)
                            j += 1
                        else:
                            break
                    
                    self._transpile_belt_chain(block_lines, indent, self.line_number)
                    i = j
                
                # Fallback for regular lines
                else:
                    self.output.append(line)
                    i += 1
                    
        except Exception as e:
            print(f"Error during transpilation at line {self.line_number}: {e}", file=sys.stderr)
            raise
        
        return _RUNTIME_SOURCE + "\n\n\n# --- Transpiled Code ---\n\n" + "\n".join(self.output)

    def _format_call(self, var: str, func_expr: str, op: str) -> str:
        """Format a function call, now with more robust expression parsing."""
        func_expr = func_expr.split('#', 1)[0].strip()
        
        # Handle the underscore placeholder first, as it's a special case
        if re.search(r'\b_\b', func_expr):
            call = re.sub(r'\b_\b', var, func_expr)
            # Check if the result is a simple call or a more complex expression
            func_name_match = re.match(r'([a-zA-Z_]\w*)', call)
            func_name = func_name_match.group(1) if func_name_match else ''
            is_terminal = func_name in self.TERMINAL_FUNCTIONS
            return call if is_terminal else f"{var} = {call}"
        
        # Standard function call parsing
        func_match = re.match(r'^([a-zA-Z_]\w*)\s*\((.*)\)\s*$', func_expr)
        if func_match:
            func_name, args_part = func_match.groups()
            func_name = func_name.strip()
            args_part = args_part.strip()
        else:
            func_name = func_expr
            args_part = ""
        
        # This prevents creating calls with empty function names
        if not func_name:
            return "" # Return an empty string for invalid expressions

        call_var = f'_sc_smart_cast({var}, "{func_name}")' if op == '~>' else var
        
        # Build the call string
        if args_part:
            call = f'_sc_call({call_var}, "{func_name}", {args_part})'
        else:
            call = f'_sc_call({call_var}, "{func_name}")'
        
        is_terminal = func_name in self.TERMINAL_FUNCTIONS or op == '->>'
        return call if is_terminal else f"{var} = {call}"

    def _transpile_belt_chain(self, block_lines: List[str], indent: str, original_line_num: int) -> None:
        """Transpiler for belt chains"""
        belt_str_full = ' '.join(line.strip() for line in block_lines)
        self.output.append(f"{indent}# sugarcoat:line:{original_line_num}")
        self.output.append(f"{indent}# Transpiled from: {belt_str_full}")

        has_return = False
        belt_str = belt_str_full.strip()
        if belt_str.startswith('return '):
            has_return = True
            belt_str = belt_str[7:].strip()
        
        parts = self._safe_split_on_operators(belt_str, ['->', '->>', '~>'])
        if not parts: return
            
        initial_expr = parts[0].strip()
        parsed_expr = self._parse_expression(initial_expr)

        if has_return and parsed_expr.has_assignment:
            # This is probs a good place to throw a more specific transpiler error in the future
            # Rn we'll prevent invalid Python from being generated
            self.output.append(f"{indent}# TranspilerError: Cannot combine 'return' with a new variable assignment.")
            return

        temp_var = self.get_temp_var()
        self.output.append(f"{indent}{temp_var} = {parsed_expr.processed}")
        
        i = 1
        while i < len(parts):
            if i + 1 >= len(parts): break
            op = parts[i]
            expr = parts[i + 1].strip()
            grab_match = re.match(r'^grab\s*\((.+)\)\s*$', expr)
            if grab_match:
                var_to_assign = grab_match.group(1).strip()
                self.output.append(f"{indent}{var_to_assign} = {temp_var}")
            elif (repeat_match := re.match(r'^\((.+)\)\s*\{(\d+)\}$', expr)):
                actions_str, count = repeat_match.groups()
                self.output.append(f"{indent}for _ in range({count}):")
                action_parts = self._safe_split_on_operators(actions_str, ['->', '->>', '~>'])
                k = 0
                while k < len(action_parts):
                    action_expr = action_parts[k]
                    action_op = action_parts[k + 1] if k + 1 < len(action_parts) else '->'
                    action_call = self._format_call(temp_var, action_expr, action_op)
                    self.output.append(f"{indent}    {action_call}")
                    k += 2
            elif op != '~>' and (brace_match := re.match(r'^\s*([a-zA-Z_]\w*)\s*\{(.+)\}\s*$', expr)):
                func_name, content = brace_match.groups()
                content = content.strip()
                try:
                    count = int(content)
                    self.output.append(f"{indent}for _ in range({count}):")
                    call_stmt = self._format_call(temp_var, func_name, op)
                    self.output.append(f"{indent}    {call_stmt}")
                except (ValueError, TypeError):
                    for_match = re.match(r'^\s*for\s+(.+?)\s+in\s+(.+?)\s*$', content)
                    if for_match:
                        loop_var, iterable = for_match.groups()
                        self.output.append(f"{indent}for {loop_var.strip()} in {iterable.strip()}:")
                        call_stmt = f"{temp_var} = _sc_call({temp_var}, '{func_name}', {loop_var.strip()})"
                        self.output.append(f"{indent}    {call_stmt}")
                    else:
                        condition = re.sub(r'\b_\b', temp_var, content)
                        self.output.append(f"{indent}if {condition}:")
                        call_stmt = self._format_call(temp_var, func_name, op)
                        self.output.append(f"{indent}    {call_stmt}")
            else:
                if op == '~>':
                    self.output.append(f"{indent}{temp_var} = {expr}({temp_var})")
                else:
                    self.output.append(f"{indent}{self._format_call(temp_var, expr, op)}")
            i += 2
        
        if has_return:
            self.output.append(f"{indent}return {temp_var}")
        elif parsed_expr.has_assignment:
            self.output.append(f"{indent}{parsed_expr.assignment_var} = {temp_var}")

    def _transpile_control_flow_block(self, block_lines: List[str], indent: str, original_line_num: int) -> None:
        """Transpilelelerrer"""
        self.output.append(f"{indent}# sugarcoat:line:{original_line_num}")
        self.output.append(f"{indent}# Transpiled from: {' '.join(line.strip() for line in block_lines)}")

        # 1. Deconstruct the very first line
        first_line_full = block_lines[0].strip()
        has_return = first_line_full.startswith('return ')
        if has_return:
            first_line_full = first_line_full[7:].strip()

        first_op_match = re.search(r'\s*->\s*', first_line_full)
        if not first_op_match: return

        initial_expr_str = first_line_full[:first_op_match.start()]
        control_part_str = first_line_full[first_op_match.end():]
        
        # 2. Handle the initial value
        parsed_expr = self._parse_expression(initial_expr_str)
        if has_return and parsed_expr.has_assignment:
            self.output.append(f"{indent}# TranspilerError: Cannot combine 'return' with an assignment.")
            return
            
        temp_var = self.get_temp_var()
        self.output.append(f"{indent}{temp_var} = {parsed_expr.processed}")
        
        result_var = self.get_temp_var()
        
        def format_line_content(line: str, var: str, is_case_pattern: bool = False) -> str:
            line = line.strip()
            if is_case_pattern and line.strip(':') == '_': return '_'
            line = re.sub(r'\b_\.([a-zA-Z_]\w*)\b', fr'_sc_get_attr({var}, "\1")', line)
            line = re.sub(r'\b_\b', var, line)
            if not is_case_pattern: line = line.strip('{:}')
            return line

        base_indent = indent
        clause_indent = base_indent + "    "
        case_body_indent = clause_indent + "    "
        current_block_indent = clause_indent

        # 3. Handle the first clause to set up the block
        first_clause_body_stmts = []
        keyword_match = re.match(r'^(if|match)\s*', control_part_str)
        keyword = keyword_match.group(1) if keyword_match else None
        
        if keyword == 'if':
            if_parts = control_part_str.split('->', 1)
            condition_part = if_parts[0]
            if len(if_parts) > 1:
                first_clause_body_stmts.append(self._format_call(temp_var, if_parts[1].strip(), '->'))
            condition = format_line_content(condition_part[2:], temp_var)
            self.output.append(f"{base_indent}if {condition}:")
            current_block_indent = clause_indent
        elif keyword == 'match':
            self.output.append(f"{base_indent}match {temp_var}:")
            self.output.append(f"{base_indent}    {result_var} = None # Default result for match")
        
        # 4. The Main Loop
        lines_to_process = block_lines[1:]
        current_clause_body = first_clause_body_stmts
        
        def flush_clause_body():
            nonlocal current_clause_body
            if not current_clause_body: self.output.append(f"{current_block_indent}pass")
            else: self.output.extend(current_clause_body)
            current_clause_body = []

        for line in lines_to_process:
            stripped = line.lstrip()
            if not stripped: continue

            op_match = re.match(r'^(->>|->|~>)\s*(.*)', stripped)
            if not op_match: continue
            
            op, expr = op_match.groups()
            expr = expr.strip()

            if expr.startswith(('elif ', 'else', 'case ')):
                flush_clause_body()
                
                if expr.startswith('elif '):
                    current_block_indent = clause_indent
                    condition = format_line_content(expr[4:], temp_var)
                    self.output.append(f"{base_indent}elif {condition}:")
                elif expr.startswith('else'):
                    current_block_indent = clause_indent
                    self.output.append(f"{base_indent}else:")
                elif expr.startswith('case '):
                    current_block_indent = case_body_indent
                    pattern_str = expr[4:].strip().strip(':')
                    pattern = format_line_content(pattern_str, temp_var, is_case_pattern=True)
                    self.output.append(f"{clause_indent}case {pattern}:")
            else:
                # It's a statement within the current clause body
                if keyword == 'match' and op == '->>':
                    is_call = re.match(r'^[a-zA-Z_]\w*\s*\(.*\)$', expr.strip())
                    statement = self._format_call(temp_var, expr, op) if is_call else f"{result_var} = {expr}"
                else:
                    statement = self._format_call(temp_var, expr, op)
                current_clause_body.append(f"{current_block_indent}{statement}")

        flush_clause_body()
        
        # 5. Final return or assignment
        if has_return:
            final_var = result_var if keyword == 'match' else temp_var
            self.output.append(f"{indent}return {final_var}")
        elif parsed_expr.has_assignment:
            final_var = result_var if keyword == 'match' else temp_var
            self.output.append(f"{indent}{parsed_expr.assignment_var} = {final_var}")

def main():
    """Main function with source-mapped error reporting for all error types hopefully"""
    parser = argparse.ArgumentParser(
        description="Sugarcoat Transpiler and Runner (b1.0.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python transpiler.py script.sugar              # Run script
  python transpiler.py script.sugar --to-python  # Transpile only
        """
    )
    parser.add_argument("file", nargs='?', help="The .sugar file to process")
    parser.add_argument("--to-python", "-p", action="store_true", 
                       help="Only transpile to a .py file and exit")
    parser.add_argument("--debug", "-d", action="store_true",
                       help="Enable debug output")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' not found", file=sys.stderr)
        sys.exit(1)
    try:
        with open(args.file, 'r', encoding='utf-8') as f:
            sugar_code = f.read()
    except Exception as e:
        print(f"Error reading file '{args.file}': {e}", file=sys.stderr)
        sys.exit(1)

    try:
        transpiler = Transpiler()
        python_code = transpiler.transpile(sugar_code)
        
        if args.debug:
            print("=== Generated Python Code ===")
            print(python_code)
            print("===========================")
            
    except Exception as e:
        print(f"Transpilation error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

    if args.to_python:
        py_file = os.path.splitext(args.file)[0] + ".py"
        try:
            with open(py_file, 'w', encoding='utf-8') as f:
                f.write(python_code)
            print(f"Successfully transpiled to '{py_file}'")
        except Exception as e:
            print(f"Error writing to '{py_file}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            exec_globals = { '__name__': '__main__', '__file__': args.file }
            compiled_code = compile(python_code, args.file, 'exec')
            exec(compiled_code, exec_globals)
        
        except Exception as e:
            import traceback
            py_error_line_num = None
            
            if isinstance(e, SyntaxError):
                # For SyntaxError the line number is directly on the exception object
                py_error_line_num = e.lineno
            else:
                # For runtime errors we parse the traceback
                tb = e.__traceback__
                if tb:
                    last_frame = traceback.extract_tb(tb)[-1]
                    py_error_line_num = last_frame.lineno

            sugar_line_num = None
            if py_error_line_num:
                python_lines = python_code.splitlines()
                # Search backwards for the source map comment
                if py_error_line_num <= len(python_lines):
                    for i in range(py_error_line_num - 1, -1, -1):
                        line = python_lines[i]
                        if line.strip().startswith("# sugarcoat:line:"):
                            try:
                                sugar_line_num = int(line.strip().split(':')[-1])
                                break
                            except (ValueError, IndexError):
                                continue
            
            print("="*60, file=sys.stderr)
            print("🔥 Sugarcoat Runtime Error 🔥", file=sys.stderr)
            if sugar_line_num:
                print(f"    File: \"{args.file}\", line {sugar_line_num}", file=sys.stderr)
            
            error_msg = f"{type(e).__name__}: {e}"
            # Clean up the error message for SyntaxError to avoid confusion
            if isinstance(e, SyntaxError):
                error_msg = f"SyntaxError: {e.msg}"
                
            print(f"    Error: {error_msg}", file=sys.stderr)
            print("="*60, file=sys.stderr)

            if args.debug:
                print("\n--- Python Traceback (for debugging the transpiler) ---", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)

            sys.exit(1)

if __name__ == "__main__":
    main()