"""
Prompt template system for variable substitution and processing.

Provides template processing with variable substitution, conditional logic,
and prompt composition capabilities.
"""

import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

from core.exceptions import ConfigurationError
from core.observability import get_logger


class PromptTemplate:
    """
    Template for processing prompts with variable substitution.
    
    Supports Jinja2-style template syntax with variable substitution,
    conditional logic, and loop constructs.
    """

    def __init__(self, template: str):
        """
        Initialize prompt template.
        
        Args:
            template: Template string with variable placeholders
        """
        self.template = template
        self.logger = get_logger("prompts.template")
        
        # Compile template patterns
        self._variable_pattern = re.compile(r'\{\{\s*([^}]+)\s*\}\}')
        self._conditional_pattern = re.compile(r'\{%\s*if\s+([^%]+)\s*%\}')
        self._end_conditional_pattern = re.compile(r'\{%\s*endif\s*%\}')
        self._loop_pattern = re.compile(r'\{%\s*for\s+(\w+)\s+in\s+([^%]+)\s*%\}')
        self._end_loop_pattern = re.compile(r'\{%\s*endfor\s*%\}')

    async def render(self, variables: Dict[str, Any]) -> str:
        """
        Render template with variables.
        
        Args:
            variables: Variables for substitution
            
        Returns:
            Rendered prompt text
        """
        try:
            result = self.template
            
            # Process conditionals
            result = await self._process_conditionals(result, variables)
            
            # Process loops
            result = await self._process_loops(result, variables)
            
            # Process variable substitutions
            result = await self._process_variables(result, variables)
            
            return result.strip()
            
        except Exception as e:
            self.logger.error(f"Template rendering failed: {e}")
            raise ConfigurationError(f"Template rendering failed: {e}") from e

    async def _process_variables(self, text: str, variables: Dict[str, Any]) -> str:
        """Process variable substitutions."""
        def replace_variable(match):
            var_path = match.group(1).strip()
            
            try:
                # Handle nested dictionary access (e.g., user.name)
                value = self._get_nested_value(variables, var_path)
                return str(value) if value is not None else ""
            except (KeyError, AttributeError, TypeError):
                # Return original placeholder if variable not found
                return match.group(0)
        
        return self._variable_pattern.sub(replace_variable, text)

    async def _process_conditionals(self, text: str, variables: Dict[str, Any]) -> str:
        """Process conditional blocks."""
        result = text
        
        # Find all conditional blocks
        conditional_matches = list(self._conditional_pattern.finditer(result))
        
        for match in reversed(conditional_matches):  # Process in reverse order
            start_pos = match.start()
            condition = match.group(1).strip()
            
            # Find matching endif
            endif_match = self._end_conditional_pattern.search(result, match.end())
            if not endif_match:
                continue
            
            end_pos = endif_match.end()
            content = result[match.end():endif_match.start()]
            
            # Evaluate condition
            if self._evaluate_condition(condition, variables):
                # Keep content
                result = result[:start_pos] + content + result[end_pos:]
            else:
                # Remove content
                result = result[:start_pos] + result[end_pos:]
        
        return result

    async def _process_loops(self, text: str, variables: Dict[str, Any]) -> str:
        """Process loop blocks."""
        result = text
        
        # Find all loop blocks
        loop_matches = list(self._loop_pattern.finditer(result))
        
        for match in reversed(loop_matches):  # Process in reverse order
            start_pos = match.start()
            loop_var = match.group(1)
            iterable_path = match.group(2).strip()
            
            # Find matching endfor
            endfor_match = self._end_loop_pattern.search(result, match.end())
            if not endfor_match:
                continue
            
            end_pos = endfor_match.end()
            content = result[match.end():endfor_match.start()]
            
            # Get iterable
            try:
                iterable = self._get_nested_value(variables, iterable_path)
                if not iterable or not hasattr(iterable, '__iter__'):
                    # Remove loop if iterable is empty or invalid
                    result = result[:start_pos] + result[end_pos:]
                    continue
                
                # Process loop content for each item
                loop_result = ""
                for item in iterable:
                    loop_vars = {**variables, loop_var: item}
                    item_content = await self._process_variables(content, loop_vars)
                    loop_result += item_content
                
                result = result[:start_pos] + loop_result + result[end_pos:]
                
            except (KeyError, AttributeError, TypeError):
                # Remove loop if iterable not found
                result = result[:start_pos] + result[end_pos:]
        
        return result

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value from dictionary using dot notation."""
        keys = path.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                raise KeyError(f"Key '{key}' not found in path '{path}'")
        
        return value

    def _evaluate_condition(self, condition: str, variables: Dict[str, Any]) -> bool:
        """Evaluate a conditional expression."""
        try:
            # Simple condition evaluation
            # Support: variable, variable == value, variable != value, etc.
            
            # Handle simple variable existence
            if condition in variables:
                value = variables[condition]
                return bool(value)
            
            # Handle comparison operators
            for op in ['==', '!=', '>', '<', '>=', '<=']:
                if op in condition:
                    left, right = condition.split(op, 1)
                    left = left.strip()
                    right = right.strip()
                    
                    left_value = self._get_nested_value(variables, left)
                    right_value = self._parse_value(right, variables)
                    
                    if op == '==':
                        return left_value == right_value
                    elif op == '!=':
                        return left_value != right_value
                    elif op == '>':
                        return left_value > right_value
                    elif op == '<':
                        return left_value < right_value
                    elif op == '>=':
                        return left_value >= right_value
                    elif op == '<=':
                        return left_value <= right_value
            
            return False
            
        except Exception:
            return False

    def _parse_value(self, value_str: str, variables: Dict[str, Any]) -> Any:
        """Parse a value string, handling variables and literals."""
        value_str = value_str.strip()
        
        # Remove quotes if present
        if (value_str.startswith('"') and value_str.endswith('"')) or \
           (value_str.startswith("'") and value_str.endswith("'")):
            return value_str[1:-1]
        
        # Check if it's a variable
        if value_str in variables:
            return variables[value_str]
        
        # Try to parse as number
        try:
            if '.' in value_str:
                return float(value_str)
            else:
                return int(value_str)
        except ValueError:
            pass
        
        # Try to parse as boolean
        if value_str.lower() in ['true', 'false']:
            return value_str.lower() == 'true'
        
        # Return as string
        return value_str

    def add_variable(self, name: str, value: Any) -> None:
        """Add a variable to the template (for testing)."""
        # This method is for testing purposes
        pass

    def get_variables(self) -> List[str]:
        """Get list of variables used in the template."""
        variables = set()
        
        # Find all variable references
        for match in self._variable_pattern.finditer(self.template):
            var_path = match.group(1).strip()
            # Extract base variable name (before any dots)
            base_var = var_path.split('.')[0]
            variables.add(base_var)
        
        return list(variables)

    def validate(self, variables: Dict[str, Any]) -> List[str]:
        """
        Validate template against variables.
        
        Args:
            variables: Available variables
            
        Returns:
            List of missing variables
        """
        required_vars = self.get_variables()
        missing_vars = []
        
        for var in required_vars:
            if var not in variables:
                missing_vars.append(var)
        
        return missing_vars

    def __str__(self) -> str:
        """String representation of template."""
        return f"PromptTemplate({self.template[:50]}...)" if len(self.template) > 50 else f"PromptTemplate({self.template})"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"PromptTemplate(template='{self.template}', variables={self.get_variables()})"
