# Sugarcoat

A sweet Python transpiler that adds syntactic sugar for more expressive and readable code.

# NOTE: THIS IS IN EARLY BETA IF NOT ALPHA, EXPECT THINGS TO BREAK AND SOME FEATURES NOT TO WORK!
(but hopefully they do work)

## Features

- **Belt/Pipeline Operations**: Chain operations with `->`, `->>`, and `~>` operators
- **Ruby-style String Interpolation**: Use `#{}` syntax in strings
- **Ternary Expressions**: Concise conditional expressions with `?:`
- **Method Aliases**: Intuitive method names like `length`, `empty?`, `nil?`
- **Class Syntactic Sugar**: Simplified class definitions with `@` instance variables
- **Smart Type Casting**: Automatic type conversion with `~>` operator
- **Constructor Sugar**: `.new()` calls simplified

## Installation

```bash
git clone https://github.com/yourusername/sugarcoat
cd sugarcoat
```

## Usage

### Basic Belt Operations

```sugar
# Chain operations with ->
"hello world" -> upcase -> print

# Terminal operations with ->>
[1, 2, 3] -> reverse ->> print

# Smart casting with ~>
"42" ~> int -> print
```

### String Interpolation

```sugar
name = "World"
"Hello #{name}!" -> print
```

### Ternary Expressions

```sugar
age = 25
age >= 18 ? "adult" : "minor" -> print
```

### Control Flow with Belts

```sugar
numbers = [1, 2, 3, 4, 5]
numbers -> if _ > 3
  -> multiply(2)
  ->> print
```

### Class Definitions

```sugar
class Person
  def initialize(name, age)
    @name = name
    @age = age
  
  def greet
    "Hello, I'm #{@name}" -> print
```

### Method Aliases

```sugar
my_list = [1, 2, 3]
my_list.length -> print     # same as len()
my_list.empty? -> print     # checks if empty
```

## Running Sugarcoat

```bash
# Run a .sugar file
python transpiler.py script.sugar

# Transpile to Python only
python transpiler.py script.sugar --to-python

# Debug mode
python transpiler.py script.sugar --debug
```

## Language Reference

### Belt Operators

| Operator | Description | Example |
|----------|-------------|---------|
| `->` | Standard belt | `value -> function` |
| `->>` | Terminal belt (doesn't assign result) | `value ->> print` |
| `~>` | Smart casting belt | `"42" ~> int` |

### Built-in Aliases

| Alias | Python Equivalent | Description |
|-------|-------------------|-------------|
| `trim` | `strip()` | Remove whitespace |
| `upcase` | `upper()` | Convert to uppercase |
| `downcase` | `lower()` | Convert to lowercase |
| `length` | `len()` | Get length |
| `empty?` | `len() == 0` | Check if empty |
| `nil?` | `is None` | Check if None |

### Advanced Features

**Conditional Blocks:**
```sugar
data -> if data.length > 0
  -> process
  -> validate
  ->> save
```

**Pattern Matching:**
```sugar
value -> match
  -> case 1: "one"
  -> case 2: "two"
  -> case _: "other"
```

**Repetition:**
```sugar
"Hello" -> print{5}  # Print 5 times
```

## Examples

### Data Processing Pipeline

```sugar
# Read, process, and save data
"data.txt" 
  -> read_file
  -> parse_json
  -> if _.valid?
    -> transform
    -> validate
    ->> save_to_db
```

### Web API Response

```sugar
response = api_call("/users/1")
response.status == 200 ? response.data : {} 
  -> format_user
  ->> render_template
```

## Error Handling

Sugarcoat provides error messages with line number mapping back to your original `.sugar` files:

```
🔥 Sugarcoat Runtime Error 🔥
    File: "script.sugar", line 15
    Error: AttributeError: 'str' object has no attribute 'invalid_method'
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

## License

MIT License - see LICENSE file for details
