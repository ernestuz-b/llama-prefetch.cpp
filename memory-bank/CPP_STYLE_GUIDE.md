C++ Style Guide
Based on llama.cpp Project
A practical guide for writing high-performance, portable C++ code

Core Principles
Simplicity First
Avoid fancy modern STL constructs, use basic for loops, avoid templates, keep it simple. The code should be straightforward and easy to understand for developers of all skill levels.
Cross-Platform Compatibility
Always consider cross-compatibility with other operating systems (Windows, Linux, macOS) and architectures (x86, ARM, RISC-V). Test your code on multiple platforms before submitting.
Minimal Dependencies
Avoid adding third-party dependencies, extra files, or extra headers. Each new dependency increases the maintenance burden and potential for compatibility issues.
Pragmatic Over Dogmatic
There are no strict rules for code style, but try to follow the patterns in the existing codebase (indentation, spacing, naming conventions). Consistency within the project is more important than following external style guides.

Namespace Usage
The llama.cpp project does NOT use traditional C++ namespace declarations. Instead, it follows a C-style approach with prefix-based namespacing:
    • Use prefix-based namespacing with function names like 
llama_* (e.g., llama_model_load, llama_init_from_gpt_params)
    • Use 
ggml_* (e.g., ggml_cuda_error, ggml_backend_dev_t)
    • Do NOT use 
using namespace std;
or similar declarations
    • When using standard library features, use explicit 
std::
prefix (e.g., std::vector, std::string)
Code Formatting
Formatting Tool
Use clang-format (from clang-tools v15 or later) to format code. The project includes a .clang-format file that defines all formatting rules.
Key Formatting Guidelines
    • Indentation: Follow existing patterns in the code (typically 4 spaces)
    • Line length: Pragmatic approach — readability over strict limits
    • Spaces: Consistent spacing around operators and after keywords
    • Braces: Opening brace on same line for functions and control structures

Naming Conventions
Functions
Use snake_case with prefixes for namespacing:
llama_model_load()
ggml_cuda_init()
llama_kv_cache_seq_rm()
Variables
Use snake_case for all variables:
n_gpu_layers
model_path
ctx_size
Constants and Defines
Use UPPER_CASE with prefix:
LLAMA_LOG_INFO
GGML_CUDA_FORCE_MMQ
GGML_USE_K_QUANTS
Types and Structures
Use snake_case with prefix:
llama_model_params
llama_context_params
ggml_backend_dev_t

Control Structures
Prefer Simple For Loops
// Good - simple and clear
for (size_t i = 0; i < n; i++) {
    // process
}

// Avoid overly fancy range-based loops
// unless clearly beneficial
Basic Conditionals
if (condition) {
    // action
}
Templates and Modern C++
Avoid Templates
Keep code simple and avoid template metaprogramming. Templates add complexity and can make code harder to understand and debug.
Minimize Modern C++ Features
    • Prefer simple, readable code over modern C++ idioms
    • Use C++17 features sparingly and only when they provide clear benefits
    • Plain C-style interfaces are preferred for the public API

Error Handling
Using Exceptions
Use std::runtime_error for exceptions:
throw std::runtime_error("n_gpu_layers already set");
Logging Errors
Use the LLAMA_LOG_* macros for logging:
LLAMA_LOG_INFO("%s: error message\n", __func__);

    • Include __func__ for function context
    • Use printf-style format strings
    • Use appropriate log levels (INFO, WARN, ERROR)
Memory Management
    • Use straightforward memory allocation patterns
    • Prefer stack allocation when possible
    • Ensure clear ownership semantics
    • Avoid complex smart pointer patterns unless necessary

Platform Considerations
Cross-Platform Testing
Always test on multiple platforms and architectures:
    • Operating systems: Windows, Linux, macOS
    • Architectures: x86, ARM, RISC-V
    • Use platform-specific code only when necessary
    • Isolate platform-specific code in separate modules
Build Flags
The project supports various compilation flags:
    • -std=c11 for C code
    • -std=c++11 for C++ code
    • Warning flags: -Wall -Wextra -Wpedantic
    • Platform-specific optimizations: -O3, NEON, Metal, CUDA
Comments and Documentation
Function Documentation
Document parameters and behavior, especially for public APIs. Keep documentation concise but complete.
Inline Comments
Use sparingly. Code should be self-documenting where possible. Add comments only when the intent is not obvious from the code itself.
TODO Comments
Use for marking future work:
// TODO: implement XYZ feature

File Organization
Header Guards
Use traditional header guards (not #pragma once):
#ifndef LLAMA_H
#define LLAMA_H
// ... header content ...
#endif // LLAMA_H
Include Order
    • System headers first
    • Project headers second
    • Keep includes minimal — only include what you actually use
Summary
The llama.cpp style emphasizes:
    • Simplicity over sophistication
    • Compatibility over platform-specific optimizations
    • C-style APIs with C++ implementation
    • Minimal dependencies and self-contained code
    • Prefix-based namespacing rather than C++ namespaces
    • Pragmatic over dogmatic adherence to &#x201C;modern&#x201D; C++

This style guide reflects the project&#x2019;s goal: enabling LLM inference with minimal setup and state-of-the-art performance on a wide range of hardware.

Additional Resources
Project Links
    • GitHub: https://github.com/ggml-org/llama.cpp
    • CONTRIBUTING.md: Project contribution guidelines
    • .clang-format: Automated formatting rules
External References
For anything not covered in this guide, refer to the C++ Core Guidelines with the understanding that llama.cpp prioritizes simplicity and portability over strict modern C++ practices.