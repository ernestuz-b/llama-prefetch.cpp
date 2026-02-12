# C++ Style Guide

**Based on llama.cpp Project**

A practical guide for writing high-performance, portable C++ code.

---

## Core Philosophy

The llama.cpp project prioritizes **simplicity, compatibility, and pragmatism** over modern C++ sophistication. The goal is enabling LLM inference with minimal setup and state-of-the-art performance on a wide range of hardware.

### Guiding Principles

| Principle | Description |
|-----------|-------------|
| **Simplicity First** | Avoid complex modern STL constructs, use basic for loops, avoid templates. Code should be straightforward and easy to understand for developers of all skill levels. |
| **Cross-Platform Compatibility** | Always consider compatibility with Windows, Linux, macOS and architectures (x86, ARM, RISC-V). Test on multiple platforms before submitting. |
| **Minimal Dependencies** | Avoid adding third-party dependencies, extra files, or extra headers. Each new dependency increases maintenance burden and compatibility issues. |
| **Pragmatic Over Dogmatic** | Follow existing codebase patterns (indentation, spacing, naming). Consistency within the project is more important than external style guides. |

---

## Code Style & Formatting

### Formatting Tool

Use `clang-format` (from clang-tools v15 or later). The project includes a [`.clang-format`](.clang-format) file that defines all formatting rules.

### Formatting Guidelines

- **Indentation**: Follow existing patterns (typically 4 spaces)
- **Line length**: Pragmatic approach — readability over strict limits
- **Spaces**: Consistent spacing around operators and after keywords
- **Braces**: Opening brace on same line for functions and control structures

### Namespace Usage

The project does **NOT** use traditional C++ namespace declarations. Instead, it follows a C-style approach with prefix-based namespacing:

- Use `llama_*` prefix (e.g., `llama_model_load`, `llama_init_from_gpt_params`)
- Use `ggml_*` prefix (e.g., `ggml_cuda_error`, `ggml_backend_dev_t`)
- **Do NOT** use `using namespace std;` or similar declarations
- Use explicit `std::` prefix for standard library features (e.g., `std::vector`, `std::string`)

---

## Naming Conventions

| Category | Convention | Examples |
|----------|------------|----------|
| **Functions** | `snake_case` with prefix | `llama_model_load()`, `ggml_cuda_init()`, `llama_kv_cache_seq_rm()` |
| **Variables** | `snake_case` | `n_gpu_layers`, `model_path`, `ctx_size` |
| **Constants/Defines** | `UPPER_CASE` with prefix | `LLAMA_LOG_INFO`, `GGML_CUDA_FORCE_MMQ`, `GGML_USE_K_QUANTS` |
| **Types/Structures** | `snake_case` with prefix | `llama_model_params`, `llama_context_params`, `ggml_backend_dev_t` |

---

## Language Features

### Control Structures

Prefer simple, traditional constructs:

```cpp
// Good - simple and clear
for (size_t i = 0; i < n; i++) {
    // process
}

if (condition) {
    // action
}
```

Avoid overly fancy range-based loops unless clearly beneficial.

### Templates & Modern C++

**Avoid Templates** - Keep code simple and avoid template metaprogramming. Templates add complexity and can make code harder to understand and debug.

**Minimize Modern C++ Features**:
- Prefer simple, readable code over modern C++ idioms
- Use C++17 features sparingly and only when they provide clear benefits
- Plain C-style interfaces are preferred for the public API

---

## Best Practices

### Error Handling

**Exceptions**: Use `std::runtime_error` for exceptions:

```cpp
throw std::runtime_error("n_gpu_layers already set");
```

**Logging**: Use `LLAMA_LOG_*` macros:

```cpp
LLAMA_LOG_INFO("%s: error message\n", __func__);
```

- Include `__func__` for function context
- Use printf-style format strings
- Use appropriate log levels (INFO, WARN, ERROR)

### Memory Management

- Use straightforward memory allocation patterns
- Prefer stack allocation when possible
- Ensure clear ownership semantics
- Avoid complex smart pointer patterns unless necessary

### Comments & Documentation

**Function Documentation**: Document parameters and behavior, especially for public APIs. Keep documentation concise but complete.

**Inline Comments**: Use sparingly. Code should be self-documenting where possible. Add comments only when the intent is not obvious from the code itself.

**TODO Comments**: Use for marking future work:

```cpp
// TODO: implement XYZ feature
```

---

## Platform & Build Considerations

### Cross-Platform Testing

Always test on multiple platforms and architectures:

| Category | Options |
|----------|---------|
| **Operating Systems** | Windows, Linux, macOS |
| **Architectures** | x86, ARM, RISC-V |

- Use platform-specific code only when necessary
- Isolate platform-specific code in separate modules

### Build Flags

| Flag | Purpose |
|------|---------|
| `-std=c11` | C code compilation |
| `-std=c++11` | C++ code compilation |
| `-Wall -Wextra -Wpedantic` | Warning flags |
| `-O3`, NEON, Metal, CUDA | Platform-specific optimizations |

---

## File Organization

### Header Guards

Use traditional header guards (not `#pragma once`):

```cpp
#ifndef LLAMA_H
#define LLAMA_H

// ... header content ...

#endif // LLAMA_H
```

### Include Order

1. System headers first
2. Project headers second
3. Keep includes minimal — only include what you actually use

---

## Summary

The llama.cpp style emphasizes:

- ✅ Simplicity over sophistication
- ✅ Compatibility over platform-specific optimizations
- ✅ C-style APIs with C++ implementation
- ✅ Minimal dependencies and self-contained code
- ✅ Prefix-based namespacing rather than C++ namespaces
- ✅ Pragmatic over dogmatic adherence to "modern" C++

---

## Additional Resources

### Project Links

- **GitHub**: https://github.com/ggml-org/llama.cpp
- **[CONTRIBUTING.md](CONTRIBUTING.md)**: Project contribution guidelines
- **[`.clang-format`](.clang-format)**: Automated formatting rules

### External References

For anything not covered in this guide, refer to the C++ Core Guidelines with the understanding that llama.cpp prioritizes simplicity and portability over strict modern C++ practices.
